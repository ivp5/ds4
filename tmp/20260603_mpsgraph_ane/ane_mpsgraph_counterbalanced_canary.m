#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <dispatch/dispatch.h>

static double now_seconds(void) {
    return [NSDate timeIntervalSinceReferenceDate];
}

static void die(NSString *message, NSError *error) {
    if (error) {
        fprintf(stderr, "%s: %s\n", message.UTF8String, error.localizedDescription.UTF8String);
    } else {
        fprintf(stderr, "%s\n", message.UTF8String);
    }
    exit(1);
}

static NSArray<NSNumber *> *shape_for_feature(NSDictionary<NSString *, MLFeatureDescription *> *descriptions,
                                              NSString **name_out) {
    NSString *name = descriptions.allKeys.firstObject;
    if (!name) die(@"missing feature", nil);
    MLFeatureDescription *description = descriptions[name];
    NSArray<NSNumber *> *shape = description.multiArrayConstraint.shape;
    if (shape.count != 2) {
        die([NSString stringWithFormat:@"expected rank-2 MLMultiArray for %@", name], nil);
    }
    *name_out = name;
    return shape;
}

static MLMultiArray *make_buffer_multiarray(id<MTLBuffer> buffer,
                                            NSArray<NSNumber *> *shape,
                                            MLMultiArrayDataType data_type) {
    if (shape.count != 2) die(@"rank-2 shape required", nil);
    NSArray<NSNumber *> *strides = @[shape[1], @1];
    NSError *error = nil;
    MLMultiArray *array = [[MLMultiArray alloc] initWithDataPointer:buffer.contents
                                                              shape:shape
                                                           dataType:data_type
                                                            strides:strides
                                                        deallocator:nil
                                                              error:&error];
    if (!array) die(@"MLMultiArray initWithDataPointer failed", error);
    return array;
}

static void fill_half_buffer(id<MTLBuffer> buffer, NSUInteger half_words, uint16_t value) {
    uint16_t *data = (uint16_t *)buffer.contents;
    for (NSUInteger index = 0; index < half_words; ++index) data[index] = value;
}

static double evict_cpu_cache(uint8_t *buffer, size_t bytes) {
    if (!buffer || bytes == 0) return 0.0;
    volatile uint8_t accumulator = 0;
    double start = now_seconds();
    for (size_t offset = 0; offset < bytes; offset += 64) {
        accumulator = (uint8_t)(accumulator + buffer[offset]);
        buffer[offset] = (uint8_t)(buffer[offset] + accumulator + 1u);
    }
    return now_seconds() - start;
}

static MLModel *load_model(NSString *path, id<MTLDevice> device) {
    NSError *error = nil;
    NSURL *model_url = [NSURL fileURLWithPath:path];
    NSLog(@"[coreml] compile_start path=%@", path);
    NSURL *compiled_url = [MLModel compileModelAtURL:model_url error:&error];
    if (!compiled_url) die(@"compileModelAtURL failed", error);
    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    config.preferredMetalDevice = device;
    MLModel *model = [MLModel modelWithContentsOfURL:compiled_url configuration:config error:&error];
    if (!model) die(@"modelWithContentsOfURL failed", error);
    return model;
}

static MPSGraphExecutable *make_mps_executable(uint32_t batch,
                                               uint32_t input_dim,
                                               uint32_t output_dim,
                                               MPSGraphTensor **input_tensor_out) {
    MPSGraph *graph = [[MPSGraph alloc] init];
    graph.options = MPSGraphOptionsDefault;
    MPSGraphTensor *input_tensor = [graph placeholderWithShape:@[@(batch), @(input_dim)]
                                                      dataType:MPSDataTypeFloat16
                                                          name:@"mps_input"];
    NSMutableData *weight_data =
        [NSMutableData dataWithLength:(NSUInteger)input_dim * (NSUInteger)output_dim * sizeof(uint16_t)];
    uint16_t *weight = (uint16_t *)weight_data.mutableBytes;
    for (NSUInteger index = 0; index < (NSUInteger)input_dim * (NSUInteger)output_dim; ++index) {
        weight[index] = (uint16_t)(0x3400u + (uint16_t)(index & 0x3fu));
    }
    MPSGraphTensor *weight_tensor = [graph constantWithData:weight_data
                                                      shape:@[@(input_dim), @(output_dim)]
                                                   dataType:MPSDataTypeFloat16];
    MPSGraphTensor *output_tensor = [graph matrixMultiplicationWithPrimaryTensor:input_tensor
                                                                 secondaryTensor:weight_tensor
                                                                            name:@"mps_dense"];
    MPSGraphShapedType *input_type = [[MPSGraphShapedType alloc] initWithShape:@[@(batch), @(input_dim)]
                                                                       dataType:MPSDataTypeFloat16];
    MPSGraphExecutable *executable = [graph compileWithDevice:nil
                                                        feeds:@{input_tensor : input_type}
                                                targetTensors:@[output_tensor]
                                             targetOperations:nil
                                        compilationDescriptor:nil];
    if (!executable) die(@"MPSGraph compile failed", nil);
    executable.options = MPSGraphOptionsNone;
    *input_tensor_out = input_tensor;
    return executable;
}

static void run_ane_once(MLModel *model,
                         MLDictionaryFeatureProvider *provider,
                         MLPredictionOptions *options) {
    NSError *error = nil;
    id<MLFeatureProvider> prediction = [model predictionFromFeatures:provider options:options error:&error];
    if (!prediction) die(@"CoreML predict failed", error);
}

static void run_mps_once(MPSGraphExecutable *executable,
                         id<MTLCommandQueue> queue,
                         NSArray<MPSGraphTensorData *> *inputs,
                         NSArray<MPSGraphTensorData *> *outputs) {
    [executable runWithMTLCommandQueue:queue
                           inputsArray:inputs
                          resultsArray:outputs
                   executionDescriptor:nil];
}

static void record_sample(NSMutableDictionary<NSString *, NSMutableArray<NSNumber *> *> *samples,
                          NSString *name,
                          double milliseconds) {
    NSMutableArray<NSNumber *> *values = samples[name];
    if (!values) {
        values = [NSMutableArray array];
        samples[name] = values;
    }
    [values addObject:@(milliseconds)];
}

static uint32_t next_random(uint32_t *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static void shuffle_cases(NSMutableArray<NSString *> *cases, uint32_t *state) {
    for (NSUInteger index = cases.count; index > 1; --index) {
        NSUInteger swap_index = next_random(state) % index;
        [cases exchangeObjectAtIndex:index - 1 withObjectAtIndex:swap_index];
    }
}

static double sorted_value(NSArray<NSNumber *> *sorted_values, double fraction) {
    if (sorted_values.count == 0) return 0.0;
    NSUInteger index = (NSUInteger)llround(fraction * (double)(sorted_values.count - 1));
    if (index >= sorted_values.count) index = sorted_values.count - 1;
    return sorted_values[index].doubleValue;
}

static NSDictionary<NSString *, NSNumber *> *summary_for_values(NSArray<NSNumber *> *values) {
    NSArray<NSNumber *> *sorted_values =
        [values sortedArrayUsingComparator:^NSComparisonResult(NSNumber *left, NSNumber *right) {
            return [left compare:right];
        }];
    double sum = 0.0;
    for (NSNumber *value in values) sum += value.doubleValue;
    return @{
        @"n" : @(values.count),
        @"min" : sorted_values.firstObject ?: @0,
        @"p50" : @(sorted_value(sorted_values, 0.50)),
        @"p90" : @(sorted_value(sorted_values, 0.90)),
        @"max" : sorted_values.lastObject ?: @0,
        @"mean" : values.count ? @(sum / (double)values.count) : @0,
    };
}

static double p50_for_case(NSDictionary<NSString *, NSMutableArray<NSNumber *> *> *samples, NSString *name) {
    NSDictionary<NSString *, NSNumber *> *summary = summary_for_values(samples[name] ?: @[]);
    return summary[@"p50"].doubleValue;
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "usage: %s MODEL.mlpackage [trials] [evict_mib] [same|separate] [seed]\n", argv[0]);
            return 2;
        }
        NSString *model_path = [NSString stringWithUTF8String:argv[1]];
        uint32_t trials = argc >= 3 ? (uint32_t)strtoul(argv[2], NULL, 10) : 12u;
        uint32_t evict_mib = argc >= 4 ? (uint32_t)strtoul(argv[3], NULL, 10) : 0u;
        NSString *input_mode = argc >= 5 ? [NSString stringWithUTF8String:argv[4]] : @"same";
        uint32_t random_state = argc >= 6 ? (uint32_t)strtoul(argv[5], NULL, 10) : 0x4d315341u;
        BOOL separate_input = [input_mode isEqualToString:@"separate"];
        if (![input_mode isEqualToString:@"same"] && !separate_input) {
            die(@"input mode must be same or separate", nil);
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) die(@"MTLCreateSystemDefaultDevice failed", nil);
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) die(@"newCommandQueue failed", nil);

        MLModel *model = load_model(model_path, device);
        NSString *input_name = nil;
        NSString *output_name = nil;
        NSArray<NSNumber *> *input_shape =
            shape_for_feature(model.modelDescription.inputDescriptionsByName, &input_name);
        NSArray<NSNumber *> *output_shape =
            shape_for_feature(model.modelDescription.outputDescriptionsByName, &output_name);
        uint32_t batch = input_shape[0].unsignedIntValue;
        uint32_t input_dim = input_shape[1].unsignedIntValue;
        uint32_t output_dim = output_shape[1].unsignedIntValue;
        NSLog(@"[shape] input=%@ %@ output=%@ %@ trials=%u evict_mib=%u input_mode=%@ seed=%u",
              input_name, input_shape, output_name, output_shape, trials, evict_mib, input_mode, random_state);

        NSUInteger input_words = (NSUInteger)batch * (NSUInteger)input_dim;
        NSUInteger output_words = (NSUInteger)batch * (NSUInteger)output_dim;
        id<MTLBuffer> ane_input = [device newBufferWithLength:input_words * sizeof(uint16_t)
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> mps_input = separate_input ? [device newBufferWithLength:input_words * sizeof(uint16_t)
                                                                       options:MTLResourceStorageModeShared] : ane_input;
        id<MTLBuffer> ane_output = [device newBufferWithLength:output_words * sizeof(uint16_t)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> mps_output = [device newBufferWithLength:output_words * sizeof(uint16_t)
                                                       options:MTLResourceStorageModeShared];
        if (!ane_input || !mps_input || !ane_output || !mps_output) die(@"MTLBuffer allocation failed", nil);
        fill_half_buffer(ane_input, input_words, 0x3c00u);
        if (separate_input) fill_half_buffer(mps_input, input_words, 0x3c00u);
        fill_half_buffer(ane_output, output_words, 0u);
        fill_half_buffer(mps_output, output_words, 0u);

        size_t evict_bytes = (size_t)evict_mib * 1024u * 1024u;
        uint8_t *evict_buffer = evict_bytes ? (uint8_t *)malloc(evict_bytes) : NULL;
        if (evict_bytes && !evict_buffer) die(@"evict buffer allocation failed", nil);
        for (size_t index = 0; index < evict_bytes; ++index) evict_buffer[index] = (uint8_t)index;

        MLMultiArray *input_array = make_buffer_multiarray(ane_input, input_shape, MLMultiArrayDataTypeFloat16);
        MLMultiArray *output_array = make_buffer_multiarray(ane_output, output_shape, MLMultiArrayDataTypeFloat16);
        NSError *error = nil;
        MLFeatureValue *input_value = [MLFeatureValue featureValueWithMultiArray:input_array];
        MLDictionaryFeatureProvider *provider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{input_name : input_value} error:&error];
        if (!provider) die(@"MLDictionaryFeatureProvider failed", error);
        MLPredictionOptions *options = [[MLPredictionOptions alloc] init];
        options.outputBackings = @{output_name : output_array};

        MPSGraphTensor *mps_input_tensor = nil;
        MPSGraphExecutable *mps = make_mps_executable(batch, input_dim, output_dim, &mps_input_tensor);
        MPSGraphTensorData *mps_input_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:mps_input
                                                                                    shape:@[@(batch), @(input_dim)]
                                                                                 dataType:MPSDataTypeFloat16];
        MPSGraphTensorData *mps_output_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:mps_output
                                                                                     shape:@[@(batch), @(output_dim)]
                                                                                  dataType:MPSDataTypeFloat16];
        NSArray *mps_inputs = @[mps_input_data];
        NSArray *mps_outputs = @[mps_output_data];

        for (uint32_t index = 0; index < 3u; ++index) {
            run_ane_once(model, provider, options);
            run_mps_once(mps, queue, mps_inputs, mps_outputs);
        }
        id<MLFeatureProvider> backing_check = [model predictionFromFeatures:provider options:options error:&error];
        if (!backing_check) die(@"CoreML backing check predict failed", error);
        MLMultiArray *actual_output = [backing_check featureValueForName:output_name].multiArrayValue;
        NSLog(@"[warmup] output_backing_used=%@", actual_output == output_array ? @"yes" : @"no");

        NSArray<NSString *> *base_cases = @[
            @"ane_only",
            @"mps_only",
            @"mps_evicted",
            @"mps_after_ane",
            @"mps_after_ane_evicted",
            @"ane_after_mps",
            @"serial_ane_mps",
            @"concurrent"
        ];
        NSMutableDictionary<NSString *, NSMutableArray<NSNumber *> *> *samples = [NSMutableDictionary dictionary];
        NSMutableArray<NSNumber *> *evict_samples = [NSMutableArray array];

        for (uint32_t trial = 0; trial < trials; ++trial) {
            NSMutableArray<NSString *> *trial_cases = [base_cases mutableCopy];
            shuffle_cases(trial_cases, &random_state);
            NSMutableString *order = [NSMutableString string];
            for (NSString *case_name in trial_cases) {
                [order appendFormat:@"%@%@", order.length ? @"," : @"", case_name];
                double start = 0.0;
                if ([case_name isEqualToString:@"ane_only"]) {
                    start = now_seconds();
                    run_ane_once(model, provider, options);
                    record_sample(samples, case_name, (now_seconds() - start) * 1e3);
                } else if ([case_name isEqualToString:@"mps_only"]) {
                    start = now_seconds();
                    run_mps_once(mps, queue, mps_inputs, mps_outputs);
                    record_sample(samples, case_name, (now_seconds() - start) * 1e3);
                } else if ([case_name isEqualToString:@"mps_evicted"]) {
                    double evict_s = evict_cpu_cache(evict_buffer, evict_bytes);
                    [evict_samples addObject:@(evict_s * 1e3)];
                    start = now_seconds();
                    run_mps_once(mps, queue, mps_inputs, mps_outputs);
                    record_sample(samples, case_name, (now_seconds() - start) * 1e3);
                } else if ([case_name isEqualToString:@"mps_after_ane"]) {
                    run_ane_once(model, provider, options);
                    start = now_seconds();
                    run_mps_once(mps, queue, mps_inputs, mps_outputs);
                    record_sample(samples, case_name, (now_seconds() - start) * 1e3);
                } else if ([case_name isEqualToString:@"mps_after_ane_evicted"]) {
                    run_ane_once(model, provider, options);
                    double evict_s = evict_cpu_cache(evict_buffer, evict_bytes);
                    [evict_samples addObject:@(evict_s * 1e3)];
                    start = now_seconds();
                    run_mps_once(mps, queue, mps_inputs, mps_outputs);
                    record_sample(samples, case_name, (now_seconds() - start) * 1e3);
                } else if ([case_name isEqualToString:@"ane_after_mps"]) {
                    run_mps_once(mps, queue, mps_inputs, mps_outputs);
                    start = now_seconds();
                    run_ane_once(model, provider, options);
                    record_sample(samples, case_name, (now_seconds() - start) * 1e3);
                } else if ([case_name isEqualToString:@"serial_ane_mps"]) {
                    start = now_seconds();
                    run_ane_once(model, provider, options);
                    run_mps_once(mps, queue, mps_inputs, mps_outputs);
                    record_sample(samples, case_name, (now_seconds() - start) * 1e3);
                } else if ([case_name isEqualToString:@"concurrent"]) {
                    dispatch_group_t group = dispatch_group_create();
                    start = now_seconds();
                    dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
                        run_ane_once(model, provider, options);
                    });
                    dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
                        run_mps_once(mps, queue, mps_inputs, mps_outputs);
                    });
                    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
                    record_sample(samples, case_name, (now_seconds() - start) * 1e3);
                }
            }
            NSLog(@"[trial] index=%u order=%@", trial, order);
        }

        NSArray<NSString *> *ordered_cases = @[
            @"ane_only",
            @"mps_only",
            @"mps_evicted",
            @"mps_after_ane",
            @"mps_after_ane_evicted",
            @"ane_after_mps",
            @"serial_ane_mps",
            @"concurrent"
        ];
        for (NSString *case_name in ordered_cases) {
            NSDictionary<NSString *, NSNumber *> *summary = summary_for_values(samples[case_name] ?: @[]);
            NSLog(@"[summary] mode=%@ evict_mib=%u case=%@ n=%@ min_ms=%.3f p50_ms=%.3f p90_ms=%.3f max_ms=%.3f mean_ms=%.3f",
                  input_mode,
                  evict_mib,
                  case_name,
                  summary[@"n"],
                  summary[@"min"].doubleValue,
                  summary[@"p50"].doubleValue,
                  summary[@"p90"].doubleValue,
                  summary[@"max"].doubleValue,
                  summary[@"mean"].doubleValue);
        }
        NSDictionary<NSString *, NSNumber *> *evict_summary = summary_for_values(evict_samples);
        double ane_p50 = p50_for_case(samples, @"ane_only");
        double mps_p50 = p50_for_case(samples, @"mps_only");
        double concurrent_p50 = p50_for_case(samples, @"concurrent");
        double serial_measured_p50 = p50_for_case(samples, @"serial_ane_mps");
        double mps_after_ane_p50 = p50_for_case(samples, @"mps_after_ane");
        double mps_after_ane_evicted_p50 = p50_for_case(samples, @"mps_after_ane_evicted");
        double mps_evicted_p50 = p50_for_case(samples, @"mps_evicted");
        NSLog(@"[derived] mode=%@ evict_mib=%u ane_plus_mps_p50_ms=%.3f serial_measured_p50_ms=%.3f concurrent_p50_ms=%.3f overlap_speedup_p50=%.3f",
              input_mode,
              evict_mib,
              ane_p50 + mps_p50,
              serial_measured_p50,
              concurrent_p50,
              concurrent_p50 > 0.0 ? (ane_p50 + mps_p50) / concurrent_p50 : 0.0);
        NSLog(@"[cache_effect] mode=%@ evict_mib=%u mps_only_p50_ms=%.3f mps_evicted_p50_ms=%.3f mps_after_ane_p50_ms=%.3f mps_after_ane_evicted_p50_ms=%.3f evict_p50_ms=%.3f",
              input_mode,
              evict_mib,
              mps_p50,
              mps_evicted_p50,
              mps_after_ane_p50,
              mps_after_ane_evicted_p50,
              evict_summary[@"p50"].doubleValue);

        [output_array getBytesWithHandler:^(const void *bytes, NSInteger size) {
            const uint16_t *half_words = (const uint16_t *)bytes;
            NSInteger words = size / (NSInteger)sizeof(uint16_t);
            NSInteger sample_count = words < 4 ? words : 4;
            NSMutableString *sample = [NSMutableString string];
            for (NSInteger index = 0; index < sample_count; ++index) {
                [sample appendFormat:@"%s0x%04x", index == 0 ? "" : " ", half_words[index]];
            }
            NSLog(@"[ane_output_sample] words=%ld first=%@", (long)words, sample);
        }];

        free(evict_buffer);
    }
    return 0;
}
