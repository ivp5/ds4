#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

typedef struct BenchStats {
    double total_s;
    double inner_s;
    double evict_s;
} BenchStats;

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
    uint16_t *p = (uint16_t *)buffer.contents;
    for (NSUInteger i = 0; i < half_words; ++i) p[i] = value;
}

static double evict_cpu_cache(uint8_t *buffer, size_t bytes) {
    if (!buffer || bytes == 0) return 0.0;
    volatile uint8_t acc = 0;
    double t0 = now_seconds();
    for (size_t i = 0; i < bytes; i += 64) {
        acc = (uint8_t)(acc + buffer[i]);
        buffer[i] = (uint8_t)(buffer[i] + acc + 1u);
    }
    return now_seconds() - t0;
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

static MPSGraphExecutable *make_mps_executable(id<MTLDevice> device,
                                               uint32_t batch,
                                               uint32_t input_dim,
                                               uint32_t output_dim,
                                               MPSGraphTensor **x_tensor_out,
                                               MPSGraphTensor **y_tensor_out) {
    MPSGraph *graph = [[MPSGraph alloc] init];
    graph.options = MPSGraphOptionsDefault;
    MPSGraphTensor *x = [graph placeholderWithShape:@[@(batch), @(input_dim)]
                                          dataType:MPSDataTypeFloat16
                                              name:@"x"];
    NSMutableData *weight_data =
        [NSMutableData dataWithLength:(NSUInteger)input_dim * (NSUInteger)output_dim * sizeof(uint16_t)];
    uint16_t *weight = (uint16_t *)weight_data.mutableBytes;
    for (NSUInteger i = 0; i < (NSUInteger)input_dim * (NSUInteger)output_dim; ++i) {
        weight[i] = (uint16_t)(0x3400u + (uint16_t)(i & 0x3fu));
    }
    MPSGraphTensor *w = [graph constantWithData:weight_data
                                          shape:@[@(input_dim), @(output_dim)]
                                       dataType:MPSDataTypeFloat16];
    MPSGraphTensor *y = [graph matrixMultiplicationWithPrimaryTensor:x
                                                     secondaryTensor:w
                                                                name:@"mps_dense"];
    MPSGraphShapedType *x_type = [[MPSGraphShapedType alloc] initWithShape:@[@(batch), @(input_dim)]
                                                                   dataType:MPSDataTypeFloat16];
    (void)device;
    MPSGraphExecutable *executable = [graph compileWithDevice:nil
                                                        feeds:@{x : x_type}
                                                targetTensors:@[y]
                                             targetOperations:nil
                                        compilationDescriptor:nil];
    if (!executable) die(@"MPSGraph compile failed", nil);
    executable.options = MPSGraphOptionsNone;
    *x_tensor_out = x;
    *y_tensor_out = y;
    return executable;
}

static void run_ane_once(MLModel *model,
                         MLDictionaryFeatureProvider *provider,
                         MLPredictionOptions *options) {
    NSError *error = nil;
    id<MLFeatureProvider> prediction = [model predictionFromFeatures:provider
                                                             options:options
                                                               error:&error];
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

static BenchStats time_ane(uint32_t rounds,
                           MLModel *model,
                           MLDictionaryFeatureProvider *provider,
                           MLPredictionOptions *options) {
    double t0 = now_seconds();
    for (uint32_t i = 0; i < rounds; ++i) run_ane_once(model, provider, options);
    return (BenchStats){.total_s = now_seconds() - t0, .inner_s = 0.0, .evict_s = 0.0};
}

static BenchStats time_mps(uint32_t rounds,
                           MPSGraphExecutable *executable,
                           id<MTLCommandQueue> queue,
                           NSArray<MPSGraphTensorData *> *inputs,
                           NSArray<MPSGraphTensorData *> *outputs) {
    double t0 = now_seconds();
    for (uint32_t i = 0; i < rounds; ++i) run_mps_once(executable, queue, inputs, outputs);
    return (BenchStats){.total_s = now_seconds() - t0, .inner_s = 0.0, .evict_s = 0.0};
}

static BenchStats time_mps_evicted(uint32_t rounds,
                                   uint8_t *evict_buffer,
                                   size_t evict_bytes,
                                   MPSGraphExecutable *executable,
                                   id<MTLCommandQueue> queue,
                                   NSArray<MPSGraphTensorData *> *inputs,
                                   NSArray<MPSGraphTensorData *> *outputs) {
    double total_t0 = now_seconds();
    double mps_s = 0.0;
    double evict_s = 0.0;
    for (uint32_t i = 0; i < rounds; ++i) {
        evict_s += evict_cpu_cache(evict_buffer, evict_bytes);
        double t0 = now_seconds();
        run_mps_once(executable, queue, inputs, outputs);
        mps_s += now_seconds() - t0;
    }
    return (BenchStats){.total_s = now_seconds() - total_t0, .inner_s = mps_s, .evict_s = evict_s};
}

static BenchStats time_mps_after_ane(uint32_t rounds,
                                     MLModel *model,
                                     MLDictionaryFeatureProvider *provider,
                                     MLPredictionOptions *options,
                                     MPSGraphExecutable *executable,
                                     id<MTLCommandQueue> queue,
                                     NSArray<MPSGraphTensorData *> *inputs,
                                     NSArray<MPSGraphTensorData *> *outputs) {
    double total_t0 = now_seconds();
    double mps_s = 0.0;
    for (uint32_t i = 0; i < rounds; ++i) {
        run_ane_once(model, provider, options);
        double t0 = now_seconds();
        run_mps_once(executable, queue, inputs, outputs);
        mps_s += now_seconds() - t0;
    }
    return (BenchStats){.total_s = now_seconds() - total_t0, .inner_s = mps_s, .evict_s = 0.0};
}

static BenchStats time_mps_after_ane_evicted(uint32_t rounds,
                                             uint8_t *evict_buffer,
                                             size_t evict_bytes,
                                             MLModel *model,
                                             MLDictionaryFeatureProvider *provider,
                                             MLPredictionOptions *options,
                                             MPSGraphExecutable *executable,
                                             id<MTLCommandQueue> queue,
                                             NSArray<MPSGraphTensorData *> *inputs,
                                             NSArray<MPSGraphTensorData *> *outputs) {
    double total_t0 = now_seconds();
    double mps_s = 0.0;
    double evict_s = 0.0;
    for (uint32_t i = 0; i < rounds; ++i) {
        run_ane_once(model, provider, options);
        evict_s += evict_cpu_cache(evict_buffer, evict_bytes);
        double t0 = now_seconds();
        run_mps_once(executable, queue, inputs, outputs);
        mps_s += now_seconds() - t0;
    }
    return (BenchStats){.total_s = now_seconds() - total_t0, .inner_s = mps_s, .evict_s = evict_s};
}

static BenchStats time_ane_after_mps(uint32_t rounds,
                                     MLModel *model,
                                     MLDictionaryFeatureProvider *provider,
                                     MLPredictionOptions *options,
                                     MPSGraphExecutable *executable,
                                     id<MTLCommandQueue> queue,
                                     NSArray<MPSGraphTensorData *> *inputs,
                                     NSArray<MPSGraphTensorData *> *outputs) {
    double total_t0 = now_seconds();
    double ane_s = 0.0;
    for (uint32_t i = 0; i < rounds; ++i) {
        run_mps_once(executable, queue, inputs, outputs);
        double t0 = now_seconds();
        run_ane_once(model, provider, options);
        ane_s += now_seconds() - t0;
    }
    return (BenchStats){.total_s = now_seconds() - total_t0, .inner_s = ane_s, .evict_s = 0.0};
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "usage: %s MODEL.mlpackage [rounds] [evict_mib]\n", argv[0]);
            return 2;
        }
        NSString *model_path = [NSString stringWithUTF8String:argv[1]];
        uint32_t rounds = argc >= 3 ? (uint32_t)strtoul(argv[2], NULL, 10) : 8u;
        uint32_t evict_mib = argc >= 4 ? (uint32_t)strtoul(argv[3], NULL, 10) : 0u;
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
        NSLog(@"[shape] input=%@ %@ output=%@ %@ rounds=%u evict_mib=%u",
              input_name, input_shape, output_name, output_shape, rounds, evict_mib);

        NSUInteger input_words = (NSUInteger)batch * (NSUInteger)input_dim;
        NSUInteger output_words = (NSUInteger)batch * (NSUInteger)output_dim;
        id<MTLBuffer> shared_input = [device newBufferWithLength:input_words * sizeof(uint16_t)
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> ane_output = [device newBufferWithLength:output_words * sizeof(uint16_t)
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> mps_output = [device newBufferWithLength:output_words * sizeof(uint16_t)
                                                      options:MTLResourceStorageModeShared];
        if (!shared_input || !ane_output || !mps_output) die(@"MTLBuffer allocation failed", nil);
        fill_half_buffer(shared_input, input_words, 0x3c00u);
        fill_half_buffer(ane_output, output_words, 0u);
        fill_half_buffer(mps_output, output_words, 0u);
        size_t evict_bytes = (size_t)evict_mib * 1024u * 1024u;
        uint8_t *evict_buffer = evict_bytes ? (uint8_t *)malloc(evict_bytes) : NULL;
        if (evict_bytes && !evict_buffer) die(@"evict buffer allocation failed", nil);
        for (size_t i = 0; i < evict_bytes; ++i) evict_buffer[i] = (uint8_t)i;

        MLMultiArray *input_array = make_buffer_multiarray(shared_input, input_shape, MLMultiArrayDataTypeFloat16);
        MLMultiArray *output_array = make_buffer_multiarray(ane_output, output_shape, MLMultiArrayDataTypeFloat16);
        NSError *error = nil;
        MLFeatureValue *input_value = [MLFeatureValue featureValueWithMultiArray:input_array];
        MLDictionaryFeatureProvider *provider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{input_name : input_value}
                                                              error:&error];
        if (!provider) die(@"MLDictionaryFeatureProvider failed", error);
        MLPredictionOptions *options = [[MLPredictionOptions alloc] init];
        options.outputBackings = @{output_name : output_array};

        MPSGraphTensor *x_tensor = nil;
        MPSGraphTensor *y_tensor = nil;
        MPSGraphExecutable *mps = make_mps_executable(device, batch, input_dim, output_dim, &x_tensor, &y_tensor);
        MPSGraphTensorData *x_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:shared_input
                                                                             shape:@[@(batch), @(input_dim)]
                                                                          dataType:MPSDataTypeFloat16];
        MPSGraphTensorData *y_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:mps_output
                                                                             shape:@[@(batch), @(output_dim)]
                                                                          dataType:MPSDataTypeFloat16];
        NSArray *mps_inputs = @[x_data];
        NSArray *mps_outputs = @[y_data];

        for (uint32_t i = 0; i < 3u; ++i) {
            run_ane_once(model, provider, options);
            run_mps_once(mps, queue, mps_inputs, mps_outputs);
        }
        id<MLFeatureProvider> backing_check = [model predictionFromFeatures:provider options:options error:&error];
        if (!backing_check) die(@"CoreML backing check predict failed", error);
        MLMultiArray *actual_output = [backing_check featureValueForName:output_name].multiArrayValue;
        NSLog(@"[warmup] output_backing_used=%@", actual_output == output_array ? @"yes" : @"no");

        BenchStats ane_only = time_ane(rounds, model, provider, options);
        BenchStats mps_only = time_mps(rounds, mps, queue, mps_inputs, mps_outputs);
        BenchStats mps_evicted = time_mps_evicted(rounds, evict_buffer, evict_bytes,
                                                  mps, queue, mps_inputs, mps_outputs);
        BenchStats mps_after_ane = time_mps_after_ane(rounds, model, provider, options,
                                                      mps, queue, mps_inputs, mps_outputs);
        BenchStats mps_after_ane_evicted = time_mps_after_ane_evicted(rounds, evict_buffer, evict_bytes,
                                                                      model, provider, options,
                                                                      mps, queue, mps_inputs, mps_outputs);
        BenchStats ane_after_mps = time_ane_after_mps(rounds, model, provider, options,
                                                      mps, queue, mps_inputs, mps_outputs);

        __block double ane_thread_s = 0.0;
        __block double mps_thread_s = 0.0;
        dispatch_group_t group = dispatch_group_create();
        double concurrent_t0 = now_seconds();
        dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
            double t0 = now_seconds();
            for (uint32_t i = 0; i < rounds; ++i) run_ane_once(model, provider, options);
            ane_thread_s = now_seconds() - t0;
        });
        dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
            double t0 = now_seconds();
            for (uint32_t i = 0; i < rounds; ++i) run_mps_once(mps, queue, mps_inputs, mps_outputs);
            mps_thread_s = now_seconds() - t0;
        });
        dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
        double concurrent_s = now_seconds() - concurrent_t0;

        double ane_ms = ane_only.total_s * 1e3 / rounds;
        double mps_ms = mps_only.total_s * 1e3 / rounds;
        double mps_evicted_ms = mps_evicted.inner_s * 1e3 / rounds;
        double mps_after_ane_ms = mps_after_ane.inner_s * 1e3 / rounds;
        double mps_after_ane_evicted_ms = mps_after_ane_evicted.inner_s * 1e3 / rounds;
        double ane_after_mps_ms = ane_after_mps.inner_s * 1e3 / rounds;
        double serial_ms = (ane_only.total_s + mps_only.total_s) * 1e3 / rounds;
        double concurrent_ms = concurrent_s * 1e3 / rounds;
        NSLog(@"[result] rounds=%u evict_mib=%u ane_ms=%.3f mps_ms=%.3f mps_evicted_ms=%.3f mps_after_ane_ms=%.3f mps_after_ane_evicted_ms=%.3f ane_after_mps_ms=%.3f serial_ms=%.3f concurrent_ms=%.3f speedup=%.3f ane_thread_ms=%.3f mps_thread_ms=%.3f evict_ms=%.3f",
              rounds,
              evict_mib,
              ane_ms,
              mps_ms,
              mps_evicted_ms,
              mps_after_ane_ms,
              mps_after_ane_evicted_ms,
              ane_after_mps_ms,
              serial_ms,
              concurrent_ms,
              serial_ms / concurrent_ms,
              ane_thread_s * 1e3 / rounds,
              mps_thread_s * 1e3 / rounds,
              (mps_evicted.evict_s + mps_after_ane_evicted.evict_s) * 1e3 / (2.0 * rounds));

        [output_array getBytesWithHandler:^(const void *bytes, NSInteger size) {
            const uint16_t *half_words = (const uint16_t *)bytes;
            NSInteger words = size / (NSInteger)sizeof(uint16_t);
            NSInteger n = words < 4 ? words : 4;
            NSMutableString *sample = [NSMutableString string];
            for (NSInteger i = 0; i < n; ++i) {
                [sample appendFormat:@"%s0x%04x", i == 0 ? "" : " ", half_words[i]];
            }
            NSLog(@"[ane_output_sample] words=%ld first=%@", (long)words, sample);
        }];
        free(evict_buffer);
    }
    return 0;
}
