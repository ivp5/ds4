#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <mach/mach.h>
#import <mach/mach_time.h>

static double now_seconds(void) {
    static mach_timebase_info_data_t timebase;
    if (timebase.denom == 0) mach_timebase_info(&timebase);
    return (double)mach_absolute_time() * (double)timebase.numer / (double)timebase.denom * 1e-9;
}

static double footprint_mib(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS) return 0.0;
    return (double)info.phys_footprint / (1024.0 * 1024.0);
}

static void die(NSString *message, NSError *error) {
    if (error) {
        fprintf(stderr, "%s: %s\n", message.UTF8String, error.localizedDescription.UTF8String);
    } else {
        fprintf(stderr, "%s\n", message.UTF8String);
    }
    exit(1);
}

static NSString *only_feature_name(NSDictionary<NSString *, MLFeatureDescription *> *descriptions) {
    if (descriptions.count != 1) {
        die([NSString stringWithFormat:@"expected exactly one feature, got %lu", (unsigned long)descriptions.count], nil);
    }
    NSString *name = descriptions.allKeys.firstObject;
    if (!name) die(@"missing feature name", nil);
    return name;
}

static size_t shape_element_count(NSArray<NSNumber *> *shape) {
    if (shape.count == 0) die(@"missing shape", nil);
    size_t elements = 1;
    for (NSNumber *dim in shape) {
        const unsigned long long value = dim.unsignedLongLongValue;
        if (value == 0 || value > SIZE_MAX / elements) die(@"invalid shape", nil);
        elements *= (size_t)value;
    }
    return elements;
}

static MLMultiArray *make_zero_input(NSArray<NSNumber *> *shape) {
    NSError *error = nil;
    MLMultiArray *array = [[MLMultiArray alloc] initWithShape:shape dataType:MLMultiArrayDataTypeFloat16 error:&error];
    if (!array) die(@"failed to allocate input MLMultiArray", error);
    memset(array.dataPointer, 0, shape_element_count(shape) * sizeof(uint16_t));
    return array;
}

static MLModelConfiguration *make_configuration(void) {
    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    MLOptimizationHints *hints = [[MLOptimizationHints alloc] init];
    hints.reshapeFrequency = MLReshapeFrequencyHintInfrequent;
    hints.specializationStrategy = MLSpecializationStrategyFastPrediction;
    config.optimizationHints = hints;
    return config;
}

static MLComputePlan *load_compute_plan(NSURL *compiled_url, MLModelConfiguration *config) {
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
    __block MLComputePlan *loaded_plan = nil;
    __block NSError *loaded_error = nil;
    [MLComputePlan loadContentsOfURL:compiled_url
                        configuration:config
                    completionHandler:^(MLComputePlan * _Nullable computePlan, NSError * _Nullable error) {
        loaded_plan = computePlan;
        loaded_error = error;
        dispatch_semaphore_signal(semaphore);
    }];
    dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
    if (!loaded_plan && loaded_error) NSLog(@"[plan_error] %@", loaded_error.localizedDescription);
    return loaded_plan;
}

static void summarize_compute_plan(MLComputePlan *plan, NSString *path) {
    if (!plan) {
        NSLog(@"[plan] path=%@ unavailable", path);
        return;
    }
    MLModelStructureProgram *program = plan.modelStructure.program;
    if (!program) {
        NSLog(@"[plan] path=%@ type=non_program", path);
        return;
    }
    MLModelStructureProgramFunction *function = program.functions[@"main"] ?: program.functions.allValues.firstObject;
    NSArray<MLModelStructureProgramOperation *> *operations = function.block.operations;
    NSUInteger ane_preferred = 0, gpu_preferred = 0, cpu_preferred = 0, other_preferred = 0;
    NSUInteger ane_supported = 0, unknown_usage = 0;
    NSMutableDictionary<NSString *, NSNumber *> *operators = [NSMutableDictionary dictionary];
    for (MLModelStructureProgramOperation *operation in operations) {
        operators[operation.operatorName] = @((operators[operation.operatorName].unsignedIntegerValue) + 1u);
        MLComputePlanDeviceUsage *usage = [plan computeDeviceUsageForMLProgramOperation:operation];
        if (!usage) {
            unknown_usage++;
            continue;
        }
        NSString *preferred = NSStringFromClass([usage.preferredComputeDevice class]);
        if ([preferred containsString:@"NeuralEngine"]) ane_preferred++;
        else if ([preferred containsString:@"GPU"]) gpu_preferred++;
        else if ([preferred containsString:@"CPU"]) cpu_preferred++;
        else other_preferred++;
        for (id<MLComputeDeviceProtocol> device in usage.supportedComputeDevices) {
            if ([NSStringFromClass([device class]) containsString:@"NeuralEngine"]) {
                ane_supported++;
                break;
            }
        }
    }
    NSArray<NSString *> *operator_names = [operators.allKeys sortedArrayUsingSelector:@selector(compare:)];
    NSMutableString *operator_summary = [NSMutableString string];
    for (NSString *name in operator_names) {
        [operator_summary appendFormat:@"%@%@:%@", operator_summary.length ? @"," : @"", name, operators[name]];
    }
    NSLog(@"[plan] path=%@ ops=%lu ane_preferred=%lu gpu_preferred=%lu cpu_preferred=%lu other_preferred=%lu ane_supported=%lu unknown_usage=%lu operators=%@",
          path,
          (unsigned long)operations.count,
          (unsigned long)ane_preferred,
          (unsigned long)gpu_preferred,
          (unsigned long)cpu_preferred,
          (unsigned long)other_preferred,
          (unsigned long)ane_supported,
          (unsigned long)unknown_usage,
          operator_summary);
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "usage: %s SHARED_L*.mlpackage [...]\n", argv[0]);
            return 2;
        }
        NSMutableArray<MLModel *> *models = [NSMutableArray array];
        double base_footprint = footprint_mib();
        NSLog(@"[start] packages=%d footprint_mib=%.2f", argc - 1, base_footprint);
        for (int index = 1; index < argc; ++index) {
            @autoreleasepool {
                NSString *path = [NSString stringWithUTF8String:argv[index]];
                NSError *error = nil;
                double compile_start = now_seconds();
                NSURL *compiled_url = [MLModel compileModelAtURL:[NSURL fileURLWithPath:path] error:&error];
                if (!compiled_url) die([NSString stringWithFormat:@"compile failed %@", path], error);
                double compile_done = now_seconds();
                MLModelConfiguration *config = make_configuration();
                MLComputePlan *compute_plan = load_compute_plan(compiled_url, config);
                summarize_compute_plan(compute_plan, path);
                double load_start = now_seconds();
                MLModel *model = [MLModel modelWithContentsOfURL:compiled_url configuration:config error:&error];
                if (!model) die([NSString stringWithFormat:@"load failed %@", path], error);
                double load_done = now_seconds();
                NSString *input_name = only_feature_name(model.modelDescription.inputDescriptionsByName);
                NSString *output_name = only_feature_name(model.modelDescription.outputDescriptionsByName);
                NSArray<NSNumber *> *shape = model.modelDescription.inputDescriptionsByName[input_name].multiArrayConstraint.shape;
                MLMultiArray *input = make_zero_input(shape);
                MLDictionaryFeatureProvider *provider =
                    [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{input_name : [MLFeatureValue featureValueWithMultiArray:input]}
                                                                      error:&error];
                if (!provider) die(@"feature provider failed", error);
                double predict_start = now_seconds();
                id<MLFeatureProvider> prediction = [model predictionFromFeatures:provider error:&error];
                if (!prediction) die([NSString stringWithFormat:@"predict failed %@", path], error);
                MLMultiArray *output = [prediction featureValueForName:output_name].multiArrayValue;
                volatile uint16_t first = output ? ((uint16_t *)output.dataPointer)[0] : 0u;
                (void)first;
                double predict_done = now_seconds();
                [models addObject:model];
                double current_footprint = footprint_mib();
                NSLog(@"[model] index=%d path=%@ compile_ms=%.3f load_ms=%.3f predict_ms=%.3f resident_models=%lu footprint_mib=%.2f delta_mib=%.2f",
                      index - 1,
                      path,
                      (compile_done - compile_start) * 1e3,
                      (load_done - load_start) * 1e3,
                      (predict_done - predict_start) * 1e3,
                      (unsigned long)models.count,
                      current_footprint,
                      current_footprint - base_footprint);
            }
        }
        NSLog(@"[done] resident_models=%lu final_footprint_mib=%.2f total_delta_mib=%.2f",
              (unsigned long)models.count,
              footprint_mib(),
              footprint_mib() - base_footprint);
    }
    return 0;
}
