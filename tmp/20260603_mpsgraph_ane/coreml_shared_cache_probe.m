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
                MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
                config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
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
