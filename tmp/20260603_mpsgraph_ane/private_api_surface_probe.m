#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <objc/runtime.h>

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

static bool keep_selector(const char *name) {
    if (!name) return false;
    const char *needles[] = {
        "_", "ane", "ANE", "neural", "Neural", "engine", "Engine",
        "device", "Device", "compile", "Compile", "special", "Special",
        "priority", "Priority", "placement", "Placement", "schedule", "Schedule",
        "sync", "Sync", "event", "Event", "queue", "Queue", "tensor", "Tensor",
        "cache", "Cache", "low", "Low", "fast", "Fast", "precision", "Precision",
        "memory", "Memory", "backing", "Backing", "plan", "Plan"
    };
    for (size_t index = 0; index < sizeof(needles) / sizeof(needles[0]); ++index) {
        if (strstr(name, needles[index])) return true;
    }
    return false;
}

static void print_methods(Class cls, bool meta, const char *prefix) {
    unsigned int count = 0;
    Method *methods = class_copyMethodList(meta ? object_getClass(cls) : cls, &count);
    for (unsigned int index = 0; index < count; ++index) {
        SEL selector = method_getName(methods[index]);
        const char *name = sel_getName(selector);
        if (keep_selector(name)) {
            printf("[method] class=%s kind=%s selector=%s types=%s\n",
                   prefix,
                   meta ? "class" : "instance",
                   name,
                   method_getTypeEncoding(methods[index]));
        }
    }
    free(methods);
}

static void print_properties(Class cls, const char *prefix) {
    unsigned int count = 0;
    objc_property_t *properties = class_copyPropertyList(cls, &count);
    for (unsigned int index = 0; index < count; ++index) {
        const char *name = property_getName(properties[index]);
        if (keep_selector(name)) {
            printf("[property] class=%s name=%s attrs=%s\n",
                   prefix,
                   name,
                   property_getAttributes(properties[index]));
        }
    }
    free(properties);
}

static void print_class(const char *name) {
    Class cls = NSClassFromString([NSString stringWithUTF8String:name]);
    if (!cls) {
        printf("[missing] class=%s\n", name);
        return;
    }
    printf("[class] name=%s superclass=%s\n", name, class_getName(class_getSuperclass(cls)));
    print_properties(cls, name);
    print_methods(cls, false, name);
    print_methods(cls, true, name);
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        const char *defaults[] = {
            "MPSGraph",
            "MPSGraphCompilationDescriptor",
            "MPSGraphExecutionDescriptor",
            "MPSGraphExecutable",
            "MPSGraphExecutableExecutionDescriptor",
            "MPSGraphDevice",
            "MPSGraphTensorData",
            "MLModel",
            "MLModelConfiguration",
            "MLOptimizationHints",
            "MLPredictionOptions",
            "MLComputePlan",
            "MLComputePlanDeviceUsage",
            "MLComputePlanCost",
            "MLNeuralEngineComputeDevice",
            "MLGPUComputeDevice",
            "MLCPUComputeDevice",
            "MLState"
        };
        if (argc > 1) {
            for (int index = 1; index < argc; ++index) print_class(argv[index]);
        } else {
            for (size_t index = 0; index < sizeof(defaults) / sizeof(defaults[0]); ++index) {
                print_class(defaults[index]);
            }
        }
    }
    return 0;
}
