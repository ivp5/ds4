#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <mach/mach_time.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    kGroups = 256,
    kBlock = 8,
    kInputDim = kGroups * kBlock
};

static double now_seconds(void) {
    static mach_timebase_info_data_t timebase;
    if (timebase.denom == 0) mach_timebase_info(&timebase);
    return (double)mach_absolute_time() * (double)timebase.numer / (double)timebase.denom * 1e-9;
}

static uint16_t f32_to_f16(float value) {
    _Float16 half = (_Float16)value;
    uint16_t bits = 0;
    memcpy(&bits, &half, sizeof(bits));
    return bits;
}

static float f16_to_f32(uint16_t bits) {
    _Float16 half;
    memcpy(&half, &bits, sizeof(half));
    return (float)half;
}

static uint32_t synthetic_code(uint32_t group, uint32_t row, uint32_t code_count) {
    return (uint32_t)((group * 17u + row * 37u + (group >> 2u)) & (code_count - 1u));
}

static MPSGraphTensor *build_lut(MPSGraph *graph,
                                 MPSGraphTensor *x,
                                 MPSGraphTensor *codebook,
                                 MPSGraphTensor *indices,
                                 uint32_t rows,
                                 NSString *name) {
    MPSGraphTensor *reshaped = [graph reshapeTensor:x
                                          withShape:@[@(kGroups), @(kBlock)]
                                               name:[name stringByAppendingString:@"_x"]];
    MPSGraphTensor *table = [graph matrixMultiplicationWithPrimaryTensor:reshaped
                                                         secondaryTensor:codebook
                                                                    name:[name stringByAppendingString:@"_table"]];
    MPSGraphTensor *gathered = [graph gatherAlongAxis:1
                                   withUpdatesTensor:table
                                       indicesTensor:indices
                                                name:[name stringByAppendingString:@"_gather"]];
    MPSGraphTensor *reduced = [graph reductionSumWithTensor:gathered
                                                       axis:0
                                                       name:[name stringByAppendingString:@"_reduce"]];
    return [graph reshapeTensor:reduced withShape:@[@(rows)] name:[name stringByAppendingString:@"_out"]];
}

static MPSGraphTensor *decode_packed_indices(MPSGraph *graph,
                                             MPSGraphTensor *packed,
                                             uint32_t rows,
                                             uint32_t bits) {
    MPSShape *full_shape = @[@(kGroups), @(rows)];
    MPSGraphTensor *row = [graph coordinateAlongAxis:1 withShape:full_shape name:@"packed_row_coord"];
    MPSGraphTensor *one = [graph constantWithScalar:1 dataType:MPSDataTypeInt32];
    MPSGraphTensor *two = [graph constantWithScalar:2 dataType:MPSDataTypeInt32];
    MPSGraphTensor *mask = [graph constantWithScalar:(double)((1u << bits) - 1u) dataType:MPSDataTypeInt32];
    MPSGraphTensor *byte_index = [graph bitwiseRightShiftWithPrimaryTensor:row
                                                           secondaryTensor:one
                                                                      name:@"packed_byte_index"];
    if (bits == 12u) {
        byte_index = [graph additionWithPrimaryTensor:row
                                      secondaryTensor:byte_index
                                                 name:@"packed_12bit_byte_index"];
    }
    MPSGraphTensor *byte_value = [graph gatherAlongAxis:1
                                     withUpdatesTensor:packed
                                         indicesTensor:byte_index
                                                  name:@"packed_byte_gather"];
    MPSGraphTensor *byte_i32 = [graph castTensor:byte_value toType:MPSDataTypeInt32 name:@"packed_byte_i32"];
    MPSGraphTensor *word_i32 = byte_i32;
    if (bits == 12u) {
        MPSGraphTensor *next_byte_index = [graph additionWithPrimaryTensor:byte_index
                                                           secondaryTensor:one
                                                                      name:@"packed_next_byte_index"];
        MPSGraphTensor *next_byte = [graph gatherAlongAxis:1
                                        withUpdatesTensor:packed
                                            indicesTensor:next_byte_index
                                                     name:@"packed_next_byte_gather"];
        MPSGraphTensor *next_i32 = [graph castTensor:next_byte toType:MPSDataTypeInt32 name:@"packed_next_byte_i32"];
        MPSGraphTensor *eight = [graph constantWithScalar:8 dataType:MPSDataTypeInt32];
        MPSGraphTensor *next_shifted = [graph bitwiseLeftShiftWithPrimaryTensor:next_i32
                                                                secondaryTensor:eight
                                                                           name:@"packed_next_shifted"];
        word_i32 = [graph bitwiseORWithPrimaryTensor:byte_i32
                                     secondaryTensor:next_shifted
                                                name:@"packed_word_i32"];
    }
    MPSGraphTensor *low_high = [graph bitwiseANDWithPrimaryTensor:row
                                                  secondaryTensor:one
                                                             name:@"packed_low_high"];
    MPSGraphTensor *shift = [graph bitwiseLeftShiftWithPrimaryTensor:low_high
                                                     secondaryTensor:two
                                                                name:@"packed_shift"];
    MPSGraphTensor *shifted = [graph bitwiseRightShiftWithPrimaryTensor:word_i32
                                                        secondaryTensor:shift
                                                                   name:@"packed_shifted"];
    return [graph bitwiseANDWithPrimaryTensor:shifted secondaryTensor:mask name:@"packed_codes_i32"];
}

static const char *descriptor_mode_name(uint32_t mode) {
    switch (mode) {
        case 1u: return "level0";
        case 2u: return "level1";
        case 3u: return "fastmath";
        case 4u: return "runtime_type_infer";
        case 5u: return "private_shape_cache";
        case 6u: return "private_device_placement";
        case 7u: return "private_ane_flags";
        case 8u: return "private_prefer1";
        case 9u: return "private_prefer2";
        case 10u: return "private_allowed3";
        case 11u: return "private_ane_device";
        case 12u: return "private_compile_resources";
        default: return "default";
    }
}

static uint32_t parse_descriptor_mode(const char *mode) {
    if (!mode || !mode[0] || strcmp(mode, "default") == 0) return 0u;
    if (strcmp(mode, "level0") == 0) return 1u;
    if (strcmp(mode, "level1") == 0) return 2u;
    if (strcmp(mode, "fastmath") == 0) return 3u;
    if (strcmp(mode, "runtime_type_infer") == 0) return 4u;
    if (strcmp(mode, "private_shape_cache") == 0) return 5u;
    if (strcmp(mode, "private_device_placement") == 0) return 6u;
    if (strcmp(mode, "private_ane_flags") == 0) return 7u;
    if (strcmp(mode, "private_prefer1") == 0) return 8u;
    if (strcmp(mode, "private_prefer2") == 0) return 9u;
    if (strcmp(mode, "private_allowed3") == 0) return 10u;
    if (strcmp(mode, "private_ane_device") == 0) return 11u;
    if (strcmp(mode, "private_compile_resources") == 0) return 12u;
    return UINT32_MAX;
}

static bool set_private_value(id object, NSString *key, id value) {
    @try {
        [object setValue:value forKey:key];
        return true;
    } @catch (NSException *exception) {
        fprintf(stderr, "[private_skip] key=%s reason=%s\n", key.UTF8String, exception.reason.UTF8String);
        return false;
    }
}

static void call_private_void(id object, NSString *selector_name) {
    SEL selector = NSSelectorFromString(selector_name);
    if (![object respondsToSelector:selector]) {
        fprintf(stderr, "[private_skip] selector=%s missing\n", selector_name.UTF8String);
        return;
    }
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Warc-performSelector-leaks"
    [object performSelector:selector];
#pragma clang diagnostic pop
}

static MPSGraphDevice *private_ane_device(void) {
    SEL selector = NSSelectorFromString(@"ANEDevice");
    Class cls = [MPSGraphDevice class];
    if (![cls respondsToSelector:selector]) {
        fprintf(stderr, "[private_skip] MPSGraphDevice ANEDevice missing\n");
        return nil;
    }
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Warc-performSelector-leaks"
    return [cls performSelector:selector];
#pragma clang diagnostic pop
}

static MPSGraphCompilationDescriptor *make_compilation_descriptor(uint32_t mode) {
    if (mode == 0u) return nil;
    MPSGraphCompilationDescriptor *descriptor = [[MPSGraphCompilationDescriptor alloc] init];
    descriptor.waitForCompilationCompletion = YES;
    descriptor.optimizationLevel = mode == 1u ? MPSGraphOptimizationLevel0 : MPSGraphOptimizationLevel1;
    if (mode == 3u) descriptor.reducedPrecisionFastMath = MPSGraphReducedPrecisionFastMathAllowFP16Intermediates;
    if (mode == 4u) [descriptor disableTypeInference];
    if (mode == 5u) {
        set_private_value(descriptor, @"enableShapeShifterCache", @YES);
        set_private_value(descriptor, @"shapeShifterCacheThreads", @4u);
    } else if (mode == 6u) {
        call_private_void(descriptor, @"enableDevicePlacement");
    } else if (mode == 7u) {
        set_private_value(descriptor, @"enableANEFWToFWSignal", @YES);
        set_private_value(descriptor, @"enableANELateLatch", @YES);
        set_private_value(descriptor, @"enableANECHWRankPromotion", @YES);
    } else if (mode == 8u) {
        call_private_void(descriptor, @"enableDevicePlacement");
        set_private_value(descriptor, @"preferredDevice", @1u);
        set_private_value(descriptor, @"allowedComputeDevices", @3u);
    } else if (mode == 9u) {
        call_private_void(descriptor, @"enableDevicePlacement");
        set_private_value(descriptor, @"preferredDevice", @2u);
        set_private_value(descriptor, @"allowedComputeDevices", @7u);
    } else if (mode == 10u) {
        call_private_void(descriptor, @"enableDevicePlacement");
        set_private_value(descriptor, @"allowedComputeDevices", @3u);
    } else if (mode == 12u) {
        set_private_value(descriptor, @"enableCompileResourcesForPackage", @YES);
    }
    return descriptor;
}

static MPSGraphExecutable *build_executable(bool packed_path,
                                            uint32_t rows,
                                            uint32_t bits,
                                            uint32_t code_count,
                                            const uint16_t *codebook,
                                            const int32_t *expanded_indices,
                                            const uint8_t *packed_indices,
                                            uint32_t descriptor_mode,
                                            MPSGraph **graph_out,
                                            MPSGraphTensor **x_tensor_out,
                                            MPSGraphTensor **out_tensor_out) {
    MPSGraph *graph = [[MPSGraph alloc] init];
    graph.options = MPSGraphOptionsDefault;
    MPSGraphTensor *x = [graph placeholderWithShape:@[@1, @(kInputDim)]
                                           dataType:MPSDataTypeFloat16
                                               name:packed_path ? @"packed_x" : @"expanded_x"];
    NSData *codebook_data = [NSData dataWithBytes:codebook length:(size_t)kBlock * code_count * sizeof(uint16_t)];
    MPSGraphTensor *codebook_tensor = [graph constantWithData:codebook_data
                                                        shape:@[@(kBlock), @(code_count)]
                                                     dataType:MPSDataTypeFloat16];
    MPSGraphTensor *indices = nil;
    if (packed_path) {
        const uint32_t packed_bytes = (rows * bits + 7u) / 8u + 1u;
        NSData *packed_data = [NSData dataWithBytes:packed_indices length:(size_t)kGroups * packed_bytes * sizeof(uint8_t)];
        MPSGraphTensor *packed_tensor = [graph constantWithData:packed_data
                                                          shape:@[@(kGroups), @(packed_bytes)]
                                                       dataType:MPSDataTypeUInt8];
        indices = decode_packed_indices(graph, packed_tensor, rows, bits);
    } else {
        NSData *index_data = [NSData dataWithBytes:expanded_indices length:(size_t)kGroups * rows * sizeof(int32_t)];
        indices = [graph constantWithData:index_data
                                    shape:@[@(kGroups), @(rows)]
                                 dataType:MPSDataTypeInt32];
    }
    MPSGraphTensor *out = build_lut(graph, x, codebook_tensor, indices, rows, packed_path ? @"packed" : @"expanded");
    MPSGraphShapedType *x_type = [[MPSGraphShapedType alloc] initWithShape:@[@1, @(kInputDim)]
                                                                   dataType:MPSDataTypeFloat16];
    MPSGraphCompilationDescriptor *descriptor = make_compilation_descriptor(descriptor_mode);
    MPSGraphDevice *compile_device = descriptor_mode == 11u ? private_ane_device() : nil;
    MPSGraphExecutable *executable = [graph compileWithDevice:compile_device
                                                        feeds:@{x : x_type}
                                                targetTensors:@[out]
                                             targetOperations:nil
                                        compilationDescriptor:descriptor];
    if (!executable) {
        fprintf(stderr, "compile failed for %s path\n", packed_path ? "packed" : "expanded");
        exit(1);
    }
    executable.options = MPSGraphOptionsNone;
    *graph_out = graph;
    *x_tensor_out = x;
    *out_tensor_out = out;
    return executable;
}

static const char *private_exec_mode(void) {
    const char *mode = getenv("MPSGRAPH_PRIVATE_EXEC_MODE");
    return mode && mode[0] ? mode : "default";
}

static void configure_execution_descriptor(MPSGraphExecutableExecutionDescriptor *descriptor) {
    const char *mode = private_exec_mode();
    if (strcmp(mode, "disable_sync_results") == 0) {
        set_private_value(descriptor, @"disableSynchronizeResults", @YES);
    } else if (strcmp(mode, "disable_ane_cache") == 0) {
        set_private_value(descriptor, @"disableANECaching", @YES);
    } else if (strcmp(mode, "disable_ane_fallback") == 0) {
        set_private_value(descriptor, @"disableANEFallback", @YES);
    } else if (strcmp(mode, "ane_sync") == 0) {
        set_private_value(descriptor, @"encodeANESync", @YES);
    } else if (strcmp(mode, "ane_disable_shared_events") == 0) {
        set_private_value(descriptor, @"encodeANEDisableSharedEvents", @YES);
    } else if (strcmp(mode, "ane_strict_no_cache") == 0) {
        set_private_value(descriptor, @"disableANECaching", @YES);
        set_private_value(descriptor, @"disableANEFallback", @YES);
    }
}

static double time_executable(MPSGraphExecutable *executable,
                              id<MTLCommandQueue> queue,
                              MPSGraphTensorData *x_data,
                              MPSGraphTensorData *out_data,
                              uint32_t rounds) {
    NSArray *inputs = @[x_data];
    NSArray *outputs = @[out_data];
    for (uint32_t warmup = 0; warmup < 3u; ++warmup) {
        [executable runWithMTLCommandQueue:queue inputsArray:inputs resultsArray:outputs executionDescriptor:nil];
    }
    double start = now_seconds();
    for (uint32_t round = 0; round < rounds; ++round) {
        [executable runWithMTLCommandQueue:queue inputsArray:inputs resultsArray:outputs executionDescriptor:nil];
    }
    return (now_seconds() - start) * 1.0e6 / (double)rounds;
}

static double time_executable_async_wait(MPSGraphExecutable *executable,
                                         id<MTLCommandQueue> queue,
                                         MPSGraphTensorData *x_data,
                                         MPSGraphTensorData *out_data,
                                         uint32_t rounds) {
    NSArray *inputs = @[x_data];
    NSArray *outputs = @[out_data];
    MPSGraphExecutableExecutionDescriptor *descriptor = [[MPSGraphExecutableExecutionDescriptor alloc] init];
    descriptor.waitUntilCompleted = YES;
    configure_execution_descriptor(descriptor);
    for (uint32_t warmup = 0; warmup < 3u; ++warmup) {
        [executable runAsyncWithMTLCommandQueue:queue inputsArray:inputs resultsArray:outputs executionDescriptor:descriptor];
    }
    double start = now_seconds();
    for (uint32_t round = 0; round < rounds; ++round) {
        [executable runAsyncWithMTLCommandQueue:queue inputsArray:inputs resultsArray:outputs executionDescriptor:descriptor];
    }
    return (now_seconds() - start) * 1.0e6 / (double)rounds;
}

static double time_executable_async_event_batch(MPSGraphExecutable *executable,
                                                id<MTLCommandQueue> queue,
                                                MPSGraphTensorData *x_data,
                                                MPSGraphTensorData *out_data,
                                                uint32_t rounds) {
    id<MTLSharedEvent> event = [queue.device newSharedEvent];
    if (!event) return -1.0;
    NSArray *inputs = @[x_data];
    NSArray *outputs = @[out_data];
    for (uint32_t warmup = 0; warmup < 3u; ++warmup) {
        MPSGraphExecutableExecutionDescriptor *descriptor = [[MPSGraphExecutableExecutionDescriptor alloc] init];
        configure_execution_descriptor(descriptor);
        [descriptor signalEvent:event atExecutionEvent:MPSGraphExecutionStageCompleted value:warmup + 1u];
        [executable runAsyncWithMTLCommandQueue:queue inputsArray:inputs resultsArray:outputs executionDescriptor:descriptor];
        if (![event waitUntilSignaledValue:warmup + 1u timeoutMS:60000u]) return -1.0;
    }
    event.signaledValue = 0u;
    double start = now_seconds();
    for (uint32_t round = 0; round < rounds; ++round) {
        MPSGraphExecutableExecutionDescriptor *descriptor = [[MPSGraphExecutableExecutionDescriptor alloc] init];
        configure_execution_descriptor(descriptor);
        [descriptor signalEvent:event atExecutionEvent:MPSGraphExecutionStageCompleted value:round + 1u];
        [executable runAsyncWithMTLCommandQueue:queue inputsArray:inputs resultsArray:outputs executionDescriptor:descriptor];
    }
    if (![event waitUntilSignaledValue:rounds timeoutMS:60000u]) return -1.0;
    return (now_seconds() - start) * 1.0e6 / (double)rounds;
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        uint32_t rows = argc >= 2 ? (uint32_t)strtoul(argv[1], NULL, 10) : 4096u;
        uint32_t rounds = argc >= 3 ? (uint32_t)strtoul(argv[2], NULL, 10) : 20u;
        uint32_t bits = argc >= 4 ? (uint32_t)strtoul(argv[3], NULL, 10) : 4u;
        uint32_t descriptor_mode = argc >= 5 ? parse_descriptor_mode(argv[4]) : 0u;
        if (rows == 0u || rows > 8192u) rows = 4096u;
        if (rounds == 0u) rounds = 20u;
        if ((bits != 4u && bits != 12u) || descriptor_mode == UINT32_MAX) {
            fprintf(stderr, "usage: %s [rows] [rounds] [bits=4|12] [default|level0|level1|fastmath|runtime_type_infer|private_shape_cache|private_device_placement|private_ane_flags|private_prefer1|private_prefer2|private_allowed3|private_ane_device|private_compile_resources]\n", argv[0]);
            return 2;
        }
        const uint32_t code_count = 1u << bits;
        const uint32_t packed_bytes = (rows * bits + 7u) / 8u + 1u;
        uint16_t *x_f16 = (uint16_t *)malloc((size_t)kInputDim * sizeof(uint16_t));
        uint16_t *codebook = (uint16_t *)malloc((size_t)kBlock * code_count * sizeof(uint16_t));
        int32_t *expanded_indices = (int32_t *)malloc((size_t)kGroups * rows * sizeof(int32_t));
        uint8_t *packed_indices = (uint8_t *)calloc((size_t)kGroups * packed_bytes, sizeof(uint8_t));
        if (!x_f16 || !codebook || !expanded_indices || !packed_indices) {
            fprintf(stderr, "allocation failed\n");
            return 1;
        }
        for (uint32_t index = 0; index < kInputDim; ++index) {
            float value = 0.31f * sinf((float)index * 0.017f) + 0.13f * cosf((float)index * 0.031f);
            x_f16[index] = f32_to_f16(value);
        }
        for (uint32_t dim = 0; dim < kBlock; ++dim) {
            for (uint32_t code = 0; code < code_count; ++code) {
                float value = 0.19f * sinf((float)(dim * 17u + code * 5u) * 0.071f);
                codebook[(size_t)dim * code_count + code] = f32_to_f16(value);
            }
        }
        for (uint32_t group = 0; group < kGroups; ++group) {
            for (uint32_t row = 0; row < rows; ++row) {
                uint32_t code = synthetic_code(group, row, code_count);
                expanded_indices[(size_t)group * rows + row] = (int32_t)code;
                uint32_t bit_offset = row * bits;
                uint32_t byte_index = bit_offset >> 3u;
                uint32_t shift = bit_offset & 7u;
                uint32_t encoded = code << shift;
                uint8_t *slot = packed_indices + (size_t)group * packed_bytes + byte_index;
                slot[0] = (uint8_t)(slot[0] | (uint8_t)(encoded & 0xffu));
                slot[1] = (uint8_t)(slot[1] | (uint8_t)((encoded >> 8u) & 0xffu));
                if (bits == 12u) slot[2] = (uint8_t)(slot[2] | (uint8_t)((encoded >> 16u) & 0xffu));
            }
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = device ? [device newCommandQueue] : nil;
        if (!device || !queue) {
            fprintf(stderr, "missing Metal device/queue\n");
            return 1;
        }
        id<MTLBuffer> x_buffer = [device newBufferWithBytes:x_f16
                                                     length:(size_t)kInputDim * sizeof(uint16_t)
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> expanded_out = [device newBufferWithLength:(size_t)rows * sizeof(uint16_t)
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> packed_out = [device newBufferWithLength:(size_t)rows * sizeof(uint16_t)
                                                       options:MTLResourceStorageModeShared];
        MPSGraphTensorData *x_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:x_buffer
                                                                             shape:@[@1, @(kInputDim)]
                                                                          dataType:MPSDataTypeFloat16];
        MPSGraphTensorData *expanded_out_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:expanded_out
                                                                                        shape:@[@(rows)]
                                                                                     dataType:MPSDataTypeFloat16];
        MPSGraphTensorData *packed_out_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:packed_out
                                                                                      shape:@[@(rows)]
                                                                                   dataType:MPSDataTypeFloat16];
        MPSGraph *expanded_graph = nil;
        MPSGraphTensor *expanded_x = nil;
        MPSGraphTensor *expanded_tensor = nil;
        double compile_start = now_seconds();
        MPSGraphExecutable *expanded_exec = build_executable(false, rows, bits, code_count, codebook, expanded_indices, packed_indices,
                                                             descriptor_mode, &expanded_graph, &expanded_x, &expanded_tensor);
        double expanded_compile_ms = (now_seconds() - compile_start) * 1.0e3;
        MPSGraph *packed_graph = nil;
        MPSGraphTensor *packed_x = nil;
        MPSGraphTensor *packed_tensor = nil;
        compile_start = now_seconds();
        MPSGraphExecutable *packed_exec = build_executable(true, rows, bits, code_count, codebook, expanded_indices, packed_indices,
                                                           descriptor_mode, &packed_graph, &packed_x, &packed_tensor);
        double packed_compile_ms = (now_seconds() - compile_start) * 1.0e3;
        (void)expanded_graph; (void)expanded_x; (void)expanded_tensor;
        (void)packed_graph; (void)packed_x; (void)packed_tensor;
        [expanded_exec runWithMTLCommandQueue:queue inputsArray:@[x_data] resultsArray:@[expanded_out_data] executionDescriptor:nil];
        [packed_exec runWithMTLCommandQueue:queue inputsArray:@[x_data] resultsArray:@[packed_out_data] executionDescriptor:nil];
        uint16_t *expanded_words = (uint16_t *)expanded_out.contents;
        uint16_t *packed_words = (uint16_t *)packed_out.contents;
        double max_abs = 0.0;
        double rms = 0.0;
        uint32_t bad = 0;
        for (uint32_t row = 0; row < rows; ++row) {
            double diff = fabs((double)f16_to_f32(expanded_words[row]) - (double)f16_to_f32(packed_words[row]));
            if (diff > max_abs) max_abs = diff;
            rms += diff * diff;
            if (diff > 0.0625) ++bad;
        }
        rms = sqrt(rms / (double)rows);
        double expanded_us = time_executable(expanded_exec, queue, x_data, expanded_out_data, rounds);
        double packed_us = time_executable(packed_exec, queue, x_data, packed_out_data, rounds);
        double expanded_async_wait_us = time_executable_async_wait(expanded_exec, queue, x_data, expanded_out_data, rounds);
        double packed_async_wait_us = time_executable_async_wait(packed_exec, queue, x_data, packed_out_data, rounds);
        double expanded_async_batch_us = time_executable_async_event_batch(expanded_exec, queue, x_data, expanded_out_data, rounds);
        double packed_async_batch_us = time_executable_async_event_batch(packed_exec, queue, x_data, packed_out_data, rounds);
        double expanded_index_mb = (double)((size_t)kGroups * rows * sizeof(int32_t)) / 1.0e6;
        double packed_index_mb = (double)((size_t)kGroups * packed_bytes * sizeof(uint8_t)) / 1.0e6;
        fprintf(stderr,
                "mpsgraph_packed_index_probe: rows=%u rounds=%u bits=%u descriptor=%s exec_private=%s groups=%u k=%u expanded_compile_ms=%.3f packed_compile_ms=%.3f expanded_us=%.3f packed_us=%.3f expanded_async_wait_us=%.3f packed_async_wait_us=%.3f expanded_async_batch_us=%.3f packed_async_batch_us=%.3f speedup=%.3f expanded_index_MB=%.3f packed_index_MB=%.3f index_shrink=%.3f bad=%u max_abs=%.6g rms=%.6g sample_exp=%.6g sample_pack=%.6g\n",
                rows, rounds, bits, descriptor_mode_name(descriptor_mode), private_exec_mode(), kGroups, code_count,
                expanded_compile_ms, packed_compile_ms,
                expanded_us, packed_us,
                expanded_async_wait_us, packed_async_wait_us,
                expanded_async_batch_us, packed_async_batch_us,
                packed_us > 0.0 ? expanded_us / packed_us : 0.0,
                expanded_index_mb, packed_index_mb,
                packed_index_mb > 0.0 ? expanded_index_mb / packed_index_mb : 0.0,
                bad, max_abs, rms,
                (double)f16_to_f32(expanded_words[0]), (double)f16_to_f32(packed_words[0]));
        free(packed_indices);
        free(expanded_indices);
        free(codebook);
        free(x_f16);
    }
    return 0;
}
