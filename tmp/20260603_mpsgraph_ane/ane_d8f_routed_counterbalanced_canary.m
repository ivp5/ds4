#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <dispatch/dispatch.h>
#import <mach/mach_time.h>

#include "ds4_d8f_reader.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    kBlock = 8,
    kHiddenDim = 4096,
    kMidDim = 2048,
    kOutDim = 4096,
    kMaxExperts = 6,
};

typedef struct D8FSlot {
    ds4_d8f_record gate;
    ds4_d8f_record up;
    ds4_d8f_record down;
    uint16_t *gate_cb_f16;
    uint16_t *up_cb_f16;
    uint16_t *down_cb_f16;
    float *gate_cb_f32;
    float *up_cb_f32;
    float *down_cb_f32;
    int32_t *gate_idx;
    int32_t *up_idx;
    int32_t *down_idx;
} D8FSlot;

static double now_seconds(void) {
    static mach_timebase_info_data_t timebase;
    if (timebase.denom == 0) mach_timebase_info(&timebase);
    return (double)mach_absolute_time() * (double)timebase.numer / (double)timebase.denom * 1e-9;
}

static void die(NSString *message, NSError *error) {
    if (error) {
        fprintf(stderr, "%s: %s\n", message.UTF8String, error.localizedDescription.UTF8String);
    } else {
        fprintf(stderr, "%s\n", message.UTF8String);
    }
    exit(1);
}

static uint16_t read_u16(const uint8_t *data) {
    return (uint16_t)data[0] | (uint16_t)((uint16_t)data[1] << 8);
}

static float f16_to_f32(uint16_t bits) {
    _Float16 value;
    memcpy(&value, &bits, sizeof(value));
    return (float)value;
}

static uint16_t f32_to_f16(float value) {
    _Float16 half = (_Float16)value;
    uint16_t bits = 0;
    memcpy(&bits, &half, sizeof(bits));
    return bits;
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

static MLModel *load_coreml_model(NSString *path, id<MTLDevice> device) {
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

static void run_ane_once(MLModel *model,
                         MLDictionaryFeatureProvider *provider,
                         MLPredictionOptions *options) {
    NSError *error = nil;
    id<MLFeatureProvider> prediction = [model predictionFromFeatures:provider options:options error:&error];
    if (!prediction) die(@"CoreML prediction failed", error);
}

static void run_d8f_once(MPSGraphExecutable *executable,
                         id<MTLCommandQueue> queue,
                         NSArray<MPSGraphTensorData *> *inputs,
                         NSArray<MPSGraphTensorData *> *outputs) {
    [executable runWithMTLCommandQueue:queue
                           inputsArray:inputs
                          resultsArray:outputs
                   executionDescriptor:nil];
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

static id<MTLComputePipelineState> make_merge_pipeline(id<MTLDevice> device) {
    NSString *source =
        @"#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "kernel void merge_first_row(const device half *shared [[buffer(0)]],\n"
         "                            const device half *routed [[buffer(1)]],\n"
         "                            device half *out [[buffer(2)]],\n"
         "                            uint gid [[thread_position_in_grid]]) {\n"
         "  if (gid >= 4096) return;\n"
         "  out[gid] = shared[gid] + routed[gid];\n"
         "}\n";
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (!library) die(@"merge newLibraryWithSource failed", error);
    id<MTLFunction> function = [library newFunctionWithName:@"merge_first_row"];
    if (!function) die(@"merge function missing", nil);
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) die(@"merge pipeline failed", error);
    return pipeline;
}

static void run_merge_once(id<MTLCommandQueue> queue,
                           id<MTLComputePipelineState> pipeline,
                           id<MTLBuffer> shared_output,
                           id<MTLBuffer> routed_output,
                           id<MTLBuffer> merged_output) {
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:shared_output offset:0 atIndex:0];
    [encoder setBuffer:routed_output offset:0 atIndex:1];
    [encoder setBuffer:merged_output offset:0 atIndex:2];
    NSUInteger threads = pipeline.maxTotalThreadsPerThreadgroup < 256 ? pipeline.maxTotalThreadsPerThreadgroup : 256;
    if (threads == 0) threads = 1;
    MTLSize threads_per_group = MTLSizeMake(threads, 1, 1);
    MTLSize groups = MTLSizeMake((kOutDim + threads - 1) / threads, 1, 1);
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads_per_group];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
}

static bool prepare_codebook(const ds4_d8f_file *file,
                             const ds4_d8f_record *record,
                             uint32_t codebook_cols,
                             float *codebook_f32,
                             uint16_t *codebook_f16) {
    if (!file || !record || !codebook_f32 || !codebook_f16) return false;
    if (record->block != kBlock || record->k == 0u || codebook_cols <= record->k) return false;
    if (record->codebook_bytes < (uint64_t)record->k * kBlock * 2u) return false;
    const uint8_t *source = file->map + record->codebook_offset;
    for (uint32_t code = 0; code < record->k; ++code) {
        for (uint32_t dim = 0; dim < kBlock; ++dim) {
            uint16_t bits = read_u16(source + ((uint64_t)code * kBlock + dim) * 2u);
            codebook_f16[(uint64_t)dim * codebook_cols + code] = bits;
            codebook_f32[(uint64_t)dim * codebook_cols + code] = f16_to_f32(bits);
        }
    }
    for (uint32_t dim = 0; dim < kBlock; ++dim) {
        codebook_f16[(uint64_t)dim * codebook_cols + record->k] = 0u;
        codebook_f32[(uint64_t)dim * codebook_cols + record->k] = 0.0f;
    }
    return true;
}

static bool prepare_indices(const ds4_d8f_file *file,
                            const ds4_d8f_record *record,
                            uint32_t rows,
                            uint32_t in_dim,
                            int32_t *indices) {
    if (!file || !record || !indices || in_dim == 0u || (in_dim & 7u) != 0u) return false;
    uint32_t groups = in_dim / kBlock;
    uint64_t needed_blocks = (uint64_t)rows * groups;
    uint64_t available_blocks = ((uint64_t)record->index_bytes * 8ull) / (uint64_t)record->bits;
    if (record->block != kBlock || record->k == 0u || available_blocks < needed_blocks) return false;
    for (uint32_t group = 0; group < groups; ++group) {
        for (uint32_t row = 0; row < rows; ++row) {
            uint64_t block_index = (uint64_t)row * groups + group;
            uint32_t code = ds4_d8f_code_at(file, record, block_index);
            if (code >= record->k) code = record->k;
            indices[(uint64_t)group * rows + row] = (int32_t)code;
        }
    }
    return true;
}

static void reference_lut_accum(const float *input,
                                const float *codebook,
                                const int32_t *indices,
                                uint32_t codebook_cols,
                                uint32_t in_dim,
                                uint32_t rows,
                                float *out) {
    uint32_t groups = in_dim / kBlock;
    for (uint32_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (uint32_t group = 0; group < groups; ++group) {
            int32_t code = indices[(uint64_t)group * rows + row];
            if (code < 0 || (uint32_t)code >= codebook_cols) continue;
            const float *xg = input + (uint64_t)group * kBlock;
            const float *cb = codebook + (uint64_t)code;
            for (uint32_t dim = 0; dim < kBlock; ++dim) {
                sum += (double)xg[dim] * (double)cb[(uint64_t)dim * codebook_cols];
            }
        }
        out[row] += (float)sum;
    }
}

static void reference_lut(const float *input,
                          const float *codebook,
                          const int32_t *indices,
                          uint32_t codebook_cols,
                          uint32_t in_dim,
                          uint32_t rows,
                          float *out) {
    memset(out, 0, (size_t)rows * sizeof(float));
    reference_lut_accum(input, codebook, indices, codebook_cols, in_dim, rows, out);
}

static MPSGraphTensor *build_lut(MPSGraph *graph,
                                 MPSGraphTensor *input,
                                 MPSGraphTensor *codebook,
                                 MPSGraphTensor *indices,
                                 uint32_t in_dim,
                                 uint32_t rows,
                                 NSString *name) {
    uint32_t groups = in_dim / kBlock;
    MPSGraphTensor *reshaped = [graph reshapeTensor:input
                                         withShape:@[@(groups), @(kBlock)]
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

static uint32_t parse_experts(const char *csv, uint32_t *experts, uint32_t max_experts) {
    if (!csv || !experts || max_experts == 0u) return 0u;
    char local[128];
    strlcpy(local, csv, sizeof(local));
    uint32_t count = 0;
    char *save = NULL;
    for (char *token = strtok_r(local, ",", &save); token && count < max_experts; token = strtok_r(NULL, ",", &save)) {
        char *end = NULL;
        unsigned long value = strtoul(token, &end, 10);
        if (end == token || *end != '\0' || value >= 256ul) return 0u;
        experts[count++] = (uint32_t)value;
    }
    return count;
}

static void free_slots(D8FSlot *slots, uint32_t n_experts) {
    for (uint32_t slot = 0; slot < n_experts; ++slot) {
        free(slots[slot].gate_cb_f16);
        free(slots[slot].up_cb_f16);
        free(slots[slot].down_cb_f16);
        free(slots[slot].gate_cb_f32);
        free(slots[slot].up_cb_f32);
        free(slots[slot].down_cb_f32);
        free(slots[slot].gate_idx);
        free(slots[slot].up_idx);
        free(slots[slot].down_idx);
    }
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
    return summary_for_values(samples[name] ?: @[])[@"p50"].doubleValue;
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc < 4) {
            fprintf(stderr, "usage: %s SHARED.mlpackage LAYER.d8f EXPERTS_CSV [trials] [evict_mib] [same|separate] [seed]\n", argv[0]);
            return 2;
        }
        NSString *model_path = [NSString stringWithUTF8String:argv[1]];
        const char *d8f_path = argv[2];
        uint32_t experts[kMaxExperts] = {0};
        uint32_t n_experts = parse_experts(argv[3], experts, kMaxExperts);
        if (n_experts == 0u) die(@"bad experts CSV", nil);
        uint32_t trials = argc >= 5 ? (uint32_t)strtoul(argv[4], NULL, 10) : 8u;
        uint32_t evict_mib = argc >= 6 ? (uint32_t)strtoul(argv[5], NULL, 10) : 0u;
        NSString *input_mode = argc >= 7 ? [NSString stringWithUTF8String:argv[6]] : @"same";
        uint32_t random_state = argc >= 8 ? (uint32_t)strtoul(argv[7], NULL, 10) : 0x44384631u;
        BOOL separate_input = [input_mode isEqualToString:@"separate"];
        if (![input_mode isEqualToString:@"same"] && !separate_input) die(@"input mode must be same or separate", nil);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) die(@"MTLCreateSystemDefaultDevice failed", nil);
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) die(@"newCommandQueue failed", nil);

        MLModel *model = load_coreml_model(model_path, device);
        NSString *input_name = nil;
        NSString *output_name = nil;
        NSArray<NSNumber *> *input_shape = shape_for_feature(model.modelDescription.inputDescriptionsByName, &input_name);
        NSArray<NSNumber *> *output_shape = shape_for_feature(model.modelDescription.outputDescriptionsByName, &output_name);
        uint32_t batch = input_shape[0].unsignedIntValue;
        uint32_t model_input_dim = input_shape[1].unsignedIntValue;
        uint32_t model_output_dim = output_shape[1].unsignedIntValue;
        if (model_input_dim != kHiddenDim || model_output_dim != kOutDim) die(@"CoreML shared model shape is not DS4 hidden->hidden", nil);
        NSLog(@"[shape] coreml_input=%@ %@ coreml_output=%@ %@ d8f=%s experts=%s nsel=%u trials=%u evict_mib=%u mode=%@ seed=%u",
              input_name, input_shape, output_name, output_shape, d8f_path, argv[3], n_experts, trials, evict_mib, input_mode, random_state);

        ds4_d8f_file file;
        if (!ds4_d8f_open(d8f_path, &file)) die(@"ds4_d8f_open failed", nil);
        D8FSlot slots[kMaxExperts];
        memset(slots, 0, sizeof(slots));
        bool ok = true;
        size_t gate_idx_bytes = (size_t)(kHiddenDim / kBlock) * kMidDim * sizeof(int32_t);
        size_t down_idx_bytes = (size_t)(kMidDim / kBlock) * kOutDim * sizeof(int32_t);
        for (uint32_t slot = 0; ok && slot < n_experts; ++slot) {
            ok = ds4_d8f_get_record(&file, DS4_D8F_GATE, experts[slot], &slots[slot].gate) &&
                 ds4_d8f_get_record(&file, DS4_D8F_UP, experts[slot], &slots[slot].up) &&
                 ds4_d8f_get_record(&file, DS4_D8F_DOWN, experts[slot], &slots[slot].down);
            if (!ok) break;
            uint32_t gate_cols = slots[slot].gate.k + 1u;
            uint32_t up_cols = slots[slot].up.k + 1u;
            uint32_t down_cols = slots[slot].down.k + 1u;
            slots[slot].gate_cb_f16 = (uint16_t *)calloc((size_t)kBlock * gate_cols, sizeof(uint16_t));
            slots[slot].up_cb_f16 = (uint16_t *)calloc((size_t)kBlock * up_cols, sizeof(uint16_t));
            slots[slot].down_cb_f16 = (uint16_t *)calloc((size_t)kBlock * down_cols, sizeof(uint16_t));
            slots[slot].gate_cb_f32 = (float *)calloc((size_t)kBlock * gate_cols, sizeof(float));
            slots[slot].up_cb_f32 = (float *)calloc((size_t)kBlock * up_cols, sizeof(float));
            slots[slot].down_cb_f32 = (float *)calloc((size_t)kBlock * down_cols, sizeof(float));
            slots[slot].gate_idx = (int32_t *)malloc(gate_idx_bytes);
            slots[slot].up_idx = (int32_t *)malloc(gate_idx_bytes);
            slots[slot].down_idx = (int32_t *)malloc(down_idx_bytes);
            ok = slots[slot].gate_cb_f16 && slots[slot].up_cb_f16 && slots[slot].down_cb_f16 &&
                 slots[slot].gate_cb_f32 && slots[slot].up_cb_f32 && slots[slot].down_cb_f32 &&
                 slots[slot].gate_idx && slots[slot].up_idx && slots[slot].down_idx &&
                 prepare_codebook(&file, &slots[slot].gate, gate_cols, slots[slot].gate_cb_f32, slots[slot].gate_cb_f16) &&
                 prepare_codebook(&file, &slots[slot].up, up_cols, slots[slot].up_cb_f32, slots[slot].up_cb_f16) &&
                 prepare_codebook(&file, &slots[slot].down, down_cols, slots[slot].down_cb_f32, slots[slot].down_cb_f16) &&
                 prepare_indices(&file, &slots[slot].gate, kMidDim, kHiddenDim, slots[slot].gate_idx) &&
                 prepare_indices(&file, &slots[slot].up, kMidDim, kHiddenDim, slots[slot].up_idx) &&
                 prepare_indices(&file, &slots[slot].down, kOutDim, kMidDim, slots[slot].down_idx);
        }
        if (!ok) {
            free_slots(slots, n_experts);
            ds4_d8f_close(&file);
            die(@"failed to prepare D8F slots", nil);
        }

        float *x_f32 = (float *)malloc((size_t)kHiddenDim * sizeof(float));
        uint16_t *x_f16 = (uint16_t *)malloc((size_t)kHiddenDim * sizeof(uint16_t));
        float *gate_tmp = (float *)malloc((size_t)kMidDim * sizeof(float));
        float *up_tmp = (float *)malloc((size_t)kMidDim * sizeof(float));
        float *mid_tmp = (float *)malloc((size_t)kMidDim * sizeof(float));
        float *ref = (float *)calloc(kOutDim, sizeof(float));
        float *got = (float *)calloc(kOutDim, sizeof(float));
        if (!x_f32 || !x_f16 || !gate_tmp || !up_tmp || !mid_tmp || !ref || !got) die(@"host allocation failed", nil);
        for (uint32_t index = 0; index < kHiddenDim; ++index) {
            float value = 0.35f * sinf((float)index * 0.011f) + 0.17f * cosf((float)index * 0.019f);
            x_f32[index] = value;
            x_f16[index] = f32_to_f16(value);
        }
        for (uint32_t slot = 0; slot < n_experts; ++slot) {
            uint32_t gate_cols = slots[slot].gate.k + 1u;
            uint32_t up_cols = slots[slot].up.k + 1u;
            uint32_t down_cols = slots[slot].down.k + 1u;
            reference_lut(x_f32, slots[slot].gate_cb_f32, slots[slot].gate_idx, gate_cols, kHiddenDim, kMidDim, gate_tmp);
            reference_lut(x_f32, slots[slot].up_cb_f32, slots[slot].up_idx, up_cols, kHiddenDim, kMidDim, up_tmp);
            for (uint32_t row = 0; row < kMidDim; ++row) {
                float gate = gate_tmp[row] > 10.0f ? 10.0f : gate_tmp[row];
                float up = up_tmp[row] > 10.0f ? 10.0f : up_tmp[row];
                if (up < -10.0f) up = -10.0f;
                mid_tmp[row] = (gate / (1.0f + expf(-gate))) * up;
            }
            reference_lut_accum(mid_tmp, slots[slot].down_cb_f32, slots[slot].down_idx, down_cols, kMidDim, kOutDim, ref);
        }

        id<MTLBuffer> ane_input = [device newBufferWithLength:(NSUInteger)batch * kHiddenDim * sizeof(uint16_t)
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> mps_input = separate_input ? [device newBufferWithLength:kHiddenDim * sizeof(uint16_t)
                                                                        options:MTLResourceStorageModeShared] : ane_input;
        id<MTLBuffer> ane_output = [device newBufferWithLength:(NSUInteger)batch * kOutDim * sizeof(uint16_t)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> d8f_output = [device newBufferWithLength:kOutDim * sizeof(uint16_t)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> merged_output = [device newBufferWithLength:kOutDim * sizeof(uint16_t)
                                                          options:MTLResourceStorageModeShared];
        if (!ane_input || !mps_input || !ane_output || !d8f_output || !merged_output) die(@"MTLBuffer allocation failed", nil);
        uint16_t *ane_words = (uint16_t *)ane_input.contents;
        for (uint32_t row = 0; row < batch; ++row) {
            memcpy(ane_words + (uint64_t)row * kHiddenDim, x_f16, (size_t)kHiddenDim * sizeof(uint16_t));
        }
        if (separate_input) memcpy(mps_input.contents, x_f16, (size_t)kHiddenDim * sizeof(uint16_t));
        memset(ane_output.contents, 0, (NSUInteger)batch * kOutDim * sizeof(uint16_t));
        memset(d8f_output.contents, 0, kOutDim * sizeof(uint16_t));
        memset(merged_output.contents, 0, kOutDim * sizeof(uint16_t));
        id<MTLComputePipelineState> merge_pipeline = make_merge_pipeline(device);

        MLMultiArray *ane_input_array = make_buffer_multiarray(ane_input, input_shape, MLMultiArrayDataTypeFloat16);
        MLMultiArray *ane_output_array = make_buffer_multiarray(ane_output, output_shape, MLMultiArrayDataTypeFloat16);
        NSError *error = nil;
        MLFeatureValue *input_value = [MLFeatureValue featureValueWithMultiArray:ane_input_array];
        MLDictionaryFeatureProvider *provider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{input_name : input_value} error:&error];
        if (!provider) die(@"MLDictionaryFeatureProvider failed", error);
        MLPredictionOptions *options = [[MLPredictionOptions alloc] init];
        options.outputBackings = @{output_name : ane_output_array};

        MPSGraph *graph = [[MPSGraph alloc] init];
        graph.options = MPSGraphOptionsDefault;
        MPSDataType data_type = MPSDataTypeFloat16;
        MPSGraphTensor *x = [graph placeholderWithShape:@[@1, @(kHiddenDim)] dataType:data_type name:@"d8f_hidden"];
        MPSGraphTensor *limit = [graph constantWithScalar:10.0 dataType:data_type];
        MPSGraphTensor *neg_limit = [graph constantWithScalar:-10.0 dataType:data_type];
        MPSGraphTensor *total = nil;
        for (uint32_t slot = 0; slot < n_experts; ++slot) {
            uint32_t gate_cols = slots[slot].gate.k + 1u;
            uint32_t up_cols = slots[slot].up.k + 1u;
            uint32_t down_cols = slots[slot].down.k + 1u;
            NSData *gate_cb_data = [NSData dataWithBytes:slots[slot].gate_cb_f16 length:(size_t)kBlock * gate_cols * sizeof(uint16_t)];
            NSData *up_cb_data = [NSData dataWithBytes:slots[slot].up_cb_f16 length:(size_t)kBlock * up_cols * sizeof(uint16_t)];
            NSData *down_cb_data = [NSData dataWithBytes:slots[slot].down_cb_f16 length:(size_t)kBlock * down_cols * sizeof(uint16_t)];
            NSData *gate_idx_data = [NSData dataWithBytes:slots[slot].gate_idx length:gate_idx_bytes];
            NSData *up_idx_data = [NSData dataWithBytes:slots[slot].up_idx length:gate_idx_bytes];
            NSData *down_idx_data = [NSData dataWithBytes:slots[slot].down_idx length:down_idx_bytes];
            MPSGraphTensor *gate_cb = [graph constantWithData:gate_cb_data shape:@[@(kBlock), @(gate_cols)] dataType:data_type];
            MPSGraphTensor *up_cb = [graph constantWithData:up_cb_data shape:@[@(kBlock), @(up_cols)] dataType:data_type];
            MPSGraphTensor *down_cb = [graph constantWithData:down_cb_data shape:@[@(kBlock), @(down_cols)] dataType:data_type];
            MPSGraphTensor *gate_idx = [graph constantWithData:gate_idx_data shape:@[@(kHiddenDim / kBlock), @(kMidDim)] dataType:MPSDataTypeInt32];
            MPSGraphTensor *up_idx = [graph constantWithData:up_idx_data shape:@[@(kHiddenDim / kBlock), @(kMidDim)] dataType:MPSDataTypeInt32];
            MPSGraphTensor *down_idx = [graph constantWithData:down_idx_data shape:@[@(kMidDim / kBlock), @(kOutDim)] dataType:MPSDataTypeInt32];
            NSString *prefix = [NSString stringWithFormat:@"slot_%u", slot];
            MPSGraphTensor *gate_raw = build_lut(graph, x, gate_cb, gate_idx, kHiddenDim, kMidDim, [prefix stringByAppendingString:@"_gate"]);
            MPSGraphTensor *up_raw = build_lut(graph, x, up_cb, up_idx, kHiddenDim, kMidDim, [prefix stringByAppendingString:@"_up"]);
            MPSGraphTensor *gate_clamped = [graph minimumWithPrimaryTensor:gate_raw secondaryTensor:limit name:[prefix stringByAppendingString:@"_gate_clamp"]];
            MPSGraphTensor *up_clamped = [graph clampWithTensor:up_raw minValueTensor:neg_limit maxValueTensor:limit name:[prefix stringByAppendingString:@"_up_clamp"]];
            MPSGraphTensor *sig = [graph sigmoidWithTensor:gate_clamped name:[prefix stringByAppendingString:@"_sigmoid"]];
            MPSGraphTensor *silu = [graph multiplicationWithPrimaryTensor:gate_clamped secondaryTensor:sig name:[prefix stringByAppendingString:@"_silu"]];
            MPSGraphTensor *mid = [graph multiplicationWithPrimaryTensor:silu secondaryTensor:up_clamped name:[prefix stringByAppendingString:@"_mid"]];
            MPSGraphTensor *down = build_lut(graph, mid, down_cb, down_idx, kMidDim, kOutDim, [prefix stringByAppendingString:@"_down"]);
            total = total ? [graph additionWithPrimaryTensor:total secondaryTensor:down name:[prefix stringByAppendingString:@"_sum"]] : down;
        }
        MPSGraphShapedType *x_type = [[MPSGraphShapedType alloc] initWithShape:@[@1, @(kHiddenDim)] dataType:data_type];
        NSLog(@"[mpsgraph] compile_start nsel=%u", n_experts);
        MPSGraphExecutable *executable = [graph compileWithDevice:nil
                                                            feeds:@{x : x_type}
                                                    targetTensors:@[total]
                                                 targetOperations:nil
                                            compilationDescriptor:nil];
        if (!executable) die(@"MPSGraph compile failed", nil);
        executable.options = MPSGraphOptionsNone;
        MPSGraphTensorData *x_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:mps_input
                                                                             shape:@[@1, @(kHiddenDim)]
                                                                          dataType:data_type];
        MPSGraphTensorData *out_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:d8f_output
                                                                               shape:@[@(kOutDim)]
                                                                            dataType:data_type];
        NSArray *d8f_inputs = @[x_data];
        NSArray *d8f_outputs = @[out_data];
        run_d8f_once(executable, queue, d8f_inputs, d8f_outputs);
        uint16_t *got_h = (uint16_t *)d8f_output.contents;
        for (uint32_t index = 0; index < kOutDim; ++index) got[index] = f16_to_f32(got_h[index]);
        double max_abs = 0.0, max_rel = 0.0, rms = 0.0;
        uint32_t bad = 0;
        for (uint32_t index = 0; index < kOutDim; ++index) {
            double diff = fabs((double)got[index] - (double)ref[index]);
            double rel = diff / fmax(1.0, fabs((double)ref[index]));
            if (diff > max_abs) max_abs = diff;
            if (rel > max_rel) max_rel = rel;
            rms += diff * diff;
            if (diff > 0.75 && rel > 0.05) bad++;
        }
        rms = sqrt(rms / (double)kOutDim);
        NSLog(@"[validate] bad=%u max_abs=%.6g max_rel=%.6g rms=%.6g sample_ref=%.6g sample_got=%.6g",
              bad, max_abs, max_rel, rms, (double)ref[0], (double)got[0]);

        for (uint32_t index = 0; index < 3u; ++index) {
            run_ane_once(model, provider, options);
            run_d8f_once(executable, queue, d8f_inputs, d8f_outputs);
            run_merge_once(queue, merge_pipeline, ane_output, d8f_output, merged_output);
        }
        id<MLFeatureProvider> backing_check = [model predictionFromFeatures:provider options:options error:&error];
        if (!backing_check) die(@"CoreML backing check failed", error);
        MLMultiArray *actual_output = [backing_check featureValueForName:output_name].multiArrayValue;
        NSLog(@"[warmup] output_backing_used=%@", actual_output == ane_output_array ? @"yes" : @"no");

        size_t evict_bytes = (size_t)evict_mib * 1024u * 1024u;
        uint8_t *evict_buffer = evict_bytes ? (uint8_t *)malloc(evict_bytes) : NULL;
        if (evict_bytes && !evict_buffer) die(@"evict allocation failed", nil);
        for (size_t index = 0; index < evict_bytes; ++index) evict_buffer[index] = (uint8_t)index;

        NSArray<NSString *> *base_cases = @[
            @"ane_only",
            @"d8f_only",
            @"d8f_evicted",
            @"d8f_after_ane",
            @"d8f_after_ane_evicted",
            @"ane_after_d8f",
            @"serial_ane_d8f",
            @"concurrent",
            @"merge_only",
            @"serial_ane_d8f_merge",
            @"concurrent_then_merge"
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
                } else if ([case_name isEqualToString:@"d8f_only"]) {
                    start = now_seconds();
                    run_d8f_once(executable, queue, d8f_inputs, d8f_outputs);
                    record_sample(samples, case_name, (now_seconds() - start) * 1e3);
                } else if ([case_name isEqualToString:@"d8f_evicted"]) {
                    double evict_s = evict_cpu_cache(evict_buffer, evict_bytes);
                    [evict_samples addObject:@(evict_s * 1e3)];
                    start = now_seconds();
                    run_d8f_once(executable, queue, d8f_inputs, d8f_outputs);
                    record_sample(samples, case_name, (now_seconds() - start) * 1e3);
                } else if ([case_name isEqualToString:@"d8f_after_ane"]) {
                    run_ane_once(model, provider, options);
                    start = now_seconds();
                    run_d8f_once(executable, queue, d8f_inputs, d8f_outputs);
                    record_sample(samples, case_name, (now_seconds() - start) * 1e3);
                } else if ([case_name isEqualToString:@"d8f_after_ane_evicted"]) {
                    run_ane_once(model, provider, options);
                    double evict_s = evict_cpu_cache(evict_buffer, evict_bytes);
                    [evict_samples addObject:@(evict_s * 1e3)];
                    start = now_seconds();
                    run_d8f_once(executable, queue, d8f_inputs, d8f_outputs);
                    record_sample(samples, case_name, (now_seconds() - start) * 1e3);
                } else if ([case_name isEqualToString:@"ane_after_d8f"]) {
                    run_d8f_once(executable, queue, d8f_inputs, d8f_outputs);
                    start = now_seconds();
                    run_ane_once(model, provider, options);
                    record_sample(samples, case_name, (now_seconds() - start) * 1e3);
                } else if ([case_name isEqualToString:@"serial_ane_d8f"]) {
                    start = now_seconds();
                    run_ane_once(model, provider, options);
                    run_d8f_once(executable, queue, d8f_inputs, d8f_outputs);
                    record_sample(samples, case_name, (now_seconds() - start) * 1e3);
                } else if ([case_name isEqualToString:@"concurrent"]) {
                    dispatch_group_t group = dispatch_group_create();
                    start = now_seconds();
                    dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
                        run_ane_once(model, provider, options);
                    });
                    dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
                        run_d8f_once(executable, queue, d8f_inputs, d8f_outputs);
                    });
                    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
                    record_sample(samples, case_name, (now_seconds() - start) * 1e3);
                } else if ([case_name isEqualToString:@"merge_only"]) {
                    start = now_seconds();
                    run_merge_once(queue, merge_pipeline, ane_output, d8f_output, merged_output);
                    record_sample(samples, case_name, (now_seconds() - start) * 1e3);
                } else if ([case_name isEqualToString:@"serial_ane_d8f_merge"]) {
                    start = now_seconds();
                    run_ane_once(model, provider, options);
                    run_d8f_once(executable, queue, d8f_inputs, d8f_outputs);
                    run_merge_once(queue, merge_pipeline, ane_output, d8f_output, merged_output);
                    record_sample(samples, case_name, (now_seconds() - start) * 1e3);
                } else if ([case_name isEqualToString:@"concurrent_then_merge"]) {
                    dispatch_group_t group = dispatch_group_create();
                    start = now_seconds();
                    dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
                        run_ane_once(model, provider, options);
                    });
                    dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
                        run_d8f_once(executable, queue, d8f_inputs, d8f_outputs);
                    });
                    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
                    run_merge_once(queue, merge_pipeline, ane_output, d8f_output, merged_output);
                    record_sample(samples, case_name, (now_seconds() - start) * 1e3);
                }
            }
            NSLog(@"[trial] index=%u order=%@", trial, order);
        }
        for (NSString *case_name in base_cases) {
            NSDictionary<NSString *, NSNumber *> *summary = summary_for_values(samples[case_name] ?: @[]);
            NSLog(@"[summary] mode=%@ evict_mib=%u nsel=%u case=%@ n=%@ min_ms=%.3f p50_ms=%.3f p90_ms=%.3f max_ms=%.3f mean_ms=%.3f",
                  input_mode,
                  evict_mib,
                  n_experts,
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
        double d8f_p50 = p50_for_case(samples, @"d8f_only");
        double concurrent_p50 = p50_for_case(samples, @"concurrent");
        double serial_p50 = p50_for_case(samples, @"serial_ane_d8f");
        NSLog(@"[derived] mode=%@ evict_mib=%u nsel=%u ane_plus_d8f_p50_ms=%.3f serial_measured_p50_ms=%.3f concurrent_p50_ms=%.3f overlap_speedup_p50=%.3f",
              input_mode,
              evict_mib,
              n_experts,
              ane_p50 + d8f_p50,
              serial_p50,
              concurrent_p50,
              concurrent_p50 > 0.0 ? (ane_p50 + d8f_p50) / concurrent_p50 : 0.0);
        double merge_p50 = p50_for_case(samples, @"merge_only");
        double serial_merge_p50 = p50_for_case(samples, @"serial_ane_d8f_merge");
        double concurrent_merge_p50 = p50_for_case(samples, @"concurrent_then_merge");
        NSLog(@"[merge_effect] mode=%@ evict_mib=%u nsel=%u merge_p50_ms=%.3f ane_plus_d8f_plus_merge_p50_ms=%.3f serial_merge_p50_ms=%.3f concurrent_then_merge_p50_ms=%.3f overlap_merge_speedup_p50=%.3f",
              input_mode,
              evict_mib,
              n_experts,
              merge_p50,
              ane_p50 + d8f_p50 + merge_p50,
              serial_merge_p50,
              concurrent_merge_p50,
              concurrent_merge_p50 > 0.0 ? (ane_p50 + d8f_p50 + merge_p50) / concurrent_merge_p50 : 0.0);
        NSLog(@"[cache_effect] mode=%@ evict_mib=%u nsel=%u d8f_only_p50_ms=%.3f d8f_evicted_p50_ms=%.3f d8f_after_ane_p50_ms=%.3f d8f_after_ane_evicted_p50_ms=%.3f evict_p50_ms=%.3f",
              input_mode,
              evict_mib,
              n_experts,
              d8f_p50,
              p50_for_case(samples, @"d8f_evicted"),
              p50_for_case(samples, @"d8f_after_ane"),
              p50_for_case(samples, @"d8f_after_ane_evicted"),
              evict_summary[@"p50"].doubleValue);
        NSLog(@"[samples] ane0=0x%04x d8f0=0x%04x merged0=0x%04x", ((uint16_t *)ane_output.contents)[0], ((uint16_t *)d8f_output.contents)[0], ((uint16_t *)merged_output.contents)[0]);

        free(evict_buffer);
        free(got);
        free(ref);
        free(mid_tmp);
        free(up_tmp);
        free(gate_tmp);
        free(x_f16);
        free(x_f32);
        free_slots(slots, n_experts);
        ds4_d8f_close(&file);
    }
    return 0;
}
