#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <mach/mach_time.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double now_seconds(void) {
    static mach_timebase_info_data_t timebase;
    if (timebase.denom == 0) mach_timebase_info(&timebase);
    return (double)mach_absolute_time() * (double)timebase.numer /
           (double)timebase.denom * 1e-9;
}

static void fill_inputs(float *query,
                        float *key,
                        float *value,
                        float *mask,
                        uint32_t heads,
                        uint32_t query_rows,
                        uint32_t key_rows,
                        uint32_t dim,
                        uint32_t causal_mask) {
    const uint64_t query_count = (uint64_t)heads * query_rows * dim;
    const uint64_t key_count = (uint64_t)heads * key_rows * dim;
    for (uint64_t i = 0; i < query_count; i++) {
        query[i] = 0.09f * sinf((float)i * 0.013f) + 0.03f * cosf((float)i * 0.071f);
    }
    for (uint64_t i = 0; i < key_count; i++) {
        key[i] = 0.08f * cosf((float)i * 0.017f) - 0.02f * sinf((float)i * 0.043f);
        value[i] = 0.07f * sinf((float)i * 0.019f) + 0.04f * cosf((float)i * 0.037f);
    }
    for (uint32_t head = 0; head < heads; head++) {
        for (uint32_t query_row = 0; query_row < query_rows; query_row++) {
            for (uint32_t key_row = 0; key_row < key_rows; key_row++) {
                float mask_value = 0.0f;
                if (causal_mask && key_row > query_row) mask_value = -INFINITY;
                mask[((uint64_t)head * query_rows + query_row) * key_rows + key_row] = mask_value;
            }
        }
    }
}

static void reference_sdpa(const float *query,
                           const float *key,
                           const float *value,
                           const float *mask,
                           float *output,
                           uint32_t heads,
                           uint32_t query_rows,
                           uint32_t key_rows,
                           uint32_t dim,
                           float scale) {
    for (uint32_t head = 0; head < heads; head++) {
        for (uint32_t query_row = 0; query_row < query_rows; query_row++) {
            float max_score = -INFINITY;
            for (uint32_t key_row = 0; key_row < key_rows; key_row++) {
                double dot = 0.0;
                for (uint32_t feature = 0; feature < dim; feature++) {
                    const uint64_t query_index = ((uint64_t)head * query_rows + query_row) * dim + feature;
                    const uint64_t key_index = ((uint64_t)head * key_rows + key_row) * dim + feature;
                    dot += (double)query[query_index] * (double)key[key_index];
                }
                const float mask_value = mask[((uint64_t)head * query_rows + query_row) * key_rows + key_row];
                const float score = (float)dot * scale + mask_value;
                if (score > max_score) max_score = score;
            }
            double denom = 0.0;
            for (uint32_t key_row = 0; key_row < key_rows; key_row++) {
                double dot = 0.0;
                for (uint32_t feature = 0; feature < dim; feature++) {
                    const uint64_t query_index = ((uint64_t)head * query_rows + query_row) * dim + feature;
                    const uint64_t key_index = ((uint64_t)head * key_rows + key_row) * dim + feature;
                    dot += (double)query[query_index] * (double)key[key_index];
                }
                const float mask_value = mask[((uint64_t)head * query_rows + query_row) * key_rows + key_row];
                denom += exp((double)((float)dot * scale + mask_value - max_score));
            }
            for (uint32_t feature = 0; feature < dim; feature++) {
                double acc = 0.0;
                for (uint32_t key_row = 0; key_row < key_rows; key_row++) {
                    double dot = 0.0;
                    for (uint32_t inner = 0; inner < dim; inner++) {
                        const uint64_t query_index = ((uint64_t)head * query_rows + query_row) * dim + inner;
                        const uint64_t key_index = ((uint64_t)head * key_rows + key_row) * dim + inner;
                        dot += (double)query[query_index] * (double)key[key_index];
                    }
                    const float mask_value = mask[((uint64_t)head * query_rows + query_row) * key_rows + key_row];
                    const double weight = exp((double)((float)dot * scale + mask_value - max_score)) / denom;
                    const uint64_t value_index = ((uint64_t)head * key_rows + key_row) * dim + feature;
                    acc += weight * (double)value[value_index];
                }
                output[((uint64_t)head * query_rows + query_row) * dim + feature] = (float)acc;
            }
        }
    }
}

static MPSGraphExecutable *compile_graph(MPSGraph *graph,
                                         MPSGraphTensor *query,
                                         MPSGraphTensor *key,
                                         MPSGraphTensor *value,
                                         MPSGraphTensor *mask,
                                         MPSGraphTensor *output,
                                         uint32_t heads,
                                         uint32_t query_rows,
                                         uint32_t key_rows,
                                         uint32_t dim) {
    MPSGraphShapedType *query_type = [[MPSGraphShapedType alloc]
        initWithShape:@[@1, @(heads), @(query_rows), @(dim)]
             dataType:MPSDataTypeFloat32];
    MPSGraphShapedType *key_type = [[MPSGraphShapedType alloc]
        initWithShape:@[@1, @(heads), @(key_rows), @(dim)]
             dataType:MPSDataTypeFloat32];
    MPSGraphShapedType *value_type = [[MPSGraphShapedType alloc]
        initWithShape:@[@1, @(heads), @(key_rows), @(dim)]
             dataType:MPSDataTypeFloat32];
    MPSGraphShapedType *mask_type = [[MPSGraphShapedType alloc]
        initWithShape:@[@1, @(heads), @(query_rows), @(key_rows)]
             dataType:MPSDataTypeFloat32];
    return [graph compileWithDevice:nil
                         feeds:@{query: query_type, key: key_type, value: value_type, mask: mask_type}
                 targetTensors:@[output]
              targetOperations:nil
          compilationDescriptor:nil];
}

static double time_executable(MPSGraphExecutable *executable,
                              id<MTLCommandQueue> queue,
                              NSArray<MPSGraphTensorData *> *inputs,
                              NSArray<MPSGraphTensorData *> *outputs,
                              uint32_t rounds) {
    for (uint32_t round = 0; round < 3; round++) {
        [executable runWithMTLCommandQueue:queue
                               inputsArray:inputs
                              resultsArray:outputs
                       executionDescriptor:nil];
    }
    const double start = now_seconds();
    for (uint32_t round = 0; round < rounds; round++) {
        [executable runWithMTLCommandQueue:queue
                               inputsArray:inputs
                              resultsArray:outputs
                       executionDescriptor:nil];
    }
    return (now_seconds() - start) * 1000.0;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const uint32_t heads = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 10) : 8u;
        const uint32_t query_rows = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 10) : 16u;
        const uint32_t key_rows = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10) : 128u;
        const uint32_t dim = argc > 4 ? (uint32_t)strtoul(argv[4], NULL, 10) : 512u;
        const uint32_t rounds = argc > 5 ? (uint32_t)strtoul(argv[5], NULL, 10) : 20u;
        const uint32_t causal_mask = argc > 6 ? (uint32_t)strtoul(argv[6], NULL, 10) : 0u;
        if (heads == 0u || query_rows == 0u || key_rows == 0u || dim == 0u || rounds == 0u) return 2;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!device || !queue) return 3;

        const uint64_t query_count = (uint64_t)heads * query_rows * dim;
        const uint64_t key_count = (uint64_t)heads * key_rows * dim;
        const uint64_t mask_count = (uint64_t)heads * query_rows * key_rows;
        float *query = (float *)malloc((size_t)query_count * sizeof(float));
        float *key = (float *)malloc((size_t)key_count * sizeof(float));
        float *value = (float *)malloc((size_t)key_count * sizeof(float));
        float *mask = (float *)malloc((size_t)mask_count * sizeof(float));
        float *reference = (float *)calloc((size_t)query_count, sizeof(float));
        float *fused_out = (float *)calloc((size_t)query_count, sizeof(float));
        float *explicit_out = (float *)calloc((size_t)query_count, sizeof(float));
        if (!query || !key || !value || !mask || !reference || !fused_out || !explicit_out) return 4;

        fill_inputs(query, key, value, mask, heads, query_rows, key_rows, dim, causal_mask);
        const float scale = 1.0f / sqrtf((float)dim);
        reference_sdpa(query, key, value, mask, reference, heads, query_rows, key_rows, dim, scale);

        id<MTLBuffer> query_buf = [device newBufferWithBytes:query length:(NSUInteger)(query_count * sizeof(float)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> key_buf = [device newBufferWithBytes:key length:(NSUInteger)(key_count * sizeof(float)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> value_buf = [device newBufferWithBytes:value length:(NSUInteger)(key_count * sizeof(float)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> mask_buf = [device newBufferWithBytes:mask length:(NSUInteger)(mask_count * sizeof(float)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> fused_buf = [device newBufferWithBytes:fused_out length:(NSUInteger)(query_count * sizeof(float)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> explicit_buf = [device newBufferWithBytes:explicit_out length:(NSUInteger)(query_count * sizeof(float)) options:MTLResourceStorageModeShared];
        if (!query_buf || !key_buf || !value_buf || !mask_buf || !fused_buf || !explicit_buf) return 5;

        NSArray *query_shape = @[@1, @(heads), @(query_rows), @(dim)];
        NSArray *key_shape = @[@1, @(heads), @(key_rows), @(dim)];
        NSArray *mask_shape = @[@1, @(heads), @(query_rows), @(key_rows)];
        MPSGraphTensorData *query_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:query_buf shape:query_shape dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *key_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:key_buf shape:key_shape dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *value_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:value_buf shape:key_shape dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *mask_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:mask_buf shape:mask_shape dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *fused_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:fused_buf shape:query_shape dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *explicit_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:explicit_buf shape:query_shape dataType:MPSDataTypeFloat32];

        MPSGraph *fused_graph = [[MPSGraph alloc] init];
        MPSGraphTensor *fq = [fused_graph placeholderWithShape:query_shape dataType:MPSDataTypeFloat32 name:@"q"];
        MPSGraphTensor *fk = [fused_graph placeholderWithShape:key_shape dataType:MPSDataTypeFloat32 name:@"k"];
        MPSGraphTensor *fv = [fused_graph placeholderWithShape:key_shape dataType:MPSDataTypeFloat32 name:@"v"];
        MPSGraphTensor *fm = [fused_graph placeholderWithShape:mask_shape dataType:MPSDataTypeFloat32 name:@"m"];
        MPSGraphTensor *fo = [fused_graph scaledDotProductAttentionWithQueryTensor:fq
                                                                         keyTensor:fk
                                                                       valueTensor:fv
                                                                        maskTensor:fm
                                                                             scale:scale
                                                                              name:@"sdpa"];
        MPSGraphExecutable *fused_executable = compile_graph(fused_graph, fq, fk, fv, fm, fo, heads, query_rows, key_rows, dim);

        MPSGraph *explicit_graph = [[MPSGraph alloc] init];
        MPSGraphTensor *eq = [explicit_graph placeholderWithShape:query_shape dataType:MPSDataTypeFloat32 name:@"q"];
        MPSGraphTensor *ek = [explicit_graph placeholderWithShape:key_shape dataType:MPSDataTypeFloat32 name:@"k"];
        MPSGraphTensor *ev = [explicit_graph placeholderWithShape:key_shape dataType:MPSDataTypeFloat32 name:@"v"];
        MPSGraphTensor *em = [explicit_graph placeholderWithShape:mask_shape dataType:MPSDataTypeFloat32 name:@"m"];
        MPSGraphTensor *ekt = [explicit_graph transposeTensor:ek permutation:@[@0, @1, @3, @2] name:@"kt"];
        MPSGraphTensor *scores = [explicit_graph matrixMultiplicationWithPrimaryTensor:eq secondaryTensor:ekt name:@"qkt"];
        MPSGraphTensor *scale_tensor = [explicit_graph constantWithScalar:(double)scale
                                                                     shape:@[@1]
                                                                  dataType:MPSDataTypeFloat32];
        MPSGraphTensor *scaled = [explicit_graph multiplicationWithPrimaryTensor:scores secondaryTensor:scale_tensor name:@"scale"];
        MPSGraphTensor *masked = [explicit_graph additionWithPrimaryTensor:scaled secondaryTensor:em name:@"mask"];
        MPSGraphTensor *prob = [explicit_graph softMaxWithTensor:masked axis:3 name:@"softmax"];
        MPSGraphTensor *eo = [explicit_graph matrixMultiplicationWithPrimaryTensor:prob secondaryTensor:ev name:@"pv"];
        MPSGraphExecutable *explicit_executable = compile_graph(explicit_graph, eq, ek, ev, em, eo, heads, query_rows, key_rows, dim);

        if (!fused_executable || !explicit_executable) return 6;
        NSArray *inputs = @[query_data, key_data, value_data, mask_data];
        [fused_executable runWithMTLCommandQueue:queue inputsArray:inputs resultsArray:@[fused_data] executionDescriptor:nil];
        [explicit_executable runWithMTLCommandQueue:queue inputsArray:inputs resultsArray:@[explicit_data] executionDescriptor:nil];
        memcpy(fused_out, fused_buf.contents, (size_t)query_count * sizeof(float));
        memcpy(explicit_out, explicit_buf.contents, (size_t)query_count * sizeof(float));

        double fused_max_abs = 0.0;
        double explicit_max_abs = 0.0;
        double fused_vs_explicit_max_abs = 0.0;
        for (uint64_t index = 0; index < query_count; index++) {
            const double fused_err = fabs((double)fused_out[index] - (double)reference[index]);
            const double explicit_err = fabs((double)explicit_out[index] - (double)reference[index]);
            const double both_err = fabs((double)fused_out[index] - (double)explicit_out[index]);
            if (fused_err > fused_max_abs) fused_max_abs = fused_err;
            if (explicit_err > explicit_max_abs) explicit_max_abs = explicit_err;
            if (both_err > fused_vs_explicit_max_abs) fused_vs_explicit_max_abs = both_err;
        }

        const double fused_ms = time_executable(fused_executable, queue, inputs, @[fused_data], rounds);
        const double explicit_ms = time_executable(explicit_executable, queue, inputs, @[explicit_data], rounds);
        printf("mpsgraph_sdpa_canary heads=%u query=%u keys=%u dim=%u rounds=%u causal=%u "
               "fused_ms=%.3f explicit_ms=%.3f speedup=%.3f "
               "fused_max_abs=%.6e explicit_max_abs=%.6e fused_vs_explicit_max_abs=%.6e\n",
               heads, query_rows, key_rows, dim, rounds, causal_mask,
               fused_ms, explicit_ms, explicit_ms > 0.0 ? explicit_ms / fused_ms : 0.0,
               fused_max_abs, explicit_max_abs, fused_vs_explicit_max_abs);
        return fused_max_abs < 1e-4 && explicit_max_abs < 1e-4 ? 0 : 1;
    }
}
