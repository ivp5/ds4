#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

typedef unsigned long ulong;

static const char *kMetalSource =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"static inline float dot8(device const half *codebook, float mid0, float mid1, float mid2, float mid3, float mid4, float mid5, float mid6, float mid7) {\n"
"  return float(codebook[0]) * mid0 + float(codebook[1]) * mid1 +\n"
"         float(codebook[2]) * mid2 + float(codebook[3]) * mid3 +\n"
"         float(codebook[4]) * mid4 + float(codebook[5]) * mid5 +\n"
"         float(codebook[6]) * mid6 + float(codebook[7]) * mid7;\n"
"}\n"
"\n"
"static inline float dot8_tg(threadgroup const half *codebook, float mid0, float mid1, float mid2, float mid3, float mid4, float mid5, float mid6, float mid7) {\n"
"  return float(codebook[0]) * mid0 + float(codebook[1]) * mid1 +\n"
"         float(codebook[2]) * mid2 + float(codebook[3]) * mid3 +\n"
"         float(codebook[4]) * mid4 + float(codebook[5]) * mid5 +\n"
"         float(codebook[6]) * mid6 + float(codebook[7]) * mid7;\n"
"}\n"
"\n"
"kernel void direct16_device(\n"
"  device const half *codebooks [[buffer(0)]],\n"
"  device const ushort *codes [[buffer(1)]],\n"
"  device const float *mid [[buffer(2)]],\n"
"  device float *out [[buffer(3)]],\n"
"  constant uint &code_count [[buffer(4)]],\n"
"  constant uint &rows [[buffer(5)]],\n"
"  constant uint &groups [[buffer(6)]],\n"
"  threadgroup float *partial [[threadgroup(0)]],\n"
"  uint tid [[thread_index_in_threadgroup]],\n"
"  ushort lane [[thread_index_in_simdgroup]],\n"
"  ushort simdgroup [[simdgroup_index_in_threadgroup]],\n"
"  uint2 tile [[threadgroup_position_in_grid]]) {\n"
"  const uint row_base = tile.x << 4;\n"
"  const uint slot = tile.y;\n"
"  float acc[16];\n"
"  for (uint row_offset = 0; row_offset < 16u; row_offset++) acc[row_offset] = 0.0f;\n"
"  for (uint group = tid; group < groups; group += 256u) {\n"
"    const uint mid_base = slot * groups * 8u + group * 8u;\n"
"    const float mid0 = mid[mid_base + 0u];\n"
"    const float mid1 = mid[mid_base + 1u];\n"
"    const float mid2 = mid[mid_base + 2u];\n"
"    const float mid3 = mid[mid_base + 3u];\n"
"    const float mid4 = mid[mid_base + 4u];\n"
"    const float mid5 = mid[mid_base + 5u];\n"
"    const float mid6 = mid[mid_base + 6u];\n"
"    const float mid7 = mid[mid_base + 7u];\n"
"    for (uint row_offset = 0; row_offset < 16u; row_offset++) {\n"
"      const uint row = row_base + row_offset;\n"
"      if (row >= rows) continue;\n"
"      const ushort code = codes[(ulong(slot) * ulong(rows) + ulong(row)) * ulong(groups) + ulong(group)];\n"
"      if (code >= code_count) continue;\n"
"      device const half *codebook = codebooks + (ulong(slot) * ulong(code_count) + ulong(code)) * 8ul;\n"
"      acc[row_offset] += dot8(codebook, mid0, mid1, mid2, mid3, mid4, mid5, mid6, mid7);\n"
"    }\n"
"  }\n"
"  for (uint row_offset = 0; row_offset < 16u; row_offset++) {\n"
"    const float subtotal = simd_sum(acc[row_offset]);\n"
"    if (lane == 0u) partial[row_offset * 8u + uint(simdgroup)] = subtotal;\n"
"  }\n"
"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  if (tid == 0u) {\n"
"    for (uint row_offset = 0; row_offset < 16u; row_offset++) {\n"
"      const uint row = row_base + row_offset;\n"
"      if (row >= rows) continue;\n"
"      float total = 0.0f;\n"
"      for (uint sg = 0u; sg < 8u; sg++) total += partial[row_offset * 8u + sg];\n"
"      out[ulong(slot) * ulong(rows) + ulong(row)] = total;\n"
"    }\n"
"  }\n"
"}\n"
"\n"
"kernel void direct16_chunk1024(\n"
"  device const half *codebooks [[buffer(0)]],\n"
"  device const ushort *codes [[buffer(1)]],\n"
"  device const float *mid [[buffer(2)]],\n"
"  device float *out [[buffer(3)]],\n"
"  constant uint &code_count [[buffer(4)]],\n"
"  constant uint &rows [[buffer(5)]],\n"
"  constant uint &groups [[buffer(6)]],\n"
"  threadgroup float *partial [[threadgroup(0)]],\n"
"  threadgroup half *codebook_cache [[threadgroup(1)]],\n"
"  uint tid [[thread_index_in_threadgroup]],\n"
"  ushort lane [[thread_index_in_simdgroup]],\n"
"  ushort simdgroup [[simdgroup_index_in_threadgroup]],\n"
"  uint2 tile [[threadgroup_position_in_grid]]) {\n"
"  const uint row_base = tile.x << 4;\n"
"  const uint slot = tile.y;\n"
"  float acc[16];\n"
"  for (uint row_offset = 0; row_offset < 16u; row_offset++) acc[row_offset] = 0.0f;\n"
"  for (uint chunk_base = 0u; chunk_base < code_count; chunk_base += 1024u) {\n"
"    const uint chunk_count = min(1024u, code_count - chunk_base);\n"
"    const uint cache_values = chunk_count * 8u;\n"
"    device const half *chunk_source = codebooks + (ulong(slot) * ulong(code_count) + ulong(chunk_base)) * 8ul;\n"
"    for (uint cache_index = tid; cache_index < cache_values; cache_index += 256u) {\n"
"      codebook_cache[cache_index] = chunk_source[cache_index];\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint group = tid; group < groups; group += 256u) {\n"
"      const uint mid_base = slot * groups * 8u + group * 8u;\n"
"      const float mid0 = mid[mid_base + 0u];\n"
"      const float mid1 = mid[mid_base + 1u];\n"
"      const float mid2 = mid[mid_base + 2u];\n"
"      const float mid3 = mid[mid_base + 3u];\n"
"      const float mid4 = mid[mid_base + 4u];\n"
"      const float mid5 = mid[mid_base + 5u];\n"
"      const float mid6 = mid[mid_base + 6u];\n"
"      const float mid7 = mid[mid_base + 7u];\n"
"      for (uint row_offset = 0; row_offset < 16u; row_offset++) {\n"
"        const uint row = row_base + row_offset;\n"
"        if (row >= rows) continue;\n"
"        const uint code = uint(codes[(ulong(slot) * ulong(rows) + ulong(row)) * ulong(groups) + ulong(group)]);\n"
"        if (code < chunk_base || code >= chunk_base + chunk_count) continue;\n"
"        threadgroup const half *codebook = codebook_cache + ulong(code - chunk_base) * 8ul;\n"
"        acc[row_offset] += dot8_tg(codebook, mid0, mid1, mid2, mid3, mid4, mid5, mid6, mid7);\n"
"      }\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  }\n"
"  for (uint row_offset = 0; row_offset < 16u; row_offset++) {\n"
"    const float subtotal = simd_sum(acc[row_offset]);\n"
"    if (lane == 0u) partial[row_offset * 8u + uint(simdgroup)] = subtotal;\n"
"  }\n"
"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  if (tid == 0u) {\n"
"    for (uint row_offset = 0; row_offset < 16u; row_offset++) {\n"
"      const uint row = row_base + row_offset;\n"
"      if (row >= rows) continue;\n"
"      float total = 0.0f;\n"
"      for (uint sg = 0u; sg < 8u; sg++) total += partial[row_offset * 8u + sg];\n"
"      out[ulong(slot) * ulong(rows) + ulong(row)] = total;\n"
"    }\n"
"  }\n"
"}\n";

static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString *name) {
    NSError *error = nil;
    id<MTLFunction> function = [library newFunctionWithName:name];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) NSLog(@"pipeline %@ failed: %@", name, error);
    return pipeline;
}

static double run_kernel(id<MTLCommandQueue> queue,
                         id<MTLComputePipelineState> pipeline,
                         id<MTLBuffer> codebooks,
                         id<MTLBuffer> codes,
                         id<MTLBuffer> mid,
                         id<MTLBuffer> out,
                         uint32_t code_count,
                         uint32_t rows,
                         uint32_t groups,
                         uint32_t slots,
                         bool chunked,
                         uint32_t iterations) {
    double best_ms = 1.0e30;
    for (uint32_t iteration = 0; iteration < iterations; iteration++) {
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:codebooks offset:0 atIndex:0];
        [encoder setBuffer:codes offset:0 atIndex:1];
        [encoder setBuffer:mid offset:0 atIndex:2];
        [encoder setBuffer:out offset:0 atIndex:3];
        [encoder setBytes:&code_count length:sizeof(code_count) atIndex:4];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:5];
        [encoder setBytes:&groups length:sizeof(groups) atIndex:6];
        [encoder setThreadgroupMemoryLength:16u * 8u * sizeof(float) atIndex:0];
        if (chunked) {
            [encoder setThreadgroupMemoryLength:1024u * 8u * sizeof(uint16_t) atIndex:1];
        }
        [encoder dispatchThreadgroups:MTLSizeMake((rows + 15u) >> 4, slots, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        const double elapsed_ms = (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0;
        if (elapsed_ms > 0.0 && elapsed_ms < best_ms) best_ms = elapsed_ms;
    }
    return best_ms;
}

static int run_case(id<MTLDevice> device,
                    id<MTLCommandQueue> queue,
                    id<MTLComputePipelineState> direct_pipeline,
                    id<MTLComputePipelineState> chunk_pipeline,
                    uint32_t code_count,
                    uint32_t rows,
                    uint32_t groups,
                    uint32_t slots,
                    uint32_t iterations) {
    const NSUInteger codebook_bytes = (NSUInteger)slots * code_count * 8u * sizeof(uint16_t);
    const NSUInteger code_bytes = (NSUInteger)slots * rows * groups * sizeof(uint16_t);
    const NSUInteger mid_bytes = (NSUInteger)slots * groups * 8u * sizeof(float);
    const NSUInteger out_bytes = (NSUInteger)slots * rows * sizeof(float);
    id<MTLBuffer> codebooks = [device newBufferWithLength:codebook_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> random_codes = [device newBufferWithLength:code_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> fixed_codes = [device newBufferWithLength:code_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> mid = [device newBufferWithLength:mid_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> out = [device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
    if (!codebooks || !random_codes || !fixed_codes || !mid || !out) {
        fprintf(stderr, "allocation failed K=%u codebook=%.2fMiB codes=%.2fMiB\n",
                code_count, (double)codebook_bytes / 1048576.0, (double)code_bytes / 1048576.0);
        return 0;
    }

    uint16_t *codebook_values = codebooks.contents;
    uint16_t *random_code_values = random_codes.contents;
    uint16_t *fixed_code_values = fixed_codes.contents;
    float *mid_values = mid.contents;
    srand(20260604u + code_count);
    for (NSUInteger index = 0; index < (NSUInteger)slots * code_count * 8u; index++) {
        const float value = 0.015f * sinf((float)index * 0.013f) + 0.007f * cosf((float)index * 0.031f);
        const __fp16 half_value = (__fp16)value;
        codebook_values[index] = *(const uint16_t *)&half_value;
    }
    for (NSUInteger index = 0; index < (NSUInteger)slots * rows * groups; index++) {
        random_code_values[index] = (uint16_t)(rand() % code_count);
        fixed_code_values[index] = 0u;
    }
    for (NSUInteger index = 0; index < (NSUInteger)slots * groups * 8u; index++) {
        mid_values[index] = sinf((float)index * 0.0017f) + 0.2f * cosf((float)index * 0.019f);
    }

    const double random_ms = run_kernel(queue, direct_pipeline, codebooks, random_codes, mid, out,
                                        code_count, rows, groups, slots, false, iterations);
    const double fixed_ms = run_kernel(queue, direct_pipeline, codebooks, fixed_codes, mid, out,
                                       code_count, rows, groups, slots, false, iterations);
    const double chunk_random_ms = run_kernel(queue, chunk_pipeline, codebooks, random_codes, mid, out,
                                              code_count, rows, groups, slots, true, iterations);
    const double chunk_fixed_ms = run_kernel(queue, chunk_pipeline, codebooks, fixed_codes, mid, out,
                                             code_count, rows, groups, slots, true, iterations);
    printf("locality K=%u rows=%u groups=%u slots=%u it=%u codebook=%.2fMiB codes=%.2fMiB "
           "device_random=%.4fms device_fixed0=%.4fms fixed_speed=%.3fx "
           "chunk1024_random=%.4fms chunk_vs_device=%.3fx chunk_fixed0=%.4fms\n",
           code_count, rows, groups, slots, iterations,
           (double)codebook_bytes / 1048576.0, (double)code_bytes / 1048576.0,
           random_ms, fixed_ms, random_ms / fixed_ms,
           chunk_random_ms, random_ms / chunk_random_ms, chunk_fixed_ms);
    return 1;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        uint32_t iterations = 40;
        uint32_t slots = 6;
        if (argc > 1) iterations = (uint32_t)strtoul(argv[1], NULL, 10);
        if (argc > 2) slots = (uint32_t)strtoul(argv[2], NULL, 10);
        const uint32_t rows = 4096;
        const uint32_t groups = 256;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kMetalSource]
                                                       options:nil
                                                         error:&error];
        if (!library) {
            NSLog(@"library creation failed: %@", error);
            return 1;
        }
        id<MTLComputePipelineState> direct_pipeline = make_pipeline(device, library, @"direct16_device");
        id<MTLComputePipelineState> chunk_pipeline = make_pipeline(device, library, @"direct16_chunk1024");
        if (!direct_pipeline || !chunk_pipeline) return 1;

        printf("d8f direct-down codebook locality canary\n");
        if (!run_case(device, queue, direct_pipeline, chunk_pipeline, 1024, rows, groups, slots, iterations)) return 1;
        if (!run_case(device, queue, direct_pipeline, chunk_pipeline, 2048, rows, groups, slots, iterations)) return 1;
        if (!run_case(device, queue, direct_pipeline, chunk_pipeline, 4096, rows, groups, slots, iterations)) return 1;
    }
    return 0;
}
