#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned long ulong;

static const char *kMetalSource =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"static inline float dot8_device(device const half *codebook, float mid0, float mid1, float mid2, float mid3, float mid4, float mid5, float mid6, float mid7) {\n"
"  return float(codebook[0]) * mid0 + float(codebook[1]) * mid1 +\n"
"         float(codebook[2]) * mid2 + float(codebook[3]) * mid3 +\n"
"         float(codebook[4]) * mid4 + float(codebook[5]) * mid5 +\n"
"         float(codebook[6]) * mid6 + float(codebook[7]) * mid7;\n"
"}\n"
"\n"
"static inline float dot8_texture(texture2d<half, access::read> codebooks, uint slot, uint code, float mid0, float mid1, float mid2, float mid3, float mid4, float mid5, float mid6, float mid7) {\n"
"  const half4 lo = codebooks.read(uint2(code * 2u, slot));\n"
"  const half4 hi = codebooks.read(uint2(code * 2u + 1u, slot));\n"
"  return float(lo.x) * mid0 + float(lo.y) * mid1 + float(lo.z) * mid2 + float(lo.w) * mid3 +\n"
"         float(hi.x) * mid4 + float(hi.y) * mid5 + float(hi.z) * mid6 + float(hi.w) * mid7;\n"
"}\n"
"\n"
"kernel void direct16_buffer(\n"
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
"      acc[row_offset] += dot8_device(codebook, mid0, mid1, mid2, mid3, mid4, mid5, mid6, mid7);\n"
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
"kernel void direct16_texture(\n"
"  texture2d<half, access::read> codebooks [[texture(0)]],\n"
"  device const ushort *codes [[buffer(0)]],\n"
"  device const float *mid [[buffer(1)]],\n"
"  device float *out [[buffer(2)]],\n"
"  constant uint &code_count [[buffer(3)]],\n"
"  constant uint &rows [[buffer(4)]],\n"
"  constant uint &groups [[buffer(5)]],\n"
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
"      acc[row_offset] += dot8_texture(codebooks, slot, uint(code), mid0, mid1, mid2, mid3, mid4, mid5, mid6, mid7);\n"
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
"}\n";

static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString *name) {
    NSError *error = nil;
    id<MTLFunction> function = [library newFunctionWithName:name];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) NSLog(@"pipeline %@ failed: %@", name, error);
    return pipeline;
}

static double run_buffer_kernel(id<MTLCommandQueue> queue,
                                id<MTLComputePipelineState> pipeline,
                                id<MTLBuffer> codebooks,
                                id<MTLBuffer> codes,
                                id<MTLBuffer> mid,
                                id<MTLBuffer> out,
                                uint32_t code_count,
                                uint32_t rows,
                                uint32_t groups,
                                uint32_t slots,
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

static double run_texture_kernel(id<MTLCommandQueue> queue,
                                 id<MTLComputePipelineState> pipeline,
                                 id<MTLTexture> codebooks,
                                 id<MTLBuffer> codes,
                                 id<MTLBuffer> mid,
                                 id<MTLBuffer> out,
                                 uint32_t code_count,
                                 uint32_t rows,
                                 uint32_t groups,
                                 uint32_t slots,
                                 uint32_t iterations) {
    double best_ms = 1.0e30;
    for (uint32_t iteration = 0; iteration < iterations; iteration++) {
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setTexture:codebooks atIndex:0];
        [encoder setBuffer:codes offset:0 atIndex:0];
        [encoder setBuffer:mid offset:0 atIndex:1];
        [encoder setBuffer:out offset:0 atIndex:2];
        [encoder setBytes:&code_count length:sizeof(code_count) atIndex:3];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
        [encoder setBytes:&groups length:sizeof(groups) atIndex:5];
        [encoder setThreadgroupMemoryLength:16u * 8u * sizeof(float) atIndex:0];
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

static id<MTLTexture> make_texture(id<MTLDevice> device,
                                   id<MTLCommandQueue> queue,
                                   const uint16_t *texels,
                                   uint32_t width,
                                   uint32_t height,
                                   MTLStorageMode storage_mode,
                                   const char *label) {
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                          width:width
                                                                                         height:height
                                                                                      mipmapped:NO];
    descriptor.usage = MTLTextureUsageShaderRead;
    descriptor.storageMode = storage_mode;
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    if (!texture) {
        fprintf(stderr, "texture allocation failed label=%s width=%u height=%u storage=%lu\n",
                label, width, height, (unsigned long)storage_mode);
        return nil;
    }
    const NSUInteger bytes_per_row = (NSUInteger)width * 4u * sizeof(uint16_t);
    if (storage_mode == MTLStorageModePrivate) {
        const NSUInteger total_bytes = bytes_per_row * height;
        id<MTLBuffer> staging = [device newBufferWithLength:total_bytes options:MTLResourceStorageModeShared];
        if (!staging) return nil;
        memcpy(staging.contents, texels, total_bytes);
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
        [blit copyFromBuffer:staging
                sourceOffset:0
           sourceBytesPerRow:bytes_per_row
         sourceBytesPerImage:bytes_per_row * height
                  sourceSize:MTLSizeMake(width, height, 1)
                   toTexture:texture
            destinationSlice:0
            destinationLevel:0
           destinationOrigin:MTLOriginMake(0, 0, 0)];
        [blit endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
    } else {
        [texture replaceRegion:MTLRegionMake2D(0, 0, width, height)
                    mipmapLevel:0
                      withBytes:texels
                    bytesPerRow:bytes_per_row];
    }
    return texture;
}

static id<MTLTexture> make_buffer_backed_texture(id<MTLDevice> device,
                                                 id<MTLBuffer> texel_buffer,
                                                 uint32_t width,
                                                 uint32_t height) {
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                          width:width
                                                                                         height:height
                                                                                      mipmapped:NO];
    descriptor.usage = MTLTextureUsageShaderRead;
    const NSUInteger bytes_per_row = (NSUInteger)width * 4u * sizeof(uint16_t);
    return [texel_buffer newTextureWithDescriptor:descriptor offset:0 bytesPerRow:bytes_per_row];
}

static int run_case(id<MTLDevice> device,
                    id<MTLCommandQueue> queue,
                    id<MTLComputePipelineState> buffer_pipeline,
                    id<MTLComputePipelineState> texture_pipeline,
                    uint32_t code_count,
                    uint32_t rows,
                    uint32_t groups,
                    uint32_t slots,
                    uint32_t iterations) {
    const uint32_t texture_width = code_count * 2u;
    const uint32_t texture_height = slots;
    const NSUInteger codebook_bytes = (NSUInteger)slots * code_count * 8u * sizeof(uint16_t);
    const NSUInteger texel_bytes = (NSUInteger)texture_width * texture_height * 4u * sizeof(uint16_t);
    const NSUInteger code_bytes = (NSUInteger)slots * rows * groups * sizeof(uint16_t);
    const NSUInteger mid_bytes = (NSUInteger)slots * groups * 8u * sizeof(float);
    const NSUInteger out_bytes = (NSUInteger)slots * rows * sizeof(float);
    id<MTLBuffer> codebooks = [device newBufferWithLength:codebook_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> texel_buffer = [device newBufferWithLength:texel_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> random_codes = [device newBufferWithLength:code_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> fixed_codes = [device newBufferWithLength:code_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> mid = [device newBufferWithLength:mid_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_buffer = [device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_texture = [device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
    if (!codebooks || !texel_buffer || !random_codes || !fixed_codes || !mid || !out_buffer || !out_texture) {
        fprintf(stderr, "allocation failed K=%u\n", code_count);
        return 0;
    }

    uint16_t *codebook_values = codebooks.contents;
    uint16_t *texel_values = texel_buffer.contents;
    uint16_t *random_code_values = random_codes.contents;
    uint16_t *fixed_code_values = fixed_codes.contents;
    float *mid_values = mid.contents;
    srand(20260604u + code_count);
    for (NSUInteger slot = 0; slot < slots; slot++) {
        for (NSUInteger code = 0; code < code_count; code++) {
            for (NSUInteger lane = 0; lane < 8u; lane++) {
                const NSUInteger linear = (slot * code_count + code) * 8u + lane;
                const float value = 0.015f * sinf((float)linear * 0.013f) + 0.007f * cosf((float)linear * 0.031f);
                const __fp16 half_value = (__fp16)value;
                const uint16_t bits = *(const uint16_t *)&half_value;
                codebook_values[linear] = bits;
                const NSUInteger pixel = (slot * texture_width + code * 2u + lane / 4u) * 4u + lane % 4u;
                texel_values[pixel] = bits;
            }
        }
    }
    for (NSUInteger index = 0; index < (NSUInteger)slots * rows * groups; index++) {
        random_code_values[index] = (uint16_t)(rand() % code_count);
        fixed_code_values[index] = 0u;
    }
    for (NSUInteger index = 0; index < (NSUInteger)slots * groups * 8u; index++) {
        mid_values[index] = sinf((float)index * 0.0017f) + 0.2f * cosf((float)index * 0.019f);
    }

    id<MTLTexture> linear_texture = make_buffer_backed_texture(device, texel_buffer, texture_width, texture_height);
    id<MTLTexture> shared_texture = make_texture(device, queue, texel_values, texture_width, texture_height, MTLStorageModeShared, "shared");
    id<MTLTexture> private_texture = make_texture(device, queue, texel_values, texture_width, texture_height, MTLStorageModePrivate, "private");
    if (!linear_texture || !shared_texture || !private_texture) return 0;

    const double buffer_random_ms = run_buffer_kernel(queue, buffer_pipeline, codebooks, random_codes, mid, out_buffer,
                                                      code_count, rows, groups, slots, iterations);
    const double buffer_fixed_ms = run_buffer_kernel(queue, buffer_pipeline, codebooks, fixed_codes, mid, out_buffer,
                                                     code_count, rows, groups, slots, iterations);
    const double linear_random_ms = run_texture_kernel(queue, texture_pipeline, linear_texture, random_codes, mid, out_texture,
                                                       code_count, rows, groups, slots, iterations);
    const double shared_random_ms = run_texture_kernel(queue, texture_pipeline, shared_texture, random_codes, mid, out_texture,
                                                       code_count, rows, groups, slots, iterations);
    const double private_random_ms = run_texture_kernel(queue, texture_pipeline, private_texture, random_codes, mid, out_texture,
                                                        code_count, rows, groups, slots, iterations);
    const double private_fixed_ms = run_texture_kernel(queue, texture_pipeline, private_texture, fixed_codes, mid, out_texture,
                                                       code_count, rows, groups, slots, iterations);
    (void)run_buffer_kernel(queue, buffer_pipeline, codebooks, random_codes, mid, out_buffer,
                            code_count, rows, groups, slots, 1);
    (void)run_texture_kernel(queue, texture_pipeline, linear_texture, random_codes, mid, out_texture,
                             code_count, rows, groups, slots, 1);
    const float *buffer_values = out_buffer.contents;
    const float *texture_values = out_texture.contents;
    float max_abs_linear = 0.0f;
    for (NSUInteger index = 0; index < (NSUInteger)slots * rows; index++) {
        const float delta = fabsf(buffer_values[index] - texture_values[index]);
        if (delta > max_abs_linear) max_abs_linear = delta;
    }
    (void)run_texture_kernel(queue, texture_pipeline, private_texture, random_codes, mid, out_texture,
                             code_count, rows, groups, slots, 1);
    float max_abs_private = 0.0f;
    for (NSUInteger index = 0; index < (NSUInteger)slots * rows; index++) {
        const float delta = fabsf(buffer_values[index] - texture_values[index]);
        if (delta > max_abs_private) max_abs_private = delta;
    }
    printf("texture_cache K=%u rows=%u groups=%u slots=%u it=%u codebook=%.2fMiB codes=%.2fMiB "
           "buffer_random=%.4fms buffer_fixed0=%.4fms fixed_speed=%.3fx "
           "tex_linear_random=%.4fms tex_shared_random=%.4fms tex_private_random=%.4fms "
           "private_vs_buffer=%.3fx private_fixed0=%.4fms private_fixed_speed=%.3fx "
           "max_abs_linear=%.6g max_abs_private=%.6g\n",
           code_count, rows, groups, slots, iterations,
           (double)codebook_bytes / 1048576.0, (double)code_bytes / 1048576.0,
           buffer_random_ms, buffer_fixed_ms, buffer_random_ms / buffer_fixed_ms,
           linear_random_ms, shared_random_ms, private_random_ms,
           buffer_random_ms / private_random_ms, private_fixed_ms, private_random_ms / private_fixed_ms,
           max_abs_linear, max_abs_private);
    if (max_abs_linear > 1.0e-3f || max_abs_private > 1.0e-3f) {
        fprintf(stderr, "texture correctness mismatch K=%u linear=%g private=%g\n",
                code_count, max_abs_linear, max_abs_private);
        return 0;
    }
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
        if (!device) {
            fprintf(stderr, "no Metal device\n");
            return 1;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kMetalSource]
                                                       options:nil
                                                         error:&error];
        if (!library) {
            NSLog(@"library creation failed: %@", error);
            return 1;
        }
        id<MTLComputePipelineState> buffer_pipeline = make_pipeline(device, library, @"direct16_buffer");
        id<MTLComputePipelineState> texture_pipeline = make_pipeline(device, library, @"direct16_texture");
        if (!buffer_pipeline || !texture_pipeline) return 1;

        printf("d8f direct-down M1 texture-cache steelman canary device=%s\n", device.name.UTF8String);
        if (!run_case(device, queue, buffer_pipeline, texture_pipeline, 1024, rows, groups, slots, iterations)) return 1;
        if (!run_case(device, queue, buffer_pipeline, texture_pipeline, 2048, rows, groups, slots, iterations)) return 1;
        if (!run_case(device, queue, buffer_pipeline, texture_pipeline, 4096, rows, groups, slots, iterations)) return 1;
    }
    return 0;
}
