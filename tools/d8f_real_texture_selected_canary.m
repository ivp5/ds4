#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ds4_d8f_reader.h"

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
"static inline float dot8_texture_buffer(texture_buffer<half, access::read> codebooks, uint max_k, uint slot, uint code, float mid0, float mid1, float mid2, float mid3, float mid4, float mid5, float mid6, float mid7) {\n"
"  const uint base = (slot * max_k + code) * 2u;\n"
"  const half4 lo = codebooks.read(base);\n"
"  const half4 hi = codebooks.read(base + 1u);\n"
"  return float(lo.x) * mid0 + float(lo.y) * mid1 + float(lo.z) * mid2 + float(lo.w) * mid3 +\n"
"         float(hi.x) * mid4 + float(hi.y) * mid5 + float(hi.z) * mid6 + float(hi.w) * mid7;\n"
"}\n"
"\n"
"kernel void selected_buffer(\n"
"  device const half *codebooks [[buffer(0)]],\n"
"  device const ushort *codes [[buffer(1)]],\n"
"  device const float *mid [[buffer(2)]],\n"
"  device float *out [[buffer(3)]],\n"
"  constant uint &max_k [[buffer(4)]],\n"
"  constant uint &rows [[buffer(5)]],\n"
"  constant uint &groups [[buffer(6)]],\n"
"  constant uint &slots [[buffer(7)]],\n"
"  threadgroup float *partial [[threadgroup(0)]],\n"
"  uint tid [[thread_index_in_threadgroup]],\n"
"  ushort lane [[thread_index_in_simdgroup]],\n"
"  ushort simdgroup [[simdgroup_index_in_threadgroup]],\n"
"  uint row_tile [[threadgroup_position_in_grid]]) {\n"
"  const uint row_base = row_tile << 4;\n"
"  float acc[16];\n"
"  for (uint row_offset = 0; row_offset < 16u; row_offset++) acc[row_offset] = 0.0f;\n"
"  for (uint slot = 0u; slot < slots; slot++) {\n"
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
"        const ushort code = codes[(ulong(slot) * ulong(rows) + ulong(row)) * ulong(groups) + ulong(group)];\n"
"        if (code >= max_k) continue;\n"
"        device const half *codebook = codebooks + (ulong(slot) * ulong(max_k) + ulong(code)) * 8ul;\n"
"        acc[row_offset] += dot8_device(codebook, mid0, mid1, mid2, mid3, mid4, mid5, mid6, mid7);\n"
"      }\n"
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
"      out[row] = total;\n"
"    }\n"
"  }\n"
"}\n"
"\n"
"kernel void selected_texture(\n"
"  texture2d<half, access::read> codebooks [[texture(0)]],\n"
"  device const ushort *codes [[buffer(0)]],\n"
"  device const float *mid [[buffer(1)]],\n"
"  device float *out [[buffer(2)]],\n"
"  constant uint &max_k [[buffer(3)]],\n"
"  constant uint &rows [[buffer(4)]],\n"
"  constant uint &groups [[buffer(5)]],\n"
"  constant uint &slots [[buffer(6)]],\n"
"  threadgroup float *partial [[threadgroup(0)]],\n"
"  uint tid [[thread_index_in_threadgroup]],\n"
"  ushort lane [[thread_index_in_simdgroup]],\n"
"  ushort simdgroup [[simdgroup_index_in_threadgroup]],\n"
"  uint row_tile [[threadgroup_position_in_grid]]) {\n"
"  const uint row_base = row_tile << 4;\n"
"  float acc[16];\n"
"  for (uint row_offset = 0; row_offset < 16u; row_offset++) acc[row_offset] = 0.0f;\n"
"  for (uint slot = 0u; slot < slots; slot++) {\n"
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
"        const ushort code = codes[(ulong(slot) * ulong(rows) + ulong(row)) * ulong(groups) + ulong(group)];\n"
"        if (code >= max_k) continue;\n"
"        acc[row_offset] += dot8_texture(codebooks, slot, uint(code), mid0, mid1, mid2, mid3, mid4, mid5, mid6, mid7);\n"
"      }\n"
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
"      out[row] = total;\n"
"    }\n"
"  }\n"
"}\n"
"\n"
"kernel void selected_texture_buffer(\n"
"  texture_buffer<half, access::read> codebooks [[texture(0)]],\n"
"  device const ushort *codes [[buffer(0)]],\n"
"  device const float *mid [[buffer(1)]],\n"
"  device float *out [[buffer(2)]],\n"
"  constant uint &max_k [[buffer(3)]],\n"
"  constant uint &rows [[buffer(4)]],\n"
"  constant uint &groups [[buffer(5)]],\n"
"  constant uint &slots [[buffer(6)]],\n"
"  threadgroup float *partial [[threadgroup(0)]],\n"
"  uint tid [[thread_index_in_threadgroup]],\n"
"  ushort lane [[thread_index_in_simdgroup]],\n"
"  ushort simdgroup [[simdgroup_index_in_threadgroup]],\n"
"  uint row_tile [[threadgroup_position_in_grid]]) {\n"
"  const uint row_base = row_tile << 4;\n"
"  float acc[16];\n"
"  for (uint row_offset = 0; row_offset < 16u; row_offset++) acc[row_offset] = 0.0f;\n"
"  for (uint slot = 0u; slot < slots; slot++) {\n"
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
"        const ushort code = codes[(ulong(slot) * ulong(rows) + ulong(row)) * ulong(groups) + ulong(group)];\n"
"        if (code >= max_k) continue;\n"
"        acc[row_offset] += dot8_texture_buffer(codebooks, max_k, slot, uint(code), mid0, mid1, mid2, mid3, mid4, mid5, mid6, mid7);\n"
"      }\n"
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
"      out[row] = total;\n"
"    }\n"
"  }\n"
"}\n";

static float f16_to_f32(uint16_t bits) {
    __fp16 value;
    memcpy(&value, &bits, sizeof(value));
    return (float)value;
}

static uint16_t load_u16(const uint8_t *bytes) {
    return (uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8);
}

static int parse_experts(const char *text, uint32_t *experts, uint32_t *count) {
    *count = 0;
    const char *cursor = text;
    while (cursor && *cursor) {
        char *end = NULL;
        const unsigned long value = strtoul(cursor, &end, 10);
        if (end == cursor || value >= 256ul || *count >= 6u) return 0;
        experts[(*count)++] = (uint32_t)value;
        if (*end == '\0') break;
        if (*end != ',') return 0;
        cursor = end + 1;
    }
    return *count > 0;
}

static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString *name) {
    NSError *error = nil;
    id<MTLFunction> function = [library newFunctionWithName:name];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) NSLog(@"pipeline %@ failed: %@", name, error);
    return pipeline;
}

static id<MTLTexture> make_buffer_backed_texture(id<MTLBuffer> texel_buffer, uint32_t width, uint32_t height) {
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                          width:width
                                                                                         height:height
                                                                                      mipmapped:NO];
    descriptor.usage = MTLTextureUsageShaderRead;
    return [texel_buffer newTextureWithDescriptor:descriptor
                                           offset:0
                                      bytesPerRow:(NSUInteger)width * 4u * sizeof(uint16_t)];
}

static id<MTLTexture> make_texture_buffer(id<MTLBuffer> texel_buffer, uint32_t texel_count) {
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                              width:texel_count
                                                                                    resourceOptions:MTLResourceStorageModeShared
                                                                                              usage:MTLTextureUsageShaderRead];
    return [texel_buffer newTextureWithDescriptor:descriptor offset:0 bytesPerRow:(NSUInteger)texel_count * 4u * sizeof(uint16_t)];
}

static double run_buffer(id<MTLCommandQueue> queue,
                         id<MTLComputePipelineState> pipeline,
                         id<MTLBuffer> codebooks,
                         id<MTLBuffer> codes,
                         id<MTLBuffer> mid,
                         id<MTLBuffer> out,
                         uint32_t max_k,
                         uint32_t rows,
                         uint32_t groups,
                         uint32_t slots,
                         uint32_t rounds) {
    double best = 1.0e30;
    for (uint32_t round = 0; round < rounds; round++) {
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:codebooks offset:0 atIndex:0];
        [encoder setBuffer:codes offset:0 atIndex:1];
        [encoder setBuffer:mid offset:0 atIndex:2];
        [encoder setBuffer:out offset:0 atIndex:3];
        [encoder setBytes:&max_k length:sizeof(max_k) atIndex:4];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:5];
        [encoder setBytes:&groups length:sizeof(groups) atIndex:6];
        [encoder setBytes:&slots length:sizeof(slots) atIndex:7];
        [encoder setThreadgroupMemoryLength:16u * 8u * sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake((rows + 15u) >> 4, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        const double ms = (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0;
        if (ms > 0.0 && ms < best) best = ms;
    }
    return best;
}

static double run_texture(id<MTLCommandQueue> queue,
                          id<MTLComputePipelineState> pipeline,
                          id<MTLTexture> codebooks,
                          id<MTLBuffer> codes,
                          id<MTLBuffer> mid,
                          id<MTLBuffer> out,
                          uint32_t max_k,
                          uint32_t rows,
                          uint32_t groups,
                          uint32_t slots,
                          uint32_t rounds) {
    double best = 1.0e30;
    for (uint32_t round = 0; round < rounds; round++) {
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setTexture:codebooks atIndex:0];
        [encoder setBuffer:codes offset:0 atIndex:0];
        [encoder setBuffer:mid offset:0 atIndex:1];
        [encoder setBuffer:out offset:0 atIndex:2];
        [encoder setBytes:&max_k length:sizeof(max_k) atIndex:3];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
        [encoder setBytes:&groups length:sizeof(groups) atIndex:5];
        [encoder setBytes:&slots length:sizeof(slots) atIndex:6];
        [encoder setThreadgroupMemoryLength:16u * 8u * sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake((rows + 15u) >> 4, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        const double ms = (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0;
        if (ms > 0.0 && ms < best) best = ms;
    }
    return best;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "usage: %s layer.d8f expert_csv [rows=4096] [rounds=80]\n", argv[0]);
            return 2;
        }
        const char *path = argv[1];
        uint32_t experts[6] = {0};
        uint32_t slots = 0;
        if (!parse_experts(argv[2], experts, &slots)) {
            fprintf(stderr, "bad expert csv: %s\n", argv[2]);
            return 2;
        }
        uint32_t rows = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10) : 4096u;
        uint32_t rounds = argc > 4 ? (uint32_t)strtoul(argv[4], NULL, 10) : 80u;
        if (rows == 0 || rows > 4096u) rows = 4096u;
        if (rounds == 0) rounds = 1u;

        ds4_d8f_file file;
        if (!ds4_d8f_open(path, &file)) return 1;
        const uint32_t groups = 256u;
        const uint32_t in_dim = groups * 8u;
        ds4_d8f_record down[6];
        ds4_d8f_native_code_record native[6];
        uint32_t max_k = 0;
        for (uint32_t slot = 0; slot < slots; slot++) {
            if (!ds4_d8f_get_record(&file, DS4_D8F_DOWN, experts[slot], &down[slot]) ||
                !ds4_d8f_get_down_native_codes(&file, experts[slot], &native[slot])) {
                fprintf(stderr, "missing down/native expert=%u\n", experts[slot]);
                ds4_d8f_close(&file);
                return 1;
            }
            if (native[slot].rows < rows || native[slot].groups != groups) {
                fprintf(stderr, "bad native dims expert=%u rows=%u groups=%u\n",
                        experts[slot], native[slot].rows, native[slot].groups);
                ds4_d8f_close(&file);
                return 1;
            }
            if (down[slot].k > max_k) max_k = down[slot].k;
        }
        if (max_k == 0u) {
            ds4_d8f_close(&file);
            return 1;
        }
        const NSUInteger codebook_bytes = (NSUInteger)slots * max_k * 8u * sizeof(uint16_t);
        const NSUInteger texel_bytes = (NSUInteger)slots * max_k * 2u * 4u * sizeof(uint16_t);
        const NSUInteger code_bytes = (NSUInteger)slots * rows * groups * sizeof(uint16_t);
        const NSUInteger mid_bytes = (NSUInteger)slots * in_dim * sizeof(float);
        const NSUInteger out_bytes = (NSUInteger)rows * sizeof(float);
        uint16_t *codebook_host = (uint16_t *)calloc(1, codebook_bytes);
        uint16_t *texel_host = (uint16_t *)calloc(1, texel_bytes);
        uint16_t *code_host = (uint16_t *)malloc(code_bytes);
        float *mid_host = (float *)malloc(mid_bytes);
        float *ref = (float *)calloc(rows, sizeof(float));
        if (!codebook_host || !texel_host || !code_host || !mid_host || !ref) {
            free(codebook_host); free(texel_host); free(code_host); free(mid_host); free(ref);
            ds4_d8f_close(&file);
            return 1;
        }
        for (uint32_t slot = 0; slot < slots; slot++) {
            for (uint32_t code = 0; code < down[slot].k; code++) {
                const uint8_t *source = file.map + down[slot].codebook_offset + (uint64_t)code * 16u;
                for (uint32_t lane = 0; lane < 8u; lane++) {
                    const uint16_t bits = load_u16(source + (uint64_t)lane * 2u);
                    codebook_host[((uint64_t)slot * max_k + code) * 8u + lane] = bits;
                    texel_host[((uint64_t)slot * max_k * 2u + code * 2u + lane / 4u) * 4u + lane % 4u] = bits;
                }
            }
            memcpy(code_host + (uint64_t)slot * rows * groups,
                   file.map + native[slot].offset,
                   (size_t)rows * groups * sizeof(uint16_t));
            for (uint32_t index = 0; index < in_dim; index++) {
                float value = 0.50f * sinf((float)(index + slot * 17u) * 0.011f) +
                              0.25f * cosf((float)(index + slot * 29u) * 0.023f);
                if ((down[slot].flags & 1u) && down[slot].scale_offset && down[slot].scale_bytes >= (index + 1u) * 2u) {
                    value *= f16_to_f32(load_u16(file.map + down[slot].scale_offset + (uint64_t)index * 2u));
                }
                mid_host[(uint64_t)slot * in_dim + index] = value;
            }
        }
        for (uint32_t row = 0; row < rows; row++) {
            double total = 0.0;
            for (uint32_t slot = 0; slot < slots; slot++) {
                const uint16_t *codes = code_host + (uint64_t)slot * rows * groups;
                const float *mid = mid_host + (uint64_t)slot * in_dim;
                for (uint32_t group = 0; group < groups; group++) {
                    const uint32_t code = codes[(uint64_t)row * groups + group];
                    const uint16_t *cb = codebook_host + ((uint64_t)slot * max_k + code) * 8u;
                    const uint32_t mid_base = group << 3;
                    for (uint32_t lane = 0; lane < 8u; lane++) {
                        total += (double)f16_to_f32(cb[lane]) * (double)mid[mid_base + lane];
                    }
                }
            }
            ref[row] = (float)total;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kMetalSource]
                                                       options:nil
                                                         error:&error];
        if (!library) {
            NSLog(@"library creation failed: %@", error);
            free(codebook_host); free(texel_host); free(code_host); free(mid_host); free(ref);
            ds4_d8f_close(&file);
            return 1;
        }
        id<MTLComputePipelineState> buffer_pipeline = make_pipeline(device, library, @"selected_buffer");
        id<MTLComputePipelineState> texture_pipeline = make_pipeline(device, library, @"selected_texture");
        id<MTLComputePipelineState> texture_buffer_pipeline = make_pipeline(device, library, @"selected_texture_buffer");
        if (!buffer_pipeline || !texture_pipeline || !texture_buffer_pipeline) {
            free(codebook_host); free(texel_host); free(code_host); free(mid_host); free(ref);
            ds4_d8f_close(&file);
            return 1;
        }
        id<MTLBuffer> codebook_buf = [device newBufferWithBytes:codebook_host length:codebook_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> texel_buf = [device newBufferWithBytes:texel_host length:texel_bytes options:MTLResourceStorageModeShared];
        id<MTLTexture> texture = make_buffer_backed_texture(texel_buf, max_k * 2u, slots);
        id<MTLTexture> texture_buffer = make_texture_buffer(texel_buf, slots * max_k * 2u);
        id<MTLBuffer> code_buf = [device newBufferWithBytes:code_host length:code_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> mid_buf = [device newBufferWithBytes:mid_host length:mid_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buffer = [device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_texture = [device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_texture_buffer = [device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        if (!codebook_buf || !texel_buf || !texture || !texture_buffer || !code_buf || !mid_buf || !out_buffer || !out_texture || !out_texture_buffer) {
            fprintf(stderr, "Metal allocation failed\n");
            free(codebook_host); free(texel_host); free(code_host); free(mid_host); free(ref);
            ds4_d8f_close(&file);
            return 1;
        }
        (void)run_buffer(queue, buffer_pipeline, codebook_buf, code_buf, mid_buf, out_buffer, max_k, rows, groups, slots, 1);
        (void)run_texture(queue, texture_pipeline, texture, code_buf, mid_buf, out_texture, max_k, rows, groups, slots, 1);
        (void)run_texture(queue, texture_buffer_pipeline, texture_buffer, code_buf, mid_buf, out_texture_buffer, max_k, rows, groups, slots, 1);
        const float *buffer_values = out_buffer.contents;
        const float *texture_values = out_texture.contents;
        const float *texture_buffer_values = out_texture_buffer.contents;
        float max_abs_buffer = 0.0f;
        float max_abs_texture = 0.0f;
        float max_abs_texture_buffer = 0.0f;
        float max_abs_buffer_texture = 0.0f;
        float max_abs_buffer_texture_buffer = 0.0f;
        for (uint32_t row = 0; row < rows; row++) {
            const float db = fabsf(buffer_values[row] - ref[row]);
            const float dt = fabsf(texture_values[row] - ref[row]);
            const float dtb = fabsf(texture_buffer_values[row] - ref[row]);
            const float dbt = fabsf(buffer_values[row] - texture_values[row]);
            const float dbtb = fabsf(buffer_values[row] - texture_buffer_values[row]);
            if (db > max_abs_buffer) max_abs_buffer = db;
            if (dt > max_abs_texture) max_abs_texture = dt;
            if (dtb > max_abs_texture_buffer) max_abs_texture_buffer = dtb;
            if (dbt > max_abs_buffer_texture) max_abs_buffer_texture = dbt;
            if (dbtb > max_abs_buffer_texture_buffer) max_abs_buffer_texture_buffer = dbtb;
        }
        const char *measure_order = getenv("DS4_TEXTURE_CANARY_ORDER");
        if (!measure_order || !measure_order[0]) measure_order = "B2T";
        double buffer_ms = 0.0;
        double texture_ms = 0.0;
        double texture_buffer_ms = 0.0;
        for (const char *cursor = measure_order; *cursor; cursor++) {
            if (*cursor == 'B' && buffer_ms == 0.0) {
                buffer_ms = run_buffer(queue, buffer_pipeline, codebook_buf, code_buf, mid_buf, out_buffer, max_k, rows, groups, slots, rounds);
            } else if (*cursor == '2' && texture_ms == 0.0) {
                texture_ms = run_texture(queue, texture_pipeline, texture, code_buf, mid_buf, out_texture, max_k, rows, groups, slots, rounds);
            } else if (*cursor == 'T' && texture_buffer_ms == 0.0) {
                texture_buffer_ms = run_texture(queue, texture_buffer_pipeline, texture_buffer, code_buf, mid_buf, out_texture_buffer, max_k, rows, groups, slots, rounds);
            }
        }
        if (buffer_ms == 0.0) buffer_ms = run_buffer(queue, buffer_pipeline, codebook_buf, code_buf, mid_buf, out_buffer, max_k, rows, groups, slots, rounds);
        if (texture_ms == 0.0) texture_ms = run_texture(queue, texture_pipeline, texture, code_buf, mid_buf, out_texture, max_k, rows, groups, slots, rounds);
        if (texture_buffer_ms == 0.0) texture_buffer_ms = run_texture(queue, texture_buffer_pipeline, texture_buffer, code_buf, mid_buf, out_texture_buffer, max_k, rows, groups, slots, rounds);
        printf("real_texture_selected file=%s experts=%s rows=%u slots=%u max_k=%u rounds=%u "
               "order=%s padded_codebook=%.3fMiB codes=%.3fMiB buffer=%.4fms tex_linear2d=%.4fms tex_buffer=%.4fms "
               "speedup_2d=%.3fx speedup_tb=%.3fx max_abs_buffer=%.6g max_abs_texture=%.6g "
               "max_abs_texture_buffer=%.6g max_abs_buf_tex=%.6g max_abs_buf_tb=%.6g\n",
               path, argv[2], rows, slots, max_k, rounds, measure_order,
               (double)codebook_bytes / 1048576.0,
               (double)code_bytes / 1048576.0,
               buffer_ms, texture_ms, texture_buffer_ms,
               buffer_ms / texture_ms, buffer_ms / texture_buffer_ms,
               max_abs_buffer, max_abs_texture, max_abs_texture_buffer,
               max_abs_buffer_texture, max_abs_buffer_texture_buffer);
        free(codebook_host); free(texel_host); free(code_host); free(mid_host); free(ref);
        ds4_d8f_close(&file);
    }
    return 0;
}
