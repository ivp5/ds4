#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    kGroups = 256,
    kBlock = 8,
    kInputDim = kGroups * kBlock,
    kCodeCount = 4096,
    kBits = 12
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

static uint32_t synthetic_code(uint32_t group, uint32_t row) {
    return (uint32_t)((group * 17u + row * 37u + (group >> 2u)) & (kCodeCount - 1u));
}

static NSString *kernel_source(void) {
    return
        @"#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "static inline uint decode12(const device uchar *packed, uint packed_bytes, uint group, uint row) {\n"
         "  uint bit_offset = row * 12u;\n"
         "  uint byte_index = bit_offset >> 3u;\n"
         "  uint shift = bit_offset & 7u;\n"
         "  const device uchar *base = packed + group * packed_bytes + byte_index;\n"
         "  uint word = uint(base[0]) | (uint(base[1]) << 8u) | (uint(base[2]) << 16u);\n"
         "  return (word >> shift) & 4095u;\n"
         "}\n"
         "kernel void vqd8_lut_packed(const device half *x [[buffer(0)]],\n"
         "                            const device half *codebook [[buffer(1)]],\n"
         "                            const device uchar *packed [[buffer(2)]],\n"
         "                            device half *out [[buffer(3)]],\n"
         "                            constant uint &rows [[buffer(4)]],\n"
         "                            constant uint &packed_bytes [[buffer(5)]],\n"
         "                            uint row [[threadgroup_position_in_grid]],\n"
         "                            uint lane [[thread_position_in_threadgroup]]) {\n"
         "  if (row >= rows) return;\n"
         "  threadgroup float partial[4];\n"
         "  float acc = 0.0f;\n"
         "  for (uint index = lane; index < 2048u; index += 128u) {\n"
         "    uint group = index >> 3u;\n"
         "    uint dim = index & 7u;\n"
         "    uint code = decode12(packed, packed_bytes, group, row);\n"
         "    acc += float(x[index]) * float(codebook[dim * 4096u + code]);\n"
         "  }\n"
         "  acc = simd_sum(acc);\n"
         "  uint simd_lane = lane & 31u;\n"
         "  uint simd_id = lane >> 5u;\n"
         "  if (simd_lane == 0u) partial[simd_id] = acc;\n"
         "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         "  float total = lane < 4u ? partial[lane] : 0.0f;\n"
         "  total = simd_sum(total);\n"
         "  if (lane == 0u) out[row] = half(total);\n"
         "}\n";
}

static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device) {
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:kernel_source() options:nil error:&error];
    if (!library) {
        fprintf(stderr, "newLibraryWithSource failed: %s\n", error.localizedDescription.UTF8String);
        exit(1);
    }
    id<MTLFunction> function = [library newFunctionWithName:@"vqd8_lut_packed"];
    if (!function) {
        fprintf(stderr, "missing vqd8_lut_packed function\n");
        exit(1);
    }
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        fprintf(stderr, "newComputePipelineState failed: %s\n", error.localizedDescription.UTF8String);
        exit(1);
    }
    return pipeline;
}

static void run_once(id<MTLCommandQueue> queue,
                     id<MTLComputePipelineState> pipeline,
                     id<MTLBuffer> x,
                     id<MTLBuffer> codebook,
                     id<MTLBuffer> packed,
                     id<MTLBuffer> out,
                     uint32_t rows,
                     uint32_t packed_bytes) {
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:x offset:0 atIndex:0];
    [encoder setBuffer:codebook offset:0 atIndex:1];
    [encoder setBuffer:packed offset:0 atIndex:2];
    [encoder setBuffer:out offset:0 atIndex:3];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
    [encoder setBytes:&packed_bytes length:sizeof(packed_bytes) atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        uint32_t rows = argc >= 2 ? (uint32_t)strtoul(argv[1], NULL, 10) : 4096u;
        uint32_t rounds = argc >= 3 ? (uint32_t)strtoul(argv[2], NULL, 10) : 20u;
        if (rows == 0u || rows > 8192u) rows = 4096u;
        if (rounds == 0u) rounds = 20u;
        uint32_t packed_bytes = (rows * kBits + 7u) / 8u + 1u;
        uint16_t *x_f16 = (uint16_t *)malloc((size_t)kInputDim * sizeof(uint16_t));
        uint16_t *codebook_f16 = (uint16_t *)malloc((size_t)kBlock * kCodeCount * sizeof(uint16_t));
        uint8_t *packed_indices = (uint8_t *)calloc((size_t)kGroups * packed_bytes, sizeof(uint8_t));
        float *reference = (float *)calloc(rows, sizeof(float));
        if (!x_f16 || !codebook_f16 || !packed_indices || !reference) {
            fprintf(stderr, "allocation failed\n");
            return 1;
        }
        for (uint32_t index = 0; index < kInputDim; ++index) {
            float value = 0.31f * sinf((float)index * 0.017f) + 0.13f * cosf((float)index * 0.031f);
            x_f16[index] = f32_to_f16(value);
        }
        for (uint32_t dim = 0; dim < kBlock; ++dim) {
            for (uint32_t code = 0; code < kCodeCount; ++code) {
                float value = 0.19f * sinf((float)(dim * 17u + code * 5u) * 0.071f);
                codebook_f16[(size_t)dim * kCodeCount + code] = f32_to_f16(value);
            }
        }
        for (uint32_t group = 0; group < kGroups; ++group) {
            for (uint32_t row = 0; row < rows; ++row) {
                uint32_t code = synthetic_code(group, row);
                uint32_t bit_offset = row * kBits;
                uint32_t byte_index = bit_offset >> 3u;
                uint32_t shift = bit_offset & 7u;
                uint32_t encoded = code << shift;
                uint8_t *slot = packed_indices + (size_t)group * packed_bytes + byte_index;
                slot[0] = (uint8_t)(slot[0] | (uint8_t)(encoded & 0xffu));
                slot[1] = (uint8_t)(slot[1] | (uint8_t)((encoded >> 8u) & 0xffu));
                slot[2] = (uint8_t)(slot[2] | (uint8_t)((encoded >> 16u) & 0xffu));
            }
        }
        for (uint32_t row = 0; row < rows; ++row) {
            double sum = 0.0;
            for (uint32_t group = 0; group < kGroups; ++group) {
                uint32_t code = synthetic_code(group, row);
                for (uint32_t dim = 0; dim < kBlock; ++dim) {
                    uint32_t index = group * kBlock + dim;
                    sum += (double)f16_to_f32(x_f16[index]) *
                           (double)f16_to_f32(codebook_f16[(size_t)dim * kCodeCount + code]);
                }
            }
            reference[row] = (float)sum;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = device ? [device newCommandQueue] : nil;
        if (!device || !queue) {
            fprintf(stderr, "missing Metal device/queue\n");
            return 1;
        }
        id<MTLComputePipelineState> pipeline = make_pipeline(device);
        id<MTLBuffer> x_buffer = [device newBufferWithBytes:x_f16 length:(size_t)kInputDim * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> codebook_buffer = [device newBufferWithBytes:codebook_f16 length:(size_t)kBlock * kCodeCount * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> packed_buffer = [device newBufferWithBytes:packed_indices length:(size_t)kGroups * packed_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buffer = [device newBufferWithLength:(size_t)rows * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        if (!x_buffer || !codebook_buffer || !packed_buffer || !out_buffer) {
            fprintf(stderr, "MTLBuffer allocation failed\n");
            return 1;
        }
        for (uint32_t warmup = 0; warmup < 3u; ++warmup) {
            run_once(queue, pipeline, x_buffer, codebook_buffer, packed_buffer, out_buffer, rows, packed_bytes);
        }
        double start = now_seconds();
        for (uint32_t round = 0; round < rounds; ++round) {
            run_once(queue, pipeline, x_buffer, codebook_buffer, packed_buffer, out_buffer, rows, packed_bytes);
        }
        double us_per = (now_seconds() - start) * 1.0e6 / (double)rounds;
        uint16_t *got = (uint16_t *)out_buffer.contents;
        double max_abs = 0.0;
        double rms = 0.0;
        uint32_t bad = 0;
        for (uint32_t row = 0; row < rows; ++row) {
            double diff = fabs((double)f16_to_f32(got[row]) - (double)reference[row]);
            if (diff > max_abs) max_abs = diff;
            rms += diff * diff;
            if (diff > 0.125) ++bad;
        }
        rms = sqrt(rms / (double)rows);
        double packed_index_mb = (double)((size_t)kGroups * packed_bytes) / 1.0e6;
        double codebook_mb = (double)((size_t)kBlock * kCodeCount * sizeof(uint16_t)) / 1.0e6;
        fprintf(stderr,
                "metal_vqd8_lut_packed_probe: rows=%u rounds=%u groups=%u k=%u us/op=%.3f packed_index_MB=%.3f codebook_MB=%.3f bad=%u max_abs=%.6g rms=%.6g sample_ref=%.6g sample_got=%.6g\n",
                rows, rounds, kGroups, kCodeCount, us_per, packed_index_mb, codebook_mb,
                bad, max_abs, rms, (double)reference[0], (double)f16_to_f32(got[0]));
        free(reference);
        free(packed_indices);
        free(codebook_f16);
        free(x_f16);
    }
    return 0;
}
