#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>

#include "ds4_d8f_reader.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    kGroups = 256,
    kBlock = 8,
    kInputDim = kGroups * kBlock,
    kRowsMax = 4096
};

static double now_seconds(void) {
    static mach_timebase_info_data_t timebase;
    if (timebase.denom == 0) mach_timebase_info(&timebase);
    return (double)mach_absolute_time() * (double)timebase.numer / (double)timebase.denom * 1e-9;
}

static uint16_t read_u16(const uint8_t *p) {
    return (uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8);
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

static NSString *kernel_source(void) {
    return
        @"#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "static inline uint decode12(const device uchar *packed, uint group, uint row) {\n"
         "  uint block_index = row * 256u + group;\n"
         "  uint bit_offset = block_index * 12u;\n"
         "  uint byte_index = bit_offset >> 3u;\n"
         "  uint shift = bit_offset & 7u;\n"
         "  uint word = uint(packed[byte_index]) | (uint(packed[byte_index + 1u]) << 8u) |\n"
         "              (uint(packed[byte_index + 2u]) << 16u) | (uint(packed[byte_index + 3u]) << 24u);\n"
         "  return (word >> shift) & 4095u;\n"
         "}\n"
         "kernel void real_d8f_down(const device half *x [[buffer(0)]],\n"
         "                          const device half *codebook [[buffer(1)]],\n"
         "                          const device uchar *packed [[buffer(2)]],\n"
         "                          device half *out [[buffer(3)]],\n"
         "                          constant uint &rows [[buffer(4)]],\n"
         "                          uint row [[threadgroup_position_in_grid]],\n"
         "                          uint lane [[thread_position_in_threadgroup]]) {\n"
         "  if (row >= rows) return;\n"
         "  threadgroup float partial[4];\n"
         "  float acc = 0.0f;\n"
         "  for (uint index = lane; index < 2048u; index += 128u) {\n"
         "    uint group = index >> 3u;\n"
         "    uint dim = index & 7u;\n"
         "    uint code = decode12(packed, group, row);\n"
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
    id<MTLFunction> function = [library newFunctionWithName:@"real_d8f_down"];
    if (!function) {
        fprintf(stderr, "missing real_d8f_down function\n");
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
                     uint32_t bits,
                     uint32_t k) {
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:x offset:0 atIndex:0];
    [encoder setBuffer:codebook offset:0 atIndex:1];
    [encoder setBuffer:packed offset:0 atIndex:2];
    [encoder setBuffer:out offset:0 atIndex:3];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
    (void)bits;
    (void)k;
    [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "usage: %s LAYER.d8f EXPERT [rows] [rounds]\n", argv[0]);
            return 2;
        }
        const char *path = argv[1];
        uint32_t expert = (uint32_t)strtoul(argv[2], NULL, 10);
        uint32_t rows = argc >= 4 ? (uint32_t)strtoul(argv[3], NULL, 10) : kRowsMax;
        uint32_t rounds = argc >= 5 ? (uint32_t)strtoul(argv[4], NULL, 10) : 20u;
        if (rows == 0u || rows > kRowsMax) rows = kRowsMax;
        if (rounds == 0u) rounds = 20u;
        ds4_d8f_file file;
        if (!ds4_d8f_open(path, &file)) return 1;
        ds4_d8f_record record;
        if (!ds4_d8f_get_record(&file, DS4_D8F_DOWN, expert, &record)) {
            fprintf(stderr, "missing down expert %u\n", expert);
            ds4_d8f_close(&file);
            return 1;
        }
        if (record.block != kBlock || record.k != 4096u || record.bits != 12u) {
            fprintf(stderr, "unsupported record block=%u k=%u bits=%u\n", record.block, record.k, record.bits);
            ds4_d8f_close(&file);
            return 1;
        }
        uint64_t needed_blocks = (uint64_t)rows * kGroups;
        uint64_t available_blocks = ((uint64_t)record.index_bytes * 8ull) / record.bits;
        if (available_blocks < needed_blocks || record.codebook_bytes < (uint64_t)record.k * kBlock * sizeof(uint16_t)) {
            fprintf(stderr, "record too small available_blocks=%llu needed=%llu codebook_bytes=%u\n",
                    (unsigned long long)available_blocks, (unsigned long long)needed_blocks, record.codebook_bytes);
            ds4_d8f_close(&file);
            return 1;
        }
        uint16_t *x_f16 = (uint16_t *)malloc((size_t)kInputDim * sizeof(uint16_t));
        float *x_f32 = (float *)malloc((size_t)kInputDim * sizeof(float));
        uint16_t *codebook_f16 = (uint16_t *)malloc((size_t)kBlock * record.k * sizeof(uint16_t));
        float *reference = (float *)calloc(rows, sizeof(float));
        if (!x_f16 || !x_f32 || !codebook_f16 || !reference) {
            fprintf(stderr, "allocation failed\n");
            ds4_d8f_close(&file);
            return 1;
        }
        for (uint32_t index = 0; index < kInputDim; ++index) {
            float value = 0.50f * sinf((float)index * 0.011f) +
                          0.25f * cosf((float)index * 0.023f);
            if ((record.flags & 1u) != 0u && record.scale_offset && record.scale_bytes >= kInputDim * 2u) {
                value *= f16_to_f32(read_u16(file.map + record.scale_offset + (uint64_t)index * 2u));
            }
            x_f32[index] = value;
            x_f16[index] = f32_to_f16(value);
        }
        const uint8_t *src = file.map + record.codebook_offset;
        for (uint32_t code = 0; code < record.k; ++code) {
            for (uint32_t dim = 0; dim < kBlock; ++dim) {
                codebook_f16[(size_t)dim * record.k + code] = read_u16(src + ((uint64_t)code * kBlock + dim) * 2u);
            }
        }
        for (uint32_t row = 0; row < rows; ++row) {
            double sum = 0.0;
            for (uint32_t group = 0; group < kGroups; ++group) {
                uint64_t block_index = (uint64_t)row * kGroups + group;
                uint32_t code = ds4_d8f_code_at(&file, &record, block_index);
                if (code >= record.k) continue;
                for (uint32_t dim = 0; dim < kBlock; ++dim) {
                    sum += (double)x_f32[group * kBlock + dim] *
                           (double)f16_to_f32(codebook_f16[(size_t)dim * record.k + code]);
                }
            }
            reference[row] = (float)sum;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = device ? [device newCommandQueue] : nil;
        if (!device || !queue) {
            fprintf(stderr, "missing Metal device/queue\n");
            ds4_d8f_close(&file);
            return 1;
        }
        id<MTLComputePipelineState> pipeline = make_pipeline(device);
        id<MTLBuffer> x_buffer = [device newBufferWithBytes:x_f16 length:(size_t)kInputDim * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> codebook_buffer = [device newBufferWithBytes:codebook_f16 length:(size_t)kBlock * record.k * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> packed_buffer = [device newBufferWithBytes:file.map + record.index_offset length:record.index_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buffer = [device newBufferWithLength:(size_t)rows * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        if (!x_buffer || !codebook_buffer || !packed_buffer || !out_buffer) {
            fprintf(stderr, "MTLBuffer allocation failed\n");
            ds4_d8f_close(&file);
            return 1;
        }
        for (uint32_t warmup = 0; warmup < 3u; ++warmup) {
            run_once(queue, pipeline, x_buffer, codebook_buffer, packed_buffer, out_buffer, rows, record.bits, record.k);
        }
        double start = now_seconds();
        for (uint32_t round = 0; round < rounds; ++round) {
            run_once(queue, pipeline, x_buffer, codebook_buffer, packed_buffer, out_buffer, rows, record.bits, record.k);
        }
        double us_per = (now_seconds() - start) * 1.0e6 / (double)rounds;
        uint16_t *got = (uint16_t *)out_buffer.contents;
        double max_abs = 0.0, max_rel = 0.0, rms = 0.0;
        uint32_t bad = 0;
        for (uint32_t row = 0; row < rows; ++row) {
            double got_f = (double)f16_to_f32(got[row]);
            double ref = (double)reference[row];
            double diff = fabs(got_f - ref);
            double rel = diff / fmax(1.0, fabs(ref));
            if (diff > max_abs) max_abs = diff;
            if (rel > max_rel) max_rel = rel;
            rms += diff * diff;
            if (diff > 0.25 && rel > 0.025) ++bad;
        }
        rms = sqrt(rms / (double)rows);
        fprintf(stderr,
                "metal_vqd8_real_d8f_down_probe: path=%s expert=%u rows=%u rounds=%u k=%u bits=%u us/op=%.3f index_MB=%.3f codebook_MB=%.3f bad=%u max_abs=%.6g max_rel=%.6g rms=%.6g sample_ref=%.6g sample_got=%.6g\n",
                path, expert, rows, rounds, record.k, record.bits, us_per,
                (double)record.index_bytes / 1.0e6,
                (double)record.codebook_bytes / 1.0e6,
                bad, max_abs, max_rel, rms,
                (double)reference[0], (double)f16_to_f32(got[0]));
        free(reference);
        free(codebook_f16);
        free(x_f32);
        free(x_f16);
        ds4_d8f_close(&file);
    }
    return 0;
}
