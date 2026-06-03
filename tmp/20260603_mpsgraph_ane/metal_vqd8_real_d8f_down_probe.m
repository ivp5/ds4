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
    kRowsMax = 4096,
    kMaxExperts = 6,
    kCodeCount = 4096
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

static uint32_t parse_experts(const char *csv, uint32_t *experts, uint32_t max_experts) {
    if (!csv || !experts || max_experts == 0u) return 0u;
    uint32_t count = 0u;
    const char *p = csv;
    while (*p && count < max_experts) {
        char *end = NULL;
        unsigned long value = strtoul(p, &end, 10);
        if (end == p || value > 255ul) return 0u;
        experts[count++] = (uint32_t)value;
        if (*end == ',') {
            p = end + 1;
            continue;
        }
        if (*end == '\0') return count;
        return 0u;
    }
    return *p ? 0u : count;
}

static bool parse_weights(const char *csv, float *weights, uint32_t n_weights) {
    if (!weights || n_weights == 0u) return false;
    for (uint32_t index = 0; index < n_weights; ++index) weights[index] = 1.0f;
    if (!csv || !csv[0] || strcmp(csv, "-") == 0 || strcmp(csv, "unit") == 0) return true;
    const char *p = csv;
    for (uint32_t index = 0; index < n_weights; ++index) {
        char *end = NULL;
        float value = strtof(p, &end);
        if (end == p) return false;
        weights[index] = value;
        if (index + 1u == n_weights) return *end == '\0';
        if (*end != ',') return false;
        p = end + 1;
    }
    return false;
}

static void weights_csv(char *out, size_t out_bytes, const float *weights, uint32_t n_weights) {
    if (!out || out_bytes == 0u) return;
    out[0] = '\0';
    for (uint32_t index = 0; index < n_weights; ++index) {
        char item[32];
        snprintf(item, sizeof(item), "%s%.6g", index ? "," : "", (double)weights[index]);
        strlcat(out, item, out_bytes);
    }
}

static NSString *kernel_source(void) {
    return
        @"#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "static inline uint decode_var(const device uchar *packed, uint base_byte, uint bits, uint group, uint row) {\n"
         "  uint block_index = row * 256u + group;\n"
         "  uint bit_offset = block_index * bits;\n"
         "  uint byte_index = base_byte + (bit_offset >> 3u);\n"
         "  uint shift = bit_offset & 7u;\n"
         "  uint word = uint(packed[byte_index]) | (uint(packed[byte_index + 1u]) << 8u) |\n"
         "              (uint(packed[byte_index + 2u]) << 16u) | (uint(packed[byte_index + 3u]) << 24u);\n"
         "  return (word >> shift) & ((1u << bits) - 1u);\n"
         "}\n"
         "kernel void real_d8f_down(const device half *x [[buffer(0)]],\n"
         "                          const device half *codebook [[buffer(1)]],\n"
         "                          const device uchar *packed [[buffer(2)]],\n"
         "                          device half *out [[buffer(3)]],\n"
         "                          constant uint &rows [[buffer(4)]],\n"
         "                          constant uint &n_experts [[buffer(5)]],\n"
         "                          constant uint *packed_offsets [[buffer(6)]],\n"
         "                          constant float *weights [[buffer(7)]],\n"
         "                          constant uint *bits_values [[buffer(8)]],\n"
         "                          constant uint *k_values [[buffer(9)]],\n"
         "                          uint row [[threadgroup_position_in_grid]],\n"
         "                          uint lane [[thread_position_in_threadgroup]]) {\n"
         "  if (row >= rows) return;\n"
         "  threadgroup float partial[4];\n"
         "  float acc = 0.0f;\n"
         "  for (uint expert = 0u; expert < n_experts; ++expert) {\n"
         "    const device half *expert_x = x + expert * 2048u;\n"
         "    const device half *expert_codebook = codebook + expert * 8u * 4096u;\n"
         "    float route_weight = weights[expert];\n"
         "    uint packed_offset = packed_offsets[expert];\n"
         "    uint bits = bits_values[expert];\n"
         "    uint k = k_values[expert];\n"
         "    for (uint index = lane; index < 2048u; index += 128u) {\n"
         "      uint group = index >> 3u;\n"
         "      uint dim = index & 7u;\n"
         "      uint code = decode_var(packed, packed_offset, bits, group, row);\n"
         "      if (code < k) acc += route_weight * float(expert_x[index]) * float(expert_codebook[dim * 4096u + code]);\n"
         "    }\n"
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
                     id<MTLBuffer> packed_offsets,
                     id<MTLBuffer> weights,
                     id<MTLBuffer> bits_values,
                     id<MTLBuffer> k_values,
                     id<MTLBuffer> out,
                     uint32_t rows,
                     uint32_t n_experts) {
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:x offset:0 atIndex:0];
    [encoder setBuffer:codebook offset:0 atIndex:1];
    [encoder setBuffer:packed offset:0 atIndex:2];
    [encoder setBuffer:out offset:0 atIndex:3];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
    [encoder setBytes:&n_experts length:sizeof(n_experts) atIndex:5];
    [encoder setBuffer:packed_offsets offset:0 atIndex:6];
    [encoder setBuffer:weights offset:0 atIndex:7];
    [encoder setBuffer:bits_values offset:0 atIndex:8];
    [encoder setBuffer:k_values offset:0 atIndex:9];
    [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "usage: %s LAYER.d8f EXPERTS_CSV [rows] [rounds] [WEIGHTS_CSV]\n", argv[0]);
            return 2;
        }
        const char *path = argv[1];
        uint32_t experts[kMaxExperts] = {0};
        uint32_t n_experts = parse_experts(argv[2], experts, kMaxExperts);
        if (n_experts == 0u) {
            fprintf(stderr, "bad experts CSV: %s\n", argv[2]);
            return 2;
        }
        uint32_t rows = argc >= 4 ? (uint32_t)strtoul(argv[3], NULL, 10) : kRowsMax;
        uint32_t rounds = argc >= 5 ? (uint32_t)strtoul(argv[4], NULL, 10) : 20u;
        float route_weights[kMaxExperts] = {0};
        if (!parse_weights(argc >= 6 ? argv[5] : NULL, route_weights, n_experts)) {
            fprintf(stderr, "bad route weights CSV\n");
            return 2;
        }
        char route_weights_text[192];
        weights_csv(route_weights_text, sizeof(route_weights_text), route_weights, n_experts);
        if (rows == 0u || rows > kRowsMax) rows = kRowsMax;
        if (rounds == 0u) rounds = 20u;
        ds4_d8f_file file;
        if (!ds4_d8f_open(path, &file)) return 1;
        ds4_d8f_record records[kMaxExperts];
        memset(records, 0, sizeof(records));
        uint32_t packed_offsets[kMaxExperts] = {0};
        uint32_t bits_values[kMaxExperts] = {0};
        uint32_t k_values[kMaxExperts] = {0};
        uint64_t packed_total_bytes = 0u;
        uint64_t needed_blocks = (uint64_t)rows * kGroups;
        for (uint32_t slot = 0; slot < n_experts; ++slot) {
            if (!ds4_d8f_get_record(&file, DS4_D8F_DOWN, experts[slot], &records[slot])) {
                fprintf(stderr, "missing down expert %u\n", experts[slot]);
                ds4_d8f_close(&file);
                return 1;
            }
            ds4_d8f_record *record = &records[slot];
            if (record->block != kBlock || record->k == 0u || record->k > kCodeCount ||
                record->bits == 0u || record->bits > 16u) {
                fprintf(stderr, "unsupported record expert=%u block=%u k=%u bits=%u\n",
                        experts[slot], record->block, record->k, record->bits);
                ds4_d8f_close(&file);
                return 1;
            }
            uint64_t available_blocks = ((uint64_t)record->index_bytes * 8ull) / record->bits;
            if (available_blocks < needed_blocks || record->codebook_bytes < (uint64_t)record->k * kBlock * sizeof(uint16_t)) {
                fprintf(stderr, "record too small expert=%u available_blocks=%llu needed=%llu codebook_bytes=%u\n",
                        experts[slot], (unsigned long long)available_blocks, (unsigned long long)needed_blocks, record->codebook_bytes);
                ds4_d8f_close(&file);
                return 1;
            }
            if (packed_total_bytes > UINT32_MAX || packed_total_bytes + record->index_bytes > UINT32_MAX) {
                fprintf(stderr, "packed index arena too large\n");
                ds4_d8f_close(&file);
                return 1;
            }
            packed_offsets[slot] = (uint32_t)packed_total_bytes;
            bits_values[slot] = record->bits;
            k_values[slot] = record->k;
            packed_total_bytes += record->index_bytes;
        }
        uint16_t *x_f16 = (uint16_t *)malloc((size_t)n_experts * kInputDim * sizeof(uint16_t));
        float *x_f32 = (float *)malloc((size_t)n_experts * kInputDim * sizeof(float));
        uint16_t *codebook_f16 = (uint16_t *)calloc((size_t)n_experts * kBlock * kCodeCount, sizeof(uint16_t));
        uint8_t *packed_indices = (uint8_t *)malloc((size_t)packed_total_bytes);
        float *reference = (float *)calloc(rows, sizeof(float));
        if (!x_f16 || !x_f32 || !codebook_f16 || !packed_indices || !reference) {
            fprintf(stderr, "allocation failed\n");
            ds4_d8f_close(&file);
            return 1;
        }
        for (uint32_t slot = 0; slot < n_experts; ++slot) {
            const ds4_d8f_record *record = &records[slot];
            for (uint32_t index = 0; index < kInputDim; ++index) {
                float value = 0.50f * sinf((float)(index + slot * 17u) * 0.011f) +
                              0.25f * cosf((float)(index + slot * 29u) * 0.023f);
                if ((record->flags & 1u) != 0u && record->scale_offset && record->scale_bytes >= kInputDim * 2u) {
                    value *= f16_to_f32(read_u16(file.map + record->scale_offset + (uint64_t)index * 2u));
                }
                x_f32[(size_t)slot * kInputDim + index] = value;
                x_f16[(size_t)slot * kInputDim + index] = f32_to_f16(value);
            }
            const uint8_t *src = file.map + record->codebook_offset;
            for (uint32_t code = 0; code < record->k; ++code) {
                for (uint32_t dim = 0; dim < kBlock; ++dim) {
                    codebook_f16[(size_t)slot * kBlock * kCodeCount + (size_t)dim * kCodeCount + code] =
                        read_u16(src + ((uint64_t)code * kBlock + dim) * 2u);
                }
            }
            memcpy(packed_indices + packed_offsets[slot], file.map + record->index_offset, record->index_bytes);
        }
        for (uint32_t row = 0; row < rows; ++row) {
            double total = 0.0;
            for (uint32_t slot = 0; slot < n_experts; ++slot) {
                const ds4_d8f_record *record = &records[slot];
                double sum = 0.0;
                for (uint32_t group = 0; group < kGroups; ++group) {
                    uint64_t block_index = (uint64_t)row * kGroups + group;
                    uint32_t code = ds4_d8f_code_at(&file, record, block_index);
                    if (code >= record->k) continue;
                    for (uint32_t dim = 0; dim < kBlock; ++dim) {
                        sum += (double)x_f32[(size_t)slot * kInputDim + group * kBlock + dim] *
                               (double)f16_to_f32(codebook_f16[(size_t)slot * kBlock * kCodeCount + (size_t)dim * kCodeCount + code]);
                    }
                }
                total += (double)route_weights[slot] * sum;
            }
            reference[row] = (float)total;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = device ? [device newCommandQueue] : nil;
        if (!device || !queue) {
            fprintf(stderr, "missing Metal device/queue\n");
            ds4_d8f_close(&file);
            return 1;
        }
        id<MTLComputePipelineState> pipeline = make_pipeline(device);
        id<MTLBuffer> x_buffer = [device newBufferWithBytes:x_f16 length:(size_t)n_experts * kInputDim * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> codebook_buffer = [device newBufferWithBytes:codebook_f16 length:(size_t)n_experts * kBlock * kCodeCount * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> packed_buffer = [device newBufferWithBytes:packed_indices length:(size_t)packed_total_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> offsets_buffer = [device newBufferWithBytes:packed_offsets length:(size_t)n_experts * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> weights_buffer = [device newBufferWithBytes:route_weights length:(size_t)n_experts * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bits_buffer = [device newBufferWithBytes:bits_values length:(size_t)n_experts * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> k_buffer = [device newBufferWithBytes:k_values length:(size_t)n_experts * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buffer = [device newBufferWithLength:(size_t)rows * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        if (!x_buffer || !codebook_buffer || !packed_buffer || !offsets_buffer || !weights_buffer || !bits_buffer || !k_buffer || !out_buffer) {
            fprintf(stderr, "MTLBuffer allocation failed\n");
            ds4_d8f_close(&file);
            return 1;
        }
        for (uint32_t warmup = 0; warmup < 3u; ++warmup) {
            run_once(queue, pipeline, x_buffer, codebook_buffer, packed_buffer, offsets_buffer, weights_buffer,
                     bits_buffer, k_buffer, out_buffer, rows, n_experts);
        }
        double start = now_seconds();
        for (uint32_t round = 0; round < rounds; ++round) {
            run_once(queue, pipeline, x_buffer, codebook_buffer, packed_buffer, offsets_buffer, weights_buffer,
                     bits_buffer, k_buffer, out_buffer, rows, n_experts);
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
        char expert_csv[128];
        expert_csv[0] = '\0';
        for (uint32_t slot = 0; slot < n_experts; ++slot) {
            char item[24];
            snprintf(item, sizeof(item), "%s%u", slot ? "," : "", experts[slot]);
            strlcat(expert_csv, item, sizeof(expert_csv));
        }
        char format_csv[192];
        format_csv[0] = '\0';
        for (uint32_t slot = 0; slot < n_experts; ++slot) {
            char item[32];
            snprintf(item, sizeof(item), "%s%u/k%ub%u", slot ? "," : "", experts[slot], k_values[slot], bits_values[slot]);
            strlcat(format_csv, item, sizeof(format_csv));
        }
        double index_mb = (double)packed_total_bytes / 1.0e6;
        double codebook_mb = (double)((size_t)n_experts * kBlock * kCodeCount * sizeof(uint16_t)) / 1.0e6;
        fprintf(stderr,
                "metal_vqd8_real_d8f_down_probe: path=%s experts=%s formats=%s route_weights=%s nsel=%u rows=%u rounds=%u us/op=%.3f index_MB=%.3f codebook_MB=%.3f bad=%u max_abs=%.6g max_rel=%.6g rms=%.6g sample_ref=%.6g sample_got=%.6g\n",
                path, expert_csv, format_csv, route_weights_text, n_experts, rows, rounds, us_per,
                index_mb, codebook_mb, bad, max_abs, max_rel, rms,
                (double)reference[0], (double)f16_to_f32(got[0]));
        free(reference);
        free(packed_indices);
        free(codebook_f16);
        free(x_f32);
        free(x_f16);
        ds4_d8f_close(&file);
    }
    return 0;
}
