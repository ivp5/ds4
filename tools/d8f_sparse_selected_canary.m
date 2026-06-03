#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ds4_d8f_reader.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned long ulong;

static const char *kMetalSource =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct SparseRecord {\n"
"  uint unique_offset;\n"
"  uint unique_count;\n"
"  uint prefix_offset;\n"
"  uint inverse_offset;\n"
"  uint score_offset;\n"
"  uint sidecar_rank;\n"
"  uint sidecar_a_lo;\n"
"  uint sidecar_a_hi;\n"
"  ulong codebook_offset;\n"
"  ulong scale_offset;\n"
"};\n"
"\n"
"struct Args {\n"
"  uint rows;\n"
"  uint groups;\n"
"  uint in_dim;\n"
"  uint n_slots;\n"
"  uint n_tokens;\n"
"  uint mid_token_stride;\n"
"  uint mid_slot_stride;\n"
"  uint score_token_stride;\n"
"};\n"
"\n"
"inline float mid_value(device const uchar *pack, ulong scale_offset, device const float *mid, uint idx) {\n"
"  if (scale_offset == 0ul) return mid[idx];\n"
"  device const half *scale = (device const half *)(pack + scale_offset);\n"
"  return mid[idx] * float(scale[idx]);\n"
"}\n"
"\n"
"inline float dot_code(device const uchar *pack, ulong codebook_offset, ulong scale_offset, device const float *mid, uint group, uint code) {\n"
"  const uint x = group << 3;\n"
"  device const half *cb = (device const half *)(pack + codebook_offset + ulong(code) * 16ul);\n"
"  return float(cb[0]) * mid_value(pack, scale_offset, mid, x + 0u) +\n"
"         float(cb[1]) * mid_value(pack, scale_offset, mid, x + 1u) +\n"
"         float(cb[2]) * mid_value(pack, scale_offset, mid, x + 2u) +\n"
"         float(cb[3]) * mid_value(pack, scale_offset, mid, x + 3u) +\n"
"         float(cb[4]) * mid_value(pack, scale_offset, mid, x + 4u) +\n"
"         float(cb[5]) * mid_value(pack, scale_offset, mid, x + 5u) +\n"
"         float(cb[6]) * mid_value(pack, scale_offset, mid, x + 6u) +\n"
"         float(cb[7]) * mid_value(pack, scale_offset, mid, x + 7u);\n"
"}\n"
"\n"
"kernel void sparse_selected_score(\n"
"    device const ushort *unique_codes [[buffer(0)]],\n"
"    device const ushort *unique_groups [[buffer(1)]],\n"
"    device const uchar *pack [[buffer(2)]],\n"
"    device const float *mid [[buffer(3)]],\n"
"    device float *score [[buffer(4)]],\n"
"    device const SparseRecord *records [[buffer(5)]],\n"
"    constant Args &args [[buffer(6)]],\n"
"    uint3 pos [[thread_position_in_grid]]) {\n"
"  const uint unique_local = pos.x;\n"
"  const uint token = pos.y;\n"
"  const uint slot = pos.z;\n"
"  if (slot >= args.n_slots || token >= args.n_tokens) return;\n"
"  const SparseRecord rec = records[slot];\n"
"  if (unique_local >= rec.unique_count) return;\n"
"  const uint unique_index = rec.unique_offset + unique_local;\n"
"  const uint group = uint(unique_groups[unique_index]);\n"
"  const uint code = uint(unique_codes[unique_index]);\n"
"  device const float *slot_mid = mid + ulong(token) * ulong(args.mid_token_stride) + ulong(slot) * ulong(args.mid_slot_stride);\n"
"  score[ulong(token) * ulong(args.score_token_stride) + ulong(rec.score_offset) + ulong(unique_local)] =\n"
"      dot_code(pack, rec.codebook_offset, rec.scale_offset, slot_mid, group, code);\n"
"}\n"
"\n"
"kernel void sparse_selected_gather(\n"
"    device const ushort *inverse [[buffer(0)]],\n"
"    device const uint *prefix [[buffer(1)]],\n"
"    device const float *score [[buffer(2)]],\n"
"    device const SparseRecord *records [[buffer(3)]],\n"
"    device const float *sidecar_dot [[buffer(4)]],\n"
"    device const uchar *pack [[buffer(5)]],\n"
"    device float *out [[buffer(6)]],\n"
"    constant Args &args [[buffer(7)]],\n"
"    threadgroup float *partial [[threadgroup(0)]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    ushort tiisg [[thread_index_in_simdgroup]],\n"
"    ushort sgitg [[simdgroup_index_in_threadgroup]],\n"
"    uint2 pos [[threadgroup_position_in_grid]]) {\n"
"  const uint row = pos.x;\n"
"  const uint token = pos.y;\n"
"  float acc = 0.0f;\n"
"  if (row < args.rows && token < args.n_tokens && tid < args.groups) {\n"
"    const uint group = tid;\n"
"    for (uint slot = 0u; slot < args.n_slots; slot++) {\n"
"      const SparseRecord rec = records[slot];\n"
"      const uint local = uint(inverse[ulong(rec.inverse_offset) + ulong(row) * ulong(args.groups) + ulong(group)]);\n"
"      const uint prefix_index = prefix[ulong(rec.prefix_offset) + ulong(group)];\n"
"      acc += score[ulong(token) * ulong(args.score_token_stride) + ulong(rec.score_offset) + ulong(prefix_index) + ulong(local)];\n"
"    }\n"
"  }\n"
"  if (row < args.rows && token < args.n_tokens && tid == 0u) {\n"
"    for (uint slot = 0u; slot < args.n_slots; slot++) {\n"
"      const SparseRecord rec = records[slot];\n"
"      if (rec.sidecar_rank == 1u) {\n"
"        const ulong a_offset = ulong(rec.sidecar_a_lo) | (ulong(rec.sidecar_a_hi) << 32);\n"
"        device const half *a = (device const half *)(pack + a_offset);\n"
"        acc += sidecar_dot[ulong(token) * ulong(args.n_slots) + ulong(slot)] * float(a[row]);\n"
"      }\n"
"    }\n"
"  }\n"
"  acc = simd_sum(acc);\n"
"  if (tiisg == 0u) partial[sgitg] = acc;\n"
"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  if (tid == 0u) {\n"
"    float total = 0.0f;\n"
"    for (uint simd_index = 0u; simd_index < 8u; simd_index++) total += partial[simd_index];\n"
"    out[ulong(token) * ulong(args.rows) + ulong(row)] = total;\n"
"  }\n"
"}\n";

typedef struct SparseMap {
    uint16_t *unique_codes;
    uint16_t *unique_groups;
    uint16_t *inverse;
    uint32_t *prefix;
    uint32_t unique_total;
} SparseMap;

typedef struct SparseRecordHost {
    uint32_t unique_offset;
    uint32_t unique_count;
    uint32_t prefix_offset;
    uint32_t inverse_offset;
    uint32_t score_offset;
    uint32_t sidecar_rank;
    uint32_t sidecar_a_lo;
    uint32_t sidecar_a_hi;
    uint64_t codebook_offset;
    uint64_t scale_offset;
} SparseRecordHost;

typedef struct SparseArgsHost {
    uint32_t rows;
    uint32_t groups;
    uint32_t in_dim;
    uint32_t n_slots;
    uint32_t n_tokens;
    uint32_t mid_token_stride;
    uint32_t mid_slot_stride;
    uint32_t score_token_stride;
} SparseArgsHost;

static float f16_to_f32(uint16_t bits) {
    __fp16 value;
    memcpy(&value, &bits, sizeof(value));
    return (float)value;
}

static uint16_t load_u16(const uint8_t *bytes) {
    return (uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8);
}

static float ref_mid_at(const ds4_d8f_file *file,
                        const ds4_d8f_record *rec,
                        const float *mid,
                        uint32_t index) {
    if ((rec->flags & 1u) && rec->scale_offset && rec->scale_bytes >= (index + 1u) * 2u) {
        return mid[index] * f16_to_f32(load_u16(file->map + rec->scale_offset + (uint64_t)index * 2u));
    }
    return mid[index];
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

static int build_sparse_map(const uint16_t *codes,
                            uint32_t rows,
                            uint32_t groups,
                            uint32_t k,
                            SparseMap *out) {
    memset(out, 0, sizeof(*out));
    const uint64_t max_unique = (uint64_t)groups * (uint64_t)k;
    if (max_unique == 0 || max_unique > UINT32_MAX) return 0;
    uint16_t *unique_codes = (uint16_t *)malloc((size_t)max_unique * sizeof(uint16_t));
    uint16_t *unique_groups = (uint16_t *)malloc((size_t)max_unique * sizeof(uint16_t));
    uint16_t *inverse = (uint16_t *)malloc((size_t)rows * groups * sizeof(uint16_t));
    uint32_t *prefix = (uint32_t *)calloc((size_t)groups + 1u, sizeof(uint32_t));
    uint8_t *present = (uint8_t *)calloc(k, 1);
    uint16_t *remap = (uint16_t *)malloc((size_t)k * sizeof(uint16_t));
    if (!unique_codes || !unique_groups || !inverse || !prefix || !present || !remap) {
        free(unique_codes); free(unique_groups); free(inverse); free(prefix); free(present); free(remap);
        return 0;
    }
    uint32_t cursor = 0;
    for (uint32_t group = 0; group < groups; group++) {
        memset(present, 0, k);
        for (uint32_t row = 0; row < rows; row++) {
            const uint16_t code = codes[(uint64_t)row * groups + group];
            if (code < k) present[code] = 1u;
        }
        prefix[group] = cursor;
        uint32_t local = 0;
        for (uint32_t code = 0; code < k; code++) {
            if (!present[code]) continue;
            remap[code] = (uint16_t)local;
            unique_codes[cursor] = (uint16_t)code;
            unique_groups[cursor] = (uint16_t)group;
            cursor++;
            local++;
        }
        if (local > UINT16_MAX) {
            free(unique_codes); free(unique_groups); free(inverse); free(prefix); free(present); free(remap);
            return 0;
        }
        for (uint32_t row = 0; row < rows; row++) {
            const uint16_t code = codes[(uint64_t)row * groups + group];
            inverse[(uint64_t)row * groups + group] = remap[code];
        }
    }
    prefix[groups] = cursor;
    free(present);
    free(remap);
    out->unique_codes = unique_codes;
    out->unique_groups = unique_groups;
    out->inverse = inverse;
    out->prefix = prefix;
    out->unique_total = cursor;
    return 1;
}

static void free_sparse_map(SparseMap *map) {
    if (!map) return;
    free(map->unique_codes);
    free(map->unique_groups);
    free(map->inverse);
    free(map->prefix);
    memset(map, 0, sizeof(*map));
}

static double run_sparse(id<MTLCommandQueue> queue,
                         id<MTLComputePipelineState> score_pipeline,
                         id<MTLComputePipelineState> gather_pipeline,
                         id<MTLBuffer> unique_codes,
                         id<MTLBuffer> unique_groups,
                         id<MTLBuffer> inverse,
                         id<MTLBuffer> prefix,
                         id<MTLBuffer> pack,
                         id<MTLBuffer> mid,
                         id<MTLBuffer> score,
                         id<MTLBuffer> records,
                         id<MTLBuffer> sidecar_dot,
                         id<MTLBuffer> output,
                         id<MTLBuffer> args,
                         uint32_t max_unique,
                         uint32_t rows,
                         uint32_t n_tokens,
                         uint32_t n_slots) {
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:score_pipeline];
    [encoder setBuffer:unique_codes offset:0 atIndex:0];
    [encoder setBuffer:unique_groups offset:0 atIndex:1];
    [encoder setBuffer:pack offset:0 atIndex:2];
    [encoder setBuffer:mid offset:0 atIndex:3];
    [encoder setBuffer:score offset:0 atIndex:4];
    [encoder setBuffer:records offset:0 atIndex:5];
    [encoder setBuffer:args offset:0 atIndex:6];
    [encoder dispatchThreads:MTLSizeMake(max_unique, n_tokens, n_slots)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder setComputePipelineState:gather_pipeline];
    [encoder setBuffer:inverse offset:0 atIndex:0];
    [encoder setBuffer:prefix offset:0 atIndex:1];
    [encoder setBuffer:score offset:0 atIndex:2];
    [encoder setBuffer:records offset:0 atIndex:3];
    [encoder setBuffer:sidecar_dot offset:0 atIndex:4];
    [encoder setBuffer:pack offset:0 atIndex:5];
    [encoder setBuffer:output offset:0 atIndex:6];
    [encoder setBuffer:args offset:0 atIndex:7];
    [encoder setThreadgroupMemoryLength:8u * sizeof(float) atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(rows, n_tokens, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    return (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "usage: %s layer.d8f expert_csv [rows=4096] [tokens=1] [rounds=60]\n", argv[0]);
            return 2;
        }
        const char *path = argv[1];
        uint32_t experts[6] = {0};
        uint32_t n_slots = 0;
        if (!parse_experts(argv[2], experts, &n_slots)) {
            fprintf(stderr, "bad expert csv: %s\n", argv[2]);
            return 2;
        }
        uint32_t rows = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10) : 4096u;
        uint32_t n_tokens = argc > 4 ? (uint32_t)strtoul(argv[4], NULL, 10) : 1u;
        uint32_t rounds = argc > 5 ? (uint32_t)strtoul(argv[5], NULL, 10) : 60u;
        if (rows == 0 || rows > 4096u) rows = 4096u;
        if (n_tokens == 0 || n_tokens > 16u) n_tokens = 1u;
        if (rounds == 0) rounds = 1u;

        ds4_d8f_file file;
        if (!ds4_d8f_open(path, &file)) return 1;
        const uint32_t groups = 256u;
        const uint32_t in_dim = groups * 8u;
        ds4_d8f_record down_records[6];
        ds4_d8f_native_code_record native_records[6];
        ds4_d8f_sidecar_record sidecar_records[6];
        uint8_t sidecar_live[6] = {0};
        SparseMap maps[6];
        memset(maps, 0, sizeof(maps));

        uint32_t total_unique = 0;
        uint32_t max_unique = 0;
        uint64_t sparse_map_bytes = 0;
        for (uint32_t slot = 0; slot < n_slots; slot++) {
            if (!ds4_d8f_get_record(&file, DS4_D8F_DOWN, experts[slot], &down_records[slot]) ||
                !ds4_d8f_get_down_native_codes(&file, experts[slot], &native_records[slot])) {
                fprintf(stderr, "missing down/native expert=%u\n", experts[slot]);
                ds4_d8f_close(&file);
                return 1;
            }
            if (native_records[slot].rows < rows || native_records[slot].groups != groups) {
                fprintf(stderr, "bad native dims expert=%u rows=%u groups=%u\n",
                        experts[slot], native_records[slot].rows, native_records[slot].groups);
                ds4_d8f_close(&file);
                return 1;
            }
            const uint16_t *native_codes = (const uint16_t *)(file.map + native_records[slot].offset);
            if (!build_sparse_map(native_codes, rows, groups, down_records[slot].k, &maps[slot])) {
                fprintf(stderr, "sparse map failed expert=%u\n", experts[slot]);
                ds4_d8f_close(&file);
                return 1;
            }
            if (maps[slot].unique_total > max_unique) max_unique = maps[slot].unique_total;
            total_unique += maps[slot].unique_total;
            sparse_map_bytes += (uint64_t)maps[slot].unique_total * 2u * 2u +
                                (uint64_t)rows * groups * sizeof(uint16_t) +
                                (uint64_t)(groups + 1u) * sizeof(uint32_t);
            if (ds4_d8f_get_down_sidecar(&file, experts[slot], &sidecar_records[slot])) {
                if (sidecar_records[slot].rank != 1u || sidecar_records[slot].in_dim != in_dim ||
                    sidecar_records[slot].out_dim < rows) {
                    fprintf(stderr, "unsupported sidecar expert=%u rank=%u\n",
                            experts[slot], sidecar_records[slot].rank);
                    ds4_d8f_close(&file);
                    return 1;
                }
                sidecar_live[slot] = 1u;
            }
        }

        float *mid = (float *)malloc((size_t)n_tokens * n_slots * in_dim * sizeof(float));
        float *sidecar_dot = (float *)calloc((size_t)n_tokens * n_slots, sizeof(float));
        float *ref = (float *)calloc((size_t)n_tokens * rows, sizeof(float));
        if (!mid || !sidecar_dot || !ref) {
            free(mid); free(sidecar_dot); free(ref); ds4_d8f_close(&file);
            return 1;
        }
        for (uint32_t token = 0; token < n_tokens; token++) {
            for (uint32_t slot = 0; slot < n_slots; slot++) {
                float *slot_mid = mid + ((uint64_t)token * n_slots + slot) * in_dim;
                for (uint32_t index = 0; index < in_dim; index++) {
                    slot_mid[index] =
                        0.50f * sinf((float)(index + slot * 17u + token * 11u) * 0.011f) +
                        0.25f * cosf((float)(index + slot * 29u + token * 7u) * 0.023f);
                }
                if (sidecar_live[slot]) {
                    double dot = 0.0;
                    const uint8_t *u = file.map + sidecar_records[slot].u_offset;
                    for (uint32_t index = 0; index < in_dim; index++) {
                        dot += (double)f16_to_f32(load_u16(u + (uint64_t)index * 2u)) *
                               (double)ref_mid_at(&file, &down_records[slot], slot_mid, index);
                    }
                    sidecar_dot[(uint64_t)token * n_slots + slot] = (float)dot;
                }
            }
        }
        for (uint32_t token = 0; token < n_tokens; token++) {
            for (uint32_t row = 0; row < rows; row++) {
                double sum = 0.0;
                for (uint32_t slot = 0; slot < n_slots; slot++) {
                    const uint16_t *native_codes = (const uint16_t *)(file.map + native_records[slot].offset);
                    const float *slot_mid = mid + ((uint64_t)token * n_slots + slot) * in_dim;
                    const ds4_d8f_record *down = &down_records[slot];
                    for (uint32_t group = 0; group < groups; group++) {
                        const uint32_t code = native_codes[(uint64_t)row * groups + group];
                        const uint8_t *half = file.map + down->codebook_offset + (uint64_t)code * 16u;
                        const uint32_t x_base = group << 3;
                        for (uint32_t delta = 0; delta < 8u; delta++) {
                            sum += (double)f16_to_f32(load_u16(half + (uint64_t)delta * 2u)) *
                                   (double)ref_mid_at(&file, down, slot_mid, x_base + delta);
                        }
                    }
                    if (sidecar_live[slot]) {
                        const uint8_t *a = file.map + sidecar_records[slot].a_offset;
                        sum += (double)sidecar_dot[(uint64_t)token * n_slots + slot] *
                               (double)f16_to_f32(load_u16(a + (uint64_t)row * 2u));
                    }
                }
                ref[(uint64_t)token * rows + row] = (float)sum;
            }
        }

        uint16_t *unique_codes = (uint16_t *)malloc((size_t)total_unique * sizeof(uint16_t));
        uint16_t *unique_groups = (uint16_t *)malloc((size_t)total_unique * sizeof(uint16_t));
        uint16_t *inverse = (uint16_t *)malloc((size_t)n_slots * rows * groups * sizeof(uint16_t));
        uint32_t *prefix = (uint32_t *)malloc((size_t)n_slots * (groups + 1u) * sizeof(uint32_t));
        SparseRecordHost records[6];
        memset(records, 0, sizeof(records));
        if (!unique_codes || !unique_groups || !inverse || !prefix) {
            free(unique_codes); free(unique_groups); free(inverse); free(prefix);
            free(mid); free(sidecar_dot); free(ref); ds4_d8f_close(&file);
            return 1;
        }
        uint32_t unique_cursor = 0;
        for (uint32_t slot = 0; slot < n_slots; slot++) {
            memcpy(unique_codes + unique_cursor, maps[slot].unique_codes, (size_t)maps[slot].unique_total * sizeof(uint16_t));
            memcpy(unique_groups + unique_cursor, maps[slot].unique_groups, (size_t)maps[slot].unique_total * sizeof(uint16_t));
            memcpy(inverse + (uint64_t)slot * rows * groups, maps[slot].inverse, (size_t)rows * groups * sizeof(uint16_t));
            memcpy(prefix + (uint64_t)slot * (groups + 1u), maps[slot].prefix, (size_t)(groups + 1u) * sizeof(uint32_t));
            records[slot].unique_offset = unique_cursor;
            records[slot].unique_count = maps[slot].unique_total;
            records[slot].prefix_offset = slot * (groups + 1u);
            records[slot].inverse_offset = slot * rows * groups;
            records[slot].score_offset = unique_cursor;
            records[slot].sidecar_rank = sidecar_live[slot] ? 1u : 0u;
            records[slot].sidecar_a_lo = sidecar_live[slot] ? (uint32_t)(sidecar_records[slot].a_offset & 0xffffffffull) : 0u;
            records[slot].sidecar_a_hi = sidecar_live[slot] ? (uint32_t)(sidecar_records[slot].a_offset >> 32) : 0u;
            records[slot].codebook_offset = down_records[slot].codebook_offset;
            records[slot].scale_offset = (down_records[slot].flags & 1u) ? down_records[slot].scale_offset : 0ull;
            unique_cursor += maps[slot].unique_total;
        }
        SparseArgsHost args = {
            rows,
            groups,
            in_dim,
            n_slots,
            n_tokens,
            n_slots * in_dim,
            in_dim,
            total_unique,
        };

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError *err = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kMetalSource]
                                                      options:nil
                                                        error:&err];
        if (!library) {
            NSLog(@"%@", err);
            free(unique_codes); free(unique_groups); free(inverse); free(prefix);
            free(mid); free(sidecar_dot); free(ref); ds4_d8f_close(&file);
            return 1;
        }
        id<MTLComputePipelineState> score_pipeline =
            [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"sparse_selected_score"] error:&err];
        id<MTLComputePipelineState> gather_pipeline =
            [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"sparse_selected_gather"] error:&err];
        if (!score_pipeline || !gather_pipeline) {
            NSLog(@"pipeline failed: %@", err);
            free(unique_codes); free(unique_groups); free(inverse); free(prefix);
            free(mid); free(sidecar_dot); free(ref); ds4_d8f_close(&file);
            return 1;
        }
        id<MTLBuffer> packBuf = [device newBufferWithBytesNoCopy:(void *)file.map
                                                          length:file.size
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];
        id<MTLBuffer> uniqueCodeBuf = [device newBufferWithBytes:unique_codes length:(NSUInteger)total_unique * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> uniqueGroupBuf = [device newBufferWithBytes:unique_groups length:(NSUInteger)total_unique * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> inverseBuf = [device newBufferWithBytes:inverse length:(NSUInteger)n_slots * rows * groups * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> prefixBuf = [device newBufferWithBytes:prefix length:(NSUInteger)n_slots * (groups + 1u) * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> midBuf = [device newBufferWithBytes:mid length:(NSUInteger)n_tokens * n_slots * in_dim * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> scoreBuf = [device newBufferWithLength:(NSUInteger)n_tokens * total_unique * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> recordBuf = [device newBufferWithBytes:records length:(NSUInteger)n_slots * sizeof(records[0]) options:MTLResourceStorageModeShared];
        id<MTLBuffer> sidecarDotBuf = [device newBufferWithBytes:sidecar_dot length:(NSUInteger)n_tokens * n_slots * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> outBuf = [device newBufferWithLength:(NSUInteger)n_tokens * rows * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> argsBuf = [device newBufferWithBytes:&args length:sizeof(args) options:MTLResourceStorageModeShared];
        if (!packBuf || !uniqueCodeBuf || !uniqueGroupBuf || !inverseBuf || !prefixBuf ||
            !midBuf || !scoreBuf || !recordBuf || !sidecarDotBuf || !outBuf || !argsBuf) {
            fprintf(stderr, "Metal buffer allocation failed\n");
            free(unique_codes); free(unique_groups); free(inverse); free(prefix);
            free(mid); free(sidecar_dot); free(ref); ds4_d8f_close(&file);
            return 1;
        }
        run_sparse(queue, score_pipeline, gather_pipeline, uniqueCodeBuf, uniqueGroupBuf, inverseBuf, prefixBuf,
                   packBuf, midBuf, scoreBuf, recordBuf, sidecarDotBuf, outBuf, argsBuf,
                   max_unique, rows, n_tokens, n_slots);
        double total_ms = 0.0;
        for (uint32_t round_index = 0; round_index < rounds; round_index++) {
            total_ms += run_sparse(queue, score_pipeline, gather_pipeline, uniqueCodeBuf, uniqueGroupBuf, inverseBuf, prefixBuf,
                                   packBuf, midBuf, scoreBuf, recordBuf, sidecarDotBuf, outBuf, argsBuf,
                                   max_unique, rows, n_tokens, n_slots);
        }
        float *gpu = outBuf.contents;
        uint32_t mismatch = 0;
        double max_abs = 0.0;
        double max_rel = 0.0;
        double sum_abs = 0.0;
        double sum_sq = 0.0;
        const uint64_t output_count = (uint64_t)n_tokens * rows;
        for (uint64_t output_index = 0; output_index < output_count; output_index++) {
            const double diff = fabs((double)gpu[output_index] - (double)ref[output_index]);
            const double denom = fabs((double)ref[output_index]) > 1e-4 ? fabs((double)ref[output_index]) : 1e-4;
            const double rel = diff / denom;
            if (diff > max_abs) max_abs = diff;
            if (rel > max_rel) max_rel = rel;
            sum_abs += diff;
            sum_sq += diff * diff;
            if (diff > 7e-4 && rel > 7e-4) mismatch++;
        }
        const double mean_abs = sum_abs / (double)output_count;
        const double rms_abs = sqrt(sum_sq / (double)output_count);
        fprintf(stderr,
                "d8f_sparse_selected_canary layer=%u nsel=%u rows=%u tokens=%u rounds=%u unique=%u max_unique=%u "
                "unique_mean=%.2f score=%.2fMiB map=%.2fMiB native=%.2fMiB sparse %.3fms total %.3fms/op %.3fus/token-row "
                "mismatch=%u max_abs=%.6e max_rel=%.6e mean_abs=%.6e rms_abs=%.6e rc=%d experts=",
                file.layer, n_slots, rows, n_tokens, rounds, total_unique, max_unique,
                (double)total_unique / (double)(n_slots * groups),
                (double)n_tokens * total_unique * sizeof(float) / 1048576.0,
                (double)sparse_map_bytes / 1048576.0,
                (double)n_slots * rows * groups * sizeof(uint16_t) / 1048576.0,
                total_ms,
                total_ms / (double)rounds,
                total_ms * 1000.0 / ((double)n_tokens * rows * rounds),
                mismatch, max_abs, max_rel, mean_abs, rms_abs, mismatch == 0 ? 1 : 0);
        for (uint32_t slot = 0; slot < n_slots; slot++) fprintf(stderr, "%s%u", slot ? "," : "", experts[slot]);
        fprintf(stderr, "\n");

        free(unique_codes);
        free(unique_groups);
        free(inverse);
        free(prefix);
        for (uint32_t slot = 0; slot < n_slots; slot++) free_sparse_map(&maps[slot]);
        free(mid);
        free(sidecar_dot);
        free(ref);
        ds4_d8f_close(&file);
        return mismatch == 0 ? 0 : 1;
    }
}
