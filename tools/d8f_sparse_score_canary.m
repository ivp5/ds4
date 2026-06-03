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
"struct Args {\n"
"  uint rows;\n"
"  uint groups;\n"
"  uint in_dim;\n"
"  uint pad0;\n"
"  ulong codebook_offset;\n"
"  ulong scale_offset;\n"
"};\n"
"\n"
"inline float mid_value(device const uchar *pack, ulong scale_offset, device const float *mid, uint idx) {\n"
"  if (scale_offset == 0ul) return mid[idx];\n"
"  device const half *scale = (device const half *)(pack + scale_offset);\n"
"  return mid[idx] * float(scale[idx]);\n"
"}\n"
"\n"
"inline float dot_code(device const uchar *pack, constant Args &args, device const float *mid, uint group, uint code) {\n"
"  const uint x = group << 3;\n"
"  device const half *cb = (device const half *)(pack + args.codebook_offset + ulong(code) * 16ul);\n"
"  return float(cb[0]) * mid_value(pack, args.scale_offset, mid, x + 0u) +\n"
"         float(cb[1]) * mid_value(pack, args.scale_offset, mid, x + 1u) +\n"
"         float(cb[2]) * mid_value(pack, args.scale_offset, mid, x + 2u) +\n"
"         float(cb[3]) * mid_value(pack, args.scale_offset, mid, x + 3u) +\n"
"         float(cb[4]) * mid_value(pack, args.scale_offset, mid, x + 4u) +\n"
"         float(cb[5]) * mid_value(pack, args.scale_offset, mid, x + 5u) +\n"
"         float(cb[6]) * mid_value(pack, args.scale_offset, mid, x + 6u) +\n"
"         float(cb[7]) * mid_value(pack, args.scale_offset, mid, x + 7u);\n"
"}\n"
"\n"
"kernel void sparse_score(\n"
"    device const ushort *unique_codes [[buffer(0)]],\n"
"    device const ushort *unique_groups [[buffer(1)]],\n"
"    device const uchar *pack [[buffer(2)]],\n"
"    device const float *mid [[buffer(3)]],\n"
"    device float *score [[buffer(4)]],\n"
"    constant Args &args [[buffer(5)]],\n"
"    constant uint &unique_total [[buffer(6)]],\n"
"    uint tid [[thread_position_in_grid]]) {\n"
"  if (tid >= unique_total) return;\n"
"  const uint group = uint(unique_groups[tid]);\n"
"  const uint code = uint(unique_codes[tid]);\n"
"  score[tid] = dot_code(pack, args, mid, group, code);\n"
"}\n"
"\n"
"kernel void sparse_gather_rows(\n"
"    device const ushort *inverse [[buffer(0)]],\n"
"    device const uint *prefix [[buffer(1)]],\n"
"    device const float *score [[buffer(2)]],\n"
"    device float *out [[buffer(3)]],\n"
"    constant Args &args [[buffer(4)]],\n"
"    threadgroup float *partial [[threadgroup(0)]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    ushort tiisg [[thread_index_in_simdgroup]],\n"
"    ushort sgitg [[simdgroup_index_in_threadgroup]],\n"
"    uint row [[threadgroup_position_in_grid]]) {\n"
"  float acc = 0.0f;\n"
"  if (row < args.rows && tid < args.groups) {\n"
"    const uint group = tid;\n"
"    const uint local = uint(inverse[ulong(row) * ulong(args.groups) + ulong(group)]);\n"
"    acc = score[ulong(prefix[group]) + ulong(local)];\n"
"  }\n"
"  acc = simd_sum(acc);\n"
"  if (tiisg == 0u) partial[sgitg] = acc;\n"
"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  if (tid == 0u) {\n"
"    float total = 0.0f;\n"
"    for (uint sg = 0u; sg < 8u; sg++) total += partial[sg];\n"
"    out[row] = total;\n"
"  }\n"
"}\n"
"\n"
"kernel void direct_rows(\n"
"    device const ushort *codes [[buffer(0)]],\n"
"    device const uchar *pack [[buffer(1)]],\n"
"    device const float *mid [[buffer(2)]],\n"
"    device float *out [[buffer(3)]],\n"
"    constant Args &args [[buffer(4)]],\n"
"    threadgroup float *partial [[threadgroup(0)]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    ushort tiisg [[thread_index_in_simdgroup]],\n"
"    ushort sgitg [[simdgroup_index_in_threadgroup]],\n"
"    uint row [[threadgroup_position_in_grid]]) {\n"
"  float acc = 0.0f;\n"
"  if (row < args.rows && tid < args.groups) {\n"
"    const uint group = tid;\n"
"    const uint code = uint(codes[ulong(row) * ulong(args.groups) + ulong(group)]);\n"
"    acc = dot_code(pack, args, mid, group, code);\n"
"  }\n"
"  acc = simd_sum(acc);\n"
"  if (tiisg == 0u) partial[sgitg] = acc;\n"
"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  if (tid == 0u) {\n"
"    float total = 0.0f;\n"
"    for (uint sg = 0u; sg < 8u; sg++) total += partial[sg];\n"
"    out[row] = total;\n"
"  }\n"
"}\n";

typedef struct SparseMap {
    uint16_t *unique_codes;
    uint16_t *unique_groups;
    uint16_t *inverse;
    uint32_t *prefix;
    uint32_t unique_total;
} SparseMap;

typedef struct MetalArgs {
    uint32_t rows;
    uint32_t groups;
    uint32_t in_dim;
    uint32_t pad0;
    uint64_t codebook_offset;
    uint64_t scale_offset;
} MetalArgs;

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

static double run_direct(id<MTLCommandQueue> queue,
                         id<MTLComputePipelineState> pipeline,
                         id<MTLBuffer> codes,
                         id<MTLBuffer> pack,
                         id<MTLBuffer> mid,
                         id<MTLBuffer> out,
                         id<MTLBuffer> args,
                         uint32_t rows) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:codes offset:0 atIndex:0];
    [enc setBuffer:pack offset:0 atIndex:1];
    [enc setBuffer:mid offset:0 atIndex:2];
    [enc setBuffer:out offset:0 atIndex:3];
    [enc setBuffer:args offset:0 atIndex:4];
    [enc setThreadgroupMemoryLength:8u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
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
                         id<MTLBuffer> out,
                         id<MTLBuffer> args,
                         uint32_t unique_total,
                         uint32_t rows) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:score_pipeline];
    [enc setBuffer:unique_codes offset:0 atIndex:0];
    [enc setBuffer:unique_groups offset:0 atIndex:1];
    [enc setBuffer:pack offset:0 atIndex:2];
    [enc setBuffer:mid offset:0 atIndex:3];
    [enc setBuffer:score offset:0 atIndex:4];
    [enc setBuffer:args offset:0 atIndex:5];
    [enc setBytes:&unique_total length:sizeof(unique_total) atIndex:6];
    [enc dispatchThreads:MTLSizeMake(unique_total, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc setComputePipelineState:gather_pipeline];
    [enc setBuffer:inverse offset:0 atIndex:0];
    [enc setBuffer:prefix offset:0 atIndex:1];
    [enc setBuffer:score offset:0 atIndex:2];
    [enc setBuffer:out offset:0 atIndex:3];
    [enc setBuffer:args offset:0 atIndex:4];
    [enc setThreadgroupMemoryLength:8u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
}

static double avg_time_direct(id<MTLCommandQueue> queue,
                              id<MTLComputePipelineState> pipeline,
                              id<MTLBuffer> codes,
                              id<MTLBuffer> pack,
                              id<MTLBuffer> mid,
                              id<MTLBuffer> out,
                              id<MTLBuffer> args,
                              uint32_t rows,
                              uint32_t rounds) {
    double total = 0.0;
    for (uint32_t i = 0; i < rounds; i++) {
        total += run_direct(queue, pipeline, codes, pack, mid, out, args, rows);
    }
    return total / (double)rounds;
}

static double avg_time_sparse(id<MTLCommandQueue> queue,
                              id<MTLComputePipelineState> score_pipeline,
                              id<MTLComputePipelineState> gather_pipeline,
                              id<MTLBuffer> unique_codes,
                              id<MTLBuffer> unique_groups,
                              id<MTLBuffer> inverse,
                              id<MTLBuffer> prefix,
                              id<MTLBuffer> pack,
                              id<MTLBuffer> mid,
                              id<MTLBuffer> score,
                              id<MTLBuffer> out,
                              id<MTLBuffer> args,
                              uint32_t unique_total,
                              uint32_t rows,
                              uint32_t rounds) {
    double total = 0.0;
    for (uint32_t i = 0; i < rounds; i++) {
        total += run_sparse(queue, score_pipeline, gather_pipeline,
                            unique_codes, unique_groups, inverse, prefix,
                            pack, mid, score, out, args, unique_total, rows);
    }
    return total / (double)rounds;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "usage: %s layer.d8f expert [rounds]\n", argv[0]);
            return 2;
        }
        const char *path = argv[1];
        const uint32_t expert = (uint32_t)strtoul(argv[2], NULL, 10);
        const uint32_t rounds = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10) : 80u;

        ds4_d8f_file file;
        if (!ds4_d8f_open(path, &file)) return 1;
        ds4_d8f_record down;
        ds4_d8f_native_code_record native;
        if (!ds4_d8f_get_record(&file, DS4_D8F_DOWN, expert, &down) ||
            !ds4_d8f_get_down_native_codes(&file, expert, &native)) {
            fprintf(stderr, "missing down/native codes for expert %u\n", expert);
            ds4_d8f_close(&file);
            return 1;
        }
        const uint16_t *native_codes = (const uint16_t *)(file.map + native.offset);
        SparseMap sparse;
        if (!build_sparse_map(native_codes, native.rows, native.groups, down.k, &sparse)) {
            fprintf(stderr, "sparse map build failed\n");
            ds4_d8f_close(&file);
            return 1;
        }
        float *mid = (float *)malloc((size_t)native.groups * 8u * sizeof(float));
        if (!mid) {
            free_sparse_map(&sparse);
            ds4_d8f_close(&file);
            return 1;
        }
        for (uint32_t i = 0; i < native.groups * 8u; i++) {
            mid[i] = sinf((float)i * 0.017f) + 0.125f * cosf((float)i * 0.071f);
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError *err = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kMetalSource]
                                                      options:nil
                                                        error:&err];
        if (!library) {
            NSLog(@"%@", err);
            free(mid); free_sparse_map(&sparse); ds4_d8f_close(&file);
            return 1;
        }
        id<MTLComputePipelineState> direct_pipeline =
            [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"direct_rows"] error:&err];
        id<MTLComputePipelineState> score_pipeline =
            [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"sparse_score"] error:&err];
        id<MTLComputePipelineState> gather_pipeline =
            [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"sparse_gather_rows"] error:&err];
        if (!direct_pipeline || !score_pipeline || !gather_pipeline) {
            NSLog(@"pipeline failed: %@", err);
            free(mid); free_sparse_map(&sparse); ds4_d8f_close(&file);
            return 1;
        }

        MetalArgs args = {
            native.rows,
            native.groups,
            native.groups * 8u,
            0u,
            down.codebook_offset,
            (down.flags & 1u) ? down.scale_offset : 0ull,
        };
        id<MTLBuffer> packBuf = [device newBufferWithBytesNoCopy:(void *)file.map
                                                          length:file.size
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];
        id<MTLBuffer> codesBuf = [device newBufferWithBytes:native_codes
                                                     length:(NSUInteger)native.bytes
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> uniqueCodeBuf = [device newBufferWithBytes:sparse.unique_codes
                                                          length:(NSUInteger)sparse.unique_total * sizeof(uint16_t)
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> uniqueGroupBuf = [device newBufferWithBytes:sparse.unique_groups
                                                           length:(NSUInteger)sparse.unique_total * sizeof(uint16_t)
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> inverseBuf = [device newBufferWithBytes:sparse.inverse
                                                       length:(NSUInteger)native.rows * native.groups * sizeof(uint16_t)
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> prefixBuf = [device newBufferWithBytes:sparse.prefix
                                                      length:(NSUInteger)(native.groups + 1u) * sizeof(uint32_t)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> midBuf = [device newBufferWithBytes:mid
                                                   length:(NSUInteger)native.groups * 8u * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> scoreBuf = [device newBufferWithLength:(NSUInteger)sparse.unique_total * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> directOut = [device newBufferWithLength:(NSUInteger)native.rows * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> sparseOut = [device newBufferWithLength:(NSUInteger)native.rows * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> argsBuf = [device newBufferWithBytes:&args length:sizeof(args) options:MTLResourceStorageModeShared];
        if (!packBuf || !codesBuf || !uniqueCodeBuf || !uniqueGroupBuf || !inverseBuf || !prefixBuf ||
            !midBuf || !scoreBuf || !directOut || !sparseOut || !argsBuf) {
            fprintf(stderr, "Metal buffer allocation failed\n");
            free(mid); free_sparse_map(&sparse); ds4_d8f_close(&file);
            return 1;
        }

        run_direct(queue, direct_pipeline, codesBuf, packBuf, midBuf, directOut, argsBuf, native.rows);
        run_sparse(queue, score_pipeline, gather_pipeline, uniqueCodeBuf, uniqueGroupBuf, inverseBuf, prefixBuf,
                   packBuf, midBuf, scoreBuf, sparseOut, argsBuf, sparse.unique_total, native.rows);
        float *direct_values = directOut.contents;
        float *sparse_values = sparseOut.contents;
        double max_abs = 0.0, rms = 0.0;
        uint32_t mismatch = 0;
        for (uint32_t row = 0; row < native.rows; row++) {
            const double diff = fabs((double)direct_values[row] - (double)sparse_values[row]);
            if (diff > max_abs) max_abs = diff;
            rms += diff * diff;
            if (diff > 1e-4) mismatch++;
        }
        rms = sqrt(rms / (double)native.rows);

        const double direct_ms = avg_time_direct(queue, direct_pipeline, codesBuf, packBuf, midBuf,
                                                 directOut, argsBuf, native.rows, rounds);
        const double sparse_ms = avg_time_sparse(queue, score_pipeline, gather_pipeline,
                                                 uniqueCodeBuf, uniqueGroupBuf, inverseBuf, prefixBuf,
                                                 packBuf, midBuf, scoreBuf, sparseOut, argsBuf,
                                                 sparse.unique_total, native.rows, rounds);
        fprintf(stderr,
                "d8f_sparse_score_canary layer=%u expert=%u k=%u rows=%u groups=%u unique=%u "
                "unique_mean=%.2f direct=%.4fms sparse=%.4fms speed=%.3fx "
                "score=%.2fMiB map=%.2fMiB native=%.2fMiB mismatch=%u max_abs=%.6e rms=%.6e rc=%d\n",
                file.layer, expert, down.k, native.rows, native.groups, sparse.unique_total,
                (double)sparse.unique_total / (double)native.groups,
                direct_ms, sparse_ms, direct_ms / sparse_ms,
                (double)sparse.unique_total * sizeof(float) / 1048576.0,
                ((double)sparse.unique_total * 2.0 * 2.0 +
                 (double)native.rows * native.groups * 2.0 +
                 (double)(native.groups + 1u) * 4.0) / 1048576.0,
                (double)native.bytes / 1048576.0,
                mismatch, max_abs, rms, mismatch == 0 ? 1 : 0);

        free(mid);
        free_sparse_map(&sparse);
        ds4_d8f_close(&file);
        return mismatch == 0 ? 0 : 1;
    }
}
