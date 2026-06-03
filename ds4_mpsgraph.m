#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <mach/mach_time.h>

#include "ds4_d8f_reader.h"
#include "ds4_gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t ds4_mpsgraph_u16(const uint8_t *p) {
    return (uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8);
}

static float ds4_mpsgraph_f16_to_f32(uint16_t bits) {
    _Float16 h;
    memcpy(&h, &bits, sizeof(h));
    return (float)h;
}

static uint16_t ds4_mpsgraph_f32_to_f16(float value) {
    _Float16 h = (_Float16)value;
    uint16_t bits = 0;
    memcpy(&bits, &h, sizeof(bits));
    return bits;
}

static double ds4_mpsgraph_seconds(void) {
    static mach_timebase_info_data_t timebase;
    if (timebase.denom == 0) mach_timebase_info(&timebase);
    return (double)mach_absolute_time() * (double)timebase.numer / (double)timebase.denom * 1e-9;
}

static bool ds4_mpsgraph_prepare_down_inputs(const ds4_d8f_file *file,
                                             const ds4_d8f_record *record,
                                             uint32_t rows,
                                             uint32_t slot,
                                             float *x_f32,
                                             uint16_t *x_f16,
                                             int32_t *indices) {
    enum { block = 8, down_in_dim = 2048 };
    if (!file || !record || !x_f32 || !x_f16 || !indices) return false;
    if (record->block != block || rows == 0) return false;
    const uint32_t groups = down_in_dim / block;
    if (((record->flags & 1u) != 0u) && record->scale_bytes < down_in_dim * 2u) return false;
    for (uint32_t i = 0; i < down_in_dim; i++) {
        float value = 0.50f * sinf((float)(i + slot * 17u) * 0.011f) +
                      0.25f * cosf((float)(i + slot * 29u) * 0.023f);
        if ((record->flags & 1u) != 0u && record->scale_offset) {
            const uint8_t *scale = file->map + record->scale_offset + (uint64_t)i * 2u;
            value *= ds4_mpsgraph_f16_to_f32(ds4_mpsgraph_u16(scale));
        }
        x_f32[i] = value;
        x_f16[i] = ds4_mpsgraph_f32_to_f16(value);
    }
    for (uint32_t group = 0; group < groups; group++) {
        for (uint32_t row = 0; row < rows; row++) {
            const uint64_t block_index = (uint64_t)row * groups + group;
            uint32_t code = ds4_d8f_code_at(file, record, block_index);
            if (code >= record->k) code = record->k;
            indices[(uint64_t)group * rows + row] = (int32_t)code;
        }
    }
    return true;
}

static bool ds4_mpsgraph_prepare_codebook(const ds4_d8f_file *file,
                                          const ds4_d8f_record *record,
                                          float *codebook_f32,
                                          uint16_t *codebook_f16) {
    enum { block = 8 };
    if (!file || !record || !codebook_f32 || !codebook_f16) return false;
    if (record->codebook_bytes < (uint64_t)record->k * block * 2u) return false;
    const uint8_t *src = file->map + record->codebook_offset;
    for (uint32_t code = 0; code < record->k; code++) {
        for (uint32_t d = 0; d < block; d++) {
            const uint16_t bits = ds4_mpsgraph_u16(src + ((uint64_t)code * block + d) * 2u);
            codebook_f16[(uint64_t)d * record->k + code] = bits;
            codebook_f32[(uint64_t)d * record->k + code] = ds4_mpsgraph_f16_to_f32(bits);
        }
    }
    return true;
}

static void ds4_mpsgraph_reference_down_lut_accum(const float *x,
                                                  const float *codebook,
                                                  const int32_t *indices,
                                                  uint32_t k,
                                                  uint32_t rows,
                                                  float *out) {
    enum { block = 8, down_in_dim = 2048 };
    const uint32_t groups = down_in_dim / block;
    for (uint32_t row = 0; row < rows; row++) {
        double sum = 0.0;
        for (uint32_t group = 0; group < groups; group++) {
            const int32_t code = indices[(uint64_t)group * rows + row];
            if (code < 0 || (uint32_t)code >= k) continue;
            const float *xg = x + (uint64_t)group * block;
            const float *cb = codebook + (uint64_t)code;
            for (uint32_t d = 0; d < block; d++) {
                sum += (double)xg[d] * (double)cb[(uint64_t)d * k];
            }
        }
        out[row] += (float)sum;
    }
}

static MPSGraphTensor *ds4_mpsgraph_build_down_lut(MPSGraph *graph,
                                                   MPSGraphTensor *x,
                                                   MPSGraphTensor *codebook,
                                                   MPSGraphTensor *indices,
                                                   uint32_t rows) {
    enum { block = 8, down_in_dim = 2048 };
    const uint32_t groups = down_in_dim / block;
    MPSGraphTensor *xr = [graph reshapeTensor:x
                                    withShape:@[@(groups), @(block)]
                                         name:@"d8f_lut_x_reshaped"];
    MPSGraphTensor *table = [graph matrixMultiplicationWithPrimaryTensor:xr
                                                         secondaryTensor:codebook
                                                                    name:@"d8f_lut_table"];
    MPSGraphTensor *gathered = [graph gatherAlongAxis:1
                                   withUpdatesTensor:table
                                       indicesTensor:indices
                                                name:@"d8f_lut_gather"];
    MPSGraphTensor *out = [graph reductionSumWithTensor:gathered
                                                   axis:0
                                                   name:@"d8f_lut_reduce"];
    return [graph reshapeTensor:out withShape:@[@(rows)] name:@"d8f_lut_out"];
}

int ds4_gpu_mpsgraph_d8f_down_lut_canary(const char *d8f_path,
                                         uint32_t expert,
                                         uint32_t rows,
                                         uint32_t rounds,
                                         uint32_t mode) {
    return ds4_gpu_mpsgraph_d8f_down_lut_selected_canary(d8f_path, &expert, 1u, rows, rounds, mode);
}

int ds4_gpu_mpsgraph_d8f_down_lut_selected_canary(const char *d8f_path,
                                                  const uint32_t *experts,
                                                  uint32_t n_experts,
                                                  uint32_t rows,
                                                  uint32_t rounds,
                                                  uint32_t mode) {
    enum { block = 8, down_in_dim = 2048, down_out_dim = 4096 };
    @autoreleasepool {
        if (!d8f_path || !experts || n_experts == 0u || n_experts > 6u) return 0;
        if (rows == 0u || rows > down_out_dim) rows = 128u;
        if (rounds == 0u) rounds = 20u;
        const bool fp32_path = mode != 0u;
        ds4_d8f_file file;
        if (!ds4_d8f_open(d8f_path, &file)) return 0;
        const uint32_t groups = down_in_dim / block;
        const uint64_t needed_blocks = (uint64_t)rows * groups;
        ds4_d8f_record records[6];
        uint32_t max_k = 0;
        for (uint32_t slot = 0; slot < n_experts; slot++) {
            if (experts[slot] >= 256u ||
                !ds4_d8f_get_record(&file, DS4_D8F_DOWN, experts[slot], &records[slot])) {
                fprintf(stderr, "ds4_mpsgraph: missing down expert %u in %s\n", experts[slot], d8f_path);
                ds4_d8f_close(&file);
                return 0;
            }
            const uint64_t available_blocks = ((uint64_t)records[slot].index_bytes * 8ull) / (uint64_t)records[slot].bits;
            if (records[slot].block != block || records[slot].k == 0u || available_blocks < needed_blocks) {
                fprintf(stderr,
                        "ds4_mpsgraph: invalid down record expert=%u k=%u block=%u available_blocks=%llu needed=%llu\n",
                        experts[slot], records[slot].k, records[slot].block,
                        (unsigned long long)available_blocks,
                        (unsigned long long)needed_blocks);
                ds4_d8f_close(&file);
                return 0;
            }
            if (records[slot].k > max_k) max_k = records[slot].k;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "ds4_mpsgraph: no Metal device\n");
            ds4_d8f_close(&file);
            return 0;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            fprintf(stderr, "ds4_mpsgraph: no Metal command queue\n");
            ds4_d8f_close(&file);
            return 0;
        }
        const size_t x_f32_bytes = (size_t)down_in_dim * sizeof(float);
        const size_t x_f16_bytes = (size_t)down_in_dim * sizeof(uint16_t);
        const size_t idx_bytes = (size_t)groups * rows * sizeof(int32_t);
        const size_t out_bytes = (size_t)rows * sizeof(float);
        float *x_f32[6] = {0};
        uint16_t *x_f16[6] = {0};
        float *cb_f32[6] = {0};
        uint16_t *cb_f16[6] = {0};
        int32_t *indices[6] = {0};
        float *ref = (float *)calloc(rows, sizeof(float));
        float *got = (float *)calloc(rows, sizeof(float));
        bool ok = ref && got;
        for (uint32_t slot = 0; ok && slot < n_experts; slot++) {
            const size_t cb_f32_bytes = (size_t)block * records[slot].k * sizeof(float);
            const size_t cb_f16_bytes = (size_t)block * records[slot].k * sizeof(uint16_t);
            x_f32[slot] = (float *)malloc(x_f32_bytes);
            x_f16[slot] = (uint16_t *)malloc(x_f16_bytes);
            cb_f32[slot] = (float *)malloc(cb_f32_bytes);
            cb_f16[slot] = (uint16_t *)malloc(cb_f16_bytes);
            indices[slot] = (int32_t *)malloc(idx_bytes);
            ok = x_f32[slot] && x_f16[slot] && cb_f32[slot] && cb_f16[slot] && indices[slot] &&
                 ds4_mpsgraph_prepare_down_inputs(&file, &records[slot], rows, slot,
                                                  x_f32[slot], x_f16[slot], indices[slot]) &&
                 ds4_mpsgraph_prepare_codebook(&file, &records[slot], cb_f32[slot], cb_f16[slot]);
            if (ok) {
                ds4_mpsgraph_reference_down_lut_accum(x_f32[slot], cb_f32[slot], indices[slot],
                                                      records[slot].k, rows, ref);
            }
        }
        if (!ok) fprintf(stderr, "ds4_mpsgraph: failed to allocate or prepare selected D8F LUT tensors\n");
        if (!ok) {
            for (uint32_t slot = 0; slot < n_experts; slot++) {
                free(indices[slot]); free(cb_f16[slot]); free(cb_f32[slot]); free(x_f16[slot]); free(x_f32[slot]);
            }
            free(got); free(ref);
            ds4_d8f_close(&file);
            return 0;
        }
        MPSGraph *graph = [[MPSGraph alloc] init];
        graph.options = MPSGraphOptionsDefault;
        MPSDataType data_type = fp32_path ? MPSDataTypeFloat32 : MPSDataTypeFloat16;
        NSMutableDictionary *feeds = [NSMutableDictionary dictionaryWithCapacity:n_experts];
        NSMutableDictionary *feed_types = [NSMutableDictionary dictionaryWithCapacity:n_experts];
        NSMutableArray *inputs = [NSMutableArray arrayWithCapacity:n_experts];
        MPSGraphTensor *out = nil;
        for (uint32_t slot = 0; slot < n_experts; slot++) {
            NSString *x_name = [NSString stringWithFormat:@"d8f_lut_x_%u", slot];
            MPSGraphTensor *x = [graph placeholderWithShape:@[@1, @(down_in_dim)]
                                                    dataType:data_type
                                                        name:x_name];
            const size_t cb_f32_bytes = (size_t)block * records[slot].k * sizeof(float);
            const size_t cb_f16_bytes = (size_t)block * records[slot].k * sizeof(uint16_t);
            NSData *cb_data = fp32_path ?
                [NSData dataWithBytes:cb_f32[slot] length:cb_f32_bytes] :
                [NSData dataWithBytes:cb_f16[slot] length:cb_f16_bytes];
            MPSGraphTensor *codebook = [graph constantWithData:cb_data
                                                         shape:@[@(block), @(records[slot].k)]
                                                      dataType:data_type];
            NSData *idx_data = [NSData dataWithBytes:indices[slot] length:idx_bytes];
            MPSGraphTensor *idx = [graph constantWithData:idx_data
                                                    shape:@[@(groups), @(rows)]
                                                 dataType:MPSDataTypeInt32];
            MPSGraphTensor *slot_out = ds4_mpsgraph_build_down_lut(graph, x, codebook, idx, rows);
            out = out ? [graph additionWithPrimaryTensor:out secondaryTensor:slot_out name:@"d8f_lut_selected_sum"] : slot_out;
            id<MTLBuffer> xbuf = [device newBufferWithBytes:(fp32_path ? (const void *)x_f32[slot] : (const void *)x_f16[slot])
                                                     length:(fp32_path ? x_f32_bytes : x_f16_bytes)
                                                    options:MTLResourceStorageModeShared];
            MPSGraphTensorData *xdata = [[MPSGraphTensorData alloc] initWithMTLBuffer:xbuf
                                                                                shape:@[@1, @(down_in_dim)]
                                                                             dataType:data_type];
            MPSGraphShapedType *x_type = [[MPSGraphShapedType alloc] initWithShape:@[@1, @(down_in_dim)]
                                                                           dataType:data_type];
            feeds[x] = xdata;
            feed_types[x] = x_type;
            [inputs addObject:xdata];
        }
        NSDictionary *result = [graph runWithMTLCommandQueue:queue
                                                       feeds:feeds
                                               targetTensors:@[out]
                                            targetOperations:nil];
        MPSGraphTensorData *out_data = result[out];
        if (!out_data) {
            fprintf(stderr, "ds4_mpsgraph: no output tensor data\n");
            for (uint32_t slot = 0; slot < n_experts; slot++) {
                free(indices[slot]); free(cb_f16[slot]); free(cb_f32[slot]); free(x_f16[slot]); free(x_f32[slot]);
            }
            free(got); free(ref);
            ds4_d8f_close(&file);
            return 0;
        }
        MPSNDArray *nd = [out_data mpsndarray];
        if (fp32_path) {
            [nd readBytes:got strideBytes:nil];
        } else {
            uint16_t *got_h = (uint16_t *)calloc(rows, sizeof(uint16_t));
            if (!got_h) {
                fprintf(stderr, "ds4_mpsgraph: output allocation failed\n");
                for (uint32_t slot = 0; slot < n_experts; slot++) {
                    free(indices[slot]); free(cb_f16[slot]); free(cb_f32[slot]); free(x_f16[slot]); free(x_f32[slot]);
                }
                free(got); free(ref);
                ds4_d8f_close(&file);
                return 0;
            }
            [nd readBytes:got_h strideBytes:nil];
            for (uint32_t i = 0; i < rows; i++) got[i] = ds4_mpsgraph_f16_to_f32(got_h[i]);
            free(got_h);
        }
        double max_abs = 0.0, max_rel = 0.0, rms = 0.0;
        uint32_t bad = 0;
        const double tol_abs = fp32_path ? 2.0e-3 : 2.5e-1 * (double)n_experts;
        const double tol_rel = fp32_path ? 2.0e-3 : 2.5e-2;
        for (uint32_t i = 0; i < rows; i++) {
            const double diff = fabs((double)got[i] - (double)ref[i]);
            const double denom = fmax(1.0, fabs((double)ref[i]));
            const double rel = diff / denom;
            if (diff > max_abs) max_abs = diff;
            if (rel > max_rel) max_rel = rel;
            rms += diff * diff;
            if (diff > tol_abs && rel > tol_rel) bad++;
        }
        rms = sqrt(rms / (double)rows);
        MPSGraphExecutable *executable = [graph compileWithDevice:nil
                                                            feeds:feed_types
                                                    targetTensors:@[out]
                                                 targetOperations:nil
                                            compilationDescriptor:nil];
        executable.options = MPSGraphOptionsNone;
        const size_t timed_out_bytes = fp32_path ? out_bytes : (size_t)rows * sizeof(uint16_t);
        id<MTLBuffer> timed_out = [device newBufferWithLength:timed_out_bytes
                                                      options:MTLResourceStorageModeShared];
        MPSGraphTensorData *timed_out_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:timed_out
                                                                                     shape:@[@(rows)]
                                                                                  dataType:data_type];
        NSArray *outputs = @[timed_out_data];
        for (uint32_t i = 0; i < 3u; i++) {
            [executable runWithMTLCommandQueue:queue
                                   inputsArray:inputs
                                  resultsArray:outputs
                           executionDescriptor:nil];
        }
        const double t0 = ds4_mpsgraph_seconds();
        for (uint32_t i = 0; i < rounds; i++) {
            [executable runWithMTLCommandQueue:queue
                                   inputsArray:inputs
                                  resultsArray:outputs
                           executionDescriptor:nil];
        }
        const double elapsed = ds4_mpsgraph_seconds() - t0;
        const double us_per = elapsed * 1.0e6 / (double)rounds;
        double logical_bytes = (double)out_bytes;
        for (uint32_t slot = 0; slot < n_experts; slot++) {
            logical_bytes += (double)(x_f16_bytes +
                                      (size_t)block * records[slot].k * sizeof(uint16_t) +
                                      idx_bytes +
                                      (size_t)groups * records[slot].k * (fp32_path ? 4u : 2u));
        }
        char expert_csv[128];
        expert_csv[0] = '\0';
        for (uint32_t slot = 0; slot < n_experts; slot++) {
            char item[24];
            snprintf(item, sizeof(item), "%s%u", slot ? "," : "", experts[slot]);
            strlcat(expert_csv, item, sizeof(expert_csv));
        }
        fprintf(stderr,
                "ds4_mpsgraph: d8f_down_lut_selected path=%s experts=%s nsel=%u rows=%u max_k=%u mode=%s rounds=%u us/op=%.3f logical_MB/op=%.3f bad=%u max_abs=%.6g max_rel=%.6g rms=%.6g sample_ref=%.6g sample_got=%.6g\n",
                d8f_path, expert_csv, n_experts, rows, max_k,
                fp32_path ? "fp32" : "fp16", rounds, us_per, logical_bytes / 1.0e6,
                bad, max_abs, max_rel, rms, (double)ref[0], (double)got[0]);
        for (uint32_t slot = 0; slot < n_experts; slot++) {
            free(indices[slot]); free(cb_f16[slot]); free(cb_f32[slot]); free(x_f16[slot]); free(x_f32[slot]);
        }
        free(got); free(ref);
        ds4_d8f_close(&file);
        return bad == 0 ? 1 : 0;
    }
}
