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

static bool ds4_mpsgraph_prepare_indices(const ds4_d8f_file *file,
                                         const ds4_d8f_record *record,
                                         uint32_t rows,
                                         uint32_t in_dim,
                                         int32_t *indices);

static void ds4_mpsgraph_free_down_arrays(uint32_t n_experts,
                                          int32_t **indices,
                                          float **cb_f32,
                                          uint16_t **cb_f16,
                                          float **x_f32,
                                          uint16_t **x_f16) {
    for (uint32_t slot = 0; slot < n_experts; slot++) {
        free(indices[slot]);
        free(cb_f16[slot]);
        free(cb_f32[slot]);
        free(x_f16[slot]);
        free(x_f32[slot]);
    }
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
    return ds4_mpsgraph_prepare_indices(file, record, rows, down_in_dim, indices);
}

static bool ds4_mpsgraph_prepare_codebook(const ds4_d8f_file *file,
                                          const ds4_d8f_record *record,
                                          uint32_t codebook_cols,
                                          float *codebook_f32,
                                          uint16_t *codebook_f16) {
    enum { block = 8 };
    if (!file || !record || !codebook_f32 || !codebook_f16) return false;
    if (codebook_cols <= record->k) return false;
    if (record->codebook_bytes < (uint64_t)record->k * block * 2u) return false;
    const uint8_t *src = file->map + record->codebook_offset;
    for (uint32_t code = 0; code < record->k; code++) {
        for (uint32_t dim = 0; dim < block; dim++) {
            const uint16_t bits = ds4_mpsgraph_u16(src + ((uint64_t)code * block + dim) * 2u);
            codebook_f16[(uint64_t)dim * codebook_cols + code] = bits;
            codebook_f32[(uint64_t)dim * codebook_cols + code] = ds4_mpsgraph_f16_to_f32(bits);
        }
    }
    for (uint32_t dim = 0; dim < block; dim++) {
        codebook_f16[(uint64_t)dim * codebook_cols + record->k] = 0u;
        codebook_f32[(uint64_t)dim * codebook_cols + record->k] = 0.0f;
    }
    return true;
}

static void ds4_mpsgraph_reference_down_lut_accum(const float *x,
                                                  const float *codebook,
                                                  const int32_t *indices,
                                                  uint32_t codebook_cols,
                                                  uint32_t in_dim,
                                                  uint32_t rows,
                                                  float *out) {
    enum { block = 8 };
    const uint32_t groups = in_dim / block;
    for (uint32_t row = 0; row < rows; row++) {
        double sum = 0.0;
        for (uint32_t group = 0; group < groups; group++) {
            const int32_t code = indices[(uint64_t)group * rows + row];
            if (code < 0 || (uint32_t)code >= codebook_cols) continue;
            const float *xg = x + (uint64_t)group * block;
            const float *cb = codebook + (uint64_t)code;
            for (uint32_t dim = 0; dim < block; dim++) {
                sum += (double)xg[dim] * (double)cb[(uint64_t)dim * codebook_cols];
            }
        }
        out[row] += (float)sum;
    }
}

static void ds4_mpsgraph_reference_lut(const float *x,
                                       const float *codebook,
                                       const int32_t *indices,
                                       uint32_t codebook_cols,
                                       uint32_t in_dim,
                                       uint32_t rows,
                                       float *out) {
    memset(out, 0, (size_t)rows * sizeof(float));
    ds4_mpsgraph_reference_down_lut_accum(x, codebook, indices, codebook_cols, in_dim, rows, out);
}

static MPSGraphTensor *ds4_mpsgraph_build_lut(MPSGraph *graph,
                                              MPSGraphTensor *x,
                                              MPSGraphTensor *codebook,
                                              MPSGraphTensor *indices,
                                              uint32_t in_dim,
                                              uint32_t rows) {
    enum { block = 8 };
    const uint32_t groups = in_dim / block;
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
            const uint32_t codebook_cols = records[slot].k + 1u;
            const size_t cb_f32_bytes = (size_t)block * codebook_cols * sizeof(float);
            const size_t cb_f16_bytes = (size_t)block * codebook_cols * sizeof(uint16_t);
            x_f32[slot] = (float *)malloc(x_f32_bytes);
            x_f16[slot] = (uint16_t *)malloc(x_f16_bytes);
            cb_f32[slot] = (float *)malloc(cb_f32_bytes);
            cb_f16[slot] = (uint16_t *)malloc(cb_f16_bytes);
            indices[slot] = (int32_t *)malloc(idx_bytes);
            ok = x_f32[slot] && x_f16[slot] && cb_f32[slot] && cb_f16[slot] && indices[slot] &&
                 ds4_mpsgraph_prepare_down_inputs(&file, &records[slot], rows, slot,
                                                  x_f32[slot], x_f16[slot], indices[slot]) &&
                 ds4_mpsgraph_prepare_codebook(&file, &records[slot], codebook_cols, cb_f32[slot], cb_f16[slot]);
            if (ok) {
                ds4_mpsgraph_reference_down_lut_accum(x_f32[slot], cb_f32[slot], indices[slot],
                                                      codebook_cols, down_in_dim, rows, ref);
            }
        }
        if (!ok) fprintf(stderr, "ds4_mpsgraph: failed to allocate or prepare selected D8F LUT tensors\n");
        if (!ok) {
            ds4_mpsgraph_free_down_arrays(n_experts, indices, cb_f32, cb_f16, x_f32, x_f16);
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
            const uint32_t codebook_cols = records[slot].k + 1u;
            NSString *x_name = [NSString stringWithFormat:@"d8f_lut_x_%u", slot];
            MPSGraphTensor *x = [graph placeholderWithShape:@[@1, @(down_in_dim)]
                                                    dataType:data_type
                                                        name:x_name];
            const size_t cb_f32_bytes = (size_t)block * codebook_cols * sizeof(float);
            const size_t cb_f16_bytes = (size_t)block * codebook_cols * sizeof(uint16_t);
            NSData *cb_data = fp32_path ?
                [NSData dataWithBytes:cb_f32[slot] length:cb_f32_bytes] :
                [NSData dataWithBytes:cb_f16[slot] length:cb_f16_bytes];
            MPSGraphTensor *codebook = [graph constantWithData:cb_data
                                                         shape:@[@(block), @(codebook_cols)]
                                                      dataType:data_type];
            NSData *idx_data = [NSData dataWithBytes:indices[slot] length:idx_bytes];
            MPSGraphTensor *idx = [graph constantWithData:idx_data
                                                    shape:@[@(groups), @(rows)]
                                                 dataType:MPSDataTypeInt32];
            MPSGraphTensor *slot_out = ds4_mpsgraph_build_lut(graph, x, codebook, idx, down_in_dim, rows);
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
            ds4_mpsgraph_free_down_arrays(n_experts, indices, cb_f32, cb_f16, x_f32, x_f16);
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
                ds4_mpsgraph_free_down_arrays(n_experts, indices, cb_f32, cb_f16, x_f32, x_f16);
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
            const uint32_t codebook_cols = records[slot].k + 1u;
            logical_bytes += (double)(x_f16_bytes +
                                      (size_t)block * codebook_cols * sizeof(uint16_t) +
                                      idx_bytes +
                                      (size_t)groups * codebook_cols * (fp32_path ? 4u : 2u));
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
        ds4_mpsgraph_free_down_arrays(n_experts, indices, cb_f32, cb_f16, x_f32, x_f16);
        free(got); free(ref);
        ds4_d8f_close(&file);
        return bad == 0 ? 1 : 0;
    }
}

static bool ds4_mpsgraph_prepare_indices(const ds4_d8f_file *file,
                                         const ds4_d8f_record *record,
                                         uint32_t rows,
                                         uint32_t in_dim,
                                         int32_t *indices) {
    enum { block = 8 };
    if (!file || !record || !indices || in_dim == 0u || (in_dim & 7u) != 0u) return false;
    const uint32_t groups = in_dim / block;
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

static void ds4_mpsgraph_free_gateup_arrays(uint32_t n_experts,
                                            int32_t **gate_idx,
                                            int32_t **up_idx,
                                            float **gate_cb_f32,
                                            float **up_cb_f32,
                                            uint16_t **gate_cb_f16,
                                            uint16_t **up_cb_f16) {
    for (uint32_t slot = 0; slot < n_experts; slot++) {
        free(up_idx[slot]);
        free(gate_idx[slot]);
        free(up_cb_f16[slot]);
        free(up_cb_f32[slot]);
        free(gate_cb_f16[slot]);
        free(gate_cb_f32[slot]);
    }
}

int ds4_gpu_mpsgraph_d8f_gateup_lut_selected_canary(const char *d8f_path,
                                                    const uint32_t *experts,
                                                    uint32_t n_experts,
                                                    uint32_t rows,
                                                    uint32_t rounds,
                                                    uint32_t mode,
                                                    float swiglu_limit) {
    enum { block = 8, gateup_in_dim = 4096, gateup_out_dim = 2048 };
    @autoreleasepool {
        if (!d8f_path || !experts || n_experts == 0u || n_experts > 6u) return 0;
        if (rows == 0u || rows > gateup_out_dim) rows = 128u;
        if (rounds == 0u) rounds = 20u;
        if (!(swiglu_limit > 0.0f)) swiglu_limit = 10.0f;
        const bool fp32_path = mode != 0u;
        ds4_d8f_file file;
        if (!ds4_d8f_open(d8f_path, &file)) return 0;
        const uint32_t groups = gateup_in_dim / block;
        const uint64_t needed_blocks = (uint64_t)rows * groups;
        ds4_d8f_record gate_records[6];
        ds4_d8f_record up_records[6];
        uint32_t max_k = 0;
        for (uint32_t slot = 0; slot < n_experts; slot++) {
            if (experts[slot] >= 256u ||
                !ds4_d8f_get_record(&file, DS4_D8F_GATE, experts[slot], &gate_records[slot]) ||
                !ds4_d8f_get_record(&file, DS4_D8F_UP, experts[slot], &up_records[slot])) {
                fprintf(stderr, "ds4_mpsgraph: missing gate/up expert %u in %s\n", experts[slot], d8f_path);
                ds4_d8f_close(&file);
                return 0;
            }
            const uint64_t gate_blocks = ((uint64_t)gate_records[slot].index_bytes * 8ull) / (uint64_t)gate_records[slot].bits;
            const uint64_t up_blocks = ((uint64_t)up_records[slot].index_bytes * 8ull) / (uint64_t)up_records[slot].bits;
            if (gate_records[slot].block != block || up_records[slot].block != block ||
                gate_records[slot].k == 0u || up_records[slot].k == 0u ||
                gate_blocks < needed_blocks || up_blocks < needed_blocks) {
                fprintf(stderr, "ds4_mpsgraph: invalid gate/up record expert=%u rows=%u\n", experts[slot], rows);
                ds4_d8f_close(&file);
                return 0;
            }
            if (gate_records[slot].k > max_k) max_k = gate_records[slot].k;
            if (up_records[slot].k > max_k) max_k = up_records[slot].k;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = device ? [device newCommandQueue] : nil;
        if (!device || !queue) {
            fprintf(stderr, "ds4_mpsgraph: no Metal device/queue\n");
            ds4_d8f_close(&file);
            return 0;
        }
        const size_t x_f32_bytes = (size_t)gateup_in_dim * sizeof(float);
        const size_t x_f16_bytes = (size_t)gateup_in_dim * sizeof(uint16_t);
        const size_t idx_bytes = (size_t)groups * rows * sizeof(int32_t);
        const size_t out_count = (size_t)n_experts * rows;
        const size_t out_bytes = out_count * sizeof(float);
        float *x_f32 = (float *)malloc(x_f32_bytes);
        uint16_t *x_f16 = (uint16_t *)malloc(x_f16_bytes);
        float *ref = (float *)malloc(out_bytes);
        float *got = (float *)calloc(out_count, sizeof(float));
        float *gate_tmp = (float *)malloc((size_t)rows * sizeof(float));
        float *up_tmp = (float *)malloc((size_t)rows * sizeof(float));
        float *gate_cb_f32[6] = {0}, *up_cb_f32[6] = {0};
        uint16_t *gate_cb_f16[6] = {0}, *up_cb_f16[6] = {0};
        int32_t *gate_idx[6] = {0}, *up_idx[6] = {0};
        bool ok = x_f32 && x_f16 && ref && got && gate_tmp && up_tmp;
        for (uint32_t i = 0; ok && i < gateup_in_dim; i++) {
            const float value = 0.35f * sinf((float)i * 0.011f) + 0.17f * cosf((float)i * 0.019f);
            x_f32[i] = value;
            x_f16[i] = ds4_mpsgraph_f32_to_f16(value);
        }
        for (uint32_t slot = 0; ok && slot < n_experts; slot++) {
            const uint32_t gate_cols = gate_records[slot].k + 1u;
            const uint32_t up_cols = up_records[slot].k + 1u;
            const size_t gate_cb_f32_bytes = (size_t)block * gate_cols * sizeof(float);
            const size_t gate_cb_f16_bytes = (size_t)block * gate_cols * sizeof(uint16_t);
            const size_t up_cb_f32_bytes = (size_t)block * up_cols * sizeof(float);
            const size_t up_cb_f16_bytes = (size_t)block * up_cols * sizeof(uint16_t);
            gate_cb_f32[slot] = (float *)malloc(gate_cb_f32_bytes);
            gate_cb_f16[slot] = (uint16_t *)malloc(gate_cb_f16_bytes);
            up_cb_f32[slot] = (float *)malloc(up_cb_f32_bytes);
            up_cb_f16[slot] = (uint16_t *)malloc(up_cb_f16_bytes);
            gate_idx[slot] = (int32_t *)malloc(idx_bytes);
            up_idx[slot] = (int32_t *)malloc(idx_bytes);
            ok = gate_cb_f32[slot] && gate_cb_f16[slot] && up_cb_f32[slot] && up_cb_f16[slot] &&
                 gate_idx[slot] && up_idx[slot] &&
                 ds4_mpsgraph_prepare_codebook(&file, &gate_records[slot], gate_cols, gate_cb_f32[slot], gate_cb_f16[slot]) &&
                 ds4_mpsgraph_prepare_codebook(&file, &up_records[slot], up_cols, up_cb_f32[slot], up_cb_f16[slot]) &&
                 ds4_mpsgraph_prepare_indices(&file, &gate_records[slot], rows, gateup_in_dim, gate_idx[slot]) &&
                 ds4_mpsgraph_prepare_indices(&file, &up_records[slot], rows, gateup_in_dim, up_idx[slot]);
            if (ok) {
                ds4_mpsgraph_reference_lut(x_f32, gate_cb_f32[slot], gate_idx[slot], gate_cols,
                                           gateup_in_dim, rows, gate_tmp);
                ds4_mpsgraph_reference_lut(x_f32, up_cb_f32[slot], up_idx[slot], up_cols,
                                           gateup_in_dim, rows, up_tmp);
                for (uint32_t row = 0; row < rows; row++) {
                    float gv = gate_tmp[row];
                    float uv = up_tmp[row];
                    if (gv > swiglu_limit) gv = swiglu_limit;
                    if (uv > swiglu_limit) uv = swiglu_limit;
                    if (uv < -swiglu_limit) uv = -swiglu_limit;
                    ref[(uint64_t)slot * rows + row] = (gv / (1.0f + expf(-gv))) * uv;
                }
            }
        }
        if (!ok) {
            fprintf(stderr, "ds4_mpsgraph: failed to allocate or prepare gate/up LUT tensors\n");
            ds4_mpsgraph_free_gateup_arrays(n_experts, gate_idx, up_idx, gate_cb_f32, up_cb_f32, gate_cb_f16, up_cb_f16);
            free(up_tmp); free(gate_tmp); free(got); free(ref); free(x_f16); free(x_f32);
            ds4_d8f_close(&file);
            return 0;
        }
        MPSGraph *graph = [[MPSGraph alloc] init];
        graph.options = MPSGraphOptionsDefault;
        MPSDataType data_type = fp32_path ? MPSDataTypeFloat32 : MPSDataTypeFloat16;
        MPSGraphTensor *x = [graph placeholderWithShape:@[@1, @(gateup_in_dim)] dataType:data_type name:@"d8f_gateup_lut_x"];
        id<MTLBuffer> xbuf = [device newBufferWithBytes:(fp32_path ? (const void *)x_f32 : (const void *)x_f16)
                                                 length:(fp32_path ? x_f32_bytes : x_f16_bytes)
                                                options:MTLResourceStorageModeShared];
        MPSGraphTensorData *xdata = [[MPSGraphTensorData alloc] initWithMTLBuffer:xbuf
                                                                            shape:@[@1, @(gateup_in_dim)]
                                                                         dataType:data_type];
        NSMutableArray *slot_mids = [NSMutableArray arrayWithCapacity:n_experts];
        MPSGraphTensor *limit = [graph constantWithScalar:(double)swiglu_limit dataType:data_type];
        MPSGraphTensor *neg_limit = [graph constantWithScalar:(double)-swiglu_limit dataType:data_type];
        for (uint32_t slot = 0; slot < n_experts; slot++) {
            const uint32_t gate_cols = gate_records[slot].k + 1u;
            const uint32_t up_cols = up_records[slot].k + 1u;
            const size_t gate_cb_bytes = (size_t)block * gate_cols * (fp32_path ? sizeof(float) : sizeof(uint16_t));
            const size_t up_cb_bytes = (size_t)block * up_cols * (fp32_path ? sizeof(float) : sizeof(uint16_t));
            NSData *gate_cb = [NSData dataWithBytes:(fp32_path ? (const void *)gate_cb_f32[slot] : (const void *)gate_cb_f16[slot])
                                             length:gate_cb_bytes];
            NSData *up_cb = [NSData dataWithBytes:(fp32_path ? (const void *)up_cb_f32[slot] : (const void *)up_cb_f16[slot])
                                           length:up_cb_bytes];
            NSData *gate_idx_data = [NSData dataWithBytes:gate_idx[slot] length:idx_bytes];
            NSData *up_idx_data = [NSData dataWithBytes:up_idx[slot] length:idx_bytes];
            MPSGraphTensor *gate_cb_tensor = [graph constantWithData:gate_cb shape:@[@(block), @(gate_cols)] dataType:data_type];
            MPSGraphTensor *up_cb_tensor = [graph constantWithData:up_cb shape:@[@(block), @(up_cols)] dataType:data_type];
            MPSGraphTensor *gate_idx_tensor = [graph constantWithData:gate_idx_data shape:@[@(groups), @(rows)] dataType:MPSDataTypeInt32];
            MPSGraphTensor *up_idx_tensor = [graph constantWithData:up_idx_data shape:@[@(groups), @(rows)] dataType:MPSDataTypeInt32];
            MPSGraphTensor *gate_raw = ds4_mpsgraph_build_lut(graph, x, gate_cb_tensor, gate_idx_tensor, gateup_in_dim, rows);
            MPSGraphTensor *up_raw = ds4_mpsgraph_build_lut(graph, x, up_cb_tensor, up_idx_tensor, gateup_in_dim, rows);
            MPSGraphTensor *gate_clamped = [graph minimumWithPrimaryTensor:gate_raw secondaryTensor:limit name:@"d8f_gate_clamp"];
            MPSGraphTensor *up_clamped = [graph clampWithTensor:up_raw minValueTensor:neg_limit maxValueTensor:limit name:@"d8f_up_clamp"];
            MPSGraphTensor *sig = [graph sigmoidWithTensor:gate_clamped name:@"d8f_gate_sigmoid"];
            MPSGraphTensor *silu = [graph multiplicationWithPrimaryTensor:gate_clamped secondaryTensor:sig name:@"d8f_gate_silu"];
            MPSGraphTensor *mid = [graph multiplicationWithPrimaryTensor:silu secondaryTensor:up_clamped name:@"d8f_gateup_mid"];
            [slot_mids addObject:mid];
        }
        MPSGraphTensor *out = n_experts == 1u ? slot_mids[0] :
            [graph concatTensors:slot_mids dimension:0 name:@"d8f_gateup_selected_concat"];
        NSDictionary *feeds = @{ x: xdata };
        NSDictionary *result = [graph runWithMTLCommandQueue:queue feeds:feeds targetTensors:@[out] targetOperations:nil];
        MPSGraphTensorData *out_data = result[out];
        if (!out_data) {
            fprintf(stderr, "ds4_mpsgraph: no gate/up output tensor data\n");
            ds4_mpsgraph_free_gateup_arrays(n_experts, gate_idx, up_idx, gate_cb_f32, up_cb_f32, gate_cb_f16, up_cb_f16);
            free(up_tmp); free(gate_tmp); free(got); free(ref); free(x_f16); free(x_f32);
            ds4_d8f_close(&file);
            return 0;
        }
        MPSNDArray *nd = [out_data mpsndarray];
        if (fp32_path) {
            [nd readBytes:got strideBytes:nil];
        } else {
            uint16_t *got_h = (uint16_t *)calloc(out_count, sizeof(uint16_t));
            if (!got_h) {
                fprintf(stderr, "ds4_mpsgraph: gate/up output allocation failed\n");
                ds4_mpsgraph_free_gateup_arrays(n_experts, gate_idx, up_idx, gate_cb_f32, up_cb_f32, gate_cb_f16, up_cb_f16);
                free(up_tmp); free(gate_tmp); free(got); free(ref); free(x_f16); free(x_f32);
                ds4_d8f_close(&file);
                return 0;
            }
            [nd readBytes:got_h strideBytes:nil];
            for (size_t i = 0; i < out_count; i++) got[i] = ds4_mpsgraph_f16_to_f32(got_h[i]);
            free(got_h);
        }
        double max_abs = 0.0, max_rel = 0.0, rms = 0.0;
        uint32_t bad = 0;
        const double tol_abs = fp32_path ? 2.0e-3 : 4.0e-2;
        const double tol_rel = fp32_path ? 2.0e-3 : 2.5e-2;
        for (size_t i = 0; i < out_count; i++) {
            const double diff = fabs((double)got[i] - (double)ref[i]);
            const double denom = fmax(1.0, fabs((double)ref[i]));
            const double rel = diff / denom;
            if (diff > max_abs) max_abs = diff;
            if (rel > max_rel) max_rel = rel;
            rms += diff * diff;
            if (diff > tol_abs && rel > tol_rel) bad++;
        }
        rms = sqrt(rms / (double)out_count);
        MPSGraphShapedType *x_type = [[MPSGraphShapedType alloc] initWithShape:@[@1, @(gateup_in_dim)] dataType:data_type];
        MPSGraphExecutable *executable = [graph compileWithDevice:nil feeds:@{ x: x_type } targetTensors:@[out] targetOperations:nil compilationDescriptor:nil];
        executable.options = MPSGraphOptionsNone;
        const size_t timed_out_bytes = fp32_path ? out_bytes : out_count * sizeof(uint16_t);
        id<MTLBuffer> timed_out = [device newBufferWithLength:timed_out_bytes options:MTLResourceStorageModeShared];
        MPSGraphTensorData *timed_out_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:timed_out
                                                                                     shape:@[@(out_count)]
                                                                                  dataType:data_type];
        NSArray *inputs = @[xdata];
        NSArray *outputs = @[timed_out_data];
        for (uint32_t i = 0; i < 3u; i++) {
            [executable runWithMTLCommandQueue:queue inputsArray:inputs resultsArray:outputs executionDescriptor:nil];
        }
        const double t0 = ds4_mpsgraph_seconds();
        for (uint32_t i = 0; i < rounds; i++) {
            [executable runWithMTLCommandQueue:queue inputsArray:inputs resultsArray:outputs executionDescriptor:nil];
        }
        const double elapsed = ds4_mpsgraph_seconds() - t0;
        const double us_per = elapsed * 1.0e6 / (double)rounds;
        double logical_bytes = (double)(x_f16_bytes + out_count * sizeof(uint16_t));
        for (uint32_t slot = 0; slot < n_experts; slot++) {
            const uint32_t gate_cols = gate_records[slot].k + 1u;
            const uint32_t up_cols = up_records[slot].k + 1u;
            logical_bytes += (double)((size_t)block * (gate_cols + up_cols) * sizeof(uint16_t) +
                                      idx_bytes * 2u +
                                      (size_t)groups * (gate_cols + up_cols) * (fp32_path ? 4u : 2u));
        }
        char expert_csv[128];
        expert_csv[0] = '\0';
        for (uint32_t slot = 0; slot < n_experts; slot++) {
            char item[24];
            snprintf(item, sizeof(item), "%s%u", slot ? "," : "", experts[slot]);
            strlcat(expert_csv, item, sizeof(expert_csv));
        }
        fprintf(stderr,
                "ds4_mpsgraph: d8f_gateup_lut_selected path=%s experts=%s nsel=%u rows=%u max_k=%u mode=%s rounds=%u clamp=%.1f us/op=%.3f logical_MB/op=%.3f bad=%u max_abs=%.6g max_rel=%.6g rms=%.6g sample_ref=%.6g sample_got=%.6g\n",
                d8f_path, expert_csv, n_experts, rows, max_k, fp32_path ? "fp32" : "fp16",
                rounds, swiglu_limit, us_per, logical_bytes / 1.0e6,
                bad, max_abs, max_rel, rms, (double)ref[0], (double)got[0]);
        ds4_mpsgraph_free_gateup_arrays(n_experts, gate_idx, up_idx, gate_cb_f32, up_cb_f32, gate_cb_f16, up_cb_f16);
        free(up_tmp); free(gate_tmp); free(got); free(ref); free(x_f16); free(x_f32);
        ds4_d8f_close(&file);
        return bad == 0 ? 1 : 0;
    }
}
