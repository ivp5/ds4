/* vqb2_score_gather_metal_canary.m — Metal canary for VQB2 score-gather.
 *
 * silv 2026-05-28 Direction C step 2: GPU kernel + measurement.
 * Step 1 (tools/vqb2_score_gather_canary.c) proved the CPU math identity
 * (6/6 PASS, max_abs=0, K=4/16/256 covered). This canary writes the MSL
 * kernel, dispatches it on the production VQB2 pack, and compares against
 * a CPU reference output.
 *
 * Two MSL kernels:
 *   1. kernel_vqb2_score_table:
 *        score[r, k] = input[r,0] * codebook[k,0] + input[r,1] * codebook[k,1]
 *        Output: device buffer (n_rows × K) fp16 values, fp32 accumulation.
 *        H2244 verified this precision policy on real 37GB VQB2 pack.
 *
 *   2. kernel_vqb2_score_gather_gate_up:
 *        For each output pair p:
 *          acc = 0 (fp32)
 *          for each row r:
 *            code = codes[expert_off + r * n_pairs + p]
 *            acc += score[r * K + code]
 *          output[p] = acc
 *
 * Bit-width handling: codes are packed 2/4/6/8-bit per code. For score-gather,
 * we decode bits-to-code in-kernel (no separate decode pass). K=4→2-bit,
 * K=16→4-bit, K=64→6-bit, K=256→8-bit. The kernel takes bit_width as constant
 * and does the inline bit extract.
 *
 * Build:
 *   clang -O2 -framework Foundation -framework Metal -o vqb2_score_gather_metal_canary \
 *     tools/vqb2_score_gather_metal_canary.m ds4_vqb2_pack.c ds4_vqb2_reader.c ds4_expert_table.c
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "../ds4_vqb2_pack.h"
#include "../ds4_vqb2_reader.h"

/* ============================================================================
 * MSL source — score-table + score-gather, with inline bit-width-aware code
 * extraction. Single source string compiled at runtime via newLibraryWithSource.
 * ============================================================================ */
static const char *msl_source =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    "// Score table: per (row r, code k) compute input[r] . codebook[k].\n"
    "// fp16 storage, fp32 accumulation per H2244 finding.\n"
    "kernel void kernel_vqb2_score_table(\n"
    "    device const float *input        [[buffer(0)]],  // n_rows * 2 floats\n"
    "    device const float *codebook     [[buffer(1)]],  // K * 2 floats\n"
    "    device       half  *score        [[buffer(2)]],  // n_rows * K halfs\n"
    "    constant      uint &n_rows       [[buffer(3)]],\n"
    "    constant      uint &K            [[buffer(4)]],\n"
    "    uint2 gid [[thread_position_in_grid]])\n"
    "{\n"
    "    const uint r = gid.y;\n"
    "    const uint k = gid.x;\n"
    "    if (r >= n_rows || k >= K) return;\n"
    "    const float in_re = input[r * 2 + 0];\n"
    "    const float in_im = input[r * 2 + 1];\n"
    "    const float cb_re = codebook[k * 2 + 0];\n"
    "    const float cb_im = codebook[k * 2 + 1];\n"
    "    score[r * K + k] = (half)(in_re * cb_re + in_im * cb_im);\n"
    "}\n"
    "\n"
    "// Gather + accumulate. Bit-width-aware code extraction lets one kernel\n"
    "// serve all K-tiers (2/4/6/8-bit). For each output pair p, walk rows,\n"
    "// decode the (expert, r, p) code, index into score table.\n"
    "kernel void kernel_vqb2_score_gather_gate_up(\n"
    "    device const half    *score        [[buffer(0)]],  // n_rows * K halfs\n"
    "    device const uchar   *codes        [[buffer(1)]],  // bit-packed\n"
    "    device       float   *output       [[buffer(2)]],  // n_pairs floats\n"
    "    constant      uint   &n_rows       [[buffer(3)]],\n"
    "    constant      uint   &n_pairs      [[buffer(4)]],\n"
    "    constant      uint   &K            [[buffer(5)]],\n"
    "    constant      uint   &bit_width    [[buffer(6)]],\n"
    "    constant      uint   &expert_off   [[buffer(7)]],  // expert * n_rows * n_pairs (in CODES)\n"
    "    uint p [[thread_position_in_grid]])\n"
    "{\n"
    "    if (p >= n_pairs) return;\n"
    "    const uint code_mask = (1u << bit_width) - 1u;\n"
    "    float acc = 0.0f;  // fp32 accumulator\n"
    "    for (uint r = 0; r < n_rows; r++) {\n"
    "        // linear code index for (expert, r, p):\n"
    "        const uint linear = expert_off + r * n_pairs + p;\n"
    "        const uint bit_off = linear * bit_width;\n"
    "        const uint byte_off = bit_off >> 3;\n"
    "        const uint shift = bit_off & 7u;\n"
    "        // Fetch 16 bits little-endian; shift; mask. (bit_width <= 8 fits in 2 bytes.)\n"
    "        const uint b0 = (uint)codes[byte_off + 0];\n"
    "        const uint b1 = (uint)codes[byte_off + 1];\n"
    "        const uint code = ((b0 | (b1 << 8)) >> shift) & code_mask;\n"
    "        acc += (float)score[r * K + code];\n"
    "    }\n"
    "    output[p] = acc;\n"
    "}\n";

/* ============================================================================
 * CPU reference paths (mirrors the validated tools/vqb2_score_gather_canary.c
 * for cross-check at the canary level — if Metal output ≡ CPU reference, both
 * the algorithm AND the GPU dispatch are correct).
 * ============================================================================ */
static void cpu_score_gather(const ds4_vqb2_file *view, uint32_t expert,
                              const float *input, float *output) {
    const uint32_t n_rows  = view->n_rows;
    const uint32_t n_pairs = view->n_pairs;
    const uint32_t K       = view->k;
    float *score = (float *)malloc((size_t)n_rows * K * sizeof(float));
    for (uint32_t r = 0; r < n_rows; r++) {
        const float in_re = input[r * 2 + 0];
        const float in_im = input[r * 2 + 1];
        for (uint32_t k = 0; k < K; k++) {
            score[r * K + k] = in_re * view->codebook[k * 2 + 0] +
                               in_im * view->codebook[k * 2 + 1];
        }
    }
    for (uint32_t p = 0; p < n_pairs; p++) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < n_rows; r++) {
            const uint32_t code = ds4_vqb2_get_code(view, expert, r, p);
            acc += score[r * K + code];
        }
        output[p] = acc;
    }
    free(score);
}

/* Deterministic LCG (no global RNG state to share with CPU canary) */
static uint64_t g_lcg = 0x100000000ULL;
static void lcg_seed(uint64_t s) { g_lcg = s | 1ULL; }
static float lcg_normal(void) {
    g_lcg = g_lcg * 6364136223846793005ULL + 1442695040888963407ULL;
    float u1 = (float)((g_lcg >> 32) & 0xFFFFFFFF) / 4294967296.0f;
    if (u1 < 1e-30f) u1 = 1e-30f;
    g_lcg = g_lcg * 6364136223846793005ULL + 1442695040888963407ULL;
    float u2 = (float)((g_lcg >> 32) & 0xFFFFFFFF) / 4294967296.0f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ============================================================================
 * Metal pipeline + dispatch
 * ============================================================================ */
typedef struct {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> lib;
    id<MTLComputePipelineState> pipe_score;
    id<MTLComputePipelineState> pipe_gather;
} metal_ctx_t;

static bool metal_init(metal_ctx_t *m) {
    m->dev = MTLCreateSystemDefaultDevice();
    if (!m->dev) { fprintf(stderr, "no Metal device\n"); return false; }
    m->queue = [m->dev newCommandQueue];

    NSError *err = nil;
    NSString *src = [NSString stringWithUTF8String:msl_source];
    m->lib = [m->dev newLibraryWithSource:src options:nil error:&err];
    if (!m->lib) {
        fprintf(stderr, "MSL compile failed: %s\n",
                err ? err.localizedDescription.UTF8String : "(unknown)");
        return false;
    }

    id<MTLFunction> fn_s = [m->lib newFunctionWithName:@"kernel_vqb2_score_table"];
    id<MTLFunction> fn_g = [m->lib newFunctionWithName:@"kernel_vqb2_score_gather_gate_up"];
    if (!fn_s || !fn_g) {
        fprintf(stderr, "kernel function lookup failed\n");
        return false;
    }
    m->pipe_score  = [m->dev newComputePipelineStateWithFunction:fn_s error:&err];
    m->pipe_gather = [m->dev newComputePipelineStateWithFunction:fn_g error:&err];
    if (!m->pipe_score || !m->pipe_gather) {
        fprintf(stderr, "pipeline build failed: %s\n",
                err ? err.localizedDescription.UTF8String : "(unknown)");
        return false;
    }
    return true;
}

/* Run the Metal score-gather path against one (view, expert, input). */
static double metal_score_gather(metal_ctx_t *m,
                                  const ds4_vqb2_file *view,
                                  uint32_t expert,
                                  const float *input,
                                  float *output) {
    const uint32_t n_rows  = view->n_rows;
    const uint32_t n_pairs = view->n_pairs;
    const uint32_t K       = view->k;
    const uint32_t bit_width = view->bit_width;
    const uint32_t expert_off = expert * n_rows * n_pairs;  /* in codes, not bytes */

    @autoreleasepool {
        /* Input buffers — newBufferWithBytes copies once, no zero-copy here for
         * simplicity; production wiring will use newBufferWithBytesNoCopy on
         * pack mmap. */
        id<MTLBuffer> buf_input    = [m->dev newBufferWithBytes:input
                                                         length:n_rows * 2 * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_codebook = [m->dev newBufferWithBytes:view->codebook
                                                         length:K * 2 * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_codes    = [m->dev newBufferWithBytes:view->codes
                                                         length:view->codes_bytes
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_score    = [m->dev newBufferWithLength:n_rows * K * sizeof(uint16_t)
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_output   = [m->dev newBufferWithLength:n_pairs * sizeof(float)
                                                         options:MTLResourceStorageModeShared];

        const double t0 = now_seconds();

        id<MTLCommandBuffer> cmd = [m->queue commandBuffer];

        /* Dispatch 1: score table */
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:m->pipe_score];
            [enc setBuffer:buf_input    offset:0 atIndex:0];
            [enc setBuffer:buf_codebook offset:0 atIndex:1];
            [enc setBuffer:buf_score    offset:0 atIndex:2];
            [enc setBytes:&n_rows length:sizeof(n_rows) atIndex:3];
            [enc setBytes:&K      length:sizeof(K)      atIndex:4];
            MTLSize grid = MTLSizeMake(K, n_rows, 1);
            /* Threadgroup size: pick a 2D shape that divides K and n_rows. */
            const uint32_t tg_x = (K >= 64) ? 64 : K;
            const uint32_t tg_y = (n_rows >= 4) ? 4 : n_rows;
            MTLSize tg = MTLSizeMake(tg_x, tg_y, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        /* Dispatch 2: gather */
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:m->pipe_gather];
            [enc setBuffer:buf_score    offset:0 atIndex:0];
            [enc setBuffer:buf_codes    offset:0 atIndex:1];
            [enc setBuffer:buf_output   offset:0 atIndex:2];
            [enc setBytes:&n_rows    length:sizeof(n_rows)    atIndex:3];
            [enc setBytes:&n_pairs   length:sizeof(n_pairs)   atIndex:4];
            [enc setBytes:&K         length:sizeof(K)         atIndex:5];
            [enc setBytes:&bit_width length:sizeof(bit_width) atIndex:6];
            [enc setBytes:&expert_off length:sizeof(expert_off) atIndex:7];
            MTLSize grid = MTLSizeMake(n_pairs, 1, 1);
            MTLSize tg = MTLSizeMake(64, 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        [cmd commit];
        [cmd waitUntilCompleted];

        const double t1 = now_seconds();

        memcpy(output, buf_output.contents, n_pairs * sizeof(float));
        return t1 - t0;
    }
}

/* ============================================================================
 * Main: run Metal + CPU paths, compare.
 * ============================================================================ */
int main(int argc, char **argv) {
    const char *pack_path = (argc > 1) ? argv[1] :
        "/Users/silv/cl/tlp/montyneg/ds4/vqb2/DeepSeek-V4-Flash/"
        "nonrotated_layer22_k256_gateup_top4_20260528/pack/"
        "ds4_flash_nonrotated_layer22_k256_gateup_top4.vqb2pack";
    const char *index_path = (argc > 2) ? argv[2] :
        "/Users/silv/cl/tlp/montyneg/ds4/vqb2/DeepSeek-V4-Flash/"
        "nonrotated_layer22_k256_gateup_top4_20260528/pack/"
        "ds4_flash_nonrotated_layer22_k256_gateup_top4.vqb2pack.index.csv";

    fprintf(stderr, "pack:  %s\n", pack_path);
    fprintf(stderr, "index: %s\n", index_path);

    ds4_vqb2_pack pk = {0};
    if (!ds4_vqb2_pack_open(pack_path, index_path, &pk)) {
        fprintf(stderr, "pack open failed\n");
        return 1;
    }

    metal_ctx_t metal = {0};
    if (!metal_init(&metal)) return 2;
    fprintf(stderr, "Metal init OK: %s\n", metal.dev.name.UTF8String);

    /* Sample selection: first entry of each K-tier present in pack. */
    uint32_t seen_first[4] = {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
    for (uint32_t i = 0; i < pk.n_entries; i++) {
        const uint32_t k_log = (pk.entries[i].k == 4) ? 0 :
                               (pk.entries[i].k == 16) ? 1 :
                               (pk.entries[i].k == 64) ? 2 :
                               (pk.entries[i].k == 256) ? 3 : UINT32_MAX;
        if (k_log == UINT32_MAX) continue;
        if (seen_first[k_log] == UINT32_MAX) seen_first[k_log] = i;
    }

    int n_runs = 0, n_pass = 0;
    double t_cpu_total = 0, t_metal_total = 0;

    /* Margin gates per H2244: K=4 highfreq margin-flagged L40 down 3072.
     * General tolerance is fp16-storage-permissive (~3e-4 rel_L2). */
    const float TOL_ABS_FP16 = 5e-3f;  /* fp16 storage adds ~1e-3 abs noise */
    const float TOL_REL_L2   = 1e-3f;

    for (uint32_t k_log = 0; k_log < 4; k_log++) {
        if (seen_first[k_log] == UINT32_MAX) continue;
        const uint32_t entry_idx = seen_first[k_log];

        ds4_vqb2_file view = {0};
        if (!ds4_vqb2_pack_view_from_entry(&pk, entry_idx, &view)) {
            fprintf(stderr, "view %u failed\n", entry_idx);
            continue;
        }
        const ds4_vqb2_pack_entry *e = &pk.entries[entry_idx];
        fprintf(stderr, "\n--- entry %u: layer=%u kind=%u K=%u n_rows=%u n_pairs=%u bit_width=%u ---\n",
                entry_idx, e->layer, e->kind_id, view.k, view.n_rows, view.n_pairs, view.bit_width);

        const uint32_t expert = 0;
        const uint32_t n_rows = view.n_rows;
        const uint32_t n_pairs = view.n_pairs;

        lcg_seed(0xC0DE0000 + entry_idx);
        float *input  = (float *)malloc(n_rows * 2 * sizeof(float));
        for (uint32_t i = 0; i < n_rows * 2; i++) input[i] = lcg_normal();

        float *cpu_out   = (float *)calloc(n_pairs, sizeof(float));
        float *metal_out = (float *)calloc(n_pairs, sizeof(float));

        const double t_cpu = now_seconds();
        cpu_score_gather(&view, expert, input, cpu_out);
        const double t_cpu_dur = now_seconds() - t_cpu;
        const double t_metal_dur = metal_score_gather(&metal, &view, expert, input, metal_out);

        /* Compare element-wise */
        float max_abs = 0.0f;
        double ssq_diff = 0.0, ssq_cpu = 0.0;
        uint32_t arg_cpu = 0, arg_metal = 0;
        float max_cpu = cpu_out[0], max_metal = metal_out[0];
        for (uint32_t i = 0; i < n_pairs; i++) {
            const float d = fabsf(cpu_out[i] - metal_out[i]);
            if (d > max_abs) max_abs = d;
            ssq_diff += (double)d * d;
            ssq_cpu  += (double)cpu_out[i] * cpu_out[i];
            if (cpu_out[i]   > max_cpu)   { max_cpu   = cpu_out[i];   arg_cpu = i; }
            if (metal_out[i] > max_metal) { max_metal = metal_out[i]; arg_metal = i; }
        }
        const float rel_l2 = (ssq_cpu > 0) ? (float)sqrt(ssq_diff / ssq_cpu) : 0.0f;
        const bool argmax_ok = (arg_cpu == arg_metal);
        const bool abs_ok    = (max_abs < TOL_ABS_FP16);
        const bool rel_ok    = (rel_l2 < TOL_REL_L2);
        const bool pass      = abs_ok && rel_ok && argmax_ok;

        fprintf(stderr, "  CPU   score-gather: %.3f ms  (out[0..2]=%.4f,%.4f,%.4f, argmax=%u v=%.4f)\n",
                t_cpu_dur * 1000.0, cpu_out[0], cpu_out[1], cpu_out[2], arg_cpu, max_cpu);
        fprintf(stderr, "  Metal score-gather: %.3f ms  (out[0..2]=%.4f,%.4f,%.4f, argmax=%u v=%.4f)\n",
                t_metal_dur * 1000.0, metal_out[0], metal_out[1], metal_out[2], arg_metal, max_metal);
        fprintf(stderr, "  cmp: max_abs=%.3e rel_L2=%.3e argmax=%s tol_abs=%.0e tol_rel=%.0e -> %s\n",
                max_abs, rel_l2, argmax_ok ? "match" : "DIFFER",
                TOL_ABS_FP16, TOL_REL_L2, pass ? "PASS" : "FAIL");
        fprintf(stderr, "  Metal-vs-CPU ratio: %.2fx %s\n",
                t_cpu_dur / t_metal_dur, t_cpu_dur > t_metal_dur ? "(Metal faster)" : "(CPU faster)");

        if (pass) n_pass++;
        n_runs++;
        t_cpu_total += t_cpu_dur;
        t_metal_total += t_metal_dur;
        free(input); free(cpu_out); free(metal_out);
    }

    fprintf(stderr, "\n=== SUMMARY ===\n");
    fprintf(stderr, "runs: %d, pass: %d\n", n_runs, n_pass);
    fprintf(stderr, "total CPU time:   %.1f ms\n", t_cpu_total * 1000.0);
    fprintf(stderr, "total Metal time: %.1f ms\n", t_metal_total * 1000.0);
    fprintf(stderr, "Metal/CPU ratio:  %.2fx %s\n",
            t_cpu_total / t_metal_total,
            t_cpu_total > t_metal_total ? "(Metal faster)" : "(CPU faster)");
    fprintf(stderr, "(H2231 AMD projection: K=256 2.88x, K=16 3.88x, K=4 3.97x vs DECODE-MATERIALIZE baseline,\n");
    fprintf(stderr, " not vs CPU score-gather; this canary's ratio is buffer-creation-overhead dominated.)\n");

    ds4_vqb2_pack_close(&pk);
    return (n_pass == n_runs) ? 0 : 1;
}
