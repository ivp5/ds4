/* vqb2_score_gather_canary.c — CPU canary for VQB2 score-gather math.
 *
 * silv 2026-05-28 Direction C step 1: prove the math BEFORE touching Metal.
 *
 * Codex H2231 measured 3.72-4.01× speedup for K4/K16, 2.88× for K256 on AMD
 * for score-gather vs decode-materialize. H2244 verified margin-safe precision
 * on real 37GB VQB2 pack (fp16-store + fp32-accum, rel_L2 ~2e-4). This canary
 * is the prerequisite for the Metal kernel: it proves both paths produce the
 * same output on the production K=256 pack, so the Metal kernel just has to
 * match a known-correct reference.
 *
 * Algorithm comparison:
 *
 *   DECODE-MATERIALIZE (current production):
 *     For each (row r, pair p):
 *       code = codes[expert_off + r * n_pairs + p]
 *       (re, im) = codebook[code]
 *     Then matmul: output[p] = sum_r input[r,0]*tile[r,p,0] + input[r,1]*tile[r,p,1]
 *
 *   SCORE-GATHER (this canary):
 *     For each (row r, code k): score[r,k] = input[r,0]*cb[k,0] + input[r,1]*cb[k,1]
 *     For each pair p: output[p] = sum_r score[r, codes[expert_off + r*n_pairs + p]]
 *
 * FLOPs (n_rows=128, n_pairs=2048, K=256):
 *   Decode-materialize: 128 × 2048 × 4 = 1,048,576 mul-add FLOPs (analytic)
 *   Score-gather:        128 × 256 × 4 + 128 × 2048 × 1 = 393,216 FLOPs
 *   Ratio: 2.67× (matches H2231 K256 measurement of 2.88× within noise)
 *
 * What this canary measures:
 *   - max_abs_diff between output_decode and output_score (should be 0 since
 *     same math, same precision)
 *   - rel_L2 between outputs (sanity)
 *   - argmax preservation across paths (the actual softmax-feeding metric)
 *   - wall-clock time for each path (CPU baseline; GPU will be faster)
 *
 * If max_abs_diff > 1e-4 with fp32-accum, the math identity is broken.
 * If argmax differs, score-gather is unsafe even before precision concerns.
 *
 * Build: bundled as a static-link standalone with ds4_vqb2_pack.c + ds4_vqb2_reader.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>

#include "../ds4_vqb2_pack.h"
#include "../ds4_vqb2_reader.h"

/* ----- Deterministic LCG (no rand() RNG-state global state) ----- */
static uint64_t g_lcg = 0x100000000ULL;
static void lcg_seed(uint64_t s) { g_lcg = s | 1ULL; }
static float lcg_normal(void) {
    /* Box-Muller from two uniform [0,1) draws. */
    g_lcg = g_lcg * 6364136223846793005ULL + 1442695040888963407ULL;
    float u1 = (float)((g_lcg >> 32) & 0xFFFFFFFF) / 4294967296.0f;
    if (u1 < 1e-30f) u1 = 1e-30f;
    g_lcg = g_lcg * 6364136223846793005ULL + 1442695040888963407ULL;
    float u2 = (float)((g_lcg >> 32) & 0xFFFFFFFF) / 4294967296.0f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
}

/* ----- Time helpers ----- */
static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ============================================================================
 * Path A: decode-materialize (mirrors kernel_vqb2_fp16_gate_up at line 57 of
 * ds4_metal_vqb2_fp16.m). Decode every (row, pair) for one expert into a
 * materialized half-tile, then matmul.
 *
 * For honest CPU-side comparison, we use plain float instead of half. The
 * Metal kernel internally accumulates as float; the half->float conversion
 * round-trip would only add quantization noise, not the algorithmic property
 * we want to compare.
 * ============================================================================ */
static double path_a_decode_materialize(
    const ds4_vqb2_file *view,
    uint32_t expert,
    const float *input,         /* n_rows * 2 floats (re/im pairs) */
    float       *output         /* n_pairs floats */
) {
    const uint32_t n_rows  = view->n_rows;
    const uint32_t n_pairs = view->n_pairs;

    /* Materialize the entire expert tile into local memory (mimics what the
     * decode pass writes into the hot-store). For n_rows=128, n_pairs=2048,
     * tile = 524288 elements × 4 bytes float = 2 MB. */
    float *tile = (float *)malloc((size_t)n_rows * n_pairs * 2 * sizeof(float));
    if (!tile) { perror("tile alloc"); exit(1); }

    const double t0 = now_seconds();

    /* Decode pass */
    for (uint32_t r = 0; r < n_rows; r++) {
        for (uint32_t p = 0; p < n_pairs; p++) {
            float re, im;
            if (!ds4_vqb2_decode_pair(view, expert, r, p, &re, &im)) {
                fprintf(stderr, "decode failed at (e=%u,r=%u,p=%u)\n",
                        expert, r, p);
                exit(1);
            }
            tile[(r * n_pairs + p) * 2 + 0] = re;
            tile[(r * n_pairs + p) * 2 + 1] = im;
        }
    }

    /* Matmul pass — matches the kernel's loop structure */
    for (uint32_t p = 0; p < n_pairs; p++) {
        float acc = 0.0f;  /* fp32 accumulator */
        for (uint32_t r = 0; r < n_rows; r++) {
            const float in_re = input[r * 2 + 0];
            const float in_im = input[r * 2 + 1];
            const size_t base = ((size_t)r * n_pairs + p) * 2;
            acc += in_re * tile[base + 0] + in_im * tile[base + 1];
        }
        output[p] = acc;
    }

    const double t1 = now_seconds();
    free(tile);
    return t1 - t0;
}

/* ============================================================================
 * Path B: score-gather. Build score table per row × codebook entry, then
 * gather. For K=256, score table is n_rows × K × sizeof(float) = 128 × 256 × 4
 * = 128 KB. CPU L2 cache easily fits this; on GPU this won't fit in TG memory
 * (32 KB limit) so the Metal kernel will use device buffer for score table.
 *
 * Math is mathematically identical to path A — both compute
 * output[p] = sum_r input[r,0] * cb[code(r,p),0] + input[r,1] * cb[code(r,p),1].
 *
 * Score-gather just factors the codebook lookup out of the inner loop:
 *   score[r,k] precomputes the dot product input[r] . codebook[k]
 *   gather    uses code(r,p) as index into score[r,*] for the accumulation
 * ============================================================================ */
static double path_b_score_gather(
    const ds4_vqb2_file *view,
    uint32_t expert,
    const float *input,
    float       *output
) {
    const uint32_t n_rows  = view->n_rows;
    const uint32_t n_pairs = view->n_pairs;
    const uint32_t K       = view->k;

    /* Score table: n_rows × K floats */
    float *score = (float *)malloc((size_t)n_rows * K * sizeof(float));
    if (!score) { perror("score alloc"); exit(1); }

    const double t0 = now_seconds();

    /* Step 1: build score table */
    for (uint32_t r = 0; r < n_rows; r++) {
        const float in_re = input[r * 2 + 0];
        const float in_im = input[r * 2 + 1];
        for (uint32_t k = 0; k < K; k++) {
            const float cb_re = view->codebook[k * 2 + 0];
            const float cb_im = view->codebook[k * 2 + 1];
            score[r * K + k] = in_re * cb_re + in_im * cb_im;
        }
    }

    /* Step 2: gather + accumulate.
     * For each output pair p, walk rows, fetch code, index into score. */
    for (uint32_t p = 0; p < n_pairs; p++) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < n_rows; r++) {
            const uint32_t code = ds4_vqb2_get_code(view, expert, r, p);
            acc += score[r * K + code];
        }
        output[p] = acc;
    }

    const double t1 = now_seconds();
    free(score);
    return t1 - t0;
}

/* ============================================================================
 * Comparison metrics.
 * ============================================================================ */
typedef struct {
    float    max_abs_diff;
    float    rel_l2;
    uint32_t argmax_a;
    uint32_t argmax_b;
    float    max_a;
    float    max_b;
    uint32_t n_differ_above_eps;
} compare_t;

static compare_t compare(const float *a, const float *b, uint32_t n, float eps) {
    compare_t r = {0};
    double ssq_diff = 0.0, ssq_a = 0.0;
    float max_a = a[0], max_b = b[0];
    uint32_t arg_a = 0, arg_b = 0;
    for (uint32_t i = 0; i < n; i++) {
        const float d = fabsf(a[i] - b[i]);
        if (d > r.max_abs_diff) r.max_abs_diff = d;
        if (d > eps) r.n_differ_above_eps++;
        ssq_diff += (double)d * d;
        ssq_a    += (double)a[i] * a[i];
        if (a[i] > max_a) { max_a = a[i]; arg_a = i; }
        if (b[i] > max_b) { max_b = b[i]; arg_b = i; }
    }
    r.rel_l2 = (ssq_a > 0) ? (float)sqrt(ssq_diff / ssq_a) : 0.0f;
    r.argmax_a = arg_a;
    r.argmax_b = arg_b;
    r.max_a = max_a;
    r.max_b = max_b;
    return r;
}

/* ============================================================================
 * Main: walk a sample of pack entries, run both paths, report.
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
        fprintf(stderr, "ds4_vqb2_pack_open failed\n");
        return 1;
    }
    fprintf(stderr, "pack opened: %u entries\n", pk.n_entries);

    /* Sample selection: ensure coverage of each K-tier present in the pack
     * (K=4, K=16, K=256 per H2203's K-mix policy). For each K, take first
     * occurrence + a deep-layer occurrence (layer >= 30) to expose any
     * layer-specific encoding drift. Plus first and last entries as anchors. */
    uint32_t sample_indices[16];
    uint32_t n_samples = 0;

    /* First/last anchors */
    sample_indices[n_samples++] = 0;
    sample_indices[n_samples++] = pk.n_entries - 1;

    /* Find first occurrence of each distinct K + a deep-layer occurrence */
    uint32_t seen_first[4] = {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
    uint32_t seen_deep[4]  = {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
    for (uint32_t i = 0; i < pk.n_entries; i++) {
        const uint32_t k_log = (pk.entries[i].k == 4) ? 0 :
                               (pk.entries[i].k == 16) ? 1 :
                               (pk.entries[i].k == 64) ? 2 :
                               (pk.entries[i].k == 256) ? 3 : UINT32_MAX;
        if (k_log == UINT32_MAX) continue;
        if (seen_first[k_log] == UINT32_MAX) seen_first[k_log] = i;
        if (pk.entries[i].layer >= 30 && seen_deep[k_log] == UINT32_MAX)
            seen_deep[k_log] = i;
    }
    for (uint32_t k = 0; k < 4 && n_samples < 16; k++) {
        if (seen_first[k] != UINT32_MAX) sample_indices[n_samples++] = seen_first[k];
        if (seen_deep[k]  != UINT32_MAX && n_samples < 16)
            sample_indices[n_samples++] = seen_deep[k];
    }

    int n_pass = 0;
    int n_argmax_match = 0;
    int n_runs = 0;
    double t_a_total = 0, t_b_total = 0;

    /* Per-cell tolerance: same input + same math should give zero diff.
     * Float-add reordering can produce ~1e-5 noise on 128-element sums; set
     * the alarm at 1e-3 to catch real algorithmic divergence. */
    const float TOL_ABS = 1e-3f;
    const float TOL_REL_L2 = 1e-3f;

    for (uint32_t s = 0; s < n_samples; s++) {
        const uint32_t entry_idx = sample_indices[s];
        if (entry_idx >= pk.n_entries) continue;

        ds4_vqb2_file view = {0};
        if (!ds4_vqb2_pack_view_from_entry(&pk, entry_idx, &view)) {
            fprintf(stderr, "view_from_entry %u failed\n", entry_idx);
            continue;
        }
        const ds4_vqb2_pack_entry *e = &pk.entries[entry_idx];
        fprintf(stderr, "\n--- entry %u: layer=%u kind=%u K=%u row_start=%u "
                "n_experts=%u n_rows=%u n_pairs=%u bit_width=%u ---\n",
                entry_idx, e->layer, e->kind_id, view.k, e->row_start,
                view.n_experts, view.n_rows, view.n_pairs, view.bit_width);

        /* Pick one expert (use 0 for simplicity; could iterate later) */
        const uint32_t expert = 0;
        const uint32_t n_rows = view.n_rows;
        const uint32_t n_pairs = view.n_pairs;

        /* Generate deterministic input (Gaussian) */
        lcg_seed(0x5e7e51d50 + entry_idx);
        float *input = (float *)malloc(n_rows * 2 * sizeof(float));
        for (uint32_t i = 0; i < n_rows * 2; i++) input[i] = lcg_normal();

        /* Run both paths */
        float *out_a = (float *)calloc(n_pairs, sizeof(float));
        float *out_b = (float *)calloc(n_pairs, sizeof(float));

        const double t_a = path_a_decode_materialize(&view, expert, input, out_a);
        const double t_b = path_b_score_gather       (&view, expert, input, out_b);

        compare_t cmp = compare(out_a, out_b, n_pairs, TOL_ABS);

        const bool argmax_ok = (cmp.argmax_a == cmp.argmax_b);
        const bool abs_ok    = (cmp.max_abs_diff < TOL_ABS);
        const bool rel_ok    = (cmp.rel_l2 < TOL_REL_L2);
        const bool pass      = abs_ok && rel_ok && argmax_ok;

        fprintf(stderr, "  decode-materialize: %.3f ms  (output[0..2]=%.4f,%.4f,%.4f, argmax=%u v=%.4f)\n",
                t_a * 1000.0, out_a[0], out_a[1], out_a[2], cmp.argmax_a, cmp.max_a);
        fprintf(stderr, "  score-gather      : %.3f ms  (output[0..2]=%.4f,%.4f,%.4f, argmax=%u v=%.4f)\n",
                t_b * 1000.0, out_b[0], out_b[1], out_b[2], cmp.argmax_b, cmp.max_b);
        fprintf(stderr, "  cmp: max_abs=%.3e rel_L2=%.3e n_diff>%.0e=%u argmax_match=%s  -> %s\n",
                cmp.max_abs_diff, cmp.rel_l2, TOL_ABS, cmp.n_differ_above_eps,
                argmax_ok ? "yes" : "NO",
                pass ? "PASS" : "FAIL");
        fprintf(stderr, "  speedup-projection (CPU): %.2fx %s\n",
                t_a / t_b, t_a > t_b ? "(score-gather faster)" : "(materialize faster)");

        if (pass) n_pass++;
        if (argmax_ok) n_argmax_match++;
        n_runs++;
        t_a_total += t_a;
        t_b_total += t_b;

        free(input); free(out_a); free(out_b);
    }

    fprintf(stderr, "\n=== SUMMARY ===\n");
    fprintf(stderr, "runs: %d, pass: %d, argmax-match: %d\n", n_runs, n_pass, n_argmax_match);
    fprintf(stderr, "total time decode-materialize: %.1f ms\n", t_a_total * 1000.0);
    fprintf(stderr, "total time score-gather      : %.1f ms\n", t_b_total * 1000.0);
    fprintf(stderr, "aggregate CPU speedup        : %.2fx\n", t_a_total / t_b_total);
    fprintf(stderr, "(GPU/Metal projection per H2231 AMD: K=256 2.88x, K=16 3.88x, K=4 3.97x)\n");

    ds4_vqb2_pack_close(&pk);
    return (n_pass == n_runs) ? 0 : 1;
}
