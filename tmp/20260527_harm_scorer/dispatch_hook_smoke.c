/* silv 2026-05-27 task #651 Phase A.1 — CPU dispatch hook smoke.
 *
 * Verifies that the organ-skip flags produce SEMANTICALLY DISTINCT output
 * vectors via ds4_hot_dispatch_layer_cpu, refuting the naive
 * "zero-organ-output" collapse hypothesis. The honest semantics:
 *   GATE skip → gate_out := 1.0 → mid = silu(1)*up*ew = 0.731*up*ew
 *   UP   skip → up_out   := 1.0 → mid = silu(gate)*ew
 *   DOWN skip → expert contributes nothing
 *
 * Builds a synthetic hot-store with one expert at (layer=0, expert=0),
 * populates the heap with deterministic FP16 values, runs dispatch four
 * times (baseline, skip_gate, skip_up, skip_down), asserts all four
 * outputs are MUTUALLY DISTINCT.
 *
 * Build (from ivp5_ds4):
 *   cc -O3 -ffast-math -mcpu=native -Wall -Wextra -I. \
 *      tmp/20260527_harm_scorer/dispatch_hook_smoke.c \
 *      ds4_expert_table.o ds4_vqb2_reader.o ds4_polar_reader.o \
 *      ds4_vqb1_reader.o \
 *      -lm -o tmp/20260527_harm_scorer/dispatch_hook_smoke
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ds4_expert_table.h"

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); return 1; } \
    printf("  PASS: %s\n", msg); \
} while (0)

/* Tile shapes per the dispatch's hardcoded DS4-Flash sizes. */
#define N_ROWS  128
#define N_PAIRS 2048
#define N_DOWN  1024
#define GATE_HALVES (N_ROWS * N_PAIRS * 2)
#define UP_HALVES   GATE_HALVES
#define DOWN_HALVES (N_DOWN * N_ROWS * 2)
#define GATE_BYTES  (GATE_HALVES * sizeof(_Float16))
#define UP_BYTES    UP_BYTES_VAL
#define UP_BYTES_VAL (UP_HALVES * sizeof(_Float16))
#define DOWN_BYTES  (DOWN_HALVES * sizeof(_Float16))
#define EXPERT_BYTES (GATE_BYTES + UP_BYTES_VAL + DOWN_BYTES)

/* Fill an FP16 tile with reproducible pseudo-random values in [-0.5, 0.5]. */
static void fill_tile_random(_Float16 *tile, size_t n_halves, uint32_t seed) {
    /* Simple xorshift32 → small float for repro. */
    uint32_t s = seed;
    for (size_t i = 0; i < n_halves; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        const float v = (float)(int32_t)(s) / (float)0x80000000u;  /* [-1, 1) */
        tile[i] = (_Float16)(v * 0.5f);  /* [-0.5, 0.5) */
    }
}

static double l2_diff(const float *a, const float *b, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        s += d * d;
    }
    return sqrt(s);
}

static double l2_norm(const float *a, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; i++) s += (double)a[i] * (double)a[i];
    return sqrt(s);
}

int main(void) {
    /* Allocate store with budget for one expert. */
    const uint64_t budget = (uint64_t)EXPERT_BYTES + 1024;
    ds4_hot_expert_store *store = ds4_hot_expert_store_alloc(budget);
    CHECK(store != NULL, "store alloc");

    /* Manually pin one expert at (layer=0, expert=0) by writing to the
     * heap + offsets directly. This bypasses the production pin path
     * (which requires real IQ2_XXS data) and gives us deterministic
     * synthetic FP16 tiles. */
    const uint32_t L = 0, E = 0;
    const size_t idx = (size_t)L * 256 + E;
    _Float16 *heap = (_Float16 *)ds4_hot_store_heap_ptr(store);
    CHECK(heap != NULL, "heap ptr non-NULL");

    /* Layout: gate | up | down */
    _Float16 *gate_dst = heap;
    _Float16 *up_dst   = heap + GATE_HALVES;
    _Float16 *down_dst = heap + GATE_HALVES + UP_HALVES;
    fill_tile_random(gate_dst, GATE_HALVES, 0xC0FFEE01);
    fill_tile_random(up_dst,   UP_HALVES,   0xC0FFEE02);
    fill_tile_random(down_dst, DOWN_HALVES, 0xC0FFEE03);

    store->gate_offset[idx] = 0;
    store->up_offset[idx]   = (int64_t)GATE_BYTES;
    store->down_offset[idx] = (int64_t)(GATE_BYTES + UP_BYTES_VAL);
    store->gate_row_blocks[idx] = DS4_VQB2_GATE_UP_FULL_ROW_MASK;
    store->up_row_blocks[idx]   = DS4_VQB2_GATE_UP_FULL_ROW_MASK;
    store->down_row_blocks[idx] = DS4_VQB2_DOWN_FULL_ROW_MASK;
    store->heap_bytes = EXPERT_BYTES;
    store->n_pinned = 1;
    printf("  setup: pinned one expert (%.2f MB total tiles)\n",
           EXPERT_BYTES / (1024.0 * 1024.0));

    /* Deterministic input vector: 7168 floats. The dispatch reshapes
     * this as [n_rows × 2] = [128 × 56], so the relevant dims are
     * input_fp32[r*2 + 0..1]. We supply input[*]. */
    const uint32_t n_embd = 256;  /* dispatch only reads first n_rows*2 = 256 */
    float input[256];
    {
        uint32_t s = 0xABCD1234;
        for (int i = 0; i < 256; i++) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            input[i] = ((float)(int32_t)s / (float)0x80000000u) * 0.5f;
        }
    }

    const int32_t selected[1] = { (int32_t)E };
    const float expert_weights[1] = { 1.0f };
    const uint32_t n_selected = 1;

    float out_baseline[N_ROWS * 2] = {0};
    float out_skip_gate[N_ROWS * 2] = {0};
    float out_skip_up[N_ROWS * 2]   = {0};
    float out_skip_down[N_ROWS * 2] = {0};

    /* (1) Baseline — no skip flags. */
    ds4_organ_skip_reset();
    int rc = ds4_hot_dispatch_layer_cpu(store, L, selected, expert_weights,
                                        n_selected, input, out_baseline, n_embd);
    CHECK(rc == 0, "baseline dispatch rc=0");
    const double bn = l2_norm(out_baseline, N_ROWS * 2);
    printf("  baseline ||out||_2 = %.4f\n", bn);
    CHECK(bn > 1e-3, "baseline produces non-trivial output");

    /* (2) Skip GATE only — gate_out := 1.0 (silu(1)≈0.731 scaling on up). */
    ds4_organ_skip_reset();
    ds4_organ_skip_set(L, E, DS4_ORGAN_GATE, 1);
    rc = ds4_hot_dispatch_layer_cpu(store, L, selected, expert_weights,
                                    n_selected, input, out_skip_gate, n_embd);
    CHECK(rc == 0, "skip_gate dispatch rc=0");
    const double gn = l2_norm(out_skip_gate, N_ROWS * 2);
    printf("  skip_gate ||out||_2 = %.4f\n", gn);

    /* (3) Skip UP only — up_out := 1.0 (silu(gate) flows through ew). */
    ds4_organ_skip_reset();
    ds4_organ_skip_set(L, E, DS4_ORGAN_UP, 1);
    rc = ds4_hot_dispatch_layer_cpu(store, L, selected, expert_weights,
                                    n_selected, input, out_skip_up, n_embd);
    CHECK(rc == 0, "skip_up dispatch rc=0");
    const double un = l2_norm(out_skip_up, N_ROWS * 2);
    printf("  skip_up ||out||_2 = %.4f\n", un);

    /* (4) Skip DOWN — expert is fully ablated; output untouched (zero). */
    ds4_organ_skip_reset();
    ds4_organ_skip_set(L, E, DS4_ORGAN_DOWN, 1);
    rc = ds4_hot_dispatch_layer_cpu(store, L, selected, expert_weights,
                                    n_selected, input, out_skip_down, n_embd);
    CHECK(rc == 0, "skip_down dispatch rc=0");
    const double dn = l2_norm(out_skip_down, N_ROWS * 2);
    printf("  skip_down ||out||_2 = %.4f\n", dn);

    /* (5) Skip DOWN means full-expert ablation — output should be ZERO. */
    CHECK(dn < 1e-6, "skip_down → output is zero (whole-expert ablation)");

    /* (6) GATE skip and UP skip should both produce NON-ZERO outputs
     *     distinct from baseline (refutes the SwiGLU collapse hypothesis). */
    CHECK(gn > 1e-3, "skip_gate output non-zero (SwiGLU coupling NOT collapsed)");
    CHECK(un > 1e-3, "skip_up output non-zero (SwiGLU coupling NOT collapsed)");

    /* (7) All four outputs should be MUTUALLY DISTINCT. */
    const double d_base_gate = l2_diff(out_baseline, out_skip_gate, N_ROWS * 2);
    const double d_base_up   = l2_diff(out_baseline, out_skip_up,   N_ROWS * 2);
    const double d_base_down = l2_diff(out_baseline, out_skip_down, N_ROWS * 2);
    const double d_gate_up   = l2_diff(out_skip_gate, out_skip_up,  N_ROWS * 2);
    const double d_gate_down = l2_diff(out_skip_gate, out_skip_down, N_ROWS * 2);
    const double d_up_down   = l2_diff(out_skip_up,  out_skip_down, N_ROWS * 2);
    printf("  pairwise distances:\n");
    printf("    base-gate=%.4f  base-up=%.4f  base-down=%.4f\n",
           d_base_gate, d_base_up, d_base_down);
    printf("    gate-up=%.4f   gate-down=%.4f  up-down=%.4f\n",
           d_gate_up, d_gate_down, d_up_down);

    const double thresh = 1e-3;
    CHECK(d_base_gate > thresh, "baseline ≠ skip_gate");
    CHECK(d_base_up   > thresh, "baseline ≠ skip_up");
    CHECK(d_base_down > thresh, "baseline ≠ skip_down");
    CHECK(d_gate_up   > thresh, "skip_gate ≠ skip_up");
    CHECK(d_gate_down > thresh, "skip_gate ≠ skip_down");
    CHECK(d_up_down   > thresh, "skip_up ≠ skip_down");

    ds4_hot_expert_store_free(store);
    printf("\nALL PASS — per-organ skip flags produce 4 distinct outputs\n");
    printf("           SwiGLU coupling collapse refuted via direct dispatch test\n");
    return 0;
}
