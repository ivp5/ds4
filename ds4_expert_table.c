/* DS4 expert-table — runtime per-(layer, expert) tier classification.
 *
 * : implements the unified substrate spec from
 * audits/expert_table_design_20260523.md. Storage + loader + router-intercept
 * helper. Integration into ds4.c documented in ds4_expert_table.h.
 *
 * Per CLAUDE.md STICKY HAZARD: any DS4 launch using this code must include
 * --prefill-metal-phases auto. The mask infrastructure does NOT change the
 * wired-memory cap; it changes WHICH experts are dispatched.
 */

#include "ds4_expert_table.h"
#include "ds4_vqb2_reader.h"  /* silv 2026-05-26: VQB2 → FP16 populator */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

uint8_t g_expert_tier[DS4_N_LAYER * DS4_N_EXPERT] = {0};
int g_expert_table_initialized = 0;

/* Set by ds4.c at engine init after weights_bind. The intercept in
 * layer_topk_selected_experts_from_probs uses pointer arithmetic against
 * this base to derive the layer index. Forward-declared in ds4.c as well. */
const void *g_ds4_layer_base_for_mask = NULL;

static int load_mask_csv(const char *path) {
 FILE *f = fopen(path, "r");
 if (!f) {
 fprintf(stderr, "ds4_load_expert_table: cannot open mask file '%s'\n", path);
 return -1;
 }
 int n_masked = 0;
 int n_lines = 0;
 char line[256];
 while (fgets(line, sizeof(line), f)) {
 n_lines++;
 /* Skip header / comment / blank lines */
 if (line[0] == '#' || line[0] == '\n' || line[0] == '\r' || line[0] == '\0') continue;
 int layer = -1, expert = -1;
 if (sscanf(line, "%d,%d", &layer, &expert) != 2) {
 /* Skip any malformed line silently — manifests may have header
 * variations from build_expert_mask.py / pin_top*.csv etc. */
 continue;
 }
 if (layer < 0 || layer >= DS4_N_LAYER) continue;
 if (expert < 0 || expert >= DS4_N_EXPERT) continue;
 /* Optional safety: refuse to mask layers 0-2 (hash layers per
 * trim_ladder_to_64gb_20260523.md). Comment-out if needed for
 * testing. */
 if (layer < 3) {
 fprintf(stderr, "ds4_load_expert_table: refusing to mask hash-layer %d expert %d\n",
 layer, expert);
 continue;
 }
 g_expert_tier[layer * DS4_N_EXPERT + expert] = DS4_EXPERT_TIER_MASKED;
 n_masked++;
 }
 fclose(f);
 fprintf(stderr, "ds4_load_expert_table: %d experts MASKED from %s (%d lines parsed)\n",
 n_masked, path, n_lines);
 return n_masked;
}

static int load_hot_csv(const char *path) {
 FILE *f = fopen(path, "r");
 if (!f) {
 fprintf(stderr, "ds4_load_expert_table: cannot open hot file '%s'\n", path);
 return -1;
 }
 int n_hot = 0;
 int n_lines = 0;
 char line[256];
 while (fgets(line, sizeof(line), f)) {
 n_lines++;
 if (line[0] == '#' || line[0] == '\n' || line[0] == '\r' || line[0] == '\0') continue;
 int layer = -1, expert = -1;
 if (sscanf(line, "%d,%d", &layer, &expert) != 2) continue;
 if (layer < 0 || layer >= DS4_N_LAYER) continue;
 if (expert < 0 || expert >= DS4_N_EXPERT) continue;
 /* Mask wins over hot — leave masked entries alone */
 if (g_expert_tier[layer * DS4_N_EXPERT + expert] != DS4_EXPERT_TIER_MASKED) {
 g_expert_tier[layer * DS4_N_EXPERT + expert] = DS4_EXPERT_TIER_HOT;
 n_hot++;
 }
 }
 fclose(f);
 fprintf(stderr, "ds4_load_expert_table: %d experts marked HOT from %s (%d lines parsed)\n",
 n_hot, path, n_lines);
 return n_hot;
}

int ds4_load_expert_table(const char *mask_path, const char *hot_path) {
 memset(g_expert_tier, 0, sizeof(g_expert_tier));
 g_expert_table_initialized = 0;

 int err = 0;
 if (mask_path && *mask_path) {
 if (load_mask_csv(mask_path) < 0) err = -1;
 }
 if (hot_path && *hot_path) {
 if (load_hot_csv(hot_path) < 0) err = -1;
 }

 if (err) return -1;

 /* Mark initialized only if at least one file was loaded; otherwise
 * the table is logically inactive (all NORMAL). */
 if ((mask_path && *mask_path) || (hot_path && *hot_path)) {
 g_expert_table_initialized = 1;
 }
 return 0;
}

void ds4_apply_expert_mask_to_selection(uint32_t layer, float *selection) {
 if (!g_expert_table_initialized) return;
 if (layer >= DS4_N_LAYER) return;
 const uint8_t *tier_row = &g_expert_tier[layer * DS4_N_EXPERT];
 for (int i = 0; i < DS4_N_EXPERT; i++) {
 if (tier_row[i] == DS4_EXPERT_TIER_MASKED) {
 selection[i] = -INFINITY;
 }
 }
}

/* =========================================================================
 * HOT EXPERT PRE-DEQUANT (silv 2026-05-26: advance further)
 *
 * Storage + diagnostic counters. The dispatch wiring lives in ds4.c (which
 * has access to the IQ2_XXS dequant helpers and the CPU-MoE inner loop).
 * ========================================================================= */

uint64_t g_ds4_hot_hits = 0;
uint64_t g_ds4_hot_misses = 0;
uint64_t g_ds4_masked_skips = 0;
uint64_t g_ds4_total_dispatches = 0;

ds4_hot_expert_store *ds4_hot_expert_store_alloc(uint64_t budget_bytes) {
    ds4_hot_expert_store *store = (ds4_hot_expert_store *)calloc(1, sizeof(*store));
    if (!store) return NULL;
    const size_t offsets_bytes = (size_t)DS4_N_LAYER * DS4_N_EXPERT * sizeof(int64_t);
    store->gate_offset = (int64_t *)malloc(offsets_bytes);
    store->up_offset   = (int64_t *)malloc(offsets_bytes);
    store->down_offset = (int64_t *)malloc(offsets_bytes);
    if (!store->gate_offset || !store->up_offset || !store->down_offset) {
        ds4_hot_expert_store_free(store);
        return NULL;
    }
    /* Initialize all offsets to -1 (not pinned). */
    for (uint64_t i = 0; i < (uint64_t)DS4_N_LAYER * DS4_N_EXPERT; i++) {
        store->gate_offset[i] = -1;
        store->up_offset[i]   = -1;
        store->down_offset[i] = -1;
    }
    store->fp16_heap = NULL;
    store->heap_bytes = 0;
    store->budget_bytes = budget_bytes;
    store->n_pinned = 0;
    return store;
}

void ds4_hot_expert_store_free(ds4_hot_expert_store *store) {
    if (!store) return;
    if (store->gate_offset) free(store->gate_offset);
    if (store->up_offset)   free(store->up_offset);
    if (store->down_offset) free(store->down_offset);
    if (store->fp16_heap)   free(store->fp16_heap);
    free(store);
}

/* Stub populator: validates budget, returns -1 because actual IQ2_XXS → FP16
 * dequant lives in ds4.c. The wiring contract is here; the data lives there.
 *
 * Future work: when ds4.c calls this, it passes the pre-dequantized FP16
 * buffer (computed via ds4_dequantize_iq2_xxs_to_f16 or equivalent), and
 * we copy into the heap. For now the function returns -1 to signal
 * "not yet implemented" without breaking the calling code.
 */
int ds4_hot_pin_expert(ds4_hot_expert_store *store,
                       uint32_t layer, uint32_t expert,
                       const void *gate_iq2, const void *up_iq2, const void *down_iq2,
                       uint64_t gate_bytes, uint64_t up_bytes, uint64_t down_bytes) {
    (void)layer; (void)expert;
    (void)gate_iq2; (void)up_iq2; (void)down_iq2;
    if (!store) return -1;
    const uint64_t needed = gate_bytes + up_bytes + down_bytes;
    if (store->heap_bytes + needed > store->budget_bytes) {
        fprintf(stderr, "ds4_hot_pin_expert: budget exceeded (need %llu, have %llu/%llu)\n",
                (unsigned long long)needed,
                (unsigned long long)store->heap_bytes,
                (unsigned long long)store->budget_bytes);
        return -1;
    }
    /* TODO(silv-approval): wire actual IQ2_XXS → FP16 dequant from ds4.c.
     * Until then, return -1 so callers know the pin didn't succeed. */
    return -1;
}

void *ds4_hot_store_heap_ptr(const ds4_hot_expert_store *store) {
    return store ? store->fp16_heap : NULL;
}

uint64_t ds4_hot_store_heap_bytes_get(const ds4_hot_expert_store *store) {
    return store ? store->heap_bytes : 0ULL;
}

const void *ds4_hot_get_gate_fp16(const ds4_hot_expert_store *store,
                                  uint32_t layer, uint32_t expert) {
    if (!store || !store->fp16_heap) return NULL;
    if (layer >= DS4_N_LAYER || expert >= DS4_N_EXPERT) return NULL;
    const int64_t off = store->gate_offset[layer * DS4_N_EXPERT + expert];
    if (off < 0) return NULL;
    return (const void *)((const char *)store->fp16_heap + off);
}

const void *ds4_hot_get_up_fp16(const ds4_hot_expert_store *store,
                                uint32_t layer, uint32_t expert) {
    if (!store || !store->fp16_heap) return NULL;
    if (layer >= DS4_N_LAYER || expert >= DS4_N_EXPERT) return NULL;
    const int64_t off = store->up_offset[layer * DS4_N_EXPERT + expert];
    if (off < 0) return NULL;
    return (const void *)((const char *)store->fp16_heap + off);
}

const void *ds4_hot_get_down_fp16(const ds4_hot_expert_store *store,
                                  uint32_t layer, uint32_t expert) {
    if (!store || !store->fp16_heap) return NULL;
    if (layer >= DS4_N_LAYER || expert >= DS4_N_EXPERT) return NULL;
    const int64_t off = store->down_offset[layer * DS4_N_EXPERT + expert];
    if (off < 0) return NULL;
    return (const void *)((const char *)store->fp16_heap + off);
}

void ds4_hot_print_stats(void) {
    if (g_ds4_total_dispatches == 0) {
        fprintf(stderr, "ds4_hot: no dispatches counted (counter wiring incomplete?)\n");
        return;
    }
    const double total = (double)g_ds4_total_dispatches;
    fprintf(stderr,
        "ds4_hot dispatch stats: total=%llu, hot=%llu (%.2f%%), normal=%llu (%.2f%%), "
        "masked_skips=%llu (%.4f%%)\n",
        (unsigned long long)g_ds4_total_dispatches,
        (unsigned long long)g_ds4_hot_hits,   100.0 * (double)g_ds4_hot_hits / total,
        (unsigned long long)g_ds4_hot_misses, 100.0 * (double)g_ds4_hot_misses / total,
        (unsigned long long)g_ds4_masked_skips, 100.0 * (double)g_ds4_masked_skips / total);
    if (g_ds4_masked_skips > 0) {
        fprintf(stderr,
            "ds4_hot: WARNING — masked_skips > 0 means dispatch path is calling "
            "into MASKED experts. Check that ds4_apply_expert_mask_to_selection() "
            "runs BEFORE expert dispatch.\n");
    }
}

void ds4_hot_reset_stats(void) {
    g_ds4_hot_hits = 0;
    g_ds4_hot_misses = 0;
    g_ds4_masked_skips = 0;
    g_ds4_total_dispatches = 0;
}

/* =========================================================================
 * Global active hot-store + dispatch helpers (silv 2026-05-26).
 * ========================================================================= */

static ds4_hot_expert_store *g_active_hot_store = NULL;

void ds4_hot_store_set_active(ds4_hot_expert_store *store) {
    g_active_hot_store = store;
    if (store) {
        fprintf(stderr,
            "ds4_hot_store_set_active: %u tiles, heap %.1f MB ready for dispatch\n",
            store->n_pinned, store->heap_bytes / 1e6);
    }
}

ds4_hot_expert_store *ds4_hot_store_get_active(void) {
    return g_active_hot_store;
}

int ds4_hot_layer_all_pinned(const ds4_hot_expert_store *store,
                             uint32_t layer,
                             const int32_t *selected,
                             uint32_t n_selected) {
    if (!store || !selected || layer >= DS4_N_LAYER) return 0;
    for (uint32_t i = 0; i < n_selected; i++) {
        const int32_t e = selected[i];
        if (e < 0 || (uint32_t)e >= DS4_N_EXPERT) return 0;
        const uint64_t idx = (uint64_t)layer * DS4_N_EXPERT + (uint64_t)e;
        if (store->gate_offset[idx] < 0) return 0;
        if (store->up_offset[idx]   < 0) return 0;
        if (store->down_offset[idx] < 0) return 0;
    }
    return 1;
}

/* SwiGLU clamp (matches DS4_SWIGLU_CLAMP_EXP behavior in ds4.c — keep
 * synchronized with that value). For the dispatch fallback we use a
 * conservative default; ds4.c can override per layer if needed. */
#ifndef DS4_HOT_SWIGLU_CLAMP
#define DS4_HOT_SWIGLU_CLAMP 1.0e4f
#endif

int ds4_hot_dispatch_layer_cpu(const ds4_hot_expert_store *store,
                               uint32_t layer,
                               const int32_t *selected,
                               const float *expert_weights,
                               uint32_t n_selected,
                               const float *input_fp32,
                               float *output_fp32,
                               uint32_t n_embd) {
    if (!store || !selected || !expert_weights || !input_fp32 || !output_fp32) return -1;
    if (layer >= DS4_N_LAYER || n_selected == 0) return -1;

    /* Per-expert tile dims pulled from a representative entry. All experts
     * within a (layer, kind) share dims, so we use selected[0]'s gate to
     * sample n_rows / n_pairs. The VQB2 reader stored these in n_rows and
     * n_pairs at file-open time; the FP16 tile layout is
     *   gate: [n_rows × n_pairs × 2] FP16 with pair-interleaving (re, im)
     *   up:   same shape as gate
     *   down: [n_pairs/2 × n_rows × 2] (down has half n_pairs per VQB2 header)
     *
     * For DS4-Flash routed FFN: input_fp32 [n_embd=7168] is reshaped as
     * [n_rows × 2] = [128 × 56] where n_rows=128 and the pair-dim of input
     * is 56 = n_embd / n_rows.
     *
     * NOTE: the tile layout differs from a standard dense matmul because
     * VQB2 stores 2-tuples. The matmul math here matches the
     * vqb2_to_fp16_test.c primitive (validated at 118 t/s single thread).
     *
     * For the FIRST integration version we do the simplest correctness-
     * preserving math: 6 sequential expert matvecs accumulated into output.
     */

    (void)n_embd; /* not used directly; layout encoded in pinned tile shape */

    /* For each selected expert: gate × input → gate_out; up × input → up_out;
     * SwiGLU(gate_out, up_out) → mid; down × mid → expert_output;
     * output += expert_weight × expert_output */

    /* Scratch buffers sized for the worst-case layout (gate/up n_pairs=2048,
     * down n_rows=128). The actual values come from store-pinned tile
     * dimensions. */
    enum { MAX_PAIRS = 2048 };
    float gate_out[MAX_PAIRS];
    float up_out[MAX_PAIRS];
    float mid[MAX_PAIRS];

    for (uint32_t s = 0; s < n_selected; s++) {
        const int32_t exp_id = selected[s];
        if (exp_id < 0) continue;
        const _Float16 *gate_tile = (const _Float16 *)ds4_hot_get_gate_fp16(store, layer, (uint32_t)exp_id);
        const _Float16 *up_tile   = (const _Float16 *)ds4_hot_get_up_fp16(store, layer, (uint32_t)exp_id);
        const _Float16 *down_tile = (const _Float16 *)ds4_hot_get_down_fp16(store, layer, (uint32_t)exp_id);
        if (!gate_tile || !up_tile || !down_tile) return -1;

        /* For DS4-Flash: n_rows=128, n_pairs=2048 for gate/up; n_pairs=1024
         * for down. The values are encoded in the tile dimensions when the
         * store was populated. For simplicity at this stage we hardcode the
         * DS4-Flash values; the production version reads them from
         * store->n_rows[layer] / store->n_pairs[layer][kind]. */
        const uint32_t n_rows  = 128;
        const uint32_t n_pairs = 2048;     /* gate/up */
        const uint32_t n_down  = 1024;     /* down n_pairs (half) */

        /* gate_out[p] = sum over rows: input[r*2+0] * gate[r*np+p][0] +
         *                              input[r*2+1] * gate[r*np+p][1] */
        for (uint32_t p = 0; p < n_pairs; p++) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < n_rows; r++) {
                const uint32_t base = (r * n_pairs + p) * 2;
                acc += input_fp32[r * 2 + 0] * (float)gate_tile[base + 0]
                     + input_fp32[r * 2 + 1] * (float)gate_tile[base + 1];
            }
            gate_out[p] = acc;
        }

        /* up_out[p] = sum over rows: input · up[r*np+p] */
        for (uint32_t p = 0; p < n_pairs; p++) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < n_rows; r++) {
                const uint32_t base = (r * n_pairs + p) * 2;
                acc += input_fp32[r * 2 + 0] * (float)up_tile[base + 0]
                     + input_fp32[r * 2 + 1] * (float)up_tile[base + 1];
            }
            up_out[p] = acc;
        }

        /* SwiGLU fused with expert weight: mid[i] = swiglu(gate, up) * weight */
        const float ew = expert_weights[s];
        for (uint32_t p = 0; p < n_pairs; p++) {
            float g = gate_out[p];
            if (g >  DS4_HOT_SWIGLU_CLAMP) g =  DS4_HOT_SWIGLU_CLAMP;
            if (g < -DS4_HOT_SWIGLU_CLAMP) g = -DS4_HOT_SWIGLU_CLAMP;
            const float sig = 1.0f / (1.0f + expf(-g));
            mid[p] = g * sig * up_out[p] * ew;
        }

        /* down: output += mid · down_tile (transposed layout: down_pair index
         * outer, row inner). down tile shape: [n_down × n_rows × 2] FP16 */
        for (uint32_t r = 0; r < n_rows; r++) {
            float acc_re = 0.0f, acc_im = 0.0f;
            for (uint32_t p = 0; p < n_down; p++) {
                const uint32_t base = (p * n_rows + r) * 2;
                acc_re += mid[p] * (float)down_tile[base + 0];
                acc_im += mid[p] * (float)down_tile[base + 1];
            }
            /* Note: this writes back into a (n_rows × 2) reshape of output.
             * The caller must align with this layout convention. */
            output_fp32[r * 2 + 0] += acc_re;
            output_fp32[r * 2 + 1] += acc_im;
        }
    }

    return 0;
}

/* =========================================================================
 * VQB2 → FP16 populator (silv 2026-05-26 "not iq2_xxs" path)
 *
 * Apple M1 P-cores have native FP16 support (ARMv8.2-a+fp16). The compiler
 * exposes _Float16 as a built-in scalar type when -mcpu=native is on.
 * We use it directly to avoid an explicit conversion routine.
 * ========================================================================= */

/* Grow the FP16 heap if needed. Reallocates and returns 0 on success or
 * -1 on alloc failure / budget overrun. */
static int hot_store_grow_to(ds4_hot_expert_store *store, uint64_t new_bytes) {
    if (new_bytes <= store->heap_bytes) return 0;
    if (new_bytes > store->budget_bytes) {
        fprintf(stderr, "ds4_hot: budget overrun (need %llu, budget %llu)\n",
                (unsigned long long)new_bytes,
                (unsigned long long)store->budget_bytes);
        return -1;
    }
    /* Round up to 64-byte alignment for SIMD. */
    uint64_t aligned = (new_bytes + 63u) & ~(uint64_t)63u;
    if (aligned > store->budget_bytes) aligned = store->budget_bytes;
    void *new_heap = realloc(store->fp16_heap, (size_t)aligned);
    if (!new_heap) {
        fprintf(stderr, "ds4_hot: realloc fail (need %llu bytes)\n",
                (unsigned long long)aligned);
        return -1;
    }
    /* Zero the newly-grown region. */
    if (aligned > store->heap_bytes) {
        memset((char *)new_heap + store->heap_bytes, 0,
               (size_t)(aligned - store->heap_bytes));
    }
    store->fp16_heap = new_heap;
    store->heap_bytes = aligned;
    return 0;
}

/* Forward declaration for the recursive directory walker. */
static int corpus_load_recursive(ds4_hot_expert_store *store, const char *dir);

int ds4_vqb2_corpus_load(ds4_hot_expert_store *store, const char *corpus_dir) {
    if (!store || !corpus_dir) return -1;
    return corpus_load_recursive(store, corpus_dir);
}

/* =========================================================================
 * Candidate-keyed VQB2 loader (codex H1884 patch verbatim).
 *
 * Manifest CSV format:
 *   candidate,layer,kind,k,path
 * Loader validates VQB2 header against layer/kind/k; rejects duplicate
 * (layer, kind) rows for the selected candidate.
 * ========================================================================= */

static void ds4_csv_trim(char *s) {
    if (!s) return;
    char *p = s;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (p != s) memmove(s, p, strlen(p) + 1);
    size_t n = strlen(s);
    while (n > 0 && (s[n-1] == ' ' || s[n-1] == '\t' || s[n-1] == '\r' || s[n-1] == '\n')) {
        s[--n] = '\0';
    }
}

static int ds4_parse_kind_token(const char *kind) {
    if (!kind) return -1;
    if (strcmp(kind, "gate") == 0 || strcmp(kind, "0") == 0) return 0;
    if (strcmp(kind, "up")   == 0 || strcmp(kind, "1") == 0) return 1;
    if (strcmp(kind, "down") == 0 || strcmp(kind, "2") == 0) return 2;
    return -1;
}

static int ds4_candidate_manifest_pin_file(ds4_hot_expert_store *store,
                                           const char *path,
                                           uint32_t expected_layer,
                                           uint32_t expected_kind,
                                           uint32_t expected_k) {
    ds4_vqb2_file f;
    if (!ds4_vqb2_open(path, &f)) {
        fprintf(stderr, "ds4_vqb2_candidate_manifest_load: open failed: %s\n", path);
        return -1;
    }
    if (f.layer != expected_layer || f.kind_id != expected_kind || f.k != expected_k) {
        fprintf(stderr,
                "ds4_vqb2_candidate_manifest_load: header mismatch path=%s "
                "manifest=(L%u kind=%u K%u) header=(L%u kind=%u K%u)\n",
                path, expected_layer, expected_kind, expected_k,
                f.layer, f.kind_id, f.k);
        ds4_vqb2_close(&f);
        return -1;
    }
    int pinned = 0;
    for (uint32_t e = 0; e < f.n_experts; e++) {
        if (ds4_hot_pin_expert_from_vqb2(store, f.layer, f.kind_id, &f, e) != 0) {
            fprintf(stderr,
                    "ds4_vqb2_candidate_manifest_load: pin failed path=%s expert=%u "
                    "after %d experts; aborting candidate load\n",
                    path, e, pinned);
            ds4_vqb2_close(&f);
            return -1;
        }
        pinned++;
    }
    ds4_vqb2_close(&f);
    return pinned;
}

int ds4_vqb2_candidate_manifest_load(ds4_hot_expert_store *store,
                                     const char *manifest_csv,
                                     const char *candidate_name) {
    if (!store || !manifest_csv || !candidate_name || !*candidate_name) return -1;
    FILE *file = fopen(manifest_csv, "r");
    if (!file) {
        fprintf(stderr, "ds4_vqb2_candidate_manifest_load: cannot open %s\n", manifest_csv);
        return -1;
    }

    uint8_t seen[DS4_N_LAYER][3] = {{0}};
    char line[4096];
    int total_pinned = 0;
    int selected_rows = 0;
    int line_no = 0;
    while (fgets(line, sizeof(line), file)) {
        line_no++;
        ds4_csv_trim(line);
        if (line[0] == '\0' || line[0] == '#') continue;
        if (line_no == 1 && strstr(line, "candidate") && strstr(line, "layer")) continue;

        char candidate[128] = {0};
        char kind_token[32] = {0};
        char path[3072] = {0};
        int layer = -1;
        int k = -1;
        if (sscanf(line, "%127[^,],%d,%31[^,],%d,%3071[^\n]",
                   candidate, &layer, kind_token, &k, path) != 5) {
            fprintf(stderr, "ds4_vqb2_candidate_manifest_load: malformed line %d: %s\n",
                    line_no, line);
            fclose(file);
            return -1;
        }
        ds4_csv_trim(candidate);
        ds4_csv_trim(kind_token);
        ds4_csv_trim(path);
        if (strcmp(candidate, candidate_name) != 0) continue;

        const int kind = ds4_parse_kind_token(kind_token);
        if (layer < 0 || layer >= DS4_N_LAYER || kind < 0 || kind > 2 || (k != 16 && k != 64 && k != 256)) {
            fprintf(stderr, "ds4_vqb2_candidate_manifest_load: bad key at line %d: %s\n",
                    line_no, line);
            fclose(file);
            return -1;
        }
        if (seen[layer][kind]) {
            fprintf(stderr,
                    "ds4_vqb2_candidate_manifest_load: duplicate candidate row for "
                    "%s L%d kind=%d; refusing ambiguous overwrite\n",
                    candidate_name, layer, kind);
            fclose(file);
            return -1;
        }
        seen[layer][kind] = 1;

        const int pinned = ds4_candidate_manifest_pin_file(store, path, (uint32_t)layer, (uint32_t)kind, (uint32_t)k);
        if (pinned < 0) {
            fclose(file);
            return -1;
        }
        total_pinned += pinned;
        selected_rows++;
    }
    fclose(file);
    if (selected_rows == 0) {
        fprintf(stderr, "ds4_vqb2_candidate_manifest_load: candidate '%s' has no rows in %s\n",
                candidate_name, manifest_csv);
        return -1;
    }
    fprintf(stderr,
            "ds4_vqb2_candidate_manifest_load: candidate=%s rows=%d pinned=%d from %s\n",
            candidate_name, selected_rows, total_pinned, manifest_csv);
    return total_pinned;
}

void ds4_hot_expert_store_print(const ds4_hot_expert_store *store) {
    if (!store) { fprintf(stderr, "ds4_hot_expert_store: (null)\n"); return; }
    fprintf(stderr,
        "ds4_hot_expert_store: %u pinned, heap %.1f / %.1f MB (budget %.1f MB)\n",
        store->n_pinned,
        store->heap_bytes / 1e6,
        store->heap_bytes / 1e6,
        store->budget_bytes / 1e6);
    /* Per-layer pin distribution */
    int per_layer_pins[DS4_N_LAYER] = {0};
    for (uint32_t L = 0; L < DS4_N_LAYER; L++) {
        for (uint32_t E = 0; E < DS4_N_EXPERT; E++) {
            const uint64_t idx = (uint64_t)L * DS4_N_EXPERT + E;
            if (store->gate_offset && store->gate_offset[idx] >= 0) per_layer_pins[L]++;
            if (store->up_offset   && store->up_offset[idx]   >= 0) per_layer_pins[L]++;
            if (store->down_offset && store->down_offset[idx] >= 0) per_layer_pins[L]++;
        }
    }
    fprintf(stderr, "  per-layer pin counts (gate+up+down):\n");
    for (uint32_t L = 0; L < DS4_N_LAYER; L++) {
        if (per_layer_pins[L] > 0) {
            fprintf(stderr, "    L%-2u: %4d pins\n", L, per_layer_pins[L]);
        }
    }
}

int ds4_hot_pin_expert_from_vqb2(ds4_hot_expert_store *store,
                                 uint32_t layer,
                                 uint32_t kind,
                                 const struct ds4_vqb2_file *vqb2_opaque,
                                 uint32_t expert) {
    if (!store || !vqb2_opaque) return -1;
    if (layer >= DS4_N_LAYER || expert >= DS4_N_EXPERT) return -1;
    if (kind > 2) return -1;
    const ds4_vqb2_file *f = (const ds4_vqb2_file *)vqb2_opaque;
    if (expert >= f->n_experts) return -1;

    /* Compute the per-expert tile size in FP16 bytes:
     *   n_rows × n_pairs × 2 (re, im) × sizeof(_Float16) */
    const uint64_t per_expert_floats = (uint64_t)f->n_rows * f->n_pairs * 2;
    const uint64_t per_expert_bytes  = per_expert_floats * 2; /* fp16 */

    /* Reserve space at the current heap tail. */
    const uint64_t off = store->heap_bytes; /* before grow */
    const uint64_t new_bytes = off + per_expert_bytes;
    if (hot_store_grow_to(store, new_bytes) != 0) return -1;

    /* Dequant: walk codes, look up codebook, write FP16 into heap. */
    _Float16 *dst = (_Float16 *)((char *)store->fp16_heap + off);
    const uint32_t bw = f->bit_width;
    const uint8_t mask = f->code_mask;
    const uint8_t *codes = f->codes;
    const float *cb = f->codebook;
    const uint64_t base = (uint64_t)expert * f->n_rows * f->n_pairs;

    for (uint64_t i = 0; i < (uint64_t)f->n_rows * f->n_pairs; i++) {
        const uint64_t bit_off = (base + i) * bw;
        const size_t byte_off = (size_t)(bit_off >> 3);
        const uint32_t shift = (uint32_t)(bit_off & 7ull);
        const uint32_t lo = codes[byte_off];
        const uint32_t hi = (bw + shift > 8u) ? codes[byte_off + 1u] : 0u;
        const uint32_t window = lo | (hi << 8);
        const uint32_t code = (window >> shift) & (uint32_t)mask;
        dst[i * 2 + 0] = (_Float16)cb[code * 2 + 0];
        dst[i * 2 + 1] = (_Float16)cb[code * 2 + 1];
    }

    /* Record offset in the appropriate kind's offset table. */
    int64_t *offsets = NULL;
    switch (kind) {
        case 0: offsets = store->gate_offset; break;
        case 1: offsets = store->up_offset;   break;
        case 2: offsets = store->down_offset; break;
    }
    if (offsets) {
        offsets[layer * DS4_N_EXPERT + expert] = (int64_t)off;
    }
    store->n_pinned++;
    return 0;
}

/* =========================================================================
 * Recursive corpus directory walker (silv 2026-05-26)
 *
 * Opens every `*.vqb2` file under corpus_dir, reads its header to route by
 * (layer, kind_id), and pins all experts via ds4_hot_pin_expert_from_vqb2.
 * Header is the source of truth — no naming convention required.
 * ========================================================================= */
#include <dirent.h>
#include <sys/stat.h>

static int has_vqb2_ext(const char *name) {
    const size_t n = strlen(name);
    if (n < 5) return 0;
    return strcmp(name + n - 5, ".vqb2") == 0;
}

static int corpus_load_file(ds4_hot_expert_store *store, const char *path) {
    ds4_vqb2_file f;
    if (!ds4_vqb2_open(path, &f)) {
        fprintf(stderr, "ds4_vqb2_corpus_load: skip %s (open failed)\n", path);
        return 0;
    }
    fprintf(stderr, "  load %s (layer=%u kind=%u %u experts)\n",
            path, f.layer, f.kind_id, f.n_experts);
    int n_pinned_this_file = 0;
    for (uint32_t e = 0; e < f.n_experts; e++) {
        if (ds4_hot_pin_expert_from_vqb2(store, f.layer, f.kind_id, &f, e) == 0) {
            n_pinned_this_file++;
        } else {
            /* Budget overrun → stop trying further experts in this file. */
            fprintf(stderr, "  ds4_vqb2_corpus_load: budget exhausted at file %s, "
                            "expert %u; pinned %d so far this file\n",
                    path, e, n_pinned_this_file);
            break;
        }
    }
    ds4_vqb2_close(&f);
    return n_pinned_this_file;
}

static int corpus_load_recursive(ds4_hot_expert_store *store, const char *dir) {
    DIR *d = opendir(dir);
    if (!d) {
        fprintf(stderr, "ds4_vqb2_corpus_load: opendir(%s) failed\n", dir);
        return -1;
    }
    int total_pinned = 0;
    struct dirent *entry;
    while ((entry = readdir(d)) != NULL) {
        if (entry->d_name[0] == '.') continue;  /* skip ., .., hidden */

        /* Construct full path */
        char path[2048];
        int n = snprintf(path, sizeof(path), "%s/%s", dir, entry->d_name);
        if (n <= 0 || (size_t)n >= sizeof(path)) continue;

        struct stat st;
        if (stat(path, &st) != 0) continue;

        if (S_ISDIR(st.st_mode)) {
            const int sub = corpus_load_recursive(store, path);
            if (sub > 0) total_pinned += sub;
        } else if (S_ISREG(st.st_mode) && has_vqb2_ext(entry->d_name)) {
            total_pinned += corpus_load_file(store, path);
        }
    }
    closedir(d);
    return total_pinned;
}

int ds4_expert_table_print_stats(void) {
 if (!g_expert_table_initialized) {
 fprintf(stderr, "ds4_expert_table: not initialized (no manifests loaded)\n");
 return 0;
 }
 int total_masked = 0, total_hot = 0;
 fprintf(stderr, "ds4_expert_table per-layer stats:\n");
 fprintf(stderr, " L masked hot normal\n");
 for (int L = 0; L < DS4_N_LAYER; L++) {
 int n_m = 0, n_h = 0;
 for (int e = 0; e < DS4_N_EXPERT; e++) {
 uint8_t t = g_expert_tier[L * DS4_N_EXPERT + e];
 if (t == DS4_EXPERT_TIER_MASKED) n_m++;
 else if (t == DS4_EXPERT_TIER_HOT) n_h++;
 }
 total_masked += n_m;
 total_hot += n_h;
 if (n_m > 0 || n_h > 0) {
 fprintf(stderr, " %2d %5d %4d %5d\n", L, n_m, n_h, DS4_N_EXPERT - n_m - n_h);
 }
 }
 fprintf(stderr, " TOTAL: %d masked, %d hot, %d normal (of %d total)\n",
 total_masked, total_hot, DS4_N_LAYER * DS4_N_EXPERT - total_masked - total_hot,
 DS4_N_LAYER * DS4_N_EXPERT);
 return total_masked;
}
