/* DS4 expert-table — per-(layer, expert) tier classification + hot-store.
 *
 * Three tiers:
 *   NORMAL = 0  IQ2_XXS lazy-mmap dispatch (default)
 *   HOT    = 1  pre-dequantized FP16, fast dispatch
 *   MASKED = 2  never routed; router skips and re-normalizes top-K
 *
 * Loaded from CSV manifests at engine open. See ds4/masks/.
 *
 * SAFETY: any DS4 launch using this code MUST include
 * --prefill-metal-phases auto (or --cpu-moe). The IQ2_XXS model exceeds
 * the M1 Max wired-memory cap (~48 GB) without phase-split prefill.
 */
#ifndef DS4_EXPERT_TABLE_H
#define DS4_EXPERT_TABLE_H

#include <stdint.h>

#ifndef DS4_EXPERT_TABLE_USES_EXTERNAL_DIMS
#ifndef DS4_N_LAYER
#define DS4_N_LAYER 43
#endif
#ifndef DS4_N_EXPERT
#define DS4_N_EXPERT 256
#endif
#endif

typedef enum {
    DS4_EXPERT_TIER_NORMAL = 0,
    DS4_EXPERT_TIER_HOT    = 1,
    DS4_EXPERT_TIER_MASKED = 2,
} ds4_expert_tier;

/* Per-(layer, expert) tier array; 11008 B, L1-resident. */
extern uint8_t g_expert_tier[DS4_N_LAYER * DS4_N_EXPERT];
extern int     g_expert_table_initialized;

/* Load mask + hot CSV manifests. NULL path → skip that tier. MASK wins
 * over HOT if both list the same (layer, expert). Returns 0/-1. */
int ds4_load_expert_table(const char *mask_path, const char *hot_path);

static inline int ds4_expert_is_masked(uint32_t layer, uint32_t expert) {
    extern uint8_t g_expert_tier[DS4_N_LAYER * DS4_N_EXPERT];
    extern int g_expert_table_initialized;
    if (!g_expert_table_initialized) return 0;
    if (layer >= DS4_N_LAYER || expert >= DS4_N_EXPERT) return 0;
    return g_expert_tier[layer * DS4_N_EXPERT + expert] == DS4_EXPERT_TIER_MASKED;
}

static inline int ds4_expert_is_hot(uint32_t layer, uint32_t expert) {
    extern uint8_t g_expert_tier[DS4_N_LAYER * DS4_N_EXPERT];
    extern int g_expert_table_initialized;
    if (!g_expert_table_initialized) return 0;
    if (layer >= DS4_N_LAYER || expert >= DS4_N_EXPERT) return 0;
    return g_expert_tier[layer * DS4_N_EXPERT + expert] == DS4_EXPERT_TIER_HOT;
}

/* Sets masked-expert selection scores to -INFINITY before topk. No-op if
 * table not initialized. Layers 0-2 (hash layers) refused by build script. */
void ds4_apply_expert_mask_to_selection(uint32_t layer, float *selection);
int  ds4_expert_table_print_stats(void);

/* ============================================================================
 * HOT EXPERT PRE-DEQUANT — FP16 tiles in a heap, addressed by per-(layer,
 * expert) byte offsets. Memory budget: hot tile = 88 MB per expert. Realistic
 * cap on M1 Max 64 GB ≈ 30-50 hot experts (~3-5 GB heap).
 * ============================================================================ */

typedef struct ds4_hot_expert_store {
    /* DS4_N_LAYER × DS4_N_EXPERT entries. -1 = not pinned. */
    int64_t  *gate_offset;
    int64_t  *up_offset;
    int64_t  *down_offset;
    /* Row-block coverage bitmasks. Current VQB2 packets encode row block 0
     * only; full hot replacement needs all 16 (gate/up) / 32 (down). */
    uint64_t *gate_row_blocks;
    uint64_t *up_row_blocks;
    uint64_t *down_row_blocks;
    void     *fp16_heap;
    uint64_t  heap_bytes;
    uint64_t  budget_bytes;
    uint32_t  n_pinned;
} ds4_hot_expert_store;

ds4_hot_expert_store *ds4_hot_expert_store_alloc(uint64_t budget_bytes);
void                  ds4_hot_expert_store_free (ds4_hot_expert_store *store);

/* Pin one (layer, expert) by dequantizing IQ2_XXS → FP16 into the heap.
 * Caller passes the IQ2 tile pointers + sizes (ds4.c owns dequant). */
int ds4_hot_pin_expert(ds4_hot_expert_store *store,
                       uint32_t layer, uint32_t expert,
                       const void *gate_iq2, const void *up_iq2, const void *down_iq2,
                       uint64_t gate_bytes, uint64_t up_bytes, uint64_t down_bytes);

/* Metal MTLBuffer wrapping accessors — heap base + length for newBufferWithBytesNoCopy. */
void    *ds4_hot_store_heap_ptr     (const ds4_hot_expert_store *store);
uint64_t ds4_hot_store_heap_bytes_get(const ds4_hot_expert_store *store);

/* Pointer to pinned FP16 tile; NULL if not pinned. */
const void *ds4_hot_get_gate_fp16(const ds4_hot_expert_store *store, uint32_t layer, uint32_t expert);
const void *ds4_hot_get_up_fp16  (const ds4_hot_expert_store *store, uint32_t layer, uint32_t expert);
const void *ds4_hot_get_down_fp16(const ds4_hot_expert_store *store, uint32_t layer, uint32_t expert);

/* Row-block constants. Full row mask must equal these for hot-path. */
#define DS4_VQB2_ROW_BLOCK_ROWS          128ULL
#define DS4_VQB2_GATE_UP_FULL_ROW_BLOCKS 16ULL
#define DS4_VQB2_DOWN_FULL_ROW_BLOCKS    32ULL
#define DS4_VQB2_GATE_UP_FULL_ROW_MASK   ((1ULL << DS4_VQB2_GATE_UP_FULL_ROW_BLOCKS) - 1ULL)
#define DS4_VQB2_DOWN_FULL_ROW_MASK      ((1ULL << DS4_VQB2_DOWN_FULL_ROW_BLOCKS)    - 1ULL)

struct ds4_vqb2_file;  /* opaque forward decl */

/* Pin one tile from a VQB2 packet. `kind` = 0/1/2 (gate/up/down). */
int ds4_hot_pin_expert_from_vqb2(ds4_hot_expert_store *store,
                                 uint32_t layer, uint32_t kind,
                                 const struct ds4_vqb2_file *vqb2, uint32_t expert);

/* Walk a directory recursively; pin every .vqb2 found. Diagnostic only —
 * Hazard: multiple K values per (layer, kind) cause silent overwrites. Use
 * ds4_vqb2_candidate_manifest_load() for runtime paths. */
int ds4_vqb2_corpus_load(ds4_hot_expert_store *store, const char *corpus_dir);

/* Candidate-keyed loader. Manifest format: candidate,layer,kind,k,path.
 * Validates header vs row; rejects K-aliases for same (layer, kind). */
int ds4_vqb2_candidate_manifest_load(ds4_hot_expert_store *store,
                                     const char *manifest_csv,
                                     const char *candidate_name);

void ds4_hot_expert_store_print(const ds4_hot_expert_store *store);

/* Pin every routed expert of one layer to the FP16 hot-store by dequantizing
 * from the IQ2_XXS mmap. silv 2026-05-27 hot-store full-coverage encoder
 * priority. Each layer ≈ 6.4 GB heap (256 experts × 3 tiles × ~8 MB FP16).
 * Defined in ds4.c (uses the iq2_xxs grid + tensor_expert_bytes helpers).
 * Model + weights are passed as void* since the typedefs are anonymous in
 * ds4.c and not exposed here; ds4.c casts back at the function entry.
 * Returns 0 on success, -1 on budget exceeded or non-IQ2_XXS layer. */
int ds4_hot_pin_layer_iq2xxs_full(
    ds4_hot_expert_store *store,
    const void *model,
    const void *weights,
    uint32_t layer);

/* silv 2026-05-27 — pair-AVG pin: stores (L_dst + L_src) / 2 averaged tiles
 * at layer_dst's hot-store slot. Both layers must share compress_ratio
 * (same parity). Saves 50% storage vs pinning both, with predicted-preserve
 * capability per the pair-AVG matrix-norm finding (rel_err 0.71 < DUP 1.41).
 * Returns 0 success, -1 budget exceeded, -2 parity mismatch. */
int ds4_hot_pin_layer_pair_avg(
    ds4_hot_expert_store *store,
    const void *model,
    const void *weights,
    uint32_t layer_dst,
    uint32_t layer_src);

/* ============================================================================
 * Global active hot-store. Set once at engine init; read O(1) during dispatch.
 * NULL = no hot store; dispatch falls back to IQ2_XXS path.
 * ============================================================================ */
void                   ds4_hot_store_set_active(ds4_hot_expert_store *store);
ds4_hot_expert_store  *ds4_hot_store_get_active(void);

/* O(n_selected) check: all gate+up+down tiles pinned AND row masks full.
 * Returns 1 iff hot path is safe for this (layer, expert-set). */
int ds4_hot_layer_all_pinned(const ds4_hot_expert_store *store, uint32_t layer,
                             const int32_t *selected, uint32_t n_selected);

/* CPU FP16 routed-FFN dispatch. Fallback when Metal unavailable; also
 * correctness reference. Output is ADDED to output_fp32. ~118 t/s single
 * thread per vqb2_to_fp16_test.c. */
int ds4_hot_dispatch_layer_cpu(const ds4_hot_expert_store *store, uint32_t layer,
                               const int32_t *selected, const float *expert_weights,
                               uint32_t n_selected,
                               const float *input_fp32, float *output_fp32,
                               uint32_t n_embd);

/* ============================================================================
 * Diagnostic counters. Increment during routing dispatch via the inline
 * helper; print/reset at session boundaries.
 * ============================================================================ */
extern uint64_t g_ds4_hot_hits;        /* dispatch matched HOT-classified expert */
extern uint64_t g_ds4_hot_misses;      /* dispatch matched non-HOT expert */
extern uint64_t g_ds4_masked_skips;    /* dispatch attempted MASKED (bug if > 0) */
extern uint64_t g_ds4_total_dispatches;

static inline void ds4_hot_count_dispatch(uint32_t layer, uint32_t expert) {
    extern uint8_t g_expert_tier[DS4_N_LAYER * DS4_N_EXPERT];
    extern int     g_expert_table_initialized;
    extern uint64_t g_ds4_hot_hits, g_ds4_hot_misses,
                    g_ds4_masked_skips, g_ds4_total_dispatches;
    if (layer >= DS4_N_LAYER || expert >= DS4_N_EXPERT) return;
    g_ds4_total_dispatches++;
    if (!g_expert_table_initialized) return;
    uint8_t t = g_expert_tier[layer * DS4_N_EXPERT + expert];
    if      (t == DS4_EXPERT_TIER_HOT)    g_ds4_hot_hits++;
    else if (t == DS4_EXPERT_TIER_MASKED) g_ds4_masked_skips++;
    else                                  g_ds4_hot_misses++;
}

void ds4_hot_print_stats(void);
void ds4_hot_reset_stats(void);

#endif /* DS4_EXPERT_TABLE_H */
