/* DS4 expert-table — runtime per-(layer, expert) tier classification.
 *
 * : unified substrate for expert-mask trim +
 * hot-expert pre-dequant. See audits/expert_table_design_20260523.md.
 *
 * Three tiers per (layer, expert_id):
 * NORMAL = 0 default IQ2_XXS lazy-mmap dispatch
 * HOT = 1 pre-dequantized FP16, fast dispatch (not yet implemented)
 * MASKED = 2 never routed; router skips and re-normalizes top-K
 *
 * Loaded from CSV manifests at engine open. See ds4/masks/ directory for
 * masks derived from the 82001-event distinct router trace + pin_top CSV
 * inversions.
 *
 * Integration into ds4.c:
 * 1. Add `#include "ds4_expert_table.h"` at top of ds4.c
 * 2. In ds4_engine_open or similar startup, call ds4_load_expert_table(
 * cfg->expert_mask_path, cfg->hot_expert_pin_path);
 * 3. In layer_topk_selected_experts_from_probs (ds4.c ~line 6192), AFTER
 * bias-add and BEFORE topk_desc, call:
 * uint32_t il = (uint32_t)(layer - &model->layers[0]);
 * ds4_apply_expert_mask_to_selection(il, selection);
 * 4. Wire CLI flags --expert-mask FILE / --hot-expert-pin FILE in
 * ds4_cli.c, ds4_bench.c, ds4_server.c, ds4_eval.c, ds4_logitlens.c.
 *
 * Per CLAUDE.md STICKY HAZARD: any DS4 launch using this code must include
 * --prefill-metal-phases auto (or --cpu-moe). The mask infrastructure does
 * NOT change the wired-memory cap behavior.
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
 DS4_EXPERT_TIER_HOT = 1,
 DS4_EXPERT_TIER_MASKED = 2,
} ds4_expert_tier;

/* Per-(layer, expert) tier classification.
 * Indexed by `layer * DS4_N_EXPERT + expert`. Default = 0 (NORMAL).
 * 11008 bytes total — fits comfortably in L1 cache. */
extern uint8_t g_expert_tier[DS4_N_LAYER * DS4_N_EXPERT];
extern int g_expert_table_initialized;

/* Load mask + hot manifests. Either path may be NULL (skip that tier).
 *
 * Mask CSV format (DROP set — entries listed are MASKED):
 * # any-comment-line
 * 3,0 <- layer 3, expert 0 → MASKED
 * 3,1
 * ...
 *
 * Hot CSV format (KEEP set — entries listed are HOT):
 * # any-comment-line
 * 3,42 <- layer 3, expert 42 → HOT (pre-dequant FP16)
 * ...
 *
 * If an (layer, expert) appears in BOTH manifests, MASK wins (mask is
 * a stronger directive than hot).
 *
 * Returns 0 on success, -1 on error.
 * Idempotent: re-calling resets the table. */
int ds4_load_expert_table(const char *mask_path, const char *hot_path);

/* Test if a given (layer, expert) is masked. Returns 0 if table uninit or
 * out-of-range. */
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

/* Apply mask to selection scores BEFORE topk_desc. Sets masked-expert scores
 * to -INFINITY so they're never selected. No-op if table not initialized.
 *
 * Layers 0-2 (hash layers, broad-spectrum routing per CLAUDE.md /
 * audits/trim_ladder_to_64gb_20260523.md) must never be masked; the CSV
 * loader does not explicitly enforce this but build_expert_mask.py does. */
void ds4_apply_expert_mask_to_selection(uint32_t layer, float *selection);

/* Diagnostics: print per-layer mask + hot counts. Returns total masked. */
int ds4_expert_table_print_stats(void);

/* =========================================================================
 * HOT EXPERT PRE-DEQUANT (silv 2026-05-26: advance further)
 * =========================================================================
 *
 * Pre-dequantize HOT experts at engine init so the CPU-MoE dispatch path
 * can use a FP16 matmul instead of the IQ2_XXS dequant+matmul each token.
 *
 * MEMORY BUDGET (CRITICAL):
 *   Per-expert FP16 storage = 3 kinds × (7168 × 2048) × 2 bytes ≈ 88 MB
 *   Full HOT for all 11008 experts: 950 GB (untenable)
 *   Practical HOT budget on M1 Max 64 GB:
 *     - Model mmap: 86.7 GB (sharing with OS page cache)
 *     - Live heap budget for hot store: ~3-5 GB realistic ceiling
 *     - That's ~30-50 HOT experts maximum (~0.3-0.5% of all experts)
 *
 * STRATEGY:
 *   Use HOT manifest to pin ONLY the top-K most-frequently-routed experts
 *   observed in a router trace. Top-K capped by `--hot-budget-mb N` flag
 *   (default 4096 MB = 4 GB ≈ 46 hot experts).
 *
 * STATUS (2026-05-26):
 *   Scaffolding shipped: store header + diagnostic counters + budget-aware
 *   loader API. Actual dispatch wiring + FP16 store population deferred
 *   until measurement shows HOT-hit rate justifies the engineering.
 *   Counters fire during routing; check via ds4_hot_print_stats() at
 *   gen-end.
 */

typedef struct ds4_hot_expert_store {
    /* Per-(layer, expert) FP16 store offsets. -1 = not pinned. */
    int64_t *gate_offset;      /* DS4_N_LAYER × DS4_N_EXPERT entries */
    int64_t *up_offset;
    int64_t *down_offset;
    /* Per-(layer, expert) row-block coverage bitmasks. Current VQB2 packets
     * encode the first 128-row block only; a complete hot replacement must
     * have all row blocks for each kind, not merely a nonnegative offset. */
    uint64_t *gate_row_blocks;
    uint64_t *up_row_blocks;
    uint64_t *down_row_blocks;
    /* The FP16 heap. NULL if no experts pinned. */
    void *fp16_heap;
    uint64_t heap_bytes;
    uint64_t budget_bytes;     /* user-specified cap */
    uint32_t n_pinned;
} ds4_hot_expert_store;

/* Allocate a hot-expert store with the given budget. Returns NULL on alloc
 * failure. The store starts empty; call ds4_hot_pin_expert() to add. */
ds4_hot_expert_store *ds4_hot_expert_store_alloc(uint64_t budget_bytes);

/* Free the store and all pinned weights. */
void ds4_hot_expert_store_free(ds4_hot_expert_store *store);

/* Pin the (layer, expert) tile by dequantizing IQ2_XXS → FP16 into the
 * store's heap. Returns:
 *   0  = success (now pinned)
 *  -1  = error (out of budget, or dequant function unavailable)
 *
 * Implementation note: requires access to the model's IQ2_XXS weight tensors,
 * which lives in ds4.c. This declaration is the contract; ds4.c provides the
 * actual populator wrapping its dequant helpers.
 */
int ds4_hot_pin_expert(ds4_hot_expert_store *store,
                       uint32_t layer, uint32_t expert,
                       const void *gate_iq2, const void *up_iq2, const void *down_iq2,
                       uint64_t gate_bytes, uint64_t up_bytes, uint64_t down_bytes);

/* Accessors for Metal MTLBuffer wrapping (silv 2026-05-26).
 * Expose the heap base pointer + length so ds4_metal_vqb2_fp16.m can
 * create a newBufferWithBytesNoCopy MTLBuffer over it. */
void    *ds4_hot_store_heap_ptr(const ds4_hot_expert_store *store);
uint64_t ds4_hot_store_heap_bytes_get(const ds4_hot_expert_store *store);

/* Lookup pinned FP16 pointer; returns NULL if expert not HOT or not pinned. */
const void *ds4_hot_get_gate_fp16(const ds4_hot_expert_store *store,
                                  uint32_t layer, uint32_t expert);
const void *ds4_hot_get_up_fp16(const ds4_hot_expert_store *store,
                                uint32_t layer, uint32_t expert);
const void *ds4_hot_get_down_fp16(const ds4_hot_expert_store *store,
                                  uint32_t layer, uint32_t expert);

/* Populate one (layer, kind, expert) tile from a VQB2 packet by dequantizing
 * to FP16 and copying into the store's heap. silv 2026-05-26 directive
 * "not iq2_xxs": this is the primary VQB2 → in-memory FP16 path.
 *
 * Per-expert FP16 size = n_rows × n_pairs × 2 (re, im) × 2 bytes
 *   gate (n_pairs=2048): 128 × 2048 × 2 × 2 = 1 MB FP16
 *   up   (n_pairs=2048): 128 × 2048 × 2 × 2 = 1 MB FP16
 *   down (n_pairs=1024): 128 × 1024 × 2 × 2 = 0.5 MB FP16
 *
 * Forward-declared vqb2 type to avoid include cycle. The caller passes
 * the opened ds4_vqb2_file pointer; the implementation includes ds4_vqb2_reader.h
 * internally and treats it as opaque.
 *
 * `kind` selects gate/up/down (matches ds4_vqb2_kind_t IDs):
 *   0 = gate → updates store->gate_offset[layer*N_EXPERT + expert]
 *   1 = up   → updates store->up_offset[layer*N_EXPERT + expert]
 *   2 = down → updates store->down_offset[layer*N_EXPERT + expert]
 *
 * Returns:
 *   0  = success
 *  -1  = out of budget, bad indices, or file/dequant error
 */
#define DS4_VQB2_ROW_BLOCK_ROWS 128ULL
#define DS4_VQB2_GATE_UP_FULL_ROW_BLOCKS 16ULL
#define DS4_VQB2_DOWN_FULL_ROW_BLOCKS 32ULL
#define DS4_VQB2_GATE_UP_FULL_ROW_MASK ((1ULL << DS4_VQB2_GATE_UP_FULL_ROW_BLOCKS) - 1ULL)
#define DS4_VQB2_DOWN_FULL_ROW_MASK ((1ULL << DS4_VQB2_DOWN_FULL_ROW_BLOCKS) - 1ULL)

struct ds4_vqb2_file;  /* opaque forward decl */
int ds4_hot_pin_expert_from_vqb2(ds4_hot_expert_store *store,
                                 uint32_t layer,
                                 uint32_t kind,
                                 const struct ds4_vqb2_file *vqb2,
                                 uint32_t expert);

/* Walk a directory recursively for `*.vqb2` files, open each, read header
 * for (layer, kind_id, n_experts), and pin every expert into the store.
 *
 * H1883 candidate-identity hazard: when multiple .vqb2 files exist for the
 * same (layer, kind) but different K values (e.g., L25 gate has both K16
 * and K64 packets after H1882), this loader silently overwrites the
 * hot-store offset with whichever file is walked second. Use
 * ds4_vqb2_candidate_manifest_load() for any runtime path.
 *
 * The header is the source of truth — file names don't have to follow any
 * convention. This means codex's varied naming (`L25_gate.vqb2` vs
 * `L22_up_s777_n1000k_i50000.vqb2`) just works.
 *
 * Returns the total number of expert tiles pinned, or -1 on hard error.
 * Per-file failures (bad header, budget overrun) skip the file with a
 * warning but don't abort the load.
 */
int ds4_vqb2_corpus_load(ds4_hot_expert_store *store, const char *corpus_dir);

/* Candidate-keyed VQB2 loader (codex H1884).
 *
 * The recursive directory loader is for diagnostics only. Once the same
 * (layer, kind) exists at multiple K values (e.g., L25 has both K16 and
 * K64 valid packets after H1882), directory traversal silently overwrites
 * the hot-store offset. Manifest rows must identify the candidate and the
 * exact packet key:
 *
 *   candidate,layer,kind,k,path
 *
 * `kind` accepts gate/up/down or 0/1/2. The loader validates the opened
 * VQB2 header against layer/kind/k before pinning. Duplicate (layer, kind)
 * rows for the selected candidate are rejected so K16/K64 aliases cannot
 * silently overwrite each other in the hot store.
 *
 * Returns total tiles pinned, or -1 on hard error.
 */
int ds4_vqb2_candidate_manifest_load(ds4_hot_expert_store *store,
                                     const char *manifest_csv,
                                     const char *candidate_name);

/* Print store contents summary: per-layer pin counts + heap usage. */
void ds4_hot_expert_store_print(const ds4_hot_expert_store *store);

/* =========================================================================
 * Global active hot-store (silv 2026-05-26 / codex H1885/H1886).
 *
 * The dispatch path in ds4.c needs to check whether a VQB2 FP16 hot store
 * is loaded and whether the current (layer, expert) tile is pinned. Rather
 * than thread the store pointer through every call site, expose a single
 * global accessor. Set once at engine init (from CLI flag), read O(1)
 * during dispatch.
 *
 * Active store is `NULL` until ds4_hot_store_set_active() is called. The
 * fallback path (IQ2_XXS + CPU-MoE) remains the default when no store is
 * active or when the layer's experts are not all pinned.
 * ========================================================================= */

void ds4_hot_store_set_active(ds4_hot_expert_store *store);
ds4_hot_expert_store *ds4_hot_store_get_active(void);

/* Returns 1 iff ALL of `n_selected` experts have gate+up+down tiles pinned
 * at `layer` in `store`. Returns 0 if store is NULL, layer out of range,
 * or any expert is missing any kind.
 *
 * This is the O(6) check the dispatch site uses to decide between the
 * VQB2-FP16 path and the IQ2_XXS fallback. With n_selected=6 (DS4 active
 * expert count), the call is ~18 pointer lookups — cache-resident.
 */
int ds4_hot_layer_all_pinned(const ds4_hot_expert_store *store,
                             uint32_t layer,
                             const int32_t *selected,
                             uint32_t n_selected);

/* CPU-side FP16 routed-FFN dispatch.
 *
 * Used as a fallback when Metal dispatch is unavailable AND as a
 * correctness reference for the Metal path. Single-thread scalar; runs at
 * ~118 t/s per `vqb2_to_fp16_test.c` benchmark.
 *
 * Computes routed-FFN output for one token from selected experts using the
 * pinned FP16 tiles. The output is ADDED to `output_fp32` (not overwritten).
 *
 * Returns 0 on success, -1 if any selected expert is missing from store.
 */
int ds4_hot_dispatch_layer_cpu(const ds4_hot_expert_store *store,
                               uint32_t layer,
                               const int32_t *selected,     /* [n_selected] */
                               const float *expert_weights,  /* [n_selected] */
                               uint32_t n_selected,
                               const float *input_fp32,      /* [DS4_N_EMBD=4096] */
                               float *output_fp32,           /* [DS4_N_EMBD=4096] (added to) */
                               uint32_t n_embd);

/* =========================================================================
 * Diagnostic counters (silv "advance further" 2026-05-26)
 *
 * These counters are incremented during routing dispatch to MEASURE whether
 * HOT pinning would help BEFORE the dispatch path is rewritten. Use them
 * to validate "is the top-K most-routed-experts assumption holding?"
 *
 * Increment via the helper macros; print at session end via
 * ds4_hot_print_stats().
 * ========================================================================= */
extern uint64_t g_ds4_hot_hits;       /* dispatch hit a HOT-classified expert */
extern uint64_t g_ds4_hot_misses;     /* dispatch hit a non-HOT expert */
extern uint64_t g_ds4_masked_skips;   /* dispatch attempted to use MASKED (bug) */
extern uint64_t g_ds4_total_dispatches;

static inline void ds4_hot_count_dispatch(uint32_t layer, uint32_t expert) {
    extern uint8_t g_expert_tier[DS4_N_LAYER * DS4_N_EXPERT];
    extern int g_expert_table_initialized;
    extern uint64_t g_ds4_hot_hits;
    extern uint64_t g_ds4_hot_misses;
    extern uint64_t g_ds4_masked_skips;
    extern uint64_t g_ds4_total_dispatches;
    if (layer >= DS4_N_LAYER || expert >= DS4_N_EXPERT) return;
    /* Always increment totals so the counter is informative even without a
     * manifest loaded — useful for baselining how many dispatches happen. */
    g_ds4_total_dispatches++;
    if (!g_expert_table_initialized) return;
    uint8_t t = g_expert_tier[layer * DS4_N_EXPERT + expert];
    if (t == DS4_EXPERT_TIER_HOT) g_ds4_hot_hits++;
    else if (t == DS4_EXPERT_TIER_MASKED) g_ds4_masked_skips++;
    else g_ds4_hot_misses++;
}

/* Print HOT-hit rate diagnostic to stderr. Safe to call at any time. */
void ds4_hot_print_stats(void);

/* Reset HOT diagnostic counters to zero (call at gen-start to scope counts
 * to a single benchmark run). */
void ds4_hot_reset_stats(void);

#endif /* DS4_EXPERT_TABLE_H */
