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

#ifndef DS4_N_LAYER
#define DS4_N_LAYER 43
#endif
#ifndef DS4_N_EXPERT
#define DS4_N_EXPERT 256
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

#endif /* DS4_EXPERT_TABLE_H */
