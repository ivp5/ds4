/* ds4_pillars.h — single-header for the three speedup-ladder pillars.
 *
 * Pillar A — ICB record→replay
 * Pillar B — Hot-expert F16 cache with margin gate
 * Pillar C — Spec-decode propose+verify
 *
 * FROZEN-organ candidates for ML-accelerator packaging:
 *   - Shared-expert MLP: ffn_gate_shexp / ffn_up_shexp / ffn_down_shexp,
 *     Q8_0, runs every token, fixed per-layer (ds4.c L2281-2283).
 *   - Output head: output_hc_base + output_hc_fn + output_hc_scale (L2288-2290).
 *   - Attention output projection: attn_output_a + attn_output_b
 *     Q8_0 matmuls (L2259-2260, L2733-2734).
 *   - HC split/expand kernels: kernel_dsv4_hc_split_sinkhorn,
 *     kernel_dsv4_hc_expand, kernel_dsv4_hc_split_weighted_sum
 *     (metal/dsv4_hc.metal). Fixed-structure dispatch every layer.
 *
 * Dispatch paths by hardware (corrected 2026-05-25 per codex H1722/H1723):
 *   - M5/A19+ : Full MTL4MachineLearningCommandEncoder support.
 *   - M1-M4   : MTL4 compute + ML encoder IS usable when the correct
 *               lifecycle is followed: newCommandAllocator →
 *               newCommandBuffer (from device, not from queue) →
 *               beginCommandBufferWithAllocator → encoder →
 *               endCommandBuffer → MTL4 queue commit. Vector-add canary
 *               on M1 Max: maxerr=0, 0.0567 ms for 1024 floats. Apple8/9-
 *               only features (specific Tensor ops) remain unavailable,
 *               but the general MTL4 compute + ML packaging surface works.
 *               Earlier "MTL4 disabled on Apple7/Apple8" conclusion
 *               measured misuse of old command-buffer assumptions, not
 *               absence of API.
 *   - Fallback: classic Metal compute encoders + ICB record→replay (the
 *               currently-shipping pillar A path) when MTL4 ML packaging
 *               doesn't fit a given organ's shape constraints.
 *
 * All organs are non-dynamic (no per-token surgery), ideal accelerator
 * targets once the boundary-deletion path (ICB) lands.
 *
 * Spaghetti consolidation: was 6 files (3 .c + 3 .h),
 * now 2 files. Single grep finds all pillar state at top of the .c file.
 * All caches global static. No malloc, no allocation behind closures.
 *
 * Each pillar is env-gated:
 * DS4_ICB_ACTIVE — pillar A live
 * DS4_ICB_TEST=1 — run record→replay self-test at init
 * DS4_ICB_MAX_COMMANDS=N — override default 4096 command capacity
 * DS4_HOT_EXPERT_ACTIVE — pillar B live
 * DS4_HOT_EXPERT_MANIFEST=path — CSV (layer,expert_id) one per line
 * DS4_HOT_EXPERT_MARGIN=F — override default 0.04 threshold
 * DS4_SPEC_ACTIVE — pillar C live
 */
#ifndef DS4_PILLARS_H
#define DS4_PILLARS_H

#include <stdbool.h>
#include <stdint.h>

/* Pillar A — ICB ----------------------------------------------------------*/
int ds4_icb_init(void);
int ds4_icb_record_decode(uint32_t n_tokens, uint32_t shape_hash);
int ds4_icb_record_prefill(uint32_t n_tokens, uint32_t shape_hash);
int ds4_icb_execute(uint32_t n_tokens, uint32_t shape_hash);
void ds4_icb_print_stats(void);
void ds4_icb_cleanup(void);
uint32_t ds4_icb_max_commands(void);

/* Pillar B — Hot-expert F16 cache -----------------------------------------*/
int ds4_hot_expert_init(const char *manifest_path);
bool ds4_hot_expert_is_cached(uint32_t layer, uint32_t file_pos);
bool ds4_hot_expert_is_cached_margin_gated(uint32_t layer, uint32_t file_pos, float margin);
float ds4_hot_expert_margin_threshold(void);
uint64_t ds4_hot_expert_f16_offset(uint32_t layer, uint32_t file_pos);
void ds4_hot_expert_print_stats(void);
void ds4_hot_expert_cleanup(void);

/* Pillar C — Spec-decode --------------------------------------------------*/
int ds4_spec_decode_init(int max_draft, float min_margin);
int ds4_spec_decode_propose(uint32_t *out_drafts, int n_draft);
int ds4_spec_decode_verify(const uint32_t *drafts, int n_draft,
 uint32_t *out_accept_prefix_len,
 uint32_t *out_main_token);
void ds4_spec_decode_print_stats(void);
void ds4_spec_decode_cleanup(void);

#endif
