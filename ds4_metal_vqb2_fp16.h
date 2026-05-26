/* ds4_metal_vqb2_fp16.h — Metal FP16 routed-FFN dispatch.
 *
 * Path:
 *   VQB2 packets on disk (~5-15 GB) → load-time dequant → FP16 hot store
 *   (~29 GB) → Metal FP16 matvec dispatch on routed experts.
 *
 * Hot store populated via ds4_vqb2_candidate_manifest_load. Dispatch looks
 * up FP16 tiles via ds4_hot_get_{gate,up,down}_fp16. Falls back to the
 * IQ2_XXS path when the hot store doesn't cover a (layer, expert) tile.
 *
 * Routed FFN is per-token-sparse (6 active of 256 experts). We use standard
 * MTLCompute with FP16 matvec kernels by default; the MTL4Compute path
 * (argument tables + residency sets) is selectable via env DS4_VQB2_FP16_PATH.
 */
#ifndef DS4_METAL_VQB2_FP16_H
#define DS4_METAL_VQB2_FP16_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Forward declarations to avoid Metal/ObjC header cycle. */
struct ds4_hot_expert_store;

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize Metal pipelines for the FP16 routed-FFN kernels.
 * Idempotent. Returns 0 on success, non-zero on failure. */
int ds4_metal_vqb2_fp16_init(void);

/* MTL4Compute path init: MTL4Compiler + MTL4CommandQueue +
 * MTL4CommandAllocator + MTL4ArgumentTable. Calls ds4_metal_vqb2_fp16_init
 * internally to reuse the device + MSL source. Idempotent. */
int ds4_metal_vqb2_fp16_init_mtl4(void);

/* Bind a hot store as Metal-resident MTLBuffer via newBufferWithBytesNoCopy.
 * On Apple Silicon unified memory with storageModeShared this is zero-copy:
 * no GPU-side mirror, no transfer cost. Store remains heap owner; Metal
 * just borrows (deallocator callback left empty).
 *
 * Must be called AFTER store is populated. Idempotent. Returns 0 / -1. */
int ds4_metal_vqb2_fp16_bind_store(struct ds4_hot_expert_store *store);

/* Per-layer dispatch: route through (hot_store, layer, expert_ids[K], weights[K])
 * to compute the routed-FFN output.
 *
 * NOT YET IMPLEMENTED — this is the integration point. The function returns
 * a sentinel value indicating "fall back to IQ2_XXS" until the Metal kernel
 * + dispatch is wired.
 *
 * Parameters (when implemented):
 *   store          — populated FP16 hot store (from ds4_vqb2_candidate_manifest_load)
 *   layer          — DS4 layer index 0..42
 *   n_tokens       — number of tokens in the batch (1 for gen)
 *   selected_exps  — [n_tokens × N_EXPERT_USED] expert IDs (host pointer)
 *   expert_weights — [n_tokens × N_EXPERT_USED] weights
 *   input          — [n_tokens × DS4_N_EMBD] activation rows
 *   output         — [n_tokens × DS4_N_EMBD] (added to)
 *
 * Returns:
 *   0  = dispatch succeeded
 *  -1  = at least one selected expert not in hot store (caller must fall back)
 *  -2  = Metal error
 *
 * Status (2026-05-26): returns -1 unconditionally (skeleton stub). Caller
 * MUST handle the fallback path.
 */
int ds4_metal_vqb2_fp16_dispatch(struct ds4_hot_expert_store *store,
                                 uint32_t layer,
                                 uint32_t n_tokens,
                                 const int32_t *selected_exps,
                                 const float *expert_weights,
                                 const void *input_fp32,
                                 void *output_fp32);

/* Optional: enable per-stage profiling (writes to stderr in same format as
 * DS4_METAL_PROFILE_DECODE_STAGE macro). MTL4 instrumentation hook —
 * captures dispatch time of gate/up/swiglu/down per expert. */
void ds4_metal_vqb2_fp16_set_profile(bool enabled);

/* Backend variants exposed for benchmarking / A/B. Top-level dispatcher
 * ds4_metal_vqb2_fp16_dispatch selects one via env DS4_VQB2_FP16_PATH=
 * legacy|icb|mtl4. */
int ds4_metal_vqb2_fp16_dispatch_legacy(struct ds4_hot_expert_store *store,
                                        uint32_t layer, uint32_t n_tokens,
                                        const int32_t *selected_exps,
                                        const float *expert_weights,
                                        const void *input_fp32, void *output_fp32);
int ds4_metal_vqb2_fp16_dispatch_icb(struct ds4_hot_expert_store *store,
                                     uint32_t layer, uint32_t n_tokens,
                                     const int32_t *selected_exps,
                                     const float *expert_weights,
                                     const void *input_fp32, void *output_fp32);
int ds4_metal_vqb2_fp16_dispatch_mtl4(struct ds4_hot_expert_store *store,
                                      uint32_t layer, uint32_t n_tokens,
                                      const int32_t *selected_exps,
                                      const float *expert_weights,
                                      const void *input_fp32, void *output_fp32);

/* ICB cache telemetry. hits/misses/evicts are monotonic counters; subtract
 * snapshots to get per-window deltas. */
void ds4_metal_vqb2_fp16_icb_stats(uint64_t *hits, uint64_t *misses, uint64_t *evicts);

#ifdef __cplusplus
}
#endif

#endif /* DS4_METAL_VQB2_FP16_H */
