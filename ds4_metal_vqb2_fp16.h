/* ds4_metal_vqb2_fp16.h — Metal FP16 routed-FFN dispatch (silv 2026-05-26).
 *
 * Per the architecture validated this session (VQB2_PATH_FOUND.md):
 *   VQB2 on disk (~5-15 GB) → load-time dequant → FP16 hot store (~29 GB)
 *   → Metal FP16 matvec dispatch on routed experts → 20-30 t/s target.
 *
 * Per codex H1884:
 *   Hot store is populated via ds4_vqb2_candidate_manifest_load (already
 *   shipped). Dispatch looks up tiles via ds4_hot_get_{gate,up,down}_fp16.
 *
 * Per codex H1666:
 *   MTL4 ML is a dense-organ route, not a universal backend. The routed
 *   FFN is per-token-sparse (6 active of 256 experts). Use standard MTLCompute
 *   with FP16 matvec kernels rather than MTL4 ML.
 *
 * THIS FILE: declarations + Metal kernel source + dispatch shape. The
 * `ds4_metal_vqb2_fp16_dispatch` function is the integration point that
 * `metal_graph_encode_decode_layer` (ds4.c:10823) calls when the hot store
 * has a tile for (layer, expert) — falling back to the IQ2_XXS path
 * otherwise. Integration wiring is the NEXT engineering step.
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

/* MTL4Compute path init (silv 2026-05-26 correction + codex H1723/H1779).
 * Uses modern MTL4Compiler + MTL4CommandQueue + MTL4CommandAllocator +
 * MTL4ArgumentTable. Required for the modern dispatch path.
 * Calls ds4_metal_vqb2_fp16_init internally to reuse device + MSL source.
 * Idempotent. Returns 0 on success, non-zero on failure.
 */
int ds4_metal_vqb2_fp16_init_mtl4(void);

/* Bind a hot store as Metal-resident MTLBuffer (zero-copy on M1 unified
 * memory via newBufferWithBytesNoCopy). The Metal kernels read FP16 tiles
 * directly from this buffer using the store's offset tables.
 *
 * Must be called AFTER store is populated (ds4_vqb2_candidate_manifest_load).
 * Idempotent: re-calling rebinds. Returns 0 on success, -1 on Metal alloc
 * failure or NULL store.
 *
 * Per codex H1717: Apple Silicon unified memory means storageModeShared
 * MTLBuffer wrapping a malloc'd heap is zero-copy — no GPU-side mirror,
 * no transfer cost. The deallocator callback is left empty so the store
 * remains the owner of the heap (Metal just borrows).
 */
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

#ifdef __cplusplus
}
#endif

#endif /* DS4_METAL_VQB2_FP16_H */
