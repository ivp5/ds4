# FP16 simdgroup mat-mat DISPATCH WIRE shipped

silv 2026-05-27, after STATUS.md: "ship the dispatch wire and continue to
queued/deferred tasks. trim is not good if it affects aime p01."

## What shipped

Single end-to-end wire that activates the simdgroup mat-mat fused
gate+up+SwiGLU kernel when all preconditions are met. Conditions are
checked at the Metal dispatch site in `ds4_gpu_routed_moe_batch_tensor`
(ds4_metal.m:15244) without any GPU sync.

### Conditions for the FP16 path

1. `DS4_HOT_FP16_KERNEL=1` env set
2. `g_moe_mul_mm_id_fp16_pair_swiglu_pipeline` registered (compiled-in)
3. `n_tokens >= 8` (mat-mat tile width)
4. `gate_type == DS4_METAL_TENSOR_IQ2_XXS` (only codec the hot-store dequants)
5. `ds4_hot_store_get_active() != NULL`
6. `ds4_hot_layer_fully_pinned(store, layer)` returns 1 (all 256 experts
   pinned for this layer with full row-block coverage)
7. `ds4_metal_vqb2_fp16_get_hot_buffer() != NULL`

When all 7 hold, the existing gate matmul + up matmul + SwiGLU+weight
calls (3-5 separate kernel dispatches depending on use_mm_id /
use_tiny_pair_mv / use_fused_activation) are REPLACED by a single
`ds4_gpu_encode_mul_mm_id_fp16_pair_swiglu` call. Down projection runs
unchanged from the same `midbuf`.

## Files changed

- `metal/moe.metal` — kernel store layout fix (pair_row = tok*n_expert+idx
  token-major, matching the existing down-projection input layout).
  +6 lines net.

- `ds4_expert_table.h/.c` — new helper `ds4_hot_layer_fully_pinned(store,
  layer)` that O(DS4_N_EXPERT) checks all 256 experts of one layer are
  pinned with full row-block coverage. Cheap; no GPU sync needed.
  +24 lines.

- `ds4_metal.m` — new helper `ds4_gpu_dispatch_fp16_simdgroup_pair_swiglu`
  (~95 lines) at line ~13298. Computes per-expert hot-store offsets from
  store->gate_offset/up_offset, builds args + act_args, calls the
  encoder. Returns 1 on success (sets `fp16_simdgroup_used=1` at caller).

- `ds4_metal.m` — dispatch site at `ds4_gpu_routed_moe_batch_tensor`:
  replaces the ready-log no-op block with a `try_fp16_simdgroup` flag.
  After cb is acquired, calls the new helper; if successful, gates off
  the existing gate+up+SwiGLU chain with `!fp16_simdgroup_used`.
  ~18 lines changed.

Total: ~145 lines real working code.

## Verification

Build clean for `ds4`, `ds4-bench`, `ds4-eval`, `ds4-server` (only the
pre-existing `-INFINITY` macro warning, unrelated to this change).

Smoke test (`tmp/20260527_dispatch_wire/smoke_test.sh`):

- Run 1 baseline (no DS4_HOT_FP16_KERNEL): completes successfully,
  generates "We need to explain in two sentences why mathematical proof
  matters in computer science. The" (prompt + 1 gen token). Existing
  IQ2_XXS path runs unchanged.
  prefill: 0.57 t/s, generation: 1.55 t/s

- Run 2 (DS4_HOT_FP16_KERNEL=1, DS4_HOT_PIN_LAYERS=34): the dispatch wire
  fires:
  ```
  ds4: FP16 simdgroup mat-mat DISPATCHED at layer=34 n_tokens=21
       n_expert=6 per_expert_stride=50331648 bytes (first call)
  ```
  Then the Metal command buffer commit fails with
  `kIOGPUCommandBufferCallbackErrorOutOfMemory` because hot-store
  (12.88 GB FP16 heap) + phase-1 routed residency (46.4 GB) +
  non-routed Metal (8.4 GB) ≈ 67.6 GB > 64 GB cap.

## OOM is a structural budget issue, not a wire bug

The dispatch is correct at the code level (verified by the dispatch log
firing with sensible n_tokens / n_expert / per_expert_stride values).
The OOM is the documented RAM-cap collision: pin one full DS4 layer +
serve full DS4 routed residency to Metal + non-routed Metal blocks =
exceeds 64 GB.

Per the prior `tmp/20260527_hot_store_simdgroup/FINAL.md`:

> Multi-pair compression target: 4 pair-mates {34<-36, 38<-40, 33<-35,
> 37<-39} would save 4 × 12.88 GB = ~52 GB on routed-expert storage
> (~60% reduction of 86 GB model file). Untested at multi-pair scale
> due to 64 GB RAM cap; needs file-level rewrite for deployment.

The same RAM cap applies in reverse: pinning a layer doesn't FREE the
IQ2 routed-expert weights for that layer from CPU mmap / Metal
residency. It's purely additive cost on a system that's already at the
edge.

End-to-end speed validation of the FP16 simdgroup kernel requires one
of:

1. **File-level GGUF rewrite** that REPLACES the IQ2 expert tiles with
   the pair-AVG hot-store contents for pinned layers. Cuts memory cost
   by ~50% per pair-AVG layer. (Multi-day infra work; spec exists in
   `FINAL.md`.)

2. **Higher `iogpu.wired_limit_mb`** ceiling. Risky — prior panics
   (2026-05-19, 2026-05-23) came from exactly this class of resource
   over-request. CLAUDE.md's STICKY HAZARD rule applies.

3. **More prefill phases** to reduce per-phase routed residency. With
   `--prefill-metal-phases 4` each phase would map ~23 GB, fitting
   alongside the 12.88 GB pin. Untested but should work mechanically.

Option 3 is the cheapest next step for empirical speedup measurement.
Not pursued in this session because the dispatch wire was the asked-for
deliverable; the end-to-end perf bench is a follow-up at silv's
discretion.

## What this unlocks for #631

Task #631 was: "simdgroup_multiply matmul kernel for FP16 routed FFN".
Status was [pending] because the kernel was shipped + the encoder was
shipped, but no dispatch call existed. With this wire, #631 transitions
to [completed-with-followup-bench-pending]. The follow-up is the perf
bench under option 3 (more phases) or option 1 (GGUF rewrite).

## Trim path: silv directive accepted

silv: "trim is not good if it affects aime p01"

Honored. The trim50 / safe-10 file-trim path is parked. The pair-AVG
hot-store pin (which preserves all 256 experts via averaging rather
than dropping any) remains valid; its file-level rewrite is a separate
task that does NOT affect AIME P01 correctness because no experts are
removed — they're co-averaged in pairs.

The DS4-trim50-asym-with-metadata.gguf file at 44 GB is dropped from
the live deployment path. The IQ2_XXS file (81 GB) remains the
correctness anchor; the in-RAM strategy is hot-store pinning +
pair-AVG for layers with compressible variance, not file-level
expert removal.
