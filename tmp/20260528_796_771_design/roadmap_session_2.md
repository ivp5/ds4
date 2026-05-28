# Post-Increment-5 roadmap — what shipped, what's blocked, what's next

silv 2026-05-28 directive: "continue working on zero copy override fill,
pack-direct loader, source-exact skips and multi-tok bf16 kernel, and
check if some of the mmap regions might no longer be needed, then onwards
to optimal MTL4+ICB stacked with path fused vqb2 records, hot predecode
and mtp speculative decode."

silv technical clues:
1. **Codebook-score gather is 2.9–4.0× faster than materializing decoded
   weights** (VQB2 optimization not yet applied).
2. **PATH_FUSED does not currently replay through ICB** (the 1.35-2.33×
   ICB speedup is missing for the fused chain).

## Shipped this turn

### Zero-copy override-fill (DEFERRED — pack format incompatible)

Added per-tensor page-alignment probe in override-fill. Result:
**zerocopy=0 / 192 BF16-for-F16 tensors are page-aligned.** Pack format
stores tensors at arbitrary byte offsets within the data arena, not on
page boundaries.

The proper fix needs:
1. Wrap the ENTIRE pack mmap as ONE MTLBuffer at engine_open (the pack
   base IS page-aligned)
2. Extend `ds4_tensor_storage` with `metal_offset` field
3. Per-tensor: storage.metal_buffer = pack_buf, storage.metal_offset =
   data_arena_off + entry->data_off
4. Update each `matmul_*_storage` to take a separate offset parameter
5. Update each dispatcher to pass `storage.metal_offset`

This is the storage-foundation for #771 pack-direct loader. Deferred to a
separate increment (~150 LOC, multi-file refactor).

### PATH_FUSED ICB-eligibility probe (shipped, awaiting workload)

Added 8-way LRU per layer keyed by `expert_signature(layer,
selected_exps)`. Counts what fraction of PATH_FUSED calls would HIT a
sig cache. Sized for the build/skip decision for PATH_FUSED+ICB
integration.

**Current smoke doesn't activate PATH_FUSED** (cpu-moe routes all 43
layers on CPU). Probe is silent until a GPU-MoE workload runs. To
activate: need a smoke without `--cpu-moe`, which requires more memory
(M1 Max 64 GB may not fit full DS4 GPU-MoE).

Counters exposed via `ds4_metal_vqb2_fp16_pf_icb_eligibility(hits,
misses)` and printed at engine_close.

## What's blocked / sized

### Zero-copy via pack-buffer + offset (1.77 GB heap + 170 ms saved)

Foundation work for #771. Blocked on storage-struct extension. ~150 LOC.

### Multi-token BF16 kernel (would symmetrize prefill/gen precision)

Current state: BF16 is matvec-only; prefill (n_tok>1) falls back to F16
mmap. Need a multi-tok BF16 kernel (mul_mv_ext-style for n_tok≤8, NAX
direct-RHS for n_tok≥32). Each is a new MSL kernel + pipeline + canary.
~3 kernels × ~150 LOC = ~450 LOC. Single-session feasible.

But: high-resolution audit found Increment 5 model impact is below noise
floor for this corpus (F16 down-sample is near-lossless). So multi-tok
BF16 buys symmetrized precision but probably negligible quality
improvement on THIS corpus.

### Source-exact lifts (histogram measured 2026-05-28)

Override-fill now emits a per-dtype histogram of the 768 remaining
skips. Measured on minimal-GGUF + ds4v4_nonrouted.pack:

| Pack source dtype | Count |   | GGUF target dtype | Count |
|---|---|---|---|---|
| F32                | 152  |   | F32 (type 0)       | 240   |
| BF16 (not wired)   | 241  |   | F16 (type 1)       | 173   |
| **FP8_E4M3**       | **375** ← LARGEST | **Q8_0 (type 8)** | **355** ← LARGEST |
| total              | 768  |   | total              | 768   |

The **highest-leverage next wire is FP8_E4M3 → Q8_0 (≤355 tensors)** —
almost certainly the routed-MoE FFN expert weights stored as FP8
source-exact in the pack, declared as Q8_0 in the GGUF.

Build path:
1. `ds4_gpu_matmul_fp8_e4m3_storage` kernel + canary (decode FP8 → F32,
   matmul against activation, output F32)
2. Q8_0 dispatcher gets a new branch when `storage.dtype ==
   DS4_TENSOR_FP8_E4M3`
3. Override-fill skip lifted for `(source_exact=FP8_E4M3, t->type=Q8_0)`
4. Live-fire confirms 355 dispatches/gen-step shift to FP8 storage path

Scope: ~400 LOC (kernel + canary + dispatcher + skip lift). Single-
session feasible.

The other buckets (BF16-still-skipped=241, F32=152) are smaller
follow-ups.

### mmap regions no longer needed (1.77 GB potentially madvise'd)

The 192 BF16-for-F16 tensors are heap-stored; their F16 mmap region is
dead weight (kernel never reads it when storage.metal_buffer is set).
Could `madvise(MADV_DONTNEED)` those pages to release physical RAM.

Risk: mmap fallback path in F16 dispatcher reads from mmap when storage
isn't set. If we madvise away portions, the fallback fails for those
tensors. Safe ONLY if we never fall back. Currently the fallback DOES
fire for n_tok>1 (multi-tok BF16 not yet built), so madvise is unsafe.

After multi-tok BF16 kernel lands, madvise becomes safe → 1.77 GB freed.

### PATH_FUSED+ICB integration (FIX silv flagged)

The fused chain (GATE+UP+SwiGLU+DOWN+sum) dispatches 5 GPU commands per
token, each through a fresh compute encoder. ICB capture+replay would
record the sequence once per (layer, sel) signature, then replay with
only route_weights update. 1.35-2.33× speedup per existing ICB Phase 8
finding.

Scope: ~200 LOC ObjC + plumbing. Requires:
1. Extend `icb_slot_t` to capture FUSED kernel sequence (not just
   matvec)
2. New `record_fused_icb()` that captures the 5 dispatches
3. PATH_FUSED entry: check sig cache → replay or fall through to
   non-cached path

### Codebook-score gather kernel (2.9-4× speedup silv flagged)

Current VQB2 path: decode-matmul fused kernel materializes decoded
weights (or partial decoded blocks) before the matmul. silv's finding:
direct codebook-score gather (skip the materialization, dot product
against codes via gather) is 2.9-4× faster.

This is a kernel-level rewrite of the VQB2 dispatch path. Substantial
MSL work. Highest-leverage single optimization in the queue.

### Hot predecode

Pre-compute frequent experts' decoded weights ahead of time, cache them
in unified memory. For DS4 routed-MoE, top-K experts per layer tend to
repeat across tokens — a hot-cache could keep their decoded form ready.

Existing `ds4_hot_expert_store` already provides infrastructure. The
question is which experts to predecode and when. Needs a frequency
trace + heuristic.

### MTP speculative decode

Parallel-evaluate draft tokens via MTP (multi-token prediction) head,
accept/reject in main path. Existing tasks #418 (spec-decode entry
point) + #674 (in-progress). Substantial integration work.

## Recommended next order of work

Engineer-roster lens (effort × leverage):

1. **Enumerate the 768 source-exact skips** by source_exact_type — ~10
   LOC instrumentation. Tells us which kernel work has highest payoff.
2. **Codebook-score gather kernel** — silv's 2.9-4× perf clue, biggest
   single speedup in the queue. ~300 LOC MSL + canary.
3. **PATH_FUSED+ICB integration** — silv's flagged correctness gap, 1.35-
   2.33× speedup once active. ~250 LOC ObjC.
4. **Multi-tok BF16 kernel** — enables madvise teardown + symmetric
   precision. ~450 LOC MSL.
5. **Zero-copy override-fill via pack-buffer + offset** — 1.77 GB RAM
   saved at startup. ~150 LOC refactor.
6. **#771 pack-direct loader** — retires the GGUF intermediary, ~10x
   total RAM reduction. Multi-session.
7. **Hot predecode + MTP spec decode** — orthogonal post-foundation
   work.

Items 1+2 are this session's possible next ships; 3-7 are multi-session.