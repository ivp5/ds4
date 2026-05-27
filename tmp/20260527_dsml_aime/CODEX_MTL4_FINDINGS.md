# Codex MTL4 research — corrected my prior assumption

silv 2026-05-27: "reread codex research to make sure you exhaust all
potential alternative protocols/permutations".

## Prior assumption REFUTED

In SPEEDUP_LADDER_2026_05_27.md and ROADMAP_DEEP_OPTIMIZATIONS.md
I wrote: "MTL4 ICB may not exist on M1 pre-M5; pivot to classic
MTLIndirectCommandBuffer on M1."

Codex CODEX_SHIFTS.md H1777 (2026-05-25) refutes that:
> On local M1 Max, `MTLGPUFamilyMetal4=true`. MTL4 command queue,
> allocator, command buffer, compute encoder, compiler, ordinary
> library-from-source, ordinary compute pipeline, argument table,
> placement sparse, and pipeline data-set serializer all work.
> `MTL4MachineLearningCommandEncoder` can also be created.

What DOESN'T work on M1:
- `MTL4MachineLearningPipelineState` from ordinary Metal kernel
  (NSInvalidArgumentException: `-[_MTLLibrary executableWithDeviceSelection:]`)
- `MTLTensor` segfaults on `tensorSizeAndAlignWithDescriptor:` fp16 compute-tensor

What DOES work (confirmed by my 20 ports this session):
- `MTL4ComputeCommandEncoder` ✓
- `MTL4ArgumentTable` with setAddress via gpuAddress ✓
- `MTL4Compiler` + `MTL4Library` + `MTL4ComputePipelineState` ✓
- `MTLResidencySet` ✓
- `MTL4CommitOptions` + feedback handlers ✓

## What MTL4 doesn't have (vs classic)

Classic MTL has `MTLIndirectCommandBuffer` with `indirectComputeCommandAtIndex:`
+ `setKernelBuffer:` + `concurrentDispatchThreadgroups:` + caller
`executeCommandsInBuffer:withRange:`.

Searching codex + Apple docs: **MTL4 does NOT introduce a new MTL4-specific
indirect command buffer class.** MTL4 retains classic `MTLIndirectCommandBuffer`
for indirect command recording. The indirect buffer can mix with MTL4
encoders since both use `id<MTLBuffer>` for setKernelBuffer recordings.

So the architecture for session-persistent dispatch is:

```
[Setup at model open]
  Allocate ONE MTLIndirectCommandBuffer with maxCommandCount = N_total_dispatches
  Record EVERY per-token dispatch into the ICB

[Per-token replay]
  Update only buffers that change (per-token: src1 activation, ids, weights)
  Set residency on all model buffers (one-time setup persists)
  executeCommandsInBuffer with full range

[Effect]
  Per-token CPU encoding: ~430 setKernelBuffer + dispatchThreadgroups
                          → 1 executeCommandsInBuffer call
```

## Why H1779/H1780 didn't show wall-time speedup

Codex H1779: "summed layer kernels = 3.149625001 ms (6 layer windows)"
Codex H1780: "persistent MTL4 native executor: 3.249458299 ms"

The persistent version was SLIGHTLY SLOWER. Reasons:
1. Only 6 dispatches; per-token GPU overhead is small relative to
   compute on those specific kernels
2. The N=6 case doesn't have enough dispatch density to amortize
3. GPU encoder cost is dominated by Metal API overhead, NOT recording
   format

For DS4 per-token decode at ~430 dispatches/token, the picture should
be different — that's 71× more dispatches per token than the H1780
test. The amortization curve flips.

## Refined session-persistent ICB design

Update to ROADMAP item 2: **DO build with MTL4 compute encoders +
classic MTLIndirectCommandBuffer**. Not pivot to classic everywhere;
the pool + MTL4 path STAYS for non-ICB MTL4 dispatches; the ICB is
the recording medium that wraps either classic-MTL or MTL4 pipelines
underneath.

Wait — does classic MTLIndirectCommandBuffer support MTL4 pipelines?
The `setComputePipelineState:` on `MTLIndirectComputeCommand` takes
`id<MTLComputePipelineState>`. My MTL4-compiled pipelines ARE
`id<MTLComputePipelineState>` (verified: the type isn't MTL4 specific
— see my softplus_sqrt MTL4 pipeline init that declares
`id<MTLComputePipelineState> g_softplus_sqrt_mtl4_pipeline`).

So YES: classic ICB can encode MTL4 pipelines as long as they're
expressed as MTLComputePipelineState. The buffer bindings via
setKernelBuffer work for both kernel storage classes (constant + device
const).

## Updated speedup ladder

| level | mechanism | DS4 status | rough target |
|-------|-----------|------------|--------------|
| 1 | MoE matmul + Phase 7 ICB | wire shipped, A/B blocked by --cpu-moe | ~22% if path fires |
| 2 | Router cycle kernel fusion | not started | ~0.5% direct + register-pressure gains |
| 3 | Session-persistent classic-ICB over all dispatches | architecture clarified this turn | 10-100× CPU encoding reduction |
| 4 | Speculative decode + tree search | #418 entry exists; not measured | 2-4× effective t/s at decode |
| 5 | Cached prefix activations | not started | ∞× for cached prefixes |

## Practical next moves (recommended order)

1. **Continue MTL4 ports** (incremental, no architectural risk).
   Next batch: router_finalize_one, qkv_rms_norm_f32_4, soft_max_4.

2. **Wire Level-1 A/B properly** by either:
   - (a) setting up DS4_HOT_FP16_KERNEL + hot-store + phase-split
     (multi-turn safety setup to avoid M1 kernel panic)
   - (b) using a smaller GGUF that fits in Metal without --cpu-moe
     (DS4-trim50 or DS4-MTP-Q4K-Q8_0-F32 3.5GB)

3. **Build Level-3 prototype** on a SUBSET of per-token dispatches
   first (e.g. the 4 router-cycle MTL4 pipelines I already have).
   This is the codex H1780 pattern but at higher dispatch density.

4. **Spec-decode tree-search** (Level 4) is the next big leverage.
   Has existing entry #418; needs activation + AIME A/B + tree-search
   variant.

5. **Cached prefix** (Level 5) is the biggest single end-to-end win
   but most architectural. Defer until L1-L3 measured.

## Honest summary

My prior framing was wrong about MTL4 ICB being unavailable on M1.
The codex research already documented (H1777/H1779/H1780) that:
- MTL4 compute + arg tables work
- Persistent executor pattern works (just doesn't always win at low N)
- The right architecture is classic ICB recording + MTL4 pipelines
  inside

This unblocks the session-persistent ICB path that I previously called
"deferred to Path B / M5 hardware". The path is available on M1 — I
just need to BUILD it on N=430 dispatches per token (vs H1780's N=6
where amortization didn't show wins).
