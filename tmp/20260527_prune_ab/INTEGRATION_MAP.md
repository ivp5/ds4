# DS4 ICB + MTL4 integration map — what's done, what's next

silv 2026-05-27 directive: "advance ICB and MTL4 everywhere and integrate
everything". This is the scan + the next-leverage targets, not a port of
all kernels in one turn (which would be ~10× the buffer).

## Current ICB footprint (6 phases shipped)

| phase | task | kernel(s) ICB-captured |
|-------|------|-----------------------|
| 1 | #553 | route_weights_with_remap (setBytes → setBuffer prep) |
| 2 | #555 | route_weights_with_remap (actual MTLIndirectCommandBuffer recording) |
| 3 | #557 | topk_mask + topk_mask_scatter |
| 4 | #558 | softplus_sqrt_f32_4 |
| 5 | #559 | router_finalize_one |
| 6 | #560 | router_weights_one |

What this means: per-token decode dispatches the 6 router/finalize Metal
kernels via ICB record→replay. Recording happens once at model open;
every subsequent token's router cycle replays the same ICB. Saves
encoder rebuild per token.

## Current MTL4 footprint

| pipeline | task | what it does |
|----------|------|--------------|
| `g_polar_pipeline` | #563 | polar p8_m2 dot (canary, 13.54× speedup measured) |
| `g_polar_tile_pipeline` | H1729 | tile×row×batch polar dot |
| `g_polar_fused_pipeline` | #565+ | fused polar with multi-stage organ |
| `g_polar_gud_pipeline` | (codec arc) | polar gate-up-down fused |
| `g_vq_gud_pipeline` | #574 | VQ K=256 gate-up-down |
| `g_hadamard_mtl4_pipeline` | **#653** | **Hadamard-16 widened (this session)** |

42 occurrences of MTL4 typenames in ds4_metal.m. All MTL4 work uses
the shared `g_polar_compiler/queue/allocator` — single MTL4 device
context, 7 pipelines registered against it.

## Total Metal kernels (.metal files): 81

```
metal/moe.metal:        16
metal/dsv4_misc.metal:  16
metal/dense.metal:      12
metal/dsv4_hc.metal:    8
metal/dsv4_kv.metal:    5
metal/flash_attn.metal: 5
metal/argsort.metal:    2
metal/hadamard.metal:   2 (per-block + widened, this session)
metal/norm.metal:       2
metal/softmax.metal:    2
metal/unary.metal:      2
{12 other 1-kernel files}
```

7 pipelines are MTL4 / 81 total kernels = **8.6% MTL4 coverage**. The
other ~74 kernels still use classic MTL pipelines.

## Where the highest-leverage MTL4 ports live

Per-token decode hot path (silv shipped 30 t/s target):

1. **Routed MoE GPU dispatch** — `moe.metal` has 16 kernels including
   `kernel_dsv4_mul_mm_id_fp16_pair_swiglu` (the FP16 simdgroup matmul
   from #631). Currently classic MTL. Highest-volume kernel on the
   per-token decode hot path. MTL4 port + ICB capture would amortize
   dispatch over decode. ~3-5× wall-clock speed lever per codex H1725.

2. **Attention compress matmul** — `dsv4_misc.metal` has 16 kernels
   covering attention compression, MLA, layer-norm fusion. Per-token
   this fires once per layer (43×). Lower priority than MoE but
   broader coverage.

3. **Flash attention** — `flash_attn.metal` already optimized at
   classic-MTL level. MTL4 port might not deliver big wins
   (FA is compute-bound, not dispatch-bound).

## Where the highest-leverage ICB ports live

The current ICB phases cover the router cycle. The big remaining
miss is **routed MoE dispatch itself** — per-token MoE fires 6 selected
experts × 3 organs = 18 matmuls per layer × 43 layers = ~774 kernel
dispatches per token. Dispatch overhead at ~50µs each = ~40ms per
token just in encoding cost.

ICB recording of the routed MoE dispatch sequence (with the EXPERT
SELECTION as the indirect-buffer-replayed argument) would cut this to
~0ms encoding cost. The router output IS the per-token-variable
input; everything else is fixed.

## What's now done that ENABLES the integration

The daemon (#659, this commit) is the integration vehicle:

1. **Harm scorer wire** (#651-#657): organ-skip honored on CPU MoE
   dispatch via env var `DS4_ORGAN_SKIP` / `DS4_ORGAN_SKIP_CSV`.
2. **Daemon mode** (#659): `--organ-skip-configs-file` runs N configs
   in one model load. 4-config A/B in 3 seconds, bit-exact match to
   per-launch outputs. Validated on the codex H2068-H2074 prune
   candidates (#658).
3. **MTL4 Hadamard** (#653): public `ds4_gpu_mtl4_hadamard16_apply`
   ready for dispatch-site pre-pass when basis-aware sidecars exist.
4. **Basis-aware marker** (#647): per-(layer, expert, organ)
   Hadamard-basis flags with Rule 6 calibration-domain gate.
5. **DEPLOYMENT_RULES.md** (#646): six selector rules constrain how
   the basis-aware sidecar may deploy.

## Integration target: deployment-ready codec selector

The pieces in place for a coherent next ship:

  codex H2074 no-hub prune set       ← this session #658 validated
       ↓
  daemon-driven A/B at 30 prompts    ← cells.csv × 30-prompt configs file
       ↓
  per-cell Hadamard-basis decision   ← which sparse cells warrant Hadamard sidecar
       ↓
  MTL4 dispatch-site pre-pass        ← apply ds4_gpu_mtl4_hadamard16_apply
                                       conditional on the basis-aware marker
       ↓
  pruned + basis-aware GGUF emit     ← the actual deployable file

That chain is ~3 turns of work. Each step is independently testable.

## Honest scope of "advance MTL4 everywhere"

A full MTL4 port of the remaining 74 classic-MTL kernels is multi-week
work, not single-turn. Identified high-leverage NEXT ports:

- `kernel_dsv4_mul_mm_id_fp16_pair_swiglu` (routed MoE matmul) — the
  per-token decode bottleneck
- Tile×row×batch generalization for the MoE matmul (mirror H1729's
  polar_tile_pipeline pattern but on the FP16 simdgroup matmul)
- ICB capture of the routed MoE dispatch sequence (separate from MTL4
  port — could happen at classic-MTL layer first)

## Next-turn candidates

The daemon enables this; pick one for next ship:

A. **30-prompt × N-config sweep** with daemon driving codex's
   math/knowledge/code prompts. Generalizes the #658 finding from 2
   prompts to 30.

B. **MTL4 port of `kernel_dsv4_mul_mm_id_fp16_pair_swiglu`** — the
   highest-leverage single-kernel MTL4 advance per current per-token
   decode profile. ~3-5× gen speedup possible.

C. **ICB capture of routed MoE dispatch** — encoding-overhead delete.
   Independent of MTL4; could happen first.

D. **GGUF emit pipeline** — actually build a pruned+basis-aware
   sidecar GGUF using `combo_minimax_02` + the Hadamard primitive,
   then capability-A/B the deployable variant against IQ2_XXS.
