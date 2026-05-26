# Metal-MoE FP16 hot-store wiring + flag cleanup — session 2026-05-26

## Goal recap (silv)

DS4 Flash on M1 Max 64 GB: target 20-30 t/s gen. trim50 (44 GB GGUF, RAM-fits)
is the working substrate.

## What landed this session

Commit `c454b79` "Metal-MoE FP16 hot-store hook + flag cleanup":

### Flag landscape (ds4.c CPU-MoE site)

Before: 3 overlapping env flags at the CPU-MoE dispatch site (DS4_VQB2_FP16,
DS4_VQB2_FP16_METAL, DS4_VQB2_FP16_PATH), the latter two effectively orphan
because the standalone ds4_metal_vqb2_fp16 dispatcher had its own ICB/MTL4/LEGACY
selector that the caller didn't directly engage.

After:
- `DS4_HOT_FP16=1` — canonical name for "predequant FP16 hot-store dispatch
  at the CPU-MoE site"
- `DS4_VQB2_FP16=1` — accepted as legacy alias for compat with existing scripts
- DROPPED: DS4_VQB2_FP16_METAL, DS4_VQB2_FP16_PATH at this site (the standalone
  dispatcher still owns its own DS4_VQB2_FP16_PATH=legacy|icb|mtl4 selector)

### New Metal-MoE-side hook (gen path)

Added at `ds4.c` near `ds4_gpu_routed_moe_one_tensor` caller, where layer
index `il` is in scope. Gated by `DS4_HOT_METAL_MOE=1`.

Flow:
1. Check active hot-store has all 6 selected experts pinned for this layer
2. Read sel/w/x from GPU tensor_contents (M1 unified memory = zero-copy)
3. Call `ds4_metal_vqb2_fp16_dispatch` (path selected by DS4_VQB2_FP16_PATH)
4. On success skip the IQ2_XXS pair_swiglu kernel; on failure fall through

Default off; existing IQ2_XXS gen path byte-identical when flag unset.

### Codec caveat (documented in code)

The existing `kernel_vqb2_fp16_gate_up` Metal kernel reads weights as
`[n_rows][n_pairs][2]` complex pairs and expects input formatted as
`n_rows` complex pairs. DS4 routed FFN uses vanilla matvec. Until the
codec/layout aligns the Metal hot-store hook produces numerically wrong
output. The wiring is in place for the next-iteration codec work.

## Empirical baseline (trim50 + Metal gen)

| Config | gen t/s | Notes |
|---|---|---|
| Vanilla, gen=64 | 17.5 | small-N best |
| Vanilla, gen=96 | 15.6-16.4 | |
| Vanilla, gen=128 | 9.4-10.9 | sustained-load thermal throttle |
| DS4_HOT_FP16=1 (CPU-MoE site) | 5.1 t/s | only fires when gen is on CPU |
| DS4_HOT_METAL_MOE=1 (this commit) | 9.8 t/s | hook fires; codec mismatch → wrong output |
| DS4_EXPERT_REMAP_ACTIVE=1 | -3 t/s | adds remap dispatch overhead |
| DS4_EXPERT_REMAP_ACTIVE=1 + DS4_ICB_ACTIVE=1 | -4 t/s | ICB overhead piles on remap |

silv's recalled "30 t/s prefill, 20 t/s gen" baseline appears to be small-N
noise — sustained gen=128 is 10 t/s on this hardware.

## Paths to 20-30 t/s gen

Single-flag flips have been exhausted. Real gains require kernel-level work:

1. **Codec alignment for FP16 hot-store**:
   - Option A: change hot-store encoding to vanilla row-major FP16
   - Option B: add Metal kernel that interprets polar pairs from vanilla input
   - Either unlocks the Metal-MoE FP16 hook to actually accelerate

2. **simdgroup_multiply on FP16 weights**:
   - M-series GPU has 32×32 simdgroup half matmul instruction
   - Current IQ2_XXS pair_swiglu does per-thread dequant+dot (no simdgroup)
   - FP16 row-major could feed simdgroup matmul → ~8× per-dispatch speedup
   - Requires the FP16 path from (1) to land first

3. **Fix DS4_EXPERT_REMAP_ACTIVE regression**:
   - Codex task #542 predicts "4× gen speedup on trim50 plumbing path"
   - Current implementation adds per-layer dispatch (+~500us each)
   - The win materializes only if IQ2_XXS pair_swiglu kernel ALSO learns to
     skip zero-weighted experts; remap-alone is pure overhead
   - Fix: add early-exit on weight=0 in pair_swiglu, OR pre-mask the selected
     buffer to drop dead experts entirely

4. **DS4_ICB_ACTIVE** also regresses today (-4 t/s). Codex screenshot shows
   ICB Phase 2 should give +4.3% with 6× std reduction. Local measurement
   contradicts; either my baseline is in a different operating regime or
   the ICB path has a latent bug only my version exhibits.

5. **Spec-decode** (#418 in progress) — 2-3× win when draft model accept
   rate is high

## What "continue" means

Per silv's directive "wire and continue", this commit ships the wiring; the
codec alignment work is a separate, larger move. Next session's lever:

- **Path A (smaller)**: investigate why `DS4_EXPERT_REMAP_ACTIVE` regresses
  vs codex's prediction — single-kernel-source debugging.
- **Path B (bigger)**: align FP16 hot-store layout to vanilla row-major +
  add Metal kernel that uses simdgroup_multiply.

silv's screenshot prior-message bottom: "commit, pull from antirez, merge,
push to ivp5, give minimal steps on how to run ds4 agent with trim50 and
script to generate trim50 if missing" — pending explicit re-ask per the
"no push without explicit instruction" rule.
