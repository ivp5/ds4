# Running DS4 with trim50 on M1 Max 64 GB

trim50 is the asymmetric expert-mask trimmed GGUF (~44 GiB), the working
substrate for DS4 inference / bench on M1 Max. It fits in Metal residency
under the 48 GiB wired-memory cap (with `iogpu.wired_limit_mb >= 60 GiB`),
so all routed MoE can run on GPU at gen time — no `--cpu-moe`, no
`--prefill-metal-phases` needed for safety.

## 1. Generate the file (if missing)

`scripts/build_trim50.sh` writes the trim50 GGUF if it doesn't exist.

Inputs:
- Full IQ2_XXS source GGUF (~86.7 GiB) — must already be on disk
- Asymmetric mask CSV: `masks/mask_asym_50gb_v2.csv`
- Python ≥ 3.9 on PATH

```bash
cd /Users/silv/cl/tlp/montyneg/ivp5_ds4

# Default locations checked; explicit override via env:
DS4_BASE_GGUF=/path/to/full-IQ2XXS.gguf ./scripts/build_trim50.sh

# Or with explicit output path:
./scripts/build_trim50.sh /path/to/output.gguf
```

Default output: `gguf/DS4-trim50-asym-with-metadata.gguf` (also at
`/Users/silv/cl/tlp/montyneg/ds4/gguf/DS4-trim50-asym-with-metadata.gguf`
on this machine — 44 GiB).

If the file already exists, the script is a no-op.

## 2. Build the binaries

```bash
cd /Users/silv/cl/tlp/montyneg/ivp5_ds4
make ds4-bench   # or `make ds4` for the chat CLI
```

## 3. Run with all-Metal gen (best gen t/s)

trim50 fits in Metal residency, so launch with NO safety flag. Gen-time
MoE runs on GPU:

```bash
MODEL=/Users/silv/cl/tlp/montyneg/ds4/gguf/DS4-trim50-asym-with-metadata.gguf

./ds4-bench \
  --prompt-file tmp/vqb2_smoke_prompt.txt \
  --model "$MODEL" \
  --ctx-start 256 --ctx-max 256 --gen-tokens 64
```

Expected: prefill ~100 t/s, gen ~16-18 t/s small-N, ~10 t/s sustained
(thermal). Mapped residency: 45.5 GiB (under 48 GiB cap — safe).

**Late-warning guard** (CLAUDE.md DS4 hazard protocol — watch for this
line in stderr):

```
ds4: Metal model views created in ... ms (mapped <N> MiB ...)
```

If `<N>` > 48000 MiB, KILL the process immediately. For trim50 it should
print ~45475 MiB. Wrapping script with auto-kill:

```bash
./tmp/20260526_metal_dispatch/run_metal_gen.sh
```

## 4. Run with --cpu-moe (slower but safer / for debugging)

```bash
./ds4-bench \
  --prompt-file tmp/vqb2_smoke_prompt.txt \
  --model "$MODEL" \
  --cpu-moe \
  --ctx-start 256 --ctx-max 256 --gen-tokens 64
```

Expected: prefill ~20 t/s, gen ~5 t/s. The `--cpu-moe` flag routes all
routed experts to CPU. Use this when:
- Testing the CPU-side dispatch (DS4_HOT_FP16 hot-store path)
- Debugging under deterministic single-thread CPU compute
- Running on a machine that can't fit Metal residency

## 5. Env flags (current landscape)

### CPU-MoE site (only fires when `--cpu-moe` or `--prefill-metal-phases` set)
- `DS4_HOT_FP16=1` (canonical) or `DS4_VQB2_FP16=1` (alias) — engage the
  predequant FP16 hot-store dispatch when active hot-store has all 6
  selected experts pinned for a layer. Needs `--vqb2-manifest` +
  `--vqb2-candidate` flags to load packets first.

### Metal-MoE site (fires when gen is on Metal — default)
- `DS4_HOT_METAL_MOE=1` — engage the Metal-side hot-store hook.
  **CURRENTLY OFF BY DEFAULT** because (a) the polar codec format
  doesn't match DS4's vanilla matvec and (b) the hook reads
  tensor_contents mid-batch with stale data. See ds4.c near the
  ds4_gpu_routed_moe_one_tensor call for the in-code caveat.

### trim50 plumbing
- `DS4_EXPERT_REMAP_ACTIVE=1` — engage the fused router_weights_with_remap
  kernel. Maps router-emitted IDs (0..255) → trim file's compressed IDs.
  Currently regresses gen ~3 t/s vs vanilla; codex task #542 predicts
  this should give +4× when paired with skip-zero in the MoE kernel.
- `DS4_ICB_ACTIVE=1` — record→replay route_remap commands via
  MTLIndirectCommandBuffer. Currently regresses; codex's Phase 2 work
  expects +4.3% gen with 6× std reduction.
- `DS4_ICB_WEIGHTS_ONE=1` — opt-in ICB for route_weights_one. Currently
  marked as a "losing path" — additional regression.

### Diagnostic
- `DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION=1` — disable the
  `kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32` fused kernel. The fusion
  is worth ~3 t/s gen, so this flag costs perf — use only for
  correctness debugging.

## 6. Reproducing the session bench

```bash
./tmp/20260526_metal_dispatch/run_metal_gen.sh                # baseline
bash tmp/20260526_metal_dispatch/abx_trim50_v2.sh             # CPU-MoE A/B
```

## Known issues

1. **CPU-MoE site VQB2 dispatch shape mismatch**: the hot-store FP16
   layout is encoded for a 128-row × 2048-pair probe configuration that
   doesn't match DS4-Flash's actual FFN dims. The CPU dispatcher works
   on the probe shape but doesn't replace full DS4 inference. Per-tile
   coverage is a partial row-block; full coverage needs more packets.

2. **Sustained-gen thermal throttle**: gen t/s degrades from ~17 (gen=64)
   to ~10 (gen=128) as the M1 ramps. Either set up cooler ambient
   conditions, or accept that "sustained" measurement is the load-bearing
   number.

3. **trim50 + DS4_EXPERT_REMAP_ACTIVE regression**: known. The fused
   remap kernel adds dispatch overhead; codex's predicted win materializes
   only when the IQ2_XXS MoE kernel also learns to early-exit on
   weight=0. See task #542.

4. **Metal-MoE FP16 hot-store hook produces wrong output**: known.
   Two caveats documented at the hook site in ds4.c — codec mismatch +
   GPU/CPU sync. Default off; needs next-iteration kernel work to enable.
