# Running ds4-agent with trim50 — minimal steps

DS4 trim50 is the ~47.7 GiB DeepSeek V4 Flash with ~50% of routed experts dropped
via a layer-asymmetric mask. It fits Metal residency below the M1 Max
`iogpu.wired_limit_mb` raised cap without `--cpu-moe`.

## 1. One-time prerequisites

```bash
# Build ds4-agent (and ds4, ds4-server, etc.)
make ds4-agent

# Verify Metal cap on M1 Max — must be >= 60 GiB for trim50 + MTP residency.
# Default is ~48 GiB; raise it (silv 2026-05-25 confirmed safe at 60 GiB):
sudo sysctl iogpu.wired_limit_mb=61440
```

## 2. Build trim50 if missing

```bash
# Will skip if gguf/DS4-trim50-asym-with-metadata.gguf already exists.
./scripts/build_trim50.sh

# Custom paths:
DS4_BASE_GGUF=/some/path/DeepSeek-V4-Flash-IQ2XXS-...gguf \
    ./scripts/build_trim50.sh /custom/out.gguf
```

Requires the full IQ2_XXS GGUF (~86.7 GiB). Source:
[DevQuasar/deepseek-ai.DeepSeek-V3.1-Flash-GGUF](https://huggingface.co/DevQuasar/deepseek-ai.DeepSeek-V3.1-Flash-GGUF).
Mask shipped at `masks/mask_asym_50gb_v2.csv`.

## 3. Run the agent

ds4-agent does NOT take `--mtl4-moe` (that's a `ds4` flag only). Without
`--prefill-metal-phases auto` or `--cpu-moe`, the engine goes full-Metal
directly. Verified: `mapped 45475.62 MiB` for trim50.

Minimal interactive (full-Metal MoE, ICB on, MTP off):

```bash
DS4_EXPERT_REMAP_ACTIVE=1 DS4_ICB_ACTIVE=1 \
    ./ds4-agent \
    -m ./gguf/DS4-trim50-asym-with-metadata.gguf
```

With MTP draft model (faster on structured prompts, see [perf table](#perf-table)):

```bash
DS4_EXPERT_REMAP_ACTIVE=1 DS4_ICB_ACTIVE=1 \
    ./ds4-agent \
    -m ./gguf/DS4-trim50-asym-with-metadata.gguf \
    --mtp ./gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
    --mtp-draft 2
```

Non-interactive one-shot with a prompt:

```bash
DS4_EXPERT_REMAP_ACTIVE=1 DS4_ICB_ACTIVE=1 \
    ./ds4-agent \
    -m ./gguf/DS4-trim50-asym-with-metadata.gguf \
    --non-interactive \
    --temp 0 \
    -p "Write a Python function that returns the n-th Fibonacci number."
```

For raw `ds4` (non-agent, no DSML tool system prompt) use `--mtl4-moe` per [§4](#4-env-flags-reference):

```bash
DS4_EXPERT_REMAP_ACTIVE=1 DS4_ICB_ACTIVE=1 \
    ./ds4 -m ./gguf/DS4-trim50-asym-with-metadata.gguf --mtl4-moe \
    --temp 0 -p "Hello, my name is" -n 16
```

### Quality note on trim50

The trim50 model has ~50% of its routed experts dropped via the layer-asymmetric
mask. Quality is acceptable on chat-shaped continuations but degrades on
out-of-distribution prompts and short-form factual queries — the agent may
emit repetition loops or DSML-system-prompt continuations instead of the
expected answer. The inference engine is correct; this is a model-quality
ceiling that is inherent to the trim. For full quality, use the un-trimmed
IQ2_XXS file directly (requires either `--cpu-moe` or wired_limit_mb >= 90 GiB).

## 4. Env flags reference

| flag | purpose |
|---|---|
| `DS4_EXPERT_REMAP_ACTIVE=1` | engages the trim50 inference plumbing (fused `router_weights_with_remap` kernel + inverse expert table) |
| `DS4_ICB_ACTIVE=1` | enables MTLIndirectCommandBuffer record→replay for the route remap (+10% prefill, +4% gen, 6-30× lower variance) |
| `--mtl4-moe` | takes the Metal-MoE path. On M1 Max falls back to legacy SIMD MoE (Metal 4 tensor API is M5+ only); needed to avoid the `--cpu-moe` fallback in `--prefill-metal-phases auto` |
| `--mtp <path>` | optional MTP draft model for speculative decode |
| `--mtp-draft N` | max draft tokens per speculative step (1-16, default 1) |
| `--mtp-margin F` | min draft confidence to invoke verify (default 3.0; lower for more aggressive drafting) |
| `--quality` / `DS4_MTP_STRICT=1` | force exact-decode MTP verifier (slower but bit-stable) |

## 5. Perf table {#perf-table}

M1 Max 64 GiB, sysctl `iogpu.wired_limit_mb=61440`, 3-run mean ± std, trim50 model:

| config | prefill (t/s) | gen (t/s) |
|---|---|---|
| `--cpu-moe` (legacy) | 3.90 | 4.80 |
| full-Metal | 30.30 ± noise | 19.20 ± noise |
| full-Metal + ICB | **30.87 ± 0.08** | **19.67 ± 0.10** |
| full-Metal + ICB + MTP-m3 (struct prompt) | ~30 | **27-41** (1.4-2.1×) |
| full-Metal + ICB + MTP-m3 (creative prompt) | ~30 | ~17-18 (LOSS) |

MTP is net-positive on structured/predictable continuations (factual, math, code,
narrative); net-negative on open-ended creative text where draft accept rate is
below ~70%. See `tmp/20260525_ane_probe/findings.md` for the full sweep.

## 6. Memory ceiling at 60 GiB cap

Linear context scaling past ctx=4K at ~12 KiB/token (DS4 uses MLA latent compression):

| ctx | total wired | notes |
|---|---|---|
| 32K | 46.1 GiB | huge headroom |
| 128K | 47.2 GiB | comfortable |
| 1M | 56.6 GiB | 3.4 GiB headroom |
| **1.34M** | 60.0 GiB | **ceiling without MTP** |
| **1.03M** | 60.0 GiB | **ceiling with MTP (+3.6 GiB)** |
| 2M | 68.7 GiB | overshoots cap |

## 7. Sticky hazard (M1 Max-specific)

Pre-cap-raise, DS4 binaries running without `--prefill-metal-phases auto` or
`--cpu-moe` caused two kernel panics (2026-05-19, 2026-05-23) by attempting to
wire 80+ GiB into Metal residency on a 48 GiB cap. With cap raised to 60 GiB
and `trim50` (44.4 GiB residency), the panic condition can no longer trigger.
The `--mtl4-moe` flag in §3 still works because it doesn't activate `--cpu-moe`
nor request phase-split prefill on the trim50 model.

If you build a custom GGUF that exceeds 60 GiB residency, restore the phase-split
safeguard:

```bash
./ds4 -m /big.gguf --prefill-metal-phases auto ...
```

Per-launch sanity check: total residency (model GGUF size + ~12 KiB × ctx + MTP
size if loaded) MUST stay below your sysctl-set cap. The engine logs
`mapped <N> MiB` after Metal init — if N exceeds the cap, kill immediately.
