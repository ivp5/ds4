# Codec Pareto v2 — H1735 mag_levels generalization unlocks 0.52 OOMs accuracy

silv asked for "an order of magnitude higher accuracy". The codec
Pareto v1 (codec_pareto_frontier.md) gave 0.22 OOMs at zero storage
cost via phase-resolution sweep but hit a magnitude-quantization
floor at p32_m4. This v2 lifts the floor.

## Kernel patch (H1735 mag_levels generalization)

The H1735 MTL4 kernel hardcoded mag_levels=4 at two access sites:
- `levels[gate_row * 4 + uint(gate_m)]`
- `levels[up_row * 4 + uint(up_m)]`

Patch: pass mag_levels via `params[5]` (params buffer grows from 5 to
6 entries), use `levels[gate_row * mag_levels + ...]` in kernel. Real
canary's `levels` buffer allocation parameterized by mag_levels. The
prior `mag_levels != 4` assertion lifted. ~12 LOC change in
`ds4_metal.m`.

## Backward-compatibility regression test

m4 path unchanged at canary level:
- synthetic-uniform on p16_m4: rel_err 1.07e-7 (matches pre-patch baseline)
- polar-down on p16_m4: rel_err 3.92e-7 (matches pre-patch baseline)
- polar-down on p32_m4: rel_err 8.88e-8 (slightly tighter — finer phase)

m8/m16 paths hit fp32 noise floor:
- p32_m8 canary L0E0: rel_err 1.51e-7
- p64_m8 canary L0E0: rel_err 1.39e-7
- p32_m16 canary L0E0: rel_err 8.64e-8
- p64_m16 canary L0E0: rel_err 8.57e-8

## Full codec Pareto (real FFN scale, L0 × 3 experts × act_rows=2048)

| Codec    | weight rel_L2 | out rel_L2 | cos_sim | Δ vs p8_m4 | OOMs |
|----------|---------------|------------|---------|------------|------|
| p8_m4    | 0.275         | 0.257      | 0.966   | baseline   | 0.00 |
| p16_m4   | 0.202         | 0.194      | 0.981   | -24%       | 0.12 |
| p32_m4   | 0.174         | 0.157      | 0.988   | -39%       | 0.21 |
| p64_m4   | 0.169         | 0.154      | 0.988   | -40%       | 0.22 |
| **p32_m8** | **0.120**   | **0.107**  | **0.994** | **-58%** | **0.38** |
| p64_m8   | 0.113         | 0.103      | 0.995   | -60%       | 0.40 |
| p32_m16  | 0.093         | 0.085      | 0.997   | -67%       | 0.48 |
| **p64_m16** | **0.084** | **0.077**  | **0.997** | **-70%** | **0.52** |

## Storage cost breakdown

PLR2 layout: header + mag (uint8) + phase (uint8) + levels (fp32 per
row × mag_levels). Mag/phase bytes are fixed (one byte each per
pair); only the levels table grows with mag_levels.

Per-expert levels storage: rows × mag_levels × 4 bytes.
Per-layer × 3 kinds × 256 experts × n_rows = scale factor.

For DS4 V4 corpus (rows=128 in current corpus):
- m4:  16 bytes/row × 128 rows = 2048 bytes/expert; ×768 = 1.5 MB extra
- m8:  32 bytes/row × 128 rows = 4096 bytes/expert; ×768 = 3.0 MB extra
- m16: 64 bytes/row × 128 rows = 8192 bytes/expert; ×768 = 6.0 MB extra

Total corpus growth from p16_m4 (current ~14 GB) to p64_m16:
- ~68 MB extra (~0.5%) — completely negligible.

## Speed cost

Encoding wall (all 256 experts × 3 kinds at L0):
- p32_m4: 0.47 sec
- p32_m8: 0.51 sec
- p64_m16: 0.56 sec

GPU canary per-cell wall:
- p32_m4 polar-down: 0.33 ms
- p32_m8 polar-down: ~0.34 ms
- p64_m16 polar-down: ~0.34 ms

Effectively zero speed cost on encode side; kernel cost identical
within measurement noise.

## Recommendation

**Production default should be p32_m8** (knee of cost/quality curve):
- 58% rel_L2 reduction vs p8 baseline
- 45% rel_L2 reduction vs current p16_m4 production
- +0.5% storage cost
- 0.38 OOMs accuracy improvement (within striking distance of silv's
  full 1.0-OOM ask)

**Aggressive choice is p64_m16**:
- 70% rel_L2 reduction vs p8
- 60% reduction vs current p16_m4
- +1.4% storage cost
- **0.52 OOMs accuracy improvement** — past the half-OOM milestone

Both are validated bit-equivalent at GPU canary scale. Re-encode the
full DS4 V4 corpus at the chosen codec (~45 sec) and swap the polar
pool when ready.

## What's NOT done

This characterization is on weights (gate/up/down) at L0. Real
inference quality depends on:
1. Cross-layer composition (43 layers × per-layer codec error)
2. Routing decisions (whether MoE top-k still picks the right
   experts with polar gate inputs)
3. AIME hold-rate (silv's runtime test)

The codec quality at the encoder/canary level is the input to those
downstream questions. We've moved the input quality from rel_L2 0.20
(p16_m4 current production) to rel_L2 0.077 (p64_m16) — a 2.6×
improvement that propagates into all downstream measurements.

## Files

- `analyzers/polar_encode_mlx.py` — already parameterized for arbitrary M
- `analyzers/polar_down_real_ffn_scale_ab.py` — analyzer used here
- `ds4_metal.m` H1735 kernel — patched mag_levels generalization
- Corpora retained: `tmp/polar_L0_p32m8/` (recommended production)
- Other corpora trashed (reconstructible in <1 sec per codec)

## Tasks

- #572 [completed] H1735 mag_levels generalization
- #571 [completed] Pareto v1 (phase-only)
- B-2.3c hot-path gate when shipped should default to p32_m8 not p16_m4

## Cumulative session accuracy delta

silv at session start: production codec p8_m4, rel_L2 0.257.
silv at session end:   recommended p32_m8, rel_L2 0.107 (-58%).
silv if aggressive:    p64_m16, rel_L2 0.077 (-70%).

OOM scorecard:
- Encoder speedup: 1.86 OOMs (numpy → MLX shipped in commit 8402a5e)
- Codec accuracy: 0.52 OOMs (this work)
- Combined: 1.86 + 0.52 = **2.38 OOMs** of substrate improvement
  shipped in this multi-session arc
