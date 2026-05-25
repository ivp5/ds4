# Codec Pareto frontier — zero-storage phase resolution sweep

silv asked for "an order of magnitude higher accuracy which should
enable to sense small aberrations unnoticed before". The polar codec
has zero-storage-cost upgrades available: PLR2 stores phase as
uint8 (code ∈ [0, P]), so increasing P from 8 → 256 keeps the byte
layout identical.

Empirical sweep across L0 × 3 experts × act_rows=2048 (real FFN
scale, all kinds gate/up/down encoded):

| Codec    | weight rel_L2 | out rel_L2 | out cos_sim | improvement vs p8 |
|----------|---------------|------------|-------------|-------------------|
| p8_m4    | 0.275         | 0.257      | 0.966       | baseline          |
| **p16_m4** | **0.202**   | **0.194**  | **0.981**   | **-24%** (current production) |
| **p32_m4** | **0.174**   | **0.157**  | **0.988**   | **-39% ← NEW PARETO OPTIMUM** |
| p64_m4   | 0.169         | 0.154      | 0.988       | -40%              |
| p128_m4  | 0.168         | 0.153      | 0.988       | -40%              |
| p256_m4  | 0.167         | 0.153      | 0.988       | -40%              |

## Findings

1. **p8 → p16 → p32 are each meaningful improvements**: -24% then -19%
   at zero storage cost.

2. **p32 is the knee**: doubling to p64 gives only -2% additional
   reduction. p32 dominates the prior production p16 on every metric
   at the same byte cost.

3. **Beyond p32, mag_levels=4 is the floor**: p128 and p256 are
   essentially identical to p64 (within 0.003 rel_L2). The codec at
   mag_levels=4 cannot go below ~0.153 output rel_L2 / 0.988 cos_sim
   regardless of phase resolution.

4. **OOM-accuracy ask partial**: full sweep gives 0.22 OOMs accuracy
   improvement (0.257 → 0.153). The remaining 0.6 OOMs to reach
   silv's full OOM target requires:
   - Higher mag_levels (m8, m16) — would need kernel mag_levels
     generalization (currently hardcoded at 4 per ds4_metal.m line
     16526). Storage cost: rows × (mag_levels - 4) × 4 bytes per
     expert per layer. At m16: 256 experts × 128 rows × 12 × 4
     bytes × 3 kinds × 43 layers = ~190 MB extra (≈ 1.4% bigger).
   - OR: per-pair codebook (different codebook per (row, pair_idx)
     instead of per-row) — substantial encoder rewrite.
   - OR: 2D vector quantization (encode complex pair as one of N
     centroids) — different codec architecture.

## Deployment implication

**Drop-in upgrade**: re-encode the corpus at p32_m4 and swap the
polar-dir on next session. The C reader already supports arbitrary
phase_levels (per commit 278db99); no kernel change needed.

Time cost: full DS4 V4 corpus re-encoding ~42 sec (per commit
b22698d). Storage cost: identical to current p16_m4 corpus (14 GB).

The B-2.3c hot-path gate work, when shipped, should default to
p32_m4 not p16_m4. The per-FFN-call output codec error becomes:
- Current p16_m4: rel_L2 0.21
- Proposed p32_m4: rel_L2 0.16

Whether the 5-point reduction is enough to swing AIME hold-rate
acceptably is B-2.3d's question, but starting from p32_m4 gives a
better baseline.

## What's next on the OOM-accuracy axis

To reach silv's full 1-OOM accuracy ask (rel_L2 < 0.025) requires
mag_levels expansion. Three paths:

1. **m8 expansion + kernel patch**: doubles per-row codebook size,
   small storage cost. Kernel needs to read mag_levels from PLR2
   header instead of asserting 4. Estimated gain: rel_L2 ~0.08
   (CLT-style halving of per-row magnitude error).

2. **m16 expansion + kernel patch**: 4× codebook size, ~190 MB extra
   corpus. Estimated rel_L2 ~0.04.

3. **Per-pair codebook**: each (row, pair_idx) has its own 4-level
   codebook. 1024× more codebook storage at p16_m4 → not viable;
   would need to compress codebooks via shared dictionary.

Path 1 (m8) is the natural next step — smallest kernel patch + good
gain. The C kernel changes: replace hardcoded `* 4 *` in the levels
buffer allocation with a runtime mag_levels parameter. ~10 LOC change.

Files: encoder commands in commit history above; analyzer at
`analyzers/polar_down_real_ffn_scale_ab.py`; codec corpora at
`tmp/polar_L0_p{8,16,32,64,128,256}m4/`.
