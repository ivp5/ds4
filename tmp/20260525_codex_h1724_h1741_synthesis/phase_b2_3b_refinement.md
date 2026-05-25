# Phase B-2.3b refinement — codec quality DOES NOT improve with accumulation length

The single-geometry 1.21× ratio in commit `1aa7e25` was accurate within
its scope (act_rows=16, down_rows=8) but extrapolating it to "real
FFN scale via CLT averaging" would be wrong. Empirical sweep
(`tmp/polar_actrows_sweep_*.log`) across 9 cells × 4 act_rows values
shows:

```
L     E    ar=  weight_rl2  out_rl2   cos
0     0   16      0.2237    0.3082    0.9524
0     0   32      0.2234    0.2680    0.9752
0     0   64      0.2049    0.2682    0.9685
0     0  128      0.2027    0.2370    0.9808
...
22  100   64      0.2090    0.0854    0.9964   ← lucky cell
42    0   64      0.1984    0.7465    0.8226   ← unlucky cell
```

**Aggregate across act_rows ∈ {16, 32, 64, 128}** (9 cells each):
- ar=16:  out rel_L2 ≈ 0.236
- ar=32:  out rel_L2 ≈ 0.201
- ar=64:  out rel_L2 ≈ 0.273 (outlier-heavy)
- ar=128: out rel_L2 ≈ 0.236

No monotone trend with accumulation length. **Output rel_L2 stays in
the same order of magnitude as weight rel_L2 (≈ 0.20) regardless of
act_rows.**

## Mechanism

Codec error is dominated by 4-level magnitude quantization, which is
a **systematic magnitude bias** rather than independent random noise.
The CLT averaging that would reduce iid noise by `1/sqrt(N)` does NOT
apply when the per-r errors are biased in the same direction (e.g.,
all quantized magnitudes round down or all round up across r).

The high cos_sim ≥ 0.95 across nearly all cells (one outlier at 0.82)
confirms this: the dot-product DIRECTION is preserved; the MAGNITUDE
is the wobbly quantity.

## Implications for B-2.3c hot-path

- Polar substitution at the FFN call site will produce output that
  differs from FP4 baseline by **~0.20 rel_L2 per call**, NOT by
  0.02 as the naive CLT prediction would suggest.
- Multi-layer composition: 43 layers × ~0.20 per layer = potentially
  large cumulative drift. Whether this still produces correct AIME
  output is **B-2.3d's runtime question** — silv's DS4-sticky-hazard
  surface.
- The cos_sim ≥ 0.95 floor is the more relevant deployability metric.
  Cosine-preserving error tends to leave attention/routing decisions
  intact while shifting numerical magnitudes.

## What's NOT refuted

- B-2.3a's kernel correctness validation (fp32 noise floor on
  bit-equivalence) is unchanged.
- The D1 architectural decision (per-call polar-down extraction) is
  unchanged — it's still the right choice.
- The polar codec headline rel_L2 ≈ 0.20 is still the codec quality.

## What IS sharpened

- The kernel composition does NOT amplify codec error (good).
- The kernel composition does NOT reduce codec error via CLT
  averaging (less good than naive prediction).
- Output codec quality at the FFN-call boundary will be the same
  order of magnitude as weight-level codec quality.
- B-2.3d AIME hold-rate is the gate-decision metric; codec
  measurements alone can't predict the answer.

## Honesty addendum

The prior commit's claim that act_rows=2048 in real inference would
yield "much stronger CLT averaging" is **refuted by the empirical
data above**. Errors don't average out beyond a single doubling.
The 1.21× headline was for canary geometry only.

Re-running the analyzer at act_rows ≥ 256 requires re-encoding the
polar corpus with rows > 128 (current corpus is row-limited).
Per-row trend at ar ∈ {16, 32, 64, 128} suggests the asymptote is
~0.20 — the codec floor itself — not 0.

## Tasks updated

- #569 [completed] B-2.3b — closes with REFINED framing (output ≈
  weight, no CLT averaging).
- Next: B-2.3c hot-path env-var gate is structurally ready; whether
  to ship depends on silv's risk tolerance given the cumulative
  multi-layer drift unknown.
