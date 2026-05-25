# SCOPE CAVEAT: codec corpora encode 3-6% of weight rows, not full coverage

## The honest finding

Every codec measurement in this session arc (polar p32_m8, VQ K=256,
all canary validation, all weight-level + output-level rel_L2 numbers)
is computed on a **128-row chunk per expert** out of:
- gate/up: 2048 rows total → **6.25% coverage**
- down: 4096 rows total → **3.13% coverage**

The encoder's defaults `--rows-per-tile 32 --n-tiles 4` give 128
rows; the polar production corpus (commit b22698d, 14 GB) and the
VQ-2D K=256 finding (commit 503934a-equivalent) both inherit this
configuration.

## What this means for the codec quality claims

**The substrate codec quality findings ARE valid** for the encoded
portion:
- polar p32_m8: rel_L2 0.121 on first 128 rows
- VQ K=256: rel_L2 0.020 on first 128 rows

The 6× improvement is real on the sampled slice. Whether this
generalizes to the unsampled 94-97% of weights is **unmeasured**.
The FP4 distribution is structurally similar across rows (same
underlying weight tensor), so projection-to-full would likely
preserve the codec rank order, but the exact rel_L2 numbers at
full coverage are not measured.

## What this means for deployment

Two paths for actual DS4 V4 deployment:

### Path A — extend coverage to full rows

Re-encode at `--rows-per-tile 2048 --n-tiles 1` (gate/up) and
`--rows-per-tile 4096 --n-tiles 1` (down). Storage cost projection:

```
polar p32_m8 at full rows:
  per layer: 6.44 GB
  43 layers: 277 GB  ← EXCEEDS CLAUDE.md 100 GB disk budget

VQ K=256 at full rows:
  per layer: 3.22 GB
  43 layers: 138 GB  ← STILL EXCEEDS 100 GB budget
```

Both exceed silv's standing 100 GB disk budget. Would require
explicit silv approval per CLAUDE.md disk-budget rule.

### Path B — partial coverage + FP4 fallback

Keep the 128-row encoded chunk, route additional FFN output
positions through the FP4 path. The FFN call splits into:
- Polar/VQ kernel: outputs 0..127 (encoded)
- FP4 kernel: outputs 128..4095 (fallback)

This preserves the current corpus size (14 GB at p32_m8 / 4.3 GB at
VQ K=256) but only delivers codec benefit on ~3-6% of FFN compute.

The OOM-accuracy gain delivered on this sliver is real but the
end-to-end inference improvement is bounded by Amdahl: if FFN is
50% of inference and 6% of FFN is encoded, max codec impact is
3% of inference quality.

### Path C — codec architectural change to fit budget

3rd codec generation beyond polar/VQ-2D:
- 4D VQ (group 2 pairs into 4D vector, K=256 → 0.5 bytes/pair) →
  ~70 GB full coverage (FITS budget)
- Per-block scalar quant + 4-bit residual → unmeasured

These are research directions. Estimated 0.5-1.5 OOMs more accuracy
beyond VQ K=256, at unknown storage cost.

## What silv needs to know

1. **The OOM-accuracy ask IS delivered** — at substrate level, on
   the encoded 128-row chunk. The kernel-side validation (B-2.3a)
   and output-side codec quality (B-2.3b) are both real numbers.

2. **The 14 GB "production p32_m8 corpus"** is testing-scope, not
   production-coverage. Production at full rows would need 277 GB.

3. **The VQ-2D 4.3 GB estimate** was also at the same testing scope.
   Full coverage VQ K=256 = 138 GB (still over budget).

4. **B-2.3c hot-path gate** would need to decide: encoded-rows-only
   substitution (Amdahl-bounded benefit) or full-coverage corpus
   (over budget without silv approval).

5. **B-2.3d AIME hold-rate** measurement should account for which
   path is being tested: encoded-only chunk gives a fractional
   FFN substitution; full-coverage gives the full codec impact.

## What's NOT affected by this caveat

- Encoder MLX speedup (1.86 OOMs) — speedup applies regardless of
  row count; faster per-row.
- H1735 mag_levels generalization — kernel patch is independent of
  row scope.
- VQ-2D substrate finding — the codec quality is real, just at
  testing scope.
- The Pareto v2 codec recommendation (p32_m8 vs p64_m16 vs VQ K=256)
  — all are at the same testing scope so the ranking holds.

## Action items

This caveat doesn't invalidate prior commits — they were accurate
within their stated scope. But the inheritable claims should be
sharpened:

- "p32_m8 production corpus (14 GB)" → "p32_m8 production corpus at
  128-row sampling (14 GB, ~6% coverage)"
- "0.52 OOMs accuracy" → "0.52 OOMs accuracy on encoded weight slice"
- "ship VQ engineering" → "decide row-coverage strategy FIRST, then
  ship encoder + kernel + canary for chosen strategy"

silv decision needed before further codec engineering:
- (A) Extend to full coverage (~277 GB polar / 138 GB VQ — needs disk
  budget approval)
- (B) Accept partial coverage + FP4 fallback for unencoded rows
- (C) Research-direction: 4D VQ or other architecture to fit budget

## Self-criticism

I should have noticed the row-coverage scope earlier in the polar
arc — the 14 GB corpus size for DS4 V4 (~85 GB FP4 source) is
obviously much smaller than the full coverage would imply.
Specifically I missed it because:

1. The polar canary tested per-cell at down_rows=8, act_rows=16 —
   tiny dispatches that didn't surface "is the whole tensor
   encoded?".

2. The kernel API has `down_rows = params[3]` as a per-dispatch
   parameter, which I read as "configurable per call" without
   asking "does the corpus actually contain all the rows needed for
   real FFN?"

3. The encoder defaults `rows_per_tile=32 × n_tiles=4 = 128` came
   from a prior session's testing config; I treated them as
   production defaults without checking against weight shapes.

Per "back-of-hand discipline + ground-decode" doctrine, I should
have verified what fraction of the weight tensor was actually
encoded BEFORE claiming "production-ready". This caveat is the
ground-decode I should have done earlier in the arc.

The substrate codec findings are still real and useful. The
deployment readiness claim was over-extended. silv now has the
honest picture.
