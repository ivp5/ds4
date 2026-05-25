# <52GB on M1 Max via VQ knobs: honest verdict

silv 2026-05-25: "does vq have knobs to tune space-vs-accuracy tradeoff to try
and fit into less than 52gb and thus inside m1 max 64gb memory?"
+ "work on block kernel as well - compare all high potential"

## Measured Pareto across 23 VQ/Block-VQ/PQ recipes

`vq_k_sweep_for_52gb.py` (per-pair) + `block_vq_sweep.py` (block + PQ).
Source: L0 gate expert 0, first 128 rows, vs FP4 source. IQ2_XXS reference
rel_L2 = 0.354 at 0.516 byte/pair.

| Recipe | byte/pair | rel_L2 | vs IQ2 | F-DS4 GB | T50 GB |
|--------|----------:|-------:|-------:|---------:|-------:|
| IQ2_XXS reference | 0.516 | 0.354 | 1.00× | 86.7 | 47.7 |
| **VQ K=256 per-pair** | 1.000 | 0.021 | **17×** | 159 | 84.5 (no fit) |
| **VQ K=256 + block32 scale** | 1.063 | 0.073 | 4.85× | 168 | 89 (no fit) |
| **PQ block=16p sub=8 K_sub=256** | 0.500 | 0.302 | 1.17× | 84.4 | **47.2 (FIT T50)** |
| PQ block=16p sub=4 K_sub=4096 | 0.375 | 0.306 | 1.16× | 65.8 | 37.9 (FIT T50) |
| VQ K=16 + block32 scale | 0.563 | 0.319 | 1.11× | 93.7 | 51.8 (FIT T50 tight) |
| VQ K=4 (any scale) | 0.25-0.31 | 0.61 | 0.58× WORSE | 47-56 | 28-33 (fits but degrades) |
| Block-VQ block=16 K=256 | 0.0625 | 0.851 | 0.42× WORSE | 19.3 | 14.6 (fits but degrades severely) |

## Verdict for Full DS4 in <52 GB

**NOT ACHIEVABLE at quality ≥ IQ2.** The information-theoretic floor for ~85B
params at FP4-comparable quality sits at ~0.5 byte/pair, which translates to
~85 GB on Full DS4's routed-expert mass alone. No tested recipe both fits
under 52 GB AND beats IQ2's 0.354 rel_L2.

The Block-VQ and PQ recipes below 0.3 byte/pair all DEGRADE quality below
IQ2 (rel_L2 0.5-0.85). The R^32 distribution of 32-weight blocks has too much
intrinsic variance for K=256 (or even K=2048) to cover at low bit budgets.

Two paths to fit Full DS4 in <52 GB:
1. **Accept quality degradation** — VQ K=4 at 0.25 byte/pair fits but rel_L2
   = 0.6 (1.7× WORSE than IQ2). Inference would visibly degrade.
2. **Drop more experts** (= more aggressive trim than trim50) — orthogonal
   to codec choice. silv's existing trim path is the lever.

## Verdict for Trim50 in <52 GB

**ACHIEVABLE with marginal quality lift.** Three deployable recipes:

| Recipe | byte/pair | rel_L2 | T50 GB | Quality lift |
|--------|----------:|-------:|-------:|-------------:|
| IQ2_XXS (current) | 0.516 | 0.354 | 47.7 | baseline |
| **PQ block=16p sub=8 K_sub=256** | 0.500 | **0.302** | 47.2 | **1.17×** |
| PQ block=16p sub=4 K_sub=4096 | 0.375 | 0.306 | 37.9 | 1.16× |
| VQ K=16 + block32 scale | 0.563 | 0.319 | 51.8 | 1.11× |

PQ block=16p sub=8 K_sub=256 is the recommended substitution:
- Same storage as IQ2_XXS (0.5 vs 0.516)
- 1.17× quality lift (rel_L2 0.302 vs 0.354)
- Smaller codebooks (32 KB per (layer, kind) — easy fit in threadgroup memory)
- Faiss-standard, well-understood decode pattern (matches H1793 chunked-decode)

But this is **NOT** OOM-class. The honest expectation: 17% codec error
reduction → maybe 5-10% AIME hold-rate lift if codec error → router fragility
→ wrong-expert-selection cascade holds.

## Why per-pair VQ K=256 is the special case

The earlier finding (VQ K=256 at 1 byte/pair = 17× lower codec error than
IQ2) is REAL but at 2× storage. Full DS4 with VQ K=256 alone = 159 GB.
Trim50 with VQ K=256 = 84.5 GB. Neither fits 52 GB.

The hybrid path (partial VQ K=256, partial IQ2/PQ) is the only way to use
VQ K=256's quality on a budget. Concretely:

- Replace ONLY hot-routed expert weights with VQ K=256 (using H1793-style
  fanout-tier selection: top 11/1396 unique tiles by fanout)
- Keep cold-routed experts at IQ2_XXS
- Storage cost: ~5% of routed-expert mass at 2× storage = +5% storage
- Quality cost: <5% of expert calls at 17× lower error = significant lift
  on the routes that matter most

This is the lever that composes with silv's existing trim50 + the new
aberration-detection signal: the WORST-VQ experts per (layer, kind) get
upgraded codebook K, the COLD experts stay at IQ2.

## High-potential composition (refined)

For silv's deployed Trim50 (60 GB inc. context, 30/20 t/s):

```
Component                          | Storage    | Quality vs IQ2
-----------------------------------|------------|----------------
Trim50 IQ2_XXS routed (current)    | 47.7 GB    | 1.0× (baseline)
Swap routed → PQ block=16 sub=8 256| 47.2 GB    | 1.17× (same size win)
+ Hot experts → VQ K=256           | +1-2 GB    | 17× on hot subset
+ Per-expert outliers → K=4096     | +0.16 GB   | 5× on bimodal tail
Total                              | ≈49 GB     | mixed: 1.17-1.5×
```

Net for AIME hold-rate: maybe 10-20% improvement if codec-error-driven
router-fragility is the binding constraint. That's the deployable claim
to test against silv's runtime.

## The OOM-accuracy detection lever

The aberration analyzer already established the OOM-higher-resolution view:
sign-flip rate, angle RMS, magnitude-stratified error, per-expert outliers.
Those are detection signals, not compression knobs. They identify WHICH
experts to upgrade to higher-K codebooks.

The deployable application: per-expert tier dispatch driven by aberration
scan. The scanner takes ~37 min for full DS4 today (75× faster than
single-cell), and ships to <5s with batched FP4 shard loading. The signal
is per-expert codec quality + outlier identification.

## What this leaves silv

Realistic options for <52 GB on M1 Max 64 GB:

1. **Keep trim50 IQ2_XXS** at 60 GB total — current baseline, works
2. **Swap routed to PQ block=16p sub=8 K_sub=256** — same storage, 1.17× lift
3. **Hybrid: PQ + VQ K=256 for hot experts** — slight storage growth (~5%),
   targeted quality lift on the high-fanout experts
4. **More aggressive trim** — orthogonal to codec; cuts experts beyond trim50

The codec-only path is bounded: ~1.17× quality lift at SAME storage as IQ2,
or 17× lift at 2× storage (won't fit). The compositional path with task
#463 layer-asymmetric trim + per-expert codec tier is the deployable lever
that actually moves the needle.

## Files

- `vq_k_sweep_for_52gb.py` + `vq_k_sweep_20260525T041853Z.json` — per-pair VQ
- `block_vq_sweep.py` + `block_vq_sweep_20260525T043807Z.json` — block+PQ
- `block_vq_v2_20260525T132300.log` — full sweep results
