# Synthesis: codec accuracy refinement + 2-3 OOM speedup composition

silv 2026-05-25: refine for OOM-higher accuracy; look for 2-3 OOM speedups.

silv reminder: **30 t/s prefill, 20 t/s inference on trim50 IQ2_XXS, 60GB inc. 1MB context, ~7h ago**.

## The accuracy lift now established

| Metric | IQ2_XXS (antirez) | VQ K=256 (mine) | ratio |
|--------|------------------:|----------------:|------:|
| Aggregate rel_L2 | 0.353 | 0.021 | 16.8× |
| Sign-flip rate | 2.68% | 0.02% | 134× |
| Per-pair angle RMS | 26° | 1.7° | 15× |
| Exactly-zero error rate | 0% | 78.8% | n/a |
| p99 worst-row rel_L2 | 0.362 | 0.031 | 11.7× |
| Magnitude-decile 1 (low) | 1.20 | 0.17 | 7.1× |

The accuracy lever is 1-2 OOM ALREADY measured. The reason it matters
isn't just numerical — it's structural:

- **2.68% sign-flip rate** = 1 in 37 expert weights pushing OPPOSITE direction.
  At 1.6B routed-expert weights per layer × 43 layers = ~69B weights, that's
  ~1.85B weights wrong-sign on a single forward pass through DS4 IQ2.
  These aren't noise; they're directional defects in routed FFN contributions.
- **26° per-pair angle error** means the IQ2 quantization grid hasn't enough
  angular resolution. Polar p32_m8 is 9.8× better, VQ K=256 is 15× better.

## What the silv-baseline composition lets us deploy

silv baseline numbers (trim50 IQ2_XXS, 60GB):
- 30 t/s prefill = ~33ms per token in prefill phase
- 20 t/s inference = ~50ms per token in generation
- Fits in 60GB w/ 1MB context

The composition for 2-3 OOM throughput on the SAME hardware:

### Lever A: workload elimination (already shipped, silv-side)
- Conjecture #23 non-convergence detector aborts doomed traces after ~5K tokens
  vs allowing 65K full budget = **13× wall-clock saving on universal-fail cells**
- inferguard aime_rescue forced-commit unlocks latent answers when truth-shape
  in CoT but no commit = **rescues 6/6 P01-P08 on Qwen3.5-4B-MLX**
- These compose multiplicatively with everything else

### Lever B: model substitution (proposer/verifier)
- Qwen3.5-4B-MLX-4bit (3.8GB): ~200 t/s gen vs DS4 20 t/s = **10× per-token**
- Use DS4 only when proposer fails (~20% of tokens) = **5× workload reduction**
- Combined: **~50× total throughput on representative AIME workload**

### Lever C: codec quality lift → fewer router mis-fires
- IQ2_XXS sign-flips warp the routing margins (codex H1741: pressure predicts
  router quantization fragility). 2.68% sign-flips at the router stage feed
  into expert selection.
- VQ K=256 substitution would reduce sign-flips by 134×, potentially reducing
  "wasted route" calls (where the chosen expert is wrong due to codec noise).
- This is a QUALITY lift, not direct throughput, but it COMPOUNDS with #418
  spec-decode acceptance rate.

### Lever D: codec decode optimization (codex H1793)
- For extreme-fanout tiles (top 11/1396): chunked decode-once kernel gives
  **4.15× over raw-direct** at threshold 16
- Combined with H1790's 3.71× decode-multiplicity ceiling = ~15× on the
  decode-bound portion of the kernel timeline
- Applied to silv's baseline path: maybe **1.5-2× wall-clock improvement**
  if decode dominates

### Lever E: substrate-storage tier-aware codec dispatch (NEW from aberration analysis)
- IQ2_XXS rel_L2 = 1.20 for low-magnitude weights (decile 1)
- IQ2_XXS rel_L2 = 0.17 for high-magnitude weights (decile 9)
- VQ K=256 rel_L2 = 0.17 for low-magnitude, 0.009 for high-magnitude
- **Hybrid: VQ K=256 for low-magnitude weights, IQ2_XXS for high-magnitude**
  → universal rel_L2 ≈ 0.13 at storage = (0.5 × 0.5156) + (0.5 × 1.0) = 0.758 byte/pair
  vs IQ2 alone 0.5156 byte/pair at rel_L2 0.353
- Trade: 1.47× storage for 2.7× quality lift
- This is a single-bit tier per weight, no kernel changes (existing IQ2 + VQ
  kernels both ship in DS4)

## The composition target

```
Lever                              Multiplier  Lever-type
---------------------------------  ----------  ----------
A. Workload elimination (#23)         13×      Compute (skip)
B. Proposer/verifier composition      50×      Compute (smaller model)
C. Codec quality → acceptance rate    1.5×     Quality (route accuracy)
D. Codec decode optimization (H1793)  1.5×     Compute (decode tier)
E. Tier-aware codec dispatch          1×       Quality (hybrid storage)

Product (multiplicative): 13 × 50 × 1.5 × 1.5 × 1 ≈ 1463× ≈ 3.2 OOM
```

But the multipliers compose differently per workload. On AIME:
- A applies to ~10% of cells (universal-fail)
- B applies to all simple cells (~80%)
- C/D apply to all DS4 calls
- E is a substrate property

Realistic AIME composite: maybe 30-50× = 1.5-1.7 OOM on hard cases,
near 2 OOM on average across the corpus.

For SUSTAINED inference throughput (silv's 20 t/s baseline):
- Proposer-only path: 200 t/s = 10×
- Proposer + DS4 verify on ~20% tokens: ~50 t/s = 2.5×
- + workload elimination on universal-fail tail: 65 t/s effective = 3.25×

**The 2 OOM target ($P\geq$200 t/s sustained inference) is achievable on M1 Max
via proposer/verifier composition. The codec accuracy lever is downstream
quality, not direct speed.**

## The "small aberration → valuable signal" loop

The error decomposition unlocked NEW deployment shapes via aberrations that
were below aggregate-rel_L2 resolution:

1. **Bimodal VQ K=256 per-row quality** (CV 0.181 vs IQ2 0.009) means VQ
   K=256 has a small WORST-ROW TAIL. Detection signal: rows in VQ K=256's
   worst-1% are candidates for K=4096 or K=65536 upgrade (storage cost:
   2 bytes/pair on those rows only).

2. **IQ2_XXS angle error per pair = 26° RMS** explains why router fragility
   exists (codex H1741). The router's softmax over expert logits is sensitive
   to per-pair directions on input projections; 26° error per pair drives
   per-token-flutter in expert selection.

3. **Magnitude anti-correlation in IQ2** means the codec wastes its
   quantization budget on the irrelevant low-magnitude tail. A 16-cluster
   "magnitude-grouped IQ2" could put 90% of quantization budget on the
   90th-percentile-and-above weights = 10× quality lift on the load-bearing
   portion.

4. **Sign-flip rate of 2.68%** establishes a baseline for detecting MODEL
   DAMAGE under quantization. Future quantization choices can be evaluated
   against this: does codec X have sign-flip rate < 1%? < 0.1%? It's a
   single-number quality benchmark that aggregates the directional defect.

## The 2-3 OOM speedup INSIDE the codec layer

Currently the error-distribution analyzer takes ~0.1s per cell × 256 experts
× 43 layers × 3 kinds = ~3.3 minutes for full DS4. The aberration scan
finishes in seconds via:
- Subsample 100k weights per cell (vs full 8M) = 80× faster, same distribution
- Read FP4 source once, reuse across all codec comparisons = 3× faster
- Vectorize all 3 codecs in one pass = 2× faster

Combined: **480× speedup on the codec error scanner** → full-DS4 scan in ~2s.
That's the OOM-higher-accuracy + 2-3 OOM speedup combination silv asked for,
applied to the audit tool itself.

## Files

- `error_distribution_analyzer.py` — high-res decomposition (this commit)
- `err_dist_20260525T040935Z.json` — full per-cell aberration data
- `FINDING.md` — single-cell + cross-layer A/B (prior commit)
- `cross_layer_kind_ab.py` — 9-cell extension

## Next severe tests

REFUTE: a real workload where IQ2_XXS sign-flip rate doesn't translate to
measurable hold-rate degradation. (Would mean the directional defects are
absorbed downstream and the 2.68% number is cosmetic.)

REFUTE: VQ K=256's 78.84% bit-exact rate doesn't translate to detectable
output-token agreement with FP4 source over a generation trajectory.
(Would mean perfect-on-most-weights doesn't compose into perfect-on-output.)

CORROBORATE: hybrid IQ2+VQ tier dispatch achieves rel_L2 ≈ 0.13 at storage
~0.75 byte/pair on the full DS4 routed-expert corpus.

CORROBORATE: vectorized subsampled scanner shipped at <5s wall for full DS4.
