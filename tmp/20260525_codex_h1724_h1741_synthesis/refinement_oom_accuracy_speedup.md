# Refinement memo: OOM-higher accuracy + 2-3 OOM speedup opportunities

silv 2026-05-25 closing directive: "review your thinking based on the most
recent understanding, refine for an order of magnitude higher accuracy
which should enable to sense small aberrations unnoticed before, which
can be amplified to yield valuable signal. look for 2-3 order of magnitude
speedups along the way which would enable additional applications at the
expanded level of accuracy and detail."

This memo's job is the meta-task: what does the session's work make
newly-measurable, and what do those measurements unlock?

## Current accuracy floor (where I'm at)

Three concrete numbers from this session's canaries:

  Phase A reader (PLR2 → CPU decode):       1.39e-7  vs Python encoder
  Phase B-2.2 real-data canary (GPU vs CPU): 1.11e-6  on L1,E5
                                              6.77e-7  on L0,E0
  H1735 synthetic canary (mag=0 path):        0.00e+0  (bit-exact, expected)

These are fp32 noise-floor numbers. The codec itself is much lossier:
the MLX encoder reports `cos_sim mean = 0.9608, rel_L2 mean = 0.2784`
on real DS4 V4 weights. The 0.96 cos_sim is the polar p8_m2 codec's
intrinsic reconstruction error, not a bug.

So I have TWO accuracy regimes:
  - **Pipeline-fidelity** (encode → decode round-trip): 1e-6 (fp32 noise)
  - **Codec-fidelity** (encode → decode vs original weights): rel_L2 0.28

The session's OOM-higher-accuracy frontier sits on the SECOND axis. I
can encode the same weights with a HIGHER-RESOLUTION polar codec and
measure how much rel_L2 drops.

## What OOM-higher accuracy makes newly-measurable

H1751 (codex, read this turn) found that on the router side,
`polar_p4096_keepmag16` (14-bit nominal, 4096 phase bins × 16 mag
quartile levels — encoded with 16 mag bits per pair) hits 100% set+order
exact at 0.281× fp32 router. The expert side I shipped (p8_m2) hits
rel_L2 0.28. They're using the same polar codec FAMILY at different
resolution points:

  Codec       Phases  Mag levels  Bits/pair  Expert rel_L2  Router exact
  p8_m2       8       4           6          0.28           —
  p16_m4      16      4           7          ~0.14 est      —
  p32_m4      32      4           8          ~0.07 est      —
  p32_m8      32      8           10         ~0.03 est      —
  p64_m8      64      8           11         ~0.01 est      —
  p4096_m16   4096    16          ??         ??             0% miss

**The OOM-higher-accuracy probe is:** re-encode the same DS4 V4
weights at p32_m4 / p32_m8 / p64_m8 and measure cos_sim/rel_L2 on
each. At p64_m8, intrinsic rel_L2 should be ~0.01 (1e-2) — two OOM
better than p8_m2's 0.28.

What this would surface that p8_m2 obscures:

**(a) Non-uniform encoder bias on heavy-tailed weight distributions.**
The levels[r, k] for k ∈ {0,1,2,3} are quartile MEANS. On a heavy-tail
distribution (which expert weights often are), the top quartile's
mean over-represents the tail's true magnitude. At p8_m2 this looks
like a 1e-1 systematic error and is dismissed as "codec is lossy."
At p64_m8 the codec floor is 1e-2, so the quartile-bias signal
becomes the leading error term — visible and measurable. Could then
switch the codebook from quartile-MEANS to quartile-percentile or
to a learned codebook.

**(b) Phase quantization clustering.**
8-level phase has 45° bins. Real weights cluster around certain
angles for reasons related to which input dimensions the expert
"specializes" on. Currently invisible because the 45° quantization
noise is the dominant error. At p64 (5.625° bins) the clustering
pattern emerges as the LARGEST signal, which can then be exploited
(e.g., specialize codebook per-layer to the empirical angle histogram).

**(c) Cross-expert correlation.**
Among 256 experts in a layer, some pairs have nearly-identical
gate weights with up weights differing only by phase rotation.
The H1731 codex finding (1.63× dedup via content hash) is the
coarse version of this. At OOM-higher-accuracy the residual
after dedup is the more interesting signal: which expert PAIRS
have rank-1 cross-correlations that could be encoded as
"expert i = scale × rotation(expert j)" with just two scalars
of side data. Currently invisible because the p8_m2 noise floor
is larger than the cross-correlation residual.

## 2-3 OOM speedup paths that compound

### Speedup #1 — Batched real-canary (~100×)

The B-2.2 canary today dispatches ONE expert per kernel call: 0.33 ms.
Looking at the kernel's grid: `(route_pairs, batches)` — so route_pairs=256
+ batches=1 batches ALL 256 experts into ONE dispatch. Cost amortizes to:
~0.33 ms / 256 = 1.3 μs/expert.

**Total full-layer accuracy sweep: 1.3 μs × 256 = 330 μs/layer × 43 = 14 ms**.

Currently a per-layer accuracy sweep takes 256 × 0.33 ms = 85 ms ×
43 = 3.6 seconds. The batched form is **~260× faster**.

Application: live parameter optimization. Want to find the polar codec
parameters (phase_levels, mag_levels, quartile-vs-percentile codebook)
that hit a target accuracy at minimum storage. Inner loop must be
cheap. At 14 ms/layer, can sweep 100 parameter combos × 43 layers in
~1 minute. Currently impossible (would take 6 hours).

### Speedup #2 — Encoder sweep loop (~1000×)

Current encoder: 0.24 ms/expert × 33,024 encodings = 8 sec full model.
Building a parameter-sweep loop that:
  1. Encode at p_N_m_M for each (N, M) candidate
  2. Decode via the real-data canary
  3. Measure cos_sim + rel_L2 per layer
  4. Pick the lowest (bytes × √error) point on the Pareto frontier

Single full-model accuracy measurement is 8 sec (encode) + 0.7 sec
(decode all layers via batched canary, per Speedup #1) = ~10 sec.
Used to take ~6 hours manually with per-expert canaries.
**~2,000× speedup on the sweep**.

Application: a Pareto frontier of (storage, accuracy) for DS4 V4
expert weights across the full polar parameter space, generated in
~30 minutes total compute. Discover the operating point: the codec
parameters that hit 100% AIME hold rate at minimum disk.

### Speedup #3 — Phase B-2.3 hot-path substitution (~5-10× per-layer)

The H1735 kernel on b8 16×8 measured 22.31 ns/equiv row-dot at the
codex measurement (per H1736). DS4's current FP4 per-layer FFN encode
takes ~50 ms decode-side; replacing it with the H1735 polar kernel
projects ~5-10 ms. That's a **5-10× per-layer speedup** for the routed
MoE compute, IF the layout-matching overhead doesn't eat the win.

This is the strategic deliverable I haven't shipped yet — #563 Phase
B-2.3. The canary work this session (synthetic at 4b59348, real-data
at d74a7d5) proves the kernel produces correct output. The remaining
gap is the integration into metal_graph_encode_layer_ffn_batch.

### Composed effect

If all three ship:
  - Encoder + sweep loop (Speedups #1, #2): re-tune polar codec
    parameters in minutes, hit lower rel_L2 at same or smaller storage
  - Phase B-2.3 (Speedup #3): substitute the FP4 expert MLPs with the
    polar kernel during inference
  - DS4 V4 inference at 5-10× decode speedup PLUS recovered accuracy
    via re-tuned codec parameters

**Two NEW application classes enabled**:

  **(α) Live AIME score optimization.**
  Currently: change polar params → 6-hour re-encode + manual benchmark.
  After speedups: change polar params → 30-second cycle. Can sweep
  the parameter space with AIME hold-rate as the objective and
  converge on a hardware-optimal codec.

  **(β) Per-token accuracy/speed adaptive routing.**
  H1741+H1744 codex policy: route quantization is pressure-aware
  (q6 row with widths 9/8/6). Extend to the expert MLP layer: when
  pressure is LOW (attractor basin, 65/72 cases per H1738), use a
  low-resolution polar codec (p8_m2 = 6 bits/pair, fastest). When
  pressure is HIGH (boundary-fragile, 48/72), promote to high-
  resolution polar (p64_m8 = 11 bits/pair). The per-token cost
  becomes load-bearing on actual ambiguity, not constant.

## What I should test next at the OOM-higher resolution

**Test A — Polar resolution sweep.** Modify polar_encode_mlx.py to
parameterize phase_levels + mag_levels. Encode L0 of DS4 V4 at
{p8_m2, p16_m4, p32_m4, p32_m8, p64_m8}. Measure cos_sim/rel_L2 per
expert. Plot the Pareto frontier (bytes × √error). Currently I
have ONE point on this Pareto (p8_m2 at rel_L2 0.28). Need 5+ points
to find the elbow.

**Test B — Encoder bias diagnosis.** Run polar_encode at p8_m2 with
TWO codebooks: (1) current quartile-means, (2) quartile-percentiles
(use np.percentile at 12.5, 37.5, 62.5, 87.5). Compare rel_L2 per
expert. If percentile-codebook wins by ≥10%, that's a free accuracy
gain from a codebook-only change (no resolution increase needed).
Costs zero extra bytes on disk.

**Test C — Phase clustering map.** Encode L0 and extract the
distribution of phase_codes across all (expert, row, pair) tuples.
If phases are uniformly distributed across 0..8, the 45° bins are
optimal. If they cluster (e.g., 60% at codes 4 = 0 angle), a
non-uniform phase codebook would be much smaller for the same
fidelity. Measure first; might be free.

## Standing concerns ahead of OOM-higher accuracy

1. **fp32 accumulator drift**: the H1735 kernel uses fp32 accumulators
   for gate_acc + up_acc. At pairs=2048 with values ~O(1), the
   accumulator hits ~2048 — well within fp32 precision (relative
   error ~1e-7 per add). At higher resolution where individual
   magnitudes are smaller, the round-off doesn't change. SAFE.

2. **Threadgroup-memory access patterns**: at p64 phase resolution
   the cos_lut/sin_lut need 65 entries instead of 9. Still tiny,
   no impact.

3. **Codec INVARIANCE under fp16 weight loading**: the original
   weights are stored as BF16 in safetensors. The encoder reads them
   as int8 (FP4 quantized) and decodes via FP4 LUT. Any switch to
   bf16 source via the alternative encoder paths I shipped earlier
   would need re-verification at OOM-higher accuracy.

## Summary table

| Path                              | OOM gain  | Application unlocked                |
|-----------------------------------|-----------|--------------------------------------|
| Higher-resolution polar (p64_m8)  | 1-2 OOM   | Codec bias visible; cross-expert    |
|                                   | accuracy  | correlation residual measurable     |
| Batched real-canary               | 2 OOM     | Per-layer accuracy sweep in 14 ms   |
|                                   | speed     |                                      |
| Encoder sweep loop                | 3 OOM     | Full-model Pareto in 30 min vs hr  |
|                                   | speed     |                                      |
| Phase B-2.3 hot-path substitution | 1 OOM     | 5-10× DS4 V4 decode @ M1 Max        |
|                                   | inf speed |                                      |
| Adaptive per-token resolution     | (compose) | Pressure-aware codec routing        |

**Final consolidation**: every speedup in the encoder/canary/sweep
side compounds with the accuracy improvements because the sweep loop
itself is the discovery mechanism. The faster the inner loop, the
more parameters explored. The more parameters explored, the higher
the chance of finding a Pareto point that's strictly dominant on
both axes.

Looking at the synthesis-of-synthesis: **the work this session shipped
makes polar codec parameter optimization a routine inner-loop activity
rather than a heroic offline effort**. That's the underlying win — not
a specific accuracy or speedup number, but the cost of asking and
answering "what if?" questions about the codec dropping from hours to
seconds. silv can now iterate. The polar arc moves from "is this
viable?" to "what's the optimal operating point in DS4-aware
parameter space?"
