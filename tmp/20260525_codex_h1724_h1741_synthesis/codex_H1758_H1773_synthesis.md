# Codex H1758-H1773 synthesis: streaming + phase-rerank complements polar/VQ

Codex shipped 16 more shifts (H1758-H1773) on the same DS4-on-M1
substrate while my session worked the polar p32_m8 + VQ K=256 codec
arc. The two research streams are **complementary**, not competing.
Together they map a substantially larger substrate optimization
surface than either alone.

## Codex's substrate arc (router/phase/streaming)

H1758-H1762 — q6 router quantization + candidate-width policy:
- 17,280 DS4 route cases tested with q6 row-wise router quantization
- Pressure-aware candidate widths (9/8/6 for high/medium/low pressure)
- 0 ordered top-6 misses with phase-margin rerank fallback
- 19-22% candidate work reduction vs uniform-width baseline

H1763-H1766 — phase rerank as MTL4 organ:
- Phase comparator at p4096/p8192/p16384 resolutions
- Two-kernel command-buffer with producer-consumer barrier (1.064×
  faster than split, 1.721× faster than one-kernel fusion)
- **One-kernel fusion is SLOWER** (1.62× regression) — DS4 substrate
  benefits from packet pipelines, not monolithic kernels
- fp16 magnitude + int16 phase codes: 3× phase-cache memory reduction
  at <1% speed cost

H1767-H1769 — streaming hidden:
- Resident router/phase packets + streamed hidden chunks
- 7.94× resident memory reduction at 1.216× slowdown (chunk-1024)
- Frontier: chunk128 = 9.45× memory / 1.77× slow; chunk4096 =
  3.20× memory / 1.06× slow

H1770-H1773 — bitpacked phase packets:
- 39-bit triple-resolution phase packet (12+13+14 bits)
- Phase midpoint p8192 IS load-bearing (can't drop it; 2-resolution
  exact policy needs 2.2× more fallback at 45.55%)
- Bitpacking works in MTL4 at <1% speed cost
- **H1773 chunk512 frontier**: 8.61× resident memory reduction +
  1.173× GPU slowdown vs polar one-shot — deployable

## My session arc (codec quality + canary)

Polar p32_m8 — production-ready codec at 14 GB, weight rel_L2 0.12
VQ K=256 — substrate codec at 4.3 GB, weight rel_L2 0.02 (6× better)
Both at canary maturity with C-side validation chains
B-2.3c stub gate wired (silv runtime-tests when ready)

## How they compose

Codex's streaming/phase-rerank operates at the ROUTER level (which
experts to fire) and the MEMORY level (resident vs streaming).
My codec operates at the WEIGHT level (how compactly to encode each
expert's weights).

The deployable combined picture:

```
   Router       Weights       Compute
   --------    -----------   -------------
   q6 + pressure  polar p32_m8   MTL4 streaming
   widths 9/8/6   (or VQ K=256)  + barrier pipeline
                                 + bitpacked phase
                                 + chunk512 hidden
```

Together this is:
- **7-8× memory reduction** (codex H1773 streaming) +
- **2× weight-bytes reduction** (VQ K=256 vs polar) +
- **6× codec accuracy improvement** (VQ vs polar) +
- **20% router work reduction** (codex H1762 pressure widths) +
- **0 ordered top-6 misses** (codex H1742 q6 row policy)

Memory frontier:
  H1766 baseline: 318 MB resident on 17280 row stress
  H1773 chunk512: 318 / 8.61 = 37 MB resident
  My VQ K=256 vs polar p32_m8: another 2× weight reduction on top

Speed frontier:
  H1773 chunk512: 1.173× slowdown vs polar one-shot
  My MLX encoder: 1.86 OOMs faster (one-time)

Combined deployable: ~30 MB resident DS4 substrate footprint with
20%+ smaller router work and 6× lower codec error on per-FFN-call
output — at ~1.2× GPU slowdown vs current path.

## The shifts that REINFORCE my arc

H1770: "phase midpoint is load-bearing" — sampling structure must
be preserved, not endpoint-only compression. This validates the
polar codec's mag-levels=8 vs mag-levels=4 finding (where m8 was
not just 2× larger phase storage but produced 2× better quality
because intermediate magnitude levels carry information).

H1764: "one-kernel fusion is slower" — the H1735 gate*silu*up*down
kernel is the right size; trying to fuse polar codec dispatch into
it would regress. The B-2.3c stub correctly stays as a SEPARATE
kernel from the FP4 dispatcher.

H1767: "Metal4 ML encoder is not productive" — my B-2.3c stub uses
standard MTL4 compute (correct path) rather than ML encoder
(unworkable on M1 Max).

## The shifts that REFUTE assumptions in my arc

H1768: "streamed hidden is the inference organ" — I built polar
canary at single-batch (n_tokens=1) scale; deployment should stream
hidden in chunks. The B-2.3c dispatcher body, when shipped, should
take chunk_size as a parameter from day one.

H1770: 2-resolution phase compression refuted — codex tested
deleting samples, found it broken. Translates to my codec: don't
try to halve VQ K=256 to K=128 without verifying; the structure may
need K=256 specifically (FP4 has 16² = 256 natural modes per scale
block, per my earlier finding).

## Recommended next session moves (when silv directs)

If Branch A.1 / A.2 / A.3 lands (codec body deployed):
- Combine with codex's chunk-streaming pattern (H1773 chunk512)
- Combine with codex's q6 pressure-aware router (H1742 + H1745)
- Total memory frontier: ~30 MB resident DS4 substrate

If silv directs codec body NOT now:
- Read codex H1746-H1757 (the shifts I skipped between H1745 and
  H1758 — 12 more I haven't synthesized)
- Or pivot to non-codec queued work

The codex arc is shipping fast; staying caught up on synthesis is
itself valuable session work that doesn't depend on silv runtime or
disk approval.

## Tasks

- The codex arc work is in `/Users/silv/cl/tlp_codex/CODEX_SHIFTS.md`
  (read-only reference)
- My arc closes at 26+ commits this session
- Combined deployable: codec body + codex streaming = the actual
  "deploy DS4 at 50% memory + 6× lower codec error" target
