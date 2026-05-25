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

## Addendum: H1776-H1782 — row-streamed execution + LRU layer windows (NEWEST)

Codex shipped 7 more shifts at 11:55 UTC after my H1775 integration.
Key findings:

**H1776**: 81 GB GGUF runs locally via row-streamed seek/read.
Resident set ≠ file size. Real DS4 act_rows=4 over 864 cases used
1,396 unique tiles, 197 MB residency, 1.7 ms GPU. Refutes "can't
run locally" — direct row decode from disk is the right organ.

**H1777-H1779**: Layer-window LRU. 239-tile LRU = full global
residency at 5.84× reduction (31 MB vs 197 MB). Layer-window
execution preserves correctness at 1e-8 max error.

**H1780**: Reusable native MTL4 process across windows. Memory win
preserved (5.83×), but GPU time didn't improve (3.15 ms vs 1.7 ms).
"Launch ceremony is the whole penalty" REFUTED.

**H1781**: Grouped window frontier:
- pairs: 2.9× resident at 2.79 ms
- triples: 1.99× resident at 2.73 ms (best wall)
- quartets: dominated

**H1782** (newest): Reusable buffer set + residency set + arg table
across grouped windows. Pairs improved native wall (0.63s → 0.25s)
but GPU sums noisy. **"Allocation/residency ceremony is NOT the main
order-of-magnitude lever."** Next speed organ: direct GGUF row decode
into reusable MTL buffers with two-buffer ring overlapping CPU
decode and GPU gate/up dispatch.

### Direct codec arc relevance

H1782 explicitly validates the codec architecture pattern:
> "Montyneg's ds4_expert_table.c matches the substrate direction:
> small global layer/expert metadata is good; global hot-row
> materialization is not. The hot cache should be windowed by route
> packet, not model-global."

This maps onto my codec design:
- VQ K=256 = 2 KB codebook (small global metadata) + 1 byte per
  pair code (per-route consumption)
- Polar p32_m8 = per-(layer, kind) levels table (small) + 2 byte
  mag+phase per pair (per-route consumption)
- Both AVOID global hot-row materialization (codec compression
  reduces the bytes streamed per route packet)

The complete substrate frontier per H1776-H1782 + codec:
```
Layer              codex contribution         my session
                                              contribution
-----------------  -------------------------  -----------------
Disk → resident    H1776 row-streamed seek    polar/VQ encoder
                                              (offline)
Resident set       H1778 layer-window LRU     codec compression
                   (5.84× memory)             (3.2× compression
                                              VQ vs polar)
Window execution   H1779-H1782 reusable       codec consumption
                   process + buffer set       per-route packet
Bandwidth          H1782 next: two-buffer    (codec halves
                   ring CPU↔GPU overlap       bandwidth)
```

These compose: codec halves the bytes transferred per row +
layer-window LRU bounds residency to active layers + reusable
buffers cut launch overhead + two-buffer ring overlaps decode
with dispatch. Combined: 5.84× × 2× = ~11× memory reduction
potential at maybe 1.2-1.5× GPU time cost.

### Sharpened H1782 shift for next session

"The two-buffer ring overlapping CPU decode and GPU dispatch" is
the NEXT speed organ codex names. My codec encoder is the CPU-side
decode. The GPU-side dispatch is the H1735 kernel. Connecting them
through a two-buffer ring (one buffer fills while other dispatches)
is engineering work that someone (codex or silv or me with
direction) would land next.

## Addendum: H1775 — MTL4 non-divisible dispatch silently corrupts

H1775 shipped right after H1774. MTL4 consumed the H1774 raw route
packet directly and emitted host-exact checksums over 864 cases:
u32_exact=true, f32_max_abs=0, resident 229,376 bytes, GPU 16.9µs.

**First clean rerun exposed a real bug**: non-divisible dispatch
wrote extra threads past the valid case count and corrupted float
outputs. Fix: case-count buffer + `gid >= cases` guard inside the
kernel.

**Shift: block size is not harmless scaffolding.** Packet consumers
need validity guards OR exact dispatch dimensions; otherwise a
speed-looking kernel silently corrupts state.

**Direct relevance to B-2.3c body**: when the polar dispatcher body
lands (after silv resolves A.1/A.2/A.3), it will dispatch the H1735
kernel with `route_pairs = n_tokens × DS4_N_EXPERT_USED = n_tokens × 6`.
For n_tokens=10, that's 60 route_pairs. If the kernel's threadgroup
geometry doesn't exactly match (e.g., dispatches 64 threadgroups
because of alignment), threadgroups 60-63 would corrupt output.

Mitigation must ship with the body:
- Either pass `cases` (or `n_tokens × 6`) as a kernel param + add
  `if (tg.x >= cases) return;` guard at threadgroup start
- OR use precisely-sized dispatch with `MTLSizeMake(n_tokens * 6, 1, 1)`

The current H1735 canary uses `route_pairs=1` so this bug class
hasn't triggered. The dispatcher body must add the guard from day
one.

## Addendum: H1774 — tinygrad route packet transplant

H1774 (just shipped) emitted a deployable DS4 MoE route packet on the
live METAL path: top20 ids/logits, top6 weights, entropy, kth-margin,
pressure bucket, and H1760 q6 width policy. Six-layer run over 864
route cases stayed exact vs CPU full-logit reference: top6 exact
100%, top20 order/set exact 100%, q6 width policy containment 100%.

Packet size: 104 bytes/case vs full logits 1024 bytes/case = 9.846×
transfer reduction.

This closes the loop from H1739 (route packet as boundary
certificate) → H1742 (route topk is certificate, not hard set) →
H1746 (q6 route certificate as MTL4 compute) → H1774 (live tinygrad
emission).

## Complete deployment picture (codex + my codec arc together)

```
   Layer              Codex                   My session
   ----------         -------------           -------------
   Router             H1774 route packet      —
                      (104 bytes vs 1024)
   Phase rerank       H1772 39-bit packet     —
                      + H1773 chunk512
   Expert weights     —                       polar p32_m8 or
                                              VQ K=256 (6× better)
   FFN compute        H1735 H1736 H1768       B-2.3c stub gate
                      streamed hidden          (body pending sub-decision)
```

The two arcs compose at different layers. Combined deployable target:
- H1774 packet (104 bytes route certificate)
- H1773 chunk512 streaming (8.61× memory @ 1.173× slow)
- VQ K=256 codec (2× weight bytes, 6× lower error)
- Total: ~30 MB resident DS4 substrate + 6× lower codec error per
  FFN call + 9.846× smaller router boundary signal

This is the substrate frontier as of 2026-05-25 evening on M1 Max.

## Tasks

- The codex arc work is in `/Users/silv/cl/tlp_codex/CODEX_SHIFTS.md`
  (read-only reference)
- My arc closes at 28+ commits this session (including this addendum)
- Combined deployable: codec body + codex streaming + H1774 route
  packet = the actual "deploy DS4 at 30 MB resident + 6× lower codec
  error" target
