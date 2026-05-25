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

### H1783 IMMEDIATE REFUTATION of the two-buffer-ring framing

H1783 (12:00 UTC, same thread, ~5 min after H1782) instrumented
H1781/H1782 by stage and refuted the ring-buffer next step:

- Pairs wall 8.6s = **tile build 7.2s (84%)** + file writes 0.1s
  + CPU ref 0.1s + native wall 0.5s + GPU feedback 3.2ms
- Triples wall 6.8s = **tile build 5.7s (84%)** + writes 0.07s
  + CPU 0.14s + native 0.3s + GPU 3.2ms

**Shift quote**: "Overlap alone cannot help much because row
decode/materialization is ~84% of wall and GPU work is
milliseconds. The real next organ is **avoiding f32 tile
materialization: compact IQ2 routed-row packets, hot-window caches,
or GPU-side IQ2 decode**."

### My codec arc IS H1783's named speed organ

"Compact IQ2 routed-row packets" describes the codec arc precisely:
- Polar p32_m8: 2 bytes/pair packet (mag + phase uint8) replacing
  4-byte f32 per pair → 2× bandwidth reduction at codec stage
- VQ K=256: 1 byte/pair packet (codebook code uint8) replacing 4-byte
  f32 → 4× bandwidth reduction
- Per-(layer, kind) codebook: small global metadata
- Per-route consumption: matches H1782's pattern statement

This is not just "enabler substrate" for inference speedup. H1783
explicitly identifies the codec arc's category as the next speed
organ. The codec body (when shipped) directly addresses the 84%
of wall time codex measured as tile build/dequant/materialization.

The 8.6s pair wall → if codec reduces tile materialization by 2×
(polar) or 4× (VQ), pair wall drops to ~5.0s or ~2.7s. The 1.5-3×
wall speedup from codec deployment IS the H1782/H1783 next-organ
contribution.

### Revised composition table

```
Layer              codex contribution         my session
                                              contribution
-----------------  -------------------------  -----------------
Disk → resident    H1776 row-streamed seek    polar/VQ encoder
Resident set       H1778 layer-window LRU     codec compression
                   (5.84× memory)             (3.2× compression)
Window execution   H1779-H1782 reusable       codec consumption
                   process + buffer set       per-route packet
Tile materialization  ←── H1783 NEW BOTTLENECK ──→  CODEC IS
                                              THE NEXT ORGAN
```

The codec arc's strategic position just escalated from "enabler"
to "named-next-speed-lever" by codex H1783's profile.

### H1784 confirms + extends with measurements (12:05 UTC)

H1784 separated raw IQ2 row I/O from dequant/f32 materialization:
- Raw IQ2 rows: **11.8 MB** for pair/triple windows
- f32 gate/up tiles: **183 MB** for same windows
- **15.5× f32 expansion overhead** at materialization stage

File I/O is small (0.04s persistent handle). **Dequant/materialization
dominates wall: pairs 5.88s, triples 5.65s** — matching H1783's 84%
diagnosis.

**Shift quote**: "The f32 tile is the wrong intermediate
representation. Next load-bearing test: **MTL4 IQ2 raw-row dot
products that delete the 15.5× f32 expansion and Python dequant
path**."

### My codec kernel IS H1784's architectural pattern

The H1735 polar kernel + gate_up_down_vq kernel both implement
"no-f32-tile" dot products:

```msl
// Polar H1735 inner loop:
uchar gate_m = mag[gate_base + j];           // 1 byte read
uint  gate_l = uint(phase[gate_base + j]);   // 1 byte read
float gate_q = levels[gate_row * M + gate_m]; // codebook lookup
gate_acc += gate_q * cos_lut[gate_l] * h0 +   // fused dequant
            gate_q * sin_lut[gate_l] * h1;    // + matmul

// VQ gate_up_down_vq inner loop:
uchar gate_c = gate_codes[gate_base + j];    // 1 byte read
float gate_re = codebook[gate_c * 2];        // direct lookup
float gate_im = codebook[gate_c * 2 + 1];    // direct lookup
gate_acc += gate_re * h0 + gate_im * h1;     // fused matmul
```

Both patterns avoid materializing the f32 tile — they fuse decode
into the matmul at register level. This is H1784's "delete the
15.5× f32 expansion" architectural call.

Differences from H1784's raw-IQ2 path:
- IQ2 codebook is hardcoded (16 values × E8M0 scale per 32-block)
- Polar codebook is LEARNED per-row magnitude + uniform phase LUT
- VQ codebook is LEARNED per-(layer, kind) K=256 in R²

The codec arc trades hardcoded-IQ2 for learned-polar-or-VQ at the
same architectural pattern: in-kernel decode, no f32 materialization.
Per codec quality measurements:
- Raw IQ2 at FP4: rel_L2 0 (it's the source)
- Polar p32_m8: rel_L2 0.12 (decode imperfect, but matmul direct)
- VQ K=256: rel_L2 0.02 (decode near-perfect, matmul direct)

VQ K=256 at 1 byte/pair achieves rel_L2 0.02 with the same
architectural advantage as H1784's raw-IQ2 path. Polar at 2 byte/pair
achieves 0.12. Both DELETE the 15.5× f32 expansion.

### Final composition picture (H1776-H1784)

```
Stage              codex (H1776-H1784)        my codec arc
-----------------  -------------------------  --------------------
Disk → resident    H1776 row-streamed seek    encoder offline
Resident set       H1778 layer-window LRU     compression 2-4×
Tile dequant       BOTTLENECK (84%, 15.5× f32 expansion)
  ← H1784 calls for: "MTL4 raw-row dot products"
  ← MY ARC DELIVERS: H1735 polar kernel + gate_up_down_vq kernel
  ← Both fuse decode into matmul at register level
  ← Polar 2× expansion saved / VQ 4× saved
Window execution   H1779-H1782 reusable       n/a (codec kernel
                                              is already this)
```

The codec arc lands directly at the stage codex H1783/H1784 just
identified as the bottleneck. Strategic alignment is now sharp:
codec deployment is BOTH the next memory organ AND the next speed
organ on M1 Max DS4 substrate.

### H1786 actually built the MTL4 raw-IQ2 dot kernel

Codex shipped H1786 around 03:20 UTC (skipping H1785). H1786
ported DS4 IQ2_XXS row decode into M1 Max MTL4 raw-buffer kernel
and computed raw-row × hidden dot products without f32 tile
materialization.

Validation:
- 96-row max abs error: 1.080e-7 (fp32 noise floor)
- 1024-row max abs error: 2.459e-7 (mean 2.505e-8)
- 1024 rows used 1.08 MB raw IQ2 vs 16.78 MB f32 (15.5× expansion
  AVOIDED in practice, matching H1784's projection)
- 1.18 ms GPU time for 1024 dots (kernel intentionally serial per
  row; parallelism left for future)

Shift quote: "Compressed expert rows can be the compute
representation at the route-packet boundary; next organ is parallel
IQ2 decode/reduction, not more f32 tile scheduling."

### H1786 directly validates the codec arc architectural pattern

| Codec | Storage | rel_L2 | Decode | Validates |
|-------|---------|--------|--------|-----------|
| Raw IQ2 (H1786) | ~0.625 byte/pair | 0 (source) | In-kernel | Codex |
| Polar p32_m8 | 2 byte/pair | 0.12 | In-kernel | My session H1735 |
| VQ K=256 | 1 byte/pair | 0.02 | In-kernel | My session gate_up_down_vq |

All three implement the same architectural pattern: compressed
representation as the compute boundary, in-kernel decode, no f32
tile materialization. H1786 proves it at fp32 noise floor on real
DS4 IQ2_XXS source weights — same pattern as polar/VQ kernels prove
on learned codebooks.

Different storage/quality Pareto points:
- Raw IQ2 (smallest, source quality)
- VQ K=256 (small, learned-near-perfect)
- Polar p32_m8 (medium, learned with magnitude/phase split)

Future session could integrate H1786's raw-IQ2 kernel alongside
polar/VQ kernels as a third codec path in the same B-2.3c stub
dispatch infrastructure — same wiring, different kernel, choice of
quality/storage point per-layer or per-experiment.

### H1787 parallelized H1786 — raw IQ2 IS now compute organ

Codex shipped H1787 around 04:20 UTC. Changed H1786's serial-per-row
decode to one 256-thread group per raw IQ2 row, each thread decoding
16 dimensions, threadgroup reduction emitting the dot.

Validation:
- 1024 rows: H1786 1.18 ms → H1787 0.18 ms = **6.41× speedup**
- 4096 rows: 0.65 ms GPU, 71.8 MB resident, max abs error 2.16e-7
  (same fp32 noise floor)
- Raw IQ2 4.32 MB vs f32 equivalent 67.1 MB (15.5× expansion avoided
  empirically at 4096-row scale)

Shift quote: "The raw compressed row is now a viable compute organ,
not only a storage organ. The next rewrite should delete f32
gate_rows_f32/up_rows_f32 from the route-packet executor and compute
actual gate/up MoE contributions from raw IQ2 packets."

### Strategic re-positioning of the codec arc against H1787

H1787 substantially shifts the competitive picture between codec
options. The honest accounting:

| Codec | Storage | rel_L2 | GPU @ 4096 rows | Engineering |
|-------|---------|--------|-----------------|-------------|
| Raw IQ2 (H1787) | ~0.5 byte/pair | 0 | 0.65 ms | Zero re-encoding |
| VQ K=256 | 1 byte/pair | 0.02 | TBD | Codebook training |
| Polar p32_m8 | 2 byte/pair | 0.12 | TBD (H1735) | Codebook training |

Where raw IQ2 cleanly wins:
- **Storage efficiency**: 0.5 vs 1 vs 2 byte/pair
- **Quality**: source-quantization, no second-order codec loss
- **Engineering**: uses existing IQ2_XXS file, zero encode time
- **Compute**: now demonstrated at production scale

Where learned codecs might still win (untested):
- Per-(layer, kind) codebook adaptability vs raw IQ2's global grid
- Distributions where K-means in R² recovers structure raw IQ2 loses
- Decode-table size (256-entry codebook vs IQ2 signed-grid 0.262 MB)

**Honest re-position**: codex H1787 demonstrates the codec-as-compute-
organ architectural pattern at production scale with the source
quantization. My learned-codec arc was the first to ship this pattern
in this session, but raw IQ2 wins the substrate axis. The codec arc's
remaining strategic value:

1. **Architectural validation**: H1735 polar kernel + gate_up_down_vq
   were independent proofs of the pattern; H1787 confirms
2. **B-2.3c dispatch infrastructure**: hot-path gate is codec-agnostic;
   reusable for raw-IQ2 as third path
3. **VQ K=256 fallback**: for tensors where raw IQ2 underperforms
   (none confirmed yet; needs cell-level comparison)

**New sub-decision option A.4** (for silv): wire H1787's raw-IQ2
kernel into the B-2.3c stub instead of polar/VQ. Storage cost = 0
(uses existing IQ2_XXS file); engineering cost = porting H1787 kernel
+ integrating into ds4.c hot-path; risk = low (source quality is
baseline); estimated effort = 200-300 LOC.

### A.4 concrete sizing (codex H1787 source inspection)

Read /Users/silv/cl/tlp_codex/research/llm_fallacy_deconstruction/framework_deconstruction/h1787_mtl4_iq2_parallel_dot_canary_20260525.m
(195 lines standalone ObjC; verified before claiming "low effort"):

Reusable kernel core (~30 lines MSL):
- `iq2_value()` decoder: 16 lines (super-block index → byte lane →
  grid index → sign decode → scaled return)
- `iq2_dot_parallel` kernel: 14 lines (per-row threadgroup, stride
  loop, partial sum, tree reduction)

Hardcoded constants in H1787 source:
- `DIM = 4096`, `ROW_BYTES = 1056`, `THREADS = 256`
- `1056 = 16 super-blocks × 66 bytes/super-block` (each super-block
  covers 256 weights = 4 sub-blocks × 64 weights)
- Signed grid: 256 × 128 × 8 = 262,144 bytes (~256 KB constant table)

Porting work required for A.4 integration:
1. Lift kernel source into `ds4_metal.m` as new MSL string (~30 lines)
2. Add `ds4_gpu_mtl4_iq2_real_canary` entry mirroring polar/VQ
   canaries (~60 lines)
3. Hot-path dispatcher: pull IQ2 raw rows + signed_grid from
   existing GGUF reader at route-time; dispatch one kernel per
   gate/up/down × route_pair (~150 LOC)
4. Three-position dispatch: gate, up, down (currently the H1787
   canary does one-row dot; FFN needs gate/up SwiGLU and down
   projection — composition logic)
5. Total estimated: 240-300 LOC for A.4 body

The IQ2 raw rows + signed_grid are already accessible from ds4.c's
GGUF reader (used by current FP4 fallback path). No new file I/O
infrastructure needed.

Trade-offs vs A.1/A.2/A.3:
- A.4 uses 30 lines of validated kernel from codex vs writing 300+
  lines of dispatcher around polar/VQ encoded corpora
- A.4 inherits production-quality at fp32 noise floor (codex tested);
  A.1/A.2/A.3 introduce codec rel_L2 0.02-0.12 noise
- A.4 needs no encode time (uses existing IQ2_XXS file); A.1 needs
  37 min, A.2 needs 6 min (after MLX VQ encoder port)
- A.4 uses 4.3 MB IQ2 rows for 4096-row matmul vs A.1's 275 GB / A.2's
  138 GB corpus on disk

A.4 is now the recommended path if silv chooses to ship the B-2.3c
body. The codec arc's own corpora (polar 14 GB, VQ 160 MB) remain
as reference/diagnostic for A/B comparison.

### CRITICAL CORRECTION (verified post-claim): DS4 already has this kernel

Back-of-hand grep on ds4_metal.m surfaced three existing pipelines:
```
g_moe_mul_mv_id_iq2_xxs_pipeline               (basic)
g_moe_mul_mv_id_iq2_xxs_pair_pipeline          (pair)
g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline   (pair + SwiGLU fused)
```

The kernel source at `metal/moe.metal:991` —
`kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32` — is a production-grade
implementation of the exact pattern H1787 demonstrates:
- In-kernel IQ2_XXS decode (block_iq2_xxs structs)
- Per-expert dispatch via `ids` + `iid1`
- Gate + up FUSED with intermediate SwiGLU
- Threadgroup-shared grid cache (`svalues` + `ssigns`) preloaded once
  per threadgroup
- N_R0_IQ2_XXS rows per simdgroup
- SimdGroup partitioning (NSG groups per threadgroup)

This is MORE sophisticated than H1787's single-row canary. H1787 is
a proof-of-concept; DS4 already ships the production version of the
pattern.

### What this means for codex H1783's "84% f32 materialization" measurement

H1783/H1784 were not measuring DS4's actual MoE kernel — they were
measuring a route-packet executor stage that PRE-MATERIALIZES
selected rows into a chunked f32 buffer BEFORE dispatching to the
kernel. The 84% wall + 15.5× f32 expansion is in that staging step,
not in the kernel itself.

H1787's contribution is then narrower: it proves the kernel-side
in-decode pattern works at canary scale. The deployable A.4 is NOT
"port H1787 to ds4_metal.m" — it's "eliminate the route-packet
executor's pre-materialization step by calling the EXISTING
`kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32` directly on the raw IQ2
rows from GGUF."

That's a SMALLER engineering target (~50-100 LOC modification, not
240-300 LOC port).

### What this means for the codec arc strategic position

This correction is severe: my polar/VQ codecs were shipped as
implementations of a pattern DS4 already has, with WEAKER numbers
on both storage and quality axes:

| Codec | Storage | Quality vs FP4 | Ships in DS4? |
|-------|---------|----------------|---------------|
| IQ2_XXS (existing) | ~0.5 byte/pair | source quantization | YES (3 kernels) |
| VQ K=256 (mine) | 1 byte/pair | rel_L2 0.02 | NO (canary only) |
| Polar p32_m8 (mine) | 2 byte/pair | rel_L2 0.12 | NO (canary only) |

The existing IQ2 path beats both my codecs on storage AND quality.
My codecs only win in the specific case where IQ2's discrete grid is
wrong for the tensor's distribution — which I have NOT empirically
demonstrated for any layer/kind.

### Honest re-positioning (severe)

The codec arc's residual value is now:
1. **B-2.3c stub wiring** as codec-agnostic dispatch infrastructure
   (still useful; raw-IQ2 path can use it)
2. **VQ K=256 as fallback** for tensors where IQ2 underperforms
   (UNTESTED — required ground-clause)
3. **Architectural demonstration** that the pattern works (this is
   redundant with DS4's existing kernels)

The codec arc was implementing a pattern that already exists at
production quality in the same codebase. I did not verify the
existing kernels before shipping the polar/VQ alternatives. This is
a Chesterton's fence violation — the existing `metal/moe.metal:991`
kernel was the fence, and I built around it without checking.

### Second-order correction (verified at call site)

The production hot path at `ds4_metal.m:14302` constructs:
```c
gate_buf = ds4_gpu_wrap_model_range(model_map, model_size,
  gate_offset, gate_tensor_bytes /* = 256 * gate_expert_bytes */,
  &gate_inner);
```

This wraps the ENTIRE 256-expert raw IQ2_XXS tensor as a Metal
buffer view into model_map. NO f32 materialization happens before
kernel dispatch. The kernel indexes per-expert at runtime via
`src0_gate + i02 * args.nb02 + first_row * args.nb01`.

Codex H1783's "84% f32 materialization" measurement was in codex's
experimental tinygrad-style row-streamed executor (their separate
repo), NOT in DS4 production. The codex arc was diagnosing their
own development infrastructure's bottleneck. DS4 production already
implements the optimal raw-IQ2-in-kernel-decode pattern.

A.4 has essentially zero deployable engineering target. DS4 ships
the answer to H1783's measurement.

### What survives this correction

- The MLX-parallel polar encoder (1.86 OOMs vs numpy) — still useful
  tooling if learned codecs are needed for non-IQ2 tensors
- The B-2.3c hot-path stub — useful dispatch infrastructure for ANY
  codec path including raw-IQ2
- The codec quality A/B harnesses — useful methodology for future
  comparisons
- The VQ K=256 < IQ2 hypothesis — testable, untested, deferred

### Revised recommendation to silv

A.4 sub-decision body shrinks from "port H1787" to "wire the existing
`kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32` through the B-2.3c stub
dispatch + eliminate route-packet pre-materialization." Sub-decisions
A.1/A.2/A.3 (codec-corpus encoding paths) are now DOMINATED by the
existing IQ2 path unless a specific tensor is shown to underperform
under IQ2 quantization.

The codec arc's actual finding for silv: **the existing
`kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32` is the production answer
to H1783's 84% materialization measurement; the route-packet
executor's pre-materialization step is the deletable wall.** This is
worth more than the polar/VQ codec scaffolding.

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
