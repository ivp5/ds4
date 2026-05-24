# Codex H1724 → H1741 sequential-read synthesis (2026-05-25)

Reading scope: codex shifts H1724 (first MTL4 polar kernel execution) through
H1741 (router quant pressure-fragility prediction). Sequential, in author
order. Source: `/Users/silv/cl/tlp_codex/CODEX_SHIFTS.md` lines 4646-4744.

This memo extracts what's load-bearing for the queued DS4 inference-side
integration work (#563 / #565 / #566) so future sessions inherit findings
without re-reading 100 lines of codex shifts.

## What survived sequential read

### Hardware reality (silent-failure-mode discipline)

- **H1724**: MTL4 ArgumentTable buffers silently read as ZERO unless their
  allocations are committed/requested through `MTLResidencySet` and attached
  to the command buffer/queue. Tiny successful kernels are not evidence the
  memory model is understood — they may be reading happy-path defaults
  rather than the buffer you bound. **First test of any new MTL4 binding
  must compare against CPU reference on non-trivial input**, not just check
  "kernel returned non-zero."
- **H1728**: `MTL4MachineLearningPipelineState` cannot be built from a
  normal `kernel void` MSL function — `executableWithDeviceSelection:` is
  unrecognized for `_MTLLibrary`. ML encoder is for ML-network pipelines;
  it does NOT subsume the compute encoder for custom kernels. Stay on
  MTL4 compute path.
- **H1734**: M1 Max reports `Metal4`, `Apple7`, `Mac2`. All MTL4 compute
  primitives instantiate (queue, allocator, buffer, compute encoder, ML
  encoder objects). ML encoder is object-proven only, NOT functionally
  proven for arbitrary kernels.

### Kernel evolution (the actual shipping shape)

The shipping kernel is **H1735's tiled gate/up/down partial fusion**, NOT
H1733's bare gate/up. The path of refutation:

1. **H1724** — single-thread per packet. Correctness-first but slow (8.2 ms
   for 7776 packets).
2. **H1725** — 256-thread threadgroup per packet. 13.54× speedup (0.6 ms).
   Same accuracy.
3. **H1727** — Layout change: drop 32× hidden duplication via route-tile
   structure (1.70 GB → 384 MB resident, 4.43× win). Kernel time band
   unchanged.
4. **H1729** — Tile × row × batch dispatch. Mag/phase/levels stored once;
   hidden streams. b8 = 46.70 ns/output (1.45× over H1727). b32 = 31.81×
   layout reduction vs row-packet repetition.
5. **H1731** — Indexed code-tile dedup. 1590 unique codes vs 2592 tiles
   (1.63×). **Conditional throughput**: b2 improves, b8 REGRESSES, b32
   improves. Use indexed only under memory pressure / high reuse; direct
   layout wins at peak b8.
6. **H1733** — Fused `silu(gate)*up*route_weight` MLP packet. b32 = 2.20×
   faster than H1731 at 1.77× lower allocation. Matched decoded reference
   to 7.55e-09. **First real organ-level representation.**
7. **H1735** — Gate/up/down PARTIAL fusion (32 activation rows + real
   Q2_K down slices). b2 lost 0.62× (serialized 32 rows starved row-level
   parallelism). b8 / b32 recovered (1.07× / 1.05×) because batch
   occupancy filled the machine. **Shipping shape is TILED partial fusion,
   not monolithic.**
8. **H1737** — Two-stage materialized split with intra-pass barrier.
   Correctness held (1e-10 to 1e-09). **Performance REGRESSED**: b8 16×8
   slowed 7.40 → 9.81 ms. **Refuted**: keep monolithic tiled fusion
   unless activation reuse has multiple consumers.

### Tile policy table (H1736 / H1738)

The tile shape is a measured scheduling surface, not a constant. From
the policy run:

| Batch | Objective           | Best tile  | ns/equiv row-dot |
|-------|---------------------|------------|------------------|
| b8    | latency / memory    | 8×8        | (lowest mem)     |
| b8    | gate-up throughput  | **16×8**   | 22.31 (1.45× H1733) |
| b8    | down / balanced     | 8×64       | 8.59             |
| b32   | latency / memory    | 16×8       | —                |
| b32   | gate-up / down / bal| 32×64      | 21.03 (1.01× H1733) |

`activation_rows × down_rows`. The down-fanout is non-monotonic:
8×64 beats 8×32 and 8×16 at b8 because the extra MACs fit in occupancy
that would otherwise be idle. **Don't pick "fused expert" as a tile;
pick (batch, objective) → tile from this table at dispatch time.**

### Route-packet content (H1739)

For top-k routing, `topk(k+1)` is the load-bearing live route packet:

- selected experts
- selected logits / weights
- kth-vs-(k+1)th boundary margin
- entropy

Transfer fall: **18.29× less than copying all 256 router logits**. Capture
is exact on selected-set (216/216 cases). Max kth-margin error 2.6e-06,
max entropy error 3.8e-06.

**Why k+1**: selected-expert IDs alone are an impoverished route
certificate. One extra competitor exposes whether the token is in a
stable attractor basin (large margin) or near a fragile routing
boundary (small margin).

### Pressure semantics (H1738 / H1740 / H1741)

- **What pressure measures**: kth-margin + entropy norm. Concentrated
  attractor states (router_attractor hidden family) → 65/72 LOW pressure,
  mean margin 0.021. Near-uniform states (gaussian) → 48/72 HIGH, mean
  margin 0.0033.
- **Pressure does NOT change tile winner** (H1740). Gate-up = 16×8 in
  both high and low buckets. **Splitting hurts occupancy** — full b8
  16×8 = 22.31 ns but high subset = 29.50 ns, low subset = 37.16 ns.
- **Pressure DOES predict router-quant fragility** (H1741):
  - q6 row: high/medium/low flips 33%/12%/0% (AUC 0.857)
  - q6 tensor: 68%/26%/4.6% (AUC 0.873)
  - q4 tensor: 100%/75%/33% (AUC 0.872)
  - noise 0.03: 39%/14%/0.9% (AUC 0.869)
- **Saturation kills the signal**: q2 row flips 210/216; q2 tensor flips
  216/216. Pressure is a boundary-fragility predictor, not a rescue
  after destructive compression.
- **Operational claim**: router precision should be pressure-aware. Low-
  pressure attractor routes tolerate lower precision / cached resident
  packets; high-pressure routes need stricter precision or candidate
  expansion. **Row-wise router quant is much safer than tensor-wise at
  the same nominal bits.**

### Routing organ choice (H1730)

TinyJit top-k on GPU vs logits-to-host crossover:

| Cases/layer | Winner            | Ratio        |
|-------------|-------------------|--------------|
| 36          | logits-to-host    | 2.17× (despite 14.22× more data) |
| 384         | TinyJit top-k     | 1.10×        |
| 1536        | TinyJit top-k     | 3.11× (4.82 vs 14.96 ms) |

**Shift**: don't cargo-cult "keep routing on GPU." Route top-k is a
batch-size/lifetime organ. For tiny diagnostic batches the host wins
even with much higher transfer. Keep on GPU only when H1729-style
resident expert codes feed many hidden states.

### Semantic metadata is load-bearing (H1732)

`tile_meta_head` + `row_meta_head` hashes alone are sufficient for
metrics but INSUFFICIENT for semantic gate/up fusion. The load-bearing
state is `(layer, expert, kind, route_case, rank, route_weight)`.

For #563 integration: the polar file format (PLR2 magic) currently
carries `(layer, kind_id, n_experts)` in the header but lacks per-tile
route-case/rank context. That's fine for static expert weights but
inadequate for runtime-routed tiles. Carry route metadata in the
dispatch packet, not in the weight file.

## Integration implications for queued tasks

### #563 (integrate codex polar p8_m2 MTL4 kernel into DS4 inference)

The codex artifacts to pull from (per H1735 / H1736 / H1738):

- **Kernel**: `h1735_mtl4_gate_up_down_partial_kernel_20260525.m` —
  tiled partial-down fusion, the shipping shape.
- **Tile policy**: `h1738_ds4_tile_policy_route_pressure_20260525.py`
  — measured (batch, objective) → tile table. Port to a C lookup.
- **Route capture**: `h1739_tinygrad_live_route_pressure_20260525.py`
  — `topk(k+1)` packet format. Adapt the C router to emit this packet.
- **Resident layout**: H1729 (tile × row × batch dispatch) + H1731
  conditional dedup. Default = direct resident; enable dedup under
  memory pressure (env var or measured RSS threshold).

Integration plan (phased, env-gated like the ICB phases):

1. **Phase A — load .polar files alongside FP4**. Mmap the combined
   .polar files (PLR2 format from `polar_encode_mlx.py`). Match
   tensors to existing `weights->layer[il]` slots by (layer, kind).
   Hold both representations in memory.
2. **Phase B — opt-in dispatch for single layer**. Env var
   `DS4_POLAR_LAYERS="l1,l2,..."` selects layers to encode via polar.
   For each polar layer, call the H1735 kernel from `metal_graph_
   encode_layer_batch`. Compare output to FP4 path on first call;
   require bit-equivalence ≤ 1e-6 or refuse the substitution.
3. **Phase C — pressure-aware tier selection**. Hook the router emit
   to capture `topk(k+1)` packets. Compute pressure (kth-margin,
   entropy). Route low-pressure tokens through polar (smaller code,
   cached), high-pressure tokens through FP4 (stricter).
4. **Phase D — eliminate FP4 fallback for proven layers**. Once
   long-running confidence is established (AIME end-to-end + sympy
   verification on rescue corpus), drop the FP4 representation for
   polar-validated layers. Disk + RAM win.

Guard rails (from H1724 / H1727 / H1737):

- Every new MTL4 binding compares against CPU reference on non-trivial
  input (mid-corpus AIME, not just zero vector).
- ALL polar buffer allocations attached via `MTLResidencySet` to the
  command queue. Audit at dispatch time, fail loud if any unattached.
- Do NOT split into two-stage materialized kernels (H1737 refuted).

### #565 (polar p8_m2 down-projection canary)

Already H1735 — that IS the down-projection canary, in the codex repo.
Task is to bring the kernel into this repo's `ds4_metal.m` as a working
canary first, then integrate.

### #566 (multi-command finalize-sequence ICB)

The route packet at the finalize end of layer N feeds the dispatch at
layer N+1. If we capture H1739's `topk(k+1)` in the ICB-recorded
sequence, the replay can include both the route-finalize call AND the
next-layer's pressure-aware tile-selection call. That's where the
multi-command finalize wins.

But ICB doctrine (silv 2026-05-23, codex H1696 / sticky-hazard in
CLAUDE.md): ICB is net-loss for tiny kernels. The finalize+next-tile
sequence is two tiny kernels — likely still net-loss vs direct
dispatch. Test as opt-in env-gated only (`DS4_ICB_FINALIZE_SEQUENCE`).

## What's NOT in scope yet

- H1741's pressure-aware router quantization is at the **router**, not
  the **expert weights**. DS4 V4 Flash router is already at F16 (per
  GGUF layout: 0.1% of bulk at F16). Won't gain from polar tiering.
  This finding applies to future architectures or to a separately-quantized
  router (which IQ2_XXS does not do).
- H1730's GPU-vs-host route-top-k crossover applies to the routing
  decision, not to the expert MLP. The DS4 inference engine already
  decides routing on host CPU for small batches; this finding
  corroborates that choice but doesn't action anything.

## Standing risks

- **H1737 monolithic-vs-staged refutation may be M1-Max-specific.** The
  ANE on A17 Pro or M4 may invert the result. Don't generalize beyond
  M1 Max without re-measuring.
- **H1731 dedup is locality-thresholded.** If DS4 routing changes
  workload shape (e.g., very long contexts → fewer unique tiles per
  batch), dedup may flip from "regression at b8" to "win at b8."
  Re-measure when the routing pattern changes.
- **H1740 pressure-splitting hurts occupancy** at the H1735 tile shape.
  A different tile family (say a routing-aware fusion that NEEDS the
  pressure split for correctness) might invert this. The shift's
  practical advice — keep pressure on packets but not on tile scheduler
  — assumes the H1735 family of kernels.

## Cross-link to local artifacts already shipped

- `polar_encode_mlx.py` (commit 8402a5e): full DS4 V4 encoding in 8 s at
  0.24 ms/expert. Produces PLR2 combined files. Phase A in the
  integration plan above LOADS these files.
- `polar_encode_bulk.py` (commit 4a529d0): numpy fallback. Useful for
  re-encoding on machines without MLX (Linux servers).
- `analyzers/polar_encode_safetensors.py` + `analyzers/polar_encode_expert.py`:
  per-expert encoders for diagnostic / single-cell probes.
- `ds4_metal.m`: contains H1724-style canary entry points
  (`ds4_gpu_mtl4_polar_dot_canary`, `_tile_canary`, `_tile_real`,
  `_fused_canary`). All pipelines created with
  `supportIndirectCommandBuffers=YES`.

## Next action (immediate)

Begin **Phase A** of the #563 integration: add a polar loader that
mmaps PLR2 files and attaches their data buffers via MTLResidencySet,
HOLDING them alongside the existing FP4 weight tensors. No dispatch
change yet. This lets us measure load-time + memory cost separately
from compute-time gain.

## Addendum: H1742 / H1743 / H1744 (read 2026-05-25, after H1741)

### H1742 — Route `topk` is a boundary certificate, not a hard set

Quantized DS4 router perturbations can be rescued by expanding perturbed
candidates until they contain the clean top-6 set. Width sweep tested
on widths {6, 7, 8, 10, 12, 16, 24, 32, 48, 64} per the artifact
`h1742_ds4_router_candidate_expansion_rescue_20260525.py`. Pressure
buckets: high (small kth-margin / high entropy), medium, low.

| Profile     | Width policy (high/med/low) | Mean | Multiple of top-6 | Savings vs uniform |
|-------------|------------------------------|------|-------------------|--------------------|
| q6 row      | 9 / 8 / 6                   | 7.236 | 1.206×           | 19.6% vs width 9   |
| q6 tensor   | 11 / 12 / 7                 | 9.264 | 1.544×           | 22.8% vs width 12  |
| q4 row      | 32 / 14 / 11                | —    | large            | (cold path)        |
| q4 tensor   | 91 / 47 / 14                | —    | very large       | (cold path)        |
| q3          | rejected for hot path       | —    | —                | —                  |

**Shift**: expert IDs alone are an over-compressed route trace. Keep
selected experts + boundary pressure + optional extra candidates;
rerank/verify at higher precision only when the boundary is fragile.

### H1743 — M1 Max Metal4 surface is compute-first, not ML-encoder-first

Re-mapped on macOS 26.5 / SDK 26.4. Five separate probe binaries
narrow the boundary:

**Runtime-proven productive (compute path):**
- `MTL4CommandQueue`, `MTL4CommandAllocator`, `MTL4CommandBuffer`
- `MTL4Compiler` (compiled an `add_one` kernel end-to-end, `bad=0`)
- `MTL4ComputePipelineDescriptor`, `MTL4ComputeCommandEncoder`
- `MTL4ArgumentTable` (raw `gpuAddress` binding works)
- `MTL4CommitFeedback`, `MTL4CounterHeap`
- `MTLResidencySet` (already known critical per H1724)
- Placement sparse buffers: `supportsPlacementSparse=yes`,
  tile sizes 16/64/256 KiB pages verified; allocation returns a
  valid `gpuAddress`.

**Not productive (ML encoder path):**
- `MTL4MachineLearningCommandEncoder` exists, but a no-pipeline /
  noop probe CRASHES at `end_ml_encoder`.
- `MTLTensor` selectors exist, but direct tensor canary CRASHES at
  `tensorSizeAndAlignWithDescriptor:`.
- Plus H1728 already established normal `kernel void` MSL functions
  are NOT valid ML pipeline inputs.

**Selectors enumerated** (h1743 selector probe): 10 of 10 respond on
the device class, but "responds" ≠ "executes without crashing." The
selector_probe binary is the cheap-existence check; the canary
binaries are the actual functional probes.

**Shift**: stay on raw `MTLBuffer` + arg tables + residency + counters
+ placement sparse + transparent compute. Don't block on opaque ML/ANE
packaging while the inspectable compute path is already proven.

### H1744 — Compression policy follows route boundary geometry

H1742's measurement turned into an executable C policy table:
`h1744_ds4_route_pressure_policy_table_20260525.inc`. The struct
`ds4_route_policy_t` carries `{profile, bits, axis_tensor, target_pct,
width_high, width_medium, width_low}` for 8 quant profiles
(q3/q4/q6/q8 × row/tensor). For DS4 V4 Flash inference, the deployable
target is **q6 row** with widths `{9, 8, 6}` per pressure bucket.

The `.inc` file is pre-shipped (codex side) and pulled into our repo
this turn at `analyzers/ds4_route_policy.inc` for direct C inclusion.
Per-token flow at inference time:

  1. Router emits topk(k+1) packet per H1739 (selected + kth-margin
     + entropy + weights).
  2. Compute pressure bucket from kth-margin/entropy (thresholds TBD).
  3. Look up `width_{high,medium,low}` from policy table.
  4. Run approximate q6-row router → extract `width` candidates.
  5. Rerank top-`width` at higher precision (the rerank cost is the
     1.206× top-6 multiplier).
  6. Emit route certificate = (selected_6, rerank_logits, pressure).

**Shift**: a global quantization setting is the wrong abstraction.
Low-pressure attractor tokens stay narrow (width 6 = exact top-6);
fragile boundary tokens buy a small candidate halo (width 9).
Row-wise router quant much safer than tensor-wise at same nominal
bits (H1741 corroboration).

### Composite integration implications

The polar (#563 / #565) work is for **expert MLP weights** (gate / up /
down). The H1742-1744 work is for the **router** (the layer that
decides which experts fire). They compose:

  pressure-aware router (q6-row + width 9/8/6) →
    selected experts (per token) →
      polar gate / up / down (per expert) →
        decoded activations →
          high-precision rerank ONLY if pressure says fragile

When pressure says LOW (router_attractor case from H1738: 65/72 cases),
we run narrow router AND can also skip the rerank — fastest path.
When pressure says HIGH (gaussian case from H1738: 48/72), we widen
both the router and the rerank — slower but boundary-correct path.

The next end-to-end deliverable is therefore: read PLR2 polar weights
(Phase A ✓ shipped commit aad0608) + dispatch H1735 tiled kernel
(Phase B pending) + emit topk(k+1) at the router (#566 prereq) +
gate the rerank by pressure (post-#566 win). Per-token cost moves
from "constant" to "load-bearing on actual ambiguity," which is what
H1738 first measured and H1744 turned into a policy table.

### H1745 — Q6 approximate pressure can select its own safe width

H1745 closes the main implementation leak H1744 left open: the H1744
policy was generated assuming access to clean-precision pressure (full
fp16 router logits), but at inference time the runtime only has the q6
approximate logits. H1745 ran the rescue probe with pressure computed
from q6-approximate entropy + q6 kth-vs-next margin (no clean-side
peek) and got:

  - widths 9/8/6 chosen per pressure bucket — same as H1744
  - 0 under-allocation cases
  - 100% clean-top-6 containment after rerank
  - 1,553 candidates vs H1744's 1,563 (slightly fewer)

**Why the approximate pressure still works**: q6 row preserves the
boundary ordering with 88.89% top-6 set exact, 66.20% order exact,
98.15% top-1 exact. The pressure signal (entropy norm, kth-margin) is
a robust summary statistic that doesn't depend on perfect ranks —
small router perturbations don't move pressure across buckets unless
the boundary is actually fragile, and that's the case we wanted to
catch anyway.

**Budget shift made operational by H1745** (no clean-side dependency):

  - High-precision router logit compute: 55,296 → 1,553 (**35.61× fewer**)
  - Total byte traffic vs fp16 full scan: q6-full-scan + fp16-rerank
    = 0.403× = **2.48× lower router weight traffic**

**Integration consequence**: the H1744 `.inc` policy table (shipped
this turn at `analyzers/ds4_route_policy.inc`) is correct AS-IS. No
need to thread clean-pressure access through the pipeline. The
runtime hot path:

  q6 router scan (cheap)
    → q6 pressure (entropy_q6, kth_margin_q6)
      → width lookup from policy table
        → top-`width` candidates
          → fp16 rerank on the small candidate set
            → route certificate emitted to expert dispatcher

**Next proof codex pre-declared**: MTL4 q6-route certificate kernel
that produces the certificate in one dispatch. That's the q6 router
analog of H1735's polar tiled fusion for the expert MLPs — and
together they form the deployable inference kernel for DS4 V4 on M1
Max with the polar + pressure-aware-router stack.

Memo scope is now H1724 → H1745.

### H1746 — Q6 route certificate runs as real MTL4 compute

Moves H1745 from Python proof to executable M1 Max MTL4 compute kernel
over raw buffers — the companion to H1735 on the router side. Source
files: `h1746_mtl4_q6_route_certificate_kernel_20260525.m` (kernel +
dispatch) and `h1746_ds4_mtl4_q6_route_certificate_20260525.py`
(harness over 216 real DS4 route cases).

**Kernel shape**: one threadgroup per route case computes q6 row-wise
logits for all 256 experts, selects top-9 candidates, then exact-reranks
only the pressure-width subset (high=9, medium=8, low=6 per the H1744
policy table that codex shipped earlier this batch).

**Verification on 216 real DS4 route cases**:

  - Approximate top-N containment: **100%**
  - Rerank set exactness:           **100%**
  - Rerank order exactness:         **100%**
  - Misses:                         **0**

**MTL4 commit feedback**: 6.356 ms GPU time for all 216 cases. Width
distribution: 6 → 110 cases, 8 → 61, 9 → 45 (matches the H1738 pressure
split — most tokens are in a low-pressure attractor).

**Compute-path doctrine corroborated**: the transparent compute path
is sufficient. No opaque `MTL4MachineLearningCommandEncoder` / `MTLTensor`
needed for the route certificate. H1727 / H1734 / H1743 picture survives.

**Remaining memory residency frontier**:
  - Current canary: full fp32 router resident for exact rerank simplicity.
  - Staged q6 + scales: 0.250× fp32 router bytes.
  - Ideal packed q6 + scales: 0.188× fp32 router bytes.

**Next memory proof codex pre-declared**: remove full fp32 residency
via (a) candidate-row packets (only the top-`width` rows extracted at
high precision), (b) placement sparse pages (`MTLSparsePageSize16/64/256`
per H1743), or (c) a route-aware exact-row cache (LRU on the
boundary-fragile rows).

**Composite picture with H1735** (this session's expert-side canary
shipped commit 4b59348):

  Router side  (H1746 q6 + rerank certificate)  → emit (selected_6,
                                                       rerank_logits,
                                                       pressure)
  Expert side  (H1735 gate/up/down tiled fusion) → per (batch, route_pair)
                                                   produce 64 down_rows
                                                   in one dispatch

Both kernels:
  - MTL4 compute path only (no ML encoder / no MTLTensor)
  - MTLResidencySet for all buffers (H1724 silent-zero hazard)
  - 256 threads/threadgroup, MTL4ArgumentTable raw gpuAddress binding
  - Dispatch sized by route_pairs + batches; tile policy from H1738

Together they form the DEPLOYABLE INFERENCE STACK for DS4 V4 Flash on
M1 Max with polar + pressure-aware-router. The Phase B integration
plan (real-data hookup) is now structurally complete on the codex
side; my Phase B-1 loader work in this repo brings the polar weights
into the engine; B-2 will dispatch H1735; a future #566 / topk(k+1)
hookup brings H1746 into the router slot.

Memo scope now H1724 → H1746.
