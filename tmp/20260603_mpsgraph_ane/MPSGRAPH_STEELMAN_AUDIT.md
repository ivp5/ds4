# MPSGraph Steelman Audit — DS4 D8F on M1 Max

## Verdict

MPSGraph has not been exhausted. The current evidence is deep enough to reject the naive “standard gather reaches the packed D8F memory floor” claim, but not deep enough to reject MPSGraph as a useful organ. The defensible architecture is:

- ANE/CoreML owns high-B shared-expert prefill where fixed per-layer weights and batch arithmetic fit its native regime.
- GPU/Metal or MPSGraph owns routed D8F work.
- Same-layer shared and routed branches overlap, then merge on GPU without CPU readback.
- MPSGraph remains useful for exact D8F graph canaries and possible down-only graph-cache production, but only if its expanded-index gather path is bypassed or proven acceptable.

## Proven Boundaries

- Exact MPSGraph D8F gate/up/SwiGLU/down canaries pass with route weights and real hidden rows.
- ANE shared expert plus exact routed D8F overlap survives a GPU merge.
- Layer-matched L26 shared CoreML plus L26 routed D8F on a real hidden row still gives useful overlap+merge speedup.
- Cache warmup is not promotion-grade for exact D8F: D8F-after-ANE repeatedly fails to beat D8F-only in the clean no-evict cases.
- Standard MPSGraph LUT implementation expands packed D8F codes into `int32` `[groups, rows]` gather indices. That is O(n) over touched rows, but it is not the encoded memory floor because index traffic is inflated before execution.
- Packed-code decode is expressible in MPSGraph for both 4-bit synthetic and 12-bit VQ-D8 LUT indices using `coordinateAlongAxis`, byte gathers, bit shifts, and masks. Correctness passes, but performance does not promote: 12-bit VQ-D8 packed decode was slower than expanded gather in both measured runs.

## Steelman Lenses

- Page/Brin: one serving path with explicit cold caches; no pile of experimental overlays on the hot path.
- Buterin: local invariants and replayable gates; every runtime organ must carry shape, layer, route, and cache assumptions.
- Krzakala/Schniter/Donoho: do not tune to AIME; preserve signal under real hidden rows, router weights, and layer replications.
- Sherwood/Gabay: compute is not abstract; DRAM/SLC/TLB/fence topology is part of the algorithm.
- Thiel: discard commodity variants; keep only paths with a plausible monopoly advantage on M1 Max unified memory.
- geohot/Carmack: fewer abstractions, direct buffers, direct counters, no framework worship.
- Knuth/Tao: prove the asymptotic object first; hot work must be O(n) over touched codes/rows and avoid hidden full-model scans.

## Remaining MPSGraph Steelman Tests

1. Packed-index expression test: done. MPSGraph can decode packed 4-bit and 12-bit codes in-graph.
2. Fusion test: first result is negative. The 12-bit VQ-D8 path shrank index storage `2.666x` but ran slower than expanded gather, so the compiler is not making this a memory-floor path.
3. Down-only production test: keep gate/up on the existing best GPU/Metal path and use MPSGraph only for selected routed down where the current isolated result is strongest.
4. Executable cache footprint: compile and retain layer/expert/shape-specialized MPSGraph executables for a representative layer set; measure cold compile, warm run, and resident memory.
5. Layout sweep: test codebook `[block,k]` versus alternative table orientations, gather axis choice, and contiguous versus strided hidden-state buffers.
6. Queue/fence policy: compare one shared command queue, separate queues, and explicit event fences when paired with CoreML output backing.
7. Shape cache: test `B=1`, `B=256`, `B=512`, `B=1024`, and `B=2048` graph caches instead of assuming one prefill shape generalizes.
8. Actual-byte baseline: compare MPSGraph logical traffic to a custom Metal packed-index kernel to quantify the exact gap to the memory floor.
9. Replication gate: repeat real hidden/router rows across more prompts and layers before treating one L26 row as representative.

## Current Architecture Decision

Do not promote generic MPSGraph gather or packed-index MPSGraph decode as the final D8F runtime. The first custom Metal packed VQ-D8 baseline is already near MPSGraph expanded-gather timing while retaining packed indices, so keep MPSGraph as an exact graph oracle, overlap probe, and possible down-only cache organ while the production path moves toward:

1. cold-loaded CoreML shared-expert package cache;
2. cold-compiled routed D8F graph or fused Metal kernel cache;
3. same hidden-state backing for shared/routed branches;
4. parallel ANE shared + GPU routed launch;
5. GPU merge into the layer FFN output;
6. no CPU readback and no hot-path model/package loading.
