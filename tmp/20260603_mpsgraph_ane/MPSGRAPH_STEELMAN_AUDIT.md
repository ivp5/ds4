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

## 2026-06-04T01:23 JST — deeper low-level API findings

- Header audit: local SDK 26.5 exposes `MPSGraphCompilationDescriptor` with `optimizationLevel`, `waitForCompilationCompletion`, `disableTypeInference`, and macOS/iOS 26 `reducedPrecisionFastMath`; `MPSGraphExecutableExecutionDescriptor` supports `waitUntilCompleted`, completion/scheduled handlers, and `MTLSharedEvent` wait/signal at `MPSGraphExecutionStageCompleted`.
- Header audit: `MPSGraphDevice` is publicly Metal-only. MPSGraph may internally place work under optimization level 1, but there is no public ANE selector comparable to CoreML `MLComputeUnitsCPUAndNeuralEngine`.
- Descriptor canary: compile modes are noisy and should remain an env-gated experiment. They can affect one shape, but did not produce stable universal wins across real down/gateup canaries.
- Execution canary: shared-event async batching is stable enough to keep. Synthetic 12-bit VQ-D8 r50 showed expanded `735.787 -> 398.767 us/op` and packed `1421.571 -> 695.074 us/op` when enqueueing runs asynchronously and waiting once on an `MTLSharedEvent`.
- Real canary: H3355 L26 selected-six MPSGraph r50 improved down `1609.519 -> 1081.396 us/op` and gate/up `1719.212 -> 1242.824 us/op`, both `bad=0`, using `DS4_MPSGRAPH_ASYNC_BATCH=1`.
- CoreML plan canary: `MLComputePlan` on `shared_l0_b2048_8bit` reports `ane_preferred=9`, `ane_supported=9`, `unknown_usage=12`, proving the ANE path is live for the compiled shared package rather than inferred from timing alone.
- Steelman conclusion: MPSGraph's deepest useful lever for this project is not in-graph packed decode; it is scheduling/fence control. Use it to batch/overlap GPU graph work while CoreML handles ANE shared-prefill/high-B work.

## 2026-06-04T01:34 JST — private API permutation audit

- Runtime selector discovery found real private ANE-facing MPSGraph selectors: `MPSGraphDevice.ANEDevice`, `MPSGraphCompilationDescriptor.enableDevicePlacement`, `preferredDevice`, `allowedComputeDevices`, `enableANEFWToFWSignal`, `enableANELateLatch`, `enableANECHWRankPromotion`, and execution flags such as `disableANECaching`, `disableANEFallback`, `encodeANESync`, and `encodeANEDisableSharedEvents`.
- Guarded synthetic 12-bit VQ-D8 tests showed these are callable but not a clear win. `private_ane_device` compiled and ran, but did not outperform default async batching; `private_prefer2` changed numerics slightly and was much slower, so the private device enum values are not safe to infer casually.
- CoreML private config is more informative than MPSGraph private placement. `e5rtComputeDeviceTypeMask=1/3` forced CPU placement (`ane_preferred=0`, `cpu_preferred=9`) and proved the ANE package speed comes from actual ANE placement, not merely model shape or cache artifacts.
- Private `aneExecutionPriority` is dangerous: naive string values `high`, `low`, and `realtime` throw uncaught `NSInvalidArgumentException`. Keep it out unless valid tokens are discovered from a primary source or runtime-verified with exception containment.
- Steelman conclusion: undocumented APIs are useful as falsification instruments, not current production levers. The best tested path remains CoreML public ANE placement + MPSGraph async/shared-event GPU scheduling + packed Metal for memory-floor routed D8F.
