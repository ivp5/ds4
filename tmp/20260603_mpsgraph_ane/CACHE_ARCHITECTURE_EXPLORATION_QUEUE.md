# Cache Architecture Exploration Queue — M1 Max DS4 ANE/MPSGraph

## Working Architecture Model

- M1 Max uses unified DRAM shared by CPU, GPU, ANE, media engines, and I/O; the useful question is not VRAM copy avoidance but which engine path can reuse cache-resident lines and which path forces cache/controller contention.
- CoreML/ANE does not expose public `MTLBuffer` feature inputs. Public low-copy paths are `MLMultiArray initWithDataPointer`, `MLMultiArray initWithPixelBuffer:shape:` for IOSurface-backed FP16/INT8, and output backings through `MLPredictionOptions`.
- MPSGraph consumes `MTLBuffer` through `MPSGraphTensorData`. A data-pointer `MLMultiArray` can wrap the same `MTLBuffer.contents` pointer on Apple unified memory, so ANE/CoreML and MPSGraph can be scheduled against the same hidden-state allocation.
- Engine-private caches are not equivalent to a CPU L1/L2 cache warmup story. The plausible high-value mechanisms are shared system-level cache residency, DRAM row/page locality, page-table/TLB residency, driver command scheduling, and avoiding extra CPU/CoreML staging copies.
- ANE is useful in high-B prefill because its arithmetic latency can overlap GPU work. It is not yet useful for exact D8F decode gather because raw CoreML `gather_along_axis` expands indices and is too slow.

## Evidence So Far

- `coreml_iosurface_ingress_canary.m`: CoreML uses IOSurface-backed output, and DS4-style `MTLBuffer -> IOSurface texture -> CoreML input` copy cost is only about `0.111 ms` at `B=2048,D=4096`.
- `ane_gpu_prefill_ios26.py`: synthetic CoreML palettized ANE + MLX GPU split-expert prefill at `B=2048,D=4096,N=2048,experts=6` overlapped to `1.306x`.
- `ane_mpsgraph_schedule_canary.m`: same shared input `MTLBuffer` for CoreML/ANE and MPSGraph/GPU is viable; CoreML output backing is used.
- High-B B=2048 r5 result: ANE `4.601 ms`, MPSGraph `6.230 ms`, ANE→MPSGraph inner `5.248 ms`, MPSGraph→ANE inner `10.201 ms`, concurrent `8.651 ms`, speedup `1.252x`.
- High-B B=2048 r20 result: ANE `4.561 ms`, MPSGraph `6.026 ms`, ANE→MPSGraph inner `8.066 ms`, MPSGraph→ANE inner `12.381 ms`, concurrent `5.944 ms`, speedup `1.781x`.
- Eviction bracket B=2048 r8: ANE→MPSGraph with CPU eviction after ANE lands around `9.1-9.7 ms` for 64/256/512 MiB sweeps, while no-evict ANE→MPSGraph often lands around `5-7 ms`. This supports a cache/scheduling-state effect, but the current canary has order contamination and must be counterbalanced before a production decision.
- Real H3355 shared-expert CoreML export is viable. Layer-0 B=2048 8-bit palettized shared expert measured `11.618 ms` on CoreML/ANE with IOSurface/output backing, and overlapped with same-shape MPSGraph dense work at `1.696x` speedup (`27.096 ms` serial, `15.980 ms` concurrent).
- 4-bit CoreML palettization is not fidelity-safe on the current shared-expert canary (`rms=0.0841229`, `ref_rms=0.427915`), while 8-bit is plausible (`rms=0.0050416`, `ref_rms=0.427915`). Cache/overlap work should therefore measure 8-bit shared experts first, not force 4-bit through an AIME gate.
- `ane_mpsgraph_counterbalanced_canary.m` tests the virtual-bandwidth hypothesis directly with randomized per-trial case order and `same` versus `separate` input allocations. Real H3355 layer-0 B=2048 8-bit, no eviction: same-buffer ANE→MPS p50 was faster than same-buffer MPS-only (`17.840 ms` vs `23.452 ms`), while separate-buffer ANE→MPS was slower than separate-buffer MPS-only (`19.674 ms` vs `15.075 ms`). This is the first direct evidence that the useful effect is shared allocation/cache/page/scheduler state, not generic ANE warmup.
- Counterbalanced overlap p50 speedups stayed positive across same/separate and 0/256MiB eviction (`1.495x` to `1.882x`), but eviction controls remain noisy under current load. Treat the signal as high-potential, not yet production-proof.
- `ane_d8f_routed_counterbalanced_canary.m` replaces synthetic MPSGraph dense with exact real D8F gate/up/SwiGLU/down LUT graph work. Exactness passed for L26/E165 (`bad=0`, `rms=1.45568e-05`) and selected six experts `165,0,1,2,3,4` (`bad=0`, `rms=3.30178e-05`).
- Real six-expert D8F did not inherit the synthetic same-buffer ANE→MPS warmup. Same/no-evict D8F-only p50 was `7.270 ms`, D8F-after-ANE was `7.760 ms`; separate/no-evict D8F-only was `6.914 ms`, D8F-after-ANE was `7.273 ms`. The robust production win is parallel overlap, not proven D8F cache warmup.
- Real six-expert ANE shared + exact D8F routed overlap remains valuable: same/no-evict p50 speedup `1.383x`; same/256MiB `1.613x`; separate/no-evict `1.600x`; separate/256MiB `1.355x`. This is the best current evidence for same-layer ANE/GPU prefill overlap.

## Highest-Potential Experiments

1. Counterbalanced schedule matrix: randomize or rotate the order of ANE-only, MPS-only, ANE→MPS, MPS→ANE, evict→MPS, ANE→evict→MPS, and concurrent in every trial; record per-round p50/p90, not only averages.
2. Eviction-size sweep: 0, 16, 32, 64, 128, 256, 512, 1024 MiB with CPU eviction and Metal eviction separately. If CPU and GPU eviction differ, the mechanism is not just unified cache residency.
3. Stride/locality sweep: keep B=2048 but vary input stride/tensor layout and MPSGraph access pattern. If ANE warming helps only contiguous rows, shared input-line locality is implicated.
4. Same-buffer versus separate-buffer A/B: CoreML data-pointer `MTLBuffer`, CoreML IOSurface, and separate numpy-style input. This separates shared allocation cache effects from generic ANE/GPU overlap.
5. Page/TLB warm test: run ANE on a model that only reads input with minimal compute, then MPSGraph dense matmul. If MPSGraph still speeds up, page/TLB/cache warmth is enough; if not, ANE compute path is the lever.
6. Real shared-expert CoreML export: dequant H3355 non-routed shared gate/up/down for a single layer into palettized CoreML, then overlap that ANE shared expert with existing D8F routed GPU prefill. This is the cleanest production seam because shared experts are not router-dynamic.
7. Real routed-output merge: ANE writes shared or partial-routed output into IOSurface/data-pointer backing; GPU merges with `batch_routed_out` without CPU readback. Measure whether merge serialization destroys overlap.
8. MPSGraph D8F down graph cache: pair exact MPSGraph down with ANE shared expert in the same layer and test whether ANE warming improves MPSGraph down, not only dense synthetic matmul.
9. Thermal/power guard: run short repeated blocks separated by cool-down gaps. Some observed ANE-after-MPS regressions look like engine contention or scheduler priority, not cache.
10. M1 Max cache-size inference: pointer-chase/sweep hidden-state buffers before MPSGraph and after ANE to estimate the effective shared-cache cliff for DS4-shaped tensors.
11. Real shared-expert counterbalance: repeat the schedule matrix with the 8-bit H3355 layer-0 shared CoreML model, not only synthetic palettized matmul. This separates real weight-stream/cache behavior from model-shape-only overlap.
12. Merge-buffer cache test: write ANE shared output into an output backing, run a GPU merge/add kernel immediately, then evict before merge. If eviction alone destroys merge speed, shared-output residency is a production lever.
13. Real-D8F pairing: replace the synthetic MPSGraph dense leg with the exact MPSGraph D8F selected-down and gate/up LUT executables. The production question is whether ANE shared prefill warms or overlaps the actual D8F graph, not a dense stand-in.
14. Metal eviction control: add a GPU eviction kernel over a large `MTLBuffer` with contiguous, strided, and random-ish page walks. CPU eviction proves host-visible cache/TLB sensitivity; Metal eviction separates GPU/SLC pressure from CPU-side page warming.
15. Cache cliff sweep: run the counterbalanced harness at eviction sizes `0,16,32,64,128,256,512,1024,2048 MiB` with at least 48 trials under low load, logging p50/p90/p99. The goal is to infer the effective M1 Max shared-cache/page-residency cliff for DS4-shaped buffers.
16. Trivial-CoreML page warm: compare real shared-expert ANE against a cheap CoreML identity/read model that touches the same input with minimal math. If both help MPSGraph, page/TLB/SLC warmth is enough; if only real ANE helps, scheduler or weight-stream side effects matter.
17. Microbatch interleave: test `B=256,512,1024,2048` shared-expert ANE while GPU runs routed D8F for the same layer. Same-layer shared and routed branches are dependency-parallel; cross-layer pipelining is not valid until the merge/residual is complete.
18. Command-queue/fence policy: test one shared Metal command queue versus separate queues plus explicit events/fences around input readiness and output merge. The target is overlap without accidental serialization or cache-destructive waits.
19. Merge survival canary: allocate CoreML shared output backing as a GPU-visible buffer/IOSurface, run exact D8F routed graph concurrently, then launch a Metal add/merge into the FFN output. Measure serial versus overlapped+merge; if merge erases the overlap, the architecture is not ready.
20. Route-weighted exact graph: extend the D8F routed MPSGraph canary to multiply selected expert outputs by router weights before summing. Current graph uses unit expert weights; performance shape is valid, but runtime integration needs route-weighted output.

## Production Direction

- Do not target ANE for exact D8F gather/decode yet.
- Target ANE for high-B prefill shared experts first: fixed per-layer weights, no router dynamism, natural overlap with D8F routed GPU work, and low-copy ingress is proven.
- Keep MPSGraph for exact D8F down/gateup canaries and graph-cache exploration. Treat MPSGraph + ANE overlap as a scheduling/cache optimization, not as a replacement for reaching the packed-index memory floor.
- Treat the user’s “virtual bandwidth” hypothesis as plausible but unproven: the likely win is cache/SLC/TLB/DRAM-controller residency plus free parallel compute, not ANE magically warming GPU private cache. Promotion requires counterbalanced evidence and eviction controls.
- The near-term runtime architecture should be same-layer parallelism: GPU computes routed D8F, ANE computes 8-bit shared expert from the same hidden-state allocation, GPU merges shared+routed outputs from CoreML output backing, then the normal layer dependency continues. Cache correctness matters at the input and merge buffers; violating that turns theoretical bandwidth into extra serialization.
- After the exact routed canary, prioritize overlap integration over cache folklore: cache warmth helped synthetic dense, but exact D8F routed mainly benefits from concurrent ANE/GPU execution. The next go/no-go is merge/fence cost, then route-weighted exact graph caching.
