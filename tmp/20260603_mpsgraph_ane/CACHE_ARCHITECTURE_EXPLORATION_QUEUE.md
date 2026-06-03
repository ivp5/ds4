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

## Production Direction

- Do not target ANE for exact D8F gather/decode yet.
- Target ANE for high-B prefill shared experts first: fixed per-layer weights, no router dynamism, natural overlap with D8F routed GPU work, and low-copy ingress is proven.
- Keep MPSGraph for exact D8F down/gateup canaries and graph-cache exploration. Treat MPSGraph + ANE overlap as a scheduling/cache optimization, not as a replacement for reaching the packed-index memory floor.
