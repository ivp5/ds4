# Architecture Simplification Journal — 2026-06-04

Append-only transaction log. Scope: IVP5 DS4 MPSGraph/ANE runtime architecture, code compactness, and data-structure discipline.

## 2026-06-04T00:42 JST — Review lenses installed

- Request translated into concrete lenses, not outbound consultation: Page/Brin = remove accidental complexity from the serving path; Buterin = make invariants explicit and locally checkable; Krzakala/Schniter/Donoho = preserve signal and avoid overfit-to-test; Sherwood/Gabay = keep a physical model of memory traffic; Thiel = remove non-monopoly/commodity paths; geohot/Carmack = fewer layers, direct measurement, no framework worship; Knuth/Tao = prove the algorithmic shape and keep hot paths O(1) per scalar / O(n) per touched tensor.
- Constraint installed: no code path should need more than O(n) over the touched rows/blocks; no hidden O(layer × expert × full-model) work in hot decode; model/package loading must be cold-path only.
- Initial architecture finding: production `ds4_mpsgraph.m` has one legitimate core abstraction, `ds4_mpsgraph_build_lut`, but duplicated index preparation and cleanup around down/gateup canaries make future MPSGraph experiments harder to audit.
- Chosen first patch: compact duplicated selected-down cleanup and reuse the generic D8F index preparation routine so index layout has one owner.

## 2026-06-04T00:44 JST — First compile feedback

- Command: `make ds4_mpsgraph.o`.
- Result: translation unit compiled, but warned on stale `groups` local left behind after replacing the bespoke down-index loop with `ds4_mpsgraph_prepare_indices`.
- Action: removed the stale local; this keeps the compactness patch warning-clean instead of accepting harmless residue.

## 2026-06-04T00:45 JST — Cache architecture and MPSGraph steelman update

- Read `coreml_shared_cache_probe.m`, its six-layer cache log, `CACHE_ARCHITECTURE_EXPLORATION_QUEUE.md`, and the fidelity journal tail.
- Finding: all-layer shared CoreML cache looks like a ~1.7 GiB resident overlay, not a model-fit blocker; the real architectural problem is avoiding ~2.4 s per-layer lazy loads in the token path.
- Code action: tightened `coreml_shared_cache_probe.m` so feature cardinality is explicit and zero-input sizing is O(rank) over shape dimensions instead of hard-coded rank 2.
- Documentation action: added `MPSGRAPH_STEELMAN_AUDIT.md` and updated the cache/fidelity ledgers with the exact boundary: MPSGraph has been explored enough to reject naive expanded gather as memory-floor, but not enough to reject packed-index/fusion/down-only/executable-cache variants.

## 2026-06-04T00:46 JST — Probe validation

- Command: recompiled `coreml_shared_cache_probe.m` with `cc -O3 -Wall -Wextra -fobjc-arc -framework Foundation -framework CoreML`.
- Command: ran one real L26 package through the stricter probe and saved `coreml_shared_cache_probe_l26_rankcheck_20260604T004600.log`.
- Result: rank/feature invariants passed; L26 package loaded and predicted with `compile_ms=46.952`, `load_ms=2449.682`, `predict_ms=19.432`, and `delta_mib=79.23`.

## 2026-06-04T00:47 JST — Exact routed canary compactness patch

- Read the main timing body of `ane_d8f_routed_counterbalanced_canary.m`.
- Finding: the concurrent ANE+D8F launch appeared twice with identical dispatch-group mechanics, once without merge and once with merge. That duplicated the fence/queue policy and made future queue experiments more error-prone.
- Code action: extracted `run_concurrent_ane_d8f_ms`, keeping the same O(1) scheduling operation per timed case and the same O(n) D8F/CoreML work underneath.

## 2026-06-04T00:48 JST — Exact routed canary validation

- Command: recompiled `ane_d8f_routed_counterbalanced_canary.m` with `ds4_d8f_reader.c` and the CoreML/Metal/MPSGraph frameworks.
- Command: ran one L26/E165 smoke with the layer-matched L26 shared CoreML package and saved `ane_d8f_routed_refactor_smoke_l26e165_same_e0_20260604T004800.log`.
- Result: exactness still passed (`bad=0`, `rms=1.45568e-05`), CoreML output backing remained used, and both `concurrent` and `concurrent_then_merge` cases executed through the new helper.

## 2026-06-04T00:49 JST — Architecture review artifact

- Added `ARCHITECTURE_REVIEW.md` to consolidate the current architecture decision instead of leaving it distributed across timing logs.
- Decision captured: H3355/D8F remains active; ANE is for shared high-B prefill overlap; MPSGraph remains an oracle/probe/down-only candidate until packed-index and executable-cache tests settle it.
- Complexity rule captured: hot path O(n) over touched tensor data, O(1) over unrelated layers/packs, and no hot lazy CoreML loads.

## 2026-06-04T00:59 JST — VQ-D8 LUT and packed-index impact

- Built `mpsgraph_packed_index_probe.m` to test the remaining MPSGraph steelman: in-graph packed-code decode instead of host-expanded `int32` index materialization.
- Result: MPSGraph can express the packed decode. Correctness passed for both 4-bit and 12-bit VQ-D8 shapes with `bad=0`.
- Result: MPSGraph does not make it fast. 12-bit VQ-D8 index storage shrank `2.666x`, but packed decode ran slower than expanded gather (`766.167 us` vs `668.612 us` in one run; `1408.538 us` vs `724.956 us` in the clean rerun).
- Retested real H3355 L26 VQ-D8 LUT canaries through `ds4`: down E165 `808.583 us`, down selected-six `1903.469 us`, gate/up E165 `1187.917 us`, gate/up selected-six `2871.323 us`; all passed `bad=0`.
- Architecture update: direct packed-index MPSGraph is not the routed memory-floor path. The next compact high-probability route is a custom Metal packed VQ-D8 LUT kernel baseline, while MPSGraph remains useful as exact oracle and overlap probe.

## 2026-06-04T01:05 JST — Metal packed LUT baseline

- Added `metal_vqd8_lut_packed_probe.m` as the direct custom-Metal counterpart to the MPSGraph packed-index probe.
- Finding: one simdgroup per row was too serial (`1725.340 us/op`), but four simdgroups per row with a threadgroup reduction reached `730.958 us/op` with `bad=0`.
- Architecture update: the packed-index memory-floor path is custom Metal, not MPSGraph. The next useful patch is to feed real D8F records/codebooks into this kernel and compare against `ds4_mpsgraph` down E165 `808.583 us/op`.

## 2026-06-04T01:16 JST — Real D8F down baseline

- Added `metal_vqd8_real_d8f_down_probe.m` and fed it the real H3355 L26/E165 down record.
- Correctness: all real-record runs passed `bad=0`; after fixing the input seed to match `ds4_mpsgraph`, sample reference is `0.174939`.
- Performance: specialized packed Metal is noisy but in contact with the right wall: `880-1435 us/op` in comparable r20/r50 runs versus same-window MPSGraph expanded r50 `748.555 us/op`.
- Decision: single-expert packed Metal is not yet enough; continue by fusing selected-six experts/route weights or improving memory/coalescing. It remains the right path because it preserves packed indices and removes host-expanded gather materialization.

## 2026-06-04T01:23 JST — MPSGraph/ANE low-level API pass

- Local SDK is Xcode macOS/iPhoneOS 26.5. Relevant primary APIs: `MPSGraphCompilationDescriptor`, `MPSGraphExecutableExecutionDescriptor`, `MTLSharedEvent`, `MLModelConfiguration.optimizationHints`, and `MLComputePlan`.
- MPSGraph direct device control is narrow: public `MPSGraphDevice` is Metal-only. The usable loopholes are compile descriptors, reduced-precision flags, async executable calls, shared events, MPSGraph packages, and MTLTensor aliasing in macOS 16/iOS 19 headers; prior project notes still mark MTLTensor/MTL4-ML as brittle.
- Added real canary gates in `ds4_mpsgraph.m`: `DS4_MPSGRAPH_COMPILE_MODE` and `DS4_MPSGRAPH_ASYNC_BATCH`. Defaults stay conservative; the async gate is the high-value path.
- Synthetic async batching with shared events showed the clearest MPSGraph gain: for 12-bit VQ-D8 rows=4096 r50, expanded sync `735.787 us` to async batch `398.767 us`, packed sync `1421.571 us` to async batch `695.074 us`.
- Real H3355 selected-six async batching is also useful: down r50 `1609.519 -> 1081.396 us/op`, gate/up r50 `1719.212 -> 1242.824 us/op`, all `bad=0`.
- CoreML `MLComputePlan` confirms actual ANE placement for shared packages: L0 b2048 8-bit reports `ane_preferred=9/21` ops and `ane_supported=9/21`, with operators `const:9,identity:1,ios18.matmul:3,ios18.silu:1,ios19.constexpr_lut_to_dense:3,ios19.maximum:1,ios19.minimum:2,ios19.mul:1`.
- Architecture update: stop treating ANE and MPSGraph as one opaque offload. CoreML owns ANE placement; MPSGraph owns GPU graph execution and queue/fence shape. The overlap organ should use CoreML output backings plus MPSGraph async-batch/event fences, then merge on GPU.
