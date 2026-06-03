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
