# IVP5 DS4 Architecture Review — 2026-06-04

## Target Shape

The clean runtime architecture is a small set of cold-owned organs:

1. H3355/D8F routed pack reader owns packed codes, codebooks, route weights, and layer/expert metadata.
2. GPU routed path owns exact routed D8F gate/up/down work, either through custom Metal at the packed-index memory floor or through MPSGraph only where its graph cache proves faster.
3. CoreML/ANE shared path owns fixed per-layer shared experts in high-B prefill, with packages compiled/loaded outside the hot token path.
4. Scheduler owns same-layer overlap: shared expert on ANE and routed experts on GPU consume the same hidden-state backing, then a GPU merge writes the FFN output.
5. Journals/logs own proof: every promoted path needs exactness, real hidden/router rows, p50/p90 timing, and cache/fence evidence.

## Current Decisions

- Keep H3355 as the active SOTA pack family; do not regress into obsolete VQB2-style exploration unless a new test proves it beats H3355 on both fidelity and runtime.
- Keep MPSGraph as an exact graph oracle, cache/scheduling probe, and possible down-only organ. Do not call expanded `int32` gather the memory floor.
- Promote ANE only for shared-expert high-B prefill and overlap, not direct routed D8F gather.
- Treat AIME as an outcome test, not an optimization target. The low-level gates are exactness against reference, real hidden/router replication, and no hot-path load/compile.

## Complexity Discipline

- Hot path must be O(n) over touched blocks/rows/tensors and O(1) over unrelated layers/packs.
- Cold path may scan packages/layers, but must expose resident footprint and load/compile time.
- No hidden full-model scans in token decode; no lazy `2.4 s` CoreML model loads inside generation.
- Expanded MPSGraph gather remains O(n), but with inflated constants from materialized `int32` indices; direct packed-index MPSGraph decode is correct but slower; packed-index Metal now has a first viable baseline near expanded-gather speed.

## Simplifications Applied

- `ds4_mpsgraph.m`: down-path index layout now reuses the generic D8F index-preparation owner; duplicated down cleanup is centralized.
- `coreml_shared_cache_probe.m`: feature cardinality and shape rank are explicit; zeroing is O(rank) for shape analysis and O(n) for tensor bytes.
- `ane_d8f_routed_counterbalanced_canary.m`: duplicated concurrent ANE+D8F dispatch/fence code is centralized in `run_concurrent_ane_d8f_ms`.

## Figure-Lens Findings

- Page/Brin: the architecture needs one serving story, not a directory full of equal-looking overlays.
- Buterin: runtime packages need typed invariants: layer, expert set, route weights, shape, backing, and compile/cache status.
- Krzakala/Schniter/Donoho: avoid overfitting to AIME; real hidden rows and router weights are better low-level proxies.
- Sherwood/Gabay: cache, SLC, DRAM, and fences are physical parts of the computation, not post-hoc explanations.
- Thiel: discard commodity paths; keep the M1-specific unified-memory/ANE-GPU overlap path because it is the non-obvious advantage.
- geohot/Carmack: if MPSGraph cannot express packed-index decode without bad intermediates, write the Metal kernel.
- Knuth/Tao: prove the asymptotic object first; exact D8F work is linear in touched blocks, but materialization constants decide whether it reaches the memory floor.

## Next Compacting Targets

1. Packed-index MPSGraph probe: first pass rejects direct in-graph packed-code decode as the memory-floor route; it is correct for 12-bit VQ-D8 but slower than expanded gather.
2. Runtime sidecar scaffold: one default-off owner for CoreML model cache, D8F graph/kernel cache, hidden backing, parallel launch, and merge.
3. Package cache manifest: record layer package path, compile URL, resident footprint, load state, and validation status in one typed table.
4. Canary harness library: if more ANE/MPSGraph canaries are added, extract shared CoreML feature/backing helpers instead of copy-pasting per file.
5. Selected-six Metal fusion: extend the real packed down kernel from one H3355 expert to selected-six with route weights, then compare against MPSGraph selected-six and the full ANE+D8F overlap canary.

## Low-Level API Architecture Update

- MPSGraph should be treated as a GPU graph scheduler with useful compile and execution descriptors, not as the primary packed-index memory-floor engine.
- CoreML is the public ANE surface. `MLComputePlan` is now the placement verifier; `MLModelConfiguration.optimizationHints` is the specialization lever.
- The highest-probability overlap structure is: CoreML ANE shared branch with output backing, MPSGraph routed branch enqueued through async/shared-event batches, then a GPU merge. This avoids CPU readback and exploits queue headroom.
- Packed Metal remains the memory-floor candidate for routed D8F because it preserves packed indices and can own mixed `k/bits` metadata directly.

## Private API Boundary

- Undocumented MPSGraph/CoreML knobs are now probes, not defaults. They can falsify placement assumptions, but private enum values and string priorities are unsafe to infer.
- `e5rtComputeDeviceTypeMask` is useful to force CPU fallback and prove ANE placement; the measured target remains default CoreML ANE placement with public optimization hints.
- MPSGraph private ANE selectors are present but did not beat async/shared-event batching. Keep MPSGraph's role as GPU scheduling/fencing until a private mode passes real H3355 canaries repeatedly.
