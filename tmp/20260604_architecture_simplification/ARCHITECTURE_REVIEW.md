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
- Expanded MPSGraph gather remains O(n), but with inflated constants from materialized `int32` indices; packed-index Metal is the baseline for memory-floor comparison.

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

1. Packed-index MPSGraph probe: prove or reject in-graph packed-code decode.
2. Runtime sidecar scaffold: one default-off owner for CoreML model cache, D8F graph/kernel cache, hidden backing, parallel launch, and merge.
3. Package cache manifest: record layer package path, compile URL, resident footprint, load state, and validation status in one typed table.
4. Canary harness library: if more ANE/MPSGraph canaries are added, extract shared CoreML feature/backing helpers instead of copy-pasting per file.
5. Packed Metal baseline: quantify actual bytes/op against MPSGraph expanded gather so “memory floor” has a measured reference.
