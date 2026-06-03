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
