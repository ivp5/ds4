# Ship status — silv 5-part directive 2026-05-27

silv: "work on tool dsml and aime. work on the 74 remaining classic
mtl kernels to mtl4 ports, including mtl4 moe-matmul, then icb moe
dispatch and gguf emit pipeline."

Two ships landed this turn (commits a88bad7 + 179701a), three design
memos with implementation paths.

## Done this turn

### Item 1 — Tool DSML A/B (commit a88bad7)
6 DSML prompts × 4 candidates via daemon, 51s wall. Result table:

```
prompt              h2074  h2071  h2068
dsml_func_call         10     10     10
dsml_json_obj          10     10     10
dsml_python_dec        10     10     10
dsml_typescript        10     10     10
dsml_xml_tag           10     10     10
dsml_yaml_kv           10     10      0
```

**0 top-1 token flips across all 24 measurements.** The 10-line diffs
are confidence shifts in ranks 2-3; top-1 commit stable. H2074's
deployable claim survives the DSML arm that codex H2072 explicitly
flagged.

Surprise finding: `dsml_yaml_kv` inverts (E52-included candidate
shows 0 diff while no-hub + replay-best show 10). YAML routing at
L09 doesn't touch E52 — diagnostic for the route-role atlas (H2075).

### Item 2 — AIME prefill A/B (commit a88bad7)
3 AIME 2026 prompts prefilled to `\boxed{` × 4 candidates × daemon,
72s wall. Result: 0 top-1 flips. Top-1 stable: P01="?", P02="53",
P03="79". L09 prune doesn't break answer-commit position.

**Note**: this is PREFILL only (next-token logit at boxed position).
Full generation-level test (multi-token rollout × K samples) is
hours of compute per (problem × candidate). Deferred to next-turn
budget when the deployable GGUF emit lands.

### Cumulative gate
18 prompts × 5 domains:

| domain      | prompts | top-1 flips (H2074) | top-1 flips (H2071) | top-1 flips (H2068) |
|-------------|--------:|--------------------:|--------------------:|--------------------:|
| math        | 3       | 0                   | 0                   | 0                   |
| knowledge   | 3       | 0                   | 0                   | 1 (Paris)           |
| code        | 3       | 0                   | 0                   | 0                   |
| DSML        | 6       | 0                   | 0                   | 0                   |
| AIME-pre    | 3       | 0                   | 0                   | 0                   |
| **TOTAL**   | **18**  | **0**               | **0**               | **1**               |

H2074 `[55,71,188,231,254]` is the validated deployable prune set.
H2068's e52_included produces the 1 flip (knowledge), empirically
validating H2077's `full_hit_domains` protection rule.

## Designed but not implemented this turn

### Item 3 — 74 classic-MTL kernels → MTL4 ports
Honest scope: full port is multi-week (~74 × 0.5 day each).

Shipped (commit 179701a, NEXT_INTEGRATION.md):
- Pipeline registration template mirroring `g_hadamard_mtl4_pipeline` (#653)
- Storage-class migration rule: `constant` → `device const` (R1 lesson)
- ArgumentTable encoder helper signature
- Canary test design

Highest-leverage kernel for first port:
`kernel_mul_mm_id_fp16_pair_swiglu_f32` (moe.metal:1282). Per-token
decode fires 774 of these calls × ~50µs encoding = 40ms/token.

Next-turn implementer starts from the template; first port is
1-2 turns (port + barrier debug per R1 lesson).

### Item 4 — ICB MoE dispatch (Phase 7)
Shipped (commit 179701a): Phase 7 design covering record-once at
model open + per-token replay with routed-expert IDs in
indirectArgumentBuffer.

Recommended order: ship MTL4 MoE-matmul (Item 3) for ONE expert,
measure speedup against classic. ICB capture against winner. Avoids
debugging two new things on the hot path.

### Item 5 — GGUF emit pipeline
Shipped (commit 179701a): Python skeleton + open-question list.

**Adjacent prior work**: `analyzers/trim_experts_gguf.py` (400 lines)
implements WHOLE-EXPERT file-trim (Path A from the trim-ladder).
Tested against trim50 deployment (#531, #538).

**The H2074 prune set is ORGAN-level not expert-level**: only DOWN
organ pruned at L09, experts {55,71,188,231,254}. The runtime wire
(#651-#657) implements this via `DS4_ORGAN_SKIP` env. File-level
equivalent: zero the DOWN tensor rows for those (layer, expert)
pairs.

The deployable shape needs a NEW tool (`trim_organs_gguf.py`) that:
1. Reads source GGUF tensor table
2. Per (layer, expert, organ) decision:
   - Whole-expert removal → reuse trim_experts_gguf.py
   - Per-organ zero → new logic (zero DOWN rows in-shard)
   - Basis-aware Hadamard → new logic (apply H, requantize)
3. Re-emit GGUF with sidecar JSON
4. Update tensor table offsets

This is 1-2 turns of focused work; the skeleton + adjacent prior
art compress the architecture decisions.

## Roadmap dependency graph

```
DSML+AIME A/B (done)
       │
       ▼
H2074 prune validated  ──────────→  GGUF emit (Item 5, next-turn)
       │                                    ↑
       │                                    │ uses Hadamard primitive
       │                                    │ from #653
       │                                    │
       ▼                                    │
MTL4 MoE-matmul (Item 3, next-turn) ────────┤
       ↓                                    │
       ICB MoE dispatch (Item 4, after MTL4)│
                                            │
       Deployable pruned + basis-aware GGUF ┘
       (5× gen speedup target combined)
```

## Tasks updated

- #661 Tool DSML A/B → completed
- #662 AIME prefill A/B → completed
- #663 MTL4 MoE-matmul port design → in_progress
- #664 ICB MoE dispatch design → in_progress
- #665 GGUF emit pipeline design → in_progress

## Engineering posture (jjj/codex pattern)

What worked this session:
1. Ship small atomic units (DSML + AIME A/Bs as separate prompts
   batched in daemon)
2. Design memo when implementation spans turns
3. Reference existing adjacent code (trim_experts_gguf.py) instead
   of re-architecting from scratch
4. Honest scope-stating (74-kernel port is multi-week, not
   one-turn)

What deferred to next turn:
1. Generation-level AIME (vs prefill A/B) — heavier compute
2. First MTL4 MoE-matmul port (~half day)
3. trim_organs_gguf.py (~half day)
4. ICB MoE Phase 7 (depends on MTL4 winner choice)
