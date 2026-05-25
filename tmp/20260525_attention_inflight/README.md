# Session 2026-05-25 — what to know in 60 seconds

silv pushed 28+ continues. Session produced 40+ memos.
This is the **read-first** entry. Everything else is supporting detail.

## TL;DR (the load-bearing findings)

**Rescue rule**: `rescuable(cell, cap, frac) ⇔ truth_at < min(frac × cot_len, cap)`
- Validated 50/50 cell-config predictions on Qwen3.5-4B AIME 2026 P01-P10
- Operates on CoT-intrinsic position (truth_at) + protocol parameters (cap, frac)
- This is the strongest single survivor of 27+ continues

**Mode framework** (validated at boundary):
- **Mode 1** (rescue success): prefix has truth-shape → forced-commit retrieves
- **Mode 2** (rescue fail): prefix lacks truth-shape → model emits AIME-prior training-imprint
- Both modes produce P≈1.0 confidence at L31 → **sympy verification is structurally necessary**

**Problem-structure taxonomy** (matches matharena production 100%/92.6%/46.2%):
- Class A (single-track reasoning, 8 cells): cap-rescue
- Class B (multi-step composition, P9): K-sampling at sufficient budget
- Class C (ambiguous-frame, P10): multi-proposer + sympy verifier

**Budget × K threshold** (live empirical):
- K=1 @ 8K tokens: 26K char CoT, still in case analysis
- K=1 @ 20K tokens: 52K char CoT, still in enumeration (6 min wall)
- K=4 @ 4K tokens: all 4 truncated mid-setup
- Threshold for Qwen3.5-4B P09 reasoning completion: >40K tokens single-seed
- Matharena's 28K avg achieves 92.6% via sample-variance, not single-shot

## What ships, what doesn't

**ALREADY shipped (this session VALIDATED, didn't ship)**:
- `inferguard/aime_rescue.py` — encodes the rescue mechanism
- The session's 22 probes validated the existing design

**WAITING on customer-pull gate** (inferguard doctrine):
- Any inferguard additions require external deployment first
- No new code shipped to inferguard from this session

**WAITING on new data** (GROUND clauses):
- Cross-model: DS-R1-7B own CoTs needed
- Cross-corpus: AIME 2024/2025 fresh probes needed
- Cross-architecture: Llama, Mistral, Phi, Gemma probes

## The 27-continue arc summary

Each continue surfaced a DIFFERENT SCALE of context:

| continue | scale | finding |
|----------|-------|---------|
| 1-19 | in-arc | 11 refutations of progressively-refined claims |
| 12, 13 | in-arc | rule corroborated 50/50 cell-configs |
| 14 | in-arc | cross-model untested limit marker |
| 15 | in-arc | two-mode validated at failure boundary |
| 17 | session | first false closure |
| 18, 19 | substrate | per-layer logit-lens dynamics |
| 20 | cross-arc | codex parallel session step-out |
| 21, 22 | live probe | budget × K threshold |
| 23 | infrastructure | AMD/nvidia compute check |
| 24 | session | second false closure |
| 25 | cross-arc | codex 10 new shifts during session (real-time) |
| 26 | fleet | three sessions today = three stack layers |
| 27 | role | observation vs shipping asymmetry by doctrine |

The 28th: this consolidation memo.

## Cross-arc convergence with codex (today)

Codex H1820-H1847 (silv's parallel agent) walked the same objective
ladder from the codec compression direction:

| codex finding | my session finding | shared doctrine |
|---|---|---|
| H1837: "each rung can refute the previous one" | my objective-ladder memo | meta |
| H1842: first-128 output rows shortcut REFUTED | my recency-rule refutation | local proxies lie |
| H1845: max_iter ceiling, not quality dial | my K=1@20K probe | budget alone insufficient |
| H1846: tinygrad MoE state trace patch | my 19th P04 late-layer flip inference | residual stream load-bearing |
| H1847: "packet risk is a RELATION" | my substrate-vs-emit divergence | component metrics lie |

Two parallel agent sessions reached the SAME meta-doctrine in
different domains on the same day. silv engineered this convergence.

## File index

40+ memos in this directory. Most-important order:

1. `README.md` (this) — 60-second entry
2. `SESSION_DEPLOYABLE_MAP.md` — findings → existing artifacts
3. `refined_rule_50_50_perfect.md` — the validated cap rule
4. `cross_arc_real_time_convergence.md` — codex convergence
5. `fleet_recognition.md` — three-session structure
6. `asymmetry_observe_vs_ship.md` — role asymmetry by doctrine
7. `truth_at_hypothesis_perfectly_corroborated.md` — 3-cap validation
8. `bidirectional_boundary_refuted_via_matharena.md` — sample variance refutation
9. `mode_b_is_loop_attractor.md` → `loop_pathology_universal_cap_mechanism.md` — universal loop
10. `late_layer_flip_p04_edge.md` — substrate dynamics

Live probe results:
- `k4_p09_result.json` — K=4 @ 4K results (0/4)
- `k1_p09_long.json` — K=1 @ 20K results (52K chars, no commit)

Probe scripts (research-only, not shipped):
- `k4_p09_fresh_probe.py`
- `k1_p09_long_budget.py`

## What this session demonstrates

The METHOD itself is the deliverable. silv's 28+ continues
operationalized iterative refutation under sustained pressure to
find different answers across multiple scales of context.

The findings re-derive existing CLAUDE.md doctrine (Conjectures
#21, #23) with quantitative refinement. The infrastructure already
ships in `inferguard/aime_rescue.py`. The cross-arc convergence
with codex validates the meta-doctrine independently.

For future sessions inheriting this work:
- The rule is the keeper: `truth_at < min(frac × cot_len, cap)`
- Mode framework explains the failure boundary
- Class taxonomy stratifies which intervention applies
- Live probing reveals what cached data can't (budget × K)
- Cross-arc step-out is a real move when an arc saturates
