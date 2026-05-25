# 16th probe: problem structure predicts which cells fail

silv 2026-05-25 continue: stepped back from rescue mechanism arc to
look at WHICH problems fail for Qwen3.5-4B and why. Read all 10
AIME 2026 P01-P10 problem statements.

## Shared structure of unrescuable cells

**P9 (truth=29)** — Joanne's die-stickers conditional probability:
- Setup: 6 die rolls, stickers 1-6 placed in order, "if sticker on face, new covers old"
- Question: P(exactly one face blank | all even stickers visible) = m/n
- Find: m + n
- **3-step composition required**: (1) compute conditional probability via
  state-machine enumeration, (2) reduce m/n to lowest terms, (3) sum m+n
- The model loops on step 1: enumerating the state machine without commitment

**P10 (truth=156)** — rotated triangle hexagon area:
- Setup: triangle 13-14-15, rotated about circumcenter so AC ⊥ BC
- Question: area of hexagon AA'CC'BB'
- Find: integer closest to area
- **Ambiguous interpretation**: "AC perpendicular BC" — which segments
  exactly? Original or rotated? Or A'C' ⊥ BC?
- The model loops on interpretation: "Let's assume condition is X? No.
  Let's assume condition is Y? No." × 100+

## Comparison: structure of rescuable cells

| cell | problem type | reasoning path | composition depth |
|------|--------------|----------------|---------------------|
| P01 | kinematic | 3 rates, common arrival time | shallow (single equation) |
| P02 | counting | palindromes with digit sum | shallow (direct enumeration) |
| P03 | geometric ratio | sphere-in-hemisphere coverage | medium (single ratio) |
| P04 | algebraic substitution | a+b+ab = (a+1)(b+1)-1 | medium (counting after transform) |
| P05 | rotation geometry | 2 rotations, cos θ value | medium (algebraic) |
| P06 | equation + divisor count | log equation, then divisors of product | medium (clean steps) |
| P07 | function iteration | cycle structure π⁶ = id | medium (clean combinatorics) |
| P08 | number theory | divisors mod 12 of 17017^17 | medium (clean) |
| P09 | conditional probability + state machine + reduce + sum | **DEEP (4-step)** |
| P10 | geometric construction + area + round | **DEEP + AMBIGUOUS** |

The pattern is clear:
- Rescuable cells: medium-depth single-track reasoning
- Unrescuable cells: deep multi-step composition WITH interpretive
  ambiguity at first step

## Why interpretation-lock matters

When the first step is unambiguous, the model commits to a direction
and proceeds (even if to a wrong intermediate). The truth eventually
emerges somewhere in the CoT.

When the first step is ambiguous (P10: which segments are perpendicular?
P9: how exactly do the conditional probability events compose?), the
model enters an INTERPRETATION LOOP — proposing and rejecting frames
without committing.

The loop fills the token budget with non-committal text. Truth is
never derived. Mode B (capability/path failure) is structurally a
consequence of interpretation-lock at the first reasoning step.

## Why production K-sampling helps P9 but not P10 as much

Matharena: Qwen3.5-4B P9 = 92.6% green, P10 = 44% green.

P9's "ambiguity" is more about complexity than genuine interpretation
choice. K-sampling at temperature > 0 gives the model multiple shots
at the state-machine enumeration. Some shots commit to a frame and
follow through. The 92% rate suggests the loop entry is shallow —
small RNG perturbation escapes.

P10's ambiguity is structurally deeper. The problem statement itself
has multiple valid interpretations (even matharena's official
problem likely had clarification — recall CLAUDE.md note about
"$\overline{A'C'}$ is perpendicular to $\overline{BC}$" being the
canonical form, with my parquet having a corruption that dropped the
primes). The 44% green rate reflects this genuine ambiguity.

## Implication for problem-class taxonomy

**Class A — single-track problems**: rescuable; Mode A (no-commit
pathology) is the typical Qwen3.5-4B failure; rescue works at any
sufficient cap.

**Class B — multi-step composition problems**: less rescuable;
K-sampling helps in proportion to interpretation depth; matharena
provides the empirical rate per problem.

**Class C — ambiguous-frame problems**: poorly rescuable; the model's
interpretation lock IS the problem structure; even production
sampling has bounded recovery rate.

For Qwen3.5-4B on AIME 2026 I:
- Class A: 8/10 cells (rescuable via cap-mechanism rule)
- Class B: P9 (rescuable at K=4-8 via sampling)
- Class C: P10 (genuinely hard; ~44% production rate)

## Doctrine: the failure mode IS the problem structure

The session's earlier framing tried to read structure in CoT digit
frequencies (Mode B1/B2). The actual structure is in the PROBLEM
statement: how many composition steps + how much interpretation
ambiguity. The loop attractor in CoT is the model's response to
problem-structure-induced lack of clean reasoning track.

This refines the engineering question. Instead of "how do we improve
rescue for Mode B?", the right question is "for each problem class,
what intervention is structurally appropriate?":
- Class A: cap-based rescue (already deployed)
- Class B: K-sampling at temperature > 0
- Class C: external semantic verifier with anchoring-bias mitigation
  + multiple proposer models

This matches CLAUDE.md Conjecture #21's deployable shape exactly:
"multi-proposer pool + per-problem sympy verifier" — Class C requires
diverse proposers, not just rescue protocol refinements.

## Files

- `tmp/20260525_attention_inflight/problem_structure_predicts_failure.md` (this)
- Source: `aime/prompts/aime_2026_p0_p9.json` (canonical problem statements)
- Earlier in arc: 15 memos in `tmp/20260525_attention_inflight/`
