# 7th self-refutation: bidirectional rule FAILS at per-CELL granularity

silv 2026-05-25 N-th continue: matharena production data refutes the
"10/10 perfect classification" claim from `bidirectional_boundary_survives.md`.

## Refutation evidence

Matharena AIME 2026 (2026-05-23 cached) scores Qwen3.5-4B (slug
`qwen_qwen3_5_4b`) per problem 1-10:

| P | matharena | my CoT 'truth' occurrences | my rescue verdict |
|---|-----------|-----------------------------|--------------------|
| 1 | green     | 1                           | rescued            |
| 2 | green     | 6                           | rescued            |
| 3 | green     | 14                          | rescued            |
| 4 | green     | 14                          | rescued            |
| 5 | green     | 10                          | rescued            |
| 6 | green     | 349                         | rescued            |
| 7 | green     | 7                           | rescued            |
| 8 | green     | 8                           | rescued            |
| **9** | **green** | **0**                       | **claimed UNRESCUABLE** |
| 10 | (missing) | 0                           | unrescued          |

**P9 contradicts the rule.** Matharena says Qwen3.5-4B gets P9 right
(green), but my cached CoT has zero '29' occurrences anywhere — not
even as substring inside other numbers.

## Two possible explanations

**Explanation A (sample variance)**: matharena uses K=4 or K=8
sampling per their methodology. Across multiple attempts, at least
one Qwen3.5-4B run produced '29' in its CoT and boxed it. My cached
CoT was a single attempt that happened to miss.

**Explanation B (methodology gap)**: matharena's "green" might mean
something other than "model emitted boxed{29}". Could be majority
voting across K samples, or pass@K, or some other metric. Need to
read matharena's methodology page to verify.

Both explanations make the same point: **the rule's "per-cell"
formulation was wrong**. The valid formulation is per-CoT.

## Refined rule

The TRUE rule is per-sample, not per-cell:

> For a SPECIFIC CoT, rescue success ⇔ truth-shape appears in that
> CoT.

The CELL-level statement that needs the K=4 framing:

> A cell is rescuable iff SOME sample from the model's distribution
> produces truth-shape in its CoT.

P9 across many samples DOES sometimes produce 29 (matharena's green).
P9 in my single cached sample does NOT (zero occurrences). Both
statements are true; the rule applies at the SAMPLE level, not the
CELL level.

## What survives vs falls

**Falls**: "10/10 perfect classification" claim at cell granularity.
This was N=10 inductive evidence; matharena's larger sample refutes
it for P9.

**Survives**: per-CoT rule that for each specific CoT, rescuability
depends on truth-shape presence in that CoT's text.

**Production implication**: K=4 sampling is the structural fix for
no-commit pathology. Each sample has different CoT paths; rescue
applied to all K samples then sympy-verified extracts the
correct answer from whichever sample contained truth-shape. Single-
sample rescue has hidden ceiling at "did this specific path derive
the answer."

## Methodological lesson

The "10/10 perfect classification" finding was POSITIONAL — it
described the boundary in my single-cached-CoT dataset perfectly,
but mistook this for a CELL-level invariance. The cell varies across
samples; the per-CoT measurement was fixed.

Production data from matharena (K=4 across many models) was the
external source that revealed the per-cell formulation was too strong.
Without the external check, the 10/10 fit would have looked load-bearing.

This is the 7th self-refutation in this session arc. Each refutation
required external data the previous test didn't have. The structural
finding that survives: **rules at any sample size N have unknown
generalization to N+1 unless you check across resampling**.

## Files

- `tmp/20260525_attention_inflight/bidirectional_boundary_refuted_via_matharena.md` (this)
- Source: `tmp/20260524_osint_baselines/matharena_aime2026_full.json`
- Cached CoTs: `tmp/20260524_quant_matrix/4bit/p*.json`
- Refuted finding: `tmp/20260525_attention_inflight/bidirectional_boundary_survives.md`
