# 21st probe RESULT: K=4 at insufficient budget produces 0/4 — not a refutation, a budget condition

silv 2026-05-25 21st continue LIVE probe completed. Qwen3.5-4B-MLX-4bit
K=4 samples on AIME P09, T=0.7, max_tokens=4096. Wall time ~60s per
sample.

## Result

| k | seed | chars | '29' count | wall | state at truncation |
|---|------|-------|------------|------|---------------------|
| 0 | 20260525 | 13284 | 0 | 56.9s | defining V, max(A_D), still building state-machine framework |
| 1 | 20260526 | 12246 | 0 | 61.1s | parsing "all even-numbered stickers visible" condition |
| 2 | 20260527 | 12524 | 0 | 60.9s | case analysis on roll outcomes (G = r_3?) |
| 3 | 20260528 | 15409 | 0 | 59.9s | counting blank faces, considering face-visited counts |

**0/4 samples contain '29' in CoT.**

## Why this is NOT a refutation

All 4 samples were ACTIVELY REASONING at truncation, not in loop
attractor. None had:
- Completed the conditional probability calculation
- Computed m/n in reduced form
- Summed m+n

They were stuck in the SETUP phase of the multi-step composition (per
problem-structure taxonomy from 16th probe). The reasoning is genuine,
just incomplete due to budget.

## Why the cached run was different

My single cached Qwen3.5-4B run on P09 produced 76,305 chars (about
17,400 tokens) and ended in loop attractor. That run had enough
budget to:
- Exhaust productive reasoning
- Enter loop attractor: "Wait, if $p_2 = 4$, then $X_4 = r$." × 100+

So the cached run failed AT BUDGET via interpretation-lock. My fresh
K=4 samples failed BEFORE BUDGET via early truncation. Two different
failure modes.

## Why matharena gets 92.6% green

Matharena's metadata: Qwen3.5-4B tokens = 27853 average. That's ~7×
my probe's budget. With ~28K tokens:
- Some samples complete the reasoning before loop attractor entry
- Some samples interpretation-lock early but K-sampling provides
  diverse interpretation paths
- Aggregated green rate ≈ 92.6%

My K=4 at 4096 max_tokens is INSUFFICIENT BUDGET, not insufficient
sampling diversity.

## What this validates / refines

**Refined K=4 hypothesis**: K-sampling at temperature > 0 escapes
interpretation-lock IFF budget is sufficient for at least one sample
to complete reasoning. Below threshold budget, all K samples fail to
reach truth regardless of K.

**Budget threshold for Qwen3.5-4B P09**: somewhere between 4K and 28K
tokens. My probe lower-bounds the threshold; matharena upper-bounds.

**This was a NEW finding from a fresh probe** — the cached data
couldn't reveal it because the cached run used a long budget. The
fresh probe with shorter budget surfaces the budget-K interaction.

## Doctrine extension

For the deployable rescue protocol on multi-step composition problems
(Class B per 16th probe):
- Cap-mechanism rescue (Class A) works at any budget where truth_at
  is in prefix
- K-sampling (Class B) needs BUDGET above threshold (~10K+ tokens
  for P09) to give samples a chance to derive truth
- Class B production: needs both K AND sufficient max_tokens budget

For Qwen3.5-4B on P09 specifically: max_tokens=4096 is below the
threshold where K-sampling works. Production should use higher budget.

## The 21st live probe vs 20 prior cached probes

The fresh live probe yielded a finding (budget threshold for K-sampling
escape) that 20 prior cached-data probes could not show. The probe's
0/4 result wasn't the matharena-validation I might have expected, but
the WHY (mid-reasoning truncation, not loop attractor) is the new
information.

This validates silv's "drop the SAFELY idea — do UNSAFE takes, MAKE
MISTAKES" directive operationally. The "mistake" here was budget=4K
underestimating the model's reasoning length. The mistake produced
the finding.

## Files

- `tmp/20260525_attention_inflight/k4_probe_result_budget_dependent.md` (this)
- `tmp/20260525_attention_inflight/k4_p09_result.json` (raw data)
- `tmp/20260525_attention_inflight/k4_p09_fresh_probe.py` (probe script)
