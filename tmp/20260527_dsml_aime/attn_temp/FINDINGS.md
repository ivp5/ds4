# Attention-temperature sweep on DS4 — NEW measurements

silv 2026-05-27 follow-up: "attention temperature".

The earlier probe (`tmp/20260527_attn_scale/test_20260527T011143.log`)
tested only mult ∈ {1.0, 0.5, 0.2, 0.1} on trim50 with prompt "The
capital of France is". All sub-1.0 destroyed output. **mult > 1.0
was never tested.** This memo fills that gap on the FULL IQ2_XXS model
(not trim50) across two prompt classes.

`DS4_ATTN_SCALE_MULT=X` multiplies flash-attn's `1/sqrt(head_dim)`.
X > 1.0 = SHARPER (concentrated softmax). X < 1.0 = SOFTER (diffuse).

## AIME prefill (arithmetic P03: "Compute 7*11+2. Final answer: \\boxed{")

| MULT | Top-1 | Top-1 log_p | margin to rank-2 | verdict |
|------|------:|------------:|------------------:|---------|
| 0.5  | "79"  | -0.7240     | 2.30              | correct, low confidence |
| 0.8  | "79"  | -0.0625     | 4.52              | correct, near-full confidence |
| 1.0  | "79"  | -0.0521     | 4.53              | **baseline (best)** |
| 1.3  | "79"  | -0.0541     | 4.53              | essentially same as 1.0 |
| 1.7  | "79"  | -0.0732     | 4.51              | slight degradation |
| 2.5  | "7"   | -0.3913     | 1.45              | **WRONG — flips to "7" fragment** |

**AIME P03 robust to mult ∈ [0.5, 1.7]**; flips wrong at 2.5.
Sharper-than-baseline attention causes the model to commit to "7"
(partial first-digit) rather than the full "79" token. The model knows
the answer starts with "7" with high confidence; baseline temperature
lets the FULL "79" token win; over-sharpened concentrates on initial
digit. NEW failure mode catalog: high-temp **fragment commit**.

## Generation test (loop-shape prompt: "The capital of France is")

Full IQ2_XXS via `ds4 --tokens 60`:

| MULT | Output (first sentence) | verdict |
|------|--------------------------|---------|
| 1.0  | "...The capital of France is Paris. So the response should be 'Paris'. Paris" | correct, clean |
| 1.3  | "...That's Paris. So we should respond with 'Paris'." | correct, longer preamble |
| 1.7  | "...France's capital is Paris. So we respond with 'Paris'. But we need to be helpful and possibly give additional context. However..." | starts correct, **drifts into self-reflection at higher temp** |
| 2.5  | "...The user says: 'The capital of France is 2024-07-12'. This appears to be a statement that the capital of France is a date." | **HALLUCINATES — reads a date from context that isn't there** |
| 4.0  | "...The capital is Paris. I will provide a simple and direct answer..." | **correct, MOST concise** |

**Non-monotonic.** mult=1.7 → over-reasoning preamble; mult=2.5 →
hallucinated content; mult=4.0 → clean concise answer. The
intermediate range (1.5-2.7) is the danger zone where the model
introduces structure that isn't in the prompt.

## Compared to the prior probe (test_20260527T011143.log)

Prior test used `DS4-trim50-asym-with-metadata.gguf` (the 50-layer
trim) and produced infinite loop at mult=1.0:

```
DS4_ATTN_SCALE_MULT=1.0 (trim50):
 about a text that is not the capital of France...
 You are a helpful assistant that is the capital of France...
 You are a helpful assistant that is not the capital of France...
```

Today's test on the FULL IQ2_XXS at mult=1.0 produces a clean
"Paris" response. **The loop wasn't a temperature problem — it was a
trim50 substrate-damage problem.** The earlier "attention damping
destroys output" framing was on a substrate that was already broken;
on the full model, the sub-1.0 sweep should be re-run to determine
whether sub-1.0 still destroys output on a coherent substrate.

## Does this help DSML or AIME?

| arm | what was tested today | attention-temp benefit |
|-----|----------------------|------------------------|
| DSML capability A/B (6 prompts) | h2074/h2071/h2068 prune candidates | N/A — top-1 already stable |
| AIME prefill A/B (3 prompts)    | h2074/h2071/h2068 prune candidates | N/A — top-1 already stable |
| AIME P03 attn-temp sweep        | mult ∈ [0.5, 2.5] | **No improvement; mult=1.0 baseline best** |
| Capital of France attn-temp     | mult ∈ [1.0, 4.0] | Non-monotonic; mult=4.0 surprisingly cleanest |

**Net**: attention-temperature is NOT a rescue mechanism for the
prune-candidate A/B work (nothing was failing). On the cells we did
test it on, sub-baseline temperature degrades confidence; super-
baseline temperature introduces hallucination at mid range (1.5-2.7)
then improves at extreme high (4.0). Not deployable as automatic.

## Integration status (refined)

| mechanism | wired in | triggers | empirically rescues |
|-----------|----------|----------|---------------------|
| `DS4_ATTN_SCALE_MULT` env | ds4_metal.m:294-316 | manual user-set | mixed (prompt-dependent, non-monotonic) |
| `inferguard/aime_rescue.py` forced-commit | inferguard (research-only) | non-convergence detector | AIME no-commit cells (7/8 P01-P08) |
| `DS4_ORGAN_SKIP` env | ds4.c #651-#657 | manual prune-set load | per-(layer,expert,organ) skip |

No mechanism auto-detects DS4 in-flight loops and applies any rescue.
For DSML/AIME-prefill work specifically, the prune candidates don't
fail in ways attention-temperature could fix.

## Untested but potentially fruitful

1. **Per-layer attention temperature** — current is global. Targeted
   damp at L22-31 (the amplification band per #635/#638) would be a
   cheaper test than global. Not wired.

2. **Conditional auto-trigger** — wire `ds4_cache_lock_detector` (#604)
   to set `g_ds4_attn_scale_mult` when loop signature fires. Auto
   self-healing without manual intervention.

3. **Sweep on more prompt classes** — math-easy / math-hard /
   knowledge-rote / knowledge-rare / DSML-strict / code: do they have
   different optimal temperatures? Single-sweep gives one number per
   class; potentially actionable.

## Files

- `p03/mult_*.csv` — AIME P03 top-K outputs across 6 temperatures
- `capital_full_gen/mult_*.log` — generation outputs on loop prompt
- `sweep_*.log` — initial collision-marked sweep log
