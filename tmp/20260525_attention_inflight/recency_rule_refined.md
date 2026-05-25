# Recency rule refined — two-mode substrate emission

silv 2026-05-25 OOM-finer accuracy: the recency rule has a fallback mode.

## P10 data refutes pure-recency

P10 cached CoT (105074 chars, truth=156): the model's emits across
prefix-frac sweep were {100, 126, 105, 100, 126, 168, 100 at cap=24k}.

Number-shape occurrence counts in CoT:
- '156' (truth): **0**
- '100': **0** (emitted at cap≤16000)
- '126': **0** (emitted at frac=8000)
- '105': **12** (emitted at frac=2000)
- '168': **0** (emitted at cap=24000)
- '90': 11 (angle context, not answer-shape)
- '13', '14', '15': 63, 38, 39 (side lengths)

**100, 126, 168 ALL have count=0 in CoT** — they come from training
imprint, NOT context retrieval.

## Two-mode substrate emission rule (refined)

**Mode 1 (text-retrieval)**: if proximal context contains a number-shape
candidate (any 1-3 digit integer plausible-as-answer), emit the
MOST RECENT one. The recency rule.

**Mode 2 (prior fallback)**: if NO recent number-shape candidate
exists, emit a TRAINING-IMPRINT prior conditioned on (problem context
+ context fingerprint). The prior is MULTI-MODAL: different contexts
trigger different priors from the same training imprint.

P10 mode switches:
- frac=2000 (4k chars): includes '105' mentions (12 in CoT) → Mode 1, emit 105
- frac=8000+: no recent answer-shape → Mode 2, emit prior {100, 126, 168}

## The training-imprint multi-modality

For "hexagon area with 60° angle and small integer side lengths",
the model's training has SEVERAL plausible answer shapes it can emit:
- 100 (common round number)
- 126 (common AIME-shape answer)
- 168 (common geometric area shape)

The model picks among these based on CONTEXT FINGERPRINT — what the
preceding text suggests as the "type" of answer expected. The prior
distribution isn't a single canonical answer; it's a learned manifold
of plausible answers from training.

## Why P09 and P10 are unrescuable

For both cells, truth NEVER appears in CoT. So Mode 1 never fires.
The model defaults to Mode 2 prior. Since neither model's training
imprint contains the correct answer (29, 156), no rescue protocol
can recover.

P09 model wandered into combinatorial arithmetic (top numbers
{4, 2, 1, 6, 5, 3} not the algebraic geometry the problem requires).
The model misread the PROBLEM CLASS.

P10 model emitted partial geometric calculations but never reached
the correct hexagon area. The model's prior for this problem class
includes plausible candidates {100, 126, 168} but not truth 156.

## Unifying with the wrong-belief doctrine

The wrong-belief doctrine + two-mode emission unify:

- Confident wrong commits are STRUCTURED: they come from coherent
  training-imprint priors (Mode 2) or recent-context retrieval (Mode 1)
- The model is NEVER hallucinating "randomly" — every emit has a
  mechanism: retrieval or imprint
- The "calibration problem" — model is confident on wrong answers —
  is structural because BOTH modes produce equally-confident outputs

## Operational consequence

For deployable rescue:
1. Detect if truth-shape candidate is in proximal context (Mode 1 viable)
2. If yes → forced-commit at last occurrence position → likely emits truth
3. If no → forced-commit emits a Mode 2 prior; not truth-likely
4. External verifier (sympy) needed in both cases

For Mode 2 fallback cells, no inference-stack adjustment helps. Need:
- Bigger model with better training imprint for the problem class
- External problem-solver (not LLM) to compute truth
- Different prompt that better activates the right Mode 2 prior

## Files

- `tmp/20260525_attention_inflight/recency_rule_refined.md` (this)
- P10 prior sweep data showing {100, 126, 105, 168} multi-modal emit
- P09/P10 CoT number-frequency audit
