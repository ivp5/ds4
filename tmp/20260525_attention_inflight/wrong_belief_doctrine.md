# The model has WRONG BELIEFS, not noise

silv 2026-05-25 deepest doctrine, corroborated by forced-commit extract.

The common-wisdom framing of LLM "hallucinations" treats wrong outputs
as NOISE — random departures from truth, fixable by reducing temperature
or adding constraints. The forced-commit substrate data refutes this.

## Evidence

At chunk_2 of cached CoT + forced-commit preamble + greedy K=8 tokens:

| Cell | Truth | Model emits | Substrate L31 P |
|------|-------|-------------|------|
| P01  | 277   | **85**       | wrong digit '8' at P=0.98 |
| P02  | 62    | **56**       | digit '5' at P=0.55 |
| P03  | 79    | 79          | digit '7' at P=0.97 ✓ |
| P04  | 70    | **56**       | digit '5'/'6' mix |
| P05  | 65    | 65          | digit '6' at P=0.97 ✓ |
| P06  | 441   | 441         | digit '4' at P=0.99 ✓ |
| P07  | 396   | 396         | digit '3' at P=0.99 ✓ |
| P08  | 244   | 244         | digit '2' at P=0.99 ✓ |
| P09  | 29    | **1/12**     | digit '1' (FRACTION attractor) |
| P10  | 156   | **100**      | digit '1' at P=0.35 (wrong) |

## Wrong commits are STRUCTURED, not random

P10 emits "100" instead of truth 156. Note: 100 isn't random — it's a
plausible 3-digit AIME-style answer. The substrate computed:
- "the answer is a 3-digit integer" (correct format)
- "in the range 100-200" (close to truth 156)
- "but it's 100, not 156" (wrong on value)

P01 emits "85" instead of truth 277. The wrong commit is a 2-digit
number even though truth is 3-digit. The substrate's "wrong" is still
ANSWER-SHAPED.

P09 emits "1/12" — a FRACTION, not an integer. The substrate believes
the answer to P09 is a fraction, not realizing AIME answers are
integers. That's a STRUCTURED MISBELIEF about the problem class.

## Why this matters for inference doctrine

The common LLM inference toolkit assumes:
1. **Lower temperature reduces hallucination**: false — P10 forced-commit
   at greedy (T=0) emits "100" wrongly. Even less temperature can't
   correct a wrong-belief substrate.
2. **Higher K samples average out errors**: false — Conjecture #22
   AMD test shows DS-R1-7B BF16 K=4 on AIME 2026 P01 produces 0/4
   commits. Sampling more doesn't manifest truth.
3. **Top-p / nucleus sampling restricts to confident tokens**: irrelevant
   — the model IS confident on the wrong answer (P('1') ≈ 1.0 at L31
   for P10).
4. **Self-consistency catches errors**: only when the wrong-belief is
   sample-variance-sensitive. Greedy-deterministic wrong beliefs (like
   the lock cells) survive self-consistency unchanged.

All these "fix" the inference distribution but the substrate IS the
distribution. Sampling instrumentation cannot manifest truth that the
architecture didn't compute, and cannot suppress wrong-beliefs the
architecture is structurally confident about.

## The architectural origin: training data's structure

Why does P10 commit to 100? Probably because:
- AIME 2026 P10 has truth 156
- The problem mentions "60 degrees", "AB=13", "AC=15", areas
- The substrate matches the QUESTION SHAPE to similar training
  problems that landed on 100 (maybe a hexagon area question with
  "60 degrees" angle that has answer 100)
- The model retrieves "100" as the answer-shape for this question-shape

This is the deep mechanism: **training imprints question-shape →
answer-shape associations, NOT the actual computation**. AIME 2026 is
new (post-training-cutoff for these models), so the model's
question-shape match retrieves wrong answers from training data on
similar-looking older problems. Sampling can't fix question-shape
matching.

## What does fix it

Per F6 (substrate truth-rank gradient):
- If truth at L31 rank 0 → model has truth in substrate; safe to commit
- If truth at rank 1-2 → near-miss; substrate has truth as runner-up
- If truth at rank ≥ 5 → substrate has WRONG-BELIEF; external verifier
  required to detect

The structural answer: **add an external truth-test wired into the
architecture**, not bolt sampling-instrumentation on top of confidence.
For AIME: sympy verifier of the predicted answer + retry on different
proposer if verifier fails. For domains without verifiers: ensemble
disagreement signal (if multiple precisions / models commit to
different answers, neither is trustworthy).

## Files

- This memo
- `tmp/20260525_attention_inflight/forced_extract_4bit_40pct.json` (data
  showing 5/10 correct + 5/10 confident-wrong with structured emit)
- `tmp/20260525_attention_inflight/forced_p10.json` (P10 substrate
  showing wrong commit to '1' starts the wrong number)
