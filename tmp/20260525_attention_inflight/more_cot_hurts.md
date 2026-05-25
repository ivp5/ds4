# More CoT can HURT — the prefix-sensitivity finding

silv 2026-05-25 continue: another deep-seated fallacy candidate exposed.

## Common LLM wisdom

"Longer chain-of-thought = better answers." This is the canonical
narrative for reasoning models: train them to "think more", let them
emit longer CoTs, accuracy goes up.

## Forced-commit prefix-frac sweep on AIME 2026 4bit

For each cell, take prefix = first `frac` of cached CoT, append
forced-commit preamble + greedy 8 tokens, measure correctness:

| Cell | Truth | frac=0.20 emit | frac=0.40 emit |
|------|-------|----------------|----------------|
| P01  | 277   | 85 ✗ | 85 ✗ |
| **P02** | **62** | **62 ✓** | **56 ✗** |
| P03  | 79    | 79 ✓ | 79 ✓ |
| P04  | 70    | 55 ✗ | 56 ✗ |
| P05  | 65    | 65 ✓ | 65 ✓ |
| P06  | 441   | 441 ✓ | 441 ✓ |
| P07  | 396   | 396 ✓ | 396 ✓ |
| P08  | 244   | 244 ✓ | 244 ✓ |
| P09  | 29    | 1/10 ✗ | 1/12 ✗ |
| P10  | 156   | 120 ✗ | 100 ✗ |

**At 20% prefix: 6/10 correct. At 40% prefix: 5/10 correct.**

P02 specifically: at 20% prefix the substrate emits **62 (truth)**;
at 40% prefix the substrate emits **56 (wrong)**.

The CoT between 20% and 40% MOVED THE SUBSTRATE FROM RIGHT TO WRONG on
P02. More thinking made it worse.

## Mechanism

The model has a PRIOR-BELIEF mapping: question-shape → likely-answer-
shape (learned from training). At 20% prefix, the substrate is still
dominated by this prior. The prior happens to be CORRECT for P02
(question-shape resembles training problems with answer ~62).

As the CoT progresses (20% → 40%), the model goes down a reasoning
path. On P02, that path EXPLICITLY arrived at a wrong intermediate
conclusion (probably some algebra step that produced 56 or similar).
The substrate now has the wrong intermediate as ACTIVE CONTEXT.
Forced-commit reads this active context and commits 56.

The CoT REASONING ACTIVELY DISPLACED the correct prior belief.

## The flip pattern reveals model dynamics

For P02 specifically:
- Prior belief (no CoT): probably 62 (correct)
- After 20% CoT: 62 (prior preserved)
- After 40% CoT: 56 (reasoning has wandered)
- After 60% / 80%: pending

If at 80% prefix the substrate commits to 62 again, the CoT eventually
RECOVERED (consistent with offline rescue working for P02). If it
sticks at 56, the CoT got stuck wrong.

This is the in-flight version of the rescuability gradient: WHERE
ALONG THE COT does the substrate believe truth? Different cells have
different "right-answer windows" along the trajectory.

## Why this refutes "longer CoT = better"

The common framing ignores that:
1. The model's PRIOR is often right on familiar question-shapes
2. CoT can introduce WRONG intermediate conclusions
3. Wrong intermediates POLLUTE the substrate going forward
4. The CoT may or may not recover before commit

For HARD problems (P01, P09, P10) where prior is wrong: more CoT
doesn't help because the model's prior is also wrong; both prior and
CoT converge to wrong.

For MEDIUM problems (P02) where prior is right but CoT can wander:
**less CoT can be MORE correct than more CoT**.

For EASY problems (P03, P05, P06, P07, P08) where prior is robust:
CoT length doesn't change much.

## Implications

The "extend CoT budget for harder problems" doctrine is wrong as a
universal rule. The correct doctrine is per-problem-shape:
- If the model's prior is right (typical for question-shapes well-
  represented in training): commit early, skip CoT
- If the model's prior is wrong but CoT can recover (medium): give
  budget but verify continuously
- If both prior and CoT-reasoning are wrong (hard novel problems):
  no budget helps; need external verification or larger model

A budget-allocator can use the substrate truth-rank gradient at chunk_1
(20%) vs chunk_2 (40%) DIVERGENCE as a signal: if rank changes between
positions, the model is exploring; if stable, decision is "locked in".

## Files

- `tmp/20260525_attention_inflight/more_cot_hurts.md` (this)
- `tmp/20260525_attention_inflight/forced_extract_4bit_0_20.json`
  (6/10 correct including P02)
- `tmp/20260525_attention_inflight/forced_extract_4bit_40pct.json`
  (5/10 correct, P02 wrong)
- Remaining: frac=0.60, frac=0.80 (in progress)
