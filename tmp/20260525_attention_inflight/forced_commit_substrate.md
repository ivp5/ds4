# Forced-commit substrate: confident-wrong ≠ confident-right at substrate level

silv 2026-05-25: deepest sampling-fallacy critique testable directly.

## Setup

For each cell, take prefix at chunk_2 (40% of cached CoT for success cell;
20% for lock cell), append forced-commit preamble:

```
\\n\\nAfter careful analysis, the final answer is \\boxed{
```

Run forward, capture per-layer hidden state at the position right after
the opening `{`. Project through final_norm + lm_head. Inspect L23-L31
commit-tier predictions.

## Results

### P01 LOCK forced-commit (prefix=20% of 62k chars = 12000 chars cap, truth=277, d1='2')

| Layer | top_1 | P(top1) | P(truth='2') | rank(2) |
|-------|-------|---------|--------------|---------|
| L23   | '8'   | 0.034   | 0           | 109846 |
| L27   | '8'   | 0.053   | 0           | 145245 |
| L28   | '8'   | 0.030   | 0           | 179763 |
| L29   | '8'   | **0.922** | 0         | 112998 |
| L30   | '8'   | **0.999** | 0.001       | 471    |
| L31   | '8'   | **0.984** | 0.001       | 7      |

**P01 forced-commit: final-3 unanimously commit `'8'` at P > 0.92.** Truth
digit `'2'` is essentially absent (P ≈ 0, rank 7-179763 across layers).
The substrate at P01-chunk_2 has NO truth latent; when forced to commit,
it picks `'8'` confidently and wrongly.

### P05 NATIVE-SUCCESS forced-commit (prefix=40% of 33k chars = 12000 chars cap, truth=65, d1='6')

| Layer | top_1 | P(top1) | P(truth='6') | rank(6) |
|-------|-------|---------|--------------|---------|
| L23   | '6'   | 0.006   | 0.006       | 0      |
| L27   | '6'   | 0.079   | 0.079       | 0      |
| L28   | '6'   | 0.081   | 0.081       | 0      |
| L29   | '6'   | **0.572** | 0.572     | 0      |
| L30   | '6'   | **0.9995** | 0.9995   | 0      |
| L31   | '6'   | **0.970** | 0.970     | 0      |

**P05 forced-commit: final-3 unanimously commit `'6'` at P > 0.57.** Truth
digit `'6'` IS the top_1 at every commit-tier layer. The substrate at
P05-chunk_2 has truth latent.

## What this means

**Both cells show "final-3 unanimous high-P" commit signature.** The
SHAPE of commit-confidence at substrate level is identical:
- P01: L29/L30/L31 all '8' at P > 0.92 — confident WRONG commit
- P05: L29/L30/L31 all '6' at P > 0.57 — confident RIGHT commit

The substrate cannot distinguish "I have the truth" from "I have a
confident wrong answer". Both produce the same final-3-unanimous
signature. The DIFFERENCE between confident-right and confident-wrong
is INVISIBLE at the substrate level.

## Implication for inference doctrine

silv's deeper fallacy named: **sampling instrumentation cannot fix
substrate-level confident-wrong**. Temperature, top_k, top_p, beam
search — all read the final L31 logits where P01 says `'8'` with P=0.984
and P05 says `'6'` with P=0.97. The presentation layer is identical;
the underlying correctness differs.

The ONLY way to discriminate confident-right from confident-wrong is
EXTERNAL VERIFICATION:
- sympy / programmatic verifier on the predicted answer
- Ensemble of proposers (if model A and model B both confidently say '6',
  trust higher; if they confidently disagree, neither trustworthy)
- Multi-precision crosscheck (4bit, BF16 commit to same digit → trust;
  different digits → uncertain)

LLM-as-verifier was already refuted in Conjecture #20 (anchoring bias).
The structural answer: external NON-LLM verifiers.

## Cross-substrate signature: commit confidence is uniform

The "calibration problem" in LLMs is often framed as "the model is
overconfident". What this experiment shows is sharper: the model is
JUST AS CONFIDENT when wrong as when right. The architectural commit
mechanism (L29-L31 unanimous-vote) doesn't track its own correctness.

Common-sense LLM inference protocols treat confidence as a signal
(use temperature to introduce uncertainty, use beam search to explore
alternatives, use top_p to limit confidence). These don't address the
CALIBRATION INVISIBILITY at substrate level.

The honest reading: a single LLM's commit confidence has near-zero
information about correctness. You either trust the model unconditionally
(brittle on hard tasks) or you bring in external verification (sympy,
second proposer, programmatic check). The "thinking" tokens, the chain
of thought, the temperature — all are presentation flourishes around
this structural fact.

## Files

- `codec_audit/forced_commit_substrate.py`
- `tmp/20260525_attention_inflight/forced_p01.json` (lock forced-commit
  data — wrong commit '8')
- `tmp/20260525_attention_inflight/forced_p05.json` (success forced-commit
  data — right commit '6')
- `tmp/20260525_attention_inflight/forced_commit_substrate.md` (this)

## Next falsifier

- Run same probe on rescue-succeeded cells (P02-P08): does forced-commit
  at chunk_2 ALWAYS extract truth on those cells? If yes, the rescue
  worked because truth was in the substrate; if no, the rescue
  relied on TEXT-RETRIEVAL of truth-shape-in-CoT.
- Cross-precision: does P01 BF16 forced-commit also commit '8' (showing
  the lock-substrate computation is precision-invariant)? Or does BF16
  produce different wrong-commit (precision-specific attractor)?
- Cross-model: does DS-R1-7B-BF16 forced-commit on P01 also commit
  confidently to some wrong digit? Or does the sharper commit-tier
  produce different behavior?
