# Recency rule re-refuted at exact-prediction granularity

silv 2026-05-25 6th self-refutation: my "rescue emits LAST number-shape
in prefix" rule predicts behavior at CELL CATEGORY level but FAILS
at exact-number prediction granularity.

## Tests run

**Test 1**: "last 1-3 digit number in prefix" predictor across 18 (cell,cap) pairs
- Matches actual emit: 1/18 (only P05@16000 where last=65=truth)
- Most cells: last number was a random digit, NOT what model emits

**Test 2**: "last COMMIT-CONTEXTUAL number" predictor (regex: `m+n = N`,
`boxed{N`, `answer is N`, `= N.`, etc.):
- Matches actual emit: 5/14 cells (better than naive)
- P01, P04@cap=24000, P06@cap=16000, P07: rule predicts correctly
- P02, P05, P08, P10: rule misses; model emits something else

## What survives vs falls

**Survives at category level**: 
- Cells WITH truth-shape in CoT → rescuable (7-8/10 ceiling)
- Cells WITHOUT truth-shape → unrescuable (P09, P10)
- Cells where truth EVENTUALLY appears in CoT need cap to reach it (P04)

**Falls at exact prediction**:
- Specific NUMBER the substrate emits is NOT predictable by simple
  "last number" or "last commit-context number" regex
- The model's attention reads CoT with more nuance than regex
- Mode 2 prior fallback (P10) is genuinely model-internal, not
  text-retrievable

## Better hypothesis

The substrate emit at forced-commit is the LM's full attention readout
of the prefix — weighted by all positions and all heads. The "last
mention" intuition captures the dominant signal on many cells but
misses cases where:
- Multiple commit-contextual candidates compete (P02 has "= 62" and
  "= 56" both in early CoT; model may pick 62 because of upstream
  algebra structure, not pure recency)
- The model integrates over MANY mentions (P06 has 441 mentioned in
  many places; final emit reflects aggregate signal)
- Mode 2 prior overrides retrieval (P10's "168" comes from training
  imprint, no commit-context anywhere in CoT)

## Methodological lesson

A "rule" that predicts cell category correctly (rescuable vs not) is
NOT the same as a rule that predicts exact emit. Conflating these is
the seductive mistake.

The CELL-CATEGORY claim ("rescue ⇔ truth-shape in CoT") survives.
The EXACT-PREDICTION claim ("emit = last commit-contextual number")
is too strong.

## What this means for production

The deployable rescue protocol can predict rescuability at CELL-LEVEL
(does any truth-shape candidate exist?) but not at EXACT-EMIT level.
The production flow still works:
1. Find truth-shape candidates in CoT
2. Forced-commit at each candidate position
3. Sympy-verify each emit
4. Return any cert-passing candidate

The verification step IS necessary — the recency rule can't shortcut
it. My earlier framing implied recency could replace verification at
prediction step; it cannot.

## Files

- `tmp/20260525_attention_inflight/recency_rule_re_refuted.md` (this)
- Earlier files where the rule was claimed:
  `recency_rule.md`, `recency_rule_refined.md`, `recency_compare.py`
- The 1/18 vs 5/14 prediction match data is in this analysis
