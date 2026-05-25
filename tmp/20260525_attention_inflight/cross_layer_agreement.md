# Cross-layer agreement on commit-position top_1

silv 2026-05-25: deployable narrow signal from L29 refutation.

## Setup

On P05 native success commit at `\boxed{` position, examine top_1
prediction at every layer. Commit-tier = L23-L31.

## Result

| Layer | top1 | P(top1) | rank('6') |
|-------|------|---------|-----------|
| L23   | '6'  | 0.006   | 0 (top-1) |
| L24   | 'مائة' | 0.0069 | 2 |
| L25   | 'مائة' | 0.0075 | 4 |
| L26   | 'مائة' | 0.0137 | 2 |
| L27   | '6'  | 0.019   | 0 |
| L28   | ' sixty' | 0.0548 | 1 |
| L29   | '6'  | 0.554   | 0 |
| L30   | '6'  | 0.999   | 0 |
| L31   | '6'  | 0.9998  | 0 |

**5/9 commit-tier layers** predict literal `'6'` at top_1.

L24-L26 detour to `'مائة'` (Arabic "hundred") — the substrate's
INTERMEDIATE representation explores multiple ENCODINGS of the answer:
- L23: digit '6' (initial commit)
- L24-L26: alternate encoding 'hundred' (positional thinking?)
- L27: back to '6'
- L28: English word ' sixty' (semantic encoding)
- L29-L31: unanimous '6'

This is a substrate-level finding: **commit installation is NOT
monotonic toward the final form**. The model considers multiple
encodings of the answer at intermediate layers before converging on
the final digit form.

## Deployable narrow signal

**Final-3 unanimous agreement**: at the emission position, read L29,
L30, L31 top_1. If all three predict the same digit token, commit is
confident.

On P05: L29='6', L30='6', L31='6' → unanimous → confident. P05 was
correct.

Untested on confident-wrong / no-commit cells. The cleanest cross-cell
falsifier would be:
- Take P10 (confident-wrong native), force-commit position, check
  final-3 agreement. If unanimous on wrong digit → "confident-wrong" =
  same substrate signature as "confident-right". If divergent → the
  wrong commit might be flagged before emission.
- Take a "rescue-success" cell, force-commit at the rescue position,
  check final-3 agreement. Should be unanimous on truth digit if
  rescue extracted truth.

## Why this matters for inference doctrine

silv's "sampling instrumentation fallacy" — sampling temperature/top_k
are post-hoc presentations of L31 logits. But the COMMIT-TIER STRUCTURE
(L23-L31 in Qwen3.5-4B) carries information BEYOND just L31. Reading
final-3 unanimous-agreement is a probe of the model's COMMITMENT shape
that the L31-only sample reader can't see.

The "final-3 unanimous" signal is essentially free at inference time
(needs the hidden states from L29/L30 anyway since the forward computes
them). Adding a check costs ~3 lm_head projections per emission position.

## Composition with L19 detector

- L19 attention growth detector: predicts commit-concentration "soon"
  (at chunk_2 of generation)
- Final-3 cross-layer agreement: confirms commit-confidence "now" (at
  the emission position)

Combined: a model that has been detected at L19 as "concentrating" and
has cross-layer agreement at emission → confidently committing (might
be right OR confidently-wrong; doesn't distinguish, but flags as commit
attempt).

The truth-vs-confident-wrong discrimination would need ANOTHER signal —
possibly comparing model's commit to a sympy verifier (deterministic) or
a second proposer (ensemble disagreement).

## Files

- `tmp/20260525_attention_inflight/lens_at_boxed_p05.json` (raw per-layer
  data on P05 commit position)
- `tmp/20260525_attention_inflight/cross_layer_agreement.md` (this)
