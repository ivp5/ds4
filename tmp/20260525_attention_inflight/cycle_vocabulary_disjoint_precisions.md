# Cross-precision cycle vocabularies are DISJOINT — two attractor classes

silv 2026-05-25: cheap text-IoU analysis on the 3-precision cycle data
discovered a sharper claim than I made in `precision_attractor_3axis_complete.md`.

## The measurement

200 tokens captured at pos 12000 in greedy decode of Qwen3.5-4B on
AIME 2026 P01 at three precisions. IoU = |A ∩ B| / |A ∪ B| of n-gram
sets.

| n-gram | 4bit∩8bit | 4bit∩bf16 | 8bit∩bf16 |
|--------|-----------|-----------|-----------|
| 1-gram | 0.615 | 0.040 | 0.043 |
| 2-gram | 0.389 | **0.000** | **0.000** |
| 3-gram | 0.300 | **0.000** | **0.000** |

**bf16 shares ZERO bigrams or trigrams with either 4bit or 8bit cycle.**

## The sharper claim

There are at least TWO qualitatively distinct attractor classes for
Qwen3.5-4B on AIME P01 by precision:

### Class A — Post-truth meta-comment (4bit + 8bit)

Shared vocabulary: `{"The", "is", "277.", "I", "will", "write", "solution", "the"}`

Both 4bit and 8bit cycles express "the solution is 277, I will write
the solution" with minor rotation differences. They live in the SAME
attractor basin with similar surface vocabulary (30% trigram overlap).

This is the regime where:
- Truth (277) is DERIVED in the CoT
- Truth APPEARS in the cycle vocabulary
- The model "knows the answer" but can't escape the meta-cycle to commit
- Inferguard rescue PROTOCOL works (forced-commit retrieves 277)

### Class B — Pre-truth interpretation paralysis (bf16)

Unique vocabulary: `{"Tanya", "left", "park", "interpreted", "Wait,", "If", "departure", "school"}`

bf16 cycle expresses "If Tanya left road she's at park. If at park,
arrived. If arrived, not running. Wait, could be interpreted as..."
This is a COMPLETELY DIFFERENT semantic content type.

This is the regime where:
- Truth (277) is NEVER DERIVED
- Truth ABSENT from cycle vocabulary
- The model is stuck PRIOR to derivation, doing semantic-ambiguity analysis
- Inferguard rescue would FAIL (no truth in CoT to retrieve)
- Would need fundamentally different protocol: force INTERPRETATION RESOLUTION first

## The discontinuous jump 8bit → bf16

The period jump is also discontinuous:
- 4bit: 31 tokens
- 8bit: 32 tokens (+1 from 4bit)
- bf16: 90 tokens (+58 = 2.8× scale change)

Period AND vocabulary BOTH show qualitative-not-continuous shift at the
8bit→bf16 boundary. This refines codex H1853's "precision continuously
deforms attractor" — precision actually triggers DIFFERENT ATTRACTOR
CLASSES at qualitative threshold.

## Implication: attractor-class-aware rescue protocol

The inferguard `aime_rescue.py` works for Class A but would fail on
Class B. Different precisions of the same model on the same cell
require different rescue strategies:

| precision | attractor class | rescue protocol |
|-----------|-----------------|-----------------|
| 4bit | A (post-truth meta) | force-commit at any truncation; cap=12K |
| 8bit | A (post-truth meta) | force-commit at any truncation; cap=12K |
| bf16 | B (pre-truth paralysis) | force INTERPRETATION RESOLUTION first; then derive; then commit |

A multi-precision ensemble couldn't just take consensus — it would
need different rescue logic per precision. Or: degrade to 4bit/8bit
when bf16 enters Class B (unlikely to be useful for production
deployment since lower precision was supposed to be the degradation).

## Cross-cell prediction

If this 2-class structure holds across cells, the prediction is:
- Some cells will be Class A in all precisions (truth derivable; meta-commit fails)
- Some cells will be Class B at high precision only (interpretation paralysis
  blocks derivation at bf16; lower precision "lucks into" derivation)
- Some cells might be Class B at ALL precisions (universally non-derivable
  on this model — matharena's P9/P10 candidates)

Untested. Would require running each AIME cell at each precision and
classifying the cycle vocabulary. Cost: 3 precisions × 10 cells × ~5min
each = ~2.5 hours wall.

## What the naive engineer would have missed

Naive engineer: "bf16 is higher precision than 4bit, so bf16 cycle
should be a higher-fidelity version of 4bit cycle."

Reality: bf16 cycle is in a STRUCTURALLY DIFFERENT REGION OF MODEL'S
TRAJECTORY SPACE. 4bit lives in basin A, bf16 lives in basin B. No
amount of precision improvement crosses A→B in the gradual sense; it's
a regime transition.

This grounds silv's "doubt every step" doctrine and his "make glorious
mistakes" — my pre-experiment hypothesis was that bf16 would be a
sharper version of 4bit. Refuted; bf16 is a STRUCTURALLY DIFFERENT
failure mode.

## Files

- This memo: tmp/20260525_attention_inflight/cycle_vocabulary_disjoint_precisions.md
- Source data: cycle_period_{result, 8bit_result, bf16_result}.json
- Companion: precision_attractor_3axis_complete.md
