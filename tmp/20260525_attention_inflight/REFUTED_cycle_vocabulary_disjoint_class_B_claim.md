# SELF-REFUTATION: "Class B = truth never derived" claim from prior memo

silv 2026-05-25: my own claim refuted by checking earlier M1 bf16 CoT
positions. Per silv's "doubt every step" + "make glorious mistakes" —
this is exactly the kind of mistake that comes from inferring from a
narrow window.

## What I claimed (in cycle_vocabulary_disjoint_precisions.md)

> **Class B — Pre-truth interpretation paralysis (bf16)**
> - Truth (277) is NEVER DERIVED
> - Truth ABSENT from cycle vocabulary
> - The model is stuck PRIOR to derivation, doing semantic-ambiguity analysis
> - Inferguard rescue would FAIL (no truth in CoT to retrieve)

## What the data shows (counter-evidence)

### Counter-evidence 1: M1 bf16 cot at pos 4000

From `bufnicjxt.output` (bf16 probe run):
```
pos=4000 t=128.1s recent=' 10/5)(14/5) = (18/5)(14/5) = 252/25$.'
```

This is **the answer derivation**. P01 truth = m+n where m/n is the
distance. The computation `(18/5)(14/5) = 252/25` gives m=252, n=25,
m+n=277 (the truth). bf16 DERIVED truth around pos 4000.

### Counter-evidence 2: AMD bf16 forced-extract result

From `amd_forced_extract_bf16_FULL_results.json`:
- Model: Qwen3.5-4B BF16 (AMD pytorch, not M1 MLX)
- Prefix: 60% of CoT, capped at 16000 chars
- Result: **7/10 rescued correctly via forced-extract**
- P01 specifically: `predicted=277, is_correct=true, emit="277}.\n\nLet's double"`

If "truth was never derived" were true, AMD forced-extract would FAIL on
P01. It SUCCEEDED. Truth WAS in the CoT at the 60% prefix point.

## What the refutation reveals (the actual mechanism)

bf16 DOES derive truth (around pos 4000 in M1 greedy decode). The
INTERPRETATION-PARALYSIS cycle at pos 12000 is what the model does
AFTER having derived truth and then drifted past it into a different
problem (semantic disambiguation of the prompt language).

**Both 4bit AND bf16 derive truth in CoT**, but at different positions:
- 4bit: derives at ~pos 8K, cycles into post-truth meta at pos 12K (~4K distance)
- 8bit: derives at ~pos 6-8K, cycles at pos 12K (~4-6K distance)
- bf16: derives at ~pos 4K, cycles at pos 12K (**~8K distance — drifts FURTHER past derivation**)

The cycle vocabulary differs because the cycle is reached at a
DIFFERENT POINT IN THE COT's TRAJECTORY through cognitive subspaces.

## The corrected classification

| precision | truth derivation pos | cycle entry pos | distance | cycle content |
|-----------|---------------------|------------------|-----------|----------------|
| 4bit | ~8K | ~12K | 4K | post-truth meta (truth in vocab) |
| 8bit | ~6-8K | ~12K | 4-6K | post-truth meta + prompt-leak |
| bf16 | ~4K | ~12K | **8K** | interpretation-doubt (truth-DRIFTED-PAST) |

The disjoint cycle vocabularies are explained NOT by "truth never derived"
but by "cycle reached at different drift-distance from derivation point."

## Implications corrected

### My "Class A vs Class B" framing — REFUTED

There aren't two attractor classes by precision. There's a CONTINUUM:
all three precisions derive truth, all three cycle, but the cycle
ENTRY POINT relative to derivation differs. The cycle vocabulary
diversity is downstream of derivation-drift-distance, not of
whether-truth-was-derived.

### Inferguard rescue protocol — bf16 IS still rescuable

The AMD result confirms 7/10 cells rescued at 60% prefix on bf16.
The protocol works because truth IS in the CoT prefix at 60% — just
needs to use the prefix-position rather than the chronological-cycle-
position to extract.

### Cycle vocabulary disjoint — STRUCTURAL FACT REMAINS

The IoU = 0.000 finding between 4bit/8bit and bf16 cycle vocabularies
remains accurate. Just the INTERPRETATION was wrong:
- True: cycle vocabularies are disjoint
- False: bf16 cycle is "pre-truth"

The deeper mechanism: cycle vocabulary reflects what the model is
"talking to itself about" — a function of WHERE in CoT trajectory the
cycle forms. 4bit/8bit cycle forms close to derivation (so meta-comment
about the answer); bf16 cycle forms past derivation (so secondary
semantic analysis of the prompt language).

## silv's discipline validated

"Doubt every step" caught this. The 200-token cycle window I analyzed
was the window AT cycle entry; checking the FULL CoT trajectory
required examining pre-cycle positions. The 5-minute IoU analysis got
the structural finding right but the SEMANTIC INFERENCE wrong.

"Make glorious mistakes" — this is one. The vocabulary-IoU result
ships unchanged; the "Class B = pre-truth" framing is wrong; the
sharper picture (precision moves derivation-position, not derivation-
absence) is the actual finding.

## Files to read first

- This refutation: REFUTED_cycle_vocabulary_disjoint_class_B_claim.md
- Refuted memo: cycle_vocabulary_disjoint_precisions.md (kept as
  artifact; the IoU finding is correct, the Class A/B inference is not)
- Counter-evidence sources:
  - bufnicjxt.output (M1 bf16 probe pos 4000 derivation)
  - amd_forced_extract_bf16_FULL_results.json (7/10 rescue)
- Companion: precision_attractor_3axis_complete.md (also needs minor
  update — bf16 DOES derive, just drifts further)
