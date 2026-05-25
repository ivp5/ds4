# 10th refutation: loop-pathology is UNIVERSAL; cap-truncation is loop-escape

silv 2026-05-25 continue: inspected end-state of ALL 10 cached CoTs.
9/10 are in loop attractors. P05 is the only clean derivation+commit.
Prior understanding of WHY rescue works was incomplete.

## Data

End-state line-repetition counts in last 1500 chars per cell:

| cell | loops? | max-line repeats | loop snippet |
|------|--------|-------------------|--------------|
| P01  | YES    | 3                | "Wait, $t_T = 2 + t_J$ means..." |
| P02  | (enumerative) | 1 (different shape) | "Could there be a case where $x_i$ is 1123? No. ...1124? No. ...1125? No." |
| P03  | YES    | 5                | "The problem says 'placed on top of the disk at P'." |
| P04  | YES    | **36**           | "Wait, $a=1, b=3 \\implies n = 1+3+3 = 7$." |
| P05  | NO     | 1                | (clean commit, ends with `\\boxed{65}`) |
| P06  | YES    | 10               | "Let's check if $x_1 x_2$ could be something else." |
| P07  | YES    | 19               | "No, (1, 1, 1, 1, 1, 1) is 6 parts." |
| P08  | YES    | 12               | "Wait, I listed it as case 7." |
| P09  | YES    | **36**           | "Wait, if $p_2 = 4$, then $X_4 = r$." (the literal Mode B P9 loop) |
| P10  | YES    | 14               | "Let's assume the condition is: A'C' perpendicular B'C'? No." |

**9/10 cells have terminal loop pathology.** Only P05 (the natural
commit case) breaks free.

P02's loop is ENUMERATIVE — the line varies (1123, 1124, 1125) so
my line-counting heuristic missed it, but it's still a loop attractor
with content-incrementing variation. Same mechanism, different
surface.

## Refined unified picture

Three categories, all unified under one mechanism (Conjecture #23
loop-attractor + truth-derivation timing):

**Category 1 (P05)**: derive truth → commit cleanly. No loop. No
rescue needed.

**Category 2 (P01, P03-P08; rescuable Mode A)**: derive truth →
enter loop → exhaust budget. Truth is in CoT BEFORE the loop section.
Rescue at proximal cap=16k truncates the late loop, queries forced-
commit position. Model's pre-loop state has truth latent. Extracts.

**Category 3 (P09, P10; unrescuable Mode B)**: enter loop BEFORE
deriving truth. Truth never appears in CoT (zero occurrences).
Rescue extracts garbage because the proximal state has no truth.

## The cap mechanism reframed

Yesterday's understanding: cap=16k is a length budget for the rescue
prompt.

Today's understanding: cap=16k is the LOOP-ESCAPE mechanism. The
truncation removes the late-position loop section that would
otherwise dominate the model's emission via attention recency.

- At full length (e.g., 76k chars for P9): the loop section dominates
  attention; the model's emission is locked into the loop pattern.
- At cap=16k: the loop section is TRUNCATED OFF the prefix. The
  forced-commit position sees only the pre-loop reasoning state. If
  truth was derived before that cutoff, the model emits it.

This explains why cap=16k works for P01-P08 but not P09/P10:
- P01-P08: truth derived in first 16k chars; loop happens later → cap=16k
  captures pre-loop state with truth → rescue extracts.
- P09/P10: model enters loop early (interpretation lock) → at cap=16k,
  prefix is already 100% loop content → rescue extracts garbage.

## Why matharena's 92% green on P9 still works

K=4 sampling at temperature > 0 escapes the initial interpretation
lock STOCHASTICALLY. Some of those 4 samples don't enter the loop
at all → derive truth → commit. The cap-truncation rescue applied to
those samples extracts truth.

For my single greedy run: hit the loop deterministically. Cap-rescue
doesn't help. Production K-sampling does.

## What does this refute?

The earlier framing of why rescue works was wrong by omission.
"Forced-commit extracts the latent truth at proximal CoT" → correct
but missed that the FORCED-COMMIT position must AVOID the loop section
that would otherwise dominate via attention recency.

The cap parameter is THE critical lever. Setting cap=∞ (full CoT) would
defeat rescue on P01-P08 even though those cells have truth in CoT —
because the full prefix's terminal loop dominates the model's emission
direction.

This means the deployable rescue's cap value should NOT be tied to
"how much context can fit" but to "where does the loop start in this
specific cell's CoT." A loop-detector-based cap would be sharper than
the fixed 16k cap.

## The 10-refutation chain

1. Substrate truth latent (refuted: position artifact)
2. Quantization helps rescue (refuted: runtime)
3. MLX library helps (refuted: prep-truncation)
4. Recency exact-prediction (refuted)
5. Two-mode framework (incomplete)
6. Recency category-level (refuted)
7. 10/10 bidirectional rule (refuted by matharena)
8. Arithmetic-form truth (did not fire, surfaced sub-taxonomy)
9. Mode B sub-taxonomy (refuted; same loop-attractor mechanism)
10. Loop pathology specific to Mode B (refuted; UNIVERSAL on 9/10 cells)

Doctrine extension: **the rescue protocol's cap parameter is loop-
escape, not budget control**. The mechanism is "truncate before the
loop entry" — and the appropriate cap is per-cell, set by where the
loop starts, not by a fixed character count.

## Files

- `tmp/20260525_attention_inflight/loop_pathology_universal_cap_mechanism.md` (this)
- Refuted: `mode_b_is_loop_attractor.md` (which claimed loop was a Mode B feature; it's universal)
- Cached CoTs inspected directly: `tmp/20260524_quant_matrix/4bit/p*.json`
