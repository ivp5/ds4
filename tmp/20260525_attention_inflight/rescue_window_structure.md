# Rescue window structure — truth-emit < cap < loop-onset

silv 2026-05-25 continue: measured first-truth-position and loop-onset
position per cell. The rescue window has clean structure.

## Per-cell positions

| cell | truth | cot_len | first truth_at | loop_onset | rescue_window |
|------|-------|---------|----------------|------------|---------------|
| P01  | 277   | 62257   | 13,471         | 14,500     | tight (1k)    |
| P02  | 62    | 73840   | 12,533         | (enum loop) | medium       |
| P03  | 79    | 113529  | 5,485          | 36,000     | wide (31k)    |
| P04  | 70    | 58075   | 20,149         | (36-line loop late) | inverted! |
| P05  | 65    | 33699   | 5,349          | NONE (clean commit) | unbounded |
| P06  | 441   | 75527   | 6,707          | 23,000     | wide (16k)    |
| P07  | 396   | 52516   | 6,432          | 25,000     | wide (19k)    |
| P08  | 244   | 68435   | 4,795          | 19,500     | wide (15k)    |
| P09  | 29    | 76305   | absent         | (early loop) | none         |
| P10  | 156   | 105074  | absent         | 18,000     | none          |

## P04 is the EDGE CASE

P04 has truth_at = 20,149. **This is ABOVE the standard cap=16,000.**
The truth derivation in P04 happens AFTER 20k chars. Cap=16k would
TRUNCATE BEFORE the truth-emit position, defeating rescue.

This matches CLAUDE.md Conjecture #23 corpus-scale rescue corroboration
2026-05-24, where P04 rescue at cap=24,000 (not 16,000) succeeded.

The standard cap=16k captures 6/7 of Mode A rescuable cells. P04 needs
cap=24k+. Dynamic cap (set per-cell) would resolve this.

## The "rescue window" mechanism

For Mode A cells, three structural positions in the CoT:
1. **Reasoning prefix** (chars 0 to truth_at): model derives toward truth
2. **Truth-emit position** (truth_at): truth-shape first appears in CoT
3. **Loop-onset position** (loop_onset): model enters infinite repetition

The rescue window is [truth_at, loop_onset]. Forced-commit at any cap
in this window:
- Prefix contains truth (so model has it latent)
- Prefix does NOT contain late loop (so attention recency points to
  productive reasoning, not loop pattern)

Per-cell optimal cap = anywhere in this window. Standard cap=16k
works for 6/7 Mode A cells; P04 needs larger.

## Dynamic cap deployable refinement

Per cell, set cap = max(16k, truth_at + 1000) AND cap < loop_onset.

Without ground-truth knowledge, can approximate via:
1. Stream the CoT generation
2. Detect loop-onset via Conjecture #23 reframe-density detector
3. Cap immediately before detected loop entry
4. Rescue at that cap

This avoids both failure modes:
- Cap too small (P04): truth not in prefix → rescue fails
- Cap too large: loop dominates attention → rescue gets garbage

## What this clarifies about the 10-refutation arc

Previous picture: cap is loop-escape (10th memo).
Refined picture: cap is the RESCUE WINDOW BOUNDARY.

The window has TWO boundaries (truth_at lower, loop_onset upper).
Cap must be in between. Fixed cap=16k is the median-fit; doesn't
adapt to cells with high truth_at (P04) or early loop_onset
(P09/P10 where it's zero by definition).

Mode B fails not because "no truth" alone but because the truth
boundary (truth_at) is at +∞ — the model never derives truth, so
no valid rescue-window cap exists.

## Engineering closure

The deployable rescue protocol has TWO levers:
- Cap parameter: should be dynamic per-cell, set by loop-onset detection
- Sample count K: addresses Mode B1 (basin escape via temperature)

Mode B2 (composition gap, if it exists separate from loop-attractor)
would need a third lever: composition-prompt injection. Earlier memo
collapsed B2 into loop-attractor too; deeper structural analysis on
more cells needed to confirm.

## Files

- `tmp/20260525_attention_inflight/rescue_window_structure.md` (this)
- Earlier: `loop_pathology_universal_cap_mechanism.md` (10th refutation)
- Earlier: `mode_b_is_loop_attractor.md` (9th refutation)
- Cached data: `tmp/20260524_quant_matrix/4bit/p*.json`
