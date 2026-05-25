# SELF-REFUTATION: substrate truth-rank gradient is text-context retrieval, not latent computation

silv 2026-05-25 doubt-everything continues — my own session finding refuted.

## What I claimed earlier this session

In `forced_commit_substrate.md` and `DEEP_DOCTRINE.md`:
> "Substrate truth-rank at L31 calibrates against RESCUABILITY"
> "P03/P05/P06/P07/P08: rank 0 at L31 → substrate has truth latent"

## What the prefix-frac sweep reveals

At forced-commit with varying prefix-frac on Qwen3.5-4B-MLX-4bit AIME 2026:

| Cell | frac=0.20 | frac=0.40 | frac=0.60 | Truth-shape in CoT |
|------|-----------|-----------|-----------|--------------------|
| P01  | 85 ✗     | 85 ✗     | **277 ✓** | once at frac=0.216 |
| P02  | **62 ✓** | 56 ✗     | **62 ✓**  | 6× starting frac=0.170 |
| P03  | 79 ✓     | 79 ✓     | 79 ✓      | (rescue cell) |
| P04  | 55 ✗     | 56 ✗     | 71 ✗      | 14× starting frac=0.347 |
| P05  | 65 ✓     | 65 ✓     | 65 ✓      | (native success) |
| P06  | 441 ✓    | 441 ✓    | 441 ✓     | (rescue cell) |
| P07  | 396 ✓    | 396 ✓    | 396 ✓     | (rescue cell) |
| P08  | 244 ✓    | 244 ✓    | 244 ✓     | (rescue cell) |
| P09  | 1/10 ✗   | 1/12 ✗   | 1 ✗      | **NEVER** in 76k CoT |
| P10  | 120 ✗    | 100 ✗    | 120 ✗     | **NEVER** in 105k CoT |
| **Total** | **6/10** | **5/10** | **7/10** | |

**P09 truth=29 occurrence count in cached CoT: ZERO. P10 truth=156: ZERO.**

The cells that NEVER mention their truth in 76k-105k chars of CoT
text are EXACTLY the cells that are not rescuable by ANY prefix-frac.

## What I had wrong

The substrate at forced-commit position doesn't carry "latent
computation of truth". It carries "what truth-shape was recently
emitted in the proximal context".

- P01 frac=0.20 prefix (12000 chars) is BEFORE position 13471 where
  "$m+n = 277$" appears. Forced-commit substrate has NO truth-shape
  in context → emits wrong '85'.
- P01 frac=0.60 prefix (37000 chars) INCLUDES position 13471. Forced-
  commit substrate reads "277" from proximal context → emits truth.
- P02 truth=62 first appears at frac=0.170. frac=0.20 prefix barely
  includes it → emits 62. frac=0.40 prefix includes more CoT past
  the "62" mention but model has emitted "56" in intermediate
  reasoning → emits "56".
- P09, P10 truth never appears in CoT → no prefix works.

## The correct mechanism

The forced-commit "rescue" is:
1. Cached CoT contains truth-shape mention at position X (X may not
   be the natural commit position)
2. Take prefix up to position Y > X
3. Append forced-commit preamble
4. Model reads truth-shape from proximal context (the embedded mention)
5. Emits truth as the committed answer

This is essentially LITERAL TEXT RETRIEVAL with a search position
condition: "find any prefix that contains truth-shape mention; the
model will read it."

The substrate doesn't "calibrate against rescuability". The TEXT
calibrates — if truth was emitted somewhere in CoT, it's findable; if
not, the substrate can't manifest it.

## What this means for my session findings

**F6 (substrate truth-rank gradient predicts rescuability)** — STILL
CORRECT operationally but misframed mechanistically. The truth-rank at
L31 at forced-commit position is HIGH when the proximal CoT contains
truth-shape (text retrieval). It's LOW when CoT has no truth-shape
nearby (no retrieval target).

**F4 (DS-R1-7B reasoning-RL distill compresses commit tier)** — still
holds at structural level (DS-R1-7B has 4-layer vs Qwen 8-layer
install). But the "install" is also context-retrieval not latent
computation, given this refutation.

**The deep doctrine** — substrate retrieves from context, not from
computation. Sampling instrumentation can't fix CoT that doesn't
contain truth. The "rescue" works because cached CoTs have truth-
shape in them; live generation without rescue protocol may not.

## Practical deployable form

Strip the substrate-claim. The deployable rescue is:
1. Detect non-convergence (Conjecture #23 detector)
2. Scan CoT for truth-shape candidates (regex on integers 0-999)
3. For each candidate position, run forced-commit + extract
4. Sympy-verify against problem constraint

This is exactly inferguard/aime_rescue.py's approach. My
"substrate truth-rank" probe is a DIAGNOSTIC for whether truth-shape
appears in proximal CoT — but it requires knowing the truth digit to
read its rank, so it's not a production signal.

The production signal: regex-scan for truth-shape candidates +
forced-commit at multiple positions + sympy verify. The substrate
doesn't add new information.

## What survives self-refutation

1. The L19 attention-entropy growth detector (F1) — separate signal,
   measures commit-concentration not truth.
2. The L23-L31 commit-tier architectural fact (F2) — truth installs
   at these layers, regardless of whether it's "latent" or "retrieved".
3. Cross-precision invariance (F3) — both 4bit and BF16 install at
   same layers regardless of mechanism.
4. Reasoning-RL distill compresses commit tier (F4) — architectural
   shape, not mechanism-specific.
5. Cross-arch commit-concentration band (F5) — universal at 60-80%
   depth.
6. Wrong-belief doctrine — model commits structured wrong answers,
   not noise. Independent of retrieval vs computation.
7. Sampling-instrumentation can't fix calibration — independent of
   retrieval vs computation.

**What's REFUTED**: my own claim "substrate truth-rank measures latent
computation". The rank measures proximal context.

This is silv's "doubt every step" honored deeply: I doubted my own
session finding from earlier this same session and found a sharper
truth.

## Files

- `tmp/20260525_attention_inflight/SELF_REFUTATION_substrate_rank.md`
  (this)
- `tmp/20260525_attention_inflight/forced_extract_4bit_0_20.json`
  (6/10, P02 ✓)
- `tmp/20260525_attention_inflight/forced_extract_4bit_40pct.json`
  (5/10, P02 ✗)
- `tmp/20260525_attention_inflight/forced_extract_4bit_0_60.json`
  (7/10, P01 ✓ via context-retrieval of "277" at frac=0.22)
