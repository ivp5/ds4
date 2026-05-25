# L29 confidence-trajectory REFUTATION + cross-layer agreement direction

silv 2026-05-25: continue queued tasks. The L29 confidence signal was
named in SESSION_SYNTHESIS open follow-ups. Test:

**H_DECIDE_BEFORE_EMIT** (formulated this session): L29 P(top_1) rises
SUSTAINED above 0.5 BEFORE the model emits `\boxed{N}` — giving an
early-warning of imminent commit.

## Result on P05 native success (truth=65, response 33699 chars)

L29 trajectory at every 200th token across full response:
- First sustained-high L29 P(top_1)>0.5: char_pos=608 (very early in CoT)
- First `\boxed{` emission: char_pos=33696 (at end)
- Gap: **33088 chars between "sustained high L29" and actual commit**

L29 P(top_1) is HIGH at MANY natural-language tokens throughout the CoT:
- pos 200: P=0.99 on `'$` (closing LaTeX math)
- pos 400: P=0.90 on `' ='` (variable assignment)
- pos 1000: P=0.98 on `' e'`
- pos 4200: P=0.99 on `' them'`
- pos 9200: P=0.9998 on `"'^"` (LaTeX exponent)

**REFUTED**: L29 P(top_1) > 0.5 is not a "commit-imminent" signal. It's
a "next-token-is-predictable-from-LM-history" signal — fires on natural
language continuations as well as on commits.

## What this teaches

silv's sampling-instrumentation fallacy applies AT THE PROBE LEVEL too.
I treated L29 P(top_1) as a global signal of substrate-decision, but
the substrate's "decisions" are CONTEXT-CONDITIONAL on every position.
The model predicts most tokens with high confidence (that's what
language modeling IS); the commit positions are NOT structurally
different at the L29 measurement.

## The narrower deployable still survives

What would actually work: **L29 P(top_1) at positions where the model
is ABOUT TO EMIT a digit** (next token will be a digit class). At those
positions specifically, L29 P(top_1) on the predicted digit IS the
substrate's commitment.

Cheaper alternative: **cross-layer agreement on top_1 digit at the
emission position**. On P05 \\boxed{ commit, all 8 commit-tier layers
(L23-L31) predicted `'6'` as top_1 (with L29 picking ` sixty` synonym).
Agreement count ≥ 7/8 = strong commit; disagreement < 5/8 = uncertain
commit.

This is a single-forward, single-position probe — runs at the actual
emission position to predict confidence.

## Architecture comparison

DS-R1-7B has SHARP L24→L27 install (4 layers); Qwen3.5-4B has SMOOTH
L23→L31 install (8 layers). For DS-R1-7B, the cross-layer-agreement
metric would have FEWER layers to vote (4 vs 8) — possibly less robust
but faster.

The "L_commit_install_width" is itself a substrate property worth
tracking.

## Files

- `codec_audit/l29_trajectory.py`
- `tmp/20260525_attention_inflight/l29_trajectory_p05.json`
- `tmp/20260525_attention_inflight/l29_trajectory_p05_full.json`
