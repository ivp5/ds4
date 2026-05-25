# Wandering CoT doctrine — rescue must hit the RIGHT mention

silv 2026-05-25 OOM-accuracy probe at P04 anomaly.

## The wrong-then-revise pattern

P04 (truth=70) cached CoT contains a wandering mid-reasoning trajectory:

**Early CoT (frac 0.216-0.338)**: model computed `75 - 4 = 71`, mentions
'71' 13× across this range.

**Later CoT (frac 0.347-0.425)**: model REVISED to `74 - 4 = 70`, mentions
'70' 14× across this range.

The truth IS in the CoT — but only after the revision at frac=0.347+.

## Why P04 fails rescue at low prefix-cap

At prefix-frac=0.60 with n_chars_cap=16000 (the session's standard):
- P04 response = 58075 chars
- prefix = min(0.60 * 58075, 16000) = **16000 chars**
- 16000 / 58075 = frac=0.276 of response
- At frac=0.276, only '71' (wrong) mentions exist; '70' first appears at frac=0.347

So forced-commit at prefix=16000 reads '71' from proximal context and
emits 71. The substrate CORRECTLY retrieves the most recently mentioned
answer-shape — which happens to be the WRONG ANSWER the model computed
before revising.

## Predicted fix: raise n_chars_cap > 21000

If cap > 20149 (where '70' first appears), prefix includes the revision,
and substrate emits truth=70.

## The doctrine refinement

The rescue protocol works when:
1. Truth-shape mention exists in cached CoT (P09/P10 fail this)
2. Prefix-frac × cap includes that mention (P04 fails this at cap=16000)
3. Proximal-context retrieval at forced-commit reads the truth-shape token

The "rescue ceiling 7/10" in my session was at cap=16000. With cap=40000
(test pending), the ceiling could rise to 8/10 because P04 truth=70 is
at char 20149.

P09 (truth=29 never in CoT) and P10 (truth=156 never in CoT) remain
unrescuable at any cap.

## Wandering as a signature of reasoning-RL training

The "wrong-then-revise" pattern is exactly what reasoning-RL training
produces: "Wait, let me check..." → revise. The model EXPLORES wrong
hypotheses then corrects. This is how DS-R1-Distill-Qwen-7B and
similar reasoning-RL models work.

The rescue protocol EXPLOITS reasoning-RL's verbose self-correction:
the cached CoT contains both wrong intermediate and right revision.
Rescue picks up whichever the prefix-cap reaches.

**Operational implication**: rescue should DEFAULT to high prefix-cap
(e.g., cap=full-response-length, no truncation). My initial 16000-char
cap was a measurement artifact; the production rescue should not
artificially truncate.

This composes with the "more CoT can hurt" finding: P02 at frac=0.40
emits wrong '56' because intermediate wrong reasoning came AFTER the
'62' truth mention. P02 at frac=0.20 (early) and frac=0.60 (after the
intermediate-wrong but with more '62' mentions later) BOTH rescue.

The cleanest rescue protocol: scan FULL response for all truth-shape
candidates, take the LAST occurrence of each, run forced-commit at
each, sympy-verify, return any cert-passing candidate.

This is essentially inferguard/aime_rescue.py's approach. My session's
"substrate truth-rank" framing was misnomer; the production protocol
was right all along.

## OOM-accuracy probe insight

The runtime_divergence_probe revealed per-layer cosine 0.9999+ across
MLX-BF16 vs HF-BF16 for L0-L30. This is **5 nines of agreement** —
the substrates are essentially identical. The 4th-level refutation
this session (rescue is runtime-invariant) was confirmed at the
sub-bit level.

OOM-better deployable: focus engineering on **CoT truth-mention
detection** (regex, semantic, or specialized classifier), NOT on
runtime tuning. The substrate-rescue mechanism is solved; the
context-search heuristic is where wins remain.

## Pending verification

Run forced_extract with n_chars_cap=40000 on all 10 cells. Predict:
- P04 (truth=70 at char 20149): rescues → 8/10 ceiling
- P09 (no truth): still fails
- P10 (no truth): still fails

If P04 doesn't rescue at cap=40000, then the rescue mechanism has
ANOTHER constraint I haven't identified (e.g., recency vs. specific
phrasing context).

## Files

- `tmp/20260525_attention_inflight/wandering_cot_doctrine.md` (this)
- pending: `tmp/20260525_attention_inflight/forced_extract_4bit_cap40k.json`
