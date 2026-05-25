# Recency rule — rescue emits the LAST number-shape in prefix

silv 2026-05-25 OOM-accuracy refinement of the rescue mechanism.

## The rule

At forced-commit position, the substrate emits the digit at the start
of the MOST RECENT 1-3-digit number mention in the prefix context.

## Evidence across cells × caps (recency predictions)

| Cell | Truth | Wrong | Cap | Last truth pos | Last wrong pos | Predicted | Actual |
|------|-------|-------|-----|----------------|----------------|-----------|--------|
| P01 | 277 | 85 | 12000 | -1 | 3232 | 85 | 85 ✓ |
| P01 | 277 | 85 | 16000 | 13471 | 3232 | 277 | 277 ✓ |
| P02 | 62 | 56 | 12000 | -1 | -1 | tie/prior | 56 (prior bias) |
| P02 | 62 | 56 | 16000 | 15112 | -1 | 62 | 62 ✓ |
| P02 | 62 | 56 | 24000 | 22265 | -1 | 62 | (pending) |
| P02 | 62 | 56 | 32000+ | 24818 | 24548 | 62 (recent) | (pending) |
| P04 | 70 | 71 | 16000 | -1 | 13406 | 71 | 71 ✓ |
| P04 | 70 | 71 | 24000 | 22693 | 22333 | 70 | (pending) |
| P04 | 70 | 71 | 32000+ | 26894 | 26017 | 70 | (pending) |

For all cases where one number-shape is in prefix and the other isn't,
the present one wins. When BOTH are in prefix, the MORE RECENT wins.
When NEITHER is in prefix, the model falls back to its prior — which
for AIME-shape question is often a wrong canonical (P02→56, P04→55).

## Refines previous doctrines

1. The "rescue ⇔ truth-shape in prefix context" doctrine is correct
   but needed sharpening. **Rescue ⇔ truth-shape is the LAST number
   in prefix context.**

2. The "more CoT can hurt" finding on P02 (frac=0.20 ✓, frac=0.40 ✗,
   frac=0.60 ✓) now mechanistically explained:
   - frac=0.20 cap=16000: only '62' (early); recency = truth → ✓
   - frac=0.40 cap=12000: neither in prefix; prior fallback → emit '56'
   - frac=0.60 cap=16000: only '62'; recency = truth → ✓

3. The wandering CoT doctrine maps to recency: model emits wrong
   first, revises to truth. Cap that catches the REVISION wins truth.

## Operational implication

The production rescue should iterate the cap to maximize the
"last-truth-shape-mention" position relative to the forced-commit
position. Implementation:

1. Find ALL truth-shape candidates in cached CoT (regex 1-3 digit)
2. For each candidate, find its LAST position in CoT
3. For each (candidate, last_pos), construct prefix = full CoT up to
   last_pos + epsilon
4. Force-commit at that prefix, sympy-verify
5. The candidate whose last_pos is AFTER any competing mention wins

This is more sophisticated than fixed cap. Composes with
inferguard/aime_rescue.py's multi-position scan.

## OOM-finer accuracy gain

Earlier I measured rescue ceiling as 7/10. Sharpened recency rule
predicts ceiling rises to **8/10 at cap=24000** (P04 rescues since
'70' last at 26894 > '71' at 26017 — both in 24000-char prefix).

Test pending; result will validate or refute the recency rule.

If refuted, the rule has more nuance (maybe POSITION RELATIVE TO
COMMIT, not absolute recency). If corroborated, the rescue ceiling
on AIME 2026 corpus rises to 8/10 — the cells without truth-in-CoT
(P09, P10) remain the only fundamental failures.

## Speedup opportunity

The recency-rule deployable doesn't need MULTIPLE forced-commit forwards.
Instead:
1. ONE forward at the LAST truth-shape mention position
2. ONE sympy verification

vs. multi-prefix scan at K different caps × forced-commits = K forwards.

For K=4 prefix-frac sweep, this is 4× speedup. Per cell, forced-commit
forward is ~10s on M1 MLX, so saving 30s per cell × 10 cells = 5 min
per AIME 2026 run.

Combined with AMD's 3-7× speedup on the underlying forward (BF16 HF
on RX 7900 XTX vs M1 MLX), production rescue can be 12-28× faster than
the current default cap=16000 multi-prefix-sweep approach.

## Files

- `tmp/20260525_attention_inflight/recency_rule.md` (this)
- pending: `tmp/20260525_attention_inflight/forced_extract_4bit_cap24k.json`
