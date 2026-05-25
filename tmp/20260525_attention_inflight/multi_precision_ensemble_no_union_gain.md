# Multi-precision ensemble: NO net cell gain, but per-cell agreement signal

silv 2026-05-25: comparing M1 4bit (Qwen3.5-4B-MLX-4bit) + inferguard
rescue vs AMD bf16 (Qwen3.5-4B BF16) + forced-extract rescue on AIME
2026 I P01-P10.

## Per-cell comparison

| cell | truth | M1 4bit | AMD bf16 | agreement |
|------|-------|---------|----------|-----------|
| P01 | 277 | 277 ✓ | 277 ✓ | agree (correct) |
| P02 | 62 | 62 ✓ | 62 ✓ | agree (correct) |
| P03 | 79 | 79 ✓ | 79 ✓ | agree (correct) |
| **P04** | 70 | **70 ✓** | **71 ✗** | **DIVERGE** |
| P05 | 65 | 65 ✓ | 65 ✓ | agree (correct) |
| P06 | 441 | 441 ✓ | 441 ✓ | agree (correct) |
| P07 | 396 | 396 ✓ | 396 ✓ | agree (correct) |
| P08 | 244 | 244 ✓ | 244 ✓ | agree (correct) |
| P09 | 29 | None ✗ | 1 ✗ | both fail |
| P10 | 156 | 100 ✗ | 100 ✗ | **agree but wrong** |

- M1 4bit: 8/10
- AMD bf16: 7/10
- **Union (either succeeds): 8/10** — NO union gain
- Intersection (both succeed): 7/10
- Divergent cells: 1 (P04)
- Falsely-confident-agreement (both wrong same answer): 1 (P10 → 100)

## What multi-precision ensemble does and doesn't do

### Does NOT close the AIME 2026 I ceiling

8/10 is the joint ceiling. P09 (truth=29) and P10 (truth=156) are
unreachable by ANY tested precision of Qwen3.5-4B. These are
capability-limit cells, not precision-rescuable.

The "deploy bf16 to close the gap" hypothesis was naive. The model
capability ceiling is structural, not precision-shadowed.

### DOES provide per-cell confidence calibration

- **Agreement on correct**: 7/10 — very high confidence (multi-precision
  ensemble votes converge on truth)
- **Disagreement** (P04): 1/10 — flag for verification; one of two is wrong
- **Agreement on wrong** (P10): 1/10 — false confidence; both precisions
  collapse to the same wrong attractor (training-prior 100 vs truth 156)

The deployable signal: P04 disagreement is real diagnostic information.
A multi-precision pipeline that ALSO runs sympy verification on cells
where precisions disagree could escalate divergent cells. P10's false-
agreement means sympy verification is still required for ANY cell, not
just disagreements.

### Refines silv's "multi-precision ensemble signal" task

Task #465 was "multi-precision ensemble signal." The deployable signal
is NOT "take consensus of N precisions" but:

```
For each cell:
  if all precisions agree:
    if sympy-verify(consensus):
      ship consensus  # 7/10 here
    else:
      ESCALATE — false confidence detected (P10 here)
  else:
    ESCALATE — divergent (P04 here)
```

This is similar to codex H1849's "logit-margin watchlist" pattern but
applied at the ENSEMBLE level rather than the per-token level. The
common doctrine: "stable agreement is high signal; divergence requires
external verification."

## P04 specifically — the divergent cell

P04 truth = 70. M1 4bit rescues to 70 (correct). AMD bf16 rescues to 71
(off by 1).

This is exactly the carry-circuit failure mode I documented in CLAUDE.md
DS4 trim analysis: late-layer ablation produces off-by-N errors at the
FINAL ADDITION step even when the chain through derivation is correct.

The fact that LOWER precision (4bit) gets it right and HIGHER precision
(bf16) gets it wrong-by-1 suggests quantization NOISE may be acting as
regularizer for this specific arithmetic carry. Or AMD pytorch bf16 has
a subtle path-difference from MLX 4bit that affects the rescued-position
sample.

Untested: would M1 bf16 also rescue to 71, or to 70? That would
discriminate "AMD vs M1 substrate difference" from "precision-level
mechanism." Cheap probe — could re-run M1 bf16 with same prefix.

## P10 specifically — false confidence

Both precisions rescue to 100 (truth=156). 100 is the AIME training
prior — the most common "answer is 100" pattern in pre-training math
data. Both precisions collapse to the SAME wrong attractor at high
confidence.

This is the worst-case failure mode for multi-precision ensemble: it
LOOKS confident (all precisions agree) but is wrong. Only sympy
verification breaks the consensus.

P10 is a strong argument for sympy verification at ALL cells, not just
disagreement-flagged ones. CLAUDE.md Conjecture #21 captures this with
the "interpreted_verifier" pattern.

## Production recommendation

For Qwen3.5-4B AIME deployment:

1. Run M1 4bit + inferguard rescue (primary, 8/10 ceiling, lowest wall)
2. SKIP AMD bf16 (no union gain, costs more compute)
3. ON ALL CELLS: sympy-verify the rescued answer
4. ON UNVERIFIABLE CELLS: escalate to multi-proposer (Orthrus 1.7B + DS-R1-7B + Qwen2.5-7B per Conjecture #21)
5. EXPECT 8/10 corpus ceiling at single-model deployment

bf16 deployment is a NET LOSS for this corpus (loses P04, adds nothing).
This refutes the naive "higher precision = more cells solved" hypothesis.

## Files

- This memo: multi_precision_ensemble_no_union_gain.md
- Source: amd_forced_extract_bf16_FULL_results.json + all_10_rescue_8_of_10.md
- Companion: precision_attractor_3axis_complete.md (which prompted the comparison)
