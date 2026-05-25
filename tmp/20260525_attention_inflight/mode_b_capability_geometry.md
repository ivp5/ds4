# Mode B has structural geometry — truth lives outside model's quantity space

silv 2026-05-25 continue#N+2: tested whether truth could be hiding in
non-string-search arithmetic forms (sqrt(N²), N/1, etc). Refutation
candidate did NOT fire — truth is genuinely absent in all forms tested.
The investigation surfaced what Mode B (capability failure) actually
looks like inside the model.

## Test outcomes

Both unrescuable cells:

| form tested | P9 (truth=29) | P10 (truth=156) |
|-------------|---------------|-----------------|
| Integer token (not part of bigger num) | 0 | 0 |
| sqrt(N²) form | 0 (sqrt(841)) | 0 (sqrt(24336)) |
| Fraction N/1 | 0 | 0 |
| Any embedding in larger digits | 0 substring | 0 substring |

Truth literally never enters either CoT in ANY form across 181K chars.
The 8th refutation candidate (string-search miss) does not fire.

## What the model DID compute — the geometry of failure

P9 (truth=29 = prime): CoT's top number tokens are 1, 2, 3, 4, 5, 6
(rampant base counting), then jumps to {15, 24, 48, 120, 144, 216,
360, 720}. **These are FACTORIALS and factorial-derived composites**
(5!=120, 6!=720, 6!/3!=120, 4!=24, 6!/2!=360). The model is in a
combinatorial-counting basin computing permutations.

Truth=29 is a **PRIME with no factorization**. Its only factors are 1
and 29. The model's quantity-space (factorials and their factor
products) cannot REACH 29 by any composition operation. The reasoning
path is structurally orthogonal to the truth.

P10 (truth=156 = 12 × 13): CoT's top numbers are {2, 65, 13, 56, 15,
14, 84, 33}. **The factor 13 appears 63 times.** The model is doing
geometry with sides 13, 14, 15. The truth 156 = 12 × 13 would require
the model to multiply 12 (which doesn't appear at all in the top
frequency) with 13.

**The model has half the truth.** 12 is missing from its reasoning. The
final multiplication step (12 × 13 = 156) is never composed.

## Mode A vs Mode B — three sub-modes

Refining the binary Mode A / Mode B from yesterday:

**Mode A (no-commit pathology, rescuable)**:
- Model derives truth in CoT (P06: 349 mentions)
- Pathological no-box-commit
- Rescue extracts via forced-commit preamble

**Mode B1 (quantity-space-orthogonal, unrescuable)**: P9 type
- Truth has properties outside the model's chosen reasoning basin
- P9: truth is prime, model is doing factorials → no path crosses 29
- Refinement: increase K-sampling at temperature > 0 to escape basin

**Mode B2 (composition-step-missing, unrescuable)**: P10 type  
- Model has the FACTORS but never performs the COMPOSITION
- P10: 13 appears 63 times, 12 doesn't appear at all, 156 = 12×13 missing
- Refinement: inject "what is X × Y?" prompt to force composition

The matharena cross-model distribution maps cleanly:
- P9 92.6% green: most production models on K=4 ESCAPE the factorial
  basin and find some path through 29 (likely via direct enumeration
  or algebraic substitution other than factorials)
- P10 44% green: composition step (factor × factor = product) is
  harder; even production models bisect

## Why this matters for production

Generic K-sampling helps Mode B1 (basin escape). It does NOT
necessarily help Mode B2 (composition gap) — if the model has a
SYSTEMIC bias toward forgetting one factor, more samples reproduce
the same forgetting.

Mode B2 fix is structural: force composition prompts. Not a generic
rescue protocol parameter; a problem-class-specific intervention.

## Files

- `tmp/20260525_attention_inflight/mode_b_capability_geometry.md` (this)
- Earlier in arc: `cross_model_p9_p10_distribution.md` (the 92%/44% framing)
- Cached CoTs: `tmp/20260524_quant_matrix/4bit/p09*.json`, `p10*.json`
