# 17-probe session arc closure

silv 2026-05-25 final synthesis: the session's 17 continues produced
a complete characterization of the Qwen3.5-4B AIME 2026 P01-P10
failure-mode + rescue-mechanism within available data scope.

## The arc

| probe | finding | refute / corroborate? |
|-------|---------|-----------------------|
| 1 | L31 substrate truth latent | refuted (position artifact) |
| 2 | quantization helps rescue | refuted (runtime) |
| 3 | MLX library helps | refuted (prep-truncation) |
| 4 | last-number recency rule | refuted (1/18 exact match) |
| 5 | two-mode framework | incomplete |
| 6 | recency category-level | refuted on exact-emit |
| 7 | 10/10 bidirectional rule | refuted by matharena |
| 8 | arithmetic-form truth presence | did not fire |
| 9 | Mode B1/B2 sub-taxonomy | refuted (same loop) |
| 10 | loop pathology Mode B specific | refuted (universal 9/10) |
| 11 | cap as budget control | refuted (rescue-window) |
| 12 | truth_at < cap rule | **corroborated 3/3 caps** |
| 13 | truth_at < min(frac×cot, cap) | **corroborated 50/50 cell-configs** |
| 14 | cross-model generalization | UNTESTED (honest limit) |
| 15 | Mode 2 prior at failure boundary | **validated** (emit not in last-500) |
| 16 | problem-structure taxonomy | derived (3 classes) |
| 17 | taxonomy predicts matharena rates | **validated** (100% / 92.6% / 46.2%) |

11 refutations, 5 corroborations, 1 honest untested marker.

## The complete mechanism

```
                  problem statement → reasoning structure
                          │
              ┌───────────┴───────────┐
            single-track            multi-step or ambiguous
            (Class A)               (Class B/C)
                │                      │
                ▼                      ▼
        Qwen3.5-4B reasons        Qwen3.5-4B interpretation-locks
        derives truth in CoT      enters loop attractor
                │                      │
                ▼                      ▼
        no-box-commit pathology   truth NEVER in CoT
        loop attractor at end                      
                │                      │
                ▼                      ▼
        truth in pre-loop CoT     loop dominates whole CoT
                │                      │
                ▼                      ▼
     FORCED-COMMIT RESCUE         RESCUE FAILS (Mode 2 prior)
     truth_at < min(frac×cot, cap)
                │
                ▼
     Mode 1 text retrieval
     emit = truth (confident)
     sympy verifies
```

## Validated structural claims

1. **Cap mechanism**: `rescuable(cell, cap, frac) ⇔ truth_at < min(frac × cot_len, cap)` — 50/50 cell-config predictions perfect.

2. **Two modes at boundary**: Mode 1 (text retrieval) when prefix contains truth; Mode 2 (prior fallback) when prefix lacks truth. Both confident emissions, sympy discriminates.

3. **Problem-structure taxonomy**:
   - Class A (single-track): cap-rescue works
   - Class B (multi-step composition): K-sampling helps
   - Class C (ambiguous frame): needs multi-proposer + verifier

4. **Production stratification matches taxonomy**: Class A = 100% across 27 models; Class B = 92.6%; Class C = 46.2%.

## Where the 17 continues actually went

The session's escalation produced a complete TRAVERSAL of the rescue
mechanism surface, from substrate-layer (L31 latent claim refuted)
through text-rules (recency refuted multiple times) through CoT-
position rules (cap mechanism corroborated) through problem-structure
(taxonomy predicts production data).

Each layer refuted the previous as deeper inspection revealed the
real mechanism. The rule that ULTIMATELY held was the one operating
on:
- CoT-intrinsic deterministic positions (truth_at, cot_len)
- Continuous protocol parameters (cap, frac)
- A `min()` operation capturing the interaction

This is the substrate-level mechanism with maximum predictive power
within demonstrated scope.

## What the session did NOT produce

- Cross-model validation (would need DS-R1-7B own CoTs)
- Cross-corpus validation (would need AIME 2024/2025 CoTs)
- A novel mechanism beyond existing CLAUDE.md Conjecture #21/#23
- An OOM-class speedup (the broader directive in the initial prompt)

The session deeply re-validated and quantified existing project
doctrine rather than producing OOM-class new findings. The "doubt
EVERYTHING" prompt produced rigorous internal validation, not
external breakthrough.

## Closure recognition

The arc reaches saturation. Further continues without new data would
re-validate the same rule on the same 10 cells with diminishing
returns. The structurally appropriate next moves require:
- DS-R1-7B own AIME 2026 CoT cache → cross-model rule test
- AIME 2024/2025 corpus → cross-corpus
- Different architecture (Llama, Mistral, Phi) → cross-arch
- Production deployment + red-team adversarial data → external validation

Within the available data scope, the 17-probe arc has produced the
fullest characterization possible. Production-ready deployable form
matches CLAUDE.md Conjecture #21's existing architecture:
multi-proposer pool + per-problem sympy verifier, with the rescue
protocol as the per-proposer post-processing step.

## Files

- `tmp/20260525_attention_inflight/SESSION_ARC_CLOSURE.md` (this)
- 17 individual probe memos in same directory
- 23 commits in git log on `main` from 2026-05-25 session
