# 9th refutation: Mode B1/B2 sub-taxonomy WRONG — both are loop-attractors

silv 2026-05-25 continue: inspected end-state of P9 and P10. Both are
in IDENTICAL loop-attractor pathology, not the distinct B1/B2 modes I
hypothesized.

## End-state evidence

**P9 (truth=29) last 2000 chars**:
```
Wait, if $p_2 = 4$, then $X_4 = r$.
Wait, if $p_2 = 4$, then $X_4 = r$.
Wait, if $p_2 = 4$, then $X_4 = r$.
[× 100+ identical repetitions]
```

**P10 (truth=156) last 2000 chars**:
```
*   Let's assume the condition is: The segment $A'C'$ is perpendicular to the segment $B'C'$? No.
*   Let's assume the condition is: The segment $A'C'$ is perpendicular to the segment $B'C'$? No.
[× 100+ identical repetitions]
```

Both cells hit the max-token budget INSIDE an infinite loop attractor.
This is Conjecture #23's pathology in its purest form: low-entropy
deterministic repetition.

## Why my Mode B1/B2 framing was wrong

I wrote (8th memo): "Mode B1 quantity-space-orthogonal" because P9's
top numbers were factorial-derived; "Mode B2 composition-gap" because
P10 has factor 13 but never 12. **Both framings read deep structure
into surface artifacts.**

The factorial-counting (P9) and the perpendicularity-loop (P10) are
artifacts of the model getting STUCK in its initial interpretation
direction. The model didn't FAIL to reach truth because of quantity-
space; it failed because it never escaped the first reasoning frame
to even start computing toward truth.

P9 stuck on the wrong problem interpretation: `p_2 = 4, X_4 = r` is
likely a misreading of the problem's structure. The model picked an
interpretation, locked in, looped.

P10 stuck on interpretation ambiguity: `A'C' ⊥ B'C'?` keeps being
proposed and rejected. The model can't commit to an interpretation
to proceed past it.

**The mechanism is the same**: loop-attractor entered after
interpretation lock; model emits same sentence forever; budget
exhausts before any productive computation reaches truth.

## Refined Mode B (single mode, not two)

Mode B (loop-attractor failure, unrescuable):
- Model enters low-entropy repetition before deriving truth
- Loop sentence varies by cell (interpretation-specific) but mechanism
  is identical: deterministic basin attracts emission
- Conjecture #23 detector fires at >3 reframes in budget
- Force-commit rescue extracts whatever is at \boxed{ position, but
  the model never derives truth, so the rescue returns garbage

## Why matharena's 92% green on P9 vs my 0% in single sample

Production K=4+ sampling at temperature > 0 GENERATES STOCHASTIC
DIVERGENCE from my greedy single sample. Some of those 4 samples
escape the initial interpretation-lock that triggers the loop.

P9 92.6% means: across K samples + sampling RNG, at least one sample
escapes the loop AND derives 29 AND commits to it. P10 44% means:
escape probability is lower (interpretation-ambiguity is genuine
problem-structure, not just initial-frame lock).

My Mode B1/B2 framing predicted: "K-sampling fixes B1 (basin escape)
but not B2 (composition gap)." Wrong — the LOOP-ATTRACTOR mechanism
makes K-sampling helpful for BOTH, just at different rates. P9's
basin is shallower (92% escape); P10's is deeper (44% escape) because
the problem statement itself is ambiguous.

## The 9th refutation in chain

The arc:
1. Substrate truth latent (refuted: position artifact)
2. Quantization helps rescue (refuted: runtime)
3. MLX helps (refuted: prep-truncation)
4. Recency rule (refuted: 5/14 exact)
5. Two-mode framework (incomplete)
6. Recency at category-level (refuted on exact-emit)
7. 10/10 bidirectional rule (refuted by matharena)
8. Arithmetic-form truth (did not fire, surfaced Mode B sub-taxonomy)
9. Mode B sub-taxonomy (REFUTED — both cells are same loop-attractor mechanism)

Each refutation required deeper reading of the actual data. My 8th
memo introduced a sub-taxonomy from FREQUENCY COUNTS of digits;
inspecting the END-STATE TEXT (which I should have done immediately)
shows the mechanism is uniform.

**Doctrine**: frequency analysis lies; trajectory inspection
discriminates. Top-N digit counts hide the path the model took.

## Files

- `tmp/20260525_attention_inflight/mode_b_is_loop_attractor.md` (this)
- Refuted: `mode_b_capability_geometry.md` (8th memo's sub-taxonomy)
- Earlier in arc: `cross_model_p9_p10_distribution.md` (production rates)
- Cached CoT end-states inspected directly: `tmp/20260524_quant_matrix/4bit/p09*.json`, `p10*.json`
