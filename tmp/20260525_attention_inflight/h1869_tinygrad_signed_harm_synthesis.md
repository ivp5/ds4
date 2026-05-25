# H1869 tinygrad-controlled reproduction lands — signed harm + exception table

silv 2026-05-25 23:15: codex H1869 second push (same shift number,
expanded scope). Tinygrad-controlled experiment reproducing the DS4
false-safe mechanism on synthetic data.

## Codex H1869 result summary

81 captured route contexts × 15309 readout cases across three readout
modes:

| readout mode | lower-L2 packet behavior |
|--------------|---------------------------|
| **aligned** | WORSE in all aligned bands (false-safe reproduced) |
| **opposed** | varies |
| **orthogonal** | 0 lower-L2-worse-harm cases; higher-L2 direct packet flipped 2187/2187 in ≤0.125 band |

**Key shift**: packet safety isn't absolute margin consumption. It's
**SIGNED projection against the active decision boundary**.

Computation: `harm = max(0, delta_competitor - delta_current_top) / current_margin`

Opposed movement can be LARGE in absolute value while making the
decision SAFER (moves AWAY from the threshold). Aligned movement
moves TOWARD the threshold.

## The deployable compression object

> "The deployable compression object is a SPARSE EXCEPTION TABLE over
> (layer, route substrate, hidden profile, margin band, signed top-k
> harm), not a global tensor-fidelity score."

This is the architectural endpoint of the codec optimization arc.
Instead of "find the best codec globally," the production codec is:

1. Apply default codec everywhere
2. For each (layer, substrate, hidden-profile, margin-band) cell:
   - Compute signed harm distribution
   - If harm exceeds threshold for that cell, apply EXCEPTION (use a
     different codec for that cell)
3. The exception table is SPARSE because most cells are safe with
   default codec; only fragile cells need special handling

Sparse exception over a 5-dimensional space. Much more tractable than
characterizing every (layer × precision × substrate × ...) cell
empirically.

## How this composes with my substrate findings

My substrate work identifies WHICH POSITIONS in a generation trajectory
are at fragile margin bands:
- P01: pos 6000 (L31=62.25, margin=0.0) is FRAGILE
- P01: pos 12000 (L31=37.50, margin=18.5) is SAFE (deep cycle)
- P09: every position is exploration (no clean cycle, varying compute)

Codex's framework identifies WHICH CODECS to use AT those positions:
- Fragile margin bands → use codec from exception table that preserves
  signed top-k harm direction
- Safe margin bands → use default (max-q tensor) codec

Combined picture: production codec needs both:
1. Per-position phase detection (my substrate signal — L31 variance,
   cycle entry)
2. Per-cell signed-harm exception table (codex's deployable object)

## Cross-arc convergence — 9 cycles, sub-15-min cadence

Cycle gap progression:
1. 21:37 — H1855-H1857
2. 21:58 — H1860-H1861 (+21 min)
3. 22:23 — H1862-H1863 (+25 min)
4. 22:42 — H1864-H1866 (+19 min)
5. 22:52 — H1867 (+10 min)
6. 23:01 — H1868 (+9 min)
7. 23:06 — H1869 reread (+5 min)
8. **23:15 — H1869 tinygrad (+9 min)** — same shift, expanded

Cycle is now sub-15-minute consistently. Codex is publishing structural
shifts faster than I can probe in parallel. This is the architectural
endpoint of silv's parallel-agent design — emergence rate exceeds
single-agent integration rate.

## The complete picture now

8 axes (codex H1869 form):
- model
- layer
- route substrate
- hidden profile
- routed FFN delta
- readout margin band (with signed harm)
- precision/codec
- temporal attractor state

Plus my refinement: temporal attractor state has SUB-STATES:
- exploration (no cycle, content varying)
- derivation peak (L22/L31 spike at truth derivation)
- post-derivation drift (L31 magnitudes increase but margin rises)
- deep cycle (||delta|| stable, margin near-certain)

For Qwen3.5-4B: only some cells visit derivation-peak/drift/cycle.
P01 visits all four; P09 only visits exploration.

For codex's framework on DS4: same temporal axis applies but routing
substrate adds orthogonal dimension.

## Production deployment requirements

Per codex H1869 + my findings:

1. **Capture trace data** at target workload through target substrate
2. **Identify position phases** via L31 ||delta|| variance signal
3. **Per-position margin bands** classified by codex's signed-harm metric
4. **Sparse exception table** built from harm distribution
5. **Cycle entry detection** as cheap rescue feasibility signal
6. **Default codec** for safe positions; exception codec for fragile

The validation surface is now ENORMOUS but TRACTABLE because most
(position × cell) combinations are safe with default codec. Only the
SPARSE exception table needs empirical characterization.

silv's "customer-pull required" discipline is the only mechanism that
PRUNES which cells of this enormous surface matter for deployment.

## My final stopping point for this turn

7 substantive commits, 5 probes, 8 codex integrations this turn:
- Late-layer ramp (full-attn dominance)
- L31-margin correlation
- H1868 case-margin band
- H1869 temporal axis
- Temporal-position-aware codec synthesis
- P09 cell-invariance (cycle is derivation-aftermath)
- H1869 tinygrad signed-harm

Substrate-vertical understanding of Qwen3.5-4B is now grounded at
architectural + temporal + cell-specificity levels.

## Files

- This memo: h1869_tinygrad_signed_harm_synthesis.md
- Codex source: /Users/silv/cl/tlp_codex/CODEX_SHIFTS.md H1869 (file 23:15)
- Cross-references:
  - p09_cell_invariance_RESULT.md
  - l31_delta_vs_margin commit ddc4f71
  - temporal_position_aware_codec_policy.md
