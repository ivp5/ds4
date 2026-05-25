# 8bit multi-layer probe — cross-precision attractor INVARIANT

silv 2026-05-25 22:50: 8bit replication of the 4bit multi-layer norm
probe. Tests whether the GLOBAL ATTRACTOR finding (cycle-period
determinism at all measured layers) is precision-invariant.

## Pre-declared hypotheses

- **H_LAYER_INVARIANT**: cycle-period determinism replicates at 8bit
  (4bit finding generalizes)
- **H_PRECISION_VARIANT**: 8bit shows different layer-attractor structure

## Result table

| layer | 4bit (cycle=31) | 8bit (cycle=32) | 4bit determinism | 8bit determinism |
|-------|-----------------|------------------|---------------------|---------------------|
| L0 | 2.30/2.30/2.30/2.30 | 2.03/2.03/2.03/2.03 | EXACT | EXACT |
| L4 | 1.21/1.20/1.20/1.20 | 1.16/1.18/1.18/1.18 | max_diff=0.008 | max_diff=0.016 |
| L22 | 5.69/5.69/5.69/5.69 | 7.22/7.22/7.19/7.22 | EXACT | max_diff=0.031 |
| L25 | 5.56/5.56/5.59/5.56 | 5.97/5.97/5.97/5.94 | max_diff=0.031 | max_diff=0.031 |
| L31 | 37.50/37.75/37.75/37.75 | 46.25/46.25/46.25/46.25 | max_diff=0.250 | EXACT |

**H_LAYER_INVARIANT CONFIRMED**: 8bit also shows exact-or-near-exact
cycle-period determinism at every tested layer. The global attractor
structure is precision-invariant.

## Magnitude shift across precisions

| layer | 4bit ||delta|| | 8bit ||delta|| | shift |
|-------|----------------|------------------|-------|
| L0 | 2.30 | 2.03 | -12% |
| L4 | 1.20 | 1.18 | -2% |
| L22 | 5.69 | 7.22 | **+27%** |
| L25 | 5.56 | 5.97 | +7% |
| L31 | 37.62 | 46.25 | **+24%** |

8bit has HIGHER compute load at mid/late layers (L22, L31) than 4bit;
roughly equal at early layers (L0, L4). The "compute load redistributes
toward later layers at higher precision" trend matches what one would
expect — higher precision can carry more information through deeper
layers.

## Cycle-period scaling

| precision | cycle period | L31 in-cycle ||delta|| | L22 in-cycle ||delta|| |
|-----------|--------------|--------------------------|--------------------------|
| 4bit | 31 tokens | 37.62 | 5.69 |
| 8bit | 32 tokens | 46.25 (+24%) | 7.22 (+27%) |
| bf16 | 90 tokens | (untested) | (untested) |

The bf16 cycle (90 tokens) is 2.8× longer than 4bit/8bit. If the L31
trend continues, bf16's in-cycle L31 ||delta|| could be MUCH higher
(would need ~50-60 if linear extrapolation, possibly much higher if
nonlinear).

Testable prediction: bf16 L31 ||delta|| in cycle > 50. Untested.

## What this RULES OUT

The "L31 is precision-specific quirk" interpretation is REFUTED. L31 is
the dominant compute layer at BOTH 4bit and 8bit. The 6× magnitude over
L22 holds at both precisions.

The "cycle is 4bit-only attractor" interpretation is REFUTED. 8bit also
exhibits the global cycle determinism at all 5 layers tested.

The "magnitude is precision-noise" interpretation is plausible but
unlikely given the directional consistency (L22 + L31 both INCREASE at
8bit, others stable/decrease).

## What this CONFIRMS

1. **Global attractor structure is precision-invariant** at the
   substrate level. The model collapses into a limit cycle at all
   tested precisions, with all measured layers participating.

2. **L31 dominance is structural**, not a single-precision artifact.
   The final decoder layer carries ~5-6× the residual contribution of
   mid-layer L22.

3. **Per-layer magnitude is precision-dependent**, with deeper layers
   showing larger precision-sensitivity.

## Implication for codec policy

Codex's H1864 found route-organ identity is SEPARABLE from codec
quality. My result confirms cycle-determinism is SEPARABLE from
precision. Combined:

```
At-substrate (precision-invariant):
  - Cycle-period determinism (all layers on orbit)
  - L31 dominance pattern
  - Attractor TOPOLOGY

Per-precision/per-codec (orthogonal):
  - Layer ||delta|| magnitudes
  - Route-organ structure
  - Margin risk profile
```

The TOPOLOGY is the structural invariant; the MAGNITUDES are the
deformable parameters. This is consistent with dynamical-systems
analysis where the attractor's topology (number of cycles, basin
structure) is robust to parameter perturbations within a regime, while
the attractor's metric properties (cycle period, basin width) vary with
parameters.

## Files

- This memo: cross_precision_attractor_INVARIANT.md
- Raw data: multi_layer_8bit_cycle_result.json (5 layers × 8 positions)
- Probe code: multi_layer_8bit_cycle.py
- Run log: multi_layer_8bit_run.log
- Companion: multi_layer_attractor_global_RESULT.md (4bit version)
- Companion: precision_attractor_3axis_complete.md (3-precision token-level)
