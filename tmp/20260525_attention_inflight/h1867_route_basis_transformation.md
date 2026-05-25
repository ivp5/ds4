# H1867 lands — cross-arc cycle 6 at 10 min cadence

silv 2026-05-25 22:52: codex H1867 shipped, 10 min after H1864-H1866.
Cycle is TIGHTENING (25 → 19 → 10 min). H1867 separates route-basis
transformation from row-count amputation cleanly.

## H1867 finding

L23 in trim50 retains 225/256 router rows BUT shows 0/144 exact route
matches with IQ2 baseline. Mean overlap 0.1181/6. This proves:

**Trim50's effect is ROUTE-BASIS TRANSFORMATION, not row truncation.**

Even keeping most experts (225/256 = 88%) the routing basis transforms
into something fundamentally different.

Same tensor-best packet at L23:
- Under IQ2 routing: clean positive transfer (top-k p95 0.017→0.001)
- Under trim50 routing: margin-risky (top-k p95 0.030→0.119 = **3.92× worse**)

The packet is the same; the routing substrate transforms safety.

## Codex H1867 verbatim shift

> "Keeping most experts is not enough. Route-basis transformation alone
> can invert packet safety; tensor/case fidelity remain false-safe
> metrics unless tied to substrate-specific readout margins."

## Architectural orthogonality observation

My probes have been on Qwen3.5-4B (no MoE, no routing substrate).
Codex's probes are on DS4 (MoE, routed FFN).

The cross-arc convergence ISN'T on specific quantitative findings (we
can't directly compare L22 substrate ||delta|| in Qwen3.5-4B to L22
route-basis transformation in DS4). The convergence is on:

1. **Layer-conditioning is required** (both arcs)
2. **Axes are separable** (codec ≠ precision ≠ route-organ ≠ ...)
3. **Naïve global metrics are false-safe** (both arcs)
4. **Substrate attractors are global** (my finding; codex extends via
   route-organ universality)

These are SHARED STRUCTURAL DOCTRINES emerging from different architectures.
The cross-arc convergence pattern is meta-doctrine convergence, not
specific-quantity convergence.

## Cycle 6 status

| cycle | time | gap (min) | codex axis added |
|-------|------|-----------|-------------------|
| 1 | 21:37 | -- | L22 codec |
| 2 | 21:58 | 21 | precision |
| 3 | 22:23 | 25 | early layers |
| 5 | 22:42 | 19 | route-organ |
| 6 | 22:52 | 10 | route-basis transformation (sharpens 5) |

Cycle is tightening as cumulative findings accelerate codex's
testable questions. Each new finding reduces the cost of the next
hypothesis.

## Live unit at H1867

`(layer, route-basis transformation, route-organ structure, route substrate,
 hidden profile, packet codec, routed FFN delta, readout margin band)`

8 axes. Codex's H1866 said 7; H1867 separates "route-basis transformation"
from "route-organ structure" as orthogonal sub-axes of the routing
substrate.

## Production codec implication

Codex's "false-safe metrics" is now empirically grounded:
- Tensor rel-L2 lies about case FFN
- Case FFN lies about logit margin
- Logit margin lies without precision context
- Precision lies without route-substrate context
- Route substrate (count) lies without route-basis transformation context

Five levels of false-safety. Production deployment requires ALL FIVE
PROBED EMPIRICALLY for the target substrate.

## My next action

This is a natural stopping point for this turn. I've shipped:
- L22 norm probe (Task #602 complete)
- Multi-layer probe (refuted "L22 is peak", found L31 dominance)
- 8bit cross-precision probe (cycle invariance + magnitude shift)
- 5 cross-arc cycle integration memos
- Cross-arc cycle period analysis

The next high-leverage probe: my multi-layer attractor finding does NOT
test route-substrate (Qwen3.5-4B has no routing). To extend to codex's
finding, I'd need to use the DS4 binary on M1 — task #603 was queued
for this. That's a real probe requiring binary execution; deferred for
now.

## Files

- This memo: h1867_route_basis_transformation.md
- Codex source: /Users/silv/cl/tlp_codex/CODEX_SHIFTS.md H1867 (file 22:52)
- Cycle period documentation: cross_arc_cycle_period_analysis.md
- Prior cycles: h1855_h1857, h1860_h1861, h1862_h1863, h1864_h1866
- My findings: l22_norm, multi_layer_attractor, cross_precision_attractor
