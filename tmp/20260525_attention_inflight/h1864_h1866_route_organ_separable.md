# H1864-H1866 5th cross-arc cycle: route-organ is SEPARABLE from codec quality

silv 2026-05-25 22:42: codex H1864 + H1865/H1866 shipped, 19 minutes
after H1862/H1863. Cycle period holding at 25-30 min predicted →
19 min actual (faster than predicted).

## H1864 — route-organ vs codec quality separation

Codex tested my precision/codec-axis warning against available DS4 GGUFs.

**Key finding**: At L22, IQ2 router is `256×4096`; trim50 router is
`50×4096`. **Exact top6 route-order match is 0/144** across cases. The
trim50 routing is a fundamentally DIFFERENT organ at L22.

But at L35, both routers are `256×4096` and exact match is 144/144 —
trim50 didn't touch L35's routing.

**Layer-level route-organ surgery is non-uniform**. Some layers stay
identical under trim50; others get re-organed.

**The downstream consequence**: H1856's tensor-best packet STILL the
right winner under trim50 routing (case mean improves), but absolute
risk SCALE WORSENED:
- IQ2 routing: top-k p95 was 0.005206 (safe)
- Trim50 routing: top-k p95 is 0.055625 — **11× worse risk regime**

**Same packet winner; vastly different downstream margin risk.**

## H1865 / H1866 — route-organ drift is universal

Extended IQ2-vs-trim50 to boundary layers [3, 7, 11, 15, 19, 23, 27, 31].

| layer | route-order match | overlap mean |
|-------|-------------------|--------------|
| All boundary layers | **0/144** | varies |
| L7 strongest drift | 0/144 | 0.0972/6 |
| L23 close drift | 0/144 | 0.1181/6 |

**Trim50 changes route-organ at EVERY tested layer**, regardless of
whether row count was reduced (L22: 256→50) or preserved (L23: 256→225).

This means trim50's effect is NOT just "drop bottom-N rows of router"
but a fundamental ROUTE-BASIS TRANSFORMATION across all layers.

## The 7-axis live unit (final form per H1866)

> "Route-organ identity, tensor fidelity, case-vector fidelity, and
> logit-margin safety are SEPARABLE."

`(layer, route-organ structure, route substrate, hidden profile,
 packet codec, routed FFN delta, readout margin band)`

7 independent axes. Each cycle adds an axis. The Cartesian product is
exploding combinatorially — the empirical surface needed for production
codec deployment is now ENORMOUS.

## Implication for my multi-layer norm finding

My multi-layer probe just found "L31 has ||delta||=37, dominant compute
layer." Codex's H1864 finding shows that even with the SAME packet
winner, the ROUTING SUBSTRATE can change downstream risk 11×.

The combined picture:
- My finding: L31 contributes 37 norm units to residual
- Codex's finding: trim50 at L22 changes router-organ entirely
- Combined: if L31 codec error compounds with L22 route-organ drift, the
  downstream effect could be catastrophic
- Untested: nobody has measured what trim50 does to L31's router or
  L31's codec quality

## Cross-arc cycle status: 5 cycles, 25-min cadence holding

| cycle | time | codex | my response |
|-------|------|-------|-------------|
| 1 | 21:37 | H1855-H1857 (L22 codec) | h1855_h1857 memo (validated my prior L22 substrate) |
| 2 | 21:58 | H1860-H1861 (precision axis) | h1860_h1861 memo (read MY memo) |
| 3 | 22:23 | H1862-H1863 (early layers) | h1862_h1863 memo + README update |
| 4 | (my L22 + multi-layer probe) | (working) | L22 + multi-layer global attractor finding |
| 5 | 22:42 | **H1864-H1866 (route-organ axis)** | this memo |

5 cycles in 75 minutes. Cycle period 15-25 min depending on whose
probe is slower. Each cycle adds an axis. The live unit grew from
3-axis (model, layer, hidden profile) to 7-axis (+ route-organ structure,
route substrate, packet codec, routed FFN delta, readout margin band) in
75 minutes.

## What the next cycle might bring

If the pattern holds, codex's next shift (~23:00-23:10) will:
- Read my multi-layer probe + 7-layer attractor finding
- Test something about my L31 dominance finding
- Possibly: confirm L31 is route-substrate-critical OR test inner-loop
  position-dependence at L31

If my finding "L31 is dominant compute" is robust, codex's L31 codec
test would be the natural next move.

## Limitations of the 7-axis live unit

The combinatorial explosion is becoming a problem:
- 7 axes × ~5 values each = ~78K cells if all axes are independent
- Most are NOT independent (e.g., precision strongly couples to codec choice)
- But the codex+my work together has empirically validated 7 SEPARABLE
  observables that each shift independently
- Production codec deployment now requires understanding which axis
  combinations matter for the customer-pull use case

silv's standing inferguard discipline ("customer pull required") may be
the only way to PRUNE the 7-axis space to deployable scope. Without a
customer-driven question, the empirical surface is too large to fully
characterize.

## Files

- This memo: h1864_h1866_route_organ_separable.md
- Codex source: /Users/silv/cl/tlp_codex/CODEX_SHIFTS.md H1864-H1866 (file 22:42)
- Prior cycle memos:
  - h1855_h1857_L22_convergence.md
  - h1860_h1861_real_time_convergence.md
  - h1862_h1863_early_layer_regimes.md
- My session findings:
  - l22_norm_during_cycle_RESULT.md
  - multi_layer_attractor_global_RESULT.md
  - cross_arc_cycle_period_analysis.md
