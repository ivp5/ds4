# H1860/H1861 real-time cross-arc convergence with precision-attractor finding

silv 2026-05-25 21:58: codex H1860/H1861 just shipped (file modified 21:58:25,
my prior commit at 21:55). H1861 explicitly READS my session memo
`precision_attractor_bifurcation_CONFIRMED.md` and extends the live unit
to include precision/codec as a first-class axis.

## What codex H1860/H1861 found

**L42 regime** (fourth):
- Pressure surface: high=61, medium=83, low=0 (same "no low" pattern as L35)
- Tensor winner is H1830 base (s42_n100k_i50000), NOT H1853 max-q
- H1853 max-q REGRESSES at L42 (0.165 → 0.173)
- Target `kind:router_row_mix|pressure:high`: "positive-transfer but NOT SAFE"
  - case mean improved 0.195 → 0.188
  - top2 p95: 2.80 → 2.15 (improved)
  - top-k p95: 2.80 → 2.27 (improved but watchlist flip rate stays 1/15)

**The codex shift quote** (verbatim, the meta-finding):

> "Read Montyneg `precision_attractor_bifurcation_CONFIRMED.md`: Qwen3.5-4B
> AIME P01 has distinct greedy limit cycles at 4bit period 31 and 8bit
> period 32, with different content and pre-cycle drift.
>
> Shift: the layer map must include precision/codec as a first-class axis.
> The current live unit expands to `(model, layer, hidden profile, routed
> FFN delta, readout margin band, precision/codec)`."

## The fourth regime closes the layer-class taxonomy

| layer | tensor winner | case FFN | logit margin | flip rate | classification |
|-------|---------------|----------|--------------|-----------|----------------|
| **L22** | H1853 max-q | wins | **wins** | (best) | **positive transfer, safe** |
| **L25** | H1853 max-q tensor only | mixed | **LOSES** | (worsens) | negative transfer (tensor lies) |
| **L35** | H1853 max-q | wins 8.30% | **LOSES 5.55×** | (worsens) | partial transfer (case wins, margin loses) |
| **L42** | H1830 BASE | wins | wins | (stays 1/15) | base-wins (max-q regresses; not safe) |

The taxonomy now has FOUR distinct regimes from the codec audit alone.

## Real-time bidirectional cross-arc influence

This session has now demonstrated bidirectional cross-arc influence WITHIN THE SAME DAY:

**Phase 1** (this session ~21:00-21:30): I probed precision-attractor
bifurcation at inference level. Found 4bit=31, 8bit=32, predicted bf16
would shift attractor again.

**Phase 2** (codex ~21:37): codex H1855-H1857 found L22 codec regime
INDEPENDENTLY of my L22 substrate-vertical finding. Two arcs converged
on "L22 is structurally special."

**Phase 3** (this session ~21:50): I extended to bf16 (90-token period,
2.9× scale jump, pre-truth interpretation manifold).

**Phase 4** (codex ~21:58): codex H1860/H1861 READS my Phase 3 memo and
extends its own model to include precision/codec as first-class axis.

**Phase 5** (this session 22:00+ — now): I document the convergence + use
codex's L42 finding to refine my own production codec deployment plan.

This is the closest real-time cross-agent feedback loop I've observed in
the project. silv's pattern of running parallel agents at different
abstraction levels (substrate inference vs codec quality) produces
emergent shared findings without explicit coordination.

## Updated production codec policy (now precision-aware)

Per H1859 + H1861 + my 3-precision attractor finding:

```
Live unit (codex's final form):
  (model, layer, hidden profile, routed FFN delta, readout margin band, precision/codec)

Layer class deployment policy:
  - L22-like (positive-transfer, safe): use H1853 max-q
  - L25-like (negative-transfer): use H1830 base
  - L35-like (partial-transfer): use H1830 base
  - L42-like (base-wins): use H1830 base
  - Other layers: per-layer empirical validation required

Precision class deployment policy:
  - 4bit: attractor period 31 tokens, post-truth manifold, cap=12K for rescue
  - 8bit: attractor period 32 tokens, prompt-leak intermediate, cap=12K
  - bf16: attractor period 90 tokens, PRE-truth interpretation manifold, cap >16K (rescuability degrades)
```

The Cartesian product (4 layer classes × 3 precisions = 12 cells) is the
new minimum-viable empirical surface for production codec deployment.

## The compounding implication

The session's META-DOCTRINE keeps growing:

1. **Layer-function differentiation** (my prior session + codex H1857)
2. **Objective-ladder rung-refutation** (codex H1859 case-but-not-margin)
3. **Codec-precision-layer triple-axis live unit** (H1861 + my 3-precision)
4. **Bidirectional real-time cross-arc influence** (this convergence)

The original mistake of naive engineering: "pick one number, optimize,
deploy." Each level of refinement reveals a NEW axis the previous level
didn't see. The "isolated knobs lie" maxim is now grounded in:
- Tensor lies (about case FFN)
- Case FFN lies (about logit margin)
- Logit margin lies (without precision context)
- Precision lies (without layer-class context)
- Layer class lies (without model-family context)

Every claimed "optimization" must specify ITS axis AND the unseen axes
it COMPRESSES. silv's "doubt every step" doctrine is empirically
necessary at every rung.

## Pending: extend the test surface

The 4-layer × 3-precision Cartesian needs more cells filled. Currently
known:
- L22 × 4bit/8bit/bf16: my substrate finding (L22 norm-explosion was tested at 4bit only)
- L22 × codec: codex H1857
- L25 × codec: codex H1854
- L35 × codec: codex H1859
- L42 × codec: codex H1861

UNTESTED cells worth probing:
- L22/L25/L35/L42 norm-explosion behavior at 8bit AND bf16 (test if codec
  layer regimes are precision-invariant)
- 4bit vs 8bit vs bf16 attractor period at OTHER AIME cells (P02, P05, P09)
- DS4 substrate-vertical decomposition (DS4 is its own architecture; the
  L22/L25/L35/L42 codex finding might not map 1:1)

## Files

- This memo: tmp/20260525_attention_inflight/h1860_h1861_real_time_convergence.md
- Codex source: /Users/silv/cl/tlp_codex/CODEX_SHIFTS.md H1860/H1861 (file 21:58:25)
- Cross-references:
  - h1858_h1859_three_regime_layer_map.md
  - h1855_h1857_L22_convergence.md
  - precision_attractor_3axis_complete.md
  - precision_attractor_bifurcation_CONFIRMED.md (read BY codex)
