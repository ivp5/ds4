# H1862/H1863 early-layer codec regimes + L4 emerges as best positive-safe point

silv 2026-05-25 22:23: codex H1862/H1863 shipped (file modified 22:23:56).
Codex READ my updated `README.md` (which I just updated at 22:00 with the
3-axis precision finding + 4-regime layer taxonomy) and extended the
layer-class probe to EARLY layers (L0/L4/L10).

## Bidirectional cross-arc feedback loop continues

Real-time sequence:
- 21:58: codex shipped H1860/H1861 reading my `precision_attractor_bifurcation_CONFIRMED.md`
- 22:00ish: I committed README update with 3-axis attractor + 4-regime taxonomy
- 22:13ish: I committed multi-precision ensemble + self-refutation memos
- 22:23: codex shipped H1862/H1863 reading my README.md

The feedback loop is at ~25-30 minute intervals — each agent reads the
other's latest memo and extends. silv's parallel-agent architecture is
producing emergent structural cartography of the layer-codec space.

## The seven-regime full-depth taxonomy

| layer | depth | pressure | tensor winner | margin behavior | classification |
|-------|-------|----------|---------------|-----------------|----------------|
| **L0** | early | high=62/med=81/low=1 | s42_n100k +3.7% | **WORSENS top-k p95 0.256→0.268** | early-risky (tensor lies again) |
| **L4** | early | high=46/med=97/low=1 | s777_n500k +4.1% | **wins 0.064→0.016 (4× improvement)** | **EARLY POSITIVE-SAFE** |
| **L10** | early | (not reported) | direct/reference | no improvement | direct-best (codec saturated) |
| **L22** | mid | high=64/med=78/low=2 | s777_n1000k +3.7% | wins | mid positive-safe |
| **L25** | mid | high=55/med=85/low=4 | (max-q) | **LOSES** | mid negative-transfer |
| **L35** | deep | high=54/med=90/low=0 | s777_n1000k | **LOSES 5.55× top-k p95** | deep partial-transfer |
| **L42** | deep | high=61/med=83/low=0 | **H1830 base wins** | partial (margin gains, watchlist unchanged) | deep base-wins-risky |

## The codex H1863 shift verbatim

> "Layer-specific codec behavior starts near the BOTTOM of the model,
> not only around L22/L25/L35/L42. The current coarse map now contains
> positive transfer, case/margin split, objective-disagreement,
> direct-best, and positive-but-risky regimes."

## L4 is the surprise — the most positive-safe layer in the network

L4 codec choice produces:
- Tensor rel-L2 improvement: +4.129% (0.165 → 0.159)
- Top-k margin p95 improvement: **4× better** (0.064 → 0.016)

L4 has the HIGHEST safe-improvement ratio of any layer tested. Better
than L22 (which only gained 3.735% tensor, top-k 0.0087→0.0052 = 1.7×).

This is COUNTERINTUITIVE if you think "early layers are simple, easy
to compress, no risk." The data shows L4 is a particularly LOAD-BEARING
layer where the right codec produces structural improvement.

Hypothesis (codex didn't say this; I'm extending): L4 might be the
"input understanding" layer where lexical/syntactic features stabilize
into semantic representations. Codec error at L4 would propagate
through entire stack as semantic-feature-noise. Getting L4 right
matters disproportionately.

L0 in contrast is the EMBEDDING-IMMEDIATE layer; codec error there
gets corrected by subsequent layers (negative-transfer at margin means
the model COMPENSATES for L0 codec noise downstream, but at cost of
margin precision).

## Implication for production codec policy

Updated production codec deployment map:

```
Layer 0 (post-embedding):    H1830 base (BUT MONITOR — tensor wins at margin cost)
Layer 4:                     s777_n500k (BEST SAFE GAIN, 4× margin improvement)
Layer 10:                    direct/reference (no codec gain possible)
Layers 11-21:                empirical validation per layer (not tested by codex yet)
Layer 22:                    s777_n1000k_i50000 (positive-safe)
Layers 23-24:                empirical validation (not tested by codex yet)
Layer 25:                    H1830 base (max-q margin-NEGATIVE)
Layers 26-34:                empirical validation
Layer 35:                    H1830 base (max-q case-positive but margin-NEGATIVE)
Layer 36-41:                 empirical validation
Layer 42:                    H1830 base (max-q REGRESSES)
```

7 layers tested; 36 layers (84%) still empirically untested. The
"choose one codec policy globally" approach is REFUTED at 7 of 7
empirically-tested layers. Per-layer codec is structurally necessary
for production deployment.

## Continued meta-doctrine reinforcement

Each level of refinement (early layers tested) reveals NEW axes the
previous level didn't see:
- Mid + deep layers tested → assumed pattern was "max-q OK at some
  layers, base elsewhere"
- Early layers tested → revealed L4 SAFEST + L10 SATURATED + L0
  RISKY-at-margin
- ~80% of network depth still UNTESTED

silv's "doubt every step" is operationally necessary here. A naive
deployment team that tested 7 of 43 layers and claimed "we have a
codec policy" would be deploying a policy for which they had EMPIRICAL
EVIDENCE for 16% of the relevant decisions.

## What the live unit looks like now

Codex H1861's expanded live unit was:
`(model, layer, hidden profile, routed FFN delta, readout margin band, precision/codec)`

H1862/H1863 didn't change the axes, but populated more LAYER values.
The unit now spans the full depth axis with 7 anchor points.

For each anchor point, the precision axis (4bit/8bit/bf16) remains
COMPLETELY UNTESTED at the codec quality level. That's 7 anchors × 3
precisions = 21 codec-precision Cartesian cells that need empirical
validation. Currently 7 cells filled (the layer-only axis at codex's
default precision).

## My L22 norm probe (running while this memo was composed)

PID b63di95ha is running L22 norm-during-cycle probe on Qwen3.5-4B-4bit.
The probe captures L22 input + output hidden norms at pre-cycle vs
in-cycle positions to test "L22 is structurally always firing" vs
"L22 fires content-dependent."

If H1862/H1863's L4 finding generalizes, L22 might ALSO be a specific
positive-safe codec point (which matches my prior substrate-vertical
finding) but L22 might NOT be the most load-bearing layer in the
network — L4 could be.

Expected probe completion ~5min wall. The result will inform whether
"compute injection" happens at L22 specifically or distributed across
both early (L4) and mid (L22) anchor points.

## Files

- This memo: tmp/20260525_attention_inflight/h1862_h1863_early_layer_regimes.md
- Codex source: /Users/silv/cl/tlp_codex/CODEX_SHIFTS.md H1862/H1863 (file 22:23:56)
- Cross-references:
  - README.md (read BY codex)
  - h1860_h1861_real_time_convergence.md (prior cross-arc memo)
  - h1858_h1859_three_regime_layer_map.md (L35 third regime)
  - h1855_h1857_L22_convergence.md (L22 codec finding matched my substrate)
