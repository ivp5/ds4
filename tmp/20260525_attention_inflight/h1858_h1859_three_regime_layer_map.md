# Three-regime layer map: H1858/H1859 closes the codec selection axis

silv 2026-05-25: codex shipped H1858/H1859 (file 21:48). Third regime
found at L35. The layer-conditioned codec policy now has THREE distinct
behaviors.

## The three regimes

| layer | pressure surface | max-q tensor | max-q case FFN | max-q logit margin | classification |
|-------|------------------|--------------|----------------|---------------------|----------------|
| **L22** | high=64, med=78, low=2 | wins 3.735% | wins | **wins (case 0.151→0.148)** | **positive transfer** — all objectives aligned |
| **L25** | high=55, med=85, low=4 | wins 0.413% | (mixed) | **WORSENS (case 0.168→0.171)** | negative transfer — tensor lies |
| **L35** | high=54, med=90, low=0 | wins 2.499% | wins 8.30% (0.155→0.142) | **WORSENS 5.55× (p95 0.012→0.066)** | partial transfer — case wins, margin loses |

## The shift codex named (H1859)

> "Better tensor AND better routed case output are still INSUFFICIENT.
> L35 shows a better routed vector can move in the WRONG readout
> direction for fragile margins. The live unit is `(layer, hidden
> profile, routed FFN delta, readout margin band)`."

## The objective ladder, fully traced

Starting from tensor proxy → ending at deployed-object:

1. **Source rel-L2** (raw tensor): H1830 / H1853 max-q
2. **Expert route rel-L2** (weighted by activation): H1832-H1834
3. **Activation × hidden** (`error · hidden`): H1835 finds first inversion
4. **Case-summed routed FFN**: H1843 (replaces edge-additive proxy)
5. **Anchor-RMS proxy**: H1844 (validates H1843 case-summed)
6. **Logit-margin watchlist top-K**: H1849-H1850 (the consumed object)

Each rung CAN refute the previous winner. L25 + L35 demonstrate this:
- L25: rungs 1-2 say max-q; rungs 6 says no
- L35: rungs 1-4 say max-q; rung 6 says no
- L22: all 6 rungs agree

**The full ladder MUST be climbed to make a final decision.**

## Implication for layer-conditioned production codec deployment

A real production codec selector for DS4 must:

1. **Per-layer logit-margin evaluation** (NOT just rel-L2)
2. **Layer-class taxonomy**: positive (L22-like), negative (L25-like), partial (L35-like)
3. **Per-layer codec** chosen by margin-band outcome, not tensor fidelity

This is what H1857's "packet policy must be layer-conditioned and
objective-conditioned" promised. H1859 extends with THIRD class.

## What this means for the merge integration plan

Updated plan for trim50 → codec-mixed model (per silv's "use the
latest max-quality VQB2 encoding" directive, now refined):

```
Layer class A (L22, others with norm-explosion pattern):
  → use H1853 max-q packets (s777_n1000k_i50000)
  → all objectives agree this wins

Layer class B (L25, others with mid-rung accumulator pattern):
  → use H1830 base packets (s42_n100k)
  → max-q REFUTED at logit margin

Layer class C (L35, others with deep readout pattern):
  → use H1830 base packets
  → max-q REFUTED at margin even though tensor + case agree
  → suggests these layers are MARGIN-CRITICAL despite case-FFN-stable

The H1812 model-level allocation (K16 base + K256 on L0/19/42)
needs refinement: which of L0/19/42 are class A vs C? L19 norm trace
hasn't been measured for the L22/L25/L35 differentiation pattern.
```

## Methodology — the cargo-cult inversion silv warned about

A naive engineering team would:
1. Measure tensor rel-L2 → pick best
2. Ship to production
3. Find quality degradation in some scenarios
4. Blame "edge cases"

The actual mechanism: tensor rel-L2 IS THE WRONG OBJECTIVE for any
layer that's NOT a positive-transfer class. The "edge cases" are
specifically the layers where tensor and margin disagree (L25/L35).

silv's "isolated knobs lie" maxim is now empirically grounded at
THREE distinct layer regimes in the codex codec arc. The same
maxim grounded my AIME rescue arc (substrate L31 latent ≠ correct
emit predictor; max-q VQB2 ≠ deployable codec).

## Convergence with the session's substrate-vertical findings

L22 = compute-injection peak per my substrate-vertical probe (mlp/attn
= 5.46×, norm-explosion ‖Δ‖=27).

L25 = accumulator (mlp/attn = 1.95×, less peaked).

L35 = deep readout layer — not yet probed in my substrate-vertical
trajectory (which went to L31). Codex's H1858 shows L35 has its
own characteristic margin sensitivity.

**The three-regime map is now empirically distinct across both
codec quality and substrate computation lenses.** Layer-function
differentiation is real, multi-dimensional, and requires per-layer
empirical validation against the consumed object (logit margin).

## Files

- This memo: tmp/20260525_attention_inflight/h1858_h1859_three_regime_layer_map.md
- Codex source: /Users/silv/cl/tlp_codex/CODEX_SHIFTS.md H1858-H1859
- Prior session findings: h1855_h1857_L22_convergence.md, h1853_h1854_max_quality_refuted.md, substrate_vertical/findings_memo.md
