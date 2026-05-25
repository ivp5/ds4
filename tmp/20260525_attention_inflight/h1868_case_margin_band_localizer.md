# H1868 — case-level margin band localizer + cycle 7 at 9 min cadence

silv 2026-05-25 23:01: codex H1868 ships, 9 min after H1867. Cycle
continues to tighten. H1868 adds CASE-LOCAL margin band as the
deployable axis.

## H1868 finding

Codex shipped `h1868_case_margin_record_exporter_20260525.py` — a thin
H1851/H1850 reuse layer writing per-case JSONL records with route top6,
hidden kind, source margin, case rel-L2, top2/top-k consumption.

L23 case-level comparison (IQ2 routing vs trim50 routing):

| substrate | target cases | harmed | helped | mean Δ top-k | mean Δ case rel-L2 |
|-----------|-------------|--------|--------|--------------|---------------------|
| IQ2 | 9 | 2 | 2 | **-0.003** (improvement) | -0.010 |
| trim50 | 9 | **6** | **0** | **+0.046** (degradation) | -0.008 |

**Same tensor-best packet. Opposite outcomes per substrate.**

**Localizing by source margin band**:

| substrate | margin band distribution | mean Δ top-k in band |
|-----------|--------------------------|-----------------------|
| IQ2 | 8/9 in >8 (high margin) | -0.003 (safe) |
| trim50 | 1 in ≤0.125, 4 in ≤1, 4 in ≤8 | **+0.082** in ≤1 band |

The harm is concentrated in low-margin cases. Worst case (#14): source
margin 0.3125, route overlap 0, packet improves case rel-L2 by 0.002
but **worsens top-k consumption by 0.125**.

## H1868 shift verbatim

> "The deployable unit is case/band-local. A packet can be selected per
> layer/substrate only after checking local source-margin bands;
> Euclidean case reconstruction is not a safety proxy in fragile
> readout bands."

## The 9-axis live unit

The 7-axis live unit from H1866 + H1867's route-basis-transformation
sub-axis + H1868's source-margin-band sub-axis:

`(layer, route-basis transformation, route-organ structure, route substrate,
 hidden profile, packet codec, routed FFN delta, readout margin band,
 source-margin band per case)`

9 axes. **The case-local axis is now first-class** — packets aren't
just safe by layer/codec/precision but require per-case margin checking.

## Connection to my multi-layer ||delta|| finding

My L31 ||delta|| varies 31-62 across positions:
- pos 4000: 34.50
- pos 6000: 62.25 (peak)
- pos 8000: 31.00
- pos 12000: 37.50 (cycle entry)

H1868's framework predicts: positions with HIGH ||delta|| might
correspond to LOW source-margin bands (model is doing harder work,
output is less certain). And vice versa.

Testable: pair my ||delta|| measurements with L_final softmax top-1
margin at the same positions. If high ||delta|| correlates with low
margin → my positions are in the fragile band where codec errors
would be dangerous.

This is an untested prediction. I haven't measured logit margins at the
captured positions; they were saved but only token-IDs, not the
top-k margin distributions.

## Cross-arc cycle 7

| cycle | time | gap (min) | codex axis |
|-------|------|-----------|-------------|
| 1 | 21:37 | -- | L22 codec |
| 2 | 21:58 | 21 | precision |
| 3 | 22:23 | 25 | early layers |
| 5 | 22:42 | 19 | route-organ |
| 6 | 22:52 | 10 | route-basis transformation |
| **7** | **23:01** | **9** | **case-local margin band** |

Cycle gaps: 21 → 25 → 19 → 10 → 9. Cycles tightening. Each cycle adds
or sharpens an axis. The live unit has grown from 3-axis to 9-axis in
~85 minutes.

## Going to ground — what is the deployable unit now?

Codex's H1868 shift: "the deployable unit is case/band-local." This
collapses the explosion: instead of 9-axis full Cartesian product
(thousands of cells), the deployable surface is:

**For each (layer, substrate, codec) tuple, characterize the
source-margin distribution of cases that arrive at that point. If
fragile-band cases dominate, the codec is unsafe regardless of average
metrics.**

This is the CASE-LOCAL discipline. silv's "isolated knobs lie" framing
now grounded at 5 levels:
1. Tensor lies about case FFN
2. Case FFN lies about logit margin
3. Logit margin lies without precision context
4. Precision lies without route-substrate context
5. **Substrate average lies without per-case margin band check**

The 5th level is the deepest because it forbids ANY averaged metric
from being a deployment criterion. Per-case scrutiny is required.

## Implication for production codec deployment

The full validation surface for any new codec/precision choice is:

1. Run sample workload through target substrate at target precision
2. Export per-case records (route top6, hidden, margins, etc.)
3. Distribute cases into source-margin bands (≤0.125, ≤1, ≤8, >8)
4. For each band: measure packet's Δ top-k
5. If FRAGILE band (e.g., ≤1) has positive Δ top-k (degradation),
   REJECT packet regardless of high-band performance

This is significantly more rigorous than "test packet on N samples,
take average." The averaging hides per-band failures.

## My next move

The integration probe: measure L_final softmax top-k margins at my
captured positions, correlate with L31 ||delta||. Would test whether
my high-compute positions ARE in fragile readout bands.

Wall time: ~3 min if I just need to capture during the existing decode
loop. Worth doing as it bridges my substrate finding to codex's
deployable framework.

## Files

- This memo: h1868_case_margin_band_localizer.md
- Codex source: /Users/silv/cl/tlp_codex/CODEX_SHIFTS.md H1868 (23:01:35)
- Cycle history: cross_arc_cycle_period_analysis.md
- My multi-layer/late-layer findings: full_attn_vs_ssm_compute_split.md
