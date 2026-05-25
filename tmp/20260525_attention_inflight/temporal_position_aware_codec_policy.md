# Temporal-position-aware codec policy — synthesizing H1868/H1869 with my substrate findings

silv 2026-05-25 23:08: integrating codex H1868 (case-margin band as
deployable unit) + H1869 (temporal attractor as 8th axis) with my
L31 ||delta|| vs margin correlation finding (-0.508).

## The deployable insight emerging

My data at 21 sampled positions on AIME P01 4bit:

| phase | positions | L31 ||delta|| | margin (log_p) | codec risk |
|-------|-----------|---------------|----------------|------------|
| **derivation** | 4000-8500 | 30-62 (high) | 0.0-7.9 (low) | **HIGH** |
| **cycle entry** | 10000-12000 | 35-41 (moderate-high) | 6-18 (high) | **LOW** |
| **deep cycle** | 12000-12031 | 37.5 (stable) | 18.5 (very high) | **LOW** |

The phases differ STRUCTURALLY:
- Derivation: model is doing high-compute work on uncertain content
- Cycle: model is doing moderate compute on highly-deterministic content
- The compute MAGNITUDE doesn't differ much (37 vs 37.5); but the
  MARGIN BAND differs MASSIVELY (low vs very high)

## H1868's framework predicts position-dependent codec aggressiveness

Codex's H1868: "the deployable unit is case/band-local. A packet can be
selected per layer/substrate only after checking local source-margin
bands."

My data ADDS temporal axis: the source-margin band varies POSITIONALLY
within a single sample. The same model on the same prompt has
high-margin positions (cycle) and low-margin positions (derivation).

**Codec policy implication**: codec aggressiveness should be
TEMPORALLY GATED:
- Cycle positions (high margin, deterministic content): aggressive codec
  is SAFE — model is committed; small numerical errors won't flip the
  output
- Derivation positions (low margin, uncertain content): conservative
  codec REQUIRED — small numerical errors WILL flip the output

The same packet (same codec) is safe at one phase and risky at another.
This is the TEMPORAL extension of codex's case-local discipline.

## How this anticipates codex's next move

Codex H1869 announced their next probe: "H1869 should be a tinygrad-
controlled reproduction of the relation, not a generic margin toy. It
must deliberately construct lower-Euclidean-error packets that harm
fragile margins and higher-error packets that preserve them."

My finding adds the temporal frame: in REAL workloads, the fragile-vs-
safe distinction tracks POSITIONS, not just cases. Constructing
synthetic adversarial packets to validate H1868 is necessary, but the
NATURAL TEMPORAL FRAGMENTATION I measured is the EMPIRICAL form of
the same phenomenon.

When codex's synthetic probe lands, my naturalistic data will be the
real-workload corroboration.

## The unified picture — 8 axes + temporal granularity

Codex H1869's 8-axis live unit:
```
(model, layer, route substrate, hidden profile, routed FFN delta,
 readout margin band, precision/codec, temporal attractor state)
```

Each axis is empirically separable. The deployable codec MUST be
characterized along all 8 axes. My L31 measurement is:
- model = Qwen3.5-4B
- layer = 31 (last, full-attn)
- route substrate = N/A (Qwen3.5-4B has no MoE routing)
- hidden profile = residual stream pre-L31
- routed FFN delta = N/A
- readout margin band = empirically varies 0.0-18.5 across positions
- precision/codec = 4bit MLX
- **temporal attractor state**: pre-cycle (positions 2000-8500),
  cycle-entry (10000-11500), deep-cycle (12000+)

For Qwen3.5-4B, axes 3 + 5 are not applicable (no MoE). The active
axes are 1, 2, 4, 6, 7, 8 — six independent dimensions per substrate
characterization.

For DS4 (codex's substrate), all 8 axes are active.

## Cross-arc bidirectional explicit citation

H1869 is the first codex shift that EXPLICITLY incorporates my
temporal-attractor finding as a NAMED axis in their model. This is
beyond the implicit convergence of the prior 7 cycles — codex is now
USING my axis name as a load-bearing element of their framework.

The reverse: I've been integrating codex's "isolated knobs lie",
"axis separability", "false-safe metrics" doctrines throughout my
work. Bidirectional explicit borrowing.

This is the architectural endpoint of silv's parallel-agent design.
Both agents are now using the OTHER agent's findings as load-bearing
inputs to their own framework. The cross-arc isn't just convergence
— it's INTEGRATED OPERATING PICTURE.

## What's still untested

For Qwen3.5-4B specifically:
- L31 attn vs MLP decomposition (what KIND of compute dominates the
  37.5 ||delta||?)
- Cycle ENTRY mechanism (at what specific position does the model
  transition from derivation to cycle?)
- KV cache decoding at L31 cycle positions (what context has the
  model built up?)
- Cell invariance: does this finding replicate on AIME P02, P05, P09?

For DS4 (codex's domain):
- Codex's tinygrad-controlled synthetic adversarial packet construction
- Whether temporal attractor concept maps to DS4 routing (DS4 has
  different attractor dynamics due to MoE)

## My next probe candidate

Cell-invariance test on Qwen3.5-4B AIME P09 (which fails to commit in
all tested precisions). If P09 also exhibits the global cycle attractor
+ L31 dominance, the substrate finding generalizes across cells. If
P09 shows different structure, the cycle finding is P01-specific.

This is the cell-axis (axis 0 of the 8-axis live unit). Currently
untested.

Wall: ~5 min on 4bit. Pre-declared:
- H_CELL_INVARIANT: P09 shows same cycle-period determinism at L31
- H_CELL_SPECIFIC: P09 has different attractor structure or no cycle

## Files

- This memo: temporal_position_aware_codec_policy.md
- Source data: l31_delta_vs_margin_result.json
- Codex H1869: /Users/silv/cl/tlp_codex/CODEX_SHIFTS.md (file 23:06:24)
- Companion: h1868_case_margin_band_localizer.md
- Companion: l31_delta_vs_margin commit ddc4f71
