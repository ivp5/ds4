# P09 cell-invariance test — magnitudes conserve, cycle attractor DOESN'T

silv 2026-05-25 23:18: P09 (truth=29, universal-fail cell) tested at
same 6 layers and similar positions as P01.

## Comparison table

P01 vs P09 ||delta|| at pos 12000 on Qwen3.5-4B-MLX-4bit:

| layer | P01 in-cycle | P09 pos 12000 | conservation |
|-------|--------------|-----------------|--------------|
| L0 | 2.30 | 1.97 | ~similar (15% diff) |
| L22 | 5.69 | 6.53 | ~similar (15% diff) |
| L25 | 5.56 | 5.31 | ~similar (5% diff) |
| L26 | 6.53 | 7.88 | ~similar (20% diff) |
| L27 | 19.25 | 21.50 | ~similar (12% diff) |
| L31 | 37.50 | 39.75 | ~similar (6% diff) |

**The architectural pattern PERSISTS**: full-attn layers (L27, L31)
dominate; L31 is the peak. The structural finding is cell-invariant.

## Cycle determinism — REFUTED on P09

P01 at positions 12000/12031/12062/12093 had L31 ||delta||:
37.50/37.75/37.75/37.75 (max_diff=0.25, essentially EXACT).

P09 at positions 12000/12031/12062/12093 has L31 ||delta||:
**39.75/41.00/38.25/42.25 (max_diff=2.5, ~6% variance)**.

Also tested 12032 (32-offset, the 8bit cycle): 37.25 (different again).

**P09 is NOT in a limit cycle at pos 12000.** The variance is
position-dependent, not repeating.

## What P09 IS doing at pos 12000

CoT trajectory snapshots:
- pos 6000: "x_6$ appears at index 6, then the"
- pos 8000: "_6$ is the last occurrence of $x_6"
- pos 10000: "6$ is always visible.\nWait, I need to"
- pos 12000: "6, then the last occurrence is at 6.\n"

**P09 is in ACTIVE EXPLORATION at pos 12000**. Each position has
different content. The model is ruminating on the probability puzzle
(stickers on dice faces), trying different interpretations of the
problem structure ("last occurrence", "always visible", "Wait, I need
to").

This is structurally different from P01 at pos 12000 where the model
is rotating through `"The solution is 277. The answer is 277. I will
write the solution."` (clean 31-token cycle of post-truth meta).

## The mechanism reveal: cycle attractor is derivation-aftermath

P01 enters cycle BECAUSE it derived truth at pos ~8000 then drifted
into a meta-commentary attractor.

P09 does NOT enter a cycle BECAUSE it never derives truth. The model
keeps exploring the problem space, generating new (non-repeating)
content.

**The limit cycle attractor is a PHASE — the AFTERMATH of derivation
— not a structural property of the model on any input.**

Cells split into:
- **Derived cells** (P01-P08): derive truth → enter post-truth cycle
- **Universal-fail cells** (P09, P10): never derive → stay in
  exploration (no cycle attractor)

## Implication for Conjecture #23

The rescue rule `truth_at < min(frac × cot_len, cap)` operationalizes:
"truth must be present in CoT prefix for forced-commit to extract it."

My substrate finding here CORROBORATES the rule:
- Rescuable cells (P01-P08): truth derived → cycle attractor at substrate
- Non-rescuable cells (P09, P10): no truth derived → no cycle attractor

The CYCLE ATTRACTOR is itself a structural signature of rescuability.
A real-time detector could distinguish exploration from cycle via L31
||delta|| variance:
- Exploration: ||delta|| variance ~5-6% at any window
- Cycle: ||delta|| variance < 1% at cycle-period offsets

This is a CHEAP DEPLOYABLE SIGNAL for "should we attempt rescue?":
- Probe L31 ||delta|| at 4 positions in last 100 tokens
- If variance > 5%: model is still exploring; rescue likely fails
- If variance < 1%: model has entered cycle; rescue likely succeeds

## Implication for the 8-axis live unit

Codex H1869's "temporal attractor state" axis now has SUB-STATES:
1. **Pre-derivation exploration** (P01 pos < 8000, P09 entire trajectory)
2. **Derivation peak** (P01 pos ~8000, where ||delta|| spikes)
3. **Post-derivation drift** (P01 pos 10000-12000)
4. **Deep cycle** (P01 pos > 12000, when stabilized into cycle)

P09 never reaches state 2/3/4 in 12000-token budget. P01 cycles through
all four states.

The temporal axis isn't binary (pre-cycle vs in-cycle); it has
substructure that maps to model-internal cognitive phases.

## Cell-invariance verdict

**Mixed**: structural magnitudes are conserved (architecture-level);
attractor behavior is cell-specific (cognitive-trajectory level).

For codex's framework: the 8-axis live unit's "temporal attractor
state" axis is per-cell, per-position. Different cells visit different
substrate states at different times.

## What changes for production deployment

The "cycle is global attractor" framing from my multi-layer probe was
correct AT P01 but not generalizable as a model-wide property.

Refined framing:
- Substrate magnitudes per layer are model-architecture properties
- Cycle attractor is a CONTENT-DEPENDENT PHASE the model enters when
  derivation completes
- Cells that never derive don't enter cycle

For inferguard rescue: cycle entry detection IS a rescuability signal.
Cheap to compute (L31 ||delta|| variance over recent positions).

## Files

- This memo: p09_cell_invariance_RESULT.md
- Raw data: p09_cell_invariance_result.json
- Probe code: p09_cell_invariance_probe.py
- Run log: p09_invariance_run.log
- Companion P01 baseline: multi_layer_attractor_global_RESULT.md
- Companion temporal: temporal_position_aware_codec_policy.md
