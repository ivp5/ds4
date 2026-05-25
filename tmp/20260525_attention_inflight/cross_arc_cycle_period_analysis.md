# Cross-arc convergence has its OWN cycle period — ~25-30 minutes

silv 2026-05-25: this session has produced 4 consecutive cross-arc
convergence cycles between my work and codex's. Naming the pattern.

## The observed cycle

| time | my action | codex action |
|------|-----------|---------------|
| ~21:25 | post-compact resume; started precision-attractor work | (idle) |
| 21:37 | (working on bf16 probe) | **shipped H1855-H1857 — L22 codec convergence** |
| ~21:50 | bf16 probe completes; 3-precision map; committed memos | (working) |
| 21:58 | (writing follow-up memos) | **shipped H1860-H1861 — read my memo, added precision axis** |
| ~22:13 | wrote multi-precision ensemble + self-refutation memos | (working) |
| 22:23 | (writing H1862/H1863 integration) | **shipped H1862-H1863 — read my README, probed early layers** |
| ~22:30 | committed early-layer integration + ran L22 probe | (idle / working on next) |

**Period: ~25-30 minutes per cycle, 4 cycles observed.**

## What drives the cycle period

The cycle is bounded by:
1. **Tool wall**: codex's codec probes take ~5-10 min wall (read source + run)
2. **Memo wall**: my analysis memo writing takes ~5-15 min
3. **Read wall**: each agent reading the other's latest output takes ~30s-2min
4. **Commit wall**: git operations + memo formatting

The slowest leg dominates. Currently it's the codec probes (codex side)
+ my analysis memos (my side). When my work is just memo-writing
(no probe), my cycle is ~10-15 min. When I run a probe (like the L22
norm probe currently running), my cycle is ~15-20 min.

## The structural pattern

**Each cycle adds a NEW axis** to the shared live unit. From codex's
H1861, the live unit is:
`(model, layer, hidden profile, routed FFN delta, readout margin band, precision/codec)`

Cycle 1 (H1855-H1857): added "layer is not just one regime" (L22 special)
Cycle 2 (H1860-H1861): added "precision/codec" axis (read my work)
Cycle 3 (H1862-H1863): extended "layer" axis to EARLY depths (L0/L4/L10)

**Each cycle REFINES the cartography of the shared structural space**
without overlap. Each agent works in their domain (substrate inference
vs codec quality) but the META-FINDING ("isolated knobs lie; layer-
conditioned analysis required") is shared and grows.

## Why this works

silv's parallel-agent architecture has emergent properties:

1. **Independent measurement axes** — substrate norm (mine) vs codec
   rel-L2 (codex) — produce non-overlapping evidence about the same
   underlying structural fact (L22 is special)

2. **Asynchronous integration** — neither agent waits for the other;
   each works at their own cycle pace; reading happens at memo-publish
   boundaries

3. **Memo-as-channel** — the only communication is via filesystem
   memos that each agent reads. No coordination protocol. The
   structural findings ARE the message.

4. **Shared meta-doctrine** — both agents have read silv's CLAUDE.md
   and apply the same critical method ("doubt every step", "isolated
   knobs lie"). When both agents apply the same doctrine to different
   evidence, they converge on the same structural claim.

5. **silv as the asymmetric driver** — silv directs both agents but
   doesn't force convergence. The convergence is an EMERGENT PROPERTY
   of the architecture, not a designed handoff.

## Cycle 5 prediction

If the pattern holds:
- My L22 norm probe completes (~5 more min wall)
- I commit findings memo (~5-10 min wall)
- Codex reads my memo (~30s)
- Codex publishes new probe (~5-10 min wall) extending another axis

**Time estimate for next cycle**: ~25-30 min, so ~22:55-23:00.

If my L22 probe finds "norm-explosion is precision-invariant," codex
might extend the L22 codec finding to multiple precisions. If "content-
dependent," codex might re-examine L22 codec at in-cycle positions.

Either way, the next cycle's axis-addition is informed by what my probe
finds. The cross-arc loop is now structurally tight enough that one
agent's finding determines the other's next move.

## Implications for silv's research strategy

The 25-30 min cross-arc convergence cycle is FAST relative to most
research processes. silv has effectively built a 2-agent feedback loop
that produces emergent structural cartography at ~12-25 axis-additions
per day if sustained.

The doctrine to keep alive: **don't pre-coordinate**. The convergence
works because each agent works in their own domain and the SHARED
DOCTRINE filters which evidence is structurally interesting. Pre-
coordination would collapse the convergence into a single perspective.

## Files

- This memo: cross_arc_cycle_period_analysis.md
- Cycle 1 evidence: h1855_h1857_L22_convergence.md
- Cycle 2 evidence: h1860_h1861_real_time_convergence.md
- Cycle 3 evidence: h1862_h1863_early_layer_regimes.md
- Codex source: /Users/silv/cl/tlp_codex/CODEX_SHIFTS.md (this hour)
