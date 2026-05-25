# H1855-H1857 + my substrate-vertical L22 finding — independent convergence

silv 2026-05-25: codex shipped H1855-H1857 just now (file 21:37).

## Codex H1855-H1857 finding

**H1855**: arbitrary-layer route/hidden cache generator. L22 pressure
surface differs from L25:
- L22: high=64, medium=78, low=2
- L25: high=55, medium=85, low=4

**H1856**: L22 tensor winner DIFFERS from L25:
- H1853's max-q packet `s777_n1000k_i50000`:
  - L25: tensor improves but logit-margin REFUTED (H1854)
  - L22: tensor improves 3.735% (0.166 → 0.160)
- H1830's `s42_n100k_i50000`:
  - L25: wins
  - L22: WORSE at 0.174

**H1857**: L22 max-q tensor winner ALSO wins logit-margin:
- case mean: 0.151 → 0.148 (improved)
- top2 p95: 0.0039 → 0.0027 (improved)
- top-k p95: 0.0087 → 0.0052 (improved)

**Shift**: "H1854's L25 result was NOT a universal law. Layers are NOT
interchangeable repeated blocks; packet policy must be LAYER-CONDITIONED
and OBJECTIVE-CONDITIONED. L25 is an objective-disagreement negative
control, while L22 is a positive transfer case."

## My session's L22 finding (prior session, 2026-05-23)

From `tmp/20260523_substrate_vertical/findings_memo.md`:

| L | type | ‖x_in‖ | ‖attn_r‖ | ‖mlp_r‖ | ‖Δ‖ | mlp/attn | cos(attn,mlp) |
|--:|:-----|-------:|---------:|--------:|-----:|---------:|---------------:|
| 21 | LIN | 12.19 | 4.75 | 6.94 | 8.38 | 1.46 | -0.009 |
| **22** | **LIN** | **15.31** | **4.62** | **25.25** | **27.00** | **5.46** | **+0.295** |
| 23 | FULL | 34.00 | 21.62 | 8.38 | 27.00 | 0.39 | +0.538 |
| 24 | LIN | 23.62 | 4.78 | 8.06 | 9.56 | 1.69 | +0.043 |
| 25 | LIN | 25.88 | 6.38 | 12.44 | 14.38 | 1.95 | +0.074 |
| 26 | LIN | 31.75 | 11.12 | 53.25 | 62.75 | 4.79 | +0.807 |

**L22 = norm-explosion MLP-dominant peak**: mlp/attn = 5.46×, ‖Δ‖=27.
**L23 = correction**: full-attn retrieves to integrate L22's MLP injection.
**L25 = mid-rung continued growth**: mlp/attn = 1.95× (much less peaked).

## The convergence

My ATTENTION-PROBE finding: L22 is structurally special (norm explosion,
MLP-dominant 5.46× peak).

Codex CODEC-PROBE finding: L22 is structurally special (max-q packet
behavior DIFFERS from L25; max-q WORKS at L22, REFUTED at L25).

**Independent measurement axes converge on the same answer**:
"L22 ≠ L25; the layer position MATTERS for both representation and
quality of compression."

## Why this matters for the merge / production deployment

The merged binary ships with hooks for layer-conditioned codec
selection (per ds4_polar_reader.c + ds4_vqb1_reader.c). The
implication of H1857:

**Production codec deployment should NOT apply one packet to all layers.**
Per-layer codec choice matters:
- L22-class (norm-explosion / MLP-dominant): max-q packets WORK
- L25-class (mid-rung accumulator): max-q packets REFUTE; use H1830 base

This is silv's earlier directive about "use the latest max-quality
VQB2 encoding" — RIGHT for SOME layers (L22), WRONG for OTHERS (L25).
The per-layer table is needed, not a global policy.

## Methodology lesson — same as my session's

Both arcs derived from independent observations the same meta-doctrine:
- "Same mathematical shape ≠ same performance decision" (codex H1852)
- "Layer-function differentiation is real" (my substrate-vertical L22 finding)
- "Isolated proxy metrics lie; downstream object discriminates" (both arcs)

The convergence is not coincidence. It's the structural fact: in any
deep network with distributed-encoding-accumulator computation, the
load-bearing layers ARE different from the support layers, and quality
metrics applied uniformly across layers MISMEASURE.

## What this means for next session

The L22 finding is now ELEVATED to a load-bearing structural claim
backed by TWO independent measurement axes (attention probes,
codec quality probes).

The next-session integration plan:
1. For codex's per-layer codec ladder: use H1857-derived recipe
   - L22-class: max-q VQB2 (s777_n1000k_i50000)
   - L25-class: base VQB2 (s42_n100k_i50000)
   - L0/L19/L42: K256 high-fidelity (H1812 risk-weighted)
   - Other layers: K16 base (still missing per H1813)
2. For my session's per-layer ablation queue: L22 specifically deserves
   single-layer ablation experiments to confirm it carries the
   "compute injection" mechanism
3. For temporal dynamics: does the L22 norm-explosion happen at EVERY
   token in the limit cycle, or does it quiesce inside the cycle?

## Files

- This memo: tmp/20260525_attention_inflight/h1855_h1857_L22_convergence.md
- Codex source: /Users/silv/cl/tlp_codex/CODEX_SHIFTS.md H1855-H1857
- My prior L22 finding: tmp/20260523_substrate_vertical/findings_memo.md
