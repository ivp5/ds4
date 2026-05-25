# H1853/H1854: "Max-quality VQB2" REFUTED — use H1830, not max-q tensor winner

silv 2026-05-25 directive said: "use the latest max-quality vqb2 encoding."

Codex shipped H1853/H1854 minutes ago (file timestamp 21:22). The finding:

## H1853 — pushed visible quality knobs hard
- max_iter=50000, fit_sample up to 1M, multiple seeds
- All candidates converged in 6-7 effective Lloyd iterations
- max_iter REMAINS inert (ceiling, not quality dial)
- A pure-tensor winner: `s777_n1000k_i50000` improved tensor rel-L2
  from H1830's 0.159477131512 → 0.158817773556 = **0.4134% better**

## H1854 — projected through routed-FFN/logit-margin scorer
- Target `kind:router_row_mix|pressure:high`:
  - H1830 stays best: case mean 0.168064517238, top-k p95 0.001051254920
  - H1853 max-q WORSE: case mean 0.171582112701, top-k p95 0.003010844562

## Shift
> "Max quality is not max tensor fidelity. Larger samples and lower
> global rel-L2 can damage the selected routed cases that matter
> for logits. The packet objective must remain downstream route/logit-
> margin aware; tensor rel-L2 winners are now useful negative controls,
> not defaults."

## Implication for silv's directive

silv asked me to "use the latest max-quality vqb2 encoding". The
NAMING is misleading — H1853 IS the latest max-quality VQB2 by tensor
rel-L2, but H1854 REFUTED it as a packet to ship.

The CORRECT encoding to use per H1854 is **H1830** (s42_n100k), the
prior winner that holds under downstream routed-FFN/logit-margin
evaluation.

If integrating codex packets into trim50 via Q4-splice:
- L25 K64 VQB2: use H1830's s42_n100k packet (NOT H1853 max-q)
- L19 K256: use original (codex hasn't run max-q sweep there yet)
- L42 K256: use original
- Layers needing K16 (120+ packets): not yet generated

## Convergence with my session's meta-doctrine

Both arcs reached identical structural insight TODAY:
- My session: substrate L31 rank-1 truth-latent ≠ predictor of correct emit
- Codex H1854: tensor rel-L2 winner ≠ logit-margin winner
- H1849: full-vocab lens ≠ what the model actually consumes
- H1842: first-128 output rows ≠ representative of full output

**Isolated proxy metrics LIE about downstream behavior. The only valid
selector is the consumed downstream object.**

This is the SAME meta-doctrine in TWO independent arcs in real time.
silv's running both sessions; both surfaced the same mechanism.

## What this means for the merge integration plan

Item H from the synthesis (trim50 codec replacement with codex H1812
K16+K256 mix) now reads:
- Use H1830 packets where available (L25 K64 VQB2 ships from AMD)
- DO NOT use H1853 max-q packets — refuted at logit-margin objective
- L19 + L42 K256 packets still needed at base quality (codex hasn't
  re-run those with margin-aware objective)
- 120+ K16 packets still pending generation

The MERGED BINARY can already load polar/VQB1/VQB2 packets via
ds4_polar_reader.c + ds4_vqb1_reader.c (per merged source). The
remaining work is:
1. Generate the missing K16 packets on AMD with margin-aware objective
2. scp packets to M1
3. Configure ds4 with DS4_POLAR_DIR / DS4_VQB1_DIR / similar env var
4. Smoke test the new layout

## Files

- This memo: tmp/20260525_attention_inflight/h1853_h1854_max_quality_refuted.md
- Codex source: /Users/silv/cl/tlp_codex/CODEX_SHIFTS.md H1853-H1854 (lines 5354+)
- Prior synthesis: H1850_directive_synthesis.md (H1830 selection corrected)
