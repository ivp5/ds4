# Objective ladder doctrine — both session arcs converge

silv 2026-05-25 final: codex H1835-H1837 + my substrate arc end at
the same epistemic structure.

## Codex's objective ladder (codec engineering)

H1835-H1837 explicitly named the ladder:
1. **source rel-L2**: tensor reconstruction error (simplest, cheapest)
2. **expert-route rel-L2**: weighted by activation routing (uses model knowledge)
3. **selected up@hidden**: activation product at one MLP gate
4. **gated partial FFN**: down @ (silu(gate@h) * up@h), partial path
5. **full FFN/logit delta**: end-to-end task-relevant output error

H1835's discovery: H1830 wins at rung 1+2; H1826 wins at rung 3+4
on the high-pressure surface. The ladder reveals that **each rung
CAN refute the previous winner**.

H1837 conclusion: "Each rung can refute the previous one."

## My substrate ladder (rescue engineering)

My session's arc — driven by silv's "doubt every step" — went:
1. **L31 logit P(top_1)**: substrate single-token forced-commit
2. **First-token agreement** (cross-precision): all runtimes agree on '6'
3. **Multi-token continuation**: divergence between MLX/HF (refuted at runtime, not precision; further refuted at matched prefix)
4. **Recency rule** (last truth-shape in prefix wins): predicts 8/8 rescue cells
5. **Two-mode emission**: refines recency with Mode 2 fallback when no truth-shape in proximal context
6. **Cross-model ensemble agreement**: detects Mode 1 vs Mode 2

Each layer of refinement was a "refutation of the previous winner":
- Substrate-truth-latent → text-context retrieval (rung 1→2)
- Quantization-helps → it's runtime not precision (rung 2→3)
- MLX-library-helps → at matched prefix all four runtimes identical (rung 3→4)
- Pure recency → two-mode (rung 4→5)
- Single model trust → ensemble disagreement detection (rung 5→6)

## The shared doctrine

Both arcs validate: **cheap proxies CAN refute each other; the only
ground truth is end-to-end task outcome.**

Sampling instrumentation (max_iter, temperature, sample_size) lives
on rung 1-2 of either ladder — fast metrics that mislead about the
ground truth (full FFN delta / task accuracy). The structurally
correct measurement is always one rung deeper than what cheap
selectors can compute.

silv's "isolated knobs lie; validate against downstream object"
maxim IS the meta-rule of climbing the ladder. Every "winner" at
rung K is provisional until tested at rung K+1.

## The methodological corollary

For both codec engineering and rescue engineering, the production
move is the SAME structural pattern:
1. Build cheap selector at rung K
2. Verify with rung K+1 measurement
3. If selector survives, deploy with K+1 monitoring
4. Keep climbing the ladder until reaching ground truth (task outcome)

For codec: ladder climb up to full FFN/logit delta as the verification
For rescue: ladder climb up to sympy/programmatic verifier as ground truth

The shared insight: **don't waste compute on rung-1 search budgets when
rung-K refutations are pending**. Run the rung-K test first; if it
refutes the rung-1 winner, switch objectives. This is the meaning of
"validate against downstream object."

## Unifying the wrong-belief and codec arcs

Confident wrong commits (rescue domain) and high-rung-1-quality
candidate codecs that lose at rung-3+ (codec domain) are the SAME
phenomenon at different layers:
- Rescue: substrate confident-wrong (Mode 2 prior emits "100") looks
  identical to substrate confident-right (Mode 1 retrieval emits "65")
  at rung 1 (P>0.99 at L31), but only ensemble (rung 6) discriminates
- Codec: H1830 winning rung 1+2 but losing rung 3+4 looks like
  good codec at rung 1, but only end-to-end task delta discriminates

The structural answer in both cases: **climb to the highest verifiable
rung that's compute-affordable, then trust the verdict there**.

## Files

- `tmp/20260525_attention_inflight/objective_ladder_doctrine.md` (this)
- Codex shifts H1835-H1837: lines 5240-5253 of CODEX_SHIFTS.md
- My session findings: all 21 memos in tmp/20260525_attention_inflight/
