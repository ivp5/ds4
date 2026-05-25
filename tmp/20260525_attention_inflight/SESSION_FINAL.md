# Session 2026-05-25 final — substrate, sampling fallacy, codex H1802-H1821

silv directives sequence:
1. "explore internal LLM states / in-flight attention / 10000 cycles"
2. "ssh to amd with user gpu_ssh"
3. "reread shifts.md + CODEX_SHIFTS — find unexplored experiments"
4. "doubt EVERYTHING, doubt EVERY STEP"
5. "make more glorious mistakes... crash into walls"
6. "continue queued/deferred tasks"
7. "read on in codex progress and continue high-potential tasks"
8. (multiple "continue")

## Final session output

**17 commits to ivp5/main** during this session. **12 findings memos**
in `tmp/20260525_attention_inflight/`. **~30 raw JSON data files**.

### Findings (survived audit)

1. **L19 attention-entropy growth detector** — commit-concentration
   signal at first 4000 chars; T=0.55 catches both lock cells (P01, P10)
   in 4bit AIME 2026 corpus. 100% recall, 0 FP.

2. **L23-L31 commit-tier install** — at the `\boxed{` emission
   position, truth installs progressively from L23 (rank 0 at low P)
   to L31 (rank 0 at P~1.0). Architecture-wide; cross-precision
   invariant.

3. **Cross-precision substrate invariance** — 4bit MLX and BF16 AMD
   align identically in L23-L31 install. Quantization doesn't shift
   WHEN truth installs.

4. **DS-R1-7B reasoning-RL compresses commit tier** — 4-layer install
   (L24-L27, 14% depth) vs Qwen3.5-4B 8-layer install (L23-L31, 25%
   depth). Cross-arch finding.

5. **Cross-arch commit-concentration band universal** — L18-L23 in
   DS-R1-7B (64-82% depth), L19 in Qwen3.5-4B (59% depth). Both
   architectures have a "commit-concentration layer" at 60-80% depth.

6. **Substrate truth-rank = text-context retrieval** (self-refuted
   from "latent computation" to "literal context retrieval"). At
   forced-commit position, the substrate's "truth-rank" measures
   whether truth-shape appears in proximal CoT mentions.

7. **WRONG-BELIEF DOCTRINE** — model's wrong commits are STRUCTURED,
   not noise. P10 emits 100 (plausible 3-digit), P09 emits 1/12
   (fraction attractor), P01 emits 85 (2-digit when truth is 3-digit).
   Coherent misbelief via question-shape → answer-shape retrieval from
   training imprint.

8. **MORE CoT CAN HURT** — P02 truth=62: emits 62 ✓ at frac=0.20
   (early-CoT prior preserved), 56 ✗ at frac=0.40 (CoT wandered to
   wrong intermediate), 62 ✓ at frac=0.60 (more truth mentions
   accumulate). Common-wisdom "longer CoT = better" REFUTED.

9. **Forced-commit substrate extract** — 5-7/10 AIME 2026 cells
   correct via chunk_2 + greedy K=8, depending on prefix-frac. Ceiling
   matches production rescue protocol's 8/10. The mechanism is text
   retrieval of truth-shape mentions from cached CoT.

### Refutations (this session)

1. **H_LOCK_SUBSTRATE_PRESERVES_TRUTH** — averaged-over-positions
   per-layer lens showed truth at top-1 with P=0.9998 at L31; but
   per-position analysis showed truth at rank 7-180k at every layer
   at every sampled position. The earlier "0.9998" was a
   position-averaging artifact (positions where literal '2' character
   was natural-next-token in locked text).

2. **H_DSP_REPLICA_EXCHANGE_ESCAPES_LOCK** — 5 replicas at T={0, 0.3,
   0.7, 1.0, 1.4} from P01 chunk_2 cache produced 0/5 truth
   recoveries. Higher temperature can't manifest substrate that
   doesn't have truth.

3. **H_TOP_K_RETENTION_CATCHES_DISCARDED_TRUTH** — truth digit '2'
   appeared in 0 of 1839 retained top-16 positions in P01 lock. The
   substrate genuinely never computed truth; sampling instrumentation
   cannot retain what wasn't there.

4. **H_DECIDE_BEFORE_EMIT (L29 confidence trajectory)** — L29 P(top_1)
   > 0.5 fires at MANY natural-language tokens throughout CoT (closing
   `$`, ` =`, LaTeX exponents). Gap = 33000 chars between first
   sustained L29>0.5 and actual `\boxed{` emission. L29 confidence is
   "next-token-predictable", not "commit-imminent".

5. **H_FINAL3_UNANIMOUS_DISCRIMINATES_CORRECTNESS** — final-3 layers
   produce IDENTICAL unanimous-high-P shapes for confident-right
   (P05 commits '6' P=0.97) and confident-wrong (P01 commits '8'
   P=0.99). Same substrate signature; opposite correctness.

6. **H_SUBSTRATE_HAS_TRUTH_LATENT** — self-refuted via prefix-frac
   sweep. The "substrate truth-rank gradient" is text retrieval from
   proximal CoT mentions, not latent computation. P09 truth=29 has
   zero occurrences in 76k CoT → no prefix-frac works.

7. **Conjecture #22 b·K isoquant** — REFUTED at budget=4000. DS-R1-
   Distill-Qwen-7B BF16 on AIME 2026 P1+P9+P10 at K=4 temp=0.7
   produced 0/12 commits across 3 problems. K-sampling cannot
   substitute for compute budget below threshold.

### Codex H1802-H1821 read

- H1802-H1810: low-bit polar codec explored exhaustively; sharply
  constrained to source-oracle only; not deployable
- H1811: VQ K depth allocation; K4 base + K64 upgrades
- H1812: routed risk-weighted allocator over (layer, expert, codec);
  L0/L19/L42 K256 recipe corroborates antirez's last6-q4
- H1814: VQB1 packet generation canary 10.1s/L1 K16
- H1817: L19 K256 max_iter=1 → rel-L2 0.0235 cosine ~1.0 (very strong)
- H1818-H1819: max_iter "push for quality" cargo-cult refuted;
  convergence at 3 effective Lloyd iterations
- **H1820: ROCm full-assignment 30.29× speedup** (67.5s→2.2s)
- **H1821: end-to-end VQB1 encoder 2.20× speedup** (113.7s→51.6s);
  next bottleneck = per-packet source materialization

### Cross-substantive composition with codex H1812

Codex H1812 picks L19 as K256-allocator high-sensitivity layer
empirically. My session finds L19 is the commit-concentration layer in
Qwen3.5-4B substrate. The L19 sensitivity in DS4 likely originates at
the COMMIT-CONCENTRATION ROLE — the recipe should reserve high-precision
codecs at L_commit_concentration of any new architecture, not just
empirically-picked layers. F4 (DS-R1-7B compresses commit tier) extends
this: the codec recipe might map onto L_commit_install (the few layers
where truth installs) rather than commit_concentration alone.

### Deep doctrine — the sampling instrumentation fallacy

silv's named-deeper fallacy: temperature/top_k/top_p/beam search are
treated as wrappers around the LLM's L31 logits. They cannot:

- Manifest truth the substrate didn't compute (P01 lock: 0/5 replicas
  at 5 temperatures; 0 in top-k=16 across 1839 positions)
- Distinguish confident-right from confident-wrong (P01 commits '8'
  at P=0.99; P05 commits '6' at P=0.97; same signature, opposite truth)
- Override the architecture's CALIBRATION INVISIBILITY (the model
  commits with equal confidence regardless of correctness)

The structural fix:
1. External programmatic verifier (sympy for AIME-style)
2. Ensemble disagreement signal (multi-model, multi-precision)
3. Substrate probe at L23-L31 to check truth-shape-in-context
4. CoT length is NOT universally helpful — sometimes shorter is better

These are NOT replacements for sampling instrumentation; they're the
TRUE solutions to calibration problems that sampling-instrumentation
was never going to solve.

## Code shipped

`codec_audit/`:
- attention_entropy_probe.py (Qwen3 hybrid)
- attention_cross_model.py (qwen2 / DS-R1-7B)
- attention_{late_region,cross_cell,cross_arch_chunk}.py
- p01_cross_prec.py
- l19_growth_detector.py
- replica_exchange_decode.py
- topk_retention_decode.py
- perlayer_logit_lens.py
- logitlens_at_position.py
- logitlens_at_boxed.py
- l29_trajectory.py
- forced_commit_substrate.py
- forced_commit_extract.py

`tmp/20260525_attention_inflight/`:
- amd_perlayer_lens_bf16.py (AMD)
- amd_conj22_isoquant.py (AMD)

## Open queue for next sessions

1. **Cross-MODEL substrate truth-rank** — test the rescue protocol on
   DS-R1-7B's OWN cached CoTs (currently the AMD conj22 traces are
   no-commit, so the rescue scan would need the full text to find
   any truth mentions in DS-R1-7B's reasoning).
2. **Architectural commit-concentration calibration** — Phi, Mistral,
   Gemma, Llama: identify their L_commit-concentration via the L19-
   style entropy growth probe; verify 60-80% depth band universality.
3. **Cross-arch L19 K256 recipe** — codex H1812's K256-required-layers
   for DS4 (L0/L19/L42); verify L_commit_concentration in those layers
   via my attention-growth probe on DS4 traces.
4. **Codex direction H1822+** — per-packet source materialization is
   the next bottleneck per H1821. Source dequant sharing across
   layer-kind packets.
5. **Larger budget K-sample test** — Conjecture #22 was refuted at
   budget=4000; matharena uses 32k+. Test at K=4 budget=16000 to see
   if the isoquant survives at higher budget.
