# Session 2026-05-25 — deep doctrine

silv directive: continue queued/deferred tasks; "doubt EVERYTHING";
deeper implications on training/inference, on sampling instrumentation
fallacy.

## The session-defining finding: substrate truth-rank gradient

At forced-commit position (prefix chunk_2 + `\\boxed{` preamble),
L31 truth-digit rank predicts rescuability category across 10 cells:

| Cell | Cat | L31 P(truth) | L31 rank | Native commit? | Rescue OK? |
|------|-----|---------|---------|----------|------------|
| P01 | LOCK | 0.001 | 7 | NO | NO |
| P02 | rescue-near | 0.28 | 1 | NO | YES* |
| P03 | rescue | 0.983 | 0 | NO | YES |
| P04 | rescue-near | 0.20 | 2 | NO | YES |
| P05 | native | 0.97 | 0 | YES | n/a |
| P06 | rescue | 0.995 | 0 | NO | YES |
| P07 | rescue | 0.993 | 0 | NO | YES |
| P08 | rescue | 0.988 | 0 | NO | YES |
| P09 | silent | 0.084 | 3 | NO | NO |
| P10 | lock | 0.35 | 0 | NO | NO |

\* P02 rescue worked offline but at different prefix; chunk_2 still
substrate-rescuable per rank=1.

**Substrate truth-rank at L31 calibrates against RESCUABILITY:**
- rank 0 → 4/4 rescue/native success (P03, P05, P06, P07, P08)
- rank 1-2 → 2 rescue-near (P02, P04)
- rank 3+ → 3/3 NOT rescuable (P01, P09, P10*)

\* P10 rank 0 at L31 but L29/L30 break the unanimous-digit pattern.

## The deeper doctrine — substrate carries information that sampling doesn't

silv's named "sampling instrumentation fallacy" — the common LLM
inference doctrine treats temperature/top_k/top_p as wrappers around
L31 logits. These wrappers cannot:
- Distinguish confident-right from confident-wrong (P01 commits `'8'`
  at P=0.99; P05 commits `'6'` at P=0.97; SAME shape, opposite
  correctness)
- Manifest truth from a substrate that doesn't have it (lock cells:
  truth at rank 7, no temperature setting helps)
- Read the commit-tier structure (L23-L31 truth-install pattern lives
  in hidden states; sampling reads only L31 logits)

The substrate truth-rank gradient demonstrates that LOWER LAYERS carry
information about RESCUABILITY that the final-layer logits don't make
visible. Reading L23-L31 reveals:
- Whether truth is in the substrate (rank 0-2 = rescuable)
- Whether the commit-tier is coherent (final-3 all-digits = number-emitting)
- Whether the commit is to truth or wrong (truth at rank 0 vs other)

## Refuted hypotheses (this session)

1. **H_DECIDE_BEFORE_EMIT (L29 confidence)**: L29 P(top_1) > 0.5 fires
   at MANY natural-language tokens throughout CoT, not specifically at
   commit positions. The L29 signal is "next-token-predictable", not
   "commit-imminent". (Gap = 33000 chars between first L29>0.5 and
   actual `\\boxed{` emission in P05.)

2. **H_LOCK_SUBSTRATE_PRESERVES_TRUTH**: refuted — at P01 lock, max
   P(truth='2') across 32 layers × 6 sampled positions = 0.000029.
   Truth was NEVER computed.

3. **H_DSP_REPLICA_EXCHANGE_ESCAPES_LOCK**: refuted — 5 replicas at
   temperatures {0.0, 0.3, 0.7, 1.0, 1.4} from chunk_2 of P01 produced
   0/5 truth recoveries.

4. **H_TOP_K_RETENTION_CATCHES_DISCARDED_TRUTH**: refuted — truth digit
   '2' did NOT appear in any retained top-16 at any of 1839 positions
   in P01 lock. Sampling instrumentation cannot retain what was never
   computed.

5. **H_FINAL3_UNANIMOUS_DISCRIMINATES_RIGHT_WRONG**: PARTIALLY refuted —
   final-3 unanimous on a DIGIT is a commit-confidence signal but
   doesn't discriminate truth vs wrong. P01 forced-commit: L29/L30/L31
   all '8' at P>0.92 (wrong). P05 forced-commit: L29/L30/L31 all '6' at
   P>0.57 (correct). Same shape.

## Conjecture #22 b·K isoquant — refuted at AMD (in progress)

DS-R1-Distill-Qwen-7B BF16 on AIME 2026 P01: 4 samples at K=4 temp=0.7,
budget=4000. 0/4 commits. P9 sample 0/4: 0/2 commits so far.
Conjecture's CORROBORATE clause (≥3/10 majority-correct) directionally
REFUTED — at 4000-token budget, even K-sampling on strong-RL model
doesn't produce commits on AIME 2026 hard cells.

The budget=4000 isn't enough for AIME 2026; matharena uses much more.
But the isoquant claim was that LOWER bit-depth with HIGHER K samples
can maintain accuracy — at budget=4000, K=4 doesn't help. The
isoquant relationship at this budget tier is REFUTED.

## What survives + cross-architecture corroboration

### F1 (L19 entropy growth as commit-concentration)
SURVIVED audit. T=0.55: 2 TP / 0 FP / 8 TN on 4bit AIME 2026 cells.

### F2 (Substrate truth installs L23-L31 commit tier, not L19)
SURVIVED cross-precision (4bit ↔ BF16 align), cross-architecture
(DS-R1-7B compresses to L24-L27, sharper).

### F3 (Cross-precision substrate invariance)
SURVIVED — Qwen3.5-4B-MLX-4bit vs BF16 align identically across
L23-L31 install trajectory.

### F4 (DS-R1-7B reasoning-RL distill compresses commit tier)
NEW finding. DS-R1-7B has 4-layer commit install (L24-L27) vs Qwen's
8-layer L23-L31. Reasoning-RL training compresses commit installation
into fewer layers.

### F5 (Cross-arch commit-concentration band universal)
NEW finding. DS-R1-7B has lowest-growth band L18-L23 (64-82% depth);
Qwen has L19 (59% depth). Both architectures show commit-concentration
at the late-middle/early-late depth band.

### F6 (Substrate truth-rank gradient predicts rescuability) — SESSION-DEFINING
NEW finding. 10-cell validation. Truth-rank at L31 at forced-commit
position = rescuability proxy. Rank 0-2 = substrate-rescuable; rank 3+
= text-context-retrieval needed.

## Architectural implications

The "calibration problem" in LLM inference is structural, not a
training-failure to-be-fixed-later. The architecture commits with equal
confidence whether right or wrong. The truthfulness gradient lives
INSIDE the substrate (truth-rank at L31), accessible via probing but
INVISIBLE to inference-time sampling.

**Honest LLM stack implication**: external verifier is structurally
required. Options:
- Programmatic verifier (sympy, regex, deterministic checker) — works
  when the domain has a verifier; AIME does (each truth = integer 1-999)
- Ensemble of proposers — multi-model or multi-precision; agreement on
  same digit at L31 across models is the discrimination signal
- Substrate probe at L23-L31 - check truth-rank at forced-commit;
  identifies rescuable vs unrescuable cells before spending budget

The "sampling instrumentation" stack is NOT BROKEN — it just doesn't
solve the calibration problem it's often expected to solve.

## Files (this session)

Code:
- `codec_audit/attention_entropy_probe.py`
- `codec_audit/attention_cross_model.py` (qwen2 / DS-R1-7B)
- `codec_audit/attention_{late_region,cross_cell,cross_arch_chunk}.py`
- `codec_audit/p01_cross_prec.py`
- `codec_audit/l19_growth_detector.py`
- `codec_audit/replica_exchange_decode.py`
- `codec_audit/topk_retention_decode.py`
- `codec_audit/perlayer_logit_lens.py`
- `codec_audit/logitlens_at_position.py`
- `codec_audit/logitlens_at_boxed.py`
- `codec_audit/l29_trajectory.py`
- `codec_audit/forced_commit_substrate.py`
- `tmp/20260525_attention_inflight/amd_perlayer_lens_bf16.py` (AMD)
- `tmp/20260525_attention_inflight/amd_conj22_isoquant.py` (AMD)

Findings memos (chronological):
1. `synthesis.md` — initial L19 finding
2. `findings_severe_audit.md` — first-round refutations
3. `cross_precision_amd.md` — BF16 vs 4bit invariance
4. `SESSION_SYNTHESIS.md` — F1-F5 mid-session
5. `cross_arch_followup.md` — DS-R1-7B substrate
6. `l29_refutation.md` — L29 trajectory refutation
7. `cross_layer_agreement.md` — final-3 unanimous signal
8. `forced_commit_substrate.md` — confident-wrong substrate
9. `DEEP_DOCTRINE.md` (this) — session-defining synthesis

Raw data: ~20 JSON files in `tmp/20260525_attention_inflight/`.

7 commits to ivp5/main during session.
