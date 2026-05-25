# In-flight attention matrix probe — L19 entropy growth as commit-concentration signal

silv 2026-05-25 directive: "explore internal LLM training/inference states,
building in-flight representations of attention matrices or applying various
dsp sampling metaphors consistently in-depth."

## Method

Class-patched `Qwen3NextAttention.__call__` (and `qwen2.Attention.__call__`
for DS-R1-7B) to manually compute attention via Q@K^T/sqrt(d) → softmax,
exposing per-(layer, head, query_position) attention WEIGHTS. The standard
fused `scaled_dot_product_attention` is opaque — manual computation costs
~2× wall but gives the matrices.

Compute per-(head, query_position) attention entropy and self-attention
mass; aggregate by layer × generation chunk (0-20%, 20-40%, ..., 80-100%).

## Substrate findings

### Both architectures share LAST-LAYER COPY SIGNATURE

| Model            | Last attn layer | self_mass | high_self frac |
|------------------|-----------------|-----------|----------------|
| Qwen3.5-4B-MLX-4bit | L31         | 0.22      | 18.6%          |
| DS-R1-Distill-Qwen-7B-4bit | L27 | 0.19      | 17.4%          |

In both models, ~18% of (head, position) cells at the LAST attention layer
have >50% attention mass on the diagonal (self-attend). This is structural —
the last layer fires copy/induction patterns broadly across the token
sequence.

### L19 entropy GROWTH suppression signals commitment-concentration

**Finding**: in Qwen3.5-4B-MLX, at L19 (full-attention norm-explosion layer),
average attention entropy NATURALLY grows from chunk_0 (prompt) to chunk_2
(40% of generation) by 0.6-1.2 nats during normal exploration. In cells
where the model concentrates into a commit (cyclic lock OR confident-wrong
commit), this growth is suppressed to ~0.5 nats.

Cross-precision P01 (HC_LOCK most_pers in parentheses):

| Precision | early_ent (0-20%) | mid_ent (20-40%) | growth | HC_LOCK pers |
|-----------|-------------------|-------------------|--------|--------------|
| 4bit      | 2.941             | 3.462             | +0.521 | 0.852        |
| bf16      | 2.874             | 3.454             | +0.581 | 0.294        |
| 8bit      | 2.879             | 3.817             | +0.938 | 0.111        |

Pearson r(growth, HC_LOCK pers) = **-0.85** (n=3). 4bit lock suppresses
growth by ~50% vs 8bit no-lock; bf16 partial-lock shows intermediate.

### Cross-cell distribution (10 cells × Qwen3.5-4B-MLX-4bit, 2026-05-25)

| Cell | Category         | growth_02 | Verdict (T=0.55)  |
|------|------------------|-----------|-------------------|
| P01  | LOCK             | +0.521    | CONCENTRATING ✓   |
| P02  | rescue           | +1.016    | EXPLORING         |
| P03  | rescue           | +0.912    | EXPLORING         |
| P04  | rescue           | +1.131    | EXPLORING         |
| P05  | native success   | +0.872    | EXPLORING         |
| P06  | rescue           | +0.866    | EXPLORING         |
| P07  | rescue           | +0.602    | borderline        |
| P08  | rescue           | +0.690    | EXPLORING         |
| P09  | silent fail      | +1.281    | EXPLORING         |
| P10  | confident-wrong  | +0.525    | CONCENTRATING ✓   |

At threshold T=0.55: 2 TP (P01 lock, P10 confident-wrong) / 0 FP / 8 TN.

Cross-precision controls:
- P02 in {4bit, 8bit, bf16}: growth = 1.02 / 0.95 / 0.96 — uniformly EXPLORING ✓
- P09 in {4bit, 8bit, bf16}: growth = 1.28 / 1.17 / 0.92 — uniformly EXPLORING ✓

## What this is + what it isn't

**IS**:
- A cheap (single forward, single layer, single statistic) in-flight signal
  detecting commitment-concentration BEFORE the lock pattern fully emerges
- Catches BOTH lock (P01) AND confident-wrong commits (P10) at threshold
  0.55, while HC_LOCK fires only on full cyclic pattern repetition
- Can be computed after only 40% of generation budget — early-warning
- Reproducible across precision (n=3 P01, n=3 P02, n=3 P09 = 9 controls)

**ISN'T** (not yet validated):
- Cross-MODEL generalized: DS-R1-7B has 28 all-attention layers, NOT 32
  with 8 full-attention; the analog of L19 may be at a different relative
  depth. DS-R1-7B reading Qwen's locked text showed normal entropy growth
  at all layers — so the lock-recognition is internal-state-dependent, not
  text-property-dependent, which is GOOD (the signal is real) but doesn't
  validate the cross-model deployability.
- Tested across multiple problems with locked behavior: P01 is the ONE
  cell with a tight cyclic lock in the 30-cell × 3-precision validation;
  no other cell in the corpus exhibits the same pattern.

## DSP sampling angle

The L19 entropy growth signal IS a sampling-theoretic statistic about the
attention DISTRIBUTION's spread. Applying replica exchange / particle
filter / importance sampling on top of this signal:

- **Trigger-driven rescue**: when growth_02 < 0.55 at 40% budget, IMMEDIATELY
  truncate cache to chunk_2 and invoke inferguard/aime_rescue (forced-commit
  at most-recent truth-shape position). Saves the remaining 60% of compute
  budget that would otherwise burn on the lock continuation.
- **Replica exchange decoding**: spawn N continuation replicas at temperatures
  T = {0.0, 0.5, 1.0}; periodically measure L19 growth in each; swap
  low-growth replicas with high-T replicas to escape commitment-basin.

The trigger-driven rescue is the IMMEDIATELY DEPLOYABLE move; replica
exchange decoding requires N× compute and a multi-temperature decoding
infrastructure.

## Files

- `codec_audit/attention_entropy_probe.py` — class-patched manual attention
  for Qwen3NextAttention, captures per-(head, position) entropy + self-mass
- `codec_audit/attention_cross_model.py` — same probe for qwen2 architecture
  (DS-R1-7B); promotes Q/K to fp32 before matmul (avoids fp16 overflow on
  no-norm models with dot products reaching 3000+)
- `codec_audit/attention_late_region.py` — chunk-wise analysis tool
- `codec_audit/attention_cross_cell.py` — cross-cell aggregation with
  CATEGORIES (lock, rescue, native, silent_fail, confident_wrong)
- `codec_audit/cross_arch_chunk_analysis.py` — Qwen vs DS-R1-7B comparison
- `codec_audit/l19_growth_detector.py` — deployable detector module
- `tmp/20260525_attention_inflight/p{01..10}_{4bit,8bit,bf16}.json` —
  per-position attention statistics (where available)

## Cross-architecture substrate truth

The LAST-LAYER COPY signature (~18% high self-mass at the last attention
layer) replicates from Qwen3.5-4B's L31 (hybrid arch) to DS-R1-7B's L27
(all-attention arch). This is structural — the last attention layer in
both architectures fires copy/induction patterns.

The L19 ENTROPY-GROWTH SUPPRESSION signal that catches lock/confident-wrong
in Qwen3.5-4B 4bit is potentially architecture-specific; the equivalent
layer in DS-R1-7B (likely L20-L24) was NOT tested with DS-R1-7B's own
generation. This is the next test: generate DS-R1-7B on a problem where
it fails to commit, measure entropy growth at its candidate analog layer,
see if the signal replicates.
