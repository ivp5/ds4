# Session 2026-05-25 — in-flight attention + substrate truth-install

silv directives (latest in chronological order):
1. "explore internal LLM training/inference states... in-flight representations
   of attention matrices... 10000 proper cycles"
2. "ssh to amd with user gpu_ssh"
3. "reread your own shifts.md as well as CODEX_SHIFTS.md - you have already
   explored this direction, check which experiments have you not yet run...
   poke at the unknown unknown with maximal informational benefit"
4. "read sequentially, without skipping or jumping ahead"
5. "doubt EVERYTHING, doubt EVERY STEP. make more glorious mistakes"
6. "do UNSAFE takes, MAKE MISTAKES! crash into the wall if you have to"

## What I built

| Probe | What it measures | Status |
|-------|-----------------|--------|
| `attention_entropy_probe.py` | Per-(layer, head, position) attention entropy on Qwen3.5-4B-MLX via class-patched manual SDPA | Working |
| `attention_cross_model.py` | Same for qwen2 (DS-R1-7B) with fp32 matmul (no q_norm/k_norm) | Working |
| `attention_late_region.py` | Chunk-wise aggregation of saved attention data | Working |
| `attention_cross_cell.py` | Cross-cell aggregation with categories | Working |
| `cross_arch_chunk_analysis.py` | Qwen3.5-4B vs DS-R1-7B comparison | Working |
| `l19_growth_detector.py` | Deployable L19 entropy-growth commit-concentration detector | T=0.55, 100% recall on lock cells |
| `replica_exchange_decode.py` | DSP parallel-tempering with N replicas at different temperatures | REFUTED on P01 lock |
| `topk_retention_decode.py` | Top-k logit retention + back-decode at non-greedy positions | REFUTED: truth not in retained top_k |
| `perlayer_logit_lens.py` | All-layer logit-lens at last N positions | Initially MISLEADING (averaged over positions) |
| `logitlens_at_position.py` | All-layer logit-lens at fractional positions | Showed P01 substrate doesn't have truth |
| `logitlens_at_boxed.py` | All-layer logit-lens at EXACT `\boxed{` commit position | Revealed L23-L31 commit-tier install |
| `amd_perlayer_lens_bf16.py` (AMD) | BF16 reference via HF `output_hidden_states=True` | Cross-precision corroboration |

## Findings (rank-ordered by load-bearing)

### F1. Substrate truth installs at L23-L31 commit tier (cross-precision)
P05 native success (truth=65, d1='6') at the position right after `\boxed{`:
- L19: P('6') = 3.4e-5 (4bit) / 5e-6 (BF16) — essentially zero
- L23 (aligned): P('6') = 0.006 (4bit) / 0.012 (BF16) — first top-1
- L29 (aligned): P('6') = 0.554 (4bit) / 0.614 (BF16) — crosses 0.5
- L31 final: P('6') = 0.9998 (4bit) / 1.0000 (BF16)

**The commit tier L23-L31 is where the answer enters the substrate. The
L19 norm-explosion full-attention layer is BEFORE truth-encoding.**

### F2. L19 attention-entropy growth detects commit-concentration
At first 4000 chars of AIME 2026 4bit cells, growth from chunk_0 to
chunk_2 is suppressed (< 0.55) in commit-attempt cells, normal (> 0.65)
in still-exploring cells.

Original (mis-categorized): TP=2 (P01 lock + P10 confident-wrong), FP=0,
TN=8 (10-cell × 4bit).

Correct categorization (after audit): 4bit cells with native `\boxed{}`
commits in first 4000 chars are ONLY P01 lock + P10 lock (both confident-
cycling). The 8 cells without commits at first 4000 chars don't fire.
Recall on commit-attempts = 2/2.

Cross-precision: 4bit growth 0.521, bf16 0.581, 8bit 0.938 on P01 prompt.
Lock signal degrades smoothly with precision; non-locking precisions
don't fire.

### F3. Lock substrate has no truth (refutes my own earlier claim)
P01 lock at 6 sampled positions × 32 layers: max P(truth_digit '2')
= 0.000029. Truth=277 was NEVER computed in P01's substrate at any
layer at any tested position. My earlier "P=0.9998 at L31" was a
position-averaging artifact (positions where literal '2' character is
the natural next token in the locked text).

### F4. Sampling-instrumentation REFUTED for lock escape
- **Replica exchange** (T=0.0, 0.3, 0.7, 1.0, 1.4 from chunk_2 cache):
  0/5 reached truth on P01.
- **Top-k retention** (k=16 across 1839 positions): no `277` substring
  in any retained token; truth never in top_k at any position.
- **Forced-commit rescue** (the only protocol that works): DOES work on
  cells where TRUTH-SHAPE appears earlier in the CoT text; FAILS on P01
  because P01 substrate genuinely never had truth.

The deeper fallacy silv named: sampling/temperature is treated as
post-hoc inference-time correction. It cannot manifest what the model
never computed. The lock is a substrate-computation failure, not a
decoding/sampling failure.

### F5. AMD heavy sweep cross-precision SUBSTRATE invariance
BF16 truth-install trajectory matches 4bit MLX within layer-alignment
shift. Quantization doesn't shift WHEN truth installs. The 4bit accuracy
gap (where it exists) is at PRE-COMMIT feature processing, not at
commit-tier install.

## Refutations of session-prior claims

- "Lock substrate preserves truth at L31" — REFUTED at position level
- "Replica exchange escapes lock via high-T" — REFUTED (0/5)
- "Top-k retention catches discarded truth" — REFUTED (truth never present)
- My cell categorization (P02-P09 as "rescue") — REFUTED (those are
  native explore-no-commit cells; rescue is an OFFLINE post-hoc protocol)
- "P10 confident-wrong-100" — REFUTED in native 4bit (P10 is a LOCK
  cell with 1801-line literal repetition; the "100" came from offline
  forced-commit rescue, not native generation)

## What survived audit

- L19 attention-entropy growth as a commit-concentration signal
- Substrate truth-install at L23-L31 commit tier
- Cross-precision substrate invariance (4bit ≈ BF16 at commit-tier)
- Forced-commit rescue protocol (where truth is in CoT text, not just
  substrate)
- HC_LOCK n-gram persistence detector (separate signal, also catches lock)

## Deeper lesson (sampling-as-instrumentation fallacy)

The default LLM inference stack treats logits as intermediate and
sampling (temperature, top_p, top_k) as the "presentation" layer.
silv's deeper directive named this fallacy: keep the model's actual
distribution, not the sampling wrapper.

What I found by following the fallacy:
- For LOCK cells, sampling-instrumentation cannot help. The lock isn't
  exploration-deficit; it's substrate-computation-deficit. No sampling
  manipulation manifests truth that was never computed.
- For COMMIT-tier inspection, the L31 logits ARE the model's certainty
  (P(truth)>0.999 at success). The "sampling" wrapper is irrelevant when
  the substrate is confident.
- The PROBE that matters is L29 (where P crosses 0.5) — that's the
  layer where the commit could go either way. Reading L29 confidence
  via logit-lens is a CHEAPER (single-position) confidence signal than
  multi-sample averaging.

## Infra notes

- AMD RX 7900 XTX 25.6 GB free: ROCm/HIP backend with transformers
  5.8.1; HuggingFace `output_hidden_states=True` works natively (no
  class-patching needed)
- AMD BF16 14280-token forward with all-layer capture: ~5s wall
- NVIDIA 4090 had only 2.6 GB free (silv's VLLM EngineCore running);
  cannot co-tenant; heavy work went to AMD instead
- MLX 4bit on M1 Max: per-cell attention probe ~3.5s wall for 1800 tokens

## Open directions (next sessions)

1. **L29 logit-lens confidence as deployable**: at the position right
   before generating `\boxed{N}`, read L29 hidden state through
   final_norm + lm_head; if P(top_1) > X, commit is confident; if
   not, flag for additional compute. Untested.
2. **Conjecture #22 b·K isoquant** (pending from 2026-05-18): K=4
   temp=0.7 on DS-R1-Distill-Qwen-7B AIME 2026 — affordable on AMD now.
   Would test majority-vote lift via bit-budget × oversampling.
3. **Cross-architecture L19 analog**: DS-R1-7B's analog of L19 (the
   norm-explosion full-attention layer) wasn't identified. Need
   per-layer attention pattern probe on DS-R1-7B own-generated
   responses.
4. **L19 detector inflight**: incremental streaming form during
   generation; truncate-and-rescue when growth signal fires below
   threshold. Untested in production-pipeline form.

## File index

- `synthesis.md` (initial L19 finding)
- `findings_severe_audit.md` (refutations of session claims)
- `cross_precision_amd.md` (AMD BF16 vs MLX 4bit substrate)
- `SESSION_SYNTHESIS.md` (this file)
- `replica_p01.json` (5 replicas, 0/5 truth recovery)
- `topk_p01.json` (16-deep top_k, 0 truth occurrences)
- `lens_at_pos_p01.json`, `lens_at_pos_p02.json`, `lens_at_pos_p05.json`
- `lens_at_boxed_p05.json` (the substrate truth-install trajectory)
- `lens_at_boxed_p10_d1_1.json` (P10 also has no commits)
- `p05_amd_bf16_result.json` (AMD reference, n_layers 32, hidden_states 33)
- `p{01..10}_4bit.json` (per-position attention entropy/self-mass)
- `p01_{4bit,8bit,bf16}.json` (cross-precision L19 data)
- `dsr1_7b_4bit_qwen_p01.json` (cross-arch attention reading lock text)
