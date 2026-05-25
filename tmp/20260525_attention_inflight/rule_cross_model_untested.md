# Honest scope: cross-model generalization is UNTESTED

silv 2026-05-25 continue: searched for DS-R1-7B's own AIME 2026 CoT
cache. Doesn't exist. The 13-probe validated rule is corroborated
on Qwen3.5-4B but cross-model untested.

## What I have

- Qwen3.5-4B-MLX-4bit AIME 2026 P01-P10 CoTs: `tmp/20260524_quant_matrix/4bit/p*.json`
- Qwen3.5-4B substrate probes (per-layer activations): various
- DS-R1-7B substrate probes RUN ON Qwen's CoT prefixes: `dsr1_7b_4bit_qwen_p01.json`
  (cross-model substrate at Qwen-CoT-prefix; NOT DS-R1-7B's own CoT)

The data I would need: DS-R1-7B generates AIME 2026 P01-P10 with full
response saved, like the `tmp/20260524_quant_matrix/4bit/p*.json` files
but for DS-R1-7B. Doesn't exist in the filesystem.

## What this means for the rule

The 13-probe arc validated:
> rescuable(cell, cap, frac) ⇔ truth_at < min(frac × cot_len, cap)

on Qwen3.5-4B with 50/50 cell-config predictions perfect.

Cross-model generalization claim — that the same rule holds for
DS-R1-7B's own CoTs with DS-R1-7B's own truth_at values — is
UNTESTED with available data.

Refutation candidates that would need DS-R1-7B own CoTs:
- DS-R1-7B may have different truth_at distribution (per Conjecture
  #21 CLAUDE.md notes DeepSeek's confident-preamble extracts P1
  truth=588 only at K=8/no-commit cases — different mechanism)
- DS-R1-7B may have different loop-onset pattern (reasoning-RL
  distill, no-commit posture vs Qwen3.5-4B no-box-commit pathology)
- The rule's threshold dependence may differ across architectures

## Two possible interpretations

**Interpretation A (rule is universal)**: any model whose CoT contains
truth-shape at some position X can have truth extracted via forced-
commit at any cap > X. The rule is architecture-independent because
it operates on the deterministic prefix-content level.

**Interpretation B (rule is Qwen3.5-4B-specific)**: Qwen3.5-4B's
no-commit pathology + forced-commit preamble produces a specific
mechanism (latent-truth-extraction-from-prefix) that doesn't
generalize to models with different posture (e.g., DS-R1-7B's
no-commit may be RL-trained at a different boundary).

Without DS-R1-7B own CoT data, I cannot distinguish A from B.

## What survives unambiguously

The rule is corroborated as a Qwen3.5-4B AIME 2026 P01-P10-specific
finding with 5 cap-frac configurations tested. That scope is honest.
Claims of broader universality require new data.

## Closing honest limit-marker

The session arc's 13 probes converged on a rule. The rule's
DEMONSTRATED scope is:
- Model: Qwen3.5-4B-MLX-4bit (CoT origin)
- Rescue model: invariant across {4bit, 8bit, bf16} of same model
- Corpus: AIME 2026 P01-P10 (single model × 10 cells)
- Configurations: 5 (cap, frac) pairs

The CLAIMED universality (any model + any corpus + any cap-frac)
exceeds the demonstrated scope. Future sessions running DS-R1-7B's
own CoTs against the rule would be the right validation.

This is the 14th probe: an HONEST UNTESTED MARKER, not a refutation
nor a corroboration. The session arc reaches its data-scope limit.

## Files

- `tmp/20260525_attention_inflight/rule_cross_model_untested.md` (this)
- Existing data inventory: tmp/20260525_attention_inflight/*
- Missing: DS-R1-7B own AIME 2026 P01-P10 CoT cache
