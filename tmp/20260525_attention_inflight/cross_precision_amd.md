# Cross-precision substrate sweep — AMD BF16 vs M1 MLX-4bit

silv 2026-05-25: offload heavy sweeps to AMD. AMD has Qwen/Qwen3.5-4B
BF16 cached locally; RX 7900 XTX 25.6 GB free.

## Result — P05 (AIME 2026, native success, truth=65, d1='6')

Per-layer P('6') at the position right after `\boxed{` (the commit point):

| Layer | 4bit MLX | BF16 AMD (offset −1 to align) |
|-------|----------|-------------------------------|
| L18   | 0.000010 (rank 20003) | 0.000001 (rank 168676) |
| L19   | 0.000034 (rank 3768)  | 0.000005 (rank 41585)  |
| L20   |                       | 0.000052 (rank 1750)   |
| L21   |                       | 0.000184 (rank 167)    |
| L22   |                       | 0.000169 (rank 213)    |
| L23   | **0.006330 (rank 0)** | 0.000033 (rank 3709)   |
| L24   |                       | **0.011912 (rank 0)**  |
| L27   | 0.019311 (rank 0)     | 0.008977 (rank 0)      |
| L28   |                       | 0.032238 (rank 0)      |
| L29   | 0.554226 (rank 0)     | 0.043836 (rank 1)      |
| L30   | **0.998928 (rank 0)** | 0.614205 (rank 0)      |
| L31   | 0.999785 (rank 0)     | **0.998689 (rank 0)**  |
| L32   | —                     | 0.999982 (rank 0)      |

Note: BF16 hidden_states includes the embedding output as L0, so BF16 L_k
corresponds to MLX L_{k-1}. After alignment (BF16 layer offset by −1):

| Aligned layer | 4bit P | BF16 P (BF16_idx − 1) |
|----------|--------|-------------|
| 19 | 0.000034 | 0.000005 (BF16_L20) |
| 23 | 0.006330 | 0.011912 (BF16_L24) — TRUTH FIRST AT TOP-1 |
| 27 | 0.019311 | 0.032238 (BF16_L28) |
| 29 | 0.554226 | 0.614205 (BF16_L30) — TRUTH P > 0.5 |
| 30 | 0.998928 | 0.998689 (BF16_L31) — TRUTH P > 0.99 |
| 31 (final) | 0.999785 | 0.999982 (BF16_L32) — FINAL |

## Substrate finding

**Truth-install pattern is CROSS-PRECISION INVARIANT** on this cell:
- Truth first becomes top-1 at aligned layer 23-24 (4bit & BF16 within 1 layer)
- Truth crosses P=0.5 at aligned layer 29-30 (both precisions identical)
- Final layer P ≈ 0.999 (both precisions)

**Quantization does NOT shift WHEN truth installs.** The substrate
function over depth is invariant to precision (at least for this cell).
This is consistent with the matharena leaderboard observation that 4bit
Qwen3.5-4B matches BF16 on most AIME cells — at the commit-position
substrate level, the answer is computed at the same depth.

The 4bit BF16 final-layer P('6'): 0.9998 vs 1.000 (0.0001% relative
loss). Negligible at this cell.

## What this corroborates

Earlier this session's finding: truth installs at L23-L31 commit tier,
NOT at L19. BF16 confirms: at L19 P('6')=5e-6 (BF16) vs 3.4e-5 (4bit) —
both essentially zero. The L19 attention-entropy signal IS NOT about
truth-encoding; it's about commit-concentration.

## What this refutes (or sharpens)

My earlier "the lock destroys truth at late layers" interpretation was
based on Ministral P04 (different model). On Qwen3.5-4B P05 (success
cell), there's no peak-then-decay: P('6') monotonically rises from
L23 onward. So Ministral's peak-then-decay is model-specific, not a
universal substrate property.

The 4bit-vs-BF16 substrate equivalence at commit position implies:
- 4bit accuracy gap on AIME (where it exists) comes from earlier
  (pre-commit) feature processing, NOT from the commit tier
- Quantization-as-saving-truth in the commit tier was never the bottleneck

## Files

- AMD code: `tmp/20260525_attention_inflight/amd_perlayer_lens_bf16.py`
- AMD result: `tmp/20260525_attention_inflight/p05_amd_bf16_result.json`
- M1 4bit result: `tmp/20260525_attention_inflight/lens_at_boxed_p05.json`

## Hardware/infra notes

- AMD RX 7900 XTX: 25.6 GB free, transformers 5.8.1, ROCm/HIP backend
- BF16 load: 426 weight files in ~7 seconds (vs M1 MLX 1-2s)
- Per-cell forward at 14280 tokens with output_hidden_states=True: ~5s
  total
- HuggingFace `output_hidden_states=True` is a built-in feature; no
  class-patching needed (unlike MLX where I had to wrap DecoderLayer
  __call__)
- UTF-8 encoding issue: Windows default cp1252 can't print CJK tokens;
  fixed via PYTHONIOENCODING=utf-8

## Next AMD sweep (queued)

Same probe on cells P02-P04, P06-P10 where applicable (note: only P05
has `\boxed{}` in the 4bit cache; for other cells, the native generation
didn't commit). To get per-layer at commit-position for those cells,
need BF16-native generations on AMD (which may NOT lock the same way
that 4bit does, since lock is precision-specific).

For cross-precision lock detection: generate BF16 P01 on AMD, check
whether it locks. CLAUDE.md says bf16 P01 most_pers=0.294 (intermediate;
doesn't fully lock). The behavior could be tested by running BF16 P01
through AMD and checking response.
