# Runtime divergence — MLX vs HF transformers at multi-token continuation

silv 2026-05-25 deepest disambiguation of cross-precision rescue finding.

## What the matrix shows

Forced-commit + greedy K=8 on identical cached responses (frac=0.60,
n_chars_cap=16000):

| Runtime | Precision | Result |
|---------|-----------|--------|
| MLX     | 4bit      | 7/10   |
| MLX     | 8bit      | 7/10 (identical commits to 4bit) |
| MLX     | BF16      | 7/10 (identical commits to 4bit) |
| HF transformers | BF16 (AMD ROCm) | 5/10 |

## Single-token logit comparison at P02 forced-commit position

The position is right after `\boxed{` in `(prefix + preamble)`. Truth=62.

4bit-MLX top-6:
- rank 1: '6' P=0.9959 ✓
- rank 2: '5' P=0.0015

BF16-MLX top-6:
- rank 1: '6' P=0.9981 ✓
- rank 2: '2' P=0.0006

BOTH MLX precisions pick '6' (correct first digit of truth=62) at P > 0.99.

The HF BF16 result was `emit='60'` — also picks '6' first. So **at the
FIRST forced-commit position, ALL four runtimes agree on '6'**.

## The divergence is in the SECOND digit

After committing '6':
- MLX continues to '2' → emits "62" (truth ✓)
- HF continues to '0' → emits "60" (wrong, off by 2)

The KV-cache extension after the first emitted token diverges between
MLX and HF.

## Likely mechanisms

1. **KV-cache precision difference**: MLX stores cache in the model's
   natural precision (4bit-quantized or bf16); HF may upcast to fp32
   or apply different storage layout.
2. **Different rotary positional encoding update** when extending the
   cache by one token.
3. **Position index handling**: when the cache shifts from prompt-fill
   to autoregressive-extension, position offset differs subtly.
4. **Numerical stability in attention softmax**: HF uses PyTorch's
   `nn.functional.scaled_dot_product_attention` which may apply
   different bf16/fp32 promotion rules than MLX's native SDPA.

## Refined doctrine

The earlier "MLX retrieves context better than HF" was approximately
correct but mechanistically MISFRAMED. The actual finding:

**Initial commit (first token after `\boxed{`) is IDENTICAL across all
tested runtimes and precisions on the test cells. The divergence is in
multi-token autoregressive continuation.**

So:
- The substrate truth-rank measurement: invariant across precision and
  runtime
- The single-token rescue: invariant
- The multi-token rescue (extracting the full multi-digit answer):
  diverges between MLX and HF at the 2nd+ token

Implication for production: if you use HF for rescue, expect 5/10 vs
MLX's 7/10. The difference is real but is at the autoregressive layer,
not the substrate layer.

## What this teaches about silv's sampling-instrumentation fallacy

Even at SINGLE-TOKEN level, the substrate is more invariant than the
multi-token continuation. The "instrumentation" (the autoregressive
loop, KV-cache, position handling) is where library implementations
diverge. The pure substrate read is consistent.

The fallacy isn't only "sampling can't fix calibration" — it's also
"the inference STACK (cache management, RoPE application, attention
backend) has more degrees of freedom than the substrate computation
itself." Production stacks differ in places that aren't visible at
the API layer.

## Files

- `tmp/20260525_attention_inflight/{forced_extract_4bit_0_60,
  forced_extract_8bit_0_60, forced_extract_bf16mlx_0_60,
  amd_forced_extract_bf16_results}.json` (all four runtimes)
- `tmp/20260525_attention_inflight/runtime_divergence.md` (this)

## Untested

- Where EXACTLY does HF diverge from MLX in the autoregressive loop?
  Could be:
  a) Apply RoPE to new key  
  b) Concat new key to cache
  c) Softmax over extended attention
  d) Sample/argmax from new logits
- Forcing HF to use eager attention vs sdpa may close the gap
- Forcing HF to use float32 KV cache may close the gap
- These are codex/engineering-level investigations
