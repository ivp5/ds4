"""Attention-entropy probe — in-flight attention pattern analysis.

silv 2026-05-25: "explore internal LLM training/inference states... in-flight
representations of attention matrices... 10000 proper cycles."

Captures per-(layer, head, query_position) ATTENTION ENTROPY across the 8
full-attention layers (L3/7/11/15/19/23/27/31) of Qwen3.5-4B-MLX-4bit.

Method:
  1. Class-patch Qwen3NextAttention.__call__ to manually compute attention
     weights via Q @ K^T / sqrt(d) → softmax (replacing the fused SDPA so
     weights are accessible).
  2. Capture per-(layer, head, query_pos) attention distribution.
  3. Compute: entropy, self-attention mass, top-1-attended position,
     short-range vs long-range mass split.
  4. Compare across prompts (locked vs success).

Per substrate vertical map:
  L19 = norm-explosion layer (full-attn RoPE; norm 15→16 jumping into late tier).
  L18 = commit-installation (largest cross-prompt cosine drop, GatedDeltaNet).
  Hypothesis: at L19 during locked positions, attention concentrates on the
  PRIOR OCCURRENCE of the locked pattern (induction-head signature). At
  success positions, attention is more distributed.

Compute cost: manual attention = O(L²) per layer per query. For 2000 tokens
and 8 full-attn layers: ~32M attention ops per cell. ~30-60s wall.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
proxy_port = random.choice(range(10001, 10150))
os.environ.setdefault("HTTPS_PROXY", f"socks5h://127.0.0.1:{proxy_port}")
os.environ.setdefault("HTTP_PROXY", f"socks5h://127.0.0.1:{proxy_port}")
os.environ.setdefault("ALL_PROXY", f"socks5h://127.0.0.1:{proxy_port}")

import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from codec_audit.db import open_db, journal

MODEL_ID = "mlx-community/Qwen3.5-4B-MLX-4bit"

# Full-attention layer indices (0-indexed) per Qwen3.5-4B architecture
FULL_ATTN_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]


# Storage for captured attention statistics per layer
# captured[layer_idx] = {"entropy": (n_pos, n_heads), "self_mass": (n_pos, n_heads),
#                        "topk_dist": (n_pos, n_heads, K), ...}
_captured = {}


def make_patched_attention_call(layer_idx, n_positions_target):
    """Create a class-patch for Qwen3NextAttention.__call__ that
    manually computes attention + captures per-head entropy."""

    def patched(self, x, mask=None, cache=None):
        B, L, D = x.shape
        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, L, -1)
        keys, values = self.k_proj(x), self.v_proj(x)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)  # (B, H_q, L, head_dim)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(
            0, 2, 1, 3)  # (B, H_kv, L, head_dim)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3)  # (B, H_kv, L, head_dim)
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # GQA: expand keys/values to match query heads via repeat
        n_q = queries.shape[1]
        n_kv = keys.shape[1]
        repeat = n_q // n_kv
        if repeat > 1:
            keys_full = mx.repeat(keys, repeat, axis=1)
            values_full = mx.repeat(values, repeat, axis=1)
        else:
            keys_full = keys
            values_full = values

        # Manual attention: (B, H, L_q, head_dim) @ (B, H, head_dim, L_k)
        # = (B, H, L_q, L_k)
        scale = self.scale
        attn_logits = mx.matmul(queries, keys_full.transpose(0, 1, 3, 2)) * scale
        # Causal mask: L_q × L_k
        L_q = attn_logits.shape[-2]
        L_k = attn_logits.shape[-1]
        if L_q > 1:
            # Causal: q at pos i can attend to k at pos j ≤ i
            row_idx = mx.arange(L_q)[:, None]
            col_idx = mx.arange(L_k)[None, :]
            mask_arr = (col_idx > row_idx + (L_k - L_q)).astype(attn_logits.dtype) * (-1e9)
            attn_logits = attn_logits + mask_arr
        attn_weights = mx.softmax(attn_logits.astype(mx.float32), axis=-1)
        # attn_weights: (B, H, L_q, L_k)

        # Per-(head, query) entropy: -sum p log p
        # safe_p where p > 1e-30; entropy in nats
        log_p = mx.log(mx.maximum(attn_weights, 1e-30))
        entropy = -mx.sum(attn_weights * log_p, axis=-1)  # (B, H, L_q)
        # Self-attention mass: weight on the SAME position (diagonal)
        # For prefill, query i attends to key positions 0..L-1 (causal),
        # self is the diagonal element
        if L_q == L_k:
            diag_idx = mx.arange(L_q)
            self_mass = attn_weights[:, :, diag_idx, diag_idx]  # (B, H, L_q)
        else:
            self_mass = mx.zeros((B, attn_weights.shape[1], L_q))

        # Top-1 attended position (for the LAST quarter of context, where induction
        # heads would fire most strongly)
        top1_pos = mx.argmax(attn_weights, axis=-1)  # (B, H, L_q)

        # Store as numpy for later analysis
        mx.eval(entropy, self_mass, top1_pos)
        _captured[layer_idx] = {
            "entropy": np.array(entropy[0], copy=True),         # (H, L_q)
            "self_mass": np.array(self_mass[0], copy=True),     # (H, L_q)
            "top1_pos": np.array(top1_pos[0], copy=True),       # (H, L_q)
        }

        # Continue with the rest of the attention computation
        output = mx.matmul(attn_weights.astype(values_full.dtype), values_full)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output * mx.sigmoid(gate))

    return patched


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text-from", required=True)
    p.add_argument("--field", default="response")
    p.add_argument("--n-chars", type=int, default=4000)
    p.add_argument("--label", default="")
    p.add_argument("--out", type=str)
    args = p.parse_args()

    text = json.loads(Path(args.text_from).read_text())[args.field][:args.n_chars]

    from mlx_lm import load
    print(f"[{time.strftime('%H:%M:%S')}] loading {MODEL_ID}")
    model, tok = load(MODEL_ID)
    internal = model.language_model.model
    layers = internal.layers
    n_layers = len(layers)

    # Class-patch the full-attention layers' attention modules
    AttnCls = None
    for L_idx in FULL_ATTN_LAYERS:
        layer = layers[L_idx]
        if AttnCls is None:
            AttnCls = type(layer.self_attn)
            orig_call = AttnCls.__call__

    # Need to dispatch to the right captured slot per layer
    # Use id-based registry
    attn_layer_idx = {id(layers[L].self_attn): L for L in FULL_ATTN_LAYERS}

    def dispatched_call(self, x, mask=None, cache=None):
        layer_idx = attn_layer_idx.get(id(self))
        if layer_idx is None:
            return orig_call(self, x, mask, cache)
        return make_patched_attention_call(layer_idx, x.shape[1])(self, x, mask, cache)

    AttnCls.__call__ = dispatched_call

    ids = tok.encode(text)
    n_tok = len(ids)
    print(f"  tokens: {n_tok}")

    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    t0 = time.time()
    _ = model(mx.array([ids]), cache=cache)
    mx.eval(*[c.state for c in cache if hasattr(c, "state")])
    print(f"  prefill+attention capture: {time.time()-t0:.1f}s")

    # Restore
    AttnCls.__call__ = orig_call

    # Aggregate per layer
    print(f"\n=== {args.label or Path(args.text_from).stem} ===")
    print(f"{'L':>3}  {'mean_ent':>8} {'min_ent':>8} {'max_ent':>8} "
          f"{'mean_self':>9} {'high_self_frac':>14}")
    aggregate = []
    for L_idx in FULL_ATTN_LAYERS:
        if L_idx not in _captured:
            continue
        cap = _captured[L_idx]
        ent = cap["entropy"]  # (H, L_q)
        self_mass = cap["self_mass"]  # (H, L_q)
        # Aggregate over (H, L_q): mean entropy across all heads and positions
        mean_ent = float(ent.mean())
        min_ent = float(ent.min())
        max_ent = float(ent.max())
        mean_self = float(self_mass.mean())
        # Fraction of (head, position) cells where self-attention mass > 0.5
        high_self_frac = float((self_mass > 0.5).mean())
        aggregate.append({
            "layer": L_idx,
            "mean_entropy": mean_ent,
            "min_entropy": min_ent,
            "max_entropy": max_ent,
            "mean_self_mass": mean_self,
            "high_self_frac": high_self_frac,
        })
        print(f"  {L_idx:>2}  {mean_ent:>8.4f} {min_ent:>8.4f} {max_ent:>8.4f} "
              f"{mean_self:>9.4f} {100*high_self_frac:>13.1f}%")

    if args.out:
        # Save per-position arrays for deeper analysis
        out_data = {
            "label": args.label,
            "source": str(args.text_from),
            "n_chars": args.n_chars,
            "n_tokens": n_tok,
            "full_attn_layers": FULL_ATTN_LAYERS,
            "aggregate": aggregate,
            "per_layer": {},
        }
        for L_idx, cap in _captured.items():
            out_data["per_layer"][str(L_idx)] = {
                "mean_entropy_per_pos": cap["entropy"].mean(axis=0).tolist(),
                "mean_self_mass_per_pos": cap["self_mass"].mean(axis=0).tolist(),
            }
        Path(args.out).write_text(json.dumps(out_data, indent=2))
        print(f"\nsaved {args.out}")

    with open_db() as conn:
        journal(conn, "attention_entropy_probe", str(args.text_from),
                {"label": args.label, "n_tokens": n_tok, "aggregate": aggregate})


if __name__ == "__main__":
    main()
