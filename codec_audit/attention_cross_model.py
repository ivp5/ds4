"""Cross-model attention entropy: Qwen3.5-4B vs DS-R1-Distill-Qwen-7B.

silv 2026-05-25: "have you tested on sufficient different models to aggregate
confidence?" The L19 attention-entropy lock signature has been measured on
Qwen3.5-4B only. To claim cross-architecture confidence, the same probe
must run on a different architecture.

Qwen3.5-4B: qwen3_next, 32 layers, hybrid (8 full-attn + 24 GatedDeltaNet),
            16 heads per attention, partial RoPE (0.25).
DS-R1-Distill-7B: qwen2, 28 layers, all-attention, 28 heads, full RoPE.

This script:
  1. Generates a P01 CoT response on DS-R1-7B-4bit (~3000 tokens greedy).
  2. Runs an analogous attention-entropy probe on every attention layer
     (all 28, since DS-R1-7B is all-attention).
  3. Bins into 5 chunks; compares chunk-2 entropy across cells.

If a similar entropy-drop pattern appears at some characteristic mid-layer
in DS-R1-7B during a hard-to-converge cell, the substrate finding holds
cross-architecture. If not, the finding is Qwen3.5-4B-specific.
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from codec_audit.db import open_db, journal


_captured = {}


def make_patched_attention_call(layer_idx):
    """Patched call works for both qwen3 and qwen2 attention shapes."""

    def patched(self, x, mask=None, cache=None):
        B, L, D = x.shape
        # Qwen2 style: q_proj, k_proj, v_proj all separate, no gate split
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        # head_dim from self attributes
        n_heads = self.n_heads if hasattr(self, "n_heads") else self.num_attention_heads
        n_kv_heads = self.n_kv_heads if hasattr(self, "n_kv_heads") else self.num_key_value_heads
        head_dim = queries.shape[-1] // n_heads
        queries = queries.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # GQA expand
        repeat = n_heads // n_kv_heads
        if repeat > 1:
            keys = mx.repeat(keys, repeat, axis=1)
            values = mx.repeat(values, repeat, axis=1)

        # Promote Q/K to fp32 before matmul (DS-R1-7B has no q_norm/k_norm,
        # so dot products can hit ~3000+ and overflow fp16).
        scale = self.scale
        q32 = queries.astype(mx.float32)
        k32 = keys.astype(mx.float32)
        attn_logits = mx.matmul(q32, k32.transpose(0, 1, 3, 2)) * scale
        L_q = attn_logits.shape[-2]
        L_k = attn_logits.shape[-1]
        if L_q > 1:
            row_idx = mx.arange(L_q)[:, None]
            col_idx = mx.arange(L_k)[None, :]
            causal_mask = (col_idx > row_idx + (L_k - L_q))
            attn_logits = mx.where(
                causal_mask,
                mx.array(-1e30, dtype=mx.float32),
                attn_logits,
            )
        attn_weights = mx.softmax(attn_logits, axis=-1)

        log_p = mx.log(mx.maximum(attn_weights, 1e-30))
        entropy = -mx.sum(attn_weights * log_p, axis=-1)
        if L_q == L_k:
            diag_idx = mx.arange(L_q)
            self_mass = attn_weights[:, :, diag_idx, diag_idx]
        else:
            self_mass = mx.zeros((B, attn_weights.shape[1], L_q))

        mx.eval(entropy, self_mass)
        _captured[layer_idx] = {
            "entropy": np.array(entropy[0], copy=True),
            "self_mass": np.array(self_mass[0], copy=True),
        }
        output = mx.matmul(attn_weights.astype(values.dtype), values)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

    return patched


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit")
    p.add_argument("--text-from", required=True)
    p.add_argument("--field", default="response")
    p.add_argument("--n-chars", type=int, default=4000)
    p.add_argument("--label", default="")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    text = json.loads(Path(args.text_from).read_text())[args.field][:args.n_chars]

    from mlx_lm import load
    print(f"[{time.strftime('%H:%M:%S')}] loading {args.model}")
    model, tok = load(args.model)
    internal = model.model
    layers = internal.layers
    n_layers = len(layers)
    print(f"  n_layers: {n_layers}")
    print(f"  attn class: {type(layers[0].self_attn).__name__}")

    # Patch all attention layers
    AttnCls = type(layers[0].self_attn)
    orig_call = AttnCls.__call__
    attn_layer_idx = {id(layers[L].self_attn): L for L in range(n_layers)}

    def dispatched_call(self, x, mask=None, cache=None):
        L_idx = attn_layer_idx.get(id(self))
        if L_idx is None:
            return orig_call(self, x, mask, cache)
        return make_patched_attention_call(L_idx)(self, x, mask, cache)

    AttnCls.__call__ = dispatched_call

    ids = tok.encode(text)
    n_tok = len(ids)
    print(f"  tokens: {n_tok}")

    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    t0 = time.time()
    _ = model(mx.array([ids]), cache=cache)
    mx.eval(*[c.state for c in cache if hasattr(c, "state")])
    print(f"  prefill+capture: {time.time()-t0:.1f}s")

    AttnCls.__call__ = orig_call

    # Aggregate
    aggregate = []
    print(f"\n{'L':>3}  {'mean_ent':>8} {'min_ent':>8} {'mean_self':>9} {'high_self%':>10}")
    for L in range(n_layers):
        if L not in _captured:
            continue
        cap = _captured[L]
        ent = cap["entropy"]
        sm = cap["self_mass"]
        aggregate.append({
            "layer": L,
            "mean_entropy": float(ent.mean()),
            "min_entropy": float(ent.min()),
            "max_entropy": float(ent.max()),
            "mean_self_mass": float(sm.mean()),
            "high_self_frac": float((sm > 0.5).mean()),
        })
        # Print every 4th
        if L % 4 == 0 or L == n_layers - 1:
            print(f"  {L:>2}  {ent.mean():>8.4f} {ent.min():>8.4f} "
                  f"{sm.mean():>9.4f} {100*(sm > 0.5).mean():>9.1f}%")

    # Per-position per-layer for offline analysis
    per_layer = {}
    for L, cap in _captured.items():
        per_layer[str(L)] = {
            "mean_entropy_per_pos": cap["entropy"].mean(axis=0).tolist(),
            "mean_self_mass_per_pos": cap["self_mass"].mean(axis=0).tolist(),
        }
    out_data = {
        "label": args.label,
        "model": args.model,
        "source": str(args.text_from),
        "n_chars": args.n_chars,
        "n_tokens": n_tok,
        "n_layers": n_layers,
        "aggregate": aggregate,
        "per_layer": per_layer,
    }
    Path(args.out).write_text(json.dumps(out_data, indent=2))
    print(f"\nsaved {args.out}")

    with open_db() as conn:
        journal(conn, "attention_entropy_cross_model", str(args.text_from),
                {"label": args.label, "model": args.model,
                 "n_tokens": n_tok, "n_layers": n_layers})


if __name__ == "__main__":
    main()
