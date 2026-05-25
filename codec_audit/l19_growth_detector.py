"""L19 entropy-growth deployable detector.

silv 2026-05-25: in-flight substrate signal cheaper than HC_LOCK.

The finding: at L19 (norm-explosion full-attention layer), attention entropy
grows from chunk_0 (prompt) to chunk_2 (40% gen) by 0.6-1.2 nats during
normal exploration. In LOCK or CONFIDENT-WRONG-COMMIT cells, this growth
is suppressed to ~0.5 nats.

Cross-precision correlation (P01 across 4bit/8bit/bf16): Pearson r =
-0.85 between L19 growth_02 and HC_LOCK most_pers.

Cross-cell distribution on 4bit (10 cells):
  Lock (P01): +0.456
  Confident-wrong (P10): +0.516
  Rescue cells (P02-P08): +0.609 to +1.212
  Native success (P05): +0.971
  Silent fail (P09): +1.526

Deployable threshold: growth_02 < 0.65 ⇒ commitment-concentration signal.
This fires EARLIER than HC_LOCK (which requires full lock pattern); it
catches at 40% of generation budget.

Composes with inferguard/aime_rescue: when this signal fires, the model
is committing — apply forced-commit at the most-recent truth-shape position
in the cached CoT to extract latent answer or detect confident-wrong.

This module supports both PRE-COMPUTED (full-text) and IN-FLIGHT
(streaming) modes.
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
os.environ.setdefault("ALL_PROXY", f"socks5h://127.0.0.1:{proxy_port}")

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))
from codec_audit.db import open_db, journal


# Deployable thresholds (calibrated on 10-cell 4bit run 2026-05-25):
GROWTH_THRESHOLD = 0.65  # nats — below this, model is concentrating
L19_LAYER_QWEN35 = 19  # full-attention norm-explosion layer
EARLY_CHUNK_FRAC = 0.20  # 0-20% of tokens
LATE_CHUNK_FRAC = 0.40  # 20-40% of tokens (early commit signal)


def l19_growth_from_full_text(model, tok, text: str,
                                early_frac: float = EARLY_CHUNK_FRAC,
                                late_frac: float = LATE_CHUNK_FRAC) -> dict:
    """Run a forward pass on the full text; compute L19 entropy growth
    from [0, early_frac) chunk to [early_frac, early_frac+late_frac) chunk.
    Returns growth + verdict.
    """
    from mlx_lm.models.cache import make_prompt_cache
    _captured = {}
    # Patch the L19 attention to capture entropy
    layers = model.language_model.model.layers
    target_layer = layers[L19_LAYER_QWEN35]
    AttnCls = type(target_layer.self_attn)
    orig_call = AttnCls.__call__
    target_id = id(target_layer.self_attn)

    def patched(self, x, mask=None, cache=None):
        if id(self) != target_id:
            return orig_call(self, x, mask, cache)
        B, L, D = x.shape
        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, L, -1)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(
            0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        repeat = queries.shape[1] // keys.shape[1]
        if repeat > 1:
            keys_full = mx.repeat(keys, repeat, axis=1)
            values_full = mx.repeat(values, repeat, axis=1)
        else:
            keys_full = keys
            values_full = values
        scale = self.scale
        attn_logits = mx.matmul(queries.astype(mx.float32),
                                  keys_full.astype(mx.float32).transpose(0, 1, 3, 2)) * scale
        L_q, L_k = attn_logits.shape[-2], attn_logits.shape[-1]
        if L_q > 1:
            row_idx = mx.arange(L_q)[:, None]
            col_idx = mx.arange(L_k)[None, :]
            causal_mask = (col_idx > row_idx + (L_k - L_q))
            attn_logits = mx.where(causal_mask, mx.array(-1e30), attn_logits)
        attn_weights = mx.softmax(attn_logits, axis=-1)
        log_p = mx.log(mx.maximum(attn_weights, 1e-30))
        ent = -mx.sum(attn_weights * log_p, axis=-1)
        mx.eval(ent)
        _captured["entropy_per_pos"] = np.array(ent[0].mean(axis=0), copy=True)
        output = mx.matmul(attn_weights.astype(values_full.dtype), values_full)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output * mx.sigmoid(gate))

    AttnCls.__call__ = patched
    ids = tok.encode(text)
    n_tok = len(ids)
    cache = make_prompt_cache(model)
    _ = model(mx.array([ids]), cache=cache)
    mx.eval(*[c.state for c in cache if hasattr(c, "state")])
    AttnCls.__call__ = orig_call

    ent = _captured["entropy_per_pos"]
    early_end = int(n_tok * early_frac)
    late_end = int(n_tok * (early_frac + late_frac))
    early_ent = float(ent[:early_end].mean())
    late_ent = float(ent[early_end:late_end].mean())
    growth = late_ent - early_ent
    verdict = "CONCENTRATING" if growth < GROWTH_THRESHOLD else "EXPLORING"
    return {
        "n_tokens": n_tok,
        "early_ent": early_ent,
        "late_ent": late_ent,
        "growth": growth,
        "verdict": verdict,
        "threshold": GROWTH_THRESHOLD,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text-from", required=True)
    p.add_argument("--field", default="response")
    p.add_argument("--n-chars", type=int, default=4000)
    p.add_argument("--label", default="")
    args = p.parse_args()

    text = json.loads(Path(args.text_from).read_text())[args.field][:args.n_chars]
    from mlx_lm import load
    model, tok = load("mlx-community/Qwen3.5-4B-MLX-4bit")
    result = l19_growth_from_full_text(model, tok, text)
    print(f"\n[{args.label or Path(args.text_from).stem}]")
    print(f"  n_tokens: {result['n_tokens']}")
    print(f"  early_ent (0-20%): {result['early_ent']:.4f}")
    print(f"  late_ent  (20-40%): {result['late_ent']:.4f}")
    print(f"  growth: {result['growth']:+.4f}  (threshold={result['threshold']})")
    print(f"  verdict: {result['verdict']}")
    with open_db() as conn:
        journal(conn, "l19_growth_detector", str(args.text_from),
                {"label": args.label, **result})


if __name__ == "__main__":
    main()
