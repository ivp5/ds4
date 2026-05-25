"""Per-token compute requirement probe — full 4-signal version.

silv 2026-05-25: extend the per-token detector beyond min_layer to also
record entropy + margin + layer-changes. Compare distributions across
prompts to surface coherent-vs-loop signature.

Method:
  1. Load Qwen3.5-4B-MLX-4bit ONCE.
  2. Tokenize input text + run prefill with per-layer residual capture.
  3. For each position: project EVERY layer's residual through lm_head;
     record per-layer top-1 token.
  4. Compute 4 signals per position via codec_audit.compute_per_token_signals
  5. Classify via codec_audit.classify_compute
  6. Write each detection to the DB through insert_token_compute (append-only).

Usage:
  python3 -m codec_audit.probe_per_token \
    --text-from /path/to/p02_cached_response.json \
    --field response --n-chars 1500
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Proxy + offline mode BEFORE HF imports
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
proxy_port = random.choice(range(10001, 10150))
os.environ.setdefault("HTTPS_PROXY", f"socks5h://127.0.0.1:{proxy_port}")
os.environ.setdefault("HTTP_PROXY", f"socks5h://127.0.0.1:{proxy_port}")
os.environ.setdefault("ALL_PROXY", f"socks5h://127.0.0.1:{proxy_port}")

import numpy as np
import mlx.core as mx
import mlx.nn as nn

# Make the codec_audit package importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from codec_audit.db import open_db, journal
from codec_audit.per_token_compute import (
    register_prompt, compute_per_token_signals, classify_compute,
    insert_token_compute, class_histogram,
)
from codec_audit.sweep import new_run, end_run

MODEL_ID = "mlx-community/Qwen3.5-4B-MLX-4bit"
TOP_K_KEEP = 3   # keep top-3 per layer for change-counting


def main():
    p = argparse.ArgumentParser("probe_per_token")
    p.add_argument("--text-from", required=True,
                    help="JSON file containing the text to analyze")
    p.add_argument("--field", default="response",
                    help="JSON field name to extract text from")
    p.add_argument("--n-chars", type=int, default=1500,
                    help="how many leading chars to analyze")
    p.add_argument("--notes", default="",
                    help="freeform notes for prompt_run row")
    args = p.parse_args()

    text = json.loads(Path(args.text_from).read_text())[args.field]
    text = text[: args.n_chars]
    print(f"text head ({len(text)} chars): {text[:80]!r}...")

    from mlx_lm import load
    print(f"[{time.strftime('%H:%M:%S')}] loading {MODEL_ID}")
    t0 = time.time()
    model, tok = load(MODEL_ID)
    print(f"loaded in {time.time()-t0:.1f}s")

    ids = tok.encode(text)
    n_tok = len(ids)
    print(f"tokens: {n_tok}")

    internal = model.language_model.model
    layers = internal.layers
    n_layers = len(layers)
    layer_idx = {id(l): i for i, l in enumerate(layers)}
    captured = [None] * n_layers

    def patched_layer_call(self, x, mask=None, cache=None):
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        mlp_r = self.mlp(self.post_attention_layernorm(h))
        out = h + mlp_r
        idx = layer_idx.get(id(self))
        if idx is not None:
            captured[idx] = out
        return out

    LayerCls = type(layers[0])
    orig = LayerCls.__call__
    LayerCls.__call__ = patched_layer_call

    print(f"[{time.strftime('%H:%M:%S')}] prefill forward")
    t0 = time.time()
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    inp = mx.array([ids])  # (1, n_tok)
    _ = model(inp, cache=cache)
    mx.eval(*[c.state for c in cache if hasattr(c, "state")])
    print(f"prefill done in {time.time()-t0:.1f}s")

    LayerCls.__call__ = orig

    # Final-layer hidden + lm_head projection
    final_norm = internal.norm
    embed = internal.embed_tokens
    final_h = mx.eval(final_norm(captured[-1]))  # (1, n_tok, hidden)

    # Project each layer's residual through final_norm + tied lm_head
    print(f"[{time.strftime('%H:%M:%S')}] computing per-layer top-1")
    per_layer_top1 = []  # per layer: array of (n_tok,) ints
    for L in range(n_layers):
        h = final_norm(captured[L])
        logits = embed.as_linear(h)  # (1, n_tok, vocab)
        top1 = mx.argmax(logits, axis=-1)  # (1, n_tok)
        per_layer_top1.append(np.asarray(top1[0]))
    # Final logits (n_tok, vocab) — cast to fp32 then numpy
    final_logits = embed.as_linear(final_norm(captured[-1]))
    final_logits_f32 = final_logits[0].astype(mx.float32)
    mx.eval(final_logits_f32)
    final_logits_np = np.array(final_logits_f32, copy=True, dtype=np.float32)

    print(f"[{time.strftime('%H:%M:%S')}] classifying {n_tok} positions")
    rows = []
    for pos in range(n_tok):
        per_layer_topk = [[int(per_layer_top1[L][pos])] for L in range(n_layers)]
        sig = compute_per_token_signals(per_layer_topk, final_logits_np[pos])
        cls = classify_compute(
            min_stab_L=sig["min_stabilize_layer"],
            entropy_nats=sig["final_entropy_nats"],
            margin_nats=sig["final_top1_margin"],
            n_layer_changes=sig["layer_top1_changes"],
            n_layers=n_layers,
        )
        rows.append((pos, ids[pos], sig, cls))

    # Decode token text once
    print(f"[{time.strftime('%H:%M:%S')}] writing to DB")
    text_for_each = [tok.decode([tid]) for tid in ids]

    with open_db() as conn:
        rid = new_run(conn, f"probe_per_token::{Path(args.text_from).name}",
                       params={"source": str(args.text_from), "n_chars": args.n_chars,
                               "n_tokens": n_tok, "notes": args.notes})
        text_proxy = f"{MODEL_ID}::{Path(args.text_from).name}::{args.n_chars}chars"
        prompt_id = register_prompt(conn, model=MODEL_ID, text=text_proxy,
                                      n_tokens=n_tok, notes=args.notes)
        for (pos, tid, sig, cls) in rows:
            insert_token_compute(conn, prompt_id=prompt_id, position=pos,
                                  token_id=tid, token_text=text_for_each[pos],
                                  signals=sig, compute_class=cls, run_id=rid)
        hist = class_histogram(conn, prompt_id)
        end_run(conn, rid, {"n_tokens": n_tok, "histogram": hist})

    print(f"\n=== class histogram for {Path(args.text_from).name} ===")
    for r in hist:
        print(f"  {r['compute_class']:<10}  n={r['n']:>4}  "
              f"mean_L={r['mean_L']:.1f}  range=[{r['min_L']},{r['max_L']}]")
    print(f"\nprompt_id: {prompt_id}, total wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
