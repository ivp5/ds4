"""Forced-commit substrate probe — does L23-L31 install truth when rescue forces commit?

silv 2026-05-25: continue queued tasks + sampling-fallacy critique.

Setup: take a no-commit cell (lock or explore-no-commit), truncate CoT
to chunk_2 (40% of tokens), append forced-commit preamble
'\\n\\nAfter careful analysis, the final answer is \\boxed{', then read
per-layer logit-lens at the position right after '{'.

Hypothesis test:
  H_FORCED_INSTALLS_TRUTH: at the forced-commit position, L23-L31 will
  install whatever digit the model retrieves (truth if in CoT context,
  wrong otherwise). The substrate-install mechanism doesn't care WHETHER
  the answer is right.

  REFUTE: per-layer at forced-commit position is FLAT (no install
  pattern) — would mean forced-commit doesn't engage the install tier.

This is the substrate-level analog of the rescue protocol; tests whether
the rescue extracts truth from the SUBSTRATE or just from TEXT-context.
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


MODEL_ID = "mlx-community/Qwen3.5-4B-MLX-4bit"
FORCED_PREAMBLE = "\n\nAfter careful analysis, the final answer is \\boxed{"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text-from", required=True)
    p.add_argument("--field", default="response")
    p.add_argument("--prefix-frac", type=float, default=0.40,
                    help="Fraction of cached response to use as prefix")
    p.add_argument("--n-chars-cap", type=int, default=12000,
                    help="Cap prefix at N chars (avoid huge prompts)")
    p.add_argument("--truth-digit", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    d = json.loads(Path(args.text_from).read_text())
    full_text = d["response"]
    print(f"  full response: {len(full_text)} chars; truth={d.get('truth')}")

    prefix_len = min(int(args.prefix_frac * len(full_text)), args.n_chars_cap)
    prefix = full_text[:prefix_len]
    forced_prompt = prefix + FORCED_PREAMBLE
    print(f"  prefix: {prefix_len} chars; forced prompt: {len(forced_prompt)} chars")
    print(f"  prefix tail: ...{prefix[-80:]!r}")
    print(f"  forced preamble: {FORCED_PREAMBLE!r}")

    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    print(f"[{time.strftime('%H:%M:%S')}] loading {MODEL_ID}")
    model, tok = load(MODEL_ID)
    internal = model.language_model.model
    layers = internal.layers
    n_layers = len(layers)
    final_norm = internal.norm
    proj_lm = lambda x: internal.embed_tokens.as_linear(x)

    captured = {}
    DecCls = type(layers[0])
    orig_call = DecCls.__call__
    layer_idx_by_id = {id(layers[L]): L for L in range(n_layers)}

    def patched(self, x, mask=None, cache=None):
        out = orig_call(self, x, mask, cache)
        L_idx = layer_idx_by_id.get(id(self))
        if L_idx is None:
            return out
        last = out[:, -1:, :]
        mx.eval(last)
        captured[L_idx] = np.array(last.astype(mx.float32), copy=True)
        return out

    DecCls.__call__ = patched
    ids = tok.encode(forced_prompt)
    print(f"  total prompt tokens: {len(ids)}")
    cache = make_prompt_cache(model)
    t0 = time.time()
    _ = model(mx.array([ids]), cache=cache)
    mx.eval(*[c.state for c in cache if hasattr(c, "state")])
    DecCls.__call__ = orig_call
    print(f"  forward + capture: {time.time()-t0:.1f}s")

    truth_id = tok.encode(args.truth_digit)[0]
    print(f"  truth_digit='{args.truth_digit}' → token_id={truth_id}")

    # Per-layer projection at last position
    per_layer = {}
    for L in range(n_layers):
        if L not in captured: continue
        h = mx.array(captured[L], dtype=mx.float16)
        h_norm = final_norm(h)
        logits = proj_lm(h_norm)
        mx.eval(logits)
        lg = np.array(logits[0, 0], copy=True, dtype=np.float32)
        z = lg - lg.max()
        probs = np.exp(z) / np.exp(z).sum()
        top1_id = int(np.argmax(lg))
        per_layer[L] = {
            "p_truth": float(probs[truth_id]),
            "rank_truth": int(np.sum(lg > lg[truth_id])),
            "top1_id": top1_id,
            "top1_token": tok.decode([top1_id]),
            "top1_prob": float(probs[top1_id]),
        }

    print(f"\n=== {args.label} (truth_digit='{args.truth_digit}') ===")
    print(f"{'L':>3}  {'top1':>10}  {'top1_prob':>9}  {'p_truth':>9}  {'rank':>6}")
    for L in sorted(per_layer.keys()):
        r = per_layer[L]
        if L < 18: continue
        print(f"  {L:>2}  {r['top1_token'][:10]!r:>12}  {r['top1_prob']:>9.4f}  {r['p_truth']:>9.4f}  {r['rank_truth']:>6d}")

    # Cross-layer agreement
    digit_top1 = sum(1 for L in per_layer if per_layer[L]['top1_token'].strip() == args.truth_digit)
    final3 = [L for L in [29, 30, 31] if L in per_layer]
    final3_agree = sum(1 for L in final3 if per_layer[L]['top1_token'].strip() == args.truth_digit)
    print(f"\n  Truth-digit top_1 count overall: {digit_top1}/{n_layers}")
    print(f"  Final-3 ({final3}) agree on '{args.truth_digit}': {final3_agree}/{len(final3)}")

    Path(args.out).write_text(json.dumps({
        "label": args.label,
        "truth_digit": args.truth_digit,
        "prefix_chars": prefix_len,
        "total_prompt_tokens": len(ids),
        "per_layer": per_layer,
        "digit_top1_count": digit_top1,
        "final3_agreement": final3_agree,
    }, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
