"""L_commit-tier P(top_1) trajectory — does the substrate decide BEFORE emission?

silv 2026-05-25: sampling-instrumentation fallacy — the model's actual
distribution at each position is the deliverable, not the sampled token.

Hypothesis H_DECIDE_BEFORE_EMIT: at some layer in the commit tier
(L25-L31 on Qwen3.5-4B), P(top_1) rises substantially BEFORE the model
emits `\\boxed{N}`. If so, that layer's P(top_1) is a CHEAP early-warning
signal of imminent commit.

Method:
  - Single forward on response text via class-patched DecoderLayer to
    capture L29 hidden state at EVERY position.
  - Project through final_norm + tied embed.as_linear → per-position logits.
  - Compute P(top_1) at each position.
  - Plot trajectory across the generation.

Test: at P05 native success, find positions where L29 P(top_1) > 0.5
SUSTAINED, then compare to actual `\\boxed{` emission position.
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

import re
import numpy as np
import mlx.core as mx


MODEL_ID = "mlx-community/Qwen3.5-4B-MLX-4bit"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text-from", required=True)
    p.add_argument("--field", default="response")
    p.add_argument("--n-chars", type=int, default=20000)
    p.add_argument("--layer", type=int, default=29, help="Layer to project (commit-tier)")
    p.add_argument("--stride", type=int, default=50, help="Sample every N positions")
    p.add_argument("--out", required=True)
    p.add_argument("--label", default="")
    args = p.parse_args()

    text = json.loads(Path(args.text_from).read_text())[args.field][:args.n_chars]
    from mlx_lm import load
    print(f"[{time.strftime('%H:%M:%S')}] loading {MODEL_ID}")
    model, tok = load(MODEL_ID)
    internal = model.language_model.model
    layers = internal.layers
    final_norm = internal.norm
    proj = lambda x: internal.embed_tokens.as_linear(x)

    captured = {}
    DecCls = type(layers[0])
    orig_call = DecCls.__call__
    target_id = id(layers[args.layer])

    def patched(self, x, mask=None, cache=None):
        out = orig_call(self, x, mask, cache)
        if id(self) == target_id:
            mx.eval(out)
            captured["hidden"] = np.array(out[0].astype(mx.float32), copy=True)
        return out

    DecCls.__call__ = patched
    ids = tok.encode(text)
    n_tok = len(ids)
    print(f"  tokens: {n_tok}; sampling every {args.stride} positions = {n_tok // args.stride} positions")
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    t0 = time.time()
    _ = model(mx.array([ids]), cache=cache)
    mx.eval(*[c.state for c in cache if hasattr(c, "state")])
    print(f"  forward: {time.time()-t0:.1f}s")
    DecCls.__call__ = orig_call

    h_all = captured["hidden"]  # (n_tok, hidden)
    print(f"  captured L{args.layer} hidden states: shape {h_all.shape}")

    # Sample positions
    positions = list(range(0, n_tok, args.stride))
    trajectory = []
    for p in positions:
        h_p = mx.array(h_all[p:p+1], dtype=mx.float16)  # (1, hidden)
        h_norm = final_norm(h_p[None, ...])  # (1, 1, hidden)
        logits = proj(h_norm)
        mx.eval(logits)
        lg = np.array(logits[0, 0], copy=True, dtype=np.float32)
        z = lg - lg.max()
        probs = np.exp(z) / np.exp(z).sum()
        top1_id = int(np.argmax(lg))
        top1_prob = float(probs[top1_id])
        top1_tok = tok.decode([top1_id])
        # Also: what character position is this in original text?
        char_pos = len(tok.decode(ids[:p+1])) if p < n_tok - 1 else len(text)
        trajectory.append({
            "tok_pos": p,
            "char_pos": char_pos,
            "top1_id": top1_id,
            "top1_token": top1_tok,
            "top1_prob": top1_prob,
        })

    # Where is `\boxed{` in the text?
    boxed_chars = [m.end() for m in re.finditer(r"\\boxed\{", text)]
    print(f"  \\boxed{{ positions: {boxed_chars}")

    # Summary
    print(f"\n  Trajectory of L{args.layer} top_1 confidence:")
    print(f"  {'pos':>5}  {'char':>6}  {'P(top1)':>9}  {'top1':>15}  {'near_boxed?':>11}")
    for t in trajectory[::max(1, len(trajectory)//40)]:  # ~40 rows
        char_near_boxed = "BOXED" if any(abs(t["char_pos"] - b) < 50 for b in boxed_chars) else ""
        print(f"  {t['tok_pos']:>5}  {t['char_pos']:>6}  {t['top1_prob']:>9.4f}  {t['top1_token'][:15]!r:>17}  {char_near_boxed:>11}")

    # Test H_DECIDE_BEFORE_EMIT: find first position where P(top_1) > 0.5 sustained for 3 consecutive samples
    high_runs = []
    cur_run = 0
    for i, t in enumerate(trajectory):
        if t["top1_prob"] > 0.5:
            cur_run += 1
            if cur_run >= 3 and not high_runs:
                # found first sustained high
                first_sustained = trajectory[i - 2]
                high_runs.append(first_sustained)
        else:
            cur_run = 0
    if high_runs and boxed_chars:
        print(f"\n  First sustained-high P(top_1)>0.5 at L{args.layer}: char_pos={high_runs[0]['char_pos']}")
        print(f"  First \\boxed{{ emission at char_pos={boxed_chars[0]}")
        print(f"  Gap: {boxed_chars[0] - high_runs[0]['char_pos']} chars BEFORE commit emission")

    Path(args.out).write_text(json.dumps({
        "label": args.label,
        "model_id": MODEL_ID,
        "layer": args.layer,
        "stride": args.stride,
        "n_tok": n_tok,
        "boxed_positions": boxed_chars,
        "trajectory": trajectory,
    }, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
