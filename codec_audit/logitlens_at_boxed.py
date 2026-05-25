"""Logit-lens at the EXACT commit position — does substrate encode truth there?

silv 2026-05-25: doubt every step. My session's "P01 lock substrate
contains truth" was REFUTED on closer look. Now test substrate at the
REAL commit position: text up to and including '\\boxed{', last token
predicts the first digit of the answer. Per-layer P(truth digit) at THAT
position.

Method:
  1. Find `\\boxed{` in the full response text (not truncated 4000-char view).
  2. Take prefix up to right after '{' opening brace.
  3. Forward-pass on that prefix; capture all-layer hidden states at the
     LAST position (the position that will predict the first answer digit).
  4. Project through final_norm + embed.as_linear; compute P(truth digit).

Falsifier:
  H_NATIVE_SUCCESS_HAS_TRUTH_AT_COMMIT: in P05 (native correct), at the
  position right after '\\boxed{', truth digit '6' (for 65) is at top-1 with
  P > 0.5 at L31.
  REFUTE: P(truth) < 0.1 at all layers — would mean even success cells
  don't have the answer in their substrate; they LATCH onto correct via
  context-completion.

Composition with earlier findings:
  - L19 attention growth signal fires when model concentrates on commit-attempt
  - per-layer logit-lens at commit position reveals what THAT commit encodes
  - Together: signal predicts "commit attempt incoming"; lens confirms
    whether the substrate ACTUALLY HAS the truth.
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


def find_boxed_emission_positions(text: str) -> list[int]:
    """Return character positions right after each '\\boxed{' opening."""
    positions = []
    for m in re.finditer(r"\\boxed\{", text):
        positions.append(m.end())  # right after '{'
    return positions


def perlayer_at_position(model, tok, prefix_text: str,
                         truth_digit: str) -> dict:
    """Forward on prefix_text, capture all-layer hidden states at LAST position,
    project to logits, return per-layer P(truth_digit) and top1."""
    from mlx_lm.models.cache import make_prompt_cache
    internal = model.language_model.model
    layers = internal.layers
    n_layers = len(layers)
    final_norm = internal.norm
    proj = lambda x: internal.embed_tokens.as_linear(x)

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
    ids = tok.encode(prefix_text)
    cache = make_prompt_cache(model)
    _ = model(mx.array([ids]), cache=cache)
    mx.eval(*[c.state for c in cache if hasattr(c, "state")])
    DecCls.__call__ = orig_call

    truth_id = tok.encode(truth_digit)[0]
    print(f"  truth_digit='{truth_digit}' → token_id={truth_id} (decode='{tok.decode([truth_id])}')")
    print(f"  prefix len: {len(prefix_text)} chars, {len(ids)} tokens")

    result = {}
    for L in range(n_layers):
        if L not in captured:
            continue
        h = mx.array(captured[L], dtype=mx.float16)
        h_norm = final_norm(h)
        logits = proj(h_norm)
        mx.eval(logits)
        lg = np.array(logits[0, 0], copy=True, dtype=np.float32)
        z = lg - lg.max()
        probs = np.exp(z) / np.exp(z).sum()
        p_truth = float(probs[truth_id])
        rank = int(np.sum(lg > lg[truth_id]))
        top1_id = int(np.argmax(lg))
        top1_tok = tok.decode([top1_id])
        result[L] = {
            "p_truth": p_truth,
            "rank_truth": rank,
            "top1_id": top1_id,
            "top1_token": top1_tok,
            "top1_prob": float(probs[top1_id]),
        }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text-from", required=True)
    p.add_argument("--field", default="response")
    p.add_argument("--truth-digit", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    d = json.loads(Path(args.text_from).read_text())
    full_text = d[args.field]
    print(f"  full response: {len(full_text)} chars")
    boxed_positions = find_boxed_emission_positions(full_text)
    print(f"  \\boxed{{ openings: {len(boxed_positions)}")
    if not boxed_positions:
        print(f"  NO commit positions in response — cell didn't emit \\boxed{{}}")
        Path(args.out).write_text(json.dumps({"label": args.label, "no_boxed": True}, indent=2))
        return

    from mlx_lm import load
    print(f"[{time.strftime('%H:%M:%S')}] loading {MODEL_ID}")
    model, tok = load(MODEL_ID)

    # For each \boxed{ emission, run per-layer lens at the position right after '{'
    all_results = []
    for pos_idx, pos in enumerate(boxed_positions[:3]):  # first 3 commit positions
        prefix = full_text[:pos]
        # Show what comes right after this prefix in the original
        print(f"\n  commit {pos_idx} at char {pos}, prefix ends with: ...{prefix[-80:]!r}")
        print(f"  following 30 chars: '{full_text[pos:pos+30]}'")
        per_layer = perlayer_at_position(model, tok, prefix, args.truth_digit)
        # Show key layers
        print(f"\n  per-layer at commit position {pos_idx}:")
        for L in [0, 5, 10, 15, 18, 19, 23, 27, 29, 30, 31]:
            if L in per_layer:
                r = per_layer[L]
                print(f"    L{L:>2}  P({args.truth_digit})={r['p_truth']:.6f} rank={r['rank_truth']:>5d}  "
                      f"top1={r['top1_token'][:8]!r} p_top1={r['top1_prob']:.4f}")
        all_results.append({
            "commit_position_idx": pos_idx,
            "char_position": pos,
            "context_before_50": prefix[-50:],
            "context_after_30": full_text[pos:pos+30],
            "per_layer": per_layer,
        })

    out_data = {
        "label": args.label,
        "text_from": args.text_from,
        "truth_digit": args.truth_digit,
        "n_boxed_emissions": len(boxed_positions),
        "commits_analyzed": len(all_results),
        "results": all_results,
    }
    Path(args.out).write_text(json.dumps(out_data, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
