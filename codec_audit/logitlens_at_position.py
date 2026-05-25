"""Per-layer logit-lens at a SPECIFIC position (not just last_n).

silv 2026-05-25: test whether truth-digit is encoded at L19+ at chunk_2
boundary (the L19-detector trigger position) BEFORE the lock crystallizes.

P01 lock: truth digit '2' is at top-1 with P=0.9998 at L31 in the LOCKED tail
(perlayer_logit_lens.py result). The lock substrate KNOWS truth. The
question is: was it ALREADY encoded at chunk_2, before the lock cycled
many times?

Method: encode the FULL response, run forward, capture hidden state at
EVERY layer at a SPECIFIC position (e.g., chunk_2 boundary = position
int(0.4 * n_tok)). Project through final_norm + tied embed.as_linear for
each layer's hidden state. Get P(truth) per layer at that single position.

Falsifier:
  H_TRUTH_BEFORE_LOCK: at chunk_2 boundary in P01, P(truth=='2') > 0.5 at
  L25+ (truth already encoded at chunk_2 before lock crystallizes).

REFUTE: P(truth) < 0.01 at all layers at chunk_2 — model truly didn't know
  truth yet at chunk_2.
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

MODEL_ID = "mlx-community/Qwen3.5-4B-MLX-4bit"


def perlayer_lens_at_positions(model, tok, text: str, position_fracs: list[float],
                                  truth_digits: list[str]) -> dict:
    """For each (position_frac, truth_digit), measure P(truth_digit_first_token)
    at every layer.
    """
    from mlx_lm.models.cache import make_prompt_cache
    internal = model.language_model.model
    layers = internal.layers
    n_layers = len(layers)
    final_norm = internal.norm
    proj = lambda x: internal.embed_tokens.as_linear(x)

    DecCls = type(layers[0])
    orig_call = DecCls.__call__
    layer_idx_by_id = {id(layers[L]): L for L in range(n_layers)}

    # Pre-encode token ids
    ids = tok.encode(text)
    n_tok = len(ids)

    # Pre-resolve positions (clamped to valid range)
    target_positions = []
    for f in position_fracs:
        p = max(0, min(n_tok - 1, int(f * n_tok)))
        target_positions.append(p)
    print(f"  n_tok={n_tok}, target positions: {target_positions}")

    captured = {L: {} for L in range(n_layers)}  # L -> {position: hidden_state}

    def patched(self, x, mask=None, cache=None):
        out = orig_call(self, x, mask, cache)
        L_idx = layer_idx_by_id.get(id(self))
        if L_idx is None:
            return out
        for p in target_positions:
            if p < out.shape[1]:
                slice_p = out[:, p:p+1, :]
                mx.eval(slice_p)
                captured[L_idx][p] = np.array(slice_p.astype(mx.float32), copy=True)
        return out

    DecCls.__call__ = patched

    cache = make_prompt_cache(model)
    _ = model(mx.array([ids]), cache=cache)
    mx.eval(*[c.state for c in cache if hasattr(c, "state")])

    DecCls.__call__ = orig_call

    # Resolve truth token IDs
    truth_token_ids = []
    for td in truth_digits:
        tid = tok.encode(td)[0]
        truth_token_ids.append(tid)
        print(f"  truth_digit='{td}' -> token_id={tid} decode='{tok.decode([tid])}'")

    # For each (layer, position, truth_digit): compute P(truth) and rank
    result = {f"frac_{f:.2f}": {"position": tp, "per_layer": {}}
              for f, tp in zip(position_fracs, target_positions)}
    for f, p in zip(position_fracs, target_positions):
        key = f"frac_{f:.2f}"
        for L in range(n_layers):
            if p not in captured[L]:
                continue
            h = captured[L][p]  # (1, 1, hidden)
            h_mx = mx.array(h, dtype=mx.float16)
            h_norm = final_norm(h_mx)
            logits = proj(h_norm)
            mx.eval(logits)
            lg = np.array(logits[0, 0], copy=True, dtype=np.float32)
            # softmax
            z = lg - lg.max()
            probs = np.exp(z) / np.exp(z).sum()
            # For each truth digit
            per_truth = {}
            for td, tid in zip(truth_digits, truth_token_ids):
                p_truth = float(probs[tid])
                rank = int(np.sum(lg > lg[tid]))
                per_truth[td] = {"p": p_truth, "rank": rank}
            top1_id = int(np.argmax(lg))
            top1_tok = tok.decode([top1_id])
            result[key]["per_layer"][L] = {
                "truths": per_truth,
                "top1_id": top1_id,
                "top1_token": top1_tok,
                "top1_prob": float(probs[top1_id]),
            }

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text-from", required=True)
    p.add_argument("--field", default="response")
    p.add_argument("--n-chars", type=int, default=4000)
    p.add_argument("--truth-digits", nargs="+", required=True,
                    help="Truth first digits to track, e.g. '2' '7'")
    p.add_argument("--position-fracs", nargs="+", type=float,
                    default=[0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0])
    p.add_argument("--label", default="")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    text = json.loads(Path(args.text_from).read_text())[args.field][:args.n_chars]
    from mlx_lm import load
    print(f"[{time.strftime('%H:%M:%S')}] loading {MODEL_ID}")
    model, tok = load(MODEL_ID)
    print(f"  text: {len(text)} chars")
    result = perlayer_lens_at_positions(model, tok, text, args.position_fracs, args.truth_digits)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"  saved {args.out}")

    # Pretty-print per position
    print(f"\n=== {args.label} ===")
    for frac_key, info in result.items():
        print(f"\n  {frac_key} (position={info['position']}):")
        print(f"  {'L':>3}  " + "  ".join(f"P({td})".rjust(10) for td in args.truth_digits)
               + f"  {'top1':>10} {'top1_p':>8}")
        for L in sorted(info["per_layer"].keys()):
            r = info["per_layer"][L]
            tprobs = [f"{r['truths'][td]['p']:.4f}" for td in args.truth_digits]
            t1 = repr(r['top1_token'][:8])
            print(f"  {L:>2}  " + "  ".join(tp.rjust(10) for tp in tprobs)
                   + f"  {t1:>10} {r['top1_prob']:>8.4f}")


if __name__ == "__main__":
    main()
