"""Per-layer logit-lens — is truth ENCODED at L19 and DESTROYED by late layers?

silv 2026-05-25 directive: read shifts.md to find unexplored direction.

Shift #287 found on Ministral-3-3B P04 (truth=279): peak L21 P=0.571, final
L25 P=0.231. Truth was ENCODED in mid-stack but DESTROYED by last 4 layers.
"The 'reasoning degrades retrieval' mechanism made microscopic."

My L19 attention-entropy-growth finding shows that at L19 the attention
distribution stops spreading during commitment-concentration. The natural
question: DOES THE LOGIT-LENS PROJECTION OF L19 HIDDEN STATE ENCODE TRUTH?

If P01 lock cell shows: P(truth) peaks at L19-ish then DECAYS by L31:
  ⇒ the lock IS the late-layer truth-destruction mechanism

If P01 lock cell shows: P(truth) is LOW at all layers:
  ⇒ the model genuinely doesn't have truth; lock is unrelated

Compose with codex H1812 finding (K256 sensitivity at L0/L19/L42 in DS4):
the SAME L19 is sensitive across two independent measurement paths.

Method:
  Project EVERY layer's hidden state at the LAST position through
  final_RMSNorm + tied_lm_head, get logits over vocab. Truth = first-digit
  token of the answer (e.g., '2' for truth=277).

Single forward pass with class-patched DecoderLayer wrapping to capture
post-layer hidden states. Cheap.
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


MODEL_ID = "mlx-community/Qwen3.5-4B-MLX-4bit"


def perlayer_logit_lens(model, tok, text: str, truth_digit: str,
                          last_n_positions: int = 16) -> dict:
    """For every layer, capture the hidden state at the last N positions.
    Project through final_norm + tied lm_head to get next-token logits.

    Returns:
      per_layer: dict[L_idx] = {"p_truth_per_pos": [...], "rank_truth_per_pos": [...]}
      where positions are the LAST last_n_positions tokens.
    """
    from mlx_lm.models.cache import make_prompt_cache
    internal = model.language_model.model
    layers = internal.layers
    n_layers = len(layers)

    # We will capture post-layer residual via class-patch on the DecoderLayer
    captured = {}  # L_idx -> (B, last_n, hidden_dim)

    DecCls = type(layers[0])
    orig_call = DecCls.__call__
    # Index by id to avoid mis-dispatch
    layer_idx_by_id = {id(layers[L]): L for L in range(n_layers)}

    def patched(self, x, mask=None, cache=None):
        out = orig_call(self, x, mask, cache)
        L_idx = layer_idx_by_id.get(id(self))
        if L_idx is not None:
            # capture last_n positions
            n_tok = out.shape[1]
            tail = out[:, max(0, n_tok - last_n_positions):, :]
            mx.eval(tail)
            captured[L_idx] = np.array(tail.astype(mx.float32), copy=True)
        return out

    DecCls.__call__ = patched

    ids = tok.encode(text)
    n_tok = len(ids)
    cache = make_prompt_cache(model)
    _ = model(mx.array([ids]), cache=cache)
    mx.eval(*[c.state for c in cache if hasattr(c, "state")])

    DecCls.__call__ = orig_call

    # Find truth digit token IDs
    truth_id = tok.encode(truth_digit)[0]  # often the bare digit is one token
    print(f"  truth_digit='{truth_digit}', token_id={truth_id}")
    print(f"  truth-string token decode: '{tok.decode([truth_id])}'")

    # Resolve final_norm + lm_head (Qwen3.5-4B uses tied embedding via .as_linear)
    final_norm = internal.norm  # RMSNorm
    if hasattr(model.language_model, "lm_head"):
        proj = lambda x: model.language_model.lm_head(x)
        print(f"  projection: untied lm_head")
    else:
        proj = lambda x: internal.embed_tokens.as_linear(x)
        print(f"  projection: tied embed_tokens.as_linear")

    per_layer = {}
    for L_idx in range(n_layers):
        if L_idx not in captured:
            continue
        h = captured[L_idx]  # (1, last_n, hidden)
        h_mx = mx.array(h, dtype=mx.float16)
        h_norm = final_norm(h_mx)  # apply final RMSNorm
        logits = proj(h_norm)
        mx.eval(logits)
        logits_np = np.array(logits[0], copy=True, dtype=np.float32)  # (last_n, vocab)
        # log-softmax for truth probability
        z = logits_np - logits_np.max(axis=-1, keepdims=True)
        ez = np.exp(z.astype(np.float64))
        sz = ez.sum(axis=-1, keepdims=True)
        probs = (ez / sz).astype(np.float32)
        p_truth = probs[:, truth_id]
        # Rank of truth in each position
        ranks = []
        for p in range(probs.shape[0]):
            ranks.append(int(np.sum(logits_np[p] > logits_np[p, truth_id])))
        per_layer[L_idx] = {
            "p_truth_per_pos": p_truth.tolist(),
            "rank_truth_per_pos": ranks,
            "mean_p_truth": float(p_truth.mean()),
            "max_p_truth": float(p_truth.max()),
            "mean_rank": float(np.mean(ranks)),
            "min_rank": int(min(ranks)),
        }

    return {
        "n_layers": n_layers,
        "n_tokens": n_tok,
        "last_n_positions": min(last_n_positions, n_tok),
        "truth_digit": truth_digit,
        "truth_token_id": int(truth_id),
        "per_layer": per_layer,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text-from", required=True)
    p.add_argument("--field", default="response")
    p.add_argument("--n-chars", type=int, default=4000)
    p.add_argument("--truth-digit", required=True, help="First digit of truth, e.g. '2' for 277")
    p.add_argument("--last-n", type=int, default=16)
    p.add_argument("--label", default="")
    p.add_argument("--out", type=str)
    args = p.parse_args()

    text = json.loads(Path(args.text_from).read_text())[args.field][:args.n_chars]
    from mlx_lm import load
    print(f"[{time.strftime('%H:%M:%S')}] loading {MODEL_ID}")
    model, tok = load(MODEL_ID)
    print(f"  text: {len(text)} chars")

    t0 = time.time()
    result = perlayer_logit_lens(model, tok, text, args.truth_digit, args.last_n)
    print(f"  perlayer logit-lens: {time.time()-t0:.1f}s")

    print(f"\n=== {args.label} (truth digit '{args.truth_digit}') ===")
    print(f"{'L':>3}  {'mean_p_truth':>12} {'max_p_truth':>11} {'min_rank':>9}")
    n_layers = result["n_layers"]
    for L in range(n_layers):
        if L not in result["per_layer"]: continue
        d = result["per_layer"][L]
        print(f"  {L:>2}  {d['mean_p_truth']:>12.6f} {d['max_p_truth']:>11.6f} {d['min_rank']:>9d}")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nsaved {args.out}")

    with open_db() as conn:
        journal(conn, "perlayer_logit_lens", str(args.text_from),
                {"label": args.label,
                 "truth_digit": args.truth_digit,
                 "max_p_truth_by_layer": {L: result["per_layer"][L]["max_p_truth"]
                                           for L in result["per_layer"]}})


if __name__ == "__main__":
    main()
