"""Top-K retention re-decode — doubting the single-token sampling fallacy.

silv 2026-05-25: "the deeper implications on training and inference, on sampling
the final/interim logits/weights and keeping only the temp/top_k instead of
employing sampling instrumentation."

Standard LLM decoding collapses logits → 1 token per step (argmax or sample).
Information loss: at each step, top_2..top_k positions and their probs are
DISCARDED. The single chosen token determines the prefix for next step;
ALL alternative branches die.

What if we RETAIN the top_k (k=16, say) logits/tokens at every position?
Then we have a branching tree of completions. Truth may live in a non-greedy
branch a few positions before the lock crystallizes.

This probe:
  1. Re-prefill the full cached response text.
  2. At each position, capture top_k tokens + log-probs.
  3. Search: did truth-shape (e.g., '277') ever appear in any top_k at any
     position?
  4. If yes: trace which positions had truth-shape at rank > 1; what was the
     log-prob ratio?
  5. Test BACK-DECODE: at the earliest position where truth-shape was in
     top_k, force-substitute that token and re-generate forward. Does the
     continuation reach truth?

This DIRECTLY refutes the sampling-as-instrumentation worldview if truth
lies in retained top_k but was discarded by argmax/sample at decode time.

The deeper fallacy named: LLM inference treats sampling as a WRAPPER around
the model (post-hoc temperature/top_k/top_p on argmax). The model produces
logits; sampling is "presentation." But the SEQUENTIAL collapse means each
decision is irreversible. Top_k retention + back-decode breaks that.
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


def capture_topk_per_position(model, tok, text: str, k: int = 16) -> dict:
    """Single prefill; capture top_k token IDs + log_probs at every position.

    Returns:
      ids: full token sequence (n_tok,)
      topk_ids: (n_tok, k) — top-k token IDs at each position
      topk_logp: (n_tok, k) — log-probs of those tokens
      argmax_ids: (n_tok,) — what argmax would have picked
    """
    from mlx_lm.models.cache import make_prompt_cache
    ids = tok.encode(text)
    n_tok = len(ids)
    cache = make_prompt_cache(model)
    t0 = time.time()
    out = model(mx.array([ids]), cache=cache)
    logits = out[0].astype(mx.float32)  # (n_tok, vocab)
    mx.eval(logits)
    logits_np = np.array(logits, copy=True, dtype=np.float32)
    print(f"  prefill+logits: {time.time()-t0:.1f}s ({n_tok} positions, vocab={logits_np.shape[1]})")

    # log_softmax for numerical stability
    z = logits_np - logits_np.max(axis=-1, keepdims=True)
    ez = np.exp(z.astype(np.float64))
    sz = ez.sum(axis=-1, keepdims=True)
    log_probs = (z - np.log(sz)).astype(np.float32)

    # Top-k at each position
    topk_idx = np.argpartition(-logits_np, k, axis=-1)[:, :k]
    # Sort top-k by logit value desc
    sort_order = np.argsort(-np.take_along_axis(logits_np, topk_idx, axis=-1), axis=-1)
    topk_idx_sorted = np.take_along_axis(topk_idx, sort_order, axis=-1)
    topk_logp_sorted = np.take_along_axis(log_probs, topk_idx_sorted, axis=-1)

    return {
        "ids": ids,
        "topk_ids": topk_idx_sorted,
        "topk_logp": topk_logp_sorted,
        "argmax_ids": topk_idx_sorted[:, 0],
        "k": k,
        "n_tok": n_tok,
    }


def find_truth_in_topk(captured: dict, tok, truth_str: str = "277") -> list[dict]:
    """For each position, check whether ANY of the top-k token decodings
    contain truth_str as a substring of the decoded text.

    Returns: [{"position": p, "rank": r, "logp": lp, "logp_ratio_to_top1": ...}]
    """
    n_tok = captured["n_tok"]
    k = captured["k"]
    topk_ids = captured["topk_ids"]
    topk_logp = captured["topk_logp"]
    found = []
    # For each position, decode each top-k token and check substring
    # Cheaper version: pre-build set of token IDs whose decoded form contains truth_str
    print(f"  building truth-shape token vocab (this finds tokens that decode to substrings containing '{truth_str}')...")
    # Sweep vocab is expensive; do it once
    truth_token_ids = set()
    vocab_size = topk_ids.max() + 1  # upper bound
    # Iterate only over unique top_k IDs across positions to bound work
    unique_ids = np.unique(topk_ids)
    print(f"  unique top-{k} IDs across all positions: {len(unique_ids)}")
    for tid in unique_ids:
        try:
            s = tok.decode([int(tid)])
            if truth_str in s:
                truth_token_ids.add(int(tid))
        except Exception:
            pass
    print(f"  tokens whose decode contains '{truth_str}': {len(truth_token_ids)}")
    if not truth_token_ids:
        return []
    # Now scan top_k array for any rank where token in truth_token_ids
    for p in range(n_tok):
        for r in range(k):
            tid = int(topk_ids[p, r])
            if tid in truth_token_ids:
                lp = float(topk_logp[p, r])
                lp_top1 = float(topk_logp[p, 0])
                tid_top1 = int(topk_ids[p, 0])
                found.append({
                    "position": p,
                    "rank": r,
                    "token_id": tid,
                    "token_text": tok.decode([tid]),
                    "logp": lp,
                    "logp_top1": lp_top1,
                    "token_top1_text": tok.decode([tid_top1]),
                    "logp_ratio": lp - lp_top1,
                })
    return found


def back_decode_from_position(model, tok, captured: dict, position: int,
                                substitute_token_id: int, n_gen: int = 200) -> str:
    """Re-decode from `position`, substituting the given token at that position
    instead of what argmax/sample chose. Returns the generated continuation.

    Equivalent to: take cached prefix [:position], append substitute_token_id,
    then generate forward.
    """
    from mlx_lm.models.cache import make_prompt_cache
    ids = captured["ids"]
    prefix = list(ids[:position]) + [substitute_token_id]
    cache = make_prompt_cache(model)
    _ = model(mx.array([prefix]), cache=cache)
    mx.eval(*[c.state for c in cache if hasattr(c, "state")])

    eos = set()
    if tok.eos_token_id is not None:
        eos.add(tok.eos_token_id)
    try:
        eos.add(tok.encode("<|im_end|>")[0])
    except Exception:
        pass

    generated = []
    last_token = mx.array([[substitute_token_id]])
    for step in range(n_gen):
        out = model(last_token, cache=cache)
        logits = out[0, -1, :].astype(mx.float32)
        next_id = int(mx.argmax(logits))
        if next_id in eos:
            break
        generated.append(next_id)
        last_token = mx.array([[next_id]])
    return tok.decode(generated)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text-from", required=True)
    p.add_argument("--field", default="response")
    p.add_argument("--n-chars", type=int, default=4000)
    p.add_argument("--truth", required=True, help="Truth value as string, e.g. '277'")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--n-back-decode", type=int, default=5, help="Number of back-decode attempts at earliest positions where truth was in top_k")
    p.add_argument("--n-gen", type=int, default=300)
    p.add_argument("--out", type=str)
    p.add_argument("--label", default="")
    args = p.parse_args()

    text = json.loads(Path(args.text_from).read_text())[args.field][:args.n_chars]
    from mlx_lm import load
    print(f"[{time.strftime('%H:%M:%S')}] loading {MODEL_ID}")
    model, tok = load(MODEL_ID)

    print(f"  text: {len(text)} chars\n")
    captured = capture_topk_per_position(model, tok, text, k=args.k)
    print(f"\n  searching for truth='{args.truth}' in top-{args.k} ...")
    found = find_truth_in_topk(captured, tok, args.truth)
    print(f"  occurrences in top-{args.k}: {len(found)}")

    if not found:
        print(f"\n  NO truth-shape token found at any rank — truth not in retained top_k.")
        print(f"  this REFUTES the top_k retention hypothesis for this cell.")
        if args.out:
            Path(args.out).write_text(json.dumps({
                "label": args.label,
                "found_in_topk": False,
                "n_positions": captured["n_tok"],
                "k": args.k,
            }, indent=2))
        return

    # Sort by EARLIEST position (back-decode is more interesting near the start)
    found_sorted = sorted(found, key=lambda x: (x["position"], x["rank"]))
    print(f"\n  first 10 occurrences of truth-shape in top-{args.k}:")
    for f in found_sorted[:10]:
        print(f"    pos={f['position']:>5} rank={f['rank']:>2} "
              f"token='{f['token_text']!r}' logp={f['logp']:+.4f} "
              f"top1='{f['token_top1_text']!r}' lp_diff={f['logp_ratio']:+.4f}")

    # Back-decode from a few EARLY positions where truth was in top_k but NOT top_1
    interesting = [f for f in found_sorted if f["rank"] > 0]
    print(f"\n  positions where truth-shape in top_k but NOT top_1: {len(interesting)}")
    back_decode_results = []
    # Try the FIRST n_back_decode such positions
    for f in interesting[:args.n_back_decode]:
        print(f"\n  back-decoding from pos={f['position']} with token='{f['token_text']!r}' "
              f"(was rank {f['rank']}, top1 was '{f['token_top1_text']!r}')")
        t0 = time.time()
        suffix = back_decode_from_position(model, tok, captured,
                                              position=f["position"],
                                              substitute_token_id=f["token_id"],
                                              n_gen=args.n_gen)
        elapsed = time.time() - t0
        truth_present = args.truth in suffix
        boxed_match = f"\\boxed{{{args.truth}}}" in suffix
        print(f"    elapsed={elapsed:.1f}s, truth_in_suffix={truth_present}, "
              f"boxed_match={boxed_match}")
        if boxed_match or truth_present:
            print(f"    SUFFIX TAIL: ...{suffix[-300:]}")
        back_decode_results.append({
            "position": f["position"],
            "rank": f["rank"],
            "substitute_token": f["token_text"],
            "elapsed_s": elapsed,
            "truth_in_suffix": truth_present,
            "boxed_match": boxed_match,
            "suffix_tail": suffix[-400:],
        })

    n_recoveries = sum(1 for r in back_decode_results if r["boxed_match"] or r["truth_in_suffix"])
    print(f"\n=== {args.label} ===")
    print(f"  Total top_k occurrences of truth-shape: {len(found_sorted)}")
    print(f"  Positions where truth in top_k but not top_1: {len(interesting)}")
    print(f"  Back-decode recoveries (truth in continuation): {n_recoveries}/{len(back_decode_results)}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "label": args.label,
            "found_in_topk": True,
            "n_top_k_hits": len(found_sorted),
            "n_off_top1_hits": len(interesting),
            "n_back_decode_recoveries": n_recoveries,
            "back_decode_results": back_decode_results,
            "first_10_hits": found_sorted[:10],
        }, indent=2))

    with open_db() as conn:
        journal(conn, "topk_retention_decode",
                str(args.text_from),
                {"label": args.label, "truth": args.truth,
                 "n_top_k_hits": len(found_sorted),
                 "n_off_top1_hits": len(interesting),
                 "n_recoveries": n_recoveries})


if __name__ == "__main__":
    main()
