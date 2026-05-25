"""Entropy trajectory probe — does the model encounter shallow-basin moments?

silv 2026-05-25: "burn 5 OOM energy to find the spark."

Following basin_geometry_probe finding: basin depth correlates with token
confidence (entropy/margin), NOT with locked-vs-success. The substrate
hypothesis is that LOOP positions never reach shallow basins (uncertain
moments where the model could explore alternatives).

This probe: per-position entropy + top-2 margin across the full response.
For each prompt: count "shallow-basin moments" (entropy > 2.0 nats AND
margin < 0.5 nats).

Hypothesis:
  H_LOCK_HAS_NO_UNCERTAINTY: P01 locked region has FEWER shallow-basin
  positions per 1000 tokens than P02/P05.

  REFUTE: P01 locked region has equal or more shallow-basin positions than
  commit-success cells. Would mean the substrate-mechanism claim is wrong;
  the lock-vs-success distinction lives somewhere else.

Method: ~3000 token forward, get logits at every position, compute entropy
+ margin. O(N) — single prefill pass. No perturbation needed.
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

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))
from codec_audit.db import open_db, journal

MODEL_ID = "mlx-community/Qwen3.5-4B-MLX-4bit"


def entropy_margin_trajectory(model, tok, text: str) -> dict:
    """Single prefill forward → per-position entropy + margin + top1."""
    ids = tok.encode(text)
    n_tok = len(ids)
    print(f"  tokens: {n_tok}")

    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    inp = mx.array([ids])
    t0 = time.time()
    out = model(inp, cache=cache)  # (1, n_tok, vocab) raw logits
    # Ensure compute completes
    out_f32 = out[0].astype(mx.float32)
    mx.eval(out_f32)
    logits = np.array(out_f32, copy=True, dtype=np.float32)  # (n_tok, vocab)
    print(f"  prefill+collect: {time.time()-t0:.1f}s")

    # Per-position: top1, entropy, margin
    top1 = np.argmax(logits, axis=-1)  # (n_tok,)
    # entropy: numerically stable softmax + entropy
    z = logits - logits.max(axis=-1, keepdims=True)
    ez = np.exp(z.astype(np.float64))
    sz = ez.sum(axis=-1, keepdims=True)
    probs = (ez / sz).astype(np.float32)
    # log-prob of top-2 → margin
    log_probs = np.log(np.maximum(probs, 1e-30))
    # top-2 indices per row
    top2_idx = np.argpartition(-logits, 2, axis=-1)[:, :2]
    # sort top-2 by logit value
    sorted_top2 = np.take_along_axis(
        top2_idx, np.argsort(-np.take_along_axis(logits, top2_idx, axis=-1), axis=-1), axis=-1
    )
    margin = (np.take_along_axis(log_probs, sorted_top2[:, :1], axis=-1)
              - np.take_along_axis(log_probs, sorted_top2[:, 1:2], axis=-1))[:, 0]
    # entropy
    safe_p = np.where(probs > 0, probs, 1.0)
    entropy = -np.sum(probs * np.log(safe_p), axis=-1)

    return {
        "n_tok": n_tok,
        "top1": top1.tolist(),
        "entropy_nats": entropy.tolist(),
        "margin_nats": margin.tolist(),
        "ids": ids,
    }


def summarize(traj: dict, shallow_entropy_threshold: float = 2.0,
              shallow_margin_threshold: float = 0.5,
              n_chunks: int = 20) -> dict:
    """Bin trajectory into chunks; report shallow-basin density."""
    n = traj["n_tok"]
    chunk_size = max(1, n // n_chunks)
    ent = np.asarray(traj["entropy_nats"])
    mar = np.asarray(traj["margin_nats"])
    shallow = (ent > shallow_entropy_threshold) & (mar < shallow_margin_threshold)
    chunks = []
    for k in range(n_chunks):
        lo = k * chunk_size
        hi = min((k + 1) * chunk_size, n) if k < n_chunks - 1 else n
        if hi <= lo:
            continue
        slc = slice(lo, hi)
        chunks.append({
            "chunk": k,
            "range": [lo, hi],
            "size": hi - lo,
            "mean_entropy": float(ent[slc].mean()),
            "mean_margin": float(mar[slc].mean()),
            "shallow_count": int(shallow[slc].sum()),
            "shallow_density": float(shallow[slc].mean()),
        })
    overall_shallow = float(shallow.mean())
    return {
        "n_tokens": n,
        "shallow_total": int(shallow.sum()),
        "shallow_density_overall": overall_shallow,
        "mean_entropy_overall": float(ent.mean()),
        "mean_margin_overall": float(mar.mean()),
        "chunks": chunks,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text-from", required=True)
    p.add_argument("--field", default="response")
    p.add_argument("--n-chars", type=int, default=10000)
    p.add_argument("--n-chunks", type=int, default=20)
    p.add_argument("--shallow-entropy", type=float, default=2.0)
    p.add_argument("--shallow-margin", type=float, default=0.5)
    p.add_argument("--label", default="")
    args = p.parse_args()

    text = json.loads(Path(args.text_from).read_text())[args.field][:args.n_chars]
    print(f"[{time.strftime('%H:%M:%S')}] loading model")
    from mlx_lm import load
    model, tok = load(MODEL_ID)

    print(f"  text: {len(text)} chars")
    traj = entropy_margin_trajectory(model, tok, text)
    summ = summarize(traj, args.shallow_entropy, args.shallow_margin, args.n_chunks)

    print(f"\n=== {args.label or Path(args.text_from).stem} ===")
    print(f"  n_tokens: {summ['n_tokens']}")
    print(f"  mean_entropy: {summ['mean_entropy_overall']:.4f}")
    print(f"  mean_margin: {summ['mean_margin_overall']:.4f}")
    print(f"  shallow positions (ent>{args.shallow_entropy} AND margin<{args.shallow_margin}): "
          f"{summ['shallow_total']} ({100*summ['shallow_density_overall']:.2f}%)")
    print(f"\n  chunk  range          ent    margin   shallow%")
    for c in summ['chunks']:
        bar = "#" * int(c['shallow_density'] * 50)
        print(f"  {c['chunk']:>2}     {c['range'][0]:>5}-{c['range'][1]:<5}  "
              f"{c['mean_entropy']:>5.2f}  {c['mean_margin']:>6.3f}  "
              f"{100*c['shallow_density']:>5.1f}%  {bar}")

    with open_db() as conn:
        journal(conn, "entropy_trajectory", str(args.text_from),
                {"label": args.label, "summary": summ})


if __name__ == "__main__":
    main()
