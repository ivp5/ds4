"""Forced-commit + greedy continuation — extract model's predicted answer.

silv 2026-05-25: deployable form of forced-commit substrate finding.

For each cell, take prefix chunk_2 + '\\boxed{' preamble + greedy decode
K=8 tokens. The model emits a number (the answer it commits to under
duress). Then check if it matches truth.

This tests the OPERATIONAL form: at the L31 substrate truth-rank check
position, the model emits a multi-digit number. Verify against truth.

Composes with the 10-cell substrate truth-rank gradient:
- rank 0 → expect truth-digit emit
- rank 1-2 → expect near-truth or truth emit
- rank ≥ 5 → expect wrong commit
"""
import argparse
import json
import os
import random
import re
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


def forced_extract(model, tok, prefix: str, k_tokens: int = 8) -> str:
    """Append forced preamble, greedy decode k tokens, return the continuation."""
    from mlx_lm.models.cache import make_prompt_cache
    full = prefix + FORCED_PREAMBLE
    ids = tok.encode(full)
    cache = make_prompt_cache(model)
    _ = model(mx.array([ids]), cache=cache)
    mx.eval(*[c.state for c in cache if hasattr(c, "state")])
    generated = []
    last_token = mx.array([[ids[-1]]])  # use last prompt token
    eos = set()
    if tok.eos_token_id is not None:
        eos.add(tok.eos_token_id)
    try:
        eos.add(tok.encode("<|im_end|>")[0])
    except Exception:
        pass
    # Stop on closing '}'
    close_brace_id = tok.encode("}")[0]
    for step in range(k_tokens):
        out = model(last_token, cache=cache)
        logits = out[0, -1, :]
        next_id = int(mx.argmax(logits))
        if next_id in eos:
            break
        generated.append(next_id)
        if next_id == close_brace_id:
            break
        last_token = mx.array([[next_id]])
    return tok.decode(generated)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells-dir", default="/Users/silv/cl/tlp/montyneg/tmp/20260524_quant_matrix/4bit")
    p.add_argument("--prefix-frac", type=float, default=0.40)
    p.add_argument("--n-chars-cap", type=int, default=12000)
    p.add_argument("--k-tokens", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from mlx_lm import load
    print(f"[{time.strftime('%H:%M:%S')}] loading {MODEL_ID}")
    model, tok = load(MODEL_ID)

    cells = sorted([f for f in Path(args.cells_dir).iterdir() if f.suffix == ".json"])
    results = []
    for cell_path in cells:
        d = json.loads(cell_path.read_text())
        cell_id = cell_path.stem.split("_")[0]
        truth = d.get("truth")
        if truth is None: continue
        full_text = d["response"]
        prefix_len = min(int(args.prefix_frac * len(full_text)), args.n_chars_cap)
        prefix = full_text[:prefix_len]
        t0 = time.time()
        emit = forced_extract(model, tok, prefix, args.k_tokens)
        elapsed = time.time() - t0
        # Extract the number from emit (digits possibly followed by '}')
        m = re.match(r"(\d+)", emit.strip())
        predicted = int(m.group(1)) if m else None
        is_correct = (predicted == truth)
        print(f"  {cell_id}: truth={truth} predicted={predicted} correct={is_correct} "
              f"emit={emit[:30]!r} elapsed={elapsed:.1f}s")
        results.append({
            "cell": cell_id,
            "truth": truth,
            "predicted": predicted,
            "is_correct": is_correct,
            "emit": emit,
            "prefix_chars": prefix_len,
            "elapsed_s": elapsed,
        })
    n_correct = sum(1 for r in results if r['is_correct'])
    print(f"\nTotal: {n_correct}/{len(results)} cells correct via forced-commit + greedy K={args.k_tokens}")
    Path(args.out).write_text(json.dumps({
        "prefix_frac": args.prefix_frac,
        "n_chars_cap": args.n_chars_cap,
        "k_tokens": args.k_tokens,
        "model_id": MODEL_ID,
        "n_correct": n_correct,
        "n_total": len(results),
        "results": results,
    }, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
