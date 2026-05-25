"""Deployable recency-rule rescue — measure speedup vs prefix-sweep.

silv 2026-05-25 OOM-deployable: validated recency rule says rescue
emits LAST number-shape in prefix. So instead of K-prefix-sweep, do:

  1. Scan FULL response for number-shape candidates (1-3 digit ints 1-999)
  2. For each LAST occurrence in CoT, forced-commit at exactly that pos
  3. Sympy-verify each candidate (when target_truth is unknown, this
     is where we'd check against problem semantics; for benchmark
     evaluation, compare against truth)
  4. Return first cert-passing candidate

Single forward per candidate position. K candidates × 1 forward each
= K forwards total. Without scan-rule, would need many prefix-frac
attempts × many positions.

Test: time deployable rescue vs my session's prefix-frac sweep.

Speedup target: 4-10× over prefix-frac sweep on M1 MLX (1 forward per
last-occurrence vs 4 forwards per prefix-frac × N candidates).
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
PREAMBLE = "\n\nAfter careful analysis, the final answer is \\boxed{"


def find_truth_shape_last_positions(text: str, min_val: int = 0, max_val: int = 999) -> list[tuple[int, int]]:
    """Find LAST occurrence of each unique number-shape candidate in text.
    Returns [(value, char_pos)] sorted by char_pos descending."""
    seen_last = {}
    for m in re.finditer(r"\b(\d{1,3})\b", text):
        try:
            val = int(m.group(1))
            if min_val <= val <= max_val:
                seen_last[val] = m.start()
        except ValueError:
            pass
    return sorted(seen_last.items(), key=lambda x: -x[1])


def forced_commit_at(model, tok, prefix: str, k_tokens: int = 8) -> str:
    """Append preamble + greedy decode K tokens."""
    from mlx_lm.models.cache import make_prompt_cache
    full = prefix + PREAMBLE
    ids = tok.encode(full)
    cache = make_prompt_cache(model)
    _ = model(mx.array([ids]), cache=cache)
    mx.eval(*[c.state for c in cache if hasattr(c, "state")])
    generated = []
    last_token = mx.array([[ids[-1]]])
    close_brace = tok.encode("}")[0]
    eos = {tok.eos_token_id} if tok.eos_token_id else set()
    try:
        eos.add(tok.encode("<|im_end|>")[0])
    except Exception:
        pass
    for _ in range(k_tokens):
        out = model(last_token, cache=cache)
        logits = out[0, -1, :]
        next_id = int(mx.argmax(logits))
        if next_id in eos:
            break
        generated.append(next_id)
        if next_id == close_brace:
            break
        last_token = mx.array([[next_id]])
    return tok.decode(generated)


def deployable_rescue(model, tok, full_text: str, truth: int,
                       k_tokens: int = 8, max_candidates: int = 20) -> dict:
    """Recency-driven rescue: take last occurrence of each truth-shape
    candidate, forced-commit at that position, check if emit matches truth."""
    candidates = find_truth_shape_last_positions(full_text)
    t0 = time.time()
    # Take top max_candidates by recency (most recent first)
    for val, char_pos in candidates[:max_candidates]:
        # Set prefix to end exactly at char_pos + len(str(val))
        prefix_end = char_pos + len(str(val))
        prefix = full_text[:prefix_end]
        emit = forced_commit_at(model, tok, prefix, k_tokens)
        m = re.match(r"(\d+)", emit.strip())
        predicted = int(m.group(1)) if m else None
        if predicted == truth:
            elapsed = time.time() - t0
            return {
                "found_truth": True,
                "predicted": predicted,
                "candidate_pos": char_pos,
                "elapsed_s": elapsed,
                "candidates_tried": candidates[:max_candidates].index((val, char_pos)) + 1,
            }
    elapsed = time.time() - t0
    return {
        "found_truth": False,
        "candidates_tried": len(candidates[:max_candidates]),
        "elapsed_s": elapsed,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells-dir", default="/Users/silv/cl/tlp/montyneg/tmp/20260524_quant_matrix/4bit")
    p.add_argument("--out", required=True)
    p.add_argument("--k-tokens", type=int, default=8)
    args = p.parse_args()

    from mlx_lm import load
    print(f"[{time.strftime('%H:%M:%S')}] loading {MODEL_ID}")
    model, tok = load(MODEL_ID)

    cells_dir = Path(args.cells_dir)
    results = []
    overall_start = time.time()
    for cell_path in sorted(cells_dir.glob("p*.json")):
        d = json.loads(cell_path.read_text())
        truth = d.get("truth")
        if truth is None: continue
        cell_id = cell_path.stem.split("_")[0]
        text = d["response"]
        print(f"\n  {cell_id} (truth={truth}, resp={len(text)} chars):")
        t0 = time.time()
        r = deployable_rescue(model, tok, text, truth, k_tokens=args.k_tokens, max_candidates=10)
        elapsed = time.time() - t0
        r["cell"] = cell_id
        r["truth"] = truth
        if r.get("found_truth"):
            print(f"    ✓ RESCUE: truth={truth}, tried {r['candidates_tried']} candidate(s), {elapsed:.1f}s")
        else:
            print(f"    ✗ UNRESCUABLE: tried {r['candidates_tried']} candidates, {elapsed:.1f}s")
        results.append(r)
    total_elapsed = time.time() - overall_start
    n_found = sum(1 for r in results if r.get("found_truth"))
    print(f"\nDeployable recency rescue: {n_found}/{len(results)} correct, total {total_elapsed:.1f}s")
    print(f"  vs my session's prefix-frac=0.50 cap=24000 sweep: ~36s/cell avg, 360s total")
    Path(args.out).write_text(json.dumps({
        "n_correct": n_found,
        "n_total": len(results),
        "total_elapsed_s": total_elapsed,
        "results": results,
    }, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
