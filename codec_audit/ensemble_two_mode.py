"""Ensemble two-mode detection — Qwen + DS-R1-7B on all 10 cells.

silv 2026-05-25: validate cross-model agreement as Mode 1 vs Mode 2
detection signal.

For each cell at cap=16000 prefix-frac=1.0:
- Both models run forced-commit + greedy K=8
- Compare emits
- If both emit truth → MODE_1_agree_correct
- If both emit same non-truth → MODE_2_collusion (rare)
- If they emit different non-truths → MODE_2_disagreement
- If one emits truth, one wrong → mixed
"""
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("ALL_PROXY", "socks5h://127.0.0.1:10001")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


PREAMBLE = "\n\nAfter careful analysis, the final answer is \\boxed{"


def forced_emit(model, tok, prompt: str, k_tokens: int = 8) -> str:
    ids = tok.encode(prompt)
    cache = make_prompt_cache(model)
    _ = model(mx.array([ids]), cache=cache)
    mx.eval(*[c.state for c in cache if hasattr(c, "state")])
    last_token = mx.array([[ids[-1]]])
    generated = []
    close = tok.encode("}")[0]
    for _ in range(k_tokens):
        out = model(last_token, cache=cache)
        next_id = int(mx.argmax(out[0, -1, :]))
        if next_id == close:
            generated.append(next_id); break
        generated.append(next_id)
        last_token = mx.array([[next_id]])
    return tok.decode(generated)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cells-dir", default="/Users/silv/cl/tlp/montyneg/tmp/20260524_quant_matrix/4bit")
    p.add_argument("--n-chars-cap", type=int, default=16000)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    print("loading Qwen3.5-4B-MLX-4bit...")
    qwen_m, qwen_t = load("mlx-community/Qwen3.5-4B-MLX-4bit")
    print("loading DS-R1-7B-MLX-4bit...")
    dsr1_m, dsr1_t = load("mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit")

    cells_dir = Path(args.cells_dir)
    results = []
    for cell_path in sorted(cells_dir.glob("p*.json")):
        d = json.loads(cell_path.read_text())
        truth = d.get("truth");
        if truth is None: continue
        cell_id = cell_path.stem.split("_")[0]
        text = d["response"]
        prefix_len = min(int(0.6 * len(text)), args.n_chars_cap)
        prompt = text[:prefix_len] + PREAMBLE
        t0 = time.time()
        qe = forced_emit(qwen_m, qwen_t, prompt)
        t1 = time.time()
        de = forced_emit(dsr1_m, dsr1_t, prompt)
        t2 = time.time()
        qm = re.match(r"(\d+)", qe.strip())
        dm = re.match(r"(\d+)", de.strip())
        qp = int(qm.group(1)) if qm else None
        dp = int(dm.group(1)) if dm else None
        agree = qp == dp
        q_correct = qp == truth
        d_correct = dp == truth
        if agree and q_correct:
            mode = "MODE_1_correct"
        elif agree and not q_correct:
            mode = "MODE_2_collusion"
        elif not agree:
            mode = "MODE_2_disagreement"
        else:
            mode = "mixed"
        print(f"  {cell_id}: truth={truth} qwen={qp} ({t1-t0:.1f}s) dsr1={dp} ({t2-t1:.1f}s) → {mode}")
        results.append({"cell": cell_id, "truth": truth, "qwen_emit": qp,
                          "dsr1_emit": dp, "agree": agree, "q_correct": q_correct,
                          "d_correct": d_correct, "mode": mode})
    n_correct_q = sum(1 for r in results if r["q_correct"])
    n_correct_d = sum(1 for r in results if r["d_correct"])
    n_agree = sum(1 for r in results if r["agree"])
    n_agree_correct = sum(1 for r in results if r["agree"] and r["q_correct"])
    print(f"\nQwen alone: {n_correct_q}/{len(results)}")
    print(f"DS-R1 alone: {n_correct_d}/{len(results)}")
    print(f"Agree on emit: {n_agree}/{len(results)}")
    print(f"Agree AND correct: {n_agree_correct}/{len(results)}")
    Path(args.out).write_text(json.dumps({
        "n_cells": len(results),
        "n_qwen_correct": n_correct_q,
        "n_dsr1_correct": n_correct_d,
        "n_agree": n_agree,
        "n_agree_correct": n_agree_correct,
        "results": results,
    }, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
