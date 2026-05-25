"""Fresh K=4 probe on AIME 2026 P09.

silv 2026-05-25 21st continue: previously matharena claimed Qwen3.5-4B
P9 at 92.6% green; my single cached run had zero '29' occurrences.
Does K=4 at temperature 0.7 produce at least one sample with truth=29 in CoT?

Conservative: max_tokens=4096 per sample (vs matharena's 27853 average).
Tests escape rate at minimal compute budget.
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("ALL_PROXY", "socks5h://127.0.0.1:10001")

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

PROBLEM_P09 = """Joanne has a blank fair six-sided die and six stickers each displaying a different integer from 1 to 6. Joanne rolls the die and then places the sticker labeled 1 on the top face of the die. She then rolls the die again, places the sticker labeled 2 on the top face, and continues this process to place the rest of the stickers in order. If the die ever lands with a sticker already on its top face, the new sticker is placed to cover the old sticker. Let $p$ be the conditional probability that at the end of the process exactly one face has been left blank, given that all the even-numbered stickers are visible on faces of the die. Then $p$ can be written as $\\tfrac mn,$ where $m$ and $n$ are relatively prime positive integers. Find $m + n.$"""

SYSTEM = "You are a careful mathematician. Solve the problem step by step. At the end, present the final integer answer enclosed in \\boxed{}."

def main():
    print(f"loading Qwen3.5-4B-MLX-4bit...")
    t0 = time.time()
    model, tok = load("mlx-community/Qwen3.5-4B-MLX-4bit")
    print(f"loaded in {time.time()-t0:.1f}s")

    chat = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": PROBLEM_P09}],
        tokenize=False,
        add_generation_prompt=True,
    )

    results = []
    for k in range(4):
        seed = 20260525 + k
        mx.random.seed(seed)
        sampler = make_sampler(temp=0.7, top_p=1.0)
        t1 = time.time()
        text = generate(
            model, tok,
            prompt=chat,
            max_tokens=4096,
            sampler=sampler,
            verbose=False,
        )
        t_elapsed = time.time() - t1
        n_truth = text.count("29")
        n_chars = len(text)
        # Where does 29 first appear?
        first_at = text.find("29") if "29" in text else -1
        print(f"  k={k} seed={seed}: chars={n_chars}, '29' count={n_truth}, first_at={first_at}, wall={t_elapsed:.1f}s")
        results.append({
            "k": k,
            "seed": seed,
            "n_chars": n_chars,
            "n_truth_occurrences": n_truth,
            "truth_first_at": first_at,
            "wall_s": t_elapsed,
            "text_tail": text[-300:],
        })

    # Summary
    n_with_truth = sum(1 for r in results if r["n_truth_occurrences"] > 0)
    print(f"\n{n_with_truth}/{len(results)} samples contain '29' in CoT")

    out = Path("tmp/20260525_attention_inflight/k4_p09_result.json")
    out.write_text(json.dumps({
        "n_samples": len(results),
        "n_with_truth": n_with_truth,
        "results": results,
    }, indent=2))
    print(f"saved {out}")

if __name__ == "__main__":
    main()
