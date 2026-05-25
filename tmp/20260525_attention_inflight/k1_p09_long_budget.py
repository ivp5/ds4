"""K=1 P09 at 20K max_tokens — bracket budget threshold for K-sampling escape.

silv 2026-05-25 22nd continue: 21st probe showed 0/4 at max_tokens=4096
but all samples were still actively reasoning at truncation. matharena
runs Qwen3.5-4B at ~27853 tokens average. Run K=1 at 20K to bracket
the budget threshold.
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

    seed = 20260525
    mx.random.seed(seed)
    sampler = make_sampler(temp=0.7, top_p=1.0)
    t1 = time.time()
    text = generate(
        model, tok,
        prompt=chat,
        max_tokens=20000,
        sampler=sampler,
        verbose=False,
    )
    t_elapsed = time.time() - t1
    n_truth = text.count("29")
    n_chars = len(text)
    first_at = text.find("29") if "29" in text else -1
    has_boxed = "\\boxed{" in text
    boxed_pos = text.find("\\boxed{") if "\\boxed{" in text else -1

    print(f"seed={seed}: chars={n_chars}, '29' count={n_truth}, first_at={first_at}")
    print(f"has_boxed={has_boxed}, boxed_pos={boxed_pos}, wall={t_elapsed:.1f}s")

    out = Path("tmp/20260525_attention_inflight/k1_p09_long.json")
    out.write_text(json.dumps({
        "seed": seed,
        "max_tokens": 20000,
        "n_chars": n_chars,
        "n_truth_occurrences": n_truth,
        "truth_first_at": first_at,
        "has_boxed": has_boxed,
        "boxed_position": boxed_pos,
        "wall_s": t_elapsed,
        "text_tail_500": text[-500:],
        "text_first_2000": text[:2000],
    }, indent=2))
    print(f"saved {out}")
    print(f"\ntext_tail_500:")
    print(text[-500:])

if __name__ == "__main__":
    main()
