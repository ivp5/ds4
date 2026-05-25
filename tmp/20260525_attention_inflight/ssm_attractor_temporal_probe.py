"""SSM attractor temporal dynamics probe.

silv 2026-05-25: through-time analysis to nearest coherent state.

Run Qwen3.5-4B-MLX-4bit on AIME P01 (the cell that's documented to
enter 1801-line repetition at 65K tokens). Capture hidden state at
sampled positions during the attractor entry. Compute pairwise L2
distance between states at distant positions.

Hypothesis: if state converges to identical values, fixed-point
attractor. If cycling in low-dim manifold, limit cycle.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("ALL_PROXY", "socks5h://127.0.0.1:10001")

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

# AIME 2026 P01 — known to enter 1801-line attractor on Qwen3.5-4B-4bit
P01 = ("Patrick started walking at a constant rate along a straight road from school to the park. "
       "One hour after Patrick left, Tanya started running along the same road from school to the park. "
       "One hour after Tanya left, Jose started bicycling along the same road from school to the park. "
       "Tanya ran at a constant rate of $2$ miles per hour faster than Patrick walked, "
       "Jose bicycled at a constant rate of $7$ miles per hour faster than Tanya ran, "
       "and all three arrived at the park at the same time. The distance from the school to the park "
       "is $\\frac{m}{n}$ miles, where $m$ and $n$ are relatively prime positive integers. Find $m + n$.")

SYSTEM = ("You are a careful mathematician. Solve the problem step by step. "
          "At the end, present the final integer answer enclosed in \\boxed{}.")


def main():
    print("loading model...")
    t0 = time.time()
    model, tok = load("mlx-community/Qwen3.5-4B-MLX-4bit")
    print(f"loaded in {time.time()-t0:.1f}s")

    chat = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": P01}],
        tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tok.encode(chat)
    print(f"prompt tokens: {len(prompt_ids)}")

    # Custom generation loop with hidden-state capture
    cache = make_prompt_cache(model)
    # Run prefill
    t1 = time.time()
    out = model(mx.array([prompt_ids]), cache=cache)
    mx.eval(out)
    print(f"prefill: {time.time()-t1:.1f}s")

    # Get last hidden state from lm_head input — we need to reach in for that
    # Use logits' top-1 as proxy for "what state produces", track over time.
    # Capture: per-token top-1 token + top-1 logit value + entropy

    SAMPLE_POSITIONS = [500, 1000, 2000, 4000, 8000, 12000, 15000]
    MAX_TOKENS = 16000

    last_token = mx.array([[prompt_ids[-1]]])
    # Initial logits from prefill's last position
    state_snapshots = []  # list of (pos, top1_id, top1_token, top1_logit, top2_logit, top1_prob, recent_tokens)

    recent_tokens = []  # ring buffer of last 16 token ids for repetition detection
    t_gen = time.time()
    last_print = t_gen

    for pos in range(MAX_TOKENS):
        # Run one step
        out = model(last_token, cache=cache)
        logits = out[0, -1, :]

        # Capture stats
        probs = mx.softmax(logits)
        sorted_idx = mx.argsort(-logits)
        top1_id = int(sorted_idx[0])
        top1_logit = float(logits[top1_id])
        top2_logit = float(logits[int(sorted_idx[1])])
        top1_prob = float(probs[top1_id])

        next_token = top1_id
        recent_tokens.append(next_token)
        if len(recent_tokens) > 64:
            recent_tokens = recent_tokens[-64:]

        # Check if at a sample position
        if pos in SAMPLE_POSITIONS or pos == MAX_TOKENS - 1:
            top1_tok_str = tok.decode([top1_id])
            snap = {
                "pos": pos,
                "top1_id": top1_id,
                "top1_token": top1_tok_str,
                "top1_logit": top1_logit,
                "top2_logit": top2_logit,
                "top1_prob": top1_prob,
                "margin": top1_logit - top2_logit,
                "recent_tokens": list(recent_tokens),
                "recent_text": tok.decode(recent_tokens),
            }
            state_snapshots.append(snap)
            print(f"pos={pos}: top1={top1_id}/'{top1_tok_str}' logit={top1_logit:.2f} margin={top1_logit-top2_logit:.2f} p={top1_prob:.3f}")
            print(f"  recent_text: {snap['recent_text'][-150:]!r}")

        # Progress every 60s
        now = time.time()
        if now - last_print > 60:
            print(f"  [progress] pos={pos}, wall={now-t_gen:.1f}s, {pos/(now-t_gen):.1f} t/s")
            last_print = now

        last_token = mx.array([[next_token]])

    total_wall = time.time() - t_gen
    print(f"\nDone. {MAX_TOKENS} tokens in {total_wall:.1f}s = {MAX_TOKENS/total_wall:.1f} t/s")

    out_path = Path("tmp/20260525_attention_inflight/ssm_attractor_temporal_result.json")
    out_path.write_text(json.dumps({
        "model": "mlx-community/Qwen3.5-4B-MLX-4bit",
        "prompt": "AIME 2026 P01",
        "max_tokens": MAX_TOKENS,
        "wall_s": total_wall,
        "tokens_per_sec": MAX_TOKENS/total_wall,
        "snapshots": state_snapshots,
    }, indent=2))
    print(f"saved {out_path}")

    # Cross-position analysis
    print("\n=== Cross-position recent-token comparison ===")
    for i in range(len(state_snapshots)):
        for j in range(i+1, len(state_snapshots)):
            si = state_snapshots[i]
            sj = state_snapshots[j]
            common = sum(1 for a, b in zip(si["recent_tokens"][-32:], sj["recent_tokens"][-32:]) if a == b)
            print(f"  pos {si['pos']} ↔ pos {sj['pos']}: last-32 tokens match {common}/32")


if __name__ == "__main__":
    main()
