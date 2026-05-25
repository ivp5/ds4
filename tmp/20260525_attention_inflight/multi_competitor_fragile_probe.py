"""Multi-competitor distribution at fragile L31 positions.

H1870 critiques top-2 margin as scalar collapse. Test on my data:
do fragile positions (margin < 1) have MANY near-tied competitors
or just ONE alternative?

Capture top-10 logits at known fragile positions (6000, 7000) and
known safe positions (5000, 12000) for comparison.

H_BIMODAL: fragile positions have 2 competitors near-tied (top-2),
           rest are noise → top-2 margin captures the dynamics
H_MULTIMODAL: fragile positions have 3-5+ competitors near-tied →
              multi-competitor watchlist is structurally necessary
"""
import json, os, time, math
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("ALL_PROXY", "socks5h://127.0.0.1:10001")
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

P01 = ("Patrick started walking at a constant rate along a straight road from school to the park. "
       "One hour after Patrick left, Tanya started running along the same road from school to the park. "
       "One hour after Tanya left, Jose started bicycling along the same road from school to the park. "
       "Tanya ran at a constant rate of $2$ miles per hour faster than Patrick walked, "
       "Jose bicycled at a constant rate of $7$ miles per hour faster than Tanya ran, "
       "and all three arrived at the park at the same time. The distance from the school to the park "
       "is $\\frac{m}{n}$ miles, where $m$ and $n$ are relatively prime positive integers. Find $m + n$.")
SYSTEM = ("You are a careful mathematician. Solve the problem step by step. "
          "At the end, present the final integer answer enclosed in \\boxed{}.")

MODEL_PATH = "mlx-community/Qwen3.5-4B-MLX-4bit"
print(f"loading {MODEL_PATH}...")
t0 = time.time()
model, tok = load(MODEL_PATH)
print(f"loaded in {time.time()-t0:.1f}s")

chat = tok.apply_chat_template(
    [{"role":"system","content":SYSTEM},{"role":"user","content":P01}],
    tokenize=False, add_generation_prompt=True)
prompt_ids = tok.encode(chat)
print(f"prompt: {len(prompt_ids)} tokens")

cache = make_prompt_cache(model)
out = model(mx.array([prompt_ids]), cache=cache)
mx.eval(out)

# Capture top-10 at known fragile + safe positions
capture_positions = {6000, 7000, 5000, 8000, 12000, 12031, 10500, 11000}
last = mx.array([[prompt_ids[-1]]])
t1 = time.time()
position_data = {}
for pos in range(12100):
    out = model(last, cache=cache)
    logits = out[0, -1, :]
    tid = int(mx.argmax(logits).item())
    if pos in capture_positions:
        # Capture top-10
        probs = mx.softmax(logits)
        probs_np = probs.tolist()
        top_idx = sorted(range(len(probs_np)), key=lambda i: -probs_np[i])[:10]
        top_probs = [probs_np[i] for i in top_idx]
        top_log_p = [math.log(p) if p > 0 else -100 for p in top_probs]
        top_tokens = [tok.decode([i]) for i in top_idx]
        position_data[pos] = {
            "top10_tokens": top_tokens,
            "top10_probs": top_probs,
            "top10_log_p": top_log_p,
        }
    last = mx.array([[tid]])
    if pos % 2000 == 0:
        print(f"  pos={pos} t={time.time()-t1:.1f}s")

print(f"\nfinished in {time.time()-t1:.1f}s")

# Print top-10 distribution at each captured position
print(f"\nTop-10 logit distribution per position:")
for pos in sorted(capture_positions):
    d = position_data.get(pos)
    if not d: continue
    print(f"\n  pos={pos}:")
    # Compute multi-competitor stats
    log_ps = d['top10_log_p']
    top1 = log_ps[0]
    top2 = log_ps[1]
    top5 = log_ps[4]
    top10 = log_ps[9]
    margin_top2 = top1 - top2
    margin_top5 = top1 - top5  # gap to 5th
    n_within_2 = sum(1 for lp in log_ps[1:] if (top1 - lp) < 2)  # competitors within 2 log_p
    n_within_5 = sum(1 for lp in log_ps[1:] if (top1 - lp) < 5)
    print(f"    margin(top1-top2)={margin_top2:.3f}, margin(top1-top5)={margin_top5:.3f}")
    print(f"    competitors within 2 log_p of top1: {n_within_2}")
    print(f"    competitors within 5 log_p of top1: {n_within_5}")
    for i, (t, p, lp) in enumerate(zip(d['top10_tokens'], d['top10_probs'], d['top10_log_p'])):
        marker = "  <-- top1" if i == 0 else ""
        print(f"      rank{i+1}: {t!r:>20} p={p:.4f} log_p={lp:7.3f}{marker}")

# H_BIMODAL vs H_MULTIMODAL verdict
print(f"\n--- H_BIMODAL vs H_MULTIMODAL VERDICT ---")
for pos in [6000, 7000]:  # known fragile
    d = position_data.get(pos)
    if d:
        log_ps = d['top10_log_p']
        n_within_1 = sum(1 for lp in log_ps[1:] if (log_ps[0] - lp) < 1)
        if n_within_1 == 1:
            verdict = "H_BIMODAL (only 1 competitor within 1 log_p of top1)"
        elif n_within_1 >= 2:
            verdict = f"H_MULTIMODAL ({n_within_1} competitors within 1 log_p)"
        else:
            verdict = "no competitors within 1 log_p"
        print(f"  pos={pos}: {verdict}")

from pathlib import Path
Path("tmp/20260525_attention_inflight/multi_competitor_fragile_result.json").write_text(json.dumps(position_data, indent=2))
print("\nsaved multi_competitor_fragile_result.json")
