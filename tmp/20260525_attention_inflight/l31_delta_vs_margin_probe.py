"""L31 ||delta|| vs softmax margin band integration probe.

Tests H1868's prediction: do high-||delta|| positions correlate with
LOW source-margin bands at the output side?

Captures at each position:
- L31 ||delta|| (via class-level patch on DecoderLayer)
- L_final top1/top2 log_p and margin (top1 - top2)
- top1 token + top2 token

If correlation: my high-compute positions ARE in fragile output bands,
matching codex's H1868 framework.

H_HIGH_DELTA_LOW_MARGIN: corr(||delta||_L31, top1_margin) < -0.3
H_INDEPENDENT: corr < |0.1|
"""
import json, os, time
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

L31 = model.layers[31]
captures = []
current_pos = [0]

DecoderLayerClass = type(model.layers[0])
orig_call = DecoderLayerClass.__call__
def patched_call(self, x, *args, **kwargs):
    out = orig_call(self, x, *args, **kwargs)
    if self is L31:
        try:
            pre_vec = x[0, -1, :] if x.ndim == 3 else x[-1, :]
            post_vec = out[0, -1, :] if out.ndim == 3 else out[-1, :]
            delta_vec_norm = float(mx.linalg.norm(post_vec - pre_vec).item())
            captures.append({"pos": current_pos[0], "L31_delta": delta_vec_norm})
        except Exception:
            pass
    return out
DecoderLayerClass.__call__ = patched_call

chat = tok.apply_chat_template(
    [{"role":"system","content":SYSTEM},{"role":"user","content":P01}],
    tokenize=False, add_generation_prompt=True)
prompt_ids = tok.encode(chat)
print(f"prompt: {len(prompt_ids)} tokens")

cache = make_prompt_cache(model)
out = model(mx.array([prompt_ids]), cache=cache)
mx.eval(out)

# Sample positions more densely to get a richer correlation
capture_positions = set(range(2000, 12100, 500))  # every 500 tokens, 20 samples
last = mx.array([[prompt_ids[-1]]])
t1 = time.time()

# For each position, capture logits at L_final too
position_logits = {}
for pos in range(12100):
    current_pos[0] = pos
    out = model(last, cache=cache)
    logits = out[0, -1, :]  # (vocab,)
    tid = int(mx.argmax(logits).item())
    if pos in capture_positions:
        # Capture top-2 softmax probabilities
        logits_np = mx.softmax(logits).tolist()  # may be expensive; only at sampled positions
        # Get top-2 indices and log_probs
        top_idx = sorted(range(len(logits_np)), key=lambda i: -logits_np[i])[:5]
        top_probs = [logits_np[i] for i in top_idx]
        import math
        top_log_p = [math.log(p) if p > 0 else -100 for p in top_probs]
        margin = top_log_p[0] - top_log_p[1]
        position_logits[pos] = {
            "top1_id": top_idx[0], "top1_token": tok.decode([top_idx[0]]),
            "top1_p": top_probs[0], "top1_log_p": top_log_p[0],
            "top2_id": top_idx[1], "top2_token": tok.decode([top_idx[1]]),
            "top2_p": top_probs[1], "top2_log_p": top_log_p[1],
            "margin_log_p": margin,
        }
    last = mx.array([[tid]])
    if pos % 2000 == 0:
        print(f"  pos={pos} t={time.time()-t1:.1f}s")

print(f"finished in {time.time()-t1:.1f}s")

# Build correlated data
combined = []
for c in captures:
    pos = c["pos"]
    if pos in position_logits:
        row = {**c, **position_logits[pos]}
        combined.append(row)

print(f"\nCombined L31 ||delta|| × output margin (n={len(combined)} positions):")
print(f"{'pos':>6}  {'L31_delta':>10}  {'top1':>20}  {'top1_p':>8}  {'margin':>8}  {'top2':>15}")
for c in combined:
    print(f"  {c['pos']:5d}  {c['L31_delta']:10.2f}  {c['top1_token']!r:>20}  {c['top1_p']:8.4f}  {c['margin_log_p']:8.3f}  {c['top2_token']!r:>15}")

# Pearson correlation
if len(combined) > 3:
    import statistics
    deltas = [c["L31_delta"] for c in combined]
    margins = [c["margin_log_p"] for c in combined]
    n = len(deltas)
    mean_d, mean_m = statistics.mean(deltas), statistics.mean(margins)
    sd_d, sd_m = statistics.stdev(deltas), statistics.stdev(margins)
    cov = sum((d-mean_d)*(m-mean_m) for d, m in zip(deltas, margins)) / (n-1)
    corr = cov / (sd_d * sd_m) if (sd_d * sd_m) > 0 else 0
    print(f"\nL31 ||delta|| stats: mean={mean_d:.2f}, stdev={sd_d:.2f}")
    print(f"Margin (log_p) stats: mean={mean_m:.2f}, stdev={sd_m:.2f}")
    print(f"Pearson correlation: {corr:.3f}")
    if corr < -0.3:
        print(f"VERDICT: H_HIGH_DELTA_LOW_MARGIN survives (negative correlation)")
    elif abs(corr) < 0.1:
        print(f"VERDICT: H_INDEPENDENT survives (no significant correlation)")
    else:
        print(f"VERDICT: intermediate")

from pathlib import Path
Path("tmp/20260525_attention_inflight/l31_delta_vs_margin_result.json").write_text(json.dumps({
    "model": MODEL_PATH,
    "combined": combined,
}, indent=2))
print("\nsaved l31_delta_vs_margin_result.json")
