"""Multi-layer norm-during-cycle probe — extend L22 finding across layers.

Test: is cycle-period-determinism (exact ||delta|| repeat at 31-token
offsets) L22-specific or does it appear at multiple layers?

Captures L0, L4, L10, L22, L25, L26, L31 at same positions as L22 probe.
Same probe architecture (class-level patch on DecoderLayer).

H_LAYER_SPECIFIC: only L22 shows exact cycle-determinism
H_DISTRIBUTED: multiple layers show determinism (attractor is global, not L22-local)
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

TARGET_LAYERS = [0, 4, 10, 22, 25, 26, 31]
target_layer_objs = {i: model.layers[i] for i in TARGET_LAYERS}
for i in TARGET_LAYERS:
    print(f"L{i}: {type(target_layer_objs[i]).__name__}")

captures = {i: [] for i in TARGET_LAYERS}
current_pos = [0]

DecoderLayerClass = type(model.layers[0])
orig_call = DecoderLayerClass.__call__
def patched_call(self, x, *args, **kwargs):
    out = orig_call(self, x, *args, **kwargs)
    for li, lobj in target_layer_objs.items():
        if self is lobj:
            try:
                pre_vec = x[0, -1, :] if x.ndim == 3 else x[-1, :]
                post_vec = out[0, -1, :] if out.ndim == 3 else out[-1, :]
                pre_norm = float(mx.linalg.norm(pre_vec).item())
                post_norm = float(mx.linalg.norm(post_vec).item())
                delta_vec_norm = float(mx.linalg.norm(post_vec - pre_vec).item())
                captures[li].append({"pos": current_pos[0], "pre": pre_norm, "post": post_norm,
                                     "delta_vec_norm": delta_vec_norm})
            except Exception as e:
                captures[li].append({"pos": current_pos[0], "error": str(e)})
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

last = mx.array([[prompt_ids[-1]]])
t1 = time.time()
capture_positions = set([2000, 4000, 6000, 8000, 10000, 12000, 12031, 12062, 12093, 12120])
recent = []
for pos in range(12125):
    current_pos[0] = pos
    out = model(last, cache=cache)
    tid = int(mx.argmax(out[0,-1,:]))
    recent.append(tid)
    if len(recent) > 32: recent = recent[-32:]
    last = mx.array([[tid]])
    if pos % 2000 == 0:
        print(f"  pos={pos} t={time.time()-t1:.1f}s recent={tok.decode(recent[-10:])!r}")

print(f"finished decode in {time.time()-t1:.1f}s")

# Print summary table
print("\n||delta|| per layer × position:")
print(f"{'pos':>6}  " + "  ".join(f"L{i:>2}" for i in TARGET_LAYERS))
target_data = {}
for pos in sorted(capture_positions):
    row = [f"{pos:6d}"]
    target_data[pos] = {}
    for li in TARGET_LAYERS:
        caps_at_pos = [c for c in captures[li] if c["pos"] == pos]
        if caps_at_pos and "delta_vec_norm" in caps_at_pos[0]:
            d = caps_at_pos[0]["delta_vec_norm"]
            row.append(f"{d:5.2f}")
            target_data[pos][li] = d
        else:
            row.append("  N/A")
    print("  ".join(row))

# Check cycle-determinism per layer
print("\nCycle-period determinism per layer (positions 12000/12031/12062/12093):")
print(f"{'layer':>6}  {'12000':>7}  {'12031':>7}  {'12062':>7}  {'12093':>7}  determinism")
for li in TARGET_LAYERS:
    vals = []
    for p in [12000, 12031, 12062, 12093]:
        v = target_data.get(p, {}).get(li)
        vals.append(v)
    if all(v is not None for v in vals):
        # determinism = all equal?
        det = "EXACT" if len(set(round(v, 3) for v in vals)) == 1 else f"max_diff={max(vals)-min(vals):.3f}"
        print(f"  L{li:2d}    " + "  ".join(f"{v:7.2f}" for v in vals) + f"  {det}")
    else:
        print(f"  L{li:2d}    MISSING DATA")

from pathlib import Path
out_data = {"model": MODEL_PATH, "captures": {str(k): v for k, v in captures.items()}, "target_layers": TARGET_LAYERS}
Path("tmp/20260525_attention_inflight/multi_layer_norm_during_cycle_result.json").write_text(json.dumps(out_data, indent=2))
print("\nsaved multi_layer_norm_during_cycle_result.json")
