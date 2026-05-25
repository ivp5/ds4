"""Multi-layer norm probe at 8bit — extends 4bit finding to test precision invariance.

8bit cycle period = 32 tokens (vs 4bit's 31). Probe captures L0, L4, L22, L25, L31
at positions matching 32-token cycle: 12000, 12032, 12064, 12096.

H_LAYER_INVARIANT: cycle-period determinism (4bit finding) replicates at 8bit
H_PRECISION_VARIANT: 8bit shows different layer-attractor structure
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

MODEL_PATH = "mlx-community/Qwen3.5-4B-MLX-8bit"
print(f"loading {MODEL_PATH}...")
t0 = time.time()
model, tok = load(MODEL_PATH)
print(f"loaded in {time.time()-t0:.1f}s")

TARGET_LAYERS = [0, 4, 22, 25, 31]
target_layer_objs = {i: model.layers[i] for i in TARGET_LAYERS}

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

# 8bit cycle is 32 tokens — capture at 12000, 12032, 12064, 12096
# Also pre-cycle: 4000, 6000, 8000
capture_positions = set([4000, 6000, 8000, 10000, 12000, 12032, 12064, 12096])

last = mx.array([[prompt_ids[-1]]])
t1 = time.time()
recent = []
for pos in range(12100):
    current_pos[0] = pos
    out = model(last, cache=cache)
    tid = int(mx.argmax(out[0,-1,:]))
    recent.append(tid)
    if len(recent) > 32: recent = recent[-32:]
    last = mx.array([[tid]])
    if pos % 2000 == 0:
        print(f"  pos={pos} t={time.time()-t1:.1f}s recent={tok.decode(recent[-10:])!r}")

print(f"finished decode in {time.time()-t1:.1f}s")

# Print summary
print("\n8bit ||delta|| per layer × position (cycle period = 32):")
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

print("\n8bit cycle-period determinism (positions 12000/12032/12064/12096):")
for li in TARGET_LAYERS:
    vals = [target_data.get(p, {}).get(li) for p in [12000, 12032, 12064, 12096]]
    if all(v is not None for v in vals):
        det = "EXACT" if len(set(round(v, 3) for v in vals)) == 1 else f"max_diff={max(vals)-min(vals):.3f}"
        print(f"  L{li:2d}    " + "  ".join(f"{v:7.2f}" for v in vals) + f"  {det}")

from pathlib import Path
Path("tmp/20260525_attention_inflight/multi_layer_8bit_cycle_result.json").write_text(json.dumps({
    "model": MODEL_PATH,
    "captures": {str(k): v for k, v in captures.items()},
    "target_layers": TARGET_LAYERS,
}, indent=2))
print("\nsaved multi_layer_8bit_cycle_result.json")
