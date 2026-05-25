"""Late-layer ||delta|| ramp probe — what is L31 doing that L22 isn't?

My multi-layer probe found:
  L22:  5.69 (GatedDeltaNet, mid)
  L25:  5.56 (GatedDeltaNet, mid)
  L26:  6.53 (GatedDeltaNet, late-mid)
  L31: 37.62 (full-attn, LAST)

What's the trajectory L26→L27→L28→L29→L30→L31? Two hypotheses:
- H_FINAL_LAYER: L31 specifically dominant (jump from L30 to L31)
- H_LATE_RAMP: ||delta|| ramps up smoothly across L26→L31

Architecture: L27 is full-attn (in {3,7,11,15,19,23,27,31}); L28/29/30 are GatedDeltaNet.

If full-attn layers consistently have higher ||delta||:
  L27 > L26, L28
  L31 > L30
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

# Late layer trajectory + a few early/mid for reference
TARGET_LAYERS = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
target_layer_objs = {i: model.layers[i] for i in TARGET_LAYERS}
print(f"target layers: {TARGET_LAYERS}")
print(f"L31 sub-modules:")
for name in ['self_attn', 'linear_attn', 'mlp']:
    if hasattr(model.layers[31], name):
        print(f"  L31.{name}: {type(getattr(model.layers[31], name)).__name__}")
print(f"L27 sub-modules:")
for name in ['self_attn', 'linear_attn', 'mlp']:
    if hasattr(model.layers[27], name):
        print(f"  L27.{name}: {type(getattr(model.layers[27], name)).__name__}")

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

last = mx.array([[prompt_ids[-1]]])
t1 = time.time()
# Reduced capture points — just need the ramp at key positions
capture_positions = set([4000, 6000, 8000, 12000, 12031])
recent = []
for pos in range(12035):
    current_pos[0] = pos
    out = model(last, cache=cache)
    tid = int(mx.argmax(out[0,-1,:]))
    recent.append(tid)
    if len(recent) > 16: recent = recent[-16:]
    last = mx.array([[tid]])
    if pos % 2000 == 0:
        print(f"  pos={pos} t={time.time()-t1:.1f}s")

print(f"finished decode in {time.time()-t1:.1f}s")

# Print ramp table
print("\n||delta|| late-layer ramp:")
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
    print("  ".join(row))

# Architectural classification
FULL_ATTN = {3, 7, 11, 15, 19, 23, 27, 31}
print("\nArchitecture: layer_type per target")
for i in TARGET_LAYERS:
    lt = "FullAttn" if i in FULL_ATTN else "GatedDelta"
    print(f"  L{i}: {lt}")

# Test H_LATE_RAMP vs H_FINAL_LAYER
print("\nIn-cycle ||delta|| trajectory L22→L31 (at pos 12000):")
in_cycle_ramp = []
for li in TARGET_LAYERS:
    v = target_data.get(12000, {}).get(li, None)
    if v is not None:
        in_cycle_ramp.append((li, v))
        layer_type = "[Attn]" if li in FULL_ATTN else "[SSM]"
        print(f"  L{li} {layer_type}: {v:7.2f}")

# Compute jump from L30 to L31
v30 = target_data.get(12000, {}).get(30)
v31 = target_data.get(12000, {}).get(31)
if v30 and v31:
    jump = v31 / v30
    print(f"\nL30 -> L31 jump factor: {jump:.2f}x")
    print(f"H_FINAL_LAYER predicts jump > 3x")
    print(f"H_LATE_RAMP predicts jump ~ 1-1.5x")
    if jump > 3:
        print("VERDICT: H_FINAL_LAYER survives (L31 specifically dominant)")
    elif jump < 1.5:
        print("VERDICT: H_LATE_RAMP survives (smooth ramp through late layers)")
    else:
        print(f"VERDICT: intermediate")

from pathlib import Path
Path("tmp/20260525_attention_inflight/late_layer_ramp_result.json").write_text(json.dumps({
    "model": MODEL_PATH,
    "captures": {str(k): v for k, v in captures.items()},
    "target_layers": TARGET_LAYERS,
}, indent=2))
print("\nsaved late_layer_ramp_result.json")
