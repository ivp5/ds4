"""P09 cell-invariance test for L22/L31 substrate finding.

My P01 multi-layer probe found:
  - L22 ||delta|| ~5.69 in cycle
  - L31 ||delta|| ~37.5 in cycle
  - Cycle period 31 tokens at all layers
  - L31 dominant, full-attn layers > SSM layers

Does P09 (universal-fail cell, truth=29) exhibit the same structure?

H_CELL_INVARIANT: P09 shows cycle-period determinism + L31 dominance
H_CELL_SPECIFIC: P09 has different attractor or no cycle
"""
import json, os, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("ALL_PROXY", "socks5h://127.0.0.1:10001")
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

# Load canonical P09
with open('/tmp/p09_data.json') as f:
    p09 = json.load(f)
P09 = p09['problem']
P09_TRUTH = p09['truth']
print(f"P09 truth: {P09_TRUTH}, problem len: {len(P09)} chars")

SYSTEM = ("You are a careful mathematician. Solve the problem step by step. "
          "At the end, present the final integer answer in the form \\boxed{N} where N is the integer answer between 0 and 999.")

MODEL_PATH = "mlx-community/Qwen3.5-4B-MLX-4bit"
print(f"loading {MODEL_PATH}...")
t0 = time.time()
model, tok = load(MODEL_PATH)
print(f"loaded in {time.time()-t0:.1f}s")

TARGET_LAYERS = [0, 22, 25, 26, 27, 31]
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
                delta_vec_norm = float(mx.linalg.norm(post_vec - pre_vec).item())
                captures[li].append({"pos": current_pos[0], "delta_vec_norm": delta_vec_norm})
            except Exception:
                pass
    return out
DecoderLayerClass.__call__ = patched_call

chat = tok.apply_chat_template(
    [{"role":"system","content":SYSTEM},{"role":"user","content":P09}],
    tokenize=False, add_generation_prompt=True)
prompt_ids = tok.encode(chat)
print(f"prompt: {len(prompt_ids)} tokens")

cache = make_prompt_cache(model)
out = model(mx.array([prompt_ids]), cache=cache)
mx.eval(out)

last = mx.array([[prompt_ids[-1]]])
t1 = time.time()
# Capture at pre-cycle + multiple cycle offsets to test for determinism
# Don't know P09's cycle period a priori — test multiple
capture_positions = set([2000, 4000, 6000, 8000, 10000, 12000, 12031, 12032, 12062, 12064, 12093])
recent = []
for pos in range(12100):
    current_pos[0] = pos
    out = model(last, cache=cache)
    tid = int(mx.argmax(out[0,-1,:]))
    recent.append(tid)
    if len(recent) > 32: recent = recent[-32:]
    last = mx.array([[tid]])
    if pos % 2000 == 0:
        print(f"  pos={pos} t={time.time()-t1:.1f}s recent={tok.decode(recent[-12:])!r}")

print(f"finished in {time.time()-t1:.1f}s")

# Print result table
print(f"\nP09 ||delta|| per layer × position:")
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

# Test cycle-period determinism at L31 across both 31 and 32 offsets
print("\nP09 cycle-period determinism candidates at L31:")
for offset in [31, 32]:
    base = target_data.get(12000, {}).get(31)
    off = target_data.get(12000 + offset, {}).get(31)
    if base and off:
        det = "EXACT" if abs(base - off) < 0.01 else f"diff={abs(base-off):.3f}"
        print(f"  L31 12000 vs 12000+{offset}: base={base:.2f}, off={off:.2f}  {det}")

# Compare to P01 numbers
print(f"\nP01 baseline (from prior probes) for comparison:")
print(f"  L22 in-cycle: 5.69, L25: 5.56, L26: 6.53, L27: 19.25, L31: 37.50")
print(f"P09 measured at pos 12000:")
for li in TARGET_LAYERS:
    v = target_data.get(12000, {}).get(li)
    if v is not None:
        print(f"  L{li}: {v:.2f}")

from pathlib import Path
Path("tmp/20260525_attention_inflight/p09_cell_invariance_result.json").write_text(json.dumps({
    "model": MODEL_PATH,
    "captures": {str(k): v for k, v in captures.items()},
    "target_layers": TARGET_LAYERS,
}, indent=2))
print("\nsaved p09_cell_invariance_result.json")
