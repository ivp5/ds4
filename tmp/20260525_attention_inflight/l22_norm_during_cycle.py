"""L22 norm during cycle vs pre-cycle on Qwen3.5-4B-MLX-4bit.

Hypothesis: if L22 is structural compute-injection peak, norm
explosion is precision/content-invariant. If content-dependent,
norm drops inside the limit cycle (no new info to inject).

Probe: monkey-patch L22's __call__ to capture pre-input + post-output
hidden state norms at every forward. Decode greedy to pos 12000+,
record norms at 5 pre-cycle positions and 5 in-cycle positions.

H_STRUCTURAL: ||delta_L22|| stable across positions (CV < 0.2)
H_CONTENT: ||delta_L22|| collapses in cycle (mean_in_cycle < 0.5 * mean_pre_cycle)
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

# Find L22, the GatedDeltaNet layer at index 22
layers = model.layers
print(f"n_layers = {len(layers)}")
L22 = layers[22]
print(f"L22 type = {type(L22).__name__}")

# Storage for captures
captures = []
current_pos = [0]

# CLASS-LEVEL patch (instance-level doesn't take because MLX Module uses class dispatch)
DecoderLayerClass = type(L22)
orig_call = DecoderLayerClass.__call__
def patched_call(self, x, *args, **kwargs):
    out = orig_call(self, x, *args, **kwargs)
    if self is L22:
        # Compute norms at last sequence position
        try:
            pre_vec = x[0, -1, :] if x.ndim == 3 else x[-1, :]
            post_vec = out[0, -1, :] if out.ndim == 3 else out[-1, :]
            pre_norm = float(mx.linalg.norm(pre_vec).item())
            post_norm = float(mx.linalg.norm(post_vec).item())
            delta_vec_norm = float(mx.linalg.norm(post_vec - pre_vec).item())
            captures.append({"pos": current_pos[0], "pre": pre_norm, "post": post_norm,
                             "post_minus_pre_norm": post_norm - pre_norm,
                             "delta_vec_norm": delta_vec_norm})
        except Exception as e:
            captures.append({"pos": current_pos[0], "error": str(e)})
    return out
DecoderLayerClass.__call__ = patched_call

# Encode
chat = tok.apply_chat_template(
    [{"role":"system","content":SYSTEM},{"role":"user","content":P01}],
    tokenize=False, add_generation_prompt=True)
prompt_ids = tok.encode(chat)
print(f"prompt: {len(prompt_ids)} tokens")

# Run prompt through to warm KV cache; captures here are for prompt positions
cache = make_prompt_cache(model)
out = model(mx.array([prompt_ids]), cache=cache)
mx.eval(out)
print(f"after prompt: {len(captures)} captures (one per prompt-end)")
captures_at_prompt = len(captures)

# Now decode position-by-position; track positions
last = mx.array([[prompt_ids[-1]]])
t1 = time.time()
captured_text_chunks = {}
recent = []
# Capture detailed at positions 2K, 4K, 6K, 8K, 10K, 12K (entry), 12030, 12060, 12090, 12120
capture_positions = set([2000, 4000, 6000, 8000, 10000, 12000, 12031, 12062, 12093, 12120])

# Run decode loop, but only KEEP captures at the target positions
# We'll filter captures list at the end based on current_pos[0] when recorded
for pos in range(12125):
    current_pos[0] = pos
    out = model(last, cache=cache)
    tid = int(mx.argmax(out[0,-1,:]))
    recent.append(tid)
    if len(recent) > 32: recent = recent[-32:]
    last = mx.array([[tid]])
    if pos in capture_positions:
        captured_text_chunks[pos] = tok.decode(recent)
    if pos % 2000 == 0:
        print(f"  pos={pos} t={time.time()-t1:.1f}s recent={tok.decode(recent[-12:])!r}")

print(f"finished decode in {time.time()-t1:.1f}s")
print(f"total captures: {len(captures)}")

# Filter to only the target positions
target_captures = [c for c in captures if c["pos"] in capture_positions]
print(f"\nL22 norm captures at target positions:")
for c in target_captures:
    if "error" in c:
        print(f"  pos={c['pos']:5d}: ERROR {c['error']}")
    else:
        print(f"  pos={c['pos']:5d}: pre={c['pre']:7.2f}  post={c['post']:7.2f}  ||delta||={c['delta_vec_norm']:7.2f}")

# Save full data
out_data = {
    "model": MODEL_PATH,
    "captures": target_captures,
    "captured_text_chunks": captured_text_chunks,
}
from pathlib import Path
Path("tmp/20260525_attention_inflight/l22_norm_during_cycle_result.json").write_text(json.dumps(out_data, indent=2))
print("\nsaved l22_norm_during_cycle_result.json")

# Quick analysis
pre_cycle = [c["delta_vec_norm"] for c in target_captures if c["pos"] < 12000]
in_cycle = [c["delta_vec_norm"] for c in target_captures if c["pos"] >= 12000]
if pre_cycle and in_cycle:
    import statistics
    pc_mean = statistics.mean(pre_cycle)
    ic_mean = statistics.mean(in_cycle)
    pc_stdev = statistics.stdev(pre_cycle) if len(pre_cycle) > 1 else 0
    ic_stdev = statistics.stdev(in_cycle) if len(in_cycle) > 1 else 0
    print(f"\npre-cycle:  delta mean = {pc_mean:+7.2f}  stdev = {pc_stdev:.2f}  CV = {pc_stdev/abs(pc_mean) if pc_mean else 0:.3f}")
    print(f"in-cycle:   delta mean = {ic_mean:+7.2f}  stdev = {ic_stdev:.2f}  CV = {ic_stdev/abs(ic_mean) if ic_mean else 0:.3f}")
    ratio = ic_mean / pc_mean if pc_mean else 0
    print(f"ratio in/pre: {ratio:.3f}")
    if ratio < 0.5:
        print("VERDICT: H_CONTENT survives (in-cycle delta < 50% of pre-cycle)")
    elif abs(ratio - 1.0) < 0.2:
        print("VERDICT: H_STRUCTURAL survives (in-cycle delta within 20% of pre-cycle)")
    else:
        print("VERDICT: intermediate — neither extreme")
