"""Cross-precision cycle period probe: bf16 vs 4bit.

Per H1853 codex finding: precision shifts attractor. Test if cycle
period differs between Qwen3.5-4B-MLX-bf16 and 4bit on AIME P01.

Hypothesis: different precision → different cycle content/period.
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

# Try 8bit first (smaller than bf16, still higher precision than 4bit)
MODEL_PATH = "mlx-community/Qwen3.5-4B-MLX-8bit"
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

print("fast-forwarding to pos 12000...")
last = mx.array([[prompt_ids[-1]]])
t1 = time.time()
recent = []
for pos in range(12000):
    out = model(last, cache=cache)
    tid = int(mx.argmax(out[0,-1,:]))
    recent.append(tid)
    if len(recent) > 64: recent = recent[-64:]
    last = mx.array([[tid]])
    if pos % 2000 == 0:
        print(f"  pos={pos} t={time.time()-t1:.1f}s recent={tok.decode(recent[-32:])!r}")
print(f"reached pos 12000 in {time.time()-t1:.1f}s")

# Capture 200 consecutive
print("capturing 200 consecutive tokens from pos 12000...")
captured = []
for i in range(200):
    out = model(last, cache=cache)
    tid = int(mx.argmax(out[0,-1,:]))
    captured.append(tid)
    last = mx.array([[tid]])

print("\nCycle-length hypothesis test:")
for L in [13, 20, 25, 30, 31, 32, 35, 40, 43, 50, 62, 93]:
    matches = sum(1 for i in range(len(captured)-L) if captured[i] == captured[i+L])
    total = len(captured) - L
    pct = 100.0 * matches / total
    marker = "  <<< MATCH" if pct >= 95 else ""
    print(f"  L={L:3d}: {matches}/{total} = {pct:.1f}%{marker}")

print("\nDecoded 100 tokens captured:")
print(repr(tok.decode(captured[:100])))

from pathlib import Path
Path("tmp/20260525_attention_inflight/cycle_period_8bit_result.json").write_text(json.dumps({
    "model": MODEL_PATH,
    "captured_tokens": captured,
    "captured_text": tok.decode(captured),
}, indent=2))
print("\nsaved cycle_period_8bit_result.json")
