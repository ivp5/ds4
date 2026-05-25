"""Locate EXACT layer of MLX↔HF divergence.

silv 2026-05-25: OOM-higher accuracy. The MLX-vs-HF rescue gap (7/10 vs 5/10)
originates somewhere in autoregressive multi-token continuation. Pinpoint
the layer.

Setup:
  - Prompt: prefix + preamble + '6' (the truth d1 that all 4 runtimes agree on)
  - Run forward in MLX 4bit, MLX bf16, HF bf16
  - Capture per-layer hidden state at the LAST position (which predicts d2)
  - Cosine similarity between MLX-bf16 and HF-bf16 per layer
  - Cosine similarity between MLX-4bit and MLX-bf16 per layer
  - Layer where cos drops sharpest = divergence locus

MLX picks d2='2' (truth=62). HF picks d2='0' (giving 60).
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
proxy_port = random.choice(range(10001, 10150))
os.environ.setdefault("HTTPS_PROXY", f"socks5h://127.0.0.1:{proxy_port}")
os.environ.setdefault("ALL_PROXY", f"socks5h://127.0.0.1:{proxy_port}")

import numpy as np


def capture_per_layer_mlx(model_id: str, prompt: str) -> tuple[list[np.ndarray], dict]:
    """Return [n_layers] of (1, 1, hidden) hidden states at LAST position, +
    final logits top tokens."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, tok = load(model_id)
    internal = model.language_model.model
    layers = internal.layers
    n_layers = len(layers)
    captured = {}

    DecCls = type(layers[0])
    orig_call = DecCls.__call__
    layer_idx_by_id = {id(layers[L]): L for L in range(n_layers)}

    def patched(self, x, mask=None, cache=None):
        out = orig_call(self, x, mask, cache)
        L_idx = layer_idx_by_id.get(id(self))
        if L_idx is None:
            return out
        last = out[:, -1:, :]
        mx.eval(last)
        captured[L_idx] = np.array(last.astype(mx.float32), copy=True)
        return out

    DecCls.__call__ = patched
    ids = tok.encode(prompt)
    cache = make_prompt_cache(model)
    out = model(mx.array([ids]), cache=cache)
    mx.eval(*[c.state for c in cache if hasattr(c, "state")])
    DecCls.__call__ = orig_call

    # Final logits
    logits = out[0, -1, :].astype(mx.float32)
    mx.eval(logits)
    np_lg = np.array(logits, copy=True)
    top_idx = np.argpartition(-np_lg, 5)[:5]
    top_idx = sorted(top_idx, key=lambda i: -np_lg[i])
    top_info = [(int(i), tok.decode([int(i)]), float(np_lg[i])) for i in top_idx]

    layer_hidden = [captured[L][0, 0] for L in range(n_layers)]  # list of (hidden,)
    return layer_hidden, {"top": top_info, "n_layers": n_layers, "n_tokens": len(ids)}


def capture_per_layer_hf_via_ssh(prompt: str, model_id: str, host: str = "gpu_ssh@10.10.0.17") -> tuple[list[np.ndarray], dict]:
    """Run HF forward on AMD via ssh, return same shape data."""
    import subprocess
    # Send prompt + script to AMD
    script = f'''
import json, sys
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

prompt = open("c:/LLMTEST/runtime_div_prompt.txt", encoding="utf-8").read()
print(f"prompt loaded: {{len(prompt)}} chars", file=sys.stderr)
tok = AutoTokenizer.from_pretrained("{model_id}", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("{model_id}", dtype=torch.bfloat16,
    device_map="cuda:0", trust_remote_code=True)
model.eval()

ids = tok.encode(prompt, return_tensors="pt").to("cuda:0")
with torch.no_grad():
    out = model(ids, output_hidden_states=True, return_dict=True)
hs = out.hidden_states  # tuple of n_layers+1 tensors
# Skip embedding output (index 0), take post-layer states
last_pos_per_layer = [h[0, -1, :].float().detach().cpu().numpy() for h in hs[1:]]
# Final logits
logits = out.logits[0, -1, :].float().detach().cpu().numpy()
top_idx = np.argpartition(-logits, 5)[:5]
top_idx = sorted(top_idx, key=lambda i: -logits[i])
top_info = [(int(i), tok.decode([int(i)]), float(logits[i])) for i in top_idx]

np.savez("c:/LLMTEST/runtime_div_hf.npz",
         hidden=np.stack(last_pos_per_layer),
         top_ids=np.array([t[0] for t in top_info]),
         top_tokens=np.array([t[1] for t in top_info]),
         top_logits=np.array([t[2] for t in top_info]))
print(f"saved {{len(last_pos_per_layer)}} layers", file=sys.stderr)
'''
    # Write prompt and script
    Path("/tmp/runtime_div_prompt.txt").write_text(prompt)
    Path("/tmp/runtime_div_hf.py").write_text(script)
    subprocess.run(["scp", "/tmp/runtime_div_prompt.txt",
                     f"{host}:c:/LLMTEST/runtime_div_prompt.txt"], check=True)
    subprocess.run(["scp", "/tmp/runtime_div_hf.py",
                     f"{host}:c:/LLMTEST/runtime_div_hf.py"], check=True)
    print(f"  running on AMD...")
    t0 = time.time()
    result = subprocess.run(["ssh", host,
                              "cd c:/LLMTEST && set PYTHONIOENCODING=utf-8 && c:/LLMTEST/python312/python.exe -u runtime_div_hf.py"],
                              capture_output=True, text=True, timeout=300)
    print(f"  AMD elapsed: {time.time()-t0:.1f}s")
    print(f"  stderr: {result.stderr[-500:]}")
    # Pull result back
    subprocess.run(["scp", f"{host}:c:/LLMTEST/runtime_div_hf.npz",
                     "/tmp/runtime_div_hf.npz"], check=True)
    data = np.load("/tmp/runtime_div_hf.npz")
    hidden = [data["hidden"][L] for L in range(data["hidden"].shape[0])]
    top_info = [(int(data["top_ids"][i]), str(data["top_tokens"][i]),
                  float(data["top_logits"][i])) for i in range(5)]
    return hidden, {"top": top_info, "n_layers": len(hidden)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text-from", required=True)
    p.add_argument("--prefix-frac", type=float, default=0.60)
    p.add_argument("--n-chars-cap", type=int, default=16000)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    d = json.loads(Path(args.text_from).read_text())
    text = d["response"]
    prefix_len = min(int(args.prefix_frac * len(text)), args.n_chars_cap)
    preamble = "\n\nAfter careful analysis, the final answer is \\boxed{"
    # Append '6' which is the agreed-on first digit for P02 (truth=62)
    prompt = text[:prefix_len] + preamble + "6"
    print(f"  prompt: {len(prompt)} chars (prefix={prefix_len} + preamble + '6')")

    print(f"\n[MLX bf16]")
    mlx_h, mlx_info = capture_per_layer_mlx("mlx-community/Qwen3.5-4B-MLX-bf16", prompt)
    print(f"  top-5 next-token logits:")
    for tid, tok_s, lg in mlx_info["top"]:
        print(f"    id={tid} token={tok_s!r} logit={lg:.4f}")

    print(f"\n[HF bf16 on AMD]")
    hf_h, hf_info = capture_per_layer_hf_via_ssh(prompt, "Qwen/Qwen3.5-4B")
    print(f"  top-5 next-token logits:")
    for tid, tok_s, lg in hf_info["top"]:
        print(f"    id={tid} token={tok_s!r} logit={lg:.4f}")

    # Per-layer cosine similarity
    print(f"\nPer-layer cosine(MLX bf16, HF bf16) at last position:")
    n_layers = min(len(mlx_h), len(hf_h))
    cosines = []
    for L in range(n_layers):
        a = mlx_h[L].astype(np.float64)
        b = hf_h[L].astype(np.float64)
        if a.shape != b.shape:
            print(f"  L{L}: shape mismatch a={a.shape} b={b.shape}")
            continue
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        cosines.append((L, cos, float(np.linalg.norm(a)), float(np.linalg.norm(b))))
    for L, cos, na, nb in cosines:
        marker = " <-- DIVERGENCE" if cos < 0.99 else ""
        print(f"  L{L:>2}: cos={cos:.6f}  ||MLX||={na:.2f} ||HF||={nb:.2f}{marker}")

    # First layer where cos drops below 0.99
    diverge_L = next((L for L, c, *_ in cosines if c < 0.99), None)
    print(f"\nFirst divergence layer: L{diverge_L}")
    Path(args.out).write_text(json.dumps({
        "mlx_top": mlx_info["top"],
        "hf_top": hf_info["top"],
        "per_layer_cosine": [(L, c, na, nb) for L, c, na, nb in cosines],
        "diverge_layer": diverge_L,
    }, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
