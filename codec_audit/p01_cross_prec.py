"""P01 cross-precision attention chunk-2 analysis.

silv 2026-05-25: verify L19 chunk-2 low-entropy is lock-specific.

P01 locks in 4bit (most_pers=0.852) but NOT in 8bit (0.111) or bf16 (0.294).
If the lock causes the L19 chunk-2 entropy drop:
  - 4bit P01 chunk-2 L19 entropy < 8bit P01 chunk-2 L19 entropy
  - 4bit P01 chunk-2 L19 entropy < bf16 P01 chunk-2 L19 entropy

If the entropy drop is prompt-property (just "this prompt is hard"):
  - all three precisions show same chunk-2 L19 entropy
"""
import json
import sys
from pathlib import Path

import numpy as np


def per_chunk_l19(data: dict, n_chunks: int = 5) -> list:
    n = data["n_tokens"]
    chunk_size = n // n_chunks
    ent = np.asarray(data["per_layer"]["19"]["mean_entropy_per_pos"])
    out = []
    for k in range(n_chunks):
        lo = k * chunk_size
        hi = (k + 1) * chunk_size if k < n_chunks - 1 else n
        out.append({
            "chunk": k,
            "range": [lo, hi],
            "ent_mean": float(ent[lo:hi].mean()),
            "ent_std": float(ent[lo:hi].std()),
        })
    return out


def main():
    base = Path("tmp/20260525_attention_inflight")
    files = {
        "4bit": base / "p01_4bit.json",
        "8bit": base / "p01_8bit.json",
        "bf16": base / "p01_bf16.json",
    }
    print("P01 cross-precision L19 entropy per chunk:\n")
    print(f"{'chunk':>5}  {'4bit':>10} {'8bit':>10} {'bf16':>10}  "
          f"{'4-8':>8} {'4-bf16':>8}")
    chunks = {p: per_chunk_l19(json.loads(fp.read_text())) for p, fp in files.items()}
    for k in range(5):
        e4 = chunks["4bit"][k]["ent_mean"]
        e8 = chunks["8bit"][k]["ent_mean"]
        ebf = chunks["bf16"][k]["ent_mean"]
        print(f"  {k:>3}  {e4:>10.4f} {e8:>10.4f} {ebf:>10.4f}  "
              f"{e4-e8:>+8.4f} {e4-ebf:>+8.4f}")


if __name__ == "__main__":
    main()
