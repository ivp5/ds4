"""Cross-architecture chunk-wise entropy: does Qwen's L19 lock signature
replicate at some DS-R1-7B layer when reading the same locked text?

silv 2026-05-25: cross-model confidence on the attention-entropy finding.
"""
import json
from pathlib import Path
import numpy as np


def per_chunk(per_layer: dict, n_tok: int, n_chunks: int = 5) -> dict:
    chunk_size = n_tok // n_chunks
    out = {}
    for L_str, ld in per_layer.items():
        L = int(L_str)
        ent = np.asarray(ld["mean_entropy_per_pos"])
        sm = np.asarray(ld["mean_self_mass_per_pos"])
        out[L] = []
        for k in range(n_chunks):
            lo = k * chunk_size
            hi = (k + 1) * chunk_size if k < n_chunks - 1 else n_tok
            out[L].append({
                "ent_mean": float(ent[lo:hi].mean()),
                "self_mean": float(sm[lo:hi].mean()),
            })
    return out


def main():
    base = Path("tmp/20260525_attention_inflight")
    qwen = json.loads((base / "p01_4bit.json").read_text())
    dsr1 = json.loads((base / "dsr1_7b_4bit_qwen_p01.json").read_text())

    qwen_chunks = per_chunk(qwen["per_layer"], qwen["n_tokens"])
    dsr1_chunks = per_chunk(dsr1["per_layer"], dsr1["n_tokens"])

    print("Qwen3.5-4B (P01 4bit, locked) — entropy per chunk:\n")
    for L in [3, 7, 11, 15, 19, 23, 27, 31]:
        if L not in qwen_chunks: continue
        ents = [c["ent_mean"] for c in qwen_chunks[L]]
        print(f"  L{L:>2}  " + "  ".join(f"{e:.3f}" for e in ents))

    print("\nDS-R1-7B (reading Qwen P01 4bit text) — entropy per chunk:\n")
    for L in [0, 3, 7, 11, 15, 19, 23, 27]:
        if L not in dsr1_chunks: continue
        ents = [c["ent_mean"] for c in dsr1_chunks[L]]
        print(f"  L{L:>2}  " + "  ".join(f"{e:.3f}" for e in ents))

    # Find each model's MINIMUM-entropy chunk and compare to chunk 2 (where Qwen lock fires)
    print("\nQwen3.5-4B per-layer chunk-2 vs chunk-0 (lock vs prompt):")
    for L in [3, 7, 11, 15, 19, 23, 27, 31]:
        if L not in qwen_chunks: continue
        c0 = qwen_chunks[L][0]["ent_mean"]
        c2 = qwen_chunks[L][2]["ent_mean"]
        delta = c2 - c0
        print(f"  L{L:>2}: chunk_0={c0:.3f}, chunk_2={c2:.3f}, delta={delta:+.3f}")

    print("\nDS-R1-7B per-layer chunk-2 vs chunk-0 (reading the locked text):")
    for L in range(0, 28):
        if L not in dsr1_chunks: continue
        c0 = dsr1_chunks[L][0]["ent_mean"]
        c2 = dsr1_chunks[L][2]["ent_mean"]
        delta = c2 - c0
        marker = "  <-- entropy DROPS in lock-region of input" if delta < -0.1 else ""
        if L % 4 == 0 or marker:
            print(f"  L{L:>2}: chunk_0={c0:.3f}, chunk_2={c2:.3f}, delta={delta:+.3f}{marker}")


if __name__ == "__main__":
    main()
