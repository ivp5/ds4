"""Attention statistics in late-region (locked-vs-success contrast).

silv 2026-05-25: dive deeper — global L31 self-mass is identical across
locked/success (0.22-0.22, 18.5%/18.2%/18.6%). The lock signature must be
in the late-region tail, not the global aggregate.

Method: for each cell's saved per-position arrays, bin into 5 chunks
(0-20%, 20-40%, ..., 80-100%) and compare attention entropy + self-mass
between locked tail (P01 last chunk) and success tail (P02/P05 last chunk).
"""
import json
import sys
from pathlib import Path

import numpy as np


def analyze(json_path: Path, label: str) -> dict:
    data = json.loads(json_path.read_text())
    n_tok = data["n_tokens"]
    per_layer = data["per_layer"]
    # For each layer, get mean_entropy_per_pos and mean_self_mass_per_pos
    n_chunks = 5
    chunk = n_tok // n_chunks
    out = {"label": label, "n_tokens": n_tok, "chunks": {}}
    for L_str, layer_data in per_layer.items():
        L = int(L_str)
        ent = np.asarray(layer_data["mean_entropy_per_pos"])
        sm = np.asarray(layer_data["mean_self_mass_per_pos"])
        out["chunks"][L] = []
        for k in range(n_chunks):
            lo = k * chunk
            hi = (k + 1) * chunk if k < n_chunks - 1 else n_tok
            out["chunks"][L].append({
                "chunk": k,
                "range": [lo, hi],
                "ent_mean": float(ent[lo:hi].mean()),
                "ent_min": float(ent[lo:hi].min()),
                "self_mean": float(sm[lo:hi].mean()),
                "self_max": float(sm[lo:hi].max()),
            })
    return out


def main():
    base = Path("tmp/20260525_attention_inflight")
    if len(sys.argv) >= 2 and sys.argv[1] == "summary":
        # Compare P01 vs P02 vs P05 late chunk for each layer
        results = {}
        for cell in ["p01", "p02", "p05"]:
            jp = base / f"{cell}_4bit.json"
            if jp.exists():
                results[cell] = analyze(jp, cell)
        # Pretty-print the LAST chunk
        print("Last 20% (commit / lock region):\n")
        print(f"{'L':>3}  ", end="")
        for cell in ["p01", "p02", "p05"]:
            print(f"{cell+'_ent':>10} {cell+'_self':>10}  ", end="")
        print()
        for L_idx in [3, 7, 11, 15, 19, 23, 27, 31]:
            print(f"  {L_idx:>2}  ", end="")
            for cell in ["p01", "p02", "p05"]:
                if cell in results and L_idx in results[cell]["chunks"]:
                    last = results[cell]["chunks"][L_idx][-1]
                    print(f"{last['ent_mean']:>10.4f} {last['self_mean']:>10.4f}  ", end="")
                else:
                    print(f"{'---':>10} {'---':>10}  ", end="")
            print()

        # Same for FIRST chunk (prompt-processing region) for contrast
        print("\nFirst 20% (prompt region):\n")
        print(f"{'L':>3}  ", end="")
        for cell in ["p01", "p02", "p05"]:
            print(f"{cell+'_ent':>10} {cell+'_self':>10}  ", end="")
        print()
        for L_idx in [3, 7, 11, 15, 19, 23, 27, 31]:
            print(f"  {L_idx:>2}  ", end="")
            for cell in ["p01", "p02", "p05"]:
                if cell in results and L_idx in results[cell]["chunks"]:
                    first = results[cell]["chunks"][L_idx][0]
                    print(f"{first['ent_mean']:>10.4f} {first['self_mean']:>10.4f}  ", end="")
                else:
                    print(f"{'---':>10} {'---':>10}  ", end="")
            print()

        # Per-chunk trajectory at L31 (where induction-head fires)
        print("\nL31 trajectory per chunk:\n")
        print(f"{'chunk':>5}  ", end="")
        for cell in ["p01", "p02", "p05"]:
            print(f"{cell+'_ent':>10} {cell+'_self':>10}  ", end="")
        print()
        for k in range(5):
            print(f"  {k:>2}  ", end="")
            for cell in ["p01", "p02", "p05"]:
                if cell in results and 31 in results[cell]["chunks"]:
                    c = results[cell]["chunks"][31][k]
                    print(f"{c['ent_mean']:>10.4f} {c['self_mean']:>10.4f}  ", end="")
                else:
                    print(f"{'---':>10} {'---':>10}  ", end="")
            print()
        return

    # Single-cell analysis
    for cell in ["p01", "p02", "p05"]:
        jp = base / f"{cell}_4bit.json"
        if jp.exists():
            out = analyze(jp, cell)
            print(f"\n=== {cell} 4bit ({out['n_tokens']} tokens) ===")
            for L in [19, 31]:
                print(f"  L{L}:")
                for c in out["chunks"][L]:
                    print(f"    chunk {c['chunk']} [{c['range'][0]:>5}-{c['range'][1]:<5}]"
                          f" ent={c['ent_mean']:.4f} self={c['self_mean']:.4f} self_max={c['self_max']:.4f}")


if __name__ == "__main__":
    main()
