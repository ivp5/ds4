"""Aggregate attention entropy/self-mass across 10 cells × 5 chunks × 8 layers.

silv 2026-05-25: confirm L19 lock-vs-success entropy gap holds across N>3 cells.

Loads: tmp/20260525_attention_inflight/p{01..10}_4bit.json
Computes per-(layer, chunk) mean across success cells, then ranks P01 lock
against the population.

The classification of cells (locked vs success) comes from the HC_LOCK
30-cell validation:
  - LOCKED in 4bit: P01 only (most_pers=0.852)
  - SUCCESS commit (no rescue needed): P05 (truth in CoT, native commit)
  - RESCUE-RECOVERED: P01 P02 P03 P04 P06 P07 P08 (truth-in-CoT cells)
  - SILENT-FAIL: P09 (no commit, no truth in CoT)
  - CONFIDENT-WRONG: P10 (committed wrong answer 100)

For the locked-vs-explore axis, P01 (lock) vs everything else (explore) is
the cleanest contrast — the others all "explore but don't commit cleanly"
or "commit correctly" but none exhibit P01's tight cyclic lock.
"""
import json
import sys
from pathlib import Path

import numpy as np


CATEGORIES = {
    "lock": ["p01"],          # HC_LOCK fired
    "rescue": ["p02", "p03", "p04", "p06", "p07", "p08"],  # rescued via aime_rescue
    "native_success": ["p05"],  # native commit success
    "silent_fail": ["p09"],
    "confident_wrong": ["p10"],
}


def load_cells(base: Path) -> dict:
    out = {}
    for cell in ["p01", "p02", "p03", "p04", "p05", "p06", "p07", "p08", "p09", "p10"]:
        jp = base / f"{cell}_4bit.json"
        if jp.exists():
            out[cell] = json.loads(jp.read_text())
    return out


def per_chunk_stats(data: dict, n_chunks: int = 5) -> dict:
    """{layer: [{chunk: k, ent_mean: float, self_mean: float}, ...]}"""
    n = data["n_tokens"]
    chunk_size = n // n_chunks
    out = {}
    for L_str, layer_data in data["per_layer"].items():
        L = int(L_str)
        ent = np.asarray(layer_data["mean_entropy_per_pos"])
        sm = np.asarray(layer_data["mean_self_mass_per_pos"])
        out[L] = []
        for k in range(n_chunks):
            lo = k * chunk_size
            hi = (k + 1) * chunk_size if k < n_chunks - 1 else n
            out[L].append({
                "ent_mean": float(ent[lo:hi].mean()),
                "self_mean": float(sm[lo:hi].mean()),
                "ent_std": float(ent[lo:hi].std()),
            })
    return out


def main():
    base = Path("tmp/20260525_attention_inflight")
    cells = load_cells(base)
    print(f"loaded {len(cells)} cells: {sorted(cells.keys())}\n")

    stats = {c: per_chunk_stats(data) for c, data in cells.items()}

    # For each chunk k, compute (L19 entropy gap of lock-vs-success)
    print("L19 entropy per chunk (lock=p01 vs success-population mean ± std):\n")
    print(f"{'chunk':>5}  {'lock_p01':>10} {'succ_mean':>10} {'succ_std':>10} "
          f"{'z_score':>10} {'gap':>10}")
    success_cells = CATEGORIES["rescue"] + CATEGORIES["native_success"]
    for k in range(5):
        lock_ent = stats["p01"][19][k]["ent_mean"]
        succ_ents = [stats[c][19][k]["ent_mean"] for c in success_cells if c in stats]
        succ_mean = float(np.mean(succ_ents))
        succ_std = float(np.std(succ_ents))
        z = (lock_ent - succ_mean) / (succ_std + 1e-9)
        gap = succ_mean - lock_ent
        print(f"  {k:>3}    {lock_ent:>10.4f} {succ_mean:>10.4f} {succ_std:>10.4f} "
              f"{z:>10.3f} {gap:>10.4f}")

    # Same for other late layers
    print("\nLate-region (chunk 4) entropy gap per layer (lock vs success):\n")
    print(f"{'L':>3}  {'lock_p01':>10} {'succ_mean':>10} {'succ_std':>10} "
          f"{'z_score':>10} {'gap':>10}")
    for L in [3, 7, 11, 15, 19, 23, 27, 31]:
        lock_ent = stats["p01"][L][4]["ent_mean"]
        succ_ents = [stats[c][L][4]["ent_mean"] for c in success_cells if c in stats]
        succ_mean = float(np.mean(succ_ents))
        succ_std = float(np.std(succ_ents))
        z = (lock_ent - succ_mean) / (succ_std + 1e-9)
        gap = succ_mean - lock_ent
        print(f"  {L:>2}  {lock_ent:>10.4f} {succ_mean:>10.4f} {succ_std:>10.4f} "
              f"{z:>10.3f} {gap:>10.4f}")

    # Same analysis for OTHER abnormal cells (silent_fail, confident_wrong)
    print("\nL19 chunk 4 across CATEGORIES:")
    for cat, cell_list in CATEGORIES.items():
        for c in cell_list:
            if c in stats:
                e = stats[c][19][4]["ent_mean"]
                s = stats[c][19][4]["self_mean"]
                n_tok = cells[c]["n_tokens"]
                print(f"  {cat:>16} {c}: ent={e:.4f} self={s:.4f} n_tok={n_tok}")

    # Self-mass instead of entropy — does it carry signal?
    print("\nL31 chunk 4 self-mass per CATEGORY:")
    for cat, cell_list in CATEGORIES.items():
        for c in cell_list:
            if c in stats:
                s = stats[c][31][4]["self_mean"]
                e = stats[c][31][4]["ent_mean"]
                print(f"  {cat:>16} {c}: self={s:.4f} ent={e:.4f}")


if __name__ == "__main__":
    main()
