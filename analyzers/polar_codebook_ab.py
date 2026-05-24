#!/usr/bin/env python3
"""Test B — codebook A/B: quartile MEANS vs quartile PERCENTILES.

Per refinement_oom_accuracy_speedup.md "Test B": the current PLR2 encoder
uses quartile MEANS as the M=4 magnitude codebook. Each row's mag values
are sorted into 4 chunks, each chunk's mean becomes a code. This biases
toward the centroid of each quartile.

Alternative: percentile-based codebook (e.g., 12.5%, 37.5%, 62.5%, 87.5%
percentiles of |mag| per row). These are the midpoints of the quartile
ranges by ORDER, not by mean. On heavy-tail distributions where the
top-quartile mean over-represents the tail, percentile-mid should give
lower reconstruction error.

Free if it wins: zero extra bytes on disk, same kernel.

Sweep at p8_m4 (current PLR2 default) on L0, 32 experts, 128 rows.
Reports rel_L2 + cos_sim for both codebooks.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from polar_encode_bulk import dequant_fp4_tensor
from polar_resolution_sweep import load_layer_gate, measure


def encode_p8_m4_mean(rows_fp32):
    """Original encoder: per-row quartile-mean codebook."""
    n_rows, in_dim = rows_fp32.shape
    pairs = in_dim // 2
    re = rows_fp32[:, 0::2]
    im = rows_fp32[:, 1::2]
    mag = np.sqrt(re * re + im * im)
    angle = np.arctan2(im, re)
    phase_step = 2.0 * math.pi / 8.0
    phase_codes = np.mod(np.rint(angle / phase_step).astype(np.int32), 8)
    qangle = phase_codes.astype(np.float64) * phase_step
    sorted_idx = np.argsort(mag, axis=-1, kind='stable')
    sorted_mag = np.take_along_axis(mag, sorted_idx, axis=-1)
    levels = np.zeros((n_rows, 4), dtype=np.float64)
    mag_codes = np.zeros_like(mag, dtype=np.int32)
    for k in range(4):
        a = k * pairs // 4
        b = (k + 1) * pairs // 4
        levels[:, k] = sorted_mag[:, a:b].mean(axis=-1)  # ← MEAN of quartile
        idx_slice = sorted_idx[:, a:b]
        rows_arange = np.arange(n_rows)[:, None]
        mag_codes[rows_arange, idx_slice] = k
    qmag = np.take_along_axis(levels, mag_codes.astype(np.int64), axis=-1)
    qrows = np.empty_like(rows_fp32, dtype=np.float64)
    qrows[:, 0::2] = qmag * np.cos(qangle)
    qrows[:, 1::2] = qmag * np.sin(qangle)
    return qrows.astype(np.float32)


def encode_p8_m4_percentile(rows_fp32):
    """Alternative encoder: per-row percentile-mid codebook (12.5, 37.5, 62.5, 87.5)."""
    n_rows, in_dim = rows_fp32.shape
    pairs = in_dim // 2
    re = rows_fp32[:, 0::2]
    im = rows_fp32[:, 1::2]
    mag = np.sqrt(re * re + im * im)
    angle = np.arctan2(im, re)
    phase_step = 2.0 * math.pi / 8.0
    phase_codes = np.mod(np.rint(angle / phase_step).astype(np.int32), 8)
    qangle = phase_codes.astype(np.float64) * phase_step
    sorted_idx = np.argsort(mag, axis=-1, kind='stable')
    sorted_mag = np.take_along_axis(mag, sorted_idx, axis=-1)
    # Percentile midpoints of each quartile: 12.5, 37.5, 62.5, 87.5
    levels = np.zeros((n_rows, 4), dtype=np.float64)
    for k in range(4):
        # Index of percentile midpoint within the row
        pct = (k + 0.5) / 4.0   # 0.125, 0.375, 0.625, 0.875
        idx = int(round(pct * pairs))
        if idx >= pairs: idx = pairs - 1
        levels[:, k] = sorted_mag[:, idx]
    mag_codes = np.zeros_like(mag, dtype=np.int32)
    for k in range(4):
        a = k * pairs // 4
        b = (k + 1) * pairs // 4
        idx_slice = sorted_idx[:, a:b]
        rows_arange = np.arange(n_rows)[:, None]
        mag_codes[rows_arange, idx_slice] = k
    qmag = np.take_along_axis(levels, mag_codes.astype(np.int64), axis=-1)
    qrows = np.empty_like(rows_fp32, dtype=np.float64)
    qrows[:, 0::2] = qmag * np.cos(qangle)
    qrows[:, 1::2] = qmag * np.sin(qangle)
    return qrows.astype(np.float32)


def encode_p8_m4_optimal_l2(rows_fp32):
    """Codebook that minimizes per-quartile L2 reconstruction error.
    For each quartile q ∈ {0,1,2,3}, the optimal scalar level is the
    quartile's L2-optimal point, which for uniform distribution-of-codes
    over the quartile range is the MEAN of mag^2 / mean of mag.
    For complex-pair reconstruction where decoded = level × (cos+sin),
    the per-quartile L2 minimizer is the quartile MEAN — same as
    encode_p8_m4_mean. So this is identical to that function but kept
    as a sanity-comparison anchor."""
    return encode_p8_m4_mean(rows_fp32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--n-experts", type=int, default=32)
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    print(f"polar_ab: loading L{args.layer} gate, {args.n_experts} experts × {args.rows} rows",
          file=sys.stderr)
    t0 = time.time()
    weights = load_layer_gate(args.model_dir, args.layer, args.n_experts, args.rows)
    print(f"  loaded in {time.time()-t0:.1f}s  shape={weights.shape}", file=sys.stderr)

    results = {}
    for name, fn in [("mean", encode_p8_m4_mean), ("percentile_mid", encode_p8_m4_percentile)]:
        t1 = time.time()
        cos_acc, rel_acc = [], []
        for e in range(args.n_experts):
            qrows = fn(weights[e])
            cos, rel = measure(weights[e], qrows)
            cos_acc.append(cos); rel_acc.append(rel)
        cos_all = np.concatenate(cos_acc)
        rel_all = np.concatenate(rel_acc)
        elapsed = time.time() - t1
        results[name] = {
            "cos_sim_mean": float(cos_all.mean()),
            "cos_sim_min":  float(cos_all.min()),
            "rel_l2_mean":  float(rel_all.mean()),
            "rel_l2_p95":   float(np.percentile(rel_all, 95)),
            "rel_l2_max":   float(rel_all.max()),
            "elapsed_s": elapsed,
        }
        print(f"  {name:>15}  cos_mean={results[name]['cos_sim_mean']:.5f}  "
              f"rel_L2_mean={results[name]['rel_l2_mean']:.5f}  "
              f"rel_L2_p95={results[name]['rel_l2_p95']:.5f}  "
              f"({elapsed:.1f}s)", file=sys.stderr)

    delta = (results["percentile_mid"]["rel_l2_mean"] - results["mean"]["rel_l2_mean"]) / results["mean"]["rel_l2_mean"]
    print(f"\npolar_ab: percentile_mid vs mean delta_rel_L2 = {delta*100:+.2f}%", file=sys.stderr)
    if delta < -0.01:
        print(f"polar_ab: PERCENTILE WINS by {-delta*100:.2f}% (free accuracy gain)", file=sys.stderr)
    elif delta > 0.01:
        print(f"polar_ab: MEAN WINS by {delta*100:.2f}% (keep current codebook)", file=sys.stderr)
    else:
        print(f"polar_ab: TIE — within ±1% of each other", file=sys.stderr)

    with open(args.out_json, "w") as f:
        json.dump({
            "model_dir": args.model_dir,
            "layer": args.layer,
            "n_experts": args.n_experts,
            "rows": args.rows,
            "codec": "p8_m4",
            "results": results,
            "delta_rel_L2_pct": delta * 100,
            "winner": "percentile_mid" if delta < -0.01 else ("mean" if delta > 0.01 else "tie"),
        }, f, indent=2)
    print(f"\npolar_ab: wrote {args.out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
