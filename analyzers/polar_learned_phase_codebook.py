#!/usr/bin/env python3
"""Test D — learned non-uniform phase codebook A/B vs uniform bins.

Test C showed phase distribution is 21% non-uniform: top-8 of 128 bins
hold 37% of mass, peaks at FP4-induced angles (multiples of 45°), and
15.7% of fine bins are empty.

This probe builds a LEARNED phase codebook by clustering the actual
angle distribution. For each candidate code count N ∈ {8, 16, 32, 48,
64}, run circular k-means (k-means in (cos, sin) space then map back
to angles) to find N cluster centers. Encode the test weights using
both:
  - uniform_N:  angles → nearest of N equally-spaced bins
  - learned_N:  angles → nearest of N empirically-clustered bins

Compare rel_L2 + cos_sim. If learned beats uniform at same N, the
gain is at FIXED storage cost (same code count = same bits per pair).

CRITICAL: the codec stores the codebook itself as side data (N×4
bytes per layer for cluster centers). Negligible vs the per-expert
mag/phase arrays.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from polar_resolution_sweep import load_layer_gate, measure


def cluster_phases_kmeans(angles_flat, n_codes, max_iter=50, seed=42):
    """Cluster angles via k-means in (cos, sin) Euclidean space.

    Returns the N cluster center angles (in radians).
    angles_flat: 1D array of angles in radians (any subset of the data).
    n_codes: number of cluster centers.
    """
    rng = np.random.default_rng(seed)
    # Start with uniform initialization: equispaced angles
    centers_ang = np.linspace(-math.pi, math.pi, n_codes, endpoint=False)
    cos_pts = np.cos(angles_flat)
    sin_pts = np.sin(angles_flat)
    for it in range(max_iter):
        cos_c = np.cos(centers_ang)
        sin_c = np.sin(centers_ang)
        # Compute distances from each point to each center (in unit-circle space)
        # dist² = (cp - cc)² + (sp - sc)² = 2 - 2*(cp*cc + sp*sc)
        # Minimum dist = maximum dot product cp*cc + sp*sc
        dots = cos_pts[:, None] * cos_c[None, :] + sin_pts[:, None] * sin_c[None, :]
        assigns = np.argmax(dots, axis=1)
        # Recompute centers as the angle of the mean (cos, sin) of each cluster
        new_centers = np.copy(centers_ang)
        for k in range(n_codes):
            mask = (assigns == k)
            if not np.any(mask):
                continue
            mean_c = cos_pts[mask].mean()
            mean_s = sin_pts[mask].mean()
            new_centers[k] = math.atan2(mean_s, mean_c)
        shift = float(np.max(np.abs(np.unwrap(new_centers) - np.unwrap(centers_ang))))
        centers_ang = new_centers
        if shift < 1e-4:
            break
    # Sort by angle for stable comparison
    centers_ang = np.sort(centers_ang)
    return centers_ang


def encode_with_phase_codebook(rows_fp32, phase_codebook, mag_levels=4):
    """Encode with given phase codebook + quartile-mean mag codebook.

    phase_codebook: array of N angles in radians (cluster centers).
    """
    n_rows, in_dim = rows_fp32.shape
    pairs = in_dim // 2
    re = rows_fp32[:, 0::2]
    im = rows_fp32[:, 1::2]
    mag = np.sqrt(re * re + im * im)
    angle = np.arctan2(im, re)

    # Assign each angle to nearest codebook entry via cos/sin
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    cos_c = np.cos(phase_codebook)
    sin_c = np.sin(phase_codebook)
    # Dot products: [n_rows, pairs, N_codes]
    # For memory, do it in chunks of rows
    qangle = np.empty_like(angle, dtype=np.float64)
    chunk = 32
    for r0 in range(0, n_rows, chunk):
        r1 = min(r0 + chunk, n_rows)
        ca = cos_a[r0:r1, :, None]  # [c, p, 1]
        sa = sin_a[r0:r1, :, None]
        dots = ca * cos_c[None, None, :] + sa * sin_c[None, None, :]
        idx = np.argmax(dots, axis=-1)  # [c, p]
        qangle[r0:r1] = phase_codebook[idx]

    # Mag quartile-mean codebook (same as p8_m4)
    sorted_idx = np.argsort(mag, axis=-1, kind='stable')
    sorted_mag = np.take_along_axis(mag, sorted_idx, axis=-1)
    levels = np.zeros((n_rows, mag_levels), dtype=np.float64)
    mag_codes = np.zeros_like(mag, dtype=np.int32)
    for k in range(mag_levels):
        a = k * pairs // mag_levels
        b = (k + 1) * pairs // mag_levels
        levels[:, k] = sorted_mag[:, a:b].mean(axis=-1)
        idx_slice = sorted_idx[:, a:b]
        rows_arange = np.arange(n_rows)[:, None]
        mag_codes[rows_arange, idx_slice] = k
    qmag = np.take_along_axis(levels, mag_codes.astype(np.int64), axis=-1)
    qrows = np.empty_like(rows_fp32, dtype=np.float64)
    qrows[:, 0::2] = qmag * np.cos(qangle)
    qrows[:, 1::2] = qmag * np.sin(qangle)
    return qrows.astype(np.float32)


def encode_with_uniform_phase(rows_fp32, n_codes, mag_levels=4):
    """Encode with uniform N-bin phase codebook + quartile-mean mag.
    Direct call into the existing uniform encoder."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from polar_resolution_sweep import encode_polar_pN_mM
    qrows, _ = encode_polar_pN_mM(rows_fp32, n_codes, mag_levels)
    return qrows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--n-experts", type=int, default=32)
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--code-counts", default="8,16,32,48,64")
    ap.add_argument("--mag-levels", type=int, default=4)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    print(f"polar_learned: loading L{args.layer} gate, {args.n_experts} experts × {args.rows} rows",
          file=sys.stderr)
    t0 = time.time()
    weights = load_layer_gate(args.model_dir, args.layer, args.n_experts, args.rows)
    print(f"  loaded in {time.time()-t0:.1f}s  shape={weights.shape}", file=sys.stderr)

    # Build the angle pool for codebook training: subsample to keep k-means tractable
    rng = np.random.default_rng(42)
    all_angles = []
    for e in range(args.n_experts):
        re = weights[e][:, 0::2]
        im = weights[e][:, 1::2]
        angles = np.arctan2(im, re).flatten()
        # subsample 1% per expert to keep total tractable for kmeans
        n_sub = max(1, len(angles) // 100)
        idx = rng.choice(len(angles), n_sub, replace=False)
        all_angles.append(angles[idx])
    angle_pool = np.concatenate(all_angles)
    print(f"  k-means training pool: {len(angle_pool):,} angles (1% subsample)", file=sys.stderr)

    code_counts = [int(x) for x in args.code_counts.split(",")]
    results = []
    for N in code_counts:
        # Learn codebook
        t1 = time.time()
        codebook = cluster_phases_kmeans(angle_pool, N)
        kmeans_time = time.time() - t1
        # Encode all experts with both uniform and learned at N codes
        cos_u, rel_u = [], []
        cos_l, rel_l = [], []
        t2 = time.time()
        for e in range(args.n_experts):
            qu = encode_with_uniform_phase(weights[e], N, args.mag_levels)
            ql = encode_with_phase_codebook(weights[e], codebook, args.mag_levels)
            cu, ru = measure(weights[e], qu)
            cl, rl = measure(weights[e], ql)
            cos_u.append(cu); rel_u.append(ru)
            cos_l.append(cl); rel_l.append(rl)
        encode_time = time.time() - t2
        ru_all = np.concatenate(rel_u)
        rl_all = np.concatenate(rel_l)
        cu_all = np.concatenate(cos_u)
        cl_all = np.concatenate(cos_l)
        delta_rel = (rl_all.mean() - ru_all.mean()) / max(ru_all.mean(), 1e-12)
        bits_per_pair = math.log2(N) + math.log2(args.mag_levels)
        entry = {
            "code_count": N,
            "bits_per_pair": bits_per_pair,
            "uniform_cos_mean": float(cu_all.mean()),
            "learned_cos_mean": float(cl_all.mean()),
            "uniform_rel_L2_mean": float(ru_all.mean()),
            "learned_rel_L2_mean": float(rl_all.mean()),
            "delta_rel_L2_pct": float(delta_rel * 100),
            "uniform_rel_L2_p95": float(np.percentile(ru_all, 95)),
            "learned_rel_L2_p95": float(np.percentile(rl_all, 95)),
            "kmeans_time_s": kmeans_time,
            "encode_time_s": encode_time,
            "codebook_angles_deg": [float(math.degrees(a)) for a in codebook],
        }
        results.append(entry)
        winner = "LEARNED" if delta_rel < -0.005 else ("UNIFORM" if delta_rel > 0.005 else "TIE")
        print(f"  N={N:>3}  bits={bits_per_pair:>4.1f}  "
              f"uniform rel_L2={ru_all.mean():.4f}  "
              f"learned rel_L2={rl_all.mean():.4f}  "
              f"Δ={delta_rel*100:+.2f}%  {winner}  "
              f"(k-means {kmeans_time:.1f}s, encode {encode_time:.1f}s)",
              file=sys.stderr)

    # Find the operating point where learned beats uniform by the most
    biggest_win = min(results, key=lambda r: r["delta_rel_L2_pct"])
    print(f"\npolar_learned: biggest learned-codebook win at N={biggest_win['code_count']}: "
          f"{-biggest_win['delta_rel_L2_pct']:.2f}% rel_L2 reduction at same bits", file=sys.stderr)

    with open(args.out_json, "w") as f:
        json.dump({
            "model_dir": args.model_dir,
            "layer": args.layer,
            "n_experts": args.n_experts,
            "rows": args.rows,
            "mag_levels": args.mag_levels,
            "results": results,
        }, f, indent=2)
    print(f"\npolar_learned: wrote {args.out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
