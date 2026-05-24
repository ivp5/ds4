#!/usr/bin/env python3
"""Test C — phase angular distribution. Does it cluster?

Per refinement_oom_accuracy_speedup.md "Test C": encode L0 weights at
high phase resolution (p128 = 2.8° bins), compute the histogram of
phase codes, check uniformity. Non-uniform → a learned/skewed phase
codebook could match the data better at the same storage.

Reports:
  - Phase histogram (128 bins)
  - Entropy of the distribution (vs uniform = log2(128) = 7 bits)
  - Chi-square test against uniform
  - Top-N most-populated phase bins

Per-layer: load gate weights for L0, build histogram across all
(expert, row, pair) angles.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from polar_resolution_sweep import load_layer_gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--n-experts", type=int, default=32)
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--phase-bins", type=int, default=128)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    print(f"polar_phase: loading L{args.layer} gate, {args.n_experts} experts × {args.rows} rows",
          file=sys.stderr)
    t0 = time.time()
    weights = load_layer_gate(args.model_dir, args.layer, args.n_experts, args.rows)
    print(f"  loaded in {time.time()-t0:.1f}s  shape={weights.shape}", file=sys.stderr)

    P = args.phase_bins
    phase_step = 2.0 * math.pi / P
    hist = np.zeros(P, dtype=np.int64)
    total = 0
    for e in range(args.n_experts):
        rows_fp32 = weights[e]
        re = rows_fp32[:, 0::2]
        im = rows_fp32[:, 1::2]
        angle = np.arctan2(im, re)
        codes = np.mod(np.rint(angle / phase_step).astype(np.int32), P)
        hist += np.bincount(codes.flatten(), minlength=P)
        total += codes.size

    # Stats
    p = hist.astype(np.float64) / max(total, 1)
    uniform_p = 1.0 / P
    entropy_bits = -float(np.sum(p[p > 0] * np.log2(p[p > 0])))
    uniform_entropy_bits = float(math.log2(P))
    entropy_ratio = entropy_bits / uniform_entropy_bits  # 1.0 = perfectly uniform

    # Chi-square statistic
    expected = total / P
    chisq = float(np.sum((hist - expected) ** 2 / max(expected, 1)))
    # Degrees of freedom = P - 1; rough threshold at p=0.01 for large P is ~P + 3√(2P)
    chisq_threshold_01 = P - 1 + 3 * math.sqrt(2 * (P - 1))

    # Top + bottom 5 bins
    sorted_idx = np.argsort(hist)
    top5 = [(int(i), int(hist[i]), float(p[i])) for i in sorted_idx[-5:][::-1]]
    bot5 = [(int(i), int(hist[i]), float(p[i])) for i in sorted_idx[:5]]

    # Concentration in top-{8, 32}: how much of the mass sits in the top-N bins?
    sorted_p = np.sort(p)[::-1]
    top8_mass = float(sorted_p[:8].sum())
    top32_mass = float(sorted_p[:32].sum())
    uniform_top8 = 8.0 / P
    uniform_top32 = 32.0 / P

    print(f"\npolar_phase: histogram stats over {total:,} (expert, row, pair) angles", file=sys.stderr)
    print(f"  entropy:       {entropy_bits:.4f} bits  (uniform={uniform_entropy_bits:.4f} bits)", file=sys.stderr)
    print(f"  entropy_ratio: {entropy_ratio:.5f}  (1.0 = perfectly uniform)", file=sys.stderr)
    print(f"  chi-square:    {chisq:.1f}  (uniform-null reject at p<0.01 if > ~{chisq_threshold_01:.1f})", file=sys.stderr)
    print(f"  top-8 mass:    {top8_mass*100:.2f}%  (uniform={uniform_top8*100:.2f}%)", file=sys.stderr)
    print(f"  top-32 mass:   {top32_mass*100:.2f}%  (uniform={uniform_top32*100:.2f}%)", file=sys.stderr)
    print(f"\n  top 5 bins (most populated):", file=sys.stderr)
    for i, c, prob in top5:
        deg = (i * 360.0 / P) - 180.0  # bin center angle in degrees, wrapped to [-180, 180)
        if deg > 180: deg -= 360
        print(f"    bin {i:>3} (~{deg:+7.2f}°)  count={c:>7,}  p={prob*100:.4f}%", file=sys.stderr)
    print(f"\n  bottom 5 bins (least populated):", file=sys.stderr)
    for i, c, prob in bot5:
        deg = (i * 360.0 / P) - 180.0
        if deg > 180: deg -= 360
        print(f"    bin {i:>3} (~{deg:+7.2f}°)  count={c:>7,}  p={prob*100:.4f}%", file=sys.stderr)

    # Verdict
    print("", file=sys.stderr)
    if entropy_ratio > 0.999:
        verdict = "UNIFORM — angles are evenly distributed; p8 bins are already optimal on this axis"
    elif entropy_ratio > 0.99:
        verdict = "NEAR-UNIFORM — only ~1% information loss vs ideal; non-uniform codebook unlikely to help much"
    elif entropy_ratio > 0.95:
        verdict = "MILDLY NON-UNIFORM — non-uniform codebook could save 5-10% storage"
    else:
        verdict = f"SIGNIFICANTLY NON-UNIFORM — non-uniform codebook could save {(1-entropy_ratio)*100:.0f}% storage"
    print(f"polar_phase: VERDICT — {verdict}", file=sys.stderr)

    with open(args.out_json, "w") as f:
        json.dump({
            "model_dir": args.model_dir,
            "layer": args.layer,
            "n_experts": args.n_experts,
            "rows": args.rows,
            "phase_bins": P,
            "total_samples": int(total),
            "entropy_bits": entropy_bits,
            "uniform_entropy_bits": uniform_entropy_bits,
            "entropy_ratio": entropy_ratio,
            "chi_square": chisq,
            "chi_square_threshold_p01": chisq_threshold_01,
            "top_8_mass": top8_mass,
            "top_32_mass": top32_mass,
            "histogram": hist.tolist(),
            "verdict": verdict,
        }, f, indent=2)
    print(f"\npolar_phase: wrote {args.out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
