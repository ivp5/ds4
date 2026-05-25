#!/usr/bin/env python3
"""Phase B-2.3b extension — down codec quality at REAL FFN scale.

The polar corpus limits gate/down to n_rows=128, so the prior A/B
analyzer caps at act_rows=128. To measure the down codec at the
real-FFN accumulation length (act_rows=2048 = moe_intermediate_size),
we substitute the act vector with one derived from the FP4 gate
(which has all 2048 rows). Then:

  out_polar[d] = sum_{r=0..2047} act[r] * polar_dwn[d, r]
  out_fp4[d]   = sum_{r=0..2047} act[r] * fp4_dwn[d, r]

The act vector is IDENTICAL on both sides (same FP4 gate), so the
delta isolates **down-codec quality at real-FFN scale**. This is the
deployment-relevant measurement: if codec error averages out via CLT
at act_rows=2048, we'll see rel_L2 << 0.20; if codec error is
systematic, rel_L2 stays ≈ 0.20.

Pure numpy, no DS4 runtime needed.
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
from polar_down_decode_validator import load_plr2_expert
from polar_vs_fp4_kernel_output_ab import load_fp4_expert, polar_decode_down_tile


def silu(x):
    return x / (1.0 + np.exp(-x))


def compute_real_ffn_ab(polar_dir, model_dir, layer, expert,
                          act_rows=2048, down_rows=128,
                          hidden_dim=4096, hidden_value=1.0):
    """Real-FFN-scale down codec A/B.

    act[r] = silu(gate_fp4[r] · hidden) * (up_fp4[r] · hidden)
      for r ∈ [0, act_rows), using FP4 gate AND up
    out_polar[d] = sum_r act[r] * polar_dwn[d, r]
    out_fp4[d]   = sum_r act[r] * fp4_dwn[d, r]

    The same act on both sides isolates down-codec quality.
    """
    # 1. Load FP4 gate + up to compute act at real FFN scale
    gate_fp4 = load_fp4_expert(model_dir, layer, expert, "gate",
                                 n_rows=act_rows, n_cols=hidden_dim).astype(np.float32)
    up_fp4   = load_fp4_expert(model_dir, layer, expert, "up",
                                 n_rows=act_rows, n_cols=hidden_dim).astype(np.float32)
    # Hidden is uniform `hidden_value` for canary-style determinism
    hidden = np.full(hidden_dim, hidden_value, dtype=np.float32)
    gate_dot = gate_fp4 @ hidden  # [act_rows]
    up_dot   = up_fp4   @ hidden  # [act_rows]
    act = (silu(gate_dot) * up_dot).astype(np.float32)

    # 2. Load polar-decoded down (corpus rows up to 128) for first `down_rows` outputs
    pd = Path(polar_dir)
    down_path = pd / f"L{layer:02d}_down.polar"
    plr2_down = load_plr2_expert(str(down_path), expert)
    if down_rows > plr2_down["n_rows"]:
        down_rows = plr2_down["n_rows"]
    polar_dwn = polar_decode_down_tile(plr2_down, down_rows, act_rows)  # [down_rows, act_rows]

    # 3. Load FP4 down for same (layer, expert), first `down_rows × act_rows` tile
    fp4_dwn = load_fp4_expert(model_dir, layer, expert, "down",
                                n_rows=down_rows, n_cols=act_rows).astype(np.float32)

    # 4. Compute outputs through both down codecs with the SAME act
    out_polar = polar_dwn @ act
    out_fp4   = fp4_dwn   @ act

    # 5. Metrics
    diff = out_polar - out_fp4
    abs_err_max = float(np.abs(diff).max())
    norm_fp4 = float(np.linalg.norm(out_fp4)) + 1e-12
    rel_l2 = float(np.linalg.norm(diff) / norm_fp4)
    cos_sim = float(np.dot(out_polar, out_fp4) /
                    (np.linalg.norm(out_polar) * np.linalg.norm(out_fp4) + 1e-12))

    # Weight-level for cross-check
    wdiff = polar_dwn.ravel() - fp4_dwn.ravel()
    weight_rel_l2 = float(np.linalg.norm(wdiff) / (np.linalg.norm(fp4_dwn.ravel()) + 1e-12))
    weight_cos_sim = float(np.dot(polar_dwn.ravel(), fp4_dwn.ravel()) /
                           (np.linalg.norm(polar_dwn.ravel()) * np.linalg.norm(fp4_dwn.ravel()) + 1e-12))

    # CLT naive prediction (if errors were independent random):
    # rel_L2 should scale as weight_rel_L2 / sqrt(act_rows / 1)
    # ... but that's also wrong because weight rel_L2 is already per-element
    # The right CLT prediction: var(sum) = sum(var) so the "expected" output
    # rel_L2 if independent random is ≈ weight_rel_L2 (the per-element
    # standard deviation propagates additively in the sum-of-squares sense).
    # Actually for elementwise iid errors with sigma_e, the output error is
    # |sigma_e * sum(|act_r|)| / |sum(act_r * w_r)|, which depends on act
    # distribution. Skipping the prediction — just report measured.

    return {
        "layer": layer, "expert": expert,
        "act_rows": act_rows, "down_rows": down_rows,
        "polar_codec": f"p{plr2_down['phase_levels']}_m{plr2_down['mag_levels']}",
        "out_polar_norm": float(np.linalg.norm(out_polar)),
        "out_fp4_norm":   float(np.linalg.norm(out_fp4)),
        "abs_err_max":    abs_err_max,
        "out_rel_l2":     rel_l2,
        "out_cos_sim":    cos_sim,
        "weight_rel_l2":  weight_rel_l2,
        "weight_cos_sim": weight_cos_sim,
        "act_norm":       float(np.linalg.norm(act)),
        "act_min":        float(act.min()),
        "act_max":        float(act.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--polar-dir", required=True)
    ap.add_argument("--model-dir",
                    default="/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash")
    ap.add_argument("--layers", default="0,22,42")
    ap.add_argument("--experts", default="0,100,255")
    ap.add_argument("--act-rows", type=int, default=2048,
                    help="real moe_intermediate_size for DS4 V4 = 2048")
    ap.add_argument("--down-rows", type=int, default=128,
                    help="capped at polar corpus n_rows (typically 128)")
    ap.add_argument("--hidden-dim", type=int, default=4096)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    experts = [int(x) for x in args.experts.split(",")]

    print(f"# polar_down_real_ffn_scale_ab — act_rows={args.act_rows} (REAL moe_int_size)",
          file=sys.stderr)
    print(f"# polar_dir={args.polar_dir} layers={layers} experts={experts}", file=sys.stderr)
    print(f"# {'L':>3} {'E':>4} {'wrl2':>7} {'orl2':>7} {'cos':>7} {'act|':>10}", file=sys.stderr)

    results = []
    t0 = time.time()
    for L in layers:
        for E in experts:
            r = compute_real_ffn_ab(args.polar_dir, args.model_dir, L, E,
                                     act_rows=args.act_rows, down_rows=args.down_rows,
                                     hidden_dim=args.hidden_dim)
            results.append(r)
            print(f"  {L:>3} {E:>4} {r['weight_rel_l2']:>7.4f} {r['out_rel_l2']:>7.4f} "
                  f"{r['out_cos_sim']:>7.4f} {r['act_norm']:>10.2f}", file=sys.stderr)
    wall = time.time() - t0

    out_rls = [r["out_rel_l2"] for r in results]
    out_cos = [r["out_cos_sim"] for r in results]
    weight_rls = [r["weight_rel_l2"] for r in results]
    weight_cos = [r["weight_cos_sim"] for r in results]
    print(f"\n# Aggregate across {len(results)} cells at REAL FFN scale "
          f"(act_rows={args.act_rows}):", file=sys.stderr)
    print(f"#  out rel_L2:   mean={np.mean(out_rls):.4f}  min={min(out_rls):.4f}  max={max(out_rls):.4f}",
          file=sys.stderr)
    print(f"#  out cos_sim:  mean={np.mean(out_cos):.4f}  min={min(out_cos):.4f}",
          file=sys.stderr)
    print(f"#  weight rel_L2: mean={np.mean(weight_rls):.4f}", file=sys.stderr)
    print(f"#  weight cos_sim: mean={np.mean(weight_cos):.4f}", file=sys.stderr)
    ratio = np.mean(out_rls) / max(np.mean(weight_rls), 1e-12)
    print(f"#  output_rel_L2 / weight_rel_L2 = {ratio:.3f}", file=sys.stderr)
    print(f"#  wall: {wall:.1f}s", file=sys.stderr)

    # CLT-comparison verdict
    canary_ratio = 1.21  # from prior B-2.3b at act_rows=16
    if ratio < canary_ratio * 0.5:
        print("#  VERDICT: real-FFN-scale CLT averaging HELPS — codec quality improves "
              "at real scale ✓", file=sys.stderr)
    elif ratio < canary_ratio * 1.2:
        print("#  VERDICT: real-FFN-scale ≈ canary-scale — codec error is systematic, "
              "no CLT free lunch", file=sys.stderr)
    else:
        print("#  VERDICT: real-FFN-scale WORSE than canary — codec error compounds "
              "at real scale", file=sys.stderr)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({
                "polar_dir": args.polar_dir, "model_dir": args.model_dir,
                "layers": layers, "experts": experts,
                "act_rows": args.act_rows, "down_rows": args.down_rows,
                "hidden_dim": args.hidden_dim,
                "results": results,
                "aggregate": {
                    "out_rel_l2_mean": float(np.mean(out_rls)),
                    "out_cos_sim_mean": float(np.mean(out_cos)),
                    "weight_rel_l2_mean": float(np.mean(weight_rls)),
                    "weight_cos_sim_mean": float(np.mean(weight_cos)),
                    "ratio_output_to_weight": float(ratio),
                    "wall_sec": float(wall),
                },
            }, f, indent=2)
        print(f"#  wrote {args.out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
