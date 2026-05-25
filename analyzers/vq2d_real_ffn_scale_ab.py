#!/usr/bin/env python3
"""VQ-2D codec quality at REAL FFN scale (act_rows=2048).

Mirrors polar_down_real_ffn_scale_ab.py protocol so VQ results are
directly comparable to polar p32_m8 at the kernel-output level.

For each (layer, expert):
  act[r] = silu(gate_fp4 @ hidden) * (up_fp4 @ hidden)  for r ∈ [0, 2048)
  out_vq[d]  = sum_r act[r] * vq_decoded_down[d, r]
  out_fp4[d] = sum_r act[r] * fp4_down[d, r]

Reports per-cell + aggregate rel_L2 + cos_sim, comparable line-by-line
to the polar real_ffn_scale results.
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
from polar_vs_fp4_kernel_output_ab import load_fp4_expert
from vq2d_codec_explore import (
    lloyd_max_2d, assign_codes, load_fp4_layer_kind, fp4_to_pairs
)


def silu(x):
    return x / (1.0 + np.exp(-x))


def vq_encode_decode_layer_kind(model_dir, layer, kind, k=256, max_iter=12,
                                 rows_per_expert=128, n_cols=None, seed=42):
    """Encode + decode FP4 weights via VQ-2D, return decoded[E,R,C] + codebook + stats."""
    tensor = load_fp4_layer_kind(model_dir, layer, kind,
                                  n_rows_per_expert=rows_per_expert,
                                  n_cols=n_cols)
    pairs = fp4_to_pairs(tensor)
    codebook, codes, mse = lloyd_max_2d(pairs, k=k, max_iter=max_iter, seed=seed)
    decoded_pairs = codebook[codes]
    decoded = decoded_pairs.reshape(tensor.shape[0], tensor.shape[1],
                                     tensor.shape[2] // 2, 2).reshape(*tensor.shape)
    return decoded, codebook, mse, tensor


def compute_real_ffn_ab(model_dir, layer, expert, k, max_iter=12,
                         act_rows=2048, down_rows=128, hidden_dim=4096,
                         hidden_value=1.0, vq_decoded_down=None, fp4_down=None):
    """One cell: real-FFN-scale output A/B (VQ vs FP4 down).
    Pass pre-computed vq_decoded_down + fp4_down if available to avoid repeat encode.
    """
    if vq_decoded_down is None or fp4_down is None:
        # Encode this layer's down via VQ
        vq_decoded_full, _, _, fp4_full = vq_encode_decode_layer_kind(
            model_dir, layer, "down", k=k, max_iter=max_iter,
            rows_per_expert=down_rows, n_cols=act_rows)
        vq_decoded_down = vq_decoded_full[expert]
        fp4_down = fp4_full[expert]

    # Compute act via FP4 gate + up (same on both VQ and FP4 sides)
    gate_fp4 = load_fp4_expert(model_dir, layer, expert, "gate",
                                 n_rows=act_rows, n_cols=hidden_dim).astype(np.float32)
    up_fp4 = load_fp4_expert(model_dir, layer, expert, "up",
                               n_rows=act_rows, n_cols=hidden_dim).astype(np.float32)
    hidden = np.full(hidden_dim, hidden_value, dtype=np.float32)
    gate_dot = gate_fp4 @ hidden
    up_dot = up_fp4 @ hidden
    act = (silu(gate_dot) * up_dot).astype(np.float32)

    # Compute out through both down codecs
    out_vq = vq_decoded_down @ act
    out_fp4 = fp4_down @ act

    diff = out_vq - out_fp4
    rel_l2 = float(np.linalg.norm(diff) / max(np.linalg.norm(out_fp4), 1e-12))
    cos_sim = float(np.dot(out_vq, out_fp4) /
                    (np.linalg.norm(out_vq) * np.linalg.norm(out_fp4) + 1e-12))

    # Weight-level for cross-check
    wdiff = vq_decoded_down.ravel() - fp4_down.ravel()
    weight_rel_l2 = float(np.linalg.norm(wdiff) /
                          (np.linalg.norm(fp4_down.ravel()) + 1e-12))
    weight_cos_sim = float(np.dot(vq_decoded_down.ravel(), fp4_down.ravel()) /
                           (np.linalg.norm(vq_decoded_down.ravel()) *
                            np.linalg.norm(fp4_down.ravel()) + 1e-12))

    return {
        "layer": layer, "expert": expert, "k": k,
        "act_rows": act_rows, "down_rows": down_rows,
        "out_rel_l2": rel_l2,
        "out_cos_sim": cos_sim,
        "weight_rel_l2": weight_rel_l2,
        "weight_cos_sim": weight_cos_sim,
        "act_norm": float(np.linalg.norm(act)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir",
                    default="/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash")
    ap.add_argument("--layers", default="0,22,42")
    ap.add_argument("--experts", default="0,100,255")
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--max-iter", type=int, default=12)
    ap.add_argument("--act-rows", type=int, default=2048)
    ap.add_argument("--down-rows", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=4096)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    experts = [int(x) for x in args.experts.split(",")]

    print(f"# vq2d_real_ffn_scale_ab — K={args.k} act_rows={args.act_rows}", file=sys.stderr)
    print(f"# layers={layers} experts={experts}", file=sys.stderr)
    print(f"# {'L':>3} {'E':>4} {'wrl2':>7} {'orl2':>7} {'cos':>7} {'act|':>10}", file=sys.stderr)

    results = []
    t0 = time.time()
    # Encode each layer's down ONCE, then test multiple experts within it
    for L in layers:
        print(f"  encoding L{L} down (K={args.k})...", file=sys.stderr)
        t_enc = time.time()
        vq_decoded_full, codebook, mse, fp4_full = vq_encode_decode_layer_kind(
            args.model_dir, L, "down", k=args.k, max_iter=args.max_iter,
            rows_per_expert=args.down_rows, n_cols=args.act_rows)
        print(f"    encoded in {time.time()-t_enc:.1f}s, mse={mse:.6f}", file=sys.stderr)
        for E in experts:
            r = compute_real_ffn_ab(args.model_dir, L, E, args.k,
                                     act_rows=args.act_rows, down_rows=args.down_rows,
                                     hidden_dim=args.hidden_dim,
                                     vq_decoded_down=vq_decoded_full[E],
                                     fp4_down=fp4_full[E])
            results.append(r)
            print(f"  {L:>3} {E:>4} {r['weight_rel_l2']:>7.4f} {r['out_rel_l2']:>7.4f} "
                  f"{r['out_cos_sim']:>7.4f} {r['act_norm']:>10.2f}", file=sys.stderr)
    wall = time.time() - t0

    out_rls = [r["out_rel_l2"] for r in results]
    out_cos = [r["out_cos_sim"] for r in results]
    weight_rls = [r["weight_rel_l2"] for r in results]
    weight_cos = [r["weight_cos_sim"] for r in results]
    print(f"\n# Aggregate across {len(results)} cells (VQ K={args.k}, "
          f"act_rows={args.act_rows}):", file=sys.stderr)
    print(f"#  out rel_L2:   mean={np.mean(out_rls):.4f}  min={min(out_rls):.4f}  max={max(out_rls):.4f}",
          file=sys.stderr)
    print(f"#  out cos_sim:  mean={np.mean(out_cos):.4f}  min={min(out_cos):.4f}",
          file=sys.stderr)
    print(f"#  weight rel_L2: mean={np.mean(weight_rls):.4f}", file=sys.stderr)
    print(f"#  weight cos_sim: mean={np.mean(weight_cos):.4f}", file=sys.stderr)
    ratio = np.mean(out_rls) / max(np.mean(weight_rls), 1e-12)
    print(f"#  output_rel_L2 / weight_rel_L2 = {ratio:.3f}", file=sys.stderr)
    print(f"#  wall: {wall:.1f}s", file=sys.stderr)

    # Direct comparison to polar p32_m8 at same protocol
    POLAR_P32M8_OUT_RL2_MEAN = 0.121  # from commit (production-validated)
    POLAR_P32M8_OUT_COS_MEAN = 0.993
    vq_vs_polar = np.mean(out_rls) / POLAR_P32M8_OUT_RL2_MEAN
    print(f"\n# vs polar p32_m8 (same protocol, validated production):", file=sys.stderr)
    print(f"#  polar p32_m8 out rel_L2: {POLAR_P32M8_OUT_RL2_MEAN:.4f}", file=sys.stderr)
    print(f"#  VQ K={args.k}      out rel_L2: {np.mean(out_rls):.4f}", file=sys.stderr)
    print(f"#  improvement ratio:        {1.0/vq_vs_polar:.2f}× lower",
          file=sys.stderr)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({
                "model_dir": args.model_dir,
                "layers": layers, "experts": experts, "k": args.k,
                "act_rows": args.act_rows, "down_rows": args.down_rows,
                "results": results,
                "aggregate": {
                    "out_rel_l2_mean":   float(np.mean(out_rls)),
                    "out_cos_sim_mean":  float(np.mean(out_cos)),
                    "weight_rel_l2_mean": float(np.mean(weight_rls)),
                    "weight_cos_sim_mean": float(np.mean(weight_cos)),
                    "ratio_output_to_weight": float(ratio),
                    "polar_p32m8_baseline_rel_l2": POLAR_P32M8_OUT_RL2_MEAN,
                    "vq_vs_polar_improvement": float(1.0 / vq_vs_polar),
                    "wall_sec": float(wall),
                },
            }, f, indent=2)
        print(f"\n#  wrote {args.out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
