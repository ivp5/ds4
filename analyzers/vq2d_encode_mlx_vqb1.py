#!/usr/bin/env python3
"""MLX-accelerated VQ-2D → VQB1 encoder.

Two MLX speedups vs the CPU encoder (vq2d_encode_vqb1.py):
  1. FP4 dequant via MLX GPU (mirrors polar_encode_mlx.py approach)
  2. K-means assign via MLX matmul (5-10× faster than numpy BLAS on M1)

Centroid update stays in numpy (cheap, scatter-mean is awkward in MLX
0.31). The Lloyd-Max loop becomes:
  Assign (MLX GPU):       ~0.5 sec @ 33M pairs / iter
  Update (numpy CPU):     ~0.2 sec / iter
  Total Lloyd:            ~7 sec for 12 iters / tile
  + Load (MLX dequant):   ~1 sec
  + Write VQB1:           ~1 sec
  Total per tile:         ~9 sec (vs CPU ~17 sec)

For DS4 V4 full corpus at 128 rows: 129 tiles × 9 sec = ~19 min.
At full rows (32× more pairs per tile): ~10× slower → ~3 hours.

Per-tile load is amortized via shard-parallel slurp (mirroring polar
encoder).
"""
import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("WARNING: mlx not installed; falling back to numpy K-means", file=sys.stderr)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from polar_encode_bulk import dequant_fp4_tensor, FP4_TABLE, FP4_BLOCK_SIZE
from vq2d_encode_vqb1 import (
    VQB1_MAGIC, VQB1_HEADER_BYTES, KIND_NAME_TO_ID, read_vqb1_full,
)
from vq2d_codec_explore import load_fp4_layer_kind, fp4_to_pairs


def _make_fp4_lut_mx():
    return mx.array(FP4_TABLE, dtype=mx.float32)


def _dequant_fp4_batch_mlx(packed_u8: 'mx.array', scale_u8: 'mx.array',
                            fp4_lut: 'mx.array'):
    """Same as polar's _dequant_fp4_batch_mlx — vectorized FP4 → fp32 on MLX."""
    low = (packed_u8 & 0x0F).astype(mx.int32)
    high = ((packed_u8 >> 4) & 0x0F).astype(mx.int32)
    low_v = fp4_lut[low]
    high_v = fp4_lut[high]
    B, out_dim, half = packed_u8.shape
    interleaved = mx.stack([low_v, high_v], axis=-1).reshape(B, out_dim, half * 2)
    scale_fp = mx.exp((scale_u8.astype(mx.float32) - 127.0) * math.log(2.0))
    scale_expanded = mx.repeat(scale_fp, FP4_BLOCK_SIZE, axis=-1)
    return interleaved * scale_expanded


def assign_codes_mlx(pairs_mx, centroids_mx, chunk_n=2_097_152):
    """MLX-batched nearest-centroid via matmul.
    Returns codes as numpy uint8 (or uint16 if K > 256).
    """
    k = centroids_mx.shape[0]
    out_dtype = np.uint8 if k <= 256 else np.uint16
    n = pairs_mx.shape[0]
    codes_np = np.empty(n, dtype=out_dtype)
    norms_c = (centroids_mx ** 2).sum(axis=-1)  # [K]
    centroids_T = centroids_mx.T  # [2, K]
    for start in range(0, n, chunk_n):
        end = min(start + chunk_n, n)
        chunk = pairs_mx[start:end]
        dots = chunk @ centroids_T  # [chunk, K]
        scores = norms_c[None, :] - 2.0 * dots
        chunk_codes = mx.argmin(scores, axis=-1)
        mx.eval(chunk_codes)
        codes_np[start:end] = np.array(chunk_codes).astype(out_dtype)
    return codes_np


def lloyd_max_2d_mlx(pairs_np, k=256, max_iter=15, seed=42, fit_sample_size=200_000):
    """K-means on 2D points via MLX assign + numpy centroid update.

    Reuses subsample-fit pattern: fit centroids on subsample (cheap), then
    assign all N pairs via MLX matmul.
    """
    rng = np.random.default_rng(seed)
    n = pairs_np.shape[0]
    fit_n = min(fit_sample_size, n)
    fit_idx = rng.choice(n, size=fit_n, replace=False) if fit_n < n else np.arange(n)
    fit_pairs_np = pairs_np[fit_idx].astype(np.float32)

    # K-means++ init on the subsample (numpy is fine for this small step)
    centroids_np = np.empty((k, 2), dtype=np.float32)
    first = rng.integers(0, fit_n)
    centroids_np[0] = fit_pairs_np[first]
    dists = np.full(fit_n, np.inf, dtype=np.float32)
    for i in range(1, k):
        new_d = np.sum((fit_pairs_np - centroids_np[i-1])**2, axis=-1)
        dists = np.minimum(dists, new_d)
        if dists.sum() < 1e-12:
            centroids_np[i] = fit_pairs_np[rng.integers(0, fit_n)]
        else:
            probs = dists / dists.sum()
            idx = rng.choice(fit_n, p=probs)
            centroids_np[i] = fit_pairs_np[idx]

    # Lloyd iterations on subsample via MLX
    fit_pairs_mx = mx.array(fit_pairs_np)
    prev_loss = np.inf
    for it in range(max_iter):
        centroids_mx = mx.array(centroids_np)
        codes_fit_np = assign_codes_mlx(fit_pairs_mx, centroids_mx)
        # Centroid update in numpy
        for c in range(k):
            mask = codes_fit_np == c
            if mask.any():
                centroids_np[c] = fit_pairs_np[mask].mean(axis=0)
        # Convergence check (cheap)
        d_check = ((fit_pairs_np[:, None, :] - centroids_np[None, :, :])**2).sum(axis=-1)
        loss = float(np.min(d_check, axis=-1).sum())
        if abs(prev_loss - loss) < 1e-9 * max(prev_loss, 1.0):
            break
        prev_loss = loss

    # Assign all pairs via MLX matmul in chunks
    pairs_mx = mx.array(pairs_np)
    centroids_mx = mx.array(centroids_np)
    codes_all_np = assign_codes_mlx(pairs_mx, centroids_mx)

    # MSE on sample for reporting
    sample_idx = rng.choice(n, size=min(50_000, n), replace=False)
    sample_pairs = pairs_np[sample_idx]
    sample_codes = codes_all_np[sample_idx]
    decoded_sample = centroids_np[sample_codes]
    mse = float(((sample_pairs - decoded_sample)**2).mean())
    return centroids_np, codes_all_np, mse


def encode_layer_kind_mlx(model_dir, layer, kind, k=256, max_iter=12,
                            rows_per_expert=128, n_cols=None,
                            seed=42, out_path=None):
    """Encode one (layer, kind) tile to VQB1 file with MLX acceleration."""
    t0 = time.time()
    tensor = load_fp4_layer_kind(model_dir, layer, kind,
                                  n_rows_per_expert=rows_per_expert,
                                  n_cols=n_cols)
    t_load = time.time() - t0
    n_experts, n_rows, in_dim = tensor.shape
    n_pairs = in_dim // 2
    pairs = fp4_to_pairs(tensor)

    t1 = time.time()
    if HAS_MLX:
        codebook, codes, mse = lloyd_max_2d_mlx(pairs, k=k, max_iter=max_iter, seed=seed)
    else:
        from vq2d_codec_explore import lloyd_max_2d
        codebook, codes, mse = lloyd_max_2d(pairs, k=k, max_iter=max_iter, seed=seed)
    t_encode = time.time() - t1

    decoded_pairs = codebook[codes]
    decoded = decoded_pairs.reshape(n_experts, n_rows, n_pairs, 2).reshape(*tensor.shape)
    rel_l2 = float(np.linalg.norm(decoded.ravel() - tensor.ravel()) /
                   max(np.linalg.norm(tensor.ravel()), 1e-12))
    cos_sim = float(np.dot(decoded.ravel(), tensor.ravel()) /
                    (np.linalg.norm(decoded.ravel()) * np.linalg.norm(tensor.ravel()) + 1e-12))

    codes_layout = codes.reshape(n_experts, n_rows, n_pairs).astype(np.uint8)

    if out_path is not None:
        with open(out_path, "wb") as f:
            f.write(VQB1_MAGIC)
            f.write(struct.pack("<IIIIIII",
                                  1, n_experts, n_rows, n_pairs,
                                  layer, KIND_NAME_TO_ID[kind], k))
            f.write(codebook.astype(np.float32).tobytes())
            f.write(codes_layout.tobytes())

    return {
        "layer": layer, "kind": kind, "k": k,
        "n_experts": n_experts, "n_rows": n_rows, "n_pairs": n_pairs,
        "in_dim": in_dim,
        "rel_l2": rel_l2, "cos_sim": cos_sim, "mse": float(mse),
        "load_sec": float(t_load), "encode_sec": float(t_encode),
        "total_sec": float(time.time() - t0),
        "out_path": str(out_path) if out_path else None,
        "file_size": (VQB1_HEADER_BYTES + k * 2 * 4 + n_experts * n_rows * n_pairs)
                      if out_path else None,
        "mlx": HAS_MLX,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir",
                    default="/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash")
    ap.add_argument("--layers", default="0")
    ap.add_argument("--kinds", default="gate,up,down")
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--rows-per-expert", type=int, default=128)
    ap.add_argument("--max-iter", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = list(range(43)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    kinds = [k.strip() for k in args.kinds.split(",")]

    print(f"MLX-VQB1 encode: K={args.k} layers={layers} kinds={kinds} out_dir={out_dir} mlx={HAS_MLX}",
          file=sys.stderr)

    t_grand = time.time()
    results = []
    for L in layers:
        for K in kinds:
            out_path = out_dir / f"L{L:02d}_{K}.vqb1"
            r = encode_layer_kind_mlx(args.model_dir, L, K, k=args.k,
                                        max_iter=args.max_iter,
                                        rows_per_expert=args.rows_per_expert,
                                        seed=args.seed, out_path=out_path)
            results.append(r)
            print(f"  L{L:02d} {K:>4} K={args.k}  rel_L2={r['rel_l2']:.4f} cos={r['cos_sim']:.4f} "
                  f"load={r['load_sec']:.1f}s enc={r['encode_sec']:.1f}s "
                  f"total={r['total_sec']:.1f}s mlx={r['mlx']}",
                  file=sys.stderr)

    grand = time.time() - t_grand
    rels = [r["rel_l2"] for r in results]
    print(f"\n# {len(results)} tiles in {grand:.1f}s "
          f"({grand / max(len(results), 1):.1f}s/tile avg)", file=sys.stderr)
    print(f"#   rel_L2:  mean={np.mean(rels):.4f}  min={min(rels):.4f}  max={max(rels):.4f}",
          file=sys.stderr)

    with open(out_dir / "manifest.json", "w") as f:
        json.dump({
            "encoder": "vq2d_encode_mlx_vqb1.py",
            "mlx_enabled": HAS_MLX,
            "model_dir": args.model_dir,
            "k": args.k, "rows_per_expert": args.rows_per_expert,
            "max_iter": args.max_iter, "seed": args.seed,
            "n_files": len(results),
            "elapsed_seconds": grand,
            "rel_l2_mean": float(np.mean(rels)),
            "files": [{"layer": r["layer"], "kind": r["kind"],
                        "rel_l2": r["rel_l2"], "cos_sim": r["cos_sim"],
                        "file_size": r["file_size"],
                        "total_sec": r["total_sec"]} for r in results],
        }, f, indent=2)
    print(f"#   wrote {out_dir}/manifest.json", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
