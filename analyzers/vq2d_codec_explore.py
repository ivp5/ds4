#!/usr/bin/env python3
"""Alternative codec: 2D vector quantization of complex pairs (re, im).

Polar codec stores 2 bytes per pair (uint8 mag code + uint8 phase code).
VQ codec stores 1 byte per pair (uint8 centroid index into K=256
codebook in R²). Two advantages potentially:

  1. Storage: 1 byte/pair vs 2 → 50% reduction
  2. Accuracy: K-means optimizes JOINTLY over (re, im) rather than
     separately quantizing magnitude and angle. If the data has
     non-radial structure (e.g., concentration along specific
     directions), VQ can exploit it; polar can't.

Codebook granularity trade-off:
  - global: 1 codebook  (lowest storage, may lose expert-specific structure)
  - per-layer: 43 codebooks (best balance)
  - per-(layer, kind): 129 codebooks
  - per-expert: 11,008 codebooks (huge overhead)

This script tests per-(layer, kind) at L0 to find feasibility, with
the analyzer protocol matching polar_down_real_ffn_scale_ab.py for
direct comparison.
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


def assign_codes(pairs, centroids, chunk_n=1_048_576):
    """Vectorized nearest-centroid via BLAS matmul, chunked for memory.

    argmin_j ||p - c_j||^2 = argmin_j (||c_j||^2 - 2 p·c_j)
    (||p||^2 is constant in j, drops out of argmin)

    Chunk pairs by chunk_n so intermediate [chunk_n, K] fits in RAM.
    Per chunk: matmul + element-wise + argmin (all BLAS-fast).
    Returns codes[N] uint8 (or uint16 if K > 256).
    """
    n = pairs.shape[0]
    k = centroids.shape[0]
    out_dtype = np.uint8 if k <= 256 else np.uint16
    codes = np.empty(n, dtype=out_dtype)
    norms_c = (centroids ** 2).sum(axis=-1)  # [K]
    centroids_T = centroids.T  # [2, K]
    for start in range(0, n, chunk_n):
        end = min(start + chunk_n, n)
        chunk = pairs[start:end]
        dots = chunk @ centroids_T  # [chunk_n, K]
        scores = norms_c[None, :] - 2.0 * dots  # [chunk_n, K]
        codes[start:end] = np.argmin(scores, axis=-1).astype(out_dtype)
    return codes


def lloyd_max_2d(pairs, k=256, max_iter=50, seed=42,
                  fit_sample_size=200_000):
    """K-means on 2D points via subsample-fit + batched-assign.

    Fits codebook on a subsample (default 200k random pairs) — enough
    for stable centroids — then assigns all N pairs in one batched
    pass. Avoids O(N*K*iters) full-data cost when N >> fit_sample_size.

    Returns (codebook[k,2], codes[N], mse_total).
    """
    rng = np.random.default_rng(seed)
    n = pairs.shape[0]
    # Subsample for fit
    fit_n = min(fit_sample_size, n)
    fit_idx = rng.choice(n, size=fit_n, replace=False) if fit_n < n else np.arange(n)
    fit_pairs = pairs[fit_idx].astype(np.float32)

    # K-means++ init on the subsample
    centroids = np.empty((k, 2), dtype=np.float32)
    first = rng.integers(0, fit_n)
    centroids[0] = fit_pairs[first]
    dists = np.full(fit_n, np.inf, dtype=np.float32)
    for i in range(1, k):
        new_d = np.sum((fit_pairs - centroids[i-1])**2, axis=-1)
        dists = np.minimum(dists, new_d)
        if dists.sum() < 1e-12:
            centroids[i] = fit_pairs[rng.integers(0, fit_n)]
        else:
            probs = dists / dists.sum()
            idx = rng.choice(fit_n, p=probs)
            centroids[i] = fit_pairs[idx]

    # Lloyd iterations on the subsample
    prev_loss = np.inf
    for it in range(max_iter):
        codes_fit = assign_codes(fit_pairs, centroids)
        loss = 0.0
        for c in range(k):
            mask = codes_fit == c
            if mask.any():
                centroids[c] = fit_pairs[mask].mean(axis=0)
        # Recompute loss for convergence check (cheap on fit subsample)
        d = ((fit_pairs[:, None, :] - centroids[None, :, :])**2).sum(axis=-1)
        loss = float(np.min(d, axis=-1).sum())
        if abs(prev_loss - loss) < 1e-9 * max(prev_loss, 1.0):
            break
        prev_loss = loss

    # Assign all pairs in one batched pass with final centroids
    codes_all = assign_codes(pairs, centroids).astype(np.uint8)
    # Compute MSE on a sample for reporting
    sample_idx = rng.choice(n, size=min(50_000, n), replace=False)
    sample_pairs = pairs[sample_idx]
    sample_codes = codes_all[sample_idx]
    decoded_sample = centroids[sample_codes]
    mse = float(((sample_pairs - decoded_sample)**2).mean())
    return centroids, codes_all, mse


def load_fp4_layer_kind(model_dir, layer, kind, n_rows_per_expert=128, n_cols=None):
    """Load FP4 for (layer, kind) across all 256 experts.
    Returns [256, n_rows, n_cols] fp32.
    """
    from safetensors import safe_open
    md = Path(model_dir)
    wm = json.load(open(md / "model.safetensors.index.json")).get("weight_map", {})
    kind_to_w = {"gate": "w1", "up": "w3", "down": "w2"}
    w_suffix = kind_to_w[kind]
    all_experts = []
    for e in range(256):
        wname = f"layers.{layer}.ffn.experts.{e}.{w_suffix}.weight"
        sname = f"layers.{layer}.ffn.experts.{e}.{w_suffix}.scale"
        wshard = wm[wname]
        sshard = wm[sname]
        with safe_open(str(md / wshard), framework="numpy", device="cpu") as f:
            w = f.get_tensor(wname)
        w_u8 = w.view(np.uint8) if w.dtype != np.uint8 else w
        try:
            with safe_open(str(md / sshard), framework="numpy", device="cpu") as f:
                s_t = f.get_tensor(sname)
            s_u8 = s_t.view(np.uint8) if s_t.dtype != np.uint8 else s_t
        except Exception:
            import torch
            from safetensors.torch import safe_open as st
            with st(str(md / sshard), framework="pt", device="cpu") as tf:
                s_torch = tf.get_tensor(sname)
                s_u8 = s_torch.view(torch.uint8).numpy()
        full = dequant_fp4_tensor(w_u8, s_u8).astype(np.float32)
        full = full[:n_rows_per_expert]
        if n_cols is not None:
            full = full[:, :n_cols]
        all_experts.append(full)
    return np.stack(all_experts, axis=0)  # [256, n_rows, n_cols]


def fp4_to_pairs(tensor):
    """[E, R, C] → [E * R * C/2, 2] complex pair array."""
    E, R, C = tensor.shape
    assert C % 2 == 0
    pairs = tensor.reshape(E, R, C // 2, 2).reshape(-1, 2)
    return pairs


def reconstruct_from_codes(codebook, codes, shape):
    """Inverse of fp4_to_pairs + lloyd_max_2d: reconstruct [E, R, C] from codes + codebook."""
    decoded_pairs = codebook[codes]  # [N, 2]
    return decoded_pairs.reshape(shape[0], shape[1], shape[2] // 2, 2).reshape(*shape)


def compute_metrics(orig, recon):
    """Per-tensor rel_L2 + cos_sim, plus aggregate."""
    flat_orig = orig.ravel()
    flat_recon = recon.ravel()
    rel_l2 = np.linalg.norm(flat_recon - flat_orig) / max(np.linalg.norm(flat_orig), 1e-12)
    cos_sim = float(np.dot(flat_orig, flat_recon) /
                    (np.linalg.norm(flat_orig) * np.linalg.norm(flat_recon) + 1e-12))
    return float(rel_l2), cos_sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir",
                    default="/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--kind", choices=["gate", "up", "down"], default="down")
    ap.add_argument("--k", type=int, default=256, help="codebook size")
    ap.add_argument("--rows-per-expert", type=int, default=128)
    ap.add_argument("--n-cols", type=int, default=None, help="None = use all in_dim")
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    t0 = time.time()
    print(f"Loading FP4 layer={args.layer} kind={args.kind} ({256} experts)...", file=sys.stderr)
    tensor = load_fp4_layer_kind(args.model_dir, args.layer, args.kind,
                                  args.rows_per_expert, args.n_cols)
    print(f"  loaded shape {tensor.shape} in {time.time()-t0:.1f}s", file=sys.stderr)
    print(f"  dtype={tensor.dtype} total_bytes={tensor.nbytes:,}", file=sys.stderr)

    pairs = fp4_to_pairs(tensor)
    print(f"  pairs shape {pairs.shape} ({pairs.shape[0]:,} 2D points)", file=sys.stderr)

    print(f"\nLloyd-Max K={args.k} (max_iter={args.max_iter})...", file=sys.stderr)
    t1 = time.time()
    codebook, codes, mse = lloyd_max_2d(pairs, k=args.k,
                                          max_iter=args.max_iter, seed=args.seed)
    print(f"  done in {time.time()-t1:.1f}s, mse={mse:.6f}", file=sys.stderr)

    recon = reconstruct_from_codes(codebook, codes, tensor.shape)
    rel_l2, cos_sim = compute_metrics(tensor, recon)

    # Storage accounting
    code_bytes = codes.nbytes  # 1 byte per pair
    codebook_bytes = codebook.nbytes  # K × 2 × 4 bytes
    polar_code_bytes = pairs.shape[0] * 2  # 2 bytes per pair
    polar_levels_bytes = 256 * args.rows_per_expert * 8 * 4  # per-expert per-row mag codebook

    result = {
        "layer": args.layer, "kind": args.kind,
        "k": args.k, "max_iter": args.max_iter, "seed": args.seed,
        "n_pairs": int(pairs.shape[0]),
        "tensor_shape": list(tensor.shape),
        "rel_l2": rel_l2,
        "cos_sim": cos_sim,
        "mse": float(mse),
        "storage": {
            "vq_code_bytes":         int(code_bytes),
            "vq_codebook_bytes":     int(codebook_bytes),
            "vq_total_bytes":        int(code_bytes + codebook_bytes),
            "polar_code_bytes_p32_m8": int(polar_code_bytes),
            "polar_levels_bytes_m8":   int(polar_levels_bytes),
            "polar_total_bytes_p32_m8": int(polar_code_bytes + polar_levels_bytes),
            "vq_vs_polar_ratio":     round((code_bytes + codebook_bytes) /
                                            max(polar_code_bytes + polar_levels_bytes, 1), 4),
        },
        "wall_sec": float(time.time() - t0),
    }

    print(f"\nResults:", file=sys.stderr)
    print(f"  VQ rel_L2: {rel_l2:.4f}  cos_sim: {cos_sim:.4f}", file=sys.stderr)
    print(f"  vs polar p32_m8 weight rel_L2: ~0.120 (codec_pareto_v2_m8m16.md)", file=sys.stderr)
    print(f"\nStorage (one (layer, kind) tile, {tensor.shape[0]} experts × {tensor.shape[1]} rows × {tensor.shape[2]} cols):",
          file=sys.stderr)
    print(f"  VQ:         {(code_bytes + codebook_bytes)/1e6:.1f} MB  "
          f"(codes {code_bytes/1e6:.1f} MB + codebook {codebook_bytes/1e3:.1f} KB)",
          file=sys.stderr)
    print(f"  polar p32_m8: {(polar_code_bytes + polar_levels_bytes)/1e6:.1f} MB  "
          f"(codes {polar_code_bytes/1e6:.1f} MB + levels {polar_levels_bytes/1e6:.1f} MB)",
          file=sys.stderr)
    print(f"  VQ / polar  = {result['storage']['vq_vs_polar_ratio']:.3f}",
          file=sys.stderr)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.out_json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
