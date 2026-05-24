#!/usr/bin/env python3
"""Bulk polar-encode DS4 V4 expert weights — vectorized + multiprocessing.

Speed-up vs polar_encode_safetensors.py:
  - Single Python startup for many experts (saves 100-300 ms each)
  - Tensor read once per expert, not per-row
  - Vectorized FP4 unpack + dequant across all out_dim rows at once
  - Vectorized polar encoding: vectorize phase quantization across all
    (row, pair) at once; per-row 4-level quantile-fit uses np.quantile +
    np.searchsorted (no Python loop)
  - Multiprocessing across experts (default: cpu_count // 2 workers)

Typical: ~0.05-0.10 sec/expert (vs 1.0 sec/expert in single-expert script).
Full layer (256 experts × 3 kinds = 768 encodings) ≈ 1-2 min instead of 13.

Output per expert/kind: same layout as polar_encode_safetensors.py
  <out_dir>/L{LL}_E{EEE}_{kind}.{mag,phase,levels,meta.json}.bin
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

FP4_TABLE = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=np.float32)
FP4_BLOCK_SIZE = 32


def decode_e8m0_to_fp32_array(scale_u8: np.ndarray) -> np.ndarray:
    """F8_E8M0 byte b → 2^(b - 127) via fp32 exponent reinterpret. Vectorized."""
    bits = (scale_u8.astype(np.uint32) << np.uint32(23))
    return bits.view(np.float32).reshape(scale_u8.shape)


def dequant_fp4_tensor(packed_int8: np.ndarray, scale_u8: np.ndarray) -> np.ndarray:
    """Full-tensor FP4 dequant. Vectorized — no Python row loop.
    packed_int8: [out_dim, in_dim/2] int8/uint8
    scale_u8:    [out_dim, in_dim/32] uint8 (F8_E8M0)
    Returns: [out_dim, in_dim] float32
    """
    packed_u8 = packed_int8.view(np.uint8) if packed_int8.dtype != np.uint8 else packed_int8
    out_dim, in_dim_half = packed_u8.shape
    in_dim = in_dim_half * 2
    low = packed_u8 & 0x0F
    high = (packed_u8 >> 4) & 0x0F
    # Stack into [out_dim, in_dim_half, 2] then reshape to [out_dim, in_dim]
    unpacked = np.stack([FP4_TABLE[low.astype(np.int64)],
                          FP4_TABLE[high.astype(np.int64)]], axis=-1).reshape(out_dim, in_dim)
    scale_fp32 = decode_e8m0_to_fp32_array(scale_u8)
    # Broadcast: each scale covers 32 in-axis elements
    scale_expanded = np.repeat(scale_fp32, FP4_BLOCK_SIZE, axis=1)
    return (unpacked * scale_expanded).astype(np.float32)


def polar_encode_p8_m2_vectorized(rows_fp32: np.ndarray):
    """Vectorized p8_m2 encoder. Returns:
      mag_codes uint8 [n_rows, pairs]
      phase_codes uint8 [n_rows, pairs] in 0..8
      levels float32 [n_rows, 4]
      qrows float32 [n_rows, in_dim]
    """
    n_rows, in_dim = rows_fp32.shape
    pairs = in_dim // 2
    # Adjacent-channel complex pair encoding
    z = rows_fp32[:, 0::2] + 1j * rows_fp32[:, 1::2]
    mag = np.abs(z)            # [n_rows, pairs]
    angle = np.angle(z)
    # Phase quantize to 8 levels (signed -4..4 → LUT index 0..8)
    phase_step = 2.0 * math.pi / 8.0
    signed_phase_code = np.rint(angle / phase_step).astype(np.int16)
    signed_phase_code = np.clip(signed_phase_code, -4, 4)
    phase_lut_code = (signed_phase_code + 4).astype(np.uint8)
    # 4-level quantile-fit per row (vectorized — np.quantile + searchsorted)
    # quantile gives the 25/50/75/100% boundaries; we use the per-quartile MEANS
    # for compatibility with codex's quantile_encode.
    levels = np.empty((n_rows, 4), dtype=np.float32)
    mag_codes = np.empty((n_rows, pairs), dtype=np.uint8)
    # Vectorized argsort: sort each row's mag, get 4 equal chunks, take per-chunk mean
    sorted_idx = np.argsort(mag, axis=1, kind='stable')  # [n_rows, pairs]
    chunk = pairs // 4
    # Build mag_codes: for each row, the indices in each quartile get codes 0..3
    row_arange = np.arange(n_rows)[:, None]
    for level in range(4):
        a = level * pairs // 4
        b = (level + 1) * pairs // 4
        idx_slice = sorted_idx[:, a:b]              # [n_rows, chunk]
        # Scatter level into mag_codes at those positions
        mag_codes[row_arange, idx_slice] = level
        # Per-row mean of magnitudes in this quartile
        chunk_mags = np.take_along_axis(mag, idx_slice, axis=1)  # [n_rows, chunk]
        levels[:, level] = chunk_mags.mean(axis=1).astype(np.float32)
    # Reconstruct qrows for diagnostics
    qangle = signed_phase_code.astype(np.float64) * phase_step
    # Per-row level lookup: qmag[r, p] = levels[r, mag_codes[r, p]]
    qmag = np.take_along_axis(levels, mag_codes.astype(np.int64), axis=1).astype(np.float64)
    qrows = np.empty_like(rows_fp32, dtype=np.float64)
    qrows[:, 0::2] = qmag * np.cos(qangle)
    qrows[:, 1::2] = qmag * np.sin(qangle)
    return mag_codes, phase_lut_code, levels, qrows.astype(np.float32)


def find_tensor_shard(model_dir: Path, name: str):
    """Return (shard_path, key) for the named tensor via the index, or scan."""
    idx_path = model_dir / "model.safetensors.index.json"
    if idx_path.exists():
        with open(idx_path) as f:
            wm = json.load(f).get("weight_map", {})
        if name in wm:
            return model_dir / wm[name], name
    # Fall back: scan shards
    from safetensors import safe_open
    for shard in sorted(model_dir.glob("model-*.safetensors")):
        try:
            with safe_open(str(shard), framework="numpy", device="cpu") as f:
                if name in f.keys():
                    return shard, name
        except Exception:
            continue
    return None, None


def load_int8_tensor(shard_path: Path, key: str) -> np.ndarray:
    from safetensors import safe_open
    with safe_open(str(shard_path), framework="numpy", device="cpu") as f:
        return f.get_tensor(key)


def load_uint8_tensor(shard_path: Path, key: str) -> np.ndarray:
    """Load a tensor of any dtype as raw uint8 bytes (for F8_E8M0 scales)."""
    from safetensors import safe_open
    with safe_open(str(shard_path), framework="numpy", device="cpu") as f:
        try:
            t = f.get_tensor(key)
            return t.view(np.uint8) if t.dtype != np.uint8 else t
        except Exception:
            import torch
            from safetensors.torch import safe_open as st
            with st(str(shard_path), framework="pt", device="cpu") as tf:
                return tf.get_tensor(key).view(torch.uint8).numpy()


KIND_TO_W = {"gate": "w1", "up": "w3", "down": "w2"}


def encode_one_expert(args):
    """Worker function: encode one (layer, expert, kind). Returns meta dict."""
    model_dir_s, layer, expert, kind, rows_per_tile, n_tiles, out_dir_s = args
    model_dir = Path(model_dir_s)
    out_dir = Path(out_dir_s)
    w = KIND_TO_W[kind]
    name = f"layers.{layer}.ffn.experts.{expert}.{w}.weight"
    scale_name = f"layers.{layer}.ffn.experts.{expert}.{w}.scale"

    wp, wk = find_tensor_shard(model_dir, name)
    sp, sk = find_tensor_shard(model_dir, scale_name)
    if wp is None or sp is None:
        return {"layer": layer, "expert": expert, "kind": kind, "error": "tensor not found"}

    w_packed = load_int8_tensor(wp, wk)            # [out_dim, in_dim/2] int8
    scale_u8 = load_uint8_tensor(sp, sk)           # [out_dim, in_dim/32] uint8
    rows_fp32 = dequant_fp4_tensor(w_packed, scale_u8)  # [out_dim, in_dim] fp32

    rows_to_encode = rows_per_tile * n_tiles
    if rows_to_encode > rows_fp32.shape[0]:
        rows_to_encode = rows_fp32.shape[0]
    rows_fp32 = rows_fp32[:rows_to_encode]

    mag_codes, phase_codes, levels, qrows = polar_encode_p8_m2_vectorized(rows_fp32)
    out_prefix = out_dir / f"L{layer:02d}_E{expert:03d}_{kind}"
    mag_codes.tofile(str(out_prefix) + ".mag.bin")
    phase_codes.tofile(str(out_prefix) + ".phase.bin")
    levels.tofile(str(out_prefix) + ".levels.bin")

    rel_l2 = float(np.linalg.norm(qrows - rows_fp32) / np.linalg.norm(rows_fp32))
    cos_sim = float((qrows * rows_fp32).sum() / (np.linalg.norm(qrows) * np.linalg.norm(rows_fp32)))
    meta = {
        "layer": layer, "expert": expert, "kind": kind,
        "shard": wp.name, "out_prefix": out_prefix.name,
        "rows_encoded": rows_to_encode,
        "pairs": rows_fp32.shape[1] // 2,
        "in_dim": rows_fp32.shape[1],
        "rel_l2": rel_l2, "cos_sim": cos_sim,
    }
    with open(str(out_prefix) + ".meta.json", "w") as f:
        json.dump(meta, f)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash")
    ap.add_argument("--layers", default="0",
                    help='comma-separated layer indices or "all" for 0..42')
    ap.add_argument("--experts", default="0-15",
                    help='comma-separated expert IDs OR range like "0-15" OR "all" for 0..255')
    ap.add_argument("--kinds", default="gate,up",
                    help='comma-separated subset of {gate,up,down}')
    ap.add_argument("--rows-per-tile", type=int, default=32)
    ap.add_argument("--n-tiles", type=int, default=4)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 4) // 2))
    args = ap.parse_args()

    layers = list(range(43)) if args.layers == "all" else [int(s) for s in args.layers.split(",")]
    if args.experts == "all":
        experts = list(range(256))
    elif "-" in args.experts:
        a, b = args.experts.split("-", 1)
        experts = list(range(int(a), int(b) + 1))
    else:
        experts = [int(s) for s in args.experts.split(",")]
    kinds = [k.strip() for k in args.kinds.split(",")]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(args.model_dir, L, E, K, args.rows_per_tile, args.n_tiles, str(out_dir))
            for L in layers for E in experts for K in kinds]

    print(f"polar_bulk: {len(jobs)} encodings × {args.workers} workers", file=sys.stderr)
    t0 = time.time()
    if args.workers <= 1:
        results = [encode_one_expert(j) for j in jobs]
    else:
        with mp.Pool(args.workers) as p:
            results = p.map(encode_one_expert, jobs, chunksize=4)
    elapsed = time.time() - t0

    n_ok = sum(1 for r in results if "error" not in r)
    n_err = len(results) - n_ok
    cos_sims = [r["cos_sim"] for r in results if "error" not in r]
    rel_l2s  = [r["rel_l2"]  for r in results if "error" not in r]
    print(f"polar_bulk: {n_ok}/{len(jobs)} ok ({n_err} err) in {elapsed:.1f}s "
          f"= {elapsed*1000/len(jobs):.1f} ms/expert", file=sys.stderr)
    if cos_sims:
        print(f"polar_bulk: cos_sim mean={np.mean(cos_sims):.4f}  "
              f"rel_L2 mean={np.mean(rel_l2s):.4f}", file=sys.stderr)
    # Write a manifest
    manifest = {
        "model_dir": args.model_dir,
        "layers": layers, "experts": experts, "kinds": kinds,
        "rows_per_tile": args.rows_per_tile, "n_tiles": args.n_tiles,
        "workers": args.workers,
        "elapsed_seconds": elapsed,
        "n_ok": n_ok, "n_err": n_err,
        "results": results,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
