#!/usr/bin/env python3
"""Encode one DS4 expert weight as p8_m2 polar binaries from safetensors source.

Source: deepseek-ai/DeepSeek-V4-Flash safetensors (downloaded to
        /Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash/).

DS4 expert storage (per inference/convert.py FP4_TABLE):
  - Weight: int8 tensor [out_dim, in_dim/2], each byte packs TWO fp4 codes
    (low nibble = first value, high nibble = second value)
  - Scale: float32 tensor [out_dim, in_dim/32] — one scale per 32 in-axis elements
  - FP4 LUT (16 entries, signed):
      0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
      0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0
  - Dequant: fp32 = FP4_TABLE[code] * scale[out, in//32]

Output layout matches the H1729 tile×row×batch canary in ds4_metal.m:
  <out>.mag.bin       tiles*rows*pairs uint8
  <out>.phase.bin     tiles*rows*pairs uint8
  <out>.levels.bin    tiles*rows*4    float32 (per-row quantile-fit levels)
  <out>.cos_lut.bin   9 float32
  <out>.sin_lut.bin   9 float32
  <out>.meta.json     tiles, rows, pairs + source tensor name + layer/expert/kind
"""
import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np

FP4_TABLE = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=np.float32)

FP4_BLOCK_SIZE = 32


def decode_e8m0_to_fp32(scale_u8: np.ndarray) -> np.ndarray:
    """F8_E8M0 (8 exponent bits, 0 mantissa, no sign) → fp32.
    Value = 2^(byte - 127). Byte 255 = NaN per IEEE 754. Byte 0 = 2^-127 (denorm).
    Used by DS4 V4 as block-scale for FP4 expert weights.
    """
    # Use IEEE-754 fp32 bit reinterpret: set exponent field directly.
    scale_u8 = scale_u8.astype(np.uint32)
    # Treat byte as exponent field of fp32 (bits 23..30), mantissa zero, sign zero.
    # bits = byte << 23 — gives 2^(byte - 127)
    bits = scale_u8 << 23
    out = bits.view(np.uint32).astype(np.uint32).tobytes()
    return np.frombuffer(out, dtype=np.float32).reshape(scale_u8.shape)


def unpack_fp4_row(packed_int8: np.ndarray, scale_row_u8: np.ndarray) -> np.ndarray:
    """Unpack one weight row from FP4-packed int8 + per-block F8_E8M0 scale.
    packed_int8: shape (in_dim/2,) int8/uint8 (two FP4 codes per byte)
    scale_row_u8: shape (in_dim/32,) uint8 (F8_E8M0 codes)
    Returns: shape (in_dim,) float32
    """
    # Treat packed as uint8 for nibble extraction
    packed_u8 = packed_int8.view(np.uint8) if packed_int8.dtype != np.uint8 else packed_int8
    in_dim_half = packed_u8.shape[0]
    in_dim = in_dim_half * 2
    low = packed_u8 & 0x0F
    high = (packed_u8 >> 4) & 0x0F
    pairs = np.stack([FP4_TABLE[low.astype(np.int64)],
                       FP4_TABLE[high.astype(np.int64)]], axis=-1).reshape(in_dim)
    # F8_E8M0 → fp32 then broadcast over 32 elements
    scale_fp32 = decode_e8m0_to_fp32(scale_row_u8.view(np.uint8))
    scale_expanded = np.repeat(scale_fp32, FP4_BLOCK_SIZE)
    return (pairs * scale_expanded).astype(np.float32)


def polar_encode_p8_m2(row: np.ndarray):
    """Per-row p8_m2 encoder, matching codex h1724.polar_encode_p8_m2.
    Returns mag_codes uint8, phase_codes uint8 (in 0..8), levels float32[4], qrow float32.
    """
    assert row.shape[0] % 2 == 0
    z = row[0::2] + 1j * row[1::2]
    mag = np.abs(z)
    angle = np.angle(z)
    phase_step = 2.0 * math.pi / 8.0
    signed_phase_code = np.rint(angle / phase_step).astype(np.int16)
    signed_phase_code = np.clip(signed_phase_code, -4, 4)
    phase_lut_code = (signed_phase_code + 4).astype(np.uint8)
    # 4-level quantile-fit per row
    order = np.argsort(mag, kind='stable')
    n = mag.size
    mag_codes = np.empty(n, dtype=np.uint8)
    levels = np.empty(4, dtype=np.float32)
    for level in range(4):
        a = level * n // 4
        b = (level + 1) * n // 4
        idx = order[a:b]
        levels[level] = float(mag[idx].mean()) if idx.size else 0.0
        mag_codes[idx] = level
    qangle = signed_phase_code.astype(np.float64) * phase_step
    qmag = levels[mag_codes].astype(np.float64)
    qrow = np.empty_like(row, dtype=np.float64)
    qrow[0::2] = qmag * np.cos(qangle)
    qrow[1::2] = qmag * np.sin(qangle)
    return mag_codes, phase_lut_code, levels, qrow.astype(np.float32)


def find_tensor_shard(model_dir: Path, tensor_name: str):
    """Locate which shard contains the named tensor. Returns (shard_path, key)
    or (None, None) if not found.
    """
    idx_path = model_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        return None, None
    with open(idx_path) as f:
        idx = json.load(f)
    wm = idx.get("weight_map", {})
    if tensor_name in wm:
        return model_dir / wm[tensor_name], tensor_name
    return None, None


def load_tensor_metadata(shard_path: Path, key: str):
    """Use safetensors to read a tensor's shape + dtype without materializing it."""
    from safetensors import safe_open
    with safe_open(str(shard_path), framework="numpy", device="cpu") as f:
        s = f.get_slice(key)
        return s.get_shape(), s.get_dtype()


def load_tensor(shard_path: Path, key: str):
    """Materialize the named tensor as numpy. Handles int8 (FP4-packed) + fp32 + bf16."""
    from safetensors import safe_open
    with safe_open(str(shard_path), framework="numpy", device="cpu") as f:
        s = f.get_slice(key)
        dtype = s.get_dtype()
        if dtype == "BF16":
            # safetensors won't materialize BF16 directly as numpy. Use torch path.
            import torch
            from safetensors.torch import safe_open as st
            with st(str(shard_path), framework="pt", device="cpu") as tf:
                return tf.get_tensor(key).to(torch.float32).numpy()
        # Otherwise return as-is from safetensors numpy view
        return f.get_tensor(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--expert", type=int, default=0)
    ap.add_argument("--kind", choices=("gate", "up", "down"), default="gate")
    ap.add_argument("--rows-per-tile", type=int, default=32)
    ap.add_argument("--n-tiles", type=int, default=4,
                    help="how many 32-row tiles to encode (default 4 = 128 rows)")
    ap.add_argument("--hidden-seed", type=int, default=20260525)
    ap.add_argument("--out", required=True, help="output prefix")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    # DS4 V4 expert naming convention (confirmed via real safetensors inspection):
    #   layers.{L}.ffn.experts.{E}.{w1,w2,w3}.weight  (FP4-packed int8)
    #   layers.{L}.ffn.experts.{E}.{w1,w2,w3}.scale   (F8_E8M0)
    # gate = w1, up = w3, down = w2 (per inference/convert.py mapping).
    kind_to_w = {"gate": "w1", "up": "w3", "down": "w2"}
    w = kind_to_w[args.kind]

    # If model.safetensors.index.json hasn't landed yet, fall back to scanning shards.
    candidates = [
        f"layers.{args.layer}.ffn.experts.{args.expert}.{w}.weight",
        f"layers.{args.layer}.mlp.experts.{args.expert}.{w}.weight",
        f"model.layers.{args.layer}.ffn.experts.{args.expert}.{w}.weight",
    ]
    weight_path = scale_path = None
    weight_key = scale_key = None
    for c in candidates:
        sp, key = find_tensor_shard(model_dir, c)
        if sp is None:
            # Manual scan if index missing
            for shard in sorted(model_dir.glob("model-*.safetensors")):
                if shard.stat().st_size < 3.5e9:
                    continue  # skip incomplete
                from safetensors import safe_open
                try:
                    with safe_open(str(shard), framework="numpy", device="cpu") as f:
                        if c in f.keys():
                            sp = shard
                            key = c
                            break
                except Exception:
                    continue
        if sp is not None:
            weight_path, weight_key = sp, key
            scale_name = c.replace(".weight", ".scale")
            sp2, key2 = find_tensor_shard(model_dir, scale_name)
            if sp2 is None:
                # Manual scan for scale too
                from safetensors import safe_open
                for shard in sorted(model_dir.glob("model-*.safetensors")):
                    if shard.stat().st_size < 3.5e9:
                        continue
                    try:
                        with safe_open(str(shard), framework="numpy", device="cpu") as f:
                            if scale_name in f.keys():
                                sp2, key2 = shard, scale_name
                                break
                    except Exception:
                        continue
            if sp2 is not None:
                scale_path, scale_key = sp2, key2
            break
    if weight_path is None:
        print(f"polar_encode: weight not found. Tried: {candidates}", file=sys.stderr)
        sys.exit(2)
    print(f"polar_encode: weight = {weight_path.name}:{weight_key}", file=sys.stderr)
    print(f"polar_encode: scale  = {scale_path.name if scale_path else '<MISSING>'}:{scale_key}", file=sys.stderr)

    w_shape, w_dtype = load_tensor_metadata(weight_path, weight_key)
    print(f"polar_encode: weight shape={w_shape} dtype={w_dtype}", file=sys.stderr)

    if w_dtype in ("I8", "INT8"):
        if scale_path is None:
            print("polar_encode: int8 weight requires a scale tensor; couldn't find one", file=sys.stderr)
            sys.exit(3)
        w_packed = load_tensor(weight_path, weight_key)
        # Scale dtype is F8_E8M0 — load as raw uint8 via safetensors framework=numpy
        # safetensors may not directly support F8_E8M0 numpy view, so manual byte read
        from safetensors import safe_open
        with safe_open(str(scale_path), framework="numpy", device="cpu") as f:
            try:
                scale_u8 = f.get_tensor(scale_key)
            except Exception:
                # Fallback: torch view as uint8
                import torch
                from safetensors.torch import safe_open as st
                with st(str(scale_path), framework="pt", device="cpu") as tf:
                    scale_u8 = tf.get_tensor(scale_key).view(torch.uint8).numpy()
        out_dim, in_dim_half = w_packed.shape
        in_dim = in_dim_half * 2
        print(f"polar_encode: FP4-packed; out_dim={out_dim} in_dim={in_dim} scale_shape={scale_u8.shape} scale_dtype={scale_u8.dtype}", file=sys.stderr)
        def get_row(r):
            return unpack_fp4_row(w_packed[r], scale_u8[r])
    elif w_dtype in ("BF16", "F32", "FLOAT32"):
        # Non-experts (bf16) — already non-quantized
        weight_full = load_tensor(weight_path, weight_key).astype(np.float32)
        out_dim, in_dim = weight_full.shape
        print(f"polar_encode: BF16 → fp32; out_dim={out_dim} in_dim={in_dim}", file=sys.stderr)
        def get_row(r):
            return weight_full[r]
    else:
        print(f"polar_encode: unsupported dtype {w_dtype}", file=sys.stderr)
        sys.exit(4)

    rows_to_encode = args.rows_per_tile * args.n_tiles
    if rows_to_encode > out_dim:
        rows_to_encode = out_dim
        print(f"polar_encode: clamped rows_to_encode to {rows_to_encode}", file=sys.stderr)

    pairs = in_dim // 2
    mag_codes = np.empty((rows_to_encode, pairs), dtype=np.uint8)
    phase_codes = np.empty((rows_to_encode, pairs), dtype=np.uint8)
    levels = np.empty((rows_to_encode, 4), dtype=np.float32)
    qrows = np.empty((rows_to_encode, in_dim), dtype=np.float32)
    rows_full = np.empty((rows_to_encode, in_dim), dtype=np.float32)
    for r in range(rows_to_encode):
        row = get_row(r)
        rows_full[r] = row
        mc, pc, lv, qr = polar_encode_p8_m2(row)
        mag_codes[r] = mc
        phase_codes[r] = pc
        levels[r] = lv
        qrows[r] = qr

    # Deterministic synthetic hidden vector for end-to-end dot validation
    rng = np.random.default_rng(args.hidden_seed)
    # Tile layout: hidden indexed by (batch, tile) — for n_tiles tiles with batch=1 we
    # emit n_tiles distinct hidden vectors
    hidden = rng.standard_normal((args.n_tiles, in_dim), dtype=np.float32)

    # Expected: per-row dot via polar vs fp32 reference
    expected_polar = np.empty(rows_to_encode, dtype=np.float32)
    expected_fp32  = np.empty(rows_to_encode, dtype=np.float32)
    for t in range(args.n_tiles):
        for r in range(args.rows_per_tile):
            global_r = t * args.rows_per_tile + r
            if global_r >= rows_to_encode: break
            expected_polar[global_r] = (qrows[global_r] * hidden[t]).sum().astype(np.float32)
            expected_fp32[global_r]  = (rows_full[global_r] * hidden[t]).sum().astype(np.float32)

    # cos/sin LUT
    cos_lut = np.empty(9, dtype=np.float32)
    sin_lut = np.empty(9, dtype=np.float32)
    for code in range(-4, 5):
        a = float(code) * (2.0 * math.pi / 8.0)
        cos_lut[code + 4] = math.cos(a)
        sin_lut[code + 4] = math.sin(a)

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    mag_codes.tofile(str(out_prefix) + ".mag.bin")
    phase_codes.tofile(str(out_prefix) + ".phase.bin")
    levels.tofile(str(out_prefix) + ".levels.bin")
    cos_lut.tofile(str(out_prefix) + ".cos_lut.bin")
    sin_lut.tofile(str(out_prefix) + ".sin_lut.bin")
    hidden.tofile(str(out_prefix) + ".hidden.bin")
    expected_polar.tofile(str(out_prefix) + ".expected_polar.bin")
    expected_fp32.tofile(str(out_prefix) + ".expected_fp32.bin")

    rel_l2 = float(np.linalg.norm(qrows - rows_full) / np.linalg.norm(rows_full))
    dot_err = float(np.abs(expected_polar - expected_fp32).mean())
    cos_sim = float((qrows * rows_full).sum() / (np.linalg.norm(qrows) * np.linalg.norm(rows_full)))
    meta = {
        "model": str(model_dir),
        "tensor": weight_key,
        "shard": weight_path.name,
        "layer": args.layer,
        "expert": args.expert,
        "kind": args.kind,
        "tiles": args.n_tiles,
        "rows_per_tile": args.rows_per_tile,
        "pairs": pairs,
        "in_dim": in_dim,
        "out_dim_total": out_dim,
        "rows_encoded": rows_to_encode,
        "rel_l2_polar_vs_fp32": rel_l2,
        "cos_sim_polar_vs_fp32": cos_sim,
        "mean_abs_dot_err": dot_err,
    }
    with open(str(out_prefix) + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
