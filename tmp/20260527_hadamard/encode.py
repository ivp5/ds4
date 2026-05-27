#!/usr/bin/env python3
"""Hadamard-16 encode-side tool for DS4 basis-aware codec.

silv 2026-05-27 task #645 (after #643 runtime kernel landed): the encode-side
counterpart that applies the same 16-point Walsh-Hadamard transform (scaled
by 1/sqrt(16) for orthogonality) to FP16 weight tiles ahead of basis-aware
quantization or pair-AVG hot-store storage.

Codex H1953/H1955 found that DCT-16 / Hadamard-16 basis transforms applied
before adjacent-pair VQB2 encoding improve DS4 routed-FFN reconstruction
quality by 1-3% at K=16. Deployment requires both ends to apply the
SAME orthogonal transform — runtime applies it to input activations
(kernel from #643), encode-side applies it to weights (this tool). Then
the quantized weights live in the transformed basis and the inner
product matches the original-basis matmul:

    <H_norm(w), H_norm(x)>  =  <w, x>

where H_norm = H_16 / sqrt(16) = H_16 * 0.25.

## Usage

Standalone (raw FP16 file):
    python3 encode.py --in W.fp16 --shape 256x7168 --out Wh.fp16

NPY input/output:
    python3 encode.py --in W.npy --out Wh.npy

IQ2_XXS GGUF tensor → Hadamard-transformed FP16:
    python3 encode.py --gguf model.gguf --tensor blk.0.ffn_gate_exps.weight \\
        --out Wh.fp16

Verify (round-trip + matmul equivalence):
    python3 encode.py --in W.fp16 --shape 256x7168 --verify
"""

from __future__ import annotations
import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np


HADAMARD_BLOCK = 16
HADAMARD_NORM = 1.0 / np.sqrt(HADAMARD_BLOCK)  # = 0.25 for N=16


# ============================================================================
# Hadamard transform (matches metal/hadamard.metal exactly)
# ============================================================================

def hadamard16_block(x: np.ndarray) -> np.ndarray:
    """Apply 16-point Walsh-Hadamard transform to last-dim blocks of 16.

    Input shape: (..., n_blocks * 16) — last dim must be a multiple of 16.
    Output shape: same as input, with H_16 applied to each 16-element block,
    scaled by 1/sqrt(16) for orthogonal normalization.

    This is the in-place butterfly that matches the Metal kernel
    `kernel_hadamard16_fp16_batched`:
        for stride in [1, 2, 4, 8]:
            for i in [0, 16):
                if i & stride == 0:
                    base = (i & ~(2*stride - 1)) + (i & (stride - 1))
                    (x[base], x[base+stride]) = (x[base]+x[base+stride], x[base]-x[base+stride])
        x *= 1/sqrt(16)
    """
    if x.shape[-1] % HADAMARD_BLOCK != 0:
        raise ValueError(f"last dim {x.shape[-1]} not divisible by {HADAMARD_BLOCK}")
    orig_shape = x.shape
    n_blocks = x.shape[-1] // HADAMARD_BLOCK
    # Reshape so last two dims are (n_blocks, 16); we'll butterfly within each block.
    work = x.reshape(*x.shape[:-1], n_blocks, HADAMARD_BLOCK).astype(np.float32, copy=True)

    # 4-stage butterfly. Vectorize by working on pairs at each stride level.
    for stride in (1, 2, 4, 8):
        # For each block, pair (work[..., base], work[..., base+stride]).
        # Index `base` ranges over positions where (i & stride) == 0:
        #   base = (i & ~(2*stride - 1)) + (i & (stride - 1))   for i with bit `stride` = 0
        # Equivalently: indices i = j_high * (2*stride) + j_low for j_low in [0, stride)
        # and j_high in [0, 16 / (2*stride)). The pair is (i, i+stride).
        group = 2 * stride
        n_groups = HADAMARD_BLOCK // group
        # Reshape last dim (16) → (n_groups, group). Within each group, the lower
        # half (0..stride-1) pairs with the upper half (stride..group-1).
        reshaped = work.reshape(*work.shape[:-1], n_groups, group)
        low  = reshaped[..., :stride].copy()
        high = reshaped[..., stride:].copy()
        reshaped[..., :stride] = low + high
        reshaped[..., stride:] = low - high
        work = reshaped.reshape(*work.shape[:-1], HADAMARD_BLOCK)

    work *= HADAMARD_NORM
    return work.reshape(orig_shape).astype(x.dtype, copy=False)


def hadamard16_block_inverse(x: np.ndarray) -> np.ndarray:
    """Inverse Hadamard-16 transform. Since H_norm is orthogonal and symmetric,
    H_norm × H_norm = I, so the inverse is the forward transform applied
    again. This wrapper exists for self-documenting code."""
    return hadamard16_block(x)


# ============================================================================
# I/O loaders
# ============================================================================

def load_input(args) -> tuple[np.ndarray, str]:
    """Load the input tensor. Returns (array, source_description)."""
    if args.gguf and args.tensor:
        return load_gguf_tensor(args.gguf, args.tensor), f"GGUF[{args.gguf}::{args.tensor}]"
    if args.input is None:
        sys.exit("error: --in PATH (or --gguf PATH --tensor NAME) is required")
    path = Path(args.input)
    if path.suffix == ".npy":
        arr = np.load(path).astype(np.float16, copy=False)
        return arr, f"npy[{path}]"
    # Raw FP16 binary
    if args.shape is None:
        sys.exit("error: --shape is required for raw binary input")
    shape = parse_shape(args.shape)
    raw = path.read_bytes()
    expected = int(np.prod(shape)) * 2
    if len(raw) != expected:
        sys.exit(f"error: file size {len(raw)} != expected {expected} for shape {shape}")
    arr = np.frombuffer(raw, dtype=np.float16).reshape(shape).copy()
    return arr, f"fp16[{path}, shape={shape}]"


def parse_shape(spec: str) -> tuple[int, ...]:
    parts = spec.replace("x", ",").replace(",", " ").split()
    return tuple(int(p) for p in parts)


def load_gguf_tensor(gguf_path: str, tensor_name: str) -> np.ndarray:
    """Load a single tensor from a GGUF file, dequantizing IQ2_XXS or
    reading FP16 directly. Returns FP16 numpy array."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                          "20260525_codec_vs_iq2_audit"))
    try:
        from codec_quality_ab import _read_gguf_meta  # type: ignore
        from dequant_iq2_xxs import dequant_iq2_xxs_fast  # type: ignore
    except ImportError as exc:
        sys.exit(f"error: cannot import GGUF/IQ2 helpers from codec_vs_iq2_audit ({exc})")

    meta, tinfo, data_start = _read_gguf_meta(Path(gguf_path))
    match = [t for t in tinfo if t["name"] == tensor_name]
    if not match:
        sys.exit(f"error: tensor '{tensor_name}' not in GGUF")
    info = match[0]
    n_elem = int(np.prod(info["dims"]))
    with open(gguf_path, "rb") as f:
        f.seek(data_start + info["offset"])
        if info["type"] == 1:   # F16
            raw = f.read(n_elem * 2)
            arr = np.frombuffer(raw, dtype=np.float16).reshape(info["dims"][::-1])
        elif info["type"] == 16:  # IQ2_XXS
            n_blocks = n_elem // 256
            raw = f.read(n_blocks * 66)
            fp32 = dequant_iq2_xxs_fast(raw, n_elem)
            arr = fp32.reshape(info["dims"][::-1]).astype(np.float16, copy=False)
        else:
            sys.exit(f"error: unsupported tensor type {info['type']} for {tensor_name}")
    return arr


def save_output(arr: np.ndarray, args) -> str:
    """Save transformed tensor. Returns output path description."""
    if args.output is None:
        sys.exit("error: --out is required (use --verify-only to skip writing)")
    path = Path(args.output)
    if path.suffix == ".npy":
        np.save(path, arr)
        return f"npy[{path}]"
    path.write_bytes(arr.astype(np.float16, copy=False).tobytes())
    return f"fp16[{path}]"


# ============================================================================
# Verification
# ============================================================================

def verify_roundtrip(W: np.ndarray, tol: float = 1.0e-2) -> tuple[bool, float, float]:
    """Verify H(H(W)) == W within FP16 precision.

    Returns (passed, max_abs_err, rel_l2)."""
    Wh = hadamard16_block(W.astype(np.float16))
    Whh = hadamard16_block(Wh)
    err = np.abs(Whh.astype(np.float32) - W.astype(np.float32))
    max_err = float(err.max())
    norm = float(np.linalg.norm(W.astype(np.float32)))
    rel_l2 = float(np.linalg.norm(err) / (norm + 1.0e-12))
    return (rel_l2 < tol, max_err, rel_l2)


def verify_matmul(W: np.ndarray, n_test_vec: int = 32, tol: float = 5.0e-2) -> tuple[bool, float, float]:
    """Verify <H(W), H(x)> ≈ <W, x> for random x.

    Treats W as (out, in) shape. Last dim of W is the K dim. Generates
    random FP16 inputs x of shape (in,) and compares W @ x vs Wh @ xh.

    Returns (passed, worst_rel_L2, mean_rel_L2).

    Uses rel_L2 = ||y_hada - y_orig|| / ||y_orig|| as the metric. Per-element
    max_rel is unreliable because dot-product output occasionally lands
    near zero (Gaussian-vector dot products have ~Normal(0, sqrt(N))
    distribution; ~3% of components are smaller than 0.03 × stddev), where
    even tiny absolute noise inflates relative error meaninglessly. rel_L2
    measures the structural agreement that matters downstream."""
    if W.ndim != 2:
        return (True, 0.0, 0.0)  # Skip — only meaningful for 2D
    in_dim = W.shape[-1]
    if in_dim % HADAMARD_BLOCK != 0:
        return (True, 0.0, 0.0)

    Wh = hadamard16_block(W)
    rng = np.random.default_rng(0xc0ffee)
    worst_l2 = 0.0
    sum_l2 = 0.0
    for _ in range(n_test_vec):
        x = rng.standard_normal(in_dim).astype(np.float16)
        y_orig = (W.astype(np.float32) @ x.astype(np.float32))
        xh = hadamard16_block(x.copy())
        y_hada = (Wh.astype(np.float32) @ xh.astype(np.float32))
        diff = y_hada - y_orig
        rel_l2 = float(np.linalg.norm(diff) /
                       (np.linalg.norm(y_orig) + 1.0e-12))
        worst_l2 = max(worst_l2, rel_l2)
        sum_l2 += rel_l2
    mean_l2 = sum_l2 / n_test_vec
    return (worst_l2 < tol, worst_l2, mean_l2)


# ============================================================================
# CLI
# ============================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g_src = ap.add_argument_group("input")
    g_src.add_argument("--in", dest="input", help="path to FP16 binary or .npy file")
    g_src.add_argument("--shape", help="tensor shape for raw FP16 input (e.g. 256x7168)")
    g_src.add_argument("--gguf", help="GGUF file (use with --tensor)")
    g_src.add_argument("--tensor", help="tensor name within --gguf (e.g. blk.0.ffn_gate_exps.weight)")

    g_dst = ap.add_argument_group("output")
    g_dst.add_argument("--out", dest="output", help="output path (.fp16 or .npy)")

    g_mode = ap.add_argument_group("mode")
    g_mode.add_argument("--verify", action="store_true",
                        help="run round-trip + matmul-equivalence verification "
                             "(in addition to writing output)")
    g_mode.add_argument("--verify-only", action="store_true",
                        help="run verification only (skip output write)")
    g_mode.add_argument("--inverse", action="store_true",
                        help="apply inverse Hadamard (same as forward since H×H=I, "
                             "kept for documentation)")

    args = ap.parse_args(argv)

    arr, src_desc = load_input(args)
    print(f"loaded: {src_desc}  shape={arr.shape}  dtype={arr.dtype}")

    # Verify if requested (round-trip + matmul) — uses ORIGINAL weights.
    do_verify = args.verify or args.verify_only
    if do_verify:
        print("\n--- verify: H(H(W)) = W round-trip ---")
        ok, max_err, rel_l2 = verify_roundtrip(arr)
        print(f"  max|err|={max_err:.4e}  rel_L2={rel_l2:.4e}  "
              f"{'PASS' if ok else 'FAIL'}")

        if arr.ndim == 2:
            print("\n--- verify: <H(W), H(x)> = <W, x> over 32 random vectors ---")
            ok2, worst_l2, mean_l2 = verify_matmul(arr)
            print(f"  worst_rel_L2={worst_l2:.4e}  mean_rel_L2={mean_l2:.4e}  "
                  f"{'PASS' if ok2 else 'FAIL'}")
        else:
            print(f"\n--- skipping matmul verify (need 2D tensor, got {arr.ndim}D) ---")

    # Apply transform + write
    if not args.verify_only:
        transformed = hadamard16_block(arr)
        if args.inverse:
            # Inverse = forward, but mark the saved output as already-inverse-applied
            pass
        out_desc = save_output(transformed, args)
        print(f"\nwrote: {out_desc}  shape={transformed.shape}  dtype={transformed.dtype}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
