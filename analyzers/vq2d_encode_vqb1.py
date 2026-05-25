#!/usr/bin/env python3
"""VQ-2D encoder → VQB1 file format.

Mirrors polar_encode_mlx.py / PLR2 architecture for VQ-2D codec.
One VQB1 file per (layer, kind) carrying all experts at K=256.

VQB1 file format (header 32 bytes + payload):
  bytes 0-3:   magic "VQB1"
  bytes 4-7:   uint32 version (=1)
  bytes 8-11:  uint32 n_experts
  bytes 12-15: uint32 n_rows
  bytes 16-19: uint32 n_pairs
  bytes 20-23: uint32 layer
  bytes 24-27: uint32 kind_id (0=gate, 1=up, 2=down)
  bytes 28-31: uint32 k (codebook size)
  payload:
    k * 2 * sizeof(float) bytes: codebook[k][2] (re, im) fp32
    n_experts * n_rows * n_pairs bytes: codes (uint8, row-major per expert)

Total per file: 32 + K*8 + E*R*P bytes.

For DS4 V4 typical at K=256, E=256, R=128, P_gate=2048, P_down=1024:
  gate: 32 + 2048 + 67,108,864 = 67.1 MB
  up:   same
  down: 32 + 2048 + 33,554,432 = 33.6 MB
"""
import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from polar_encode_bulk import dequant_fp4_tensor
from vq2d_codec_explore import (
    lloyd_max_2d, assign_codes, load_fp4_layer_kind, fp4_to_pairs
)


VQB1_MAGIC = b"VQB1"
VQB1_HEADER_BYTES = 32

KIND_GATE = 0
KIND_UP = 1
KIND_DOWN = 2
KIND_NAME_TO_ID = {"gate": KIND_GATE, "up": KIND_UP, "down": KIND_DOWN}


def encode_layer_kind_vqb1(model_dir, layer, kind, k=256, max_iter=12,
                            rows_per_expert=128, n_cols=None,
                            seed=42, out_path=None):
    """Encode one (layer, kind) tile to VQB1 file.
    Returns dict with stats; writes file if out_path given.
    """
    t0 = time.time()
    tensor = load_fp4_layer_kind(model_dir, layer, kind,
                                  n_rows_per_expert=rows_per_expert,
                                  n_cols=n_cols)
    t_load = time.time() - t0
    n_experts, n_rows, in_dim = tensor.shape
    n_pairs = in_dim // 2

    pairs = fp4_to_pairs(tensor)
    t1 = time.time()
    codebook, codes, mse = lloyd_max_2d(pairs, k=k, max_iter=max_iter, seed=seed)
    t_encode = time.time() - t1

    # Quality check
    decoded_pairs = codebook[codes]
    decoded = decoded_pairs.reshape(n_experts, n_rows, n_pairs, 2).reshape(*tensor.shape)
    rel_l2 = float(np.linalg.norm(decoded.ravel() - tensor.ravel()) /
                   max(np.linalg.norm(tensor.ravel()), 1e-12))
    cos_sim = float(np.dot(decoded.ravel(), tensor.ravel()) /
                    (np.linalg.norm(decoded.ravel()) * np.linalg.norm(tensor.ravel()) + 1e-12))

    # Reshape codes to per-expert layout [E, R, P] for canonical file write
    codes_layout = codes.reshape(n_experts, n_rows, n_pairs).astype(np.uint8)

    if out_path is not None:
        # Write VQB1 file
        with open(out_path, "wb") as f:
            f.write(VQB1_MAGIC)
            f.write(struct.pack("<IIIIIII",
                                  1,  # version
                                  n_experts, n_rows, n_pairs,
                                  layer, KIND_NAME_TO_ID[kind], k))
            # Codebook
            cb_fp32 = codebook.astype(np.float32)
            f.write(cb_fp32.tobytes())
            # Codes (E*R*P uint8)
            f.write(codes_layout.tobytes())

    return {
        "layer": layer, "kind": kind, "k": k,
        "n_experts": n_experts, "n_rows": n_rows, "n_pairs": n_pairs,
        "in_dim": in_dim,
        "rel_l2": rel_l2, "cos_sim": cos_sim,
        "mse": float(mse),
        "load_sec": float(t_load),
        "encode_sec": float(t_encode),
        "total_sec": float(time.time() - t0),
        "out_path": str(out_path) if out_path else None,
        "file_size": (VQB1_HEADER_BYTES + k * 2 * 4 + n_experts * n_rows * n_pairs) if out_path else None,
    }


def read_vqb1_header(path):
    """Read VQB1 header without loading payload."""
    with open(path, "rb") as f:
        head = f.read(VQB1_HEADER_BYTES)
    if head[:4] != VQB1_MAGIC:
        raise RuntimeError(f"{path}: bad VQB1 magic {head[:4]}")
    fields = struct.unpack("<IIIIIII", head[4:])
    version, n_experts, n_rows, n_pairs, layer, kind_id, k = fields
    return {
        "magic": "VQB1", "version": version,
        "n_experts": n_experts, "n_rows": n_rows, "n_pairs": n_pairs,
        "layer": layer, "kind_id": kind_id, "k": k,
    }


def read_vqb1_full(path):
    """Read entire VQB1 into memory. Returns (header, codebook[K,2], codes[E,R,P])."""
    with open(path, "rb") as f:
        head_bytes = f.read(VQB1_HEADER_BYTES)
        if head_bytes[:4] != VQB1_MAGIC:
            raise RuntimeError(f"{path}: bad VQB1 magic")
        version, n_experts, n_rows, n_pairs, layer, kind_id, k = \
            struct.unpack("<IIIIIII", head_bytes[4:])
        cb_bytes = f.read(k * 2 * 4)
        codes_bytes = f.read(n_experts * n_rows * n_pairs)
    codebook = np.frombuffer(cb_bytes, dtype=np.float32).reshape(k, 2)
    codes = np.frombuffer(codes_bytes, dtype=np.uint8).reshape(n_experts, n_rows, n_pairs)
    header = {
        "version": version, "n_experts": n_experts, "n_rows": n_rows,
        "n_pairs": n_pairs, "layer": layer, "kind_id": kind_id, "k": k,
    }
    return header, codebook, codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir",
                    default="/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash")
    ap.add_argument("--layers", default="0", help="comma-separated or 'all'")
    ap.add_argument("--kinds", default="gate,up,down")
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--rows-per-expert", type=int, default=128)
    ap.add_argument("--max-iter", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True,
                    help="dir for L{LL}_{kind}.vqb1 files")
    ap.add_argument("--verify-readback", action="store_true",
                    help="open back each file and verify codebook+codes shape match")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = list(range(43)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    kinds = [k.strip() for k in args.kinds.split(",")]

    print(f"VQB1 encode: K={args.k}, layers={layers}, kinds={kinds}, out_dir={out_dir}",
          file=sys.stderr)

    grand_t0 = time.time()
    results = []
    for L in layers:
        for K in kinds:
            out_path = out_dir / f"L{L:02d}_{K}.vqb1"
            r = encode_layer_kind_vqb1(args.model_dir, L, K, k=args.k,
                                         max_iter=args.max_iter,
                                         rows_per_expert=args.rows_per_expert,
                                         seed=args.seed, out_path=out_path)
            results.append(r)
            print(f"  L{L:02d} {K:>4}  rel_L2={r['rel_l2']:.4f} cos_sim={r['cos_sim']:.4f} "
                  f"file={r['file_size']/1e6:.1f} MB  ({r['total_sec']:.1f}s)",
                  file=sys.stderr)

            if args.verify_readback:
                hdr, cb, codes = read_vqb1_full(out_path)
                assert cb.shape == (args.k, 2), f"codebook shape {cb.shape}"
                assert codes.shape == (r["n_experts"], r["n_rows"], r["n_pairs"]), \
                    f"codes shape {codes.shape}"
                print(f"    verify OK: cb {cb.shape}  codes {codes.shape}", file=sys.stderr)

    grand = time.time() - grand_t0
    rels = [r["rel_l2"] for r in results]
    coss = [r["cos_sim"] for r in results]
    total_bytes = sum(r["file_size"] for r in results)
    print(f"\n# Aggregate: {len(results)} files in {grand:.1f}s", file=sys.stderr)
    print(f"#   rel_L2:  mean={np.mean(rels):.4f}  min={min(rels):.4f}  max={max(rels):.4f}",
          file=sys.stderr)
    print(f"#   cos_sim: mean={np.mean(coss):.4f}  min={min(coss):.4f}",
          file=sys.stderr)
    print(f"#   total bytes: {total_bytes/1e9:.2f} GB", file=sys.stderr)

    # Manifest
    with open(out_dir / "manifest.json", "w") as f:
        json.dump({
            "encoder": "vq2d_encode_vqb1.py",
            "model_dir": args.model_dir,
            "k": args.k, "rows_per_expert": args.rows_per_expert,
            "max_iter": args.max_iter, "seed": args.seed,
            "n_files": len(results),
            "elapsed_seconds": grand,
            "rel_l2_mean": float(np.mean(rels)),
            "cos_sim_mean": float(np.mean(coss)),
            "total_bytes": total_bytes,
            "files": [{"layer": r["layer"], "kind": r["kind"],
                        "rel_l2": r["rel_l2"], "cos_sim": r["cos_sim"],
                        "file_size": r["file_size"]} for r in results],
        }, f, indent=2)
    print(f"\n#   wrote {out_dir}/manifest.json", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
