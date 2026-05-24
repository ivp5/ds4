#!/usr/bin/env python3
"""Encode a real DS4 expert's gate weight as p8_m2 polar binary files.

Per codex H1717-H1726: polar p8_m2 (8-level signed phase + 4-level magnitude
per adjacent-channel pair = 2.5 bits/scalar) beats IQ2/Q2 on routed
dot-output. Per H1727 the deployable layout is route-tiled with one hidden
vector per tile + 32 output rows; this script emits the simpler per-packet
layout matched to the canary kernel for first-pass validation.

Input:
  Trim50 GGUF (default: /Users/silv/cl/tlp/montyneg/ds4/gguf/DS4-trim50-asym-with-metadata.gguf)
  Layer index + tensor name + expert index
Output (binary files, same dir as --out):
  <out>.mag.bin       packets×pairs uint8     mag code (0..3)
  <out>.phase.bin     packets×pairs uint8     phase code (0..8, signed→[-4,4] + 4)
  <out>.levels.bin    packets×4    float32    per-packet magnitude levels (quantile-mean)
  <out>.hidden.bin    packets×pairs×2 float32 synthetic hidden vector (same shape as
                                              expected per-packet hidden) — used by the
                                              canary kernel for the dot product
  <out>.expected.bin  packets        float32  expected polar dot vs the synthetic hidden
                                              (computed in numpy from the quantized weight)
  <out>.meta.json                              packets, pairs, source layer/expert/kind

Caller:
  python3 analyzers/polar_encode_expert.py --layer 0 --expert 0 --kind gate \\
      --out tmp/polar_l0_e0_gate
"""
import argparse
import importlib.util
import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np

# Borrow codex's dequant + polar encoder. They share the same lineage with
# my h1714 reading of the GGUF format.
CODEX_HERE = Path('/Users/silv/cl/tlp_codex/research/llm_fallacy_deconstruction/framework_deconstruction')
for mod_name, mod_path in [
    ('h1714', CODEX_HERE / 'h1714_ds4_real_gguf_row_quant_canary_20260525.py'),
    ('h1724', CODEX_HERE / 'h1724_ds4_mtl4_polar_packet_kernel_20260525.py'),
]:
    if not mod_path.exists():
        print(f"polar_encode: missing codex helper {mod_path}", file=sys.stderr)
        sys.exit(2)
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    globals()[mod_name] = mod


def find_tensor(gguf_path, name):
    """Return (offset, dims, ggml_type) for the named tensor."""
    with open(gguf_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            raise ValueError(f"not a GGUF file: {gguf_path}")
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        # Skip metadata
        for _ in range(n_kv):
            klen = struct.unpack('<Q', f.read(8))[0]
            f.read(klen)
            vt = struct.unpack('<I', f.read(4))[0]
            _ = h1714.read_value(f, vt) if hasattr(h1714, 'read_value') else _skip_value(f, vt)
        # Tensor index
        tensors = []
        for _ in range(n_tensors):
            nlen = struct.unpack('<Q', f.read(8))[0]
            tname = f.read(nlen).decode()
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            ggml_type = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensors.append((tname, dims, ggml_type, offset))
        # Find data start (aligned)
        cur = f.tell()
        ALIGN = 32
        data_start = (cur + ALIGN - 1) & ~(ALIGN - 1)
        for tname, dims, ggml_type, off in tensors:
            if tname == name:
                return data_start + off, dims, ggml_type, tensors
        return None, None, None, tensors


def _skip_value(f, vt):
    if vt == 8:  # string
        slen = struct.unpack('<Q', f.read(8))[0]
        f.read(slen)
    elif vt == 9:  # array
        et = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            _skip_value(f, et)
    elif vt in (0, 1):  # u8/i8
        f.read(1)
    elif vt in (2, 3):  # u16/i16
        f.read(2)
    elif vt in (4, 5, 6, 7):  # u32/i32/f32/bool
        f.read(4)
    elif vt in (10, 11, 12):  # u64/i64/f64
        f.read(8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gguf', default='/Users/silv/cl/tlp/montyneg/ds4/gguf/DS4-trim50-asym-with-metadata.gguf')
    ap.add_argument('--layer', type=int, default=0)
    ap.add_argument('--expert', type=int, default=0)
    ap.add_argument('--kind', choices=('gate', 'up'), default='gate')
    ap.add_argument('--rows', type=int, default=32,
                    help='how many rows of the expert weight to encode as packets (default 32)')
    ap.add_argument('--hidden-seed', type=int, default=20260525)
    ap.add_argument('--out', required=True, help='output prefix')
    args = ap.parse_args()

    tensor_name = f"blk.{args.layer}.ffn_{args.kind}_exps.weight"
    print(f"polar_encode: gguf={args.gguf}", file=sys.stderr)
    print(f"polar_encode: tensor={tensor_name}", file=sys.stderr)

    offset, dims, ggml_type, _ = find_tensor(args.gguf, tensor_name)
    if offset is None:
        print(f"polar_encode: tensor not found: {tensor_name}", file=sys.stderr)
        sys.exit(3)

    print(f"polar_encode: tensor dims={dims} ggml_type={ggml_type} offset={offset}", file=sys.stderr)
    # Dims convention: gate/up exps are [hidden, ff_exp, n_experts_kept_for_layer]
    # (column-major). For polar encoding we want ROWS of (hidden, ff_exp).
    # The codex helper packs per-block already.

    if ggml_type != 16:  # IQ2_XXS
        print(f"polar_encode: ggml_type {ggml_type} not IQ2_XXS; aborting", file=sys.stderr)
        sys.exit(4)

    # Dims for DS4 trim50: ffn_gate_exps weight is typically [N_EMBD, N_FF_EXP, n_kept]
    # = [4096, 2048, n_kept_at_this_layer]. We want N rows of length 4096 elements each,
    # for one specific expert. Convention: row r of expert e is at byte offset
    #   offset + (e * n_kept_stride_bytes) + r * row_bytes_one_row
    # IQ2_XXS row of 4096 elements = 16 blocks of 256 elements × 66 bytes = 1056 bytes.

    # Reconstruct shape. Many GGUFs reverse dim order.
    if len(dims) == 3:
        d0, d1, d2 = dims  # likely n_experts, hidden, ff_exp OR hidden, ff_exp, n_experts
    else:
        raise ValueError(f"unexpected dims: {dims}")

    # Per the trim50 file: blk.0.ffn_gate_exps.weight has dims [n_experts, hidden, ff_exp]
    # in GGUF storage order (last dim varies fastest in mmap). For IQ2_XXS, the "row"
    # is dim[-1] elements wide.
    row_elements = d0  # ggml convention: row width = first listed dim
    n_rows_per_expert = d1
    n_experts_in_file = d2 if len(dims) == 3 else 1

    # row_elements should be 4096; n_rows_per_expert = 2048; n_experts varies.
    print(f"polar_encode: row_elements={row_elements} n_rows_per_expert={n_rows_per_expert} n_experts={n_experts_in_file}", file=sys.stderr)

    if row_elements % 256 != 0:
        print(f"polar_encode: row width {row_elements} not multiple of 256 (IQ2 block size)", file=sys.stderr)
        sys.exit(5)
    blocks_per_row = row_elements // 256
    bytes_per_row = blocks_per_row * 66

    if args.expert >= n_experts_in_file:
        print(f"polar_encode: expert {args.expert} >= {n_experts_in_file}", file=sys.stderr)
        sys.exit(6)
    if args.rows > n_rows_per_expert:
        print(f"polar_encode: rows {args.rows} > {n_rows_per_expert}", file=sys.stderr)
        sys.exit(7)

    # Load the IQ2_XXS signed-grid LUT once (cached). Reads the tables from ds4.c.
    DS4_C = '/Users/silv/cl/tlp/montyneg/ivp5_ds4/ds4.c'
    signed_grid = h1714.load_iq2_tables(DS4_C)

    # Read the requested rows.
    rows_fp32 = []
    with open(args.gguf, 'rb') as f:
        for r in range(args.rows):
            row_offset = offset + args.expert * n_rows_per_expert * bytes_per_row + r * bytes_per_row
            f.seek(row_offset)
            row_bytes = f.read(bytes_per_row)
            if len(row_bytes) != bytes_per_row:
                print(f"polar_encode: short read for row {r}", file=sys.stderr)
                sys.exit(8)
            row_fp32 = h1714.dequant_iq2_xxs_row(row_bytes, signed_grid)
            rows_fp32.append(row_fp32)
    rows_fp32 = np.stack(rows_fp32)  # [packets, scalars]
    print(f"polar_encode: dequantized rows shape={rows_fp32.shape} dtype={rows_fp32.dtype}", file=sys.stderr)

    # Per-packet polar encode using codex's primitive
    packets = rows_fp32.shape[0]
    pairs = rows_fp32.shape[1] // 2
    mag_codes = np.empty((packets, pairs), dtype=np.uint8)
    phase_codes = np.empty((packets, pairs), dtype=np.uint8)
    levels = np.empty((packets, 4), dtype=np.float32)
    approx = np.empty_like(rows_fp32, dtype=np.float64)
    for p in range(packets):
        mc, pc, lv, ap = h1724.polar_encode_p8_m2(rows_fp32[p].astype(np.float64))
        mag_codes[p] = mc
        phase_codes[p] = pc
        levels[p] = lv
        approx[p] = ap

    # Deterministic synthetic hidden vector
    rng = np.random.default_rng(args.hidden_seed)
    hidden = rng.standard_normal(rows_fp32.shape).astype(np.float32)
    # Expected polar dot per packet (reference, computed in fp64)
    expected_polar = (approx * hidden).sum(axis=1).astype(np.float32)
    # Also compute the "ground truth" IQ2-dequant dot for comparison
    expected_iq2 = (rows_fp32.astype(np.float64) * hidden).sum(axis=1).astype(np.float32)

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    mag_codes.astype(np.uint8).tofile(str(out_prefix) + '.mag.bin')
    phase_codes.astype(np.uint8).tofile(str(out_prefix) + '.phase.bin')
    levels.astype(np.float32).tofile(str(out_prefix) + '.levels.bin')
    hidden.astype(np.float32).tofile(str(out_prefix) + '.hidden.bin')
    expected_polar.astype(np.float32).tofile(str(out_prefix) + '.expected.bin')
    expected_iq2.astype(np.float32).tofile(str(out_prefix) + '.iq2.bin')

    meta = {
        'gguf': str(args.gguf),
        'tensor': tensor_name,
        'layer': args.layer,
        'expert': args.expert,
        'kind': args.kind,
        'packets': packets,
        'pairs': pairs,
        'row_elements': row_elements,
        'rel_l2_polar_vs_iq2': float(np.linalg.norm(approx - rows_fp32) / np.linalg.norm(rows_fp32)),
        'max_abs_dot_diff_polar_vs_iq2': float(np.abs(expected_polar - expected_iq2).max()),
        'mean_abs_dot_diff_polar_vs_iq2': float(np.abs(expected_polar - expected_iq2).mean()),
    }
    with open(str(out_prefix) + '.meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"polar_encode: wrote packets={packets} pairs={pairs}", file=sys.stderr)
    print(f"polar_encode: rel_L2(polar vs iq2)={meta['rel_l2_polar_vs_iq2']:.6f}", file=sys.stderr)
    print(f"polar_encode: max|dot_polar - dot_iq2|={meta['max_abs_dot_diff_polar_vs_iq2']:.6e}", file=sys.stderr)
    print(f"polar_encode: mean|dot_polar - dot_iq2|={meta['mean_abs_dot_diff_polar_vs_iq2']:.6e}", file=sys.stderr)
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
