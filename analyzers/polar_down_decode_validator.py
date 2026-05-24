#!/usr/bin/env python3
"""Validate polar-DOWN decode matches FP4-source down weights.

Prereq for Phase B-2.3 Option D (extend H1735 MSL kernel to read polar-down).
Before writing the kernel, confirm the polar pool's down weights round-trip
to the same fp32 values the FP4 source would yield.

Workflow:
  1. Load polar-down PLR2 for (layer, expert).
  2. Decode the full row × pair matrix to fp32 via polar codec.
  3. Load FP4 down weights from safetensors for same (layer, expert).
  4. Dequantize FP4 → fp32 (same path the encoder used as input).
  5. Compare per-row rel_L2 + cos_sim.

If the per-row rel_L2 matches the codec floor (~0.20 at p16_m4), the
encoder did the right thing for down weights — same as for gate/up.
If it's significantly worse, there's a down-specific bug to fix
BEFORE the kernel work lands.

Output: per-row rel_L2 + cos_sim table + aggregate stats.
"""
import argparse
import json
import math
import struct
import sys
import time
import mmap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from polar_encode_bulk import dequant_fp4_tensor, FP4_BLOCK_SIZE


PLR2_MAGIC = b"PLR2"
PLR2_HEADER_BYTES = 64


def load_plr2_expert(path, expert_idx):
    """Load (layer, kind) PLR2 file, return (header, mag, phase, levels) for expert."""
    with open(path, "rb") as f:
        header_bytes = f.read(PLR2_HEADER_BYTES)
        rest = f.read()
    if header_bytes[:4] != PLR2_MAGIC:
        raise RuntimeError(f"bad magic in {path}")
    version, n_experts, n_rows, n_pairs, layer, kind_id, phase_levels, mag_levels = \
        struct.unpack("<IIIIIIII", header_bytes[4:36])
    if phase_levels == 0: phase_levels = 8
    if mag_levels == 0: mag_levels = 4
    if expert_idx >= n_experts:
        raise RuntimeError(f"expert {expert_idx} >= n_experts {n_experts}")
    mag_bytes_per_expert = n_rows * n_pairs
    phase_bytes_per_expert = n_rows * n_pairs
    levels_bytes_per_expert = n_rows * mag_levels * 4
    mag_off = expert_idx * mag_bytes_per_expert
    phase_off = n_experts * mag_bytes_per_expert + expert_idx * phase_bytes_per_expert
    levels_off = 2 * n_experts * mag_bytes_per_expert + expert_idx * levels_bytes_per_expert
    mag = np.frombuffer(rest[mag_off:mag_off + mag_bytes_per_expert], dtype=np.uint8).reshape(n_rows, n_pairs)
    phase = np.frombuffer(rest[phase_off:phase_off + phase_bytes_per_expert], dtype=np.uint8).reshape(n_rows, n_pairs)
    levels = np.frombuffer(rest[levels_off:levels_off + levels_bytes_per_expert], dtype=np.float32).reshape(n_rows, mag_levels)
    return {
        "version": version, "n_experts": n_experts, "n_rows": n_rows, "n_pairs": n_pairs,
        "layer": layer, "kind_id": kind_id, "phase_levels": phase_levels, "mag_levels": mag_levels,
        "mag": mag, "phase": phase, "levels": levels,
    }


def polar_decode_to_fp32(plr2, n_rows=None):
    """Reconstruct fp32 matrix [n_rows, in_dim] from polar codes."""
    P = plr2["phase_levels"]
    M = plr2["mag_levels"]
    phase_step = 2.0 * math.pi / P
    half_P = P // 2
    rows = n_rows if n_rows is not None else plr2["n_rows"]
    mag = plr2["mag"][:rows]
    phase = plr2["phase"][:rows]
    levels = plr2["levels"][:rows]
    # qmag[r,p] = levels[r, mag[r,p]]
    qmag = np.take_along_axis(levels, mag.astype(np.int64), axis=-1)
    # qangle[r,p] = (phase[r,p] - half_P) * phase_step
    qangle = (phase.astype(np.int64) - half_P) * phase_step
    qre = qmag * np.cos(qangle)
    qim = qmag * np.sin(qangle)
    # Interleave to [rows, n_pairs * 2 = in_dim]
    out = np.empty((rows, qmag.shape[1] * 2), dtype=np.float32)
    out[:, 0::2] = qre
    out[:, 1::2] = qim
    return out


def load_fp4_down(model_dir, layer, expert, n_rows):
    """Load FP4 down for (layer, expert), dequantize to fp32, return [n_rows, in_dim]."""
    from safetensors import safe_open
    md = Path(model_dir)
    wm = json.load(open(md / "model.safetensors.index.json")).get("weight_map", {})
    wname = f"layers.{layer}.ffn.experts.{expert}.w2.weight"
    sname = f"layers.{layer}.ffn.experts.{expert}.w2.scale"
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
    full = dequant_fp4_tensor(w_u8, s_u8)
    return full[:n_rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash")
    ap.add_argument("--polar-dir", required=True,
                    help="dir containing L{LL}_down.polar files")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--experts", default="0,5,100,200,255",
                    help="comma-separated expert IDs to test")
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    down_path = Path(args.polar_dir) / f"L{args.layer:02d}_down.polar"
    if not down_path.exists():
        print(f"ERR: {down_path} not found", file=sys.stderr)
        return 1

    expert_ids = [int(x) for x in args.experts.split(",")]
    results = []
    grand_t0 = time.time()
    for expert in expert_ids:
        t0 = time.time()
        plr2 = load_plr2_expert(str(down_path), expert)
        plr_decode = polar_decode_to_fp32(plr2, n_rows=args.rows)
        t_dec = time.time() - t0

        t1 = time.time()
        fp4_decoded = load_fp4_down(args.model_dir, args.layer, expert, args.rows)
        t_fp4 = time.time() - t1

        # Compare per-row
        rows = min(plr_decode.shape[0], fp4_decoded.shape[0])
        plr = plr_decode[:rows].astype(np.float64)
        fp4 = fp4_decoded[:rows].astype(np.float64)
        rel_l2 = np.linalg.norm(plr - fp4, axis=-1) / np.maximum(np.linalg.norm(fp4, axis=-1), 1e-12)
        cos = np.einsum('rk,rk->r', plr, fp4) / np.maximum(
            np.linalg.norm(plr, axis=-1) * np.linalg.norm(fp4, axis=-1), 1e-12)

        entry = {
            "layer": args.layer, "expert": expert,
            "polar_codec": f"p{plr2['phase_levels']}_m{plr2['mag_levels']}",
            "rows_compared": int(rows),
            "in_dim": int(plr.shape[1]),
            "rel_l2_mean": float(rel_l2.mean()),
            "rel_l2_max":  float(rel_l2.max()),
            "rel_l2_p95":  float(np.percentile(rel_l2, 95)),
            "cos_sim_mean": float(cos.mean()),
            "cos_sim_min":  float(cos.min()),
            "decode_polar_s": float(t_dec),
            "load_fp4_s": float(t_fp4),
        }
        results.append(entry)
        print(f"  L{args.layer:02d} E{expert:>3}  rel_L2 mean={entry['rel_l2_mean']:.4f} "
              f"max={entry['rel_l2_max']:.4f}  cos mean={entry['cos_sim_mean']:.4f} "
              f"min={entry['cos_sim_min']:.4f}  ({t_dec:.1f}s polar + {t_fp4:.1f}s fp4)",
              file=sys.stderr)

    grand = time.time() - grand_t0
    rels = [r["rel_l2_mean"] for r in results]
    print(f"\nAggregate across {len(results)} experts:", file=sys.stderr)
    print(f"  rel_L2 mean: {np.mean(rels):.4f}  min: {min(rels):.4f}  max: {max(rels):.4f}",
          file=sys.stderr)
    print(f"  Total wall: {grand:.1f}s", file=sys.stderr)
    if all(r["rel_l2_mean"] < 0.35 for r in results):
        print("\n  VERDICT: polar-down decode is consistent with FP4 source ✓", file=sys.stderr)
        print("  (rel_L2 < 0.35 = within expected codec noise for p16_m4)", file=sys.stderr)
    else:
        print("\n  VERDICT: polar-down decode has anomalies — investigate before Phase B-2.3 kernel work",
              file=sys.stderr)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({
                "model_dir": args.model_dir,
                "polar_dir": args.polar_dir,
                "layer": args.layer,
                "experts": expert_ids,
                "rows": args.rows,
                "results": results,
            }, f, indent=2)
        print(f"\n  wrote {args.out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
