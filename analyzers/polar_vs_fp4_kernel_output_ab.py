#!/usr/bin/env python3
"""Phase B-2.3b kernel-output codec quality measurement.

Mirrors ds4_gpu_mtl4_polar_real_canary's CPU computation (which we
already verified via B-2.3a is bit-equivalent to GPU at fp32 noise
floor) and runs it TWICE — once with polar-decoded down, once with
FP4-source down. The difference is the codec quality at the kernel
output level.

WHY this works without C-side FP4 loading:
- B-2.3a proved GPU(polar gate + polar-decoded fp32 down) ≈ CPU(polar
  gate + polar-decoded fp32 down) at rel_err 1e-7.
- The H1735 kernel doesn't care if the fp32 down came from polar
  decode or FP4 decode — it just consumes fp32 tiles.
- So GPU(polar gate + FP4-decoded fp32 down) ≈ CPU(polar gate +
  FP4-decoded fp32 down) at the same fp32 noise floor.
- The codec-quality question is: |CPU_polar - CPU_FP4| / |CPU_FP4|.
  Pure numpy can answer it.

Composes with B-2.3a — does NOT replace the GPU canary; the canary
proved the kernel is correct, this proves the codec is acceptable.

Output: per-(layer, expert) cos_sim + rel_L2 between out_polar and
out_fp4. Aggregate stats across cells. The expected codec floor is
~0.05 (the validator's rel_L2 0.20 weight-level error projects down
via the sum-over-r averaging in the H1735 dot product).
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
from polar_down_decode_validator import load_plr2_expert


def polar_decode_gate_dot_rows(plr2_gate, expert_unused, n_rows):
    """Mirror C-side gate decode: gate_dot[r] = sum_p (re(r,p) + im(r,p)).

    expert_unused — already baked into plr2_gate via load_plr2_expert.
    Returns gate_dot[n_rows] fp32.
    """
    P = plr2_gate["phase_levels"]
    M = plr2_gate["mag_levels"]
    phase_step = 2.0 * math.pi / P
    half_P = P // 2
    rows = min(n_rows, plr2_gate["n_rows"])
    mag = plr2_gate["mag"][:rows]
    phase = plr2_gate["phase"][:rows]
    levels = plr2_gate["levels"][:rows]
    qmag = np.take_along_axis(levels, mag.astype(np.int64), axis=-1)
    qangle = (phase.astype(np.int64) - half_P) * phase_step
    re = qmag * np.cos(qangle)
    im = qmag * np.sin(qangle)
    # gate_dot[r] = sum over pairs of (re + im)
    return (re.sum(axis=-1) + im.sum(axis=-1)).astype(np.float32)


def polar_decode_down_tile(plr2_down, down_rows, act_rows):
    """Mirror C-side polar-down extraction:
    dwn[d * act_rows + a] where:
      a = 2*k   → re of pair k for row d
      a = 2*k+1 → im of pair k for row d
    Returns [down_rows, act_rows] fp32.
    """
    P = plr2_down["phase_levels"]
    M = plr2_down["mag_levels"]
    phase_step = 2.0 * math.pi / P
    half_P = P // 2
    needed_pairs = (act_rows + 1) // 2
    mag = plr2_down["mag"][:down_rows, :needed_pairs]
    phase = plr2_down["phase"][:down_rows, :needed_pairs]
    levels = plr2_down["levels"][:down_rows]
    qmag = np.take_along_axis(levels, mag.astype(np.int64), axis=-1)
    qangle = (phase.astype(np.int64) - half_P) * phase_step
    re = qmag * np.cos(qangle)
    im = qmag * np.sin(qangle)
    out = np.empty((down_rows, act_rows), dtype=np.float32)
    out[:, 0::2] = re[:, :(act_rows + 1) // 2]
    if act_rows >= 2:
        out[:, 1::2] = im[:, :act_rows // 2]
    return out


def load_fp4_expert(model_dir, layer, expert, kind, n_rows=None, n_cols=None):
    """Load FP4-source weight tile for (layer, expert, kind).
    kind ∈ {"gate","up","down"} → safetensors name maps to {gate→w1, up→w3, down→w2}.
    """
    from safetensors import safe_open
    md = Path(model_dir)
    wm = json.load(open(md / "model.safetensors.index.json")).get("weight_map", {})
    kind_to_w = {"gate": "w1", "up": "w3", "down": "w2"}
    w_suffix = kind_to_w[kind]
    wname = f"layers.{layer}.ffn.experts.{expert}.{w_suffix}.weight"
    sname = f"layers.{layer}.ffn.experts.{expert}.{w_suffix}.scale"
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
    if n_rows is not None:
        full = full[:n_rows]
    if n_cols is not None:
        full = full[:, :n_cols]
    return full


def compute_canary_outputs(polar_dir, model_dir, layer, expert, down_rows, act_rows):
    """Mirror canary computation TWICE:
    out_polar[d] = sum_r act[r] * polar_dwn[d, r]
    out_fp4[d]   = sum_r act[r] * fp4_dwn[d, r]
    Returns dict with both outputs + comparison stats.

    act[r] is computed from polar-decoded gate (canary uses gate==up).
    """
    pd = Path(polar_dir)
    gate_path = pd / f"L{layer:02d}_gate.polar"
    down_path = pd / f"L{layer:02d}_down.polar"
    plr2_gate = load_plr2_expert(str(gate_path), expert)
    plr2_down = load_plr2_expert(str(down_path), expert)

    # 1. Gate-side activation: mirror canary's silu(g)*g for each row
    gate_dot = polar_decode_gate_dot_rows(plr2_gate, expert, act_rows)
    # silu(x) = x / (1 + e^-x); act = silu(g) * g
    silu = gate_dot / (1.0 + np.exp(-gate_dot.astype(np.float64))).astype(np.float32)
    act = silu * gate_dot  # since gate==up in canary

    # 2. Polar-decoded down tile
    polar_dwn = polar_decode_down_tile(plr2_down, down_rows, act_rows)

    # 3. FP4-source down tile
    fp4_dwn = load_fp4_expert(model_dir, layer, expert, "down",
                                n_rows=down_rows, n_cols=act_rows)
    fp4_dwn = fp4_dwn.astype(np.float32)

    # 4. Per-d outputs (matches canary's CPU expected loop)
    out_polar = (act[None, :] * polar_dwn).sum(axis=-1)
    out_fp4   = (act[None, :] * fp4_dwn).sum(axis=-1)

    # 5. Codec quality at output level
    diff = out_polar - out_fp4
    abs_err = np.abs(diff)
    norm_fp4 = np.linalg.norm(out_fp4) + 1e-12
    rel_l2 = np.linalg.norm(diff) / norm_fp4
    cos_sim = float(np.dot(out_polar, out_fp4) /
                    (np.linalg.norm(out_polar) * np.linalg.norm(out_fp4) + 1e-12))

    # Also report weight-level codec quality on this tile for cross-check
    wdiff = polar_dwn.ravel() - fp4_dwn.ravel()
    weight_rel_l2 = float(np.linalg.norm(wdiff) / (np.linalg.norm(fp4_dwn.ravel()) + 1e-12))
    weight_cos_sim = float(np.dot(polar_dwn.ravel(), fp4_dwn.ravel()) /
                           (np.linalg.norm(polar_dwn.ravel()) * np.linalg.norm(fp4_dwn.ravel()) + 1e-12))

    return {
        "layer": layer, "expert": expert,
        "down_rows": down_rows, "act_rows": act_rows,
        "polar_codec": f"p{plr2_gate['phase_levels']}_m{plr2_gate['mag_levels']}",
        "out_polar_first": float(out_polar[0]),
        "out_fp4_first":   float(out_fp4[0]),
        "abs_err_max":     float(abs_err.max()),
        "rel_l2":          float(rel_l2),
        "cos_sim":         cos_sim,
        "weight_rel_l2":   weight_rel_l2,
        "weight_cos_sim":  weight_cos_sim,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--polar-dir", required=True)
    ap.add_argument("--model-dir",
                    default="/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash")
    ap.add_argument("--layers", default="0,22,42",
                    help="comma-separated layer indices")
    ap.add_argument("--experts", default="0,100,255",
                    help="comma-separated expert indices")
    ap.add_argument("--down-rows", type=int, default=8)
    ap.add_argument("--act-rows", type=int, default=16)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    experts = [int(x) for x in args.experts.split(",")]

    print(f"# polar_vs_fp4_kernel_output_ab — polar_dir={args.polar_dir}", file=sys.stderr)
    print(f"# layers={layers} experts={experts} down_rows={args.down_rows} act_rows={args.act_rows}",
          file=sys.stderr)

    results = []
    t0 = time.time()
    for L in layers:
        for E in experts:
            r = compute_canary_outputs(args.polar_dir, args.model_dir, L, E,
                                        args.down_rows, args.act_rows)
            results.append(r)
            print(f"  L{L:02d} E{E:>3}  out_polar[0]={r['out_polar_first']:+.4f}  "
                  f"out_fp4[0]={r['out_fp4_first']:+.4f}  "
                  f"abs_err_max={r['abs_err_max']:.3e}  "
                  f"out_rel_L2={r['rel_l2']:.4f} cos_sim={r['cos_sim']:.4f}  "
                  f"(weight rel_L2={r['weight_rel_l2']:.4f} cos={r['weight_cos_sim']:.4f})",
                  file=sys.stderr)
    wall = time.time() - t0

    out_rel_l2s = [r["rel_l2"] for r in results]
    weight_rel_l2s = [r["weight_rel_l2"] for r in results]
    out_cos_sims = [r["cos_sim"] for r in results]
    weight_cos_sims = [r["weight_cos_sim"] for r in results]
    print(f"\n# Aggregate across {len(results)} cells:", file=sys.stderr)
    print(f"#  out rel_L2:  mean={np.mean(out_rel_l2s):.4f}  min={min(out_rel_l2s):.4f}  max={max(out_rel_l2s):.4f}",
          file=sys.stderr)
    print(f"#  out cos_sim: mean={np.mean(out_cos_sims):.4f}  min={min(out_cos_sims):.4f}",
          file=sys.stderr)
    print(f"#  weight rel_L2: mean={np.mean(weight_rel_l2s):.4f}  (validator headline = 0.2018)",
          file=sys.stderr)
    print(f"#  weight cos_sim: mean={np.mean(weight_cos_sims):.4f}", file=sys.stderr)
    print(f"#  wall: {wall:.1f}s", file=sys.stderr)

    # Verdict: kernel-output codec rel_L2 should compose linearly with weight rel_L2
    # — i.e., be in the same order. If output rel_L2 >> weight rel_L2, the kernel
    # amplifies codec error (alarm). If output rel_L2 << weight rel_L2, the sum-over-r
    # averaging is helping (expected for dot products).
    ratio = np.mean(out_rel_l2s) / max(np.mean(weight_rel_l2s), 1e-12)
    print(f"\n#  output_rel_L2 / weight_rel_L2 = {ratio:.2f}", file=sys.stderr)
    if ratio < 1.5:
        print("#  VERDICT: kernel does not amplify codec error (acceptable for B-2.3c hot-path) ✓",
              file=sys.stderr)
    else:
        print("#  VERDICT: kernel amplifies codec error by ≥1.5× — investigate before B-2.3c",
              file=sys.stderr)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({
                "polar_dir": args.polar_dir, "model_dir": args.model_dir,
                "layers": layers, "experts": experts,
                "down_rows": args.down_rows, "act_rows": args.act_rows,
                "results": results,
                "aggregate": {
                    "out_rel_l2_mean": float(np.mean(out_rel_l2s)),
                    "out_cos_sim_mean": float(np.mean(out_cos_sims)),
                    "weight_rel_l2_mean": float(np.mean(weight_rel_l2s)),
                    "weight_cos_sim_mean": float(np.mean(weight_cos_sims)),
                    "ratio_output_to_weight": float(ratio),
                    "wall_sec": float(wall),
                },
            }, f, indent=2)
        print(f"\n# wrote {args.out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
