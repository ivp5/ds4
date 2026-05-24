#!/usr/bin/env python3
"""Test A — polar resolution sweep on real DS4 V4 expert weights.

Standalone probe per refinement_oom_accuracy_speedup.md.

For each (phase_levels, mag_levels) pair, encode a slice of L0 gate
weights and measure cos_sim + rel_L2 against the original FP4-dequantized
weights. Outputs a JSON Pareto table; the elbow tells silv where the
operating point lives on the (storage_bytes_per_pair × √error) frontier.

Codec parameters explored:
  phase_levels ∈ {4, 8, 16, 32, 64, 128} — angular bins, powers of 2
  mag_levels   ∈ {2, 4, 8, 16}           — magnitude code levels

Storage cost: bits_per_pair = log2(phase_levels) + log2(mag_levels).
  p8_m2  (current PLR2 default): 6 bits/pair  (3+2 — phase signed code uses
                                                4 bits 0..8 in PLR2 but the
                                                effective info is log2(9))
  p16_m4: 7 bits/pair
  p32_m4: 8 bits/pair
  p32_m8: 10 bits/pair
  p64_m8: 11 bits/pair
  p128_m16: 13 bits/pair

For each (P, M) the per-row codebook is M quartile-percentile means.
Phase is rounded to the nearest 2π/P bin.

CPU-only — no GPU kernel needed at higher resolution. Validates the
accuracy/storage tradeoff before committing to a higher-res GPU kernel.

Usage:
  python3.14 analyzers/polar_resolution_sweep.py \\
      --model-dir /Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash \\
      --layer 0 --n-experts 32 --rows 128 \\
      --out-json tmp/polar_resolution_sweep_L0.json
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# Reuse the bulk encoder's FP4 dequant
sys.path.insert(0, str(Path(__file__).resolve().parent))
from polar_encode_bulk import dequant_fp4_tensor, FP4_BLOCK_SIZE


def encode_polar_pN_mM(rows_fp32, phase_levels, mag_levels):
    """Encode rows_fp32 [n_rows, in_dim] at p{phase_levels}_m{mag_levels}.

    Returns:
      qrows fp32 [n_rows, in_dim] — decoded reconstruction
      bits_per_pair float — storage cost
    """
    n_rows, in_dim = rows_fp32.shape
    pairs = in_dim // 2
    re = rows_fp32[:, 0::2]
    im = rows_fp32[:, 1::2]
    mag = np.sqrt(re * re + im * im)
    angle = np.arctan2(im, re)

    # Phase quantize: P bins, codes ∈ {0, .., P-1}, decoded angle = 2π * code / P - π.
    phase_step = 2.0 * math.pi / phase_levels
    phase_codes = np.mod(np.rint(angle / phase_step).astype(np.int32), phase_levels)
    qangle = phase_codes.astype(np.float64) * phase_step
    # Wrap qangle back into [-π, π] for stable cos/sin (not strictly necessary
    # since cos/sin are 2π-periodic).
    # Mag quantize: M codes via quartile-percentile means (matches polar_encode_mlx).
    sorted_idx = np.argsort(mag, axis=-1, kind='stable')
    sorted_mag = np.take_along_axis(mag, sorted_idx, axis=-1)
    levels = np.zeros((n_rows, mag_levels), dtype=np.float64)
    mag_codes = np.zeros_like(mag, dtype=np.int32)
    for k in range(mag_levels):
        a = k * pairs // mag_levels
        b = (k + 1) * pairs // mag_levels
        levels[:, k] = sorted_mag[:, a:b].mean(axis=-1)
        idx_slice = sorted_idx[:, a:b]
        rows_arange = np.arange(n_rows)[:, None]
        mag_codes[rows_arange, idx_slice] = k
    qmag = np.take_along_axis(levels, mag_codes.astype(np.int64), axis=-1)
    qre = qmag * np.cos(qangle)
    qim = qmag * np.sin(qangle)
    qrows = np.empty_like(rows_fp32, dtype=np.float64)
    qrows[:, 0::2] = qre
    qrows[:, 1::2] = qim
    bits_per_pair = math.log2(phase_levels) + math.log2(mag_levels)
    return qrows.astype(np.float32), bits_per_pair


def measure(rows_fp32, qrows):
    """Return (cos_sim, rel_L2) — both ndarrays [n_rows]."""
    rows = rows_fp32.astype(np.float64)
    q = qrows.astype(np.float64)
    rn = np.linalg.norm(rows, axis=-1)
    qn = np.linalg.norm(q, axis=-1)
    cos = np.einsum('rk,rk->r', rows, q) / np.maximum(rn * qn, 1e-12)
    rel = np.linalg.norm(q - rows, axis=-1) / np.maximum(rn, 1e-12)
    return cos.astype(np.float32), rel.astype(np.float32)


def load_layer_gate(model_dir, layer, n_experts, rows):
    """Load L{layer} gate weights for first n_experts. Returns
    [n_experts, rows, in_dim] fp32 after FP4 dequant."""
    import json as _json
    md = Path(model_dir)
    idx_path = md / "model.safetensors.index.json"
    if not idx_path.exists():
        raise RuntimeError(f"no safetensors index at {idx_path}")
    wm = _json.load(open(idx_path)).get("weight_map", {})
    from safetensors import safe_open
    blobs = []
    for e in range(n_experts):
        wname = f"layers.{layer}.ffn.experts.{e}.w1.weight"
        sname = f"layers.{layer}.ffn.experts.{e}.w1.scale"
        wshard = wm[wname]
        sshard = wm[sname]
        with safe_open(str(md / wshard), framework="numpy", device="cpu") as f:
            w = f.get_tensor(wname)
        w_u8 = w.view(np.uint8) if w.dtype != np.uint8 else w
        # Scale via torch (E8M0)
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
        blobs.append(full[:rows])
    return np.stack(blobs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--n-experts", type=int, default=32)
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--phase-levels", default="4,8,16,32,64,128")
    ap.add_argument("--mag-levels", default="2,4,8,16")
    args = ap.parse_args()

    Ps = [int(x) for x in args.phase_levels.split(",")]
    Ms = [int(x) for x in args.mag_levels.split(",")]

    print(f"polar_sweep: loading L{args.layer} gate, {args.n_experts} experts × {args.rows} rows",
          file=sys.stderr)
    t0 = time.time()
    weights = load_layer_gate(args.model_dir, args.layer, args.n_experts, args.rows)
    print(f"  loaded in {time.time()-t0:.1f}s  shape={weights.shape}", file=sys.stderr)

    results = []
    for P in Ps:
        for M in Ms:
            t1 = time.time()
            cos_acc = []
            rel_acc = []
            for e in range(args.n_experts):
                qrows, bpp = encode_polar_pN_mM(weights[e], P, M)
                cos, rel = measure(weights[e], qrows)
                cos_acc.append(cos)
                rel_acc.append(rel)
            cos_all = np.concatenate(cos_acc)
            rel_all = np.concatenate(rel_acc)
            elapsed = time.time() - t1
            entry = {
                "phase_levels": P,
                "mag_levels": M,
                "bits_per_pair": bpp,
                "cos_sim_mean": float(cos_all.mean()),
                "cos_sim_min": float(cos_all.min()),
                "cos_sim_p05": float(np.percentile(cos_all, 5)),
                "rel_l2_mean": float(rel_all.mean()),
                "rel_l2_max": float(rel_all.max()),
                "rel_l2_p95": float(np.percentile(rel_all, 95)),
                "n_rows_sampled": int(cos_all.shape[0]),
                "elapsed_s": elapsed,
            }
            results.append(entry)
            print(f"  p{P:>3}_m{M:>2}  bits={bpp:>4.1f}  cos_mean={entry['cos_sim_mean']:.4f}  "
                  f"rel_L2_mean={entry['rel_l2_mean']:.4f}  ({elapsed:.1f}s)",
                  file=sys.stderr)

    # Build Pareto: (bits_per_pair, rel_l2_mean) — minimize both
    points = [(r["bits_per_pair"], r["rel_l2_mean"], r) for r in results]
    points.sort()
    pareto = []
    best_rel = float("inf")
    for bpp, rel, r in points:
        if rel < best_rel:
            pareto.append(r)
            best_rel = rel
    print(f"\npolar_sweep: Pareto frontier ({len(pareto)} points):", file=sys.stderr)
    for r in pareto:
        print(f"  p{r['phase_levels']:>3}_m{r['mag_levels']:>2}  "
              f"bits={r['bits_per_pair']:>4.1f}  "
              f"rel_L2_mean={r['rel_l2_mean']:.4f}  "
              f"cos_sim_mean={r['cos_sim_mean']:.4f}",
              file=sys.stderr)

    with open(args.out_json, "w") as f:
        json.dump({
            "model_dir": args.model_dir,
            "layer": args.layer,
            "n_experts": args.n_experts,
            "rows": args.rows,
            "results": results,
            "pareto": pareto,
        }, f, indent=2)
    print(f"\npolar_sweep: wrote {args.out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
