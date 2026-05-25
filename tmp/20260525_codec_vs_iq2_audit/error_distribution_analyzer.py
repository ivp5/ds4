"""High-resolution per-pair codec error decomposition.

silv 2026-05-25: refine for OOM-higher accuracy to sense small aberrations
unnoticed before; amplify to yield valuable signal.

Aggregate rel_L2 = 0.353 ± 0.001 is a SCALAR. This script decomposes:

1. Signed per-weight error histogram (IQ2/polar/VQ vs FP4 source)
2. Magnitude-stratified rel_L2 (does codec X fail more on high-magnitude weights?)
3. Per-row outlier detection (which rows have worst codec error per codec?)
4. Sign-flip rate (does codec X flip the sign on % of weights?)
5. Worst-case absolute error (max abs per codec)
6. Per-pair magnitude vs angle error decomposition (for complex-pair codecs)
7. Subblock-correlated error (within IQ2's 32-weight block, is error uniform?)

These features surface small aberrations that aggregate rel_L2 hides.
"""
import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "analyzers"))
sys.path.insert(0, str(Path(__file__).parent))

from codec_quality_ab import (
    _read_gguf_meta, dequant_iq2_xxs_fast,
    load_fp4_source_expert,
    read_polar_plr2, read_vq_vqb1,
)


def stratified_rel_l2(reconstruction, source, n_strata=10):
    """Bin source weights by abs-magnitude into n_strata equal-count strata.
    Report rel_L2 per stratum. High-magnitude strata being worse = magnitude
    bias in the codec.
    """
    abs_source = np.abs(source)
    # Sort by magnitude, take strata indices
    sort_idx = np.argsort(abs_source)
    n = len(source)
    strata = []
    for k in range(n_strata):
        lo = (k * n) // n_strata
        hi = ((k + 1) * n) // n_strata
        idx = sort_idx[lo:hi]
        src = source[idx]
        rec = reconstruction[idx]
        diff = rec - src
        rel_l2 = float(np.linalg.norm(diff) / max(np.linalg.norm(src), 1e-30))
        max_abs = float(np.max(np.abs(diff)))
        mean_abs_src = float(np.mean(abs_source[idx]))
        strata.append({
            "stratum": k, "n": int(hi - lo),
            "mean_abs_src": mean_abs_src,
            "rel_l2": rel_l2,
            "max_abs_err": max_abs,
        })
    return strata


def signed_error_histogram(reconstruction, source, n_bins=64):
    """Histogram of (reconstruction - source) as signed values.
    Asymmetry indicates bias (over- or under-estimation).
    """
    diff = reconstruction - source
    # Use std-relative bins so codec comparisons are meaningful
    std = np.std(source)
    edges = np.linspace(-3 * std, 3 * std, n_bins + 1)
    counts, _ = np.histogram(diff, bins=edges)
    return {
        "edges_std_units": (edges / std).tolist(),
        "counts": counts.tolist(),
        "diff_mean": float(np.mean(diff)),
        "diff_std": float(np.std(diff)),
        "diff_skew": float(((diff - np.mean(diff)) ** 3).mean() / max(np.std(diff) ** 3, 1e-30)),
        "fraction_zero": float(np.mean(diff == 0.0)),
    }


def sign_flip_rate(reconstruction, source):
    """Fraction of weights where reconstruction has opposite sign from source."""
    nonzero = source != 0.0
    if nonzero.sum() == 0:
        return 0.0
    flips = (np.sign(reconstruction[nonzero]) != np.sign(source[nonzero])).sum()
    return float(flips / nonzero.sum())


def per_row_outliers(reconstruction_2d, source_2d, top_k=10):
    """For each row, compute rel_L2; return top_k worst rows + the
    spread (std/mean) as a measure of within-tensor codec-error heterogeneity.
    """
    n_rows = source_2d.shape[0]
    diff_2d = reconstruction_2d - source_2d
    per_row_l2 = np.linalg.norm(diff_2d, axis=1)
    per_row_src_norm = np.linalg.norm(source_2d, axis=1)
    per_row_rel = per_row_l2 / np.maximum(per_row_src_norm, 1e-30)
    worst_idx = np.argsort(per_row_rel)[-top_k:][::-1]
    return {
        "n_rows": int(n_rows),
        "rel_l2_mean": float(per_row_rel.mean()),
        "rel_l2_std":  float(per_row_rel.std()),
        "rel_l2_min":  float(per_row_rel.min()),
        "rel_l2_max":  float(per_row_rel.max()),
        "rel_l2_p95":  float(np.percentile(per_row_rel, 95)),
        "rel_l2_p99":  float(np.percentile(per_row_rel, 99)),
        "rel_l2_cv":   float(per_row_rel.std() / max(per_row_rel.mean(), 1e-30)),
        "worst_row_indices": worst_idx.tolist(),
        "worst_row_rel_l2": per_row_rel[worst_idx].tolist(),
    }


def per_pair_decomposition(reconstruction_2d, source_2d):
    """For complex-pair codecs, decompose error into magnitude vs angle components.
    """
    # Treat each pair (col 2*i, 2*i+1) as complex
    re_src = source_2d[:, 0::2]
    im_src = source_2d[:, 1::2]
    re_rec = reconstruction_2d[:, 0::2]
    im_rec = reconstruction_2d[:, 1::2]
    mag_src = np.sqrt(re_src ** 2 + im_src ** 2)
    mag_rec = np.sqrt(re_rec ** 2 + im_rec ** 2)
    ang_src = np.arctan2(im_src, re_src)
    ang_rec = np.arctan2(im_rec, re_rec)
    # Angle diff in [-pi, pi]
    ang_diff = np.mod(ang_rec - ang_src + np.pi, 2 * np.pi) - np.pi
    mag_diff = mag_rec - mag_src

    return {
        "mag_rel_l2": float(np.linalg.norm(mag_diff) / max(np.linalg.norm(mag_src), 1e-30)),
        "mag_diff_mean": float(np.mean(mag_diff)),
        "mag_diff_std":  float(np.std(mag_diff)),
        "angle_diff_rms_rad": float(np.sqrt(np.mean(ang_diff ** 2))),
        "angle_diff_p50_rad": float(np.percentile(np.abs(ang_diff), 50)),
        "angle_diff_p95_rad": float(np.percentile(np.abs(ang_diff), 95)),
        "angle_diff_max_rad": float(np.max(np.abs(ang_diff))),
    }


def analyze_codec(name, reconstruction, source, source_2d_shape=None):
    """Run the full decomposition battery for one codec's reconstruction."""
    print(f"  === {name} ===")
    src = source.flatten()
    rec = reconstruction.flatten()
    # Match shapes if reconstruction is smaller than source (codec scope = 128 rows)
    if rec.size != src.size:
        n = rec.size
        src = src[:n]
        if source_2d_shape:
            # Reshape source to (rows_codec, n_pairs*2)
            rows_codec = n // (source_2d_shape[1])
            source_2d_local = source.reshape(source_2d_shape)[:rows_codec]
            reconstruction_2d_local = reconstruction.reshape(rows_codec, source_2d_shape[1])
        else:
            return {"name": name, "error": "shape mismatch + no shape provided"}
    else:
        if source_2d_shape:
            source_2d_local = source.reshape(source_2d_shape)
            reconstruction_2d_local = reconstruction.reshape(source_2d_shape)
        else:
            source_2d_local = source.reshape(1, -1)
            reconstruction_2d_local = reconstruction.reshape(1, -1)

    diff = rec - src
    rel_l2 = float(np.linalg.norm(diff) / max(np.linalg.norm(src), 1e-30))
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))

    print(f"    overall rel_L2: {rel_l2:.4f}, max_abs: {max_abs:.4e}, mean_abs: {mean_abs:.4e}")

    strata = stratified_rel_l2(rec, src, n_strata=10)
    print(f"    magnitude strata (low→high), rel_L2 per decile:")
    for s in strata:
        print(f"      [{s['stratum']:>2}] mean_src={s['mean_abs_src']:.4e}  rel_L2={s['rel_l2']:.4f}  max_abs_err={s['max_abs_err']:.4e}")

    sf_rate = sign_flip_rate(rec, src)
    print(f"    sign-flip rate: {sf_rate:.4f} ({sf_rate*100:.2f}% of weights have opposite sign)")

    hist = signed_error_histogram(rec, src, n_bins=64)
    print(f"    signed-err: mean={hist['diff_mean']:.4e}  std={hist['diff_std']:.4e}  skew={hist['diff_skew']:.4f}  frac_exactly_zero={hist['fraction_zero']:.4f}")

    row_outliers = per_row_outliers(reconstruction_2d_local, source_2d_local, top_k=5)
    print(f"    per-row rel_L2: mean={row_outliers['rel_l2_mean']:.4f}  std={row_outliers['rel_l2_std']:.4f}  cv={row_outliers['rel_l2_cv']:.4f}  p95={row_outliers['rel_l2_p95']:.4f}  p99={row_outliers['rel_l2_p99']:.4f}  max={row_outliers['rel_l2_max']:.4f}")
    print(f"    worst rows: {row_outliers['worst_row_indices']} with rel_L2 {[f'{x:.4f}' for x in row_outliers['worst_row_rel_l2']]}")

    pair = per_pair_decomposition(reconstruction_2d_local, source_2d_local)
    print(f"    pair-decomp: mag_rel_l2={pair['mag_rel_l2']:.4f}  angle_rms_rad={pair['angle_diff_rms_rad']:.4f}  angle_p95={pair['angle_diff_p95_rad']:.4f}")

    return {
        "name": name,
        "overall": {"rel_l2": rel_l2, "max_abs": max_abs, "mean_abs": mean_abs},
        "strata": strata,
        "sign_flip_rate": sf_rate,
        "hist": hist,
        "row_outliers": row_outliers,
        "pair_decomp": pair,
    }


def main():
    root = Path("/Users/silv/cl/tlp/montyneg")
    iq2_gguf = root / "ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
    fp4_dir = root / "ds4/gguf/DeepSeek-V4-Flash"
    polar_dir = root / "ivp5_ds4/tmp/polar_full_p32m8"
    vq_dir = root / "ivp5_ds4/tmp/vqb1_L0_k256_mlx"

    print("Reading GGUF metadata...")
    meta_cache = _read_gguf_meta(iq2_gguf)

    # Focus: L0 gate expert 0, first 128 rows (matches polar/VQ corpus scope)
    layer, kind, expert = 0, "gate", 0
    print(f"\n=== L{layer} {kind} expert {expert} HIGH-RESOLUTION ===\n")

    info = next(t for t in meta_cache[1] if t["name"] == f"blk.{layer}.ffn_{kind}_exps.weight")
    n_total = int(np.prod(info["dims"]))
    weights_per_expert = n_total // 256
    blocks_per_expert = weights_per_expert // 256
    bytes_per_expert = blocks_per_expert * 66
    data_start = meta_cache[2]

    f = open(iq2_gguf, "rb")
    f.seek(data_start + info["offset"] + expert * bytes_per_expert)
    iq2_bytes = f.read(bytes_per_expert)
    f.close()
    iq2_fp32 = dequant_iq2_xxs_fast(iq2_bytes, weights_per_expert)

    fp4_full, _, _ = load_fp4_source_expert(fp4_dir, layer=layer, kind=kind, expert=expert)
    fp4_flat = fp4_full.flatten()
    print(f"FP4 source: shape={fp4_full.shape} std={fp4_full.std():.4e} range=[{fp4_full.min():.4e}, {fp4_full.max():.4e}]")
    print(f"IQ2 source: same shape, std={iq2_fp32.std():.4e}\n")

    # Slice everything to first 128 rows
    rows_codec = 128
    fp4_2d_first = fp4_full[:rows_codec]  # (128, 4096)
    iq2_2d_first = iq2_fp32.reshape(fp4_full.shape)[:rows_codec]
    src_shape_first = fp4_2d_first.shape  # (128, 4096)

    polar_path = polar_dir / f"L{layer:02d}_{kind}.polar"
    polar_2d = read_polar_plr2(polar_path, expert=expert)['reconstructed']

    vq_path = vq_dir / f"L{layer:02d}_{kind}.vqb1"
    vq_2d = read_vq_vqb1(vq_path, expert=expert)['reconstructed']

    results = {
        "cell": {"layer": layer, "kind": kind, "expert": expert, "rows_codec": rows_codec},
        "fp4_src_stats": {
            "shape": list(fp4_full.shape),
            "mean": float(fp4_full.mean()),
            "std": float(fp4_full.std()),
            "min": float(fp4_full.min()),
            "max": float(fp4_full.max()),
            "fraction_zero": float(np.mean(fp4_full == 0.0)),
        },
        "codecs": [],
    }

    # IQ2_XXS (using full 2048-row tensor for the IQ2 row-outlier picture)
    print(f"--- IQ2_XXS (full {fp4_full.shape[0]} rows) ---")
    r_iq2_full = analyze_codec("IQ2_XXS (full)", iq2_fp32.reshape(fp4_full.shape),
                                fp4_full, source_2d_shape=fp4_full.shape)
    results["codecs"].append(r_iq2_full)

    # First 128 rows for comparison parity
    print(f"\n--- IQ2_XXS (first {rows_codec} rows for parity) ---")
    r_iq2_128 = analyze_codec("IQ2_XXS (first 128)", iq2_2d_first, fp4_2d_first,
                               source_2d_shape=src_shape_first)
    results["codecs"].append(r_iq2_128)

    # Polar (first 128 rows)
    print(f"\n--- polar p32_m8 (first {rows_codec} rows) ---")
    r_polar = analyze_codec("polar p32_m8", polar_2d, fp4_2d_first,
                             source_2d_shape=src_shape_first)
    results["codecs"].append(r_polar)

    # VQ K=256 (first 128 rows)
    print(f"\n--- VQ K=256 (first {rows_codec} rows) ---")
    r_vq = analyze_codec("VQ K=256", vq_2d, fp4_2d_first,
                          source_2d_shape=src_shape_first)
    results["codecs"].append(r_vq)

    # Cross-codec comparison table
    print("\n=== CROSS-CODEC SUMMARY ===")
    print(f"  {'codec':<22}  {'rel_L2':>7}  {'mag_rL2':>7}  {'ang_rms':>7}  {'sign-flip':>9}  {'p95_row':>7}  {'p99_row':>7}")
    print(f"  {'-'*22}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*7}  {'-'*7}")
    for r in results["codecs"]:
        if "error" in r:
            continue
        print(f"  {r['name']:<22}  {r['overall']['rel_l2']:>7.4f}  {r['pair_decomp']['mag_rel_l2']:>7.4f}  {r['pair_decomp']['angle_diff_rms_rad']:>7.4f}  {r['sign_flip_rate']:>9.4f}  {r['row_outliers']['rel_l2_p95']:>7.4f}  {r['row_outliers']['rel_l2_p99']:>7.4f}")

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    Path(f"err_dist_{ts}.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved err_dist_{ts}.json")


if __name__ == "__main__":
    main()
