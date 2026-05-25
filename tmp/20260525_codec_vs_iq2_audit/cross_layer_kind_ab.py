"""Cross-(layer, kind) extension of codec_quality_ab.py.

Test the "VQ dominates IQ2" claim on multiple (layer, kind, expert) combos
to check if the single-cell result generalizes.
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


def measure_cell(iq2_gguf, fp4_dir, polar_dir, vq_dir,
                  meta_cache, layer, kind, expert):
    """Return dict of rel_L2 for each codec on (layer, kind, expert)."""
    meta, tinfo, data_start = meta_cache
    target_name = f"blk.{layer}.ffn_{kind}_exps.weight"
    info = next((t for t in tinfo if t["name"] == target_name), None)
    if info is None:
        return {"error": f"no tensor {target_name}"}

    n_total = int(np.prod(info["dims"]))
    weights_per_expert = n_total // 256
    blocks_per_expert = weights_per_expert // 256
    bytes_per_expert = blocks_per_expert * 66

    # 1. IQ2_XXS slice for this expert
    f = open(iq2_gguf, "rb")
    f.seek(data_start + info["offset"] + expert * bytes_per_expert)
    iq2_bytes = f.read(bytes_per_expert)
    f.close()
    iq2_fp32 = dequant_iq2_xxs_fast(iq2_bytes, weights_per_expert)

    # 2. FP4 source for this expert
    try:
        fp4_full, _, _ = load_fp4_source_expert(fp4_dir, layer=layer, kind=kind, expert=expert)
        fp4_flat = fp4_full.flatten()
    except Exception as e:
        return {"error": f"FP4 load: {type(e).__name__}: {e}"}

    if fp4_flat.size != iq2_fp32.size:
        return {"error": f"size mismatch fp4={fp4_flat.size} iq2={iq2_fp32.size}"}

    # 3. IQ2 vs FP4 on full expert
    diff_iq2 = iq2_fp32 - fp4_flat
    rel_l2_iq2_full = float(np.linalg.norm(diff_iq2) / max(np.linalg.norm(fp4_flat), 1e-30))

    # 4. polar / VQ corpora encode only first 128 rows. Slice for fair comparison.
    # out_dim (rows) × in_dim — depends on kind. For gate/up: (2048, 4096). For down: (4096, 2048).
    # IQ2 tensor has layout matching the GGUF tensor dims. For DS4 V4:
    #   gate_exps: dims = [in_dim=4096, out_dim=2048, n_experts=256], so per expert is (2048, 4096).
    # FP4 expert shape is (out_dim, in_dim).
    fp4_2d = fp4_flat.reshape(fp4_full.shape)  # use FP4's actual shape
    iq2_2d = iq2_fp32.reshape(fp4_full.shape)

    # Slice first 128 OUTPUT rows (the polar/VQ corpus convention)
    n_rows_codec = 128
    fp4_first = fp4_2d[:n_rows_codec].flatten()
    iq2_first = iq2_2d[:n_rows_codec].flatten()
    diff_iq2_first = iq2_first - fp4_first
    rel_l2_iq2_first = float(np.linalg.norm(diff_iq2_first) / max(np.linalg.norm(fp4_first), 1e-30))

    # 5. polar p32_m8
    polar_path = polar_dir / f"L{layer:02d}_{kind}.polar"
    polar_rel_l2 = None
    polar_bpp = None
    if polar_path.exists():
        try:
            polar = read_polar_plr2(polar_path, expert=expert)
            polar_flat = polar['reconstructed'].flatten()
            if polar_flat.size == fp4_first.size:
                diff = polar_flat - fp4_first
                polar_rel_l2 = float(np.linalg.norm(diff) / max(np.linalg.norm(fp4_first), 1e-30))
                polar_bpp = float(polar['storage_bytes_per_pair'])
        except Exception as e:
            polar_rel_l2 = f"err: {type(e).__name__}"

    # 6. VQ K=256
    vq_path = vq_dir / f"L{layer:02d}_{kind}.vqb1"
    vq_rel_l2 = None
    vq_bpp = None
    if vq_path.exists():
        try:
            vq = read_vq_vqb1(vq_path, expert=expert)
            vq_flat = vq['reconstructed'].flatten()
            if vq_flat.size == fp4_first.size:
                diff = vq_flat - fp4_first
                vq_rel_l2 = float(np.linalg.norm(diff) / max(np.linalg.norm(fp4_first), 1e-30))
                vq_bpp = float(vq['storage_bytes_per_pair'])
        except Exception as e:
            vq_rel_l2 = f"err: {type(e).__name__}"

    return {
        "layer": layer, "kind": kind, "expert": expert,
        "fp4_shape": list(fp4_full.shape),
        "iq2_full_rel_l2": rel_l2_iq2_full,
        "iq2_first128_rel_l2": rel_l2_iq2_first,
        "iq2_bytes_per_pair": 0.5156,
        "polar_rel_l2": polar_rel_l2,
        "polar_bytes_per_pair": polar_bpp,
        "vq_rel_l2": vq_rel_l2,
        "vq_bytes_per_pair": vq_bpp,
    }


def main():
    root = Path("/Users/silv/cl/tlp/montyneg")
    iq2_gguf = root / "ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
    fp4_dir = root / "ds4/gguf/DeepSeek-V4-Flash"
    polar_dir = root / "ivp5_ds4/tmp/polar_full_p32m8"
    vq_dir = root / "ivp5_ds4/tmp/vqb1_L0_k256_mlx"  # only has L0; others get N/A

    print("Reading GGUF metadata once...")
    meta_cache = _read_gguf_meta(iq2_gguf)
    print(f"  {len(meta_cache[1])} tensors loaded")

    cells = [
        # (layer, kind, expert) — mix early/mid/late layers, multiple experts, all 3 kinds
        (0, "gate", 0),
        (0, "gate", 100),
        (0, "gate", 255),
        (0, "up", 0),
        (0, "down", 0),
        (5, "gate", 0),
        (10, "gate", 0),
        (20, "gate", 0),
        (42, "gate", 0),  # last layer
    ]
    results = []
    print(f"\n{'layer':>5}  {'kind':<6}  {'exp':>4}  {'iq2_full':>9}  {'iq2_128':>8}  {'polar':>8}  {'vq':>8}")
    print(f"  {'-'*5}  {'-'*6}  {'-'*4}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*8}")
    for layer, kind, expert in cells:
        t0 = time.time()
        r = measure_cell(iq2_gguf, fp4_dir, polar_dir, vq_dir, meta_cache, layer, kind, expert)
        elapsed = time.time() - t0
        if "error" in r:
            print(f"  {layer:>5}  {kind:<6}  {expert:>4}  ERROR: {r['error']}")
            results.append(r)
            continue
        polar_s = f"{r['polar_rel_l2']:.4f}" if isinstance(r['polar_rel_l2'], float) else str(r['polar_rel_l2'])[:8] if r['polar_rel_l2'] is not None else "N/A"
        vq_s = f"{r['vq_rel_l2']:.4f}" if isinstance(r['vq_rel_l2'], float) else str(r['vq_rel_l2'])[:8] if r['vq_rel_l2'] is not None else "N/A"
        print(f"  {layer:>5}  {kind:<6}  {expert:>4}  "
              f"{r['iq2_full_rel_l2']:>9.4f}  {r['iq2_first128_rel_l2']:>8.4f}  "
              f"{polar_s:>8}  {vq_s:>8}    ({elapsed:.1f}s)")
        results.append(r)

    # Summary
    print(f"\n=== Summary ===")
    valid_iq2 = [r for r in results if "error" not in r and isinstance(r.get("iq2_full_rel_l2"), float)]
    valid_vq = [r for r in valid_iq2 if isinstance(r.get("vq_rel_l2"), float)]
    valid_polar = [r for r in valid_iq2 if isinstance(r.get("polar_rel_l2"), float)]

    if valid_iq2:
        iq2_avg = np.mean([r["iq2_full_rel_l2"] for r in valid_iq2])
        iq2_min = min(r["iq2_full_rel_l2"] for r in valid_iq2)
        iq2_max = max(r["iq2_full_rel_l2"] for r in valid_iq2)
        print(f"  IQ2_XXS rel_L2 (full expert): mean={iq2_avg:.4f}  range=[{iq2_min:.4f}, {iq2_max:.4f}]  n={len(valid_iq2)}")
    if valid_vq:
        vq_avg = np.mean([r["vq_rel_l2"] for r in valid_vq])
        ratios = [r["iq2_first128_rel_l2"] / r["vq_rel_l2"] for r in valid_vq if r["vq_rel_l2"] > 0]
        ratio_avg = np.mean(ratios)
        print(f"  VQ K=256 rel_L2 (first 128 rows): mean={vq_avg:.4f}  n={len(valid_vq)}")
        print(f"  IQ2/VQ ratio (first 128): mean={ratio_avg:.2f}× (VQ this much better)")
    if valid_polar:
        polar_avg = np.mean([r["polar_rel_l2"] for r in valid_polar])
        ratios = [r["iq2_first128_rel_l2"] / r["polar_rel_l2"] for r in valid_polar if r["polar_rel_l2"] > 0]
        ratio_avg = np.mean(ratios)
        print(f"  polar p32_m8 rel_L2 (first 128 rows): mean={polar_avg:.4f}  n={len(valid_polar)}")
        print(f"  IQ2/polar ratio (first 128): mean={ratio_avg:.2f}× (polar this much better)")

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = Path(f"cross_layer_kind_{ts}.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
