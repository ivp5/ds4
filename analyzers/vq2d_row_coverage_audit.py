#!/usr/bin/env python3
"""Test whether 128-row codec finding generalizes to full row coverage.

Two tests:
1. Fit codebook on first N rows, measure rel_L2 on different row ranges
   of the SAME expert. If consistent → 128-row testing is fine.
2. Fit per-chunk codebooks vs single shared, compare quality.

If the FP4-dequant pair distribution is consistent across rows of the
same weight tensor (likely, since FP4 has only 16 dequant values per
scale), then K=256 codebook should work identically on rows 0-128
and rows 3000-3128. The "row coverage" concern would dissolve.

If per-row scale variation creates per-row codebook drift, the
testing-scope finding doesn't generalize without per-row codebooks.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from polar_encode_bulk import dequant_fp4_tensor
from vq2d_codec_explore import lloyd_max_2d, assign_codes


def load_full_expert_kind(model_dir, layer, expert, kind):
    """Load FULL FP4-decoded weight for one (layer, expert, kind)."""
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
    return dequant_fp4_tensor(w_u8, s_u8).astype(np.float32)


def rel_l2(orig, recon):
    return float(np.linalg.norm(recon.ravel() - orig.ravel()) /
                 max(np.linalg.norm(orig.ravel()), 1e-12))


def cos_sim(a, b):
    fa, fb = a.ravel(), b.ravel()
    return float(np.dot(fa, fb) / (np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-12))


def encode_chunk_vq(chunk, k=256, max_iter=12, seed=42):
    """Encode one chunk [R, C] as pairs → fit + assign + decode."""
    pairs = chunk.reshape(-1, 2)
    cb, codes, mse = lloyd_max_2d(pairs, k=k, max_iter=max_iter, seed=seed)
    decoded = cb[codes].reshape(*chunk.shape)
    return decoded, cb, codes


def assign_with_codebook(chunk, codebook):
    """Assign pairs in chunk to nearest centroid in given codebook."""
    pairs = chunk.reshape(-1, 2)
    codes = assign_codes(pairs, codebook)
    return codebook[codes].reshape(*chunk.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir",
                    default="/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--expert", type=int, default=0)
    ap.add_argument("--kind", choices=["gate", "up", "down"], default="down")
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--max-iter", type=int, default=12)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    print(f"Loading full {args.kind} for L{args.layer} E{args.expert}...", file=sys.stderr)
    full = load_full_expert_kind(args.model_dir, args.layer, args.expert, args.kind)
    print(f"  shape {full.shape}  total bytes {full.nbytes/1e6:.1f} MB", file=sys.stderr)

    R, C = full.shape
    # Test 1: fit codebook on first 128 rows, apply to multiple row-ranges
    chunk_size = 128
    print(f"\nTest 1: codebook fit on rows[0:128], applied to multiple ranges", file=sys.stderr)
    print(f"  fitting codebook on rows[0:{chunk_size}]...", file=sys.stderr)
    t0 = time.time()
    _, cb_first, _ = encode_chunk_vq(full[:chunk_size], k=args.k, max_iter=args.max_iter)
    print(f"  fit complete in {time.time()-t0:.1f}s", file=sys.stderr)

    # Test rows[0:128], rows[1024:1152], rows[R-128:R]
    ranges_to_test = []
    if R >= 256:  ranges_to_test.append((0, 128))
    if R >= 1152: ranges_to_test.append((1024, 1152))
    if R >= 256:  ranges_to_test.append((R-128, R))

    test1_results = []
    for r0, r1 in ranges_to_test:
        chunk = full[r0:r1]
        recon = assign_with_codebook(chunk, cb_first)
        rl = rel_l2(chunk, recon)
        cs = cos_sim(chunk, recon)
        test1_results.append({"r0": r0, "r1": r1, "rel_l2": rl, "cos_sim": cs})
        print(f"  rows[{r0:>5}:{r1:>5}]: rel_L2={rl:.4f}  cos_sim={cs:.4f}", file=sys.stderr)

    # Test 2: per-chunk codebook for each 128-row chunk vs shared single
    print(f"\nTest 2: per-chunk codebook vs shared on rows[0:128] / [1024:1152] / [{R-128}:{R}]", file=sys.stderr)
    test2_results = []
    for r0, r1 in ranges_to_test:
        chunk = full[r0:r1]
        per_chunk_recon, _, _ = encode_chunk_vq(chunk, k=args.k, max_iter=args.max_iter)
        per_rl = rel_l2(chunk, per_chunk_recon)
        shared_recon = assign_with_codebook(chunk, cb_first)
        sh_rl = rel_l2(chunk, shared_recon)
        test2_results.append({
            "r0": r0, "r1": r1,
            "per_chunk_rel_l2": per_rl,
            "shared_rel_l2": sh_rl,
            "shared_vs_per": sh_rl / max(per_rl, 1e-12),
        })
        print(f"  rows[{r0:>5}:{r1:>5}]: per-chunk rel_L2={per_rl:.4f}  shared rel_L2={sh_rl:.4f}  "
              f"shared/per ratio={sh_rl/max(per_rl,1e-12):.3f}", file=sys.stderr)

    # Test 3: full-tensor codebook (fit on ALL rows) vs first-128 codebook
    print(f"\nTest 3: full-tensor codebook ({R} rows) vs first-128 codebook on full tensor", file=sys.stderr)
    t0 = time.time()
    full_recon, cb_full, _ = encode_chunk_vq(full, k=args.k, max_iter=args.max_iter)
    full_rl = rel_l2(full, full_recon)
    print(f"  full-fit codebook ({time.time()-t0:.1f}s): rel_L2={full_rl:.4f}", file=sys.stderr)

    # Apply first-128 codebook to full tensor
    full_via_first128 = assign_with_codebook(full, cb_first)
    via_first_rl = rel_l2(full, full_via_first128)
    print(f"  first-128 codebook on full: rel_L2={via_first_rl:.4f}  "
          f"shared/full ratio={via_first_rl/max(full_rl,1e-12):.3f}", file=sys.stderr)

    test3 = {
        "full_fit_rel_l2": full_rl,
        "first128_on_full_rel_l2": via_first_rl,
        "ratio_first128_to_full": via_first_rl / max(full_rl, 1e-12),
    }

    # Verdict
    print(f"\n# Verdict:", file=sys.stderr)
    ratios = [r["shared_vs_per"] for r in test2_results]
    if max(ratios) < 1.2 and test3["ratio_first128_to_full"] < 1.2:
        print(f"  CODEBOOK GENERALIZES: shared/per ratio max {max(ratios):.3f}, "
              f"first128/full ratio {test3['ratio_first128_to_full']:.3f}", file=sys.stderr)
        print(f"  → 128-row testing IS representative of full-coverage codec quality", file=sys.stderr)
        print(f"  → SCOPE CAVEAT can be retired: deployment at FIRST128 codebook + per-chunk", file=sys.stderr)
        print(f"     codes covers full rows at same quality as testing", file=sys.stderr)
    else:
        print(f"  CODEBOOK DOES NOT GENERALIZE: ratios {ratios}, "
              f"first128/full ratio {test3['ratio_first128_to_full']:.3f}", file=sys.stderr)
        print(f"  → 128-row testing OVER-estimates codec quality at full scope", file=sys.stderr)
        print(f"  → Need per-chunk codebooks OR row-region-specific codebooks", file=sys.stderr)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({
                "model_dir": args.model_dir,
                "layer": args.layer, "expert": args.expert, "kind": args.kind,
                "k": args.k, "tensor_shape": list(full.shape),
                "test1_shared_codebook_across_rows": test1_results,
                "test2_per_chunk_vs_shared": test2_results,
                "test3_full_fit_vs_first128": test3,
            }, f, indent=2)
        print(f"\n  wrote {args.out_json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
