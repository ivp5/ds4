"""Block-VQ + Product-Quantization sweep — encode N-pair blocks as single index.

silv 2026-05-25: "work on block kernel as well - compare all high potential"

Variants tested:

A. Block-VQ: encode N-pair block (= 2N weights) as ONE log2(K)-bit index.
   - K-means in R^(2N), K codebook entries of dim 2N.
   - Storage: log2(K)/N bits per pair + codebook cost
   - Decode: lookup codebook[index] = 2N floats → fp32 weights

B. Product-Quantization (PQ): split N-pair block into M sub-blocks of N/M pairs,
   each independently VQ-encoded with K_sub codebook.
   - Storage: M × log2(K_sub) bits per block = M × log2(K_sub) / N bits per pair
   - Decode: M sub-lookups, concat
   - PQ is the standard Faiss trick — separable, fast, often near-optimal at fixed bit budget

C. Block-VQ with per-block scale: encode normalized block + 1 scale byte
   - Adds 8/N_pair bits to storage but covers dynamic range across blocks

Block sizes tested: 4-pair (8 weights), 8-pair (16 weights), 16-pair (32 weights).
The 16-pair = 32-weight matches IQ2_XXS sub-block structure.
"""
import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path("/Users/silv/cl/tlp/montyneg/ivp5_ds4/analyzers")))
sys.path.insert(0, str(Path(__file__).parent))

from polar_encode_bulk import dequant_fp4_tensor


def kmeans_nd(vectors, K, max_iter=20, seed=42):
    """K-means on n-dimensional vectors. Returns (codebook K×D, codes N)."""
    rng = np.random.default_rng(seed)
    N, D = vectors.shape
    if K >= N:
        K = N
    init_idx = rng.choice(N, size=K, replace=False)
    codebook = vectors[init_idx].copy()
    prev_inertia = float("inf")
    for it in range(max_iter):
        chunk = max(10_000, 1_000_000 // max(D, 1))
        codes = np.empty(N, dtype=np.int32)
        inertia = 0.0
        for i in range(0, N, chunk):
            sub = vectors[i:i+chunk]
            d = np.sum((sub[:, None, :] - codebook[None, :, :]) ** 2, axis=-1)
            codes[i:i+chunk] = np.argmin(d, axis=-1)
            inertia += d[np.arange(d.shape[0]), codes[i:i+chunk]].sum()
        new_cb = np.zeros_like(codebook)
        for k in range(K):
            mask = codes == k
            if mask.sum() > 0:
                new_cb[k] = vectors[mask].mean(axis=0)
            else:
                new_cb[k] = codebook[k]
        if it > 0 and abs(prev_inertia - inertia) / max(prev_inertia, 1e-30) < 1e-5:
            break
        codebook = new_cb
        prev_inertia = inertia
    return codebook, codes


def encode_block_vq(pairs_flat, n_pairs_block, K, with_scale=False, name=""):
    """Block-VQ: encode each n_pairs_block block as one log2(K)-bit index."""
    pairs = pairs_flat.reshape(-1, 2)
    n_pairs = pairs.shape[0]
    n_blocks = n_pairs // n_pairs_block
    block_dim = 2 * n_pairs_block  # weights per block
    blocks = pairs[:n_blocks * n_pairs_block].reshape(n_blocks, block_dim)

    if with_scale:
        scale = np.linalg.norm(blocks, axis=-1).reshape(-1, 1)
        scale_safe = np.maximum(scale, 1e-30)
        blocks_unit = blocks / scale_safe
        cb, codes = kmeans_nd(blocks_unit, K)
        # Quantize scale to fp16 (could use u8 with log-encoding for less space)
        scale_q = scale.astype(np.float16).astype(np.float32)  # fp16 round-trip
        recon_unit = cb[codes]
        recon = recon_unit * scale_q
    else:
        cb, codes = kmeans_nd(blocks, K)
        recon = cb[codes]
        scale_q = None

    n_kept = recon.shape[0] * recon.shape[1]
    diff = recon - blocks
    rel_l2 = float(np.linalg.norm(diff) / max(np.linalg.norm(blocks), 1e-30))

    bits_index = np.log2(K)
    bits_scale = 16.0 if with_scale else 0.0
    bits_per_pair = (bits_index + bits_scale) / n_pairs_block
    cb_bytes = K * block_dim * 4  # fp32 codebook

    return {
        "name": name, "n_pairs_block": n_pairs_block, "K": K, "with_scale": with_scale,
        "rel_l2": rel_l2,
        "bits_per_pair": float(bits_per_pair),
        "byte_per_pair": float(bits_per_pair / 8),
        "codebook_bytes": int(cb_bytes),
    }


def encode_product_quant(pairs_flat, n_pairs_block, n_subblocks, K_sub, name=""):
    """Product Quantization: split each n_pairs_block block into n_subblocks sub-blocks,
    each VQ-encoded with K_sub codebook."""
    pairs = pairs_flat.reshape(-1, 2)
    n_pairs = pairs.shape[0]
    assert n_pairs_block % n_subblocks == 0
    pairs_per_sub = n_pairs_block // n_subblocks
    sub_dim = 2 * pairs_per_sub
    n_blocks = n_pairs // n_pairs_block
    blocks = pairs[:n_blocks * n_pairs_block].reshape(n_blocks, n_subblocks, sub_dim)

    # Encode each sub-block index independently using all blocks' sub-vectors at that index
    codebooks = []
    codes_all = []
    recon = np.empty_like(blocks)
    for s in range(n_subblocks):
        sub_vecs = blocks[:, s, :]  # (n_blocks, sub_dim)
        cb, codes = kmeans_nd(sub_vecs, K_sub)
        codebooks.append(cb)
        codes_all.append(codes)
        recon[:, s, :] = cb[codes]

    diff = recon - blocks
    rel_l2 = float(np.linalg.norm(diff) / max(np.linalg.norm(blocks), 1e-30))

    bits_per_block = n_subblocks * np.log2(K_sub)
    bits_per_pair = bits_per_block / n_pairs_block
    cb_bytes = sum(cb.size * 4 for cb in codebooks)

    return {
        "name": name, "n_pairs_block": n_pairs_block, "n_subblocks": n_subblocks,
        "K_sub": K_sub,
        "rel_l2": rel_l2,
        "bits_per_pair": float(bits_per_pair),
        "byte_per_pair": float(bits_per_pair / 8),
        "codebook_bytes": int(cb_bytes),
    }


def main():
    sys.path.insert(0, str(Path("/Users/silv/cl/tlp/montyneg/ivp5_ds4/analyzers")))
    from polar_encode_mlx import _load_index, _load_expert_bytes

    fp4_dir = Path("/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash")
    weight_map = _load_index(fp4_dir)
    blob, scale = _load_expert_bytes(fp4_dir, layer=0, expert=0, kind="gate", weight_map=weight_map)
    fp4_full = dequant_fp4_tensor(np.asarray(blob), np.asarray(scale))
    fp4_first128 = fp4_full[:128].flatten()
    print(f"FP4 source: {fp4_first128.size} weights = {fp4_first128.size//2} pairs")

    iq2_rel_l2 = 0.3543

    results = []
    print(f"\n{'recipe':<45}  {'byte/pair':>10}  {'rel_L2':>8}  {'lift':>5}  {'cb_KB':>6}")
    print(f"  {'-'*45}  {'-'*10}  {'-'*8}  {'-'*5}  {'-'*6}")

    # A. Pure Block-VQ — joint encoding of N-pair blocks
    block_configs = [
        # (n_pairs_block, K)
        (4,    256),    # 8-weight block × K=256 = 8 bits / 4 pairs = 0.25 byte/pair
        (4,    1024),   # 8-weight block × K=1024 = 10 bits / 4 pairs = 0.3125 byte/pair
        (4,    4096),   # 0.375 byte/pair
        (4,    16384),  # 0.4375
        (8,    256),    # 16-weight block × K=256 = 8/8 = 0.125 byte/pair
        (8,    1024),   # 0.156
        (8,    4096),   # 0.1875
        (8,    16384),  # 0.21875
        (16,   256),    # 32-weight block × K=256 = 8/16 = 0.0625 byte/pair (matches IQ2 block)
        (16,   1024),   # 0.078
        (16,   4096),   # 0.094
        (16,   16384),  # 0.109
    ]
    print(f"\n--- Block-VQ (joint encoding of N-pair blocks) ---")
    for n_pb, K in block_configs:
        name = f"Block-VQ block={n_pb}pairs K={K}"
        t0 = time.time()
        try:
            r = encode_block_vq(fp4_first128, n_pb, K, with_scale=False, name=name)
            elapsed = time.time() - t0
            lift = iq2_rel_l2 / r["rel_l2"]
            r["quality_lift_vs_iq2"] = float(lift)
            print(f"  {name:<45}  {r['byte_per_pair']:>10.4f}  {r['rel_l2']:>8.4f}  {lift:>5.2f}×  {r['codebook_bytes']/1024:>6.1f}    ({elapsed:.1f}s)")
            results.append(r)
        except Exception as e:
            print(f"  {name:<45}  ERROR: {type(e).__name__}: {e}")

    # A'. Block-VQ + scale (covers wider dynamic range with fewer K)
    print(f"\n--- Block-VQ + per-block fp16 scale ---")
    for n_pb, K in [(8, 256), (8, 1024), (16, 256), (16, 1024), (16, 4096)]:
        name = f"Block-VQ block={n_pb}pairs K={K} +scale"
        t0 = time.time()
        try:
            r = encode_block_vq(fp4_first128, n_pb, K, with_scale=True, name=name)
            elapsed = time.time() - t0
            lift = iq2_rel_l2 / r["rel_l2"]
            r["quality_lift_vs_iq2"] = float(lift)
            print(f"  {name:<45}  {r['byte_per_pair']:>10.4f}  {r['rel_l2']:>8.4f}  {lift:>5.2f}×  {r['codebook_bytes']/1024:>6.1f}    ({elapsed:.1f}s)")
            results.append(r)
        except Exception as e:
            print(f"  {name:<45}  ERROR: {type(e).__name__}: {e}")

    # B. Product Quantization — separable sub-block encoding
    print(f"\n--- Product-Quantization (PQ): N-pair block split into M sub-blocks ---")
    pq_configs = [
        # (n_pairs_block, n_subblocks, K_sub)
        (8,   2, 256),    # 16-weight block → 2 × 8-weight sub × K=256 = 16 bits / 8 pairs = 0.25 byte/pair
        (8,   4, 256),    # 4 × 4-weight sub × K=256 = 32/8 = 0.5 byte/pair (same as K=256 per pair)
        (16,  2, 256),    # 32-weight block → 2 × 16-weight sub × K=256 = 16/16 = 0.125 byte/pair
        (16,  4, 256),    # 4 × 8-weight × K=256 = 32/16 = 0.25 byte/pair
        (16,  8, 256),    # 8 × 4-weight × K=256 = 64/16 = 0.5 byte/pair
        (16,  2, 1024),   # 2 × 16-weight × K=1024 = 20/16 = 0.156 byte/pair
        (16,  4, 1024),   # 4 × 8-weight × K=1024 = 40/16 = 0.3125 byte/pair
        (16,  2, 4096),   # 24/16 = 0.1875
        (16,  4, 4096),   # 48/16 = 0.375
    ]
    for n_pb, n_sub, K_sub in pq_configs:
        name = f"PQ block={n_pb}p sub={n_sub} K_sub={K_sub}"
        t0 = time.time()
        try:
            r = encode_product_quant(fp4_first128, n_pb, n_sub, K_sub, name=name)
            elapsed = time.time() - t0
            lift = iq2_rel_l2 / r["rel_l2"]
            r["quality_lift_vs_iq2"] = float(lift)
            print(f"  {name:<45}  {r['byte_per_pair']:>10.4f}  {r['rel_l2']:>8.4f}  {lift:>5.2f}×  {r['codebook_bytes']/1024:>6.1f}    ({elapsed:.1f}s)")
            results.append(r)
        except Exception as e:
            print(f"  {name:<45}  ERROR: {type(e).__name__}: {e}")

    # Storage extrapolation
    print(f"\n=== Pareto frontier (sorted by byte/pair ascending) ===")
    results.sort(key=lambda r: r["byte_per_pair"])
    full_routed_pairs = 76.7e9 / 0.5156
    trim50_routed_pairs = full_routed_pairs * 0.5
    spine_gb = 10.0

    print(f"  {'recipe':<45}  {'byte/pair':>10}  {'rel_L2':>8}  {'lift':>5}  {'F-DS4 GB':>10}  {'T50 GB':>10}  {'F-fit52':<8}  {'T-fit52':<8}")
    print(f"  {'-'*45}  {'-'*10}  {'-'*8}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
    for r in results:
        f_gb = full_routed_pairs * r["byte_per_pair"] / 1e9 + spine_gb
        t_gb = trim50_routed_pairs * r["byte_per_pair"] / 1e9 + spine_gb
        f_fit = "YES" if f_gb < 52 else "no"
        t_fit = "YES" if t_gb < 52 else "no"
        lift = r["quality_lift_vs_iq2"]
        print(f"  {r['name']:<45}  {r['byte_per_pair']:>10.4f}  {r['rel_l2']:>8.4f}  {lift:>5.2f}×  {f_gb:>10.2f}  {t_gb:>10.2f}  {f_fit:<8}  {t_fit:<8}")

    print(f"\n=== Recipes that FIT in <52 GB on FULL DS4 AND beat IQ2 quality (lift > 1×) ===")
    fits_and_wins = [r for r in results
                     if (full_routed_pairs * r["byte_per_pair"] / 1e9 + spine_gb) < 52
                     and r["quality_lift_vs_iq2"] > 1.0]
    fits_and_wins.sort(key=lambda r: -r["quality_lift_vs_iq2"])
    for r in fits_and_wins:
        f_gb = full_routed_pairs * r["byte_per_pair"] / 1e9 + spine_gb
        print(f"  {r['name']:<45}  {r['byte_per_pair']:>10.4f} byte/pair  {f_gb:>6.2f} GB  rel_L2 {r['rel_l2']:.4f} ({r['quality_lift_vs_iq2']:.2f}× IQ2)")

    if not fits_and_wins:
        print("  (none — full DS4 in <52GB with quality > IQ2 not yet achievable in this sweep)")

    print(f"\n=== Recipes that FIT in <52 GB on TRIM50 AND beat IQ2 quality ===")
    fits_and_wins_t = [r for r in results
                       if (trim50_routed_pairs * r["byte_per_pair"] / 1e9 + spine_gb) < 52
                       and r["quality_lift_vs_iq2"] > 1.0]
    fits_and_wins_t.sort(key=lambda r: -r["quality_lift_vs_iq2"])
    for r in fits_and_wins_t:
        t_gb = trim50_routed_pairs * r["byte_per_pair"] / 1e9 + spine_gb
        print(f"  {r['name']:<45}  {r['byte_per_pair']:>10.4f} byte/pair  {t_gb:>6.2f} GB  rel_L2 {r['rel_l2']:.4f} ({r['quality_lift_vs_iq2']:.2f}× IQ2)")

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    Path(f"block_vq_sweep_{ts}.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved block_vq_sweep_{ts}.json")


if __name__ == "__main__":
    main()
