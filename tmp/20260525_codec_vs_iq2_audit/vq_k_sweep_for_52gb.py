"""Quality vs storage sweep across VQ K + per-block-scale to find <52GB fits.

silv 2026-05-25: "does vq have knobs to tune space-vs-accuracy tradeoff to try
and fit into less than 52gb and thus inside m1 max 64gb memory?"

Knobs swept:
- K ∈ {4, 16, 64, 256} = {2, 4, 6, 8} bits per pair
- per-32-weight u8 scale (BLOCK_SCALE) vs no scale (NO_SCALE) vs per-row fp32 scale (ROW_SCALE)
- per-(layer,kind) shared codebook vs per-row codebook

Method: encode L0 gate expert 0 first 128 rows with each recipe (sklearn K-means
in R²), measure rel_L2 vs FP4 source, compute storage byte/pair.

This is a QUALITY sweep; storage is exact math per recipe.
"""
import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "analyzers"))
sys.path.insert(0, str(Path(__file__).parent))

# Use the bulk encoder helper to load FP4 source
from polar_encode_bulk import dequant_fp4_tensor

# Don't need sklearn; do simple K-means manually for tight control
def kmeans_2d(pairs, K, max_iter=30, seed=42):
    """Simple Lloyd-Max K-means on R² pairs. Returns (codebook K×2, codes N)."""
    rng = np.random.default_rng(seed)
    N = pairs.shape[0]
    # Init: random sample
    init_idx = rng.choice(N, size=K, replace=False)
    codebook = pairs[init_idx].copy()
    prev_inertia = float("inf")
    for it in range(max_iter):
        # Assign: chunked to avoid O(N*K) blowup
        chunk = 100_000
        codes = np.empty(N, dtype=np.int32)
        inertia = 0.0
        for i in range(0, N, chunk):
            sub = pairs[i:i+chunk]
            d = np.sum((sub[:, None, :] - codebook[None, :, :]) ** 2, axis=-1)
            codes[i:i+chunk] = np.argmin(d, axis=-1)
            inertia += d[np.arange(d.shape[0]), codes[i:i+chunk]].sum()
        # Update
        new_cb = np.zeros_like(codebook)
        for k in range(K):
            mask = codes == k
            if mask.sum() > 0:
                new_cb[k] = pairs[mask].mean(axis=0)
            else:
                new_cb[k] = codebook[k]
        if abs(prev_inertia - inertia) / max(prev_inertia, 1e-30) < 1e-5:
            break
        codebook = new_cb
        prev_inertia = inertia
    return codebook, codes


def encode_test(pairs_flat, K, scale_mode, recipe_name):
    """Encode flat-pair array under given recipe, measure rel_L2 and byte/pair cost.

    scale_mode:
      "none"        — single fit, no scale
      "block32"     — per-32-weight (16-pair) u8 scale (matches IQ2's block structure)
      "row"         — per-row fp32 scale (assumes pairs come reshape-friendly)
    """
    pairs = pairs_flat.reshape(-1, 2)  # (n_pairs, 2)
    n_pairs = pairs.shape[0]

    if scale_mode == "none":
        cb, codes = kmeans_2d(pairs, K)
        recon = cb[codes]
        rel_l2 = float(np.linalg.norm(recon - pairs) / max(np.linalg.norm(pairs), 1e-30))
        # Storage: K × 2 × 4 bytes (codebook) + n_pairs × log2(K) bits (codes)
        bits_per_pair = np.log2(K)
        cb_bytes = K * 2 * 4  # amortized over all encoders sharing this codebook
        return {
            "name": recipe_name, "K": K, "scale_mode": scale_mode,
            "rel_l2": rel_l2,
            "bits_per_pair": float(bits_per_pair),
            "byte_per_pair": float(bits_per_pair / 8),
            "codebook_bytes": int(cb_bytes),
        }

    if scale_mode == "block32":
        # Block scaling: per 16-pair block (32 weights), one u8 scale
        # First normalize within each block, then K-means on unit-normalized pairs
        n_blocks = n_pairs // 16
        pairs_b = pairs[:n_blocks * 16].reshape(n_blocks, 16, 2)
        block_max = np.linalg.norm(pairs_b, axis=-1).max(axis=-1)  # (n_blocks,)
        block_max_safe = np.maximum(block_max, 1e-30)
        unit_pairs = pairs_b / block_max_safe[:, None, None]  # (n_blocks, 16, 2) normalized
        unit_pairs_flat = unit_pairs.reshape(-1, 2)
        # K-means on unit pairs
        cb, codes = kmeans_2d(unit_pairs_flat, K)
        recon_unit = cb[codes].reshape(n_blocks, 16, 2)
        # Scale: quantize block_max to u8 using log2 (E8M0-like) or linear
        # Use E8M0: store as uint8(round(log2(block_max) + 127))
        # For simplicity, use linear u8 normalized to max(block_max)
        max_max = block_max.max()
        scale_u8 = np.clip(np.round(block_max / max_max * 255), 0, 255).astype(np.uint8)
        scale_decoded = scale_u8.astype(np.float32) * (max_max / 255)
        # Reconstruct
        recon_full = recon_unit * scale_decoded[:, None, None]
        recon_flat = recon_full.reshape(-1, 2)
        # rel_L2
        rel_l2 = float(np.linalg.norm(recon_flat - pairs[:n_blocks * 16]) /
                        max(np.linalg.norm(pairs[:n_blocks * 16]), 1e-30))
        bits_per_pair = np.log2(K) + 8.0 / 16  # K-bits per pair + 8/16 bits scale per pair
        return {
            "name": recipe_name, "K": K, "scale_mode": scale_mode,
            "rel_l2": rel_l2,
            "bits_per_pair": float(bits_per_pair),
            "byte_per_pair": float(bits_per_pair / 8),
            "codebook_bytes": int(K * 2 * 4),
            "scale_overhead_byte_per_pair": 8.0 / 16 / 8,
        }

    if scale_mode == "row":
        # Per-row scale: assume rows are sequential 2048-pair chunks (matching DS4 codec scope)
        # Need to know the row size for this to work; pass as part of recipe metadata
        n_row = 2048  # for L0 gate first 128 rows expert 0
        n_rows = n_pairs // n_row
        pairs_r = pairs[:n_rows * n_row].reshape(n_rows, n_row, 2)
        row_max = np.linalg.norm(pairs_r, axis=-1).max(axis=-1)  # (n_rows,)
        row_max_safe = np.maximum(row_max, 1e-30)
        unit_pairs = pairs_r / row_max_safe[:, None, None]
        unit_pairs_flat = unit_pairs.reshape(-1, 2)
        cb, codes = kmeans_2d(unit_pairs_flat, K)
        recon_unit = cb[codes].reshape(n_rows, n_row, 2)
        # Per-row fp32 scale
        recon_full = recon_unit * row_max[:, None, None]
        recon_flat = recon_full.reshape(-1, 2)
        rel_l2 = float(np.linalg.norm(recon_flat - pairs[:n_rows * n_row]) /
                        max(np.linalg.norm(pairs[:n_rows * n_row]), 1e-30))
        # Storage: log2(K) bits/pair + 32 bits per n_row pairs = log2(K) + 32/n_row
        bits_per_pair = np.log2(K) + 32.0 / n_row
        return {
            "name": recipe_name, "K": K, "scale_mode": scale_mode,
            "rel_l2": rel_l2,
            "bits_per_pair": float(bits_per_pair),
            "byte_per_pair": float(bits_per_pair / 8),
            "codebook_bytes": int(K * 2 * 4),
            "scale_overhead_byte_per_pair": 32.0 / n_row / 8,
        }

    raise ValueError(f"unknown scale_mode {scale_mode}")


def main():
    # Load FP4 source for L0 gate expert 0
    sys.path.insert(0, str(Path("/Users/silv/cl/tlp/montyneg/ivp5_ds4/analyzers")))
    from polar_encode_mlx import _load_index, _load_expert_bytes, KIND_TO_W

    fp4_dir = Path("/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash")
    weight_map = _load_index(fp4_dir)
    blob, scale = _load_expert_bytes(fp4_dir, layer=0, expert=0, kind="gate", weight_map=weight_map)
    fp4_full = dequant_fp4_tensor(np.asarray(blob), np.asarray(scale))
    # Use first 128 rows (codec scope)
    fp4_first128 = fp4_full[:128].flatten()
    print(f"FP4 source: 128 rows × 4096 = {fp4_first128.size} weights = {fp4_first128.size//2} pairs")
    print(f"  stats: std={fp4_first128.std():.4e} range=[{fp4_first128.min():.4e}, {fp4_first128.max():.4e}]")

    # Reference: full IQ2_XXS rel_L2 for this slice was 0.3543; VQ K=256 = 0.0212

    recipes = [
        # K, scale_mode, name
        (4,   "none",    "VQ K=4 no scale"),
        (4,   "block32", "VQ K=4 + block32 scale"),
        (4,   "row",     "VQ K=4 + per-row scale"),
        (16,  "none",    "VQ K=16 no scale"),
        (16,  "block32", "VQ K=16 + block32 scale"),
        (16,  "row",     "VQ K=16 + per-row scale"),
        (64,  "none",    "VQ K=64 no scale"),
        (64,  "block32", "VQ K=64 + block32 scale"),
        (64,  "row",     "VQ K=64 + per-row scale"),
        (256, "none",    "VQ K=256 no scale (reference)"),
        (256, "block32", "VQ K=256 + block32 scale"),
        (256, "row",     "VQ K=256 + per-row scale"),
    ]

    results = []
    print(f"\n{'recipe':<40}  {'K':>4}  {'scale':<10}  {'byte/pair':>10}  {'rel_L2':>8}  {'lift':>5}")
    print(f"  {'-'*40}  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*5}")
    iq2_rel_l2 = 0.3543  # from prior measurement on same slice

    for K, scale_mode, name in recipes:
        t0 = time.time()
        r = encode_test(fp4_first128, K, scale_mode, name)
        elapsed = time.time() - t0
        lift = iq2_rel_l2 / r["rel_l2"]
        print(f"  {name:<40}  {K:>4}  {scale_mode:<10}  {r['byte_per_pair']:>10.4f}  {r['rel_l2']:>8.4f}  {lift:>5.2f}×    ({elapsed:.1f}s)")
        r["iq2_rel_l2_baseline"] = iq2_rel_l2
        r["quality_lift_vs_iq2"] = float(lift)
        results.append(r)

    # Storage extrapolation
    print(f"\n=== Storage projection: full DS4 routed-expert size for each recipe ===")
    # Full DS4 routed: 76.7 GB IQ2 at 0.5156 byte/pair → 1.487e11 pairs in routed
    full_routed_pairs = 76.7e9 / 0.5156  # ~1.49e11 pairs
    spine_gb = 10.0  # Q8/F16 spine (config.json + non-routed)
    print(f"  {'recipe':<40}  {'byte/pair':>10}  {'routed GB':>10}  {'total GB':>10}  {'fit 52GB?':<10}")
    print(f"  {'-'*40}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for r in results:
        routed_gb = full_routed_pairs * r["byte_per_pair"] / 1e9
        total_gb = routed_gb + spine_gb
        fits = "YES" if total_gb < 52 else "no"
        print(f"  {r['name']:<40}  {r['byte_per_pair']:>10.4f}  {routed_gb:>10.2f}  {total_gb:>10.2f}  {fits:<10}")

    # Same for trim50 (half the routed experts)
    print(f"\n=== Storage projection: TRIM50 DS4 routed-expert size (silv baseline 60GB total) ===")
    trim50_routed_pairs = full_routed_pairs * 0.5
    print(f"  {'recipe':<40}  {'byte/pair':>10}  {'routed GB':>10}  {'total GB':>10}  {'fit 52GB?':<10}")
    print(f"  {'-'*40}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for r in results:
        routed_gb = trim50_routed_pairs * r["byte_per_pair"] / 1e9
        total_gb = routed_gb + spine_gb
        fits = "YES" if total_gb < 52 else "no"
        print(f"  {r['name']:<40}  {r['byte_per_pair']:>10.4f}  {routed_gb:>10.2f}  {total_gb:>10.2f}  {fits:<10}")

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    Path(f"vq_k_sweep_{ts}.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved vq_k_sweep_{ts}.json")


if __name__ == "__main__":
    main()
