"""Proper A/B: IQ2_XXS vs polar p32_m8 vs VQ K=256, all measured against FP4 source.

silv 2026-05-25: IQ2_XXS is antirez's lossy ~2-bit re-quantization of FP4 source.
Source = FP4 routed experts + FP8 E4M3 non-expert weights (per config.json).
The codec arc encoded polar/VQ from FP4 source; this script extracts IQ2_XXS
quality from the same FP4 source so all three codecs can be compared.

Method:
- Pick one tensor: blk.0.ffn_gate_exps.weight, expert 0
- Get FP4 source weights via dequant_fp4_tensor
- Get IQ2_XXS dequant via Python port of metal/moe.metal iq2_value()
- Get polar p32_m8 reconstruction via the polar_decode helper
- Get VQ K=256 reconstruction via the vq_decode helper
- Report rel_L2 + bytes/pair Pareto
"""
import sys
import struct
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "analyzers"))
sys.path.insert(0, str(Path(__file__).parent))

from polar_encode_bulk import dequant_fp4_tensor  # noqa: E402
from dequant_iq2_xxs import dequant_iq2_xxs_fast  # noqa: E402


# ---- GGUF reader (minimal, just for finding tensor offsets) ----

def _read_gguf_meta(path: Path):
    """Return (metadata_dict, tensor_info_list, data_start_offset)."""
    f = open(path, "rb")
    magic = f.read(4)
    assert magic == b"GGUF", f"bad magic: {magic}"
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]

    def read_str():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8")

    def read_value(vt):
        if vt == 0: return struct.unpack("<B", f.read(1))[0]
        if vt == 1: return struct.unpack("<b", f.read(1))[0]
        if vt == 2: return struct.unpack("<H", f.read(2))[0]
        if vt == 3: return struct.unpack("<h", f.read(2))[0]
        if vt == 4: return struct.unpack("<I", f.read(4))[0]
        if vt == 5: return struct.unpack("<i", f.read(4))[0]
        if vt == 6: return struct.unpack("<f", f.read(4))[0]
        if vt == 7: return struct.unpack("<B", f.read(1))[0] != 0  # bool
        if vt == 8: return read_str()
        if vt == 9:
            elem_vt = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            return [read_value(elem_vt) for _ in range(n)]
        if vt == 10: return struct.unpack("<Q", f.read(8))[0]
        if vt == 11: return struct.unpack("<q", f.read(8))[0]
        if vt == 12: return struct.unpack("<d", f.read(8))[0]
        raise ValueError(f"unknown vt {vt}")

    meta = {}
    for _ in range(n_kv):
        key = read_str()
        vt = struct.unpack("<I", f.read(4))[0]
        meta[key] = read_value(vt)

    # Tensor info
    tinfo = []
    for _ in range(n_tensors):
        name = read_str()
        n_dims = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
        ggml_type = struct.unpack("<I", f.read(4))[0]
        offset = struct.unpack("<Q", f.read(8))[0]
        tinfo.append({"name": name, "dims": dims, "type": ggml_type, "offset": offset})

    # Align to 32 bytes (GGUF default alignment)
    pos = f.tell()
    align = meta.get("general.alignment", 32)
    pad = (align - pos % align) % align
    data_start = pos + pad
    f.close()
    return meta, tinfo, data_start


def _read_tensor_bytes(path: Path, info: dict, data_start: int) -> bytes:
    """Read raw bytes for a tensor from the GGUF data block."""
    f = open(path, "rb")
    # IQ2_XXS: type=16, 256 weights / 66 bytes
    if info["type"] != 16:
        raise NotImplementedError(f"only IQ2_XXS (type=16) supported here, got {info['type']}")
    n_weights = int(np.prod(info["dims"]))
    assert n_weights % 256 == 0, f"n_weights={n_weights} not div by 256"
    n_blocks = n_weights // 256
    nbytes = n_blocks * 66
    f.seek(data_start + info["offset"])
    raw = f.read(nbytes)
    f.close()
    return raw


# ---- Polar p32_m8 reader (matches ds4_polar_reader.h PLR2 format) ----

def read_polar_plr2(path: Path, expert: int = 0):
    """Read PLR2 file, return reconstructed fp32 weights for one expert.

    PLR2 layout (per ds4_polar_reader.h):
      0..3   magic "PLR2"
      4..7   version
      8..11  n_experts
      12..15 n_rows
      16..19 n_pairs
      20..23 layer
      24..27 kind_id
      28..31 phase_levels (legacy=0 means 8)
      32..35 mag_levels (legacy=0 means 4)
      36..63 reserved
      64..       mag_codes uint8 [n_experts, n_rows, n_pairs]
      ...        phase_codes uint8 [n_experts, n_rows, n_pairs]
      ...        levels float32 [n_experts, n_rows, mag_levels]
    """
    with open(path, "rb") as f:
        hdr = f.read(64)
        assert hdr[:4] == b"PLR2", f"bad polar magic: {hdr[:4]}"
        version, n_experts, n_rows, n_pairs, layer, kind_id, phase_levels, mag_levels = \
            struct.unpack("<IIIIIIII", hdr[4:36])
        if phase_levels == 0: phase_levels = 8
        if mag_levels == 0: mag_levels = 4

        total_codes = n_experts * n_rows * n_pairs
        mag_all = np.frombuffer(f.read(total_codes), dtype=np.uint8).reshape(n_experts, n_rows, n_pairs)
        phase_all = np.frombuffer(f.read(total_codes), dtype=np.uint8).reshape(n_experts, n_rows, n_pairs)
        levels_all = np.frombuffer(f.read(n_experts * n_rows * mag_levels * 4), dtype=np.float32).reshape(n_experts, n_rows, mag_levels)

    mag = mag_all[expert]   # (n_rows, n_pairs)
    phase = phase_all[expert]
    levels = levels_all[expert]  # (n_rows, mag_levels)
    # Decoder must match _reconstruct_mlx in polar_encode_mlx.py:
    # qangle = (phase_codes - P/2) * phase_step   where phase_step = 2π/P
    phase_step = 2.0 * np.pi / float(phase_levels)
    angles = (phase.astype(np.int32) - (phase_levels // 2)).astype(np.float32) * phase_step
    m = levels[np.arange(n_rows)[:, None], mag]  # (n_rows, n_pairs)
    re = m * np.cos(angles)
    im = m * np.sin(angles)
    out = np.empty((n_rows, 2 * n_pairs), dtype=np.float32)
    out[:, 0::2] = re
    out[:, 1::2] = im
    return {
        "n_experts": n_experts, "n_rows": n_rows, "n_pairs": n_pairs,
        "phase_levels": phase_levels, "mag_levels": mag_levels,
        "kind_id": kind_id, "layer": layer,
        "reconstructed": out,  # (n_rows, 2*n_pairs)
        "storage_bytes_per_pair": 2,
    }


# ---- VQ K=256 reader (matches ds4_vqb1_reader.h VQB1 format) ----

def read_vq_vqb1(path: Path, expert: int = 0):
    """Read VQB1 file (32-byte header + codebook + codes per ds4_vqb1_reader.h).

    Header: magic(4) + version + n_experts + n_rows + n_pairs + layer + kind_id + k = 32 bytes
    Codebook: K × 2 fp32 floats
    Codes: n_experts × n_rows × n_pairs uint8
    """
    with open(path, "rb") as f:
        hdr = f.read(32)
        assert hdr[:4] == b"VQB1", f"bad vq magic: {hdr[:4]}"
        version, n_experts, n_rows, n_pairs, layer, kind_id, K = struct.unpack("<IIIIIII", hdr[4:32])
        codebook = np.frombuffer(f.read(K * 2 * 4), dtype=np.float32).reshape(K, 2)
        codes_all = np.frombuffer(f.read(n_experts * n_rows * n_pairs), dtype=np.uint8).reshape(n_experts, n_rows, n_pairs)
    codes = codes_all[expert]  # (n_rows, n_pairs)
    reconstructed_pairs = codebook[codes]  # (n_rows, n_pairs, 2)
    out = reconstructed_pairs.reshape(n_rows, 2 * n_pairs).astype(np.float32)
    return {
        "n_experts": n_experts, "n_rows": n_rows, "n_pairs": n_pairs, "K": K,
        "kind_id": kind_id, "layer": layer,
        "codebook": codebook,
        "reconstructed": out,
        "storage_bytes_per_pair": 1,
    }


# ---- FP4 source loader (read from safetensors via existing analyzer infra) ----

def load_fp4_source_expert(safetensors_dir: Path, layer: int, kind: str, expert: int):
    """Load FP4 source for L{layer} expert {expert} {kind}.

    Returns (fp32 numpy array, tensor_name, shape).
    Uses the same tensor-naming convention as polar_encode_mlx._load_expert_bytes:
        layers.{layer}.ffn.experts.{expert}.{w}.weight + .scale
    where w = w1 (gate) / w2 (down) / w3 (up).
    """
    sys.path.insert(0, str(safetensors_dir.parent.parent.parent / "ivp5_ds4" / "analyzers"))
    from polar_encode_mlx import _load_index, _load_expert_bytes, KIND_TO_W  # type: ignore

    weight_map = _load_index(safetensors_dir)
    if not weight_map:
        raise FileNotFoundError(f"no weight_map at {safetensors_dir}/model.safetensors.index.json")
    blob, scale = _load_expert_bytes(safetensors_dir, layer, expert, kind, weight_map)
    if blob is None:
        sample_keys = [k for k in weight_map if f"layers.{layer}" in k][:5]
        raise FileNotFoundError(f"L{layer} expert {expert} {kind} ({KIND_TO_W[kind]}) not in weight_map; sample L{layer} keys: {sample_keys}")
    # blob/scale come back as numpy arrays already in the safetensors-declared shape
    print(f"  raw blob shape: {blob.shape} dtype: {blob.dtype}")
    print(f"  raw scale shape: {scale.shape} dtype: {scale.dtype}")
    fp32 = dequant_fp4_tensor(np.asarray(blob), np.asarray(scale))
    return fp32, f"layers.{layer}.ffn.experts.{expert}.{KIND_TO_W[kind]}.weight", fp32.shape


# ---- Main A/B ----

def main():
    root = Path("/Users/silv/cl/tlp/montyneg/ivp5_ds4")
    montyneg_root = Path("/Users/silv/cl/tlp/montyneg")
    iq2_gguf = montyneg_root / "ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
    fp4_dir = montyneg_root / "ds4/gguf/DeepSeek-V4-Flash"
    polar_dir = root / "tmp/polar_full_p32m8"
    vq_dir = root / "tmp/vqb1_L0_k256_mlx"

    # 1. Read GGUF tensor info, find blk.0.ffn_gate_exps.weight
    print("=== Step 1: read GGUF metadata ===")
    meta, tinfo, data_start = _read_gguf_meta(iq2_gguf)
    target_name = "blk.0.ffn_gate_exps.weight"
    info = next((t for t in tinfo if t["name"] == target_name), None)
    if info is None:
        # Find anything matching gate_exps
        candidates = [t["name"] for t in tinfo if "gate_exps" in t["name"]][:5]
        print(f"target not found; candidates: {candidates}")
        return
    print(f"target: {target_name}")
    print(f"  type={info['type']} dims={info['dims']} offset={info['offset']}")

    # 2. Read IQ2_XXS bytes for the WHOLE tensor (256 experts), then slice expert 0
    n_total = int(np.prod(info["dims"]))
    n_blocks_total = n_total // 256
    print(f"  n_weights total: {n_total}, n_blocks total: {n_blocks_total}")
    # Per-expert: dims = (in_dim, out_dim, n_experts) or similar
    # DS4 expert shape: gate is (n_experts=256, moe_intermediate_size=2048, hidden_size=4096) typically
    # = 256 * 2048 * 4096 = 2,147,483,648 weights = 2.15 GB IQ2_XXS at 0.5 bytes/weight ≈ 553 MB
    # Per expert: 2048 * 4096 = 8,388,608 weights = 32,768 blocks = 2,162,688 bytes IQ2
    # Verify: info["dims"] should give us this
    print(f"  blocks per expert: {n_blocks_total // 256}")
    print(f"  weights per expert: {n_total // 256}")

    # Read IQ2 bytes for expert 0 only
    weights_per_expert = n_total // 256
    blocks_per_expert = weights_per_expert // 256
    bytes_per_expert = blocks_per_expert * 66
    f = open(iq2_gguf, "rb")
    f.seek(data_start + info["offset"])
    iq2_expert0_bytes = f.read(bytes_per_expert)
    f.close()
    print(f"  IQ2_XXS expert-0 bytes: {len(iq2_expert0_bytes)} = {blocks_per_expert} blocks × 66 bytes")

    # 3. Dequantize IQ2_XXS expert 0
    print("=== Step 2: dequantize IQ2_XXS ===")
    iq2_fp32 = dequant_iq2_xxs_fast(iq2_expert0_bytes, weights_per_expert)
    print(f"  iq2_fp32 shape: {iq2_fp32.shape}, dtype: {iq2_fp32.dtype}")
    print(f"  iq2 stats: mean={iq2_fp32.mean():.4e}, std={iq2_fp32.std():.4e}, range=[{iq2_fp32.min():.4e}, {iq2_fp32.max():.4e}]")

    # 4. Load FP4 source for the same expert
    print("=== Step 3: load FP4 source ===")
    try:
        fp4_full, fp4_tensor_name, fp4_shape = load_fp4_source_expert(fp4_dir, layer=0, kind="gate", expert=0)
        print(f"  FP4 tensor: {fp4_tensor_name}, shape: {fp4_shape}")
        print(f"  FP4 full size: {fp4_full.size}, ndim: {fp4_full.ndim}")
        # Slice to expert 0
        if fp4_full.ndim == 3 and fp4_full.shape[0] == 256:
            fp4_expert0 = fp4_full[0].flatten()
        elif fp4_full.ndim == 2:
            # Already single expert
            fp4_expert0 = fp4_full.flatten()
        else:
            print(f"unexpected fp4 shape: {fp4_full.shape}")
            return
        print(f"  fp4_expert0 size: {fp4_expert0.size}")
        print(f"  fp4 stats: mean={fp4_expert0.mean():.4e}, std={fp4_expert0.std():.4e}, range=[{fp4_expert0.min():.4e}, {fp4_expert0.max():.4e}]")
    except Exception as e:
        print(f"  FP4 load FAILED: {type(e).__name__}: {e}")
        print(f"  Will report only IQ2 stats")
        fp4_expert0 = None

    # 5. Compute rel_L2 for IQ2 vs FP4
    if fp4_expert0 is not None and fp4_expert0.size == iq2_fp32.size:
        diff = iq2_fp32 - fp4_expert0
        rel_l2 = np.linalg.norm(diff) / max(np.linalg.norm(fp4_expert0), 1e-30)
        max_abs = np.max(np.abs(diff))
        mean_abs = np.mean(np.abs(diff))
        print(f"\n=== IQ2_XXS vs FP4 (L0 gate expert 0) ===")
        print(f"  rel_L2:       {rel_l2:.4f}")
        print(f"  max_abs:      {max_abs:.4e}")
        print(f"  mean_abs:     {mean_abs:.4e}")
        print(f"  bytes/pair:   {bytes_per_expert / (weights_per_expert / 2):.4f}")
    elif fp4_expert0 is not None:
        print(f"  size mismatch: IQ2={iq2_fp32.size} FP4={fp4_expert0.size}")

    # The polar/VQ corpora encode only n_rows_per_expert=128 rows of the full
    # 2048-row tensor per scope-caveat from analyzers/vq2d_encode_vqb1.py.
    # To compare fairly, slice FP4 expert 0 to its first 128 rows × all in_dim cols.
    fp4_2d = fp4_expert0.reshape(2048, 4096)  # (out_dim, in_dim)
    fp4_first128 = fp4_2d[:128].flatten()  # First 128 rows = 128 * 4096 = 524288 weights
    print(f"\n  fp4_first128 size: {fp4_first128.size} ({fp4_first128.size//1024} KB worth)")

    # IQ2 doesn't encode "first 128 rows" — it encodes the WHOLE expert. So for fair comparison
    # we slice the IQ2 dequant the same way.
    iq2_2d = iq2_fp32.reshape(2048, 4096)
    iq2_first128 = iq2_2d[:128].flatten()
    diff_iq2_128 = iq2_first128 - fp4_first128
    rel_l2_iq2_128 = np.linalg.norm(diff_iq2_128) / max(np.linalg.norm(fp4_first128), 1e-30)
    print(f"  rel_L2 IQ2 vs FP4 (FIRST 128 rows, matching codec corpus scope): {rel_l2_iq2_128:.4f}")

    # 6. Polar p32_m8 — encoded for expert 0 of L0 gate at 128 rows
    polar_path = polar_dir / "L00_gate.polar"
    polar_rel_l2 = None
    polar_bytes_per_pair = None
    if polar_path.exists() and fp4_expert0 is not None:
        print(f"\n=== Polar p32_m8 reading ===")
        polar = read_polar_plr2(polar_path, expert=0)
        print(f"  polar: n_experts={polar['n_experts']} n_rows={polar['n_rows']} n_pairs={polar['n_pairs']} phase={polar['phase_levels']} mag={polar['mag_levels']}")
        polar_flat = polar['reconstructed'].flatten()
        # polar.reconstructed shape: (n_rows=128, 2*n_pairs=4096)
        # FP4 expert 0 shape: (2048, 4096) — take first 128 rows
        if polar_flat.size == fp4_first128.size:
            diff = polar_flat - fp4_first128
            polar_rel_l2 = np.linalg.norm(diff) / max(np.linalg.norm(fp4_first128), 1e-30)
            polar_bytes_per_pair = polar['storage_bytes_per_pair']
            print(f"  rel_L2 polar vs FP4 (first 128 rows): {polar_rel_l2:.4f}")
            print(f"  bytes/pair polar: {polar_bytes_per_pair}")
        else:
            print(f"  size mismatch: polar={polar_flat.size}, fp4_first128={fp4_first128.size}")
    else:
        if not polar_path.exists():
            print(f"  no polar at {polar_path}")

    # 7. VQ K=256
    vq_path = vq_dir / "L00_gate.vqb1"
    vq_rel_l2 = None
    vq_bytes_per_pair = None
    if vq_path.exists() and fp4_expert0 is not None:
        print(f"\n=== VQ K=256 reading ===")
        vq = read_vq_vqb1(vq_path, expert=0)
        print(f"  vq: n_experts={vq['n_experts']} n_rows={vq['n_rows']} n_pairs={vq['n_pairs']} K={vq['K']}")
        vq_flat = vq['reconstructed'].flatten()
        if vq_flat.size == fp4_first128.size:
            diff = vq_flat - fp4_first128
            vq_rel_l2 = np.linalg.norm(diff) / max(np.linalg.norm(fp4_first128), 1e-30)
            vq_bytes_per_pair = vq['storage_bytes_per_pair']
            print(f"  rel_L2 vq vs FP4 (first 128 rows): {vq_rel_l2:.4f}")
            print(f"  bytes/pair vq: {vq_bytes_per_pair}")
        else:
            print(f"  size mismatch: vq={vq_flat.size}, fp4_first128={fp4_first128.size}")
    else:
        if not vq_path.exists():
            print(f"  no vq at {vq_path}")

    # 8. Pareto summary
    print("\n=== Pareto frontier (L0 gate expert 0, first 128 rows) ===")
    print(f"  {'Codec':<20}  {'bytes/pair':>10}  {'rel_L2':>8}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*8}")
    print(f"  {'IQ2_XXS (antirez)':<20}  {0.5156:>10.4f}  {rel_l2_iq2_128:>8.4f}")
    if polar_rel_l2 is not None:
        print(f"  {'polar p32_m8':<20}  {float(polar_bytes_per_pair):>10.4f}  {polar_rel_l2:>8.4f}")
    if vq_rel_l2 is not None:
        print(f"  {'VQ K=256':<20}  {float(vq_bytes_per_pair):>10.4f}  {vq_rel_l2:>8.4f}")

    # Save result JSON for inheritance
    result = {
        "layer": 0, "kind": "gate", "expert": 0,
        "scope_rows": 128, "in_dim": 4096,
        "iq2_xxs": {"bytes_per_pair": 0.5156, "rel_l2": float(rel_l2_iq2_128)},
        "polar_p32_m8": {"bytes_per_pair": float(polar_bytes_per_pair) if polar_bytes_per_pair else None,
                         "rel_l2": float(polar_rel_l2) if polar_rel_l2 is not None else None},
        "vq_k256":      {"bytes_per_pair": float(vq_bytes_per_pair) if vq_bytes_per_pair else None,
                         "rel_l2": float(vq_rel_l2) if vq_rel_l2 is not None else None},
    }
    import time
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    Path(f"result_{ts}.json").write_text(json.dumps(result, indent=2))
    print(f"\nSaved result_{ts}.json")


if __name__ == "__main__":
    main()
