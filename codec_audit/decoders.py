"""codec_audit/decoders.py — consolidated codec decoders.

One decoder per codec family. All return fp32 numpy arrays at the SAME shape
as the FP4 source for the given (layer, kind, expert), or None if the codec
has no encoded data for the requested cell.

Each decoder is a pure function: same inputs → same outputs.

Caching: load_fp4_source uses functools.lru_cache for O(1) repeated lookup;
per-(layer, kind) shared codec corpora are loaded once via _open_polar / _open_vqb1.
"""
import struct
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

# Defaults for current DS4 V4 deployment
_FP4_DIR = Path("/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash")
_IQ2_GGUF = Path(
    "/Users/silv/cl/tlp/montyneg/ds4/gguf/"
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
)
_POLAR_DIR = Path("/Users/silv/cl/tlp/montyneg/ivp5_ds4/tmp/polar_full_p32m8")
_VQ_DIR = Path("/Users/silv/cl/tlp/montyneg/ivp5_ds4/tmp/vqb1_L0_k256_mlx")

# Lookup tables for IQ2_XXS (extracted from metal/moe.metal in earlier work)
_TABLES_DIR = Path("/Users/silv/cl/tlp/montyneg/ivp5_ds4/tmp/20260525_codec_vs_iq2_audit")
_IQ2XXS_GRID = None
_KSIGNS_IQ2XS = None


def _load_iq2_tables():
    global _IQ2XXS_GRID, _KSIGNS_IQ2XS
    if _IQ2XXS_GRID is None:
        _IQ2XXS_GRID = np.load(_TABLES_DIR / "iq2xxs_grid.npy")
    if _KSIGNS_IQ2XS is None:
        _KSIGNS_IQ2XS = np.load(_TABLES_DIR / "ksigns_iq2xs.npy")
    return _IQ2XXS_GRID, _KSIGNS_IQ2XS


# ============================================================================
# FP4 source loader (cached)
# ============================================================================

@lru_cache(maxsize=64)
def load_fp4_source(layer: int, kind: str, expert: int,
                     fp4_dir: Path = _FP4_DIR) -> np.ndarray:
    """Load FP4 source weights for one expert. Returns (out_dim, in_dim) fp32.

    O(1) cache hit; O(safetensors_read) cache miss. Caches up to 64 experts
    to avoid re-reading shards during full sweeps.
    """
    sys.path.insert(0, str(fp4_dir.parent.parent.parent / "ivp5_ds4" / "analyzers"))
    from polar_encode_mlx import _load_index, _load_expert_bytes  # type: ignore
    from polar_encode_bulk import dequant_fp4_tensor  # type: ignore

    weight_map = _load_index(fp4_dir)
    if not weight_map:
        raise FileNotFoundError(f"no weight_map at {fp4_dir}")
    blob, scale = _load_expert_bytes(fp4_dir, layer, expert, kind, weight_map)
    if blob is None:
        raise FileNotFoundError(
            f"FP4 source missing for L{layer} {kind} expert {expert}")
    return dequant_fp4_tensor(np.asarray(blob), np.asarray(scale))


# ============================================================================
# IQ2_XXS decoder (from metal/moe.metal layout, vectorized)
# ============================================================================

def _dequant_iq2_xxs(raw: bytes, n_weights: int) -> np.ndarray:
    """Vectorized IQ2_XXS dequant. 66 bytes per 256 weights."""
    grid, ksigns = _load_iq2_tables()
    grid_bytes = np.frombuffer(grid.tobytes(), dtype=np.int8).reshape(256, 8)

    n_blocks = n_weights // 256
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, 66)
    d_f16 = np.frombuffer(arr[:, 0:2].copy().tobytes(),
                           dtype=np.float16).astype(np.float32)
    sub = arr[:, 2:].reshape(n_blocks, 8, 8).astype(np.uint32)
    aux0 = sub[:, :, 0] | (sub[:, :, 1] << 8) | (sub[:, :, 2] << 16) | (sub[:, :, 3] << 24)
    aux1 = sub[:, :, 4] | (sub[:, :, 5] << 8) | (sub[:, :, 6] << 16) | (sub[:, :, 7] << 24)
    sub_scale = (2.0 * (aux1 >> np.uint32(28)).astype(np.float32) + 1.0) * 0.125
    eff_scale = sub_scale * d_f16[:, None]

    byte_lanes = np.arange(4, dtype=np.uint32)[None, None, :]
    grid_index = ((aux0[:, :, None] >> (8 * byte_lanes)) & 0xff).astype(np.uint8)
    sign_byte = ((aux1[:, :, None] >> (7 * byte_lanes)) & 0x7f).astype(np.uint8)

    grid_vals = grid_bytes[grid_index].astype(np.float32)  # (B,8,4,8)

    expanded = ksigns[sign_byte]  # (B,8,4) uint8 mask
    lane_bits = (1 << np.arange(8, dtype=np.uint8))
    sign_mask = (expanded[..., None] & lane_bits) > 0
    sign_factor = np.where(sign_mask, -1.0, 1.0).astype(np.float32)

    weighted = grid_vals * sign_factor * eff_scale[:, :, None, None]
    out = weighted.reshape(n_blocks, 256)
    return out.reshape(-1).astype(np.float32)


def decode_iq2_xxs(layer: int, kind: str, expert: int,
                    gguf_path: Path = _IQ2_GGUF) -> np.ndarray:
    """Decode IQ2_XXS expert from GGUF. Returns fp32 array at FP4 source shape.

    Reads only the bytes for the requested (layer, kind, expert) — no full mmap.
    O(expert_bytes) work; ~2 MB / expert.
    """
    meta_cache = _read_gguf_meta(gguf_path)
    meta, tinfo, data_start = meta_cache
    target = f"blk.{layer}.ffn_{kind}_exps.weight"
    info = next((t for t in tinfo if t["name"] == target), None)
    if info is None:
        return None
    n_total = int(np.prod(info["dims"]))
    weights_per_expert = n_total // 256
    blocks_per_expert = weights_per_expert // 256
    bytes_per_expert = blocks_per_expert * 66

    with open(gguf_path, "rb") as f:
        f.seek(data_start + info["offset"] + expert * bytes_per_expert)
        raw = f.read(bytes_per_expert)
    fp32 = _dequant_iq2_xxs(raw, weights_per_expert)
    # Reshape to source shape (out_dim, in_dim) — for gate/up DS4 V4 it's (2048, 4096)
    # The FP4 source's shape is the authority; we mirror it here.
    src = load_fp4_source(layer, kind, expert)
    return fp32.reshape(src.shape)


# ============================================================================
# Polar PLR2 decoder
# ============================================================================

def _open_plr2(path: Path):
    with open(path, "rb") as f:
        hdr = f.read(64)
        assert hdr[:4] == b"PLR2", f"bad polar magic: {hdr[:4]}"
        version, n_experts, n_rows, n_pairs, layer, kind_id, phase_levels, mag_levels = \
            struct.unpack("<IIIIIIII", hdr[4:36])
        if phase_levels == 0: phase_levels = 8
        if mag_levels == 0: mag_levels = 4
        total = n_experts * n_rows * n_pairs
        mag_all = np.frombuffer(f.read(total), dtype=np.uint8).reshape(
            n_experts, n_rows, n_pairs)
        phase_all = np.frombuffer(f.read(total), dtype=np.uint8).reshape(
            n_experts, n_rows, n_pairs)
        levels_all = np.frombuffer(
            f.read(n_experts * n_rows * mag_levels * 4), dtype=np.float32
        ).reshape(n_experts, n_rows, mag_levels)
    return {"n_experts": n_experts, "n_rows": n_rows, "n_pairs": n_pairs,
            "phase_levels": phase_levels, "mag_levels": mag_levels,
            "mag_all": mag_all, "phase_all": phase_all, "levels_all": levels_all}


def decode_polar_plr2(layer: int, kind: str, expert: int,
                       polar_dir: Path = _POLAR_DIR) -> np.ndarray | None:
    """Decode polar p32_m8 corpus for one expert. Returns fp32 (n_rows, 2*n_pairs).

    Note: codec scope = first 128 rows of full 2048-row expert (per encoder).
    Returns None if no polar corpus exists for this (layer, kind).
    """
    polar_path = polar_dir / f"L{layer:02d}_{kind}.polar"
    if not polar_path.exists():
        return None
    p = _open_plr2(polar_path)
    mag = p["mag_all"][expert]
    phase = p["phase_all"][expert]
    levels = p["levels_all"][expert]
    phase_step = 2.0 * np.pi / float(p["phase_levels"])
    angles = ((phase.astype(np.int32) - (p["phase_levels"] // 2))
              .astype(np.float32) * phase_step)
    m = levels[np.arange(p["n_rows"])[:, None], mag]
    re = m * np.cos(angles)
    im = m * np.sin(angles)
    out = np.empty((p["n_rows"], 2 * p["n_pairs"]), dtype=np.float32)
    out[:, 0::2] = re
    out[:, 1::2] = im
    return out


# ============================================================================
# VQ K=256 decoder (VQB1 format)
# ============================================================================

def _open_vqb1(path: Path):
    with open(path, "rb") as f:
        hdr = f.read(32)
        assert hdr[:4] == b"VQB1", f"bad vq magic: {hdr[:4]}"
        version, n_experts, n_rows, n_pairs, layer, kind_id, K = \
            struct.unpack("<IIIIIII", hdr[4:32])
        codebook = np.frombuffer(f.read(K * 2 * 4), dtype=np.float32).reshape(K, 2)
        codes_all = np.frombuffer(
            f.read(n_experts * n_rows * n_pairs), dtype=np.uint8
        ).reshape(n_experts, n_rows, n_pairs)
    return {"n_experts": n_experts, "n_rows": n_rows, "n_pairs": n_pairs, "K": K,
            "codebook": codebook, "codes_all": codes_all}


def decode_vq_vqb1(layer: int, kind: str, expert: int,
                    vq_dir: Path = _VQ_DIR) -> np.ndarray | None:
    """Decode VQ K=256 corpus for one expert. Returns fp32 (n_rows, 2*n_pairs).

    Returns None if no VQ corpus exists for this (layer, kind).
    """
    vq_path = vq_dir / f"L{layer:02d}_{kind}.vqb1"
    if not vq_path.exists():
        return None
    v = _open_vqb1(vq_path)
    codes = v["codes_all"][expert]
    recon_pairs = v["codebook"][codes]
    return recon_pairs.reshape(v["n_rows"], 2 * v["n_pairs"]).astype(np.float32)


# ============================================================================
# GGUF metadata cache (one read per process)
# ============================================================================

_GGUF_META_CACHE: dict = {}


def _read_gguf_meta(path: Path):
    """Minimal GGUF metadata reader, cached per-process."""
    p = str(path)
    if p in _GGUF_META_CACHE:
        return _GGUF_META_CACHE[p]
    f = open(path, "rb")
    magic = f.read(4)
    assert magic == b"GGUF", f"bad magic: {magic}"
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]

    def rs():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8")

    def rv(vt):
        if vt == 0: return struct.unpack("<B", f.read(1))[0]
        if vt == 1: return struct.unpack("<b", f.read(1))[0]
        if vt == 2: return struct.unpack("<H", f.read(2))[0]
        if vt == 3: return struct.unpack("<h", f.read(2))[0]
        if vt == 4: return struct.unpack("<I", f.read(4))[0]
        if vt == 5: return struct.unpack("<i", f.read(4))[0]
        if vt == 6: return struct.unpack("<f", f.read(4))[0]
        if vt == 7: return struct.unpack("<B", f.read(1))[0] != 0
        if vt == 8: return rs()
        if vt == 9:
            elem_vt = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            return [rv(elem_vt) for _ in range(n)]
        if vt == 10: return struct.unpack("<Q", f.read(8))[0]
        if vt == 11: return struct.unpack("<q", f.read(8))[0]
        if vt == 12: return struct.unpack("<d", f.read(8))[0]
        raise ValueError(f"unknown vt {vt}")

    meta = {}
    for _ in range(n_kv):
        key = rs()
        vt = struct.unpack("<I", f.read(4))[0]
        meta[key] = rv(vt)
    tinfo = []
    for _ in range(n_tensors):
        name = rs()
        n_dims = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
        ggml_type = struct.unpack("<I", f.read(4))[0]
        offset = struct.unpack("<Q", f.read(8))[0]
        tinfo.append({"name": name, "dims": dims, "type": ggml_type, "offset": offset})
    pos = f.tell()
    align = meta.get("general.alignment", 32)
    pad = (align - pos % align) % align
    data_start = pos + pad
    f.close()
    result = (meta, tinfo, data_start)
    _GGUF_META_CACHE[p] = result
    return result


# ============================================================================
# Codec registry (decoder dispatch)
# ============================================================================

DECODERS = {
    "iq2_xxs":      decode_iq2_xxs,
    "polar_p32_m8": decode_polar_plr2,
    "vq_k256":      decode_vq_vqb1,
}
