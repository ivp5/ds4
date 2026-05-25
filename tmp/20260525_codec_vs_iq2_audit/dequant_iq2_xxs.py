"""IQ2_XXS dequantizer in Python (numpy).

Block layout (from H1786 / metal/moe.metal):
- 66 bytes per block of 256 weights (~2.06 bits/weight)
- bytes 0-1: f16 scale `d` (little-endian)
- bytes 2-65: 8 sub-blocks of 8 bytes each
  - per sub-block: 4 bytes aux0 (4 grid indices, 1 byte each)
                 + 4 bytes aux1 (4 sign bits 7-bit each + top 4 bits = extra-scale exp)

Per-weight decode:
  scale = d * (2 * (aux1 >> 28) + 1) * 0.125
  sign  = (aux1 >> (7 * byte_lane)) & 127     # byte_lane in [0..3]
  grid_index = (aux0 >> (8 * byte_lane)) & 255
  value = signed_grid[grid_index * 1024 + sign_table[sign] * 8 + grid_lane] * scale
  (grid_lane in [0..7] = position within the 8-weight grid entry)
"""
import struct
import numpy as np
from pathlib import Path

_HERE = Path(__file__).parent

# Lookup tables from metal/moe.metal
IQ2XXS_GRID = np.load(_HERE / "iq2xxs_grid.npy")  # 256 uint64, each = 8 bytes of int8
KSIGNS_IQ2XS = np.load(_HERE / "ksigns_iq2xs.npy")  # 128 uint8, sign-bitfield expansion

# Expand IQ2_XXS grid into a signed_grid[256 * 8] int8 array (the byte view).
# Each uint64 grid entry encodes 8 int8 values (low byte first).
_GRID_BYTES = np.frombuffer(IQ2XXS_GRID.tobytes(), dtype=np.int8).reshape(256, 8)


def _decode_sign(sign_byte: int, lane: int) -> int:
    """Return -1 or +1 based on whether bit `lane` of ksigns[sign_byte] is set.

    Per metal/moe.metal:880: `signg & kmask_iq2xs[j] ? -1.f : 1.f`
    where kmask_iq2xs[j] = 1 << j (the per-lane sign bit).
    """
    expanded = KSIGNS_IQ2XS[sign_byte & 127]
    return -1 if (expanded & (1 << lane)) else 1


def dequant_iq2_xxs(blocks: bytes, n_weights: int) -> np.ndarray:
    """Dequantize n_weights of IQ2_XXS-encoded blocks to fp32.

    blocks: raw IQ2_XXS bytes (n_blocks * 66, where n_blocks = n_weights // 256)
    n_weights: must be a multiple of 256
    Returns: numpy fp32 array of shape (n_weights,)
    """
    assert n_weights % 256 == 0, f"n_weights must be multiple of 256, got {n_weights}"
    n_blocks = n_weights // 256
    expected_bytes = n_blocks * 66
    assert len(blocks) == expected_bytes, f"expected {expected_bytes} bytes, got {len(blocks)}"

    out = np.empty(n_weights, dtype=np.float32)

    # Vectorize where possible: per-block scale + per-sub-block (aux0, aux1)
    arr = np.frombuffer(blocks, dtype=np.uint8).reshape(n_blocks, 66)
    # Scale: f16 from bytes [0:2] of each block
    scale_bytes = arr[:, 0:2].copy()  # (n_blocks, 2)
    d_f16 = np.frombuffer(scale_bytes.tobytes(), dtype=np.float16).astype(np.float32)  # (n_blocks,)

    # Per sub-block (8 per block), aux0/aux1 are uint32 little-endian
    sub_bytes = arr[:, 2:].reshape(n_blocks, 8, 8)  # (n_blocks, 8 sub, 8 bytes per sub)
    aux0 = (sub_bytes[:, :, 0].astype(np.uint32)
            | (sub_bytes[:, :, 1].astype(np.uint32) << 8)
            | (sub_bytes[:, :, 2].astype(np.uint32) << 16)
            | (sub_bytes[:, :, 3].astype(np.uint32) << 24))
    aux1 = (sub_bytes[:, :, 4].astype(np.uint32)
            | (sub_bytes[:, :, 5].astype(np.uint32) << 8)
            | (sub_bytes[:, :, 6].astype(np.uint32) << 16)
            | (sub_bytes[:, :, 7].astype(np.uint32) << 24))
    # aux0/aux1 shape: (n_blocks, 8)

    # extra-scale exponent from top 4 bits of aux1
    sub_scale = (2.0 * (aux1 >> np.uint32(28)).astype(np.float32) + 1.0) * 0.125  # (n_blocks, 8)

    # Per sub-block, 4 byte_lanes (0..3); each lane gives a grid_index byte + 7-bit sign byte
    # grid_index = (aux0 >> (8*lane)) & 0xff; sign = (aux1 >> (7*lane)) & 0x7f
    # Per byte_lane, the 8-weight grid entry IQ2XXS_GRID[grid_index] (8 int8 values) is multiplied by per-lane sign-mask
    for bi in range(n_blocks):
        d_block = d_f16[bi]
        for si in range(8):  # sub-block
            ss = sub_scale[bi, si] * d_block  # effective scale for this sub-block
            a0 = int(aux0[bi, si])
            a1 = int(aux1[bi, si])
            for byte_lane in range(4):  # 4 byte_lanes per sub-block
                grid_index = (a0 >> (8 * byte_lane)) & 0xff
                sign_byte = (a1 >> (7 * byte_lane)) & 0x7f
                expanded_sign = KSIGNS_IQ2XS[sign_byte]  # uint8 with one bit per lane
                # 8 weights per byte_lane (positions 0..7 within grid)
                base = bi * 256 + si * 32 + byte_lane * 8
                for j in range(8):
                    v = _GRID_BYTES[grid_index, j]
                    sign_factor = -1.0 if (expanded_sign & (1 << j)) else 1.0
                    out[base + j] = float(v) * sign_factor * ss

    return out


def dequant_iq2_xxs_fast(blocks: bytes, n_weights: int) -> np.ndarray:
    """Vectorized variant — equivalent but ~100x faster on numpy.

    All per-element loops replaced with vectorized indexing + broadcast.
    """
    assert n_weights % 256 == 0
    n_blocks = n_weights // 256
    arr = np.frombuffer(blocks, dtype=np.uint8).reshape(n_blocks, 66)
    d_f16 = np.frombuffer(arr[:, 0:2].copy().tobytes(), dtype=np.float16).astype(np.float32)
    sub_bytes = arr[:, 2:].reshape(n_blocks, 8, 8).astype(np.uint32)
    aux0 = sub_bytes[:, :, 0] | (sub_bytes[:, :, 1] << 8) | (sub_bytes[:, :, 2] << 16) | (sub_bytes[:, :, 3] << 24)
    aux1 = sub_bytes[:, :, 4] | (sub_bytes[:, :, 5] << 8) | (sub_bytes[:, :, 6] << 16) | (sub_bytes[:, :, 7] << 24)
    sub_scale = (2.0 * (aux1 >> np.uint32(28)).astype(np.float32) + 1.0) * 0.125
    eff_scale = sub_scale * d_f16[:, None]  # (n_blocks, 8)

    # For each (block, sub, byte_lane), get grid_index + sign_byte
    # Shape: (n_blocks, 8 sub, 4 byte_lanes)
    byte_lanes = np.arange(4, dtype=np.uint32)[None, None, :]
    grid_index = ((aux0[:, :, None] >> (8 * byte_lanes)) & 0xff).astype(np.uint8)  # (B,8,4)
    sign_byte = ((aux1[:, :, None] >> (7 * byte_lanes)) & 0x7f).astype(np.uint8)   # (B,8,4)

    # Grid lookup: (B,8,4,8) int8 values from _GRID_BYTES
    grid_vals = _GRID_BYTES[grid_index]  # (B,8,4,8) int8
    grid_vals_f = grid_vals.astype(np.float32)

    # Sign expansion: ksigns[sign_byte] is a uint8 with bit-per-lane;
    # for each of the 8 lanes (j in 0..7), check if bit j is set
    expanded = KSIGNS_IQ2XS[sign_byte]  # (B,8,4) uint8 mask
    lane_bits = (1 << np.arange(8, dtype=np.uint8))  # (8,)
    sign_mask = (expanded[..., None] & lane_bits) > 0  # (B,8,4,8) bool
    sign_factor = np.where(sign_mask, -1.0, 1.0).astype(np.float32)

    weighted = grid_vals_f * sign_factor * eff_scale[:, :, None, None]  # (B,8,4,8)
    # Flatten: B blocks × 8 sub × 4 byte_lanes × 8 weights = 256 weights/block
    out = weighted.reshape(n_blocks, 256)
    return out.reshape(-1)


if __name__ == "__main__":
    # Self-test: create a synthetic block (zeros aux/scale=1.0) and verify both paths agree
    n_w = 256
    # Construct a block with d=1.0, all sub-blocks aux0=0 (grid index 0), aux1=0 (sign 0, extra-scale 0)
    block = bytearray(66)
    # f16 1.0 = 0x3c00
    struct.pack_into("<H", block, 0, 0x3c00)
    out_slow = dequant_iq2_xxs(bytes(block), n_w)
    out_fast = dequant_iq2_xxs_fast(bytes(block), n_w)
    diff = np.max(np.abs(out_slow - out_fast))
    print(f"slow vs fast max abs diff: {diff:.2e}")
    print(f"out_fast[0:16]: {out_fast[:16]}")
    print(f"sub_scale for zero aux1: {(2*0+1)*0.125}")
    # Expected: grid[0] = 0x0808080808080808 = 8 copies of int8(8); scale = 1.0 * 0.125 = 0.125
    # So each weight = 8 * 1 * 0.125 = 1.0 (no sign flip)
    expected = 1.0
    err = np.max(np.abs(out_fast - expected))
    print(f"max abs err vs expected {expected}: {err:.2e}")
    assert err < 1e-5, f"self-test FAILED: {err}"
    assert diff < 1e-5, f"slow/fast mismatch: {diff}"
    print("PASS")
