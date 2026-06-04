#!/usr/bin/env python3
"""Shared DS4D8F1 codec helpers.

The D8F file format is one organ: constants, record addressing, payload
decoding, source-weight lookup, and codec-error metrics live here so probes,
selectors, and encoders do not import each other as accidental libraries.
"""

from __future__ import annotations

import mmap
import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np

from ds4_safetensors import DEFAULT_MODEL_DIR, SafetensorStore


FUSED_MAGIC = b"DS4D8F1\0"
HEADER_BYTES = 4096
EXPERTS = 256
PROJECTIONS = {"gate": 0, "up": 1, "down": 2}
PROJECTION_NAMES = {value: key for key, value in PROJECTIONS.items()}
SOURCE_STEMS = {"gate": "w1", "up": "w3", "down": "w2"}
K_FIELDS = {"gate": "K_gate", "up": "K_up", "down": "K_down"}
RECORD_BYTES = 64
RECORD_STRUCT = struct.Struct("<IIIIIIQQQIIII")
RECORD_FIELDS = (
    "projection",
    "expert",
    "k",
    "bits",
    "block",
    "row_block",
    "codebook_offset",
    "index_offset",
    "scale_offset",
    "codebook_bytes",
    "index_bytes",
    "scale_bytes",
    "flags",
)
SHAPES = {
    "gate": (2048, 4096),
    "up": (2048, 4096),
    "down": (4096, 2048),
}


def projection_id(projection: str) -> int:
    try:
        return PROJECTIONS[projection]
    except KeyError as exc:
        raise ValueError(f"unknown D8F projection {projection!r}") from exc


def record_offset(projection: str, expert: int) -> int:
    if expert < 0 or expert >= EXPERTS:
        raise ValueError(f"expert {expert} outside [0,{EXPERTS})")
    return HEADER_BYTES + (projection_id(projection) * EXPERTS + expert) * RECORD_BYTES


def record_dict(fields: tuple[int, ...]) -> dict[str, int]:
    return dict(zip(RECORD_FIELDS, fields, strict=True))


def read_record(mapped: mmap.mmap, projection: str, expert: int) -> dict[str, int]:
    return record_dict(RECORD_STRUCT.unpack_from(mapped, record_offset(projection, expert)))


def read_record_from_file(handle: BinaryIO, projection: str, expert: int) -> dict[str, int]:
    handle.seek(record_offset(projection, expert))
    payload = handle.read(RECORD_BYTES)
    if len(payload) != RECORD_BYTES:
        raise ValueError(f"short D8F record read: {len(payload)} bytes")
    return record_dict(RECORD_STRUCT.unpack(payload))


def empty_record_table() -> bytearray:
    table = bytearray(RECORD_BYTES * len(PROJECTIONS) * EXPERTS)
    for projection in range(len(PROJECTIONS)):
        for expert in range(EXPERTS):
            offset = (projection * EXPERTS + expert) * RECORD_BYTES
            RECORD_STRUCT.pack_into(table, offset, projection, expert, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    return table


def unpack_codes(payload: memoryview, bits: int, count: int) -> np.ndarray:
    raw = np.frombuffer(payload, dtype=np.uint8)
    codes = np.empty(count, dtype=np.uint16)
    mask = (1 << bits) - 1
    for block_index in range(count):
        bit_offset = block_index * bits
        byte_offset = bit_offset >> 3
        shift = bit_offset & 7
        word = int(raw[byte_offset])
        word |= int(raw[byte_offset + 1]) << 8
        word |= int(raw[byte_offset + 2]) << 16
        word |= int(raw[byte_offset + 3]) << 24
        codes[block_index] = (word >> shift) & mask
    return codes


def decode_d8f(mapped: mmap.mmap, record: dict[str, int], projection: str) -> np.ndarray:
    rows, cols = SHAPES[projection]
    groups = rows * cols // 8
    codebook_base = int(record["codebook_offset"])
    codebook_end = codebook_base + int(record["codebook_bytes"])
    index_base = int(record["index_offset"])
    index_end = index_base + int(record["index_bytes"])

    codebook = np.frombuffer(mapped[codebook_base:codebook_end], dtype=np.float16)
    codebook = codebook.reshape(int(record["k"]), 8).astype(np.float32)
    codes = unpack_codes(memoryview(mapped)[index_base:index_end], int(record["bits"]), groups)
    decoded = codebook[codes.astype(np.int32)].reshape(rows, cols)

    if projection == "down" and int(record["flags"]) & 1:
        scale_base = int(record["scale_offset"])
        scale_end = scale_base + int(record["scale_bytes"])
        inv_scale = np.frombuffer(mapped[scale_base:scale_end], dtype=np.float16).astype(np.float32)
        if inv_scale.shape[0] != cols:
            raise ValueError(f"down inv_scale length {inv_scale.shape[0]} != cols {cols}")
        decoded = decoded * inv_scale[None, :]
    return decoded


def source_stem(layer: int, projection: str, expert: int) -> str:
    return f"layers.{layer}.ffn.experts.{expert}.{SOURCE_STEMS[projection]}"


def orient_weight(weight: np.ndarray, projection: str, stem: str) -> np.ndarray:
    wanted = SHAPES[projection]
    if weight.shape == wanted:
        return weight
    if weight.T.shape == wanted:
        return weight.T
    raise ValueError(f"{stem}: shape {weight.shape} incompatible with {wanted}")


def source_weight(
    layer: int,
    projection: str,
    expert: int,
    store: SafetensorStore | None = None,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> np.ndarray:
    store = store or SafetensorStore(model_dir)
    stem = source_stem(layer, projection, expert)
    return orient_weight(np.asarray(store.packed_fp4(stem), dtype=np.float32), projection, stem)


load_source = source_weight


def compare(source: np.ndarray, decoded: np.ndarray) -> dict[str, float]:
    source_flat = source.reshape(-1).astype(np.float64)
    decoded_flat = decoded.reshape(-1).astype(np.float64)
    source_norm = float(np.linalg.norm(source_flat))
    decoded_norm = float(np.linalg.norm(decoded_flat))
    diff_norm = float(np.linalg.norm(decoded_flat - source_flat))
    dot = float(np.dot(decoded_flat, source_flat))
    decoded_energy = float(np.dot(decoded_flat, decoded_flat))
    scalar = dot / decoded_energy if decoded_energy > 0.0 else 0.0
    scaled_diff = float(np.linalg.norm(decoded_flat * scalar - source_flat))
    cosine = dot / (source_norm * decoded_norm) if source_norm > 0.0 and decoded_norm > 0.0 else 0.0
    return {
        "source_rms": float(np.sqrt(np.mean(source_flat * source_flat))),
        "decoded_rms": float(np.sqrt(np.mean(decoded_flat * decoded_flat))),
        "source_max_abs": float(np.max(np.abs(source_flat))),
        "decoded_max_abs": float(np.max(np.abs(decoded_flat))),
        "rel_l2": diff_norm / source_norm if source_norm > 0.0 else 0.0,
        "cosine": cosine,
        "optimal_scalar_decoded_to_source": scalar,
        "rel_l2_after_optimal_scalar": scaled_diff / source_norm if source_norm > 0.0 else 0.0,
    }


class D8FFile:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.handle: BinaryIO | None = None
        self.mapped: mmap.mmap | None = None

    def __enter__(self) -> "D8FFile":
        self.open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def open(self) -> None:
        self.handle = self.path.open("rb")
        self.mapped = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)
        if self.mapped[:8] != FUSED_MAGIC:
            self.close()
            raise ValueError(f"{self.path}: not DS4D8F1")

    def close(self) -> None:
        if self.mapped is not None:
            self.mapped.close()
            self.mapped = None
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def record(self, projection: str, expert: int) -> dict[str, int]:
        if self.mapped is None:
            raise RuntimeError("D8FFile is not open")
        return read_record(self.mapped, projection, expert)

    def decode(self, record: dict[str, int], projection: str) -> np.ndarray:
        if self.mapped is None:
            raise RuntimeError("D8FFile is not open")
        return decode_d8f(self.mapped, record, projection)
