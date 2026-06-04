#!/usr/bin/env python3
"""Compare one DS4D8F expert record against the original safetensors weight.

This is a codec truth probe, not a generation benchmark.  It decodes the
bitpacked D8F payload exactly as the Metal runtime sees it:

- gate/up: fp16 codebook entries are the effective weights.
- down: fp16 codebook entries multiply fp16 inverse activation-scale values
  from the record scale payload, matching `d8f_mid_value` in Metal.

It also reports the best single scalar that maps decoded D8F weights back to
source weights.  A low `rel_l2_after_optimal_scalar` with a large scalar is a
direct signature of a missing codebook-scale field in the on-disk format.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from ds4_safetensors import DEFAULT_MODEL_DIR, SafetensorStore  # noqa: E402


HEADER_BYTES = 4096
EXPERTS = 256
RECORD_BYTES = 64
RECORD_STRUCT = struct.Struct("<IIIIIIQQQIIII")
PROJECTIONS = {"gate": 0, "up": 1, "down": 2}
SOURCE_STEMS = {"gate": "w1", "up": "w3", "down": "w2"}
SHAPES = {
    "gate": (2048, 4096),
    "up": (2048, 4096),
    "down": (4096, 2048),
}


def read_record(mapped: mmap.mmap, projection: str, expert: int) -> dict[str, int]:
    projection_id = PROJECTIONS[projection]
    offset = HEADER_BYTES + (projection_id * EXPERTS + expert) * RECORD_BYTES
    fields = RECORD_STRUCT.unpack_from(mapped, offset)
    names = (
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
    return dict(zip(names, fields, strict=True))


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


def load_source(layer: int, projection: str, expert: int, store: SafetensorStore | None = None) -> np.ndarray:
    store = store or SafetensorStore(DEFAULT_MODEL_DIR)
    stem = f"layers.{layer}.ffn.experts.{expert}.{SOURCE_STEMS[projection]}"
    source = np.asarray(store.packed_fp4(stem), dtype=np.float32)
    if projection in ("gate", "up"):
        return source if source.shape == SHAPES[projection] else source.T
    return source if source.shape == SHAPES[projection] else source.T


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d8f", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--projection", required=True, choices=sorted(PROJECTIONS))
    parser.add_argument("--expert", required=True, type=int)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.d8f.open("rb") as handle:
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            if mapped[:8] != b"DS4D8F1\0":
                raise ValueError(f"{args.d8f}: not DS4D8F1")
            store = SafetensorStore(args.model_dir)
            record = read_record(mapped, args.projection, args.expert)
            source = load_source(args.layer, args.projection, args.expert, store)
            decoded = decode_d8f(mapped, record, args.projection)
            result = {
                "d8f": os.fspath(args.d8f),
                "layer": args.layer,
                "projection": args.projection,
                "expert": args.expert,
                "record": record,
                "metrics": compare(source, decoded),
            }
            if args.projection == "down" and int(record["flags"]) & 1:
                scale_base = int(record["scale_offset"])
                scale_end = scale_base + int(record["scale_bytes"])
                inv_scale = np.frombuffer(mapped[scale_base:scale_end], dtype=np.float16).astype(np.float32)
                result["inv_scale"] = {
                    "len": int(inv_scale.shape[0]),
                    "min": float(np.min(inv_scale)),
                    "max": float(np.max(inv_scale)),
                    "mean": float(np.mean(inv_scale)),
                }
            print(json.dumps(result, indent=2, sort_keys=True))
        finally:
            mapped.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
