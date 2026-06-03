#!/usr/bin/env python3
"""Verify H3385 sparse group-code sidecars against native D8F down codes."""
from __future__ import annotations

import argparse
import json
import math
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

from d8f_down_code_cardinality import read_header, read_native_codes, read_native_records
from d8f_sparse_sidecar_pack import (
    SPARSE_HEADER_BYTES,
    SPARSE_MAGIC,
    SPARSE_RECORD,
    SPARSE_RECORD_BYTES,
    SPARSE_VERSION,
)


def read_sparse_header(handle) -> dict[str, Any]:
    handle.seek(0)
    prefix = handle.read(16)
    if len(prefix) != 16 or prefix[:8] != SPARSE_MAGIC:
        raise ValueError("bad sparse sidecar magic")
    version, header_bytes = struct.unpack("<II", prefix[8:16])
    if version != SPARSE_VERSION:
        raise ValueError(f"bad sparse sidecar version: {version}")
    if header_bytes <= 0 or header_bytes > SPARSE_HEADER_BYTES - 16:
        raise ValueError(f"bad sparse header size: {header_bytes}")
    return json.loads(handle.read(header_bytes).decode("utf-8"))


def unpack_inverse(data: bytes, rows: int, bits: int) -> np.ndarray:
    if bits <= 0:
        return np.zeros(rows, dtype=np.uint16)
    unpacked = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")
    unpacked = unpacked[: rows * bits].reshape(rows, bits).astype(np.uint16, copy=False)
    shifts = (np.arange(bits, dtype=np.uint16) & 15)[None, :]
    return np.sum(unpacked << shifts, axis=1, dtype=np.uint16)


def read_record(handle, expert: int) -> dict[str, int]:
    handle.seek(SPARSE_HEADER_BYTES + expert * SPARSE_RECORD_BYTES)
    fields = SPARSE_RECORD.unpack(handle.read(SPARSE_RECORD_BYTES))
    names = (
        "expert",
        "rows",
        "groups",
        "k",
        "unique_count",
        "max_group_unique",
        "group_prefix_offset",
        "inverse_offset_table_offset",
        "unique_code_offset",
        "inverse_bits_offset",
        "group_prefix_bytes",
        "inverse_offset_table_bytes",
        "unique_code_bytes",
        "inverse_bits_bytes",
        "inverse_bits_bits",
        "flags",
    )
    return dict(zip(names, fields, strict=True))


def read_exact_sparse_codes(handle, record: dict[str, int]) -> np.ndarray:
    rows = int(record["rows"])
    groups = int(record["groups"])
    handle.seek(int(record["group_prefix_offset"]))
    group_prefix = np.frombuffer(handle.read(int(record["group_prefix_bytes"])), dtype="<u4").copy()
    handle.seek(int(record["inverse_offset_table_offset"]))
    inverse_offsets = np.frombuffer(handle.read(int(record["inverse_offset_table_bytes"])), dtype="<u4").copy()
    handle.seek(int(record["unique_code_offset"]))
    unique_codes = np.frombuffer(handle.read(int(record["unique_code_bytes"])), dtype="<u2").copy()
    handle.seek(int(record["inverse_bits_offset"]))
    inverse_blob = handle.read(int(record["inverse_bits_bytes"]))
    if group_prefix.size != groups + 1:
        raise ValueError(f"group prefix length mismatch: {group_prefix.size} != {groups + 1}")
    if inverse_offsets.size != groups + 1:
        raise ValueError(f"inverse offset length mismatch: {inverse_offsets.size} != {groups + 1}")
    codes = np.empty((rows, groups), dtype=np.uint16)
    for group in range(groups):
        unique_start = int(group_prefix[group])
        unique_end = int(group_prefix[group + 1])
        inverse_start = int(inverse_offsets[group])
        inverse_end = int(inverse_offsets[group + 1])
        group_unique = unique_codes[unique_start:unique_end]
        bits = max(1, math.ceil(math.log2(int(group_unique.size))))
        inverse = unpack_inverse(inverse_blob[inverse_start:inverse_end], rows, bits)
        if np.any(inverse >= group_unique.size):
            raise ValueError(f"group {group}: inverse index out of range")
        codes[:, group] = group_unique[inverse]
    return codes


def verify_layer(pack_dir: Path, layer_entry: dict[str, Any]) -> dict[str, Any]:
    d8f_path = pack_dir / layer_entry["d8f"]
    sidecar_path = pack_dir / layer_entry["sidecar"]
    mismatches = 0
    with d8f_path.open("rb") as d8f_handle, sidecar_path.open("rb") as sidecar_handle:
        sparse_header = read_sparse_header(sidecar_handle)
        d8f_header = read_header(d8f_handle)
        native_records = read_native_records(d8f_handle, d8f_header)
        for expert in layer_entry["hot_experts"]:
            expert_id = int(expert)
            if expert_id not in native_records:
                raise ValueError(f"{d8f_path.name}: missing native record for expert {expert_id}")
            record = read_record(sidecar_handle, expert_id)
            if int(record["expert"]) != expert_id:
                raise ValueError(f"{sidecar_path.name}: record expert mismatch for {expert_id}")
            sparse_codes = read_exact_sparse_codes(sidecar_handle, record)
            native_codes = read_native_codes(d8f_handle, native_records[expert_id])
            mismatch_count = int(np.count_nonzero(sparse_codes != native_codes))
            mismatches += mismatch_count
            if mismatch_count:
                raise ValueError(f"L{layer_entry['layer']:02d} E{expert_id:03d}: {mismatch_count} code mismatches")
    return {
        "layer": int(layer_entry["layer"]),
        "experts": len(layer_entry["hot_experts"]),
        "mismatches": mismatches,
        "sidecar_unique_total": int(sparse_header["unique_total"]),
        "sidecar_sparse_bytes": int(sparse_header["sparse_bytes"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="H3385 sparse manifest JSON")
    parser.add_argument("--progress-every", type=int, default=5, help="print progress every N layers")
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pack_dir = Path(manifest["pack"])
    if not pack_dir.is_absolute():
        pack_dir = (manifest_path.parent / pack_dir).resolve()
    started = time.time()
    reports = []
    for index, layer_entry in enumerate(manifest["layers"], 1):
        report = verify_layer(pack_dir, layer_entry)
        reports.append(report)
        if index == 1 or index % max(args.progress_every, 1) == 0 or index == len(manifest["layers"]):
            elapsed = time.time() - started
            print(
                f"verified layers={index}/{len(manifest['layers'])} "
                f"last=L{report['layer']:02d} mismatches=0 elapsed={elapsed:.1f}s",
                flush=True,
            )
    print(
        "H3385 sparse sidecar verification ok "
        f"layers={len(reports)} experts={sum(report['experts'] for report in reports)} "
        f"elapsed={time.time() - started:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
