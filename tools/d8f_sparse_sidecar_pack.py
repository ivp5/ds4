#!/usr/bin/env python3
"""Build H3385-style exact sparse down-code sidecars from H3384 native codes.

The sidecar stores, per selected down expert and group:
- sorted unique code IDs used by the 4096 output rows
- a bitpacked row -> local-unique-code inverse map

This is exact: no weights or codes are changed.  It is a layout/encoding sidecar
for the sparse selected-down kernel proved by the canaries.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

from d8f_down_code_cardinality import (
    DOWN_GROUPS,
    EXPERTS,
    HEADER_BYTES,
    read_header,
    read_native_codes,
    read_native_records,
    read_record,
)

SPARSE_MAGIC = b"DS4D8FS\0"
SPARSE_VERSION = 1
SPARSE_HEADER_BYTES = 4096
SPARSE_RECORD_BYTES = 80
SPARSE_RECORD = struct.Struct("<IIIIIIQQQQIIIIII")


def align(handle, alignment: int = 16) -> int:
    position = handle.tell()
    padding = (-position) % alignment
    if padding:
        handle.write(b"\0" * padding)
        position += padding
    return position


def write_header(handle, header: dict[str, Any]) -> None:
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if 16 + len(raw) > SPARSE_HEADER_BYTES:
        raise ValueError(f"sparse header too large: {len(raw)}")
    handle.seek(0)
    handle.write(SPARSE_MAGIC)
    handle.write(struct.pack("<II", SPARSE_VERSION, len(raw)))
    handle.write(raw)
    handle.write(b"\0" * (SPARSE_HEADER_BYTES - 16 - len(raw)))


def pack_inverse(values: np.ndarray, bits: int) -> bytes:
    if bits <= 0:
        return b""
    shifts = np.arange(bits, dtype=np.uint16)
    planes = ((values.astype(np.uint16)[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    return np.packbits(planes.reshape(-1), bitorder="little").tobytes()


def build_sparse_blobs(codes: np.ndarray, *, k: int) -> dict[str, Any]:
    rows, groups = codes.shape
    group_prefix = np.zeros(groups + 1, dtype="<u4")
    inverse_byte_offsets = np.zeros(groups + 1, dtype="<u4")
    unique_chunks: list[np.ndarray] = []
    inverse_chunks: list[bytes] = []
    unique_cursor = 0
    inverse_cursor = 0
    max_group_unique = 0
    for group in range(groups):
        unique_codes, inverse = np.unique(codes[:, group], return_inverse=True)
        if unique_codes.size == 0 or int(unique_codes[-1]) >= k:
            raise ValueError(f"group {group}: invalid sparse codes for k={k}")
        group_prefix[group] = unique_cursor
        inverse_byte_offsets[group] = inverse_cursor
        unique_count = int(unique_codes.size)
        max_group_unique = max(max_group_unique, unique_count)
        bits = max(1, math.ceil(math.log2(unique_count)))
        packed_inverse = pack_inverse(inverse.astype(np.uint16, copy=False), bits)
        unique_chunks.append(unique_codes.astype("<u2", copy=False))
        inverse_chunks.append(packed_inverse)
        unique_cursor += unique_count
        inverse_cursor += len(packed_inverse)
    group_prefix[groups] = unique_cursor
    inverse_byte_offsets[groups] = inverse_cursor
    unique_blob = np.concatenate(unique_chunks).astype("<u2", copy=False).tobytes()
    inverse_blob = b"".join(inverse_chunks)
    return {
        "rows": rows,
        "groups": groups,
        "unique_count": unique_cursor,
        "max_group_unique": max_group_unique,
        "group_prefix": group_prefix.tobytes(),
        "inverse_byte_offsets": inverse_byte_offsets.tobytes(),
        "unique_codes": unique_blob,
        "inverse_bits": inverse_blob,
    }


def link_or_copy(source: Path, destination: Path, *, hardlink: bool) -> str:
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    if hardlink:
        os.link(source, destination)
        return "hardlink"
    shutil.copy2(source, destination)
    return "copy"


def build_layer_sidecar(layer_entry: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    source_d8f = Path(layer_entry["d8f"])
    layer = int(layer_entry["layer"])
    hot_experts = [int(expert) for expert in layer_entry["hot_experts"]]
    sidecar_name = f"ds4_L{layer:02d}_down_sparse_groupcode_top{len(hot_experts)}.d8fs"
    sidecar_path = out_dir / sidecar_name
    if sidecar_path.exists():
        raise FileExistsError(f"sidecar already exists: {sidecar_path}")
    records = [bytes(SPARSE_RECORD_BYTES) for _ in range(EXPERTS)]
    layer_unique_total = 0
    layer_sparse_bytes = 0
    layer_native_equivalent = 0
    layer_metrics: list[dict[str, Any]] = []
    with source_d8f.open("rb") as source_handle:
        source_header = read_header(source_handle)
        native_records = read_native_records(source_handle, source_header)
        with sidecar_path.open("wb") as sidecar_handle:
            sidecar_handle.write(b"\0" * (SPARSE_HEADER_BYTES + EXPERTS * SPARSE_RECORD_BYTES))
            for expert in hot_experts:
                if expert not in native_records:
                    raise ValueError(f"L{layer:02d} E{expert:03d}: missing native-code sidecar")
                down = read_record(source_handle, "down", expert)
                native = native_records[expert]
                codes = read_native_codes(source_handle, native)
                blobs = build_sparse_blobs(codes, k=int(down["k"]))
                group_prefix_offset = align(sidecar_handle)
                sidecar_handle.write(blobs["group_prefix"])
                inverse_offset_table_offset = align(sidecar_handle)
                sidecar_handle.write(blobs["inverse_byte_offsets"])
                unique_code_offset = align(sidecar_handle)
                sidecar_handle.write(blobs["unique_codes"])
                inverse_bits_offset = align(sidecar_handle)
                sidecar_handle.write(blobs["inverse_bits"])
                group_prefix_bytes = len(blobs["group_prefix"])
                inverse_offset_table_bytes = len(blobs["inverse_byte_offsets"])
                unique_code_bytes = len(blobs["unique_codes"])
                inverse_bits_bytes = len(blobs["inverse_bits"])
                sparse_bytes = group_prefix_bytes + inverse_offset_table_bytes + unique_code_bytes + inverse_bits_bytes
                native_bytes = int(native["rows"]) * int(native["groups"]) * 2
                record = SPARSE_RECORD.pack(
                    expert,
                    int(native["rows"]),
                    int(native["groups"]),
                    int(down["k"]),
                    int(blobs["unique_count"]),
                    int(blobs["max_group_unique"]),
                    group_prefix_offset,
                    inverse_offset_table_offset,
                    unique_code_offset,
                    inverse_bits_offset,
                    group_prefix_bytes,
                    inverse_offset_table_bytes,
                    unique_code_bytes,
                    inverse_bits_bytes,
                    inverse_bits_bytes * 8,
                    1,
                )
                records[expert] = record
                layer_unique_total += int(blobs["unique_count"])
                layer_sparse_bytes += sparse_bytes
                layer_native_equivalent += native_bytes
                layer_metrics.append(
                    {
                        "expert": expert,
                        "k": int(down["k"]),
                        "unique_count": int(blobs["unique_count"]),
                        "unique_mean": int(blobs["unique_count"]) / int(native["groups"]),
                        "max_group_unique": int(blobs["max_group_unique"]),
                        "sparse_bytes": sparse_bytes,
                        "native_bytes": native_bytes,
                        "sparse_vs_native": sparse_bytes / native_bytes,
                    }
                )
            payload_end = align(sidecar_handle)
            header = {
                "format": "DS4D8FS sparse down group-code sidecar",
                "encoding": "H3385 exact sparse group-code down sidecars",
                "layer": layer,
                "source_d8f": source_d8f.name,
                "selected_experts": hot_experts,
                "record_bytes": SPARSE_RECORD_BYTES,
                "records": EXPERTS,
                "rows": 4096,
                "groups": DOWN_GROUPS,
                "bitpacked_inverse": "per-group byte-aligned little-endian local code index",
                "payload_end": payload_end,
                "unique_total": layer_unique_total,
                "sparse_bytes": layer_sparse_bytes,
                "native_equivalent_bytes": layer_native_equivalent,
                "sparse_vs_native": layer_sparse_bytes / max(layer_native_equivalent, 1),
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            write_header(sidecar_handle, header)
            sidecar_handle.seek(SPARSE_HEADER_BYTES)
            for record in records:
                sidecar_handle.write(record)
    return {
        "layer": layer,
        "d8f": source_d8f.name,
        "sidecar": sidecar_name,
        "hot_experts": hot_experts,
        "unique_total": layer_unique_total,
        "sparse_bytes": layer_sparse_bytes,
        "native_equivalent_bytes": layer_native_equivalent,
        "sparse_vs_native": layer_sparse_bytes / max(layer_native_equivalent, 1),
        "experts": layer_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="H3384 manifest JSON")
    parser.add_argument("--out-dir", required=True, help="output H3385 pack directory")
    parser.add_argument("--copy-d8f", action="store_true", help="copy D8F files instead of hardlinking")
    parser.add_argument("--force", action="store_true", help="allow an existing empty output directory")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    layers = manifest.get("layers")
    if not isinstance(layers, list):
        raise SystemExit("manifest must contain layers list")
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        if any(out_dir.iterdir()):
            raise SystemExit(f"output directory already exists and is not empty: {out_dir}")
        if not args.force:
            raise SystemExit(f"output directory already exists; use --force only for an empty directory: {out_dir}")
    else:
        out_dir.mkdir(parents=True, exist_ok=False)

    layer_reports: list[dict[str, Any]] = []
    link_modes: set[str] = set()
    total_sparse_bytes = 0
    total_native_equivalent = 0
    total_unique = 0
    started = time.time()
    for layer_entry in layers:
        layer = int(layer_entry["layer"])
        source_d8f = Path(layer_entry["d8f"])
        destination_d8f = out_dir / source_d8f.name
        mode = link_or_copy(source_d8f, destination_d8f, hardlink=not args.copy_d8f)
        link_modes.add(mode)
        report = build_layer_sidecar(layer_entry, out_dir)
        layer_reports.append(report)
        total_sparse_bytes += int(report["sparse_bytes"])
        total_native_equivalent += int(report["native_equivalent_bytes"])
        total_unique += int(report["unique_total"])
        print(
            f"H3385 sparse sidecar L{layer:02d} unique={report['unique_total']} "
            f"sparse={report['sparse_bytes']/1048576.0:.2f}MiB "
            f"native={report['native_equivalent_bytes']/1048576.0:.2f}MiB "
            f"ratio={report['sparse_vs_native']:.3f}",
            flush=True,
        )
    pack_manifest = {
        "pack": ".",
        "pack_name": out_dir.name,
        "encoding": "H3385 H3384 plus exact sparse group-code down sidecars",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "self_contained": True,
        "runtime_dependencies": [],
        "provenance": {
            "base_pack_name": Path(str(manifest.get("pack", ""))).name,
            "source_manifest_name": manifest_path.name,
        },
        "d8f_link_modes": sorted(link_modes),
        "layers": layer_reports,
        "d8f_layers": len(layer_reports),
        "sparse_sidecar_total_bytes": total_sparse_bytes,
        "native_equivalent_bytes": total_native_equivalent,
        "sparse_vs_native": total_sparse_bytes / max(total_native_equivalent, 1),
        "unique_total": total_unique,
        "build_elapsed_s": time.time() - started,
    }
    (out_dir / "H3385_SPARSE_GROUPCODE_MANIFEST.json").write_text(
        json.dumps(pack_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "H3385 sparse pack complete "
        f"layers={len(layer_reports)} sparse={total_sparse_bytes/1048576.0:.2f}MiB "
        f"native={total_native_equivalent/1048576.0:.2f}MiB "
        f"ratio={pack_manifest['sparse_vs_native']:.3f} out={out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
