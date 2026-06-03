#!/usr/bin/env python3
"""Projection-specific sparse D8F splicer and hardlink pack builder."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import time
from pathlib import Path
from typing import Any

FUSED_MAGIC = b"DS4D8F1\0"
HEADER_BYTES = 4096
EXPERTS = 256
PROJECTIONS = ("gate", "up", "down")
GATEUP_ROW_BLOCKS = 16
GATEUP_OVERLAY_SENTINEL = 0xFFFFFFFF
PROJECTION_ID = {name: index for index, name in enumerate(PROJECTIONS)}
FUSED_RECORD_BYTES = 64
FUSED_RECORD = struct.Struct("<IIIIIIQQQIIII")
SIDECAR_RECORD_BYTES = 64
SIDECAR_RECORD = struct.Struct("<IIIIQQIIIIffff")


def emit(**fields: object) -> None:
    fields.setdefault("updated_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print(json.dumps(fields, sort_keys=True), flush=True)


def d8f_path(root: Path, layer: int) -> Path:
    exact = root / f"ds4_L{layer:02d}_gate_up_down_VQD8_noE8_rank1.d8f"
    if exact.exists():
        return exact
    matches = sorted(root.glob(f"ds4_L{layer:02d}_gate_up_down_*.d8f"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return exact
    raise RuntimeError(f"ambiguous D8F files for L{layer:02d} under {root}: {[p.name for p in matches]}")


def read_payload(path: Path, offset: int, size: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"{path}: truncated payload offset={offset} size={size}")
    return data


def append_aligned(handle, data: bytes, alignment: int = 16) -> int:
    position = handle.tell()
    padding = (-position) % alignment
    if padding:
        handle.write(b"\0" * padding)
        position += padding
    handle.write(data)
    return position


def rewrite_header(handle, header: dict[str, Any]) -> None:
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if 16 + len(raw) > HEADER_BYTES:
        raise ValueError(f"header too large: {len(raw)}")
    handle.seek(0)
    handle.write(FUSED_MAGIC)
    handle.write(struct.pack("<II", 1, len(raw)))
    handle.write(raw)
    handle.write(b"\0" * (HEADER_BYTES - 16 - len(raw)))


def read_d8f(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(16)
        if len(prefix) != 16 or prefix[:8] != FUSED_MAGIC:
            result.update({"valid": False, "reason": "bad_magic_or_short_header", "bytes": size})
            return result
        version = struct.unpack("<I", prefix[8:12])[0]
        header_len = struct.unpack("<I", prefix[12:16])[0]
        if version != 1 or not (0 < header_len <= HEADER_BYTES - 16):
            result.update({"valid": False, "reason": "bad_header_len", "bytes": size})
            return result
        header = json.loads(handle.read(header_len).decode("utf-8"))
        handle.seek(HEADER_BYTES)
        records = []
        for _ in range(len(PROJECTIONS) * EXPERTS):
            raw = handle.read(FUSED_RECORD_BYTES)
            if len(raw) != FUSED_RECORD_BYTES:
                result.update({"valid": False, "reason": "truncated_record_table", "bytes": size})
                return result
            records.append(FUSED_RECORD.unpack(raw))
        gateup_overlays: dict[tuple[int, int, int], tuple[Any, ...]] = {}
        overlay_slot_offset = int(header.get("gateup_overlay_slot_table_offset", 0) or 0)
        if overlay_slot_offset:
            overlay_record_offset = int(header.get("gateup_overlay_record_offset", 0) or 0)
            overlay_slot_entries = int(header.get("gateup_overlay_slot_entries", 2 * EXPERTS * GATEUP_ROW_BLOCKS) or 0)
            overlay_record_bytes = int(header.get("gateup_overlay_record_bytes", FUSED_RECORD_BYTES) or FUSED_RECORD_BYTES)
            overlay_records = int(header.get("gateup_overlay_records", 0) or 0)
            overlay_sentinel = int(header.get("gateup_overlay_sentinel", GATEUP_OVERLAY_SENTINEL) or GATEUP_OVERLAY_SENTINEL)
            if (overlay_slot_entries != 2 * EXPERTS * GATEUP_ROW_BLOCKS or
                    overlay_record_bytes != FUSED_RECORD_BYTES or
                    overlay_sentinel != GATEUP_OVERLAY_SENTINEL or
                    not overlay_record_offset or
                    overlay_slot_offset + overlay_slot_entries * 4 > size or
                    overlay_record_offset + overlay_records * FUSED_RECORD_BYTES > size):
                result.update({"valid": False, "reason": "unsupported_gateup_overlay_shape", "bytes": size})
                return result
            handle.seek(overlay_slot_offset)
            slot_table = handle.read(overlay_slot_entries * 4)
            handle.seek(overlay_record_offset)
            overlay_table = handle.read(overlay_records * FUSED_RECORD_BYTES)
            if len(slot_table) != overlay_slot_entries * 4 or len(overlay_table) != overlay_records * FUSED_RECORD_BYTES:
                result.update({"valid": False, "reason": "truncated_gateup_overlay", "bytes": size})
                return result
            for slot_index in range(overlay_slot_entries):
                slot = struct.unpack_from("<I", slot_table, slot_index * 4)[0]
                if slot == overlay_sentinel:
                    continue
                if slot >= overlay_records:
                    result.update({"valid": False, "reason": "gateup_overlay_slot_out_of_range", "bytes": size})
                    return result
                projection = slot_index // (EXPERTS * GATEUP_ROW_BLOCKS)
                within_projection = slot_index % (EXPERTS * GATEUP_ROW_BLOCKS)
                expert = within_projection // GATEUP_ROW_BLOCKS
                row_block = within_projection % GATEUP_ROW_BLOCKS
                gateup_overlays[(projection, expert, row_block)] = FUSED_RECORD.unpack_from(
                    overlay_table, slot * FUSED_RECORD_BYTES
                )
        sidecars: dict[int, tuple[Any, ...]] = {}
        sidecar_offset = int(header.get("sidecar_table_offset", 0) or 0)
        sidecar_records = int(header.get("sidecar_records", 0) or 0)
        sidecar_record_bytes = int(header.get("sidecar_record_bytes", SIDECAR_RECORD_BYTES) or SIDECAR_RECORD_BYTES)
        if sidecar_offset:
            if sidecar_records != EXPERTS or sidecar_record_bytes != SIDECAR_RECORD_BYTES:
                result.update({"valid": False, "reason": "unsupported_sidecar_table_shape", "bytes": size})
                return result
            handle.seek(sidecar_offset)
            table = handle.read(EXPERTS * SIDECAR_RECORD_BYTES)
            if len(table) != EXPERTS * SIDECAR_RECORD_BYTES:
                result.update({"valid": False, "reason": "truncated_sidecar_table", "bytes": size})
                return result
            for expert in range(EXPERTS):
                record = SIDECAR_RECORD.unpack_from(table, expert * SIDECAR_RECORD_BYTES)
                if int(record[1]) != 0:
                    sidecars[expert] = record
    result.update({
        "valid": True,
        "bytes": size,
        "header": header,
        "records": records,
        "gateup_overlays": gateup_overlays,
        "sidecars": sidecars,
    })
    return result


def copy_record(source_path: Path, source_record: tuple[Any, ...], handle, hasher) -> tuple[int, ...]:
    projection, expert, k, bits, block, row_block, codebook_offset, index_offset, scale_offset, codebook_bytes, index_bytes, scale_bytes, flags = [int(x) for x in source_record]
    if k == 0:
        return (projection, expert, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    codebook = read_payload(source_path, codebook_offset, codebook_bytes)
    index = read_payload(source_path, index_offset, index_bytes)
    new_codebook_offset = append_aligned(handle, codebook)
    new_index_offset = append_aligned(handle, index)
    scale = b""
    new_scale_offset = 0
    if scale_bytes:
        scale = read_payload(source_path, scale_offset, scale_bytes)
        new_scale_offset = append_aligned(handle, scale)
    hasher.update(codebook)
    hasher.update(index)
    hasher.update(scale)
    return (
        projection,
        expert,
        k,
        bits,
        block,
        row_block,
        new_codebook_offset,
        new_index_offset,
        new_scale_offset,
        codebook_bytes,
        index_bytes,
        scale_bytes,
        flags,
    )


def append_sidecars(
    base_path: Path,
    source_path: Path,
    base_sidecars: dict[int, tuple[Any, ...]],
    source_sidecars: dict[int, tuple[Any, ...]],
    source_down_experts: set[int],
    handle,
) -> dict[str, Any]:
    merged: dict[int, tuple[Path, tuple[Any, ...]]] = {expert: (base_path, record) for expert, record in base_sidecars.items()}
    for expert in source_down_experts:
        if expert in source_sidecars:
            merged[expert] = (source_path, source_sidecars[expert])
        elif expert in base_sidecars:
            raise RuntimeError(f"selected down projection for E{expert} requires missing source sidecar")
    if not merged:
        return {}
    table_offset = handle.tell()
    padding = (-table_offset) % 16
    if padding:
        handle.write(b"\0" * padding)
        table_offset += padding
    table = bytearray(SIDECAR_RECORD_BYTES * EXPERTS)
    handle.write(table)
    for expert, (record_path, record) in sorted(merged.items()):
        source_expert, rank, in_dim, out_dim, u_offset, a_offset, u_bytes, a_bytes, flags, reserved, eff_rank, gain, aa_frac, reserved_f32 = record
        u_payload = read_payload(record_path, int(u_offset), int(u_bytes))
        a_payload = read_payload(record_path, int(a_offset), int(a_bytes))
        new_u_offset = append_aligned(handle, u_payload)
        new_a_offset = append_aligned(handle, a_payload)
        table[int(expert) * SIDECAR_RECORD_BYTES:(int(expert) + 1) * SIDECAR_RECORD_BYTES] = SIDECAR_RECORD.pack(
            int(source_expert),
            int(rank),
            int(in_dim),
            int(out_dim),
            new_u_offset,
            new_a_offset,
            int(u_bytes),
            int(a_bytes),
            int(flags),
            int(reserved),
            float(eff_rank),
            float(gain),
            float(aa_frac),
            float(reserved_f32),
        )
    here = handle.tell()
    handle.seek(table_offset)
    handle.write(table)
    handle.seek(here)
    return {
        "sidecar_format": "DS4D8F_DOWN_RANK1",
        "sidecar_table_offset": table_offset,
        "sidecar_record_bytes": SIDECAR_RECORD_BYTES,
        "sidecar_records": EXPERTS,
        "sidecar_count": len(merged),
        "sidecar_record_struct": "<IIIIQQIIIIffff",
        "sidecar_payload_dtype": "fp16_le",
        "sidecar_in_dim": 2048,
        "sidecar_out_dim": 4096,
    }


def parse_selections(selection_json: Path | None, inline: list[str]) -> dict[int, dict[int, set[str]]]:
    selections: dict[int, dict[int, set[str]]] = {}

    def add(layer: int, expert: int, projections: list[str]) -> None:
        bad = [item for item in projections if item not in PROJECTION_ID]
        if bad:
            raise ValueError(f"unknown projections: {bad}")
        if not (0 <= expert < EXPERTS):
            raise ValueError(f"expert out of range: {expert}")
        selections.setdefault(layer, {}).setdefault(expert, set()).update(projections)

    if selection_json:
        payload = json.loads(selection_json.read_text(encoding="utf-8"))
        for item in payload.get("selected", []):
            projections = item.get("projections") or item.get("projection") or PROJECTIONS
            if isinstance(projections, str):
                projections = [projections]
            add(int(item["layer"]), int(item["expert"]), [str(projection) for projection in projections])
    for spec in inline:
        layer_s, expert_s, projection_s = spec.split(":", 2)
        add(int(layer_s), int(expert_s), projection_s.split(","))
    return selections


def splice_layer(base_dir: Path, source_dir: Path, out_dir: Path, layer: int, layer_selection: dict[int, set[str]], execute: bool) -> dict[str, Any]:
    base_info = read_d8f(d8f_path(base_dir, layer))
    source_info = read_d8f(d8f_path(source_dir, layer))
    out_path = out_dir / f"ds4_L{layer:02d}_gate_up_down_VQD8_noE8_rank1.d8f"
    report: dict[str, Any] = {
        "layer": layer,
        "selected": {str(expert): sorted(projections) for expert, projections in sorted(layer_selection.items())},
        "base": base_info["path"],
        "source": source_info["path"],
        "output": str(out_path),
        "ready": False,
    }
    if not base_info.get("valid") or not source_info.get("valid"):
        report["reason"] = "base_or_source_missing_or_invalid"
        return report
    missing = []
    for expert, projections in sorted(layer_selection.items()):
        for projection in projections:
            record = source_info["records"][PROJECTION_ID[projection] * EXPERTS + expert]
            if int(record[2]) == 0:
                missing.append({"expert": expert, "projection": projection})
    if missing:
        report["reason"] = "source_missing_selected_records"
        report["missing_records"] = missing
        return report
    source_down_experts = {expert for expert, projections in layer_selection.items() if "down" in projections}
    report["source_down_sidecar_overlap"] = sorted(source_down_experts.intersection(set(base_info["sidecars"])))
    report["ready"] = True
    if not execute:
        return report
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise FileExistsError(out_path)
    base_path = Path(base_info["path"])
    source_path = Path(source_info["path"])
    header = dict(base_info["header"])
    header.update({
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sparse_splice": "projection-specific selected-expert D8F splice",
        "sparse_splice_base": str(base_path),
        "sparse_splice_source": str(source_path),
        "sparse_splice_selection": report["selected"],
    })
    for key in list(header):
        if key.startswith("gateup_overlay_"):
            header.pop(key, None)
    for key in [
        "sidecar_table_offset",
        "sidecar_record_bytes",
        "sidecar_records",
        "sidecar_count",
        "sidecar_record_struct",
        "sidecar_payload_dtype",
        "sidecar_in_dim",
        "sidecar_out_dim",
        "sidecar_format",
    ]:
        header.pop(key, None)
    payload_hashers = {projection: hashlib.sha256() for projection in PROJECTIONS}
    new_records = []
    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with temp_path.open("wb") as handle:
        rewrite_header(handle, header)
        handle.write(b"\0" * (len(PROJECTIONS) * EXPERTS * FUSED_RECORD_BYTES))
        for projection_id, projection in enumerate(PROJECTIONS):
            for expert in range(EXPERTS):
                use_source = expert in layer_selection and projection in layer_selection[expert]
                record_info = source_info if use_source else base_info
                record_path = source_path if use_source else base_path
                record = record_info["records"][projection_id * EXPERTS + expert]
                new_records.append(copy_record(record_path, record, handle, payload_hashers[projection]))
        sidecar_header = append_sidecars(base_path, source_path, base_info["sidecars"], source_info["sidecars"], source_down_experts, handle)
        header.update(sidecar_header)
        if sidecar_header and header.get("target") in {"d8f_direct_no_sidecar", "rank1_down_sidecar_none_selected"}:
            header["target"] = "rank1_down_sidecar_direct"
        header["payload_sha256"] = {projection: payload_hashers[projection].hexdigest() for projection in PROJECTIONS}
        handle.seek(HEADER_BYTES)
        for record in new_records:
            handle.write(FUSED_RECORD.pack(*record))
        rewrite_header(handle, header)
    temp_path.replace(out_path)
    report["bytes"] = out_path.stat().st_size
    report["materialized"] = True
    return report


def hardlink_pack(base_pack: Path, out_pack: Path, spliced_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if out_pack.exists():
        raise FileExistsError(out_pack)
    out_pack.mkdir(parents=True)
    replacement_by_name = {Path(item["output"]).name: Path(item["output"]) for item in spliced_reports if item.get("materialized")}
    entries = []
    emit(phase="pack_link_start", base_pack=str(base_pack), out_pack=str(out_pack), replacements=sorted(replacement_by_name))
    for source in sorted(base_pack.iterdir()):
        dest = out_pack / source.name
        replacement = replacement_by_name.get(source.name)
        link_source = replacement or source
        if source.is_symlink():
            raise RuntimeError(f"refusing to copy symlink into self-contained pack: {source}")
        if source.is_dir():
            shutil.copytree(source, dest, copy_function=os.link)
            mode = "hardlinked_tree"
        else:
            os.link(link_source, dest)
            mode = "hardlink_replacement" if replacement else "hardlink_base"
        entries.append({"name": dest.name, "source": str(link_source), "mode": mode, "bytes": dest.stat().st_size})
        if replacement:
            emit(phase="pack_link_replacement", name=dest.name, bytes=dest.stat().st_size)
    emit(phase="pack_link_done", entries=len(entries), out_pack=str(out_pack))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-json", type=Path)
    parser.add_argument("--select", action="append", default=[], help="layer:expert:projection[,projection...]")
    parser.add_argument("--base-pack", type=Path, required=True)
    parser.add_argument("--source-pack", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pack-out-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    selections = parse_selections(args.selection_json, args.select)
    if not selections:
        raise SystemExit("empty selection")
    layer_reports = []
    for layer, layer_selection in sorted(selections.items()):
        emit(phase="splice_layer_start", layer=layer, selection={str(expert): sorted(projections) for expert, projections in sorted(layer_selection.items())}, execute=args.execute)
        layer_report = splice_layer(args.base_pack, args.source_pack, args.out_dir, layer, layer_selection, args.execute)
        layer_reports.append(layer_report)
        emit(phase="splice_layer_done", layer=layer, ready=bool(layer_report.get("ready")), materialized=bool(layer_report.get("materialized")), bytes=layer_report.get("bytes"), reason=layer_report.get("reason"))
    pack_entries = []
    if args.execute and args.pack_out_dir:
        if not all(item.get("ready") for item in layer_reports):
            raise SystemExit("not all layer reports are ready; refusing pack build")
        pack_entries = hardlink_pack(args.base_pack, args.pack_out_dir, layer_reports)
    report = {
        "phase": "ds4_sparse_projection_splice",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "execute": args.execute,
        "base_pack": str(args.base_pack),
        "source_pack": str(args.source_pack),
        "out_dir": str(args.out_dir),
        "pack_out_dir": None if args.pack_out_dir is None else str(args.pack_out_dir),
        "selection": {str(layer): {str(expert): sorted(projections) for expert, projections in sorted(layer_selection.items())} for layer, layer_selection in sorted(selections.items())},
        "layer_reports": layer_reports,
        "all_ready": all(item.get("ready") for item in layer_reports),
        "pack_entries": pack_entries,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "all_ready": report["all_ready"],
        "execute": args.execute,
        "layers": sorted(selections),
        "pack_out_dir": report["pack_out_dir"],
        "report": None if args.report is None else str(args.report),
    }, sort_keys=True))
    return 0 if report["all_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
