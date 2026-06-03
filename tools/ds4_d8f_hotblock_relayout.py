#!/usr/bin/env python3
"""Losslessly relayout D8F payloads so hot experts are physically contiguous.

The D8F runtime already reads offsets from the record table, so payload order is
an encoding/layout choice rather than a quality choice.  The default D8F layout
is projection-major: all gate experts, then all up experts, then all down
experts.  A decode token touches gate/up/down for the selected experts, so a hot
expert currently spans hundreds of MiB inside a layer file.

This tool rewrites selected layer files with the same records and payload bytes,
but copies payloads in expert-major order with the nominated hot experts first:

    hot E0 gate, hot E0 up, hot E0 down, hot E1 gate, ...
    remaining experts gate/up/down ...

The result is self-contained and runtime-readable by the existing D8F reader.
It changes locality, not quantization math or fidelity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import struct
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from ds4_sparse_projection_splice import (
    EXPERTS,
    FUSED_RECORD,
    FUSED_RECORD_BYTES,
    GATEUP_OVERLAY_SENTINEL,
    GATEUP_ROW_BLOCKS,
    HEADER_BYTES,
    PROJECTIONS,
    SIDECAR_RECORD,
    SIDECAR_RECORD_BYTES,
    append_aligned,
    copy_record,
    d8f_path,
    hardlink_pack,
    read_payload,
    read_d8f,
    rewrite_header,
)

GATEUP_OVERLAY_PREFIX = "gateup_overlay_"
GATEUP_ROWS_PER_BLOCK = 128
GATEUP_IN_DIM = 4096
GATEUP_INDEX_GUARD_BYTES = 4
DOWN_IN_DIM = 2048
DOWN_GROUPS = DOWN_IN_DIM // 8
DOWN_NATIVE_CODE_RECORD_BYTES = 32
DOWN_NATIVE_CODE_RECORD = struct.Struct("<IIIIQII")
DOWN_NATIVE_CODE_DTYPE_U16 = 1


def parse_int_set(text: str, *, limit: int, name: str) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            values = range(int(left), int(right) + 1)
        else:
            values = (int(part),)
        for value in values:
            if value < 0 or value >= limit:
                raise SystemExit(f"{name} out of range: {value}")
            if value not in seen:
                seen.add(value)
                out.append(value)
    if not out:
        raise SystemExit(f"empty {name} set")
    return out


def parse_layer_hot_json(path: Path) -> dict[int, list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_layers = payload.get("layers", payload)
    if not isinstance(raw_layers, dict):
        raise SystemExit("hot JSON must be an object or contain a 'layers' object")
    result: dict[int, list[int]] = {}
    for layer_s, experts in raw_layers.items():
        layer = int(layer_s)
        if isinstance(experts, str):
            result[layer] = parse_int_set(experts, limit=EXPERTS, name=f"L{layer} expert")
        else:
            seen: set[int] = set()
            result[layer] = []
            for expert_raw in experts:
                expert = int(expert_raw)
                if expert < 0 or expert >= EXPERTS:
                    raise SystemExit(f"L{layer} expert out of range: {expert}")
                if expert not in seen:
                    seen.add(expert)
                    result[layer].append(expert)
        if not result[layer]:
            raise SystemExit(f"L{layer} has no hot experts")
    return result


def parse_route_stages(text: str) -> set[str] | None:
    if not text or text == "all":
        return None
    stages = {part.strip() for part in text.split(",") if part.strip()}
    if not stages:
        return None
    return stages


def route_trace_hot_experts(path: Path, *, top: int, stages: set[str] | None, score_mode: str) -> dict[int, list[int]]:
    if top <= 0 or top > EXPERTS:
        raise SystemExit(f"--route-top out of range: {top}")
    scores: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[int, int] = defaultdict(int)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"layer", "stage", *(f"e{i}" for i in range(6)), *(f"w{i}" for i in range(6))}
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise SystemExit(f"{path}: route trace missing columns: {missing}")
        for row in reader:
            if stages is not None and row["stage"] not in stages:
                continue
            layer = int(row["layer"])
            counts[layer] += 1
            for slot in range(6):
                expert = int(row[f"e{slot}"])
                if expert < 0 or expert >= EXPERTS:
                    raise SystemExit(f"{path}: expert out of range in L{layer}: {expert}")
                if score_mode == "count":
                    value = 1.0
                elif score_mode == "mass":
                    value = float(row[f"w{slot}"])
                elif score_mode == "abs_mass":
                    value = abs(float(row[f"w{slot}"]))
                else:
                    raise SystemExit(f"unknown route score mode: {score_mode}")
                scores[layer][expert] += value
    if not scores:
        raise SystemExit(f"{path}: no route rows matched stages={sorted(stages) if stages else 'all'}")
    result: dict[int, list[int]] = {}
    for layer, layer_scores in scores.items():
        ranked = sorted(layer_scores.items(), key=lambda item: (-item[1], item[0]))
        result[layer] = [expert for expert, _ in ranked[:top]]
    print(json.dumps({
        "phase": "route_trace_hot_selection",
        "trace_csv": str(path),
        "layers": sorted(result),
        "route_top": top,
        "route_score": score_mode,
        "route_stages": sorted(stages) if stages else "all",
        "rows_by_layer": {str(layer): counts[layer] for layer in sorted(counts)},
        "hot_experts": {str(layer): result[layer] for layer in sorted(result)},
    }, sort_keys=True), flush=True)
    return result


def payload_span(record: tuple[Any, ...]) -> tuple[int, int, int]:
    spans: list[tuple[int, int]] = []
    codebook_offset, index_offset, scale_offset = int(record[6]), int(record[7]), int(record[8])
    codebook_bytes, index_bytes, scale_bytes = int(record[9]), int(record[10]), int(record[11])
    if codebook_bytes:
        spans.append((codebook_offset, codebook_offset + codebook_bytes))
    if index_bytes:
        spans.append((index_offset, index_offset + index_bytes))
    if scale_bytes:
        spans.append((scale_offset, scale_offset + scale_bytes))
    if not spans:
        return (0, 0, 0)
    start = min(item[0] for item in spans)
    end = max(item[1] for item in spans)
    return (start, end, end - start)


def expert_span(records: list[tuple[Any, ...]], expert: int) -> tuple[int, int, int]:
    spans: list[tuple[int, int]] = []
    for projection_id in range(len(PROJECTIONS)):
        start, end, _ = payload_span(records[projection_id * EXPERTS + expert])
        if end > start:
            spans.append((start, end))
    if not spans:
        return (0, 0, 0)
    start = min(item[0] for item in spans)
    end = max(item[1] for item in spans)
    return (start, end, end - start)


def window_span(records: list[tuple[Any, ...]], experts: list[int]) -> tuple[int, int, int]:
    spans: list[tuple[int, int]] = []
    for expert in experts:
        start, end, _ = expert_span(records, expert)
        if end > start:
            spans.append((start, end))
    if not spans:
        return (0, 0, 0)
    start = min(item[0] for item in spans)
    end = max(item[1] for item in spans)
    return (start, end, end - start)


def span_summary(spans: list[tuple[int, int]]) -> dict[str, Any]:
    valid = [(start, end) for start, end in spans if end > start]
    if not valid:
        return {
            "window_start": 0,
            "window_end": 0,
            "window_bytes": 0,
            "payload_bytes": 0,
            "density": 0.0,
            "span_count": 0,
        }
    start = min(item[0] for item in valid)
    end = max(item[1] for item in valid)
    payload_bytes = sum(end_i - start_i for start_i, end_i in valid)
    window_bytes = end - start
    return {
        "window_start": start,
        "window_end": end,
        "window_bytes": window_bytes,
        "payload_bytes": payload_bytes,
        "density": (payload_bytes / window_bytes) if window_bytes else 0.0,
        "span_count": len(valid),
    }


def runtime_hot_spans(info: dict[str, Any],
                      records: list[tuple[Any, ...]],
                      experts: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    overlays = info.get("gateup_overlays", {})
    for expert in experts:
        for projection_id in (0, 1):
            needs_base = False
            if overlays:
                for row_block in range(GATEUP_ROW_BLOCKS):
                    overlay = overlays.get((projection_id, expert, row_block))
                    if overlay:
                        start, end, _ = payload_span(overlay)
                        if end > start:
                            spans.append((start, end))
                    else:
                        needs_base = True
            else:
                needs_base = True
            if needs_base:
                start, end, _ = payload_span(records[projection_id * EXPERTS + expert])
                if end > start:
                    spans.append((start, end))
        for projection_id in (2,):
            start, end, _ = payload_span(records[projection_id * EXPERTS + expert])
            if end > start:
                spans.append((start, end))
        header = info.get("header", {})
        sidecar_table_offset = int(header.get("sidecar_table_offset", 0) or 0)
        sidecar_record_bytes = int(header.get("sidecar_record_bytes", SIDECAR_RECORD_BYTES) or SIDECAR_RECORD_BYTES)
        sidecar_records = int(header.get("sidecar_records", 0) or 0)
        if sidecar_table_offset and expert < sidecar_records:
            start = sidecar_table_offset + expert * sidecar_record_bytes
            spans.append((start, start + sidecar_record_bytes))
        sidecar = info.get("sidecars", {}).get(expert)
        if sidecar:
            _, rank, _, _, u_offset, a_offset, u_bytes, a_bytes, *_ = sidecar
            if int(rank):
                spans.append((int(u_offset), int(u_offset) + int(u_bytes)))
                spans.append((int(a_offset), int(a_offset) + int(a_bytes)))
    return spans


def runtime_hot_summary(info: dict[str, Any],
                        records: list[tuple[Any, ...]],
                        experts: list[int]) -> dict[str, Any]:
    return span_summary(runtime_hot_spans(info, records, experts))


def gateup_rowblock_hot_summaries(info: dict[str, Any],
                                  records: list[tuple[Any, ...]],
                                  experts: list[int]) -> list[dict[str, Any]]:
    overlays = info.get("gateup_overlays", {})
    summaries: list[dict[str, Any]] = []
    for row_block in range(GATEUP_ROW_BLOCKS):
        spans: list[tuple[int, int]] = []
        for expert in experts:
            for projection_id in (0, 1):
                overlay = overlays.get((projection_id, expert, row_block))
                if overlay:
                    start, end, _ = payload_span(overlay)
                    if end > start:
                        spans.append((start, end))
                    continue
                record = records[projection_id * EXPERTS + expert]
                projection, _, _, bits, block, _, codebook_offset, index_offset, _, codebook_bytes, index_bytes, _, _ = [int(x) for x in record]
                if projection != projection_id or not bits or not block or not index_bytes:
                    continue
                if codebook_bytes:
                    spans.append((codebook_offset, codebook_offset + codebook_bytes))
                row_index_bytes = ((GATEUP_IN_DIM // block) * bits + 7) // 8
                segment_bytes = GATEUP_ROWS_PER_BLOCK * row_index_bytes
                guarded_segment_bytes = segment_bytes + GATEUP_INDEX_GUARD_BYTES
                segment_offset = index_offset + row_block * segment_bytes
                if segment_offset + guarded_segment_bytes <= index_offset + index_bytes:
                    spans.append((segment_offset, segment_offset + guarded_segment_bytes))
        item = span_summary(spans)
        item["row_block"] = row_block
        summaries.append(item)
    return summaries


def compact_window_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    windows = [int(item["window_bytes"]) for item in items]
    payloads = [int(item["payload_bytes"]) for item in items]
    densities = [float(item["density"]) for item in items if int(item["window_bytes"]) > 0]
    if not windows:
        return {"min": 0, "mean": 0.0, "max": 0, "payload_mean": 0.0, "density_mean": 0.0}
    return {
        "min": min(windows),
        "mean": sum(windows) / len(windows),
        "max": max(windows),
        "payload_mean": sum(payloads) / len(payloads),
        "density_mean": (sum(densities) / len(densities)) if densities else 0.0,
    }


def copy_gateup_overlay_record(source_path: Path,
                               source_record: tuple[Any, ...],
                               row_block: int,
                               handle,
                               hasher) -> tuple[int, ...]:
    projection, expert, k, bits, block, _, codebook_offset, index_offset, _, codebook_bytes, index_bytes, _, _ = [int(x) for x in source_record]
    if projection not in (0, 1) or k == 0 or block != 8 or row_block < 0 or row_block >= GATEUP_ROW_BLOCKS:
        raise RuntimeError(f"invalid gate/up overlay source p={projection} e={expert} k={k} block={block} row_block={row_block}")
    row_index_bytes = ((GATEUP_IN_DIM // block) * bits + 7) // 8
    segment_bytes = GATEUP_ROWS_PER_BLOCK * row_index_bytes
    guarded_segment_bytes = segment_bytes + GATEUP_INDEX_GUARD_BYTES
    segment_offset = index_offset + row_block * segment_bytes
    if segment_offset + guarded_segment_bytes > index_offset + index_bytes:
        raise RuntimeError(f"gate/up overlay source too short p={projection} e={expert} row_block={row_block}")
    codebook = read_payload(source_path, codebook_offset, codebook_bytes)
    index = read_payload(source_path, segment_offset, guarded_segment_bytes)
    new_codebook_offset = append_aligned(handle, codebook)
    new_index_offset = append_aligned(handle, index)
    hasher.update(codebook)
    hasher.update(index)
    return (
        projection,
        expert,
        k,
        bits,
        block,
        row_block,
        new_codebook_offset,
        new_index_offset,
        0,
        codebook_bytes,
        guarded_segment_bytes,
        0,
        0,
    )


def begin_gateup_overlay(handle, hot_experts: list[int]) -> dict[str, Any]:
    slot_offset = handle.tell()
    padding = (-slot_offset) % 16
    if padding:
        handle.write(b"\0" * padding)
        slot_offset += padding
    slot_entries = 2 * EXPERTS * GATEUP_ROW_BLOCKS
    slot_table = bytearray()
    for _ in range(slot_entries):
        slot_table += GATEUP_OVERLAY_SENTINEL.to_bytes(4, "little")
    handle.write(slot_table)
    record_offset = handle.tell()
    padding = (-record_offset) % 16
    if padding:
        handle.write(b"\0" * padding)
        record_offset += padding
    record_count = 2 * len(hot_experts) * GATEUP_ROW_BLOCKS
    record_table = bytearray(record_count * FUSED_RECORD_BYTES)
    handle.write(record_table)
    return {
        "slot_offset": slot_offset,
        "record_offset": record_offset,
        "slot_entries": slot_entries,
        "slot_table": slot_table,
        "record_table": record_table,
        "records": 0,
    }


def add_gateup_overlay_record(state: dict[str, Any],
                              source_path: Path,
                              source_record: tuple[Any, ...],
                              row_block: int,
                              handle,
                              hasher) -> None:
    record = copy_gateup_overlay_record(source_path, source_record, row_block, handle, hasher)
    projection, expert = int(record[0]), int(record[1])
    slot = int(state["records"])
    slot_index = (projection * EXPERTS + expert) * GATEUP_ROW_BLOCKS + row_block
    state["slot_table"][slot_index * 4:(slot_index + 1) * 4] = slot.to_bytes(4, "little")
    state["record_table"][slot * FUSED_RECORD_BYTES:(slot + 1) * FUSED_RECORD_BYTES] = FUSED_RECORD.pack(*record)
    state["records"] = slot + 1


def finish_gateup_overlay(state: dict[str, Any] | None, handle) -> dict[str, Any]:
    if not state:
        return {}
    here = handle.tell()
    handle.seek(int(state["slot_offset"]))
    handle.write(state["slot_table"])
    handle.seek(int(state["record_offset"]))
    handle.write(state["record_table"])
    handle.seek(here)
    return {
        "gateup_overlay_slot_table_offset": int(state["slot_offset"]),
        "gateup_overlay_record_offset": int(state["record_offset"]),
        "gateup_overlay_slot_entries": int(state["slot_entries"]),
        "gateup_overlay_record_bytes": FUSED_RECORD_BYTES,
        "gateup_overlay_records": int(state["records"]),
        "gateup_overlay_sentinel": GATEUP_OVERLAY_SENTINEL,
        "gateup_overlay_layout": "hot_expert_rowblock_major_gate_up",
        "gateup_overlay_rows_per_block": GATEUP_ROWS_PER_BLOCK,
        "gateup_overlay_index_guard_bytes": GATEUP_INDEX_GUARD_BYTES,
    }


def d8f_code_at_blob(index: bytes, bits: int, block_index: int) -> int:
    bit_offset = block_index * bits
    byte_offset = bit_offset >> 3
    shift = bit_offset & 7
    window = int.from_bytes(index[byte_offset:byte_offset + 4], "little", signed=False)
    return (window >> shift) & ((1 << bits) - 1)


def encode_down_native_codes(source_path: Path, source_record: tuple[Any, ...]) -> tuple[int, bytes]:
    projection, expert, k, bits, block, _, _, index_offset, _, _, index_bytes, _, _ = [int(x) for x in source_record]
    if projection != 2 or k == 0 or block != 8 or bits <= 0:
        raise RuntimeError(f"invalid down native-code source p={projection} e={expert} k={k} block={block} bits={bits}")
    blocks = (index_bytes * 8) // bits
    if blocks % DOWN_GROUPS:
        raise RuntimeError(f"down native-code source has non-integral rows e={expert} blocks={blocks}")
    rows = blocks // DOWN_GROUPS
    index = read_payload(source_path, index_offset, index_bytes)
    out = bytearray(rows * DOWN_GROUPS * 2)
    for row in range(rows):
        for group in range(DOWN_GROUPS):
            code = d8f_code_at_blob(index, bits, row * DOWN_GROUPS + group)
            if code >= 65536:
                raise RuntimeError(f"down native-code overflow e={expert} code={code}")
            struct.pack_into("<H", out, (row * DOWN_GROUPS + group) * 2, code)
    return rows, bytes(out)


def begin_down_native_code_sidecars(source_path: Path,
                                    records: list[tuple[Any, ...]],
                                    hot_experts: list[int],
                                    handle) -> dict[str, Any] | None:
    if not hot_experts:
        return None
    table_offset = handle.tell()
    padding = (-table_offset) % 16
    if padding:
        handle.write(b"\0" * padding)
        table_offset += padding
    table = bytearray(DOWN_NATIVE_CODE_RECORD_BYTES * EXPERTS)
    handle.write(table)
    payload_hash = hashlib.sha256()
    count = 0
    for expert in hot_experts:
        record = records[2 * EXPERTS + expert]
        rows, payload = encode_down_native_codes(source_path, record)
        payload_offset = append_aligned(handle, payload)
        payload_hash.update(payload)
        table[expert * DOWN_NATIVE_CODE_RECORD_BYTES:(expert + 1) * DOWN_NATIVE_CODE_RECORD_BYTES] = DOWN_NATIVE_CODE_RECORD.pack(
            expert,
            rows,
            DOWN_GROUPS,
            DOWN_NATIVE_CODE_DTYPE_U16,
            payload_offset,
            len(payload),
            0,
        )
        count += 1
    return {
        "table_offset": table_offset,
        "table": table,
        "count": count,
        "payload_hash": payload_hash,
    }


def finish_down_native_code_sidecars(state: dict[str, Any] | None, handle) -> dict[str, Any]:
    if not state:
        return {}
    here = handle.tell()
    handle.seek(int(state["table_offset"]))
    handle.write(state["table"])
    handle.seek(here)
    return {
        "down_native_code_sidecar_format": "DS4D8F_DOWN_NATIVE_U16_CODES_V1",
        "down_native_code_sidecar_table_offset": int(state["table_offset"]),
        "down_native_code_sidecar_record_bytes": DOWN_NATIVE_CODE_RECORD_BYTES,
        "down_native_code_sidecar_records": EXPERTS,
        "down_native_code_sidecar_count": int(state["count"]),
        "down_native_code_sidecar_record_struct": "<IIIIQII",
        "down_native_code_sidecar_dtype": "uint16_le",
        "down_native_code_sidecar_layout": "hot_selected_expert_row_major_group_u16",
        "down_native_code_sidecar_groups": DOWN_GROUPS,
        "down_native_code_sidecar_payload_sha256": state["payload_hash"].hexdigest(),
    }


def begin_hotblock_sidecars(source_path: Path,
                            sidecars: dict[int, tuple[Any, ...]],
                            hot_experts: list[int],
                            handle) -> dict[str, Any] | None:
    if not sidecars:
        return None
    table_offset = handle.tell()
    padding = (-table_offset) % 16
    if padding:
        handle.write(b"\0" * padding)
        table_offset += padding
    table = bytearray(SIDECAR_RECORD_BYTES * EXPERTS)
    handle.write(table)
    written: set[int] = set()

    def write_one(expert: int, record: tuple[Any, ...]) -> None:
        source_expert, rank, in_dim, out_dim, u_offset, a_offset, u_bytes, a_bytes, flags, reserved, eff_rank, gain, aa_frac, reserved_f32 = record
        u_payload = read_payload(source_path, int(u_offset), int(u_bytes))
        a_payload = read_payload(source_path, int(a_offset), int(a_bytes))
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
        written.add(int(expert))

    for expert in hot_experts:
        record = sidecars.get(expert)
        if record:
            write_one(expert, record)
    return {
        "source_path": source_path,
        "sidecars": sidecars,
        "table_offset": table_offset,
        "table": table,
        "written": written,
        "write_one": write_one,
    }


def finish_hotblock_sidecars(state: dict[str, Any] | None, handle) -> dict[str, Any]:
    if not state:
        return {}
    sidecars: dict[int, tuple[Any, ...]] = state["sidecars"]
    written: set[int] = state["written"]
    write_one = state["write_one"]
    for expert, record in sorted(sidecars.items()):
        if expert not in written:
            write_one(expert, record)
    here = handle.tell()
    handle.seek(int(state["table_offset"]))
    handle.write(state["table"])
    handle.seek(here)
    return {
        "sidecar_format": "DS4D8F_DOWN_RANK1",
        "sidecar_table_offset": int(state["table_offset"]),
        "sidecar_record_bytes": SIDECAR_RECORD_BYTES,
        "sidecar_records": EXPERTS,
        "sidecar_count": len(sidecars),
        "sidecar_record_struct": "<IIIIQQIIIIffff",
        "sidecar_payload_dtype": "fp16_le",
        "sidecar_in_dim": 2048,
        "sidecar_out_dim": 4096,
        "d8f_hotblock_sidecar_payload_order": "hot_selected_sidecars_then_main_payload_then_remaining_sidecars",
    }


def ordered_experts(hot_experts: list[int]) -> list[int]:
    hot = []
    seen: set[int] = set()
    for expert in hot_experts:
        if expert not in seen:
            hot.append(expert)
            seen.add(expert)
    return hot + [expert for expert in range(EXPERTS) if expert not in seen]


def strip_rebuilt_header_keys(header: dict[str, Any]) -> None:
    for key in list(header):
        if key.startswith(GATEUP_OVERLAY_PREFIX):
            raise SystemExit("existing gate/up row-block overlays are not preserved by hotblock relayout")
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
        "down_native_code_sidecar_format",
        "down_native_code_sidecar_table_offset",
        "down_native_code_sidecar_record_bytes",
        "down_native_code_sidecar_records",
        "down_native_code_sidecar_count",
        "down_native_code_sidecar_record_struct",
        "down_native_code_sidecar_dtype",
        "down_native_code_sidecar_layout",
        "down_native_code_sidecar_groups",
        "down_native_code_sidecar_payload_sha256",
    ]:
        header.pop(key, None)


def relayout_layer(source_pack: Path,
                   out_dir: Path,
                   layer: int,
                   hot_experts: list[int],
                   execute: bool,
                   status_every: int,
                   gateup_rowblock_overlays: bool,
                   down_native_code_sidecars: bool) -> dict[str, Any]:
    source_path = d8f_path(source_pack, layer)
    info = read_d8f(source_path)
    report: dict[str, Any] = {
        "layer": layer,
        "source": str(source_path),
        "output": str(out_dir / source_path.name),
        "hot_experts": hot_experts,
        "ready": False,
    }
    if not info.get("valid"):
        report["reason"] = "source_missing_or_invalid"
        return report

    old_records = list(info["records"])
    old_window = window_span(old_records, hot_experts)
    old_expert_spans = {str(expert): expert_span(old_records, expert)[2] for expert in hot_experts}
    old_runtime = runtime_hot_summary(info, old_records, hot_experts)
    old_gateup_rowblocks = gateup_rowblock_hot_summaries(info, old_records, hot_experts)
    report.update({
        "ready": True,
        "source_bytes": info["bytes"],
        "source_hot_window_bytes": old_window[2],
        "source_hot_expert_span_bytes": old_expert_spans,
        "source_runtime_hot_window_bytes": old_runtime["window_bytes"],
        "source_runtime_hot_payload_bytes": old_runtime["payload_bytes"],
        "source_runtime_hot_density": old_runtime["density"],
        "source_runtime_hot_span_count": old_runtime["span_count"],
        "source_gateup_rowblock_hot_window_bytes": compact_window_stats(old_gateup_rowblocks),
    })
    if not execute:
        return report

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(report["output"])
    if out_path.exists():
        raise FileExistsError(out_path)

    header = dict(info["header"])
    strip_rebuilt_header_keys(header)
    header.update({
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "d8f_hotblock_layout": "expert-major hot-block relayout",
        "d8f_hotblock_source": str(source_path),
        "d8f_hotblock_hot_experts": hot_experts,
        "d8f_hotblock_payload_order": "hot_expert_major_gate_up_down_then_remaining_expert_major",
        "d8f_hotblock_gateup_rowblock_overlays": bool(gateup_rowblock_overlays),
        "d8f_hotblock_down_native_code_sidecars": bool(down_native_code_sidecars),
    })
    payload_hashers = {projection: hashlib.sha256() for projection in PROJECTIONS}
    overlay_hashers = {projection: hashlib.sha256() for projection in PROJECTIONS[:2]}
    new_records: list[tuple[Any, ...] | None] = [None] * (len(PROJECTIONS) * EXPERTS)
    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    with temp_path.open("wb") as handle:
        rewrite_header(handle, header)
        handle.write(b"\0" * (len(PROJECTIONS) * EXPERTS * FUSED_RECORD_BYTES))
        expert_order = ordered_experts(hot_experts)
        sidecar_state = begin_hotblock_sidecars(source_path, info["sidecars"], hot_experts, handle)
        down_native_state = begin_down_native_code_sidecars(source_path, old_records, hot_experts, handle) if down_native_code_sidecars else None
        overlay_state = begin_gateup_overlay(handle, hot_experts) if gateup_rowblock_overlays else None
        if overlay_state:
            for row_block in range(GATEUP_ROW_BLOCKS):
                for expert in hot_experts:
                    for projection_id, projection in enumerate(PROJECTIONS[:2]):
                        old_record = old_records[projection_id * EXPERTS + expert]
                        add_gateup_overlay_record(
                            overlay_state, source_path, old_record, row_block, handle,
                            overlay_hashers[projection],
                        )
                if status_every:
                    print(json.dumps({
                        "phase": "hotblock_gateup_overlay_progress",
                        "layer": layer,
                        "row_block_done": row_block + 1,
                        "row_blocks_total": GATEUP_ROW_BLOCKS,
                        "out_bytes": handle.tell(),
                    }, sort_keys=True), flush=True)
            for expert in hot_experts:
                old_record = old_records[2 * EXPERTS + expert]
                new_records[2 * EXPERTS + expert] = copy_record(
                    source_path, old_record, handle, payload_hashers[PROJECTIONS[2]]
                )
            hot_set = set(hot_experts)
            cold_experts = hot_experts + [expert for expert in range(EXPERTS) if expert not in hot_set]
            for expert_index, expert in enumerate(cold_experts, 1):
                projection_ids = (0, 1) if expert in hot_set else (0, 1, 2)
                for projection_id in projection_ids:
                    old_record = old_records[projection_id * EXPERTS + expert]
                    new_records[projection_id * EXPERTS + expert] = copy_record(
                        source_path, old_record, handle, payload_hashers[PROJECTIONS[projection_id]]
                    )
                if status_every and (expert_index % status_every == 0 or expert_index == len(cold_experts)):
                    print(json.dumps({
                        "phase": "hotblock_copy_progress",
                        "layer": layer,
                        "experts_done": expert_index,
                        "experts_total": len(cold_experts),
                        "out_bytes": handle.tell(),
                    }, sort_keys=True), flush=True)
        else:
            for expert_index, expert in enumerate(expert_order, 1):
                for projection_id, projection in enumerate(PROJECTIONS):
                    old_record = old_records[projection_id * EXPERTS + expert]
                    new_records[projection_id * EXPERTS + expert] = copy_record(
                        source_path, old_record, handle, payload_hashers[projection]
                    )
                if status_every and (expert_index % status_every == 0 or expert_index == len(expert_order)):
                    print(json.dumps({
                        "phase": "hotblock_copy_progress",
                        "layer": layer,
                        "experts_done": expert_index,
                        "experts_total": len(expert_order),
                        "out_bytes": handle.tell(),
                    }, sort_keys=True), flush=True)
        overlay_header = finish_gateup_overlay(overlay_state, handle)
        sidecar_header = finish_hotblock_sidecars(sidecar_state, handle)
        down_native_header = finish_down_native_code_sidecars(down_native_state, handle)
        header.update(overlay_header)
        header.update(sidecar_header)
        header.update(down_native_header)
        if overlay_state:
            header["gateup_overlay_payload_sha256"] = {
                projection: overlay_hashers[projection].hexdigest()
                for projection in PROJECTIONS[:2]
            }
        header["payload_sha256"] = {projection: payload_hashers[projection].hexdigest() for projection in PROJECTIONS}
        handle.seek(HEADER_BYTES)
        for record in new_records:
            if record is None:
                raise RuntimeError("internal missing record after relayout")
            handle.write(FUSED_RECORD.pack(*record))
        rewrite_header(handle, header)
    temp_path.replace(out_path)

    new_info = read_d8f(out_path)
    if not new_info.get("valid"):
        raise RuntimeError(f"rewritten D8F did not validate: {out_path}")
    new_records_t = list(new_info["records"])
    new_window = window_span(new_records_t, hot_experts)
    new_expert_spans = {str(expert): expert_span(new_records_t, expert)[2] for expert in hot_experts}
    new_runtime = runtime_hot_summary(new_info, new_records_t, hot_experts)
    new_gateup_rowblocks = gateup_rowblock_hot_summaries(new_info, new_records_t, hot_experts)
    old_gateup_stats = compact_window_stats(old_gateup_rowblocks)
    new_gateup_stats = compact_window_stats(new_gateup_rowblocks)
    report.update({
        "materialized": True,
        "output_bytes": out_path.stat().st_size,
        "output_hot_window_bytes": new_window[2],
        "output_hot_expert_span_bytes": new_expert_spans,
        "hot_window_reduction": (old_window[2] / new_window[2]) if new_window[2] else None,
        "output_runtime_hot_window_bytes": new_runtime["window_bytes"],
        "output_runtime_hot_payload_bytes": new_runtime["payload_bytes"],
        "output_runtime_hot_density": new_runtime["density"],
        "output_runtime_hot_span_count": new_runtime["span_count"],
        "runtime_hot_window_reduction": (old_runtime["window_bytes"] / new_runtime["window_bytes"]) if new_runtime["window_bytes"] else None,
        "output_gateup_rowblock_hot_window_bytes": new_gateup_stats,
        "gateup_rowblock_hot_window_reduction": (old_gateup_stats["mean"] / new_gateup_stats["mean"]) if new_gateup_stats["mean"] else None,
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pack-out-dir", type=Path)
    parser.add_argument("--layers", default="all", help="comma/range list, or all")
    parser.add_argument("--hot-experts", default="", help="comma/range list used for every selected layer")
    parser.add_argument("--hot-json", type=Path, help='JSON: {"layers":{"26":[165,0,1]}}')
    parser.add_argument("--route-trace-csv", type=Path, help="DS4_ROUTER_TRACE_PE csv; top experts become layer-local hot order")
    parser.add_argument("--route-top", type=int, default=6)
    parser.add_argument("--route-stages", default="all", help="all, or comma list such as prefill,decode")
    parser.add_argument("--route-score", choices=["mass", "abs_mass", "count"], default="mass")
    parser.add_argument("--gateup-rowblock-overlays", action="store_true",
                        help="materialize hot gate/up experts as 128-row overlay records for deeper locality")
    parser.add_argument("--down-native-code-sidecars", action="store_true",
                        help="materialize hot down expert codes as row-major uint16 gather-native sidecars")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--status-every", type=int, default=32)
    args = parser.parse_args()

    available_layers = sorted(
        int(path.name.split("_L", 1)[1][:2])
        for path in args.pack.glob("ds4_L??_gate_up_down_*.d8f")
    )
    if not available_layers:
        raise SystemExit(f"no D8F layers found under {args.pack}")
    if args.layers == "all":
        layers = available_layers
    else:
        layers = parse_int_set(args.layers, limit=max(available_layers) + 1, name="layer")
        missing = sorted(set(layers) - set(available_layers))
        if missing:
            raise SystemExit(f"requested missing D8F layers: {missing}")

    per_layer_hot: dict[int, list[int]] = {}
    if args.route_trace_csv:
        per_layer_hot.update(route_trace_hot_experts(
            args.route_trace_csv,
            top=args.route_top,
            stages=parse_route_stages(args.route_stages),
            score_mode=args.route_score,
        ))
    if args.hot_json:
        per_layer_hot.update(parse_layer_hot_json(args.hot_json))
    common_hot = parse_int_set(args.hot_experts, limit=EXPERTS, name="expert") if args.hot_experts else []
    reports: list[dict[str, Any]] = []
    for layer in layers:
        hot_experts = per_layer_hot.get(layer, common_hot)
        if not hot_experts:
            raise SystemExit(f"L{layer} has no hot expert list; pass --hot-experts or --hot-json")
        print(json.dumps({
            "phase": "hotblock_layer_start",
            "layer": layer,
            "hot_experts": hot_experts,
            "execute": args.execute,
            "gateup_rowblock_overlays": args.gateup_rowblock_overlays,
            "down_native_code_sidecars": args.down_native_code_sidecars,
        }, sort_keys=True), flush=True)
        report = relayout_layer(
            args.pack, args.out_dir, layer, hot_experts, args.execute,
            args.status_every, args.gateup_rowblock_overlays,
            args.down_native_code_sidecars,
        )
        reports.append(report)
        print(json.dumps({
            "phase": "hotblock_layer_done",
            "layer": layer,
            "ready": report.get("ready", False),
            "materialized": report.get("materialized", False),
            "source_hot_window_bytes": report.get("source_hot_window_bytes"),
            "output_hot_window_bytes": report.get("output_hot_window_bytes"),
            "hot_window_reduction": report.get("hot_window_reduction"),
            "source_runtime_hot_window_bytes": report.get("source_runtime_hot_window_bytes"),
            "output_runtime_hot_window_bytes": report.get("output_runtime_hot_window_bytes"),
            "runtime_hot_window_reduction": report.get("runtime_hot_window_reduction"),
            "gateup_rowblock_hot_window_reduction": report.get("gateup_rowblock_hot_window_reduction"),
            "reason": report.get("reason"),
        }, sort_keys=True), flush=True)

    pack_entries: list[dict[str, Any]] = []
    if args.execute and args.pack_out_dir:
        if not all(item.get("ready") for item in reports):
            raise SystemExit("not all layer reports are ready; refusing pack build")
        pack_entries = hardlink_pack(args.pack, args.pack_out_dir, reports)

    summary = {
        "phase": "ds4_d8f_hotblock_relayout",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "execute": args.execute,
        "pack": str(args.pack),
        "out_dir": str(args.out_dir),
        "pack_out_dir": None if args.pack_out_dir is None else str(args.pack_out_dir),
        "layers": layers,
        "common_hot_experts": common_hot,
        "gateup_rowblock_overlays": args.gateup_rowblock_overlays,
        "down_native_code_sidecars": args.down_native_code_sidecars,
        "route_trace_csv": None if args.route_trace_csv is None else str(args.route_trace_csv),
        "route_top": args.route_top,
        "route_stages": args.route_stages,
        "route_score": args.route_score,
        "layer_reports": reports,
        "all_ready": all(item.get("ready") for item in reports),
        "pack_entries": pack_entries,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "all_ready": summary["all_ready"],
        "execute": args.execute,
        "layers": layers,
        "report": None if args.report is None else str(args.report),
        "pack_out_dir": summary["pack_out_dir"],
    }, sort_keys=True))
    return 0 if summary["all_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
