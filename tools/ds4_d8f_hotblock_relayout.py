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
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from ds4_sparse_projection_splice import (
    EXPERTS,
    FUSED_RECORD,
    FUSED_RECORD_BYTES,
    HEADER_BYTES,
    PROJECTIONS,
    append_sidecars,
    copy_record,
    d8f_path,
    hardlink_pack,
    read_d8f,
    rewrite_header,
)

GATEUP_OVERLAY_PREFIX = "gateup_overlay_"


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
    ]:
        header.pop(key, None)


def relayout_layer(source_pack: Path,
                   out_dir: Path,
                   layer: int,
                   hot_experts: list[int],
                   execute: bool,
                   status_every: int) -> dict[str, Any]:
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
    report.update({
        "ready": True,
        "source_bytes": info["bytes"],
        "source_hot_window_bytes": old_window[2],
        "source_hot_expert_span_bytes": old_expert_spans,
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
    })
    payload_hashers = {projection: hashlib.sha256() for projection in PROJECTIONS}
    new_records: list[tuple[Any, ...] | None] = [None] * (len(PROJECTIONS) * EXPERTS)
    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    with temp_path.open("wb") as handle:
        rewrite_header(handle, header)
        handle.write(b"\0" * (len(PROJECTIONS) * EXPERTS * FUSED_RECORD_BYTES))
        expert_order = ordered_experts(hot_experts)
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
        sidecar_header = append_sidecars(
            source_path,
            source_path,
            info["sidecars"],
            info["sidecars"],
            set(),
            handle,
        )
        header.update(sidecar_header)
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
    report.update({
        "materialized": True,
        "output_bytes": out_path.stat().st_size,
        "output_hot_window_bytes": new_window[2],
        "output_hot_expert_span_bytes": new_expert_spans,
        "hot_window_reduction": (old_window[2] / new_window[2]) if new_window[2] else None,
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
        }, sort_keys=True), flush=True)
        report = relayout_layer(args.pack, args.out_dir, layer, hot_experts, args.execute, args.status_every)
        reports.append(report)
        print(json.dumps({
            "phase": "hotblock_layer_done",
            "layer": layer,
            "ready": report.get("ready", False),
            "materialized": report.get("materialized", False),
            "source_hot_window_bytes": report.get("source_hot_window_bytes"),
            "output_hot_window_bytes": report.get("output_hot_window_bytes"),
            "hot_window_reduction": report.get("hot_window_reduction"),
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
