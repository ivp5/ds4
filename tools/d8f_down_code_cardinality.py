#!/usr/bin/env python3
"""Analyze D8F down native-code locality for exact sparse-code layouts.

The direct fused down kernel pays one codebook dot per row/group.  A code-major
layout would instead compute one dot per unique code inside each group, then
gather/scatter that score back to rows.  This tool reads the authoritative D8F
native-code sidecars and estimates whether that exact rewrite has enough
cardinality reduction to justify the extra indirection.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

FUSED_MAGIC = b"DS4D8F1\0"
HEADER_BYTES = 4096
EXPERTS = 256
PROJECTIONS = ("gate", "up", "down")
PROJECTION_ID = {name: index for index, name in enumerate(PROJECTIONS)}
FUSED_RECORD_BYTES = 64
FUSED_RECORD = struct.Struct("<IIIIIIQQQIIII")
DOWN_NATIVE_CODE_RECORD_BYTES = 32
DOWN_NATIVE_CODE_RECORD = struct.Struct("<IIIIQII")
DOWN_NATIVE_CODE_DTYPE_U16 = 1
DOWN_GROUPS = 256


def d8f_path(root: Path, layer: int) -> Path:
    exact = root / f"ds4_L{layer:02d}_gate_up_down_VQD8_noE8_rank1.d8f"
    if exact.exists():
        return exact
    matches = sorted(root.glob(f"ds4_L{layer:02d}_gate_up_down_*.d8f"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(exact)
    raise RuntimeError(f"ambiguous D8F files for L{layer:02d}: {[p.name for p in matches]}")


def read_header(handle) -> dict[str, Any]:
    handle.seek(0)
    raw = handle.read(HEADER_BYTES)
    if len(raw) != HEADER_BYTES or raw[:8] != FUSED_MAGIC:
        raise ValueError("not a DS4D8F1 file")
    version, header_json_bytes = struct.unpack_from("<II", raw, 8)
    if version != 1 or 16 + header_json_bytes > HEADER_BYTES:
        raise ValueError(f"unsupported D8F header version={version} json={header_json_bytes}")
    return json.loads(raw[16 : 16 + header_json_bytes].decode("utf-8"))


def read_record(handle, projection: str, expert: int) -> dict[str, int]:
    index = PROJECTION_ID[projection] * EXPERTS + expert
    handle.seek(HEADER_BYTES + index * FUSED_RECORD_BYTES)
    raw = handle.read(FUSED_RECORD_BYTES)
    if len(raw) != FUSED_RECORD_BYTES:
        raise ValueError(f"truncated record projection={projection} expert={expert}")
    fields = FUSED_RECORD.unpack(raw)
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


def read_native_records(handle, header: dict[str, Any]) -> dict[int, dict[str, int]]:
    table_offset = int(header.get("down_native_code_sidecar_table_offset", 0))
    record_bytes = int(header.get("down_native_code_sidecar_record_bytes", DOWN_NATIVE_CODE_RECORD_BYTES))
    records = int(header.get("down_native_code_sidecar_records", 0))
    if not table_offset or record_bytes != DOWN_NATIVE_CODE_RECORD_BYTES:
        return {}
    result: dict[int, dict[str, int]] = {}
    for row in range(records):
        handle.seek(table_offset + row * record_bytes)
        raw = handle.read(record_bytes)
        if len(raw) != record_bytes:
            raise ValueError(f"truncated native-code record row={row}")
        expert, rows, groups, dtype, offset, byte_count, flags = DOWN_NATIVE_CODE_RECORD.unpack(raw)
        if byte_count == 0:
            continue
        if dtype != DOWN_NATIVE_CODE_DTYPE_U16 or groups != DOWN_GROUPS or byte_count != rows * groups * 2:
            raise ValueError(
                f"invalid native-code record expert={expert} rows={rows} groups={groups} "
                f"dtype={dtype} bytes={byte_count}"
            )
        result[expert] = {
            "expert": expert,
            "rows": rows,
            "groups": groups,
            "offset": offset,
            "bytes": byte_count,
            "flags": flags,
        }
    return result


def read_native_codes(handle, rec: dict[str, int]) -> np.ndarray:
    handle.seek(rec["offset"])
    raw = handle.read(rec["bytes"])
    if len(raw) != rec["bytes"]:
        raise ValueError(f"truncated native-code payload expert={rec['expert']}")
    return np.frombuffer(raw, dtype="<u2").reshape(rec["rows"], rec["groups"])


def quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values.astype(np.float64), q))


def expert_metrics(codes: np.ndarray, *, k: int, layer: int, expert: int) -> dict[str, Any]:
    rows, groups = codes.shape
    unique_counts = np.empty(groups, dtype=np.int32)
    max_freq = np.empty(groups, dtype=np.int32)
    hot_sizes = (16, 32, 64, 128, 256, 512, 1024)
    hot_mass = {hot: np.empty(groups, dtype=np.float64) for hot in hot_sizes}
    inverse_bytes = 0
    for group in range(groups):
        counts = np.bincount(codes[:, group], minlength=k)
        live = counts[counts > 0]
        live_sorted = np.sort(live)[::-1]
        unique = int(live_sorted.size)
        unique_counts[group] = unique
        max_freq[group] = int(live_sorted[0]) if unique else 0
        bits = max(1, math.ceil(math.log2(max(unique, 1))))
        inverse_bytes += (rows * bits + 7) // 8
        cumulative = np.cumsum(live_sorted, dtype=np.int64)
        for hot in hot_sizes:
            if hot >= unique:
                covered = rows
            else:
                covered = int(cumulative[hot - 1])
            hot_mass[hot][group] = covered / rows
    total_unique = int(unique_counts.sum())
    row_group_count = rows * groups
    dense_score_count = groups * k
    sparse_score_bytes = total_unique * 4
    unique_list_bytes = total_unique * 2
    group_prefix_bytes = (groups + 1) * 4
    sparse_map_bytes = unique_list_bytes + inverse_bytes + group_prefix_bytes
    return {
        "layer": layer,
        "expert": expert,
        "k": k,
        "rows": rows,
        "groups": groups,
        "row_group_count": row_group_count,
        "unique_total": total_unique,
        "unique_mean": float(unique_counts.mean()),
        "unique_min": int(unique_counts.min()),
        "unique_p10": quantile(unique_counts, 0.10),
        "unique_p50": quantile(unique_counts, 0.50),
        "unique_p90": quantile(unique_counts, 0.90),
        "unique_max": int(unique_counts.max()),
        "row_dot_reduction": row_group_count / max(total_unique, 1),
        "dense_score_count": dense_score_count,
        "sparse_score_count": total_unique,
        "sparse_vs_dense_score": total_unique / max(dense_score_count, 1),
        "sparse_score_bytes": sparse_score_bytes,
        "sparse_map_bytes": sparse_map_bytes,
        "native_code_bytes": row_group_count * 2,
        "max_freq_mean": float(max_freq.mean()),
        "max_freq_p90": quantile(max_freq, 0.90),
        "hot_mass": {str(hot): float(values.mean()) for hot, values in hot_mass.items() if hot <= k},
    }


def load_layers(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.manifest:
        payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        layers = payload.get("layers")
        if not isinstance(layers, list):
            raise SystemExit("manifest must contain a layers list")
        return layers
    if not args.pack:
        raise SystemExit("provide --manifest or --pack")
    return [
        {
            "layer": layer,
            "d8f": str(d8f_path(Path(args.pack), layer)),
            "hot_experts": [int(part) for part in args.experts.split(",") if part],
        }
        for layer in range(args.first_layer, args.last_layer + 1)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="H3384 manifest with layers/hot_experts")
    parser.add_argument("--pack", help="D8F pack root, used when no manifest is provided")
    parser.add_argument("--experts", default="", help="CSV expert list for --pack mode")
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--last-layer", type=int, default=42)
    parser.add_argument("--out", help="write full JSON report")
    args = parser.parse_args()

    layers = load_layers(args)
    per_expert: list[dict[str, Any]] = []
    started = time.time()
    for layer_entry in layers:
        layer = int(layer_entry["layer"])
        d8f = Path(layer_entry["d8f"])
        hot_experts = [int(expert) for expert in layer_entry.get("hot_experts", [])]
        if not hot_experts:
            continue
        with d8f.open("rb") as handle:
            header = read_header(handle)
            native_records = read_native_records(handle, header)
            for expert in hot_experts:
                if expert not in native_records:
                    continue
                down = read_record(handle, "down", expert)
                codes = read_native_codes(handle, native_records[expert])
                metrics = expert_metrics(codes, k=int(down["k"]), layer=layer, expert=expert)
                per_expert.append(metrics)
                print(
                    "d8f_cardinality "
                    f"L{layer:02d} E{expert:03d} k={metrics['k']} "
                    f"unique_mean={metrics['unique_mean']:.1f} "
                    f"row_dot_reduction={metrics['row_dot_reduction']:.3f} "
                    f"sparse_vs_dense={metrics['sparse_vs_dense_score']:.3f} "
                    f"hot256={metrics['hot_mass'].get('256', 0.0):.3f}",
                    flush=True,
                )
    if not per_expert:
        raise SystemExit("no native-code experts analyzed")
    unique_total = sum(item["unique_total"] for item in per_expert)
    row_group_total = sum(item["row_group_count"] for item in per_expert)
    dense_score_total = sum(item["dense_score_count"] for item in per_expert)
    sparse_score_bytes = sum(item["sparse_score_bytes"] for item in per_expert)
    sparse_map_bytes = sum(item["sparse_map_bytes"] for item in per_expert)
    native_code_bytes = sum(item["native_code_bytes"] for item in per_expert)
    summary = {
        "experts": len(per_expert),
        "elapsed_s": time.time() - started,
        "row_group_total": row_group_total,
        "unique_total": unique_total,
        "unique_mean": float(np.mean([item["unique_mean"] for item in per_expert])),
        "row_dot_reduction": row_group_total / max(unique_total, 1),
        "sparse_vs_dense_score": unique_total / max(dense_score_total, 1),
        "sparse_score_bytes": sparse_score_bytes,
        "sparse_map_bytes": sparse_map_bytes,
        "native_code_bytes": native_code_bytes,
        "sparse_map_vs_native_code_bytes": sparse_map_bytes / max(native_code_bytes, 1),
        "hot_mass": {
            str(hot): float(np.mean([item["hot_mass"][str(hot)] for item in per_expert if str(hot) in item["hot_mass"]]))
            for hot in (16, 32, 64, 128, 256, 512, 1024)
            if any(str(hot) in item["hot_mass"] for item in per_expert)
        },
    }
    report = {"summary": summary, "experts": per_expert}
    print("d8f_cardinality_summary " + json.dumps(summary, sort_keys=True))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
