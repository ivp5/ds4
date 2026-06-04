#!/usr/bin/env python3
"""Build rank1-sidecar gains data from measured route-mass x rank1-harm feedback."""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

EXPERTS = 256
HEADER_BYTES = 4096
SIDECAR_RECORD_BYTES = 64
SIDECAR_RECORD = struct.Struct("<IIIIQQIIIIffff")


def parse_layers(text: str) -> list[int]:
    layers: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first, last = part.split("-", 1)
            layers.extend(range(int(first), int(last) + 1))
        else:
            layers.append(int(part))
    return sorted(dict.fromkeys(layers))


def link_file(source: Path, destination: Path) -> str:
    if destination.exists():
        raise FileExistsError(destination)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        destination.write_bytes(source.read_bytes())
        return "copy"


def read_d8f_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        prefix = handle.read(16)
        if len(prefix) != 16 or prefix[:8] != b"DS4D8F1\0":
            raise ValueError(f"{path}: bad DS4D8F magic")
        header_bytes = struct.unpack("<I", prefix[12:16])[0]
        if header_bytes <= 0 or header_bytes > HEADER_BYTES - 16:
            raise ValueError(f"{path}: bad header length {header_bytes}")
        return json.loads(handle.read(header_bytes).decode("utf-8"))


def existing_sidecars(pack_dir: Path, layer: int) -> list[int]:
    path = pack_dir / f"ds4_L{layer:02d}_gate_up_down_VQD8_noE8_rank1.d8f"
    if not path.exists():
        return []
    header = read_d8f_header(path)
    table_offset = int(header.get("sidecar_table_offset", 0))
    record_bytes = int(header.get("sidecar_record_bytes", SIDECAR_RECORD_BYTES))
    if table_offset <= 0 or record_bytes != SIDECAR_RECORD_BYTES:
        return []
    selected: list[int] = []
    with path.open("rb") as handle:
        for expert in range(EXPERTS):
            handle.seek(table_offset + expert * record_bytes)
            record = SIDECAR_RECORD.unpack(handle.read(SIDECAR_RECORD_BYTES))
            if int(record[1]) > 0 and int(record[4]) > 0 and int(record[5]) > 0:
                selected.append(expert)
    return selected


def lognormal_fit(values: list[float]) -> dict[str, float]:
    logs = [math.log(max(float(value), 1.0e-12)) for value in values]
    mean = sum(logs) / float(len(logs)) if logs else 0.0
    var = sum((value - mean) ** 2 for value in logs) / float(len(logs)) if logs else 0.0
    return {"mean": mean, "sigma": math.sqrt(var) or 1.0}


def lognormal_z(value: float, fit: dict[str, float]) -> float:
    return (math.log(max(float(value), 1.0e-12)) - fit["mean"]) / fit["sigma"]


def load_direction_index(data_dir: Path, layers: set[int]) -> dict[int, dict[str, set[int] | set[str]]]:
    index: dict[int, dict[str, set[int] | set[str]]] = {}
    for layer in layers:
        targeting_path = data_dir / f"targeting_L{layer}.json"
        dirs_path = data_dir / f"dirs_L{layer}.npz"
        if not targeting_path.exists() or not dirs_path.exists():
            index[layer] = {"down_below": set(), "dirs": set()}
            continue
        targeting = json.loads(targeting_path.read_text(encoding="utf-8"))
        dirs = np.load(dirs_path)
        try:
            index[layer] = {
                "down_below": {int(value) for value in targeting.get("down_below", [])},
                "dirs": set(dirs.files),
            }
        finally:
            dirs.close()
    return index


def valid_direction(direction_index: dict[int, dict[str, set[int] | set[str]]], layer: int, expert: int) -> bool:
    layer_index = direction_index.get(layer)
    if not layer_index:
        return False
    return expert in layer_index["down_below"] and f"L{layer}_e{expert}_u" in layer_index["dirs"]


def priority_rows(payload: dict[str, Any], layers: set[int], score_mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for row in payload.get("top40", []):
        layer = int(row["L"])
        if layer not in layers:
            continue
        rank1_energy = float(row["r1e"])
        rank1_relerr = float(row["r1err"])
        route_mass = float(row["rmass"])
        priority = float(row["priority"])
        rows.append(
            {
                "layer": layer,
                "expert": int(row["e"]),
                "rank1_energy": rank1_energy,
                "rank1_relerr": rank1_relerr,
                "rank1_harm": rank1_energy * rank1_relerr,
                "route_mass": route_mass,
                "priority": priority,
                "already_sidecar": bool(row.get("sidecar", False)),
                "source": "agent1_priority",
            }
        )
    score_meta: dict[str, Any] = {"mode": score_mode, "fields": []}
    for row in rows:
        row["selection_score"] = float(row["priority"])
    if score_mode == "lognormal":
        score_fields = ["route_mass", "rank1_energy", "rank1_relerr"]
        fits = {field: lognormal_fit([float(row[field]) for row in rows]) for field in score_fields}
        score_meta["fields"] = [{**{"name": field}, **fits[field]} for field in score_fields]
        for row in rows:
            row["lognormal_score"] = sum(lognormal_z(float(row[field]), fits[field]) for field in score_fields)
            row["selection_score"] = float(row["lognormal_score"])
    elif score_mode != "raw":
        raise ValueError(f"unknown score mode {score_mode}")
    rows.sort(key=lambda item: (-float(item["selection_score"]), int(item["layer"]), int(item["expert"])))
    return rows, score_meta


def choose_rows(
    rows: list[dict[str, Any]],
    direction_index: dict[int, dict[str, set[int] | set[str]]],
    top_total: int,
    max_per_layer: int,
    relax_cap_to_fill_budget: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, int], bool]:
    chosen: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    delayed_by_cap: list[dict[str, Any]] = []
    chosen_pairs: set[tuple[int, int]] = set()
    per_layer_counts: dict[int, int] = {}
    for row in rows:
        layer = int(row["layer"])
        expert = int(row["expert"])
        if not valid_direction(direction_index, layer, expert):
            rejected.append({**row, "reason": "missing_direction_or_not_down_below"})
            continue
        if max_per_layer > 0 and per_layer_counts.get(layer, 0) >= max_per_layer:
            delayed_by_cap.append(row)
            continue
        chosen.append(row)
        chosen_pairs.add((layer, expert))
        per_layer_counts[layer] = per_layer_counts.get(layer, 0) + 1
        if len(chosen) >= top_total:
            rejected.extend({**row, "reason": "not_reached"} for row in delayed_by_cap)
            return chosen, rejected, per_layer_counts, False
    relaxed_fill = False
    if relax_cap_to_fill_budget and max_per_layer > 0 and len(chosen) < top_total:
        relaxed_fill = True
        for row in delayed_by_cap:
            layer = int(row["layer"])
            expert = int(row["expert"])
            if (layer, expert) in chosen_pairs:
                continue
            chosen.append({**row, "cap_relaxed": True})
            chosen_pairs.add((layer, expert))
            per_layer_counts[layer] = per_layer_counts.get(layer, 0) + 1
            if len(chosen) >= top_total:
                break
    rejected.extend(
        {**row, "reason": "max_per_layer"}
        for row in delayed_by_cap
        if (int(row["layer"]), int(row["expert"])) not in chosen_pairs
    )
    return chosen, rejected, per_layer_counts, relaxed_fill


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priority-json", type=Path, required=True)
    parser.add_argument("--source-data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layers", default="36-42")
    parser.add_argument("--top-total", type=int, default=30)
    parser.add_argument("--max-per-layer", type=int, default=0)
    parser.add_argument("--include-existing-pack-dir", type=Path, default=None)
    parser.add_argument("--score-mode", choices=("lognormal", "raw"), default="lognormal")
    parser.add_argument("--relax-cap-to-fill-budget", action="store_true")
    args = parser.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"output directory exists and is not empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    layers = set(parse_layers(args.layers))
    payload = json.loads(args.priority_json.read_text(encoding="utf-8"))
    rows, score_meta = priority_rows(payload, layers, args.score_mode)
    direction_index = load_direction_index(args.source_data_dir, layers)
    chosen, rejected, per_layer_counts, cap_relaxed = choose_rows(
        rows,
        direction_index,
        args.top_total,
        args.max_per_layer,
        args.relax_cap_to_fill_budget,
    )

    existing_added: list[dict[str, Any]] = []
    if args.include_existing_pack_dir is not None:
        chosen_pairs = {(int(row["layer"]), int(row["expert"])) for row in chosen}
        for layer in sorted(layers):
            for expert in existing_sidecars(args.include_existing_pack_dir, layer):
                pair = (layer, expert)
                if pair in chosen_pairs:
                    continue
                if not valid_direction(direction_index, layer, expert):
                    rejected.append({"layer": layer, "expert": expert, "source": "existing_sidecar", "reason": "missing_direction_or_not_down_below"})
                    continue
                row = {
                    "layer": layer,
                    "expert": expert,
                    "rank1_energy": 0.0,
                    "rank1_relerr": 0.0,
                    "rank1_harm": 0.2,
                    "route_mass": 0.0,
                    "priority": 1.0e-9,
                    "selection_score": 1.0e-9,
                    "already_sidecar": True,
                    "source": "existing_sidecar",
                }
                chosen.append(row)
                existing_added.append(row)
                chosen_pairs.add(pair)

    link_modes: set[str] = set()
    for layer in sorted(layers):
        for stem in (f"dirs_L{layer}.npz", f"targeting_L{layer}.json"):
            source = args.source_data_dir / stem
            if source.exists():
                link_modes.add(link_file(source, args.out_dir / stem))

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in chosen:
        grouped.setdefault(int(row["layer"]), []).append(row)
    for layer, layer_rows in grouped.items():
        layer_rows.sort(key=lambda item: (-float(item["selection_score"]), int(item["expert"])))
        selected = [int(row["expert"]) for row in layer_rows]
        priority_sum = sum(float(row["priority"]) for row in layer_rows if float(row["priority"]) > 0.0)
        gains = {
            "layer": layer,
            "selected": selected,
            "cum_target": 1.0,
            "selector": "route_mass_x_rank1_harm_agent1_20260604",
            "down_gain": {str(row["expert"]): max(float(row["rank1_harm"]), 0.2) for row in layer_rows},
            "down_block_reduction": {str(row["expert"]): float(row["priority"]) for row in layer_rows},
            "down_block_share": {
                str(row["expert"]): (float(row["priority"]) / priority_sum if priority_sum > 0.0 else 0.0)
                for row in layer_rows
            },
            "down_aa_frac": {str(row["expert"]): float(row["rank1_energy"]) for row in layer_rows},
            "score_meta": score_meta,
            "rank1_harm_rows": layer_rows,
        }
        (args.out_dir / f"gains_L{layer}.json").write_text(json.dumps(gains, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "priority_json": str(args.priority_json),
        "source_data_dir": str(args.source_data_dir),
        "layers": sorted(layers),
        "top_total": args.top_total,
        "max_per_layer": args.max_per_layer,
        "relax_cap_to_fill_budget": args.relax_cap_to_fill_budget,
        "cap_relaxed_to_fill_budget": cap_relaxed,
        "score_meta": score_meta,
        "include_existing_pack_dir": str(args.include_existing_pack_dir) if args.include_existing_pack_dir else None,
        "link_modes": sorted(link_modes),
        "chosen": chosen,
        "chosen_count": len(chosen),
        "existing_added": existing_added,
        "rejected": rejected,
        "rejected_count": len(rejected),
        "sidecar_bytes_rank1": len(chosen) * (2048 + 4096) * 2,
    }
    (args.out_dir / "H3386_RANK1_HARM_ALLOCATOR_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"rank1 harm allocation complete chosen={len(chosen)} rejected={len(rejected)} "
        f"sidecar_mib={manifest['sidecar_bytes_rank1']/1048576.0:.3f} out={args.out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
