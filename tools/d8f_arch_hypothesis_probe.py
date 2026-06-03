#!/usr/bin/env python3
"""Executable DS4/D8F architecture hypothesis ledger.

The goal is not to summarize the architecture; it is to turn every assumed
organ/atom invariant into a falsifiable row with current evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from d8f_down_code_cardinality import (
    DOWN_GROUPS,
    EXPERTS,
    PROJECTIONS,
    read_header,
    read_native_codes,
    read_native_records,
    read_record,
)


@dataclass
class Hypothesis:
    id: str
    layer: int | None
    atom: str
    hypothesis: str
    test: str
    status: str
    evidence: dict[str, Any]


def emit(rows: list[Hypothesis], row: Hypothesis) -> None:
    rows.append(row)
    layer = "--" if row.layer is None else f"{row.layer:02d}"
    print(
        f"{row.status:9s} {row.id:28s} L{layer} atom={row.atom} "
        f"evidence={json.dumps(row.evidence, sort_keys=True)}",
        flush=True,
    )


def resolve_pack_path(manifest_path: Path, payload: dict[str, Any]) -> Path:
    pack = Path(str(payload.get("pack", manifest_path.parent)))
    if not pack.is_absolute():
        pack = (manifest_path.parent / pack).resolve()
    return pack


def resolve_d8f_path(pack: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else pack / path


def record_offset_ok(file_size: int, offset: int, byte_count: int) -> bool:
    return offset >= 0 and byte_count >= 0 and offset + byte_count <= file_size


def expected_blocks(projection: str) -> int:
    if projection in ("gate", "up"):
        return 2048 * (4096 // 8)
    if projection == "down":
        return 4096 * (2048 // 8)
    raise ValueError(projection)


def record_hypotheses(handle, file_size: int, layer: int, rows: list[Hypothesis]) -> dict[str, Any]:
    k_counts: dict[str, int] = {}
    overflowing_gateup = 0
    down_tg_fit = 0
    down_tg_overflow = 0
    invalid_records = 0
    for expert in range(EXPERTS):
        gate = read_record(handle, "gate", expert)
        up = read_record(handle, "up", expert)
        if int(gate["k"]) * 16 + int(up["k"]) * 16 > 32768:
            overflowing_gateup += 1
        for projection in PROJECTIONS:
            rec = read_record(handle, projection, expert)
            k = int(rec["k"])
            bits = int(rec["bits"])
            key = f"{projection}_k{k}"
            k_counts[key] = k_counts.get(key, 0) + 1
            codebook_bytes = int(rec["codebook_bytes"])
            index_bytes = int(rec["index_bytes"])
            scale_bytes = int(rec["scale_bytes"])
            valid = (
                int(rec["projection"]) == PROJECTIONS.index(projection)
                and int(rec["expert"]) == expert
                and int(rec["block"]) == 8
                and k > 0
                and bits > 0
                and bits < 31
                and k == (1 << bits)
                and codebook_bytes >= k * 16
                and record_offset_ok(file_size, int(rec["codebook_offset"]), codebook_bytes)
                and record_offset_ok(file_size, int(rec["index_offset"]), index_bytes)
                and (scale_bytes == 0 or record_offset_ok(file_size, int(rec["scale_offset"]), scale_bytes))
                and (index_bytes * 8 // bits) >= expected_blocks(projection)
            )
            if not valid:
                invalid_records += 1
            if projection == "down":
                if k * 16 <= 32768:
                    down_tg_fit += 1
                else:
                    down_tg_overflow += 1
    emit(
        rows,
        Hypothesis(
            "H_RECORD_ATOMS_VALID",
            layer,
            "record",
            "Each layer has 3*256 D8F records whose k/bits/block/offset/index contracts match the kernel assumptions.",
            "Read every fused record and bounds-check codebook/index/scale payloads plus minimum block count.",
            "PASS" if invalid_records == 0 else "FAIL",
            {"invalid_records": invalid_records, "k_counts": k_counts},
        ),
    )
    emit(
        rows,
        Hypothesis(
            "H_GATEUP_TG_CACHE_REFUTED",
            layer,
            "threadgroup_memory",
            "Gate+up codebooks generally fit into M1 Max 32 KiB threadgroup memory.",
            "For each expert, compare gate.k*16 + up.k*16 against 32768 bytes.",
            "REFUTED" if overflowing_gateup else "PASS",
            {"experts_over_32KiB": overflowing_gateup, "experts": EXPERTS},
        ),
    )
    emit(
        rows,
        Hypothesis(
            "H_DOWN_SINGLE_CB_TG_FIT_MIXED",
            layer,
            "threadgroup_memory",
            "A single down codebook can be cached in 32 KiB threadgroup memory.",
            "Count down experts with down.k*16 <= 32768 and >32768.",
            "MIXED" if down_tg_fit and down_tg_overflow else ("PASS" if down_tg_fit else "REFUTED"),
            {"fit_32KiB": down_tg_fit, "overflow_32KiB": down_tg_overflow},
        ),
    )
    return {
        "k_counts": k_counts,
        "gateup_over_32KiB": overflowing_gateup,
        "down_fit_32KiB": down_tg_fit,
        "down_over_32KiB": down_tg_overflow,
    }


def native_hypotheses(handle, header: dict[str, Any], layer_entry: dict[str, Any], layer: int, rows: list[Hypothesis]) -> dict[str, Any]:
    native_records = read_native_records(handle, header)
    hot_experts = [int(expert) for expert in layer_entry.get("hot_experts", [])]
    missing = []
    invalid = []
    code_oob = []
    unique_total = 0
    row_group_total = 0
    max_unique = 0
    for expert in hot_experts:
        native = native_records.get(expert)
        down = read_record(handle, "down", expert)
        if native is None:
            missing.append(expert)
            continue
        native_valid = (
            int(native["rows"]) == 4096
            and int(native["groups"]) == DOWN_GROUPS
            and int(native["bytes"]) == 4096 * DOWN_GROUPS * 2
        )
        if not native_valid:
            invalid.append({"expert": expert, "native": native})
            continue
        codes = read_native_codes(handle, native)
        max_code = int(codes.max())
        if max_code >= int(down["k"]):
            code_oob.append({"expert": expert, "k": int(down["k"]), "max_code": max_code})
        for group in range(codes.shape[1]):
            unique = int(np.unique(codes[:, group]).size)
            unique_total += unique
            max_unique = max(max_unique, unique)
        row_group_total += int(codes.size)
    emit(
        rows,
        Hypothesis(
            "H_SELECTED_NATIVE_CODES_VALID",
            layer,
            "native_code_sidecar",
            "Every selected expert has a complete u16 down-code sidecar and all codes are < down.k.",
            "Read all selected native sidecars, validate shape/bytes, and scan every row-group code.",
            "PASS" if not missing and not invalid and not code_oob else "FAIL",
            {"missing": missing, "invalid": invalid[:3], "code_oob": code_oob[:3], "selected": len(hot_experts)},
        ),
    )
    reduction = row_group_total / max(unique_total, 1)
    emit(
        rows,
        Hypothesis(
            "H_SPARSE_CODE_REUSE_EXISTS",
            layer,
            "native_code_entropy",
            "Native down codes have repeated per-group codes that can be exploited exactly.",
            "Compute per-group unique code counts over every selected native sidecar.",
            "PASS" if reduction > 1.25 else "REFUTED",
            {"row_group_total": row_group_total, "unique_total": unique_total, "row_dot_reduction": reduction, "max_group_unique": max_unique},
        ),
    )
    return {
        "selected": len(hot_experts),
        "native_records": len(native_records),
        "native_missing": missing,
        "unique_total": unique_total,
        "row_group_total": row_group_total,
        "row_dot_reduction": reduction,
    }


def texture_hypotheses(layer_entry: dict[str, Any], layer: int, rows: list[Hypothesis]) -> None:
    hot_experts = [int(expert) for expert in layer_entry.get("hot_experts", [])]
    pack_bytes = int(layer_entry.get("bytes", 0))
    hot_window = int(layer_entry.get("runtime_hot_window_bytes", 0))
    pack2d_width = 16384
    bytes_per_row = pack2d_width * 4 * 2
    height = math.ceil(max(hot_window, 1) / bytes_per_row)
    compact_bytes = 256 * 4096 * 8 * 2
    selected_compact_bytes = len(hot_experts) * 4096 * 8 * 2
    emit(
        rows,
        Hypothesis(
            "H_TEXTURE_SHAPES_FIT",
            layer,
            "texture_cache",
            "Pack2D hot-window, all-expert compact atlas, and selected-expert atlas are legal M1 texture-cache shapes.",
            "Check 16384-wide RGBA16F pack2D height and atlas byte sizes against pack/hot bounds.",
            "PASS" if height <= 16384 and compact_bytes < pack_bytes and selected_compact_bytes < compact_bytes else "FAIL",
            {
                "pack2d_height": height,
                "hot_window_MiB": hot_window / 1048576.0,
                "all_expert_atlas_MiB": compact_bytes / 1048576.0,
                "selected_atlas_MiB": selected_compact_bytes / 1048576.0,
            },
        ),
    )


def sparse_manifest_hypotheses(sparse_manifest: Path | None, rows: list[Hypothesis]) -> None:
    if sparse_manifest is None:
        emit(
            rows,
            Hypothesis(
                "H_H3385_SPARSE_SIDECAR_PROVEN",
                None,
                "sparse_sidecar",
                "H3385 exact sparse group-code sidecars are present, self-contained, and smaller than native codes in aggregate.",
                "Inspect sparse manifest.",
                "MISSING",
                {},
            ),
        )
        return
    payload = json.loads(sparse_manifest.read_text(encoding="utf-8"))
    pack = resolve_pack_path(sparse_manifest, payload)
    layers = payload.get("layers", [])
    sparse_total = int(payload.get("sparse_sidecar_total_bytes", 0))
    native_total = int(payload.get("native_equivalent_bytes", 0))
    unique_total = int(payload.get("unique_total", 0))
    row_group_total = sum(len(layer.get("hot_experts", layer.get("experts", []))) * 4096 * DOWN_GROUPS for layer in layers)
    missing_sidecars = []
    symlinks = []
    for layer in layers:
        sidecar = layer.get("sidecar")
        if sidecar:
            sidecar_path = pack / sidecar
            if not sidecar_path.exists():
                missing_sidecars.append(sidecar)
            if sidecar_path.is_symlink():
                symlinks.append(sidecar)
    status = "PASS" if (
        payload.get("self_contained") is True
        and payload.get("runtime_dependencies") == []
        and len(layers) == 43
        and not missing_sidecars
        and not symlinks
        and native_total > 0
        and sparse_total < native_total
    ) else "FAIL"
    emit(
        rows,
        Hypothesis(
            "H_H3385_SPARSE_SIDECAR_PROVEN",
            None,
            "sparse_sidecar",
            "H3385 exact sparse group-code sidecars are present, self-contained, and smaller than native codes in aggregate.",
            "Inspect sparse manifest for self-contained pack, sidecars, and aggregate bytes.",
            status,
            {
                "pack": str(pack),
                "layers": len(layers),
                "sparse_vs_native": sparse_total / max(native_total, 1),
                "row_dot_reduction": row_group_total / max(unique_total, 1),
                "missing_sidecars": missing_sidecars[:5],
                "symlink_sidecars": symlinks[:5],
            },
        ),
    )


def global_hypotheses(manifest_path: Path, payload: dict[str, Any], rows: list[Hypothesis]) -> None:
    layers = payload.get("layers", [])
    logical_gb = float(payload.get("logical_decimal_GB", payload.get("logical_GB", 0.0)))
    emit(
        rows,
        Hypothesis(
            "H_MODEL_HAS_43_ROUTED_LAYERS",
            None,
            "model",
            "The routed DS4 pack has 43 layers.",
            "Count manifest layers.",
            "PASS" if len(layers) == 43 else "FAIL",
            {"layers": len(layers), "manifest": str(manifest_path)},
        ),
    )
    bad_hot = [
        int(layer.get("layer", -1))
        for layer in layers
        if len(layer.get("hot_experts", [])) != 6 or len(set(layer.get("hot_experts", []))) != len(layer.get("hot_experts", []))
    ]
    emit(
        rows,
        Hypothesis(
            "H_LAYER_TOP6_HOT_EXPERTS",
            None,
            "routing",
            "Each layer currently exposes six selected hot experts for the hot-block layout.",
            "Check hot_experts cardinality and uniqueness in every layer entry.",
            "PASS" if not bad_hot else "FAIL",
            {"bad_layers": bad_hot[:10], "layers_checked": len(layers)},
        ),
    )
    size_status = "MISSING"
    if logical_gb:
        size_status = "PASS" if logical_gb < 52.0 else "REFUTED"
    emit(
        rows,
        Hypothesis(
            "H_52GB_FIT_CURRENT_PACK",
            None,
            "size",
            "The current D8F routed pack contributes to a full model fitting under the 52 GB target.",
            "Use manifest logical_decimal_GB when present.",
            size_status,
            {"logical_decimal_GB": logical_gb, "over_target_GB": max(0.0, logical_gb - 52.0)},
        ),
    )
    emit(
        rows,
        Hypothesis(
            "H_AIME_FIDELITY_PRESERVED",
            None,
            "fidelity",
            "The current pack preserves enough fidelity to pass AIME.",
            "Require a current engine-faithful AIME or accepted proxy result tied to this pack.",
            "MISSING",
            {"reason": "no current AIME pass evidence was found in this manifest"},
        ),
    )
    emit(
        rows,
        Hypothesis(
            "H_50TPS_REACHED",
            None,
            "throughput",
            "DeepSeek 4 Flash currently decodes at 50 t/s on M1 Max 64GB.",
            "Require current full-engine decode measurement.",
            "REFUTED",
            {"reason": "recent journaled full-engine H3000/H336x results are single-digit t/s, not 50 t/s"},
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="H3384 native-code manifest")
    parser.add_argument("--sparse-manifest", type=Path, help="H3385 sparse manifest")
    parser.add_argument("--out", type=Path, help="write JSON evidence")
    parser.add_argument("--max-layers", type=int, default=43, help="limit layer count for canaries")
    args = parser.parse_args()

    started = time.time()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    pack = resolve_pack_path(args.manifest, payload)
    rows: list[Hypothesis] = []
    global_hypotheses(args.manifest, payload, rows)
    sparse_manifest_hypotheses(args.sparse_manifest, rows)
    layer_reports = []
    for index, layer_entry in enumerate(payload.get("layers", [])[: args.max_layers], 1):
        layer = int(layer_entry["layer"])
        d8f_path = resolve_d8f_path(pack, str(layer_entry["d8f"]))
        if index == 1 or index % 5 == 0:
            print(f"phase=layer_probe index={index} layer={layer:02d} path={d8f_path.name}", flush=True)
        if not d8f_path.exists():
            emit(
                rows,
                Hypothesis(
                    "H_LAYER_D8F_EXISTS",
                    layer,
                    "file",
                    "Layer D8F file exists.",
                    "Check path from manifest.",
                    "FAIL",
                    {"path": str(d8f_path)},
                ),
            )
            continue
        if d8f_path.is_symlink():
            file_status = "FAIL"
        else:
            file_status = "PASS"
        emit(
            rows,
            Hypothesis(
                "H_LAYER_D8F_SELF_CONTAINED",
                layer,
                "file",
                "Layer D8F file is present as a real self-contained file, not a symlink.",
                "Check exists and symlink status.",
                file_status,
                {"path": str(d8f_path), "bytes": d8f_path.stat().st_size, "is_symlink": d8f_path.is_symlink()},
            ),
        )
        with d8f_path.open("rb") as handle:
            header = read_header(handle)
            layer_match = int(header.get("layer", -1)) == layer
            emit(
                rows,
                Hypothesis(
                    "H_D8F_HEADER_LAYER_MATCH",
                    layer,
                    "header",
                    "D8F header layer ID matches manifest layer.",
                    "Read DS4D8F header JSON.",
                    "PASS" if layer_match else "FAIL",
                    {"header_layer": header.get("layer"), "manifest_layer": layer},
                ),
            )
            record_report = record_hypotheses(handle, d8f_path.stat().st_size, layer, rows)
            native_report = native_hypotheses(handle, header, layer_entry, layer, rows)
            texture_hypotheses(layer_entry, layer, rows)
            layer_reports.append({"layer": layer, "records": record_report, "native": native_report})
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1
    report = {
        "manifest": str(args.manifest),
        "pack": str(pack),
        "sparse_manifest": str(args.sparse_manifest) if args.sparse_manifest else None,
        "elapsed_s": time.time() - started,
        "status_counts": status_counts,
        "hypotheses": [asdict(row) for row in rows],
        "layers": layer_reports,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("arch_hypothesis_summary " + json.dumps(status_counts, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
