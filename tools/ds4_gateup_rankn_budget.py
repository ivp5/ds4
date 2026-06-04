#!/usr/bin/env python3
"""Budget rank-N gate/up residual sidecars against a DS4 allocation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DIM_MODEL = 4096
DIM_FFN = 2048
FP16_BYTES = 2


def residual_bytes(rank: int, projections: int = 2) -> int:
    return projections * rank * (DIM_MODEL + DIM_FFN) * FP16_BYTES


def load_promoted_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return list(manifest.get("late_route_hot_gateup_promotions", {}).get("rows", []))


def selected_from_allocation(allocation: np.ndarray, route_threshold: float) -> list[dict[str, Any]]:
    mask = (
        (allocation["layer"] >= 40) &
        (allocation["layer"] <= 42) &
        (allocation["route_mass"] >= route_threshold)
    )
    rows: list[dict[str, Any]] = []
    for row in allocation[mask][np.argsort(allocation[mask]["route_mass"])[::-1]]:
        rows.append(
            {
                "layer": int(row["layer"]),
                "expert": int(row["expert"]),
                "rank": int(row["rank"]) if "rank" in allocation.dtype.names else 0,
                "route_mass": float(row["route_mass"]),
                "K_gate": int(row["K_gate"]),
                "K_up": int(row["K_up"]),
                "K_down": int(row["K_down"]),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--allocation", type=Path)
    parser.add_argument("--rank", type=int, default=131)
    parser.add_argument("--route-threshold", type=float, default=0.02)
    parser.add_argument("--mode", choices=("manifest-promoted", "allocation-threshold"), default="manifest-promoted")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    byte_model = manifest.get("byte_model", {})
    current_total = int(byte_model["target_total_bytes_with_template_extras"])
    current_margin = int(byte_model["margin_to_52gb_decimal"])

    if args.mode == "manifest-promoted":
        rows = load_promoted_rows(manifest)
    else:
        if not args.allocation:
            raise SystemExit("--allocation is required for allocation-threshold mode")
        rows = selected_from_allocation(np.load(args.allocation, allow_pickle=False), args.route_threshold)

    per_expert = residual_bytes(args.rank)
    added = per_expert * len(rows)
    total = current_total + added
    margin = 52_000_000_000 - total
    payload = {
        "rank": args.rank,
        "format_assumption": "fp16 U[out_dim,rank] + fp16 A[rank,in_dim] for gate and up",
        "bytes_per_gateup_expert": per_expert,
        "bytes_per_gateup_expert_mib": per_expert / 1048576.0,
        "selected_experts": len(rows),
        "added_bytes": added,
        "added_mib": added / 1048576.0,
        "current_total_bytes": current_total,
        "current_margin_to_52gb_decimal": current_margin,
        "new_total_bytes": total,
        "new_margin_to_52gb_decimal": margin,
        "fits_52gb_decimal": margin >= 0,
        "selected_rows": rows,
    }
    print(json.dumps(payload, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if margin >= 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
