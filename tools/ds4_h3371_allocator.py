#!/usr/bin/env python3
"""Regenerate the H3371 general-fidelity allocation from H3216.

H3371 is the non-AIME-derived repair target recorded in the codec registry:
drop the H3230 overlay, demote L40-L42 down records above K512 to K512, then
demote the 186 weakest L40 down records from K512 to K256.  This tool makes
that rule executable and emits a byte-budget manifest for pack rebuilds.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

EXPERTS = 256
LAYERS = 43
HEADER_BYTES = 4096
FUSED_RECORD_BYTES = 64
PROJECTIONS = 3
GROUP = 8
ROUTED_ROWS = 2048
ROUTED_COLS = 4096
DOWN_SCALE_BYTES = 4096
INDEX_GUARD_BYTES = 4
WEIGHTS_PER_PROJECTION = ROUTED_ROWS * ROUTED_COLS
GROUPS_PER_PROJECTION = WEIGHTS_PER_PROJECTION // GROUP
DEFAULT_INPUT = Path("tmp/20260602_codec_general/ds4_52gb_allocation_h3216_l39_downprotected_l40_42_gateup_floor.npy")
DEFAULT_TEMPLATE_PACK = Path(
    "/Users/silv/cl/tlp/montyneg/ds4/"
    "DeepSeek-V4-Flash_H3384_H3382_all43_route_hotblock_sidecar_top6_down_native_codes_D8F_800kctx_probe_20260604"
)


def projection_bpw(k: int) -> float:
    bits = int(math.ceil(math.log2(k)))
    codebook_bits = k * GROUP * 16
    return float(bits + codebook_bits / WEIGHTS_PER_PROJECTION)


def record_payload_bytes(k: int, *, down: bool) -> int:
    bits = int(math.ceil(math.log2(k)))
    codebook_bytes = k * GROUP * 2
    index_bytes = (GROUPS_PER_PROJECTION * bits + 7) // 8 + INDEX_GUARD_BYTES
    return codebook_bytes + index_bytes + (DOWN_SCALE_BYTES if down else 0)


def routed_payload_bytes(allocation: np.ndarray) -> int:
    total = LAYERS * (HEADER_BYTES + PROJECTIONS * EXPERTS * FUSED_RECORD_BYTES)
    for row in allocation:
        total += record_payload_bytes(int(row["K_gate"]), down=False)
        total += record_payload_bytes(int(row["K_up"]), down=False)
        total += record_payload_bytes(int(row["K_down"]), down=True)
    return total


def k_hist(allocation: np.ndarray, field: str) -> dict[str, int]:
    values, counts = np.unique(allocation[field], return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts, strict=True)}


def template_bytes(template_pack: Path | None) -> dict[str, int]:
    if template_pack is None:
        return {}
    metadata = template_pack / "DeepSeek-V4-Flash.metadata-only.full-tensor-manifest.zero-tensor-data.pack-direct.gguf"
    nonrouted = template_pack / "ds4v4_nonrouted.i32_normf32_bf16matf16.pack"
    out: dict[str, int] = {}
    if metadata.exists():
        out["metadata_bytes"] = metadata.stat().st_size
    if nonrouted.exists():
        out["nonrouted_bytes"] = nonrouted.stat().st_size
    return out


def update_down_fields(allocation: np.ndarray) -> None:
    if "bpw_down" in allocation.dtype.names:
        for k in np.unique(allocation["K_down"]):
            allocation["bpw_down"][allocation["K_down"] == k] = projection_bpw(int(k))
    if "codec_down" in allocation.dtype.names:
        allocation["codec_down"][:] = "VQ-D8"
    if "cbtrain_down" in allocation.dtype.names:
        allocation["cbtrain_down"][:] = "act-aware"


def update_gateup_fields(allocation: np.ndarray) -> None:
    for projection in ("gate", "up"):
        k_field = f"K_{projection}"
        bpw_field = f"bpw_{projection}"
        codec_field = f"codec_{projection}"
        cbtrain_field = f"cbtrain_{projection}"
        if bpw_field in allocation.dtype.names:
            for k in np.unique(allocation[k_field]):
                allocation[bpw_field][allocation[k_field] == k] = projection_bpw(int(k))
        if codec_field in allocation.dtype.names:
            allocation[codec_field][:] = "VQ-D8"
        if cbtrain_field in allocation.dtype.names:
            allocation[cbtrain_field][:] = "plain"


def weakest_l40_down_rows(allocation: np.ndarray, count: int) -> np.ndarray:
    indices = np.where((allocation["layer"] == 40) & (allocation["K_down"] == 512))[0]
    if len(indices) < count:
        raise ValueError(f"only {len(indices)} L40 K512 down rows remain; cannot demote {count}")
    priority = allocation["combo_priority"][indices] if "combo_priority" in allocation.dtype.names else np.zeros(len(indices))
    route_mass = allocation["route_mass"][indices] if "route_mass" in allocation.dtype.names else np.zeros(len(indices))
    up_priority = allocation["up_priority"][indices] if "up_priority" in allocation.dtype.names else np.zeros(len(indices))
    rank = allocation["rank"][indices] if "rank" in allocation.dtype.names else np.arange(len(indices))
    order = np.lexsort((-rank, up_priority, route_mass, priority))
    return indices[order[:count]]


def build_h3371(source: np.ndarray, l40_k256_count: int) -> tuple[np.ndarray, dict[str, Any]]:
    allocation = source.copy()
    late_mask = (allocation["layer"] >= 40) & (allocation["layer"] <= 42) & (allocation["K_down"] > 512)
    late_before = allocation["K_down"][late_mask].copy()
    allocation["K_down"][late_mask] = 512
    weak = weakest_l40_down_rows(allocation, l40_k256_count)
    weak_rows = [
        {
            "layer": int(allocation["layer"][index]),
            "expert": int(allocation["expert"][index]),
            "rank": int(allocation["rank"][index]) if "rank" in allocation.dtype.names else int(index),
            "route_mass": float(allocation["route_mass"][index]) if "route_mass" in allocation.dtype.names else 0.0,
            "up_priority": float(allocation["up_priority"][index]) if "up_priority" in allocation.dtype.names else 0.0,
        }
        for index in weak
    ]
    allocation["K_down"][weak] = 256
    update_down_fields(allocation)
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "H3216 allocation",
        "target": "H3371_H3216_budget_repair_no_overlay",
        "rule": [
            "drop H3230 overlay",
            "demote L40-L42 down K>512 records to K512",
            f"demote {l40_k256_count} weakest L40 down K512 records to K256",
        ],
        "late_down_demotions": {
            "count": int(late_mask.sum()),
            "from_hist": {str(int(k)): int(v) for k, v in zip(*np.unique(late_before, return_counts=True), strict=True)},
        },
        "l40_k256_demotions": {
            "count": int(len(weak)),
            "tie_break": "combo_priority asc, route_mass asc, up_priority asc, rank desc",
            "first_rows": weak_rows[:24],
        },
        "k_hist": {
            "gate": k_hist(allocation, "K_gate"),
            "up": k_hist(allocation, "K_up"),
            "down": k_hist(allocation, "K_down"),
        },
    }
    return allocation, manifest


def build_source_base(source: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    allocation = source.copy()
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "H3216 allocation",
        "target": "H3373_H3216_late_route_hot_gateup_repair",
        "rule": [
            "start from H3216 without H3371 late-down demotion",
            "preserve H3216 down allocation to avoid over-demoting route-hot late experts",
        ],
        "k_hist": {
            "gate": k_hist(allocation, "K_gate"),
            "up": k_hist(allocation, "K_up"),
            "down": k_hist(allocation, "K_down"),
        },
    }
    return allocation, manifest


def promote_late_hot_gateup(allocation: np.ndarray,
                            manifest: dict[str, Any],
                            route_threshold: float,
                            gate_k: int,
                            up_k: int) -> None:
    if route_threshold <= 0.0:
        return
    mask = (
        (allocation["layer"] >= 40) &
        (allocation["layer"] <= 42) &
        (allocation["route_mass"] >= route_threshold)
    )
    before = allocation[mask].copy()
    allocation["K_gate"][mask] = np.maximum(allocation["K_gate"][mask], gate_k)
    allocation["K_up"][mask] = np.maximum(allocation["K_up"][mask], up_k)
    update_gateup_fields(allocation)
    rows = []
    for row in before[np.argsort(before["route_mass"])[::-1]]:
        current = allocation[(allocation["layer"] == row["layer"]) & (allocation["expert"] == row["expert"])][0]
        rows.append({
            "layer": int(row["layer"]),
            "expert": int(row["expert"]),
            "rank": int(row["rank"]) if "rank" in allocation.dtype.names else 0,
            "route_mass": float(row["route_mass"]),
            "old_K_gate": int(row["K_gate"]),
            "new_K_gate": int(current["K_gate"]),
            "old_K_up": int(row["K_up"]),
            "new_K_up": int(current["K_up"]),
            "K_down": int(current["K_down"]),
        })
    if manifest.get("target") == "H3373_H3216_late_route_hot_gateup_repair":
        pass
    else:
        manifest["target"] = "H3372_H3371_late_route_hot_gateup_repair"
    manifest["rule"].append(
        f"promote L40-L42 route_mass>={route_threshold:g} gate/up to at least K{gate_k}/K{up_k}"
    )
    manifest["late_route_hot_gateup_promotions"] = {
        "route_threshold": route_threshold,
        "gate_k_floor": gate_k,
        "up_k_floor": up_k,
        "count": int(mask.sum()),
        "rows": rows,
    }
    manifest["k_hist"] = {
        "gate": k_hist(allocation, "K_gate"),
        "up": k_hist(allocation, "K_up"),
        "down": k_hist(allocation, "K_down"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--template-pack", type=Path, default=DEFAULT_TEMPLATE_PACK)
    parser.add_argument("--base-policy", choices=("h3371", "source"), default="h3371")
    parser.add_argument("--l40-k256-count", type=int, default=186)
    parser.add_argument("--late-hot-gateup-threshold", type=float, default=0.0)
    parser.add_argument("--late-hot-gate-k", type=int, default=2048)
    parser.add_argument("--late-hot-up-k", type=int, default=4096)
    args = parser.parse_args()

    source = np.load(args.input, allow_pickle=False)
    if source.shape != (LAYERS * EXPERTS,):
        raise SystemExit(f"unexpected allocation shape {source.shape}; expected {(LAYERS * EXPERTS,)}")
    required = {"layer", "expert", "K_gate", "K_up", "K_down"}
    missing = required.difference(source.dtype.names or ())
    if missing:
        raise SystemExit(f"allocation missing required fields: {sorted(missing)}")

    if args.base_policy == "source":
        allocation, manifest = build_source_base(source)
    else:
        allocation, manifest = build_h3371(source, args.l40_k256_count)
    promote_late_hot_gateup(allocation,
                            manifest,
                            args.late_hot_gateup_threshold,
                            args.late_hot_gate_k,
                            args.late_hot_up_k)
    before_routed = routed_payload_bytes(source)
    after_routed = routed_payload_bytes(allocation)
    extras = template_bytes(args.template_pack)
    total = after_routed + sum(extras.values())
    manifest["byte_model"] = {
        "record_model": "bitpacked indices + fp16 Kx8 codebook + 4096-byte down act-scale",
        "source_routed_bytes": before_routed,
        "target_routed_bytes": after_routed,
        "routed_saved_bytes": before_routed - after_routed,
        **extras,
        "target_total_bytes_with_template_extras": total,
        "target_total_decimal_gb_with_template_extras": total / 1e9,
        "target_total_gib_with_template_extras": total / (1024**3),
        "margin_to_52gb_decimal": 52_000_000_000 - total,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.base_policy == "source" and args.late_hot_gateup_threshold > 0.0:
        tag = "h3373_source_down_late_route_hot_gateup_repair"
    elif args.late_hot_gateup_threshold > 0.0:
        tag = "h3372_late_route_hot_gateup_repair"
    else:
        tag = "h3371_general_fit_no_overlay"
    npy_path = args.out_dir / f"ds4_52gb_allocation_{tag}.npy"
    json_path = args.out_dir / f"{tag}_manifest.json"
    if npy_path.exists() or json_path.exists():
        raise SystemExit(f"refusing to overwrite existing outputs in {args.out_dir}")
    np.save(npy_path, allocation)
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"allocation": os.fspath(npy_path), "manifest": os.fspath(json_path), **manifest["byte_model"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
