#!/usr/bin/env python3
"""Correlate logit perturbation magnitude with observed flip threshold.

Input files are JSON outputs from ds4_logit_margin_eval.py.  Older files that
lack a "perturbation" block are rehydrated from their reference/candidate paths
when those paths are still available.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from ds4_logit_margin_eval import load_logits, summarize_perturbation  # noqa: E402


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x0 = x - float(np.mean(x))
    y0 = y - float(np.mean(y))
    denom = float(np.linalg.norm(x0) * np.linalg.norm(y0))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(x0, y0) / denom)


def load_eval(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    perturbation = payload.get("perturbation")
    if not isinstance(perturbation, dict):
        reference = load_logits(Path(payload["reference"]))
        candidate = load_logits(Path(payload["candidate"]))
        skip = int(payload.get("skip_positions", 0))
        perturbation = summarize_perturbation(reference[skip:], candidate[skip:], payload.get("positions", []))
    theta = perturbation.get("theta_max_flipped_margin_logits")
    return {
        "path": str(path),
        "candidate": payload.get("candidate"),
        "positions": int(payload.get("summary", {}).get("positions", len(payload.get("positions", [])))),
        "argmax_flips": int(payload.get("summary", {}).get("argmax_flips", 0)),
        "logit_delta_l2": float(perturbation.get("logit_delta_l2", math.nan)),
        "logit_delta_rms": float(perturbation.get("logit_delta_rms", math.nan)),
        "logit_delta_max_abs": float(perturbation.get("logit_delta_max_abs", math.nan)),
        "theta_max_flipped_margin_logits": None if theta is None else float(theta),
        "p95_gap_erosion_logits": float(payload.get("summary", {}).get("p95_gap_erosion_logits", math.nan)),
        "min_reference_margin_logits": float(payload.get("summary", {}).get("min_reference_margin_logits", math.nan)),
        "status": payload.get("summary", {}).get("status"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_json", nargs="+", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    rows = [load_eval(path) for path in args.eval_json]
    theta_rows = [
        row
        for row in rows
        if row["theta_max_flipped_margin_logits"] is not None and math.isfinite(row["logit_delta_l2"])
    ]
    norms = np.asarray([row["logit_delta_l2"] for row in theta_rows], dtype=np.float64)
    thetas = np.asarray([row["theta_max_flipped_margin_logits"] for row in theta_rows], dtype=np.float64)
    summary = {
        "inputs": len(rows),
        "inputs_with_flips": len(theta_rows),
        "corr_l2_theta": pearson(norms, thetas),
        "norm_min": float(np.min(norms)) if norms.size else math.nan,
        "norm_max": float(np.max(norms)) if norms.size else math.nan,
        "theta_min": float(np.min(thetas)) if thetas.size else math.nan,
        "theta_max": float(np.max(thetas)) if thetas.size else math.nan,
        "interpretation": "theta is observed max flipped reference margin; no-flip candidates are excluded from corr",
    }
    payload = {"summary": summary, "rows": rows}
    print(json.dumps(payload, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
