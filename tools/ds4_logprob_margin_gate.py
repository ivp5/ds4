#!/usr/bin/env python3
"""Score DS4 codec candidates from --dump-logprobs margin traces.

This supplies the margin side of the codec-selection inequality:

  P95(codec logit error) < P5(reasoning margin)

The AIME scoreboard is post-hoc confirmation, not the selection target.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_steps(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"{path} has no non-empty steps array")
    return steps


def selected_text(step: dict[str, Any]) -> str:
    selected = step.get("selected", {})
    if isinstance(selected, dict):
        return str(selected.get("text", ""))
    return str(selected)


def margin_entropy(step: dict[str, Any]) -> tuple[float, float]:
    top = step.get("top_logprobs", [])
    if not isinstance(top, list) or not top:
        return math.nan, math.nan
    logits = np.asarray([float(item["logit"]) for item in top if "logit" in item], dtype=np.float64)
    logprobs = np.asarray([float(item["logprob"]) for item in top if "logprob" in item], dtype=np.float64)
    margin = float(logits[0] - logits[1]) if logits.size > 1 else 99.0
    if logprobs.size == 0:
        return margin, math.nan
    probs = np.exp(logprobs - float(np.max(logprobs)))
    probs /= float(np.sum(probs))
    entropy = float(-np.sum(probs * np.log(probs + 1.0e-12)))
    return margin, entropy


def analyze(steps: list[dict[str, Any]], repeat_window: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    window: deque[str] = deque(maxlen=repeat_window)
    for fallback_step, step in enumerate(steps):
        margin, entropy = margin_entropy(step)
        text = selected_text(step)
        repeat = text in window
        window.append(text)
        rows.append(
            {
                "step": int(step.get("step", fallback_step)),
                "selected_text": text,
                "entropy": entropy,
                "margin_logits": margin,
                "repeat": repeat,
            }
        )
    first_margin = rows[0]["margin_logits"]
    bifurcation_step = None
    if math.isfinite(first_margin):
        for row in rows[1:]:
            margin = row["margin_logits"]
            if math.isfinite(margin) and margin < first_margin * 0.4 and margin < 2.0:
                bifurcation_step = row["step"]
                break
    reasoning = rows[1:] if len(rows) > 1 else rows
    reasoning_margins = np.asarray(
        [row["margin_logits"] for row in reasoning if math.isfinite(row["margin_logits"])],
        dtype=np.float64,
    )
    repeat_count = sum(1 for row in rows if row["repeat"])
    summary = {
        "steps": len(rows),
        "first_margin_logits": float(first_margin),
        "bifurcation_step": bifurcation_step,
        "reasoning_mean_margin_logits": float(np.mean(reasoning_margins)) if reasoning_margins.size else math.nan,
        "reasoning_min_margin_logits": float(np.min(reasoning_margins)) if reasoning_margins.size else math.nan,
        "reasoning_p5_margin_logits": float(np.percentile(reasoning_margins, 5)) if reasoning_margins.size else math.nan,
        "reasoning_p10_margin_logits": float(np.percentile(reasoning_margins, 10)) if reasoning_margins.size else math.nan,
        "repeat_count": int(repeat_count),
    }
    return rows, summary


def status(summary: dict[str, Any], mean_threshold: float, repeat_threshold: int) -> str:
    if summary["reasoning_mean_margin_logits"] < mean_threshold and summary["repeat_count"] >= repeat_threshold:
        return "ECHO"
    return "COHERENT"


def print_report(rows: list[dict[str, Any]], summary: dict[str, Any], verdict: str, limit: int) -> None:
    print("step selected_text entropy margin_logits repeat")
    for row in rows[:limit]:
        entropy = row["entropy"]
        margin = row["margin_logits"]
        print(
            f"{row['step']:>4} {row['selected_text']!r:<16.16} "
            f"{entropy:>7.2f} {margin:>13.2f} "
            f"{'REP' if row['repeat'] else ''}"
        )
    if len(rows) > limit:
        print(f"... {len(rows) - limit} more steps omitted")
    print("summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6g}")
        else:
            print(f"  {key}: {value}")
    print(f"  coherence_status: {verdict}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logprobs_json", type=Path)
    parser.add_argument("--mean-threshold", type=float, default=1.5)
    parser.add_argument("--repeat-threshold", type=int, default=3)
    parser.add_argument("--repeat-window", type=int, default=6)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--print-limit", type=int, default=16)
    args = parser.parse_args()
    rows, summary = analyze(load_steps(args.logprobs_json), args.repeat_window)
    verdict = status(summary, args.mean_threshold, args.repeat_threshold)
    print_report(rows, summary, verdict, args.print_limit)
    if args.json_out:
        payload = {
            "source": str(args.logprobs_json),
            "mean_threshold": args.mean_threshold,
            "repeat_threshold": args.repeat_threshold,
            "summary": {**summary, "coherence_status": verdict},
            "steps": rows,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
    return 2 if verdict == "ECHO" else 0


if __name__ == "__main__":
    raise SystemExit(main())
