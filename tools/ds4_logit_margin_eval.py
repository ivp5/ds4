#!/usr/bin/env python3
"""Compare codec logits against a reference in reasoning-margin units.

This is a static auxiliary check for the codec promotion metric.  Weight
rel-L2 and activation-weighted errors are useful triage signals, but they do
not answer whether the codec corrupts the model's actual next-token decision.
For current D8F promotion, pair this with ds4_head_margin_sensitivity.py,
which scores head-projected linearized codec influence on fragile reasoning
tokens.  This tool compares already-materialized logits directly and reports
how much the codec erodes the reference winner's gap.

Inputs can be:
  - ds4 --dump-logits JSON files with a top-level "logits" array
  - .npy arrays shaped [vocab] or [positions, vocab]

The primary scalar is worst_gap_erosion_logits:

  (ref[target] - ref[best_competitor])
    - (candidate[target] - candidate[best_competitor])

By default the target is the reference argmax at each position and the
competitor set is the union of the reference/candidate top-K tokens.  For
answer-token tests, pass --target-token and optionally --competitor-token.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


NEG_INF = -float("inf")


def _as_float_array(values: Any) -> np.ndarray:
    if isinstance(values, list):
        cleaned = [NEG_INF if value is None else float(value) for value in values]
        return np.asarray(cleaned, dtype=np.float32)
    array = np.asarray(values, dtype=np.float32)
    array = np.where(np.isfinite(array), array, NEG_INF).astype(np.float32, copy=False)
    return array


def load_logits(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        logits = np.load(path)
    else:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if "logits" not in payload:
            raise ValueError(f"{path} does not contain a top-level logits array")
        logits = _as_float_array(payload["logits"])

    logits = _as_float_array(logits)
    if logits.ndim == 1:
        logits = logits.reshape(1, logits.shape[0])
    if logits.ndim != 2:
        raise ValueError(f"{path} must be [vocab] or [positions,vocab], got {logits.shape}")
    return logits


def finite_argmax(logits: np.ndarray) -> int:
    if not np.isfinite(logits).any():
        raise ValueError("all logits are non-finite")
    return int(np.nanargmax(logits))


def top_indices(logits: np.ndarray, top_k: int) -> np.ndarray:
    finite = np.where(np.isfinite(logits), logits, NEG_INF)
    keep = min(max(top_k, 1), finite.shape[0])
    if keep == finite.shape[0]:
        return np.argsort(finite)[::-1].astype(np.int64)
    unordered = np.argpartition(finite, -keep)[-keep:]
    ordered = unordered[np.argsort(finite[unordered])[::-1]]
    return ordered.astype(np.int64)


def parse_token_csv(text: str | None) -> list[int]:
    if not text:
        return []
    out: list[int] = []
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        out.append(int(part, 0))
    return out


def token_for_position(tokens: list[int], position: int) -> int | None:
    if not tokens:
        return None
    if len(tokens) == 1:
        return tokens[0]
    if position >= len(tokens):
        raise ValueError(
            f"got {len(tokens)} target tokens but need token for position {position}"
        )
    return tokens[position]


def softmax_kl(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref_finite = np.where(np.isfinite(reference), reference, NEG_INF).astype(np.float64)
    cand_finite = np.where(np.isfinite(candidate), candidate, NEG_INF).astype(np.float64)
    ref_max = float(np.max(ref_finite))
    cand_max = float(np.max(cand_finite))
    if not math.isfinite(ref_max) or not math.isfinite(cand_max):
        return float("nan")
    ref_exp = np.exp(ref_finite - ref_max)
    cand_exp = np.exp(cand_finite - cand_max)
    ref_sum = float(ref_exp.sum())
    cand_sum = float(cand_exp.sum())
    if ref_sum <= 0.0 or cand_sum <= 0.0:
        return float("nan")
    ref_prob = ref_exp / ref_sum
    ref_logprob = ref_finite - ref_max - math.log(ref_sum)
    cand_logprob = cand_finite - cand_max - math.log(cand_sum)
    return float(np.sum(ref_prob * (ref_logprob - cand_logprob)))


def competitor_pool(
    reference: np.ndarray,
    candidate: np.ndarray,
    target_token: int,
    top_k: int,
    explicit_competitors: list[int],
) -> np.ndarray:
    if explicit_competitors:
        pool = np.asarray(explicit_competitors, dtype=np.int64)
    else:
        pool = np.unique(
            np.concatenate(
                [
                    top_indices(reference, top_k),
                    top_indices(candidate, top_k),
                ]
            )
        )
    pool = pool[(pool >= 0) & (pool < reference.shape[0]) & (pool != target_token)]
    if pool.size == 0:
        raise ValueError("empty competitor pool")
    return pool


def position_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    position: int,
    top_k: int,
    margin_budget: float,
    target_token: int | None,
    explicit_competitors: list[int],
) -> dict[str, Any]:
    ref_argmax = finite_argmax(reference)
    cand_argmax = finite_argmax(candidate)
    target = ref_argmax if target_token is None else target_token
    if target < 0 or target >= reference.shape[0]:
        raise ValueError(f"target token {target} outside vocab {reference.shape[0]}")

    pool = competitor_pool(reference, candidate, target, top_k, explicit_competitors)
    ref_gaps = reference[target] - reference[pool]
    cand_gaps = candidate[target] - candidate[pool]
    erosions = ref_gaps - cand_gaps
    worst_index = int(np.nanargmax(erosions))
    competitor = int(pool[worst_index])

    ref_best_competitor = int(pool[int(np.nanargmax(reference[pool]))])
    cand_best_competitor = int(pool[int(np.nanargmax(candidate[pool]))])
    ref_margin = float(reference[target] - reference[ref_best_competitor])
    cand_margin = float(candidate[target] - candidate[cand_best_competitor])
    worst_erosion = float(erosions[worst_index])

    union = np.unique(np.concatenate([pool, np.asarray([target], dtype=np.int64)]))
    top_abs_error = float(np.max(np.abs(candidate[union] - reference[union])))
    target_error = float(candidate[target] - reference[target])
    competitor_error = float(candidate[competitor] - reference[competitor])

    return {
        "position": position,
        "target_token": int(target),
        "reference_argmax": int(ref_argmax),
        "candidate_argmax": int(cand_argmax),
        "argmax_agree": bool(ref_argmax == cand_argmax),
        "competitor_token": competitor,
        "reference_margin_logits": ref_margin,
        "candidate_margin_logits": cand_margin,
        "worst_gap_erosion_logits": worst_erosion,
        "erosion_over_budget": worst_erosion / margin_budget if margin_budget > 0 else float("inf"),
        "target_logit_error": target_error,
        "competitor_logit_error": competitor_error,
        "top_pool_abs_logit_error": top_abs_error,
        "kl_ref_to_candidate": softmax_kl(reference, candidate),
        "status": "PASS" if cand_margin > 0.0 and worst_erosion <= margin_budget else "FAIL",
    }


def summarize(rows: list[dict[str, Any]], margin_budget: float, logit_error_budget: float) -> dict[str, Any]:
    erosions = np.asarray([row["worst_gap_erosion_logits"] for row in rows], dtype=np.float64)
    abs_errors = np.asarray([row["top_pool_abs_logit_error"] for row in rows], dtype=np.float64)
    margins = np.asarray([row["reference_margin_logits"] for row in rows], dtype=np.float64)
    flips = sum(1 for row in rows if not row["argmax_agree"])
    margin_failed = sum(1 for row in rows if row["status"] != "PASS")
    logit_over = sum(1 for row in rows if row["top_pool_abs_logit_error"] > logit_error_budget)
    return {
        "positions": len(rows),
        "margin_budget_logits": margin_budget,
        "logit_error_budget": logit_error_budget,
        "failures": margin_failed,
        "margin_failures": margin_failed,
        "logit_error_over_budget": logit_over,
        "argmax_flips": flips,
        "max_gap_erosion_logits": float(np.max(erosions)) if rows else float("nan"),
        "mean_gap_erosion_logits": float(np.mean(erosions)) if rows else float("nan"),
        "p95_gap_erosion_logits": float(np.percentile(erosions, 95)) if rows else float("nan"),
        "max_top_pool_abs_logit_error": float(np.max(abs_errors)) if rows else float("nan"),
        "p95_top_pool_abs_logit_error": float(np.percentile(abs_errors, 95)) if rows else float("nan"),
        "min_reference_margin_logits": float(np.min(margins)) if rows else float("nan"),
        "status": "PASS" if margin_failed == 0 and logit_over == 0 else "FAIL",
    }


def summarize_perturbation(reference: np.ndarray, candidate: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    mask = np.isfinite(reference) & np.isfinite(candidate)
    if not bool(np.any(mask)):
        return {
            "finite_logits": 0,
            "logit_delta_l2": float("nan"),
            "logit_delta_rms": float("nan"),
            "logit_delta_max_abs": float("nan"),
            "theta_max_flipped_margin_logits": None,
            "theta_min_unflipped_margin_logits": None,
            "flipped_reference_margin_logits": [],
        }
    delta = candidate[mask].astype(np.float64) - reference[mask].astype(np.float64)
    flipped = [float(row["reference_margin_logits"]) for row in rows if not row["argmax_agree"]]
    unflipped = [float(row["reference_margin_logits"]) for row in rows if row["argmax_agree"]]
    return {
        "finite_logits": int(delta.size),
        "logit_delta_l2": float(np.linalg.norm(delta)),
        "logit_delta_rms": float(np.sqrt(np.mean(delta * delta))),
        "logit_delta_max_abs": float(np.max(np.abs(delta))),
        "theta_max_flipped_margin_logits": max(flipped) if flipped else None,
        "theta_min_unflipped_margin_logits": min(unflipped) if unflipped else None,
        "flipped_reference_margin_logits": flipped,
    }


def emit_text(rows: list[dict[str, Any]], summary: dict[str, Any], limit: int) -> None:
    print(
        "pos target ref_arg cand_arg ref_margin cand_margin erosion budget status "
        "target_err comp_err kl"
    )
    for row in rows[:limit]:
        print(
            f"{row['position']:>3} {row['target_token']:>6} "
            f"{row['reference_argmax']:>7} {row['candidate_argmax']:>8} "
            f"{row['reference_margin_logits']:>10.4f} "
            f"{row['candidate_margin_logits']:>11.4f} "
            f"{row['worst_gap_erosion_logits']:>7.4f} "
            f"{row['erosion_over_budget']:>6.2f} "
            f"{row['status']:>4} "
            f"{row['target_logit_error']:>10.4f} "
            f"{row['competitor_logit_error']:>8.4f} "
            f"{row['kl_ref_to_candidate']:>8.4f}"
        )
    if len(rows) > limit:
        print(f"... {len(rows) - limit} more positions omitted")
    print("summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6g}")
        else:
            print(f"  {key}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--margin-budget", type=float, default=0.5)
    parser.add_argument("--logit-error-budget", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--skip-positions", type=int, default=0)
    parser.add_argument("--target-token", help="single token id or comma list, one per position")
    parser.add_argument("--competitor-token", help="comma list of explicit competitor token ids")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--print-limit", type=int, default=32)
    args = parser.parse_args()

    reference = load_logits(args.reference)
    candidate = load_logits(args.candidate)
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: reference {reference.shape}, candidate {candidate.shape}")
    if args.skip_positions < 0 or args.skip_positions >= reference.shape[0]:
        raise ValueError("--skip-positions must leave at least one position")

    targets = parse_token_csv(args.target_token)
    competitors = parse_token_csv(args.competitor_token)
    rows: list[dict[str, Any]] = []
    for position in range(args.skip_positions, reference.shape[0]):
        rows.append(
            position_metrics(
                reference[position],
                candidate[position],
                position,
                args.top_k,
                args.margin_budget,
                token_for_position(targets, position - args.skip_positions),
                competitors,
            )
        )
    summary = summarize(rows, args.margin_budget, args.logit_error_budget)
    perturbation = summarize_perturbation(reference[args.skip_positions :], candidate[args.skip_positions :], rows)
    payload = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "top_k": args.top_k,
        "skip_positions": args.skip_positions,
        "summary": summary,
        "perturbation": perturbation,
        "positions": rows,
    }
    emit_text(rows, summary, args.print_limit)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
