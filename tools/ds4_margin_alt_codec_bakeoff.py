#!/usr/bin/env python3
"""Bake off alternate expert codecs on the ASCII margin span.

The score is not rel-L2 and not activation-aware E.  It is the same local
selector used by ds4_head_margin_sensitivity.py:

  P95(|head_ASCII(selected-worst) · J_norm · δffn|) / P5(reasoning margin)

Each codec reports its own charged storage: index bits, scale bytes, codebook
bytes, and any global constants.  Deterministic lattices get no free codebook;
per-group scales are charged when the quantizer uses them.
"""

from __future__ import annotations

import argparse
import json
import math
import mmap
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from ds4_d8f_weight_compare import decode_d8f, load_source, read_record  # noqa: E402
from ds4_head_margin_sensitivity import (  # noqa: E402
    DEFAULT_ACTS,
    MODEL_DIR,
    load_acts,
    load_head_rows,
    load_index,
    load_numeric,
    load_residual,
    load_many_trace_margin_directions,
    norm_jvp,
    projection_delta,
    router_select,
)


Quantizer = Callable[[np.ndarray, np.random.Generator], tuple[np.ndarray, dict[str, Any]]]


def bits_for_k(k: int) -> int:
    return int(math.ceil(math.log2(k)))


def per_group_scale(weight: np.ndarray, group: int) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = weight.shape
    if cols % group != 0:
        raise ValueError(f"cols={cols} not divisible by group={group}")
    grouped = weight.reshape(rows, cols // group, group)
    scales = np.maximum(np.max(np.abs(grouped), axis=2, keepdims=True), 1.0e-9).astype(np.float32)
    return scales.repeat(group, axis=2).reshape(rows, cols), scales.reshape(rows, cols // group)


def charge_payload(
    name: str,
    weight: np.ndarray,
    index_bits_per_block: float,
    block_dim: int,
    scale_count: int,
    codebook_values: int = 0,
    global_values: int = 0,
) -> dict[str, Any]:
    n_weights = int(weight.size)
    blocks = int(math.ceil(n_weights / block_dim))
    index_bits = float(blocks * index_bits_per_block)
    scale_bytes = int(scale_count * 2)
    codebook_bytes = int(codebook_values * 2)
    global_bytes = int(global_values * 2)
    total_bits = index_bits + 8.0 * float(scale_bytes + codebook_bytes + global_bytes)
    return {
        "codec": name,
        "weights": n_weights,
        "block_dim": block_dim,
        "index_bits_per_block": index_bits_per_block,
        "index_bits": index_bits,
        "scale_bytes_fp16": scale_bytes,
        "codebook_bytes_fp16": codebook_bytes,
        "global_bytes_fp16": global_bytes,
        "charged_bytes": int(math.ceil(total_bits / 8.0)),
        "charged_bpw": total_bits / float(n_weights),
    }


def scalar_levels(name: str, levels: list[float], group: int) -> Quantizer:
    level_array = np.asarray(levels, dtype=np.float32)
    max_abs = float(np.max(np.abs(level_array)))
    fixed_bits = bits_for_k(len(levels))

    def quantize(weight: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
        del rng
        scale, compact = per_group_scale(weight, group)
        normalized = weight / (scale / max_abs)
        indices = np.argmin(np.abs(normalized[..., None] - level_array.reshape(1, 1, -1)), axis=2)
        quantized = level_array[indices].astype(np.float32) * (scale / max_abs)
        charge = charge_payload(name, weight, fixed_bits, 1, int(compact.size), codebook_values=len(levels))
        charge["levels"] = levels
        charge["note"] = "fixed-length symbols; ternary is charged as 2 bits/weight, not ideal entropy"
        return quantized.astype(np.float32), charge

    return quantize


def make_e8_codebook(k: int) -> np.ndarray:
    import itertools

    points: set[tuple[float, ...]] = set()
    for vector in itertools.product(range(-2, 3), repeat=8):
        if sum(vector) % 2 == 0 and sum(value * value for value in vector) <= 8:
            points.add(tuple(float(value) for value in vector))
    for vector in itertools.product((-1.5, -0.5, 0.5, 1.5), repeat=8):
        total = sum(vector)
        if abs(total - round(total)) < 1.0e-9 and round(total) % 2 == 0 and sum(value * value for value in vector) <= 8:
            points.add(tuple(float(value) for value in vector))
    ordered = sorted(points, key=lambda values: sum(value * value for value in values))
    if len(ordered) < k:
        raise ValueError(f"E8 generator produced {len(ordered)} points, need {k}")
    return np.asarray(ordered[:k], dtype=np.float32)


def nearest_codebook(points: np.ndarray, codebook: np.ndarray, chunk: int, status: str) -> np.ndarray:
    out = np.empty(points.shape[0], dtype=np.int32)
    cb_norm = np.sum(codebook * codebook, axis=1).reshape(1, -1)
    for start in range(0, points.shape[0], chunk):
        end = min(start + chunk, points.shape[0])
        dots = points[start:end] @ codebook.T
        dist = np.sum(points[start:end] * points[start:end], axis=1, keepdims=True) + cb_norm - 2.0 * dots
        out[start:end] = np.argmin(dist, axis=1)
        if start == 0 or end == points.shape[0]:
            print(f"[alt_codec] {status}: assigned {end}/{points.shape[0]}", file=sys.stderr, flush=True)
    return out


def e8_quantizer(k: int, group: int, betas: list[float]) -> Quantizer:
    base = make_e8_codebook(k)
    bits = bits_for_k(k)
    name = f"e8_k{k}"

    def quantize(weight: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
        del rng
        scale, compact = per_group_scale(weight, group)
        rows, cols = weight.shape
        blocks = (weight / scale).reshape(-1, 8).astype(np.float32)
        sample = blocks[np.linspace(0, blocks.shape[0] - 1, min(32768, blocks.shape[0])).astype(np.int64)]
        best_beta = betas[0]
        best_mse = float("inf")
        for beta in betas:
            codebook = base * np.float32(beta)
            indices = nearest_codebook(sample, codebook, 8192, f"{name} beta={beta:g} sample")
            recon = codebook[indices]
            mse = float(np.mean((sample - recon) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_beta = beta
        codebook = base * np.float32(best_beta)
        indices = nearest_codebook(blocks, codebook, 8192, f"{name} full")
        quantized = (codebook[indices].reshape(rows, cols) * scale).astype(np.float32)
        charge = charge_payload(name, weight, bits, 8, int(compact.size), global_values=1)
        charge["deterministic_codebook"] = "E8 lowest-norm point order; no codebook bytes charged"
        charge["stored_global_beta_fp16"] = best_beta
        charge["sample_mse"] = best_mse
        return quantized, charge

    return quantize


def lloyd_codebook(points: np.ndarray, k: int, iters: int, rng: np.random.Generator, name: str) -> np.ndarray:
    if points.shape[0] < k:
        raise ValueError(f"{name}: only {points.shape[0]} points for k={k}")
    initial = rng.choice(points.shape[0], size=k, replace=False)
    codebook = points[initial].astype(np.float32).copy()
    for iteration in range(iters):
        indices = nearest_codebook(points, codebook, 8192, f"{name} lloyd {iteration + 1}/{iters}")
        sums = np.zeros_like(codebook)
        counts = np.bincount(indices, minlength=k).astype(np.float32)
        np.add.at(sums, indices, points)
        live = counts > 0
        codebook[live] = sums[live] / counts[live, None]
    return codebook


def vq_d8_quantizer(k: int, group: int, sample_blocks: int, iters: int) -> Quantizer:
    bits = bits_for_k(k)
    name = f"vq_d8_k{k}"

    def quantize(weight: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
        scale, compact = per_group_scale(weight, group)
        rows, cols = weight.shape
        blocks = (weight / scale).reshape(-1, 8).astype(np.float32)
        sample_count = min(sample_blocks, blocks.shape[0])
        sample_idx = rng.choice(blocks.shape[0], size=sample_count, replace=False)
        codebook = lloyd_codebook(blocks[sample_idx], k, iters, rng, name)
        indices = nearest_codebook(blocks, codebook, 8192, f"{name} full")
        quantized = (codebook[indices].reshape(rows, cols) * scale).astype(np.float32)
        charge = charge_payload(name, weight, bits, 8, int(compact.size), codebook_values=k * 8)
        charge["sample_blocks"] = sample_count
        charge["lloyd_iters"] = iters
        return quantized, charge

    return quantize


def aqlm_m2_k32_quantizer(group: int, sample_blocks: int, iters: int) -> Quantizer:
    name = "aqlm_m2_k32"
    k = 32

    def quantize(weight: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
        scale, compact = per_group_scale(weight, group)
        rows, cols = weight.shape
        blocks = (weight / scale).reshape(-1, 8).astype(np.float32)
        sample_count = min(sample_blocks, blocks.shape[0])
        sample = blocks[rng.choice(blocks.shape[0], size=sample_count, replace=False)]
        c1 = lloyd_codebook(sample, k, max(2, iters // 2), rng, f"{name}.c1")
        r1 = sample - c1[nearest_codebook(sample, c1, 8192, f"{name}.rvq_init1")]
        c2 = lloyd_codebook(r1, k, max(2, iters // 2), rng, f"{name}.c2")
        row_ids = np.arange(sample.shape[0])
        for iteration in range(iters):
            combos = (c1[:, None, :] + c2[None, :, :]).reshape(k * k, 8)
            assign = nearest_codebook(sample, combos, 8192, f"{name} joint {iteration + 1}/{iters}")
            i1 = assign // k
            i2 = assign % k
            design = np.zeros((sample.shape[0], 2 * k), dtype=np.float32)
            design[row_ids, i1] = 1.0
            design[row_ids, k + i2] = 1.0
            solved = np.linalg.lstsq(design, sample, rcond=1e-4)[0].astype(np.float32)
            c1 = solved[:k]
            c2 = solved[k:]
        combos = (c1[:, None, :] + c2[None, :, :]).reshape(k * k, 8)
        indices = nearest_codebook(blocks, combos, 8192, f"{name} full")
        quantized = (combos[indices].reshape(rows, cols) * scale).astype(np.float32)
        charge = charge_payload(name, weight, 10, 8, int(compact.size), codebook_values=2 * k * 8)
        charge["sample_blocks"] = sample_count
        charge["joint_iters"] = iters
        return quantized, charge

    return quantize


def fwht_rows(values: np.ndarray) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    width = out.shape[1]
    step = 1
    scale = float(width) ** -0.5
    while step < width:
        view = out.reshape(out.shape[0], -1, 2 * step)
        left = view[:, :, :step].copy()
        right = view[:, :, step : 2 * step].copy()
        view[:, :, :step] = left + right
        view[:, :, step : 2 * step] = left - right
        step *= 2
    return out * scale


def qtip_quantizer(r_bits: int, group: int, state_bits: int = 6) -> Quantizer:
    name = f"qtip_r{r_bits}"
    windows = 1 << (state_bits + r_bits)
    states = 1 << state_bits
    branch = 1 << r_bits

    def encode(transformed: np.ndarray, codebook: np.ndarray) -> np.ndarray:
        rows, cols = transformed.shape
        pred_state = np.repeat(np.arange(states), branch)
        branch_bits = np.tile(np.arange(branch), states)
        window = (pred_state << r_bits) | branch_bits
        next_state = window & (states - 1)
        order = np.argsort(next_state, kind="stable")
        pred_state = pred_state[order].reshape(states, branch)
        pred_window = window[order].reshape(states, branch)
        cb_window = codebook[pred_window]
        cost = np.full((rows, states), 1.0e30, dtype=np.float32)
        cost[:, 0] = 0.0
        edges = np.zeros((cols, rows, states), dtype=np.uint8)
        for col in range(cols):
            cand = cost[:, pred_state] + (transformed[:, col].reshape(rows, 1, 1) - cb_window.reshape(1, states, branch)) ** 2
            edges[col] = np.argmin(cand, axis=2).astype(np.uint8)
            cost = np.min(cand, axis=2)
        state = np.argmin(cost, axis=1)
        out = np.empty((rows, cols), dtype=np.int32)
        row_ids = np.arange(rows)
        for col in range(cols - 1, -1, -1):
            edge = edges[col, row_ids, state]
            out[:, col] = pred_window[state, edge]
            state = pred_state[state, edge]
        return out

    def quantize(weight: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
        del rng
        rows, cols = weight.shape
        transformed = fwht_rows(weight)
        scale = np.maximum(np.max(np.abs(transformed), axis=1, keepdims=True) / 2.5, 1.0e-9).astype(np.float32)
        normalized = (transformed / scale).reshape(-1)
        quantiles = np.linspace(0.0, 1.0, windows + 2, dtype=np.float64)[1:-1]
        codebook = np.quantile(normalized, quantiles).astype(np.float32)
        codes = encode((transformed / scale).astype(np.float32), codebook)
        quantized = fwht_rows(codebook[codes] * scale).astype(np.float32)
        charge = charge_payload(name, weight, r_bits, 1, rows, codebook_values=windows)
        charge["state_bits"] = state_bits
        charge["fixed_start_state"] = 0
        charge["codebook_fit"] = "scalar quantiles over FWHT(row)/row_scale; no finetune"
        return quantized.reshape(rows, cols), charge

    return quantize


def d8f_quantizer(path: Path, layer: int, projection: str, expert: int) -> Quantizer:
    name = "current_d8f"

    def quantize(weight: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
        del weight, rng
        with path.open("rb") as handle:
            mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                record = read_record(mapped, projection, expert)
                decoded = decode_d8f(mapped, record, projection).astype(np.float32)
            finally:
                mapped.close()
        charged_bytes = int(record["codebook_bytes"]) + int(record["index_bytes"]) + int(record["scale_bytes"]) + 64
        charge = {
            "codec": name,
            "weights": int(decoded.size),
            "record": record,
            "charged_bytes": charged_bytes,
            "charged_bpw": charged_bytes * 8.0 / float(decoded.size),
            "note": "actual D8F record bytes: codebook + index + scale + 64-byte record header",
        }
        return decoded, charge

    return quantize


def codec_registry(args: argparse.Namespace) -> dict[str, Quantizer]:
    betas = [float(value) for value in args.e8_betas.split(",") if value]
    registry: dict[str, Quantizer] = {
        "scalar_binary": scalar_levels("scalar_binary", [-1.0, 1.0], args.group),
        "scalar_ternary": scalar_levels("scalar_ternary", [-1.0, 0.0, 1.0], args.group),
        "e8_k256": e8_quantizer(256, args.group, betas),
        "e8_k1024": e8_quantizer(1024, args.group, betas),
        "vq_d8_k256": vq_d8_quantizer(256, args.group, args.sample_blocks, args.lloyd_iters),
        "vq_d8_k1024": vq_d8_quantizer(1024, args.group, args.sample_blocks, args.lloyd_iters),
        "aqlm_m2_k32": aqlm_m2_k32_quantizer(args.group, args.sample_blocks, args.aqlm_iters),
        "qtip_r1": qtip_quantizer(1, args.group),
        "qtip_r2": qtip_quantizer(2, args.group),
    }
    if args.d8f:
        registry["current_d8f"] = d8f_quantizer(args.d8f, args.layer, args.projection, args.expert)
    return registry


def score_candidate(
    args: argparse.Namespace,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    candidate: np.ndarray,
    acts: np.ndarray,
    residual_exact: np.ndarray | None,
    selected: np.ndarray,
    weights: np.ndarray,
    trace_indices: np.ndarray,
    norm_weight: np.ndarray,
    margin_dirs: np.ndarray,
    p5_margin: float,
    trace_row_mode: str,
) -> dict[str, Any]:
    token_rows_all, slots_all = np.where(selected == args.expert)
    token_rows = token_rows_all
    slots = slots_all
    dir_rows: np.ndarray | None = None
    if trace_row_mode == "aligned":
        trace_index_to_dir = {int(row): offset for offset, row in enumerate(trace_indices.tolist())}
        keep = np.asarray([int(row) in trace_index_to_dir for row in token_rows_all], dtype=bool)
        token_rows = token_rows_all[keep]
        slots = slots_all[keep]
        dir_rows = np.asarray([trace_index_to_dir[int(row)] for row in token_rows.tolist()], dtype=np.int64)
    if token_rows.size == 0:
        return {"status": "unselected_fragile", "selected_tokens": 0, "selected_tokens_total": int(token_rows_all.size)}
    gate_q = candidate if args.projection == "gate" else gate
    up_q = candidate if args.projection == "up" else up
    down_q = candidate if args.projection == "down" else down
    x = acts[token_rows]
    clean, delta = projection_delta(args.projection, x, gate, up, down, gate_q, up_q, down_q)
    delta *= weights[token_rows, slots][:, None]
    residual_rows = residual_exact[token_rows] if residual_exact is not None else None
    residual = residual_rows if residual_rows is not None else x + clean
    delta_norm = norm_jvp(residual, delta, norm_weight, exact=not args.simple_rms)
    active_margin_dirs = margin_dirs[dir_rows] if dir_rows is not None else margin_dirs
    if trace_row_mode == "aligned":
        errors = np.sum(delta_norm * active_margin_dirs, axis=1)
    else:
        errors = delta_norm @ active_margin_dirs.T
    abs_errors = np.abs(errors).reshape(-1)
    p95 = float(np.percentile(abs_errors, 95))
    return {
        "status": "scored",
        "selected_tokens": int(token_rows.size),
        "selected_tokens_total": int(token_rows_all.size),
        "p95_margin_logit_error": p95,
        "p99_margin_logit_error": float(np.percentile(abs_errors, 99)),
        "mean_abs_margin_logit_error": float(np.mean(abs_errors)),
        "p95_error_lt_p5_margin": bool(p95 < p5_margin),
        "noise_margin_ratio": p95 / max(p5_margin, 1.0e-9),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--expert", required=True, type=int)
    parser.add_argument("--projection", required=True, choices=("gate", "up", "down"))
    parser.add_argument("--logprobs-json", required=True, type=Path, nargs="+")
    parser.add_argument("--d8f", type=Path)
    parser.add_argument("--acts", type=Path, default=DEFAULT_ACTS)
    parser.add_argument("--ffn-in-bin", type=Path)
    parser.add_argument("--hc-dump", type=Path)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--codecs", default="current_d8f,scalar_binary,scalar_ternary,e8_k256,e8_k1024,vq_d8_k256,aqlm_m2_k32,qtip_r1")
    parser.add_argument("--margin-token-filter", choices=("none", "ascii", "latin"), default="ascii")
    parser.add_argument("--margin-threshold", choices=("raw", "filtered", "min"), default="min")
    parser.add_argument("--fragile-margin-max", type=float, default=0.5)
    parser.add_argument("--trace-row-mode", choices=("auto", "aligned", "cross"), default="auto")
    parser.add_argument("--simple-rms", action="store_true")
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--sample-blocks", type=int, default=32768)
    parser.add_argument("--lloyd-iters", type=int, default=4)
    parser.add_argument("--aqlm-iters", type=int, default=4)
    parser.add_argument("--e8-betas", default="0.125,0.1875,0.25,0.375,0.5,0.75,1.0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    index = load_index(MODEL_DIR)
    acts = load_acts(args)
    residual_exact = load_residual(args, acts.shape[0])
    selected, route_weights = router_select(MODEL_DIR, index, args.layer, acts)
    token_pool, pair_local, trace_indices, p5_margin, raw_summary, filtered_summary = load_many_trace_margin_directions(
        args.logprobs_json,
        20,
        args.margin_token_filter,
        args.margin_threshold,
        None if args.fragile_margin_max < 0.0 else args.fragile_margin_max,
    )
    if args.trace_row_mode == "aligned":
        trace_row_mode = "aligned"
    elif args.trace_row_mode == "cross":
        trace_row_mode = "cross"
    else:
        can_align = (
            len(args.logprobs_json) == 1
            and bool(args.ffn_in_bin or args.hc_dump)
            and bool(trace_indices.size)
            and int(np.max(trace_indices)) < acts.shape[0]
        )
        trace_row_mode = "aligned" if can_align else "cross"
    head_rows = load_head_rows(MODEL_DIR, index, token_pool)
    margin_dirs = head_rows[pair_local[:, 0]] - head_rows[pair_local[:, 1]]
    norm_weight = load_numeric(MODEL_DIR, index, "norm.weight").reshape(-1)
    gate = load_source(args.layer, "gate", args.expert).astype(np.float32)
    up = load_source(args.layer, "up", args.expert).astype(np.float32)
    down = load_source(args.layer, "down", args.expert).astype(np.float32)
    source = {"gate": gate, "up": up, "down": down}[args.projection]

    registry = codec_registry(args)
    codec_names = [name.strip() for name in args.codecs.split(",") if name.strip()]
    rows: list[dict[str, Any]] = []
    for ordinal, name in enumerate(codec_names, start=1):
        if name not in registry:
            rows.append({"codec": name, "status": "unknown_codec"})
            continue
        print(f"[alt_codec] {ordinal}/{len(codec_names)} {name}: quantizing", file=sys.stderr, flush=True)
        codec_started = time.perf_counter()
        try:
            candidate, charge = registry[name](source, rng)
            score = score_candidate(
                args,
                gate,
                up,
                down,
                candidate,
                acts,
                residual_exact,
                selected,
                route_weights,
                trace_indices,
                norm_weight,
                margin_dirs,
                p5_margin,
                trace_row_mode,
            )
            score.update(
                {
                    "codec": name,
                    "charge": charge,
                    "elapsed_s": time.perf_counter() - codec_started,
                    "weight_rel_l2": float(np.linalg.norm((source - candidate).reshape(-1)) / max(np.linalg.norm(source.reshape(-1)), 1.0e-12)),
                }
            )
        except Exception as exc:  # keep bakeoff moving
            score = {"codec": name, "status": "error", "error": f"{type(exc).__name__}: {exc}", "elapsed_s": time.perf_counter() - codec_started}
        rows.append(score)
        print(f"[alt_codec] {name}: {score.get('status')} ratio={score.get('noise_margin_ratio')}", file=sys.stderr, flush=True)

    scored = [row for row in rows if row.get("status") == "scored"]
    payload = {
        "layer": args.layer,
        "expert": args.expert,
        "projection": args.projection,
        "tokens": int(acts.shape[0]),
        "margin_token_filter": args.margin_token_filter,
        "trace_row_mode": trace_row_mode,
        "trace_indices": trace_indices.tolist(),
        "reasoning_p5_margin_logits": p5_margin,
        "raw_reasoning_margin_summary": raw_summary,
        "filtered_reasoning_margin_summary": filtered_summary,
        "selection_rule": "P95(|head_ASCII(selected-worst) * J_norm * delta_ffn|) < P5(reasoning margin)",
        "rows": rows,
        "best_by_noise_margin_ratio": min(scored, key=lambda row: row["noise_margin_ratio"])["codec"] if scored else None,
        "elapsed_s": time.perf_counter() - started,
    }
    print(json.dumps(payload, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if scored and all(row["p95_error_lt_p5_margin"] for row in scored) else 2


if __name__ == "__main__":
    raise SystemExit(main())
