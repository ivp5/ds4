#!/usr/bin/env python3
"""Measure head-projected codec error against reasoning margins.

This is the codec selector:

  P95(|head_margin_direction · J_norm · δffn|) < P5(reasoning margin)

For each selected expert, the tool decodes the D8F record, subtracts the FP4
source weight, applies that weight error to real FFN-input activations X, then
projects the resulting residual error through final RMSNorm and the output-head
top1/top2 margin directions from a --dump-logprobs trace.

For L42 this is a direct terminal-layer predictor. For earlier layers it is a
local downstream-linearized proxy unless an exact residual HC dump is supplied.
"""

from __future__ import annotations

import argparse
import json
import math
import mmap
from pathlib import Path
import sys
import time
from typing import Any
import unicodedata

import numpy as np

try:
    import mlx.core as mx
except Exception:  # pragma: no cover - optional runtime backend
    mx = None


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path("/Users/silv/cl/tlp/montyneg/ds4/DeepSeek-V4-Flash")
DEFAULT_ACTS = REPO_ROOT / "tmp" / "20260601_actaware_nvidia" / "acts_L36_L42_8192.npz"

if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from ds4_d8f import decode_d8f, load_source, read_record  # noqa: E402
from ds4_logprob_margin_gate import analyze as analyze_logprobs  # noqa: E402
from ds4_logprob_margin_gate import load_steps  # noqa: E402
from ds4_safetensors import SafetensorStore  # noqa: E402


ROUTE_SCALE = 1.5
TOPK_EXPERTS = 6


def parse_csv_ints(text: str | None) -> list[int]:
    if not text:
        return []
    return [int(part.strip(), 0) for part in text.split(",") if part.strip()]


def token_payload(item: dict[str, Any]) -> dict[str, Any]:
    token = item.get("token")
    if isinstance(token, dict):
        return token
    return item


def token_text(item: dict[str, Any]) -> str:
    token = token_payload(item)
    if isinstance(token, dict):
        text = token.get("text")
        if isinstance(text, str):
            return text
        raw_bytes = token.get("bytes")
        if isinstance(raw_bytes, list):
            try:
                return bytes(int(value) & 0xFF for value in raw_bytes).decode("utf-8", errors="ignore")
            except ValueError:
                return ""
    return ""


def is_latin_margin_token(item: dict[str, Any]) -> bool:
    text = token_text(item)
    if not text:
        return False
    saw_letter = False
    for char in text:
        category = unicodedata.category(char)
        if category[0] == "L":
            saw_letter = True
            if "LATIN" not in unicodedata.name(char, ""):
                return False
        elif category[0] in {"M", "N", "P", "S", "Z"}:
            continue
        elif char in "\n\r\t":
            continue
        else:
            return False
    return saw_letter or any(char in "\n\r\t" or unicodedata.category(char)[0] in {"N", "P", "S", "Z"} for char in text)


def is_ascii_margin_token(item: dict[str, Any]) -> bool:
    text = token_text(item)
    if not text:
        return False
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        return False
    return any(not char.isspace() for char in text) or any(char in "\n\r\t" for char in text)


def margin_token_allowed(item: dict[str, Any], token_filter: str) -> bool:
    if token_filter == "none":
        return True
    if token_filter == "ascii":
        return is_ascii_margin_token(item)
    if token_filter == "latin":
        return is_latin_margin_token(item)
    raise ValueError(f"bad margin token filter {token_filter}")


def item_token_id(item: dict[str, Any]) -> int:
    return int(token_payload(item)["id"])


def item_score(item: dict[str, Any]) -> float:
    if "logit" in item:
        return float(item["logit"])
    return float(item["logprob"])


def filtered_margin_summary(margins: list[float], filtered_steps: int, fallback_steps: int) -> dict[str, Any]:
    values = np.asarray(margins, dtype=np.float64)
    reasoning = values[1:] if values.size > 1 else values
    return {
        "filtered_steps": int(filtered_steps),
        "fallback_steps": int(fallback_steps),
        "reasoning_p5_margin_logits": float(np.percentile(reasoning, 5)) if reasoning.size else math.nan,
        "reasoning_p10_margin_logits": float(np.percentile(reasoning, 10)) if reasoning.size else math.nan,
        "reasoning_mean_margin_logits": float(np.mean(reasoning)) if reasoning.size else math.nan,
        "reasoning_min_margin_logits": float(np.min(reasoning)) if reasoning.size else math.nan,
    }


def load_trace_margin_directions(
    path: Path,
    top_k: int,
    token_filter: str,
    threshold_mode: str,
    fragile_margin_max: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any], dict[str, Any]]:
    steps = load_steps(path)
    _, raw_summary = analyze_logprobs(steps, repeat_window=6)
    all_pairs: list[tuple[int, int]] = []
    used_pairs: list[tuple[int, int]] = []
    used_trace_indices: list[int] = []
    filtered_margins: list[float] = []
    token_pool: set[int] = set()
    filtered_steps = 0
    fallback_steps = 0
    skipped_nonfragile_steps = 0
    for trace_index, step in enumerate(steps):
        top = step.get("top_logprobs", [])
        if len(top) < 2:
            continue
        selected_token = step.get("selected", {})
        if not isinstance(selected_token, dict) or "id" not in selected_token:
            selected_token = top[0].get("token", {})
        selected_id = item_token_id(selected_token)
        selected_score = item_score(top[0])
        for item in top[:top_k]:
            if item_token_id(item) == selected_id:
                selected_score = item_score(item)
                break
        filtered_competitors = [
            item
            for item in top[:top_k]
            if item_token_id(item) != selected_id and margin_token_allowed(item, token_filter)
        ]
        if filtered_competitors:
            competitor = filtered_competitors[0]
            filtered_steps += 1
        else:
            competitor = next(item for item in top[:top_k] if item_token_id(item) != selected_id)
            fallback_steps += 1
        margin = selected_score - item_score(competitor)
        pair = (selected_id, item_token_id(competitor))
        all_pairs.append(pair)
        filtered_margins.append(margin)
        if len(all_pairs) > 1 and (fragile_margin_max is None or margin <= fragile_margin_max):
            used_pairs.append(pair)
            used_trace_indices.append(trace_index)
        else:
            skipped_nonfragile_steps += 1
        token_pool.add(selected_id)
        for item in filtered_competitors if filtered_competitors else top[:top_k]:
            token_pool.add(item_token_id(item))
    if not used_pairs:
        if fragile_margin_max is None:
            raise ValueError(f"{path} has no usable selected-vs-competitor margin pairs")
        raise ValueError(f"{path} has no fragile margin pairs at or below {fragile_margin_max}")
    if not all_pairs:
        raise ValueError(f"{path} has no top1/top2 margin pairs")
    for a, b in all_pairs:
        token_pool.add(a)
        token_pool.add(b)
    pool = np.asarray(sorted(token_pool), dtype=np.int64)
    row_to_local = {int(token): i for i, token in enumerate(pool.tolist())}
    pairs = np.asarray([[row_to_local[a], row_to_local[b]] for a, b in used_pairs], dtype=np.int64)
    trace_indices = np.asarray(used_trace_indices, dtype=np.int64)
    filtered_summary = {
        **filtered_margin_summary(filtered_margins, filtered_steps, fallback_steps),
        "direction_mode": "selected_vs_nearest_allowed_competitor",
        "fragile_margin_max": fragile_margin_max,
        "all_direction_count": len(all_pairs),
        "used_fragile_direction_count": len(used_pairs),
        "skipped_nonfragile_or_prefix_steps": int(skipped_nonfragile_steps),
    }
    raw_p5 = float(raw_summary["reasoning_p5_margin_logits"])
    filtered_p5 = float(filtered_summary["reasoning_p5_margin_logits"])
    if threshold_mode == "raw":
        p5_margin = raw_p5
    elif threshold_mode == "filtered":
        p5_margin = filtered_p5
    elif threshold_mode == "min":
        p5_margin = min(raw_p5, filtered_p5)
    else:
        raise ValueError(f"bad margin threshold mode {threshold_mode}")
    return pool, pairs, trace_indices, p5_margin, raw_summary, filtered_summary


def load_many_trace_margin_directions(
    paths: list[Path],
    top_k: int,
    token_filter: str,
    threshold_mode: str,
    fragile_margin_max: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any], dict[str, Any]]:
    if not paths:
        raise ValueError("at least one --logprobs-json path is required")
    if len(paths) == 1:
        return load_trace_margin_directions(paths[0], top_k, token_filter, threshold_mode, fragile_margin_max)

    token_pool: set[int] = set()
    token_pairs: list[tuple[int, int]] = []
    trace_indices: list[int] = []
    raw_summaries: list[dict[str, Any]] = []
    filtered_summaries: list[dict[str, Any]] = []
    p5_values: list[float] = []
    step_offset = 0
    for path in paths:
        local_pool, local_pairs, local_trace_indices, p5_margin, raw_summary, filtered_summary = load_trace_margin_directions(
            path, top_k, token_filter, threshold_mode, fragile_margin_max
        )
        local_tokens = local_pool.tolist()
        for pair in local_pairs.tolist():
            first = int(local_tokens[int(pair[0])])
            second = int(local_tokens[int(pair[1])])
            token_pairs.append((first, second))
            token_pool.add(first)
            token_pool.add(second)
        trace_indices.extend((local_trace_indices + step_offset).astype(np.int64).tolist())
        raw_summaries.append({"source": str(path), **raw_summary})
        filtered_summaries.append({"source": str(path), **filtered_summary})
        p5_values.append(float(p5_margin))
        step_offset += len(load_steps(path))

    pool = np.asarray(sorted(token_pool), dtype=np.int64)
    row_to_local = {int(token): i for i, token in enumerate(pool.tolist())}
    pairs = np.asarray([[row_to_local[a], row_to_local[b]] for a, b in token_pairs], dtype=np.int64)
    trace_index_array = np.asarray(trace_indices, dtype=np.int64)
    raw_p5s = [float(item["reasoning_p5_margin_logits"]) for item in raw_summaries]
    filtered_p5s = [float(item["reasoning_p5_margin_logits"]) for item in filtered_summaries]
    raw_summary = {
        "trace_count": len(paths),
        "sources": raw_summaries,
        "reasoning_p5_margin_logits": float(np.min(raw_p5s)),
        "reasoning_min_margin_logits": float(np.min([float(item["reasoning_min_margin_logits"]) for item in raw_summaries])),
        "reasoning_mean_margin_logits": float(np.mean([float(item["reasoning_mean_margin_logits"]) for item in raw_summaries])),
        "aggregation": "min_p5_across_traces",
    }
    filtered_summary = {
        "trace_count": len(paths),
        "sources": filtered_summaries,
        "reasoning_p5_margin_logits": float(np.min(filtered_p5s)),
        "reasoning_min_margin_logits": float(np.min([float(item["reasoning_min_margin_logits"]) for item in filtered_summaries])),
        "reasoning_mean_margin_logits": float(np.mean([float(item["reasoning_mean_margin_logits"]) for item in filtered_summaries])),
        "used_fragile_direction_count": int(pairs.shape[0]),
        "aggregation": "min_p5_across_traces",
    }
    return pool, pairs, trace_index_array, float(np.min(p5_values)), raw_summary, filtered_summary


def read_hc_dump(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 16:
        raise ValueError(f"{path}: too small for DS4H header")
    magic, n_tokens, hc_dim, layer = struct.unpack_from("<IIII", raw, 0)
    if magic != 0x44533448:
        raise ValueError(f"{path}: bad DS4H magic {magic:#x}")
    data = np.frombuffer(raw, dtype=np.float32, offset=16)
    expected = int(n_tokens) * int(hc_dim)
    if data.size != expected:
        raise ValueError(f"{path}: expected {expected} f32 values, got {data.size}")
    return data.reshape(int(n_tokens), int(hc_dim))


def load_acts(args: argparse.Namespace) -> np.ndarray:
    if args.ffn_in_bin:
        data = np.fromfile(args.ffn_in_bin, dtype=np.float32)
        if data.size % 4096 != 0:
            raise ValueError(f"{args.ffn_in_bin}: f32 count {data.size} not divisible by 4096")
        return data.reshape(data.size // 4096, 4096)[: args.tokens]
    acts = np.load(args.acts)[f"act_L{args.layer}"].astype(np.float32)
    return acts[: args.tokens]


def load_residual(args: argparse.Namespace, n_tokens: int) -> np.ndarray | None:
    if not args.hc_dump:
        return None
    hc = read_hc_dump(args.hc_dump)
    if hc.shape[0] < n_tokens:
        raise ValueError(f"{args.hc_dump}: only {hc.shape[0]} tokens, need {n_tokens}")
    if hc.shape[1] == 4096:
        return hc[:n_tokens]
    if hc.shape[1] % 4096 != 0:
        raise ValueError(f"{args.hc_dump}: hc_dim {hc.shape[1]} not compatible")
    return hc[:n_tokens, :4096]


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def silu_mlx(x: Any) -> Any:
    return x / (1.0 + mx.exp(-x))


def resolve_backend(args: argparse.Namespace) -> str:
    if args.backend == "auto":
        backend = "mlx" if mx is not None else "numpy"
    else:
        backend = args.backend
    if backend == "mlx":
        if mx is None:
            raise RuntimeError("MLX backend requested but mlx.core is not importable")
        device = mx.gpu if args.mlx_device == "gpu" else mx.cpu
        mx.set_default_device(device)
    return backend


def router_select(store: SafetensorStore, layer: int, acts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gate = store.numeric(f"layers.{layer}.ffn.gate.weight")
    scores = acts @ gate.T
    probs = np.sqrt(np.log1p(np.exp(-np.abs(scores))) + np.maximum(scores, 0.0))
    bias_name = f"layers.{layer}.ffn.gate.bias"
    if bias_name in store.weight_map:
        probs += store.numeric(bias_name).reshape(1, -1)
    selected = np.argpartition(-probs, TOPK_EXPERTS, axis=1)[:, :TOPK_EXPERTS]
    selected_probs = np.take_along_axis(probs, selected, axis=1)
    weights = selected_probs / np.maximum(np.sum(selected_probs, axis=1, keepdims=True), 1.0e-9) * ROUTE_SCALE
    return selected.astype(np.int32), weights.astype(np.float32)


def projection_delta(
    projection: str,
    acts: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    gate_q: np.ndarray,
    up_q: np.ndarray,
    down_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gate_pre = acts @ gate.T
    up_pre = acts @ up.T
    hidden = silu(gate_pre) * up_pre
    clean = hidden @ down.T
    if projection == "gate":
        delta = (silu(acts @ gate_q.T) * up_pre) @ down.T - clean
    elif projection == "up":
        delta = (silu(gate_pre) * (acts @ up_q.T)) @ down.T - clean
    elif projection == "down":
        delta = hidden @ down_q.T - clean
    elif projection == "all":
        delta = (silu(acts @ gate_q.T) * (acts @ up_q.T)) @ down_q.T - clean
    else:
        raise ValueError(f"bad projection {projection}")
    return clean, delta


def norm_jvp(residual: np.ndarray, delta: np.ndarray, norm_weight: np.ndarray, exact: bool) -> np.ndarray:
    rms = np.sqrt(np.mean(residual * residual, axis=1, keepdims=True) + 1.0e-6)
    if not exact:
        return (delta / rms) * norm_weight
    dot = np.mean(residual * delta, axis=1, keepdims=True)
    return (delta / rms - residual * dot / (rms * rms * rms)) * norm_weight


def norm_jvp_mlx(residual: Any, delta: Any, norm_weight: np.ndarray, exact: bool) -> Any:
    rms = mx.sqrt(mx.mean(residual * residual, axis=1, keepdims=True) + 1.0e-6)
    norm_m = mx.array(norm_weight.reshape(1, -1), dtype=mx.float32)
    if not exact:
        return (delta / rms) * norm_m
    dot = mx.mean(residual * delta, axis=1, keepdims=True)
    return (delta / rms - residual * dot / (rms * rms * rms)) * norm_m


def abs_margin_errors_mlx(
    projection: str,
    acts: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    gate_q: np.ndarray,
    up_q: np.ndarray,
    down_q: np.ndarray,
    route_weights: np.ndarray,
    residual_exact: np.ndarray | None,
    norm_weight: np.ndarray,
    margin_dirs: np.ndarray,
    exact_norm: bool,
    aligned: bool,
) -> np.ndarray:
    x = mx.array(acts, dtype=mx.float32)
    gate_m = mx.array(gate, dtype=mx.float32)
    up_m = mx.array(up, dtype=mx.float32)
    down_m = mx.array(down, dtype=mx.float32)
    gate_pre = x @ mx.transpose(gate_m)
    up_pre = x @ mx.transpose(up_m)
    hidden = silu_mlx(gate_pre) * up_pre
    clean = hidden @ mx.transpose(down_m)
    if projection == "gate":
        gate_q_m = mx.array(gate_q, dtype=mx.float32)
        delta = (silu_mlx(x @ mx.transpose(gate_q_m)) * up_pre) @ mx.transpose(down_m) - clean
    elif projection == "up":
        up_q_m = mx.array(up_q, dtype=mx.float32)
        delta = (silu_mlx(gate_pre) * (x @ mx.transpose(up_q_m))) @ mx.transpose(down_m) - clean
    elif projection == "down":
        down_q_m = mx.array(down_q, dtype=mx.float32)
        delta = hidden @ mx.transpose(down_q_m) - clean
    elif projection == "all":
        gate_q_m = mx.array(gate_q, dtype=mx.float32)
        up_q_m = mx.array(up_q, dtype=mx.float32)
        down_q_m = mx.array(down_q, dtype=mx.float32)
        delta = (silu_mlx(x @ mx.transpose(gate_q_m)) * (x @ mx.transpose(up_q_m))) @ mx.transpose(down_q_m) - clean
    else:
        raise ValueError(f"bad projection {projection}")
    delta = delta * mx.array(route_weights.reshape(-1, 1), dtype=mx.float32)
    residual = mx.array(residual_exact, dtype=mx.float32) if residual_exact is not None else x + clean
    delta_norm = norm_jvp_mlx(residual, delta, norm_weight, exact=exact_norm)
    margin_dirs_m = mx.array(margin_dirs, dtype=mx.float32)
    if aligned:
        errors = mx.sum(delta_norm * margin_dirs_m, axis=1)
    else:
        errors = delta_norm @ mx.transpose(margin_dirs_m)
    abs_errors = mx.abs(errors)
    mx.eval(abs_errors)
    return np.asarray(abs_errors).reshape(-1)


def decode_projection(mapped: mmap.mmap, layer: int, projection: str, expert: int) -> tuple[np.ndarray, np.ndarray]:
    record = read_record(mapped, projection, expert)
    decoded = decode_d8f(mapped, record, projection)
    source = load_source(layer, projection, expert)
    return source.astype(np.float32), decoded.astype(np.float32)


def score_expert(
    mapped: mmap.mmap,
    args: argparse.Namespace,
    expert: int,
    ordinal: int,
    total: int,
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
    started = time.perf_counter()
    token_rows_all, slots_all = np.where(selected == expert)
    token_rows = token_rows_all
    slots = slots_all
    dir_rows: np.ndarray | None = None
    if trace_row_mode == "aligned":
        trace_index_to_dir = {int(row): offset for offset, row in enumerate(trace_indices.tolist())}
        keep = np.asarray([int(row) in trace_index_to_dir for row in token_rows_all], dtype=bool)
        token_rows = token_rows_all[keep]
        slots = slots_all[keep]
        dir_rows = np.asarray([trace_index_to_dir[int(row)] for row in token_rows.tolist()], dtype=np.int64)
    print(
        f"[ds4_head_margin_sensitivity] expert {ordinal}/{total} E{expert} "
        f"selected_tokens={token_rows.size}/{token_rows_all.size} "
        f"mode={trace_row_mode} backend={args.backend}",
        file=sys.stderr,
        flush=True,
    )
    if token_rows.size == 0:
        return {
            "expert": expert,
            "selected_tokens": 0,
            "selected_tokens_total": int(token_rows_all.size),
            "backend": args.backend,
            "trace_row_mode": trace_row_mode,
            "status": "unselected_fragile" if trace_row_mode == "aligned" else "unselected",
        }

    gate, gate_q = decode_projection(mapped, args.layer, "gate", expert)
    up, up_q = decode_projection(mapped, args.layer, "up", expert)
    down, down_q = decode_projection(mapped, args.layer, "down", expert)
    x = acts[token_rows]
    route_weights = weights[token_rows, slots]
    residual_rows = residual_exact[token_rows] if residual_exact is not None else None
    active_margin_dirs = margin_dirs[dir_rows] if dir_rows is not None else margin_dirs
    if args.backend == "mlx":
        abs_errors = abs_margin_errors_mlx(
            args.projection,
            x,
            gate,
            up,
            down,
            gate_q,
            up_q,
            down_q,
            route_weights,
            residual_rows,
            norm_weight,
            active_margin_dirs,
            exact_norm=not args.simple_rms,
            aligned=trace_row_mode == "aligned",
        )
    else:
        clean, delta = projection_delta(args.projection, x, gate, up, down, gate_q, up_q, down_q)
        delta *= route_weights[:, None]
        residual = residual_rows if residual_rows is not None else x + clean
        delta_norm = norm_jvp(residual, delta, norm_weight, exact=not args.simple_rms)
        if trace_row_mode == "aligned":
            errors = np.sum(delta_norm * active_margin_dirs, axis=1)
        else:
            errors = delta_norm @ active_margin_dirs.T
        abs_errors = np.abs(errors).reshape(-1)
    p95 = float(np.percentile(abs_errors, 95))
    row = {
        "expert": expert,
        "selected_tokens": int(token_rows.size),
        "selected_tokens_total": int(token_rows_all.size),
        "route_weight_mean": float(np.mean(weights[token_rows, slots])),
        "backend": args.backend,
        "trace_row_mode": trace_row_mode,
        "p95_margin_logit_error": p95,
        "p99_margin_logit_error": float(np.percentile(abs_errors, 99)),
        "mean_abs_margin_logit_error": float(np.mean(abs_errors)),
        "p95_error_lt_p5_margin": bool(p95 < p5_margin),
        "noise_margin_ratio": p95 / max(p5_margin, 1.0e-9),
        "residual_mode": "exact_hc_dump" if residual_exact is not None else "local_expert_approx",
        "status": "scored",
    }
    print(
        f"[ds4_head_margin_sensitivity] expert {ordinal}/{total} E{expert} done "
        f"p95={row['p95_margin_logit_error']:.6g} ratio={row['noise_margin_ratio']:.3g} "
        f"elapsed={time.perf_counter() - started:.2f}s",
        file=sys.stderr,
        flush=True,
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d8f", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--logprobs-json", required=True, type=Path, nargs="+")
    parser.add_argument("--acts", type=Path, default=DEFAULT_ACTS)
    parser.add_argument("--ffn-in-bin", type=Path)
    parser.add_argument("--hc-dump", type=Path)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--experts", default="selected")
    parser.add_argument("--projection", choices=("gate", "up", "down", "all"), default="all")
    parser.add_argument("--logprobs-top-k", type=int, default=20)
    parser.add_argument("--margin-token-filter", choices=("none", "ascii", "latin"), default="ascii")
    parser.add_argument("--margin-threshold", choices=("raw", "filtered", "min"), default="min")
    parser.add_argument(
        "--fragile-margin-max",
        type=float,
        default=0.5,
        help="score only reasoning trace directions whose selected-vs-competitor margin is at or below this value; negative disables pruning",
    )
    parser.add_argument("--simple-rms", action="store_true", help="drop the RMSNorm mean-coupling JVP term")
    parser.add_argument("--backend", choices=("auto", "numpy", "mlx"), default="auto")
    parser.add_argument("--mlx-device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--max-experts", type=int, help="score only the first N routed/requested experts")
    parser.add_argument(
        "--trace-row-mode",
        choices=("auto", "aligned", "cross"),
        default="auto",
        help="aligned scores each exact trace row against its own fragile competitor; cross scores calibration deltas against all fragile directions",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    args.backend = resolve_backend(args)

    store = SafetensorStore(MODEL_DIR)
    acts = load_acts(args)
    residual_exact = load_residual(args, acts.shape[0])
    selected, weights = router_select(store, args.layer, acts)
    if args.experts == "selected":
        experts = sorted(int(value) for value in np.unique(selected))
    else:
        experts = parse_csv_ints(args.experts)
    if args.max_experts is not None:
        experts = experts[: args.max_experts]

    token_pool, pair_local, trace_indices, p5_margin, raw_margin_summary, filtered_margin_summary_payload = load_many_trace_margin_directions(
        args.logprobs_json,
        args.logprobs_top_k,
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
    head_rows = store.rows("head.weight", token_pool)
    margin_dirs = head_rows[pair_local[:, 0]] - head_rows[pair_local[:, 1]]
    norm_weight = store.numeric("norm.weight").reshape(-1)

    with args.d8f.open("rb") as handle:
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            rows = [
                score_expert(
                    mapped,
                    args,
                    expert,
                    ordinal,
                    len(experts),
                    acts,
                    residual_exact,
                    selected,
                    weights,
                    trace_indices,
                    norm_weight,
                    margin_dirs,
                    p5_margin,
                    trace_row_mode,
                )
                for ordinal, expert in enumerate(experts, start=1)
            ]
        finally:
            mapped.close()

    scored = [row for row in rows if row.get("status") == "scored"]
    p95s = np.asarray([row["p95_margin_logit_error"] for row in scored], dtype=np.float64)
    payload = {
        "d8f": str(args.d8f),
        "layer": args.layer,
        "projection": args.projection,
        "backend": args.backend,
        "mlx_device": args.mlx_device if args.backend == "mlx" else None,
        "tokens": int(acts.shape[0]),
        "experts_requested": args.experts,
        "experts_scored": len(scored),
        "margin_trace": [str(path) for path in args.logprobs_json],
        "reasoning_p5_margin_logits": p5_margin,
        "margin_token_filter": args.margin_token_filter,
        "margin_threshold": args.margin_threshold,
        "fragile_margin_max": None if args.fragile_margin_max < 0.0 else args.fragile_margin_max,
        "raw_reasoning_margin_summary": raw_margin_summary,
        "filtered_reasoning_margin_summary": filtered_margin_summary_payload,
        "margin_direction_count": int(margin_dirs.shape[0]),
        "trace_row_mode": trace_row_mode,
        "trace_indices": trace_indices.tolist(),
        "selection_rule": "P95(|head_margin_direction * J_norm * delta_ffn|) < P5(reasoning margin)",
        "max_expert_p95_margin_logit_error": float(np.max(p95s)) if scored else math.nan,
        "passes_selection_rule": bool(np.max(p95s) < p5_margin) if scored else False,
        "residual_mode": "exact_hc_dump" if residual_exact is not None else "local_expert_approx",
        "experts": rows,
    }
    print(json.dumps(payload, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if payload["passes_selection_rule"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
