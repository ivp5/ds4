#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


CANARY_RE = re.compile(r"ds4: d8f_organ_selected_batch_canary .*$", re.MULTILINE)
NOTICE_RE = re.compile(r"down_native_tex=(\d+)")


def parse_layers(value):
    if not value:
        return None
    layers = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start_layer = int(start_text)
            end_layer = int(end_text)
            layers.update(range(start_layer, end_layer + 1))
        else:
            layers.add(int(part))
    return layers


def parse_int_list(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def metric_after(label, line):
    match = re.search(rf"{re.escape(label)} ([0-9.]+) ms total", line)
    return float(match.group(1)) if match else None


def metric_equals(label, line, cast=float):
    match = re.search(rf"{re.escape(label)}=([^ \n]+)", line)
    if not match:
        return None
    value = match.group(1)
    if cast is int:
        return int(value)
    return cast(value)


def parse_canary_output(text):
    canary_match = CANARY_RE.search(text)
    if not canary_match:
        raise ValueError("missing d8f_organ_selected_batch_canary line")
    line = canary_match.group(0)
    notice_match = NOTICE_RE.search(text)
    return {
        "down_native_tex": int(notice_match.group(1)) if notice_match else 0,
        "rows": metric_equals("rows", line, int),
        "tokens": metric_equals("tokens", line, int),
        "rounds": metric_equals("rounds", line, int),
        "gpu_ms": metric_after("gpu", line),
        "classic_gpu_ms": metric_after("classic_gpu", line),
        "classic_graph_gpu_ms": metric_after("classic_graph_gpu", line),
        "selected_gpu_ms": metric_after("selected_gpu", line),
        "rc": metric_equals("rc", line, int),
        "classic_rc": metric_equals("classic_rc", line, int),
        "classic_graph_rc": metric_equals("classic_graph_rc", line, int),
        "selected_rc": metric_equals("selected_rc", line, int),
        "mismatch": metric_equals("mismatch", line, int),
        "classic_mismatch": metric_equals("classic_mismatch", line, int),
        "classic_graph_mismatch": metric_equals("classic_graph_mismatch", line, int),
        "selected_mismatch": metric_equals("selected_mismatch", line, int),
        "max_abs": metric_equals("max_abs", line, float),
        "classic_max_abs": metric_equals("classic_max_abs", line, float),
        "selected_max_abs": metric_equals("selected_max_abs", line, float),
        "selected_path": metric_equals("selected_path", line, str),
        "raw_line": line,
    }


def mode_env(mode):
    env = {}
    if mode == "direct-buffer":
        env["DS4_D8F_CLASSIC_PACKET_ICB"] = "0"
    elif mode == "direct-texture":
        env["DS4_D8F_CLASSIC_PACKET_ICB"] = "0"
        env["DS4_D8F_RUNTIME_NATIVE_DOWN"] = "1"
    elif mode == "icb-buffer":
        pass
    elif mode == "texture":
        env["DS4_D8F_RUNTIME_NATIVE_DOWN"] = "1"
    else:
        raise ValueError(f"unknown mode {mode}")
    return env


def summarize_pairs(results):
    by_key = {}
    for result in results:
        key = (result["layer"], result["tokens"], result["rows"], result["rounds"])
        by_key.setdefault(key, {})[result["mode"]] = result
    summaries = []
    for key, modes in sorted(by_key.items()):
        layer, tokens, rows, rounds = key
        summary = {
            "layer": layer,
            "tokens": tokens,
            "rows": rows,
            "rounds": rounds,
        }
        for left_mode, right_mode, prefix in (
            ("direct-buffer", "direct-texture", "direct"),
            ("icb-buffer", "texture", "production"),
        ):
            left = modes.get(left_mode)
            right = modes.get(right_mode)
            if not left or not right:
                continue
            for field in ("gpu_ms", "classic_gpu_ms", "classic_graph_gpu_ms", "selected_gpu_ms"):
                left_ms = left.get(field)
                right_ms = right.get(field)
                if left_ms and right_ms:
                    summary[f"{prefix}_{field}_speedup"] = left_ms / right_ms
                    summary[f"{prefix}_{field}_delta_ms"] = left_ms - right_ms
        summaries.append(summary)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--tokens", default="1,4")
    parser.add_argument("--layers", default="")
    parser.add_argument(
        "--modes",
        default="direct-buffer,direct-texture,icb-buffer,texture",
        help="comma list: direct-buffer,direct-texture,icb-buffer,texture",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    binary = args.binary if args.binary.is_absolute() else args.binary.resolve()
    wanted_layers = parse_layers(args.layers)
    tokens_list = parse_int_list(args.tokens)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_json.parent / (args.out_json.stem + "_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    layer_entries = []
    for entry in manifest["layers"]:
        layer = int(entry["layer"])
        if wanted_layers is not None and layer not in wanted_layers:
            continue
        layer_entries.append(entry)

    results = []
    total_runs = len(layer_entries) * len(tokens_list) * len(modes)
    run_index = 0
    start_time = time.time()
    for entry in layer_entries:
        layer = int(entry["layer"])
        experts = ",".join(str(expert) for expert in entry["hot_experts"])
        for tokens in tokens_list:
            for mode in modes:
                run_index += 1
                run_env = os.environ.copy()
                run_env.update(mode_env(mode))
                command = [
                    str(binary),
                    "--d8f-organ-selected-batch-canary",
                    entry["d8f"],
                    experts,
                    str(args.rows),
                    str(tokens),
                    str(args.rounds),
                ]
                log_path = log_dir / f"L{layer:02d}_T{tokens}_{mode}.log"
                elapsed = time.time() - start_time
                print(
                    f"phase=run {run_index}/{total_runs} elapsed={elapsed:.1f}s "
                    f"layer={layer:02d} tokens={tokens} mode={mode} experts={experts}",
                    flush=True,
                )
                completed = subprocess.run(command, text=True, capture_output=True, env=run_env)
                text = completed.stdout + completed.stderr
                log_path.write_text(text)
                if completed.returncode != 0:
                    print(text, end="")
                    raise SystemExit(completed.returncode)
                metrics = parse_canary_output(text)
                metrics.update(
                    {
                        "layer": layer,
                        "mode": mode,
                        "experts": entry["hot_experts"],
                        "d8f": entry["d8f"],
                        "log": str(log_path),
                    }
                )
                failures = [
                    key
                    for key in ("classic_rc", "classic_graph_rc", "selected_rc")
                    if metrics.get(key) not in (None, 1)
                ]
                mismatches = [
                    key
                    for key in ("classic_mismatch", "classic_graph_mismatch", "selected_mismatch")
                    if metrics.get(key) not in (None, 0)
                ]
                if failures or mismatches:
                    print(text, end="")
                    raise SystemExit(f"validation failed layer={layer} mode={mode} failures={failures} mismatches={mismatches}")
                results.append(metrics)
                print(
                    f"result layer={layer:02d} tokens={tokens} mode={mode} "
                    f"native_tex={metrics['down_native_tex']} gpu_ms={metrics['gpu_ms']:.3f} "
                    f"classic_graph_ms={metrics['classic_graph_gpu_ms']:.3f} "
                    f"selected_ms={metrics['selected_gpu_ms']:.3f}",
                    flush=True,
                )

    summaries = summarize_pairs(results)
    payload = {
        "manifest": str(args.manifest),
        "binary": str(binary),
        "rows": args.rows,
        "rounds": args.rounds,
        "tokens": tokens_list,
        "modes": modes,
        "results": results,
        "summaries": summaries,
    }
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")

    csv_fields = [
        "layer",
        "tokens",
        "mode",
        "down_native_tex",
        "gpu_ms",
        "classic_gpu_ms",
        "classic_graph_gpu_ms",
        "selected_gpu_ms",
        "selected_path",
        "max_abs",
        "selected_max_abs",
        "log",
    ]
    with args.out_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field) for field in csv_fields})

    direct_selected_wins = [
        item for item in summaries if item.get("direct_selected_gpu_ms_speedup", 0.0) > 1.0
    ]
    production_selected_wins = [
        item for item in summaries if item.get("production_selected_gpu_ms_speedup", 0.0) > 1.0
    ]
    print(
        f"summary runs={len(results)} layer_token_pairs={len(summaries)} "
        f"direct_selected_wins={len(direct_selected_wins)} "
        f"production_selected_wins={len(production_selected_wins)} "
        f"out_json={args.out_json} out_csv={args.out_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
