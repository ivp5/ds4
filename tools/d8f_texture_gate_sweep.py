#!/usr/bin/env python3
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


FIELD_RE = re.compile(r"([A-Za-z0-9_]+)=([^ \n]+)")


def parse_metrics(line):
    metrics = {}
    for key, value in FIELD_RE.findall(line):
        metrics[key] = value
    for key in (
        "rows",
        "slots",
        "max_k",
        "rounds",
        "pack2d",
        "pack2d_width",
        "pack2d_height",
    ):
        if key in metrics:
            metrics[key] = int(metrics[key])
    for key in (
        "padded_codebook",
        "codes",
        "buffer",
        "tex_linear2d",
        "tex_buffer",
        "tex_sample",
        "tex_pack2d_read",
        "tex_pack2d_sample",
        "speedup_2d",
        "speedup_tb",
        "speedup_sample",
        "speedup_pack2d_read",
        "speedup_pack2d_sample",
        "max_abs_buffer",
        "max_abs_texture",
        "max_abs_texture_buffer",
        "max_abs_texture_sample",
        "max_abs_pack2d_read",
        "max_abs_pack2d_sample",
        "max_abs_buf_tex",
        "max_abs_buf_tb",
        "max_abs_buf_sample",
        "max_abs_buf_pack2d_read",
        "max_abs_buf_pack2d_sample",
        "pack2d_view",
    ):
        if key in metrics:
            metrics[key] = float(metrics[key].removesuffix("MiB").removesuffix("ms").removesuffix("x"))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--rounds", type=int, default=40)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    results = []
    for entry in manifest["layers"]:
        layer = int(entry["layer"])
        experts = ",".join(str(expert) for expert in entry["hot_experts"])
        command = [str(args.binary), entry["d8f"], experts, str(args.rows), str(args.rounds)]
        print(f"phase=run layer={layer:02d} experts={experts}", flush=True)
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        if completed.returncode != 0:
            print(completed.stdout, end="")
            print(completed.stderr, end="", file=sys.stderr)
            raise SystemExit(completed.returncode)
        line = completed.stdout.strip().splitlines()[-1]
        print(line, flush=True)
        metrics = parse_metrics(line)
        metrics["layer"] = layer
        metrics["experts"] = entry["hot_experts"]
        metrics["d8f"] = entry["d8f"]
        metrics["best_texture_path"] = (
            min(
                (
                    (metrics.get("tex_linear2d", 1e30), "linear2d"),
                    (metrics.get("tex_buffer", 1e30), "texture_buffer"),
                    (metrics.get("tex_sample", 1e30), "sample_nearest"),
                    (metrics.get("tex_pack2d_read", 1e30), "pack2d_read"),
                    (metrics.get("tex_pack2d_sample", 1e30), "pack2d_sample"),
                )
            )[1]
        )
        metrics["best_texture_ms"] = min(
            metrics.get("tex_linear2d", 1e30),
            metrics.get("tex_buffer", 1e30),
            metrics.get("tex_sample", 1e30),
            metrics.get("tex_pack2d_read", 1e30),
            metrics.get("tex_pack2d_sample", 1e30),
        )
        metrics["best_texture_speedup"] = metrics["buffer"] / metrics["best_texture_ms"]
        results.append(metrics)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "rows": args.rows,
                "rounds": args.rounds,
                "layers": results,
            },
            indent=2,
        )
        + "\n"
    )
    with args.out_csv.open("w", newline="") as csv_file:
        fieldnames = [
            "layer",
            "max_k",
            "pack2d",
            "buffer",
            "tex_linear2d",
            "tex_buffer",
            "tex_sample",
            "tex_pack2d_read",
            "tex_pack2d_sample",
            "speedup_2d",
            "speedup_tb",
            "speedup_sample",
            "speedup_pack2d_read",
            "speedup_pack2d_sample",
            "best_texture_path",
            "best_texture_speedup",
            "max_abs_buffer",
            "max_abs_texture",
            "max_abs_texture_buffer",
            "max_abs_texture_sample",
            "max_abs_pack2d_read",
            "max_abs_pack2d_sample",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in results:
            writer.writerow({field: metrics.get(field) for field in fieldnames})

    wins = [metrics for metrics in results if metrics["best_texture_speedup"] > 1.05]
    mean_best_speedup = sum(metrics["best_texture_speedup"] for metrics in results) / len(results)
    print(
        f"summary layers={len(results)} texture_wins_gt_1p05={len(wins)} "
        f"mean_best_speedup={mean_best_speedup:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
