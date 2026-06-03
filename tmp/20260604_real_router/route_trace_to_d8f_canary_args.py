#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def choose_row(rows, layer, stage, pos, row_index):
    matches = []
    for item in rows:
        if int(item["layer"]) != layer:
            continue
        if stage and item["stage"] != stage:
            continue
        if pos is not None and int(item["pos"]) != pos:
            continue
        matches.append(item)
    if not matches:
        raise SystemExit(f"no route row for layer={layer} stage={stage or '*'} pos={pos if pos is not None else '*'}")
    if row_index < 0:
        row_index = len(matches) + row_index
    if row_index < 0 or row_index >= len(matches):
        raise SystemExit(f"row_index={row_index} outside {len(matches)} matches")
    return matches[row_index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_csv")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--stage", choices=["prefill", "decode", "decode_cpu"], default=None)
    parser.add_argument("--pos", type=int, default=None)
    parser.add_argument("--row-index", type=int, default=-1)
    parser.add_argument("--shared-model", default="tmp/20260603_mpsgraph_ane/shared_l0_b2048_8bit_20260603T234320.mlpackage")
    parser.add_argument("--d8f-root", default="/Users/silv/cl/tlp/montyneg/ds4/DeepSeek-V4-Flash_H3355_H3353_L26E165_gate_clean_D8F_800kctx_AIME_active_20260603")
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--evict-mib", type=int, default=0)
    parser.add_argument("--mode", choices=["same", "separate"], default="same")
    parser.add_argument("--seed", type=int, default=29001)
    parser.add_argument("--hidden-f32", default=None)
    parser.add_argument("--hidden-row", type=int, default=None)
    parser.add_argument("--evict-mode", choices=["cpu", "metal"], default="cpu")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    with open(args.trace_csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = choose_row(rows, args.layer, args.stage, args.pos, args.row_index)
    experts = ",".join(row[f"e{i}"] for i in range(6))
    weights = ",".join(row[f"w{i}"] for i in range(6))
    d8f_path = str(Path(args.d8f_root) / f"ds4_L{args.layer:02d}_gate_up_down_VQD8_noE8_rank1.d8f")
    command = [
        "tmp/20260603_mpsgraph_ane/ane_d8f_routed_counterbalanced_canary",
        args.shared_model,
        d8f_path,
        experts,
        str(args.trials),
        str(args.evict_mib),
        args.mode,
        str(args.seed),
        weights,
    ]
    if args.hidden_f32:
        command.append(args.hidden_f32)
        command.append(str(args.hidden_row if args.hidden_row is not None else int(row["pos"])))
        command.append(args.evict_mode)
    payload = {
        "trace_csv": args.trace_csv,
        "row": int(row["row"]),
        "stage": row["stage"],
        "pos": int(row["pos"]),
        "layer": int(row["layer"]),
        "experts": [int(row[f"e{i}"]) for i in range(6)],
        "weights": [float(row[f"w{i}"]) for i in range(6)],
        "experts_csv": experts,
        "weights_csv": weights,
        "d8f_path": d8f_path,
        "hidden_f32": args.hidden_f32,
        "hidden_row": args.hidden_row if args.hidden_row is not None else (int(row["pos"]) if args.hidden_f32 else None),
        "evict_mode": args.evict_mode,
        "command": command,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("experts=" + experts)
        print("weights=" + weights)
        print("d8f=" + d8f_path)
        print("cmd=" + " ".join(command))


if __name__ == "__main__":
    main()
