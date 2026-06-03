#!/usr/bin/env python3
import argparse

import coremltools as ct
import numpy as np

from make_coreml_shared_expert_from_nrpk import dequant_fp8_transposed, read_manifest


def reference_shared(pack, layer, clamp, hidden):
    entries, data_offset = read_manifest(pack)
    prefix = f"layers.{layer}.ffn.shared_experts"
    gate_t = dequant_fp8_transposed(
        pack, entries, data_offset, f"{prefix}.w1.weight", f"{prefix}.w1.scale"
    ).astype(np.float32)
    down_t = dequant_fp8_transposed(
        pack, entries, data_offset, f"{prefix}.w2.weight", f"{prefix}.w2.scale"
    ).astype(np.float32)
    up_t = dequant_fp8_transposed(
        pack, entries, data_offset, f"{prefix}.w3.weight", f"{prefix}.w3.scale"
    ).astype(np.float32)
    gate = hidden @ gate_t
    up = hidden @ up_t
    gate = np.minimum(gate, clamp)
    up = np.clip(up, -clamp, clamp)
    mid = (gate / (1.0 + np.exp(-gate))) * up
    return mid @ down_t


def load_hidden(path, row):
    data = np.fromfile(path, dtype=np.float32)
    if data.size % 4096 != 0:
        raise SystemExit(f"hidden file has non-row size: {path} floats={data.size}")
    rows = data.size // 4096
    if row < 0:
        row += rows
    if row < 0 or row >= rows:
        raise SystemExit(f"hidden row {row} outside {rows} rows")
    hidden = data.reshape(rows, 4096)[row].astype(np.float32)
    print(
        f"[hidden] source=file path={path} row={row} rows={rows} "
        f"mean={hidden.mean():.6g} rms={np.sqrt(np.mean(hidden * hidden)):.6g} "
        f"max_abs={np.max(np.abs(hidden)):.6g} first={hidden[0]:.6g}",
        flush=True,
    )
    return hidden


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pack")
    parser.add_argument("model")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--clamp", type=float, default=10.0)
    parser.add_argument("--hidden-f32", default=None)
    parser.add_argument("--hidden-row", type=int, default=0)
    args = parser.parse_args()

    print(
        f"[config] layer={args.layer} batch={args.batch} clamp={args.clamp} model={args.model}",
        flush=True,
    )
    hidden = load_hidden(args.hidden_f32, args.hidden_row) if args.hidden_f32 else np.ones(4096, dtype=np.float32)
    if not args.hidden_f32:
        print("[hidden] source=ones", flush=True)
    ref = reference_shared(args.pack, args.layer, args.clamp, hidden)
    model = ct.models.MLModel(args.model, compute_units=ct.ComputeUnit.CPU_AND_NE)
    batch = np.repeat(hidden.astype(np.float16)[None, :], args.batch, axis=0)
    pred = model.predict({"x": batch})
    out = np.asarray(pred["shared_out"], dtype=np.float32)
    row = out[0]
    diff = row - ref
    denom = np.maximum(1.0, np.abs(ref))
    max_abs = float(np.max(np.abs(diff)))
    max_rel = float(np.max(np.abs(diff) / denom))
    rms = float(np.sqrt(np.mean(diff * diff)))
    ref_rms = float(np.sqrt(np.mean(ref * ref)))
    print(
        f"[result] shape={out.shape} max_abs={max_abs:.6g} "
        f"max_rel={max_rel:.6g} rms={rms:.6g} ref_rms={ref_rms:.6g} "
        f"sample_ref={ref[0]:.6g} sample_got={row[0]:.6g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
