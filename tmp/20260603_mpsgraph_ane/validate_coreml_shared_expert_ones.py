#!/usr/bin/env python3
import argparse

import coremltools as ct
import numpy as np

from make_coreml_shared_expert_from_nrpk import dequant_fp8_transposed, read_manifest


def reference_ones(pack, layer, clamp):
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
    gate = gate_t.sum(axis=0)
    up = up_t.sum(axis=0)
    gate = np.minimum(gate, clamp)
    up = np.clip(up, -clamp, clamp)
    mid = (gate / (1.0 + np.exp(-gate))) * up
    return mid @ down_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pack")
    parser.add_argument("model")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--clamp", type=float, default=10.0)
    args = parser.parse_args()

    print(
        f"[config] layer={args.layer} batch={args.batch} clamp={args.clamp} model={args.model}",
        flush=True,
    )
    ref = reference_ones(args.pack, args.layer, args.clamp)
    model = ct.models.MLModel(args.model, compute_units=ct.ComputeUnit.CPU_AND_NE)
    pred = model.predict({"x": np.ones((args.batch, 4096), dtype=np.float16)})
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
