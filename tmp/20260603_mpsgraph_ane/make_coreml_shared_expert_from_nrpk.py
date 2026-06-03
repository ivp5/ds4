#!/usr/bin/env python3
import argparse
import json
import struct
import time

import coremltools as ct
import coremltools.optimize as cto
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


EXP_SCALE = np.array(
    [0.0, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0,
     2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0],
    dtype=np.float32,
)


def now_ms():
    return int(time.time() * 1000)


def target_from_name(name):
    target = getattr(ct.target, name, None)
    if target is None:
        raise SystemExit(f"unknown CoreML target: {name}")
    return target


def read_manifest(pack_path):
    with open(pack_path, "rb") as handle:
        header = handle.read(128)
        if header[:8] != b"DS4NRPK1":
            raise SystemExit("bad DS4NRPK1 magic")
        manifest_offset, manifest_bytes, data_offset, data_bytes, _ = struct.unpack_from(
            "<QQQQQ", header, 16
        )
        handle.seek(manifest_offset)
        manifest = json.loads(handle.read(manifest_bytes))
    return {entry["name"]: entry for entry in manifest}, data_offset


def e4m3fn_decode(raw):
    raw = raw.astype(np.uint16)
    sign = np.where((raw & 0x80) != 0, -1.0, 1.0).astype(np.float32)
    mag = np.minimum(raw & 0x7F, 126)
    exp = (mag >> 3) & 0x0F
    mant = mag & 0x07
    sub = mant.astype(np.float32) * np.float32(0.001953125)
    norm = (np.float32(1.0) + mant.astype(np.float32) * np.float32(0.125)) * EXP_SCALE[exp]
    return sign * np.where(exp == 0, sub, norm)


def e8m0_decode(raw):
    return np.exp2(raw.astype(np.float32) - np.float32(127.0))


def dequant_fp8_transposed(pack_path, entries, data_offset, weight_name, scale_name):
    weight_entry = entries[weight_name]
    scale_entry = entries[scale_name]
    out_dim, in_dim = weight_entry["shape"]
    scale_rows, scale_cols = scale_entry["shape"]
    if scale_rows != (out_dim + 127) // 128 or scale_cols != (in_dim + 127) // 128:
        raise SystemExit(
            f"scale shape mismatch for {weight_name}: {scale_entry['shape']} "
            f"vs expected {[(out_dim + 127) // 128, (in_dim + 127) // 128]}"
        )
    result_t = np.empty((in_dim, out_dim), dtype=np.float16)
    with open(pack_path, "rb") as handle:
        handle.seek(data_offset + weight_entry["data_off"])
        weight = np.frombuffer(handle.read(weight_entry["data_bytes"]), dtype=np.uint8).reshape(
            out_dim, in_dim
        )
        handle.seek(data_offset + scale_entry["data_off"])
        scale = np.frombuffer(handle.read(scale_entry["data_bytes"]), dtype=np.uint8).reshape(
            scale_rows, scale_cols
        )
    for row0 in range(0, out_dim, 128):
        row1 = min(row0 + 128, out_dim)
        for col0 in range(0, in_dim, 128):
            col1 = min(col0 + 128, in_dim)
            tile_scale = e8m0_decode(scale[row0 // 128, col0 // 128])
            tile = e4m3fn_decode(weight[row0:row1, col0:col1]) * tile_scale
            result_t[col0:col1, row0:row1] = tile.T.astype(np.float16)
    return result_t


def build_and_save(args):
    entries, data_offset = read_manifest(args.pack)
    prefix = f"layers.{args.layer}.ffn.shared_experts"
    print(f"[load] layer={args.layer} prefix={prefix} t={now_ms()}", flush=True)
    gate_t = dequant_fp8_transposed(
        args.pack, entries, data_offset, f"{prefix}.w1.weight", f"{prefix}.w1.scale"
    )
    print(f"[load] gate_t={gate_t.shape} t={now_ms()}", flush=True)
    down_t = dequant_fp8_transposed(
        args.pack, entries, data_offset, f"{prefix}.w2.weight", f"{prefix}.w2.scale"
    )
    print(f"[load] down_t={down_t.shape} t={now_ms()}", flush=True)
    up_t = dequant_fp8_transposed(
        args.pack, entries, data_offset, f"{prefix}.w3.weight", f"{prefix}.w3.scale"
    )
    print(f"[load] up_t={up_t.shape} t={now_ms()}", flush=True)

    opset_target = target_from_name(args.opset)
    deployment_target = target_from_name(args.deployment)
    batch = args.batch
    clamp = np.float16(args.clamp)
    neg_clamp = np.float16(-args.clamp)

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(batch, gate_t.shape[0]), dtype=types.fp16)],
        opset_version=opset_target,
    )
    def prog(x):
        gate = mb.matmul(x=x, y=gate_t)
        up = mb.matmul(x=x, y=up_t)
        gate_clamped = mb.minimum(x=gate, y=clamp)
        up_hi = mb.minimum(x=up, y=clamp)
        up_clamped = mb.maximum(x=up_hi, y=neg_clamp)
        sig = mb.sigmoid(x=gate_clamped)
        silu = mb.mul(x=gate_clamped, y=sig)
        mid = mb.mul(x=silu, y=up_clamped)
        y = mb.matmul(x=mid, y=down_t)
        return mb.identity(x=y, name="shared_out")

    print(
        f"[convert] start output={args.output} batch={batch} bits={args.bits} "
        f"opset={args.opset} deployment={args.deployment} t={now_ms()}",
        flush=True,
    )
    model = ct.convert(
        prog,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=deployment_target,
    )
    print(f"[convert] done t={now_ms()}", flush=True)
    if args.bits:
        config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpPalettizerConfig(nbits=args.bits, mode="kmeans")
        )
        print(f"[palettize] start bits={args.bits} t={now_ms()}", flush=True)
        model = cto.coreml.palettize_weights(model, config)
        print(f"[palettize] done t={now_ms()}", flush=True)
    spec = model.get_spec()
    print(
        f"[spec] inputs={[item.name for item in spec.description.input]} "
        f"outputs={[item.name for item in spec.description.output]}",
        flush=True,
    )
    model.save(args.output)
    print(f"[save] done path={args.output} t={now_ms()}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pack")
    parser.add_argument("output")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--clamp", type=float, default=10.0)
    parser.add_argument("--opset", default="iOS26")
    parser.add_argument("--deployment", default="macOS26")
    args = parser.parse_args()
    build_and_save(args)


if __name__ == "__main__":
    main()
