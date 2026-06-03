#!/usr/bin/env python3
import argparse
import time

import coremltools as ct
import coremltools.optimize as cto
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


def now_ms():
    return int(time.time() * 1000)


def target_from_name(name):
    target = getattr(ct.target, name, None)
    if target is None:
        raise SystemExit(f"unknown CoreML target: {name}")
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--input-dim", type=int, default=4096)
    parser.add_argument("--output-dim", type=int, default=2048)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--opset", default="iOS26")
    parser.add_argument("--deployment", default="macOS26")
    args = parser.parse_args()

    opset_target = target_from_name(args.opset)
    deployment_target = target_from_name(args.deployment)
    weight = (np.random.randn(args.input_dim, args.output_dim).astype(np.float16) * 0.01)

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(args.batch, args.input_dim), dtype=types.fp16)],
        opset_version=opset_target,
    )
    def prog(x):
        y = mb.matmul(x=x, y=weight)
        return mb.identity(x=y, name="y")

    print(
        f"[config] output={args.output} batch={args.batch} "
        f"input_dim={args.input_dim} output_dim={args.output_dim} "
        f"bits={args.bits} opset={args.opset} deployment={args.deployment}",
        flush=True,
    )
    print(f"[convert] start t={now_ms()}", flush=True)
    model = ct.convert(
        prog,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=deployment_target,
    )
    print(f"[convert] done t={now_ms()}", flush=True)
    config = cto.coreml.OptimizationConfig(
        global_config=cto.coreml.OpPalettizerConfig(nbits=args.bits, mode="kmeans")
    )
    print(f"[palettize] start t={now_ms()}", flush=True)
    palettized = cto.coreml.palettize_weights(model, config)
    print(f"[palettize] done t={now_ms()}", flush=True)
    spec = palettized.get_spec()
    print(
        f"[spec] inputs={[item.name for item in spec.description.input]} "
        f"outputs={[item.name for item in spec.description.output]}",
        flush=True,
    )
    palettized.save(args.output)
    print(f"[save] done path={args.output} t={now_ms()}", flush=True)


if __name__ == "__main__":
    main()
