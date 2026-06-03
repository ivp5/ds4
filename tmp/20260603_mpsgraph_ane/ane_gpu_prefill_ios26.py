#!/usr/bin/env python3
import argparse
import threading
import time

import coremltools as ct
import coremltools.optimize as cto
import mlx.core as mx
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


def build_coreml_model(batch, input_dim, output_dim, bits, opset_target, deployment_target):
    weight = (np.random.randn(input_dim, output_dim).astype(np.float16) * 0.01)

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(batch, input_dim), dtype=types.fp16)],
        opset_version=opset_target,
    )
    def prog(x):
        return mb.matmul(x=x, y=weight)

    print(f"[coreml] convert_start t={now_ms()}", flush=True)
    model = ct.convert(
        prog,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=deployment_target,
    )
    print(f"[coreml] convert_done t={now_ms()}", flush=True)
    config = cto.coreml.OptimizationConfig(
        global_config=cto.coreml.OpPalettizerConfig(nbits=bits, mode="kmeans")
    )
    print(f"[coreml] palettize_start bits={bits} t={now_ms()}", flush=True)
    palettized = cto.coreml.palettize_weights(model, config)
    print(f"[coreml] palettize_done t={now_ms()}", flush=True)
    return palettized


def time_call(label, fn):
    t0 = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t0
    print(f"[{label}] elapsed_ms={elapsed * 1e3:.3f} t={now_ms()}", flush=True)
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--input-dim", type=int, default=4096)
    parser.add_argument("--output-dim", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=6)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--opset", default="iOS26")
    parser.add_argument("--deployment", default="macOS26")
    args = parser.parse_args()

    opset_target = target_from_name(args.opset)
    deployment_target = target_from_name(args.deployment)
    print(
        f"[config] batch={args.batch} input_dim={args.input_dim} "
        f"output_dim={args.output_dim} experts={args.experts} bits={args.bits} "
        f"opset={args.opset} deployment={args.deployment}",
        flush=True,
    )

    model = build_coreml_model(
        args.batch,
        args.input_dim,
        args.output_dim,
        args.bits,
        opset_target,
        deployment_target,
    )
    coreml_input = {"x": np.ones((args.batch, args.input_dim), np.float16)}
    gpu_weight = mx.ones((args.input_dim, args.output_dim), dtype=mx.float16) * 0.001
    gpu_input = mx.ones((args.batch, args.input_dim), dtype=mx.float16)

    def ane_work():
        for _ in range(args.experts):
            model.predict(coreml_input)

    def gpu_work():
        for _ in range(args.experts):
            mx.eval(gpu_input @ gpu_weight)

    print("[warmup] start", flush=True)
    ane_work()
    gpu_work()
    print(f"[warmup] done t={now_ms()}", flush=True)

    ane_elapsed = time_call("ane", ane_work)
    gpu_elapsed = time_call("gpu", gpu_work)

    def concurrent_work():
        ane_thread = threading.Thread(target=ane_work)
        gpu_thread = threading.Thread(target=gpu_work)
        ane_thread.start()
        gpu_thread.start()
        ane_thread.join()
        gpu_thread.join()

    both_elapsed = time_call("concurrent", concurrent_work)
    serial_elapsed = ane_elapsed + gpu_elapsed
    ops = args.experts * args.batch * args.input_dim * args.output_dim * 2
    print(
        f"[result] serial_ms={serial_elapsed * 1e3:.3f} "
        f"concurrent_ms={both_elapsed * 1e3:.3f} "
        f"speedup={serial_elapsed / both_elapsed:.3f} "
        f"ane_GFLOPs={ops / ane_elapsed / 1e9:.1f} "
        f"gpu_GFLOPs={ops / gpu_elapsed / 1e9:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
