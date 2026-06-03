#!/usr/bin/env python3
import argparse
import time

import coremltools as ct
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


def build_b1_program(input_dim, output_dim, codes, opset_target):
    groups = input_dim // 8
    codebook = np.random.randn(8, codes).astype(np.float16)
    indices = np.random.randint(0, codes, size=(groups, output_dim), dtype=np.int32)

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, input_dim), dtype=types.fp16)],
        opset_version=opset_target,
    )
    def prog(x):
        reshaped = mb.reshape(x=x, shape=(groups, 8))
        table = mb.matmul(x=reshaped, y=codebook)
        gathered = mb.gather_along_axis(
            x=table, indices=indices, axis=1, validate_indices=False
        )
        out = mb.reduce_sum(x=gathered, axes=[0])
        return mb.expand_dims(x=out, axes=[0])

    return prog


def build_batched_program(batch, input_dim, output_dim, codes, opset_target):
    groups = input_dim // 8
    codebook = np.random.randn(8, codes).astype(np.float16)
    base_indices = np.random.randint(
        0, codes, size=(1, groups, output_dim), dtype=np.int32
    )
    indices = np.broadcast_to(base_indices, (batch, groups, output_dim)).copy()
    print(
        f"[batched] materialized_indices_MB={indices.nbytes / 1e6:.3f}",
        flush=True,
    )

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(batch, input_dim), dtype=types.fp16)],
        opset_version=opset_target,
    )
    def prog(x):
        reshaped = mb.reshape(x=x, shape=(batch, groups, 8))
        table = mb.matmul(x=reshaped, y=codebook)
        gathered = mb.gather_along_axis(
            x=table, indices=indices, axis=2, validate_indices=False
        )
        return mb.reduce_sum(x=gathered, axes=[1])

    return prog


def convert_and_time(label, prog, input_name, input_array, rounds, deployment_target):
    print(f"[{label}] convert_start t={now_ms()}", flush=True)
    model = ct.convert(
        prog,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=deployment_target,
    )
    print(f"[{label}] convert_done t={now_ms()}", flush=True)
    for i in range(3):
        model.predict({input_name: input_array})
        print(f"[{label}] warmup={i + 1}/3 t={now_ms()}", flush=True)
    t0 = time.perf_counter()
    for i in range(rounds):
        model.predict({input_name: input_array})
        if (i + 1) % max(1, rounds // 4) == 0:
            print(f"[{label}] round={i + 1}/{rounds} t={now_ms()}", flush=True)
    elapsed = time.perf_counter() - t0
    us_per = elapsed * 1e6 / rounds
    logical_bytes = input_array.nbytes
    print(
        f"[{label}] ok rounds={rounds} us_per={us_per:.3f} "
        f"input_MB={logical_bytes / 1e6:.3f}",
        flush=True,
    )
    return us_per


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dim", type=int, default=4096)
    parser.add_argument("--output-dim", type=int, default=4096)
    parser.add_argument("--codes", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--opset", default="iOS26")
    parser.add_argument("--deployment", default="macOS26")
    parser.add_argument("--skip-b1", action="store_true")
    parser.add_argument("--skip-batched", action="store_true")
    args = parser.parse_args()

    if args.input_dim % 8 != 0:
        raise SystemExit("--input-dim must be divisible by 8")

    opset_target = target_from_name(args.opset)
    deployment_target = target_from_name(args.deployment)

    print(
        f"[config] input_dim={args.input_dim} output_dim={args.output_dim} "
        f"codes={args.codes} rounds={args.rounds} batch={args.batch} "
        f"opset={args.opset} deployment={args.deployment}",
        flush=True,
    )

    if not args.skip_b1:
        b1_input = np.random.randn(1, args.input_dim).astype(np.float16)
        b1_prog = build_b1_program(
            args.input_dim, args.output_dim, args.codes, opset_target
        )
        convert_and_time("b1", b1_prog, "x", b1_input, args.rounds, deployment_target)

    if not args.skip_batched:
        batch_input = np.random.randn(args.batch, args.input_dim).astype(np.float16)
        batched_prog = build_batched_program(
            args.batch, args.input_dim, args.output_dim, args.codes, opset_target
        )
        convert_and_time(
            "batched", batched_prog, "x", batch_input, args.rounds, deployment_target
        )


if __name__ == "__main__":
    main()
