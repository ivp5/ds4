#!/usr/bin/env python3
"""Small cached safetensors reader for DS4 source-oracle tools.

The runtime work needs one general source view over the upstream HF shards:
tensor name -> shard header -> byte span -> typed/dequantized numeric view.
This module keeps that surface in one place so margin probes, codec comparators,
and future source-streaming can share the same offset/dtype rules.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import struct
from typing import Any

import numpy as np


DEFAULT_MODEL_DIR = Path("/Users/silv/cl/tlp/montyneg/ds4/DeepSeek-V4-Flash")
E4M3FN_EXP_SCALE = np.array(
    [
        0.0,
        0.015625,
        0.03125,
        0.0625,
        0.125,
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
        64.0,
        128.0,
        256.0,
    ],
    dtype=np.float32,
)
PACKED_FP4_TABLE = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


@dataclass(frozen=True)
class SafetensorSpan:
    name: str
    shard: Path
    data_base: int
    start: int
    end: int
    shape: tuple[int, ...]
    dtype: str

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    @property
    def absolute_start(self) -> int:
        return self.data_base + self.start


class SafetensorStore:
    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR):
        self.model_dir = Path(model_dir)
        with (self.model_dir / "model.safetensors.index.json").open("r", encoding="utf-8") as handle:
            self.weight_map: dict[str, str] = json.load(handle)["weight_map"]
        self.headers: dict[str, tuple[dict[str, Any], int]] = {}

    def header(self, shard_name: str) -> tuple[dict[str, Any], int]:
        cached = self.headers.get(shard_name)
        if cached is not None:
            return cached
        shard = self.model_dir / shard_name
        with shard.open("rb") as handle:
            header_bytes = struct.unpack("<Q", handle.read(8))[0]
            cached = (json.loads(handle.read(header_bytes)), 8 + header_bytes)
        self.headers[shard_name] = cached
        return cached

    def span(self, name: str) -> SafetensorSpan:
        shard_name = self.weight_map[name]
        header, data_base = self.header(shard_name)
        meta = header[name]
        start, end = meta["data_offsets"]
        return SafetensorSpan(
            name=name,
            shard=self.model_dir / shard_name,
            data_base=data_base,
            start=int(start),
            end=int(end),
            shape=tuple(int(dim) for dim in meta["shape"]),
            dtype=str(meta["dtype"]),
        )

    def raw(self, name: str) -> tuple[bytes, tuple[int, ...], str]:
        span = self.span(name)
        with span.shard.open("rb") as handle:
            handle.seek(span.absolute_start)
            return handle.read(span.nbytes), span.shape, span.dtype

    def numeric(self, name: str) -> np.ndarray:
        raw, shape, dtype = self.raw(name)
        return numeric_from_raw(raw, shape, dtype, name)

    def rows(self, name: str, rows: np.ndarray) -> np.ndarray:
        span = self.span(name)
        if len(span.shape) != 2:
            raise ValueError(f"{name}: row loader expects a matrix, got shape={span.shape}")
        if span.dtype not in {"BF16", "F16", "F32"}:
            raise ValueError(f"{name}: row loader supports BF16/F16/F32, got {span.dtype}")
        item_bytes = {"BF16": 2, "F16": 2, "F32": 4}[span.dtype]
        row_bytes = span.shape[1] * item_bytes
        out: list[np.ndarray] = []
        with span.shard.open("rb") as handle:
            for row in rows.astype(np.int64).tolist():
                handle.seek(span.absolute_start + int(row) * row_bytes)
                out.append(numeric_from_raw(handle.read(row_bytes), (span.shape[1],), span.dtype, name))
        return np.stack(out, axis=0)

    def packed_fp4(self, stem: str) -> np.ndarray:
        weight_raw, weight_shape, weight_dtype = self.raw(f"{stem}.weight")
        scale_raw, scale_shape, scale_dtype = self.raw(f"{stem}.scale")
        return dequant_packed_fp4(weight_raw, weight_shape, weight_dtype, scale_raw, scale_shape, scale_dtype, stem)

    def fp8(self, stem: str) -> np.ndarray:
        weight_raw, weight_shape, weight_dtype = self.raw(f"{stem}.weight")
        scale_raw, scale_shape, scale_dtype = self.raw(f"{stem}.scale")
        return dequant_fp8(weight_raw, weight_shape, weight_dtype, scale_raw, scale_shape, scale_dtype, stem)

    def source_weight(self, stem: str) -> np.ndarray:
        dtype = self.span(f"{stem}.weight").dtype
        if dtype == "I8":
            return self.packed_fp4(stem)
        if dtype == "F8_E4M3":
            return self.fp8(stem)
        return self.numeric(f"{stem}.weight")

    def dtype_summary(self) -> tuple[dict[str, int], dict[str, int]]:
        counts: dict[str, int] = {}
        bytes_by_dtype: dict[str, int] = {}
        for name in self.weight_map:
            span = self.span(name)
            counts[span.dtype] = counts.get(span.dtype, 0) + 1
            bytes_by_dtype[span.dtype] = bytes_by_dtype.get(span.dtype, 0) + span.nbytes
        return counts, bytes_by_dtype


def e8m0_to_f32(raw: np.ndarray) -> np.ndarray:
    return np.exp2(raw.astype(np.float32) - 127.0)


def e4m3fn_to_f32(raw: np.ndarray) -> np.ndarray:
    value = raw.astype(np.uint8)
    sign = np.where((value >> 7) != 0, -1.0, 1.0).astype(np.float32)
    exp = (value >> 3) & 0x0F
    mant = value & 0x07
    subnormal = mant.astype(np.float32) * np.float32(0.001953125)
    normal = (1.0 + mant.astype(np.float32) * np.float32(0.125)) * E4M3FN_EXP_SCALE[exp]
    return sign * np.where(exp == 0, subnormal, normal)


def numeric_from_raw(raw: bytes, shape: tuple[int, ...], dtype: str, name: str = "tensor") -> np.ndarray:
    if dtype == "BF16":
        return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)
    if dtype == "I64":
        return np.frombuffer(raw, dtype=np.int64).reshape(shape)
    if dtype == "I8":
        return np.frombuffer(raw, dtype=np.int8).reshape(shape)
    if dtype == "F8_E4M3":
        return e4m3fn_to_f32(np.frombuffer(raw, dtype=np.uint8)).reshape(shape)
    if dtype == "F8_E8M0":
        return e8m0_to_f32(np.frombuffer(raw, dtype=np.uint8)).reshape(shape)
    raise ValueError(f"{name}: unsupported safetensors dtype {dtype}")


def dequant_packed_fp4(
    weight_raw: bytes,
    weight_shape: tuple[int, ...],
    weight_dtype: str,
    scale_raw: bytes,
    scale_shape: tuple[int, ...],
    scale_dtype: str,
    name: str = "packed-fp4",
) -> np.ndarray:
    if weight_dtype != "I8" or scale_dtype != "F8_E8M0":
        raise ValueError(f"{name}: expected I8 weight + F8_E8M0 scale, got {weight_dtype}/{scale_dtype}")
    if len(weight_shape) != 2 or len(scale_shape) != 2:
        raise ValueError(f"{name}: packed FP4 expects 2D tensors")
    out_dim, packed_in = weight_shape
    in_dim = packed_in * 2
    if in_dim % 32 != 0:
        raise ValueError(f"{name}: packed FP4 in_dim {in_dim} is not divisible by 32")
    if scale_shape != (out_dim, in_dim // 32):
        raise ValueError(f"{name}: scale shape {scale_shape} does not match {(out_dim, in_dim // 32)}")
    packed = np.frombuffer(weight_raw, dtype=np.uint8).reshape(weight_shape)
    out = np.empty((out_dim, in_dim), dtype=np.float32)
    out[:, 0::2] = PACKED_FP4_TABLE[packed & 0x0F]
    out[:, 1::2] = PACKED_FP4_TABLE[packed >> 4]
    scales = np.repeat(e8m0_to_f32(np.frombuffer(scale_raw, dtype=np.uint8).reshape(scale_shape)), 32, axis=1)
    return out * scales


def dequant_fp8(
    weight_raw: bytes,
    weight_shape: tuple[int, ...],
    weight_dtype: str,
    scale_raw: bytes,
    scale_shape: tuple[int, ...],
    scale_dtype: str,
    name: str = "fp8",
) -> np.ndarray:
    if weight_dtype != "F8_E4M3" or scale_dtype != "F8_E8M0":
        raise ValueError(f"{name}: expected F8_E4M3 weight + F8_E8M0 scale, got {weight_dtype}/{scale_dtype}")
    if len(weight_shape) != 2 or len(scale_shape) != 2:
        raise ValueError(f"{name}: FP8 expects 2D tensors")
    out_dim, in_dim = weight_shape
    expected_scale_shape = ((out_dim + 127) // 128, (in_dim + 127) // 128)
    if scale_shape != expected_scale_shape:
        raise ValueError(f"{name}: scale shape {scale_shape} does not match {expected_scale_shape}")
    weight = e4m3fn_to_f32(np.frombuffer(weight_raw, dtype=np.uint8)).reshape(weight_shape)
    scales = e8m0_to_f32(np.frombuffer(scale_raw, dtype=np.uint8).reshape(scale_shape))
    scales = np.repeat(np.repeat(scales, 128, axis=0), 128, axis=1)[:out_dim, :in_dim]
    return weight * scales


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--tensor")
    args = parser.parse_args()

    store = SafetensorStore(args.model_dir)
    if args.summary:
        counts, bytes_by_dtype = store.dtype_summary()
        print(json.dumps({"dtype_counts": counts, "bytes_by_dtype": bytes_by_dtype}, indent=2, sort_keys=True))
    if args.tensor:
        span = store.span(args.tensor)
        print(json.dumps({
            "name": span.name,
            "shard": span.shard.name,
            "offset": span.absolute_start,
            "bytes": span.nbytes,
            "shape": span.shape,
            "dtype": span.dtype,
        }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
