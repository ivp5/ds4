#!/usr/bin/env python3
"""Encode DS4 routed experts into DS4D8F1 layer files.

This is the rebuild path for H3371-style allocation artifacts.  It writes the
exact record/table/payload format consumed by `ds4_d8f_reader.c` and Metal:

- projection-major 3 x 256 record table, 64 bytes per record.
- VQ-D8 blocks over `W.reshape(-1, 8)`.
- fp16 codebook payloads and little-endian bitpacked code indices with a
  4-byte guard so the runtime's final uint32 word read remains in-bounds.
- down records optionally precondition columns by deployed SwiGLU hidden RMS and
  store fp16 inverse column scale in the D8F scale payload.

Use `--experts` for a surgical canary before full-layer encoding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LIB_ROOT = REPO_ROOT / "tmp" / "20260531_codec_coherence"
if str(LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(LIB_ROOT))

import ds4_mlx_lib as mlx_lib  # noqa: E402


FUSED_MAGIC = b"DS4D8F1\0"
HEADER_BYTES = 4096
EXPERTS = 256
PROJECTIONS = {"gate": 0, "up": 1, "down": 2}
PROJECTION_NAMES = {v: k for k, v in PROJECTIONS.items()}
SOURCE_STEMS = {"gate": "w1", "up": "w3", "down": "w2"}
K_FIELDS = {"gate": "K_gate", "up": "K_up", "down": "K_down"}
RECORD_BYTES = 64
RECORD_STRUCT = struct.Struct("<IIIIIIQQQIIII")
GROUP = 8
INDEX_GUARD_BYTES = 4
SHAPES = {
    "gate": (2048, 4096),
    "up": (2048, 4096),
    "down": (4096, 2048),
}
DEFAULT_ALLOCATION = REPO_ROOT / "tmp" / "20260604_quant_runtime" / "h3371_reconstructed" / "ds4_52gb_allocation_h3371_general_fit_no_overlay.npy"
DEFAULT_ACTS = Path("/Users/silv/cl/tlp_codex/deployed_acts_richcalib_8192tok_20260530T231858.npz")


def align_handle(handle, alignment: int = 16) -> int:
    position = handle.tell()
    padding = (-position) % alignment
    if padding:
        handle.write(b"\0" * padding)
        position += padding
    return position


def append_aligned(handle, payload: bytes, alignment: int = 16) -> int:
    position = align_handle(handle, alignment)
    handle.write(payload)
    return position


def rewrite_header(handle, header: dict[str, Any]) -> None:
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if 16 + len(raw) > HEADER_BYTES:
        raise ValueError(f"header too large: {len(raw)} bytes")
    handle.seek(0)
    handle.write(FUSED_MAGIC)
    handle.write(struct.pack("<II", 1, len(raw)))
    handle.write(raw)
    handle.write(b"\0" * (HEADER_BYTES - 16 - len(raw)))


def empty_record_table() -> bytearray:
    table = bytearray(RECORD_BYTES * len(PROJECTIONS) * EXPERTS)
    for projection in range(len(PROJECTIONS)):
        for expert in range(EXPERTS):
            offset = (projection * EXPERTS + expert) * RECORD_BYTES
            RECORD_STRUCT.pack_into(table, offset, projection, expert, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    return table


def bits_for_k(k: int) -> int:
    if k not in (256, 512, 1024, 2048, 4096):
        raise ValueError(f"unsupported K={k}")
    return int(math.ceil(math.log2(k)))


def allocation_lookup(path: Path) -> dict[tuple[int, int], np.void]:
    allocation = np.load(path, allow_pickle=False)
    result: dict[tuple[int, int], np.void] = {}
    for row in allocation:
        result[(int(row["layer"]), int(row["expert"]))] = row
    return result


def source_weight(layer: int, projection: str, expert: int) -> mx.array:
    stem = f"layers.{layer}.ffn.experts.{expert}.{SOURCE_STEMS[projection]}"
    weight = mlx_lib.deq_fp4(stem)
    wanted = SHAPES[projection]
    if weight.shape == wanted:
        return weight
    if weight.T.shape == wanted:
        return weight.T
    raise ValueError(f"{stem}: shape {weight.shape} incompatible with {wanted}")


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def down_column_scale(layer: int, expert: int, acts: np.lib.npyio.NpzFile, tokens: int, mode: str) -> mx.array:
    acts_np = acts[f"act_L{layer}"].astype(np.float32, copy=False)[:tokens]
    acts_mx = mx.array(acts_np)
    gate = source_weight(layer, "gate", expert)
    up = source_weight(layer, "up", expert)
    gate_act = acts_mx @ gate.T
    up_act = acts_mx @ up.T
    if mode == "deployed":
        gate_act = mx.minimum(gate_act, 10.0)
        up_act = mx.clip(up_act, -10.0, 10.0)
    elif mode != "unclamped":
        raise ValueError(f"unknown down scale mode: {mode}")
    hidden = silu(gate_act) * up_act
    scale = mx.sqrt(mx.mean(hidden * hidden, axis=0)) + 1e-9
    mx.eval(scale)
    return scale


def lloyd(points: mx.array, k: int, *, iterations: int, fit_samples: int, seed: int) -> mx.array:
    rows = points.shape[0]
    sample_count = min(rows, max(k, fit_samples))
    rng = np.random.default_rng(seed)
    sample_idx = mx.array(rng.choice(rows, sample_count, replace=False).astype(np.int32))
    sample = points[sample_idx]
    centroid_idx = mx.array(rng.choice(sample_count, k, replace=False).astype(np.int32))
    centroids = sample[centroid_idx]
    mx.eval(centroids)
    code_ids = mx.arange(k)
    for _ in range(iterations):
        distances = -2.0 * (sample @ centroids.T) + mx.sum(centroids * centroids, axis=1)[None, :]
        assign = mx.argmin(distances, axis=1)
        mx.eval(assign)
        one_hot = (assign[:, None] == code_ids[None, :]).astype(mx.float32)
        counts = mx.maximum(mx.sum(one_hot, axis=0)[:, None], 1.0)
        centroids = (one_hot.T @ sample) / counts
        mx.eval(centroids)
    return centroids


def assign(points: mx.array, centroids: mx.array, chunk: int) -> np.ndarray:
    norm = mx.sum(centroids * centroids, axis=1)
    out: list[mx.array] = []
    for start in range(0, points.shape[0], chunk):
        part = points[start:start + chunk]
        codes = mx.argmin(-2.0 * (part @ centroids.T) + norm[None, :], axis=1)
        mx.eval(codes)
        out.append(codes)
    merged = mx.concatenate(out)
    mx.eval(merged)
    return np.asarray(merged, dtype=np.uint16)


def bitpack_codes(codes: np.ndarray, bits: int) -> bytes:
    codes64 = codes.astype(np.uint64, copy=False)
    count = codes64.shape[0]
    payload = np.zeros(((count * bits + 7) // 8) + INDEX_GUARD_BYTES, dtype=np.uint8)
    bit_offsets = np.arange(count, dtype=np.uint64) * np.uint64(bits)
    byte_offsets = (bit_offsets >> np.uint64(3)).astype(np.int64)
    shifts = bit_offsets & np.uint64(7)
    values = codes64 << shifts
    byte_count = (bits + 7) // 8 + 1
    for byte_index in range(byte_count):
        np.bitwise_or.at(payload, byte_offsets + byte_index, ((values >> np.uint64(8 * byte_index)) & np.uint64(0xff)).astype(np.uint8))
    return payload.tobytes()


def encode_weight(weight: mx.array, k: int, *, scale: mx.array | None, iterations: int, fit_samples: int, assign_chunk: int, seed: int) -> tuple[bytes, bytes, bytes, dict[str, float]]:
    working = weight * scale[None, :] if scale is not None else weight
    points = working.reshape(-1, GROUP).astype(mx.float32)
    centroids = lloyd(points, k, iterations=iterations, fit_samples=fit_samples, seed=seed)
    codes = assign(points, centroids, assign_chunk)
    centroids_np = np.asarray(centroids, dtype=np.float32)
    codebook = centroids_np.astype(np.float16).tobytes()
    packed = bitpack_codes(codes, bits_for_k(k))
    metrics: dict[str, float] = {
        "codebook_max_abs": float(np.max(np.abs(centroids_np))),
        "codebook_rms": float(np.sqrt(np.mean(centroids_np * centroids_np))),
        "unique_codes": float(np.unique(codes).shape[0]),
    }
    scale_payload = b""
    if scale is not None:
        scale_np = np.asarray(scale, dtype=np.float32)
        inv_scale = (1.0 / np.maximum(scale_np, 1e-9)).astype(np.float16)
        scale_payload = inv_scale.tobytes()
        metrics.update({
            "scale_min": float(scale_np.min()),
            "scale_max": float(scale_np.max()),
            "scale_mean": float(scale_np.mean()),
        })
    return codebook, packed, scale_payload, metrics


def write_record(table: bytearray, projection: str, expert: int, k: int, codebook_offset: int, index_offset: int, scale_offset: int, codebook_bytes: int, index_bytes: int, scale_bytes: int) -> None:
    projection_id = PROJECTIONS[projection]
    offset = (projection_id * EXPERTS + expert) * RECORD_BYTES
    flags = 1 if projection == "down" and scale_bytes else 0
    RECORD_STRUCT.pack_into(
        table,
        offset,
        projection_id,
        expert,
        k,
        bits_for_k(k),
        GROUP,
        0,
        codebook_offset,
        index_offset,
        scale_offset,
        codebook_bytes,
        index_bytes,
        scale_bytes,
        flags,
    )


def parse_experts(text: str) -> list[int]:
    if text.lower() == "all":
        return list(range(EXPERTS))
    experts = [int(x) for x in text.split(",") if x.strip()]
    for expert in experts:
        if expert < 0 or expert >= EXPERTS:
            raise ValueError(f"expert out of range: {expert}")
    return experts


def parse_projections(text: str) -> list[str]:
    projections = [x.strip() for x in text.split(",") if x.strip()]
    for projection in projections:
        if projection not in PROJECTIONS:
            raise ValueError(f"unknown projection: {projection}")
    return projections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allocation", type=Path, default=DEFAULT_ALLOCATION)
    parser.add_argument("--acts", type=Path, default=DEFAULT_ACTS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--experts", default="all")
    parser.add_argument("--projections", default="gate,up,down")
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--down-scale-mode", choices=("deployed", "unclamped"), default="deployed")
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--fit-samples", type=int, default=120000)
    parser.add_argument("--assign-chunk", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    experts = parse_experts(args.experts)
    projections = parse_projections(args.projections)
    allocation = allocation_lookup(args.allocation)
    acts = np.load(args.acts) if "down" in projections else None
    started = time.time()
    table = empty_record_table()
    payload_hash = hashlib.sha256()
    encoded: list[dict[str, Any]] = []

    with args.out.open("wb") as handle:
        handle.write(b"\0" * (HEADER_BYTES + len(table)))
        for expert in experts:
            row = allocation.get((args.layer, expert))
            if row is None:
                raise ValueError(f"missing allocation row for L{args.layer} expert {expert}")
            down_scale_cache: mx.array | None = None
            for projection in projections:
                k = int(row[K_FIELDS[projection]])
                t0 = time.time()
                weight = source_weight(args.layer, projection, expert)
                scale = None
                if projection == "down":
                    if down_scale_cache is None:
                        if acts is None:
                            raise ValueError("down projection requested without acts")
                        down_scale_cache = down_column_scale(args.layer, expert, acts, args.tokens, args.down_scale_mode)
                    scale = down_scale_cache
                codebook, index, scale_payload, metrics = encode_weight(
                    weight,
                    k,
                    scale=scale,
                    iterations=args.iters,
                    fit_samples=args.fit_samples,
                    assign_chunk=args.assign_chunk,
                    seed=args.seed + args.layer * 1000 + expert * 10 + PROJECTIONS[projection],
                )
                codebook_offset = append_aligned(handle, codebook)
                index_offset = append_aligned(handle, index)
                scale_offset = append_aligned(handle, scale_payload) if scale_payload else 0
                payload_hash.update(codebook)
                payload_hash.update(index)
                payload_hash.update(scale_payload)
                write_record(
                    table,
                    projection,
                    expert,
                    k,
                    codebook_offset,
                    index_offset,
                    scale_offset,
                    len(codebook),
                    len(index),
                    len(scale_payload),
                )
                entry = {
                    "layer": args.layer,
                    "expert": expert,
                    "projection": projection,
                    "k": k,
                    "bits": bits_for_k(k),
                    "codebook_bytes": len(codebook),
                    "index_bytes": len(index),
                    "scale_bytes": len(scale_payload),
                    "elapsed_sec": time.time() - t0,
                    **metrics,
                }
                encoded.append(entry)
                print(json.dumps(entry, sort_keys=True), flush=True)
        handle.seek(HEADER_BYTES)
        handle.write(table)
        final_size = handle.seek(0, os.SEEK_END)
        manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
        encoded_sample = encoded if len(encoded) <= 8 else encoded[:8]
        expert_header: list[int] | str = experts if len(experts) <= 64 else "all"
        header = {
            "format": "DS4D8F",
            "version": 1,
            "codec": "direct-FP4-to-VQ-D8",
            "target": "h3371_layer_encode_canary" if len(experts) < EXPERTS or len(projections) < 3 else "h3371_layer_encode",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "layer": args.layer,
            "experts": EXPERTS,
            "encoded_experts": expert_header,
            "encoded_expert_count": len(experts),
            "encoded_projections": projections,
            "encoded_record_count": len(encoded),
            "header_bytes": HEADER_BYTES,
            "table_offset": HEADER_BYTES,
            "fused_record_bytes": RECORD_BYTES,
            "record_order": "projection_major_then_expert",
            "record_struct": RECORD_STRUCT.format,
            "projections": ["gate", "up", "down"],
            "allocation": os.fspath(args.allocation),
            "acts": os.fspath(args.acts) if acts is not None else None,
            "tokens": args.tokens,
            "down_scale_mode": args.down_scale_mode,
            "iters": args.iters,
            "fit_samples": args.fit_samples,
            "assign_chunk": args.assign_chunk,
            "seed": args.seed,
            "index_guard_bytes": INDEX_GUARD_BYTES,
            "payload_sha256": payload_hash.hexdigest(),
            "encoded_sample": encoded_sample,
            "manifest": os.fspath(manifest_path),
            "elapsed_sec": time.time() - started,
            "file_bytes": final_size,
        }
        rewrite_header(handle, header)
    manifest = dict(header)
    manifest["encoded_experts"] = experts
    manifest["encoded"] = encoded
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": os.fspath(args.out), "records": len(encoded), "file_bytes": args.out.stat().st_size, "elapsed_sec": time.time() - started}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
