#!/usr/bin/env python3
"""Repack an existing DS4 M1R pack with bit-packed scale-index planes.

This is the runtime-native successor to the obsolete one-time CDX3/VQB2 pack
experiments: it reads an M1R pack, copies codebooks and index planes unchanged,
and only changes the per-expert scale-index planes from 8 bits/code to a chosen
bit width. For scale_bits=7 the full DS4 Flash routed pack saves ~256 MiB,
enough to cross the strict 52 GiB product envelope without touching VQ indices.
"""
from __future__ import annotations

import argparse
import json
import mmap
import pathlib
import struct
import sys
import time

import numpy as np

MAGIC = b"DS4M1R\0\0"
VERSION = 3
N_EXPERTS = 256
MAX_LAYERS = 43
MAX_KINDS = 3
DEFAULT_ALIGN = 16 * 1024
SECTION_TABLE_MAGIC = b"M1RSECT\0"
SECTION_TABLE_VERSION = 1
SECTION_TABLE_HEADER_OFFSET = 128
SECTION_TABLE_RECORD_OFFSET = 256
SECTION_TABLE_HEADER_FMT = "<8sIIII"
SECTION_TABLE_RECORD_FMT = "<12I2f7Q"
SECTION_TABLE_RECORD_BYTES = struct.calcsize(SECTION_TABLE_RECORD_FMT)
EXT_MAGIC = b"M1REXT\0\0"
EXT_HEADER_OFFSET = 80
EXT_HEADER_FMT = "<8sQQII"
SCALE_LP_RECORD_BYTES = 8
KIND_NAMES = {0: "gate", 1: "up", 2: "down"}


def align_up(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a if a > 1 else x


def parse_layers(text: str | None, available: set[int]) -> list[int]:
    if not text or text == "all":
        return sorted(available)
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    out = sorted(set(out))
    missing = [x for x in out if x not in available]
    if missing:
        raise SystemExit(f"requested layers not present in source: {missing}")
    return out


def u32(base: bytes | mmap.mmap, off: int) -> int:
    return struct.unpack_from("<I", base, off)[0]


def u64(base: bytes | mmap.mmap, off: int) -> int:
    return struct.unpack_from("<Q", base, off)[0]


def read_m1r_sections(src: mmap.mmap, size: int) -> tuple[dict[tuple[int, int], dict], int, int, int, int, int]:
    if size < 4096 or src[:8] != MAGIC:
        raise SystemExit("not an M1R pack")
    source_version = u32(src, 8)
    align = u32(src, 32) or DEFAULT_ALIGN
    source_size = u64(src, 68) if size >= 76 else size
    if src[SECTION_TABLE_HEADER_OFFSET:SECTION_TABLE_HEADER_OFFSET + 8] != SECTION_TABLE_MAGIC:
        raise SystemExit("missing M1R section table")
    table_version = u32(src, SECTION_TABLE_HEADER_OFFSET + 8)
    record_offset = u32(src, SECTION_TABLE_HEADER_OFFSET + 12)
    record_bytes = u32(src, SECTION_TABLE_HEADER_OFFSET + 16)
    record_count = u32(src, SECTION_TABLE_HEADER_OFFSET + 20)
    if table_version != SECTION_TABLE_VERSION or record_bytes != SECTION_TABLE_RECORD_BYTES:
        raise SystemExit(f"unsupported section table v={table_version} record_bytes={record_bytes}")
    scale_lp_offset = u64(src, EXT_HEADER_OFFSET + 8)
    scale_lp_bytes = u64(src, EXT_HEADER_OFFSET + 16)
    scale_lp_sections = u32(src, EXT_HEADER_OFFSET + 24)
    scale_lp_experts = u32(src, EXT_HEADER_OFFSET + 28)
    if scale_lp_experts != N_EXPERTS or scale_lp_sections < record_count:
        raise SystemExit("invalid scale-log table")
    records: dict[tuple[int, int], dict] = {}
    for i in range(record_count):
        raw = struct.unpack_from(SECTION_TABLE_RECORD_FMT, src, record_offset + i * record_bytes)
        rec = dict(
            layer=raw[0], kind=raw[1], k=raw[2], bits=raw[3], out_dim=raw[4], in_dim=raw[5],
            n_indices=raw[6], scale_count=raw[7], scale_stride=raw[8], index_bytes=raw[9],
            index_stride=raw[10], section_index=(raw[11] & 0xffff), source_scale_bits=(raw[11] >> 16) or 8,
            scale_log_min=raw[12], scale_log_step=raw[13],
            codebook_offset=raw[14], codebook_bytes=raw[15], scale_offset=raw[16], scale_bytes=raw[17],
            index_offset=raw[18], index_plane_bytes=raw[19], section_bytes=raw[20],
        )
        if rec["codebook_offset"] + rec["codebook_bytes"] > size:
            raise SystemExit(f"bad codebook span in record {i}")
        if rec["scale_offset"] + rec["scale_bytes"] > size:
            raise SystemExit(f"bad scale span in record {i}")
        if rec["index_offset"] + rec["index_plane_bytes"] > size:
            raise SystemExit(f"bad index span in record {i}")
        records[(rec["layer"], rec["kind"])] = rec
    return records, align, source_version, source_size, scale_lp_offset, scale_lp_bytes


def copy_from_mmap(out_f, src: mmap.mmap, off: int, n: int, chunk: int = 32 * 1024 * 1024) -> None:
    pos = off
    end = off + n
    while pos < end:
        step = min(chunk, end - pos)
        out_f.write(src[pos:pos + step])
        pos += step


def write_padding(out_f, align: int) -> int:
    pos = out_f.tell()
    want = align_up(pos, align)
    if want > pos:
        out_f.write(b"\0" * (want - pos))
    return want


def pack_7bit_scale_plane(src: mmap.mmap, rec: dict, scale_lp_old: memoryview, scale_bits: int) -> tuple[bytes, list[tuple[float, float]], dict]:
    if rec["source_scale_bits"] != 8:
        raise SystemExit("source scale planes must be 8-bit for v3 repack")
    levels = (1 << scale_bits) - 1
    scale_count = int(rec["scale_count"])
    scale_stride = int(rec["scale_stride"])
    base = int(rec["scale_offset"])
    rows = np.empty((N_EXPERTS, scale_count), dtype=np.uint8)
    for expert in range(N_EXPERTS):
        start = base + expert * scale_stride
        rows[expert, :] = np.frombuffer(src[start:start + scale_count], dtype=np.uint8)
    mins = rows.min(axis=1).astype(np.uint16)
    maxs = rows.max(axis=1).astype(np.uint16)
    span = maxs - mins
    denom = np.where(span == 0, 1, span).astype(np.float32)
    shifted = rows.astype(np.int16) - mins[:, None].astype(np.int16)
    q = np.rint(shifted.astype(np.float32) * float(levels) / denom[:, None]).astype(np.uint8)
    q = np.minimum(q, levels).astype(np.uint8)
    bit_planes = ((q[:, :, None] >> np.arange(scale_bits, dtype=np.uint8)) & 1).astype(np.uint8)
    packed = np.packbits(bit_planes.reshape(N_EXPERTS, scale_count * scale_bits), axis=1, bitorder="little")
    raw_stride = (scale_count * scale_bits + 7) // 8
    out_stride = align_up(raw_stride + 4, 64)
    out = bytearray(out_stride * N_EXPERTS)
    for expert in range(N_EXPERTS):
        off = expert * out_stride
        out[off:off + raw_stride] = packed[expert, :raw_stride].tobytes()
    lp_new: list[tuple[float, float]] = []
    max_abs_log_err = 0.0
    sum_abs_log_err = 0.0
    total_codes = N_EXPERTS * scale_count
    for expert in range(N_EXPERTS):
        old_min, old_step = struct.unpack_from("<2f", scale_lp_old, (rec["section_index"] * N_EXPERTS + expert) * SCALE_LP_RECORD_BYTES)
        new_min = old_min + float(mins[expert]) * old_step
        new_step = 0.0 if span[expert] == 0 else (float(span[expert]) * old_step) / float(levels)
        lp_new.append((new_min, new_step))
        old_log = old_min + rows[expert].astype(np.float32) * old_step
        new_log = new_min + q[expert].astype(np.float32) * new_step
        err = np.abs(old_log - new_log)
        max_abs_log_err = max(max_abs_log_err, float(err.max(initial=0.0)))
        sum_abs_log_err += float(err.sum(dtype=np.float64))
    stats = dict(
        source_scale_bits=8,
        scale_bits=scale_bits,
        raw_stride=raw_stride,
        scale_stride=out_stride,
        min_code_min=int(mins.min()),
        min_code_max=int(mins.max()),
        max_code_min=int(maxs.min()),
        max_code_max=int(maxs.max()),
        max_abs_log_err=max_abs_log_err,
        mean_abs_log_err=sum_abs_log_err / float(total_codes),
    )
    return bytes(out), lp_new, stats


def write_section_table(out_f, meta: list[dict], header_bytes: int) -> None:
    table_end = SECTION_TABLE_RECORD_OFFSET + len(meta) * SECTION_TABLE_RECORD_BYTES
    if table_end > header_bytes:
        raise SystemExit(f"section table too large: {table_end} > {header_bytes}")
    out_f.seek(SECTION_TABLE_HEADER_OFFSET)
    out_f.write(struct.pack(SECTION_TABLE_HEADER_FMT, SECTION_TABLE_MAGIC, SECTION_TABLE_VERSION,
                            SECTION_TABLE_RECORD_OFFSET, SECTION_TABLE_RECORD_BYTES, len(meta)))
    out_f.seek(SECTION_TABLE_RECORD_OFFSET)
    for section_index, m in enumerate(meta):
        reserved = (int(m["scale_bits"]) << 16) | section_index
        out_f.write(struct.pack(
            SECTION_TABLE_RECORD_FMT,
            int(m["layer"]), int(m["kind"]), int(m["k"]), int(m["bits"]),
            int(m["out_dim"]), int(m["in_dim"]), int(m["n_indices"]), int(m["scale_count"]),
            int(m["scale_stride"]), int(m["index_bytes"]), int(m["index_stride"]), reserved,
            float(m["scale_log_min"]), float(m["scale_log_step"]),
            int(m["codebook_offset"]), int(m["codebook_bytes"]),
            int(m["scale_offset"]), int(m["scale_bytes"]),
            int(m["index_offset"]), int(m["index_plane_bytes"]),
            int(m["section_bytes"]),
        ))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-pack", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--scale-bits", type=int, default=7, choices=[4, 5, 6, 7, 8])
    ap.add_argument("--layers", default="all")
    ap.add_argument("--align", type=int, default=0, help="0 means reuse source align")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--status-every", type=int, default=1)
    args = ap.parse_args()
    if args.out.exists() and not args.force:
        raise SystemExit(f"output exists; pass --force: {args.out}")
    started = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.in_pack.open("rb") as in_f:
        src = mmap.mmap(in_f.fileno(), 0, access=mmap.ACCESS_READ)
        source_size = args.in_pack.stat().st_size
        records, source_align, source_version, indexed_source_size, scale_lp_offset, scale_lp_bytes = read_m1r_sections(src, source_size)
        layers = parse_layers(args.layers, {k[0] for k in records})
        align = args.align or source_align or DEFAULT_ALIGN
        header_bytes = align_up(4096 + MAX_LAYERS * MAX_KINDS * 128, align)
        scale_lp_old = memoryview(src)[scale_lp_offset:scale_lp_offset + scale_lp_bytes]
        meta: list[dict] = []
        scale_lp_new: list[tuple[float, float]] = []
        stats: list[dict] = []
        with args.out.open("wb+") as out_f:
            out_f.write(b"\0" * header_bytes)
            section_counter = 0
            for layer in layers:
                for kind in range(MAX_KINDS):
                    rec = records.get((layer, kind))
                    if rec is None:
                        raise SystemExit(f"missing source section L{layer} kind={kind}")
                    write_padding(out_f, align)
                    cb_out = out_f.tell()
                    copy_from_mmap(out_f, src, rec["codebook_offset"], rec["codebook_bytes"])
                    write_padding(out_f, align)
                    scale_out = out_f.tell()
                    if args.scale_bits == 8:
                        scale_payload = src[rec["scale_offset"]:rec["scale_offset"] + rec["scale_bytes"]]
                        lp = [struct.unpack_from("<2f", scale_lp_old, (rec["section_index"] * N_EXPERTS + e) * SCALE_LP_RECORD_BYTES) for e in range(N_EXPERTS)]
                        scale_stride = rec["scale_stride"]
                        st = dict(source_scale_bits=rec["source_scale_bits"], scale_bits=8,
                                  raw_stride=rec["scale_count"], scale_stride=scale_stride,
                                  max_abs_log_err=0.0, mean_abs_log_err=0.0)
                    else:
                        scale_payload, lp, st = pack_7bit_scale_plane(src, rec, scale_lp_old, args.scale_bits)
                        scale_stride = st["scale_stride"]
                    out_f.write(scale_payload)
                    write_padding(out_f, align)
                    index_out = out_f.tell()
                    copy_from_mmap(out_f, src, rec["index_offset"], rec["index_plane_bytes"])
                    section_end = out_f.tell()
                    m = dict(
                        layer=layer, kind=kind, kind_name=KIND_NAMES.get(kind, str(kind)),
                        k=rec["k"], bits=rec["bits"], out_dim=rec["out_dim"], in_dim=rec["in_dim"],
                        n_indices=rec["n_indices"], scale_count=rec["scale_count"],
                        scale_bits=args.scale_bits, scale_stride=scale_stride,
                        index_bytes=rec["index_bytes"], index_stride=rec["index_stride"],
                        codebook_offset=cb_out, codebook_bytes=rec["codebook_bytes"],
                        scale_offset=scale_out, scale_bytes=scale_stride * N_EXPERTS,
                        index_offset=index_out, index_plane_bytes=rec["index_plane_bytes"],
                        scale_log_min=rec["scale_log_min"], scale_log_step=rec["scale_log_step"],
                        section_bytes=section_end - cb_out,
                    )
                    meta.append(m)
                    scale_lp_new.extend(lp)
                    stats.append(dict(layer=layer, kind=kind, kind_name=KIND_NAMES.get(kind, str(kind)), **st))
                    section_counter += 1
                    if args.status_every and section_counter % args.status_every == 0:
                        elapsed = time.time() - started
                        print(f"status section={section_counter}/{len(layers)*MAX_KINDS} L{layer} {KIND_NAMES.get(kind, kind)} out={out_f.tell()/1073741824:.3f}GiB elapsed={elapsed:.1f}s", flush=True)
            write_padding(out_f, align)
            scale_lp_offset_new = out_f.tell()
            for mn, step in scale_lp_new:
                out_f.write(struct.pack("<2f", mn, step))
            scale_lp_bytes_new = out_f.tell() - scale_lp_offset_new
            meta_json = json.dumps(meta, separators=(",", ":")).encode()
            write_padding(out_f, align)
            meta_offset = out_f.tell()
            out_f.write(meta_json)
            final_size = out_f.tell()
            out_f.seek(0)
            out_f.write(struct.pack("<8sIIIIIIIQQQQQ", MAGIC, VERSION, len(layers), MAX_LAYERS,
                                    N_EXPERTS, 8, 128, align, meta_offset, len(meta_json),
                                    header_bytes, final_size, indexed_source_size or source_size))
            out_f.seek(EXT_HEADER_OFFSET)
            out_f.write(struct.pack(EXT_HEADER_FMT, EXT_MAGIC, scale_lp_offset_new,
                                    scale_lp_bytes_new, len(meta), N_EXPERTS))
            write_section_table(out_f, meta, header_bytes)
            out_f.truncate(final_size)
    summary = dict(
        out=str(args.out), source_pack=str(args.in_pack), source_version=source_version,
        version=VERSION, layers=layers, scale_bits=args.scale_bits, align=align,
        size_bytes=final_size, size_gib=final_size / 1073741824,
        source_size_bytes=source_size, source_size_gib=source_size / 1073741824,
        saved_bytes=source_size - final_size, saved_mib=(source_size - final_size) / 1048576,
        scale_lp_offset=scale_lp_offset_new, scale_lp_bytes=scale_lp_bytes_new,
        meta_offset=meta_offset, meta_len=len(meta_json), sections=len(meta),
        elapsed_sec=time.time() - started, stats=stats, meta=meta,
    )
    summary_path = args.out.with_suffix(args.out.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"done out={args.out} size={final_size/1073741824:.6f}GiB saved={(source_size-final_size)/1048576:.3f}MiB summary={summary_path} elapsed={summary['elapsed_sec']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
