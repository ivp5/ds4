#!/usr/bin/env python3
"""Repack DS4 CDX3 routed experts into an M1-runtime-native fixed-plane pack.

CDX3 is fidelity-good but runtime-hostile for M1: each expert/kind payload is
framed and interleaved as gate/up/down per expert, so a routed token cannot bind
six experts with one fixed stride. M1R keeps the exact CDX3 D8 codebooks/scales
/indices but stores them as page-aligned layer-kind planes:

  codebook[layer][kind]
  scales[layer][kind][expert][scale_count]
  indices[layer][kind][expert][index_stride]

The result is addressable as base + expert * stride for selected-expert fused
Metal kernels. No quantization math changes here; this is codec-runtime layout.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import pathlib
import struct
import sys
import time

MAGIC = b"DS4M1R\0\0"
VERSION = 2
CDX3I_HEADER_BYTES = 1260
CDX3I_RECORD_BYTES = 64
DEFAULT_ALIGN = 16 * 1024
META_RECORD_BYTES = 128
MAX_LAYERS = 43
MAX_KINDS = 3
N_EXPERTS = 256
SECTION_TABLE_MAGIC = b"M1RSECT\0"
SECTION_TABLE_VERSION = 1
SECTION_TABLE_HEADER_OFFSET = 128
SECTION_TABLE_RECORD_OFFSET = 256
SECTION_TABLE_HEADER_FMT = "<8sIIII"
SECTION_TABLE_RECORD_FMT = "<12I2f7Q"
SECTION_TABLE_HEADER_BYTES = struct.calcsize(SECTION_TABLE_HEADER_FMT)
SECTION_TABLE_RECORD_BYTES = struct.calcsize(SECTION_TABLE_RECORD_FMT)
EXT_MAGIC = b"M1REXT\0\0"
EXT_HEADER_OFFSET = 80
EXT_HEADER_FMT = "<8sQQII"
EXT_HEADER_BYTES = struct.calcsize(EXT_HEADER_FMT)
SCALE_LP_RECORD_BYTES = 8

KIND_NAMES = {0: "gate", 1: "up", 2: "down"}


def align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def read_cdx3_index(index_path: pathlib.Path):
    raw = index_path.read_bytes()
    if len(raw) < CDX3I_HEADER_BYTES or raw[:7] != b"CDX3IDX":
        raise SystemExit(f"bad CDX3I index: {index_path}")
    version, n_records, records_total, complete, d, group = struct.unpack_from("<6I", raw, 8)
    if version != 2 or d != 8 or group != 128:
        raise SystemExit(f"unsupported CDX3I v={version} d={d} group={group}")
    first_record_offset, indexed_pack_size, created_epoch = struct.unpack_from("<3Q", raw, 32)
    k_by_layer = struct.unpack_from("<43I", raw, 56)
    codebook_offsets = struct.unpack_from("<129Q", raw, 56 + 43 * 4)
    records = {}
    for i in range(n_records):
        p = CDX3I_HEADER_BYTES + i * CDX3I_RECORD_BYTES
        layer, expert, kind, bits = struct.unpack_from("<4H", raw, p)
        out_dim, in_dim, n_indices, scale_count = struct.unpack_from("<4I", raw, p + 8)
        payload_offset, framed_bytes, scale_offset, index_offset = struct.unpack_from("<4Q", raw, p + 24)
        scale_log_min, scale_log_step = struct.unpack_from("<2f", raw, p + 56)
        records[(layer, expert, kind)] = dict(
            layer=layer, expert=expert, kind=kind, bits=bits,
            out_dim=out_dim, in_dim=in_dim, n_indices=n_indices, scale_count=scale_count,
            payload_offset=payload_offset, framed_bytes=framed_bytes,
            scale_offset=scale_offset, index_offset=index_offset,
            scale_log_min=scale_log_min, scale_log_step=scale_log_step,
        )
    return dict(
        version=version, n_records=n_records, records_total=records_total, complete=complete,
        d=d, group=group, first_record_offset=first_record_offset,
        indexed_pack_size=indexed_pack_size, created_epoch=created_epoch,
        k_by_layer=k_by_layer, codebook_offsets=codebook_offsets, records=records,
    )


def parse_layers(text: str | None):
    if not text or text == "all":
        return list(range(MAX_LAYERS))
    out = []
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    out = sorted(set(out))
    for v in out:
        if v < 0 or v >= MAX_LAYERS:
            raise SystemExit(f"layer out of range: {v}")
    return out


def copy_from_mmap(out_f, src: mmap.mmap, off: int, n: int, chunk: int = 8 * 1024 * 1024):
    end = off + n
    pos = off
    while pos < end:
        step = min(chunk, end - pos)
        out_f.write(src[pos:pos + step])
        pos += step


def write_padding(out_f, align: int):
    pos = out_f.tell()
    want = align_up(pos, align)
    if want > pos:
        out_f.write(b"\0" * (want - pos))
    return want


def write_section_table(out_f, meta: list[dict], header_bytes: int):
    table_end = SECTION_TABLE_RECORD_OFFSET + len(meta) * SECTION_TABLE_RECORD_BYTES
    if table_end > header_bytes:
        raise SystemExit(f"M1R section table too large: {table_end} > header {header_bytes}")
    out_f.seek(SECTION_TABLE_HEADER_OFFSET)
    out_f.write(struct.pack(
        SECTION_TABLE_HEADER_FMT,
        SECTION_TABLE_MAGIC,
        SECTION_TABLE_VERSION,
        SECTION_TABLE_RECORD_OFFSET,
        SECTION_TABLE_RECORD_BYTES,
        len(meta),
    ))
    out_f.seek(SECTION_TABLE_RECORD_OFFSET)
    for section_index, m in enumerate(meta):
        out_f.write(struct.pack(
            SECTION_TABLE_RECORD_FMT,
            int(m["layer"]), int(m["kind"]), int(m["k"]), int(m["bits"]),
            int(m["out_dim"]), int(m["in_dim"]), int(m["n_indices"]), int(m["scale_count"]),
            int(m["scale_stride"]), int(m["index_bytes"]), int(m["index_stride"]), section_index,
            float(m["scale_log_min"]), float(m["scale_log_step"]),
            int(m["codebook_offset"]), int(m["codebook_bytes"]),
            int(m["scale_offset"]), int(m["scale_bytes"]),
            int(m["index_offset"]), int(m["index_plane_bytes"]),
            int(m["section_bytes"]),
        ))


def write_scale_lp_table(out_f, index: dict, meta: list[dict]) -> tuple[int, int]:
    scale_lp_offset = out_f.tell()
    for m in meta:
        layer = int(m["layer"])
        kind = int(m["kind"])
        for expert in range(N_EXPERTS):
            rec = index["records"][(layer, expert, kind)]
            out_f.write(struct.pack("<2f", rec["scale_log_min"], rec["scale_log_step"]))
    scale_lp_bytes = out_f.tell() - scale_lp_offset
    return scale_lp_offset, scale_lp_bytes


def write_extension_header(out_f, scale_lp_offset: int, scale_lp_bytes: int, sections: int):
    out_f.seek(EXT_HEADER_OFFSET)
    out_f.write(struct.pack(
        EXT_HEADER_FMT,
        EXT_MAGIC,
        scale_lp_offset,
        scale_lp_bytes,
        sections,
        N_EXPERTS,
    ))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, type=pathlib.Path)
    ap.add_argument("--index", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--layers", default="all", help="all, comma list, or ranges like 0-3,25,42")
    ap.add_argument("--align", type=int, default=DEFAULT_ALIGN)
    ap.add_argument("--status-every", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    layers = parse_layers(args.layers)
    if args.out.exists() and not args.force:
        raise SystemExit(f"output exists; pass --force: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    index = read_cdx3_index(args.index)
    pack_size = args.pack.stat().st_size
    if pack_size != index["indexed_pack_size"]:
        print(f"warning: pack size {pack_size} != indexed {index['indexed_pack_size']}", file=sys.stderr, flush=True)

    header_bytes = align_up(4096 + MAX_LAYERS * MAX_KINDS * META_RECORD_BYTES, args.align)
    meta = []
    started = time.time()
    total_written_payload = 0
    with args.pack.open("rb") as pf, args.out.open("wb+") as out_f:
        src = mmap.mmap(pf.fileno(), 0, access=mmap.ACCESS_READ)
        out_f.write(b"\0" * header_bytes)
        section_count = 0
        for layer in layers:
            for kind in range(MAX_KINDS):
                rec0 = index["records"].get((layer, 0, kind))
                if rec0 is None:
                    raise SystemExit(f"missing L{layer} kind {kind} expert0")
                k = index["k_by_layer"][layer]
                bits = rec0["bits"]
                codebook_offset = index["codebook_offsets"][layer * 3 + kind]
                codebook_bytes = k * 8 * 2
                index_bytes = (rec0["n_indices"] * bits + 7) // 8
                index_stride = align_up(index_bytes + 4, 64)
                scale_stride = align_up(rec0["scale_count"], 64)
                for expert in range(N_EXPERTS):
                    rec = index["records"].get((layer, expert, kind))
                    if rec is None:
                        raise SystemExit(f"missing L{layer} E{expert} kind {kind}")
                    for field in ("bits", "out_dim", "in_dim", "n_indices", "scale_count"):
                        if rec[field] != rec0[field]:
                            raise SystemExit(f"layout mismatch L{layer} E{expert} kind {kind} field {field}: {rec[field]} != {rec0[field]}")
                write_padding(out_f, args.align)
                cb_out = out_f.tell()
                copy_from_mmap(out_f, src, codebook_offset, codebook_bytes)
                write_padding(out_f, args.align)
                scale_out = out_f.tell()
                for expert in range(N_EXPERTS):
                    rec = index["records"][(layer, expert, kind)]
                    before = out_f.tell()
                    copy_from_mmap(out_f, src, rec["scale_offset"], rec0["scale_count"])
                    pad = scale_stride - rec0["scale_count"]
                    if pad:
                        out_f.write(b"\0" * pad)
                    assert out_f.tell() - before == scale_stride
                write_padding(out_f, args.align)
                index_out = out_f.tell()
                for expert in range(N_EXPERTS):
                    rec = index["records"][(layer, expert, kind)]
                    before = out_f.tell()
                    copy_from_mmap(out_f, src, rec["index_offset"], index_bytes)
                    out_f.write(b"\0" * (index_stride - index_bytes))
                    assert out_f.tell() - before == index_stride
                section_end = out_f.tell()
                meta.append(dict(
                    layer=layer, kind=kind, kind_name=KIND_NAMES[kind], k=k, bits=bits,
                    out_dim=rec0["out_dim"], in_dim=rec0["in_dim"], n_indices=rec0["n_indices"],
                    scale_count=rec0["scale_count"], scale_stride=scale_stride,
                    index_bytes=index_bytes, index_stride=index_stride,
                    codebook_offset=cb_out, codebook_bytes=codebook_bytes,
                    scale_offset=scale_out, scale_bytes=scale_stride * N_EXPERTS,
                    index_offset=index_out, index_plane_bytes=index_stride * N_EXPERTS,
                    scale_log_min=rec0["scale_log_min"], scale_log_step=rec0["scale_log_step"],
                    section_bytes=section_end - cb_out,
                ))
                section_count += 1
                total_written_payload += section_end - cb_out
                if args.status_every and section_count % args.status_every == 0:
                    elapsed = time.time() - started
                    mib = total_written_payload / 1048576
                    rate = mib / max(elapsed, 1e-9)
                    done = section_count
                    total = len(layers) * MAX_KINDS
                    eta = (total - done) * (elapsed / max(done, 1))
                    print(f"status sections={done}/{total} layer={layer} kind={KIND_NAMES[kind]} out={out_f.tell()/1073741824:.3f}GiB payload={mib:.1f}MiB rate={rate:.1f}MiB/s eta={eta:.1f}s", flush=True)
        payload_end = out_f.tell()
        scale_lp_offset, scale_lp_bytes = write_scale_lp_table(out_f, index, meta)
        meta_json = json.dumps(meta, separators=(",", ":")).encode()
        meta_offset = out_f.tell()
        out_f.seek(meta_offset)
        out_f.write(meta_json)
        file_size = out_f.tell()
        out_f.seek(0)
        fixed_header = struct.pack(
            "<8sIIIIIIIQQQQQ",
            MAGIC, VERSION, len(layers), MAX_LAYERS, N_EXPERTS, index["d"], index["group"], args.align,
            meta_offset, len(meta_json), header_bytes, file_size, index["indexed_pack_size"],
        )
        out_f.write(fixed_header)
        write_extension_header(out_f, scale_lp_offset, scale_lp_bytes, len(meta))
        write_section_table(out_f, meta, header_bytes)
        out_f.truncate(file_size)
        src.close()
    summary_path = args.out.with_suffix(args.out.suffix + ".summary.json")
    summary = dict(
        out=str(args.out), index=str(args.index), source_pack=str(args.pack), layers=layers,
        size_bytes=file_size, size_gib=file_size / 1073741824,
        payload_end=payload_end, scale_lp_offset=scale_lp_offset, scale_lp_bytes=scale_lp_bytes,
        source_size_bytes=pack_size, source_size_gib=pack_size / 1073741824,
        align=args.align, sections=len(meta), elapsed_sec=time.time() - started,
        section_table_magic=SECTION_TABLE_MAGIC.decode("ascii", errors="replace"),
        ext_magic=EXT_MAGIC.decode("ascii", errors="replace"),
        section_table_offset=SECTION_TABLE_RECORD_OFFSET,
        section_table_record_bytes=SECTION_TABLE_RECORD_BYTES,
        scale_lp_record_bytes=SCALE_LP_RECORD_BYTES,
        meta=meta,
    )
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"done out={args.out} size={file_size/1073741824:.3f}GiB summary={summary_path} elapsed={summary['elapsed_sec']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
