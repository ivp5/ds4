#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, struct, time
import cdx3_repack_m1r as r


def build_meta(index, layers, align, header_bytes):
    meta=[]
    pos=header_bytes
    for layer in layers:
        for kind in range(r.MAX_KINDS):
            rec0=index['records'].get((layer,0,kind))
            if rec0 is None: raise SystemExit(f'missing L{layer} kind {kind} expert0')
            k=index['k_by_layer'][layer]
            bits=rec0['bits']
            codebook_bytes=k*8*2
            index_bytes=(rec0['n_indices']*bits+7)//8
            index_stride=r.align_up(index_bytes+4,64)
            scale_stride=r.align_up(rec0['scale_count'],64)
            cb_out=r.align_up(pos,align); pos=cb_out+codebook_bytes
            scale_out=r.align_up(pos,align); pos=scale_out+scale_stride*r.N_EXPERTS
            index_out=r.align_up(pos,align); pos=index_out+index_stride*r.N_EXPERTS
            meta.append(dict(
                layer=layer, kind=kind, kind_name=r.KIND_NAMES[kind], k=k, bits=bits,
                out_dim=rec0['out_dim'], in_dim=rec0['in_dim'], n_indices=rec0['n_indices'],
                scale_count=rec0['scale_count'], scale_stride=scale_stride,
                index_bytes=index_bytes, index_stride=index_stride,
                codebook_offset=cb_out, codebook_bytes=codebook_bytes,
                scale_offset=scale_out, scale_bytes=scale_stride*r.N_EXPERTS,
                index_offset=index_out, index_plane_bytes=index_stride*r.N_EXPERTS,
                scale_log_min=rec0['scale_log_min'], scale_log_step=rec0['scale_log_step'],
                section_bytes=pos-cb_out,
            ))
    return meta,pos


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pack', required=True, type=pathlib.Path)
    ap.add_argument('--index', required=True, type=pathlib.Path)
    ap.add_argument('--out', required=True, type=pathlib.Path)
    ap.add_argument('--layers', default='all')
    ap.add_argument('--align', type=int, default=r.DEFAULT_ALIGN)
    ap.add_argument('--header-bytes', type=int, default=32768)
    ap.add_argument('--force', action='store_true')
    args=ap.parse_args()
    started=time.time()
    layers=r.parse_layers(args.layers)
    index=r.read_cdx3_index(args.index)
    meta,payload_end=build_meta(index,layers,args.align,args.header_bytes)
    have=args.out.stat().st_size
    if have < payload_end:
        raise SystemExit(f'payload incomplete: have={have} expected_payload_end={payload_end}')
    if have > payload_end and not args.force:
        raise SystemExit(f'output already has tail: have={have} expected_payload_end={payload_end}; pass --force to rewrite tail')
    scale_lp_bytes=len(meta)*r.N_EXPERTS*r.SCALE_LP_RECORD_BYTES
    scale_lp_offset=payload_end
    meta_json=json.dumps(meta,separators=(',',':')).encode()
    meta_offset=scale_lp_offset+scale_lp_bytes
    final_size=meta_offset+len(meta_json)
    with args.out.open('rb+') as f:
        f.truncate(payload_end)
        f.seek(scale_lp_offset)
        r.write_scale_lp_table(f,index,meta)
        f.seek(meta_offset)
        f.write(meta_json)
        f.seek(0)
        header=struct.pack('<8sIIIIIIIQQQQQ', r.MAGIC, r.VERSION, len(layers), r.MAX_LAYERS,
                           r.N_EXPERTS, index['d'], index['group'], args.align,
                           meta_offset, len(meta_json), args.header_bytes, final_size,
                           index['indexed_pack_size'])
        f.write(header)
        r.write_extension_header(f, scale_lp_offset, scale_lp_bytes, len(meta))
        r.write_section_table(f, meta, args.header_bytes)
        f.truncate(final_size)
    summary_path=args.out.with_suffix(args.out.suffix+'.summary.json')
    summary=dict(out=str(args.out), index=str(args.index), source_pack=str(args.pack), layers=layers,
                 size_bytes=final_size, size_gib=final_size/1073741824,
                 payload_end=payload_end, scale_lp_offset=scale_lp_offset, scale_lp_bytes=scale_lp_bytes,
                 meta_offset=meta_offset, meta_len=len(meta_json),
                 source_size_bytes=args.pack.stat().st_size, source_size_gib=args.pack.stat().st_size/1073741824,
                 align=args.align, sections=len(meta), elapsed_sec=time.time()-started,
                 section_table_magic=r.SECTION_TABLE_MAGIC.decode('ascii', errors='replace'),
                 ext_magic=r.EXT_MAGIC.decode('ascii', errors='replace'),
                 section_table_offset=r.SECTION_TABLE_RECORD_OFFSET,
                 section_table_record_bytes=r.SECTION_TABLE_RECORD_BYTES,
                 scale_lp_record_bytes=r.SCALE_LP_RECORD_BYTES,
                 meta=meta)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f'finalized out={args.out} size={final_size/1073741824:.3f}GiB payload_end={payload_end} meta_len={len(meta_json)} summary={summary_path} elapsed={summary["elapsed_sec"]:.2f}s')

if __name__=='__main__': main()
