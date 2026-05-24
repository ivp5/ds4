#!/usr/bin/env python3.14
"""Drop selected experts from DS4 GGUF (Path A file-trim).

silv 2026-05-23 directive: "if you get a pathway to reducing ds4 flash size
to less than 52gb with the aggregated techniques, try that as well!"

This script implements Path A (file-trim) per audits/trim_ladder_to_64gb_20260523.md.
Path B (runtime mask via ds4_expert_table.{c,h}) lets us TEST capability with
the same file; Path A produces the smaller deployable file.

Algorithm:
  1. Parse GGUF metadata + tensor index
  2. Load mask CSV → kept-expert IDs per layer
  3. For each ffn_{gate,up,down}_exps tensor:
     - Slice along n_experts axis (last dim) to keep only mask-allowed experts
     - Recompute tensor shape and data size
  4. For ffn_gate_inp at each layer:
     - Slice the router output dim (n_experts) to keep only kept experts
  5. For ffn_exp_probs_b at each layer (if present):
     - Slice to keep only kept experts
  6. Add ds4.expert_remap.<L> = list of original IDs (for runtime translation)
  7. Update layer-specific n_experts if present (else stays at DS4_N_EXPERT=256)
  8. Re-emit GGUF

Caveats:
  - The runtime (ds4.c) must handle per-layer variable n_experts (currently
    hard-coded to 256 in tensor_expect_routed_expert at ds4.c:2599).
  - Alternative: keep n_experts=256 metadata but use the ds4.expert_remap
    metadata to translate logits at runtime. Less invasive on ds4.c.
  - The ffn_gate_inp slicing requires understanding the router weight shape;
    it's [n_embd, n_experts] in column-major. Slicing columns.

Usage:
  python3 trim_experts_gguf.py \\
    --base /path/to/DeepSeek-V4-Flash-IQ2XXS-...gguf \\
    --mask /Users/silv/cl/tlp/montyneg/ds4/masks/mask_keep50.csv \\
    --out  /path/to/DS4-trim50.gguf

For testing without DS4 launch: use --dry-run to skip file write but compute
projected output size.
"""

import argparse, os, re, struct, sys
from pathlib import Path
from collections import defaultdict

# --- Constants (mirror trim_layers_gguf.py) ---
GGUF_VALUE_UINT8, GGUF_VALUE_INT8, GGUF_VALUE_UINT16, GGUF_VALUE_INT16 = 0,1,2,3
GGUF_VALUE_UINT32, GGUF_VALUE_INT32, GGUF_VALUE_FLOAT32, GGUF_VALUE_BOOL = 4,5,6,7
GGUF_VALUE_STRING, GGUF_VALUE_ARRAY = 8, 9
GGUF_VALUE_UINT64, GGUF_VALUE_INT64, GGUF_VALUE_FLOAT64 = 10, 11, 12

SCALAR_SIZES = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
SCALAR_FMTS = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}

# (block_elements, block_bytes) per ggml type
QUANT_SIZES = {
    0: (1, 4),     # F32
    1: (1, 2),     # F16
    8: (32, 34),   # Q8_0
    10: (256, 84), # Q2_K
    12: (256, 144),# Q4_K
    14: (256, 210),# Q6_K
    16: (256, 66), # IQ2_XXS — the bulk of DS4's expert weights
    26: (1, 4),   # I32
    30: (1, 2),   # BF16
}

GGUF_DEFAULT_ALIGNMENT = 32
DS4_N_LAYER = 43
DS4_N_EXPERT = 256
PROTECTED_LAYERS = {0, 1, 2}

# Tensor name patterns that need per-expert slicing
EXP_TENSOR_RE = re.compile(r'^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$')
ROUTER_TENSOR_RE = re.compile(r'^blk\.(\d+)\.ffn_gate_inp\.weight$')
PROBS_B_TENSOR_RE = re.compile(r'^blk\.(\d+)\.exp_probs_b\.bias$')


def read_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8')

def read_value(f, vt):
    if vt in SCALAR_SIZES:
        return struct.unpack(SCALAR_FMTS[vt], f.read(SCALAR_SIZES[vt]))[0]
    if vt == GGUF_VALUE_STRING:
        return read_str(f)
    if vt == GGUF_VALUE_ARRAY:
        atype = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        return (atype, [read_value(f, atype) for _ in range(n)])
    raise ValueError(f"unknown gguf vtype {vt}")

def write_value(f, vt, value):
    if vt in SCALAR_SIZES:
        f.write(struct.pack(SCALAR_FMTS[vt], value))
    elif vt == GGUF_VALUE_STRING:
        b = value.encode('utf-8')
        f.write(struct.pack('<Q', len(b)) + b)
    elif vt == GGUF_VALUE_ARRAY:
        atype, items = value
        f.write(struct.pack('<I', atype) + struct.pack('<Q', len(items)))
        for it in items:
            write_value(f, atype, it)

def tensor_bytes(dims, ggml_type):
    n = 1
    for d in dims: n *= d
    bs, bb = QUANT_SIZES[ggml_type]
    return (n // bs) * bb

def load_mask_csv(path):
    """Return {layer: set(dropped_experts)}."""
    drops = defaultdict(set)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split(",")
            if len(parts) != 2: continue
            l, e = int(parts[0]), int(parts[1])
            if 0 <= l < DS4_N_LAYER and 0 <= e < DS4_N_EXPERT:
                drops[l].add(e)
    return dict(drops)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="source GGUF path")
    ap.add_argument("--mask", required=True, help="mask CSV (DROP list)")
    ap.add_argument("--out", required=True, help="output GGUF path")
    ap.add_argument("--dry-run", action="store_true", help="compute projected output size; do not write")
    ap.add_argument("--force", action="store_true", help="overwrite output if exists")
    args = ap.parse_args()

    base, out_path = Path(args.base), Path(args.out)
    if not base.exists():
        print(f"ERROR: base file not found: {base}", file=sys.stderr); return 1
    if out_path.exists() and not args.force and not args.dry_run:
        print(f"ERROR: output exists, use --force: {out_path}", file=sys.stderr); return 2

    drops_per_layer = load_mask_csv(args.mask)
    total_drops = sum(len(d) for d in drops_per_layer.values())
    print(f"[trim-exp] mask: {args.mask}")
    print(f"[trim-exp] total drops: {total_drops}")
    print(f"[trim-exp] per-layer drops summary: " + ", ".join(
        f"L{L}:{len(drops_per_layer.get(L, set()))}" for L in sorted(drops_per_layer.keys())[:10]) + " ...")

    # Compute kept-expert IDs per layer
    kept_per_layer = {}
    remap_per_layer = {}  # new_id -> old_id
    for L in range(DS4_N_LAYER):
        if L in PROTECTED_LAYERS:
            kept_per_layer[L] = list(range(DS4_N_EXPERT))
        else:
            drops = drops_per_layer.get(L, set())
            kept = sorted(set(range(DS4_N_EXPERT)) - drops)
            kept_per_layer[L] = kept
            remap_per_layer[L] = kept  # remap[new] = old

    # === Parse base GGUF ===
    print(f"[trim-exp] reading {base} ({base.stat().st_size/1e9:.2f} GB)")
    fin = open(base, "rb")
    magic = fin.read(4)
    if magic != b"GGUF":
        print(f"not a GGUF: {magic!r}", file=sys.stderr); return 3
    version = struct.unpack('<I', fin.read(4))[0]
    n_tensors = struct.unpack('<Q', fin.read(8))[0]
    n_kv = struct.unpack('<Q', fin.read(8))[0]
    print(f"[trim-exp] gguf v{version}: {n_tensors} tensors, {n_kv} kv")

    kvs = []
    for _ in range(n_kv):
        k = read_str(fin)
        vt = struct.unpack('<I', fin.read(4))[0]
        v = read_value(fin, vt)
        kvs.append((k, vt, v))

    # === Read tensor index ===
    tensors_in = []
    for _ in range(n_tensors):
        name = read_str(fin)
        ndim = struct.unpack('<I', fin.read(4))[0]
        dims = struct.unpack(f'<{ndim}Q', fin.read(8 * ndim))
        ggml_type = struct.unpack('<I', fin.read(4))[0]
        rel_offset = struct.unpack('<Q', fin.read(8))[0]
        tensors_in.append((name, list(dims), ggml_type, rel_offset))

    data_start = fin.tell()
    data_start = (data_start + GGUF_DEFAULT_ALIGNMENT - 1) // GGUF_DEFAULT_ALIGNMENT * GGUF_DEFAULT_ALIGNMENT

    # === Classify tensors + compute new shapes ===
    tensors_out = []  # (new_name, new_dims, ggml_type, old_rel_offset, old_name, slice_info)
    total_old_bytes = 0
    total_new_bytes = 0
    for (name, dims, ggml_type, rel_offset) in tensors_in:
        old_size = tensor_bytes(dims, ggml_type)
        total_old_bytes += old_size

        m_exp = EXP_TENSOR_RE.match(name)
        m_router = ROUTER_TENSOR_RE.match(name)
        m_probs = PROBS_B_TENSOR_RE.match(name)

        if m_exp:
            # ffn_{gate,up,down}_exps: shape [..., n_experts]; slice last dim
            L = int(m_exp.group(1))
            kept = kept_per_layer.get(L, list(range(DS4_N_EXPERT)))
            new_n_experts = len(kept)
            # dims is [d0, d1, n_experts] for gate/up or [d0, d1, n_experts] for down
            # ggml stores in row-major: each expert is a contiguous chunk
            new_dims = list(dims)
            assert new_dims[-1] == DS4_N_EXPERT, f"{name}: expected last dim={DS4_N_EXPERT}, got {new_dims[-1]}"
            new_dims[-1] = new_n_experts
            new_size = tensor_bytes(new_dims, ggml_type)
            total_new_bytes += new_size
            tensors_out.append((name, new_dims, ggml_type, rel_offset, name,
                               {'kind': 'exp', 'layer': L, 'kept': kept}))
        elif m_router:
            # ffn_gate_inp: shape [n_embd, n_experts]; slice last dim
            L = int(m_router.group(1))
            kept = kept_per_layer.get(L, list(range(DS4_N_EXPERT)))
            new_dims = list(dims)
            if new_dims[-1] == DS4_N_EXPERT:
                new_dims[-1] = len(kept)
                new_size = tensor_bytes(new_dims, ggml_type)
                total_new_bytes += new_size
                tensors_out.append((name, new_dims, ggml_type, rel_offset, name,
                                   {'kind': 'router', 'layer': L, 'kept': kept}))
            else:
                # Unexpected shape — pass through
                total_new_bytes += old_size
                tensors_out.append((name, dims, ggml_type, rel_offset, name, None))
        elif m_probs:
            # ffn_exp_probs_b: [n_experts]; slice
            L = int(m_probs.group(1))
            kept = kept_per_layer.get(L, list(range(DS4_N_EXPERT)))
            new_dims = [len(kept)]
            new_size = tensor_bytes(new_dims, ggml_type)
            total_new_bytes += new_size
            tensors_out.append((name, new_dims, ggml_type, rel_offset, name,
                               {'kind': 'probs_b', 'layer': L, 'kept': kept}))
        else:
            # Pass-through
            total_new_bytes += old_size
            tensors_out.append((name, dims, ggml_type, rel_offset, name, None))

    print(f"[trim-exp] tensor bytes: old={total_old_bytes/1e9:.2f} GB, "
          f"new={total_new_bytes/1e9:.2f} GB, saved={(total_old_bytes - total_new_bytes)/1e9:.2f} GB")

    if args.dry_run:
        print(f"[trim-exp] DRY RUN — projected output size: ~{total_new_bytes/1e9:.2f} GB (data only) + ~10 MB metadata")
        if total_new_bytes / 1e9 < 52.0:
            print(f"[trim-exp] ✓ projected file under 52 GB target")
        else:
            print(f"[trim-exp] ✗ projected file ABOVE 52 GB target; need more aggressive mask")
        return 0

    # === Add expert_remap metadata for runtime translation ===
    # New metadata key per layer: ds4.expert_remap.<L> = array<u32>[kept] of old IDs
    new_kvs = list(kvs)
    for L in range(DS4_N_LAYER):
        if L in PROTECTED_LAYERS:
            continue
        if L not in remap_per_layer:
            continue
        kept = remap_per_layer[L]
        if len(kept) == DS4_N_EXPERT:
            continue  # no change
        key = f"ds4.expert_remap.{L}"
        new_kvs.append((key, GGUF_VALUE_ARRAY, (GGUF_VALUE_UINT32, kept)))

    # NOTE: We do NOT change the model_type's n_experts metadata. Instead the
    # runtime checks for ds4.expert_remap.<L> and uses it to translate router
    # output (256 logits) → kept-expert indices.
    # Alternative: change deepseek4.expert_count to layer-specific; more invasive.

    # === Write output file ===
    print(f"[trim-exp] writing {out_path}...")
    fout = open(out_path, "wb")
    fout.write(b"GGUF")
    fout.write(struct.pack('<I', version))
    fout.write(struct.pack('<Q', len(tensors_out)))
    fout.write(struct.pack('<Q', len(new_kvs)))

    for k, vt, v in new_kvs:
        kb = k.encode('utf-8')
        fout.write(struct.pack('<Q', len(kb)) + kb)
        fout.write(struct.pack('<I', vt))
        write_value(fout, vt, v)

    # Compute new tensor offsets
    new_rel_offsets = []
    new_offset = 0
    sizes = []
    for (new_name, new_dims, ggml_type, old_rel, old_name, slice_info) in tensors_out:
        sz = tensor_bytes(new_dims, ggml_type)
        new_offset = (new_offset + GGUF_DEFAULT_ALIGNMENT - 1) // GGUF_DEFAULT_ALIGNMENT * GGUF_DEFAULT_ALIGNMENT
        new_rel_offsets.append(new_offset)
        sizes.append(sz)
        new_offset += sz

    for (new_name, new_dims, ggml_type, old_rel, old_name, _), new_rel in zip(tensors_out, new_rel_offsets):
        nb = new_name.encode('utf-8')
        fout.write(struct.pack('<Q', len(nb)) + nb)
        fout.write(struct.pack('<I', len(new_dims)))
        for d in new_dims: fout.write(struct.pack('<Q', d))
        fout.write(struct.pack('<I', ggml_type))
        fout.write(struct.pack('<Q', new_rel))

    # Pad to alignment for data start
    pos = fout.tell()
    pad = (GGUF_DEFAULT_ALIGNMENT - (pos % GGUF_DEFAULT_ALIGNMENT)) % GGUF_DEFAULT_ALIGNMENT
    fout.write(b"\x00" * pad)
    new_data_start = fout.tell()

    # Write tensor data — for sliced tensors, copy kept-expert chunks only
    for (new_name, new_dims, ggml_type, old_rel, old_name, slice_info), new_rel, new_sz in zip(
            tensors_out, new_rel_offsets, sizes):
        target = new_data_start + new_rel
        cur = fout.tell()
        if cur < target:
            fout.write(b"\x00" * (target - cur))

        if slice_info is None:
            # Pass-through copy
            fin.seek(data_start + old_rel)
            remaining = new_sz
            chunk = 16 * 1024 * 1024
            while remaining > 0:
                buf = fin.read(min(chunk, remaining))
                if not buf:
                    print(f"ERROR: short read at {old_name}", file=sys.stderr); return 4
                fout.write(buf); remaining -= len(buf)
        else:
            # Sliced — write only kept chunks
            kind = slice_info['kind']
            kept = slice_info['kept']
            if kind == 'exp':
                # Expert tensor: shape [d0, d1, n_experts], stored expert-major
                # (last dim varies slowest). Per-expert byte size = tensor_bytes
                # of [d0, d1, 1].
                old_dims = [new_dims[0], new_dims[1], DS4_N_EXPERT]
                per_expert_bytes = tensor_bytes([new_dims[0], new_dims[1], 1], ggml_type)
                for new_e, old_e in enumerate(kept):
                    fin.seek(data_start + old_rel + old_e * per_expert_bytes)
                    remaining = per_expert_bytes
                    chunk = 16 * 1024 * 1024
                    while remaining > 0:
                        buf = fin.read(min(chunk, remaining))
                        if not buf:
                            print(f"ERROR: short read on expert {old_e} of {old_name}", file=sys.stderr); return 5
                        fout.write(buf); remaining -= len(buf)
            elif kind == 'router':
                # Router weight: [n_embd, n_experts]. Stored row-major:
                # the n_experts dim is fastest-varying. To slice n_experts,
                # we need to read each row and write only kept columns.
                # F16 storage means 2 bytes per element.
                n_embd = new_dims[0]
                bs, bb = QUANT_SIZES[ggml_type]
                bytes_per_elem = bb // bs
                if bs != 1:
                    print(f"ERROR: router weight ggml_type {ggml_type} not supported for slicing (blocksize {bs})", file=sys.stderr)
                    return 6
                # Read row by row
                fin.seek(data_start + old_rel)
                for row in range(n_embd):
                    row_data = fin.read(DS4_N_EXPERT * bytes_per_elem)
                    if len(row_data) != DS4_N_EXPERT * bytes_per_elem:
                        print(f"ERROR: short read on router row {row} of {old_name}", file=sys.stderr); return 7
                    # Extract kept columns
                    out_row = bytearray()
                    for old_e in kept:
                        start = old_e * bytes_per_elem
                        out_row.extend(row_data[start:start + bytes_per_elem])
                    fout.write(out_row)
            elif kind == 'probs_b':
                # 1D F32 tensor; slice elements
                bs, bb = QUANT_SIZES[ggml_type]
                bytes_per_elem = bb // bs
                fin.seek(data_start + old_rel)
                full_data = fin.read(DS4_N_EXPERT * bytes_per_elem)
                out_data = bytearray()
                for old_e in kept:
                    start = old_e * bytes_per_elem
                    out_data.extend(full_data[start:start + bytes_per_elem])
                fout.write(out_data)
            else:
                print(f"ERROR: unknown slice kind {kind!r}", file=sys.stderr); return 8

    fin.close(); fout.close()
    out_size = out_path.stat().st_size
    print(f"[trim-exp] done. output size: {out_size/1e9:.2f} GB (saved {(base.stat().st_size - out_size)/1e9:.2f} GB)")

    # Verify size against 52 GB target
    if out_size / 1e9 < 52.0:
        print(f"[trim-exp] ✓ output file is under 52 GB (silv's target)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
