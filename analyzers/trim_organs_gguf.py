#!/usr/bin/env python3.14
"""Organ-level GGUF pruning for DS4 (Path C — per-(layer, expert, organ) zero).

silv 2026-05-27: GGUF emit pipeline for the H2074 prune set.

CONTEXT
=======
Existing analyzers/trim_experts_gguf.py removes WHOLE EXPERTS file-side.
The codex H2074 deployable prune set is finer: at layer 9, DOWN organ
only, experts {55, 71, 188, 231, 254} get the silv-mean-zero replacement.

This tool implements the per-organ pruning. Three operation modes per
(layer, expert, organ) decision:

  KEEP   — copy through unchanged (default for all cells)
  ZERO   — replace tensor row with constant fill (calibration-mean if
           known, otherwise 0; semantically equivalent to organ-skip
           at the runtime layer used by Phase A wire #651-#657)
  PRUNE  — drop the cell from the tensor (only allowed for cells
           that are also being whole-expert-removed; otherwise
           shape mismatch with the router)

DESIGN
======

Input: source GGUF + organ-skip CSV (same format as DS4_ORGAN_SKIP_CSV):
  layer,expert,organ
  9,55,2
  9,71,2
  9,188,2
  9,231,2
  9,254,2
  (organ: 0=GATE, 1=UP, 2=DOWN per ds4_expert_table.h)

For each tensor in source:
  - Identify (layer, expert, organ) tuple from tensor name + position
  - Apply organ-skip decision: keep / zero / prune
  - Write transformed tensor to output

Per-expert weights live in 3D tensors:
  blk.{L}.ffn_gate_exps.weight  shape (n_in, n_out, n_expert)  organ=0
  blk.{L}.ffn_up_exps.weight    shape (n_in, n_out, n_expert)  organ=1
  blk.{L}.ffn_down_exps.weight  shape (n_out, n_in, n_expert)  organ=2

To zero organ K for expert E at layer L: locate the n_expert-th block
slice at expert offset E in that tensor and replace its bytes.

ZERO SEMANTICS
==============

For IQ2_XXS quantization (66 bytes per 256-element block), zeroing
means writing 66 zero bytes. The decoder produces all-zero outputs for
that block (verified empirically per ds4_metal IQ2_XXS dequant). So
zeroing the down-organ weights for an expert means: when the router
selects that expert at layer L, the down-projection contributes nothing
to the residual stream → equivalent to organ-skip-DOWN at runtime.

The Phase A wire (#651-#657) implemented this semantic at runtime via
the DS4_ORGAN_SKIP env flag. This tool implements it FILE-SIDE so the
runtime flag isn't required.

CALIBRATION-MEAN OPTION
=======================

If --calibration-source PATH is provided, the zero replacement uses
calibration-mean values per (layer, expert, organ) rather than 0. This
preserves more of the expert's contribution while still removing its
input-dependence. Format: same CSV with extra column 'mean_value' or
a separate npz file with cell-keyed means.

NOT YET IMPLEMENTED for IQ2_XXS — would require re-quantizing the
constant-mean tensor to IQ2_XXS format, which needs the original
quantization metadata (block scales etc.). For now, zero-only.

INTEGRATION WITH H2074 PRUNE SET
=================================

The validated H2074 prune set is `[55,71,188,231,254]` at layer 9,
DOWN organ only. The mask CSV is:

  9,55,2
  9,71,2
  9,188,2
  9,231,2
  9,254,2

This produces a ~340 KB reduction (5 experts × 66 bytes × 1024
blocks-per-expert ≈ 340 KB at IQ2_XXS) — a small but symbolic deploy.
The real size win is in basis-aware sidecar emission (Hadamard
transformed weights) which is a separate next-turn implementation.

Usage:
  python3 trim_organs_gguf.py \\
    --base /path/to/DeepSeek-V4-Flash-IQ2XXS-...gguf \\
    --mask cells_h2074_no_hub.csv \\
    --out  /path/to/DS4-h2074-organ-pruned.gguf

  python3 trim_organs_gguf.py --base ... --mask ... --dry-run
    (analyzes affected tensors + reports projected output without writing)

STATUS
======
SKELETON — implements parsing + dry-run analysis. The full
tensor-rewrite path is deferred to next turn because it needs:
1. Correct IQ2_XXS block-stride math (66 bytes × 1024 blocks per expert
   in down_exps — verify against actual file layout)
2. Streaming-shard write (81GB source doesn't fit in memory)
3. Tensor-table offset recomputation
4. Sidecar JSON emission with prune-decision-provenance

Each of the above is 50-100 lines of focused work. The skeleton below
implements (1) parsing + dry-run analysis (2) projects output size
(3) names the missing implementation pieces.
"""

import argparse
import os
import re
import struct
import sys
from pathlib import Path
from collections import defaultdict

# Reuse constants from trim_experts_gguf.py — same GGUF format
GGUF_VALUE_UINT8, GGUF_VALUE_INT8, GGUF_VALUE_UINT16, GGUF_VALUE_INT16 = 0, 1, 2, 3
GGUF_VALUE_UINT32, GGUF_VALUE_INT32, GGUF_VALUE_FLOAT32, GGUF_VALUE_BOOL = 4, 5, 6, 7
GGUF_VALUE_STRING, GGUF_VALUE_ARRAY = 8, 9
GGUF_VALUE_UINT64, GGUF_VALUE_INT64, GGUF_VALUE_FLOAT64 = 10, 11, 12

SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
SCALAR_FMTS = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?',
               10: '<Q', 11: '<q', 12: '<d'}

QUANT_SIZES = {
    0: (1, 4),     # F32
    1: (1, 2),     # F16
    8: (32, 34),   # Q8_0
    10: (256, 84), # Q2_K
    12: (256, 144),# Q4_K
    14: (256, 210),# Q6_K
    16: (256, 66), # IQ2_XXS — DS4 routed-expert bulk
    26: (1, 4),    # I32
    30: (1, 2),    # BF16
}

DS4_N_LAYER = 43
DS4_N_EXPERT = 256

# Tensor name patterns for routed experts. Organ index: 0=GATE, 1=UP, 2=DOWN
EXP_TENSOR_RE = re.compile(r'^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$')
ORGAN_TO_IDX = {'gate': 0, 'up': 1, 'down': 2}
IDX_TO_ORGAN = {v: k for k, v in ORGAN_TO_IDX.items()}


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


def parse_organ_skip_csv(path):
    """Returns dict {(layer, expert): set(organ_idx)}."""
    cells = defaultdict(set)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.lower().startswith('layer'):
                continue
            parts = line.split(',')
            if len(parts) < 3:
                continue
            try:
                layer = int(parts[0])
                expert = int(parts[1])
                organ = int(parts[2])
            except ValueError:
                continue
            if not (0 <= layer < DS4_N_LAYER):
                continue
            if not (0 <= expert < DS4_N_EXPERT):
                continue
            if organ not in (0, 1, 2):
                continue
            cells[(layer, expert)].add(organ)
    return cells


def scan_gguf_for_routed_experts(gguf_path, cells):
    """Walk the source GGUF tensor table, identify which tensors are affected
    by the prune-cell set. Returns list of (tensor_name, layer, organ, n_experts,
    block_elems_per_expert, bytes_per_expert, total_bytes, n_affected_experts)."""

    affected = []
    with open(gguf_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            raise ValueError(f"not a GGUF file: magic={magic!r}")
        version, n_tensors, n_kv = struct.unpack('<IQQ', f.read(20))

        # Skip metadata KVs — we only care about tensor table for the scan
        for _ in range(n_kv):
            k = read_str(f)
            vt = struct.unpack('<I', f.read(4))[0]
            _ = read_value(f, vt)

        # Tensor table
        for _ in range(n_tensors):
            name = read_str(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = list(struct.unpack(f'<{n_dims}Q', f.read(8 * n_dims)))
            ttype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]

            m = EXP_TENSOR_RE.match(name)
            if not m:
                continue

            layer = int(m.group(1))
            organ = ORGAN_TO_IDX[m.group(2)]

            # Affected if any cell at this layer × organ is in the skip set
            affected_experts = []
            for (cell_layer, cell_expert), organ_set in cells.items():
                if cell_layer == layer and organ in organ_set:
                    affected_experts.append(cell_expert)

            if not affected_experts:
                continue

            # Compute bytes-per-expert for the zeroing
            block_elems, block_bytes = QUANT_SIZES.get(ttype, (1, 4))
            n_in = dims[0] if len(dims) >= 1 else 0
            n_out = dims[1] if len(dims) >= 2 else 0
            n_experts = dims[2] if len(dims) >= 3 else 1

            elems_per_expert = n_in * n_out
            blocks_per_expert = elems_per_expert // block_elems
            bytes_per_expert = blocks_per_expert * block_bytes
            total_bytes = bytes_per_expert * n_experts

            affected.append({
                'name': name,
                'layer': layer,
                'organ': organ,
                'organ_str': IDX_TO_ORGAN[organ],
                'ttype': ttype,
                'dims': dims,
                'offset': offset,
                'n_in': n_in,
                'n_out': n_out,
                'n_experts': n_experts,
                'block_elems': block_elems,
                'block_bytes': block_bytes,
                'blocks_per_expert': blocks_per_expert,
                'bytes_per_expert': bytes_per_expert,
                'total_bytes': total_bytes,
                'n_affected_experts': len(affected_experts),
                'affected_experts': sorted(affected_experts),
            })

    return affected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='source GGUF path')
    ap.add_argument('--mask', required=True, help='organ-skip CSV (layer,expert,organ)')
    ap.add_argument('--out', help='output GGUF path (omit for dry-run)')
    ap.add_argument('--dry-run', action='store_true',
                    help='analyze + project size; no file write')
    args = ap.parse_args()

    if not args.out:
        args.dry_run = True

    cells = parse_organ_skip_csv(args.mask)
    if not cells:
        print(f"[trim-organs] empty/invalid mask file: {args.mask}", file=sys.stderr)
        return 2

    print(f"[trim-organs] mask: {sum(len(o) for o in cells.values())} (layer, expert, organ) cells "
          f"across {len(cells)} (layer, expert) pairs")

    affected = scan_gguf_for_routed_experts(args.base, cells)

    if not affected:
        print("[trim-organs] no tensors in source GGUF match the mask cells", file=sys.stderr)
        return 1

    total_zero_bytes = 0
    for t in affected:
        zero_bytes = t['bytes_per_expert'] * t['n_affected_experts']
        total_zero_bytes += zero_bytes
        print(f"  {t['name']} (L={t['layer']}, organ={t['organ_str']}): "
              f"{t['n_affected_experts']}/{t['n_experts']} experts zeroed, "
              f"{zero_bytes:,} bytes ({zero_bytes / (1024 * 1024):.2f} MiB)")

    print(f"[trim-organs] total bytes to zero: {total_zero_bytes:,} "
          f"({total_zero_bytes / (1024 * 1024):.2f} MiB)")

    if args.dry_run:
        print("[trim-organs] DRY RUN — no file written")
        print()
        print("NEXT STEPS to implement full pipeline:")
        print("  1. Stream-shard write: read source shard N, emit output shard M")
        print("     (81GB source can't fit in memory on 64GB M1)")
        print("  2. Per-expert byte-range zero replacement at known offset")
        print("     (offset = tensor_data_start + expert_idx * bytes_per_expert)")
        print("  3. Tensor table offset recomputation (output offsets unchanged")
        print("     for our case — we're not removing or resizing tensors,")
        print("     just zeroing bytes within existing tensor extents)")
        print("  4. Sidecar JSON: provenance record of prune decisions")
        return 0

    raise NotImplementedError(
        "Full write path deferred to next turn. See dry-run mode for analysis.\n"
        "Skeleton implementation is in this file; complete with:\n"
        " - tensor-data shadow-copy or streaming write\n"
        " - per-cell byte-range zero at tensor_data_start + expert_idx * bytes_per_expert\n"
        " - GGUF tensor-table offset reuse (no change needed for zero-in-place)\n"
        " - sidecar JSON emission"
    )


if __name__ == '__main__':
    sys.exit(main())
