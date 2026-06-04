#!/usr/bin/env python3
"""Compare one DS4D8F expert record against the original safetensors weight.

This is a codec truth probe, not a generation benchmark.  It decodes the
bitpacked D8F payload exactly as the Metal runtime sees it:

- gate/up: fp16 codebook entries are the effective weights.
- down: fp16 codebook entries multiply fp16 inverse activation-scale values
  from the record scale payload, matching `d8f_mid_value` in Metal.

It also reports the best single scalar that maps decoded D8F weights back to
source weights.  A low `rel_l2_after_optimal_scalar` with a large scalar is a
direct signature of a missing codebook-scale field in the on-disk format.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from ds4_d8f import D8FFile, DEFAULT_MODEL_DIR, PROJECTIONS, compare, load_source  # noqa: E402
from ds4_safetensors import SafetensorStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d8f", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--projection", required=True, choices=sorted(PROJECTIONS))
    parser.add_argument("--expert", required=True, type=int)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = SafetensorStore(args.model_dir)
    with D8FFile(args.d8f) as d8f:
        record = d8f.record(args.projection, args.expert)
        source = load_source(args.layer, args.projection, args.expert, store)
        decoded = d8f.decode(record, args.projection)
        result = {
            "d8f": os.fspath(args.d8f),
            "layer": args.layer,
            "projection": args.projection,
            "expert": args.expert,
            "record": record,
            "metrics": compare(source, decoded),
        }
        if args.projection == "down" and int(record["flags"]) & 1:
            mapped = d8f.mapped
            if mapped is None:
                raise RuntimeError("D8FFile is not open")
            scale_base = int(record["scale_offset"])
            scale_end = scale_base + int(record["scale_bytes"])
            inv_scale = np.frombuffer(mapped[scale_base:scale_end], dtype=np.float16).astype(np.float32)
            result["inv_scale"] = {
                "len": int(inv_scale.shape[0]),
                "min": float(np.min(inv_scale)),
                "max": float(np.max(inv_scale)),
                "mean": float(np.mean(inv_scale)),
            }
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
