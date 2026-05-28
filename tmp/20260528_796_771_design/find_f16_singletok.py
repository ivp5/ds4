#!/usr/bin/env python3
"""List ds4_gpu_matmul_f16_tensor sites with their n_tok argument.

For Increment 2d bulk wiring: only single-tok sites can use
matmul_f16_storage (current implementation rejects n_tok > 1).
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "ds4.c"

text = SRC.read_text()
lines = text.split("\n")

# Find each ds4_gpu_matmul_f16_tensor( site + extract n_tok (8th arg)
i = 0
sites = []
while i < len(lines):
    if "ds4_gpu_matmul_f16_tensor(" in lines[i]:
        start = i
        # Walk forward, collecting until we hit closing );
        depth = 0
        buf = []
        for j in range(start, min(start + 25, len(lines))):
            buf.append(lines[j])
            depth += lines[j].count("(") - lines[j].count(")")
            if depth <= 0 and ");" in lines[j] or (depth <= 0 and ")" in lines[j] and "!= 0" in lines[j]):
                break
        full = "\n".join(buf)
        # Extract args between matmul_f16_tensor( and final )
        m = re.search(r"ds4_gpu_matmul_f16_tensor\((.*?)\)\s*(?:!=\s*0)?;", full, re.DOTALL)
        if m:
            args = [a.strip() for a in m.group(1).split(",")]
            n_tok = args[-1] if args else "?"
            sites.append((start + 1, n_tok, args))
        i = j + 1
    else:
        i += 1

single = [(ln, n, args) for (ln, n, args) in sites if n == "1"]
multi = [(ln, n, args) for (ln, n, args) in sites if n != "1"]
print(f"Single-tok (n_tok=1): {len(single)}")
for ln, n, args in single:
    # weight tensor is args[3] (after dst, map, size — actually dst is args[0])
    # signature: (dst, map, size, offset, in_dim, out_dim, src, n_tok)
    # so weight offset is args[3] which is typically "X->abs_offset"
    weight = args[3] if len(args) > 3 else "?"
    print(f"  {ln}: weight_offset={weight}")
print(f"\nMulti-tok (n_tok != 1): {len(multi)}")
for ln, n, args in multi[:8]:
    weight = args[3] if len(args) > 3 else "?"
    print(f"  {ln}: n_tok={n}  weight_offset={weight}")
