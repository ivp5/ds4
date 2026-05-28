#!/usr/bin/env python3
"""Audit for dual-representation drift candidates within structs.

The Cycle 5 bug was `override_data` + `override_bytes` + `override_source_type`
+ `override_source_active` — 4 fields tracking what should be 1 concept.
The same shape appears as siblings sharing a prefix or suffix.

This audit finds STRUCTS WHERE 3+ FIELDS SHARE A COMMON PREFIX (≥4 chars).
Strong signal of a multi-field representation that could be unified.
"""

import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
FILES = [
    "ds4.c", "ds4_metal.m", "ds4_metal_vqb2_fp16.m", "ds4_inflight.c",
    "ds4_expert_table.c", "ds4_moe_route_log.c", "ds4_polar_reader.c",
    "ds4_vqb1_reader.c", "ds4_vqb2_reader.c", "ds4_vqb2_pack.c",
    "ds4_watersic_pack.c", "ds4_nonrouted_pack.c", "ds4_prefix_cache.c",
    "ds4_nonrouted_pack.h", "ds4.h", "ds4_gpu.h",
]

# any-type field declarator inside struct body
FIELD_RE = re.compile(
    r"^\s*(?:const\s+)?[a-zA-Z_][\w\s*<>:,]+?\s+\*?\s*"
    r"([a-z_][a-z_0-9]*)\s*(?:\[[^\]]*\]\s*)*;",
    re.MULTILINE
)

# struct/typedef body extractor (same as before)
struct_body_re = re.compile(r"(?:typedef\s+struct(?:\s+\w+)?\s*{|struct\s+\w+\s*{)", re.MULTILINE)

def common_prefix(strings):
    """Longest common prefix of a list."""
    if not strings:
        return ""
    shortest = min(strings, key=len)
    for i, c in enumerate(shortest):
        if any(s[i] != c for s in strings):
            return shortest[:i]
    return shortest

multi_field_structs = []

for fname in FILES:
    path = ROOT / fname
    if not path.exists():
        continue
    text = path.read_text()

    for sm in struct_body_re.finditer(text):
        body_start = sm.end()
        depth = 1
        i = body_start
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        body_end = i - 1
        body = text[body_start:body_end]
        after = text[body_end:body_end + 80]
        sname_m = re.match(r"}\s*([A-Za-z_][A-Za-z_0-9]*)", after)
        sname = sname_m.group(1) if sname_m else "?"

        fields = [m.group(1) for m in FIELD_RE.finditer(body)]
        if not fields:
            continue
        # Group by prefix (first underscore-split component)
        groups = defaultdict(list)
        for f in fields:
            parts = f.split("_")
            if len(parts) >= 2:
                key = parts[0]
                groups[key].append(f)
        for prefix, group in groups.items():
            if len(group) >= 3 and len(prefix) >= 4:
                # Found a 3+ field group with a common prefix
                multi_field_structs.append((fname, sname, prefix, group))

print(f"Found {len(multi_field_structs)} structs with 3+ fields sharing a common prefix:")
print()
# Sort by count of fields desc
multi_field_structs.sort(key=lambda x: -len(x[3]))
for fname, sname, prefix, fields in multi_field_structs[:20]:
    print(f"  {fname}  struct {sname}  prefix='{prefix}_'  fields=[{', '.join(fields[:6])}{'...' if len(fields) > 6 else ''}]  ({len(fields)})")
