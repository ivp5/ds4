#!/usr/bin/env python3
"""Audit for Cycle-5-shape latent bugs at the struct-field layer.

The override_source_active bug was a STRUCT FIELD declared but never
written. This script extends the flag-globals audit to struct fields.

Method:
1. Parse all `typedef struct { ... } NAME;` and `struct NAME { ... };` bodies
2. Extract field names of flag-shape types (int, uint8_t, bool, etc.)
3. For each field, count assignments: `.NAME = X` or `->NAME = X`
4. For each field, count reads: any other occurrence
5. Flag any with writes == 0 and reads > 0
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # = ivp5_ds4/
FILES = [
    "ds4.c", "ds4_metal.m", "ds4_metal_vqb2_fp16.m", "ds4_inflight.c",
    "ds4_expert_table.c", "ds4_moe_route_log.c", "ds4_polar_reader.c",
    "ds4_vqb1_reader.c", "ds4_vqb2_reader.c", "ds4_vqb2_pack.c",
    "ds4_watersic_pack.c", "ds4_nonrouted_pack.c", "ds4_prefix_cache.c",
    "ds4_nonrouted_pack.h", "ds4.h", "ds4_gpu.h",
]

# Field type pattern — only check "flag-shaped" fields (int/bool/uint8_t/uint32_t/etc.)
FIELD_RE = re.compile(
    r"^\s*(?:int|uint8_t|uint16_t|uint32_t|uint64_t|bool|_Bool|atomic_int|atomic_bool|int32_t)\s+"
    r"([a-z_][a-z_0-9]*)\s*[;[]",
    re.MULTILINE
)

# Suffix filter — same heuristic as flag-global audit. The Cycle 5 bug
# (override_source_active) ends in _active. The token-store-active field
# would be the same shape.
FLAG_SUFFIXES = (
    "_active", "_enabled", "_ok", "_ready", "_inited", "_initialized",
    "_done", "_loaded", "_open", "_set", "_present", "_valid", "_attempted",
    "_armed", "_running", "_dirty", "_pending", "_locked", "_busy", "_held",
    "_recorded", "_dispatched", "_fired", "_complete",
)

FLAG_PREFIXES = ("is_", "has_", "should_", "needs_", "did_", "got_", "wants_")

def is_flag_field(name: str) -> bool:
    if any(name.endswith(suf) for suf in FLAG_SUFFIXES):
        return True
    if any(name.startswith(pre) for pre in FLAG_PREFIXES):
        return True
    return False

# Gather all source text concatenated (so we see cross-file references)
all_text = ""
file_offsets = {}
for fname in FILES:
    path = ROOT / fname
    if not path.exists():
        continue
    file_offsets[fname] = len(all_text)
    all_text += "\n" + path.read_text()

def location_of(pos: int) -> str:
    """Map offset in all_text → (filename, line)."""
    cur_file = "?"
    cur_off = 0
    for fname, off in sorted(file_offsets.items(), key=lambda x: x[1]):
        if off > pos:
            break
        cur_file = fname
        cur_off = off
    line = all_text[cur_off:pos].count("\n") + 1
    return f"{cur_file}:{line}"

# Find all struct bodies. typedef struct { ... } NAME; OR struct NAME { ... };
struct_body_re = re.compile(r"(?:typedef\s+struct(?:\s+\w+)?\s*{|struct\s+\w+\s*{)", re.MULTILINE)

suspicious = []
field_seen = set()  # dedup by (struct_name, field_name)

for sm in struct_body_re.finditer(all_text):
    body_start = sm.end()
    # Find matching close brace at depth 1 (allow nested anon unions/structs)
    depth = 1
    i = body_start
    while i < len(all_text) and depth > 0:
        c = all_text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    body_end = i - 1
    body = all_text[body_start:body_end]
    # Try to extract the struct name (after the closing brace)
    after = all_text[body_end:body_end + 80]
    sname_m = re.match(r"}\s*([A-Za-z_][A-Za-z_0-9]*)", after)
    sname = sname_m.group(1) if sname_m else "?"
    # Find flag-shaped fields
    for fm in FIELD_RE.finditer(body):
        fname_id = fm.group(1)
        if not is_flag_field(fname_id):
            continue
        key = (sname, fname_id)
        if key in field_seen:
            continue
        field_seen.add(key)
        # Count assignments: NAME.field = ... or NAME->field = ...
        # Also handle field initializers in C99 designated init: .field = ...
        # Also handle array indexing: .field[i] = ... or .field[i].sub = ...
        assign_re = re.compile(
            r"(?:[.>]" + re.escape(fname_id) + r"(?:\s*\[[^\]]*\])?\s*=\s*[^=])"
        )
        writes = len(assign_re.findall(all_text))
        # Also catch memcpy(field, ...) / memset(field, ...) — they write but
        # the assignment-detector misses them.
        memcpy_re = re.compile(
            r"\b(?:memcpy|memset|memmove)\s*\([^,)]*[.>]" + re.escape(fname_id) + r"\b"
        )
        writes += len(memcpy_re.findall(all_text))
        # Count READS: any other reference to the field name (not as assignment LHS).
        read_re = re.compile(r"[.>]" + re.escape(fname_id) + r"\b")
        reads_total = len(read_re.findall(all_text))
        reads = reads_total - writes
        if writes == 0 and reads > 0:
            decl_pos = body_start + fm.start()
            suspicious.append((location_of(decl_pos), sname, fname_id, writes, reads, fm.group(0).strip()))

print(f"Audited {len(field_seen)} flag-shaped struct fields across {len([f for f in FILES if (ROOT / f).exists()])} files")
print()
if not suspicious:
    print("CLEAN: no flag-shaped struct fields with zero writes + non-zero reads")
else:
    print(f"FOUND {len(suspicious)} suspicious fields:")
    print()
    for loc, sname, fname, writes, reads, decl in suspicious:
        print(f"  {loc}  {sname}.{fname}")
        print(f"    writes={writes}  reads={reads}")
        print(f"    decl: {decl}")
        print()
