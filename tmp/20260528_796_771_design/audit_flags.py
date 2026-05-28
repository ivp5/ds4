#!/usr/bin/env python3
"""Audit for Cycle-5-shape latent bugs across ds4 source files.

Finds: flag-shaped static globals (matching _active/_enabled/_ok/_inited/
_attempted/_loaded/... suffix) that have ZERO writes anywhere in the
source. These are the same shape as `override_source_active` — declared
but never assigned, so they always return zero-init value and the
downstream code path is dormant.

silv 2026-05-28 audit per Cycle 5 retrospective.
"""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
FILES = [
    "ds4.c", "ds4_metal.m", "ds4_metal_vqb2_fp16.m", "ds4_inflight.c",
    "ds4_expert_table.c", "ds4_moe_route_log.c", "ds4_polar_reader.c",
    "ds4_vqb1_reader.c", "ds4_vqb2_reader.c", "ds4_vqb2_pack.c",
    "ds4_watersic_pack.c", "ds4_nonrouted_pack.c", "ds4_prefix_cache.c",
]

FLAG_SUFFIXES = (
    "_active", "_enabled", "_ok", "_ready", "_inited", "_initialized",
    "_done", "_loaded", "_open", "_set", "_present", "_valid", "_attempted",
    "_armed", "_running", "_dirty", "_pending", "_locked", "_busy", "_held",
    "_init_ok", "_init_attempted",
)

DECL_RE = re.compile(
    r"^static\s+(?:int|uint8_t|bool|_Bool|uint32_t|uint16_t|uint64_t|atomic_int|atomic_bool)\s+"
    r"((?:g_|s_)[A-Za-z_][A-Za-z_0-9]*)\s*[;=]"
)

def is_flag(sym: str) -> bool:
    return any(sym.endswith(suf) for suf in FLAG_SUFFIXES)

def audit_file(path: Path):
    text = path.read_text()
    suspicious = []
    for m in DECL_RE.finditer(text):
        sym = m.group(1)
        if not is_flag(sym):
            continue
        # Count assignments (sym = SOMETHING, not sym == SOMETHING)
        # Exclude the declaration line itself
        assign_pat = re.compile(r"\b" + re.escape(sym) + r"\s*=\s*[^=]")
        all_assigns = assign_pat.findall(text)
        # Subtract initializer on declaration if present
        decl_text = text[m.start():text.find("\n", m.start())]
        has_initializer = "=" in decl_text and not decl_text.endswith(";\n")
        writes = len(all_assigns) - (1 if has_initializer else 0)
        # Subtract initializer-on-decl form like `= 0;` on the decl line
        if "= 0;" in decl_text or "= 1;" in decl_text or "= NULL;" in decl_text:
            writes -= 1
        # Count reads (any occurrence not in an assignment LHS)
        all_refs = re.findall(r"\b" + re.escape(sym) + r"\b", text)
        reads = len(all_refs) - len(all_assigns) - 1  # subtract decl
        if writes <= 0 and reads > 0:
            suspicious.append((sym, m.start(), writes, reads, decl_text.strip()))
    return suspicious

all_suspicious = []
for fname in FILES:
    path = ROOT / "ivp5_ds4" / fname
    if not path.exists():
        continue
    sus = audit_file(path)
    for sym, pos, writes, reads, decl in sus:
        line_no = path.read_text()[:pos].count("\n") + 1
        all_suspicious.append((fname, line_no, sym, writes, reads, decl))

if not all_suspicious:
    print("CLEAN: no flag-shaped globals with zero writes + non-zero reads")
else:
    print(f"FOUND {len(all_suspicious)} suspicious flag-shaped globals:")
    print()
    for fname, line_no, sym, writes, reads, decl in all_suspicious:
        print(f"  {fname}:{line_no}  {sym}")
        print(f"    writes={writes}  reads={reads}")
        print(f"    decl: {decl}")
        print()
