#!/usr/bin/env python3
"""Fix-up: re-add `NSError *err = nil;` declarations in functions where the
previous script removed it but `err` is still referenced elsewhere in the
function body.

Heuristic: walk functions (delimited by ^static int|static id<...> followed
by '{' through matching '}'). If body uses 'err' (e.g., 'error:&err' or
'err.localizedDescription') AND no 'NSError *err' declaration exists in body,
inject one after the opening brace.
"""
import re
import sys
from pathlib import Path

PATH = Path("/Users/silv/cl/tlp/montyneg/ivp5_ds4/ds4_metal.m")
src = PATH.read_text()
lines = src.split("\n")

# Find function starts
n_lines = len(lines)
fixes = 0
i = 0
out = []
while i < n_lines:
    line = lines[i]
    out.append(line)
    # Detect function opening — must match pattern like:
    # 'static int ds4_XXX_pipeline_init(void) {'
    # or 'int ds4_gpu_mtl4_XXX_canary(...) {' (skip canaries, they don't use the helper)
    if re.match(r'^static int ds4_[a-zA-Z0-9_]+_pipeline_init\(void\) \{$', line):
        # Find matching close brace + scan body
        depth = 1
        body_lines = []
        j = i + 1
        while j < n_lines and depth > 0:
            l2 = lines[j]
            # Cheap brace counter (works for our well-formatted code)
            depth += l2.count("{") - l2.count("}")
            body_lines.append((j, l2))
            j += 1
        body_text = "\n".join(l for _, l in body_lines)
        uses_err = ("&err" in body_text or "err.localizedDescription" in body_text or
                    "err ? err.localizedDescription" in body_text)
        has_decl = "NSError *err" in body_text
        if uses_err and not has_decl:
            # Inject after the line that sets *_init_attempted = 1 (typical first 2-3 lines)
            inject_after = None
            for k, (lineno, bl) in enumerate(body_lines):
                if "_init_attempted = 1;" in bl:
                    inject_after = lineno
                    break
            if inject_after is None:
                # Fallback: inject right after function opening
                inject_after = i
            # We'll insert "    NSError *err = nil;" after inject_after
            # Build out by accumulating up to + including inject_after, then inserting
            # decl, then continuing.
            # Strategy: emit body_lines but watch for inject_after
            for (lineno, bl) in body_lines:
                out.append(bl)
                if lineno == inject_after:
                    out.append("    NSError *err = nil;")
                    fixes += 1
            i = j
            continue
        else:
            # No fix needed; emit body unchanged
            for (lineno, bl) in body_lines:
                out.append(bl)
            i = j
            continue
    i += 1

if fixes > 0:
    PATH.write_text("\n".join(out))
print(f"restored NSError *err = nil; in {fixes} function(s)", file=sys.stderr)
