#!/usr/bin/env python3
"""Surgical resolver for ds4_metal.m antirez/main merge conflicts.

Conflicts 1-4 (declaration/body updates): take antirez (THEIRS).
Conflicts 5-6 (spurious; merge tool mis-placed antirez's wide-tile MoE
encode block inside my softplus_sqrt MTL4 port): keep HEAD (OURS).
The wide-tile MoE encode logic will be applied separately by manual
patch to the real MoE encode function.
"""
import re
import sys
from pathlib import Path

PATH = Path("/Users/silv/cl/tlp/montyneg/ivp5_ds4/ds4_metal.m")
src = PATH.read_text()

# We have 6 conflict blocks. Resolutions in order:
# 1: theirs (line 2793 — function declaration)
# 2: theirs (line 13171 — pipeline functions)
# 3: theirs (line 13877 — encode body)
# 4: theirs (line 13950 — encode wrapper)
# 5: ours (line 16732 — softplus_sqrt init; antirez's text is misplaced)
# 6: ours (line 16864 — softplus_sqrt canary; antirez's text is misplaced)

RESOLUTIONS = ["theirs", "theirs", "theirs", "theirs", "ours", "ours"]

pattern = re.compile(
    r"<<<<<<< HEAD\n(.*?)\n=======\n(.*?)\n>>>>>>> antirez/main\n",
    re.DOTALL,
)

matches = list(pattern.finditer(src))
print(f"Found {len(matches)} conflicts; resolutions: {RESOLUTIONS}", file=sys.stderr)
assert len(matches) == len(RESOLUTIONS), f"Mismatch: {len(matches)} vs {len(RESOLUTIONS)}"

# Build resolved file by walking matches in reverse so indexes stay stable
out = src
for i in range(len(matches) - 1, -1, -1):
    m = matches[i]
    ours, theirs = m.group(1), m.group(2)
    resolved = ours if RESOLUTIONS[i] == "ours" else theirs
    # Restore trailing newline that pattern consumed
    resolved_with_nl = resolved + "\n"
    out = out[: m.start()] + resolved_with_nl + out[m.end() :]
    print(f"  conflict #{i+1}: applied {RESOLUTIONS[i]} ({len(resolved)} chars)", file=sys.stderr)

# Verify no markers remain
remaining = re.findall(r"^<<<<<<<|^=======|^>>>>>>> ", out, re.M)
if remaining:
    print(f"ERROR: {len(remaining)} markers remain after resolution", file=sys.stderr)
    sys.exit(1)

PATH.write_text(out)
print("resolved", file=sys.stderr)
