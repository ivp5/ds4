"""Wrapper: probe per-token compute on a CHUNK starting at offset."""
import argparse, json, sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--json", required=True)
p.add_argument("--field", default="response")
p.add_argument("--offset", type=int, default=0)
p.add_argument("--n-chars", type=int, default=1500)
p.add_argument("--out", required=True)
p.add_argument("--notes", default="")
args = p.parse_args()

data = json.loads(Path(args.json).read_text())
text = data[args.field][args.offset:args.offset + args.n_chars]
out = {args.field: text}
Path(args.out).write_text(json.dumps(out))
print(f"wrote {args.out} ({len(text)} chars from offset {args.offset})")
print(f"head: {text[:120]!r}")
