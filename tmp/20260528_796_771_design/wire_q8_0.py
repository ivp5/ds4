#!/usr/bin/env python3
"""Wire remaining Q8_0 production call sites through ds4_matmul_q8_0_via_tensor.

Pattern transform:
  ds4_gpu_matmul_q8_0_tensor(DST, MODEL_ARGS, X->abs_offset, IN, OUT, SRC, N)
  →
  ds4_matmul_q8_0_via_tensor(DST, model, X, IN, OUT, SRC, N)

where MODEL_ARGS is `model->map, model->size,` (or g->cpu_model->map etc.,
preserved as the 2nd arg in the new form).

Safety:
- Skip the dispatcher itself (line ~12012) — it's the fallback.
- Skip if pattern doesn't match (e.g., offset isn't `X->abs_offset`).
- Dry-run by default; pass --apply to actually edit.
"""

import re
import sys
import pathlib

SRC = pathlib.Path(__file__).resolve().parent.parent.parent / "ds4.c"
APPLY = "--apply" in sys.argv

text = SRC.read_text()
lines = text.split("\n")

# Find every "ds4_gpu_matmul_q8_0_tensor(" call; expand to closing ")"
calls = []
i = 0
while i < len(lines):
    if "ds4_gpu_matmul_q8_0_tensor(" in lines[i]:
        start = i
        depth = 0
        end = i
        chunk = []
        for j in range(start, min(start + 25, len(lines))):
            chunk.append(lines[j])
            depth += lines[j].count("(") - lines[j].count(")")
            if depth <= 0 and j > start:
                end = j
                break
            elif depth <= 0 and lines[j].count(")") > 0 and "ds4_gpu_matmul_q8_0_tensor(" in lines[j]:
                # single-line call
                end = j
                break
        calls.append((start, end, chunk))
        i = end + 1
    else:
        i += 1

print(f"Found {len(calls)} ds4_gpu_matmul_q8_0_tensor( call sites")

# For each call, parse args. Skip if this is the dispatcher's fallback.
def is_dispatcher_fallback(chunk):
    """The dispatcher's own fallback is `return ds4_gpu_matmul_q8_0_tensor(...)`."""
    full = "\n".join(chunk)
    return "return ds4_gpu_matmul_q8_0_tensor(dst, model->map, model->size," in full

def parse_call(chunk):
    """Extract (dst, weight_pointer_expr, in_dim, out_dim, src, n_tok)."""
    full = "\n".join(chunk)
    # Find the "ds4_gpu_matmul_q8_0_tensor(" position and walk forward
    # depth-tracking to find the matching ")". Plain regex doesn't work
    # because args can contain "()" (e.g. "(uint64_t)X * Y") which the
    # non-greedy match stops at.
    start = full.find("ds4_gpu_matmul_q8_0_tensor(")
    if start < 0:
        return None
    after = start + len("ds4_gpu_matmul_q8_0_tensor(")
    depth = 1
    i = after
    while i < len(full) and depth > 0:
        c = full[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    args_str = full[after:i - 1]
    # Split on commas at top level
    args = []
    depth = 0
    cur = ""
    for c in args_str:
        if c == "(" or c == "[":
            depth += 1
        elif c == ")" or c == "]":
            depth -= 1
        if c == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += c
    if cur.strip():
        args.append(cur.strip())
    if len(args) != 8:
        return None
    dst, model_map, model_size, weight_offset, in_dim, out_dim, src, n_tok = args
    # Extract tensor* from "X->abs_offset"
    wm = re.match(r"(.+?)\s*->\s*abs_offset\s*$", weight_offset)
    if not wm:
        return None
    tensor_ptr = wm.group(1).strip()
    return {
        "dst": dst,
        "model_map": model_map,
        "model_size": model_size,
        "tensor_ptr": tensor_ptr,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "src": src,
        "n_tok": n_tok,
    }

def fmt_call(parts, leading_ws):
    """Emit replacement multi-line call respecting original indentation."""
    return (
        f"{leading_ws}ds4_matmul_q8_0_via_tensor({parts['dst']}, model,\n"
        f"{leading_ws} {parts['tensor_ptr']},\n"
        f"{leading_ws} {parts['in_dim']}, {parts['out_dim']},\n"
        f"{leading_ws} {parts['src']}, {parts['n_tok']})"
    )

# Build replacement plan
plans = []
for (start, end, chunk) in calls:
    if is_dispatcher_fallback(chunk):
        continue
    parts = parse_call(chunk)
    if parts is None:
        print(f"  SKIP line {start+1}: cannot parse")
        continue
    # The model argument in the new call has to match the original's model context.
    # For sites that use `model->map, model->size`, model is just `model`.
    # For sites using `g->cpu_model->map, g->cpu_model->size`, model is `g->cpu_model`.
    # Etc. Derive by stripping `->map` from model_map.
    model_var = parts["model_map"].replace("->map", "")
    if model_var != "model":
        print(f"  NON-STD MODEL at line {start+1}: {model_var}")
        # We still write but use that name; the helper's signature takes
        # `const ds4_model *model` so the caller can pass any model*.
    plans.append((start, end, chunk, parts, model_var))

print(f"Wirable plans: {len(plans)}")

# Write replacement
if not APPLY:
    for (start, end, chunk, parts, model_var) in plans[:3]:
        print(f"\n=== Line {start+1} preview ===")
        print("BEFORE:")
        for cl in chunk:
            print(f"  {cl}")
        print("AFTER:")
        leading_ws = chunk[0][: len(chunk[0]) - len(chunk[0].lstrip(" "))]
        # Replace tensor_ptr's abs_offset with just the tensor
        # Build the new call line by line
        # First line preserves prefix before the call
        prefix_match = re.search(r"^(.*?)ds4_gpu_matmul_q8_0_tensor", chunk[0])
        prefix = prefix_match.group(1) if prefix_match else leading_ws
        new_first = f"{prefix}ds4_matmul_q8_0_via_tensor({parts['dst']}, {model_var},"
        new_body = (
            f"{leading_ws} {parts['tensor_ptr']},\n"
            f"{leading_ws} {parts['in_dim']}, {parts['out_dim']},\n"
            f"{leading_ws} {parts['src']}, {parts['n_tok']})"
        )
        # Suffix on last line (after ")") — preserve "!= 0;" etc.
        last = chunk[-1]
        close_idx = last.rfind(")")
        suffix = last[close_idx + 1:] if close_idx >= 0 else ""
        print(new_first)
        print(new_body + suffix)
    print(f"\n(showing first 3 of {len(plans)} — rerun with --apply to write)")
    sys.exit(0)

# Apply: rewrite from bottom up to preserve line numbers
new_lines = list(lines)
for (start, end, chunk, parts, model_var) in sorted(plans, key=lambda p: -p[0]):
    leading_ws = chunk[0][: len(chunk[0]) - len(chunk[0].lstrip(" "))]
    prefix_match = re.search(r"^(.*?)ds4_gpu_matmul_q8_0_tensor", chunk[0])
    prefix = prefix_match.group(1) if prefix_match else leading_ws
    last = chunk[-1]
    close_idx = last.rfind(")")
    suffix = last[close_idx + 1:] if close_idx >= 0 else ""
    new_call = (
        f"{prefix}ds4_matmul_q8_0_via_tensor({parts['dst']}, {model_var},\n"
        f"{leading_ws} {parts['tensor_ptr']},\n"
        f"{leading_ws} {parts['in_dim']}, {parts['out_dim']},\n"
        f"{leading_ws} {parts['src']}, {parts['n_tok']}){suffix}"
    )
    # Replace lines start..end inclusive with the multi-line new_call
    new_lines[start:end + 1] = new_call.split("\n")

SRC.write_text("\n".join(new_lines))
print(f"Applied {len(plans)} edits to {SRC}")
