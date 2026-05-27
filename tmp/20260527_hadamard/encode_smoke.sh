#!/usr/bin/env bash
# silv 2026-05-27 task #645 — smoke test for the Hadamard encode tool.
#
# Generates a small synthetic FP16 weight, runs encode.py through the full
# pipeline (read → transform → verify → write), then re-reads the output
# and checks the round-trip via Python.

set -euo pipefail
cd "$(dirname "$0")"

TS=$(date +%Y%m%dT%H%M%S)

python3 - <<'PY'
import numpy as np
from pathlib import Path
rng = np.random.default_rng(0xfeed)
# 256 output × 7168 input — matches DS4 routed expert gate/up tile shape (n_ffn × n_embd).
W = rng.standard_normal((256, 7168)).astype(np.float16)
Path("W_test.fp16").write_bytes(W.tobytes())
print(f"created W_test.fp16: shape={W.shape}, dtype={W.dtype}, "
      f"min={W.min():.4f}, max={W.max():.4f}, norm={np.linalg.norm(W.astype(np.float32)):.4f}")
PY

echo
echo "=== encode + verify ==="
python3 encode.py --in W_test.fp16 --shape 256x7168 --out Wh_test.fp16 --verify

echo
echo "=== read back + compare against in-Python reference ==="
python3 - <<'PY'
import numpy as np
import sys
sys.path.insert(0, ".")
from encode import hadamard16_block

W = np.frombuffer(open("W_test.fp16", "rb").read(), dtype=np.float16).reshape(256, 7168)
Wh_disk = np.frombuffer(open("Wh_test.fp16", "rb").read(), dtype=np.float16).reshape(256, 7168)
Wh_ref = hadamard16_block(W)

# What encode.py wrote should match the in-process reference exactly (same code path).
diff = np.abs(Wh_disk.astype(np.float32) - Wh_ref.astype(np.float32))
print(f"disk-vs-ref: max|err|={diff.max():.4e}  rel_L2={np.linalg.norm(diff)/np.linalg.norm(Wh_ref.astype(np.float32)):.4e}")
assert diff.max() < 1.0e-6, "disk output differs from in-process reference"

# Also verify <Wh, xh> ≈ <W, x>.
rng = np.random.default_rng(0x42)
x = rng.standard_normal(7168).astype(np.float16)
xh = hadamard16_block(x.copy())
y_orig = (W.astype(np.float32) @ x.astype(np.float32))
y_hada = (Wh_disk.astype(np.float32) @ xh.astype(np.float32))
diff_y = np.abs(y_hada - y_orig)
rel = diff_y / np.maximum(np.abs(y_orig), 1e-3)
print(f"matmul: max_rel_err={rel.max():.4e}  mean_rel_err={rel.mean():.4e}  (FP16 floor ~5e-2)")
assert rel.max() < 5.0e-2, f"matmul equivalence broken: max_rel={rel.max()}"

print("PASS")
PY

# Clean up
rm -f W_test.fp16 Wh_test.fp16
echo
echo "=== smoke complete ==="
