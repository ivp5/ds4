#!/usr/bin/env bash
# silv 2026-05-27 task #645 — Python ↔ GPU cross-validation smoke.
#
# 1. Python generates W.fp16
# 2. Python writes Wh.fp16 = H(W) via encode.py
# 3. C program loads Wh.fp16, applies GPU Hadamard → Whh = H(Wh)
# 4. Compares Whh against W. If both code paths apply the same H,
#    H(H(W)) = W within FP16 precision.

set -euo pipefail
cd "$(dirname "$0")/../.."

OUTDIR=tmp/20260527_hadamard

# Step 1: Generate random W
python3 - <<PY
import numpy as np
rng = np.random.default_rng(0xabcdef)
W = rng.standard_normal((256, 7168)).astype(np.float16)
open("$OUTDIR/W.fp16", "wb").write(W.tobytes())
print(f"created W.fp16: shape={W.shape}")
PY

# Step 2: Hadamard-encode in Python
python3 "$OUTDIR/encode.py" --in "$OUTDIR/W.fp16" --shape 256x7168 --out "$OUTDIR/Wh.fp16"

# Step 3: GPU Hadamard via C, compare against W
"$OUTDIR/cross_gpu_python" "$OUTDIR/W.fp16" "$OUTDIR/Wh.fp16" 256 7168

# Cleanup
rm -f "$OUTDIR/W.fp16" "$OUTDIR/Wh.fp16"
echo
echo "=== cross-validation complete ==="
