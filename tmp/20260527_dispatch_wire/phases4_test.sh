#!/usr/bin/env bash
# silv 2026-05-27 — phases=4 test for the FP16 simdgroup dispatch wire.
#
# v1 used --prefill-metal-phases auto (N=2 phases of ~46 GB each); the
# 12.88 GB hot-store pin pushed total residency past the 64 GB cap → OOM.
#
# With --prefill-metal-phases 4, each phase serves ~23 GB of routed
# expert residency. Hot-store 12.88 + per-phase 23 + non-routed 8.4 =
# 44.28 GB total. Fits in 64 GB with ~12 GB headroom.
#
# Expected behavior:
#   - Dispatch fires on layer 34 when prefill reaches it during phase 3
#   - No Metal OOM at command-buffer commit
#   - Greedy first-token output matches baseline (FP16 vs IQ2 should
#     agree on top-1 for this prompt; numerical drift is small)

set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL=/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
PROMPT='Explain in two sentences why mathematical proof matters in computer science.'
TS=$(date +%Y%m%dT%H%M%S)
OUTDIR=tmp/20260527_dispatch_wire

mkdir -p "$OUTDIR"

LOGFILE="$OUTDIR/phases4_fp16_simdgroup_${TS}.log"

echo "=== phases=4 + DS4_HOT_PIN_LAYERS=34 + DS4_HOT_FP16_KERNEL=1 ===" | tee "$LOGFILE"

DS4_HOT_PIN_LAYERS=34 DS4_HOT_FP16_KERNEL=1 ./ds4 \
    --model "$MODEL" \
    --prefill-metal-phases 4 \
    --threads 10 \
    --prompt "$PROMPT" \
    --tokens 16 \
    2>&1 | tee -a "$LOGFILE"

echo
echo "=== summary ==="
echo "log: $LOGFILE"
echo
echo "DISPATCHED markers:"
grep -c "FP16 simdgroup mat-mat DISPATCHED" "$LOGFILE" || echo "0"
echo
echo "Memory errors:"
grep -c "Insufficient Memory\|OutOfMemory" "$LOGFILE" || echo "0"
echo
echo "Output completion:"
grep -E "prefill:.*generation:|prompt processing failed|ds4-bench:.*runtime" "$LOGFILE" || true
