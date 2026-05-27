#!/usr/bin/env bash
# silv 2026-05-27 — smoke test for the FP16 simdgroup mat-mat dispatch wire.
#
# Baseline: existing IQ2_XXS path runs (no DS4_HOT_FP16_KERNEL).
# Probe:    DS4_HOT_FP16_KERNEL=1 + DS4_HOT_PIN_LAYERS=34 → the new simdgroup
#           dispatch helper fires on layer 34 when n_tokens >= 8 during prefill.
#
# Both runs must produce the same first-token output (greedy, single-token
# generation) since the FP16 weights are dequantized from IQ2_XXS and the
# kernel computes the same gate*up*silu*route_w math, just in FP16 vs IQ2.
# A small numeric difference is expected but the greedy top-1 token should
# match on this short prompt.

set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL=/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
PROMPT='Explain in two sentences why mathematical proof matters in computer science.'
TS=$(date +%Y%m%dT%H%M%S)
OUTDIR=tmp/20260527_dispatch_wire

mkdir -p "$OUTDIR"

echo "=== run 1: baseline (no DS4_HOT_FP16_KERNEL) ===" | tee "$OUTDIR/baseline_${TS}.log"
./ds4 \
    --model "$MODEL" \
    --prefill-metal-phases auto \
    --threads 10 \
    --prompt "$PROMPT" \
    --tokens 16 \
    2>&1 | tee -a "$OUTDIR/baseline_${TS}.log"

echo
echo "=== run 2: FP16 simdgroup on layer 34 (DS4_HOT_PIN_LAYERS=34 + DS4_HOT_FP16_KERNEL=1) ===" | tee "$OUTDIR/fp16_simdgroup_${TS}.log"
DS4_HOT_PIN_LAYERS=34 DS4_HOT_FP16_KERNEL=1 ./ds4 \
    --model "$MODEL" \
    --prefill-metal-phases auto \
    --threads 10 \
    --prompt "$PROMPT" \
    --tokens 16 \
    2>&1 | tee -a "$OUTDIR/fp16_simdgroup_${TS}.log"

echo
echo "=== summary ==="
echo "baseline log:        $OUTDIR/baseline_${TS}.log"
echo "fp16_simdgroup log:  $OUTDIR/fp16_simdgroup_${TS}.log"
echo
echo "Looking for 'FP16 simdgroup mat-mat DISPATCHED' in the FP16 run:"
grep -c "FP16 simdgroup mat-mat DISPATCHED" "$OUTDIR/fp16_simdgroup_${TS}.log" || echo "  (no dispatch — check pin / token threshold)"
echo
echo "Looking for prefill / gen tps in both:"
grep -E "prefill:.*generation:|t/s" "$OUTDIR/baseline_${TS}.log" || true
echo "---"
grep -E "prefill:.*generation:|t/s" "$OUTDIR/fp16_simdgroup_${TS}.log" || true
