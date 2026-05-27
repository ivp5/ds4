#!/usr/bin/env bash
# silv 2026-05-27 — smoke test v2 for the FP16 simdgroup dispatch wire.
#
# v1 result: dispatch wire fires at layer=34 n_tokens=21, but OOM at the
# Metal command buffer step because hot-store (12.88 GB) + phase-1 routed
# residency (~46 GB) + non-routed (~8 GB) exceeds the 64 GB cap.
#
# v2 strategy: validate that
#   (1) DS4_HOT_FP16_KERNEL=1 with NO pin does NOT crash and falls back to IQ2
#   (2) DS4_HOT_FP16_KERNEL=1 with PARTIAL pin (cap experts) gates the
#       FP16 path off via ds4_hot_layer_fully_pinned and falls back to IQ2
#   (3) Output matches a baseline run (no DS4 envs) within greedy top-1

set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL=/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
PROMPT='Explain in two sentences why mathematical proof matters in computer science.'
TS=$(date +%Y%m%dT%H%M%S)
OUTDIR=tmp/20260527_dispatch_wire

mkdir -p "$OUTDIR"

run_one() {
    local label="$1"
    shift
    local logfile="$OUTDIR/${label}_${TS}.log"
    echo "=== run: $label ===" | tee "$logfile"
    "$@" \
        --model "$MODEL" \
        --prefill-metal-phases auto \
        --threads 10 \
        --prompt "$PROMPT" \
        --tokens 16 \
        2>&1 | tee -a "$logfile"
    echo "log: $logfile"
}

# Run 1: baseline — no FP16 env, no pin. IQ2_XXS path runs everywhere.
run_one baseline ./ds4

# Run 2: FP16 env on but NO pin. Hot-store unallocated; the dispatch
# helper returns 0 immediately (no active store); fallback to IQ2.
DS4_HOT_FP16_KERNEL=1 run_one fp16_env_no_pin ./ds4

# Run 3: FP16 env on + partial pin via DS4_HOT_PIN_EXPERTS_MAX=128.
# Hot-store has 128/256 experts pinned; ds4_hot_layer_fully_pinned
# returns 0; fallback to IQ2. Verifies the gating logic.
DS4_HOT_PIN_LAYERS=10 DS4_HOT_PIN_EXPERTS_MAX=128 DS4_HOT_FP16_KERNEL=1 \
    run_one fp16_partial_pin ./ds4 || echo "(partial pin run: nonzero exit — OOM expected at phase-2 residency)"

echo
echo "=== summary ==="
for label in baseline fp16_env_no_pin fp16_partial_pin; do
    log="$OUTDIR/${label}_${TS}.log"
    [[ -f "$log" ]] || continue
    echo
    echo "--- $label ---"
    grep -E "FP16 simdgroup mat-mat DISPATCHED|prefill:.*generation:|Insufficient Memory|fully_pinned|simdgroup mat-mat READY" "$log" | head -10 || true
done

# Output-equivalence check (greedy first-token):
echo
echo "=== first generated token comparison ==="
for label in baseline fp16_env_no_pin fp16_partial_pin; do
    log="$OUTDIR/${label}_${TS}.log"
    [[ -f "$log" ]] || continue
    # The model emits text after the input echo. Find the first non-system line.
    awk '/^processing/ {found=1; next} found && /^[A-Z]/ {print "  "FILENAME": "$0; exit}' "$log" || true
done
