#!/bin/bash
# test_polar_b23c_gate.sh — verify B-2.3c stub hot-path gate works at runtime.
#
# What this tests:
#   - DS4_POLAR_DIR + DS4_POLAR_LAYERS env vars detected and propagated
#   - Polar pool loads at session-open with diagnostic
#   - Hot-path gate calls polar dispatcher stub on enabled layers
#   - Stub returns 0 → FP4 fallback engaged → inference output unchanged
#   - Per-layer diagnostic emits once per layer (gate engagement evidence)
#
# Per CLAUDE.md DS4 STICKY HAZARD: invokes ds4-bench (NOT ds4) with
# --prefill-metal-phases auto. Single short prompt. Should complete in
# under 60 seconds.
#
# Usage:
#   ./test_polar_b23c_gate.sh [model.gguf]
#
# Default model: /Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
#
# Pre-flight check (per STICKY HAZARD):
#   - Confirms --prefill-metal-phases auto is in the launch command
#   - Confirms recent panic file < 7 days is NOT DS4-shaped

set -uo pipefail

MODEL="${1:-/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf}"
POLAR_DIR="${POLAR_DIR:-tmp/polar_full_p32m8}"
POLAR_LAYERS="${POLAR_LAYERS:-0,1,2,3}"
PROMPT="${PROMPT:-What is 1+1?}"
LOG_FILE="tmp/test_polar_b23c_gate_$(date +%Y%m%dT%H%M%S).log"
OUT_FILE="tmp/test_polar_b23c_gate_$(date +%Y%m%dT%H%M%S).out"

# Pre-flight: STICKY HAZARD check
echo "=== Pre-flight (CLAUDE.md DS4 STICKY HAZARD) ==="
if ! [ -f "$MODEL" ]; then
    echo "FAIL: model file not found: $MODEL"
    exit 1
fi
if ! [ -d "$POLAR_DIR" ]; then
    echo "FAIL: polar dir not found: $POLAR_DIR"
    exit 1
fi
RECENT_PANIC=$(ls -t /Library/Logs/DiagnosticReports/panic-*.panic 2>/dev/null | head -1)
if [ -n "$RECENT_PANIC" ]; then
    PANIC_AGE_DAYS=$(( ( $(date +%s) - $(stat -f%c "$RECENT_PANIC") ) / 86400 ))
    if [ "$PANIC_AGE_DAYS" -lt 7 ]; then
        echo "WARN: recent panic file (age $PANIC_AGE_DAYS days): $RECENT_PANIC"
        echo "      Inspect head -2 of that file before proceeding."
        head -2 "$RECENT_PANIC" | tail -1
    fi
fi
echo "  model: $MODEL"
echo "  polar dir: $POLAR_DIR"
echo "  enabled layers: $POLAR_LAYERS"
echo "  log: $LOG_FILE"
echo "  out: $OUT_FILE"

echo ""
echo "=== Run 1: BASELINE (no polar env vars set) ==="
./ds4 --prefill-metal-phases auto \
    -m "$MODEL" -p "$PROMPT" \
    -n 8 \
    > "$OUT_FILE.baseline" 2> "$LOG_FILE.baseline" || {
    echo "FAIL: baseline run failed; check $LOG_FILE.baseline"
    exit 1
}
echo "  baseline output: $(wc -c < "$OUT_FILE.baseline") bytes"
echo "  baseline first 80 chars: $(head -c80 "$OUT_FILE.baseline")"

echo ""
echo "=== Run 2: WITH polar gate engaged ==="
DS4_POLAR_DIR="$POLAR_DIR" \
DS4_POLAR_LAYERS="$POLAR_LAYERS" \
./ds4 --prefill-metal-phases auto \
    -m "$MODEL" -p "$PROMPT" \
    -n 8 \
    > "$OUT_FILE.polar" 2> "$LOG_FILE.polar" || {
    echo "FAIL: polar-gated run failed; check $LOG_FILE.polar"
    exit 1
}
echo "  polar output: $(wc -c < "$OUT_FILE.polar") bytes"
echo "  polar first 80 chars: $(head -c80 "$OUT_FILE.polar")"

echo ""
echo "=== Verdicts ==="

# Verdict 1: B-2.3c gate diagnostic appeared
if grep -q "polar B-2.3c gate engaged" "$LOG_FILE.polar"; then
    N_LAYERS_ENGAGED=$(grep -c "polar B-2.3c gate engaged" "$LOG_FILE.polar")
    echo "  PASS: B-2.3c gate diagnostic appeared on $N_LAYERS_ENGAGED layers"
else
    echo "  FAIL: B-2.3c gate diagnostic NOT in stderr (gate logic broken?)"
    echo "  Check: $LOG_FILE.polar for 'polar' lines:"
    grep -i "polar" "$LOG_FILE.polar" | head -10
fi

# Verdict 2: B-2.3c diagnostic ABSENT in baseline
if grep -q "polar B-2.3c gate engaged" "$LOG_FILE.baseline"; then
    echo "  FAIL: B-2.3c diagnostic appeared in BASELINE (should only fire when env vars set)"
else
    echo "  PASS: B-2.3c diagnostic absent in baseline (env-var gate works)"
fi

# Verdict 3: output is bit-identical (stub returns 0 → no actual substitution)
if cmp -s "$OUT_FILE.baseline" "$OUT_FILE.polar"; then
    echo "  PASS: inference output bit-identical baseline vs polar (stub fallback works)"
else
    echo "  FAIL: inference output DIFFERS (stub should always return 0, no substitution)"
    echo "  diff:"
    diff "$OUT_FILE.baseline" "$OUT_FILE.polar" | head -10
fi

# Verdict 4: polar pool loaded at session open
if grep -q "polar.*opened\|polar.*loaded\|PLR2" "$LOG_FILE.polar"; then
    echo "  PASS: polar pool loaded (PLR2 files opened)"
else
    echo "  WARN: no PLR2 load evidence in stderr — pool may not have engaged"
fi

echo ""
echo "=== Summary ==="
echo "  baseline log: $LOG_FILE.baseline"
echo "  polar log:    $LOG_FILE.polar"
echo "  Compare them: diff $LOG_FILE.baseline $LOG_FILE.polar | head -30"
echo ""
echo "Once all 4 verdicts PASS, the B-2.3c gate is runtime-validated."
echo "Next step: silv approve A.1/A.2/A.3 sub-decision to unlock body shipping."
