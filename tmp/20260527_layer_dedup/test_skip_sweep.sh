#!/usr/bin/env bash
# DS4 layer-dedup test — skip late-layer amplification candidates on trim50.
# Per prior weight-similarity finding: L33-L36 + L38-L42 are amplification
# candidates (cos sim 0.79-0.96 with neighbors); L36/37/38 are commit-triple
# discontinuities (cos ~0.26) — DO NOT skip.
set +u
cd /Users/silv/cl/tlp/montyneg/ivp5_ds4

MODEL=/Users/silv/cl/tlp/montyneg/ds4/gguf/DS4-trim50-asym-with-metadata.gguf
TS=$(date +%Y%m%dT%H%M%S)
LOG=tmp/20260527_layer_dedup/sweep_${TS}.log
mkdir -p tmp/20260527_layer_dedup

PROMPT="The quick brown fox jumps over the lazy dog. The capital of France is "

echo "=== DS4 layer-skip × quality on trim50 ===" | tee $LOG
echo "Prompt: $PROMPT" | tee -a $LOG
echo "" | tee -a $LOG

run_skip() {
    local label="$1"
    local skip="$2"
    echo "--- $label  SKIP=$skip ---" | tee -a $LOG
    if [ -n "$skip" ]; then
        DS4_LAYER_SKIP_LIST="$skip" timeout 60 ./ds4 --model "$MODEL" \
            --prompt "$PROMPT" --tokens 48 2>&1 \
            | grep -vE "^ds4: (Metal|context|drift|tokens|metal|loading)" | tail -10 | tee -a $LOG
    else
        timeout 60 ./ds4 --model "$MODEL" \
            --prompt "$PROMPT" --tokens 48 2>&1 \
            | grep -vE "^ds4: (Metal|context|drift|tokens|metal|loading)" | tail -10 | tee -a $LOG
    fi
    echo "" | tee -a $LOG
}

run_skip "baseline (no skip)"           ""
run_skip "skip 1 late-amp (L34)"        "34"
run_skip "skip 2 late-amp (L34,L35)"    "34,35"
run_skip "skip 3 late-amp (L33,L34,L35)" "33,34,35"
run_skip "skip late-amp L40-L42"        "40,41,42"
run_skip "skip commit-triple L36-L38 (UNSAFE per finding)" "36,37,38"
run_skip "skip 5 spread (33,34,40,41,42)" "33,34,40,41,42"

echo "LOG: $LOG"
