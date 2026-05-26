#!/usr/bin/env bash
# DS4 layer-dedup test on FULL untrimmed IQ2_XXS (86 GB).
# Per prior weight-cosine proxy finding:
#   L22-L31 amplification candidates (cos sim 0.78-0.87)
#   L33-L36: high amplification (cos 0.92-0.96) — best dedup candidates
#   L36/37/38: commit triple (cos 0.26) — CANNOT skip
#   L38-L42: re-amplifying (cos 0.79-0.88)
#
# Test: skip various combinations, measure quality on capital + math prompt.
# Uses --prefill-metal-phases auto (gen on CPU MoE, ~1.5 t/s but correct).
set +u
cd /Users/silv/cl/tlp/montyneg/ivp5_ds4

MODEL=/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
TS=$(date +%Y%m%dT%H%M%S)
LOG=tmp/20260527_layer_dedup/full_ds4_${TS}.log

PROMPTS=(
  "The capital of France is "
  "What is 2+2? Answer with just the number: "
)

run_skip() {
    local label="$1"
    local skip="$2"
    local prompt="$3"
    echo "" | tee -a $LOG
    echo "--- $label  SKIP=$skip ---" | tee -a $LOG
    if [ -n "$skip" ]; then
        DS4_LAYER_SKIP_LIST="$skip" timeout 90 ./ds4 --model "$MODEL" \
            --prefill-metal-phases auto \
            --prompt "$prompt" --tokens 32 2>&1 \
            | grep -vE "^ds4: (Metal|context|drift|tokens|metal|loading|MTP|graph|backend|cpu)" \
            | tail -5 | tee -a $LOG
    else
        timeout 90 ./ds4 --model "$MODEL" \
            --prefill-metal-phases auto \
            --prompt "$prompt" --tokens 32 2>&1 \
            | grep -vE "^ds4: (Metal|context|drift|tokens|metal|loading|MTP|graph|backend|cpu)" \
            | tail -5 | tee -a $LOG
    fi
}

echo "=== FULL untrimmed DS4 layer-dedup sweep ===" | tee $LOG

for PROMPT in "${PROMPTS[@]}"; do
    echo "" | tee -a $LOG
    echo "###################" | tee -a $LOG
    echo "PROMPT: $PROMPT" | tee -a $LOG
    echo "###################" | tee -a $LOG

    run_skip "baseline"                  ""              "$PROMPT"
    run_skip "skip L34 (amp candidate)"  "34"            "$PROMPT"
    run_skip "skip L34,L35"              "34,35"         "$PROMPT"
    run_skip "skip L33,L34,L35"          "33,34,35"      "$PROMPT"
    run_skip "skip L40,L41,L42"          "40,41,42"      "$PROMPT"
    run_skip "skip L36,L37,L38 (commit)" "36,37,38"      "$PROMPT"
    run_skip "skip 5 spread"             "33,34,40,41,42" "$PROMPT"
done

echo "" | tee -a $LOG
echo "LOG: $LOG"
