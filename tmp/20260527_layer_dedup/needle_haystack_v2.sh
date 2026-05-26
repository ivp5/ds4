#!/usr/bin/env bash
# v2: enough tokens to clear CoT preamble + reach answer
set +u
cd /Users/silv/cl/tlp/montyneg/ivp5_ds4

MODEL=/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
TS=$(date +%Y%m%dT%H%M%S)
LOG=tmp/20260527_layer_dedup/needle_v2_${TS}.log

FILLER_A="The Romans built aqueducts that supplied freshwater to many distant cities. These structures were engineering marvels. The aqueducts often spanned valleys and ravines. Some still stand today as monuments to ancient engineering. The Pont du Gard in southern France is one famous example. Built around the first century AD, it was part of a fifty kilometer aqueduct. Roman engineers used arch construction extensively."

FILLER_B="Modern bridges use various design philosophies. Truss bridges distribute loads through triangular members. Suspension bridges use cables to support the deck. Cable-stayed bridges combine elements of both. The Golden Gate Bridge opened in nineteen thirty seven. It was the longest suspension bridge until nineteen sixty four. Engineering progress continues to enable longer spans. Newer materials allow lighter structures."

NEEDLE="IMPORTANT FACT: the account number is 8217."
QUESTION="What is the account number? The number is "

run() {
    local label="$1"
    local prompt="$2"
    echo "" | tee -a $LOG
    echo "=== $label ===" | tee -a $LOG
    timeout 300 ./ds4 --model "$MODEL" \
        --prefill-metal-phases auto \
        --prompt "$prompt" --tokens 48 2>&1 | tail -7 | tee -a $LOG
}

echo "TS=$TS  Needle='8217'" | tee $LOG

# FRONT: needle at start
run "FRONT (needle at pos ~5)"     "$NEEDLE $FILLER_A $FILLER_B $QUESTION"
# MIDDLE: needle in middle
run "MIDDLE (needle at pos ~55)"   "$FILLER_A $NEEDLE $FILLER_B $QUESTION"
# END: needle just before question
run "END (needle at pos ~110)"     "$FILLER_A $FILLER_B $NEEDLE $QUESTION"
# SHORT baseline
run "SHORT (no filler)"            "$NEEDLE $QUESTION"
# CONTROL: no needle present at all
run "NO-NEEDLE control"            "$FILLER_A $FILLER_B $QUESTION"

echo "" | tee -a $LOG
echo "LOG: $LOG"
