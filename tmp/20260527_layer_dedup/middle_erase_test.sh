#!/usr/bin/env bash
# Middle-cache-row erasure test: at gen-time attention, set softmax weights
# of cache rows [lo,hi] to ~0. Tests silv's "80% middle disposable" claim.
#
# Test set: 5 prompts where the load-bearing info is at different positions.
# For each prompt, run baseline + mask middle 50% of cache rows. Score whether
# output remains coherent + correct.
#
# Mask range is set per-prompt based on prompt's token count: mask the middle
# 50% (rows from 25% to 75% of context length).

set +u
cd /Users/silv/cl/tlp/montyneg/ivp5_ds4

MODEL=/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
TS=$(date +%Y%m%dT%H%M%S)
LOG=tmp/20260527_layer_dedup/middle_erase_${TS}.log

FILLER="The Romans built aqueducts that supplied freshwater to many distant cities. These structures were engineering marvels. Modern bridges use various design philosophies. Truss bridges distribute loads through triangular members. The Pont du Gard in southern France is one famous example. Built around the first century AD it was part of a fifty kilometer aqueduct. Roman engineers used arch construction extensively. Cable-stayed bridges combine elements of suspension and truss design."

run() {
    local label="$1"
    local prompt="$2"
    local mask="$3"
    echo "" | tee -a $LOG
    echo "=== $label  MASK=$mask ===" | tee -a $LOG
    if [ -n "$mask" ]; then
        DS4_KV_MASK_RANGE="$mask" timeout 300 ./ds4 --model "$MODEL" \
            --prefill-metal-phases auto \
            --prompt "$prompt" --tokens 32 2>&1 | tail -6 | tee -a $LOG
    else
        timeout 300 ./ds4 --model "$MODEL" \
            --prefill-metal-phases auto \
            --prompt "$prompt" --tokens 32 2>&1 | tail -6 | tee -a $LOG
    fi
}

# Each prompt has ~130 tokens.  Mask=30-100 erases the middle 70 rows.
P1="The capital of France is Paris. $FILLER What is the capital of France?"
P2="$FILLER The account number is 8217. What is the account number?"
P3="The sum of 17 and 25 is 42. $FILLER What was the sum I mentioned?"
P4="$FILLER What is 8 times 9 minus 3?"
P5="My name is Sam. $FILLER What is my name?"

echo "TS=$TS  Middle-mask range: rows 30-100 (middle ~50% of ~130-token contexts)" | tee $LOG

run "P1-fact-at-front (baseline)" "$P1" ""
run "P1-fact-at-front (middle masked)" "$P1" "30-100"

run "P2-fact-after-filler (baseline)" "$P2" ""
run "P2-fact-after-filler (middle masked)" "$P2" "30-100"

run "P3-fact-at-front (baseline)" "$P3" ""
run "P3-fact-at-front (middle masked)" "$P3" "30-100"

run "P4-question-only (baseline)" "$P4" ""
run "P4-question-only (middle masked)" "$P4" "30-100"

run "P5-name-at-front (baseline)" "$P5" ""
run "P5-name-at-front (middle masked)" "$P5" "30-100"

echo "" | tee -a $LOG
echo "LOG: $LOG"
