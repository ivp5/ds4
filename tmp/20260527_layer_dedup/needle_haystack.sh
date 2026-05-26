#!/usr/bin/env bash
# Middle-cache-position disposability test via needle-in-haystack.
# Same needle (secret code 4221) placed at: front, middle, end. Filler is
# uncorrelated text. Question is always at the end. If middle positions are
# nearly disposable, the MIDDLE placement should fail at significantly higher
# rate than FRONT or END placements.
#
# silv conjecture: 80% of cells should survive middle-erasure → which would
# imply middle-position info IS NOT load-bearing, so middle-placement should
# WORK because the model is already skipping middle. Inverse prediction:
# middle-position info should FAIL because the model under-attends middle.
# Either way, this test bounds the answer.

set +u
cd /Users/silv/cl/tlp/montyneg/ivp5_ds4

MODEL=/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
TS=$(date +%Y%m%dT%H%M%S)
LOG=tmp/20260527_layer_dedup/needle_${TS}.log

FILLER_A="The Romans built aqueducts that supplied freshwater to many distant cities. These structures were engineering marvels. The aqueducts often spanned valleys and ravines. Some still stand today as monuments to ancient engineering. The Pont du Gard in southern France is one famous example. Built around the first century AD, it was part of a fifty kilometer aqueduct. Roman engineers used arch construction extensively."

FILLER_B="Modern bridges use various design philosophies. Truss bridges distribute loads through triangular members. Suspension bridges use cables to support the deck. Cable-stayed bridges combine elements of both. The Golden Gate Bridge opened in nineteen thirty seven. It was the longest suspension bridge until nineteen sixty four. Engineering progress continues to enable longer spans. Newer materials allow lighter structures."

NEEDLE="IMPORTANT: my account number is 8217."
QUESTION="What was my account number? Answer with just the number: "

run() {
    local label="$1"
    local prompt="$2"
    echo "" | tee -a $LOG
    echo "=== $label ===" | tee -a $LOG
    timeout 240 ./ds4 --model "$MODEL" \
        --prefill-metal-phases auto \
        --prompt "$prompt" --tokens 20 2>&1 | tail -5 | tee -a $LOG
}

echo "TS=$TS  Needle='8217'  Question expects '8217'" | tee $LOG

# FRONT: needle at start (positions ~0-20)
run "FRONT placement" \
    "$NEEDLE $FILLER_A $FILLER_B $QUESTION"

# MIDDLE: needle in middle (positions ~50-70)
run "MIDDLE placement" \
    "$FILLER_A $NEEDLE $FILLER_B $QUESTION"

# END: needle just before question (positions ~120-140)
run "END placement" \
    "$FILLER_A $FILLER_B $NEEDLE $QUESTION"

# SHORT baseline (no filler at all)
run "SHORT baseline (no filler)" "$NEEDLE $QUESTION"

# CONTROL: no needle, just filler — should NOT produce 8217
run "CONTROL no-needle (should fail to recall 8217)" \
    "$FILLER_A $FILLER_B $QUESTION"

echo "" | tee -a $LOG
echo "LOG: $LOG"
