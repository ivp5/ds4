#!/usr/bin/env bash
# Test attn scale on prompts where DS4 may have truth in CoT (Conjecture #23 case)
set +u
cd /Users/silv/cl/tlp/montyneg/ivp5_ds4

MODEL=/Users/silv/cl/tlp/montyneg/ds4/gguf/DS4-trim50-asym-with-metadata.gguf
TS=$(date +%Y%m%dT%H%M%S)
LOG=tmp/20260527_attn_scale/math_${TS}.log

# Math prompts
declare -a PROMPTS=(
  "What is 2+2? Answer with just the number: "
  "Compute 15 * 7. Show only the result: "
  "Solve: 100 - 37 = "
)

for PROMPT in "${PROMPTS[@]}"; do
  echo "" | tee -a $LOG
  echo "###################" | tee -a $LOG
  echo "PROMPT: $PROMPT" | tee -a $LOG
  echo "###################" | tee -a $LOG
  for MULT in 1.0 0.98 0.95 0.9 0.8 0.7 0.5; do
    echo "" | tee -a $LOG
    echo "--- scale=$MULT ---" | tee -a $LOG
    DS4_ATTN_SCALE_MULT=$MULT timeout 30 ./ds4 --model "$MODEL" \
      --prompt "$PROMPT" --tokens 24 2>&1 \
      | grep -vE "^ds4: (Metal|context|drift|tokens|metal|loading|MTP|graph|prefill|backend)" \
      | tail -4 | tee -a $LOG
  done
done

echo "LOG: $LOG"
