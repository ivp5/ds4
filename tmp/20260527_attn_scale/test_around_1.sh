#!/usr/bin/env bash
set +u
cd /Users/silv/cl/tlp/montyneg/ivp5_ds4

MODEL=/Users/silv/cl/tlp/montyneg/ds4/gguf/DS4-trim50-asym-with-metadata.gguf
TS=$(date +%Y%m%dT%H%M%S)
LOG=tmp/20260527_attn_scale/around_1_${TS}.log
PROMPT="The capital of France is "

echo "=== fine-grain DS4 attn scale around 1.0 ===" | tee $LOG
echo "Prompt: $PROMPT" | tee -a $LOG

for MULT in 0.97 0.98 0.985 0.99 0.995 1.0 1.005 1.01 1.015 1.02 1.03; do
  echo "" | tee -a $LOG
  echo "--- DS4_ATTN_SCALE_MULT=$MULT ---" | tee -a $LOG
  DS4_ATTN_SCALE_MULT=$MULT timeout 30 ./ds4 --model "$MODEL" \
    --prompt "$PROMPT" --tokens 40 2>&1 \
    | grep -vE "^ds4: (Metal|context|drift|tokens|metal|loading|MTP|graph)" \
    | tail -6 | tee -a $LOG
done

echo "LOG: $LOG"
