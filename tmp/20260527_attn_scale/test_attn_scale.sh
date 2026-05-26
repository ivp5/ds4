#!/usr/bin/env bash
# DS4 attention scale × quality test on trim50.
# Probes a math prompt known to loop on Qwen3.5-4B (proxy for Conjecture #23).
set +u
cd /Users/silv/cl/tlp/montyneg/ivp5_ds4

MODEL=/Users/silv/cl/tlp/montyneg/ds4/gguf/DS4-trim50-asym-with-metadata.gguf
TS=$(date +%Y%m%dT%H%M%S)
LOG=tmp/20260527_attn_scale/test_${TS}.log

PROMPT="The capital of France is "

echo "=== DS4 attention scale × quality on trim50 ===" | tee $LOG

for MULT in 1.0 0.5 0.2 0.1; do
  echo "" | tee -a $LOG
  echo "--- DS4_ATTN_SCALE_MULT=$MULT ---" | tee -a $LOG
  DS4_ATTN_SCALE_MULT=$MULT timeout 60 ./ds4 --model "$MODEL" \
    --prompt "$PROMPT" --tokens 64 2>&1 \
    | grep -vE "^ds4: (Metal|context|drift|tokens|metal|loading)" \
    | tail -10 | tee -a $LOG
done

echo "LOG: $LOG"
