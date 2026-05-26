#!/usr/bin/env bash
# Finer-grained DS4 attention scale sweep — find the boundary between
# baseline-working (1.0) and breaking-at-0.5.
set +u
cd /Users/silv/cl/tlp/montyneg/ivp5_ds4

MODEL=/Users/silv/cl/tlp/montyneg/ds4/gguf/DS4-trim50-asym-with-metadata.gguf
TS=$(date +%Y%m%dT%H%M%S)
LOG=tmp/20260527_attn_scale/finer_${TS}.log

PROMPT="The capital of France is "

echo "=== DS4 attention scale finer-grained sweep ===" | tee $LOG
echo "Prompt: $PROMPT" | tee -a $LOG

# Sharpening (sub-1.0) — find break point
# Smoothing (super-1.0) — does it ever help?
for MULT in 0.5 0.6 0.7 0.75 0.8 0.85 0.9 0.95 1.0 1.05 1.1 1.2 1.5 2.0; do
  echo "" | tee -a $LOG
  echo "--- DS4_ATTN_SCALE_MULT=$MULT ---" | tee -a $LOG
  DS4_ATTN_SCALE_MULT=$MULT timeout 45 ./ds4 --model "$MODEL" \
    --prompt "$PROMPT" --tokens 48 2>&1 \
    | grep -vE "^ds4: (Metal|context|drift|tokens|metal|loading|MTP|graph)" \
    | tail -8 | tee -a $LOG
done

echo "LOG: $LOG"
