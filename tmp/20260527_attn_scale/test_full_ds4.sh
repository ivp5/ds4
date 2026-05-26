#!/usr/bin/env bash
# Test attention scale on FULL DS4 (86 GB IQ2_XXS) to disambiguate
# whether trim50 specifically is broken vs the temp mechanism itself.
# Use --prefill-metal-phases auto (per DS4 STICKY HAZARD safety rule).
set +u
cd /Users/silv/cl/tlp/montyneg/ivp5_ds4

# FULL untrimmed file
MODEL=/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
TS=$(date +%Y%m%dT%H%M%S)
LOG=tmp/20260527_attn_scale/full_${TS}.log

PROMPTS=(
  "The capital of France is "
  "What is 2+2? Answer with just the number: "
)

echo "=== FULL untrimmed DS4 attention scale sweep ===" | tee $LOG
echo "Model: $MODEL" | tee -a $LOG

for PROMPT in "${PROMPTS[@]}"; do
  echo "" | tee -a $LOG
  echo "###################" | tee -a $LOG
  echo "PROMPT: $PROMPT" | tee -a $LOG
  echo "###################" | tee -a $LOG
  for MULT in 1.0 0.95 0.9 0.8 0.5 0.2; do
    echo "" | tee -a $LOG
    echo "--- scale=$MULT ---" | tee -a $LOG
    DS4_ATTN_SCALE_MULT=$MULT timeout 90 ./ds4 --model "$MODEL" \
      --prefill-metal-phases auto \
      --prompt "$PROMPT" --tokens 24 2>&1 \
      | grep -vE "^ds4: (Metal|context|drift|tokens|metal|loading|MTP|graph|backend|restored|cpu)" \
      | tail -5 | tee -a $LOG
  done
done

echo "LOG: $LOG"
