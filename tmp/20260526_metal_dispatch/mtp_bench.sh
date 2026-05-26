#!/usr/bin/env bash
# Time end-to-end ds4 chat-CLI gen with and without MTP.
# Goal: validate silv's stated 20-30 t/s target via MTP speculative decode.
set +u
cd /Users/silv/cl/tlp/montyneg/ivp5_ds4

MODEL=/Users/silv/cl/tlp/montyneg/ds4/gguf/DS4-trim50-asym-with-metadata.gguf
MTP=/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
TS=$(date +%Y%m%dT%H%M%S)
LOG=tmp/20260526_metal_dispatch/mtp_bench_${TS}.log

PROMPT="Write a 200-word essay about the importance of clear communication in software engineering."

echo "=== MTP gen bench on trim50 ===" | tee $LOG
echo "" | tee -a $LOG
echo "--- no MTP (single-token gen) ---" | tee -a $LOG
time ./ds4 --model "$MODEL" --prompt "$PROMPT" --tokens 100 2>&1 | tee -a $LOG | tail -10
echo "" | tee -a $LOG
echo "--- with MTP --mtp-draft 1 (1 draft per step = baseline) ---" | tee -a $LOG
time ./ds4 --model "$MODEL" --mtp "$MTP" --mtp-draft 1 --prompt "$PROMPT" --tokens 100 2>&1 | tee -a $LOG | tail -10
echo "" | tee -a $LOG
echo "--- with MTP --mtp-draft 4 (more aggressive drafting) ---" | tee -a $LOG
time ./ds4 --model "$MODEL" --mtp "$MTP" --mtp-draft 4 --prompt "$PROMPT" --tokens 100 2>&1 | tee -a $LOG | tail -10

echo "" | tee -a $LOG
echo "LOG: $LOG"
