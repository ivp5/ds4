#!/usr/bin/env bash
# DS4 layer-DUPLICATION test — same-parity pairs (ratio compatibility).
# DS4 layer ratios: L0/L1=dense, even>=2 ratio=4 (with indexer),
#                                odd >=3 ratio=128 (no indexer).
# Valid dups: even<->even, odd<->odd, L0<->L1.
#
# Tests silv's hypothesis: late-layer weight-cosine 0.92-0.96 in {33,34,35,36}
# means those layers are amplifying — dup should preserve. cosine ~0.26 at the
# commit-pair 36/37/38 means distinct work — dup should break.
set +u
cd /Users/silv/cl/tlp/montyneg/ivp5_ds4

MODEL=/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
TS=$(date +%Y%m%dT%H%M%S)
LOG=tmp/20260527_layer_dedup/dup_${TS}.log
PROMPT="The capital of France is "

run_dup() {
    local label="$1"
    local dup="$2"
    echo "" | tee -a $LOG
    echo "=== $label  DUP=$dup ===" | tee -a $LOG
    if [ -n "$dup" ]; then
        DS4_LAYER_DUP="$dup" timeout 180 ./ds4 --model "$MODEL" \
            --prefill-metal-phases auto \
            --prompt "$PROMPT" --tokens 24 2>&1 | tail -30 | tee -a $LOG
    else
        timeout 180 ./ds4 --model "$MODEL" \
            --prefill-metal-phases auto \
            --prompt "$PROMPT" --tokens 24 2>&1 | tail -30 | tee -a $LOG
    fi
}

echo "=== FULL DS4 layer-DUP (same-parity) sweep ===" | tee $LOG

run_dup "BASELINE"                                  ""
# AMPLIFICATION region (high weight cosine ~0.92-0.96):
run_dup "amp-even: L34<-L36"                        "34=36"
run_dup "amp-even: L34<-L36, L36<-L38"              "34=36,36=38"
run_dup "amp-odd:  L33<-L35"                        "33=35"
run_dup "amp-odd:  L33<-L35, L35<-L37"              "33=35,35=37"
# COMMIT region (silv's prior cos=0.26 anti-amplification):
run_dup "commit-even: L36<-L38 (post-commit)"       "36=38"
run_dup "commit-odd:  L37<-L39"                     "37=39"
# CROSS-REGION (early vs late):
run_dup "cross-even: L4<-L34 (early<-late)"         "4=34"
run_dup "cross-even: L34<-L4 (late<-early)"         "34=4"
run_dup "cross-odd:  L5<-L33"                       "5=33"
# WIDE AMP COMBO (test if 4-layer dup preserves):
run_dup "wide-amp-even: 34<-36, 38<-40, 42<-40"     "34=36,38=40"

echo "" | tee -a $LOG
echo "LOG: $LOG"
