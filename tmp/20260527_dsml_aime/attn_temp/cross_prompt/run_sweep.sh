#!/bin/bash
set -u
cd /Users/silv/cl/tlp/montyneg/ivp5_ds4
MODEL=/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf

while IFS=$'\t' read -r pname body; do
    [ -z "$pname" ] && continue
    pdir=tmp/20260527_dsml_aime/attn_temp/cross_prompt/p_${pname}
    mkdir -p $pdir
    printf '%s' "$body" > $pdir/prompt.txt
    for MULT in 0.5 0.8 1.0 1.5 2.5; do
        OUT=$pdir/mult_${MULT}.csv
        [ -s $OUT ] && continue
        for try in 1 2 3; do
            DS4_ATTN_SCALE_MULT=$MULT timeout 30 ./ds4-logitlens -m "$MODEL" \
                --prompt-file $pdir/prompt.txt -k 4 --cpu-moe > $OUT 2>/dev/null
            [ -s $OUT ] && break
            sleep 3
        done
    done
done < tmp/20260527_dsml_aime/attn_temp/cross_prompt/prompts.tsv

echo "done." >> tmp/20260527_dsml_aime/attn_temp/cross_prompt/sweep_20260527T191143.log
