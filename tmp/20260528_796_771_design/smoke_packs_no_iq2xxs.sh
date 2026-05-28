#!/usr/bin/env bash
# Smoke test using packs as silv directed (2026-05-28 evening):
#   "use this instead of the gguf /Users/silv/cl/tlp/montyneg/ds4/nonrouted/ds4v4_nonrouted.pack"
#
# IQ2_XXS GGUF (86.7 GB) remains OFF-LIMITS per silv directive.
# This uses the 9 GB minimal GGUF (structural shell) + nonrouted-pack
# (8.4 GB source-exact tensor data) + VQB2 pack (39 GB routed FFN).
#
# Goals:
#   (a) Validate this session's engineer-roster work (Cycles 1, 2, 3, 5, 6', 9, 9a-9g)
#       on a real model boot — not just the amortized canary.
#   (b) Confirm --nonrouted-pack identity-fill path works end-to-end.
#   (c) Surface any remaining bugs before #796 Increment 1.

set -uo pipefail

PEER=$(pgrep -fl "ds4 -m" 2>/dev/null | grep -v "$$" || true)
if [ -n "$PEER" ]; then
    echo "ABORT: peer ds4 detected: $PEER"
    exit 3
fi

TIMESTAMP=$(date +%Y%m%dT%H%M%S)
DIR=/Users/silv/cl/tlp/montyneg/ivp5_ds4/tmp/20260528_796_771_design
LOG="$DIR/smoke_packs_${TIMESTAMP}.log"
DS4=/Users/silv/cl/tlp/montyneg/ivp5_ds4/ds4
MINIMAL_GGUF=/Users/silv/cl/tlp/montyneg/ds4/nonrouted/ds4v4_minimal.gguf
NONROUTED_PACK=/Users/silv/cl/tlp/montyneg/ds4/nonrouted/ds4v4_nonrouted.pack
VQB2_PACK=/Users/silv/cl/tlp/montyneg/ds4/vqb2/DeepSeek-V4-Flash/nonrotated_layer22_k256_gateup_top4_20260528/pack/ds4_flash_nonrotated_layer22_k256_gateup_top4.vqb2pack

PROMPT_FILE=/tmp/smoke_packs_prompt.txt
echo "The quick brown fox jumps over the" > "$PROMPT_FILE"

echo "[$(date)] Smoke: minimal-GGUF + nonrouted-pack + VQB2 pack" | tee -a "$LOG"
echo "[$(date)] DS4_VQB2_FP16_PATH=fused (Cycle 6' default; explicit for clarity)" | tee -a "$LOG"
echo "[$(date)] --tokens 5 (smoke only)" | tee -a "$LOG"
echo "[$(date)] Models:" | tee -a "$LOG"
ls -lh "$MINIMAL_GGUF" "$NONROUTED_PACK" "$VQB2_PACK" 2>&1 | tee -a "$LOG"

DS4_VQB2_FP16_PATH=fused \
"$DS4" \
    -m "$MINIMAL_GGUF" \
    --nonrouted-pack "$NONROUTED_PACK" \
    --vqb2-pack "$VQB2_PACK" \
    --prompt-file "$PROMPT_FILE" \
    --tokens 5 \
    --metal \
    --prefill-metal-phases auto \
    > "${LOG}.ds4" 2>&1
RC=$?

echo "[$(date)] ds4 exit=$RC" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "=== Last 40 lines of ds4 log ===" | tee -a "$LOG"
tail -40 "${LOG}.ds4" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "=== SEGV check ===" | tee -a "$LOG"
if grep -i "segmentation\|sigsegv\|abort\|panic" "${LOG}.ds4" >/dev/null 2>&1; then
    echo "FAIL: SEGV/abort/panic detected" | tee -a "$LOG"
    grep -i "segmentation\|sigsegv\|abort\|panic" "${LOG}.ds4" | head -5 | tee -a "$LOG"
else
    echo "PASS: no SEGV/abort/panic" | tee -a "$LOG"
fi
echo "" | tee -a "$LOG"
echo "=== Engineer-roster sanity ===" | tee -a "$LOG"
grep -E "DS4_HOT_METAL_MOE|DS4_VQB2_FP16_PATH|nonrouted-pack|vqb2-pack|override-fill|ICB|Cycle [0-9]+'?[a-z]?" "${LOG}.ds4" 2>&1 | head -30 | tee -a "$LOG"

exit "$RC"
