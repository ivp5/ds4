#!/bin/bash
# OOM-3 diff-test canary — silv 2026-05-28
#
# Productized wrapper around the diff-test infrastructure that already
# lives in ds4 source (DS4_METAL_GRAPH_TRACE_LAYERS + DS4_METAL_GRAPH_
# TEACHER_FORCE + DS4_METAL_GRAPH_TRACE_STAGE_LAYER env vars +
# --metal-graph-prompt-test / --metal-graph-full-test flags).
#
# OOM-1 Phase Y proved this is the cleanest CPU-vs-Metal divergence
# locator. Wrapping as one-line invocation so every future kernel ship
# A/Bs at per-layer resolution in <60 seconds.
#
# Usage:
#   tools/oom3_diff_canary.sh [MODE] [LAYER] [PROMPT]
#
# MODE: prompt | full | teacher | stage   (default: full)
# LAYER: layer index for stage trace      (default: 42)
# PROMPT: text                            (default: "Hello")
#
# Examples:
#   tools/oom3_diff_canary.sh                       # full decode-graph diff
#   tools/oom3_diff_canary.sh prompt                # prefill-graph diff
#   tools/oom3_diff_canary.sh teacher               # per-layer LOCAL divergence
#   tools/oom3_diff_canary.sh stage 42 "Hello"      # L42 internal stages
#   tools/oom3_diff_canary.sh stage 21 "AIME"       # L21 stages on different prompt

set -e
MODE="${1:-full}"
LAYER="${2:-42}"
PROMPT="${3:-Hello}"

META="/Users/silv/cl/tlp/montyneg/ds4/nonrouted/ds4v4_minimal.gguf"
NRPK="/Users/silv/cl/tlp/montyneg/ds4/nonrouted/ds4v4_nonrouted.pack"
VQB2="/Users/silv/cl/tlp/montyneg/ds4/vqb2/DeepSeek-V4-Flash/nonrotated_layer22_k256_gateup_top4_20260528/pack"
LOG_DIR="tmp/$(date +%Y%m%d)_oom3"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/diff_${MODE}_L${LAYER}_$(date +%H%M%S).log"

case "$MODE" in
 prompt)
  echo "[oom3] prefill-graph diff (--metal-graph-prompt-test) on prompt='$PROMPT'" | tee "$LOG"
  ./ds4 -m "$META" --nonrouted-pack "$NRPK" --vqb2-pack "$VQB2" \
   --metal-graph-prompt-test -p "$PROMPT" -n 1 2>&1 | tee -a "$LOG" \
   | grep -E 'graph logits|first-token|router selected'
  ;;
 full)
  echo "[oom3] decode-graph diff (--metal-graph-full-test) on prompt='$PROMPT'" | tee "$LOG"
  DS4_METAL_GRAPH_TRACE_LAYERS=1 \
  ./ds4 -m "$META" --nonrouted-pack "$NRPK" --vqb2-pack "$VQB2" \
   --metal-graph-full-test -p "$PROMPT" -n 1 2>&1 | tee -a "$LOG" \
   | grep -E 'full graph layer|first-token graph diffs'
  ;;
 teacher)
  echo "[oom3] teacher-forced LOCAL per-layer divergence on prompt='$PROMPT'" | tee "$LOG"
  DS4_METAL_GRAPH_TRACE_LAYERS=1 DS4_METAL_GRAPH_TEACHER_FORCE=1 \
  ./ds4 -m "$META" --nonrouted-pack "$NRPK" --vqb2-pack "$VQB2" \
   --metal-graph-full-test -p "$PROMPT" -n 1 2>&1 | tee -a "$LOG" \
   | grep -E 'teacher hc_max|first-token graph diffs'
  ;;
 stage)
  echo "[oom3] stage-trace of L$LAYER on prompt='$PROMPT'" | tee "$LOG"
  DS4_METAL_GRAPH_TRACE_LAYERS=1 DS4_METAL_GRAPH_TRACE_STAGE_LAYER="$LAYER" \
  ./ds4 -m "$META" --nonrouted-pack "$NRPK" --vqb2-pack "$VQB2" \
   --metal-graph-full-test -p "$PROMPT" -n 1 2>&1 | tee -a "$LOG" \
   | grep -E "stage layer $LAYER|shared layer $LAYER|routed layer $LAYER|router selected mismatch|first-token graph"
  ;;
 *)
  echo "unknown MODE='$MODE'; use: prompt | full | teacher | stage"
  exit 2
  ;;
esac

echo "[oom3] log saved: $LOG"
