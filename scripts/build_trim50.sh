#!/usr/bin/env bash
# build_trim50.sh — generate the DS4 trim50 GGUF if missing.
#
# Inputs:
#   $1 (optional)  output path. Default: ./gguf/DS4-trim50-asym-with-metadata.gguf
#   DS4_BASE_GGUF  (env)  full IQ2_XXS source. Default scans common locations.
#   DS4_MASK       (env)  expert-drop mask CSV. Default: masks/mask_asym_50gb_v2.csv
#
# Output:
#   ~47.7 GiB GGUF with ~50% of routed experts dropped per the asymmetric mask.
#   Fits Metal residency under iogpu.wired_limit_mb >= 60 GiB (no --cpu-moe).
#
# Run:
#   ./scripts/build_trim50.sh
#   DS4_BASE_GGUF=/path/to/full.gguf ./scripts/build_trim50.sh /custom/out.gguf
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

OUT="${1:-${REPO_ROOT}/gguf/DS4-trim50-asym-with-metadata.gguf}"
MASK="${DS4_MASK:-${REPO_ROOT}/masks/mask_asym_50gb_v2.csv}"

# Locate base GGUF
if [ -n "${DS4_BASE_GGUF:-}" ]; then
    BASE="$DS4_BASE_GGUF"
else
    for cand in \
        "/Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf" \
        "${REPO_ROOT}/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf" \
        "${HOME}/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
    do
        if [ -f "$cand" ]; then BASE="$cand"; break; fi
    done
fi

if [ -z "${BASE:-}" ] || [ ! -f "$BASE" ]; then
    echo "build_trim50: full IQ2_XXS base GGUF not found." >&2
    echo "  Set DS4_BASE_GGUF to its path, or place it at one of the default locations." >&2
    echo "  Source: https://huggingface.co/DevQuasar/deepseek-ai.DeepSeek-V3.1-Flash-GGUF (IQ2_XXS recipe ~86.7 GiB)" >&2
    exit 1
fi

if [ ! -f "$MASK" ]; then
    echo "build_trim50: mask not found at $MASK" >&2
    exit 1
fi

if [ -f "$OUT" ]; then
    echo "build_trim50: $OUT already exists ($(du -h "$OUT" | cut -f1)); nothing to do."
    exit 0
fi

mkdir -p "$(dirname "$OUT")"

PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c "import sys; assert sys.version_info >= (3,9)" 2>/dev/null; then
    echo "build_trim50: need python >= 3.9; set PYTHON env to override." >&2
    exit 1
fi

echo "build_trim50: base=$BASE"
echo "build_trim50: mask=$MASK"
echo "build_trim50: out=$OUT"
echo "build_trim50: starting trim (~few minutes; output ~47.7 GiB)..."

"$PYTHON" "${REPO_ROOT}/analyzers/trim_experts_gguf.py" \
    --base "$BASE" \
    --mask "$MASK" \
    --out  "$OUT"

if [ -f "$OUT" ]; then
    echo "build_trim50: done. Size: $(du -h "$OUT" | cut -f1)"
else
    echo "build_trim50: FAILED — output not created" >&2
    exit 1
fi
