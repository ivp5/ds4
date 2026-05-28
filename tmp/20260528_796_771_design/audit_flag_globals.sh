#!/usr/bin/env bash
# Audit for Cycle-5-shape latent bugs: flag-shaped static globals with
# zero writes (declared but never assigned). Reads work, but the read
# always returns the zero-init value → dormant code path.
#
# silv 2026-05-28: extend the audit silv requested after Cycle 5 found
# override_source_active never written (made source-exact path inert).

set -uo pipefail
cd "$(dirname "$0")/../.."

FILES="ds4.c ds4_metal.m ds4_metal_vqb2_fp16.m ds4_inflight.c ds4_expert_table.c
       ds4_moe_route_log.c ds4_polar_reader.c ds4_vqb1_reader.c
       ds4_vqb2_reader.c ds4_vqb2_pack.c ds4_watersic_pack.c
       ds4_nonrouted_pack.c ds4_prefix_cache.c"

declare -i total=0 sus=0
for f in $FILES; do
    [ -f "$f" ] || continue
    # Find flag-shaped static globals
    while IFS= read -r line; do
        decl_line=$(echo "$line" | cut -d: -f1)
        decl_text=$(echo "$line" | cut -d: -f2-)
        # Extract symbol — last word in identifier-shape before ; or =
        sym=$(echo "$decl_text" | sed -nE 's/.*\b([gs]_[a-zA-Z_][a-zA-Z_0-9]*)\s*[;=].*/\1/p')
        [ -z "$sym" ] && continue
        total=$((total + 1))
        # Count writes (NOT in declaration, NOT a comparison)
        # Pattern: symbol = NOT_=  (assignment, not ==)
        writes=$(grep -cE "\b${sym}\s*=[^=]" "$f" | head -1)
        # Subtract initializer (if `= 0;` or similar on the decl line itself)
        if echo "$decl_text" | grep -qE "= "; then
            writes=$((writes - 1))
        fi
        if [ "$writes" -le 0 ]; then
            sus=$((sus + 1))
            echo "SUSPICIOUS: $f:$decl_line  $sym  writes=$writes"
            echo "  decl: $decl_text"
            # Show reads to confirm it's actually used
            reads=$(grep -nE "\b${sym}\b" "$f" | grep -v ":${decl_line}:" | head -3)
            if [ -n "$reads" ]; then
                echo "  reads:"
                echo "$reads" | sed 's/^/    /'
            fi
            echo ""
        fi
    done < <(grep -nE "^static (int|uint8_t|bool|_Bool|uint32_t) (g_|s_)[a-z_]*(_active|_enabled|_ok|_ready|_inited|_initialized|_done|_loaded|_open|_set|_present|_valid|_attempted|_armed|_running|_dirty|_pending|_locked|_busy|_held)" "$f" || true)
done
echo "=== summary: $sus suspicious / $total flag-shaped globals audited ==="
