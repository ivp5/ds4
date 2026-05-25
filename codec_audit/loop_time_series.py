"""Loop-emergence time-series probe.

silv 2026-05-25: "analyze the whole vertical dynamics through time to the
nearest coherent state".

Probes per-token compute distribution at multiple offsets in a long
generation. The shift in class distribution through time IS the loop-
emergence signature.

Usage:
  python3 codec_audit/loop_time_series.py \
    --json /path/to/p01_cached.json \
    --offsets 0,5000,10000,20000,30000,40000,50000,60000 \
    --n-chars 1000
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

# Run probe_per_token.py as subprocess for each offset (clean env per run)
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True)
    p.add_argument("--offsets", required=True, help="comma-separated offsets")
    p.add_argument("--n-chars", type=int, default=1000)
    args = p.parse_args()

    offsets = [int(o) for o in args.offsets.split(",")]
    base = Path(args.json).stem

    for offset in offsets:
        slice_path = f"/tmp/loop_ts_{base}_o{offset}.json"
        # Build slice file
        subprocess.run([
            "python3", "codec_audit/_probe_loop_region.py",
            "--json", args.json,
            "--offset", str(offset),
            "--n-chars", str(args.n_chars),
            "--out", slice_path,
        ], check=True)
        # Probe
        notes = f"loop_ts {base} offset={offset} n_chars={args.n_chars}"
        subprocess.run([
            "python3", "-u", "codec_audit/probe_per_token.py",
            "--text-from", slice_path,
            "--n-chars", str(args.n_chars),
            "--notes", notes,
        ], check=True)

    # Query the DB for the just-inserted prompts + show time series
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from codec_audit import open_db
    from codec_audit.db import query

    with open_db() as conn:
        # Find the most recent prompt_run entries matching this base
        rows = query(conn, """
            SELECT pr.prompt_id, pr.notes,
                   SUM(CASE WHEN tc.compute_class='trivial' THEN 1 ELSE 0 END) AS n_trivial,
                   SUM(CASE WHEN tc.compute_class='easy' THEN 1 ELSE 0 END)    AS n_easy,
                   SUM(CASE WHEN tc.compute_class='medium' THEN 1 ELSE 0 END)  AS n_medium,
                   SUM(CASE WHEN tc.compute_class='hard' THEN 1 ELSE 0 END)    AS n_hard,
                   SUM(CASE WHEN tc.compute_class='undecided' THEN 1 ELSE 0 END) AS n_undec,
                   COUNT(*) AS n_total,
                   AVG(tc.min_stabilize_layer) AS mean_L,
                   AVG(tc.final_entropy_nats)  AS mean_entropy
            FROM token_compute tc
            JOIN prompt_run pr USING(prompt_id)
            WHERE pr.notes LIKE ?
            GROUP BY pr.prompt_id
            ORDER BY pr.prompt_id
        """, (f"loop_ts {base}%",))

    print(f"\n{'='*80}\n=== TIME SERIES: {base} ===\n{'='*80}\n")
    print(f"{'prompt_id':>9} {'notes':<48} {'trivial%':>8} {'hard%':>6} "
          f"{'undec%':>7} {'mean_L':>7} {'mean_ent':>8}")
    print("-" * 100)
    for r in rows:
        t = 100 * r['n_trivial'] / r['n_total']
        h = 100 * r['n_hard'] / r['n_total']
        u = 100 * r['n_undec'] / r['n_total']
        flag = " ← LOOP" if (t > 14 and h < 10 and u < 5) else ""
        print(f"{r['prompt_id']:>9} {r['notes'][:48]:<48} {t:>7.1f}% {h:>5.1f}% "
              f"{u:>6.1f}% {r['mean_L']:>7.2f} {r['mean_entropy']:>8.4f}{flag}")


if __name__ == "__main__":
    main()
