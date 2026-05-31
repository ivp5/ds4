#!/usr/bin/env python3
"""H2247-template residency-cache cliff audit.

silv 2026-05-28: codex H2247 found L9 K4 layer-kind dispatch ran at
899-1476ms vs ~1ms for L0/L39/L42. The cliff was caused by single-slot
MTL4 residency cache thrashing on L9's split-codebook layout
(gate/up = [1,15], down = [1,31]). The fix absorbs up to 64 codebook
buffers in one residency group. Validated 800-1000× speedup.

This script generalizes that detection: across all 43 layers × 3 kinds,
surface which (layer, kind) pairs have layout patterns likely to thrash
the residency cache.

Detection heuristics:
  1. K-tier heterogeneity within (layer, kind): if a layer's row_blocks
     use multiple K values, dispatch must rotate codebooks → thrash risk
  2. Codebook fragmentation: distinct codebook signatures per (layer, kind)
     — measured via `header_k` count AND distinct row_block ranges
  3. Pack offset locality: large pack-offset jumps between row_blocks
     within the same dispatch correlate with cache miss patterns

Output: ranked list of (layer, kind) pairs by suspicion score. Top-K
candidates are the next H2247-style residency fixes to investigate.

Run:
  python3 tools/h2247_residency_audit.py [--index-csv PATH] [--top N]
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_INDEX = (
    "/Users/silv/cl/tlp/montyneg/ds4/vqb2/DeepSeek-V4-Flash/"
    "nonrotated_layer22_k256_gateup_top4_20260528/pack/"
    "ds4_flash_nonrotated_layer22_k256_gateup_top4.vqb2pack.index.csv"
)


def parse_index(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "layer":         int(r["layer"]),
                "kind":          r["kind"],
                "kind_id":       int(r["kind_id"]),
                "k":             int(r["k"]),
                "row_start":     int(r["row_start"]),
                "row_count":     int(r["row_count"]),
                "pack_offset":   int(r["pack_offset"]),
                "packet_bytes":  int(r["packet_bytes"]),
                "header_k":      int(r["header_k"]),
                "header_n_codes": int(r["header_n_codes"]),
                "expert_ids":    r["expert_ids_sha256"],
            })
    return rows


def audit(rows):
    """Return dict (layer, kind) -> dict of suspicion features."""
    by_lk = defaultdict(list)
    for r in rows:
        by_lk[(r["layer"], r["kind"])].append(r)

    findings = []
    for (layer, kind), entries in by_lk.items():
        # Sort by row_start to walk the dispatch order
        entries.sort(key=lambda e: e["row_start"])

        # Feature 1: K-tier heterogeneity within (layer, kind)
        k_values = sorted({e["k"] for e in entries})
        k_diversity = len(k_values)

        # Feature 2: distinct codebook signatures = distinct (header_k, expert_ids)
        # Two entries with same K but different expert_ids hash = different codebook
        codebook_signatures = sorted({(e["header_k"], e["expert_ids"]) for e in entries})
        codebook_diversity = len(codebook_signatures)

        # Feature 3: pack-offset locality — sum of |delta| between consecutive offsets
        # Normalized by total packet bytes (large = scattered)
        offset_jumps = 0
        for i in range(1, len(entries)):
            offset_jumps += abs(entries[i]["pack_offset"] - entries[i-1]["pack_offset"])
        total_bytes = sum(e["packet_bytes"] for e in entries)
        offset_locality = offset_jumps / max(total_bytes, 1)

        # Feature 4: row_block count + whether codebook RUNS exist (consecutive
        # blocks sharing codebook) vs interleaved. If two distinct codebooks
        # appear in alternating fashion, every dispatch triggers a swap.
        cb_run_count = 1
        cb_alternation_score = 0  # higher = more alternation = worse for cache
        prev_sig = None
        last_run_sig = None
        for e in entries:
            sig = (e["header_k"], e["expert_ids"])
            if prev_sig is not None and sig != prev_sig:
                cb_run_count += 1
                # If we're alternating back to a previous signature, that's
                # the thrash pattern H2247 caught at L9.
                if last_run_sig is not None and sig == last_run_sig:
                    cb_alternation_score += 1
                last_run_sig = prev_sig
            prev_sig = sig

        # Suspicion score: weighted sum
        # k_diversity ≥ 2 + cb_diversity ≥ 2 + alternation = classic H2247 shape
        score = (
            (k_diversity - 1) * 10 +
            (codebook_diversity - 1) * 5 +
            cb_alternation_score * 20 +
            (cb_run_count - 1) * 2 +
            offset_locality * 0.5
        )

        findings.append({
            "layer": layer,
            "kind": kind,
            "n_blocks": len(entries),
            "k_values": k_values,
            "k_diversity": k_diversity,
            "cb_diversity": codebook_diversity,
            "cb_run_count": cb_run_count,
            "cb_alternation": cb_alternation_score,
            "offset_locality": offset_locality,
            "total_bytes_mb": total_bytes / 1e6,
            "score": score,
        })

    return findings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", default=DEFAULT_INDEX)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--show-all", action="store_true")
    args = ap.parse_args()

    rows = parse_index(args.index_csv)
    print(f"index: {args.index_csv}", file=sys.stderr)
    print(f"rows:  {len(rows)}", file=sys.stderr)

    findings = audit(rows)
    findings.sort(key=lambda f: -f["score"])

    # Quick sanity: H2247 said L9 is the canonical bad case
    l9 = [f for f in findings if f["layer"] == 9]
    if l9:
        print("\n== L9 sanity (codex H2247 documented case) ==", file=sys.stderr)
        for f in l9:
            print(f"  L{f['layer']:<2} {f['kind']:<4}: K-diversity={f['k_diversity']} "
                  f"cb-diversity={f['cb_diversity']} cb-runs={f['cb_run_count']} "
                  f"alternation={f['cb_alternation']} score={f['score']:.2f}",
                  file=sys.stderr)

    print(f"\n== Top {args.top} suspicion scores ==", file=sys.stderr)
    print(f"{'rank':<5} {'L':>3} {'kind':<5} {'n_bk':>5} {'K_div':>6} {'cb_div':>7} "
          f"{'cb_runs':>8} {'alt':>4} {'loc':>7} {'MB':>6} {'score':>8}",
          file=sys.stderr)
    for i, f in enumerate(findings[:args.top]):
        ks = "/".join(str(k) for k in f["k_values"])
        print(f"{i+1:<5} {f['layer']:>3} {f['kind']:<5} {f['n_blocks']:>5} "
              f"{ks:>6} {f['cb_diversity']:>7} {f['cb_run_count']:>8} "
              f"{f['cb_alternation']:>4} {f['offset_locality']:>7.2f} "
              f"{f['total_bytes_mb']:>6.1f} {f['score']:>8.2f}",
              file=sys.stderr)

    # Score distribution percentiles
    scores = sorted(f["score"] for f in findings)
    p50 = scores[len(scores) // 2]
    p90 = scores[int(len(scores) * 0.9)]
    p99 = scores[int(len(scores) * 0.99)]
    max_s = scores[-1]
    print(f"\nscore distribution: p50={p50:.2f} p90={p90:.2f} p99={p99:.2f} max={max_s:.2f}",
          file=sys.stderr)
    print(f"outliers (score > 5× p50): "
          f"{sum(1 for s in scores if s > 5 * p50 and p50 > 0)}", file=sys.stderr)

    # ----- Cross-kind K-mix WITHIN a layer ------------------------------
    # H2247 caught intra-(layer,kind) codebook thrash. The active pack
    # eliminates that, but the cross-kind transition (gate→up→down) within
    # a single layer dispatch can ALSO thrash the residency cache if K
    # changes between kinds.
    by_layer = defaultdict(list)  # layer -> list of (kind, K, cb_hash)
    for r in rows:
        by_layer[r["layer"]].append((r["kind"], r["k"], r["expert_ids"]))

    print(f"\n== Cross-kind K-mix audit (within-layer transitions) ==", file=sys.stderr)
    print(f"Production dispatch order: gate → up → down. If K differs across kinds,",
          file=sys.stderr)
    print(f"the codebook cache must swap between kind dispatches (potential thrash).",
          file=sys.stderr)
    print(f"\n{'L':>3} {'gate_K':>7} {'up_K':>5} {'down_K':>7} {'K-mix':>6} "
          f"{'cb-distinct':>12}", file=sys.stderr)
    cross_kind_problem_layers = []
    for layer in sorted(by_layer.keys()):
        entries = by_layer[layer]
        ks_by_kind = defaultdict(set)
        cbs_by_kind = defaultdict(set)
        for kind, k, cb in entries:
            ks_by_kind[kind].add(k)
            cbs_by_kind[kind].add(cb)
        g_ks = sorted(ks_by_kind["gate"])
        u_ks = sorted(ks_by_kind["up"])
        d_ks = sorted(ks_by_kind["down"])
        all_ks = sorted(set(g_ks + u_ks + d_ks))
        k_mix = len(all_ks)
        all_cbs = set()
        for cbs in cbs_by_kind.values():
            all_cbs.update(cbs)
        n_cb_distinct = len(all_cbs)
        if k_mix > 1 or n_cb_distinct > 3:  # 3 = one per kind (uniform)
            g_str = "/".join(str(k) for k in g_ks) if g_ks else "-"
            u_str = "/".join(str(k) for k in u_ks) if u_ks else "-"
            d_str = "/".join(str(k) for k in d_ks) if d_ks else "-"
            print(f"{layer:>3} {g_str:>7} {u_str:>5} {d_str:>7} {k_mix:>6} "
                  f"{n_cb_distinct:>12}", file=sys.stderr)
            cross_kind_problem_layers.append({
                "layer": layer,
                "gate_K": g_ks, "up_K": u_ks, "down_K": d_ks,
                "k_mix": k_mix, "cb_distinct": n_cb_distinct,
            })

    if not cross_kind_problem_layers:
        print(f"  (no layers have cross-kind K-mix > 1 or cb-distinct > 3)",
              file=sys.stderr)
    else:
        print(f"\n  {len(cross_kind_problem_layers)} layers flagged for runtime "
              f"benchmarking.", file=sys.stderr)
        print(f"  Next step: instrument per-(layer, kind) Metal dispatch timing,",
              file=sys.stderr)
        print(f"  compare these flagged layers' times against uniform-K-mix peers.",
              file=sys.stderr)

    # ----- Pack-offset locality: layer-level mean dispatch distance -----
    # In production the layer dispatch reads gate/up/down packets sequentially.
    # If those packets are physically distant in the pack file, the L2/L3
    # cache + page cache get worked harder.
    layer_dispatch_jumps = []
    for layer in sorted(by_layer.keys()):
        offsets_by_kind = defaultdict(list)
        for r in rows:
            if r["layer"] == layer:
                offsets_by_kind[r["kind"]].append(r["pack_offset"])
        for kind in offsets_by_kind:
            offsets_by_kind[kind].sort()
        # Mean jump from gate-end → up-start → down-start (the production order)
        if all(k in offsets_by_kind for k in ("gate", "up", "down")):
            gate_end = max(offsets_by_kind["gate"])
            up_start = min(offsets_by_kind["up"])
            up_end = max(offsets_by_kind["up"])
            down_start = min(offsets_by_kind["down"])
            jump_gate_to_up = abs(up_start - gate_end)
            jump_up_to_down = abs(down_start - up_end)
            total_jump = jump_gate_to_up + jump_up_to_down
            layer_dispatch_jumps.append({
                "layer": layer,
                "gate_to_up_GB": jump_gate_to_up / 1e9,
                "up_to_down_GB": jump_up_to_down / 1e9,
                "total_GB": total_jump / 1e9,
            })

    layer_dispatch_jumps.sort(key=lambda x: -x["total_GB"])
    print(f"\n== Layer dispatch pack-offset locality (top 10 by jump) ==",
          file=sys.stderr)
    print(f"{'L':>3} {'gate→up GB':>12} {'up→down GB':>12} {'total GB':>10}",
          file=sys.stderr)
    for x in layer_dispatch_jumps[:10]:
        print(f"{x['layer']:>3} {x['gate_to_up_GB']:>12.3f} "
              f"{x['up_to_down_GB']:>12.3f} {x['total_GB']:>10.3f}", file=sys.stderr)

    # Spread analysis
    jumps_total = [x["total_GB"] for x in layer_dispatch_jumps]
    if jumps_total:
        print(f"\n  jump stats: min={min(jumps_total):.3f} max={max(jumps_total):.3f} "
              f"median={sorted(jumps_total)[len(jumps_total)//2]:.3f} GB", file=sys.stderr)
        # Outliers: any layer with > 3× median jump
        median = sorted(jumps_total)[len(jumps_total)//2]
        outliers = [x for x in layer_dispatch_jumps if x["total_GB"] > 3 * median]
        if outliers:
            print(f"  outliers (>3× median): {len(outliers)} layers",
                  file=sys.stderr)
            for x in outliers:
                print(f"    L{x['layer']:>2}: {x['total_GB']:.3f} GB total jump",
                      file=sys.stderr)


if __name__ == "__main__":
    main()
