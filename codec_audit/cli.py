"""codec_audit/cli.py — command-line surface.

Usage:
  python3 -m codec_audit init                          — create DB + schema
  python3 -m codec_audit register-defaults             — register the 3 codecs
  python3 -m codec_audit scan --layer 0 --kind gate     — full sweep
  python3 -m codec_audit query                          — best codec per cell
  python3 -m codec_audit journal --tail 50              — show recent actions
"""
import argparse
import json
import sys

from . import (
    open_db, init_db, register_codec, journal, scan_layer_kind,
    new_run, end_run, list_codecs, DB_PATH,
)
from .db import query


def cmd_init(args):
    init_db()
    print(f"initialized {DB_PATH}")


def cmd_register_defaults(args):
    with open_db() as conn:
        for spec in [
            ("iq2_xxs",      "iq2_xxs", 4.125,  0.5156, {"block_weights": 256, "block_bytes": 66}),
            ("polar_p32_m8", "polar",   16.0,   2.0,    {"phase_levels": 32, "mag_levels": 8}),
            ("vq_k256",      "vq",      8.0,    1.0,    {"K": 256}),
        ]:
            cid = register_codec(conn, *spec[:4], params=spec[4])
            print(f"codec_id={cid}  {spec[0]}")


def cmd_scan(args):
    with open_db() as conn:
        rid = new_run(conn, f"scan_L{args.layer}_{args.kind}",
                       params={"layer": args.layer, "kind": args.kind,
                               "experts": [args.expert_start, args.expert_end],
                               "n_subsample": args.n_subsample})
        summary = scan_layer_kind(
            conn, layer=args.layer, kind=args.kind,
            experts=range(args.expert_start, args.expert_end),
            n_subsample=args.n_subsample, run_id=rid,
        )
        end_run(conn, rid, summary)
        print(json.dumps(summary, indent=2))


def cmd_query(args):
    """Best codec per (layer, kind, expert) by mean rel_l2."""
    with open_db() as conn:
        rows = query(conn, """
            SELECT layer, kind, expert, codec_name,
                   AVG(rel_l2) AS mean_rl2,
                   AVG(sign_flip) AS mean_sf,
                   AVG(angle_rms_rad) AS mean_angle,
                   COUNT(*) AS n
            FROM v_measurement_full
            GROUP BY layer, kind, expert, codec_name
            ORDER BY layer, kind, expert, mean_rl2
        """)
        for r in rows[:args.limit]:
            print(f"L{r['layer']:>2} {r['kind']:<4} E{r['expert']:>3}  "
                  f"{r['codec_name']:<14}  rl2={r['mean_rl2']:.4f}  "
                  f"sf={r['mean_sf']:.4f}  ang={r['mean_angle']:.4f}  n={r['n']}")


def cmd_journal(args):
    with open_db() as conn:
        rows = query(conn,
            f"SELECT entry_id, ts, action, target FROM journal "
            f"ORDER BY entry_id DESC LIMIT ?", (args.tail,))
        for r in rows:
            print(f"#{r['entry_id']:>5} {r['ts']}  {r['action']:<25} {r['target']}")


def cmd_codecs(args):
    with open_db() as conn:
        for c in list_codecs(conn):
            print(f"#{c['codec_id']:>3} {c['name']:<18} {c['family']:<10} "
                  f"bits/pair={c['bits_per_pair']:>6.3f}  "
                  f"byte/pair={c['storage_byte_per_pair']:>6.4f}")


def cmd_report(args):
    """Unified DB report: codec quality + per-token compute + ngram locks."""
    with open_db() as conn:
        # 1. Codec registry
        codecs = list_codecs(conn)
        print(f"\n=== Registered codecs ({len(codecs)}) ===")
        for c in codecs:
            print(f"  #{c['codec_id']:>2} {c['name']:<18} {c['family']:<10} "
                  f"byte/pair={c['storage_byte_per_pair']:>6.4f}")

        # 2. Cross-codec quality summary
        rows = query(conn, """
            SELECT codec_name, codec_family, bits_per_pair,
                   COUNT(*) AS n, AVG(rel_l2) AS mean_rl2,
                   AVG(sign_flip) AS mean_sf, AVG(angle_rms_rad) AS mean_ang
            FROM v_measurement_full GROUP BY codec_name
            ORDER BY mean_rl2
        """)
        if rows:
            print(f"\n=== Codec quality summary (across all measurements) ===")
            print(f"  {'codec':<14} {'family':<10} {'b/pair':>7} {'n':>4} "
                  f"{'rel_L2':>8} {'sign_flip':>9} {'angle_rms':>9}")
            for r in rows:
                print(f"  {r['codec_name']:<14} {r['codec_family']:<10} "
                      f"{r['bits_per_pair']:>7.3f} {r['n']:>4} "
                      f"{r['mean_rl2']:>8.4f} {r['mean_sf']:>9.4f} "
                      f"{r['mean_ang']:>9.4f}")

        # 3. Aberration outliers (per-codec, worst K cells)
        rows = query(conn, """
            SELECT layer, kind, expert, codec_name, rel_l2, sign_flip
            FROM v_measurement_full
            WHERE codec_name = 'vq_k256'
            ORDER BY rel_l2 DESC LIMIT ?
        """, (args.top_n,))
        if rows:
            print(f"\n=== VQ K=256 worst {args.top_n} cells (aberration outliers) ===")
            print(f"  {'layer':>5} {'kind':<6} {'expert':>6} {'rel_L2':>8} {'sign_flip':>9}")
            for r in rows:
                print(f"  {r['layer']:>5} {r['kind']:<6} {r['expert']:>6} "
                      f"{r['rel_l2']:>8.4f} {r['sign_flip']:>9.4f}")

        # 4. Per-token compute distribution per prompt
        rows = query(conn, """
            SELECT pr.prompt_id, pr.notes, pr.n_tokens,
                   SUM(CASE WHEN tc.compute_class='trivial' THEN 1 ELSE 0 END) AS n_t,
                   SUM(CASE WHEN tc.compute_class='easy' THEN 1 ELSE 0 END) AS n_e,
                   SUM(CASE WHEN tc.compute_class='medium' THEN 1 ELSE 0 END) AS n_m,
                   SUM(CASE WHEN tc.compute_class='hard' THEN 1 ELSE 0 END) AS n_h,
                   SUM(CASE WHEN tc.compute_class='undecided' THEN 1 ELSE 0 END) AS n_u,
                   AVG(tc.min_stabilize_layer) AS mean_L
            FROM token_compute tc JOIN prompt_run pr USING(prompt_id)
            GROUP BY pr.prompt_id ORDER BY pr.prompt_id
        """)
        if rows:
            print(f"\n=== Per-token compute distribution (per prompt) ===")
            print(f"  {'id':>3} {'notes':<48} {'n':>4} {'triv%':>5} "
                  f"{'easy%':>5} {'med%':>5} {'hard%':>5} {'und%':>5} {'mL':>5}")
            for r in rows:
                n = r['n_tokens']
                if n == 0: continue
                print(f"  {r['prompt_id']:>3} {r['notes'][:48]:<48} {n:>4} "
                      f"{100*r['n_t']/n:>4.1f}% {100*r['n_e']/n:>4.1f}% "
                      f"{100*r['n_m']/n:>4.1f}% {100*r['n_h']/n:>4.1f}% "
                      f"{100*r['n_u']/n:>4.1f}% {r['mean_L']:>5.1f}")

        # 5. N-gram lock events
        try:
            rows = query(conn, """
                SELECT prompt_id, position, repeat_factor, top_ngram, top_count,
                       triggered_rescue
                FROM ngram_lock_event WHERE triggered_rescue = 1
                ORDER BY repeat_factor DESC LIMIT ?
            """, (args.top_n,))
            if rows:
                print(f"\n=== N-gram lock events (top {args.top_n} by repeat_factor) ===")
                print(f"  {'p_id':>4} {'pos':>6} {'rep_factor':>10} {'count':>5} {'top 5-gram':<60}")
                for r in rows:
                    print(f"  {r['prompt_id']:>4} {r['position']:>6} "
                          f"{r['repeat_factor']:>10.2f} {r['top_count']:>5} "
                          f"{(r['top_ngram'] or '')[:60]!r:<60}")
        except Exception:
            pass

        # 6. Recent journal
        rows = query(conn,
            "SELECT entry_id, ts, action, target FROM journal "
            "ORDER BY entry_id DESC LIMIT ?", (args.journal_tail,))
        if rows:
            print(f"\n=== Recent journal (last {args.journal_tail}) ===")
            for r in rows:
                print(f"  #{r['entry_id']:>4} {r['ts']}  {r['action']:<25} "
                      f"{r['target'][:50] if r['target'] else ''}")


def main():
    p = argparse.ArgumentParser("codec_audit")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init").set_defaults(func=cmd_init)
    sub.add_parser("register-defaults").set_defaults(func=cmd_register_defaults)

    sp = sub.add_parser("scan")
    sp.add_argument("--layer", type=int, default=0)
    sp.add_argument("--kind", default="gate")
    sp.add_argument("--expert-start", type=int, default=0)
    sp.add_argument("--expert-end", type=int, default=256)
    sp.add_argument("--n-subsample", type=int, default=50_000)
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser("query")
    sp.add_argument("--limit", type=int, default=20)
    sp.set_defaults(func=cmd_query)

    sp = sub.add_parser("journal")
    sp.add_argument("--tail", type=int, default=50)
    sp.set_defaults(func=cmd_journal)

    sub.add_parser("codecs").set_defaults(func=cmd_codecs)

    sp = sub.add_parser("report",
        help="unified DB report: codec quality + compute distribution + locks")
    sp.add_argument("--top-n", type=int, default=10)
    sp.add_argument("--journal-tail", type=int, default=15)
    sp.set_defaults(func=cmd_report)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
