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

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
