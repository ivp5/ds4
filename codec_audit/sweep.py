"""codec_audit/sweep.py — high-level scan orchestrators.

scan_layer_kind: one (layer, kind), all experts × all registered codecs.
scan_cross_codec: deeper per-cell A/B (with magnitude strata + outliers).

Each scan writes to the DB through batch transactions (Carmack: amortize
the syscall cost). Journal records the sweep parameters at start and the
result summary at end.
"""
import json
import os
import platform
import sqlite3
import time

import numpy as np

from .db import journal, _git_commit
from .decoders import DECODERS, load_fp4_source
from .measure import aberration_vector, insert_measurement
from .registry import register_codec, ensure_source, list_codecs


def new_run(conn: sqlite3.Connection, name: str, params: dict | None = None) -> int:
    """Open a new run record. Returns run_id."""
    cur = conn.execute(
        "INSERT INTO run (name, host, git_commit, params_json) "
        "VALUES (?, ?, ?, ?)",
        (name, platform.node(), _git_commit(), json.dumps(params or {})),
    )
    rid = cur.lastrowid
    journal(conn, "run_start", name, {"run_id": rid, "params": params or {}})
    return rid


def end_run(conn: sqlite3.Connection, run_id: int, summary: dict | None = None):
    """Close a run record with summary. (run.ended_at is updatable by design.)"""
    # Per the design, only 'run' rows are mutable (start/end timestamps).
    # If you want fully append-only, write a separate run_end journal entry instead.
    conn.execute("UPDATE run SET ended_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
                 "WHERE run_id = ?", (run_id,))
    journal(conn, "run_end", str(run_id), summary or {})


def scan_layer_kind(conn: sqlite3.Connection, layer: int, kind: str,
                     experts: range | list = range(256),
                     codecs: list[str] | None = None,
                     n_subsample: int = 50_000,
                     seed_int: int = 42,
                     run_id: int | None = None) -> dict:
    """Run aberration vector for all (expert × codec) cells of one (layer, kind).

    O(n_experts × n_codecs) per call. FP4 source cached per expert
    (lru_cache); polar/VQ corpora loaded once per (layer, kind).

    Returns summary dict with counts and aggregate stats.
    """
    if codecs is None:
        codecs = list(DECODERS.keys())
    rng = np.random.default_rng(seed_int)
    t0 = time.time()
    n_inserted = 0
    n_errors = 0
    journal(conn, "scan_layer_kind_start",
            f"L{layer}_{kind}",
            {"experts": list(experts) if not isinstance(experts, range) else
             [experts.start, experts.stop], "codecs": codecs,
             "n_subsample": n_subsample, "seed_int": seed_int})

    for expert in experts:
        try:
            src_2d = load_fp4_source(layer, kind, expert)
        except Exception as e:
            n_errors += 1
            journal(conn, "fp4_load_fail",
                    f"L{layer}_{kind}_E{expert}", {"error": str(e)})
            continue
        src = src_2d.flatten()
        source_id = ensure_source(conn, layer, kind, expert, src_2d)
        # Subsample indices reused across all codecs for this cell
        if n_subsample < src.size:
            sub_idx = rng.choice(src.size, size=n_subsample, replace=False)
        else:
            sub_idx = None
        for codec_name in codecs:
            decode_fn = DECODERS.get(codec_name)
            if decode_fn is None:
                continue
            try:
                rec = decode_fn(layer, kind, expert)
            except Exception as e:
                n_errors += 1
                journal(conn, "decode_fail",
                        f"L{layer}_{kind}_E{expert}/{codec_name}", {"error": str(e)})
                continue
            if rec is None:
                continue
            rec_flat = rec.flatten()
            # Resize source slice to match codec scope (polar/VQ may be 128-row subset)
            if rec_flat.size != src.size:
                src_slice = src[:rec_flat.size]
                sub_idx_local = sub_idx[sub_idx < rec_flat.size] if sub_idx is not None else None
            else:
                src_slice = src
                sub_idx_local = sub_idx
            metrics = aberration_vector(rec_flat, src_slice, sub_idx_local)
            codec_id = _codec_id(conn, codec_name)
            insert_measurement(conn, source_id, codec_id, metrics,
                                n_subsample=(len(sub_idx_local) if sub_idx_local is not None
                                              else src_slice.size),
                                seed_int=seed_int, run_id=run_id)
            n_inserted += 1

    elapsed = time.time() - t0
    summary = {"inserted": n_inserted, "errors": n_errors,
               "elapsed_s": elapsed,
               "experts_attempted": len(experts) if hasattr(experts, "__len__")
                                     else (experts.stop - experts.start),
               "codecs": codecs}
    journal(conn, "scan_layer_kind_end", f"L{layer}_{kind}", summary)
    return summary


def scan_cross_codec(conn: sqlite3.Connection,
                      cells: list[tuple[int, str, int]],
                      codecs: list[str] | None = None,
                      n_subsample: int = 50_000,
                      seed_int: int = 42,
                      run_id: int | None = None) -> list[dict]:
    """Cross-cell × cross-codec sweep with full metric vector.

    cells: list of (layer, kind, expert) tuples
    Returns one dict per cell with embedded codec metrics.
    """
    if codecs is None:
        codecs = list(DECODERS.keys())
    rng = np.random.default_rng(seed_int)
    out = []
    journal(conn, "scan_cross_codec_start", "",
            {"n_cells": len(cells), "codecs": codecs,
             "n_subsample": n_subsample, "seed_int": seed_int})
    for (layer, kind, expert) in cells:
        try:
            src_2d = load_fp4_source(layer, kind, expert)
        except Exception as e:
            out.append({"layer": layer, "kind": kind, "expert": expert,
                        "error": f"fp4_load: {e}"})
            continue
        src = src_2d.flatten()
        source_id = ensure_source(conn, layer, kind, expert, src_2d)
        if n_subsample < src.size:
            sub_idx = rng.choice(src.size, size=n_subsample, replace=False)
        else:
            sub_idx = None
        cell = {"layer": layer, "kind": kind, "expert": expert, "codecs": {}}
        for codec_name in codecs:
            decode_fn = DECODERS.get(codec_name)
            if decode_fn is None:
                continue
            try:
                rec = decode_fn(layer, kind, expert)
            except Exception as e:
                cell["codecs"][codec_name] = {"error": str(e)}
                continue
            if rec is None:
                cell["codecs"][codec_name] = None
                continue
            rec_flat = rec.flatten()
            if rec_flat.size != src.size:
                src_slice = src[:rec_flat.size]
                sub_idx_local = sub_idx[sub_idx < rec_flat.size] if sub_idx is not None else None
            else:
                src_slice = src
                sub_idx_local = sub_idx
            metrics = aberration_vector(rec_flat, src_slice, sub_idx_local)
            cell["codecs"][codec_name] = metrics
            codec_id = _codec_id(conn, codec_name)
            insert_measurement(conn, source_id, codec_id, metrics,
                                n_subsample=(len(sub_idx_local) if sub_idx_local is not None
                                              else src_slice.size),
                                seed_int=seed_int, run_id=run_id)
        out.append(cell)
    journal(conn, "scan_cross_codec_end", "",
            {"cells_done": len(out)})
    return out


# ============================================================================
# Helpers
# ============================================================================

def _codec_id(conn: sqlite3.Connection, name: str) -> int:
    cur = conn.execute("SELECT codec_id FROM codec WHERE name = ?", (name,))
    row = cur.fetchone()
    if row is None:
        raise ValueError(
            f"codec '{name}' not registered. Call register_codec first."
        )
    return row["codec_id"]
