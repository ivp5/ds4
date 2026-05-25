"""codec_audit/measure.py — aberration vector + composition metrics.

All measurements are pure functions of (reconstruction, source) numpy arrays.
Vectorized, O(n) in input size, O(1) extra space.

Geohotz minimum-viable-code: 6 metrics in one function, no class hierarchy.
Donoho rigor: n_subsample always required, seed always recorded.
"""
import json
import sqlite3
from typing import Optional

import numpy as np


def aberration_vector(rec: np.ndarray, src: np.ndarray,
                       subsample_idx: Optional[np.ndarray] = None) -> dict:
    """6-dim aberration measurement comparing reconstruction to source.

    Returns:
        rel_l2: ||rec - src|| / ||src||
        sign_flip: fraction of non-zero src weights where sign(rec) != sign(src)
        exactly_zero: fraction of diffs exactly equal to 0 (codec hits source exactly)
        max_abs: max(|rec - src|)
        mean_abs: mean(|rec - src|)
        angle_rms_rad: RMS angle error per (re, im) pair, in radians

    O(n) time, O(n) space (intermediate diff array).
    """
    if subsample_idx is not None:
        src = src[subsample_idx]
        rec = rec[subsample_idx]
    diff = rec - src
    src_norm = float(np.linalg.norm(src))
    if src_norm < 1e-30:
        rel_l2 = 0.0
    else:
        rel_l2 = float(np.linalg.norm(diff) / src_norm)
    nz = src != 0.0
    sign_flip = (float((np.sign(rec[nz]) != np.sign(src[nz])).mean())
                 if nz.any() else 0.0)
    exactly_zero = float((diff == 0.0).mean())
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    # Per-pair angle error (trim to even length to avoid shape mismatch
    # under odd-count subsample)
    n_pair = (src.size // 2) * 2
    re_s, im_s = src[:n_pair:2], src[1:n_pair:2]
    re_r, im_r = rec[:n_pair:2], rec[1:n_pair:2]
    ang_s = np.arctan2(im_s, re_s)
    ang_r = np.arctan2(im_r, re_r)
    ang_diff = np.mod(ang_r - ang_s + np.pi, 2 * np.pi) - np.pi
    angle_rms_rad = float(np.sqrt(np.mean(ang_diff ** 2)))
    return {
        "rel_l2": rel_l2,
        "sign_flip": sign_flip,
        "exactly_zero": exactly_zero,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "angle_rms_rad": angle_rms_rad,
    }


def magnitude_strata(rec: np.ndarray, src: np.ndarray, n_strata: int = 10) -> list:
    """rel_L2 per magnitude decile of source weights.

    Reveals magnitude-correlated error (key finding: IQ2 has rel_L2 > 1 on
    low-magnitude weights). O(n log n) for the sort.
    """
    abs_src = np.abs(src)
    sort_idx = np.argsort(abs_src)
    n = len(src)
    strata = []
    for k in range(n_strata):
        lo = (k * n) // n_strata
        hi = ((k + 1) * n) // n_strata
        idx = sort_idx[lo:hi]
        s = src[idx]
        r = rec[idx]
        s_norm = float(np.linalg.norm(s))
        rl2 = float(np.linalg.norm(r - s) / max(s_norm, 1e-30))
        strata.append({
            "stratum": k, "n": int(hi - lo),
            "mean_abs_src": float(np.mean(abs_src[idx])),
            "rel_l2": rl2,
            "max_abs_err": float(np.max(np.abs(r - s))),
        })
    return strata


def composition_metrics(gate_rec: np.ndarray, gate_src: np.ndarray,
                         up_rec: np.ndarray, up_src: np.ndarray,
                         down_rec_rows: np.ndarray, down_src_rows: np.ndarray,
                         hidden: np.ndarray,
                         selected_down_idx: list[int]) -> dict:
    """Full gate/up/silu·up/down composition error (H1795-H1801 schema).

    Per H1800/H1801: codec quality lives in the dot-product carrier under
    a specific hidden vector, not in static reconstruction.

    gate_rec/gate_src: 2D (n_rows, in_dim) for gate proj
    up_rec/up_src:    2D (n_rows, in_dim) for up proj
    down_rec_rows/down_src_rows: 2D (n_down_rows, n_rows) for SELECTED down rows
    hidden:           1D (in_dim,) hidden vector
    selected_down_idx: which down rows are in down_rec_rows

    Returns dict matching composition_measurement schema.
    """
    # Gate / up dots
    gate_dot_src = gate_src @ hidden
    gate_dot_rec = gate_rec @ hidden
    up_dot_src = up_src @ hidden
    up_dot_rec = up_rec @ hidden

    gate_rl2 = float(np.linalg.norm(gate_dot_rec - gate_dot_src) /
                      max(np.linalg.norm(gate_dot_src), 1e-30))
    up_rl2 = float(np.linalg.norm(up_dot_rec - up_dot_src) /
                    max(np.linalg.norm(up_dot_src), 1e-30))

    # silu(gate) * up
    def silu(x): return x / (1.0 + np.exp(-x))
    act_src = silu(gate_dot_src) * up_dot_src
    act_rec = silu(gate_dot_rec) * up_dot_rec
    act_rl2 = float(np.linalg.norm(act_rec - act_src) /
                     max(np.linalg.norm(act_src), 1e-30))
    cos_src_rec = float(np.dot(act_src, act_rec) /
                         max(np.linalg.norm(act_src) * np.linalg.norm(act_rec), 1e-30))

    # Down rows (selected)
    down_rl2 = float(np.linalg.norm(down_rec_rows - down_src_rows) /
                      max(np.linalg.norm(down_src_rows), 1e-30))

    # Selected down outputs
    down_out_src = down_src_rows @ act_src
    down_out_rec = down_rec_rows @ act_rec
    sel_down_rl2 = float(np.linalg.norm(down_out_rec - down_out_src) /
                          max(np.linalg.norm(down_out_src), 1e-30))

    return {
        "gate_rel_l2": gate_rl2,
        "up_rel_l2": up_rl2,
        "down_row_rel_l2": down_rl2,
        "activation_rel_l2": act_rl2,
        "activation_cos": cos_src_rec,
        "selected_down_rel_l2": sel_down_rl2,
        "selected_rows_json": json.dumps(selected_down_idx),
    }


def insert_measurement(conn: sqlite3.Connection, source_id: int, codec_id: int,
                        metrics: dict, n_subsample: int, seed_int: int,
                        hidden_id: int | None = None, run_id: int | None = None,
                        strata: list | None = None, outliers: dict | None = None,
                        notes: str | None = None) -> int:
    """Append a measurement row. Append-only (triggers prevent UPDATE/DELETE).

    O(log n) insert via B-tree index.
    """
    cur = conn.execute(
        "INSERT INTO measurement "
        "(source_id, codec_id, hidden_id, n_subsample, seed_int, "
        " rel_l2, sign_flip, exactly_zero, max_abs, mean_abs, angle_rms_rad, "
        " strata_json, outliers_json, notes, run_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (source_id, codec_id, hidden_id, n_subsample, seed_int,
         metrics["rel_l2"], metrics.get("sign_flip"), metrics.get("exactly_zero"),
         metrics.get("max_abs"), metrics.get("mean_abs"), metrics.get("angle_rms_rad"),
         json.dumps(strata) if strata else None,
         json.dumps(outliers) if outliers else None,
         notes, run_id),
    )
    return cur.lastrowid


def insert_composition(conn: sqlite3.Connection, source_id: int, codec_id: int,
                        hidden_id: int, metrics: dict,
                        run_id: int | None = None) -> int:
    """Append a composition measurement row. Append-only."""
    cur = conn.execute(
        "INSERT INTO composition_measurement "
        "(source_id, codec_id, hidden_id, gate_rel_l2, up_rel_l2, down_row_rel_l2, "
        " activation_rel_l2, activation_cos, selected_down_rel_l2, "
        " selected_rows_json, run_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (source_id, codec_id, hidden_id,
         metrics.get("gate_rel_l2"), metrics.get("up_rel_l2"),
         metrics.get("down_row_rel_l2"), metrics.get("activation_rel_l2"),
         metrics.get("activation_cos"), metrics.get("selected_down_rel_l2"),
         metrics.get("selected_rows_json"), run_id),
    )
    return cur.lastrowid
