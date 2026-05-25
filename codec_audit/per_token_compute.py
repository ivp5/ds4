"""codec_audit/per_token_compute.py — per-token computational requirement detector.

silv 2026-05-25: "Build a tool to detect per-token computational requirement."

Substrate ground (Qwen3.5-4B-MLX-4bit, established 2026-05-23/24):
  - 32-layer hybrid: 24 GatedDeltaNet linear-attn + 8 full-attention (L3/7/11/15/19/23/27/31)
  - L0-L6: prompt-processing tier (cos≥0.98 between prompts)
  - L7: first full-attention RoPE, first meaningful fracture
  - L18: COMMIT INSTALLATION (largest single-layer cosine drop)
  - L19: norm explosion (RoPE full-attention; 15→73 from L19 to L31)
  - L25-L31: class-specialized commit tier (cumulative MLP coherence)

Per-token compute classification (4 levels, derived from 4 signals):
  TRIVIAL    — early-exit possible at L≤24 (8.86% of tokens per measured P05)
  EASY       — stabilizes at L25-L28 (~25%)
  MEDIUM     — stabilizes at L29-L30 (~50%)
  HARD       — needs L=31 + low-margin top-1 (~17%)

Signals (per-token, all derivable from one prefill pass):
  - min_stabilize_layer: smallest L where projected top-1 matches L_final top-1
  - final_entropy_nats:  entropy at L_final softmax (uncertainty)
  - final_top1_margin:   logp[top-1] - logp[top-2] (decision sharpness)
  - layer_top1_changes:  # of distinct top-1 candidates across layers L18-L31

Records each (prompt, position) detection to the codec_audit DB.
Journal records the prompt registration and detection sweep parameters.
"""
import hashlib
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

from .db import open_db, journal
from .registry import _sha256_hex


# Compute-class thresholds (calibrated against the 700-token P05 substrate scan)
TRIVIAL_MAX_L = 24
EASY_MAX_L = 28
MEDIUM_MAX_L = 30
LOW_MARGIN_NATS = 0.5   # logp gap that flags low-confidence decision
HIGH_ENTROPY_NATS = 1.5  # entropy threshold for hard-token flag


def _ensure_token_compute_schema(conn: sqlite3.Connection):
    """Add token_compute table if missing. Schema is append-only via triggers."""
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS prompt_run (
        prompt_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        model       TEXT NOT NULL,
        text_sha256 TEXT NOT NULL UNIQUE,
        text_head   TEXT,
        n_tokens    INTEGER NOT NULL,
        notes       TEXT,
        ts          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    );
    CREATE INDEX IF NOT EXISTS idx_prompt_model ON prompt_run(model);

    CREATE TABLE IF NOT EXISTS token_compute (
        tc_id                INTEGER PRIMARY KEY AUTOINCREMENT,
        prompt_id            INTEGER NOT NULL REFERENCES prompt_run(prompt_id),
        position             INTEGER NOT NULL,
        token_id             INTEGER,
        token_text           TEXT,
        min_stabilize_layer  INTEGER,
        final_entropy_nats   REAL,
        final_top1_margin    REAL,
        layer_top1_changes   INTEGER,
        compute_class        TEXT NOT NULL,
        layer_top1_json      TEXT,
        run_id               INTEGER REFERENCES run(run_id),
        ts                   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_token_compute_unique
      ON token_compute(prompt_id, position);
    CREATE INDEX IF NOT EXISTS idx_token_compute_class
      ON token_compute(compute_class);

    CREATE TRIGGER IF NOT EXISTS prompt_run_no_update
      BEFORE UPDATE ON prompt_run
      BEGIN SELECT RAISE(ABORT, 'prompt_run is immutable'); END;
    CREATE TRIGGER IF NOT EXISTS prompt_run_no_delete
      BEFORE DELETE ON prompt_run
      BEGIN SELECT RAISE(ABORT, 'prompt_run is immutable'); END;
    CREATE TRIGGER IF NOT EXISTS token_compute_no_update
      BEFORE UPDATE ON token_compute
      BEGIN SELECT RAISE(ABORT, 'token_compute rows are immutable'); END;
    CREATE TRIGGER IF NOT EXISTS token_compute_no_delete
      BEFORE DELETE ON token_compute
      BEGIN SELECT RAISE(ABORT, 'token_compute rows are immutable'); END;
    """)


def register_prompt(conn: sqlite3.Connection, model: str, text: str,
                     n_tokens: int, notes: str = "") -> int:
    """Insert prompt_run row if not exists. Returns prompt_id. O(1) via UNIQUE."""
    _ensure_token_compute_schema(conn)
    sha = _sha256_hex(text.encode("utf-8"))
    cur = conn.execute(
        "INSERT OR IGNORE INTO prompt_run (model, text_sha256, text_head, n_tokens, notes) "
        "VALUES (?, ?, ?, ?, ?)",
        (model, sha, text[:256], n_tokens, notes),
    )
    if cur.rowcount:
        journal(conn, "register_prompt", model,
                {"sha256": sha, "n_tokens": n_tokens, "head": text[:80]})
    cur = conn.execute("SELECT prompt_id FROM prompt_run WHERE text_sha256 = ?", (sha,))
    return cur.fetchone()["prompt_id"]


def classify_compute(min_stab_L: int, entropy_nats: float, margin_nats: float,
                      n_layer_changes: int, n_layers: int = 32) -> str:
    """Classify per-token compute requirement from 4 signals.

    Returns: 'trivial' | 'easy' | 'medium' | 'hard' | 'undecided'
    """
    # Hard if needs full depth AND has low margin or high entropy
    if min_stab_L >= n_layers - 1:
        if margin_nats < LOW_MARGIN_NATS or entropy_nats > HIGH_ENTROPY_NATS:
            return "undecided"
        return "hard"
    if min_stab_L <= TRIVIAL_MAX_L:
        return "trivial"
    if min_stab_L <= EASY_MAX_L:
        return "easy"
    if min_stab_L <= MEDIUM_MAX_L:
        return "medium"
    return "hard"


def compute_per_token_signals(per_layer_topk: list[list[int]],
                               final_logits: np.ndarray) -> dict:
    """Derive 4-dim compute signal from per-layer top-k arrays + L_final logits.

    per_layer_topk: list of len n_layers, each item is list of top-K token IDs
                    at that layer's projection. Item 0 = layer 0, etc.
    final_logits:   1D fp32 array (vocab_size,) — raw L_final logits

    Returns dict with 4 signals.
    """
    n_layers = len(per_layer_topk)
    final_top1 = per_layer_topk[-1][0] if per_layer_topk[-1] else None

    # min_stabilize_layer: smallest L such that top-1[L..n] == final_top1 for all L'>=L
    min_stab = n_layers - 1  # default = full depth
    for L in range(n_layers - 1, -1, -1):
        if not per_layer_topk[L] or per_layer_topk[L][0] != final_top1:
            min_stab = L + 1
            break
    if min_stab > n_layers - 1:
        min_stab = n_layers - 1

    # Distinct top-1 candidates across L18-L31 (late commit tier)
    late_top1_set = set()
    for L in range(min(18, n_layers - 1), n_layers):
        if per_layer_topk[L]:
            late_top1_set.add(per_layer_topk[L][0])
    n_changes = len(late_top1_set)

    # Final-layer entropy + top-2 margin
    if final_logits is not None and final_logits.size > 0:
        # log-softmax for numerical stability
        z = final_logits - final_logits.max()
        ez = np.exp(z, dtype=np.float64)
        sz = ez.sum()
        probs = ez / sz
        # entropy in nats
        nz = probs > 0
        entropy = float(-(probs[nz] * np.log(probs[nz])).sum())
        # top-2 margin in log-prob space (nats)
        idx = np.argpartition(-final_logits, 2)[:2]
        sorted_idx = idx[np.argsort(-final_logits[idx])]
        logp = np.log(probs[sorted_idx] + 1e-30)
        margin = float(logp[0] - logp[1])
    else:
        entropy = float("nan")
        margin = float("nan")

    return {
        "min_stabilize_layer": int(min_stab),
        "final_entropy_nats": entropy,
        "final_top1_margin": margin,
        "layer_top1_changes": n_changes,
        "final_top1_id": int(final_top1) if final_top1 is not None else None,
        "late_top1_ids": sorted(late_top1_set),
    }


def insert_token_compute(conn: sqlite3.Connection, prompt_id: int, position: int,
                          token_id: int | None, token_text: str | None,
                          signals: dict, compute_class: str,
                          run_id: int | None = None) -> int:
    """Insert one per-token compute row. Append-only (triggers enforce)."""
    cur = conn.execute(
        "INSERT OR IGNORE INTO token_compute "
        "(prompt_id, position, token_id, token_text, "
        " min_stabilize_layer, final_entropy_nats, final_top1_margin, "
        " layer_top1_changes, compute_class, layer_top1_json, run_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (prompt_id, position, token_id, token_text,
         signals["min_stabilize_layer"], signals["final_entropy_nats"],
         signals["final_top1_margin"], signals["layer_top1_changes"],
         compute_class, json.dumps(signals.get("late_top1_ids", [])), run_id),
    )
    return cur.lastrowid


def ingest_min_layer_results(conn: sqlite3.Connection,
                              results_json_path: Path,
                              model: str = "mlx-community/Qwen3.5-4B-MLX-4bit",
                              run_id: int | None = None) -> dict:
    """One-shot import of existing per-token early-exit probe results.

    Reads results_*.json from tmp/20260524_per_token_early_exit/ and writes
    token_compute rows to the DB. O(n_positions).

    Returns summary dict with counts per compute_class.
    """
    _ensure_token_compute_schema(conn)
    data = json.loads(Path(results_json_path).read_text())
    mls = data.get("min_layer_per_position", [])
    cells = data.get("cells", "unknown")
    n_layers = data.get("n_layers", 32)

    text_proxy = f"{model}::{cells}::{len(mls)}_tokens"
    prompt_id = register_prompt(conn, model=model, text=text_proxy,
                                  n_tokens=len(mls),
                                  notes=f"ingested from {results_json_path.name}")

    class_counts = {}
    for pos, min_L in enumerate(mls):
        # Without entropy/margin from this probe, use minimal signal classification
        cls = classify_compute(min_stab_L=int(min_L),
                                entropy_nats=0.0, margin_nats=999.0,
                                n_layer_changes=0, n_layers=n_layers)
        sig = {"min_stabilize_layer": int(min_L),
                "final_entropy_nats": None, "final_top1_margin": None,
                "layer_top1_changes": None, "late_top1_ids": []}
        try:
            insert_token_compute(conn, prompt_id=prompt_id, position=pos,
                                  token_id=None, token_text=None,
                                  signals=sig, compute_class=cls, run_id=run_id)
            class_counts[cls] = class_counts.get(cls, 0) + 1
        except sqlite3.IntegrityError:
            pass  # already ingested

    journal(conn, "ingest_min_layer", str(results_json_path),
            {"prompt_id": prompt_id, "n_tokens": len(mls), "class_counts": class_counts})
    return {"prompt_id": prompt_id, "n_tokens": len(mls), "class_counts": class_counts}


def class_histogram(conn: sqlite3.Connection, prompt_id: int) -> dict:
    """O(n) GROUP BY on the unique index."""
    cur = conn.execute(
        "SELECT compute_class, COUNT(*) AS n, "
        "       AVG(min_stabilize_layer) AS mean_L, "
        "       MIN(min_stabilize_layer) AS min_L, "
        "       MAX(min_stabilize_layer) AS max_L "
        "FROM token_compute WHERE prompt_id = ? GROUP BY compute_class "
        "ORDER BY mean_L",
        (prompt_id,),
    )
    return [dict(r) for r in cur.fetchall()]
