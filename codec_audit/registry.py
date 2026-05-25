"""codec_audit/registry.py — codec/source/hidden registration.

All registration is idempotent (UNIQUE constraints). Returns row IDs for
foreign-key references in measurements.
"""
import hashlib
import json
import sqlite3
from typing import Any, Optional

import numpy as np

from .db import journal


def _sha256_hex(data: bytes | np.ndarray) -> str:
    if isinstance(data, np.ndarray):
        data = data.tobytes()
    return hashlib.sha256(data).hexdigest()


def register_codec(conn: sqlite3.Connection, name: str, family: str,
                    bits_per_pair: float, storage_byte_per_pair: float | None = None,
                    params: Optional[dict] = None) -> int:
    """Insert codec spec if not exists. Returns codec_id.

    family: 'iq2_xxs' | 'polar' | 'vq' | 'block_vq' | 'pq'
    bits_per_pair: data bits per (re, im) pair
    storage_byte_per_pair: actual on-disk bytes per pair (including scale overhead)
    params: codec-family-specific parameters

    O(1) lookup via name UNIQUE index.
    """
    if storage_byte_per_pair is None:
        storage_byte_per_pair = bits_per_pair / 8.0
    params_json = json.dumps(params or {}, sort_keys=True)
    spec_bytes = f"{name}|{family}|{bits_per_pair}|{params_json}".encode()
    spec_sha = _sha256_hex(spec_bytes)
    cur = conn.execute(
        "INSERT OR IGNORE INTO codec "
        "(name, family, bits_per_pair, storage_byte_per_pair, params_json, spec_sha256) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (name, family, bits_per_pair, storage_byte_per_pair, params_json, spec_sha),
    )
    if cur.rowcount:
        journal(conn, "register_codec", name,
                {"family": family, "bits_per_pair": bits_per_pair,
                 "storage_byte_per_pair": storage_byte_per_pair, "params": params})
    cur = conn.execute("SELECT codec_id FROM codec WHERE name = ?", (name,))
    return cur.fetchone()["codec_id"]


def ensure_source(conn: sqlite3.Connection, layer: int, kind: str, expert: int,
                   tensor: np.ndarray, name: str | None = None) -> int:
    """Insert source_tensor spec if not exists. Returns source_id.

    UNIQUE on (layer, kind, expert); O(1) lookup.
    """
    if name is None:
        name = f"blk.{layer}.ffn_{kind}_exps.weight[{expert}]"
    sha = _sha256_hex(tensor)
    shape_json = json.dumps(list(tensor.shape))
    cur = conn.execute(
        "INSERT OR IGNORE INTO source_tensor "
        "(name, layer, kind, expert, shape_json, sha256, n_weights) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (name, layer, kind, expert, shape_json, sha, int(tensor.size)),
    )
    if cur.rowcount:
        journal(conn, "register_source", name,
                {"layer": layer, "kind": kind, "expert": expert,
                 "shape": list(tensor.shape), "sha256": sha, "n_weights": int(tensor.size)})
    cur = conn.execute(
        "SELECT source_id FROM source_tensor WHERE layer=? AND kind=? AND expert=?",
        (layer, kind, expert),
    )
    return cur.fetchone()["source_id"]


def ensure_hidden(conn: sqlite3.Connection, kind: str, hidden: np.ndarray,
                   seed_int: int | None = None) -> int:
    """Insert hidden vector if not exists. Returns hidden_id.

    Sherwood field+seed: codec quality depends on activation context (H1800/H1801).
    Every composition measurement names exactly which hidden vector was used.
    """
    sha = _sha256_hex(hidden)
    cur = conn.execute(
        "INSERT OR IGNORE INTO hidden_vector (kind, seed_int, dim, sha256) "
        "VALUES (?, ?, ?, ?)",
        (kind, seed_int, int(hidden.size), sha),
    )
    if cur.rowcount:
        journal(conn, "register_hidden", sha,
                {"kind": kind, "dim": int(hidden.size), "seed_int": seed_int})
    cur = conn.execute("SELECT hidden_id FROM hidden_vector WHERE sha256 = ?", (sha,))
    return cur.fetchone()["hidden_id"]


def list_codecs(conn: sqlite3.Connection, family: str | None = None) -> list[dict]:
    """O(n_codecs) listing. Filter by family (optional)."""
    if family:
        cur = conn.execute(
            "SELECT * FROM codec WHERE family = ? ORDER BY bits_per_pair", (family,))
    else:
        cur = conn.execute("SELECT * FROM codec ORDER BY family, bits_per_pair")
    return [dict(r) for r in cur.fetchall()]
