"""codec_audit/db.py — SQLite plumbing + journal.

O(1) opens, O(log n) journal appends, append-only invariants enforced via
schema triggers. Knuth: one connection per session, batched writes.
"""
import json
import os
import sqlite3
import subprocess
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "codec_audit.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def init_db(path: Path = DB_PATH) -> None:
    """Create schema if not exists. Idempotent.

    Returns: None. Use open_db() afterwards.
    """
    conn = sqlite3.connect(path)
    try:
        with open(SCHEMA_PATH) as f:
            conn.executescript(f.read())
        conn.commit()
    finally:
        conn.close()


@contextmanager
def open_db(path: Path = DB_PATH):
    """Open SQLite connection with WAL + foreign_keys. Initializes schema lazily.

    Usage:
        with open_db() as conn:
            ...

    Returns sqlite3.Connection. Auto-commits on clean exit, rolls back on exception.
    """
    if not path.exists():
        init_db(path)
    conn = sqlite3.connect(path, isolation_level="DEFERRED")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


_GIT_COMMIT_CACHE = None


def _git_commit() -> str:
    """Return current git HEAD SHA. Cached per-process."""
    global _GIT_COMMIT_CACHE
    if _GIT_COMMIT_CACHE is None:
        try:
            _GIT_COMMIT_CACHE = subprocess.check_output(
                ["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            _GIT_COMMIT_CACHE = "unknown"
    return _GIT_COMMIT_CACHE


def journal(conn: sqlite3.Connection, action: str, target: str = "",
            payload: dict | list | str | None = None) -> int:
    """Append-only journal entry. Returns entry_id.

    action: short action name, e.g. 'measure', 'register_codec', 'scan'
    target: identifier (e.g. 'L0_gate_E243/vq_k256')
    payload: JSON-serializable detail dict
    """
    if not isinstance(payload, str):
        payload = json.dumps(payload, default=str)
    cur = conn.execute(
        "INSERT INTO journal (action, target, payload, git_commit, pid) "
        "VALUES (?, ?, ?, ?, ?)",
        (action, target, payload, _git_commit(), os.getpid()),
    )
    return cur.lastrowid


def query(conn: sqlite3.Connection, sql: str, params: tuple | dict = ()) -> list:
    """Convenience: run SELECT, return list of dicts."""
    cur = conn.execute(sql, params)
    return [dict(row) for row in cur.fetchall()]
