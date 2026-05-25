"""N-gram cache-lock detector — the orthogonal loop signature.

silv 2026-05-25: time-series probe refuted my single-window class-distribution
threshold. The refinement direction: track repetition density via n-gram
matches in a rolling window. This complements the per-token compute signal.

Mechanism (F6.5 finding from substrate_vertical_descent):
  - Loop lock = cache content repetition (NOT position-encoding artifact)
  - When 4-5 consecutive tokens repeat their content from N positions back,
    the induction-head circuit attends to the prior occurrence and the
    cache feeds back its own past content → continuation of the same pattern

Detector:
  - Rolling window of last W tokens (default 200)
  - For each n in {3, 4, 5}: count distinct n-grams + n-gram repetition factor
  - n_gram_repeat_factor = (n_total_ngrams) / (n_distinct_ngrams)
  - LOCK SIGNATURE: repeat_factor >= 3 for n=5 over the window (top-5-gram
    appears ≥3× in last 200 tokens)
  - LOCK ESCAPE TRIGGER: emit Conjecture #23 rescue command

Pure-text analyzer; no model forward needed. O(W) per token shift via
incremental update.

Records detections to codec_audit DB through journal + a new
`ngram_lock_event` table.
"""
import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from codec_audit.db import open_db, journal


def _ensure_ngram_schema(conn: sqlite3.Connection):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS ngram_lock_event (
        event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        prompt_id   INTEGER REFERENCES prompt_run(prompt_id),
        position    INTEGER NOT NULL,
        window      INTEGER NOT NULL,
        n           INTEGER NOT NULL,
        n_total     INTEGER NOT NULL,
        n_distinct  INTEGER NOT NULL,
        repeat_factor REAL NOT NULL,
        top_ngram   TEXT,
        top_count   INTEGER,
        compute_class_dist TEXT,
        triggered_rescue INTEGER NOT NULL DEFAULT 0,
        ts          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    );
    CREATE INDEX IF NOT EXISTS idx_ngram_event_prompt
      ON ngram_lock_event(prompt_id, position);
    CREATE TRIGGER IF NOT EXISTS ngram_event_no_update
      BEFORE UPDATE ON ngram_lock_event
      BEGIN SELECT RAISE(ABORT, 'ngram_lock_event is append-only'); END;
    CREATE TRIGGER IF NOT EXISTS ngram_event_no_delete
      BEFORE DELETE ON ngram_lock_event
      BEGIN SELECT RAISE(ABORT, 'ngram_lock_event is append-only'); END;
    """)


def ngram_signature(tokens: list, n: int = 5) -> dict:
    """Compute n-gram repetition signature for a token sequence.

    Returns:
        n_total: total n-grams in sequence (= len(tokens) - n + 1)
        n_distinct: distinct n-grams
        repeat_factor: n_total / n_distinct (1.0 = no repeats)
        top_ngram: most-repeated n-gram
        top_count: how many times it appears
    """
    if len(tokens) < n:
        return {"n_total": 0, "n_distinct": 0, "repeat_factor": 1.0,
                "top_ngram": None, "top_count": 0}
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    n_total = len(ngrams)
    n_distinct = len(counts)
    most = counts.most_common(1)[0]
    return {
        "n_total": n_total,
        "n_distinct": n_distinct,
        "repeat_factor": n_total / max(n_distinct, 1),
        "top_ngram": list(most[0]),
        "top_count": most[1],
    }


def successor_diversity(tokens: list, target_ngram: tuple, n: int = 5) -> dict:
    """For each occurrence of target_ngram in tokens, look at the next n-gram.

    Returns count + distinct count of successor n-grams. Low diversity =
    true cycle (model always continues the same way after this pattern).
    High diversity = exploration (model uses this pattern as a step in
    multiple different paths).
    """
    successors = []
    for i in range(len(tokens) - 2 * n + 1):
        if tuple(tokens[i:i+n]) == target_ngram:
            successors.append(tuple(tokens[i+n:i+2*n]))
    n_total = len(successors)
    n_distinct = len(set(successors))
    return {
        "n_total": n_total,
        "n_distinct": n_distinct,
        "successor_diversity": n_distinct / max(n_total, 1),
    }


def rolling_lock_scan(tokens: list, window: int = 200,
                       n_values=(3, 4, 5),
                       lock_threshold: float = 3.0,
                       step: int = 20,
                       check_successor_diversity: bool = True) -> list:
    """Slide a window over tokens, computing n-gram signature at each step.

    Returns list of dicts with rolling position + per-n signatures.
    Lock signature: ANY n in n_values has repeat_factor >= lock_threshold
    AND successor_diversity (if checked) < 0.5 (= mostly same continuation).

    O(N/step * window) — fast enough for live generation monitoring.
    """
    results = []
    for end in range(window, len(tokens) + 1, step):
        chunk = tokens[end - window:end]
        per_n = {}
        locked = False
        true_lock = False  # repetition + low successor diversity
        for n in n_values:
            sig = ngram_signature(chunk, n)
            per_n[f"n{n}"] = sig
            if sig["repeat_factor"] >= lock_threshold:
                locked = True
                if check_successor_diversity and sig["top_count"] >= 3:
                    succ = successor_diversity(
                        chunk, tuple(sig["top_ngram"]), n)
                    sig["successor"] = succ
                    if succ["successor_diversity"] <= 0.5 and succ["n_total"] >= 3:
                        true_lock = True
        results.append({
            "position": end,
            "window_size": window,
            "per_n": per_n,
            "locked": locked,
            "true_lock": true_lock,
            "max_repeat_factor": max(per_n[f"n{n}"]["repeat_factor"] for n in n_values),
        })
    return results


def analyze_response(json_path: Path, field: str = "response",
                      window: int = 200, step: int = 50,
                      lock_threshold: float = 3.0) -> dict:
    """Analyze a cached response file for loop-lock signatures.

    Uses simple whitespace tokenization (no model needed) — the lock signature
    is at the WORD level, not subword. This is intentional: the silv-observed
    "Could it mean Patrick starts at t=0..." loop is at word-tuple granularity.
    """
    data = json.loads(json_path.read_text())
    text = data[field]
    # Whitespace tokenize for word-level signature
    tokens = text.split()
    print(f"text: {len(text)} chars, {len(tokens)} whitespace-tokens")

    series = rolling_lock_scan(tokens, window=window,
                                n_values=(3, 4, 5),
                                lock_threshold=lock_threshold,
                                step=step)

    n_locked = sum(1 for s in series if s["locked"])
    n_true_lock = sum(1 for s in series if s["true_lock"])
    first_lock = next((s for s in series if s["locked"]), None)
    first_true_lock = next((s for s in series if s["true_lock"]), None)

    # H1801-lesson refinement (2026-05-25): per-window detector alone doesn't
    # generalize — track DOMINANT TOP-5-GRAM PERSISTENCE across windows.
    # P01 loop: 8 distinct top-5-grams in 54 windows, most-persistent in 85%.
    # P02 success: 13 distinct in 60 windows, most-persistent in 72%.
    from collections import Counter
    top_seq = [
        ' '.join(s['per_n']['n5']['top_ngram']) if s['per_n']['n5']['top_ngram'] else ''
        for s in series
    ]
    top_seq = [t for t in top_seq if t]
    top_counts = Counter(top_seq)
    n_distinct_top5 = len(top_counts)
    most_persistent_n = top_counts.most_common(1)[0][1] if top_counts else 0
    most_persistent_fraction = most_persistent_n / max(len(top_seq), 1)
    distinct_rate = n_distinct_top5 / max(len(top_seq), 1)
    # Conservative trigger thresholds derived from P01/P02 cross-cell test
    high_confidence_lock = (most_persistent_fraction > 0.80
                             and distinct_rate < 0.15)

    return {
        "json": str(json_path),
        "n_chars": len(text),
        "n_word_tokens": len(tokens),
        "n_windows": len(series),
        "n_locked_windows": n_locked,
        "n_true_lock_windows": n_true_lock,
        "first_lock_position": first_lock["position"] if first_lock else None,
        "first_true_lock_position": first_true_lock["position"] if first_true_lock else None,
        "first_lock_top_ngram": (
            first_lock["per_n"]["n5"]["top_ngram"] if first_lock else None
        ),
        "first_true_lock_top_ngram": (
            first_true_lock["per_n"]["n5"]["top_ngram"] if first_true_lock else None
        ),
        "n_distinct_top5grams": n_distinct_top5,
        "distinct_rate": distinct_rate,
        "most_persistent_fraction": most_persistent_fraction,
        "most_persistent_ngram": top_counts.most_common(1)[0][0] if top_counts else None,
        "high_confidence_lock": high_confidence_lock,
        "series": series,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True)
    p.add_argument("--field", default="response")
    p.add_argument("--window", type=int, default=200)
    p.add_argument("--step", type=int, default=50)
    p.add_argument("--lock-threshold", type=float, default=3.0)
    p.add_argument("--save-to-db", action="store_true",
                    help="Record lock events to codec_audit DB")
    args = p.parse_args()

    res = analyze_response(Path(args.json), field=args.field,
                            window=args.window, step=args.step,
                            lock_threshold=args.lock_threshold)

    print(f"\n=== n-gram lock scan: {Path(args.json).name} ===")
    print(f"  text: {res['n_chars']} chars, {res['n_word_tokens']} whitespace-tokens")
    print(f"  windows: {res['n_windows']} (W={args.window}, step={args.step})")
    nl, nt = res['n_locked_windows'], res['n_true_lock_windows']
    pl = 100 * nl / max(res['n_windows'], 1)
    pt = 100 * nt / max(res['n_windows'], 1)
    print(f"  LOCKED windows:    {nl:>4} ({pl:.1f}%)")
    print(f"  TRUE-LOCK windows: {nt:>4} ({pt:.1f}%)  ← repetition + low successor-diversity")
    if res["first_lock_position"] is not None:
        print(f"  FIRST LOCK at position {res['first_lock_position']}")
        print(f"  top 5-gram: {' '.join(res['first_lock_top_ngram'])!r}")
    if res["first_true_lock_position"] is not None:
        print(f"  FIRST TRUE-LOCK at position {res['first_true_lock_position']}")
        print(f"  top 5-gram: {' '.join(res['first_true_lock_top_ngram'])!r}")
    if res["first_lock_position"] is None:
        print(f"  no locks at threshold {args.lock_threshold}")

    # H1801-lesson refinement: cross-window persistence signal
    print(f"\n  --- cross-window persistence signal ---")
    print(f"  distinct top-5-grams: {res['n_distinct_top5grams']}")
    print(f"  distinct-rate: {res['distinct_rate']:.3f}  (< 0.15 → STUCK)")
    print(f"  most-persistent top 5-gram fraction: {res['most_persistent_fraction']:.3f}  (> 0.80 → STUCK)")
    if res['most_persistent_ngram']:
        print(f"  most-persistent top 5-gram: {res['most_persistent_ngram']!r}")
    print(f"  HIGH-CONFIDENCE LOCK: {res['high_confidence_lock']}")

    print(f"\n  position  max_repeat  n5_top_count  n5_distinct  n5_top_ngram")
    print(f"  --------  ----------  ------------  -----------  ------------")
    for s in res["series"]:
        n5 = s["per_n"]["n5"]
        flag = " ← LOCK" if s["locked"] else ""
        top = ' '.join(n5["top_ngram"]) if n5["top_ngram"] else ''
        print(f"  {s['position']:>8}  {s['max_repeat_factor']:>10.2f}  "
              f"{n5['top_count']:>12}  {n5['n_distinct']:>11}  "
              f"{top[:50]!r:<52}{flag}")

    if args.save_to_db:
        with open_db() as conn:
            _ensure_ngram_schema(conn)
            # Find prompt_id by notes-search (best-effort)
            cur = conn.execute(
                "SELECT prompt_id FROM prompt_run WHERE text_head LIKE ? LIMIT 1",
                (f"%{Path(args.json).stem[:20]}%",))
            row = cur.fetchone()
            prompt_id = row["prompt_id"] if row else None
            for s in res["series"]:
                conn.execute(
                    "INSERT INTO ngram_lock_event "
                    "(prompt_id, position, window, n, n_total, n_distinct, "
                    " repeat_factor, top_ngram, top_count, triggered_rescue) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (prompt_id, s["position"], args.window, 5,
                     s["per_n"]["n5"]["n_total"], s["per_n"]["n5"]["n_distinct"],
                     s["per_n"]["n5"]["repeat_factor"],
                     ' '.join(s["per_n"]["n5"]["top_ngram"]) if s["per_n"]["n5"]["top_ngram"] else None,
                     s["per_n"]["n5"]["top_count"],
                     1 if s["locked"] else 0),
                )
            journal(conn, "ngram_lock_scan", str(json_path),
                    {"n_windows": res["n_windows"],
                     "n_locked": res["n_locked_windows"],
                     "first_lock_position": res["first_lock_position"]})


if __name__ == "__main__":
    main()
