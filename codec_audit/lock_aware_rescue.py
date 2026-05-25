"""HC_LOCK-aware rescue wrapper.

Integrates the n-gram cache-lock detector with the existing
inferguard/aime_rescue.py forced-commit protocol.

Decision tree:
  1. If response already has a `\\boxed{N}` commit → return N (existing).
  2. If HC_LOCK fires (cyclic-failure detected):
     a. Truncate response BEFORE the locked region (find first_lock_position
        from n-gram detector, back off by W tokens).
     b. Call existing rescue_no_commit on the truncated text.
     c. Record in codec_audit DB: ngram_lock_event with triggered_rescue=1
  3. Else (no lock detected):
     a. Call existing rescue_no_commit directly.
     b. Record in DB as "rescue_no_lock".

This ADDS a path that handles the cyclic-failure mode specifically. The
existing rescue is preserved for explore-without-commit cases.
"""
import json
import sqlite3
import sys
import time
from pathlib import Path

# Make both inferguard and codec_audit importable
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path("/Users/silv/cl/tlp/montyneg/inferguard")))

from codec_audit.ngram_lock_detector import analyze_response, _ensure_ngram_schema
from codec_audit.db import open_db, journal


def lock_aware_rescue_text_analysis(response: str, truth_hint=None,
                                      window: int = 500, step: int = 250,
                                      lock_threshold: float = 3.0) -> dict:
    """Pure-text analysis (no model). Returns dict with:
      - 'hc_lock': bool
      - 'first_lock_position' (word-position): int or None
      - 'recommend_truncate_chars': int or None (char offset to truncate at)
      - 'rescue_strategy': 'commit_exists' | 'lock_truncate' | 'no_lock_rescue' | 'no_action'
    """
    # 1. Already committed?
    import re
    boxed = re.findall(r'\\boxed\{(\d+)\}', response)
    if boxed:
        return {
            "hc_lock": False,
            "rescue_strategy": "commit_exists",
            "extracted_answer": int(boxed[-1]),
            "first_lock_position": None,
            "recommend_truncate_chars": None,
        }

    # 2. Run HC_LOCK detector via the existing module (use a tmp file shim)
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"response": response}, f)
        tmp_path = Path(f.name)
    try:
        res = analyze_response(tmp_path, field="response",
                                window=window, step=step,
                                lock_threshold=lock_threshold)
    finally:
        tmp_path.unlink(missing_ok=True)

    if res["high_confidence_lock"]:
        # Find character offset of first locked word position
        # word-tokens are whitespace-split; map back to char offset
        words = response.split()
        # res["first_lock_position"] is a word position
        wpos = res["first_lock_position"]
        if wpos is not None and wpos > window:
            # back off by window // 2 to truncate before the lock starts
            target_word = max(0, wpos - window)
            # Recompose to find char position
            chars = 0
            for i, w in enumerate(words):
                if i >= target_word:
                    break
                chars += len(w) + 1
            return {
                "hc_lock": True,
                "rescue_strategy": "lock_truncate",
                "first_lock_position": wpos,
                "recommend_truncate_chars": chars,
                "most_persistent_fraction": res["most_persistent_fraction"],
                "distinct_rate": res["distinct_rate"],
                "top_ngram": res.get("most_persistent_ngram"),
            }

    return {
        "hc_lock": False,
        "rescue_strategy": "no_lock_rescue",
        "first_lock_position": None,
        "recommend_truncate_chars": None,
        "most_persistent_fraction": res["most_persistent_fraction"],
        "distinct_rate": res["distinct_rate"],
    }


def record_decision(conn: sqlite3.Connection, response_sha: str,
                     decision: dict, model_id: str = ""):
    """Append the rescue decision to the codec_audit journal."""
    journal(conn, "lock_aware_rescue", response_sha,
            {"strategy": decision["rescue_strategy"],
             "hc_lock": decision["hc_lock"],
             "first_lock_position": decision.get("first_lock_position"),
             "recommend_truncate_chars": decision.get("recommend_truncate_chars"),
             "most_persistent_fraction": decision.get("most_persistent_fraction"),
             "distinct_rate": decision.get("distinct_rate"),
             "top_ngram": decision.get("top_ngram"),
             "model_id": model_id})


def analyze_cached_cell(json_path: Path, save_to_db: bool = True) -> dict:
    """Analyze a cached AIME response file. Pure text, no model forward."""
    data = json.loads(Path(json_path).read_text())
    response = data["response"]
    correct_truth = data.get("correct")
    decision = lock_aware_rescue_text_analysis(response)

    if save_to_db:
        import hashlib
        sha = hashlib.sha256(response.encode()).hexdigest()
        with open_db() as conn:
            _ensure_ngram_schema(conn)
            record_decision(conn, sha, decision,
                             model_id=data.get("model", ""))

    return {
        "cell": json_path.stem,
        "correct": correct_truth,
        "decision": decision,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scan-quant-matrix", action="store_true",
                    help="Scan all 30 cells in tmp/20260524_quant_matrix")
    p.add_argument("--json", type=str, help="Single response JSON to analyze")
    p.add_argument("--no-save", action="store_true",
                    help="Skip writing to codec_audit DB")
    args = p.parse_args()

    if args.scan_quant_matrix:
        root = Path("/Users/silv/cl/tlp/montyneg/tmp/20260524_quant_matrix")
        files = []
        for q in ["4bit", "8bit", "bf16"]:
            for f in sorted((root / q).glob("p*.json")):
                if "truth_hits" in f.name or "results" in f.name:
                    continue
                files.append((q, f))
        print(f"\n=== HC_LOCK-aware rescue decisions for {len(files)} cells ===\n")
        print(f"{'quant':<6} {'cell':<28} {'correct':<8} {'strategy':<18} {'pos':>6}")
        print("-" * 78)
        for q, f in files:
            try:
                r = analyze_cached_cell(f, save_to_db=not args.no_save)
                d = r["decision"]
                cell = r["cell"]
                cor = str(r["correct"])
                strat = d["rescue_strategy"]
                pos = d.get("first_lock_position", "-") or "-"
                print(f"{q:<6} {cell:<28} {cor:<8} {strat:<18} {pos!s:>6}")
            except Exception as e:
                print(f"{q:<6} {f.stem:<28} ERROR: {e}")
    elif args.json:
        r = analyze_cached_cell(Path(args.json), save_to_db=not args.no_save)
        print(json.dumps(r, indent=2))
    else:
        p.print_help()


if __name__ == "__main__":
    main()
