"""Cross-precision × cross-problem HC_LOCK validation.

Tests the n-gram lock detector on 30 cells (4bit + 8bit + bf16 × P01-P10)
to severe-test the threshold generalization.

Truth table:
  correct=True  AND HC_LOCK=False → TN (success not flagged)
  correct=False AND HC_LOCK=True  → TP (cyclic-failure flagged)
  correct=True  AND HC_LOCK=True  → FP (success spuriously flagged)
  correct=False AND HC_LOCK=False → could be FN (cyclic-failure missed)
                                  OR explore-without-commit (correct TN)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from codec_audit.ngram_lock_detector import analyze_response


def scan_quant_matrix(root: Path = Path("/Users/silv/cl/tlp/montyneg/tmp/20260524_quant_matrix")):
    """Scan all cells across all precisions."""
    rows = []
    for quant in ["4bit", "8bit", "bf16"]:
        qdir = root / quant
        if not qdir.exists():
            continue
        for f in sorted(qdir.glob("p*.json")):
            if "truth_hits" in f.name or "results" in f.name:
                continue
            try:
                data = json.loads(f.read_text())
                if "response" not in data:
                    continue
                # Skip degenerate (very short or empty)
                if len(data["response"]) < 1000:
                    continue
                res = analyze_response(f, window=500, step=250, lock_threshold=3.0)
                rows.append({
                    "quant": quant,
                    "problem": f.stem.split("_")[0],
                    "correct": data.get("correct"),
                    "extracted": data.get("extracted"),
                    "truth": data.get("truth"),
                    "n_chars": res["n_chars"],
                    "n_words": res["n_word_tokens"],
                    "n_windows": res["n_windows"],
                    "n_distinct_top5": res["n_distinct_top5grams"],
                    "distinct_rate": res["distinct_rate"],
                    "most_pers_frac": res["most_persistent_fraction"],
                    "hc_lock": res["high_confidence_lock"],
                })
            except Exception as e:
                rows.append({"quant": quant, "problem": f.stem.split("_")[0],
                              "error": str(e)})
    return rows


def main():
    rows = scan_quant_matrix()
    # Cross-tabulate
    print(f"\n=== HC_LOCK cross-validation: 30 cells (Qwen3.5-4B × 3 precisions × 10 problems) ===\n")
    print(f"{'quant':<6} {'prob':<5} {'correct':<8} {'words':>6} "
          f"{'distinct':>9} {'rate':>6} {'pers':>6} {'HC':<8} {'classification':<20}")
    print("-" * 90)
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN_candidate": 0, "error": 0}
    for r in sorted(rows, key=lambda x: (x.get("quant", ""), x.get("problem", ""))):
        if "error" in r:
            print(f"{r['quant']:<6} {r['problem']:<5} ERROR: {r['error']}")
            counts["error"] += 1
            continue
        correct = r["correct"]
        hc = r["hc_lock"]
        if correct and not hc:        cls = "TN (success)"; counts["TN"] += 1
        elif not correct and hc:      cls = "TP (cyclic-fail)"; counts["TP"] += 1
        elif correct and hc:          cls = "FP (false alarm)"; counts["FP"] += 1
        else:                          cls = "explore-no-commit"; counts["FN_candidate"] += 1
        print(f"{r['quant']:<6} {r['problem']:<5} {str(correct):<8} "
              f"{r['n_words']:>6} {r['n_distinct_top5']:>9} "
              f"{r['distinct_rate']:>6.3f} {r['most_pers_frac']:>6.3f} "
              f"{str(hc):<8} {cls:<20}")
    print(f"\n=== Summary ===")
    total = sum(counts.values())
    print(f"  total: {total}")
    for k, v in counts.items():
        print(f"  {k:<15} {v:>3}  ({100*v/max(total,1):.1f}%)")
    # FP rate matters: any FP would refute the detector
    fp_rate = counts["FP"] / max(counts["TN"] + counts["FP"], 1)
    print(f"\n  FP rate (false alarm on success cells): {fp_rate:.4f}")
    print(f"  TP rate (catch among non-success): "
           f"{counts['TP'] / max(counts['TP'] + counts['FN_candidate'], 1):.4f}")
    print(f"  (note: FN_candidate may be explore-without-commit, not detector miss)")


if __name__ == "__main__":
    main()
