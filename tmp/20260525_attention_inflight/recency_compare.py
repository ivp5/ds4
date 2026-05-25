"""Compare predicted vs actual cap=24000 results — validate recency rule."""
import json, re, os, sys
from pathlib import Path

# Predicted per-cap emissions
PREDICTED = {
    "p01": "277", "p02": "62", "p03": "79", "p04": "70",
    "p05": "65", "p06": "441", "p07": "396", "p08": "244",
    "p09": "?",  "p10": "?",
}

def main():
    # Load actual cap=24000 results
    path = Path("tmp/20260525_attention_inflight/forced_extract_4bit_cap24k.json")
    if not path.exists():
        print(f"NOT YET — {path} doesn't exist")
        return
    d = json.load(path.open())
    print(f"cap=24000 results: {d['n_correct']}/{d['n_total']} correct\n")
    print(f"{'cell':>5} {'truth':>5} {'predicted':>10} {'actual':>10} {'recency_match':>14}")
    for r in d['results']:
        cell = r['cell']
        truth = r['truth']
        pred_str = PREDICTED.get(cell, "?")
        actual = r['predicted']
        actual_str = str(actual) if actual else 'None'
        match = (str(truth) == actual_str)
        recency_ok = (pred_str == actual_str) or (pred_str == "?")
        print(f"{cell:>5} {truth:>5} {pred_str:>10} {actual_str:>10} {'YES' if recency_ok else 'NO':>14}"
              + ('  ✓' if match else '  ✗'))


if __name__ == "__main__":
    main()
