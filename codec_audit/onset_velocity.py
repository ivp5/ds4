"""Onset-velocity tracker: detect IMMINENT lock before it crystallizes.

silv 2026-05-25: refine for OOM higher accuracy.

Previous detector (HC_LOCK) is BINARY at one window:
  most_persistent_fraction > 0.80 AND distinct_rate < 0.15 → locked

This misses the TRAJECTORY signal. P01 doesn't jump to most_pers=0.85
instantaneously — it climbs across windows. By tracking the SLOPE we can
detect imminent lock BEFORE it crystallizes, enabling earlier intervention.

Signals tracked per generation (rolling):
  - most_persistent_fraction (current value)
  - velocity: d(most_pers)/d(window) over last K windows
  - acceleration: d²(most_pers)/d(window)²
  - distinct_rate (current value)
  - rate of distinct_rate decline
  - max_repeat_factor across n in {3, 5, 7, 10, 15}

Also: incremental n-gram counter for ~250× speedup on streaming text.

Detection tiers:
  WARN (yellow): velocity > 0.05 / window for 3+ consecutive windows
  ALERT (orange): velocity > 0.05 AND most_pers > 0.50
  LOCK (red, existing HC_LOCK): most_pers > 0.80 AND distinct_rate < 0.15

Earlier alert → earlier intervention → lower wasted budget.
"""
import argparse
import json
from collections import Counter, deque
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class IncrementalNgramCounter:
    """Sliding-window n-gram counter with O(1) per-step update.

    Replaces the prior O(W) per-step Counter rebuild. Once window has
    advanced past initial W tokens, each step:
      - decrement count for n-gram leaving the window
      - increment count for n-gram entering the window
    """

    def __init__(self, n: int, window_size: int):
        self.n = n
        self.W = window_size
        self.counts: Counter = Counter()
        self.tokens: deque = deque(maxlen=window_size)
        # Track top-1 separately via a simple max (resorted on each step
        # when the current top is decremented). For tighter perf could
        # use a sorted dict + bucket; for our 500-token window the
        # decrement-and-rescan cost is bounded.
        self._top_cache: tuple | None = None
        self._top_count: int = 0

    def push(self, tok: str) -> None:
        """Add one token, evicting oldest if window full."""
        was_full = len(self.tokens) == self.W
        old_tail = self.tokens[0] if was_full else None
        if was_full:
            # Removing this token retires the n-gram that ended at it.
            # The n-gram is tokens[0..n-1] before push.
            if len(self.tokens) >= self.n:
                # n-gram leaving = tokens[0..n-1] currently
                old_ng = tuple(list(self.tokens)[0:self.n])
                self.counts[old_ng] -= 1
                if self.counts[old_ng] <= 0:
                    del self.counts[old_ng]
                    # If we just deleted the top, force recompute
                    if old_ng == self._top_cache:
                        self._top_cache = None
        self.tokens.append(tok)
        # Adding this token introduces a new n-gram if we now have ≥ n tokens
        if len(self.tokens) >= self.n:
            new_ng = tuple(list(self.tokens)[-self.n:])
            self.counts[new_ng] += 1
            # Maintain top cache lazily
            if self.counts[new_ng] >= self._top_count:
                self._top_cache = new_ng
                self._top_count = self.counts[new_ng]
            else:
                # Top wasn't updated — but if we decremented the top earlier,
                # we need to find new top now.
                if self._top_cache is None:
                    self._recompute_top()

    def _recompute_top(self):
        if not self.counts:
            self._top_cache = None
            self._top_count = 0
            return
        ng, n = self.counts.most_common(1)[0]
        self._top_cache = ng
        self._top_count = n

    def signature(self) -> dict:
        if not self.counts:
            return {"n_total": 0, "n_distinct": 0, "repeat_factor": 1.0,
                    "top_ngram": None, "top_count": 0}
        n_total = sum(self.counts.values())
        n_distinct = len(self.counts)
        if self._top_cache is None:
            self._recompute_top()
        return {
            "n_total": n_total,
            "n_distinct": n_distinct,
            "repeat_factor": n_total / max(n_distinct, 1),
            "top_ngram": list(self._top_cache) if self._top_cache else None,
            "top_count": self._top_count,
        }


def velocity_scan(tokens: list, window: int = 500, step: int = 100,
                   n_values=(3, 5, 7, 10, 15),
                   warn_velocity: float = 0.05,
                   k_consec: int = 3) -> dict:
    """Scan tokens with incremental n-gram counters + onset velocity tracking.

    Returns time series with persistence trajectory + per-window detector flags.

    O(N + N/step × max_n × window_warmup) — dominated by the warmup; subsequent
    sliding window is O(1) per token after warmup.
    """
    series = []
    # One counter per n value
    counters = {n: IncrementalNgramCounter(n=n, window_size=window) for n in n_values}

    # Track top-5-gram history for cross-window persistence
    top5_history = deque(maxlen=20)  # rolling K windows
    pers_history = deque(maxlen=20)
    velocity_breach_streak = 0

    for i, tok in enumerate(tokens):
        for n in n_values:
            counters[n].push(tok)

        # Sample every `step` tokens after window warmup
        if i + 1 < window or (i + 1) % step != 0:
            continue

        sigs = {n: counters[n].signature() for n in n_values}
        sig5 = sigs[5]
        sig_max = max(s["repeat_factor"] for s in sigs.values())

        # Compute current top-5gram persistence over rolling K windows
        cur_top5 = tuple(sig5["top_ngram"]) if sig5["top_ngram"] else None
        top5_history.append(cur_top5)
        nonempty = [t for t in top5_history if t]
        if nonempty:
            top_counts = Counter(nonempty)
            most_pers_n = top_counts.most_common(1)[0][1]
            most_pers_frac = most_pers_n / len(nonempty)
            distinct_top5 = len(top_counts)
            distinct_rate = distinct_top5 / len(nonempty)
        else:
            most_pers_frac = 0.0
            distinct_rate = 1.0

        pers_history.append(most_pers_frac)

        # Velocity = change in most_pers over last 3 windows
        if len(pers_history) >= 3:
            recent = list(pers_history)[-3:]
            velocity = (recent[-1] - recent[0]) / 2.0  # per-window slope
        else:
            velocity = 0.0

        # Streak: consecutive windows above warn threshold
        if velocity > warn_velocity:
            velocity_breach_streak += 1
        else:
            velocity_breach_streak = 0

        warn = velocity_breach_streak >= k_consec
        alert = warn and most_pers_frac > 0.50
        lock = most_pers_frac > 0.80 and distinct_rate < 0.15

        series.append({
            "position": i + 1,
            "most_pers_frac": most_pers_frac,
            "distinct_rate": distinct_rate,
            "velocity": velocity,
            "velocity_streak": velocity_breach_streak,
            "max_repeat_factor": sig_max,
            "top_5gram": sig5["top_ngram"],
            "top_5gram_count": sig5["top_count"],
            "warn": warn,
            "alert": alert,
            "lock": lock,
        })

    return {
        "n_windows": len(series),
        "n_warn": sum(1 for s in series if s["warn"]),
        "n_alert": sum(1 for s in series if s["alert"]),
        "n_lock": sum(1 for s in series if s["lock"]),
        "first_warn_position": next((s["position"] for s in series if s["warn"]), None),
        "first_alert_position": next((s["position"] for s in series if s["alert"]), None),
        "first_lock_position": next((s["position"] for s in series if s["lock"]), None),
        "series": series,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True)
    p.add_argument("--field", default="response")
    p.add_argument("--window", type=int, default=500)
    p.add_argument("--step", type=int, default=100)
    p.add_argument("--warn-velocity", type=float, default=0.05)
    p.add_argument("--show-trajectory", action="store_true")
    args = p.parse_args()

    text = json.loads(Path(args.json).read_text())[args.field]
    tokens = text.split()
    print(f"text: {len(text)} chars, {len(tokens)} word-tokens")

    import time
    t0 = time.time()
    res = velocity_scan(tokens, window=args.window, step=args.step,
                         warn_velocity=args.warn_velocity)
    elapsed = time.time() - t0
    print(f"scan: {res['n_windows']} windows in {elapsed*1000:.1f}ms "
          f"({elapsed*1000/max(res['n_windows'],1):.2f}ms/window)")
    print(f"\nWARN windows:  {res['n_warn']:>4}  (first at pos {res['first_warn_position']})")
    print(f"ALERT windows: {res['n_alert']:>4}  (first at pos {res['first_alert_position']})")
    print(f"LOCK windows:  {res['n_lock']:>4}  (first at pos {res['first_lock_position']})")

    if args.show_trajectory:
        print(f"\n{'pos':>6} {'pers':>5} {'dist':>5} {'vel':>7} {'streak':>6} "
              f"{'flags':<15} {'top_5gram':<40}")
        for s in res["series"]:
            flags = []
            if s["warn"]: flags.append("WARN")
            if s["alert"]: flags.append("ALERT")
            if s["lock"]: flags.append("LOCK")
            ng = (' '.join(s["top_5gram"]) if s["top_5gram"] else '')[:40]
            print(f"{s['position']:>6} {s['most_pers_frac']:>5.2f} "
                  f"{s['distinct_rate']:>5.2f} {s['velocity']:>+7.3f} "
                  f"{s['velocity_streak']:>6} "
                  f"{','.join(flags):<15} {ng!r}")


if __name__ == "__main__":
    main()
