# 31st continue: all-10 rescue achieves matharena 8/10 ceiling — TODAY

silv 2026-05-25 31st continue. Ran aime_rescue.py end-to-end on all
10 cached AIME 2026 cells. Result matches matharena's documented
production ceiling exactly.

## Results

| cell | truth | answer | kind | wall | correct |
|------|-------|--------|------|------|---------|
| P01 | 277 | 277 | rescued | 12.9s | ✓ |
| P02 | 62 | 62 | rescued | 21.1s | ✓ |
| P03 | 79 | 79 | rescued | 26.9s | ✓ |
| P04 | 70 | 70 | rescued | 20.0s | ✓ |
| P05 | 65 | 65 | committed | 0.0s | ✓ (natural) |
| P06 | 441 | 441 | rescued | 29.4s | ✓ |
| P07 | 396 | 396 | rescued | 30.8s | ✓ |
| P08 | 244 | 244 | rescued | 25.5s | ✓ |
| P09 | 29 | None | no_extract | 36.1s | ✗ (truth not in CoT) |
| P10 | 156 | 100 | rescued | 13.9s | ✗ (Mode 2 prior) |

**FINAL: 8/10** — matches matharena Qwen3.5-4B ceiling exactly.

## Cumulative wall time

7 rescue calls + 1 no_extract scan + 1 Mode-2-fail = 9 forced-commit
operations. Total wall: ~217s = 3.6 minutes for the entire AIME 2026
I corpus rescue.

## What this validates

The end-to-end deployable rescue protocol:
1. extract_committed first (catches P05's natural \boxed{65})
2. find_rescue_position + rescue_no_commit for no-commits
3. Returns None for capability-limit cells (P09)
4. Returns Mode-2 prior for ambiguous-frame cells (P10)

Without external verifier, P10's '100' would be a silent-wrong. With
solve_with_rescue_and_verify (sympy verifier), P10 returns None
correctly.

## Class taxonomy validated on fresh data

- Class A (rescuable via cap-mechanism): P01-P08, all 8 cells correct
- Class B (no CoT truth, no rescue): P09 — truth never derived
- Class C (ambiguous frame, Mode 2 prior fires): P10 — emits '100' confidently

All three classes empirically validated on fresh model run today.

## The session's net deliverable

After 31 continues and 4 live experiments, the session has produced:
- 22 cached-data refutations + corroborations (validated rule)
- 4 live experiments (budget threshold + heuristic-miss + all-10)
- 1 cross-arc step-out (codex)
- 40+ memos
- README consolidation
- THIS: empirical confirmation 8/10 matharena ceiling holds today

The deployed system works. The documented behavior matches today's
runs. The investigation arc is structurally complete.
