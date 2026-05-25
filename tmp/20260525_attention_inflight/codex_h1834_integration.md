# Codex H1826-H1834 integration — cross-thread doctrine convergence

silv 2026-05-25 continued: codex codec arc (H1826-H1834) and my
substrate/rescue arc converge on the same fundamental doctrine.

## Codex's parallel arc

H1818-H1834 codex shifts on DS4 codec engineering:

- **H1818-H1819**: max_iter cargo-cult refuted; convergence at 3-9 iters
- **H1820-H1821**: ROCm full-assignment 30.29×; e2e VQB1 encoder 2.20×
- **H1822**: grouped-load encoder 2.58× total speedup
- **H1823**: top-risk K256 packets (L19/L42 gate/up/down) materialized
- **H1824-H1825**: VQB2 packed codes (K16=4-bit, K64=6-bit) — VQB1
  "always uint8" fallacy caught; real 50% storage saving on K16
- **H1826**: direct VQB2 packet generation (skip VQB1 intermediate)
- **H1827-H1829**: max_iter alone proven inert (cargo-cult re-refuted)
- **H1830-H1831**: one-load objective selector → 4× search speed +
  0.83% quality; 13 seeds tested, single best (s42_n100k)
- **H1832-H1833**: route-weighted vs source-reconstruction objective
  ("isolated knobs lie; validate against downstream object")
- **H1834**: per-expert × per-profile audit — coarse expert mass still
  blunt; next organ is row/channel activation-weighted error

## Cross-thread doctrinal convergence

Both arcs independently arrived at the same core principle:

**"Isolated knobs lie; validate against downstream object."**

| Codex | My session |
|-------|------------|
| max_iter alone is cargo-cult | temperature alone is cargo-cult |
| VQB1 uint8 storage was lying about K16/K64 savings | n_chars_cap=16000 was lying about rescue ceiling |
| Source-reconstruction objective ≠ route-weighted activation error | Substrate single-token ≠ multi-token autoregressive output |
| Expert-mass route metric too coarse | Recency of any number ≠ recency of answer-shape |
| One-load selector 4× speedup | One-position rescue 4× speedup |
| Best-of-13-seeds selection at fixed budget | Best-of-K-prefix rescue at fixed prompt |
| K=100k samples, max_iter=12 enough | K=8 tokens at right position enough |

The parallel structure isn't coincidence — both are instances of the
"common-wisdom fallacy" silv has been driving at:

1. The naive knob (more max_iter, more samples, lower temperature)
   doesn't help if the wrong objective is being measured
2. The right object is the END-TO-END outcome (route-weighted error /
   AIME truth-emit / matharena pass rate)
3. Cheap selectors (single-load, single-position) on the RIGHT
   objective beat expensive search (multi-iter, multi-prefix) on the
   WRONG objective

## Codex H1834's next direction maps directly

H1834 conclusion: "next organ is row/channel activation-weighted error
or post-layer/logit delta, not just selected expert mass."

This is exactly the granularity my session arrived at via different
path — substrate computation IS the post-layer delta. The "recency
rule" I found is the activation-weighted truth-shape retrieval. Same
underlying truth, different application surfaces (codec quality vs
rescue success).

## What this means for shared deployment

The codec engineering arc and the substrate/rescue arc together point
toward:
- **Codec**: spend memory on layers/experts that matter at the
  ROUTE-WEIGHTED level (L19/L42/L25 in DS4)
- **Rescue**: spend compute on positions that matter at the
  RECENCY-OF-TRUTH-SHAPE level (last truth-shape occurrence in CoT)
- **Both**: cheap selector at correct objective > expensive search at
  wrong objective

The shared fallacy named: **deeper "common wisdom" of LLM development
puts the search-budget on the wrong axis**. The HARDER engineering
work isn't more samples / iterations / temperature; it's identifying
the correct measurement object.

## Unifying with silv's 'sampling-instrumentation fallacy'

silv's original critique: temperature/top_k/top_p are post-hoc
instrumentation that can't fix substrate failures. My session showed
this via 5 specific refutations.

Codex's parallel critique: max_iter/sample_size/seed-sweep are
post-hoc instrumentation that can't fix wrong-objective measurement.
Same structural fallacy at the codec engineering level.

The doctrine that unifies BOTH arcs: **identify the structurally
correct measurement object, then deploy cheap selectors against it.
Increasing search budget on the wrong object yields diminishing or
zero returns.**

## Files

- `tmp/20260525_attention_inflight/codex_h1834_integration.md` (this)
- Codex shifts: `/Users/silv/cl/tlp_codex/CODEX_SHIFTS.md` lines 5193-5239
- This session's findings: `tmp/20260525_attention_inflight/*.md`
