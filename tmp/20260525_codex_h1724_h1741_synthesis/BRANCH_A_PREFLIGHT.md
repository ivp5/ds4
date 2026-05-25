# Branch A pre-flight: dependency check before polar hot-path lands

silv chose Branch A: ship polar B-2.3c hot-path gate + runtime test.
Before writing ~300 LOC of dispatcher code, surface a critical
dependency the scope-caveat audit revealed.

## The dependency

The polar p32_m8 production corpus encodes **128 rows per expert**:
- gate: 128 of 2048 rows (6.25% coverage)
- up:   128 of 2048 rows (6.25% coverage)
- down: 128 of 4096 rows (3.13% coverage)

The H1735 kernel computes `out[batch, route_pair, tid]` for
`tid < down_rows`. With down_rows = 128 max, only the first 128 of
4096 hidden_size outputs are produced.

For real FFN inference at the hot path call site, the model
expects ALL hidden_size = 4096 down outputs. If the hot-path dispatch
calls polar with the current 128-row corpus, the output is INCOMPLETE
(first 128 polar-computed, remaining 3968 would be uninitialized
buffer memory).

This isn't fixable in the dispatcher code — it's a corpus-coverage
issue. Three resolutions:

## Option A.1: full-row corpus first (recommended)

Re-encode polar p32_m8 at full rows. Storage cost:
- 275 GB at full rows for polar p32_m8
- Exceeds CLAUDE.md 100 GB disk budget — needs explicit silv approval

Encoding time: ~37 min via MLX encoder (linear scaling from 128-row
44.7s × 32×). Pure GPU compute; no kernel changes needed.

Then the hot-path dispatcher (~300 LOC) consumes the full-row corpus
and produces complete FFN outputs.

Steps:
1. silv approves disk budget: "OK to use ~275 GB for polar_full_rows/"
2. Encode: `polar_encode_mlx.py --layers all --kinds gate,up,down \
   --phase-levels 32 --mag-levels 8 --rows-per-tile 2048 --n-tiles 1 \
   --out-dir tmp/polar_full_p32m8_full_rows`
   (37 min wall, encoder unchanged)
3. Validate: cross-cell GPU canary on the new corpus (5 min)
4. Ship hot-path dispatcher + env-var gate (~300 LOC, this is the
   actual Branch A delivery)
5. silv runtime-tests via DS4_POLAR_FFN_LAYERS env var with AIME
   hold-rate measurement

## Option A.2: VQ K=256 instead (better tradeoff)

VQ-2D K=256 has 6× lower codec error AND 2× smaller storage.

Storage cost:
- 138 GB at full rows for VQ K=256
- Still exceeds 100 GB budget — needs silv approval
- But ~50% smaller than polar A.1 option

Engineering cost: VQ canary chain is already shipped (commit 917df69).
The hot-path dispatcher mirroring would be ~300 LOC for either codec
— roughly equal effort.

The MLX-parallel VQ encoder runs at 13s/tile vs 17s/tile CPU; full
corpus at full rows would be ~129 × 13 × 32 = ~14 hours. Slow vs
polar (37 min) because the polar encoder MLX-port is more mature.

Recommendation if going VQ: first ship the polar encoder's MLX
load-step port to VQ (~150 LOC) → full corpus drops to ~6 min →
then encode → then ship hot-path dispatcher.

## Option A.3: hybrid polar + FP4 dispatch

Keep 128-row corpus. Dispatcher computes:
- Polar: hidden_size outputs 0..127
- FP4 path: hidden_size outputs 128..4095

Storage cost: 14 GB (current corpus, no change).
Engineering cost: ~500 LOC (two dispatch paths + output combining).

This is the smallest-storage option but most complex code. Quality
impact: 3% of FFN compute uses polar (faster?), 97% uses FP4
(slower but correct). Net speed unclear. Hold-rate impact minimal.

## My recommendation

**A.1 (full-row polar) is the cleanest Branch A execution.**

Reasoning:
- 275 GB is a one-time cost; the corpus can be moved to ~/.Trash
  after AIME hold-rate completes
- Disk budget approval is a single yes/no question for silv
- Engineering effort is the lowest of the three options (no hybrid
  combine code, no new encoder port)
- Validates the polar p32_m8 codec at its actual deployment scope
- Falsifies cleanly: AIME hold-rate either matches FP4 baseline
  within X% or doesn't

If silv prefers A.2 or A.3, that's their call. A.3 specifically is
the path if disk budget can't be approved.

## What I need from silv before writing dispatcher code

Pick one:
1. **A.1 approved**: I encode full-row polar corpus (37 min, 275 GB),
   then write the dispatcher (~300 LOC) + hot-path gate, ship for
   runtime test.

2. **A.2 approved**: I MLX-port the VQ encoder load step (~150 LOC),
   encode full-row VQ corpus (~6 min, 138 GB), then write VQ
   dispatcher (~300 LOC) + hot-path gate.

3. **A.3 approved**: I write the hybrid dispatcher (~500 LOC) +
   hot-path gate. Larger codebase, same disk footprint as today.

4. **Defer**: I work on a non-codec task while silv considers the
   disk-budget question.

Without one of these, writing the dispatcher would either produce
incorrect inference (if I assumed 128-row coverage was sufficient)
or block-on-silv-runtime-test for an obvious-failure case.

## What I'm doing NOW

Surfacing this dependency in a memo so silv has clear visibility
before any dispatcher code lands. I won't write ~300 LOC of code
that won't actually work without one of the above decisions.

This is the responsible "back-of-hand discipline + ground-decode"
move I should have done before committing to any of the deployment
narratives — the row-coverage caveat memo (commit a1ca099) flagged
the storage concern; this memo turns it into a forced decision.

Commit: this memo + brief note on the task list.

## ADDENDUM 2026-05-25 evening — H1682 Amdahl correction (sharpens OOM framing)

silv's prior architecture-review session (03:00 UTC, before my codec
arc started) named the H1682 Amdahl correction:

> "2-3 OOM speedup CANNOT come from swapping the matmul executor
> alone. MTL4 ML is the dense-organ lever, not the organism. Path
> forward requires graph deletion (fewer tokens via MTP/spec-decode,
> fewer layers/expert calls via skip, fewer host round trips)."

This sharpens the session's OOM scorecard honestly:

| Component | OOM | Type |
|-----------|-----|------|
| Encoder MLX speedup | 1.86 | ONE-TIME tooling (not inference) |
| Codec accuracy (VQ K=256) | 1.10 | QUALITY at substrate (not speed) |
| "Combined 2.96 OOMs" | — | misleading aggregate per H1682 |

Neither component delivers H1682's "2-3 OOM inference speedup" on
its own:
- Encoder speedup is **one-time tooling** (run once per corpus)
- Codec accuracy is **quality**, not speed (substitution at same
  compute cost gives same speed)
- Inference speed gains from the codec would only come INDIRECTLY
  via lower memory pressure enabling other wins (chunk-streaming,
  cache fitting, layer skip viability, etc.)

The codex H1773 chunk512 streaming gives 8.61× resident memory
reduction at 1.173× slowdown — that's the actual memory-axis win
that PAIRS with my codec quality. My codec contributes to the
memory frontier (4.3 GB VQ vs 14 GB polar p32_m8 vs 85 GB FP4)
which enables H1773's streaming patterns to fit more in residency.

## Sharpened honest claim

My session delivered:
- **Codec QUALITY improvement** (1.10 OOMs, VQ K=256)
- **Encoder TOOLING speedup** (1.86 OOMs, MLX-parallel encoder)

Both as enablers for the H1682-compliant deletion-based 2-3 OOM
inference frontier — NOT as standalone inference-speed contributors.

The complete deployment frontier (per silv's H1682 + codex H1773 +
my codec arc) is:

```
Layer                    Owner          Contribution
-------------------      -----------    -----------------------
Graph deletion           silv / codex   2-3 OOM inference speedup
  MTP/spec-decode                        (the H1682-compliant lever)
  Layer/expert skip                      
  Host round-trip cut                    
Memory streaming         codex H1773    7-8× resident reduction
  Chunk512 hidden                        (enables fitting under
  Bitpacked phase                         resource caps)
Substrate codec          my session     6× lower codec error +
  VQ K=256 (recommended)                 50% smaller weights
  Polar p32_m8 (deployable)              (enables tighter memory
Encoder tooling          my session      profile)
  MLX-parallel                          1.86 OOM faster corpus
                                         rebuild (one-time)
Engineering gate         my session     B-2.3c stub wired
                                        (body needs silv decision)
Rescue layer             silv           AIME hold-rate restoration
  inferguard.aime_rescue                 (codec near-miss catches)
```

Each layer is necessary but not sufficient. silv's full 2-3 OOM
deployment requires combining at least the graph-deletion lever
(H1682) with the memory streaming (H1773) and the codec quality
(my session). The codec body shipping is a prerequisite for
combining at the resident-memory layer.

silv decision remains: A.1/A.2/A.3 for codec body. The deeper
strategic frame is now H1682-aware: codec deployment is necessary
substrate for the inference-speedup frontier, not the speedup itself.
