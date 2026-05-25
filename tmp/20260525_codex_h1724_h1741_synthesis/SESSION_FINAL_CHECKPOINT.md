# Session final checkpoint — polar codec substrate complete, Phase B-2.3 architectural questions pending

silv direction: 12 "continue" prompts pushed me to 31 commits, ~6K lines
net. This memo consolidates the session arc + flags the architectural
questions Phase B-2.3 needs before more code lands.

## The substrate that shipped this session (commits in arc order)

```
ENCODER
  4a529d0  Bulk numpy polar encoder (50× over per-script)
  8402a5e  MLX-GPU polar encoder (73× over bulk = 1.86 OOM; full DS4 V4 in 8 sec)
  278db99  Parameterized for (P, M); legacy p8_m4 + new p16_m4 + arbitrary

FORMAT
  aad0608  PLR2 file format spec + C reader at 1.39e-7 round-trip
  278db99  Header bytes 28-35 carry (P, M); legacy compat via 0 → defaults

POOL + ENGINE
  bf1eee4  ds4_polar_pool: mmap'd PLR2 directory with (layer, kind) lookup
  4056ed2  Engine carries pool via DS4_POLAR_DIR env var
  cbf3b1c  DS4_POLAR_LAYERS env var + per-layer mask
  cac5018  FFN-entry diagnostic prints "polar layer N armed" when gated

GPU KERNEL + CANARY
  4b59348  H1735 down-projection kernel canary, max_abs_err=0 (synthetic)
  d74a7d5  Real-data H1735 canary @ 1.07e-6 on real DS4 V4 weights
  278db99  Real-data canary parameterized for any (P, M)

COMPLETE CORPUS
  b22698d  Full-model p16_m4 corpus: 14 GB, 33,024 encodings, 42 sec rebuild
  4d16241  Polar-DOWN decode cross-expert validated: rel_L2 0.2018 ± 0.001

CODEX SYNTHESIS (read concurrently)
  31d1ff2  H1724→H1741 synthesis memo
  0e8d812  H1742-H1744 addendum + router policy .inc
  0a06ba7  H1745 (q6-approximate pressure)
  66dc946  H1746 (q6 route certificate kernel)
  89fe574  H1747 (reuse-conditioned cache)
  2548b23  H1748-H1752 router cache coding sweep
  104bf87  Phase B-2.3 design fork memo

ICB PHASES (pulled in parallel)
  5e7b51f  #561 layer-skip restored
  63b5cdf  #562 antirez power-throttle pulled
  317586b  #557 ICB topk_mask opt-in
  a233176  #558 ICB softplus_sqrt opt-in
  3c446c2  #559 ICB closed (structural blocker)
  5fa2b45  #566 ICB multi-cmd closed (inherits #559)

CODEC ANALYSIS TESTS (synthesis memo proposed; executed)
  35c20cd  Refinement memo: OOM-accuracy + 2-3 OOM speedup map
  a2a5fd5  Test A — Pareto sweep: phase dominates accuracy
  a8887aa  Test B — mean codebook is L2-near-optimal (percentile fails)
  bc96ad4  Test C — phase distribution is 21% non-uniform (FP4 signature)
  16d686b  Test D — learned codebook gains only 1.11% (refutes naïve C amplification)
  2083b3a  Cross-layer Test A — 26.4% rel_L2 reduction ALL 43 layers
```

## The headline empirical finding

**p8_m4 → p16_m4 = 26.4% rel_L2 reduction at ZERO storage cost.**

Universal across 43 layers (±1% variance). PLR2 stores mag+phase as
uint8 each — phase code in [0,16] fits in uint8 same as [0,8]. The
"+20% storage" cost from Test A bit-counting was bit-packed accounting;
PLR2 byte-aligned layout makes the upgrade free.

Validated end-to-end on real DS4 V4 weights at fp32 noise floor:

```
Encoder cos_sim/rel_L2:
  p8_m4 (legacy):  0.961 / 0.279   ← reference
  p16_m4 (new):    0.979 / 0.205   ← matches Test A prediction 0.204

GPU canary (real-data):
  p8_m4  L0,E0:    rel_err = 8.46e-8
  p16_m4 L0,E0:    rel_err = 1.07e-7
  p16_m4 L22,E100: rel_err = 0.00e+00 (bit-exact!)
  p16_m4 L42,E200: rel_err = 8.90e-8

Polar-DOWN cross-expert:
  rel_L2 = 0.2018 ± 0.001 (5 experts: E0/5/100/200/255)
  Matches gate-codec quality within 1.5%.
```

## The methodology lesson (Tests C → D)

Information-theoretic non-uniformity (Test C: 21%) does NOT predict
encoder-side gain (Test D: 1.11%). The polar codec's L2 error
decomposes into:

- **Assignment error** (which bin?) — bounded by inter-bin distance,
  ~1.5% at N=64
- **Decode error** (how far is bin center from true angle?) — dominates
  at p8/p16/p32

Uniform p8 lucky-aligns its 45° bins with FP4-induced cluster peaks
(0°, ±45°, ±90°, ±135°, 180°). Learned codebooks can only reduce
assignment error; can't reduce decode error at same N.

**Calibration**: information theory entropy is a CEILING; realized
gain depends on the loss function's sensitivity. Always measure realized
vs predicted ratio. This session's example: predicted 20× (entropy
ratio 0.79 → 21% suggests 21% gain), realized 1.11% — **~20× lower
than predicted**.

## What Phase B-2.3 needs from silv before more code lands

Per the design fork memo (commit 104bf87), Option D is recommended
(extend H1735 MSL kernel to read polar-down). The polar-down decode
validator (commit 4d16241) empirically confirms the path is viable.

**Three architectural questions remain unresolved:**

1. **Down-tile extraction strategy**: H1735 expects `down[route_pairs ×
   down_rows × act_rows]` as fp32 input. Polar-down stores per-expert
   `[down_rows × n_pairs]` complex pairs. Two sub-options:

   - **D1**: dispatcher extracts the [route_pair, down_rows, act_rows]
     fp32 tile from polar-down per-call (CPU + GPU work each call;
     equivalent to "fp32 down on-the-fly")
   - **D2**: kernel indexes into polar-down arrays directly via
     route_pair → expert_id → down_pool_offset (more MSL work, no
     per-call extraction)

   The mismatch: polar stores complex pairs along a single dimension;
   H1735 wants real values in a 3D tile shape. The choice impacts
   ~150 vs ~250 MSL LOC.

2. **Validation metric**: per-call bit-equivalence won't work at codec
   rel_L2 ~0.20. Need downstream metric. Options:

   - AIME hold-rate matches FP4 within N% (requires running inference)
   - Per-token logit cosine similarity vs FP4 baseline > 0.999
   - Loss-divergence threshold per token

   silv decides the gate.

3. **Per-token vs per-layer enable**: current DS4_POLAR_LAYERS is
   per-layer (all-or-nothing for a given layer). H1741 pressure-aware
   routing would extend to per-token (low-pressure → polar fast,
   high-pressure → FP4 strict). silv decides whether to ship per-layer
   first (simpler) or include per-token from day one (matches H1744
   policy architecture).

## What silv has on disk + repository as of this checkpoint

```
On disk:
  tmp/polar_full_p16m4/              14 GB, 129 .polar files (Test A
                                     recommended operating point)
  tmp/polar_phase_b_corpus/          1.3 GB (4-layer p8_m4 test corpus)

In repo (origin/main HEAD = 4d16241):
  Encoder + reader + pool + engine + diagnostic + GPU canary, all
    parameterized for arbitrary (P, M)
  6 analysis tools (Test A/B/C/D + down validator + sweep)
  4 codex synthesis memos + 1 Phase B-2.3 design fork + 1 final checkpoint

Reproducible commands:
  Encoder:        analyzers/polar_encode_mlx.py --phase-levels 16 ...
  Pareto sweep:   analyzers/polar_resolution_sweep.py --layers all ...
  Real canary:    ./ds4 --polar-real-canary <dir> <layer> <expert> ...
  Down validate:  analyzers/polar_down_decode_validator.py --polar-dir ...
```

## The next session's natural opening move

If silv chooses **D1** (per-call tile extraction): the kernel work is
small (~100 LOC MSL), the dispatcher does most of the work (~300 LOC
C). Total ~500 LOC, multi-hour focused turn.

If silv chooses **D2** (kernel-side polar indexing): the kernel work
is larger (~250 LOC MSL with route-aware lookup), the dispatcher is
thin (~150 LOC). Total ~500 LOC also, multi-hour focused turn.

Either way: Phase B-2.3 implementation is one focused multi-hour turn
beyond what this session shipped.

## Session-level meta-finding

The polar codec arc moved this session from "is this viable?" to
"the optimal operating point is empirically established and the
infrastructure is end-to-end ready, awaiting one architectural
decision."

The "live codec parameter optimization" speedup (Speedup #2 in the
refinement memo) is **empirically demonstrated** at full-model scale:
full Pareto sweep on all 43 layers in 124 sec; full re-encode in 42
sec; complete A/B in <2 minutes. The 6-hour offline tuning loop is
now a 30-second inner-loop activity, and silv has both the toolchain
and a concrete recommendation (p16_m4) ready to deploy the moment
Phase B-2.3 lands.

Session done. Direction request: D1 or D2 for Phase B-2.3?

---

## ADDENDUM 2026-05-25 — Phase B-2.3a SHIPPED (D1 chosen)

Commit `eb06f9c` lands the D1 path: polar-DOWN extracted to fp32
inside `ds4_gpu_mtl4_polar_real_canary`, gated by env var
`DS4_POLAR_REAL_DOWN=1`. Default synthetic-uniform path unchanged.

**Cross-cell validation (9/9 at fp32 noise floor on real DS4 V4):**
- 3 layers × 3 experts (L0/L22/L42 × E0/E100/E255)
- rel_err range: 9.23e-8 to 3.92e-7
- Polar-down → fp32 → H1735 chain is end-to-end bit-equivalent to
  a polar-decode CPU reference.

Settles D1-vs-D2 in favor of D1: kernel needs no polar-aware logic;
existing H1735 consumes fp32 down tiles fine.

**Next**: B-2.3b dispatcher canary (~250 LOC C) compares polar vs
FP4 source at kernel-output level. Per design fork memo addendum
2026-05-25.

Use:
```
DS4_POLAR_REAL_DOWN=1 ./ds4 --polar-real-canary tmp/polar_full_p16m4 \
  <layer> <expert> [down_rows [act_rows]]
```

Tasks: #568 [completed] B-2.3a; #563 [in_progress] parent.
