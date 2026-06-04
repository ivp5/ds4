# DS4 Flash on M1 Max 64GB — Operating Recipe & Architecture

silv directive 2026-05-25: "focus on getting ds4 flash working fast and
correct on this m1 max 64gb machine, recall mtl4, icd optimizations, and
the rest that was queued"

This is the single-page operational doc. All previous scattered
findings consolidated.

## STICKY HAZARD — pre-launch check (CRITICAL)

**THREE kernel panics on file: 2026-05-19, 2026-05-23, 2026-05-30.** The
first two: DS4 launches WITHOUT phase-split wiring 86.7 GiB IQ2_XXS into
Metal on a 48 GiB cap. The **THIRD (2026-05-30)**: a legacy VQB2 hot-store
path loaded an uncapped routed sidecar under concurrency (load avg 65,
multiple workflow agents), so Metal residency blew the cap. VQB2/CDX pack
paths are now quarantined as one-time artifacts; the active under-52 GiB path
is metadata-only GGUF + exact non-routed pack + M1R fixed-plane routed pack.
Two structural defenses remain current:
- **PreToolUse hook** `~/.claude/hooks/ds4-m1-guard.sh` (registered in
  settings.json) BLOCKS unsafe ds4 model-runs and reads inside executed `.sh`
  wrappers too.
- **RULE: NEVER run live ds4 on M1 from an autonomous workflow agent.**
  Foreground/silv only, one job at a time.

EVERY DS4 binary launch on M1 MUST contain one of:

```
--prefill-metal-phases auto    # preferred — auto-resolves N
--prefill-metal-phases N       # explicit phase count
--cpu-moe                      # mutually exclusive; ALL routed MoE on CPU
```

If your binary version doesn't support these flags, pick a different
binary. The default Metal path is a guaranteed panic on M1.

Do not use historical VQB2/CDX directories as runtime inputs. If a run cannot
be expressed as `--nonrouted-pack ... --m1r-pack ...` or a deliberate
source/GGUF control, stop and re-check the pack lineage before launching.

Late-warning log line (already past the point of no return):
```
ds4: Metal model views created in ... ms (mapped <N> MiB from offset ...)
```
If N > 48000, KILL within 90 seconds or expect reboot.

## Architecture stack — what's wired

```
┌──────────────────────────────────────────────────────────────┐
│  ds4-bench / ds4-server / ds4-eval / ds4-logitlens / ds4 (CLI)│
├──────────────────────────────────────────────────────────────┤
│  ds4_pillars.c        — ICB/HotExpert/SpecDecode  (env-gated, scaffolded)│
│  ds4_journal.c        — SQLite-WAL append-only    (JOURNAL=1 opt-in)│
│  ds4_expert_table.c   — Path B runtime expert mask│
│  ds4_polar_reader.c   — legacy codec reader       │
│  ds4_vqb1_reader.c    — legacy codec reader       │
│  ds4_moe_route_log.c  — per-event router trace (v3 schema)│
├──────────────────────────────────────────────────────────────┤
│  ds4.c (~24K lines)   — main engine, MoE dispatch │
│  ds4_metal.m          — Metal kernels + ICB record/replay│
│  ds4_neon_i8mm.c      — NEON kernels (i8mm NO-OP on M1, FEAT_I8MM=0)│
└──────────────────────────────────────────────────────────────┘
```

## Build options

```bash
# Default — zero-overhead, production
make ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4-logitlens

# JOURNAL=1 — append-only SQLite-WAL forensic journal
# Adds session/token/routing/event tables with UPDATE/DELETE ABORT triggers
# Set DS4_JOURNAL_DB=/path/to/journal.db to enable runtime
JOURNAL=1 make ds4-bench
DS4_JOURNAL_DB=/tmp/ds4.db ./ds4-bench ...
```

Journal schema is **append-only enforced at SQL trigger layer**:
- `session(id, started_unix, model_path, ctx_size, backend, options_json)`
- `token(session_id, seq_pos, token_id, logp, wall_us)`
- `routing(session_id, seq_pos, layer_id, selected BLOB, weights BLOB, margin)`
- `event(id, session_id, unix_us, kind, payload_json)`

Triggers: `RAISE(ABORT, 'journal is append-only')` on UPDATE/DELETE.

## Operating recipes — pick by workload

### 1. Single-prompt benchmark (default)
```bash
./ds4-bench --model ds4flash.gguf \
  --prefill-metal-phases auto \
  --threads 10 \
  --prompt-file PROMPT \
  --gen-tokens 128 --ctx-start 2048
```
Streams progress per shift #295 (every 16 tok or 2 sec).

### 2. Chat/agentic workload — KV-disk-dir for 9× speedup
```bash
./ds4-server --model ds4flash.gguf \
  --prefill-metal-phases auto \
  --ctx 100000 \
  --kv-disk-dir /tmp/ds4-kv \
  --kv-disk-space-mb 8192
```
Per shift #293: cold/warm/warm2 = 64.82s / 7.20s / 6.87s on repeat
prompts. **9× user-facing speedup**. Default-on for chat.

### 3. Doesn't fit in 60GB Metal cap — phase-split fallback
```bash
./ds4-bench --prefill-metal-phases N --cpu-moe \
  --model ds4flash.gguf ...
```
phases-auto resolves N from `iogpu.wired_limit_mb`. Use explicit N
only if you've measured the cap yourself.

### 4. Forensic mode — full journal
```bash
JOURNAL=1 make ds4-bench
DS4_JOURNAL_DB=/tmp/ds4.db ./ds4-bench --prefill-metal-phases auto ...
# Later: sqlite3 /tmp/ds4.db 'SELECT * FROM event WHERE kind="commit"'
```

## Current best — measured benchmarks (shifts #293-#295)

| Config | Prefill t/s | Gen t/s | Wall (117s baseline) |
|--------|-------------|---------|----------------------|
| phases-auto, no MTP | 22.04 | 1.83 | 117s |
| phases-auto, MTP=2 | similar | 1.62 | longer |
| phases-auto, MTP=4 | similar | 1.04 | longer (sequential verifier) |
| phases-auto, MTP=8 | similar | 0.75 | longest |
| KV-disk cold | 22.04 | 1.53 | 64.82s |
| **KV-disk warm** | (skipped) | **4.51 t/s** | **6.87s** |

**Deployment doctrine update**:
- Retire only the old **sequential-verifier** MTP path. MTP remains the target speed path when verifier work is block/parallel.
- Promote KV-disk-dir to default-on for repeated-prefix agentic workloads.
- Max-performance target: MTL4/compute + dense Q8_0 ICB replay + compressed routed-expert replay + block verifier + MTP/specdecode.

## MPSGraph / ANE runtime status

MPSGraph is an exact routed-D8F oracle and queueing probe, not the memory-floor
packed-index path. The useful production-facing lever is async batching:
`DS4_MPSGRAPH_ASYNC_BATCH=1` queues executable runs behind one shared-event wait.
On H3355 L26 selected-six canaries it improved same-window down and gate/up
throughput while preserving `bad=0` correctness.

`scaledDotProductAttention(query,key,value,mask,scale)` is exact but not a DS4
prefill replacement on M1 Max. 2026-06-04 canary: Q=1,K=128,D=512 gained 1.068×,
Q=1,K=512 was flat/slower, and prefill-ish Q=16/32,K=512 lost about 2× vs
explicit MPSGraph QKᵀ/softmax/PV. Keep it as a decode probe, not primary MLA/SWA.

CoreML owns ANE placement for shared high-B branches. `MLComputePlan` verifies
actual ANE placement for shared packages; CoreML output backings plus MPSGraph
shared-event queueing are the current overlap route.

Private/undocumented knobs remain probes only. `DS4_MPSGRAPH_COMPILE_MODE` accepts
`private_compiler_options` for controlled experiments with
`DS4_MPSGRAPH_COMPILER_OPTIONS=N`, but synthetic wins did not promote to real
H3355 selected-six or ANE+D8F overlap. Do not default-enable private CoreML E5
masks, precompiled-E5, priority strings, or raw MPSGraph compiler options until
they pass real selected-layer D8F, overlap, and fidelity gates.

## Quantization landscape — M1 Max 64GB fit

| File | Size | Status | Capability |
|------|------|--------|------------|
| `ds4flash.gguf` (IQ2_XXS w2 imatrix) | 86.7 GB | Production | Full AIME P01-P10 reachable |
| `DeepSeek-V4-Flash_H3384_H3382_all43_route_hotblock_sidecar_top6_down_native_codes_D8F_800kctx_probe_20260604` | 41 GB file / ~17.6 GB process RSS during current smoke / ~46 GB peak system wired during prefill | Default speed candidate, **not fidelity SOTA** | H3384 hot-block D8F + top6 native down-code sidecars; no-flag CLI can discover it, but normal generation is blocked before model load because agent2 A194-A203 and local `--coherence-gate` show dynamic AIME echo collapse; temp=0.6 seed=1 fails malformed-burst at token 13, and `DS4_D8F_RUNTIME_NATIVE_DOWN_DISABLE=1` still fails greedy at token 26, so recurrence is in the base D8F codec/allocation, not only native-down sidecar runtime or greedy sampling |
| `DeepSeek-V4-Flash_H3385_H3384_sparse_groupcode_sidecars_D8F_800kctx_probe_20260604` | 487 MB hardlink overlay on H3384 | Experimental sparse sidecar overlay, **not a self-contained flat-pack** | D8F payloads are hardlinks to H3384 and the directory lacks the metadata GGUF + non-routed pack required by `--flat-pack`; sparse+rank1 Metal selected-layer canaries are exact but slower than native-code sidecars and inherit H3384 coherence risk |
| `DeepSeek-V4-Flash_H3373_H3216_source_down_late_route_hot_gateup_repair_D8F_800kctx_20260604` | Rebuild target; allocator predicts ~51.77 GB with current metadata + non-routed pack extras | Current general-fidelity rebuild target, **not locally fully encoded yet** | Agent2 identified the echo-causing loss as gate/up; H3372 then exposed that late down was over-demoted. `tools/ds4_h3371_allocator.py --base-policy source --late-hot-gateup-threshold 0.02` keeps H3216 down allocation and promotes 40 L40-L42 route-hot gate/up records to K2048/K4096, leaving ~226 MB decimal margin under 52 GB. L40 rel-L2 canaries beat H3384, but promotion now requires `tools/ds4_head_margin_sensitivity.py`: `P95(|head_Latin(selected-worst) · J_norm · (W-Wq) · X|) < P5(reasoning margin)`. |
| `DeepSeek-V4-Flash_H3372_H3371_late_route_hot_gateup_repair_D8F_800kctx_20260604` | Rebuild target; allocator predicts ~51.64 GB with current metadata + non-routed pack extras | Superseded by H3373 source-down repair | H3372 keeps H3371's late-down budget repair and promotes route-hot gate/up, leaving ~357 MB margin. L40/E104 gate/up improves to `0.450/0.414`, but top-8 hot down worsens vs H3384 by mean rel-L2 `+0.113`; H3373 restores H3216 down while still fitting. |
| `DeepSeek-V4-Flash_H3371_H3216_budget_repair_no_overlay_D8F_800kctx_20260603` | Rebuild target; allocator predicts ~51.60 GB with current metadata + non-routed pack extras | Superseded by H3372/H3373 | `tools/ds4_h3371_allocator.py` regenerates the non-AIME policy from H3216: demote L40-L42 down records to K512 and demote the 186 weakest L40 down rows to K256. Audit found it left the L40-L42 route-hot gate/up records at K256/K256, matching Agent2's gate/up-loss diagnosis. |
| `DS4-trim50-asym-with-metadata.gguf` | 26 GB | Path A trim | **4× gen speedup BUT arithmetic carry breaks** (shifts cite v_P+5 vs v_P+9 collapse) |
| Q4_K_M-XL 153 GB | 153 GB | doesn't fit | — |
| `MLX-Qwen3.5-9B-DS-V4-Flash-4bit` | 5 GB | distill, MLX | side-by-side proposer |

H3384 root-cause triage: `tools/ds4_d8f_weight_compare.py` decodes a D8F
record exactly as Metal sees it and compares it to the source safetensors
expert. The first L40 canaries refute the easy "missing codebook scalar" bug:
optimal decoded→source scalars are ~0.91-0.99 for E0/E192/E236 gate/up/down,
while rel-L2 remains ~0.46-0.60. So the sampled echo is not fixed by a global
scale multiplier; the current evidence points at allocation/codec fidelity.

Margin-sensitivity triage is now the fidelity selector. `tools/ds4_logprob_margin_gate.py`
extracts the reasoning margin tail from `--dump-logprobs`; the agent2 H3384
echo trace has `reasoning_p5_margin_logits=0.0768974` and six fragile
selected-vs-nearest-Latin-competitor directions under 0.5 logits.
`tools/ds4_head_margin_sensitivity.py` projects codec error through the output
head rows for those directions, applies the RMSNorm JVP, and scores the
linearized influence on fragile rows. Its default backend is MLX/GPU when
available; exact trace runs should supply `--ffn-in-bin` and `--hc-dump` so
`--trace-row-mode auto` uses aligned per-token influence instead of the
calibration cross-product probe.

Gate/up residual triage: global rank-1/rank-8 and row-block rank-1/rank-8
residual probes on H3371 L40/E104 are too weak for the byte cost; gate act-rel
improves only ~4-5% and up ~1-4% even at rank-8. Direct K promotion of route-hot
late gate/up is the stronger falsified path, and it already fits the 52 GB
budget. H3373 is stronger than H3372 because it also restores the H3216 late
down allocation; the byte model still fits under 52 GB.

Int8 codebook-cache remains opt-in. On H3384 L26 selected-six,
`DS4_D8F_RUNTIME_NATIVE_DOWN_I8_CBSRAM=1` kept 52,911/219,136 entries at
`max_rel=0.02` and preserved selected mismatch=0, but slowed selected organ
time 16.911 → 25.015 ms/20 rounds and raised selected max_rel to 0.2818.
Tighter gates (`0.005`, `0.001`) were still slower, so this is not primary.

The trim50 file fits comfortably in 64GB but the dropped 10 layers
{1-6, 26, 27, 30, 35} are arithmetic-load-bearing under multi-equation
problems (shifts #292-294 bisect work). Use for non-math chat; not for
AIME / numerical reasoning.

## ICB record→replay status (2026-05-26 audit + DELETION of dead facade)

The dead `ds4_pillars.c` facade (219 LOC, 14 declared functions, zero
call sites) was DELETED 2026-05-26 per shift #189 doctrine ("delete what
doesn't earn its keep"; #192 "any customer pull elevates back to mod.ts;
until then the cut stands"). Files moved to `~/.Trash/`:
- `ds4_pillars.c.<ts>.bak`
- `ds4_pillars.h.<ts>.bak`

Reversible via `mv` if a future pillar-orchestration need emerges.

Build verified clean both modes (default + JOURNAL=1). ds4-bench
shrunk 840KB → 837KB.

**The REAL ICB record→replay mechanisms live IN-PLACE in `ds4_metal.m`**.
Current PRIME/default policy is measured-path-first, not dispatch-count-first:

| ICB / replay path | Env var | Status |
|-------------------|---------|--------|
| route_remap (43 layers × per-token) | `DS4_ICB_ACTIVE` | default-on; disable with `DS4_ICB_ACTIVE=0` or `DS4_ICB_ACTIVE_DISABLE=1` |
| softplus_sqrt | `DS4_ICB_SOFTPLUS` | default-on in decode router-select fallback; fused router-select usually bypasses it |
| topk_mask (2-kernel) | `DS4_ICB_TOPK_MASK` | default-on when that mask path is selected |
| dense Q8_0 single-token matvec | `DS4_DENSE_MATVEC_ICB` | PRIME/default; disable with `DS4_DENSE_MATVEC_ICB_DISABLE=1` |
| D8F classic packet ICB replay | `DS4_D8F_CLASSIC_PACKET_ICB` | PRIME/default; native texture down is gate-only ICB plus direct textured down |
| D8F texture-free packet range replay | `DS4_D8F_PACKET_ICB_RANGE` | PRIME/default where eligible; disable with `DS4_D8F_PACKET_ICB_RANGE_DISABLE=1` |
| route_weights_one | `DS4_ICB_WEIGHTS_ONE` | opt-in only; measured loser on the 6-thread decode kernel |

**Fusion kernels (not ICB but related)**:

| Fusion | Disable env var (default-ON) |
|--------|------------------------------|
| router_select_fusion | `DS4_METAL_DISABLE_ROUTER_SELECT_FUSION` |
| routed_pair_swiglu_fusion | `DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION` |
| KV RoPE+store | `DS4_METAL_DISABLE_KV_ROPE_STORE_FUSION` |
| indexed/decode attention inverse-RoPE | `DS4_METAL_DISABLE_INDEXED_ATTN_ROPE_FUSION`, `DS4_METAL_DISABLE_DECODE_ATTN_ROPE_FUSION` |
| FP8 shared-down HC | `DS4_METAL_DISABLE_SHARED_DOWN_FP8_HC_FUSION` |
| top-only greedy argmax | `DS4_METAL_DISABLE_TOP_ONLY_ARGMAX` |

**D8F native-down texture/cache defaults**:
- Native down readers default on for layers `0,20,25,26,37` at `n_tokens<=4`.
- Compact texture atlas defaults on for layers `20,37`.
- Pack-offset 2D texture defaults on for layer `26`.
- Gather, prewarm, private compact texture, i8 CBSRAM, sparse H3385, rank1 split
  packet replay, FP8 attention-output one-CB, Q-head RMSNorm+RoPE, indexer-Q
  RoPE, router matmul+select, and compressor RMS+RoPE stay opt-in until they
  pass full-runtime or matching selected-organ gates.

**Hot-expert F16 cache (was Pillar B)**: `ds4_hot_expert_init` has zero
call sites in main inference. Bitmap + manifest loader sits unused.
Pre-dequant pool NOT allocated. Margin gate per H1703 NOT wired (task #544).

**Spec-decode (was Pillar C)**: retired per shift #293 (MTP net loss).

**Engineering honesty**: 6 ICB phases were SHIPPED into ds4_metal.m
in-place, not into the ds4_pillars.c facade. The facade was a
documentation-shape proposal that was overtaken by direct in-place
wiring in ds4_metal.m. ds4_pillars.c is harmless (no perf cost; just
takes up build time + 8244 lines of declarations).

**Operating recipe** for the ACTUAL ICB pipelines:
```bash
# Speed-only H3384 canary: GPU-resident pack, split-2 overlap, measured
# decode fusions, D8F native-down texture/cache gates, dense Q8_0 matvec ICB,
# D8F packet ICB replay/range where eligible, and top-only argmax.
# Do not treat this as the primary fidelity path: greedy and temp=0.6 sampled
# gates both collapse, so the failure is not a greedy-only attractor.
./ds4 --metal --flat-pack /path/to/H3384 ...

# Comparison baseline for speed-only canaries: turn PRIME-only replay off explicitly.
DS4_PRIME_PATH=0 DS4_DENSE_MATVEC_ICB_DISABLE=1 DS4_METAL_DISABLE_TOP_ONLY_ARGMAX=1 \
  ./ds4 --metal --flat-pack /path/to/H3384 ...

# Known loser on decode remains opt-in: caller pays useResource cost on a
# 6-thread one-shot kernel, so do not include it in PRIME without new A/B data.
DS4_ICB_WEIGHTS_ONE=1 ./ds4 --metal --flat-pack /path/to/H3384 ...
```

Dense Q8_0 matvec ICB promotion gate: `--icb-dense-canary 4096 4096`
must be bit-exact, then `--icb-dense-bench 4096 4096 258 30` must show
a forward-level speedup. 2026-06-04 M1 Max corrected-bypass result:
bit-exact, 54.408 → 52.421 ms/fwd, 1.04×.

**MTL4 status** (per ds4_pillars.h doc):
- COMPUTE path productive on M1 Max (polar_dot canary: 83 ns/packet
  at 7776 packets; max_abs_err = 0).
- Indirect compute dispatch is productive when it collapses many tiny host
  dispatches: `--indirect-dispatch-canary 262144 1 5` measured tiny 1322.936 ms
  vs direct 0.642 ms vs indirect-prefilled 0.605 ms, exact output. H3385 sparse
  score uses this; H3384 primary has no sparse sidecar by default.
- Object/mesh dispatch is not a compute-fusion win on M1 Max:
  `--mesh-dispatch-canary 1024 5` measured host 3.479 ms vs mesh 5.321 ms.
- ML pipeline path NOT productive on M1 Max (raises NSInvalidArgumentException
  on normal MSL kernels — needs ML-compatible executable shape, only
  accessible from Core ML model, not custom compute kernel).
- IMPLICATION: use COMPUTE path for FROZEN organs (shared expert MLP,
  output head, attention output projection); skip ML packaging on M1.

## Why old MTP was retired, and what replaces it

Per shifts #292-#293:
1. MTP spec-decode runs with avg 2.7 accept per call (after bench-path
   bugfix to call ds4_session_eval_speculative_argmax) BUT
2. Verifier is sequential per-draft-position — each draft costs a full
   main-model forward
3. Net effect: MTP=2 → 1.62 t/s (vs no-MTP 1.83 t/s). MTP=4 → 1.04 t/s.
   MTP=8 → 0.75 t/s. Strict net loss, scaling worse with draft depth.

Real spec-decode win requires **batched/block MoE verification**: read each
expert/codebook once per verifier block, produce top-1 rows for acceptance,
and read back only the committed continuation logits. That is now the owning
abstraction. Codec containers are disposable; the runtime object should be
whatever lets MTL4/compute + ICB replay compressed expert records without
whole-model FP16 expansion.

Immediate code direction:
- Keep the non-strict microbatch verifier as the performance path.
- Keep strict/exact decode2 only as a correctness oracle.
- Remove per-token heap churn from speculative verify scratch.
- Fuse routed expert decode, SwiGLU, down, and verifier top-1 work into stable
  MTL buffers/ICB records before optimizing pack aesthetics.

## Why hot-expert pin doesn't work (shift #292)

Static top-K hot-expert pin claim REFUTED on routing trace v2 (55K events):
- top-32 covers only 58% of routing
- prefill↔decode J@32 = 0.32 (different sets)
- adjacent-layer J = 0.012 (layers don't share hot experts)

The 86.94% hit at 15 GiB pool from prior session was **LRU temporal-burst**
NOT static concentration. Pillar B (Hot-expert F16) must use dynamic LRU,
not static manifest — margin-gated per H1703.

## Why i8mm doesn't help on M1 (shift #292)

`FEAT_I8MM=0` on M1 Max. SMMLA kernel is a runtime no-op. The WALLS_MOVED
6× prefill speedup from prior reports was **entirely from
--prefill-metal-phases + 10-thread saturation**, not i8mm.

Profiler shows `ds4_vec_dot_iq2_xxs_pair_q8_K` dominates decode CPU.
The real CPU speedup paths:
1. TBL-based lookup batching in the IQ2 kernel (task #417 measured no
   speedup at A/B but only one test run)
2. Polar-PDM re-encode (LARGER file, refuting naive bandwidth assumption)
3. Pre-decoded hot-expert tier (Pillar B)
4. Workload-level --kv-disk-dir (already shipped, 9× win)

## Legendary lens applied to the architecture

- **Carmack** — ship simplest working. ds4_pillars.c is 8244 lines for
  three separate ladders, env-gated, single grep finds all state. 
- **Knuth** — literate code, provable bounds. Append-only journal has
  SQL-trigger ABORT enforcement; verified via standalone sqlite3 probe.
- **Page/Brin** — incremental indexing. Per-event O(1) inserts into journal;
  no periodic O(n) rebuild. Streaming bench progress emits per-work-unit.
- **Buterin** — append-only ledger semantics. Journal triggers reject
  UPDATE/DELETE; routing/token/event are the ledger entries.
- **Geohotz** — radical simplicity, tinygrad-pattern. Pillars are 3
  flat C files with global static state; no inheritance, no malloc in
  hot paths.
- **Torvalds** — irreducible structure. ds4.c is single ~24K-line
  spaghetti by design (NO EXTRACTION doctrine); grep IS the navigation.
- **Krzakala/Donoho/Sherwood** — propensity-theoretic. The CPU bottleneck
  IS `ds4_vec_dot_iq2_xxs_pair_q8_K`; speedups exist in lookup batching
  (TBL) + polar basis (separating ml=magnitudes from fm=phases per
  shift #291).
- **Thiel** — "what important truth do very few agree with you on" — that
  MTP is a NET LOSS on M1 (shift #293) was contrary to received wisdom
  from cuda-side benchmarks; M1-specific verification was the work.

## Pending / queued (consolidated from tasks list)

High-leverage:
- **#542** Fusion win: `kernel_dsv4_router_weights_with_remap` fused kernel
  ALREADY DONE in ds4_metal.m line 14690. ICB record→replay for it
  is wired but not validated against baseline.
- **#544** Margin gate on hot-expert cache per H1703 — actual Pillar B
  margin threshold not yet allocated, F16 pool empty.
- **#536** Codex MTL4 ML packed MoE path — REFUTED on M1 per H1728
  (ML pipeline incompatible). MTL4 COMPUTE path remains productive.
- **#563** Polar p8_m2 MTL4 kernel into DS4 — ds4_polar_reader.c exists,
  not yet swapped in for IQ2_XXS in production hot path.

Deferred:
- #543 Architecture review through legendary lenses + journal subsystem
  WIRED — this memo IS the architecture review; journal IS wired.
- #547-#551 Scaffold consolidation/refactors — deferred until pillars
  prove themselves at measurable gen-rate lift (none currently).

## Operating doctrine summary

1. **No flags/env discover H3384 + Metal + PRIME + prefill auto** when the H3384 flat-pack is present, but auto-selected H3384 now refuses normal generation until greedy and sampled `--coherence-gate` passes; temp=0.6 sampled echo makes this a codec/allocation failure signal, not a greedy decode workaround; use explicit `--flat-pack` or `DS4_ALLOW_UNCERTIFIED_H3384=1` only for speed-only experiments
2. **Use `--prefill-metal-phases 0` only for A/B**; Metal default is `auto`, and external D8F normalizes to phase-free GPU runtime when no GGUF routed residency needs swapping
3. **For chat/agentic: add `--kv-disk-dir`** (9× speedup on repeat)
4. **For forensics: build JOURNAL=1** (append-only SQLite trace)
5. **For benchmarking: streaming progress is on by default** (set
   DS4_BENCH_QUIET=1 to disable for piped CSV-only output)
6. **MTP is retired** — don't enable speculative decoding on M1
7. **trim50 file**: use for non-math; arithmetic carry breaks
8. **Codec promotion is margin-gated** — rel-L2/act-aware are triage only;
   require `P95(head-projected fragile-token codec error) < P5(reasoning margin)`
9. **Pillars env-gated** — none auto-active; enable when you have a
   measurement target

## Anemll ds4-ssd fork notes — measured borrowing only

Checked `github.com/Anemll/ds4-ssd` `main-alpha` (pushed 2026-06-04). Its
headline SSD path is a sidecar package with routed expert slot banks, async
pread/readahead, prefill slot prefetch, optional disk KV, and machine profiles
whose environment defaults never override user exports. That is a different
artifact class from H3384: current H3384 smoke shows ~17.6 GB process RSS while
Metal wired memory grows during prefill, so "fits" must be reported as live RSS
+ wired pressure, not inferred from a 52GB file-budget slogan. SSD slot banking
is not automatically faster and should not replace the default without a local
selected-layer/end-to-end win and coherence gate pass.

Borrowed now: make the measured speed-only recipe explicit — H3384 + Metal +
PRIME + `prefill-metal-phases auto`, with dynamic coherence refuted by greedy
and sampled gates. The primary fidelity direction is the H3373 margin-gated
general-fit rebuild, not another sampler setting on H3384.
Fork lesson kept for future sidecar work:
M1-class profiles keep ANE routed prefill off unless measured; 16K prefill
chunks require raw-cap headroom (`128 + chunk`, aligned), and SSD/I/O knobs
should be profile defaults with user env winning, not hidden cargo-cult flags.

## Files

- `DS4_M1_OPERATING_RECIPE.md` (this file)
- `ds4_pillars.h` (pillar API surface)
- `ds4_journal.h` (append-only journal API)
- `Makefile` (JOURNAL=1 build option)
- `montyneg/tmp/20260521_ds4_aberration/` (deep DS4 work, shifts #292-#295)
- `montyneg/audits/POST_MORTEM_DS4_LOCKUP_20260523.md` (panic post-mortem)
