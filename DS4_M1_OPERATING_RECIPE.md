# DS4 Flash on M1 Max 64GB — Operating Recipe & Architecture

silv directive 2026-05-25: "focus on getting ds4 flash working fast and
correct on this m1 max 64gb machine, recall mtl4, icd optimizations, and
the rest that was queued"

This is the single-page operational doc. All previous scattered
findings consolidated.

## STICKY HAZARD — pre-launch check (CRITICAL)

**Two kernel panics on file: 2026-05-19, 2026-05-23. Both triggered by
DS4 launches WITHOUT phase-split, attempting to wire 86.7 GiB IQ2_XXS
into Metal on a 48 GiB cap.**

EVERY DS4 binary launch on M1 MUST contain one of:

```
--prefill-metal-phases auto    # preferred — auto-resolves N
--prefill-metal-phases N       # explicit phase count
--cpu-moe                      # mutually exclusive; ALL routed MoE on CPU
```

If your binary version doesn't support these flags, pick a different
binary. The default Metal path is a guaranteed panic on M1.

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
│  ds4_polar_reader.c   — polar p32_m8 codec        │
│  ds4_vqb1_reader.c    — VQ-2D K=256 codec         │
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
make ds4 ds4-server ds4-bench ds4-eval ds4-agent

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

**Deployment doctrine**:
- **Retire MTP from M1 recipe** (net loss at all draft depths)
- **Promote KV-disk-dir to default-on** for agentic workloads

## Quantization landscape — M1 Max 64GB fit

| File | Size | Status | Capability |
|------|------|--------|------------|
| `ds4flash.gguf` (IQ2_XXS w2 imatrix) | 86.7 GB | Production | Full AIME P01-P10 reachable |
| `DS4-trim50-asym-with-metadata.gguf` | 26 GB | Path A trim | **4× gen speedup BUT arithmetic carry breaks** (shifts cite v_P+5 vs v_P+9 collapse) |
| Q4_K_M-XL 153 GB | 153 GB | doesn't fit | — |
| `MLX-Qwen3.5-9B-DS-V4-Flash-4bit` | 5 GB | distill, MLX | side-by-side proposer |

The trim50 file fits comfortably in 64GB but the dropped 10 layers
{1-6, 26, 27, 30, 35} are arithmetic-load-bearing under multi-equation
problems (shifts #292-294 bisect work). Use for non-math chat; not for
AIME / numerical reasoning.

## Pillars — scaffolded, env-gated, NOT YET WIRED into main inference

```c
// ds4_pillars.h — three speedup ladders, env-gated
DS4_ICB_ACTIVE=1   // Pillar A — ICB record→replay (record once, dispatch many)
DS4_HOT_EXPERT_ACTIVE=1 + DS4_HOT_EXPERT_MANIFEST=path
                   // Pillar B — F16 cache for hot experts (margin gate per H1703)
DS4_SPEC_ACTIVE=1  // Pillar C — Spec-decode propose+verify (retired for MTP per shift #293)
```

**Status per pillar**:
- **A (ICB)**: 6 phases shipped, record→replay self-tested, but the
  cache slot validates and re-records on every layer per token —
  the projected 15% gen lift hasn't been measured against current
  1.83 t/s baseline.
- **B (Hot-expert F16)**: bitmap + manifest loader shipped. Pre-dequant
  pool NOT yet allocated. Margin gate per H1703 NOT YET wired (task #544).
- **C (Spec-decode)**: retired in favor of MTP retirement per shift #293.

**MTL4 status** (per ds4_pillars.h doc):
- COMPUTE path productive on M1 Max (polar_dot canary: 83 ns/packet
  at 7776 packets; max_abs_err = 0).
- ML pipeline path NOT productive on M1 Max (raises NSInvalidArgumentException
  on normal MSL kernels — needs ML-compatible executable shape, only
  accessible from Core ML model, not custom compute kernel).
- IMPLICATION: use COMPUTE path for FROZEN organs (shared expert MLP,
  output head, attention output projection); skip ML packaging on M1.

## Why MTP is retired on M1

Per shifts #292-#293:
1. MTP spec-decode runs with avg 2.7 accept per call (after bench-path
   bugfix to call ds4_session_eval_speculative_argmax) BUT
2. Verifier is sequential per-draft-position — each draft costs a full
   main-model forward
3. Net effect: MTP=2 → 1.62 t/s (vs no-MTP 1.83 t/s). MTP=4 → 1.04 t/s.
   MTP=8 → 0.75 t/s. Strict net loss, scaling worse with draft depth.

Real spec-decode win would require **batched MoE verification** (read
each expert once per batch), which DS4 doesn't currently implement.
Task #418 was the entry point design memo for this; not yet built.

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

1. **Always launch with `--prefill-metal-phases auto`** (M1 STICKY HAZARD)
2. **For chat/agentic: add `--kv-disk-dir`** (9× speedup on repeat)
3. **For forensics: build JOURNAL=1** (append-only SQLite trace)
4. **For benchmarking: streaming progress is on by default** (set
   DS4_BENCH_QUIET=1 to disable for piped CSV-only output)
5. **MTP is retired** — don't enable speculative decoding on M1
6. **trim50 file**: use for non-math; arithmetic carry breaks
7. **Pillars env-gated** — none auto-active; enable when you have a
   measurement target

## Files

- `DS4_M1_OPERATING_RECIPE.md` (this file)
- `ds4_pillars.h` (pillar API surface)
- `ds4_journal.h` (append-only journal API)
- `Makefile` (JOURNAL=1 build option)
- `montyneg/tmp/20260521_ds4_aberration/` (deep DS4 work, shifts #292-#295)
- `montyneg/audits/POST_MORTEM_DS4_LOCKUP_20260523.md` (panic post-mortem)
