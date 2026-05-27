# DS4 tmp/ inventory index

silv directive 2026-05-27: "reorganize and integrate the contents of
montyneg/ivp5_ds4/tmp directory experiments — trash obsolete
(you've got almost 32gb there), integrate and move to reasonable
locations truly beneficial insights and reference them correctly
so you can find them easily later".

## What was trashed (~29 GB → ~52 MB, 560× reduction)

All moved to `~/.Trash/ds4_tmp_*_<timestamp>` (silv empties manually).
The analyzer scripts at `analyzers/polar_encode_*.py` /
`analyzers/vq2d_encode_*.py` can regenerate any trashed encoder corpus
if needed — the corpora are scratch output, not load-bearing input.

### Phase 1 — 8 polar/vqb encoder corpora (~28.9 GB)

Encoder output from closed codec arcs (tasks #571-589, all completed).
References in C/objc are documentation comments showing how to point
`DS4_POLAR_DIR=` at one of these paths; nothing READS the binary
corpora at runtime. The codec exploration concluded that VQ-2D K=256
dominates polar p32_m8 (#574), so these corpora are dead-end paths.

| dir                          | size | what                                            |
|------------------------------|------|-------------------------------------------------|
| polar_full_p32m8             | 14 GB| full-model polar p32_m8 encoded weights         |
| polar_full_p16m4             | 14 GB| full-model polar p16_m4 alternate parameter    |
| polar_phase_b_corpus         | 1.3 GB| polar phase B exploration corpus              |
| polar_L0_p32m8               | 323 MB| Layer-0 canary corpus                         |
| polar_L0_p32m4               | 322 MB| Layer-0 canary corpus alternate parameter     |
| vqb1_L0_k256                 | 160 MB| VQB1 canary corpus K=256                      |
| vqb1_L0_k256_mlx             | 160 MB| MLX-encoded VQB1 canary                       |
| vqb2_corpus_h1887v2          | 80 MB | VQB2 H1887v2 corpus + smoke logs              |

### Phase 2 — 5 leftover polar_* result JSON dirs + loose codec files (~388 KB)

Small JSON result files from one-off codec probes; findings already
captured in shifts.md / queue tasks.

`polar_codebook_ab/`, `polar_down_validation/`, `polar_learned_phase/`,
`polar_phase_histogram/`, `polar_resolution_sweep/` + 9 loose
`polar_*.{log,json}` and `vq2d_*.json` files at tmp/ root.

## What survives (14 dirs + 1 loose file, ~52 MB)

The convention: each subdir is dated `YYYYMMDD_topic` or topic-named
and contains at least one .md memo. To find a past session:

```
grep -i <keyword> tmp/_INVENTORY.md
```

### By session date

| dir                                  | size | topic |
|--------------------------------------|------|-------|
| `20260525_antirez_v4pro_findings/`   | 8K   | antirez DS4 v4 PRO upstream findings + integration plan |
| `20260525_attention_inflight/`       | 47M  | **MAJOR SESSION** — 80 .md memos. Substrate descent / attention-inflight probes / Conjecture #23 cross-precision refinement. Topics: L22 norm during cycle, late-layer ramp, cycle vocabulary, precision attractor, bidirectional boundary, rescue P02/P04 verified live, two-mode emission, cross-arc codex synthesis. Result JSONs (top 4 are 13M + 9.6M + 6.8M + 4.9M) are regenerable from probes; the .md memos are the value. |
| `20260525_codec_vs_iq2_audit/`       | 456K | <52 GB on M1 Max via VQ knobs — honest verdict for #589 |
| `20260525_codex_h1724_h1741_synthesis/` | 192K| Branch A pre-flight: dependency check before polar hot-path lands (#577) |
| `20260526_ds4_smoke/`                | 8K   | DS4-Flash smoke test PASSED on M1 Max 64GB (#538 follow-up) |
| `20260526_metal_dispatch/`           | 112K | Metal-MoE FP16 hot-store wiring + flag cleanup (#625) |
| `20260526_vqb2_h1849_scope/`         | 1.3M | Session "go on" round 2 — codex H1875-H1884 integration + queue (#583+) |
| `20260527_attn_scale/`               | 48K  | Attention scale × quantization interaction test scripts on trim50 (#636) |
| `20260527_dispatch_wire/`            | 48K  | DS4 deferred queue audit — silv 2026-05-27 post-dispatch-wire (#631) |
| `20260527_hadamard/`                 | 1.9M | **LOAD-BEARING** — `FINAL.md`, `ENCODE_FINAL.md`, `DEPLOYMENT_RULES.md` + kernel/encoder source + Python ↔ GPU cross-validation harness. Hadamard-16 codec primitive (#643/#645/#646). The DEPLOYMENT_RULES.md captures 6 selector rules extracted from codex H1967-H2068. |
| `20260527_harm_scorer/`              | 184K | **LOAD-BEARING** — `DESIGN.md` for the per-organ signed source-top-K harm scorer (#649). OOM-precision lift (signed top-K margin delta vs rank/rel-L2). OOM speedup ladder (KV-prefix × structural-panel × batched-case = ~600×). Phase A code shipped: `organ_skip_smoke.c` 31/31 assertions pass. |
| `20260527_hot_store_simdgroup/`      | 80K  | Hot-store full-coverage + simdgroup matmul — engineering status memo |
| `20260527_layer_dedup/`              | 112K | 6-question synthesis on amplification, head-damping, limit cycles, pairs, codec, middle cache |
| `20260527_simdgroup_design/`         | 8K   | simdgroup_matrix prefill mat-mat kernel — design + stub (#631 precursor) |

### Loose files at tmp/ root

| file                       | size | what                                  |
|----------------------------|------|---------------------------------------|
| `vqb2_smoke_prompt.txt`    | 2K   | prompt fixture for VQB2 smoke (referenced by `analyzers/vqb2_to_fp16_test.c`) |

## Load-bearing pointers (DON'T re-derive these from scratch)

When iterating on the codec/selector arc post-cleanup, START HERE:

1. **DEPLOYMENT_RULES.md** at `tmp/20260527_hadamard/DEPLOYMENT_RULES.md`
   — six rules (selector before primitive, per-organ not uniform,
   route-resident not full-manifest, per-expert pruning, signed harm
   not absolute drift, domain certificate) extracted from codex
   H1967-H2068 sequential read. Re-read before any Hadamard codec /
   GGUF rewrite tool / dispatch-site pre-pass work.

2. **DESIGN.md** at `tmp/20260527_harm_scorer/DESIGN.md` — the
   per-organ signed top-K harm scorer that DEPLOYMENT_RULES requires
   but didn't exist. Names the 2-3 OOM accuracy lift (continuous
   signed top-K margin delta vs current rank/rel-L2 coarse measure)
   and 2-3 OOM speedup ladder. Five pre-committed REFUTE conditions
   inside.

3. **20260525_attention_inflight/** memos — the substrate descent
   that produced the L22 norm finding, cycle-vocabulary observations,
   rescue protocol validation, two-mode emission boundary. Many of
   these flowed into Conjectures #19-23 in `/Users/silv/cl/tlp/CLAUDE.md`;
   the memos preserve the cycle-by-cycle reasoning.

4. **ENCODE_FINAL.md** at `tmp/20260527_hadamard/` — names the three
   deferred ENCODE_FINAL items (GGUF rewrite tool, hot-store basis-
   aware marker, dispatch-site pre-pass) with their explicit
   DEPLOYMENT_RULES constraints. Item #2 (hot-store basis-aware
   marker) shipped 2026-05-27 (#647). Items #1 and #3 are blocked on
   the selector design from harm_scorer/DESIGN.md.

## Reference to externally-integrated insights

Findings already extracted upstream of tmp/ (do not re-extract here):

- **/Users/silv/cl/tlp/shifts.md** — chronological doctrine log
- **/Users/silv/cl/tlp/CLAUDE.md** — conjecture register (Conjectures
  #15-23 carry most DS4-side findings)
- **memory/MEMORY.md** at `~/.claude/projects/-Users-silv-cl-tlp/memory/`
  — cross-session memos by topic

When a dated tmp/ memo's finding has been promoted to one of the above,
the tmp/ memo is preserved as the source-of-record (where the work
actually happened) but the canonical citation is the upstream
location. This file's purpose is purely the LOOKUP INDEX from "what
was that session about" → "where in tmp/ is it".
