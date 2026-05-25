# Merge build green; M1 smoke test BLOCKED on DS4 STICKY HAZARD doctrine

silv 2026-05-25 directive: pull antirez, merge, smoke test, OOM speedups.

## Merge: DONE
Build green. 5 binaries: ds4 / ds4-server / ds4-bench / ds4-eval /
ds4-agent. All respond to `--help`. Commit 79bcc1c.

## Smoke test: BLOCKED on doctrine

Pre-launch safety check per CLAUDE.md DS4 STICKY HAZARD:
```
ls /Library/Logs/DiagnosticReports/panic-*.panic
→ panic-full-2026-05-19-220021.0002.panic
→ panic-full-2026-05-23-120018.0002.panic
```

Both panics within 7-day window. Both DS4-shaped per
`montyneg/audits/POST_MORTEM_DS4_LOCKUP_20260523.md`.

Doctrine (CLAUDE.md):
> "if a panic file < 7 days old and DS4-shaped → halt, ask silv"

Compliance: HALT.

## What silv-explicit approval would enable

Three safe smoke-test paths, all requiring silv go-ahead:

**A. Minimal CPU-MoE smoke** (`./ds4 -m <model.gguf> --cpu-moe -p "hello"`):
- Runs all routed MoE on CPU
- No Metal residency risk
- Slowest but safest M1 path
- Tests model load + non-MoE Metal path

**B. Phased Metal smoke** (`./ds4 -m <model.gguf> --prefill-metal-phases auto -p "hello"`):
- Auto-sized phases per `iogpu.wired_limit_mb`
- Metal-resident in each phase, fallback to cpu-moe for generation
- The DS4 STICKY HAZARD-compliant path
- Tests both Metal prefill + cpu-moe gen

**C. ds4-bench smoke**:
- Same flag requirements
- Reports prefill/gen tokens/sec at fixed frontiers
- Measurable speedup vs pre-merge baseline

## OOM speedup levers identified

From the H1816-H1849 read (synthesis committed b62719b):

**Layer 0 — Top-k softmax fallacy**: 4.81 OOM headline. H1849 proved
the cut at codec quality scoring (144 × 129280 × 4096 → 144 × 2 × 4096).
Same fallacy applies to inference logit computation. Standard pipeline
computes full softmax over 129,280 vocab tokens then applies
temperature/top_p/top_k as post-hoc filtering. The actual decision
uses 1-2 tokens. Sampling instrumentation IS the band-aid evidence
of this waste.

**Already deployed in this merge**:
- 4× gen speedup from fused router_weights_with_remap (silv arch review)
- 25× faster source materialization from slice-aware loader (codex H1822)
- 19× wall improvement from ICB recording (silv arch review #540)
- 2.16 OOM rescue protocol via cap-mechanism (this session's AIME work)

## What I can do without silv approval (safe within doctrine)

1. **Inspect the merged binaries' help output** — done
2. **Read the build artifacts' symbol tables** — safe
3. **Run code analysis on the merged source** — safe
4. **Look for opportunities to apply H1849 top-k optimization** — safe
5. **Test on AMD/nvidia (silv-explicit per CLAUDE.md)** — possible if
   binary cross-compiled, but the M1 build is arm64-only

## What I cannot do

- Launch ds4 / ds4-bench / ds4-eval / ds4-server / ds4-agent on M1
  without silv explicit go-ahead, even with --prefill-metal-phases auto,
  because the panic-files-within-7-days check tripped.

## Recommendation for next iteration

If silv wants the smoke test:
- Confirm safe to launch with `--prefill-metal-phases auto` despite
  recent panics
- Or provide a model path to use
- Or redirect to AMD-side cross-build

If silv wants OOM speedup work without launch:
- Identify the inference logit-computation site in ds4.c
- Implement partial-sort top-k in place of full softmax
- Measure compile-only (no panic risk)

## Repo state after directive

- 13 commits added today since silv's directive
- Merge commit 755fec5 + 4 fixup commits
- Build green, M1-runnable but launch-blocked

## Files

- This memo: tmp/20260525_attention_inflight/merge_smoke_test_blocked_on_doctrine.md
- Synthesis: tmp/20260525_attention_inflight/h1816_h1849_synthesis_oom_fallacy.md
- Merge commit: 755fec5 + 4 follow-ups
- Build binaries: ds4, ds4-server, ds4-bench, ds4-eval, ds4-agent
- Post-mortem: montyneg/audits/POST_MORTEM_DS4_LOCKUP_20260523.md
