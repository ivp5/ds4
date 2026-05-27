# DS4 deferred queue audit — silv 2026-05-27 post-dispatch-wire

After shipping the simdgroup dispatch wire (commit 327e937, task #631
practically closed), the pending DS4 task queue sorts into four buckets.

## STALE-DONE (already shipped, just unmarked)

- **#548** Journal trivially-disabled-zero-perf via JOURNAL=1 opt-in
  `ds4_journal.h` lines 39-55: when `DS4_JOURNAL_ENABLE` is undefined,
  every emit/open/close/flush is a `static inline` `(void)param;` no-op.
  `Makefile` has `JOURNAL ?=` → default build excludes the .o + libsqlite3
  link. Trivially disabled at the source level + zero perf cost when off.

- **#549** Metal kernel consolidation: rb16/rb4 share body via static
  inline helper.
  `metal/dsv4_misc.metal:745` defines `dsv4_attend_heads8_rb16_inline()`;
  `kernel_dsv4_indexed_mixed_attention_heads8_rb16` (line 845) and
  `kernel_dsv4_indexed_mixed_attention_heads8_rb4` (line 864, kept as
  ABI alias) are both 16-line shells that call the shared body.

## TRIM-PARKED per silv directive ("trim is not good if it affects AIME P01")

- **#542** Fusion win: `router_weights_with_remap` → 4× gen speedup on
  trim50 plumbing path. The trim50 file (`DS4-trim50-asym-with-metadata.gguf`,
  44 GB) is the dropped-experts variant that broke AIME P01 on
  multi-equation reasoning. Parked. The fusion win itself is decoupled
  from trim; if a non-trim path produces a router_remap signature it
  could still apply, but it isn't blocking anything today.

- **#547** Scaffold consolidation: ds4_icb + ds4_hot_expert + ds4_spec_decode
  → ds4_pillars. Files were created earlier (commit predates current
  HEAD), then deleted (commit 3ab7639 per `h1871_h1873_codex_uses_my_substrate.md`)
  in honor of the CLAUDE.md no-extraction rule ("Codebase is always
  spaghetti with static global definitions/allocations on top"). The
  attempted consolidation conflicted with the architectural-discipline
  rule and was reverted. Mark closed-as-rejected.

- **#550** ds4_pillars + ds4_journal trim — 7 refactors, -697 net lines.
  Depends on #547. Since #547 was rejected per architectural discipline,
  this is also closed-as-rejected. Spaghetti stays.

- **#551** Plumbing-port fallback + dead remap/renormalize kernels
  removed. Trim-shape refactoring against grandfathered structure; same
  rejection rationale.

## NEEDS SPEC CLARIFICATION

- **#544** Margin gate added to hot-expert cache per H1703. The H1703
  source memo isn't present in `tmp/2026052*/`. Without the spec for what
  "routing margin" gates which experts pin, the right implementation is
  unclear. Task #535 (Routing-margin analyzer for SwiGLU instability
  diagnosis) shipped; the gate is the deployment of that analyzer's
  output into pinning policy, but the threshold + dispatch site need
  silv-side decision. Parked pending H1703.

## TRACTABLE BUT DEFERRED (not pursued this turn)

- **#543** Architecture review through legendary lenses + journal
  subsystem WIRED. Task #534 marks "Architecture review + journal
  infrastructure shipped" as completed. #543's "WIRED" likely means
  active per-event emit at hot-path sites. Currently the journal is
  opt-in via JOURNAL=1; per-event call sites exist (search for
  `ds4_journal_emit_*`); whether they're considered "wired enough" is
  an opinion call. Likely closed-as-superseded by #534.

- **#608** Unified amortization subsystem design — model-load + idle +
  shadow-thread tiers. Design task; needs research thinking, not pure
  shipping. Defer.

- **#610** Cross-arch synthesis landed in codex H1948: PCA128 sidecar
  beats coord/random. Synthesis from codex into project state. Likely a
  CLAUDE.md / shifts.md text-write task. Not in this turn.

- **#611** H1953: DS4 cross-arch corroboration of session DCT/Hadamard
  substrate work. Same shape as #610. Defer.

- **#643** MTL4 Hadamard FP16 batched kernel landed (GGUF rewrite
  unblocker). This is the technical bridge that would let the pair-AVG
  GGUF rewrite ship — which would let the simdgroup mat-mat actually
  fit in RAM at scale (no double-counting expert weights). Real impact
  but multi-day work. Defer to a future session.

## What changed this turn

- Shipped: task **#631** (simdgroup dispatch wire) — commit 327e937
- Marked stale-done: **#548**, **#549**
- Marked trim-parked per silv: **#542**, **#547**, **#550**, **#551**
- Identified spec-blocked: **#544**

The remaining tractable but not-this-turn work is **#643** (Hadamard
batched kernel for GGUF rewrite), which is the engineering bridge to
the FP16 simdgroup path's actual perf demonstration. That + the
multi-pair pair-AVG GGUF rewrite (FINAL.md from prior turn) together
unlock the 4× prefill projected win without burning the 64 GB RAM cap.
