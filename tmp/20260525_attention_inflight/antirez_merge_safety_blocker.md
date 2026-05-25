# antirez/main merge BLOCKED by M1 safety field removal

silv 2026-05-25 directive: "pull latest antirez/ds4 commits to head, merge".

Fetched `antirez/main` (12 new commits since our last fetch). Attempted
merge. ABORTED due to safety-critical structural divergence.

## What antirez shipped

12 commits since our HEAD diverged:

```
ad0209f Fix PRO routed MoE expert mapping
86ffb09 Document experimental PRO support
c3b81a4 Project renamed to DwarfStar, without the "4"
747cc94 Document PRO support and model downloads
dc322b8 Merge branch 'pro'
e8ad324 Validate DS4 compression layout by shape
26ee1fa Use parsed tensor span for Metal model views
b1703f3 Document model-specific continuation data
68cd2c0 Support PRO official continuation collection
04f151d DeepSeek v4 PRO support  ← +847 lines in ds4.c
e83bb9a Terminate displayed bash output with newline
add1488 Stabilize agent prefill label
```

## Why merge aborted

The merge produces 50 conflicts in ds4.c, 17 in ds4_metal.m, 1 each
in ds4.h and ds4_gpu.h. Investigating the ds4.h conflict revealed a
SAFETY-CRITICAL structural divergence:

**antirez REMOVED from `ds4_engine_options`:**
- `bool cpu_moe`
- `int n_cpu_moe_layers`
- `int prefill_metal_phases`

These are the **DS4 STICKY HAZARD** fields per CLAUDE.md — M1 kernel
panic prevention. Without `--prefill-metal-phases auto`, launching
DS4 on M1 with the full IQ2_XXS DeepSeek model wires 86.7 GiB into
Metal residency, exceeds the M1 Max 48 GiB wired-memory cap, and
forces system reboot. Documented twice on file (2026-05-19, 2026-05-23).

**antirez REPLACED with only**: `bool inspect_only`

This means antirez's main has STRUCTURALLY MOVED AWAY from the
cpu-moe/phased-prefill safety pattern. The replacement of `--prefill-
metal-phases auto` with `inspect_only` suggests the project has
pivoted to a different memory-management strategy I don't have
visibility into.

## What the merge would do if blindly accepted

Taking antirez's version of ds4.h would:
1. Remove the `--prefill-metal-phases` CLI flag
2. Remove the `--cpu-moe` CLI flag
3. Remove the auto-phase machinery
4. Re-introduce M1 kernel-panic vulnerability per the prior 2 documented incidents

This violates the CLAUDE.md doctrine:
> Before launching ANY ds4 binary on M1, the command line MUST
> contain `--prefill-metal-phases auto` or `--cpu-moe`.

The merge as-is would break this invariant.

## What needs to happen (silv decision)

Three options:

**Option A — Keep silv's M1-safe fields, merge antirez's other changes**:
- Hand-resolve ds4.h to KEEP cpu_moe/n_cpu_moe_layers/prefill_metal_phases
- Hand-resolve ds4.c and ds4_metal.m to keep silv's M1-safety paths
- Take antirez's PRO support, project rename, compression validation
- 4+ hours of conflict resolution

**Option B — Take antirez's main wholesale**:
- Drop silv's M1-safety machinery
- Trust antirez's new memory-management strategy (whatever it is)
- Re-derive M1-safety on top if needed
- RISK: M1 kernel panic if launched without understanding antirez's new pattern

**Option C — Don't merge yet**:
- Acknowledge the structural divergence
- Wait for antirez to upstream silv's M1-safety fields, OR
- Maintain a permanent fork with the safety machinery preserved
- silv's local work continues to diverge from antirez

## Other observations

The CLAUDE.md doctrine includes prior similar merges:
- #533: "ds4.c hand-merge — 39 conflicts between silv expert_remap +
  upstream NAX/Metal4/F16-KV"

So this pattern recurs. Each time silv's local work and antirez's
upstream collide on the same hot path.

## Recommended next step

I cannot safely choose between A/B/C without silv's input. The merge
mechanics are within my capability (4+ hours of focused work for A),
but the strategic choice depends on:
- Whether silv has plans for the M1-safety pattern post-rename
- Whether antirez's PRO support is critical enough to absorb the rename
- Whether silv wants to fork permanently

Per CLAUDE.md: "Agency Boundaries — outbound actions require exact
words. 'Prepare' ≠ 'send.' 'Draft' ≠ 'file.'" Applying the same
discipline: 'pull, merge' is the directive, but the choice between
A/B/C is silv's call when safety is at stake.

## What I've done that's reversible

- `git fetch antirez` (read-only, no side effects)
- Started + ABORTED merge twice (no commits, working tree clean)
- Repo is at `On branch main, Your branch is ahead of 'origin/main' by 123 commits, nothing to commit`

The repo is in its pre-merge state. No changes shipped.

## What's also blocking (concurrent compute load)

A background process (PID 68379) is running the 31st-continue all-10
rescue probe. It completed (output: 8/10). That's reported in
`all_10_rescue_8_of_10.md` (committed). Memory pressure resolved.

## Files

- `tmp/20260525_attention_inflight/antirez_merge_safety_blocker.md` (this)
- CLAUDE.md DS4 STICKY HAZARD section (load-bearing safety doctrine)
- `montyneg/audits/POST_MORTEM_DS4_LOCKUP_20260523.md` (prior incident)
