# Disk audit — codec corpus retention analysis

Triggered by Branch A sub-decision (codec body deployment requires
138-275 GB additional disk; volume currently 90% full / 197 GiB free).

## Current tmp/ footprint (29 GB total)

```
14 GB  tmp/polar_full_p32m8         ← PRODUCTION (recommended polar codec)
14 GB  tmp/polar_full_p16m4         ← LEGACY (superseded by p32_m8 since dec43f6)
1.3 GB tmp/polar_phase_b_corpus     ← LEGACY (pre-MLX-encoder, pre-codec-Pareto)
322 MB tmp/polar_L0_p32m4           ← L0 Pareto v1 reference
323 MB tmp/polar_L0_p32m8           ← L0 Pareto v2 reference (= subset of full_p32m8)
160 MB tmp/vqb1_L0_k256_mlx         ← MLX-VQ canary corpus (KEEP)
160 MB tmp/vqb1_L0_k256             ← CPU-VQ canary (redundant with _mlx — bit-identical)
```

## Retention recommendation

**Safe to mv ~/.Trash (saves 15.5 GB IF silv empties Trash)**:
- `polar_full_p16m4` (14 GB) — superseded by p32_m8. Quality:
  rel_L2 0.121 (p32_m8) vs 0.194 (p16_m4). No new test needs p16_m4.
- `polar_phase_b_corpus` (1.3 GB) — pre-MLX-encoder legacy from
  earlier session. Not referenced by current analyzers.
- `vqb1_L0_k256` (160 MB) — bit-identical to `vqb1_L0_k256_mlx`
  per cross-validation. Keep _mlx, retire CPU version.

**Keep retained**:
- `polar_full_p32m8` (14 GB) — production-ready corpus referenced by
  B-2.3c stub diagnostic + canary tests
- `polar_L0_p32m4`, `polar_L0_p32m8` (~650 MB total) — small enough
  to keep as L0 references for Pareto-comparison work
- `vqb1_L0_k256_mlx` (160 MB) — VQ canary corpus

## Disk math if silv approves trashing + empties Trash

```
Current:        197 GiB free / 1.8 TiB total = 89.4% full
After trash:    197 + 15.5 = 212.5 GiB free
After empty:    1.8 TiB - (current used - 15.5 GB) free
```

Per CLAUDE.md "mv to Trash does NOT free space; only emptying does,
and you cannot empty silv's Trash." So trashing alone doesn't help
the 90%-full constraint — silv must also empty Trash for the space
to free.

## How this affects Branch A sub-decision

If silv:
1. Approves trashing 15.5 GB + empties Trash → 213 GiB free
2. Branch A.2 VQ full-row (138 GB) fits comfortably (75 GiB headroom)
3. Branch A.1 polar full-row (275 GB) still doesn't fit (-62 GiB)

Net: trashing alone doesn't enable A.1, but it does make A.2 the
clearly-feasible-and-recommended path.

## What I will NOT do without silv approval

Trashing user-relevant files per CLAUDE.md "User files are read-only
unless user says 'edit/add to/change/update [filename]'." Even though
I created these corpora in this session, silv references them in
CLAUDE.md sections (e.g., the "p16_m4 production corpus" framing
predates p32_m8 supersession). Conservative: surface the audit, await
silv's explicit `mv ~/.Trash/` directives.

## Quick-action commands silv can run

```
# Free 14 GB (legacy p16_m4 corpus, superseded by p32_m8):
mv tmp/polar_full_p16m4 ~/.Trash/

# Free 1.3 GB (pre-MLX legacy):
mv tmp/polar_phase_b_corpus ~/.Trash/

# Free 160 MB (CPU-VQ duplicate of MLX-VQ):
mv tmp/vqb1_L0_k256 ~/.Trash/

# Then empty Trash via Finder to actually reclaim ~15.5 GB
```

After: A.2 becomes the unambiguously-feasible path. The B-2.3c body
work can target VQ K=256 + chunk-streaming (per codex H1773
deployable frontier) for end-to-end deployment.
