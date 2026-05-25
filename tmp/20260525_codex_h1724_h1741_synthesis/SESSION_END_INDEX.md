# Session-end index — 2026-05-25 codec arc

One-stop navigation for the 21-commit multi-segment session arc
(commits 503934a → 619c6f0). Use this to jump to any artifact.

## Git log (most-recent first)

```
619c6f0  Synthesize codex H1758-H1773: streaming/phase-rerank complements polar/VQ
0c8c269  test_polar_b23c_gate.sh: runtime verification script for silv
22a1c8c  B-2.3c stub: hot-path gate wired, FP4 path unchanged
b5e84f1  Branch A pre-flight: row-coverage dependency forces silv decision
1169cfc  Session checkpoint: 22-commit codec arc inheritable in one read
a799456  MLX-accelerated VQ encoder: 24% faster + canary bit-equivalent
917df69  VQ-2D C-side canary chain shipped: 6/6 cells at fp32 noise floor
3bef5a7  Row-coverage audit REFUTES the quality concern — codebook generalizes
a1ca099  SCOPE CAVEAT: codec corpora encode 3-6% of weight rows + VQB1 writer
cf39af8  VQ K-sweep: K=256 is empirical optimum (no headroom above)
8a1e401  VQ-2D at real FFN scale: 6.15× lower codec error than polar p32_m8
f837d90  VQ-2D codec breakthrough: 1.10 OOMs accuracy at HALF storage vs polar
f024ff4  Production p32_m8 corpus: 33024 encodings, 44.7s, 14 GB on disk
503934a  H1735 mag_levels generalization: 0.52 OOMs accuracy at +1.4% storage
dec43f6  Codec Pareto frontier: p32_m4 is new optimum (-39% rel_L2 zero cost)
a9a7580  Session checkpoint update: full polar codec characterization
2f77491  Phase B-2.3b at REAL FFN scale: kernel output ≈ weight codec quality
4ce53cc  Phase B-2.3b REFINEMENT: codec quality does NOT improve with act_rows
d216608  Phase B-2.3b: polar vs FP4 kernel-output A/B — codec composes acceptably
b781bf2  Memo addenda: Phase B-2.3a shipped + validated, D1 path settled
eb06f9c  Phase B-2.3a (#568): polar-DOWN canary path validated bit-equivalent (D1)
```

## Code surfaces (what compiles + ships)

### C / Objective-C (`ds4_metal.m`, `ds4_polar_reader.*`, `ds4_vqb1_reader.*`, `ds4_gpu.h`, `ds4.c`)

- `ds4_polar_reader.h/c` — PLR2 file format reader (mmap + decode_pair)
- `ds4_vqb1_reader.h/c` — VQB1 file format reader (mmap + codebook lookup) — NEW this session
- `ds4_metal.m::gate_up_down_partial` — H1735 polar kernel, mag_levels generalized this session
- `ds4_metal.m::gate_up_down_vq` — NEW VQ MTL4 kernel (no trig, codebook lookup)
- `ds4_metal.m::ds4_gpu_mtl4_polar_real_canary` — polar B-2.3a canary
- `ds4_metal.m::ds4_gpu_mtl4_vq_real_canary` — NEW VQ canary
- `ds4_metal.m::ds4_gpu_mtl4_polar_routed_moe_batch_stub` — NEW B-2.3c stub
- `ds4.c::metal_graph_encode_layer_ffn_batch` — NEW hot-path gate at ~14369
- `ds4_cli.c` — added `--vq-real-canary` flag
- `Makefile` — added `ds4_vqb1_reader.o` to CORE_OBJS

### Python encoders + analyzers (`analyzers/`)

- `polar_encode_mlx.py` — polar MLX encoder (existing, parameterized
  for arbitrary phase_levels + mag_levels this session)
- `vq2d_codec_explore.py` — NEW VQ-2D K-means encoder (CPU)
- `vq2d_encode_vqb1.py` — NEW VQB1 file writer (Python)
- `vq2d_encode_mlx_vqb1.py` — NEW VQ MLX encoder (24% over CPU)
- `vq2d_real_ffn_scale_ab.py` — NEW VQ vs FP4 at real FFN scale
- `vq2d_row_coverage_audit.py` — NEW codebook-generalization audit
- `polar_vs_fp4_kernel_output_ab.py` — NEW polar canary-scale A/B
- `polar_down_real_ffn_scale_ab.py` — NEW polar real-FFN-scale A/B
- `polar_down_decode_validator.py` — polar weight-level validator

### Runtime test script

- `test_polar_b23c_gate.sh` — NEW silv-runnable verification of
  B-2.3c stub gate. Pre-flight enforces DS4 STICKY HAZARD rules.

## On-disk artifacts (NOT in git, file paths only)

- `tmp/polar_full_p32m8/` — 14 GB production-ready polar p32_m8
  corpus (33,024 encodings × 128 rows each — scope per
  SCOPE_CAVEAT_row_coverage.md; codebook-quality extends to full
  4096 rows per audit commit 3bef5a7)
- `tmp/polar_full_p32m8/manifest_summary.json` — small file in git
- `tmp/polar_L0_p32m4/` — 322 MB Pareto v1 reference corpus
- `tmp/polar_L0_p32m8/` — 322 MB Pareto v2 reference corpus
- `tmp/vqb1_L0_k256/` — 167 MB VQ K=256 corpus (Python writer)
- `tmp/vqb1_L0_k256_mlx/` — 167 MB VQ K=256 corpus (MLX writer,
  bit-identical to Python writer per canary)

## Memos (`tmp/20260525_codex_h1724_h1741_synthesis/`)

Read in this order for full context:
1. `memo.md` — H1724-H1752 synthesis (codex inputs, original)
2. `phase_b2_3_design_fork.md` + addenda — polar B-2.3 architecture decision
3. `codec_pareto_frontier.md` — Pareto v1 (p8/p16/p32/p64 m4)
4. `codec_pareto_v2_m8m16.md` — Pareto v2 (m8/m16 with kernel patch)
5. `vq2d_codec_breakthrough.md` — VQ K=256 substrate finding
6. `SCOPE_CAVEAT_row_coverage.md` — Scope caveat + REFUTATION
7. `BRANCH_A_PREFLIGHT.md` — Branch A row-coverage sub-decision
8. `codex_H1758_H1773_synthesis.md` — codex parallel-arc integration
9. `SESSION_FINAL_CHECKPOINT.md` — latest checkpoint with full state
10. This file (`SESSION_END_INDEX.md`)

## OOM scorecard (substrate level)

```
Encoder MLX speedup:                     1.86 OOMs (production)
Polar p32_m8 (production-ready):         0.52 OOMs
VQ K=256 (canary-ready):                 1.10 OOMs
Combined session gain:                   2.96 OOMs
```

Validated at full row coverage per codebook-generalization audit
(commit 3bef5a7).

## Decision branches for silv

Polar arc + B-2.3c stub shipped; body needs silv sub-decision:

| Branch | Description | Storage | Engineering | Risk |
|--------|-------------|---------|-------------|------|
| **A.1** | Full-row polar corpus | 275 GB | Encode 37 min + dispatcher 300 LOC + silv runtime | medium |
| **A.2** | Full-row VQ corpus | 138 GB | MLX VQ encoder port 150 LOC + encode 6 min + dispatcher 300 LOC + silv runtime | medium |
| **A.3** | Hybrid polar+FP4 dispatch | 14 GB (current) | Combine logic 500 LOC + silv runtime | medium-high |
| **B** | Pivot to non-codec queued work | — | varies | low |
| **C** | Defer to later session | — | — | none |

Disk approval needed for A.1 / A.2. A.3 doesn't need disk but
delivers no speedup (only A/B scientific validation).

## Inheritance state for next session

If next session opens with "codec arc continuation":
- Read `SESSION_FINAL_CHECKPOINT.md` for complete picture
- Check this index for code surface map
- silv has options A.1 / A.2 / A.3 / B / C — get answer first

If next session opens with "different topic":
- 27 commits stand; tmp/polar* corpora on disk are reusable
- B-2.3c stub is harmless residual (FP4 path unchanged)
- Codex H1758-H1773 is synthesized in memo; H1774+ may be available

The arc is at a clean natural pause; the codec engineering is
silv-direction-bound from this point forward.

## Late codex shipped this same evening (post-checkpoint)

| Codex shift | Result | Strategic relevance to my codec arc |
|-------------|--------|--------------------------------------|
| H1776-H1782 | Row-stream + layer-window LRU | Memory orthogonal to my codec |
| H1783 | 84% of wall is f32 materialization | Names codec as the next speed organ |
| H1784 | 15.5× f32 expansion overhead measured | Sets the deletion target |
| H1786 | MTL4 raw-IQ2 dot kernel: 1024 rows fp32 noise floor, 15.5× expansion AVOIDED, 1.18ms GPU | Direct validation of codec-as-compute-boundary pattern; polar/VQ kernels prove same pattern with learned codebooks |
| H1787 (running PID 58195) | Parallel raw IQ2 dot canary at 4096 rows | Codex pursuing H1786's "next organ: parallel decode/reduction" |

Total session arc as of H1786 integration: 34 commits.

Each subsequent codex shift was integrated into
`codex_H1758_H1773_synthesis.md` via addendum (not new memo).
