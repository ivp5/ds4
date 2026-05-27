# High-potential queue — for silv review before implementation

Standing context: Round 1 (wide-tile MSL consolidation) deferred per
pivot to "MTL4 port queue + ICB optimizations". This queue captures
what I'd pick next, ranked. Each item is a concrete, sized swing.

## MTL4 port queue — what's left to port

### Q-1: mul_mm_id f16-output family (12 pipelines)
4 quants {IQ2_XXS, Q8_0, Q4_K, Q2_K} × 3 widths {n32, n64, n128} of
`kernel_mul_mm_id_<quant>_f16{,_n64,_n128}`. The production dispatcher
(`ds4_gpu_routed_mm_f16_rhs_pipeline_for_tile`) already routes to these
classic Metal kernels for half-precision-output paths. None in MTL4 yet.
Each ~150-400 LOC. Same wide-tile mc[8] bug in upstream — I'd port WITH
the corrected NR1/k pattern from the start, never shipping the broken
form. Est. +12 pipelines × ~250 LOC = +3000 LOC if naive, OR if shipped
via the Round-1 helper pattern: +12 pipelines × ~20 LOC + 4 builders × ~250 = +1240 LOC.

### Q-2: FlashAttention kernels family
`flash_attn_ext_pad`, `flash_attn_ext_blk`, `flash_attn_ext_vec` — KV-cache
attention. Different kernel family entirely (not mm_id pattern). Higher
risk, higher novelty, blocks long-context optimization work. Est. +600-1000 LOC.

### Q-3: kernel_mul_mv_id_* family (matvec routed)
Counterpart to mul_mm_id for decode-phase (single-token routing). Used
heavily in autoregressive generation. ~6-8 pipelines. Est. +800-1200 LOC.

### Q-4: classic Metal upstream PR
File PR to antirez/ds4 with the NR1/k corrected template. Even if not
merged, documents the bug for community. Zero LOC cost on my side; pure
contribution.

## ICB optimization queue

### I-1: Wire MTL4 wide-tile kernels into production dispatch
Currently the production dispatcher (line 13164+) routes to classic Metal
kernels (which have the mc[8] bug). My MTL4 fixed kernels exist but aren't
used. Wiring requires:
- Add `if (g_use_mtl4_routed_mm)` switch in dispatcher
- Per-pipeline shmem_bytes table (8/16/32 KB)
- Verify ICB compatibility (MTL4 pipelines + classic ICB?)
Risk: medium (touches production hot path). Reward: actual fix in
production prefill correctness AND advertised wide-tile speedup.

### I-2: MoE matmul ICB capture (Phase 7+ continuation of #664)
Currently #664 [completed] shipped the design memo. Execution: capture the
sequence {map0, routed_mm_gate, routed_mm_up, swiglu_weight, routed_mm_down,
sum6} into one MTLIndirectCommandBuffer per layer. Replay per token saves
~60µs × 6 dispatches = 360µs per layer. For 60 layers, ~22ms per token.
Risk: high (ICB lifecycle subtle, segfault history on similar work — see
#545). Reward: 20+ t/s decode improvement.

### I-3: Session-persistent ICB cache (#673)
The current ICB record→replay rebuilds the ICB per layer per call. A session
cache keyed on (layer, tile_n, expert-mask-hash) would reuse the same ICB
across calls. Risk: medium. Reward: amortize ICB encoding (~10-50µs) across
calls.

### I-4: MTL4-native ICB (#673 alternative)
MTL4 has different ICB primitives (MTL4ArgumentTable + MTL4CommandBuffer).
Currently my MTL4 ports work via direct command-buffer encoding, not ICB.
Building a parallel MTL4ICB infrastructure would unify the path. Big swing,
months of work.

## Highest-leverage rewrites (deferred Round 1+)

### R-1: Wide-tile MSL consolidation (Round 1 deferred)
12 pipelines → 4 NR1-parameterized builders. -2400 LOC.

### R-2: Canary unification
12 canaries × 150 LOC → 1 unified harness + per-quant table. -1500 LOC.

### R-3: Drop dead wrappers
ds4_gpu_routed_mm_pipeline, ds4_gpu_routed_mm_f16_rhs_pipeline unused
since 9ca9013. -50 LOC, removes compiler warnings.

## Top picks if forced to choose 3 for next session

1. **I-1** (wire MTL4 wide-tile into production dispatch) — converts shipped
   code into actual production fix; high leverage; medium risk.
2. **Q-1** (f16-output family via corrected pattern) — completes the
   routed mul_mm_id family with preemptively-correct kernels.
3. **R-1** (wide-tile MSL consolidation) — large LOC reduction; demonstrates
   the "generality > kludge" pattern at scale.

ICB I-2/I-3/I-4 deferred — higher risk, less single-session deliverable.

