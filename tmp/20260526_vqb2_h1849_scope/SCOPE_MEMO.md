# VQB2 + H1849 + inferguard cross-validation — scope memo

silv directive 2026-05-26: "try the vqb2 to fit memory + 20-30 tok/sec ...
check AMD box for codex's VQB2 ... apply H1849 top-k optimization to
ds4.c logit-compute site ... apply codec-side margin-consumption objective
per H1849 ... read ds4.c logit-compute site to scope top-k retrofit ...
cross-validate inferguard rescue on merged binary"

## Findings audit

### What's actually on AMD (silv said "use that")

Codex's VQB2 corpus inventory on `gpu_ssh@10.10.0.17:G:\LLM\tlp_codex_h1815\`:
- `packets_h1826_l25_k64_direct_vqb2_v3` — L25 only (gate/up/down), K=64 → ~125 MB
- `h1853_max_quality_vqb2_*` — max-quality test
- `h1856_l22_vqb2_packet_sweep` — L22 sweep
- `h1858_l35_vqb2_packet_sweep` — L35 sweep
- `h1860_l42_vqb2_packet_sweep` — L42 sweep

**Coverage: 4 of 43 layers (9%).** ~500 MB total VQB2 packets.

To run DS4-Flash with VQB2 instead of IQ2_XXS, need:
- Full 43-layer VQB2 corpus: codex has 4; missing 39 layers × ~50 MB ≈ 2 GB unencoded
- Local VQB2 encoder (`vq2d_encode_mlx_vqb1.py` exists but produces VQB1 not VQB2 bit-packed; would need extension)
- VQB2 C reader (have ds4_vqb1_reader, need to extend for bit-packing)
- VQB2 Metal kernel (does not exist — DS4 uses IQ2_XXS pair kernel currently)
- GGUF format integration OR runtime overlay (does not exist)

**Honest assessment**: silv's premise "fit into memory" is correct — full VQB2
DS4-Flash would be ~15-20 GB total, well within M1 Max 64 GB cap, avoiding
the phase-split + CPU-MoE fallback that currently caps gen at 2.34 t/s.

But the engineering to TEST this is multi-session work (~6-10 hours
focused). Not achievable in one continue.

### H1849 top-k retrofit at ds4.c logit-compute site

Located the lm_head matmul:
```
ds4.c:9257  matvec_q8_0_decode_scratch(logits, model, weights->output,
                                       scratch->output_norm, scratch);
```

Specs:
- `weights->output` = Q8_0 tensor, shape [DS4_N_EMBD=7168, DS4_N_VOCAB=129280]
- Cost per token: 7168 × 129280 = **926.5M dot products** at Q8_0
- Output: full 129280-vocab logit vector
- Sampler: `ds4_session_argmax_excluding` (greedy argmax)

H1849's optimization (per codex shift):
> "candidate margin scoring can use source top-k embedding rows. For
> top-2 margin this reduces repeated candidate scoring from
> 144*129280*4096 dot work to 144*2*4096, a 64640× arithmetic cut."

**Direct application to inference is non-trivial** because at inference
time we DON'T know the top-K in advance — finding the top-K IS the
purpose of the matmul.

Two viable approaches:

(a) **Two-pass approximation**:
   1. Cheap pass: low-rank lm_head approximation (e.g., rank-128 SVD) →
      top-K candidates (~7168 × 128 = 0.9M ops, 1000× faster)
   2. Exact pass: full Q8_0 matmul on top-K rows only (~7168 × K = 3.7M
      ops for K=512)
   Total: ~1.252× speedup of lm_head matmul (mostly the rank-128 SVD)

(b) **Watchlist sampling**:
   - Maintain a "hot watchlist" of frequently-emitted tokens
   - At lm_head, compute logits for: watchlist (~512 tokens) + random
     sample (~512 tokens) + last K context tokens (~16)
   - Argmax over the ~1040-token subset
   - Periodically refresh watchlist
   - Risk: misses rare-correct tokens; bounded by argmax stability

**Engineering scope**: ~2-4 hours focused work. Lower-risk than VQB2
swap. Returns ~1.2-2× lm_head matmul speedup. NOT the 10-15× silv wants.

### H1849 codec-side margin-consumption objective integration

This is OFFLINE TOOLING — selecting which codec packet to ship based on
margin-consumption, not rel-L2. Per codex H1849-H1850:
> "case rel-L2 prefers h1830_s42_n100k, but margin consumption prefers
> h1827_fit1m despite worse case rel-L2"

For DS4 deployment: would inform WHICH packet codex's selector ships per
layer. Codex already implements this in their analyzers; we'd need to
adapt to local DS4 deployment if we go the VQB2 route.

**Engineering scope**: contingent on VQB2 corpus being available
locally. Without the corpus, no packet selection happens. Currently
blocked.

### inferguard rescue cross-validation on merged binary

This IS tractable. Steps:
1. Run merged ds4-bench (just shipped 2.34 t/s baseline) on AIME P01-P10
   prompts with truth-collection
2. Apply `inferguard/aime_rescue.py` to the trace
3. Verify rescue protocol still produces same 8/10 ceiling as
   pre-merge

Caveat: inferguard's aime_rescue.py runs on the Qwen3.5-4B model output,
not DS4. The "cross-validate on merged binary" likely means:
- Verify the merged ds4-bench produces equivalent gen output to pre-merge
- Confirm the streaming progress doesn't disturb output determinism

This is mostly a regression test. ~30 min wall.

## Realistic path forward

### Tractable in next continue
1. **inferguard rescue cross-validation** — regression test the merged
   binary against Qwen3.5-4B baseline (AIME 2026 I corpus)
2. **VQB2 reader stub in C** — extend ds4_vqb1_reader to handle VQB2
   bit-packing (~1-2 hours focused work; can decode codex's L25 packet)
3. **Single-layer VQB2 canary test** — pull codex's L25 packets from AMD,
   verify decode correctness against ground truth

### Multi-session work
1. **Full DS4-Flash VQB2 encoding** — 43 layers × ~45s/layer ≈ 32 minutes
   of encoding + bit-pack pass (need to extend encoder for VQB2)
2. **VQB2 Metal kernel** — write or adapt kernel for routed-FFN dispatch
3. **DS4.c integration** — swap IQ2_XXS path → VQB2 path conditionally
4. **Full-stack smoke test** at target 20-30 t/s

### NOT tractable as silv described
- "20-30 tok/sec" goal cannot be reached without the full VQB2 corpus +
  Metal kernel + integration
- The 2.34 t/s baseline is current; even fitting in Metal (no CPU-MoE
  fallback) would likely give 5-10 t/s, not 20-30 t/s
- 20-30 t/s likely requires BOTH VQB2 fit AND specialized kernels with
  ICB record→replay AND possibly speculative decode (which is retired
  on M1 per shift #293)

## My next move

The cheapest meaningful step: **inferguard rescue cross-validation**.
This validates the merge didn't break the rescue protocol — defends
the 8/10 AIME 2026 ceiling against silent regression. ~30 min wall.

If silv wants the VQB2 path despite the multi-session scope, the
2-step entry is:
1. Pull codex's L25 packets from AMD (~125 MB)
2. Write minimal VQB2 reader extension to ds4_vqb1_reader

Both reachable in ~2 hours total but neither alone produces a
measurable speedup (single-layer reader is correctness-only without
the kernel + 39 other layers).

## Files

- This memo: tmp/20260526_vqb2_h1849_scope/SCOPE_MEMO.md
- Codex VQB2 packets: gpu_ssh@10.10.0.17:G:\LLM\tlp_codex_h1815\
- DS4-Flash IQ2_XXS: /Users/silv/cl/tlp/montyneg/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-...
- Smoke test baseline: tmp/20260526_ds4_smoke/SMOKE_TEST_PASSED.md
