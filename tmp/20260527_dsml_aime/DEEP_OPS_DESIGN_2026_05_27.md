# Deep optimizations — design + first-step scaffold

silv 2026-05-27 directive: "continue mtl4 port queue, Wire Level-1 A/B
properly, work on Cached prefix activations and Spec-decode tree-search
activation, then on Level-3 session-persistent ICB on router cycle.
read recent codex work - the agent is working on newly encoded/quantized
vq2b gguf on amd box - will transfer here when ready and let you know
where it is".

## Codex catch-up this turn (relevant findings)

**H2084** (codex, 2026-05-27): H2074 no-hub `[55,71,188,231,254]` passes
the FULL DSML+AIME product-mode generation gate. 72/72 byte-identical
generated steps across all 9 prompts. Codex declares: "stop replaying
L09 and move to GGUF emit, MTL4 MoE matmul, and ICB routed-MoE dispatch."

**H2085** (codex audit of my work): caught the IQ2 dequant 2× scale
bug already (which I fixed via H2086 patch this session). Confirmed my
ICB Phase 7 wire direction is correct.

**VQB2 GGUF on AMD**: codex is encoding route-aware VQB2 sidecars at
high-pressure scope (`route_sparse_full_row`). Per H1955, K16 gate/up
+ Hadamard16 + DCT16 down beats raw adjacent-pair VQB2 by 3-3.5% on
DS4 L09 routed FFN. Transfer status: silv will signal when ready.

## Wire Level-1 A/B properly — BLOCKED status

The MoE matmul ICB wire is in `ds4_gpu_encode_mul_mm_id_fp16_pair_swiglu`
(ds4_metal.m:13396+). This function only fires when:
1. `DS4_HOT_FP16_KERNEL=1` env set
2. hot-store active + layer fully pinned

With M1-mandatory `--cpu-moe`, the routed-MoE runs on CPU → the wire's
GPU path NEVER FIRES → measured A/B was page-cache warming, not ICB.

**Two paths to unblock**:

(a) **Wait for VQB2 GGUF transfer**. The codec arc shifts the MoE
   weights to VQB2 (52GiB target). With VQB2 fit, GPU MoE may run
   without `--cpu-moe` directly. ICB wire fires automatically.

(b) **Phase-split + selective hot-store**. Use `--prefill-metal-phases auto`
   (mandatory for IQ2_XXS on M1) AND pin a few layers via hot-store at
   FP16. Per ds4.c:13452, hot-store needs all 256 experts pinned at FP16
   for a layer. Per-layer pin: ~256 × FP16 weights ≈ ~256 × ~2.75 MiB =
   ~704 MiB per layer. Pinning 1-2 layers stays well under M1 wired-cap.
   That gives a real-workload A/B with ICB on/off measurable.

**Recommended next-turn action**: wait for VQB2 transfer (path a — codec
arc is closer to deploy). If VQB2 transfer is multi-day, ship path b
as a fallback A/B harness.

## Spec-decode tree-search activation (Item 4)

**Current state** (per task #454, #633, ds4.c grep):
- MTP draft entry in ds4.c works: `e->mtp_ready`, `e->mtp_draft_tokens`
- Existing API: `ds4_engine_mtp_draft_tokens(engine)` returns active draft K
- ds4_metal.m:15394 `metal_graph_eval_mtp_draft_from_hc()` is the GPU eval
- silv 20-30 t/s target was the validation gate per #633

**What's NOT in place**:
- Tree-search variant (linear K candidates only)
- Acceptance rate measurement on AIME corpus
- Verifier prefix-match for tree topology

**Tree-search architecture** (extends current linear MTP):

```
Linear spec-decode (current):
  draft N tokens forward
  verifier eval all N+1 positions (anchor + draft)
  accept longest matching prefix

Tree spec-decode:
  draft N tokens forward at width B (B branches × N depth)
  ↓
  candidate tree:
       anchor
      /  |  \
     a1  a2  a3      (width B=3 at depth 1)
    /|   |   |\
   b1 b2 b3  b4 b5   (next depth)
  ↓
  verifier eval all N×B nodes in one parallel batch
  accept longest matching root-to-leaf path (greedy on logits)
```

**Acceptance scaling**:
- Linear K=4 at 50% accept: avg 2 accepted/iter → 2× t/s
- Tree B=4 K=4 at 30%/branch: ~0.7^4 × 4 ≈ 0.96 accepted (worse!)
- Tree B=2 K=8 at 50%/branch: ~0.5^8 × 2 paths ≈ 0.008 (much worse)

**Reality**: tree expands SEARCH BREADTH but each branch has independent
acceptance rate. Tree only wins when verifier compute is amortized over
many candidates AND acceptance per branch is high (low entropy decode).

Per Apple/Anthropic/DeepMind speculative work: tree beats linear at
**LOW-ENTROPY DECODE POSITIONS** (e.g. after `\boxed{` where one digit
is heavily favored). At HIGH-ENTROPY positions, linear is optimal.

**Concrete spec-tree implementation** (2-3 turns):
1. Add `mtp_draft_tree_width` field to engine state
2. In `metal_graph_eval_mtp_draft_from_hc`, dispatch B times instead of 1
3. Verifier `ds4_gpu_capture_verifier_prefix` extends to tree topology
4. Activation gate: only use tree when top-1 anchor confidence > 0.8

**Effort**: 2-3 turns. Validation: AIME 2026 K=4 corpus A/B.

## Cached prefix activations (Item 3)

**Architecture**:

```c
typedef struct {
    uint8_t prefix_hash[32];  /* SHA-256 of full tokenized prefix */
    uint32_t n_prefix_tokens;
    uint64_t mtime;            /* invalidation if model changes */

    /* Per-layer state snapshot — one slot per active layer */
    ds4_gpu_tensor *kv_state[DS4_N_LAYER];    /* compressed KV cache */
    ds4_gpu_tensor *raw_kv_state[DS4_N_LAYER]; /* SWA window state */
    ds4_gpu_tensor *residual_final;            /* output of last layer */
    ds4_gpu_tensor *router_state[DS4_N_LAYER]; /* compressor state */
} ds4_prefix_cache_entry;
```

**Hash design**: SHA-256 over `tokenize(prefix_text)`. Stable across runs.
First-token speculation isn't cached — only fully-prefixed prefill state.

**Storage**:
- In-memory LRU of K=8 entries (~4-8 GiB depending on prefix length)
- Disk-backed at `~/Library/Caches/ds4/prefix_cache/<hash>.bin`
- Memory-mapped restore: zero-copy reload on cache hit

**Restore flow**:
1. Tokenize input prefix
2. Hash → check LRU + disk
3. If hit: restore KV/residual/router state directly to GPU buffers
4. Skip prefill; jump to gen loop immediately

**Hit case wall-time**:
- Baseline 1k-token prefill at 1.7 t/s: ~590s prefill + 10s gen = 600s
- Cached restore: ~1s deserialize + 10s gen = 11s
- **~54× speedup** for cached prefix workloads (ChatGPT-style fixed prompts)

**Effort**: 3-5 turns. Phases:
1. Hash + serialize per-layer state to disk (1 turn)
2. Restore + invalidation logic (1 turn)
3. LRU + disk-backed storage (1 turn)
4. Integration with prefill bypass at engine load (1 turn)
5. AIME corpus A/B + cache hit/miss telemetry (1 turn)

## Level-3 session-persistent ICB on router cycle

**Current state** (this session):
- 5/5 router cycle MTL4 pipelines exist
- ArgumentTable pool with 99.9% hit rate
- Codex H1777 confirms MTL4 + classic ICB work on M1
- Classic MTLIndirectCommandBuffer accepts MTL4 pipelines (both expose
  MTLComputePipelineState)

**Design**:

```c
typedef struct {
    id<MTLIndirectCommandBuffer> router_icb;
    /* Persistent buffer bindings — re-acquired per-token if pointers
     * change but pool keeps them warm */
    id<MTLBuffer> args_probs_softplus;  /* persistent if probs stable */
    id<MTLBuffer> ...;
} ds4_router_session_icb;

static int build_router_session_icb(void) {
    /* Record 5 dispatches into ICB:
     * 0. softplus_sqrt (logit → prob)
     * 1. router_finalize_one (top-K select)
     * 2. topk_mask (-INF fill)
     * 3. topk_mask_scatter (zero selected rows)
     * 4. router_weights_one (norm weights)
     *
     * Each cmd_i gets its own setComputePipelineState + setKernelBuffer
     * + dispatchThreadgroups. ICB persists across all per-token calls.
     */
}

static int dispatch_router_session_icb(id<MTLCommandBuffer> cb, ...) {
    /* Per-token: useResource × N, then executeCommandsInBuffer range[0..5] */
}
```

**Effort**: 2-3 turns:
1. Build router_session_icb at first-token call (sig-cache on buffers)
2. Replace 5 individual dispatches with executeCommandsInBuffer
3. A/B measurement vs current per-call dispatch

**Risk**: per codex H1779/H1780, persistent encoder at N=6 didn't speed
up over per-call. But router cycle here is N=5 per token, so similar
amortization profile. If wall-time doesn't move, this is informational
not deployable.

**Higher-density target**: replace per-LAYER per-token MoE matmul +
down-projection + per-layer norm into ONE bigger ICB. That's N=~430
dispatches → 1 executeCommands. Amortization curve flips at high N.

## Continue MTL4 ports — incremental next batch

| # | port | dispatch density | priority |
|---|------|------------------|----------|
| done | qkv_rms_norm_f32_4 | 43×/token | ✓ this turn |
| 1 | rms_norm_fuse_impl (templated) | 43×/token | templated, harder |
| 2 | kernel_soft_max | per-attention | medium |
| 3 | kernel_soft_max_4 | per-attention | medium |
| 4 | kernel_dsv4_indexer_score_one_direct | per-attention | larger |
| 5 | kernel_dsv4_hc_expand | per-layer | medium |
| 6 | kernel_dsv4_fp8_kv_quantize_f32 | per-decode | needs dequant helper |
| 7-50+ | MoE/dense variants | various | mass port |

Sustainable pace: 2-3 ports per session given other architectural work.
At session start (7/81 → 22/81 this session = +15 ports), reaching
50% MTL4 (40/81) is ~6-8 more sessions of mass porting.

## What this turn shipped vs deferred

| item | shipped | deferred |
|------|---------|----------|
| MTL4 ports | qkv_rms_norm (#685) | router_finalize was prior turn |
| Level-1 A/B | blocked analysis | VQB2 transfer + phase-split path |
| Spec-tree | design memo | 2-3 turn implementation |
| Cached prefix | design memo + 5-phase plan | 3-5 turn implementation |
| Session-persistent ICB | design memo | 2-3 turn router cycle build |

## Order recommendation by cumulative benefit

When VQB2 GGUF arrives → top priority. Codec arc deploys.

Until then, parallel tracks:
1. Continue MTL4 ports (incremental, no blocker)
2. Build cached prefix activations (biggest absolute speedup;
   independent of MTL4)
3. Spec-tree variant (2-3 turn, leverages existing #418)
4. Session-persistent ICB router cycle (architectural learning, may
   not show wins at N=5)

Cached prefix is the SINGLE highest-leverage item that doesn't depend
on MTL4 or VQB2. If silv has ChatGPT-style workloads (shared system
prompt + variable user message), the ∞× prefill speedup is the most
visible win.
