# Roadmap: deep optimizations beyond MTL4 port mass

silv 2026-05-27: directive includes "continue the mtl4 ports queue,
router cycle kernel fusion, session-persistent MTL4IndirectCommandBuffer,
speculative decode + tree search (#418 entry exists), cached prefix
activations for long-prompt etc."

This memo lays out the design for each multi-turn item. The MTL4 port
queue continues incrementally; these are larger architectural moves.

## Item 1: Router cycle kernel fusion (Level 2 of speedup ladder)

**Current state**: 6 separate Metal kernels run per router cycle per
token:
- softplus_sqrt_f32_4
- router_finalize_one (256-thread bitonic sort)
- router_weights_one (6-thread weight norm)
- router_weights_with_remap (6-thread expert-remap)
- topk_mask (~256-thread mask fill)
- topk_mask_scatter (~256-thread scatter)

**Fusion target**: collapse into ONE mega-kernel that processes the
entire router pipeline for one token in one dispatch. Per-token
savings:
- 6 encoder rebuilds → 1
- 6 setup-tear-down cycles → 1
- Inter-kernel data stays in threadgroup memory (no global memory
  round-trip)

**Risk**: bigger kernel = bigger threadgroup memory footprint = lower
SM occupancy. Need to measure register pressure + verify M1 Max can
fit the fused kernel within hardware limits.

**Effort**: 2-3 turns. Build incrementally:
1. Fuse softplus_sqrt + router_finalize_one (probability→selection)
2. Add router_weights_one to that
3. Then topk_mask + scatter

Each fusion step ships an MTL4 pipeline + canary + diagnostic A/B.

**Expected gain**: at ~50µs encoder overhead per kernel × 5 saved
calls = ~250µs/token = ~0.5% at 53ms/token. Less than expected because
router cycle is small fraction of total per-token time. The win is
in keeping the data in registers/threadgroup.

## Item 2: Session-persistent MTL4IndirectCommandBuffer (Level 3)

**Concept**: record ALL per-token dispatches into ONE MTL4 indirect
command buffer at session-open. Per token: ONLY rebind buffers that
change (via persistent ArgumentTable pool that the pool API already
provides) + executeCommandsInBuffer for the full sequence.

**Per-token operations to capture** (estimated ~430 dispatches):
- 6 router cycle kernels (per layer × 43 = 258)
- 6 routed-MoE matmul (gate+up+SwiGLU fused) (per token × 43 = 258)
- 6 routed-MoE down-projection (per token × 43 = 258)
- 1 per-layer norm + residual add (× 43 = 43)
- ... attention prefill + flash attn ...
- 1 final lm_head matmul

**Architecture**:
```c
typedef struct {
    id<MTLIndirectCommandBuffer> per_token_icb;  /* recorded once */
    id<MTL4ArgumentTable> *per_layer_arg_tables; /* pooled */
    /* signature cache: per-buffer-binding gpuAddress + offset */
    ds4_per_token_signature *sig_cache;
} ds4_session_persistent_icb;
```

**Per-token replay flow**:
1. Update ArgumentTable bindings for buffers that changed (~50
   setAddress calls — cheap)
2. Set residency on all model buffers
3. executeCommandsInBuffer with range[0..N_total_dispatches]
4. Single commit, single wait

**Effort**: 4-6 turns. This is the BIG architectural ship that
delivers the Level-3 10-100× speedup. Needs:
- Per-pipeline buffer-binding metadata (already have for our 20
  ported pipelines)
- Signature cache mechanism
- Rebuild trigger on shape/buffer changes
- End-to-end validation against current decode output

**Apple constraint**: MTL4IndirectCommandBuffer may not exist on M1
Max (pre-M5 disables MTL4 tensor API per the boot message). The
classic MTLIndirectCommandBuffer DOES work on M1 — see Phases 1-6.
Path B (full MTL4 ICB) may need M2+/M5 hardware.

**Pivot**: use classic MTLIndirectCommandBuffer with the existing
classic-MTL pipelines for session-persistent recording. The pool +
MTL4 pipelines we built are still useful for ArgumentTable persistence
WITHIN the classic ICB recording.

## Item 3: Speculative decode + tree search (Level 4 — 100×+ effective)

**Existing**: task #418 (silv 2026-05-23) has the spec-decode entry
point in ds4.c. Status: completed. Per task #633 ("MTP speculative
decode end-to-end bench"), the silv 20-30 t/s target was the validation
gate.

**Tree search extension**: instead of K linear candidates, build a
tree of M branches × K depth = M×K candidate tokens forward in
parallel. Verifier accepts the longest matching prefix.

**Expected throughput at acceptance**:
- Baseline gen: ~4-19 t/s (config-dependent)
- Spec K=4, accept 50%: 8-38 t/s effective
- Tree M=4 K=4, accept 30%: 12-57 t/s effective

**Effort**: 2-4 turns. Concrete next-step:
1. Read codex H1430+ for spec-decode design notes
2. Verify task #418 entry point still compiles
3. Add tree-search variant of spec-decode
4. A/B baseline vs spec vs spec-tree on AIME 2026 corpus
5. Confirm accuracy unchanged (spec doesn't change behavior, only
   speed)

## Item 4: Cached prefix activations (Level 5 — ∞× for long-prompt)

**Concept**: for long-prompt scenarios where multiple generations
share a 1k-token system prompt, save the per-layer KV cache + final
activation state for that prefix, reload on subsequent gens. Prefill
becomes O(1) instead of O(prefix_len).

**Architecture**:
```c
typedef struct {
    uint8_t prefix_hash[32];  /* SHA-256 of tokenized prefix */
    uint32_t n_prefix_tokens;
    /* Per-layer state at end of prefix */
    ds4_gpu_tensor *kv_cache[DS4_N_LAYER];
    ds4_gpu_tensor *residual_final;
    /* Last activation state ready for first gen token */
    ds4_gpu_tensor *active_state;
} ds4_prefix_cache_entry;
```

**Implementation**:
1. After prefill completes, snapshot all per-layer state to disk
2. On subsequent gen: hash incoming prompt prefix
3. If matches: restore state, skip prefill, start gen immediately
4. If no match: normal prefill, optionally save new prefix

**Effort**: 3-5 turns. Significant disk + memory accounting.

**Expected gain**: at 1k-token prefix taking ~2 minutes to prefill at
1.7 t/s, restoring from cache makes that ~0 seconds. Effective
prefill speedup: ∞× for cached prompts. For ChatGPT-style workloads
with shared system prompt + variable user message, the per-call
prefill cost approaches just the user-message portion.

## Item 5: MTL4 port queue (incremental, no architectural change)

**Current**: 20/81 MTL4 pipelines (24.7%) shipped this session.

**Remaining high-leverage targets** (in priority order):
1. **router_finalize_one** (256-thread bitonic + hash mode + bias)
2. **qkv_rms_norm_f32_4** (6 buffers, attention norm — fires per layer)
3. **soft_max + soft_max_4** (softmax.metal, templated)
4. **15 more moe.metal kernels** (mul_mv_id variants, attn_out)
5. **dsv4_hc.metal** (head-compress, 7 more kernels)
6. **dsv4_misc.metal** (indexer variants, attention heads)

**Mass-port strategy**: each kernel follows the documented template.
~150 lines per port + canary. 3-5 kernels per session is sustainable.

**End state**: 80/81 MTL4 (one templated kernel may stay classic) at
~12-15 more sessions of mass porting.

## What this session shipped toward each item

| item | progress | unblockers |
|------|----------|------------|
| Router fusion | None yet | All 4 router-cycle MTL4 pipelines exist (#671-#673, #677) — coherent set to fuse |
| Session-persistent ICB | Pool + Phase 7 wire | Need GPU MoE actually firing in benchmark |
| Speculative decode | Entry point exists (#418) | Activate + measure on AIME corpus |
| Cached prefix | None yet | Architectural design needed |
| MTL4 port queue | 13 ports shipped (+ pool + MoE matmul pipeline) | Continue at ~3-5/session |

## Honest scope

This session focused on the FOUNDATIONAL ports + pool + MoE matmul
pipeline + ICB Phase 7 wire. The MEASURED speedup from these is not
yet end-to-end validated (the A/B I ran was invalidated by --cpu-moe
config bypassing the wired path).

Items 1-4 each require multi-turn dedicated work + careful end-to-end
validation. The MTL4 port queue (item 5) is the steady incremental
work that's mechanically tractable.

Next-turn focus options (recommend in this order):
A. Set up phase-split GPU MoE config + hot-store + measure real
   ICB effect (closes the A/B loop from this session)
B. Continue MTL4 ports queue (router_finalize_one + qkv_rms_norm)
C. Start router fusion (Item 1) — first concrete architectural win
D. Spec-decode tree-search (Item 3) — biggest end-to-end leverage
