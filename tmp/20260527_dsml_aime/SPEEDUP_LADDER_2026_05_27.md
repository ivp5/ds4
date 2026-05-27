# 2-3 OOM speedup ladder — DS4 decode hot path

silv 2026-05-27: "review your thinking based on the most recent
understanding, refine for an order of magnitude higher accuracy
which should enable to sense small aberrations unnoticed before...
look for 2-3 order of magnitude speedups along the way".

## Where time goes per token (current decode baseline)

DS4 trim50 IQ2_XXS, M1 Max, full Metal phases-auto: ~19 t/s ≈ 53 ms/token.

Per-token kernel call inventory (estimated):
- 4 router-cycle: ~50µs encoder + ~10µs compute = ~240µs total
- 18-774 routed-MoE matmul: ~50µs encoder + ~30µs compute × N — dominates at high N
- 43 layers × ~10 misc kernels = ~430 kernels × ~50µs = ~21ms
- 1 final lm_head: ~5ms compute
- CB rebuild + commit + wait: ~3ms

Of 53ms/token: dispatch encoding ~30ms (56%), compute ~23ms (44%).

## Path A revisited — why the cross-CB wire is hard

Examining the actual router_weights_one call site (ds4_metal.m:14323-14336):

```objc
if (!ds4_route_weights_one_dispatch(cb, ...)) {  /* classic ICB fall-through */
    enc = ds4_gpu_compute_encoder(cb);  /* cb is id<MTLCommandBuffer> CLASSIC */
    [enc setComputePipelineState:router_weights_pipeline];
    [enc setBuffer:...];  /* 3× */
    [enc dispatchThreads:MTLSizeMake(6,1,1) threadsPerThreadgroup:MTLSizeMake(6,1,1)];
    ds4_gpu_end_compute_encoder(cb, enc);
}
```

**Blocker**: `cb` is `id<MTLCommandBuffer>` (classic) — from `g_queue`. MTL4
dispatch needs `id<MTL4CommandBuffer>` from `g_polar_queue`. These are
SEPARATE command queues with separate dependency tracking.

To dispatch MTL4 mid-classic-CB requires:
- Insert MTLSharedEvent before MTL4 dispatch (classic queue waits)
- Submit MTL4 CB on g_polar_queue
- MTL4 CB signals event when done
- Classic CB resumes after event

Each cross-queue event ≈ 100µs overhead. For a 6-thread kernel with ~10µs
compute, the event overhead exceeds the kernel cost by 10×. **Net regression.**

The earlier session note at ds4_metal.m:15104-15111 already saw this:
classic ICB for router_weights_one regressed gen from 19.1 → 17.1 t/s
because useResource×3 + executeCommandsInBuffer overhead exceeded the
encoder-setup savings on a 6-thread kernel. The MTL4 cross-queue path
would be worse.

**Conclusion**: pool wire on TINY kernels (router_weights_one) is
net-negative. The pool's value is on LARGE per-token-amortized
sequences, not per-call.

## The actual 2-3 OOM speedup ladder

### Level 1: 1-2× from existing infrastructure
- **MoE matmul MTL4 pipeline** (shipped this session, init-canary passes)
  + ICB Phase 7 capture against it.
  - At 774 MoE dispatches/token × 50µs encoder = 38ms/token encoding
  - ICB record-once + replay: cuts to ~5ms/token (the executeCommands overhead)
  - **Per-token decoder time: 53 → 25 ms (~2×)** if MoE encoding eliminated.

### Level 2: 5-10× from kernel fusion
- Fuse the router cycle 4-6 kernels into 1 mega-kernel
- Per-token router dispatches: 6 → 1 = 5× on that segment
- Cross-kernel data stays in registers/threadgroup memory (eliminates
  global memory roundtrip between dispatches)
- Risk: bigger kernel = bigger threadgroup memory = lower occupancy.
  Net win depends on whether saved encoding > occupancy hit.

### Level 3: 10-100× from session-persistent MTL4IndirectCommandBuffer
- Build ONE MTL4 ICB at session-open recording ALL per-token dispatches
- Per token: ONLY rebind changed buffer addresses (via persistent
  ArgumentTable, exactly what the pool sets up) + executeCommands
- Per-token CPU overhead: 53ms → potentially ~5ms (10×)
- This is the FULL MTL4 path silv's directive points at

### Level 4: 100×+ from speculative decode + tree search
- Speculative decode with draft model: K candidate tokens forward in
  parallel, verifier accepts/rejects
- Effective t/s scales with acceptance rate × K
- DS4's MTP speculative decode entry point exists (#418, completed)
- Current gen: ~19 t/s; with K=4 speculative @ 50% accept: ~38-76 t/s

### Level 5: ∞× via cached prefix activations
- Long-prompt scenarios: 1k-token prefill recomputed every generation
- Persistent KV cache + prefix-hash deduplication: prefill becomes O(1)
  on cached prefixes
- For ChatGPT-style workloads with shared system prompt: effectively
  bypasses prefill entirely (the codec_audit DB pattern from #590)

## What this session's MTL4 ports actually enable

The 13 ports shipped this session are FOUNDATIONAL for Levels 1-3:

1. **Level 1** unblocked: MoE matmul MTL4 pipeline (#682) +
   ArgumentTable pool (#679) ready for ICB Phase 7 wiring.

2. **Level 2** prepared: 4 router-cycle MTL4 pipelines exist; fusion
   target is now a coherent set of MTL4-pipelines to merge into one.

3. **Level 3** built: pool + MTL4 dispatch pattern is the substrate
   for session-persistent ICB. The pool's 99.9% hit rate proves
   ArgumentTable reuse works across many dispatches.

## Honest scope on Path A this turn

I attempted to wire pool into router_weights_one classic dispatch but
discovered the cross-queue blocker. The right next-turn move is NOT
to force the wire — it's:

1. **Skip Path A for tiny kernels**: 6-thread router_weights_one is
   below the threshold where MTL4-via-pool helps. Apply pool only to
   kernels with >100 threads of compute.

2. **Apply to MoE matmul instead**: 774 dispatches × 64 threads each
   is exactly the workload pattern where pool + MTL4 + ICB compounds.
   This is also where the 5× gen speedup target lives per
   INTEGRATION_MAP.md.

3. **Build session-persistent ICB on the MoE matmul path**: record
   once at model open, replay per token. The pool's pre-allocated
   ArgumentTables become the per-dispatch bindings inside the ICB.

## Refined understanding (silv's "1 OOM higher accuracy" framing)

What changed in my view this turn:

**Before**: thought the pool would help router_weights_one because
it's called every token and ArgumentTable allocation is expensive.

**After**: realized router_weights_one is so small (6 threads, ~10µs
compute) that ANY per-call overhead — pool, ICB, MTL4 cross-queue —
exceeds the kernel cost. The pool was tested at ~99.9% hit rate but
the saved alloc time (~20µs/call × 1000 calls = 20ms) is dwarfed by
the encoder rebuild overhead which the pool doesn't address.

**The correct heuristic**: pool/ICB/MTL4 optimizations help when
**compute_time_per_dispatch ≫ encoder_overhead_per_dispatch**. For
DS4:
- MoE matmul (~30µs compute, 50µs encoder): break-even — small win
- Routed MoE × 774 calls (group of dispatches): big win from batching
- Tiny router kernels (~10µs compute, 50µs encoder): NET REGRESSION

This refines the speedup theory: chase optimization at the LARGE-dispatch
end, not the small-dispatch end. The MoE matmul + ICB Phase 7 is the
correct next concrete deliverable, not router-cycle wiring.

## What to ship next

1. Full output canary for MoE matmul (validate the pipeline works on
   real data, not just init)
2. ICB Phase 7: capture MoE dispatch sequence into MTL4 indirect CB
3. Integrate into hot path with model inference A/B
4. Measure: target 2× gen speedup at level 1, then chase level 2-3

Levels 4-5 (speculative + cached prefix) are orthogonal capabilities;
they don't depend on the MTL4 ports and can be developed in parallel.
