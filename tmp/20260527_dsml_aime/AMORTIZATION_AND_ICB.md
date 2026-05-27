# MTL4 ArgumentTable amortization + ICB optimization design

silv 2026-05-27: "also implement ArgumentTable amortization and relevant
icb optimizations".

## What shipped this turn

### ArgumentTable pool (task #679)

`ds4_metal.m` adds a free-list pool keyed by `max_buffer_bind_count`:

```c
static id<MTL4ArgumentTable> ds4_mtl4_pool_acquire(NSUInteger n_bindings);
static void ds4_mtl4_pool_release(id<MTL4ArgumentTable> at, NSUInteger n_bindings);
void ds4_gpu_mtl4_pool_stats(uint64_t *acq, uint64_t *alloc, uint64_t *rel);
```

5 buckets: 3, 4, 6, 8, 16 bindings. Each holds up to 8 cached tables.
`acquire()` returns a cached table if available; allocates fresh
otherwise. `release()` returns to free list (bindings persist).

### Validation: `--amortized-canary`

`router_weights_one` rerun N times back-to-back through the pool:

```
n_iter=100:  acquires=100  allocs=1  hit_rate=99.0%  gpu[0]=0.2500 ✓
n_iter=1000: acquires=1000 allocs=1  hit_rate=99.9%  gpu[0]=0.2500 ✓
```

**Cost amortized**: 1× ArgumentTable allocation (~10-50µs once) + N× cheap
`setAddress` calls. Per-call cost asymptotically the cost of binding-update.

### Architectural impact

For the per-token decode hot path:
- 4 router-cycle kernels × tokens_per_run × ~20µs save per dispatch
- 18-774 MoE-matmul dispatches × ~20µs save (once MoE matmul ports)
- All other MTL4 kernels in the future

Conservative estimate at 8 router + 50 misc + 774 MoE = 832 kernels/token.
At 20µs save per dispatch: 832 × 20µs = **~16 ms / token saved** in
ArgumentTable allocation alone after amortization warm-up. At baseline
gen of ~50 tokens/sec ≈ 20 ms/token, that's a substantial fraction.

(Actual GPU dispatch wall time may not be entirely allocation-bound; the
pool removes ONE source of overhead. Other sources — encoder rebuild,
buffer rebinding, residency-set churn — remain to be amortized.)

## ICB optimization paths

### Current ICB state (classic-MTL Phases 1-6)

Already shipped:
- Phase 1-2: route_weights_with_remap (#553, #555)
- Phase 3: topk_mask (#557)
- Phase 4: softplus_sqrt_f32_4 (#558)
- Phase 5: router_finalize_one (#559)
- Phase 6: router_weights_one (#560)

These use **classic** `MTLIndirectCommandBuffer`. Record at model-open;
replay per-token. Saves encoder rebuild + dispatch-shape encoding cost.

### MTL4 ICB integration paths

Three paths forward, ordered by leverage:

#### Path A — MTL4ArgumentTable + classic ICB (PARTIAL — recommended next)

The MTL4 ports SHARE pipelines with the classic-MTL path; the existing
classic ICBs (Phases 1-6) can continue to drive them. The MTL4 path
provides:
- `device const args_ptr` storage class (already shipped)
- ArgumentTable amortization via the pool (shipped this turn)

**To wire**: replace each Phase N's classic-MTL encoder with an MTL4
encoder that uses pool-acquired tables. The classic ICB schema doesn't
change; only the per-dispatch encoder swaps.

Cost: ~1 turn per phase to integrate the pool into the existing ICB
record/replay code. Risk: low — pool is correctness-preserving.

#### Path B — MTL4IndirectCommandBuffer (FULL)

Apple's MTL4 API has `id<MTL4IndirectCommandBuffer>` — the MTL4-native
version of indirect commands. Different API surface than classic ICB.
Provides:
- Per-dispatch ArgumentTable indirection (table_id baked into ICB)
- Encoder-free dispatch (record once, run forever)
- Native MTL4 compiler optimizations

To use: build an `id<MTL4IndirectComputeCommand>` per dispatch, populate
its argument table reference at record time. Per-token replay just
issues `executeCommandsInBuffer`.

Cost: 2-3 turns to design + ship for the router cycle alone (replaces
Phase 1-6 ICBs entirely with MTL4 versions). Risk: medium — API less
mature than classic ICB; debugging cycle uncertain.

#### Path C — ICB Phase 7 for MoE dispatch

Independent of MTL4. The routed MoE dispatch sequence (~774 calls per
token) is the largest remaining ICB-uncaptured surface. Phase 7 would:
- Record N expert-slot dispatches at model open
- Per-token replay sets per-expert buffer offsets via indirectArgumentBuffer
- Dispatches MoE matmul N times (one per selected expert)

This was design-memoed earlier (NEXT_INTEGRATION.md). Implementation
remains a separate effort because the MoE matmul kernel itself isn't
yet MTL4-ported (next-turn target).

### Recommended sequence

1. **NEXT TURN**: integrate the pool into one existing classic-MTL ICB
   phase (e.g. Phase 6 router_weights_one) — proves Path A end-to-end
2. **TURN +1**: ship MTL4 MoE-matmul port (the per-token hot path)
3. **TURN +2**: ICB Phase 7 captures MoE dispatch sequence against the
   new MTL4 pipeline
4. **TURN +3**: Path B exploration if Path A leaves meaningful headroom

## Honest scope

The pool by itself (this turn) is a leaf-level optimization. The real
speedup multiplier is when the pool integrates with the existing ICBs
and the new MoE-matmul port lands. Without those, the pool is just a
memory-allocation optimization that saves ~10-50µs per dispatch — useful
but not transformative.

**This turn shipped**: pool infrastructure + validation. Path A wiring,
Path C MoE port, and ICB integration are next-turn items.

## Files changed this turn

- `ds4_metal.m`: 5 pool buckets, acquire/release/stats functions,
  amortized canary using pool (~150 lines)
- `ds4_gpu.h`: public `ds4_gpu_mtl4_pool_stats` + amortized canary decl
- `ds4_cli.c`: `--amortized-canary` CLI entry

Plus the ratio4_shift_f32 port (task #678) — 16/81 MTL4 pipelines now.

## CLI verification

```
./ds4 --amortized-canary 100
ds4: router_weights_one amortized canary n_iter=100 acquires=100 allocs=1
     releases=100 pool_hit_rate=99.0% gpu[0]=0.2500 max_abs=0.0000e+00

./ds4 --amortized-canary 1000
ds4: router_weights_one amortized canary n_iter=1000 acquires=1000 allocs=1
     releases=1000 pool_hit_rate=99.9% gpu[0]=0.2500 max_abs=0.0000e+00
```

Pool achieves design goal: 1 allocation per pipeline-binding-count,
infinite reuse, correctness preserved.
