# Route ICB record→replay — design for porting codex H1716 to DS4 Metal

Date: 2026-05-25
Inherits: H1716 (codex) "tinygrad DS4 real-dim route transplant" — 16.33× at 16 layers via TinyJit capture.

## Goal

Capture DS4's route kernel chain (softplus+sqrt → router_finalize → router_weights → router_weights_with_remap) into a per-layer MTLIndirectCommandBuffer (ICB). Replay across the 43 layers per token instead of re-encoding 4 dispatches × 43 layers = 172 dispatches per token.

## Why H1716 works

Codex's tinygrad TinyJit captures the route compute graph and replays it. The capture removes:
- Per-dispatch kernel launch overhead (~10 μs each)
- Argument re-encoding cost
- Command buffer construction overhead

At 16 layers × 4 dispatches/layer = 64 dispatches in baseline. Captured = ~1 graph execute. Per-dispatch overhead amortized.

DS4's current route path issues:
- Each layer creates its own command buffer via `ds4_gpu_command_buffer(&owned)` → `ds4_gpu_finish_command_buffer(cb, owned, "trim remap")`. Per-CB submission cost is high.
- 4 kernels × 43 layers × 1+ ms CB-overhead = 170 ms/token of pure CB cost. Current decode is 52 ms/token total, so this is wrong OR I'm misreading "owned" semantics.
- Most likely: `owned=0` means the CB is shared with the broader graph (`g_batch_cb`), so only one submission per token total. But each ENCODER (computeCommandEncoder) still has setup cost.
- Actual gain from ICB: removing per-encoder setup, not per-CB submission. Per-encoder overhead is ~100 μs each. 4 × 43 = 172 encoders × 100 μs = 17 ms/token saved.

That matches H1716's scaling (~16× when overhead-bound).

## Blocker (from May 25 self-test)

ICB doesn't have `setBytes`. The Metal API for ICB-recorded compute commands (`id<MTLIndirectComputeCommand>`) only supports:
- `setComputePipelineState:`
- `setKernelBuffer:offset:atIndex:`
- `setThreadgroupMemoryLength:atIndex:`
- `concurrentDispatchThreadgroups:threadsPerThreadgroup:`
- `dispatchThreadgroups:threadsPerThreadgroup:`

DS4's route kernels currently use `[enc setBytes:&layer_index ...]` for scalar args. ICB rejects this → segfault when the self-test tried it (ds4_metal.m:4216).

## Fix shape

Convert scalar args from `setBytes` to a per-layer MTLBuffer with offsets.

For `kernel_dsv4_router_weights_with_remap`:
- 3 scalar args: `layer_index, n_expert, n_tokens` = 12 bytes
- Pack into a single `dsv4_route_args` struct
- Allocate one MTLBuffer holding 43 packed args (one slot per layer)
- Fill at engine init when DS4_N_LAYER and n_tokens are known
- Each layer's ICB dispatch uses `setKernelBuffer:offset:layer_index * 12 atIndex:3`

## Minimal patch plan (port to DS4)

Phase 1 — args buffer plumbing (no ICB yet):
1. Add struct `dsv4_route_remap_args { uint32 layer_index; uint32 n_expert; uint32 n_tokens; }` to ds4_metal.m
2. Add `static id<MTLBuffer> g_dsv4_route_remap_args_buf` (global, 43 × 12 bytes)
3. At init: fill buffer with `{i, 256, n_tokens}` for i ∈ [0, 42]. **n_tokens varies per call** — fill per call instead, but n_tokens is constant within a token's forward, so update once per token at most. Actually n_tokens=1 for decode, varies for prefill. Add the n_tokens to the args struct value, fill on first use, lazy-rebuild on n_tokens change.
4. Modify the kernel .metal signature: `constant uint& layer_index [[buffer(3)]]` → `constant struct DSRouteRemapArgs* args [[buffer(3)]]` + dereference `args->layer_index, args->n_expert, args->n_tokens`. Or keep three separate args at indices 3,4,5 each as buffer offsets into the same big buffer.
5. Update the dispatch call: replace 3 `setBytes` with `setBuffer:g_dsv4_route_remap_args_buf offset:(layer_index * 12) atIndex:3`. n_expert at offset+4, n_tokens at offset+8.
6. Verify numerical match against baseline.

Phase 2 — ICB record/replay:
1. Add `static id<MTLIndirectCommandBuffer> g_dsv4_route_remap_icb` (one ICB, 43 commands)
2. Allocate at init via `[g_device newIndirectCommandBufferWithDescriptor:... maxCommandCount:43 options:...]` with `inheritBuffers:NO, inheritPipelineState:NO, supportRayTracing:NO`
3. Set encoded commands: for layer=0..42, get `id<MTLIndirectComputeCommand> cmd = [g_icb indirectComputeCommandAtIndex:layer]`
4. Each command: `setComputePipelineState`, `setKernelBuffer` (×4: inverse_table, selected, weights, args at offset), `setThreadgroupMemoryLength`, `concurrentDispatchThreadgroups`
5. On replay: `[blitEncoder optimizeIndirectCommandBuffer:g_icb withRange:...]` once, then in compute encoder: `[enc executeCommandsInBuffer:g_icb withRange:NSMakeRange(layer, 1)]` per layer.

For batched execution across 43 layers: `[enc executeCommandsInBuffer:g_icb withRange:NSMakeRange(0, 43)]` — but this requires the inter-layer dependencies (selected, weights buffers) to be the SAME across layers, which they aren't in DS4 (each layer has its own selected/weights tensor allocation). So per-layer execution it is, but it still removes per-dispatch encoding overhead.

## Risks

1. **Buffer aliasing**: ICB-recorded commands hold pipeline state but not buffer contents. The route uses `selected, weights` buffers that vary per-token (their CONTENTS change between tokens via prior route step). Since ICB stores the BUFFER POINTER + offset, and contents are written before execute, this should work — but verify on the smoke test.

2. **threadgroup memory**: `setThreadgroupMemoryLength:n_expert * sizeof(float)` is constant (256 × 4 = 1024 bytes). Should work in ICB.

3. **Per-token n_tokens variability**: prefill has n_tokens=1..2048 (chunks), decode has n_tokens=1. Either bake n_tokens=1 ICB for decode (90% of work) and keep prefill on encoder path, or rebuild ICB args when n_tokens changes.

4. **Renormalize boolean**: dispatch already handles inside the fused kernel (per H541), so no extra branch needed.

## Smoke test before integration

1. Build patched binary with the args-buffer conversion (no ICB yet)
2. Run AIME P01 — verify output matches pre-patch
3. Add ICB recording for `router_weights_with_remap` only
4. Run AIME P01 — verify output still matches
5. Time prefill + gen — should see 5-20% win in route-bound portion

## Expected wins

Per-token DS4 decode currently: 52 ms (19.35 t/s baseline today).
Route overhead per token (estimate from H1716 analog): 4 dispatches × 43 layers × ~100 μs = 17 ms.
If ICB removes 80% of that: 14 ms saved → 38 ms/token → **26.3 t/s** (36% gen speedup).

Lower-bound: if route IS already efficient inside DS4's batched CB and the wallclock is mostly attn+MoE compute, we'll see <5% win. That's the falsifier.

## Decoupling from H1717

H1717 (polar dequant) is independent of this work. ICB recording captures kernel DISPATCH; polar dequant changes WEIGHT FORMAT. They don't conflict. Can ship in either order; ICB first because it's smaller scope and gives an immediate measurable win.

## Files to touch (estimated)

- `ds4_metal.m`: ~150-300 line patch (args buffer + ICB scaffolding + dispatch routing)
- `metal/dsv4_misc.metal`: ~10 line kernel signature change (constant arg → buffer pointer)
- `ds4_pillars.c`: remove scaffold stubs, implement real ICB API on top of new ds4_metal.m primitives
- `ds4_pillars.h`: add args struct typedef

Total: ~400-500 line patch. Single commit, env-gated by DS4_ICB_ACTIVE.
