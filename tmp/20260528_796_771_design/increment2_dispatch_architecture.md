# #796 Increment 2 — dispatch architecture finding

## The wiring problem

Increment 1 shipped `mul_mv_bf16_f32_mtl4` + canary. Both pass. The kernel is correct.

Wiring it to production requires re-architecting the matmul dispatch surface. The discovery:

**All existing matmul dispatch interfaces take `(map, size, weight_offset)` — they bypass `tensor_data()` entirely.**

From `ds4_gpu.h`:
```c
int ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size, weight_offset, ...);
int ds4_gpu_matmul_f16_tensor (out, model_map, model_size, weight_offset, ...);
int ds4_gpu_matmul_f32_tensor (out, model_map, model_size, weight_offset, ...);
int ds4_gpu_matmul_f16_pair_tensor(...);
int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(...);
```

LM head dispatch at `ds4.c:12766`:
```c
ok = ds4_gpu_matmul_q8_0_tensor(g->logits,
    model->map,
    model->size,
    weights->output->abs_offset,   ← reads GGUF mmap directly
    DS4_N_EMBD, vocab_dim, g->output_norm, 1) != 0;
```

The Phase 2b override-fill writes heap-allocated bytes to `t->storage.bytes`. These bytes are **unreachable** through the existing matmul interfaces — they go through `model->map + offset`, never `tensor_data(model, t)`.

## Engineer-roster diagnosis

Same pattern as Cycle 5 / Cycle 9 / Cycle 6': **two representations of one concept** that drift apart.

- The tensor metadata layer (`ds4_tensor`, `tensor_data`, `tensor_effective_type`) knows about overrides
- The dispatch layer (`ds4_gpu_matmul_*_tensor`) doesn't — it consumes `(map, offset)` raw
- The override mechanism affects readers that go through `tensor_data()` — but NOT the matmul dispatchers

**Linus**: the dispatch interface predates the override mechanism; nobody re-architected it when override was added (Phase 2b).
**Knuth**: data structures FIRST. The dispatch interface should consume a `ds4_tensor *`, not `(map, offset)`.
**Carmack**: one interface — `dispatch_matmul(out, weight_tensor, x)`. The dispatch internals look up storage + dtype.
**Pearl**: the override is a NEW representation that the OLD interface can't see. Cycle 5 made the metadata layer authoritative; the dispatch layer hasn't caught up.

## The refactor needed (Increment 2 actual scope)

**Option A — wrap at the call site (smallest, ~200 lines):**

At each matmul call site, before dispatch:
```c
const void *weight_data = tensor_data(model, w);
const uint32_t etype = tensor_effective_type(w);
const bool from_mmap = (weight_data == model->map + w->abs_offset);

switch (etype) {
case DS4_TENSOR_Q8_0:
    if (from_mmap) {
        ds4_gpu_matmul_q8_0_tensor(out, model->map, model->size,
                                   w->abs_offset, ...);
    } else {
        /* NEW: heap-based Q8_0 matmul wrapper */
        ds4_gpu_matmul_q8_0_heap(out, weight_data, w->bytes, ...);
    }
    break;
case DS4_TENSOR_BF16:
    /* Storage is heap. Use the new BF16 kernel. */
    ds4_gpu_matmul_bf16_heap(out, weight_data, w->bytes, ...);
    break;
...
}
```

Requires:
- New heap-aware variants of each matmul (matmul_q8_0_heap, matmul_f16_heap, matmul_bf16_heap, etc.)
- Each wraps the weight bytes as MTLBuffer via `newBufferWithBytesNoCopy` (zero-copy on shared memory)
- Dispatch at every matmul call site (LM head, Q/K/V/O × 43 layers, shared experts × 43)

Cost: ~200 lines of wrapper code + ~50 call sites updated.

**Option B — push tensor into dispatch (cleaner, but ~400 lines):**

Refactor each matmul function to take `ds4_tensor *` instead of `(map, offset)`:
```c
int ds4_gpu_matmul_tensor(
    ds4_gpu_tensor *out,
    const ds4_model *model,   /* for model->map fallback */
    const ds4_tensor *weight, /* THE source of truth */
    uint64_t in_dim, out_dim,
    const ds4_gpu_tensor *x,
    uint64_t n_tok);
```

The function pulls `tensor_data(model, weight)` + `tensor_effective_type(weight)` and dispatches internally. Old `(map, offset)` interface deprecated.

Cost: ~400 lines. Touches every matmul site + every kernel-pipeline-init path.

**Option C — heap buffer registration at engine_open (smallest runtime cost):**

When override-fill creates `storage.bytes`, ALSO register an MTLBuffer wrapping those bytes:
```c
t->storage.bytes = dst;
t->storage.length = pe->data_bytes;
t->storage.dtype = ...;
t->storage.ownership = DS4_STORAGE_HEAP;
/* NEW: pre-wrap as MTLBuffer once */
t->storage.metal_buffer = ds4_gpu_wrap_heap_bytes(dst, pe->data_bytes);
```

Then at dispatch time:
```c
if (w->storage.metal_buffer) {
    /* Use the pre-wrapped buffer + new etype-aware kernel */
    ds4_gpu_matmul_tensor_storage(out, &w->storage, ...);
} else {
    /* Existing mmap path */
    ds4_gpu_matmul_q8_0_tensor(out, model->map, ..., w->abs_offset, ...);
}
```

Cost: ~250 lines. Pre-wraps overheads paid at load; runtime branches on `storage.metal_buffer != NULL`.

## Recommendation

**Option C** is the engineer-roster pick:
- Pre-wraps the heap bytes as MTLBuffer at load time (paid once, not per-dispatch)
- Existing matmul interfaces stay (no API breakage)
- New kernel dispatch family `*_tensor_storage` handles the override path
- Dispatch logic stays simple: NULL storage → old path, non-NULL → new path

Scope: ~250 lines + new matmul_bf16_tensor_storage function + dispatch wiring at ~50 call sites.

## Two-session estimate

- Session A: pre-wrap helpers, matmul_bf16_tensor_storage, two call sites (LM head + Q matmul of layer 0)
- Session B: remaining ~48 call sites (Q/K/V/O × 43, shared experts, MTP)
- Session C: lift the Cycle 5 ground rule + smoke validation

## What's shipped today (Increment 1)

`mul_mv_bf16_f32_mtl4` kernel + `--mul-mv-bf16-canary` proves the read path. The compute is correct. The wiring is gated on Option C (or equivalent) which is silv-curated session work.

## Dead code surfaced this turn

`ds4_gpu_set_model_map_ranges` (ds4_metal.m:42939) has NO callers post-Cycle-6'-cleanup. Retire it as part of the dispatch refactor — fewer surfaces.
