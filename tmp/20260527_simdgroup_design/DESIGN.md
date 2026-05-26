# simdgroup_matrix prefill mat-mat kernel — design + stub

silv 2026-05-27 priority: "advance simdgroup matmul determination to next
level."

## Why this kernel doesn't exist yet

The current routed-FFN kernels (kernel_mul_mv_id_*) are MATRIX × VECTOR.
At gen-time (batch=1) this is correct — each token produces one output
row. The 8×8 simdgroup_matrix primitive is mat×mat (8×8 × 8×8 → 8×8) and
gives zero benefit on matvec at batch=1.

For PREFILL, n_tokens ≥ 8 typically (chunks of 2048). Each routed-FFN
matmul becomes (n_tokens × n_embd) × (n_embd × n_ffn_dim) → (n_tokens
× n_ffn_dim). That's a real mat×mat — simdgroup_matrix gives ~8× over
the current row-by-row matvec.

## Kernel signature (proposed)

```metal
kernel void kernel_mul_mm_id_fp16_pair_swiglu_f32(
    constant ds4_metal_args_mul_mm_id & args,
    constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
    device const half  * src0_gate,    // expert weights, FP16 packed row-major
    device const half  * src0_up,
    device const float * src1,         // input activations, FP32 (n_tokens × n_embd)
    device       float * dst_mid,      // output (n_tokens × n_ffn_dim)
    device const int   * ids,          // [n_tokens × N_EXPERT_USED] expert IDs
    device const float * weights,      // [n_tokens × N_EXPERT_USED] expert weights
    threadgroup char   * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]);
```

## Algorithm (per threadgroup)

Each threadgroup computes an 8×8 output tile = 8 tokens × 8 output channels.

```
// Initialize accumulator tile
simdgroup_matrix<float, 8, 8> acc(0.0);

// Loop over K (n_embd) in tiles of 8
for (int k = 0; k < n_embd; k += 8) {
    // Load 8×8 input tile (n_tokens=8 × n_embd=8, FP16 from FP32 source)
    simdgroup_matrix<half, 8, 8> A;
    simdgroup_load(A, /* input pointer */ , /* stride */);

    // Load 8×8 weight tile (n_embd=8 × n_ffn=8) for the routed expert
    simdgroup_matrix<half, 8, 8> B;
    simdgroup_load(B, /* expert weight pointer + offset */ , /* stride */);

    // Accumulate
    simdgroup_multiply_accumulate(acc, A, B, acc);
}

// SwiGLU + expert-weight scaling
// then store acc to dst_mid
```

## Engineering scope

- **Pipeline init**: ~30 lines in ds4_metal.m (mirror of existing
  kernel_mul_mv_id_fp16_pair_swiglu_f32 init)
- **Dispatch**: prefill paths currently dispatch matvec kernel per-token;
  need to batch tokens into 8-wide tiles, dispatch mat-mat per tile.
  ~100 lines in the prefill path
- **Kernel body**: ~150 lines Metal shader. Has subtleties:
  - Each token may route to DIFFERENT experts (the `ids` array). Within a
    threadgroup's 8 tokens, we may have 1-8 distinct experts. Two options:
    A) Pre-sort tokens by selected expert at prefill-time, then dispatch
       per-expert (clean simdgroup access). Adds preprocessing cost.
    B) Per-threadgroup branch on expert ID. Threadgroup divergence cost.
  - SwiGLU activation runs after gate × up multiply. Same fusion as the
    existing pair_swiglu_f32 kernel.
  - Expert weight (the gating coefficient) multiplies the final output.

- **Validation**: A/B with the matvec path on the same prefill prompt.
  Numerical equivalence within FP16 precision tolerance.
- **Bench**: target 4-8× prefill throughput on routed-FFN portion. At
  prefill 6 t/s baseline, 8× brings it to 48 t/s on routed-FFN only.
  Overall t/s improvement depends on attention/non-routed compute share.

## Implementation order

1. **Stub kernel**: declare the signature in moe.metal, return early.
   Register pipeline. Verify compile + linker.
2. **Trivial body**: per-thread scalar dot product (same as matvec,
   ignore simdgroup). Validates dispatch + arg layout.
3. **Real simdgroup body**: replace scalar inner loop with
   simdgroup_load / simdgroup_multiply_accumulate. A/B against (2).
4. **Routing optimization**: option (A) pre-sort or (B) per-tg branch.
   Bench both.
5. **Wire into prefill path**: replace per-token matvec dispatch with
   batched mat-mat at n_tokens % 8 == 0.

Steps 1-2 are ~half day; 3-4 are ~1 day; 5 is ~half day with care.
Total: ~2 days work.

## Why not ship in this session

The Metal shader rewrite + dispatch site rewrite + A/B harness all
require careful staging. Single-session ship would either be incomplete
or untested. The synthesis memo + design + skeleton commit is the
correct deliverable now; the full kernel ship is a tracked follow-up.

## Files

- `metal/moe.metal`: add stub kernel signature after the FP16 matvec
  kernel (line ~1265 of current file)
- `ds4_metal.m`: register `g_moe_mul_mm_id_fp16_pair_swiglu_pipeline`
- New dispatch site: in prefill_layer_major when `n_tokens >= 8`
- Env flag: `DS4_PREFILL_MATMUL_MM=1` to enable
