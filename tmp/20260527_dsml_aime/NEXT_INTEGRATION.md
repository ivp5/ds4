# Next-integration design — MTL4 MoE-matmul port + ICB MoE dispatch + GGUF emit

silv 2026-05-27 directive (full): "work on tool dsml and aime. work on
the 74 remaining classic mtl kernels to mtl4 ports, including mtl4
moe-matmul, then icb moe dispatch and gguf emit pipeline."

This memo documents the design for the 3 remaining items after DSML +
AIME prefill arms shipped (commit a88bad7). Single-turn budget did not
admit full implementation; the memo + commit-skeleton pattern is the
honest deliverable.

## Item 1 — MTL4 port of `kernel_mul_mm_id_fp16_pair_swiglu_f32`

### Why this kernel first

Per INTEGRATION_MAP.md: 81 total Metal kernels, 8.6% MTL4 coverage
(7 pipelines). The single highest-leverage remaining port is
`kernel_mul_mm_id_fp16_pair_swiglu_f32` (moe.metal:1282) — the FP16
simdgroup matmul that codex H1725 named as the per-token decode
bottleneck.

Per-token routed MoE dispatch fires:
- 6 selected experts × 3 organs (gate, up, down) = 18 matmuls per layer
- × 43 layers = ~774 kernel dispatches per token
- × ~50µs encoding cost = ~40ms per token in encoding alone

Cutting encoder rebuild via MTL4 + ICB combined target: ~5× wall-clock
gen speedup on decode-bound workloads.

### Port template (mirrors `g_hadamard_mtl4_pipeline` shipped #653)

#### A. Pipeline registration in ds4_metal.m

```objc
// Add near g_hadamard_mtl4_pipeline declaration:
static id<MTL4ComputePipelineState> g_moe_matmul_mtl4_pipeline = nil;

// In ds4_gpu_mtl4_init() or equivalent, alongside g_hadamard_mtl4_pipeline:
{
    NSError *err = nil;
    MTL4ComputePipelineDescriptor *desc = [MTL4ComputePipelineDescriptor new];
    desc.computeFunctionDescriptor = [[MTL4LibraryFunctionDescriptor alloc] init];
    ((MTL4LibraryFunctionDescriptor *)desc.computeFunctionDescriptor).name =
        @"kernel_mul_mm_id_fp16_pair_swiglu_f32_mtl4";
    ((MTL4LibraryFunctionDescriptor *)desc.computeFunctionDescriptor).library =
        g_polar_library;
    g_moe_matmul_mtl4_pipeline =
        [g_polar_compiler newComputePipelineStateWithDescriptor:desc error:&err];
    if (!g_moe_matmul_mtl4_pipeline) {
        fprintf(stderr, "MTL4 MoE matmul pipeline build failed: %s\n",
                err.localizedDescription.UTF8String);
        return -1;
    }
}
```

#### B. Storage-class migration

The classic-MTL kernel signature:

```metal
kernel void kernel_mul_mm_id_fp16_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights, ...)
```

MTL4 storage-class requirements (per R1 Hadamard catch — `constant`
bindings via MTL4ArgumentTable don't get gpuAddress; need `device const`):

```metal
kernel void kernel_mul_mm_id_fp16_pair_swiglu_f32_mtl4(
        device const ds4_metal_args_mul_mv_id * args_ptr,
        device const ds4_metal_dsv4_moe_swiglu_weight_args * act_ptr,
        device const half * src0_gate,
        device const half * src0_up,
        device const float * src1,
        device float * dst_mid,
        device const int32_t * ids,
        device const float * weights, ...)
```

Body re-uses the same simdgroup matmul logic — only the pointer
dereferences change (`args.foo` → `args_ptr->foo`).

#### C. Encoder + argument-table

```objc
// in ds4_gpu_mtl4_moe_matmul_swiglu_apply_encoder helper:
[encoder setComputePipelineState:g_moe_matmul_mtl4_pipeline];
MTL4ArgumentTable *argTable = [g_moe_matmul_argtable_pool acquire];
[argTable setAddress:args_buf.gpuAddress atIndex:0];
[argTable setAddress:act_buf.gpuAddress atIndex:1];
[argTable setAddress:src0_gate_buf.gpuAddress atIndex:2];
[argTable setAddress:src0_up_buf.gpuAddress atIndex:3];
[argTable setAddress:src1_buf.gpuAddress atIndex:4];
[argTable setAddress:dst_mid_buf.gpuAddress atIndex:5];
[argTable setAddress:ids_buf.gpuAddress atIndex:6];
[argTable setAddress:weights_buf.gpuAddress atIndex:7];
[encoder setArgumentTable:argTable];
[encoder setThreadgroupMemoryLength:tg_mem_bytes atIndex:0];
[encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:tpt];
// Critical: barrier between consecutive MoE matmuls (R1 lesson)
[encoder barrierAfterEncoderStages:MTLStageDispatch
                beforeEncoderStages:MTLStageDispatch
                  visibilityOptions:MTL4VisibilityOptionDevice];
```

#### D. Canary test (mirrors `ds4_gpu_mtl4_hadamard16_canary`)

The MoE matmul produces FP32 `mid` buffer. Canary: 1 token × 1 expert
× small n_in/n_out (e.g. 64×128), feed deterministic gate/up weights,
verify output ≈ silu(gate·x)*up·route_weight to FP16 precision.

### Honest scope

The full port is ~200-400 lines (pipeline reg + MTL4 kernel + encoder
helper + canary). Per the H2073 R1 lesson, the FIRST integration test
likely fails on a barrier/storage-class subtlety; debug cycle is
1-2 hours.

Not shipped this turn because: each of the 74 remaining ports has
similar shape, ~half-day each. The TEMPLATE above is the deliverable
that compresses 74×0.5d → 74×1h once the first lands.

## Item 2 — ICB capture of routed MoE dispatch

### What ICB already covers (6 phases)

| phase | task | covered |
|-------|------|---------|
| 1-2 | #553/#555 | route_weights_with_remap |
| 3 | #557 | topk_mask, topk_mask_scatter |
| 4 | #558 | softplus_sqrt_f32_4 |
| 5 | #559 | router_finalize_one |
| 6 | #560 | router_weights_one |

These cover the ROUTER cycle. The big remaining miss is the actual
expert dispatch — which fires ~774 kernel calls per token (per the
Item-1 calculation).

### ICB design

The router output is `(token, expert_id, weight)` triples. Everything
downstream of the router is fixed per model topology:
- Layer count, expert count, K-top all const
- Per-layer expert matmul shapes const
- Buffer offsets per (layer, expert, organ) const at model-load time

What VARIES per token is just the selected-expert-ids and their
weights. ICB record-once / replay-per-token works because the
per-expert dispatch loop is identical between tokens; only the
buffer-offset arithmetic depends on the routed-expert IDs.

### Phase 7 sketch

```objc
// At model open, after Phase 1-6 ICBs built:
static id<MTLIndirectCommandBuffer> g_icb_moe_dispatch = nil;

// Build phase: record N expert-slots worth of dispatches
// (N = ne0 = output dim / tile size in expert direction)
// Each slot pre-records the matmul kernel + max-tile dispatch shape.

// Per-token replay:
//   1. Read router output → selected_expert_ids[K_top]
//   2. For each expert in selected_expert_ids:
//        a. Set this expert's buffer offsets in indirectArgumentBuffer
//        b. Encode executeCommandsInBuffer for this expert's slot
//   3. One barrier at end before down-projection
```

### Pre-port vs. post-port question

The ICB capture is INDEPENDENT of MTL4 port. ICB can happen on classic
MTL kernels first (smaller blast radius), then the kernels themselves
get MTL4'd, then the ICB get regenerated against MTL4 pipelines.

Recommended order:
1. Ship MTL4 MoE-matmul (Item 1) for ONE expert + measure speedup vs classic
2. If MTL4 wins, ICB capture against the MTL4 pipeline (Item 2)
3. If MTL4 doesn't clearly win, ICB capture against classic, then revisit MTL4

This avoids debugging two new things at once on a hot-path kernel.

## Item 3 — GGUF emit pipeline (pruned + basis-aware sidecar)

### What's needed

Combine:
- Codex H2074's `[55,71,188,231,254]` prune set (validated this session, commit a88bad7)
- Basis-aware Hadamard sidecar markers (per #647)
- Multi-shard GGUF emit honoring the 52GiB target (codex H2069)

### Emit-pipeline shape

```
INPUT:
  • Source GGUF (DeepSeek-V4-Flash-IQ2XXS-w2Q2K-... 81.0 GiB)
  • Prune manifest:
      {layer: 9, experts: [55,71,188,231,254], organ: down, weight: silv-mean-zero}
  • Basis sidecar marker:
      {layer: 9, expert: <set>, organ: <set>, basis: hadamard16}

PROCESS:
  1. Read source GGUF tensor table
  2. For each routed-expert tensor (layers 0..N-1, experts 0..255, organs gate/up/down):
       a. If (layer, expert, organ) in prune manifest:
          - Replace tensor data with calibration-derived mean
            (mid-zero for DOWN; mean-of-trained-activation for GATE/UP)
          - Drop the row entirely if calibration-domain matches Rule 6
       b. If (layer, expert, organ) marked basis-aware:
          - Apply forward Hadamard-16 transform (via ds4_gpu_mtl4_hadamard16_apply)
          - Re-quantize the transformed tensor
          - Stash inverse-Hadamard marker in tensor metadata
       c. Otherwise: copy through unchanged
  3. Recompute tensor table with new offsets
  4. Write multi-shard GGUF (3-4 shards of ~17 GiB each per DS4 convention)
  5. Generate sidecar JSON:
       - Prune-manifest source
       - Basis-aware marker source
       - Calibration-domain table
       - Verification hash per shard

OUTPUT:
  • DeepSeek-V4-Flash-IQ2XXS-pruned-basis-aware-{00001-N,...}.gguf
  • DeepSeek-V4-Flash-IQ2XXS-pruned-basis-aware.sidecar.json
  • DEPLOYMENT_RULES.md instance for this build
```

### Skeleton entry-point

```python
# tools/emit_pruned_basis_aware_gguf.py
def emit(source_gguf_path, prune_manifest_path, basis_marker_path,
         output_dir, shard_size_gib=17):
    src = gguf.GGUFReader(source_gguf_path)
    prune = load_prune_manifest(prune_manifest_path)
    basis = load_basis_markers(basis_marker_path)

    writer = MultiShardGGUFWriter(output_dir, shard_size_gib)
    for tensor_name, tensor in src.iter_tensors():
        layer, expert, organ = parse_tensor_path(tensor_name)
        if (layer, expert, organ) in prune:
            data = calibration_replacement(tensor, prune[(layer, expert, organ)])
        elif (layer, expert, organ) in basis:
            data = apply_hadamard16_then_requantize(tensor, basis[(layer, expert, organ)])
        else:
            data = tensor.data
        writer.write_tensor(tensor_name, data, tensor.dtype)
    writer.finalize()
    emit_sidecar_json(...)
```

### Open questions

1. **In-place vs. shard-rewrite**: 81 GiB source → 52 GiB output isn't
   in-place-friendly. Need 2× source size headroom on a 64 GB M1.
   Stream-shard: read shard N of source → emit shard M of output.

2. **Calibration-mean source**: per-cell mean replacement requires
   calibration activations. Where do those live? Codex H2068's
   calibration set has them per-domain — need to plumb the
   calibration_domain_id rule (#647 Rule 6).

3. **Hadamard requantization soundness**: applying H, requantizing,
   inverse-applying at inference is the chain. The MTL4 Hadamard
   kernel + 2700× R1 fix means we have the primitive. Soundness of
   the round-trip at IQ2_XXS quantization tier is the open question.

## Cross-item dependency graph

```
DSML+AIME (this commit)
       │
       ├──→  Prune-set validated  ──→  GGUF emit (Item 3)
       │                                    ↑
       │                                    │ uses Hadamard primitive
       │                                    │ from #653
       │                                    │
       ├──→  MTL4 MoE-matmul (Item 1) ──────┘
       │                ↑
       │                │ once MTL4 pipeline lands,
       │                │ ICB capture (Item 2) against
       │                │ the MTL4 pipeline
       │                │
       └────────────────┘
```

## Honest ship status

| item | design | skeleton | implementation | tested |
|------|-------:|---------:|---------------:|-------:|
| MTL4 MoE-matmul port | ✓ | template above | NO | NO |
| ICB MoE dispatch | ✓ | Phase 7 sketch | NO | NO |
| GGUF emit pipeline | ✓ | Python skeleton | NO | NO |

Each item is 1-2 turns of focused implementation work. The design memo
above is the leverage point — turns 1 of each can now START from a
known shape rather than from-scratch architecture decisions.

## What this session DID ship as a coherent unit

1. **Daemon mode + harm-scorer wire** (#651-#659)
2. **MTL4 Hadamard widened pipeline** (#653) with R1 2700× precision fix
3. **9-prompt cross-domain capability A/B** validating H2074's no-hub
   prune set on math/knowledge/code
4. **6-prompt DSML capability A/B** validating top-1 stability under prune
5. **3-prompt AIME-prefill capability A/B** validating top-1 stability
6. **18-prompt cumulative deployment gate**: 0 top-1 flips for H2074,
   1 flip for H2068 (validating H2077 protection rule)
7. **This design memo** for the 3 remaining items

Engineering posture (jjj/codex pattern): ship small atomic units +
design memos for what doesn't fit in one turn. Don't ship broken
half-implementations of multi-turn items.
