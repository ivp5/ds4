# MTL4 ports — session ledger 2026-05-27

silv 2026-05-27 directive sequence:
1. "work on the 74 remaining classic mtl kernels to mtl4 ports"
2. "go on with the 70+ classic mtl to MTL4 ports"
3. "continue with the remaining mtl4 ports"

## Ports shipped this session

| # | kernel | source | canary result | CLI flag |
|---|--------|--------|---------------|----------|
| 1 | softplus_sqrt_f32_4 | metal/unary.metal:290 | max_abs=4.77e-7 | `--softplus-sqrt-canary` |
| 2 | router_weights_one | metal/dsv4_misc.metal:263 | max_abs=0 (exact) | `--router-weights-one-canary` |
| 3 | topk_mask | metal/dsv4_misc.metal:346 | 4096/4096 cells correct | `--topk-mask-canary` |
| 4 | topk_mask_scatter | metal/dsv4_misc.metal:366 | 768/768 zeroed (n_tokens=256) | `--topk-mask-scatter-canary` |

## Coverage progression

| stage | pipelines | / total | % |
|-------|----------:|--------:|----:|
| Pre-session | 7 | 81 | 8.6% |
| After softplus_sqrt | 8 | 81 | 9.9% |
| After router_weights_one | 9 | 81 | 11.1% |
| After topk_mask | 10 | 81 | 12.3% |
| After topk_mask_scatter | **11** | 81 | **13.6%** |

Net: 4 ports / +5% coverage / ~600 lines of code / 0 ICB regressions.

## Port pattern (formulaic, ~150 lines per port)

Each port follows the same template:

```c
/* 1. Pipeline state */
static id<MTLComputePipelineState> g_<kernel>_mtl4_pipeline;
static int g_<kernel>_mtl4_init_attempted;
static int g_<kernel>_mtl4_init_ok;

/* 2. Pipeline init function */
static int ds4_<kernel>_mtl4_pipeline_init(void) {
    /* gate against re-init */
    if (g_<kernel>_mtl4_init_attempted) return g_<kernel>_mtl4_init_ok;
    g_<kernel>_mtl4_init_attempted = 1;
    if (!ds4_polar_pipeline_init()) return 0;  /* shared compiler */

    /* kernel source as NSString — storage class: constant → device const */
    NSString *source = @"...";

    /* MTL4LibraryDescriptor → newLibrary → MTL4ComputePipelineDescriptor */
    /* error reporting + g_<kernel>_mtl4_init_ok = 1 on success */
}

/* 3. Public canary */
int ds4_gpu_mtl4_<kernel>_canary(...) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!ds4_<kernel>_mtl4_pipeline_init()) return 0;

    /* allocate buffers, write args + inputs */
    /* MTLResidencySetDescriptor + addAllocations + commit + requestResidency */
    /* MTL4ArgumentTableDescriptor + setAddress per buffer */
    /* newCommandBuffer + beginCommandBufferWithAllocator + useResidencySet */
    /* computeCommandEncoder + setComputePipelineState + setArgumentTable */
    /* dispatchThreadgroups + endEncoding + endCommandBuffer */
    /* commit + dispatch_semaphore_wait for completion */
    /* compare GPU output vs CPU reference */
}
```

## Bugs caught + fixed this session

1. **`MTL4ComputePipelineState` doesn't exist** — use `MTLComputePipelineState`.
   The MTL4 transition only renamed compiler/queue/encoder; pipeline state
   protocols stayed as `MTL...` (no MTL4 prefix).

2. **ARC + goto bypassing __strong inits** — Objective-C ARC's strong-reference
   lifetime tracking conflicts with `goto done`. Solution: use nested
   `if (...) {...}` blocks instead of goto, restructure error handling.

3. **`computeCommandEncoderWithDescriptor:` doesn't exist on MTL4CommandBuffer**
   — use `[cb computeCommandEncoder]` instead.

4. **`useResources:count:usage:` doesn't exist on MTL4ComputeCommandEncoder**
   — use `[cb useResidencySet:residency]` at command-buffer level.

5. **`waitForCompletion` on queue doesn't exist** — use `dispatch_semaphore`
   + `MTL4CommitOptions` with `addFeedbackHandler`.

6. **`-ffast-math` makes `host_dst[i] == -INFINITY` UB** — use `<-1e30f`
   comparison instead. Build warning: `-Wnan-infinity-disabled`.

7. **g_device not initialized in canary path** — must call `ds4_gpu_init()`
   first. Hadamard canary has this; my first softplus_sqrt port missed it
   and produced `polar MTL4 compiler init failed: (no error)`.

8. **H2086 IQ2_XXS dequant 2× off** (codex catch, fixed this session):
   `0.125 * (0.5 + h)` was half of reference `0.125 * (2h+1)`. Math:
   `(0.5+h)*0.125 = (1+2h)/16` ≠ `(2h+1)*0.125 = (1+2h)/8`. Reference
   formula confirmed in ds4.c:2112-2113.

## Remaining high-leverage ports

Per INTEGRATION_MAP.md priority:

1. **kernel_mul_mm_id_fp16_pair_swiglu_f32** (moe.metal:1282)
   - The per-token decode bottleneck — 774 calls/token
   - 8 buffer bindings + 2 args structs
   - ~130-line kernel body
   - **1-2 turn port** due to complexity
   - 5× gen speedup target when combined with ICB capture

2. **kernel_dsv4_router_finalize_one** (dsv4_misc.metal:288)
   - 256-thread bitonic sort with threadgroup scratch
   - Medium complexity (~60-line body)
   - Already ICB-captured (Phase 5 #559)

3. **kernel_dsv4_router_weights_with_remap** (dsv4_misc.metal:210)
   - Similar to router_weights_one but with expert-remap indirection
   - Already ICB-captured (Phase 2 #555)

4. **kernel_rms_norm_fuse_impl + kernel_dsv4_qkv_rms_norm_f32_4**
   (norm.metal)
   - Templated kernel + 6-buffer variant
   - Mid-priority — fires once per layer (43×) in attention path

5. **kernel_soft_max + kernel_soft_max_4** (softmax.metal)
   - Softmax over various row sizes
   - Mid-priority

## Honest scope

74 → 70 remaining classic-MTL kernels after this session. At ~150 lines + 2-3 turns per kernel for the easier ones, full 70-kernel port is 70 × 0.5 day = ~35 day-equivalents. Mass-porting is now MECHANICAL given the established template — next-turn implementer can port 3-5 simple kernels per session.

## Files modified this session

- `ds4_metal.m` — 4 new pipeline init + canary blocks (~600 lines)
- `ds4_gpu.h` — 3 public canary declarations
- `ds4_cli.c` — 4 new CLI entries

Plus the H2086 IQ2 dequant fix (2 lines) and the comment block above it.
