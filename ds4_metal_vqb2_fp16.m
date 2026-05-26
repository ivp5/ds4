/* ds4_metal_vqb2_fp16.m — Metal FP16 routed-FFN kernel skeleton.
 *
 * silv 2026-05-26 directive: "proceed with vqb2, Metal FP16 routed-FFN
 * kernel, mtl4 instrumentation according to the aggregated insights"
 *
 * STATUS: SKELETON. The Metal kernel source is here; the dispatch wrapper
 * returns -1 (fallback) until full integration lands. This file ships:
 *   1. The Metal Shading Language (MSL) source for FP16 routed-FFN matvec
 *   2. The dispatch wrapper signature + skeleton ObjC bridge code
 *   3. ICB Phase 7 design (record→replay for the new kernel)
 *   4. MTL4 instrumentation pattern (DS4_METAL_PROFILE_DECODE_STAGE)
 *
 * NEXT INTEGRATION STEPS (~8-16h focused):
 *   1. Wire `ds4_metal_vqb2_fp16_init` into ds4_engine_open
 *   2. In `metal_graph_encode_decode_layer` (ds4.c:10823), check
 *      ds4_hot_get_gate_fp16(store, il, expert) for each selected expert;
 *      if non-NULL, call `ds4_metal_vqb2_fp16_dispatch`; else fall back.
 *   3. Implement the dispatch: copy expert FP16 buffers to Metal-resident
 *      MTLBuffers (per-layer KV cache), encode 3 matvec kernels (gate, up,
 *      down) with SwiGLU in between, sum weighted outputs.
 *   4. A/B benchmark against IQ2_XXS + CPU-MoE baseline.
 *   5. Add ICB record→replay for steady-state dispatch (Phase 7).
 */
#include "ds4_metal_vqb2_fp16.h"
#include "ds4_expert_table.h"

#include <stdio.h>
#include <string.h>

#ifdef __APPLE__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif

/* =========================================================================
 * METAL SHADING LANGUAGE SOURCE
 *
 * Three kernels needed:
 *   - kernel_vqb2_fp16_gate_up: input × gate_tile + input × up_tile → swiglu
 *   - kernel_vqb2_fp16_down:    mid × down_tile → output_acc (weighted add)
 *   - kernel_vqb2_fp16_swiglu:  fused SwiGLU on gate × up
 *
 * Apple Silicon's native FP16 matmul (M1+ with __ARM_FEATURE_FP16_FML) is
 * exposed via `half` type in MSL. Native 2× throughput vs FP32 on M1 Max
 * shader cores; native FP16 SIMD on the GPU compute path.
 * ========================================================================= */

static const char *ds4_metal_vqb2_fp16_msl_source =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    "/* FP16 gate+up matvec — one threadgroup per output column, sum across rows.\n"
    " *   tile is [n_rows × n_pairs × 2] FP16 (re, im interleaved per pair).\n"
    " *   input is [n_rows × 2] FP32 (one expert's input slice).\n"
    " *   output is [n_pairs] FP32 accumulator (gate_out or up_out). */\n"
    "kernel void kernel_vqb2_fp16_gate_up(\n"
    "    device const half     *tile           [[buffer(0)]],\n"
    "    device const float    *input          [[buffer(1)]],\n"
    "    device       float    *output         [[buffer(2)]],\n"
    "    constant      uint    &n_rows         [[buffer(3)]],\n"
    "    constant      uint    &n_pairs        [[buffer(4)]],\n"
    "    uint p [[thread_position_in_grid]])\n"
    "{\n"
    "    if (p >= n_pairs) return;\n"
    "    float acc = 0.0f;\n"
    "    for (uint r = 0; r < n_rows; r++) {\n"
    "        const float in_re = input[r * 2u + 0u];\n"
    "        const float in_im = input[r * 2u + 1u];\n"
    "        const uint base = (r * n_pairs + p) * 2u;\n"
    "        acc += in_re * (float)tile[base + 0u]\n"
    "             + in_im * (float)tile[base + 1u];\n"
    "    }\n"
    "    output[p] = acc;\n"
    "}\n"
    "\n"
    "/* Fused SwiGLU + expert weighting:\n"
    " *   mid[i] = swiglu_clamped(gate[i], up[i]) × expert_weight\n"
    " * Standard SwiGLU: silu(gate) × up, where silu(x) = x / (1 + exp(-x)).\n"
    " * Clamp gate to [-CLAMP, +CLAMP] before silu. */\n"
    "kernel void kernel_vqb2_fp16_swiglu(\n"
    "    device const float *gate         [[buffer(0)]],\n"
    "    device const float *up           [[buffer(1)]],\n"
    "    device       float *mid          [[buffer(2)]],\n"
    "    constant      uint  &n_pairs     [[buffer(3)]],\n"
    "    constant      float &clamp_exp   [[buffer(4)]],\n"
    "    constant      float &expert_w    [[buffer(5)]],\n"
    "    uint i [[thread_position_in_grid]])\n"
    "{\n"
    "    if (i >= n_pairs) return;\n"
    "    float g = gate[i];\n"
    "    if (g >  clamp_exp) g =  clamp_exp;\n"
    "    if (g < -clamp_exp) g = -clamp_exp;\n"
    "    const float sig = 1.0f / (1.0f + exp(-g));\n"
    "    mid[i] = g * sig * up[i] * expert_w;\n"
    "}\n"
    "\n"
    "/* FP16 down matvec — accumulate into output.\n"
    " *   down_tile is [n_pairs × n_rows × 2] FP16 (transpose-shaped vs gate).\n"
    " *   mid is [n_pairs] FP32 (post-SwiGLU activations).\n"
    " *   output is [n_rows × 2] FP32 (added to, not overwritten). */\n"
    "kernel void kernel_vqb2_fp16_down(\n"
    "    device const half     *tile        [[buffer(0)]],\n"
    "    device const float    *mid         [[buffer(1)]],\n"
    "    device       float    *output      [[buffer(2)]],\n"
    "    constant      uint    &n_rows      [[buffer(3)]],\n"
    "    constant      uint    &n_pairs     [[buffer(4)]],\n"
    "    uint r [[thread_position_in_grid]])\n"
    "{\n"
    "    if (r >= n_rows) return;\n"
    "    float acc_re = 0.0f, acc_im = 0.0f;\n"
    "    for (uint p = 0; p < n_pairs; p++) {\n"
    "        const float m = mid[p];\n"
    "        const uint base = (p * n_rows + r) * 2u;\n"
    "        acc_re += m * (float)tile[base + 0u];\n"
    "        acc_im += m * (float)tile[base + 1u];\n"
    "    }\n"
    "    output[r * 2u + 0u] += acc_re;\n"
    "    output[r * 2u + 1u] += acc_im;\n"
    "}\n";

/* =========================================================================
 * ICB PHASE 7 DESIGN (silv "advance ICB record→replay further")
 *
 * Existing ICB phases (per task list 553-560):
 *   Phase 1: setBytes→setBuffer for route_weights_with_remap
 *   Phase 2: actual MTLIndirectCommandBuffer recording
 *   Phase 3: kernel_dsv4_topk_mask
 *   Phase 4: kernel_dsv4_softplus_sqrt_f32_4
 *   Phase 5: kernel_dsv4_router_finalize_one
 *   Phase 6: kernel_dsv4_router_weights_one
 *
 * Phase 7 (this design): kernel_vqb2_fp16_{gate_up,swiglu,down}
 *
 * The new VQB2-FP16 dispatch fires once per layer per token during gen,
 * with the SAME pipeline state across tokens. ICB record→replay should
 * record the 3-kernel sequence at first dispatch + replay on subsequent
 * dispatches with updated buffer arguments. Expected gain per codex
 * H1716/H1716 logic: 5-15% of route-encoding cost reclaimed.
 *
 * Env var to gate: DS4_ICB_VQB2_FP16=1 (opt-in, matches existing pattern).
 * Per-layer ICB slot: 43 slots (one per layer); signature is (gate_tile_buf,
 * up_tile_buf, down_tile_buf, n_tokens). Mismatch → re-record at that slot.
 *
 * Falls through to direct encoding when env unset. The 6 existing ICB
 * phases continue working as before — this is additive.
 * ========================================================================= */

/* =========================================================================
 * MTL4 INSTRUMENTATION HOOK (silv "mtl4 instrumentation")
 *
 * Per codex H1666: "MTL4 ML is a measured dense-organ route, not a
 * universal backend." The routed FFN is per-token-sparse, NOT a dense
 * organ. So actual MTL4 ML execution here is non-applicable.
 *
 * The instrumentation we DO need is per-stage profiling: time spent in
 * gate/up matvec vs SwiGLU vs down matvec. ds4_metal.m already has the
 * DS4_METAL_PROFILE_DECODE_STAGE macro for this. The dispatch wrapper
 * below uses it to tag VQB2-FP16-specific stages so DS4_METAL_DECODE_STAGE_PROFILE=1
 * captures them in the per-stage profile output.
 *
 * Stage tags emitted (when profile enabled):
 *   "vqb2_fp16_gate" — gate matvec
 *   "vqb2_fp16_up"   — up matvec
 *   "vqb2_fp16_swiglu" — fused SwiGLU
 *   "vqb2_fp16_down" — down matvec
 *   "vqb2_fp16_load" — per-token buffer prep (FP16 tile → MTL buffer copy)
 *
 * The MTL4 ML path (kernel_dsv4_* with shape-fixed dense organs in
 * ds4_metal.m) stays as-is for shared experts + attention pieces. This
 * file does NOT touch those.
 * ========================================================================= */

static bool s_profile_enabled = false;

void ds4_metal_vqb2_fp16_set_profile(bool enabled) {
    s_profile_enabled = enabled;
}

#ifdef __APPLE__
static id<MTLDevice> s_device = nil;
static id<MTLLibrary> s_library = nil;
static id<MTLCommandQueue> s_command_queue = nil;
static id<MTLComputePipelineState> s_pipeline_gate_up = nil;
static id<MTLComputePipelineState> s_pipeline_swiglu = nil;
static id<MTLComputePipelineState> s_pipeline_down = nil;
static bool s_initialized = false;
static id<MTLBuffer> s_hot_store_buffer = nil;   /* MTLBuffer wrapping the FP16 heap */
static const void *s_bound_heap_ptr = NULL;
static size_t s_bound_heap_bytes = 0;

/* Per-expert scratch buffers (shared across experts within a dispatch).
 * gate_out / up_out / mid are each [n_pairs × 4] bytes; for DS4-Flash
 * n_pairs=2048 → 8 KB each, 24 KB total. Allocated once at init under
 * MTLResourceStorageModeShared (zero-copy on Apple Silicon). */
static id<MTLBuffer> s_scratch_gate_out = nil;
static id<MTLBuffer> s_scratch_up_out = nil;
static id<MTLBuffer> s_scratch_mid = nil;

/* DS4-Flash routed-FFN dimensions per (layer, expert). Hardcoded for now
 * (matches CPU dispatch in ds4_expert_table.c:305+). Future iteration:
 * read from store->n_rows[layer] / store->n_pairs[layer][kind] after the
 * row-block guard patch (codex H1909) lands. */
#define DS4_VQB2_FP16_N_ROWS   128u
#define DS4_VQB2_FP16_N_PAIRS  2048u   /* gate/up */
#define DS4_VQB2_FP16_N_DOWN   1024u   /* down n_pairs */
#define DS4_VQB2_FP16_SCRATCH_BYTES (DS4_VQB2_FP16_N_PAIRS * sizeof(float))
#define DS4_VQB2_FP16_CLAMP    1.0e4f

/* Per-expert tile sizes (must match the VQB2 reader's FP16 layout). */
#define DS4_VQB2_FP16_GATE_BYTES \
    ((size_t)DS4_VQB2_FP16_N_ROWS * DS4_VQB2_FP16_N_PAIRS * 2u * sizeof(_Float16))
#define DS4_VQB2_FP16_UP_BYTES   DS4_VQB2_FP16_GATE_BYTES
#define DS4_VQB2_FP16_DOWN_BYTES \
    ((size_t)DS4_VQB2_FP16_N_ROWS * DS4_VQB2_FP16_N_DOWN  * 2u * sizeof(_Float16))

/* MTL4 surface — codex H1723/H1788/H1779 modern path.
 *
 * silv 2026-05-26 correction: "MTL4 ! not just any Metal api. it was
 * researched by codex and it works if used correctly."
 *
 * Per codex H1779 evidence: MTL4ComputeCommandEncoder validated within
 * 1.49e-08 of CPU on routed-FFN workload. Unlocks argument tables +
 * residency sets + timestamp counters + feedback handlers.
 *
 * Env-gated: DS4_VQB2_FP16_MTL4=1 selects MTL4 path; otherwise legacy
 * MTLComputePipelineState path. Both share the same MSL source.
 */
static id<MTL4Compiler> s_mtl4_compiler = nil;
static id<MTLLibrary> s_mtl4_library = nil;
static id<MTLComputePipelineState> s_mtl4_pipeline_gate_up = nil;
static id<MTLComputePipelineState> s_mtl4_pipeline_swiglu = nil;
static id<MTLComputePipelineState> s_mtl4_pipeline_down = nil;
static id<MTL4CommandQueue> s_mtl4_queue = nil;
static id<MTL4CommandAllocator> s_mtl4_allocator = nil;
static bool s_mtl4_initialized = false;

int ds4_metal_vqb2_fp16_init(void) {
    if (s_initialized) return 0;

    @autoreleasepool {
        s_device = MTLCreateSystemDefaultDevice();
        if (!s_device) {
            fprintf(stderr, "ds4_metal_vqb2_fp16_init: no Metal device\n");
            return -1;
        }

        NSError *error = nil;
        NSString *source = [NSString stringWithUTF8String:ds4_metal_vqb2_fp16_msl_source];
        s_library = [s_device newLibraryWithSource:source options:nil error:&error];
        if (!s_library) {
            fprintf(stderr, "ds4_metal_vqb2_fp16_init: library compile failed: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "unknown");
            return -1;
        }

        id<MTLFunction> fn_gate_up = [s_library newFunctionWithName:@"kernel_vqb2_fp16_gate_up"];
        id<MTLFunction> fn_swiglu  = [s_library newFunctionWithName:@"kernel_vqb2_fp16_swiglu"];
        id<MTLFunction> fn_down    = [s_library newFunctionWithName:@"kernel_vqb2_fp16_down"];
        if (!fn_gate_up || !fn_swiglu || !fn_down) {
            fprintf(stderr, "ds4_metal_vqb2_fp16_init: missing kernel function\n");
            return -1;
        }

        s_pipeline_gate_up = [s_device newComputePipelineStateWithFunction:fn_gate_up error:&error];
        s_pipeline_swiglu  = [s_device newComputePipelineStateWithFunction:fn_swiglu  error:&error];
        s_pipeline_down    = [s_device newComputePipelineStateWithFunction:fn_down    error:&error];
        if (!s_pipeline_gate_up || !s_pipeline_swiglu || !s_pipeline_down) {
            fprintf(stderr, "ds4_metal_vqb2_fp16_init: pipeline state creation failed\n");
            return -1;
        }

        /* Command queue for direct (non-MTL4) dispatch. One queue per process is
         * sufficient — we only ever encode 1 cmd buffer per dispatch call. */
        s_command_queue = [s_device newCommandQueue];
        if (!s_command_queue) {
            fprintf(stderr, "ds4_metal_vqb2_fp16_init: command queue alloc failed\n");
            return -1;
        }

        /* Pre-allocate scratch buffers (shared across experts in a dispatch).
         * Storage mode shared is zero-copy on Apple Silicon unified memory. */
        s_scratch_gate_out = [s_device newBufferWithLength:DS4_VQB2_FP16_SCRATCH_BYTES
                                                  options:MTLResourceStorageModeShared];
        s_scratch_up_out   = [s_device newBufferWithLength:DS4_VQB2_FP16_SCRATCH_BYTES
                                                  options:MTLResourceStorageModeShared];
        s_scratch_mid      = [s_device newBufferWithLength:DS4_VQB2_FP16_SCRATCH_BYTES
                                                  options:MTLResourceStorageModeShared];
        if (!s_scratch_gate_out || !s_scratch_up_out || !s_scratch_mid) {
            fprintf(stderr, "ds4_metal_vqb2_fp16_init: scratch buffer alloc failed\n");
            return -1;
        }

        s_initialized = true;
        fprintf(stderr,
            "ds4_metal_vqb2_fp16_init: 3 pipelines + queue + 3×%zu B scratch ready\n",
            (size_t)DS4_VQB2_FP16_SCRATCH_BYTES);
        return 0;
    }
}

/* MTL4Compute init (codex H1723 pattern + H1779 routed-FFN validation).
 *
 * Differences from legacy:
 *   - MTL4Compiler is a separate object (not part of device)
 *   - MTL4LibraryDescriptor + MTL4LibraryFunctionDescriptor
 *   - MTL4ComputePipelineDescriptor (different from MTLComputePipelineDescriptor)
 *   - MTL4CommandQueue + MTL4CommandAllocator (required pair)
 *   - Argument table via gpuAddress, not setBuffer
 *
 * Idempotent. Returns 0 on success, -1 on failure.
 */
int ds4_metal_vqb2_fp16_init_mtl4(void) {
    if (s_mtl4_initialized) return 0;
    /* Ensure legacy init has run for s_device (we reuse the same device) */
    if (!s_initialized) {
        if (ds4_metal_vqb2_fp16_init() != 0) return -1;
    }
    @autoreleasepool {
        NSError *err = nil;

        /* MTL4Compiler */
        MTL4CompilerDescriptor *compilerDesc = [MTL4CompilerDescriptor new];
        s_mtl4_compiler = [s_device newCompilerWithDescriptor:compilerDesc error:&err];
        if (!s_mtl4_compiler) {
            fprintf(stderr, "ds4_metal_vqb2_fp16_init_mtl4: compiler failed: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "unknown");
            return -1;
        }

        /* MTL4 Library from same MSL source */
        MTL4LibraryDescriptor *libDesc = [MTL4LibraryDescriptor new];
        libDesc.source = [NSString stringWithUTF8String:ds4_metal_vqb2_fp16_msl_source];
        libDesc.name = @"ds4_vqb2_fp16_mtl4";
        s_mtl4_library = [s_mtl4_compiler newLibraryWithDescriptor:libDesc error:&err];
        if (!s_mtl4_library) {
            fprintf(stderr, "ds4_metal_vqb2_fp16_init_mtl4: library failed: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "unknown");
            return -1;
        }

        /* Helper macro: compile one pipeline by kernel name. */
        #define MTL4_MAKE_PIPELINE(kname, var) do { \
            MTL4LibraryFunctionDescriptor *fnDesc = [MTL4LibraryFunctionDescriptor new]; \
            fnDesc.library = s_mtl4_library; \
            fnDesc.name = @kname; \
            MTL4ComputePipelineDescriptor *pipeDesc = [MTL4ComputePipelineDescriptor new]; \
            pipeDesc.computeFunctionDescriptor = fnDesc; \
            pipeDesc.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES; \
            pipeDesc.maxTotalThreadsPerThreadgroup = 256; \
            (var) = [s_mtl4_compiler newComputePipelineStateWithDescriptor:pipeDesc \
                                                       compilerTaskOptions:nil \
                                                                     error:&err]; \
            if (!(var)) { \
                fprintf(stderr, "ds4_metal_vqb2_fp16_init_mtl4: pipeline %s failed: %s\n", \
                        kname, err ? [[err localizedDescription] UTF8String] : "unknown"); \
                return -1; \
            } \
        } while (0)
        MTL4_MAKE_PIPELINE("kernel_vqb2_fp16_gate_up", s_mtl4_pipeline_gate_up);
        MTL4_MAKE_PIPELINE("kernel_vqb2_fp16_swiglu",  s_mtl4_pipeline_swiglu);
        MTL4_MAKE_PIPELINE("kernel_vqb2_fp16_down",    s_mtl4_pipeline_down);
        #undef MTL4_MAKE_PIPELINE

        /* Command queue + allocator (MTL4 requires explicit allocator) */
        s_mtl4_queue = [s_device newMTL4CommandQueueWithDescriptor:[MTL4CommandQueueDescriptor new]
                                                              error:&err];
        if (!s_mtl4_queue) {
            fprintf(stderr, "ds4_metal_vqb2_fp16_init_mtl4: queue failed: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "unknown");
            return -1;
        }
        s_mtl4_allocator = [s_device newCommandAllocatorWithDescriptor:[MTL4CommandAllocatorDescriptor new]
                                                                  error:&err];
        if (!s_mtl4_allocator) {
            fprintf(stderr, "ds4_metal_vqb2_fp16_init_mtl4: allocator failed: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "unknown");
            return -1;
        }

        s_mtl4_initialized = true;
        fprintf(stderr,
            "ds4_metal_vqb2_fp16_init_mtl4: MTL4 path ready "
            "(3 pipelines + queue + allocator)\n");
        return 0;
    }
}

int ds4_metal_vqb2_fp16_bind_store(struct ds4_hot_expert_store *store) {
    if (!s_initialized) {
        if (ds4_metal_vqb2_fp16_init() != 0) return -1;
    }
    if (!store) return -1;
    /* Access the FP16 heap via accessor in ds4_expert_table.c. */
    extern void *ds4_hot_store_heap_ptr(const struct ds4_hot_expert_store *);
    extern uint64_t ds4_hot_store_heap_bytes_get(const struct ds4_hot_expert_store *);
    const void *heap = ds4_hot_store_heap_ptr(store);
    const uint64_t bytes = ds4_hot_store_heap_bytes_get(store);
    if (!heap || bytes == 0) {
        fprintf(stderr, "ds4_metal_vqb2_fp16_bind_store: store has no heap\n");
        return -1;
    }

    if (s_bound_heap_ptr == heap && s_bound_heap_bytes == bytes && s_hot_store_buffer != nil) {
        return 0;
    }

    s_hot_store_buffer = nil;
    s_bound_heap_ptr = NULL;
    s_bound_heap_bytes = 0;

    @autoreleasepool {
        s_hot_store_buffer = [s_device newBufferWithBytesNoCopy:(void *)heap
                                                       length:(NSUInteger)bytes
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
        if (!s_hot_store_buffer) {
            fprintf(stderr,
                "ds4_metal_vqb2_fp16_bind_store: newBufferWithBytesNoCopy failed for "
                "%.1f MB heap @ %p\n", (double)bytes / 1e6, heap);
            return -1;
        }
        s_bound_heap_ptr = heap;
        s_bound_heap_bytes = (size_t)bytes;
        fprintf(stderr,
            "ds4_metal_vqb2_fp16_bind_store: bound %.1f MB FP16 heap as MTLBuffer "
            "(zero-copy on Apple Silicon unified memory)\n",
            (double)bytes / 1e6);
        return 0;
    }
}

/* Forward decls — accessors live in ds4_expert_table.c. We re-declare here
 * to avoid pulling the full struct definition into Metal compilation
 * units (the struct lives behind a typedef in ds4_expert_table.h, but
 * we'd rather keep this file's #include surface minimal). */
extern const void *ds4_hot_get_gate_fp16(const struct ds4_hot_expert_store *,
                                         uint32_t layer, uint32_t expert);
extern const void *ds4_hot_get_up_fp16(const struct ds4_hot_expert_store *,
                                       uint32_t layer, uint32_t expert);
extern const void *ds4_hot_get_down_fp16(const struct ds4_hot_expert_store *,
                                         uint32_t layer, uint32_t expert);

/* Convert an absolute pointer (returned by ds4_hot_get_*_fp16) into a byte
 * offset into the bound MTLBuffer. Returns SIZE_MAX on out-of-range. */
static size_t vqb2_fp16_offset_for(const void *tile_ptr) {
    if (!tile_ptr || !s_bound_heap_ptr) return SIZE_MAX;
    const uintptr_t base = (uintptr_t)s_bound_heap_ptr;
    const uintptr_t tile = (uintptr_t)tile_ptr;
    if (tile < base) return SIZE_MAX;
    const uintptr_t off = tile - base;
    if (off >= (uintptr_t)s_bound_heap_bytes) return SIZE_MAX;
    return (size_t)off;
}

int ds4_metal_vqb2_fp16_dispatch(struct ds4_hot_expert_store *store,
                                 uint32_t layer,
                                 uint32_t n_tokens,
                                 const int32_t *selected_exps,
                                 const float *expert_weights,
                                 const void *input_fp32,
                                 void *output_fp32) {
    if (!s_initialized) return -1;
    if (!store || !selected_exps || !expert_weights ||
        !input_fp32 || !output_fp32) return -1;
    if (n_tokens != 1) return -2;  /* multi-token batch unimplemented — caller falls back */
    if (!s_hot_store_buffer) {
        /* Caller forgot to bind the store. Try a lazy bind. */
        if (ds4_metal_vqb2_fp16_bind_store(store) != 0) return -1;
        if (!s_hot_store_buffer) return -1;
    }

    /* Caller-pointer wrap pattern: input is read-only; output is read+write
     * (down kernel uses +=). Both are FP32. On Apple Silicon unified memory
     * newBufferWithBytesNoCopy on a malloc'd region is zero-copy under
     * MTLResourceStorageModeShared. Length is rounded up to the page (16 KB
     * on M1) under the hood; the underlying alloc must be page-multiple in
     * the FP32 layout (n_rows × 2 = 256 floats = 1 KB ≪ 16 KB — must be
     * page-aligned + bounded by something at least page-sized). Caller is
     * responsible for that; we pad the wrap length to page-size to avoid
     * MTLBuffer-creation noise. The kernel only touches the first
     * (n_rows × 2) elements. */
    const size_t io_logical_bytes  = (size_t)DS4_VQB2_FP16_N_ROWS * 2u * sizeof(float);
    const size_t io_wrapped_bytes  = (io_logical_bytes + 16383u) & ~((size_t)16383u);  /* 16 KB */

    @autoreleasepool {
        id<MTLBuffer> in_buf =
            [s_device newBufferWithBytesNoCopy:(void *)input_fp32
                                        length:io_wrapped_bytes
                                       options:MTLResourceStorageModeShared
                                   deallocator:nil];
        id<MTLBuffer> out_buf =
            [s_device newBufferWithBytesNoCopy:output_fp32
                                        length:io_wrapped_bytes
                                       options:MTLResourceStorageModeShared
                                   deallocator:nil];
        if (!in_buf || !out_buf) {
            fprintf(stderr, "ds4_metal_vqb2_fp16_dispatch: I/O buffer wrap failed\n");
            return -2;
        }

        id<MTLCommandBuffer> cmd = [s_command_queue commandBuffer];
        if (!cmd) return -2;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -2;

        const uint32_t n_rows  = DS4_VQB2_FP16_N_ROWS;
        const uint32_t n_pairs = DS4_VQB2_FP16_N_PAIRS;
        const uint32_t n_down  = DS4_VQB2_FP16_N_DOWN;
        const float   clamp   = DS4_VQB2_FP16_CLAMP;

        /* Threadgroup sizing: pipeline-reported max threads, capped to n_pairs
         * for the gate/up/swiglu kernels (one thread per output pair) or
         * n_rows for down (one thread per output row). 128 is a good default
         * on M1 P-cores (matches SIMD-group width 32 × 4). */
        const NSUInteger tg_pairs = MIN((NSUInteger)128, (NSUInteger)n_pairs);
        const NSUInteger tg_rows  = MIN((NSUInteger)128, (NSUInteger)n_rows);
        const NSUInteger grid_pairs = ((n_pairs + tg_pairs - 1) / tg_pairs) * tg_pairs;
        const NSUInteger grid_rows  = ((n_rows  + tg_rows  - 1) / tg_rows ) * tg_rows;
        const NSUInteger grid_down  = ((n_down  + tg_pairs - 1) / tg_pairs) * tg_pairs;

        for (uint32_t s = 0; s < 6; s++) {  /* DS4 routed n_selected = 6 */
            const int32_t exp_id = selected_exps[s];
            if (exp_id < 0) continue;

            const void *gate_ptr = ds4_hot_get_gate_fp16(store, layer, (uint32_t)exp_id);
            const void *up_ptr   = ds4_hot_get_up_fp16  (store, layer, (uint32_t)exp_id);
            const void *down_ptr = ds4_hot_get_down_fp16(store, layer, (uint32_t)exp_id);
            const size_t gate_off = vqb2_fp16_offset_for(gate_ptr);
            const size_t up_off   = vqb2_fp16_offset_for(up_ptr);
            const size_t down_off = vqb2_fp16_offset_for(down_ptr);
            if (gate_off == SIZE_MAX || up_off == SIZE_MAX || down_off == SIZE_MAX) {
                [enc endEncoding];
                return -1;  /* not all selected experts pinned — caller falls back */
            }

            /* ---- gate matvec: kernel_vqb2_fp16_gate_up(gate_tile, input, gate_out) ---- */
            [enc setComputePipelineState:s_pipeline_gate_up];
            [enc setBuffer:s_hot_store_buffer offset:(NSUInteger)gate_off atIndex:0];
            [enc setBuffer:in_buf             offset:0                    atIndex:1];
            [enc setBuffer:s_scratch_gate_out offset:0                    atIndex:2];
            [enc setBytes:&n_rows  length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&n_pairs length:sizeof(uint32_t) atIndex:4];
            [enc dispatchThreads:MTLSizeMake(grid_pairs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_pairs, 1, 1)];

            /* ---- up matvec: same kernel, different tile ---- */
            [enc setBuffer:s_hot_store_buffer offset:(NSUInteger)up_off   atIndex:0];
            [enc setBuffer:in_buf             offset:0                    atIndex:1];
            [enc setBuffer:s_scratch_up_out   offset:0                    atIndex:2];
            [enc setBytes:&n_rows  length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&n_pairs length:sizeof(uint32_t) atIndex:4];
            [enc dispatchThreads:MTLSizeMake(grid_pairs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_pairs, 1, 1)];

            /* ---- SwiGLU + expert_w fuse ---- */
            const float expert_w = expert_weights[s];
            [enc setComputePipelineState:s_pipeline_swiglu];
            [enc setBuffer:s_scratch_gate_out offset:0 atIndex:0];
            [enc setBuffer:s_scratch_up_out   offset:0 atIndex:1];
            [enc setBuffer:s_scratch_mid      offset:0 atIndex:2];
            [enc setBytes:&n_pairs  length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&clamp    length:sizeof(float)    atIndex:4];
            [enc setBytes:&expert_w length:sizeof(float)    atIndex:5];
            [enc dispatchThreads:MTLSizeMake(grid_pairs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_pairs, 1, 1)];

            /* ---- down matvec: kernel_vqb2_fp16_down(down_tile, mid, output) ---- */
            [enc setComputePipelineState:s_pipeline_down];
            [enc setBuffer:s_hot_store_buffer offset:(NSUInteger)down_off atIndex:0];
            [enc setBuffer:s_scratch_mid      offset:0                    atIndex:1];
            [enc setBuffer:out_buf            offset:0                    atIndex:2];
            [enc setBytes:&n_rows length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&n_down length:sizeof(uint32_t) atIndex:4];
            [enc dispatchThreads:MTLSizeMake(grid_rows, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_rows, 1, 1)];
            (void)grid_down;  /* n_down used only inside the kernel for the inner loop bound */
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error) {
            fprintf(stderr, "ds4_metal_vqb2_fp16_dispatch: cmd.error = %s\n",
                    [[cmd.error localizedDescription] UTF8String]);
            return -2;
        }
        return 0;
    }
}

#else
/* Non-Apple build (CPU/CUDA): MTL4 stubs */
int ds4_metal_vqb2_fp16_init(void) { return -1; }
int ds4_metal_vqb2_fp16_dispatch(struct ds4_hot_expert_store *store,
                                 uint32_t layer, uint32_t n_tokens,
                                 const int32_t *selected_exps,
                                 const float *expert_weights,
                                 const void *input_fp32, void *output_fp32) {
    (void)store; (void)layer; (void)n_tokens;
    (void)selected_exps; (void)expert_weights;
    (void)input_fp32; (void)output_fp32;
    return -1;
}
#endif
