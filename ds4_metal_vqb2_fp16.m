/* ds4_metal_vqb2_fp16.m — Metal FP16 routed-FFN dispatch.
 *
 *   3 backends (legacy / ICB record-replay / MTL4 argument-table), selected
 *   at runtime via env DS4_VQB2_FP16_PATH = legacy|icb|mtl4.
 *   Hot store FP16 heap wraps zero-copy as MTLBuffer on Apple Silicon.
 *   ICB cache is per-layer 8-slot LRU keyed by sorted-expert hash.
 *   Journal events emit through ds4_journal (weak default stubs).
 *
 * Convention: all variables defined statically at top; minimal functions;
 * self-explanatory names.
 *
 * SAFETY: every DS4 binary using this MUST launch with
 * --prefill-metal-phases auto (or --cpu-moe). The IQ2_XXS model exceeds
 * the M1 Max wired-memory cap (~48 GB) without phase-split prefill.
 */
#include "ds4_metal_vqb2_fp16.h"
#include "ds4_expert_table.h"
#include "ds4_journal.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>

#ifdef __APPLE__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif

/* ==========================================================================
 * Constants — fixed for DS4-Flash routed FFN layout.
 * ========================================================================== */
#define N_ROWS              128u   /* per-expert tile rows (gate/up/down) */
#define N_PAIRS             2048u  /* gate/up output pairs */
#define N_DOWN              1024u  /* down output pairs */
#define N_EXPERT_USED       6u
#define SWIGLU_CLAMP        1.0e4f
#define SCRATCH_BYTES       (N_PAIRS * sizeof(float))
#define IO_PAGE_BYTES       16384u
#define ICB_SLOTS_PER_LAYER 8
#define ICB_CMDS_PER_DISP   24u   /* 6 experts × 4 kernels */
#define KNUTH_MULT          2654435761ULL

#ifndef DS4_N_LAYER
#define DS4_N_LAYER         43
#endif

/* ==========================================================================
 * Metal Shading Language source — three kernels: gate/up matvec, SwiGLU,
 * down matvec. `constant T &` accepts inline setBytes OR setBuffer:offset
 * uniformly, so the same MSL serves all 3 dispatch backends.
 * ========================================================================== */
static const char *msl_source =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
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
    "        acc += in_re * (float)tile[base + 0u] + in_im * (float)tile[base + 1u];\n"
    "    }\n"
    "    output[p] = acc;\n"
    "}\n"
    "\n"
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

#ifdef __APPLE__
/* ==========================================================================
 * STATICS — all named, all defined at top.
 * State lives here, functions just touch it.
 * ========================================================================== */

/* Legacy MTL pipelines + queue */
static id<MTLDevice>               s_device              = nil;
static id<MTLLibrary>              s_library             = nil;
static id<MTLCommandQueue>         s_command_queue       = nil;
static id<MTLComputePipelineState> s_pipeline_gate_up    = nil;
static id<MTLComputePipelineState> s_pipeline_swiglu     = nil;
static id<MTLComputePipelineState> s_pipeline_down       = nil;
static bool                        s_initialized         = false;

/* Hot-store MTLBuffer wrapping the FP16 heap (zero-copy on M1 unified mem) */
static id<MTLBuffer>               s_hot_store_buffer    = nil;
static const void                 *s_bound_heap_ptr      = NULL;
static size_t                      s_bound_heap_bytes    = 0;

/* Per-call scratch + persistent I/O — allocated once at init.
 * Persistent identity means ICB-recorded commands reference STABLE buffers. */
static id<MTLBuffer>               s_scratch_gate_out    = nil;
static id<MTLBuffer>               s_scratch_up_out      = nil;
static id<MTLBuffer>               s_scratch_mid         = nil;
static id<MTLBuffer>               s_persistent_input    = nil;
static id<MTLBuffer>               s_persistent_output   = nil;

/* MTL4 surface (modern path: argument tables + residency sets) */
static id<MTL4Compiler>            s_mtl4_compiler       = nil;
static id<MTLLibrary>              s_mtl4_library        = nil;
static id<MTLComputePipelineState> s_mtl4_pipeline_gate_up = nil;
static id<MTLComputePipelineState> s_mtl4_pipeline_swiglu  = nil;
static id<MTLComputePipelineState> s_mtl4_pipeline_down    = nil;
static id<MTL4CommandQueue>        s_mtl4_queue          = nil;
static id<MTL4CommandAllocator>    s_mtl4_allocator      = nil;
static id<MTL4ArgumentTable>       s_mtl4_arg_table      = nil;
static id<MTLResidencySet>         s_mtl4_residency      = nil;
static bool                        s_mtl4_initialized    = false;

/* ICB Phase 7 cache.
 * Per-layer 8-slot row, linear-probe + LRU evict (Knuth TAOCP §6.4).
 * Hot-path uniforms (n_rows / n_pairs / clamp / n_down / expert_w[6]) packed
 * into a single 256 B per-slot region of s_icb_uniforms. */
typedef struct {
    uint64_t                       key;        /* sorted-expert hash; 0 = empty */
    id<MTLIndirectCommandBuffer>   icb;
    uint64_t                       last_use;
    size_t                         gate_off[N_EXPERT_USED];
    size_t                         up_off  [N_EXPERT_USED];
    size_t                         down_off[N_EXPERT_USED];
    float                          expert_w[N_EXPERT_USED];
} icb_slot_t;

static icb_slot_t                  s_icb_cache[DS4_N_LAYER][ICB_SLOTS_PER_LAYER];
static id<MTLBuffer>               s_icb_uniforms        = nil;  /* lazy alloc */
static uint64_t                    s_icb_ticket          = 0;
static uint64_t                    s_icb_hits            = 0;
static uint64_t                    s_icb_misses          = 0;
static uint64_t                    s_icb_evicts          = 0;

/* Top-level path selection */
typedef enum {
    PATH_LEGACY = 0,
    PATH_ICB    = 1,
    PATH_MTL4   = 2,
} dispatch_path_t;
static dispatch_path_t             s_active_path         = PATH_LEGACY;
static bool                        s_path_resolved       = false;
static bool                        s_profile_enabled     = false;
#endif /* __APPLE__ */

/* ==========================================================================
 * Public API — toggles and stats.
 * ========================================================================== */
void ds4_metal_vqb2_fp16_set_profile(bool enabled) {
#ifdef __APPLE__
    s_profile_enabled = enabled;
#else
    (void)enabled;
#endif
}

void ds4_metal_vqb2_fp16_icb_stats(uint64_t *hits, uint64_t *misses, uint64_t *evicts) {
#ifdef __APPLE__
    if (hits)   *hits   = s_icb_hits;
    if (misses) *misses = s_icb_misses;
    if (evicts) *evicts = s_icb_evicts;
#else
    if (hits)   *hits   = 0;
    if (misses) *misses = 0;
    if (evicts) *evicts = 0;
#endif
}

#ifdef __APPLE__
/* ==========================================================================
 * Journal hooks — weak default stubs let non-journal build link clean.
 * Strong defs in ds4.c/ds4_cli.c override at link time.
 * ========================================================================== */
__attribute__((weak)) struct ds4_journal *ds4_get_active_journal(void) { return NULL; }
__attribute__((weak)) int64_t             ds4_get_active_session_id(void) { return 0; }

static void emit_dispatch(uint32_t layer, const char *path, uint64_t wall_us,
                          bool icb_hit, int64_t sig) {
#ifdef DS4_JOURNAL_ENABLE
    struct ds4_journal *j = ds4_get_active_journal();
    if (!j) return;
    char payload[256];
    snprintf(payload, sizeof(payload),
        "{\"layer\":%u,\"path\":\"%s\",\"wall_us\":%llu,\"icb_hit\":%s,\"sig\":%lld}",
        (unsigned)layer, path, (unsigned long long)wall_us,
        icb_hit ? "true" : "false", (long long)sig);
    ds4_journal_emit_event(j, ds4_get_active_session_id(),
                           "vqb2_fp16_dispatch", payload);
#else
    (void)layer; (void)path; (void)wall_us; (void)icb_hit; (void)sig;
#endif
}

/* ==========================================================================
 * Hash + ICB-cache lookup (O(1) amortized, O(8) worst with LRU evict).
 * ========================================================================== */
static uint64_t expert_signature(uint32_t layer, const int32_t *exps, int n) {
    int32_t sorted[8] = {0};
    int m = (n > 8) ? 8 : n;
    for (int i = 0; i < m; i++) sorted[i] = exps[i];
    for (int i = 1; i < m; i++) {
        int32_t v = sorted[i]; int j = i - 1;
        while (j >= 0 && sorted[j] > v) { sorted[j+1] = sorted[j]; j--; }
        sorted[j+1] = v;
    }
    uint64_t h = (uint64_t)layer * KNUTH_MULT;
    for (int i = 0; i < m; i++) h = (h ^ (uint32_t)sorted[i]) * KNUTH_MULT;
    return h ? h : 1;
}

static icb_slot_t *icb_lookup_or_evict(uint32_t layer, uint64_t sig, bool *hit) {
    if (layer >= DS4_N_LAYER) { *hit = false; return NULL; }
    icb_slot_t *row = s_icb_cache[layer];
    icb_slot_t *lru = &row[0];
    for (int i = 0; i < ICB_SLOTS_PER_LAYER; i++) {
        if (row[i].key == sig) { row[i].last_use = ++s_icb_ticket; *hit = true; return &row[i]; }
        if (row[i].last_use < lru->last_use) lru = &row[i];
    }
    *hit = false;
    return lru;
}

/* heap_offset_for: convert FP16 pointer (from hot store accessor) to byte
 * offset into the bound MTLBuffer. Returns SIZE_MAX on out-of-range. */
static size_t heap_offset_for(const void *tile_ptr) {
    if (!tile_ptr || !s_bound_heap_ptr) return SIZE_MAX;
    const uintptr_t base = (uintptr_t)s_bound_heap_ptr;
    const uintptr_t tile = (uintptr_t)tile_ptr;
    if (tile < base) return SIZE_MAX;
    const uintptr_t off = tile - base;
    if (off >= (uintptr_t)s_bound_heap_bytes) return SIZE_MAX;
    return (size_t)off;
}

/* Tile-offset triple resolver. Returns 0 on success, -1 if any expert is
 * not pinned in the hot store. */
extern const void *ds4_hot_get_gate_fp16(const struct ds4_hot_expert_store *, uint32_t, uint32_t);
extern const void *ds4_hot_get_up_fp16  (const struct ds4_hot_expert_store *, uint32_t, uint32_t);
extern const void *ds4_hot_get_down_fp16(const struct ds4_hot_expert_store *, uint32_t, uint32_t);

static int resolve_tile_offsets(struct ds4_hot_expert_store *store, uint32_t layer,
                                const int32_t *exps,
                                size_t *gate_off, size_t *up_off, size_t *down_off) {
    for (int s = 0; s < (int)N_EXPERT_USED; s++) {
        int32_t e = exps[s];
        if (e < 0) return -1;
        gate_off[s] = heap_offset_for(ds4_hot_get_gate_fp16(store, layer, (uint32_t)e));
        up_off  [s] = heap_offset_for(ds4_hot_get_up_fp16  (store, layer, (uint32_t)e));
        down_off[s] = heap_offset_for(ds4_hot_get_down_fp16(store, layer, (uint32_t)e));
        if (gate_off[s] == SIZE_MAX || up_off[s] == SIZE_MAX || down_off[s] == SIZE_MAX)
            return -1;
    }
    return 0;
}

/* Pack uniforms into a per-slot 256 B region of s_icb_uniforms.
 *   off+ 0  uint32  n_rows
 *   off+ 4  uint32  n_pairs
 *   off+ 8  float   clamp
 *   off+12  uint32  n_down
 *   off+16  float   expert_w[6]
 */
static void pack_uniforms_at(char *u, const float *expert_weights) {
    *(uint32_t *)(u +  0) = N_ROWS;
    *(uint32_t *)(u +  4) = N_PAIRS;
    *(float    *)(u +  8) = SWIGLU_CLAMP;
    *(uint32_t *)(u + 12) = N_DOWN;
    for (int s = 0; s < (int)N_EXPERT_USED; s++)
        *(float *)(u + 16 + 4 * s) = expert_weights[s];
}

/* ==========================================================================
 * INIT — compile legacy pipelines + alloc scratch + alloc persistent I/O.
 * ========================================================================== */
int ds4_metal_vqb2_fp16_init(void) {
    if (s_initialized) return 0;
    @autoreleasepool {
        s_device = MTLCreateSystemDefaultDevice();
        if (!s_device) { fprintf(stderr, "ds4_vqb2_fp16: no Metal device\n"); return -1; }

        NSError *err = nil;
        NSString *src = [NSString stringWithUTF8String:msl_source];
        s_library = [s_device newLibraryWithSource:src options:nil error:&err];
        if (!s_library) { fprintf(stderr, "ds4_vqb2_fp16: library: %s\n",
            err ? [[err localizedDescription] UTF8String] : "?"); return -1; }

        id<MTLFunction> fn_g = [s_library newFunctionWithName:@"kernel_vqb2_fp16_gate_up"];
        id<MTLFunction> fn_s = [s_library newFunctionWithName:@"kernel_vqb2_fp16_swiglu"];
        id<MTLFunction> fn_d = [s_library newFunctionWithName:@"kernel_vqb2_fp16_down"];
        if (!fn_g || !fn_s || !fn_d) {
            fprintf(stderr, "ds4_vqb2_fp16: missing kernel\n"); return -1;
        }
        s_pipeline_gate_up = [s_device newComputePipelineStateWithFunction:fn_g error:&err];
        s_pipeline_swiglu  = [s_device newComputePipelineStateWithFunction:fn_s error:&err];
        s_pipeline_down    = [s_device newComputePipelineStateWithFunction:fn_d error:&err];
        if (!s_pipeline_gate_up || !s_pipeline_swiglu || !s_pipeline_down) {
            fprintf(stderr, "ds4_vqb2_fp16: pipeline state\n"); return -1;
        }
        s_command_queue       = [s_device newCommandQueue];
        s_scratch_gate_out    = [s_device newBufferWithLength:SCRATCH_BYTES options:MTLResourceStorageModeShared];
        s_scratch_up_out      = [s_device newBufferWithLength:SCRATCH_BYTES options:MTLResourceStorageModeShared];
        s_scratch_mid         = [s_device newBufferWithLength:SCRATCH_BYTES options:MTLResourceStorageModeShared];
        s_persistent_input    = [s_device newBufferWithLength:IO_PAGE_BYTES options:MTLResourceStorageModeShared];
        s_persistent_output   = [s_device newBufferWithLength:IO_PAGE_BYTES options:MTLResourceStorageModeShared];
        if (!s_command_queue || !s_scratch_gate_out || !s_scratch_up_out || !s_scratch_mid ||
            !s_persistent_input || !s_persistent_output) {
            fprintf(stderr, "ds4_vqb2_fp16: buffer alloc\n"); return -1;
        }
        s_initialized = true;
        fprintf(stderr, "ds4_vqb2_fp16_init: 3 pipelines + queue + 3×%zuB scratch + 2×%uB I/O\n",
                (size_t)SCRATCH_BYTES, (unsigned)IO_PAGE_BYTES);
        return 0;
    }
}

int ds4_metal_vqb2_fp16_init_mtl4(void) {
    if (s_mtl4_initialized) return 0;
    if (!s_initialized && ds4_metal_vqb2_fp16_init() != 0) return -1;
    @autoreleasepool {
        NSError *err = nil;
        s_mtl4_compiler = [s_device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&err];
        if (!s_mtl4_compiler) { fprintf(stderr, "mtl4: compiler\n"); return -1; }

        MTL4LibraryDescriptor *libDesc = [MTL4LibraryDescriptor new];
        libDesc.source = [NSString stringWithUTF8String:msl_source];
        libDesc.name   = @"ds4_vqb2_fp16_mtl4";
        s_mtl4_library = [s_mtl4_compiler newLibraryWithDescriptor:libDesc error:&err];
        if (!s_mtl4_library) { fprintf(stderr, "mtl4: library\n"); return -1; }

        /* compile_mtl4_pipeline: helper as inline-ish macro since 3 args differ */
        #define COMPILE_MTL4(kname, var) do { \
            MTL4LibraryFunctionDescriptor *fn = [MTL4LibraryFunctionDescriptor new]; \
            fn.library = s_mtl4_library; fn.name = @kname; \
            MTL4ComputePipelineDescriptor *pd = [MTL4ComputePipelineDescriptor new]; \
            pd.computeFunctionDescriptor = fn; \
            pd.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES; \
            pd.maxTotalThreadsPerThreadgroup = 256; \
            (var) = [s_mtl4_compiler newComputePipelineStateWithDescriptor:pd \
                                                      compilerTaskOptions:nil error:&err]; \
            if (!(var)) { fprintf(stderr, "mtl4: pipeline %s\n", kname); return -1; } \
        } while (0)
        COMPILE_MTL4("kernel_vqb2_fp16_gate_up", s_mtl4_pipeline_gate_up);
        COMPILE_MTL4("kernel_vqb2_fp16_swiglu",  s_mtl4_pipeline_swiglu);
        COMPILE_MTL4("kernel_vqb2_fp16_down",    s_mtl4_pipeline_down);
        #undef COMPILE_MTL4

        s_mtl4_queue = [s_device newMTL4CommandQueueWithDescriptor:[MTL4CommandQueueDescriptor new]
                                                              error:&err];
        s_mtl4_allocator = [s_device newCommandAllocatorWithDescriptor:[MTL4CommandAllocatorDescriptor new]
                                                                  error:&err];
        if (!s_mtl4_queue || !s_mtl4_allocator) { fprintf(stderr, "mtl4: queue/allocator\n"); return -1; }
        s_mtl4_initialized = true;
        fprintf(stderr, "ds4_vqb2_fp16_init_mtl4: ready\n");
        return 0;
    }
}

int ds4_metal_vqb2_fp16_bind_store(struct ds4_hot_expert_store *store) {
    if (!s_initialized && ds4_metal_vqb2_fp16_init() != 0) return -1;
    if (!store) return -1;
    extern void    *ds4_hot_store_heap_ptr     (const struct ds4_hot_expert_store *);
    extern uint64_t ds4_hot_store_heap_bytes_get(const struct ds4_hot_expert_store *);
    const void *heap = ds4_hot_store_heap_ptr(store);
    const uint64_t bytes = ds4_hot_store_heap_bytes_get(store);
    if (!heap || bytes == 0) { fprintf(stderr, "bind_store: empty heap\n"); return -1; }
    if (s_bound_heap_ptr == heap && s_bound_heap_bytes == bytes && s_hot_store_buffer != nil)
        return 0;
    s_hot_store_buffer = nil; s_bound_heap_ptr = NULL; s_bound_heap_bytes = 0;
    @autoreleasepool {
        s_hot_store_buffer = [s_device newBufferWithBytesNoCopy:(void *)heap
                                                        length:(NSUInteger)bytes
                                                       options:MTLResourceStorageModeShared
                                                   deallocator:nil];
        if (!s_hot_store_buffer) { fprintf(stderr, "bind_store: wrap fail\n"); return -1; }
        s_bound_heap_ptr = heap; s_bound_heap_bytes = (size_t)bytes;
        fprintf(stderr, "bind_store: %.1f MB FP16 heap bound\n", (double)bytes / 1e6);
        return 0;
    }
}

/* Lazy MTL4 arg-table + residency-set init.
 * Buffers bound via MTL4 argument tables READ AS ZERO unless declared in a
 * residency set with requestResidency + addResidencySet to queue. Codified
 * here in the init path so callers get the right state by construction. */
static int init_mtl4_arg_table(void) {
    if (s_mtl4_arg_table && s_mtl4_residency) return 0;
    if (!s_mtl4_initialized && ds4_metal_vqb2_fp16_init_mtl4() != 0) return -1;
    @autoreleasepool {
        NSError *err = nil;
        if (!s_mtl4_arg_table) {
            MTL4ArgumentTableDescriptor *d = [MTL4ArgumentTableDescriptor new];
            d.maxBufferBindCount = 8; d.initializeBindings = YES;
            s_mtl4_arg_table = [s_device newArgumentTableWithDescriptor:d error:&err];
            if (!s_mtl4_arg_table) { fprintf(stderr, "mtl4: argTable\n"); return -1; }
        }
        if (!s_mtl4_residency) {
            MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
            rd.label = @"vqb2_fp16_residency"; rd.initialCapacity = 16;
            s_mtl4_residency = [s_device newResidencySetWithDescriptor:rd error:&err];
            if (!s_mtl4_residency) { fprintf(stderr, "mtl4: residency\n"); return -1; }
            id<MTLAllocation> allocs[] = {
                (id<MTLAllocation>)s_hot_store_buffer,
                (id<MTLAllocation>)s_persistent_input,
                (id<MTLAllocation>)s_persistent_output,
                (id<MTLAllocation>)s_scratch_gate_out,
                (id<MTLAllocation>)s_scratch_up_out,
                (id<MTLAllocation>)s_scratch_mid,
                (id<MTLAllocation>)s_icb_uniforms,
            };
            for (NSUInteger i = 0; i < sizeof(allocs)/sizeof(allocs[0]); i++)
                if (allocs[i]) [s_mtl4_residency addAllocation:allocs[i]];
            [s_mtl4_residency commit];
            [s_mtl4_residency requestResidency];
            [s_mtl4_queue addResidencySet:s_mtl4_residency];
        }
    }
    return 0;
}

/* ==========================================================================
 * BACKEND 1 — Legacy direct-encoding (MTLComputeCommandEncoder).
 * ========================================================================== */
int ds4_metal_vqb2_fp16_dispatch_legacy(struct ds4_hot_expert_store *store,
                                        uint32_t layer, uint32_t n_tokens,
                                        const int32_t *selected_exps,
                                        const float *expert_weights,
                                        const void *input_fp32, void *output_fp32) {
    if (!s_initialized) return -1;
    if (!store || !selected_exps || !expert_weights || !input_fp32 || !output_fp32) return -1;
    if (n_tokens != 1) return -2;
    if (!s_hot_store_buffer && ds4_metal_vqb2_fp16_bind_store(store) != 0) return -1;

    const size_t io_bytes = (size_t)N_ROWS * 2u * sizeof(float);
    const size_t io_wrap  = (io_bytes + 16383u) & ~((size_t)16383u);

    @autoreleasepool {
        id<MTLBuffer> in_buf  = [s_device newBufferWithBytesNoCopy:(void *)input_fp32
                                  length:io_wrap options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> out_buf = [s_device newBufferWithBytesNoCopy:output_fp32
                                  length:io_wrap options:MTLResourceStorageModeShared deallocator:nil];
        if (!in_buf || !out_buf) { fprintf(stderr, "legacy: I/O wrap\n"); return -2; }

        id<MTLCommandBuffer> cmd = [s_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!cmd || !enc) return -2;

        const uint32_t n_rows = N_ROWS, n_pairs = N_PAIRS, n_down = N_DOWN;
        const float    clamp  = SWIGLU_CLAMP;
        const NSUInteger tg_pairs = MIN((NSUInteger)128, (NSUInteger)n_pairs);
        const NSUInteger tg_rows  = MIN((NSUInteger)128, (NSUInteger)n_rows);
        const NSUInteger grid_pairs = ((n_pairs + tg_pairs - 1) / tg_pairs) * tg_pairs;
        const NSUInteger grid_rows  = ((n_rows  + tg_rows  - 1) / tg_rows ) * tg_rows;

        for (uint32_t s = 0; s < N_EXPERT_USED; s++) {
            const int32_t exp_id = selected_exps[s];
            if (exp_id < 0) continue;
            const size_t gate_o = heap_offset_for(ds4_hot_get_gate_fp16(store, layer, (uint32_t)exp_id));
            const size_t up_o   = heap_offset_for(ds4_hot_get_up_fp16  (store, layer, (uint32_t)exp_id));
            const size_t down_o = heap_offset_for(ds4_hot_get_down_fp16(store, layer, (uint32_t)exp_id));
            if (gate_o == SIZE_MAX || up_o == SIZE_MAX || down_o == SIZE_MAX) {
                [enc endEncoding]; return -1;
            }

            /* gate matvec */
            [enc setComputePipelineState:s_pipeline_gate_up];
            [enc setBuffer:s_hot_store_buffer offset:(NSUInteger)gate_o atIndex:0];
            [enc setBuffer:in_buf             offset:0 atIndex:1];
            [enc setBuffer:s_scratch_gate_out offset:0 atIndex:2];
            [enc setBytes:&n_rows  length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&n_pairs length:sizeof(uint32_t) atIndex:4];
            [enc dispatchThreads:MTLSizeMake(grid_pairs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_pairs, 1, 1)];
            /* up matvec — same pipeline, different tile + output */
            [enc setBuffer:s_hot_store_buffer offset:(NSUInteger)up_o atIndex:0];
            [enc setBuffer:s_scratch_up_out   offset:0 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(grid_pairs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_pairs, 1, 1)];
            /* SwiGLU fuse */
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
            /* down matvec */
            [enc setComputePipelineState:s_pipeline_down];
            [enc setBuffer:s_hot_store_buffer offset:(NSUInteger)down_o atIndex:0];
            [enc setBuffer:s_scratch_mid      offset:0 atIndex:1];
            [enc setBuffer:out_buf            offset:0 atIndex:2];
            [enc setBytes:&n_rows length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&n_down length:sizeof(uint32_t) atIndex:4];
            [enc dispatchThreads:MTLSizeMake(grid_rows, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_rows, 1, 1)];
        }
        [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
        if (cmd.error) { fprintf(stderr, "legacy: %s\n", [[cmd.error localizedDescription] UTF8String]); return -2; }
        return 0;
    }
}

/* ==========================================================================
 * BACKEND 2 — ICB Phase 7 record-replay.
 * On miss: record 24 commands into a fresh MTLIndirectCommandBuffer.
 * On hit (same expert set): memcpy 24 B of weights into uniforms; replay.
 * Persistent I/O buffers keep ICB references valid across tokens.
 * ========================================================================== */

/* Ensure ICB allocated for this slot. */
static id<MTLIndirectCommandBuffer> ensure_icb(icb_slot_t *slot) {
    if (slot->icb) return slot->icb;
    MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
    d.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
    d.inheritBuffers = NO; d.inheritPipelineState = NO;
    d.maxKernelBufferBindCount = 3;
    slot->icb = [s_device newIndirectCommandBufferWithDescriptor:d
                                                 maxCommandCount:ICB_CMDS_PER_DISP
                                                         options:MTLResourceStorageModeShared];
    return slot->icb;
}

/* Update expert_weights region of uniforms — cheap-path on cache hit. */
static void update_weights_only(icb_slot_t *slot, const float *expert_weights) {
    if (!s_icb_uniforms || !slot->icb) return;
    const ptrdiff_t base = (ptrdiff_t)((uintptr_t)slot - (uintptr_t)s_icb_cache) / (ptrdiff_t)sizeof(icb_slot_t);
    char *u = (char *)s_icb_uniforms.contents + (size_t)base * 256u;
    for (int s = 0; s < (int)N_EXPERT_USED; s++) {
        *(float *)(u + 16 + 4 * s) = expert_weights[s];
        slot->expert_w[s] = expert_weights[s];
    }
}

/* Record 24 commands; bake offsets + uniform-bindings into the ICB. */
static void record_icb(icb_slot_t *slot, uint64_t sig, struct ds4_hot_expert_store *store,
                       uint32_t layer, const int32_t *selected, const float *expert_weights,
                       id<MTLBuffer> in_buf, id<MTLBuffer> out_buf) {
    id<MTLIndirectCommandBuffer> icb = ensure_icb(slot);
    if (!icb) return;

    /* Lazy alloc shared uniforms buffer (one 256 B slot per (layer, slot)). */
    if (!s_icb_uniforms) {
        s_icb_uniforms = [s_device newBufferWithLength:(NSUInteger)(256u * DS4_N_LAYER * ICB_SLOTS_PER_LAYER)
                                              options:MTLResourceStorageModeShared];
        if (!s_icb_uniforms) return;
    }
    const ptrdiff_t base = (ptrdiff_t)((uintptr_t)slot - (uintptr_t)s_icb_cache) / (ptrdiff_t)sizeof(icb_slot_t);
    const size_t u_off = (size_t)base * 256u;
    pack_uniforms_at((char *)s_icb_uniforms.contents + u_off, expert_weights);

    if (resolve_tile_offsets(store, layer, selected,
                             slot->gate_off, slot->up_off, slot->down_off) != 0) {
        slot->key = 0; return;
    }
    for (int s = 0; s < (int)N_EXPERT_USED; s++) slot->expert_w[s] = expert_weights[s];

    const NSUInteger tg_pairs = MIN((NSUInteger)128, (NSUInteger)N_PAIRS);
    const NSUInteger tg_rows  = MIN((NSUInteger)128, (NSUInteger)N_ROWS);
    const NSUInteger gp = (N_PAIRS + tg_pairs - 1) / tg_pairs;
    const NSUInteger gr = (N_ROWS  + tg_rows  - 1) / tg_rows;

    /* 6 experts × 4 commands each (gate / up / swiglu / down). */
    for (int s = 0; s < (int)N_EXPERT_USED; s++) {
        const uint32_t base_cmd = (uint32_t)(s * 4);
        id<MTLIndirectComputeCommand> c0 = [icb indirectComputeCommandAtIndex:base_cmd + 0];
        [c0 setComputePipelineState:s_pipeline_gate_up];
        [c0 setKernelBuffer:s_hot_store_buffer offset:(NSUInteger)slot->gate_off[s] atIndex:0];
        [c0 setKernelBuffer:in_buf             offset:0 atIndex:1];
        [c0 setKernelBuffer:s_scratch_gate_out offset:0 atIndex:2];
        [c0 setKernelBuffer:s_icb_uniforms offset:(NSUInteger)(u_off + 0) atIndex:3];
        [c0 setKernelBuffer:s_icb_uniforms offset:(NSUInteger)(u_off + 4) atIndex:4];
        [c0 concurrentDispatchThreadgroups:MTLSizeMake(gp, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(tg_pairs, 1, 1)];

        id<MTLIndirectComputeCommand> c1 = [icb indirectComputeCommandAtIndex:base_cmd + 1];
        [c1 setComputePipelineState:s_pipeline_gate_up];
        [c1 setKernelBuffer:s_hot_store_buffer offset:(NSUInteger)slot->up_off[s] atIndex:0];
        [c1 setKernelBuffer:in_buf             offset:0 atIndex:1];
        [c1 setKernelBuffer:s_scratch_up_out   offset:0 atIndex:2];
        [c1 setKernelBuffer:s_icb_uniforms offset:(NSUInteger)(u_off + 0) atIndex:3];
        [c1 setKernelBuffer:s_icb_uniforms offset:(NSUInteger)(u_off + 4) atIndex:4];
        [c1 concurrentDispatchThreadgroups:MTLSizeMake(gp, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(tg_pairs, 1, 1)];

        id<MTLIndirectComputeCommand> c2 = [icb indirectComputeCommandAtIndex:base_cmd + 2];
        [c2 setComputePipelineState:s_pipeline_swiglu];
        [c2 setKernelBuffer:s_scratch_gate_out offset:0 atIndex:0];
        [c2 setKernelBuffer:s_scratch_up_out   offset:0 atIndex:1];
        [c2 setKernelBuffer:s_scratch_mid      offset:0 atIndex:2];
        [c2 setKernelBuffer:s_icb_uniforms offset:(NSUInteger)(u_off + 4) atIndex:3];
        [c2 setKernelBuffer:s_icb_uniforms offset:(NSUInteger)(u_off + 8) atIndex:4];
        [c2 setKernelBuffer:s_icb_uniforms offset:(NSUInteger)(u_off + 16 + 4*s) atIndex:5];
        [c2 concurrentDispatchThreadgroups:MTLSizeMake(gp, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(tg_pairs, 1, 1)];

        id<MTLIndirectComputeCommand> c3 = [icb indirectComputeCommandAtIndex:base_cmd + 3];
        [c3 setComputePipelineState:s_pipeline_down];
        [c3 setKernelBuffer:s_hot_store_buffer offset:(NSUInteger)slot->down_off[s] atIndex:0];
        [c3 setKernelBuffer:s_scratch_mid      offset:0 atIndex:1];
        [c3 setKernelBuffer:out_buf            offset:0 atIndex:2];
        [c3 setKernelBuffer:s_icb_uniforms offset:(NSUInteger)(u_off + 0)  atIndex:3];
        [c3 setKernelBuffer:s_icb_uniforms offset:(NSUInteger)(u_off + 12) atIndex:4];
        [c3 concurrentDispatchThreadgroups:MTLSizeMake(gr, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(tg_rows, 1, 1)];
    }
    slot->key = sig;
}

int ds4_metal_vqb2_fp16_dispatch_icb(struct ds4_hot_expert_store *store,
                                     uint32_t layer, uint32_t n_tokens,
                                     const int32_t *selected_exps,
                                     const float *expert_weights,
                                     const void *input_fp32, void *output_fp32) {
    if (!s_initialized) return -1;
    if (!store || !selected_exps || !expert_weights || !input_fp32 || !output_fp32) return -1;
    if (n_tokens != 1) return -2;
    if (!s_hot_store_buffer && ds4_metal_vqb2_fp16_bind_store(store) != 0) return -1;
    if (layer >= DS4_N_LAYER) return -1;

    struct timeval tv0; gettimeofday(&tv0, NULL);
    const uint64_t sig = expert_signature(layer, selected_exps, N_EXPERT_USED);
    bool hit = false;
    icb_slot_t *slot = icb_lookup_or_evict(layer, sig, &hit);
    if (!slot) return -1;

    const size_t io_bytes = (size_t)N_ROWS * 2u * sizeof(float);
    memcpy(s_persistent_input.contents, input_fp32, io_bytes);
    memset(s_persistent_output.contents, 0, io_bytes);

    @autoreleasepool {
        if (!hit || slot->icb == nil) {
            if (slot->key != 0) s_icb_evicts++;
            s_icb_misses++;
            record_icb(slot, sig, store, layer, selected_exps, expert_weights,
                       s_persistent_input, s_persistent_output);
            if (slot->key == 0) {
                struct timeval tv1; gettimeofday(&tv1, NULL);
                emit_dispatch(layer, "icb_reject",
                              (uint64_t)((tv1.tv_sec - tv0.tv_sec) * 1000000ULL +
                                         (tv1.tv_usec - tv0.tv_usec)),
                              false, (int64_t)sig);
                return -1;
            }
        } else {
            s_icb_hits++;
            update_weights_only(slot, expert_weights);
        }

        id<MTLCommandBuffer> cmd = [s_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!cmd || !enc) return -2;
        [enc useResource:s_hot_store_buffer  usage:MTLResourceUsageRead];
        [enc useResource:s_persistent_input  usage:MTLResourceUsageRead];
        [enc useResource:s_persistent_output usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        [enc useResource:s_scratch_gate_out  usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        [enc useResource:s_scratch_up_out    usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        [enc useResource:s_scratch_mid       usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        [enc useResource:s_icb_uniforms      usage:MTLResourceUsageRead];
        [enc executeCommandsInBuffer:slot->icb withRange:NSMakeRange(0, ICB_CMDS_PER_DISP)];
        [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
        if (cmd.error) { fprintf(stderr, "icb: %s\n", [[cmd.error localizedDescription] UTF8String]); return -2; }
        memcpy(output_fp32, s_persistent_output.contents, io_bytes);
    }

    struct timeval tv1; gettimeofday(&tv1, NULL);
    emit_dispatch(layer, "icb",
                  (uint64_t)((tv1.tv_sec - tv0.tv_sec) * 1000000ULL +
                             (tv1.tv_usec - tv0.tv_usec)),
                  hit, (int64_t)sig);
    return 0;
}

/* ==========================================================================
 * BACKEND 3 — MTL4 argument-table dispatch.
 * Per-dispatch: setAddress on persistent argTable + dispatchThreadgroups.
 * Residency set declared at init (so arg-table reads aren't silently zero).
 * ========================================================================== */
int ds4_metal_vqb2_fp16_dispatch_mtl4(struct ds4_hot_expert_store *store,
                                      uint32_t layer, uint32_t n_tokens,
                                      const int32_t *selected_exps,
                                      const float *expert_weights,
                                      const void *input_fp32, void *output_fp32) {
    if (!s_initialized) return -1;
    if (!store || !selected_exps || !expert_weights || !input_fp32 || !output_fp32) return -1;
    if (n_tokens != 1) return -2;
    if (!s_hot_store_buffer && ds4_metal_vqb2_fp16_bind_store(store) != 0) return -1;
    if (init_mtl4_arg_table() != 0) return -1;

    struct timeval tv0; gettimeofday(&tv0, NULL);
    const uint64_t sig = expert_signature(layer, selected_exps, N_EXPERT_USED);

    size_t gate_off[N_EXPERT_USED], up_off[N_EXPERT_USED], down_off[N_EXPERT_USED];
    if (resolve_tile_offsets(store, layer, selected_exps, gate_off, up_off, down_off) != 0)
        return -1;

    const size_t io_bytes = (size_t)N_ROWS * 2u * sizeof(float);
    memcpy(s_persistent_input.contents, input_fp32, io_bytes);
    memset(s_persistent_output.contents, 0, io_bytes);
    pack_uniforms_at((char *)s_icb_uniforms.contents, expert_weights);

    const uint64_t hot_b   = s_hot_store_buffer.gpuAddress;
    const uint64_t in_a    = s_persistent_input.gpuAddress;
    const uint64_t out_a   = s_persistent_output.gpuAddress;
    const uint64_t sg_a    = s_scratch_gate_out.gpuAddress;
    const uint64_t su_a    = s_scratch_up_out.gpuAddress;
    const uint64_t sm_a    = s_scratch_mid.gpuAddress;
    const uint64_t un_a    = s_icb_uniforms.gpuAddress;

    const NSUInteger tg_pairs = MIN((NSUInteger)128, (NSUInteger)N_PAIRS);
    const NSUInteger tg_rows  = MIN((NSUInteger)128, (NSUInteger)N_ROWS);
    const NSUInteger gp = (N_PAIRS + tg_pairs - 1) / tg_pairs;
    const NSUInteger gr = (N_ROWS  + tg_rows  - 1) / tg_rows;

    @autoreleasepool {
        [s_mtl4_allocator reset];
        id<MTL4CommandBuffer> cb = [s_device newCommandBuffer];
        if (!cb) return -2;
        [cb beginCommandBufferWithAllocator:s_mtl4_allocator];
        [cb useResidencySet:s_mtl4_residency];
        id<MTL4ComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) { [cb endCommandBuffer]; return -2; }
        [enc setArgumentTable:s_mtl4_arg_table];

        for (int s = 0; s < (int)N_EXPERT_USED; s++) {
            /* gate */
            [enc setComputePipelineState:s_mtl4_pipeline_gate_up];
            [s_mtl4_arg_table setAddress:(hot_b + gate_off[s]) atIndex:0];
            [s_mtl4_arg_table setAddress:in_a                   atIndex:1];
            [s_mtl4_arg_table setAddress:sg_a                   atIndex:2];
            [s_mtl4_arg_table setAddress:(un_a + 0)             atIndex:3];
            [s_mtl4_arg_table setAddress:(un_a + 4)             atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(gp, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_pairs, 1, 1)];
            /* up — same pipeline, different tile + output */
            [s_mtl4_arg_table setAddress:(hot_b + up_off[s]) atIndex:0];
            [s_mtl4_arg_table setAddress:su_a                 atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(gp, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_pairs, 1, 1)];
            /* swiglu */
            [enc setComputePipelineState:s_mtl4_pipeline_swiglu];
            [s_mtl4_arg_table setAddress:sg_a                       atIndex:0];
            [s_mtl4_arg_table setAddress:su_a                       atIndex:1];
            [s_mtl4_arg_table setAddress:sm_a                       atIndex:2];
            [s_mtl4_arg_table setAddress:(un_a + 4)                 atIndex:3];
            [s_mtl4_arg_table setAddress:(un_a + 8)                 atIndex:4];
            [s_mtl4_arg_table setAddress:(un_a + 16 + 4*s)          atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(gp, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_pairs, 1, 1)];
            /* down */
            [enc setComputePipelineState:s_mtl4_pipeline_down];
            [s_mtl4_arg_table setAddress:(hot_b + down_off[s]) atIndex:0];
            [s_mtl4_arg_table setAddress:sm_a                   atIndex:1];
            [s_mtl4_arg_table setAddress:out_a                  atIndex:2];
            [s_mtl4_arg_table setAddress:(un_a + 0)             atIndex:3];
            [s_mtl4_arg_table setAddress:(un_a + 12)            atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(gr, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_rows, 1, 1)];
        }
        [enc endEncoding]; [cb endCommandBuffer];

        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        __block NSError *fbErr = nil;
        MTL4CommitOptions *opts = [MTL4CommitOptions new];
        [opts addFeedbackHandler:^(id<MTL4CommitFeedback> fb) {
            fbErr = fb.error; dispatch_semaphore_signal(sem);
        }];
        id<MTL4CommandBuffer> bufs[1] = { cb };
        [s_mtl4_queue commit:bufs count:1 options:opts];
        if (dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, 10LL * NSEC_PER_SEC)) != 0) {
            fprintf(stderr, "mtl4: timeout\n"); return -2;
        }
        if (fbErr) { fprintf(stderr, "mtl4: %s\n", [fbErr.localizedDescription UTF8String]); return -2; }
        memcpy(output_fp32, s_persistent_output.contents, io_bytes);
    }

    struct timeval tv1; gettimeofday(&tv1, NULL);
    emit_dispatch(layer, "mtl4",
                  (uint64_t)((tv1.tv_sec - tv0.tv_sec) * 1000000ULL +
                             (tv1.tv_usec - tv0.tv_usec)),
                  false, (int64_t)sig);
    return 0;
}

/* ==========================================================================
 * Top-level dispatcher — env DS4_VQB2_FP16_PATH selects backend once.
 * ========================================================================== */
int ds4_metal_vqb2_fp16_dispatch(struct ds4_hot_expert_store *store,
                                 uint32_t layer, uint32_t n_tokens,
                                 const int32_t *selected_exps,
                                 const float *expert_weights,
                                 const void *input_fp32, void *output_fp32) {
    if (!s_path_resolved) {
        const char *e = getenv("DS4_VQB2_FP16_PATH");
        if (e && *e) {
            if      (strcmp(e, "icb")  == 0) s_active_path = PATH_ICB;
            else if (strcmp(e, "mtl4") == 0) s_active_path = PATH_MTL4;
            else                             s_active_path = PATH_LEGACY;
            fprintf(stderr, "ds4_vqb2_fp16: path = %s\n", e);
        }
        s_path_resolved = true;
    }
    switch (s_active_path) {
        case PATH_ICB:  return ds4_metal_vqb2_fp16_dispatch_icb (store, layer, n_tokens, selected_exps, expert_weights, input_fp32, output_fp32);
        case PATH_MTL4: return ds4_metal_vqb2_fp16_dispatch_mtl4(store, layer, n_tokens, selected_exps, expert_weights, input_fp32, output_fp32);
        case PATH_LEGACY:
        default:        return ds4_metal_vqb2_fp16_dispatch_legacy(store, layer, n_tokens, selected_exps, expert_weights, input_fp32, output_fp32);
    }
}

/* ==========================================================================
 * GPU-handle variant — silv 2026-05-26 directive.
 *
 * Caller passes ds4_gpu_tensor* handles instead of CPU pointers. Skips the
 * newBufferWithBytesNoCopy wrap; reads/writes the same MTLBuffers the
 * engine uses. Caller MUST have called ds4_gpu_end_commands() first so the
 * input tensors' GPU writes are visible.
 *
 * For this iteration the body still routes through the legacy CPU-pointer
 * dispatcher via tensor_contents (on M1 unified memory: same address). The
 * surface change moves the responsibility for sync from "inside dispatcher"
 * (where it doesn't know) to "at caller" (where it does), and prepares for
 * the next-iteration true-shared-cb integration.
 * ========================================================================== */
extern void *ds4_gpu_tensor_contents(struct ds4_gpu_tensor *t);

int ds4_metal_vqb2_fp16_dispatch_gpu(struct ds4_hot_expert_store *store,
                                     uint32_t layer,
                                     uint32_t n_tokens,
                                     struct ds4_gpu_tensor *selected,
                                     struct ds4_gpu_tensor *weights,
                                     struct ds4_gpu_tensor *input,
                                     struct ds4_gpu_tensor *output) {
    if (!store || !selected || !weights || !input || !output) return -1;
    if (n_tokens != 1) return -2;

    const int32_t *sel = (const int32_t *)ds4_gpu_tensor_contents(selected);
    const float   *w   = (const float   *)ds4_gpu_tensor_contents(weights);
    const float   *in  = (const float   *)ds4_gpu_tensor_contents(input);
    float         *out = (float         *)ds4_gpu_tensor_contents(output);
    if (!sel || !w || !in || !out) return -1;

    /* Delegate to the path selected by DS4_VQB2_FP16_PATH (legacy by default). */
    return ds4_metal_vqb2_fp16_dispatch(store, layer, n_tokens, sel, w, in, out);
}

#else  /* non-Apple build */
int ds4_metal_vqb2_fp16_init(void)        { return -1; }
int ds4_metal_vqb2_fp16_init_mtl4(void)   { return -1; }
int ds4_metal_vqb2_fp16_bind_store(struct ds4_hot_expert_store *s) { (void)s; return -1; }
int ds4_metal_vqb2_fp16_dispatch_legacy(struct ds4_hot_expert_store *s, uint32_t l, uint32_t n,
        const int32_t *e, const float *w, const void *i, void *o) {
    (void)s; (void)l; (void)n; (void)e; (void)w; (void)i; (void)o; return -1;
}
int ds4_metal_vqb2_fp16_dispatch_icb(struct ds4_hot_expert_store *s, uint32_t l, uint32_t n,
        const int32_t *e, const float *w, const void *i, void *o) {
    (void)s; (void)l; (void)n; (void)e; (void)w; (void)i; (void)o; return -1;
}
int ds4_metal_vqb2_fp16_dispatch_mtl4(struct ds4_hot_expert_store *s, uint32_t l, uint32_t n,
        const int32_t *e, const float *w, const void *i, void *o) {
    (void)s; (void)l; (void)n; (void)e; (void)w; (void)i; (void)o; return -1;
}
int ds4_metal_vqb2_fp16_dispatch(struct ds4_hot_expert_store *s, uint32_t l, uint32_t n,
        const int32_t *e, const float *w, const void *i, void *o) {
    (void)s; (void)l; (void)n; (void)e; (void)w; (void)i; (void)o; return -1;
}
int ds4_metal_vqb2_fp16_dispatch_gpu(struct ds4_hot_expert_store *s, uint32_t l, uint32_t n,
        struct ds4_gpu_tensor *sel, struct ds4_gpu_tensor *w,
        struct ds4_gpu_tensor *in, struct ds4_gpu_tensor *out) {
    (void)s; (void)l; (void)n; (void)sel; (void)w; (void)in; (void)out; return -1;
}
#endif  /* __APPLE__ */
