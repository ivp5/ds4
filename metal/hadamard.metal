// silv 2026-05-27 — Hadamard-16 batched FP16 kernel.
//
// GGUF-rewrite-unblocker (task #643). Codex H1953 showed that DCT-16 /
// Hadamard-16 basis transforms applied before adjacent-pair VQB2 encoding
// improve reconstruction rel-L2 by ~1-3% across DS4 routed-FFN tensors at
// K=16. To deploy basis-aware quantized weights, the runtime needs to
// apply the SAME orthogonal transform to the matmul's INPUT activation
// before the (already-transformed) weights are consumed; orthogonal
// transforms commute through inner products:
//
//   <w, x> = <H_normalized w, H_normalized x>
//
// where H_16 is the 16-point Walsh-Hadamard matrix (sequency-unordered)
// and H_normalized = H_16 / sqrt(16) = H_16 * 0.25.
//
// This kernel applies H_16 in-place to each 16-element block of an FP16
// activation buffer, scaled by 1/sqrt(16). The buffer is batched: a row
// stride lets the caller process (n_rows × blocks_per_row) blocks per
// dispatch. Each threadgroup handles one block via a 4-stage butterfly
// in threadgroup memory.
//
// Why MTL (Metal) and not MTL4: MTL4Compute is appropriate for kernels
// inside MTL4-managed packages (e.g., polar dot canary at ds4_metal.m
// line 15580+). This kernel runs in the regular DS4 graph alongside the
// existing routed-MoE dispatches, so it shares their MTLCommandBuffer
// queue. Symmetric MTL4 port is a follow-up if/when the routed-MoE path
// moves to MTL4.
//
// Self-inverse property: H_normalized × H_normalized = I, so the same
// kernel can be used for forward (encode-side) and inverse (decode-side)
// transforms. Caller specifies direction by NOT scaling — the kernel
// always applies the orthogonal H_normalized.
//
// Concatenated into the project's combined Metal library by
// ds4_gpu_full_source() in ds4_metal.m; the base source already provides
// `#include <metal_stdlib>` + `using namespace metal;`.

struct ds4_metal_args_hadamard16_batched {
    uint32_t n_rows;
    uint32_t blocks_per_row;
    uint64_t row_stride_bytes;  // distance between row starts in bytes
};

constant float kHadamard16InvSqrt16 = 0.25f;  // 1 / sqrt(16)

kernel void kernel_hadamard16_fp16_batched(
        constant ds4_metal_args_hadamard16_batched & args,
        device half * x,
        threadgroup half * scratch [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const uint row = tgpig.x;
    const uint block = tgpig.y;
    if (row >= args.n_rows || block >= args.blocks_per_row || tid >= 16u) {
        return;
    }

    // Row pointer + block offset within the row (block of 16 halves = 32 B).
    device half * row_ptr = (device half *)((device char *)x +
                                            (uint64_t)row * args.row_stride_bytes);
    device half * block_ptr = row_ptr + (uint64_t)block * 16u;

    // Load 16 halves into threadgroup scratch.
    scratch[tid] = block_ptr[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4 butterfly stages: pair-sums of stride 1, 2, 4, 8.
    // At each stage, half the threads (the "low" partner) compute
    //   scratch[base] = a + b
    //   scratch[base + stride] = a - b
    // where (a, b) = (scratch[base], scratch[base + stride]).
    for (uint stride = 1u; stride < 16u; stride <<= 1u) {
        if ((tid & stride) == 0u) {
            const uint base = (tid & ~(2u * stride - 1u)) + (tid & (stride - 1u));
            const float a = (float)scratch[base];
            const float b = (float)scratch[base + stride];
            scratch[base]          = (half)(a + b);
            scratch[base + stride] = (half)(a - b);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Orthogonal normalization: scale by 1/sqrt(16) = 0.25.
    block_ptr[tid] = (half)((float)scratch[tid] * kHadamard16InvSqrt16);
}
