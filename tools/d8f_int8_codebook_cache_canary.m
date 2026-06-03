#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

typedef unsigned long ulong;

static const char *SRC =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"inline float dot_half4(half4 q, float4 x) {\n"
"  return float(q.x) * x.x + float(q.y) * x.y + float(q.z) * x.z + float(q.w) * x.w;\n"
"}\n"
"\n"
"inline float dot_char4(char4 q, float4 x) {\n"
"  return float(q.x) * x.x + float(q.y) * x.y + float(q.z) * x.z + float(q.w) * x.w;\n"
"}\n"
"\n"
"kernel void score_half_direct(\n"
"    device const half4 *cb [[buffer(0)]],\n"
"    device const float4 *x [[buffer(1)]],\n"
"    device float *out [[buffer(2)]],\n"
"    constant uint &K [[buffer(3)]],\n"
"    constant uint &groups [[buffer(4)]],\n"
"    constant uint &in_dim [[buffer(5)]],\n"
"    uint3 gid [[thread_position_in_grid]]) {\n"
"  const uint code = gid.x;\n"
"  const uint group = gid.y;\n"
"  const uint slot = gid.z;\n"
"  if (code >= K || group >= groups) return;\n"
"  const device half4 *q = cb + code * 2u;\n"
"  const uint x4_base = (slot * in_dim + group * 8u) >> 2;\n"
"  const float acc = dot_half4(q[0], x[x4_base]) + dot_half4(q[1], x[x4_base + 1u]);\n"
"  out[(ulong(slot) * ulong(groups) + ulong(group)) * ulong(K) + ulong(code)] = acc;\n"
"}\n"
"\n"
"kernel void score_half_const(\n"
"    constant const half4 *cb [[buffer(0)]],\n"
"    device const float4 *x [[buffer(1)]],\n"
"    device float *out [[buffer(2)]],\n"
"    constant uint &K [[buffer(3)]],\n"
"    constant uint &groups [[buffer(4)]],\n"
"    constant uint &in_dim [[buffer(5)]],\n"
"    uint3 gid [[thread_position_in_grid]]) {\n"
"  const uint code = gid.x;\n"
"  const uint group = gid.y;\n"
"  const uint slot = gid.z;\n"
"  if (code >= K || group >= groups) return;\n"
"  const constant half4 *q = cb + code * 2u;\n"
"  const uint x4_base = (slot * in_dim + group * 8u) >> 2;\n"
"  const float acc = dot_half4(q[0], x[x4_base]) + dot_half4(q[1], x[x4_base + 1u]);\n"
"  out[(ulong(slot) * ulong(groups) + ulong(group)) * ulong(K) + ulong(code)] = acc;\n"
"}\n"
"\n"
"kernel void score_half_tg16(\n"
"    device const half4 *cb [[buffer(0)]],\n"
"    device const float4 *x [[buffer(1)]],\n"
"    device float *out [[buffer(2)]],\n"
"    constant uint &K [[buffer(3)]],\n"
"    constant uint &groups [[buffer(4)]],\n"
"    constant uint &in_dim [[buffer(5)]],\n"
"    threadgroup half4 *cbt [[threadgroup(0)]],\n"
"    uint3 lid [[thread_position_in_threadgroup]],\n"
"    uint3 tg [[threadgroup_position_in_grid]]) {\n"
"  const uint code = tg.x * 16u + uint(lid.x);\n"
"  const uint group = tg.y * 16u + uint(lid.y);\n"
"  const uint slot = tg.z;\n"
"  if (lid.y == 0u && code < K) {\n"
"    const device half4 *q = cb + code * 2u;\n"
"    cbt[uint(lid.x) * 2u] = q[0];\n"
"    cbt[uint(lid.x) * 2u + 1u] = q[1];\n"
"  }\n"
"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  if (code >= K || group >= groups) return;\n"
"  const uint x4_base = (slot * in_dim + group * 8u) >> 2;\n"
"  const threadgroup half4 *q = cbt + uint(lid.x) * 2u;\n"
"  const float acc = dot_half4(q[0], x[x4_base]) + dot_half4(q[1], x[x4_base + 1u]);\n"
"  out[(ulong(slot) * ulong(groups) + ulong(group)) * ulong(K) + ulong(code)] = acc;\n"
"}\n"
"\n"
"kernel void score_i8_direct(\n"
"    device const char4 *cb [[buffer(0)]],\n"
"    device const float *scales [[buffer(1)]],\n"
"    device const float4 *x [[buffer(2)]],\n"
"    device float *out [[buffer(3)]],\n"
"    constant uint &K [[buffer(4)]],\n"
"    constant uint &groups [[buffer(5)]],\n"
"    constant uint &in_dim [[buffer(6)]],\n"
"    uint3 gid [[thread_position_in_grid]]) {\n"
"  const uint code = gid.x;\n"
"  const uint group = gid.y;\n"
"  const uint slot = gid.z;\n"
"  if (code >= K || group >= groups) return;\n"
"  const device char4 *q = cb + code * 2u;\n"
"  const uint x4_base = (slot * in_dim + group * 8u) >> 2;\n"
"  const float acc = scales[code] * (dot_char4(q[0], x[x4_base]) + dot_char4(q[1], x[x4_base + 1u]));\n"
"  out[(ulong(slot) * ulong(groups) + ulong(group)) * ulong(K) + ulong(code)] = acc;\n"
"}\n"
"\n"
"kernel void score_i8_const(\n"
"    constant const char4 *cb [[buffer(0)]],\n"
"    constant const float *scales [[buffer(1)]],\n"
"    device const float4 *x [[buffer(2)]],\n"
"    device float *out [[buffer(3)]],\n"
"    constant uint &K [[buffer(4)]],\n"
"    constant uint &groups [[buffer(5)]],\n"
"    constant uint &in_dim [[buffer(6)]],\n"
"    uint3 gid [[thread_position_in_grid]]) {\n"
"  const uint code = gid.x;\n"
"  const uint group = gid.y;\n"
"  const uint slot = gid.z;\n"
"  if (code >= K || group >= groups) return;\n"
"  const constant char4 *q = cb + code * 2u;\n"
"  const uint x4_base = (slot * in_dim + group * 8u) >> 2;\n"
"  const float acc = scales[code] * (dot_char4(q[0], x[x4_base]) + dot_char4(q[1], x[x4_base + 1u]));\n"
"  out[(ulong(slot) * ulong(groups) + ulong(group)) * ulong(K) + ulong(code)] = acc;\n"
"}\n"
"\n"
"kernel void score_i8_global_scale(\n"
"    device const char4 *cb [[buffer(0)]],\n"
"    device const float4 *x [[buffer(1)]],\n"
"    device float *out [[buffer(2)]],\n"
"    constant uint &K [[buffer(3)]],\n"
"    constant uint &groups [[buffer(4)]],\n"
"    constant uint &in_dim [[buffer(5)]],\n"
"    constant float &scale [[buffer(6)]],\n"
"    uint3 gid [[thread_position_in_grid]]) {\n"
"  const uint code = gid.x;\n"
"  const uint group = gid.y;\n"
"  const uint slot = gid.z;\n"
"  if (code >= K || group >= groups) return;\n"
"  const device char4 *q = cb + code * 2u;\n"
"  const uint x4_base = (slot * in_dim + group * 8u) >> 2;\n"
"  const float acc = scale * (dot_char4(q[0], x[x4_base]) + dot_char4(q[1], x[x4_base + 1u]));\n"
"  out[(ulong(slot) * ulong(groups) + ulong(group)) * ulong(K) + ulong(code)] = acc;\n"
"}\n"
"\n"
"kernel void score_i8_tg16(\n"
"    device const char4 *cb [[buffer(0)]],\n"
"    device const float *scales [[buffer(1)]],\n"
"    device const float4 *x [[buffer(2)]],\n"
"    device float *out [[buffer(3)]],\n"
"    constant uint &K [[buffer(4)]],\n"
"    constant uint &groups [[buffer(5)]],\n"
"    constant uint &in_dim [[buffer(6)]],\n"
"    threadgroup char4 *cbt [[threadgroup(0)]],\n"
"    threadgroup float *st [[threadgroup(1)]],\n"
"    uint3 lid [[thread_position_in_threadgroup]],\n"
"    uint3 tg [[threadgroup_position_in_grid]]) {\n"
"  const uint code = tg.x * 16u + uint(lid.x);\n"
"  const uint group = tg.y * 16u + uint(lid.y);\n"
"  const uint slot = tg.z;\n"
"  if (lid.y == 0u && code < K) {\n"
"    const device char4 *q = cb + code * 2u;\n"
"    cbt[uint(lid.x) * 2u] = q[0];\n"
"    cbt[uint(lid.x) * 2u + 1u] = q[1];\n"
"    st[uint(lid.x)] = scales[code];\n"
"  }\n"
"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  if (code >= K || group >= groups) return;\n"
"  const uint x4_base = (slot * in_dim + group * 8u) >> 2;\n"
"  const threadgroup char4 *q = cbt + uint(lid.x) * 2u;\n"
"  const float acc = st[uint(lid.x)] * (dot_char4(q[0], x[x4_base]) + dot_char4(q[1], x[x4_base + 1u]));\n"
"  out[(ulong(slot) * ulong(groups) + ulong(group)) * ulong(K) + ulong(code)] = acc;\n"
"}\n";

typedef enum KernelKind {
    KernelKindHalf,
    KernelKindHalfConst,
    KernelKindHalfTg16,
    KernelKindI8,
    KernelKindI8Const,
    KernelKindI8Global,
    KernelKindI8Tg16,
} KernelKind;

static id<MTLBuffer> make_private_buffer(id<MTLDevice> device,
                                         id<MTLCommandQueue> queue,
                                         id<MTLBuffer> src,
                                         NSUInteger length) {
    id<MTLBuffer> dst = [device newBufferWithLength:length options:MTLResourceStorageModePrivate];
    if (!dst) return nil;
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:length];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return dst;
}

static double run_kernel(id<MTLCommandQueue> queue,
                         id<MTLComputePipelineState> pipeline,
                         KernelKind kind,
                         id<MTLBuffer> codebook,
                         id<MTLBuffer> scales,
                         id<MTLBuffer> x,
                         id<MTLBuffer> out,
                         uint32_t K,
                         uint32_t groups,
                         uint32_t slots,
                         uint32_t in_dim,
                         float global_scale,
                         uint32_t iterations) {
    double best = 1e30;
    for (uint32_t iter = 0; iter < iterations; iter++) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        if (kind == KernelKindHalf || kind == KernelKindHalfConst || kind == KernelKindHalfTg16) {
            [enc setBuffer:codebook offset:0 atIndex:0];
            [enc setBuffer:x offset:0 atIndex:1];
            [enc setBuffer:out offset:0 atIndex:2];
            [enc setBytes:&K length:sizeof(K) atIndex:3];
            [enc setBytes:&groups length:sizeof(groups) atIndex:4];
            [enc setBytes:&in_dim length:sizeof(in_dim) atIndex:5];
            if (kind == KernelKindHalfTg16) {
                [enc setThreadgroupMemoryLength:16u * 2u * sizeof(uint64_t) atIndex:0];
            }
        } else if (kind == KernelKindI8Global) {
            [enc setBuffer:codebook offset:0 atIndex:0];
            [enc setBuffer:x offset:0 atIndex:1];
            [enc setBuffer:out offset:0 atIndex:2];
            [enc setBytes:&K length:sizeof(K) atIndex:3];
            [enc setBytes:&groups length:sizeof(groups) atIndex:4];
            [enc setBytes:&in_dim length:sizeof(in_dim) atIndex:5];
            [enc setBytes:&global_scale length:sizeof(global_scale) atIndex:6];
        } else {
            [enc setBuffer:codebook offset:0 atIndex:0];
            [enc setBuffer:scales offset:0 atIndex:1];
            [enc setBuffer:x offset:0 atIndex:2];
            [enc setBuffer:out offset:0 atIndex:3];
            [enc setBytes:&K length:sizeof(K) atIndex:4];
            [enc setBytes:&groups length:sizeof(groups) atIndex:5];
            [enc setBytes:&in_dim length:sizeof(in_dim) atIndex:6];
            if (kind == KernelKindI8Tg16) {
                [enc setThreadgroupMemoryLength:16u * 2u * sizeof(uint32_t) atIndex:0];
                [enc setThreadgroupMemoryLength:16u * sizeof(float) atIndex:1];
            }
        }
        if (kind == KernelKindHalfTg16 || kind == KernelKindI8Tg16) {
            [enc dispatchThreadgroups:MTLSizeMake((K + 15u) >> 4, (groups + 15u) >> 4, slots)
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        } else {
            [enc dispatchThreads:MTLSizeMake(K, groups, slots)
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
        if (ms > 0.0 && ms < best) best = ms;
    }
    return best;
}

static int run_case(id<MTLDevice> device,
                    id<MTLCommandQueue> queue,
                    id<MTLComputePipelineState> half_direct,
                    id<MTLComputePipelineState> half_const,
                    id<MTLComputePipelineState> half_tg16,
                    id<MTLComputePipelineState> i8_direct,
                    id<MTLComputePipelineState> i8_const,
                    id<MTLComputePipelineState> i8_global,
                    id<MTLComputePipelineState> i8_tg16,
                    uint32_t K,
                    uint32_t in_dim,
                    uint32_t slots,
                    uint32_t iterations) {
    const uint32_t groups = in_dim >> 3;
    const float global_scale = 0.004f;
    const NSUInteger cbh_bytes = (NSUInteger)K * 8u * sizeof(uint16_t);
    const NSUInteger cbi_bytes = (NSUInteger)K * 8u;
    const NSUInteger scales_bytes = (NSUInteger)K * sizeof(float);
    const NSUInteger x_bytes = (NSUInteger)slots * in_dim * sizeof(float);
    const NSUInteger out_bytes = (NSUInteger)slots * groups * K * sizeof(float);

    id<MTLBuffer> cbh = [device newBufferWithLength:cbh_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> cbi = [device newBufferWithLength:cbi_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> scales = [device newBufferWithLength:scales_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> x = [device newBufferWithLength:x_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> out = [device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
    if (!cbh || !cbi || !scales || !x || !out) {
        fprintf(stderr, "allocation failed K=%u out=%.2f MiB\n", K, (double)out_bytes / 1048576.0);
        return 0;
    }

    uint16_t *half_codebook = cbh.contents;
    int8_t *int8_codebook = cbi.contents;
    float *scale_values = scales.contents;
    float *x_values = x.contents;
    srand(17u + K);
    for (uint32_t code = 0; code < K; code++) {
        const float scale = global_scale * (0.65f + 0.70f * (float)(code % 113u) / 112.0f);
        scale_values[code] = scale;
        for (uint32_t d = 0; d < 8u; d++) {
            const int value = (rand() % 255) - 127;
            int8_codebook[(NSUInteger)code * 8u + d] = (int8_t)value;
            const __fp16 h = (__fp16)((float)value * scale);
            half_codebook[(NSUInteger)code * 8u + d] = *(const uint16_t *)&h;
        }
    }
    for (NSUInteger i = 0; i < (NSUInteger)slots * in_dim; i++) {
        x_values[i] = sinf((float)i * 0.0019f) + 0.25f * cosf((float)i * 0.017f);
    }

    id<MTLBuffer> cbh_private = make_private_buffer(device, queue, cbh, cbh_bytes);
    id<MTLBuffer> cbi_private = make_private_buffer(device, queue, cbi, cbi_bytes);
    id<MTLBuffer> scales_private = make_private_buffer(device, queue, scales, scales_bytes);

    printf("score-cache K=%u in=%u groups=%u slots=%u out=%.2fMiB it=%u\n",
           K, in_dim, groups, slots, (double)out_bytes / 1048576.0, iterations);

    const double half_s = run_kernel(queue, half_direct, KernelKindHalf, cbh, nil, x, out, K, groups, slots, in_dim, global_scale, iterations);
    const double half_c = run_kernel(queue, half_const, KernelKindHalfConst, cbh, nil, x, out, K, groups, slots, in_dim, global_scale, iterations);
    const double half_tg = run_kernel(queue, half_tg16, KernelKindHalfTg16, cbh, nil, x, out, K, groups, slots, in_dim, global_scale, iterations);
    const double i8_s = run_kernel(queue, i8_direct, KernelKindI8, cbi, scales, x, out, K, groups, slots, in_dim, global_scale, iterations);
    const double i8_c = run_kernel(queue, i8_const, KernelKindI8Const, cbi, scales, x, out, K, groups, slots, in_dim, global_scale, iterations);
    const double i8_g = run_kernel(queue, i8_global, KernelKindI8Global, cbi, nil, x, out, K, groups, slots, in_dim, global_scale, iterations);
    const double i8_tg = run_kernel(queue, i8_tg16, KernelKindI8Tg16, cbi, scales, x, out, K, groups, slots, in_dim, global_scale, iterations);
    printf(" shared half_direct=%.4f half_const=%.4f half_tg16=%.4f i8_direct=%.4f i8_const=%.4f i8_global=%.4f i8_tg16=%.4f\n",
           half_s, half_c, half_tg, i8_s, i8_c, i8_g, i8_tg);
    printf(" shared speed_vs_half: half_const=%.3fx half_tg16=%.3fx i8_direct=%.3fx i8_const=%.3fx i8_global=%.3fx i8_tg16=%.3fx\n",
           half_s / half_c, half_s / half_tg, half_s / i8_s, half_s / i8_c, half_s / i8_g, half_s / i8_tg);

    if (cbh_private && cbi_private && scales_private) {
        const double phalf_s = run_kernel(queue, half_direct, KernelKindHalf, cbh_private, nil, x, out, K, groups, slots, in_dim, global_scale, iterations);
        const double phalf_c = run_kernel(queue, half_const, KernelKindHalfConst, cbh_private, nil, x, out, K, groups, slots, in_dim, global_scale, iterations);
        const double phalf_tg = run_kernel(queue, half_tg16, KernelKindHalfTg16, cbh_private, nil, x, out, K, groups, slots, in_dim, global_scale, iterations);
        const double pi8_s = run_kernel(queue, i8_direct, KernelKindI8, cbi_private, scales_private, x, out, K, groups, slots, in_dim, global_scale, iterations);
        const double pi8_c = run_kernel(queue, i8_const, KernelKindI8Const, cbi_private, scales_private, x, out, K, groups, slots, in_dim, global_scale, iterations);
        const double pi8_g = run_kernel(queue, i8_global, KernelKindI8Global, cbi_private, nil, x, out, K, groups, slots, in_dim, global_scale, iterations);
        const double pi8_tg = run_kernel(queue, i8_tg16, KernelKindI8Tg16, cbi_private, scales_private, x, out, K, groups, slots, in_dim, global_scale, iterations);
        printf("private half_direct=%.4f half_const=%.4f half_tg16=%.4f i8_direct=%.4f i8_const=%.4f i8_global=%.4f i8_tg16=%.4f\n",
               phalf_s, phalf_c, phalf_tg, pi8_s, pi8_c, pi8_g, pi8_tg);
        printf("private speed_vs_half: half_const=%.3fx half_tg16=%.3fx i8_direct=%.3fx i8_const=%.3fx i8_global=%.3fx i8_tg16=%.3fx\n",
               phalf_s / phalf_c, phalf_s / phalf_tg, phalf_s / pi8_s, phalf_s / pi8_c, phalf_s / pi8_g, phalf_s / pi8_tg);
    }
    return 1;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        uint32_t iterations = 60;
        uint32_t slots = 6;
        if (argc > 1) iterations = (uint32_t)strtoul(argv[1], NULL, 10);
        if (argc > 2) slots = (uint32_t)strtoul(argv[2], NULL, 10);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError *err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:nil error:&err];
        if (!lib) {
            NSLog(@"%@", err);
            return 1;
        }
        id<MTLComputePipelineState> half_direct = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"score_half_direct"] error:&err];
        id<MTLComputePipelineState> half_const = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"score_half_const"] error:&err];
        id<MTLComputePipelineState> half_tg16 = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"score_half_tg16"] error:&err];
        id<MTLComputePipelineState> i8_direct = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"score_i8_direct"] error:&err];
        id<MTLComputePipelineState> i8_const = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"score_i8_const"] error:&err];
        id<MTLComputePipelineState> i8_global = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"score_i8_global_scale"] error:&err];
        id<MTLComputePipelineState> i8_tg16 = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"score_i8_tg16"] error:&err];
        if (!half_direct || !half_const || !half_tg16 || !i8_direct || !i8_const || !i8_global || !i8_tg16) {
            NSLog(@"pipeline creation failed: %@", err);
            return 1;
        }
        printf("d8f int8 codebook cache score-phase canary\n");
        if (!run_case(device, queue, half_direct, half_const, half_tg16, i8_direct, i8_const, i8_global, i8_tg16, 1024, 4096, slots, iterations)) return 1;
        if (!run_case(device, queue, half_direct, half_const, half_tg16, i8_direct, i8_const, i8_global, i8_tg16, 2048, 7168, slots, iterations)) return 1;
    }
    return 0;
}
