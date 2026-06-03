#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned long ulong;

static const char *kMetalSource =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"inline float dot_manual(float4 q0, float4 q1, float4 x0, float4 x1) {\n"
"  return q0.x * x0.x + q0.y * x0.y + q0.z * x0.z + q0.w * x0.w +\n"
"         q1.x * x1.x + q1.y * x1.y + q1.z * x1.z + q1.w * x1.w;\n"
"}\n"
"\n"
"inline float dot_intrin(float4 q0, float4 q1, float4 x0, float4 x1) {\n"
"  return dot(q0, x0) + dot(q1, x1);\n"
"}\n"
"\n"
"kernel void score_buffer_manual(\n"
"  device const half4 *codebook [[buffer(0)]],\n"
"  device const float4 *mid [[buffer(1)]],\n"
"  device float *out [[buffer(2)]],\n"
"  constant uint &k [[buffer(3)]],\n"
"  constant uint &groups [[buffer(4)]],\n"
"  constant uint &in_dim [[buffer(5)]],\n"
"  uint3 gid [[thread_position_in_grid]]) {\n"
"  const uint code = gid.x;\n"
"  const uint group = gid.y;\n"
"  const uint slot = gid.z;\n"
"  if (code >= k || group >= groups) return;\n"
"  const device half4 *q = codebook + (ulong(slot) * ulong(k) + ulong(code)) * 2ul;\n"
"  const uint x4 = (slot * in_dim + group * 8u) >> 2;\n"
"  out[(ulong(slot) * ulong(groups) + ulong(group)) * ulong(k) + ulong(code)] =\n"
"      dot_manual(float4(q[0]), float4(q[1]), mid[x4], mid[x4 + 1u]);\n"
"}\n"
"\n"
"kernel void score_buffer_dot(\n"
"  device const half4 *codebook [[buffer(0)]],\n"
"  device const float4 *mid [[buffer(1)]],\n"
"  device float *out [[buffer(2)]],\n"
"  constant uint &k [[buffer(3)]],\n"
"  constant uint &groups [[buffer(4)]],\n"
"  constant uint &in_dim [[buffer(5)]],\n"
"  uint3 gid [[thread_position_in_grid]]) {\n"
"  const uint code = gid.x;\n"
"  const uint group = gid.y;\n"
"  const uint slot = gid.z;\n"
"  if (code >= k || group >= groups) return;\n"
"  const device half4 *q = codebook + (ulong(slot) * ulong(k) + ulong(code)) * 2ul;\n"
"  const uint x4 = (slot * in_dim + group * 8u) >> 2;\n"
"  out[(ulong(slot) * ulong(groups) + ulong(group)) * ulong(k) + ulong(code)] =\n"
"      dot_intrin(float4(q[0]), float4(q[1]), mid[x4], mid[x4 + 1u]);\n"
"}\n"
"\n"
"kernel void score_texture_buffer(\n"
"  texture_buffer<half, access::read> codebook [[texture(0)]],\n"
"  device const float4 *mid [[buffer(0)]],\n"
"  device float *out [[buffer(1)]],\n"
"  constant uint &k [[buffer(2)]],\n"
"  constant uint &groups [[buffer(3)]],\n"
"  constant uint &in_dim [[buffer(4)]],\n"
"  uint3 gid [[thread_position_in_grid]]) {\n"
"  const uint code = gid.x;\n"
"  const uint group = gid.y;\n"
"  const uint slot = gid.z;\n"
"  if (code >= k || group >= groups) return;\n"
"  const uint base = (slot * k + code) * 2u;\n"
"  const float4 q0 = float4(codebook.read(base));\n"
"  const float4 q1 = float4(codebook.read(base + 1u));\n"
"  const uint x4 = (slot * in_dim + group * 8u) >> 2;\n"
"  out[(ulong(slot) * ulong(groups) + ulong(group)) * ulong(k) + ulong(code)] = dot_intrin(q0, q1, mid[x4], mid[x4 + 1u]);\n"
"}\n"
"\n"
"kernel void score_texture2d_read(\n"
"  texture2d<half, access::read> codebook [[texture(0)]],\n"
"  device const float4 *mid [[buffer(0)]],\n"
"  device float *out [[buffer(1)]],\n"
"  constant uint &k [[buffer(2)]],\n"
"  constant uint &groups [[buffer(3)]],\n"
"  constant uint &in_dim [[buffer(4)]],\n"
"  uint3 gid [[thread_position_in_grid]]) {\n"
"  const uint code = gid.x;\n"
"  const uint group = gid.y;\n"
"  const uint slot = gid.z;\n"
"  if (code >= k || group >= groups) return;\n"
"  const float4 q0 = float4(codebook.read(uint2(code * 2u, slot)));\n"
"  const float4 q1 = float4(codebook.read(uint2(code * 2u + 1u, slot)));\n"
"  const uint x4 = (slot * in_dim + group * 8u) >> 2;\n"
"  out[(ulong(slot) * ulong(groups) + ulong(group)) * ulong(k) + ulong(code)] = dot_intrin(q0, q1, mid[x4], mid[x4 + 1u]);\n"
"}\n"
"\n"
"kernel void score_texture2d_sample(\n"
"  texture2d<float, access::sample> codebook [[texture(0)]],\n"
"  device const float4 *mid [[buffer(0)]],\n"
"  device float *out [[buffer(1)]],\n"
"  constant uint &k [[buffer(2)]],\n"
"  constant uint &groups [[buffer(3)]],\n"
"  constant uint &in_dim [[buffer(4)]],\n"
"  uint3 gid [[thread_position_in_grid]]) {\n"
"  constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);\n"
"  const uint code = gid.x;\n"
"  const uint group = gid.y;\n"
"  const uint slot = gid.z;\n"
"  if (code >= k || group >= groups) return;\n"
"  const float y = float(slot) + 0.5f;\n"
"  const float4 q0 = codebook.sample(s, float2(float(code * 2u) + 0.5f, y));\n"
"  const float4 q1 = codebook.sample(s, float2(float(code * 2u + 1u) + 0.5f, y));\n"
"  const uint x4 = (slot * in_dim + group * 8u) >> 2;\n"
"  out[(ulong(slot) * ulong(groups) + ulong(group)) * ulong(k) + ulong(code)] = dot_intrin(q0, q1, mid[x4], mid[x4 + 1u]);\n"
"}\n"
"\n"
"kernel void raw_texture2d_gather_x(\n"
"  texture2d<float, access::sample> codebook [[texture(0)]],\n"
"  device float *out [[buffer(0)]],\n"
"  constant uint &texels_per_slot [[buffer(1)]],\n"
"  constant uint &groups [[buffer(2)]],\n"
"  uint3 gid [[thread_position_in_grid]]) {\n"
"  constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);\n"
"  const uint x = min(gid.x * 2u, texels_per_slot - 1u);\n"
"  const uint group = gid.y;\n"
"  const uint slot = gid.z;\n"
"  const float4 g = codebook.gather(s, float2(float(x) + 0.5f, float(slot) + 0.5f));\n"
"  out[(ulong(slot) * ulong(groups) + ulong(group)) * ulong((texels_per_slot + 1u) >> 1) + ulong(gid.x)] =\n"
"      g.x + g.y + g.z + g.w;\n"
"}\n";

typedef enum KernelKind {
    KernelKindBufferManual,
    KernelKindBufferDot,
    KernelKindTextureBuffer,
    KernelKindTexture2DRead,
    KernelKindTexture2DSample,
    KernelKindGatherRaw,
} KernelKind;

static uint16_t f32_to_f16(float value) {
    __fp16 half_value = (__fp16)value;
    uint16_t bits = 0;
    memcpy(&bits, &half_value, sizeof(bits));
    return bits;
}

static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString *name) {
    NSError *error = nil;
    id<MTLFunction> function = [library newFunctionWithName:name];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) NSLog(@"pipeline %@ failed: %@", name, error);
    return pipeline;
}

static id<MTLBuffer> make_private_buffer(id<MTLDevice> device,
                                         id<MTLCommandQueue> queue,
                                         id<MTLBuffer> src,
                                         NSUInteger length) {
    id<MTLBuffer> dst = [device newBufferWithLength:length options:MTLResourceStorageModePrivate];
    if (!dst) return nil;
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
    [blit copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:length];
    [blit endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    return dst;
}

static id<MTLTexture> make_texture2d(id<MTLBuffer> buffer, uint32_t width, uint32_t height) {
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                          width:width
                                                                                         height:height
                                                                                      mipmapped:NO];
    descriptor.usage = MTLTextureUsageShaderRead;
    return [buffer newTextureWithDescriptor:descriptor
                                     offset:0
                                bytesPerRow:(NSUInteger)width * 4u * sizeof(uint16_t)];
}

static id<MTLTexture> make_texture_buffer(id<MTLBuffer> buffer, uint32_t texel_count) {
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                              width:texel_count
                                                                                    resourceOptions:MTLResourceStorageModeShared
                                                                                              usage:MTLTextureUsageShaderRead];
    return [buffer newTextureWithDescriptor:descriptor
                                     offset:0
                                bytesPerRow:(NSUInteger)texel_count * 4u * sizeof(uint16_t)];
}

static double run_kernel(id<MTLCommandQueue> queue,
                         id<MTLComputePipelineState> pipeline,
                         KernelKind kind,
                         id<MTLBuffer> codebook,
                         id<MTLTexture> texture,
                         id<MTLBuffer> mid,
                         id<MTLBuffer> out,
                         uint32_t k,
                         uint32_t groups,
                         uint32_t slots,
                         uint32_t in_dim,
                         uint32_t iterations) {
    double best = 1.0e30;
    const uint32_t texels_per_slot = k * 2u;
    for (uint32_t iteration = 0; iteration < iterations; iteration++) {
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        if (kind == KernelKindBufferManual || kind == KernelKindBufferDot) {
            [encoder setBuffer:codebook offset:0 atIndex:0];
            [encoder setBuffer:mid offset:0 atIndex:1];
            [encoder setBuffer:out offset:0 atIndex:2];
            [encoder setBytes:&k length:sizeof(k) atIndex:3];
            [encoder setBytes:&groups length:sizeof(groups) atIndex:4];
            [encoder setBytes:&in_dim length:sizeof(in_dim) atIndex:5];
            [encoder dispatchThreads:MTLSizeMake(k, groups, slots)
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        } else if (kind == KernelKindGatherRaw) {
            [encoder setTexture:texture atIndex:0];
            [encoder setBuffer:out offset:0 atIndex:0];
            [encoder setBytes:&texels_per_slot length:sizeof(texels_per_slot) atIndex:1];
            [encoder setBytes:&groups length:sizeof(groups) atIndex:2];
            [encoder dispatchThreads:MTLSizeMake((texels_per_slot + 1u) >> 1, groups, slots)
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        } else {
            [encoder setTexture:texture atIndex:0];
            [encoder setBuffer:mid offset:0 atIndex:0];
            [encoder setBuffer:out offset:0 atIndex:1];
            [encoder setBytes:&k length:sizeof(k) atIndex:2];
            [encoder setBytes:&groups length:sizeof(groups) atIndex:3];
            [encoder setBytes:&in_dim length:sizeof(in_dim) atIndex:4];
            [encoder dispatchThreads:MTLSizeMake(k, groups, slots)
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        }
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        const double elapsed_ms = (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0;
        if (elapsed_ms > 0.0 && elapsed_ms < best) best = elapsed_ms;
    }
    return best;
}

static void fill_inputs(uint16_t *codebook, float *mid, uint32_t k, uint32_t in_dim, uint32_t slots) {
    for (uint32_t slot = 0; slot < slots; slot++) {
        for (uint32_t code = 0; code < k; code++) {
            for (uint32_t lane = 0; lane < 8u; lane++) {
                const float phase = (float)(slot * 131u + code * 17u + lane * 29u);
                const float value = 0.06f * sinf(phase * 0.017f) + 0.04f * cosf(phase * 0.011f);
                codebook[(((ulong)slot * (ulong)k + (ulong)code) * 2ul + lane / 4u) * 4ul + lane % 4u] = f32_to_f16(value);
            }
        }
        for (uint32_t i = 0; i < in_dim; i++) {
            const float phase = (float)(slot * 257u + i);
            mid[(ulong)slot * (ulong)in_dim + i] = sinf(phase * 0.0019f) + 0.25f * cosf(phase * 0.013f);
        }
    }
}

static double max_abs_diff(const float *left, const float *right, NSUInteger count) {
    double max_abs = 0.0;
    for (NSUInteger i = 0; i < count; i++) {
        const double diff = fabs((double)left[i] - (double)right[i]);
        if (diff > max_abs) max_abs = diff;
    }
    return max_abs;
}

static int run_case(id<MTLDevice> device,
                    id<MTLCommandQueue> queue,
                    id<MTLComputePipelineState> buffer_manual,
                    id<MTLComputePipelineState> buffer_dot,
                    id<MTLComputePipelineState> texture_buffer,
                    id<MTLComputePipelineState> texture2d_read,
                    id<MTLComputePipelineState> texture2d_sample,
                    id<MTLComputePipelineState> gather_raw,
                    uint32_t k,
                    uint32_t in_dim,
                    uint32_t slots,
                    uint32_t iterations) {
    const uint32_t groups = in_dim >> 3;
    const NSUInteger codebook_bytes = (NSUInteger)slots * k * 2u * 4u * sizeof(uint16_t);
    const NSUInteger mid_bytes = (NSUInteger)slots * in_dim * sizeof(float);
    const NSUInteger out_bytes = (NSUInteger)slots * groups * k * sizeof(float);
    const NSUInteger gather_bytes = (NSUInteger)slots * groups * k * sizeof(float);

    uint16_t *codebook_host = (uint16_t *)malloc(codebook_bytes);
    float *mid_host = (float *)malloc(mid_bytes);
    if (!codebook_host || !mid_host) {
        free(codebook_host);
        free(mid_host);
        return 0;
    }
    fill_inputs(codebook_host, mid_host, k, in_dim, slots);

    id<MTLBuffer> codebook_shared = [device newBufferWithBytes:codebook_host length:codebook_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> mid = [device newBufferWithBytes:mid_host length:mid_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_ref = [device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_test = [device newBufferWithLength:MAX(out_bytes, gather_bytes) options:MTLResourceStorageModeShared];
    id<MTLBuffer> codebook_private = make_private_buffer(device, queue, codebook_shared, codebook_bytes);
    id<MTLTexture> tex2d_shared = make_texture2d(codebook_shared, k * 2u, slots);
    id<MTLTexture> texbuf_shared = make_texture_buffer(codebook_shared, slots * k * 2u);
    id<MTLTexture> tex2d_private = codebook_private ? make_texture2d(codebook_private, k * 2u, slots) : nil;
    id<MTLTexture> texbuf_private = codebook_private ? make_texture_buffer(codebook_private, slots * k * 2u) : nil;
    if (!codebook_shared || !mid || !out_ref || !out_test || !tex2d_shared || !texbuf_shared) {
        fprintf(stderr, "allocation failed k=%u in=%u slots=%u out=%.2fMiB\n", k, in_dim, slots, (double)out_bytes / 1048576.0);
        free(codebook_host);
        free(mid_host);
        return 0;
    }

    (void)run_kernel(queue, buffer_manual, KernelKindBufferManual, codebook_shared, nil, mid, out_ref, k, groups, slots, in_dim, 1);
    const float *ref = out_ref.contents;
    printf("texture-dsp-score K=%u in=%u groups=%u slots=%u out=%.2fMiB it=%u\n",
           k, in_dim, groups, slots, (double)out_bytes / 1048576.0, iterations);

    const double shared_manual = run_kernel(queue, buffer_manual, KernelKindBufferManual, codebook_shared, nil, mid, out_ref, k, groups, slots, in_dim, iterations);
    const double shared_dot = run_kernel(queue, buffer_dot, KernelKindBufferDot, codebook_shared, nil, mid, out_test, k, groups, slots, in_dim, iterations);
    const double err_dot = max_abs_diff(ref, out_test.contents, (NSUInteger)slots * groups * k);
    const double shared_texbuf = run_kernel(queue, texture_buffer, KernelKindTextureBuffer, nil, texbuf_shared, mid, out_test, k, groups, slots, in_dim, iterations);
    const double err_texbuf = max_abs_diff(ref, out_test.contents, (NSUInteger)slots * groups * k);
    const double shared_read = run_kernel(queue, texture2d_read, KernelKindTexture2DRead, nil, tex2d_shared, mid, out_test, k, groups, slots, in_dim, iterations);
    const double err_read = max_abs_diff(ref, out_test.contents, (NSUInteger)slots * groups * k);
    const double shared_sample = run_kernel(queue, texture2d_sample, KernelKindTexture2DSample, nil, tex2d_shared, mid, out_test, k, groups, slots, in_dim, iterations);
    const double err_sample = max_abs_diff(ref, out_test.contents, (NSUInteger)slots * groups * k);
    const double shared_gather = run_kernel(queue, gather_raw, KernelKindGatherRaw, nil, tex2d_shared, mid, out_test, k, groups, slots, in_dim, iterations);
    printf(" shared buffer_manual=%.4f buffer_dot=%.4f texbuf=%.4f tex2d_read=%.4f tex2d_sample=%.4f gather_raw=%.4f\n",
           shared_manual, shared_dot, shared_texbuf, shared_read, shared_sample, shared_gather);
    printf(" shared speed_vs_manual: dot=%.3fx texbuf=%.3fx tex2d_read=%.3fx tex2d_sample=%.3fx gather_raw=%.3fx err_dot=%.3g err_texbuf=%.3g err_read=%.3g err_sample=%.3g\n",
           shared_manual / shared_dot, shared_manual / shared_texbuf, shared_manual / shared_read,
           shared_manual / shared_sample, shared_manual / shared_gather,
           err_dot, err_texbuf, err_read, err_sample);

    if (codebook_private && tex2d_private && texbuf_private) {
        const double private_manual = run_kernel(queue, buffer_manual, KernelKindBufferManual, codebook_private, nil, mid, out_ref, k, groups, slots, in_dim, iterations);
        const double private_dot = run_kernel(queue, buffer_dot, KernelKindBufferDot, codebook_private, nil, mid, out_test, k, groups, slots, in_dim, iterations);
        const double private_texbuf = run_kernel(queue, texture_buffer, KernelKindTextureBuffer, nil, texbuf_private, mid, out_test, k, groups, slots, in_dim, iterations);
        const double private_read = run_kernel(queue, texture2d_read, KernelKindTexture2DRead, nil, tex2d_private, mid, out_test, k, groups, slots, in_dim, iterations);
        const double private_sample = run_kernel(queue, texture2d_sample, KernelKindTexture2DSample, nil, tex2d_private, mid, out_test, k, groups, slots, in_dim, iterations);
        const double private_gather = run_kernel(queue, gather_raw, KernelKindGatherRaw, nil, tex2d_private, mid, out_test, k, groups, slots, in_dim, iterations);
        printf("private buffer_manual=%.4f buffer_dot=%.4f texbuf=%.4f tex2d_read=%.4f tex2d_sample=%.4f gather_raw=%.4f\n",
               private_manual, private_dot, private_texbuf, private_read, private_sample, private_gather);
        printf("private speed_vs_manual: dot=%.3fx texbuf=%.3fx tex2d_read=%.3fx tex2d_sample=%.3fx gather_raw=%.3fx\n",
               private_manual / private_dot, private_manual / private_texbuf, private_manual / private_read,
               private_manual / private_sample, private_manual / private_gather);
    }

    free(codebook_host);
    free(mid_host);
    return 1;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        uint32_t iterations = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 10) : 40u;
        uint32_t slots = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 10) : 6u;
        if (iterations == 0u) iterations = 1u;
        if (slots == 0u) slots = 1u;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kMetalSource]
                                                       options:nil
                                                         error:&error];
        if (!library) {
            NSLog(@"library creation failed: %@", error);
            return 1;
        }
        id<MTLComputePipelineState> buffer_manual = make_pipeline(device, library, @"score_buffer_manual");
        id<MTLComputePipelineState> buffer_dot = make_pipeline(device, library, @"score_buffer_dot");
        id<MTLComputePipelineState> texture_buffer = make_pipeline(device, library, @"score_texture_buffer");
        id<MTLComputePipelineState> texture2d_read = make_pipeline(device, library, @"score_texture2d_read");
        id<MTLComputePipelineState> texture2d_sample = make_pipeline(device, library, @"score_texture2d_sample");
        id<MTLComputePipelineState> gather_raw = make_pipeline(device, library, @"raw_texture2d_gather_x");
        if (!buffer_manual || !buffer_dot || !texture_buffer || !texture2d_read || !texture2d_sample || !gather_raw) return 1;

        printf("d8f texture/DSP score-phase canary\n");
        if (!run_case(device, queue, buffer_manual, buffer_dot, texture_buffer, texture2d_read, texture2d_sample, gather_raw,
                      1024u, 4096u, slots, iterations)) return 1;
        if (!run_case(device, queue, buffer_manual, buffer_dot, texture_buffer, texture2d_read, texture2d_sample, gather_raw,
                      2048u, 7168u, slots, iterations)) return 1;
        if (!run_case(device, queue, buffer_manual, buffer_dot, texture_buffer, texture2d_read, texture2d_sample, gather_raw,
                      4096u, 7168u, slots, iterations)) return 1;
    }
    return 0;
}
