#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <VideoToolbox/VideoToolbox.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreMedia/CoreMedia.h>
#import <objc/runtime.h>
#import <simd/simd.h>
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int frames;
    size_t bytes;
    OSStatus last_status;
    VTEncodeInfoFlags last_flags;
} ds4_vt_encode_ctx;

static double ds4_now_ms(void) {
    static mach_timebase_info_data_t info;
    if (info.denom == 0) mach_timebase_info(&info);
    const uint64_t t = mach_absolute_time();
    return (double)t * (double)info.numer / (double)info.denom / 1.0e6;
}

static void ds4_fourcc(char out[5], CMVideoCodecType type) {
    out[0] = (char)((type >> 24) & 0xff);
    out[1] = (char)((type >> 16) & 0xff);
    out[2] = (char)((type >> 8) & 0xff);
    out[3] = (char)(type & 0xff);
    out[4] = '\0';
}

static void ds4_encode_callback(void *outputCallbackRefCon,
                                void *sourceFrameRefCon,
                                OSStatus status,
                                VTEncodeInfoFlags infoFlags,
                                CMSampleBufferRef sampleBuffer) {
    (void)sourceFrameRefCon;
    ds4_vt_encode_ctx *ctx = (ds4_vt_encode_ctx *)outputCallbackRefCon;
    ctx->last_status = status;
    ctx->last_flags = infoFlags;
    if (status == noErr && sampleBuffer && CMSampleBufferDataIsReady(sampleBuffer)) {
        CMBlockBufferRef data = CMSampleBufferGetDataBuffer(sampleBuffer);
        ctx->frames++;
        if (data) ctx->bytes += CMBlockBufferGetDataLength(data);
    }
}

static void ds4_print_cf_string(CFStringRef s) {
    if (!s) {
        printf("(null)");
        return;
    }
    char buf[256];
    if (CFStringGetCString(s, buf, sizeof(buf), kCFStringEncodingUTF8)) printf("%s", buf);
    else printf("(string)");
}

static int ds4_cfbool(CFTypeRef value) {
    if (!value || CFGetTypeID(value) != CFBooleanGetTypeID()) return -1;
    return CFBooleanGetValue((CFBooleanRef)value) ? 1 : 0;
}

static OSStatus ds4_make_pixel_buffer(int width, int height, OSType format, CVPixelBufferRef *pixel) {
    CFDictionaryRef io_surface = CFDictionaryCreate(kCFAllocatorDefault,
                                                    NULL,
                                                    NULL,
                                                    0,
                                                    &kCFTypeDictionaryKeyCallBacks,
                                                    &kCFTypeDictionaryValueCallBacks);
    const void *keys[] = {
        kCVPixelBufferMetalCompatibilityKey,
        kCVPixelBufferIOSurfacePropertiesKey,
    };
    const void *values[] = {
        kCFBooleanTrue,
        io_surface,
    };
    CFDictionaryRef attrs = CFDictionaryCreate(kCFAllocatorDefault,
                                               keys,
                                               values,
                                               2,
                                               &kCFTypeDictionaryKeyCallBacks,
                                               &kCFTypeDictionaryValueCallBacks);
    OSStatus status = CVPixelBufferCreate(kCFAllocatorDefault,
                                          width,
                                          height,
                                          format,
                                          attrs,
                                          pixel);
    if (attrs) CFRelease(attrs);
    if (io_surface) CFRelease(io_surface);
    return status;
}

static void ds4_fill_pixel_buffer(CVPixelBufferRef pixel) {
    if (!pixel) return;
    const int width = (int)CVPixelBufferGetWidth(pixel);
    const int height = (int)CVPixelBufferGetHeight(pixel);
    CVPixelBufferLockBaseAddress(pixel, 0);
    uint8_t *base = (uint8_t *)CVPixelBufferGetBaseAddress(pixel);
    const size_t row_bytes = CVPixelBufferGetBytesPerRow(pixel);
    for (int y = 0; y < height; y++) {
        uint8_t *row = base + (size_t)y * row_bytes;
        for (int x = 0; x < width; x++) {
            row[x * 4 + 0] = (uint8_t)(x * 13 + y);
            row[x * 4 + 1] = (uint8_t)(y * 17 + x);
            row[x * 4 + 2] = (uint8_t)(x ^ (y * 3));
            row[x * 4 + 3] = 255;
        }
    }
    CVPixelBufferUnlockBaseAddress(pixel, 0);
}

static void ds4_probe_vt_h264_encode_size(int width, int height, const char *label) {
    CFStringRef keys[] = { kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder };
    CFTypeRef values[] = { kCFBooleanTrue };
    CFDictionaryRef spec = CFDictionaryCreate(kCFAllocatorDefault,
                                              (const void **)keys,
                                              (const void **)values,
                                              1,
                                              &kCFTypeDictionaryKeyCallBacks,
                                              &kCFTypeDictionaryValueCallBacks);
    ds4_vt_encode_ctx ctx = {0};
    VTCompressionSessionRef session = NULL;
    OSStatus status = VTCompressionSessionCreate(kCFAllocatorDefault,
                                                 width,
                                                 height,
                                                 kCMVideoCodecType_H264,
                                                 spec,
                                                 NULL,
                                                 NULL,
                                                 ds4_encode_callback,
                                                 &ctx,
                                                 &session);
    if (spec) CFRelease(spec);
    printf("vt_h264_hw_session_create label=%s width=%d height=%d status=%d session=%d\n",
           label,
           width,
           height,
           (int)status,
           session ? 1 : 0);
    if (status != noErr || !session) return;
    VTSessionSetProperty(session, kVTCompressionPropertyKey_RealTime, kCFBooleanTrue);
    VTSessionSetProperty(session, kVTCompressionPropertyKey_AllowFrameReordering, kCFBooleanFalse);
    VTSessionSetProperty(session, kVTCompressionPropertyKey_MaxKeyFrameInterval, (__bridge CFTypeRef)@(1));
    status = VTCompressionSessionPrepareToEncodeFrames(session);
    printf("vt_h264_prepare label=%s status=%d\n", label, (int)status);

    CVPixelBufferRef pixel = NULL;
    status = ds4_make_pixel_buffer(width, height, kCVPixelFormatType_32BGRA, &pixel);
    printf("vt_pixel_create label=%s status=%d pixel=%d\n", label, (int)status, pixel ? 1 : 0);
    if (pixel) {
        ds4_fill_pixel_buffer(pixel);
        double first_ms = 0.0;
        double best_ms = 1.0e30;
        double sum_ms = 0.0;
        VTEncodeInfoFlags flags = 0;
        OSStatus complete = noErr;
        for (int frame = 0; frame < 8; frame++) {
            const double t0 = ds4_now_ms();
            flags = 0;
            status = VTCompressionSessionEncodeFrame(session,
                                                     pixel,
                                                     CMTimeMake(frame, 30),
                                                     CMTimeMake(1, 30),
                                                     NULL,
                                                     NULL,
                                                     &flags);
            complete = VTCompressionSessionCompleteFrames(session, CMTimeMake(frame, 30));
            const double t1 = ds4_now_ms();
            const double elapsed = t1 - t0;
            if (frame == 0) first_ms = elapsed;
            else {
                if (elapsed < best_ms) best_ms = elapsed;
                sum_ms += elapsed;
            }
        }
        complete = VTCompressionSessionCompleteFrames(session, kCMTimeInvalid);
        CFTypeRef using_hw = NULL;
        OSStatus prop_status = VTSessionCopyProperty(session,
                                                     kVTCompressionPropertyKey_UsingHardwareAcceleratedVideoEncoder,
                                                     kCFAllocatorDefault,
                                                     &using_hw);
        const double input_bytes = (double)width * (double)height * 4.0;
        const double warm_best_gbs = input_bytes / best_ms / 1.0e6;
        printf("vt_h264_encode label=%s status=%d complete=%d flags=%u callback_frames=%d callback_bytes=%zu callback_status=%d callback_flags=%u input_bytes=%.0f first_ms=%.3f warm_best_ms=%.3f warm_mean_ms=%.3f warm_best_input_gbs=%.3f using_hw_status=%d using_hw=%d\n",
               label,
               (int)status,
               (int)complete,
               (unsigned)flags,
               ctx.frames,
               ctx.bytes,
               (int)ctx.last_status,
               (unsigned)ctx.last_flags,
               input_bytes,
               first_ms,
               best_ms,
               sum_ms / 7.0,
               warm_best_gbs,
               (int)prop_status,
               ds4_cfbool(using_hw));
        if (using_hw) CFRelease(using_hw);
        CVPixelBufferRelease(pixel);
    }
    VTCompressionSessionInvalidate(session);
    CFRelease(session);
}

static void ds4_probe_vt_pixel_transfer_size(int width, int height, const char *label) {
    CVPixelBufferRef src = NULL;
    CVPixelBufferRef dst = NULL;
    OSStatus src_status = ds4_make_pixel_buffer(width, height, kCVPixelFormatType_32BGRA, &src);
    OSStatus dst_status = ds4_make_pixel_buffer(width, height, kCVPixelFormatType_32BGRA, &dst);
    printf("vt_pixel_transfer_create label=%s width=%d height=%d src_status=%d dst_status=%d src=%d dst=%d\n",
           label,
           width,
           height,
           (int)src_status,
           (int)dst_status,
           src ? 1 : 0,
           dst ? 1 : 0);
    if (!src || !dst) {
        if (src) CVPixelBufferRelease(src);
        if (dst) CVPixelBufferRelease(dst);
        return;
    }
    ds4_fill_pixel_buffer(src);
    VTPixelTransferSessionRef session = NULL;
    OSStatus status = VTPixelTransferSessionCreate(kCFAllocatorDefault, &session);
    printf("vt_pixel_transfer_session label=%s status=%d session=%d\n", label, (int)status, session ? 1 : 0);
    if (status != noErr || !session) {
        CVPixelBufferRelease(src);
        CVPixelBufferRelease(dst);
        return;
    }
    VTSessionSetProperty(session, kVTPixelTransferPropertyKey_RealTime, kCFBooleanTrue);
    double first_ms = 0.0;
    double best_ms = 1.0e30;
    double sum_ms = 0.0;
    for (int round = 0; round < 8; round++) {
        const double t0 = ds4_now_ms();
        status = VTPixelTransferSessionTransferImage(session, src, dst);
        const double t1 = ds4_now_ms();
        const double elapsed = t1 - t0;
        if (round == 0) first_ms = elapsed;
        else {
            if (elapsed < best_ms) best_ms = elapsed;
            sum_ms += elapsed;
        }
    }
    const double payload_bytes = (double)width * (double)height * 4.0;
    const double traffic_bytes = payload_bytes * 2.0;
    printf("vt_pixel_transfer label=%s status=%d payload_bytes=%.0f traffic_bytes=%.0f first_ms=%.3f warm_best_ms=%.3f warm_mean_ms=%.3f warm_best_payload_gbs=%.3f warm_best_traffic_gbs=%.3f\n",
           label,
           (int)status,
           payload_bytes,
           traffic_bytes,
           first_ms,
           best_ms,
           sum_ms / 7.0,
           payload_bytes / best_ms / 1.0e6,
           traffic_bytes / best_ms / 1.0e6);
    CFRelease(session);
    CVPixelBufferRelease(src);
    CVPixelBufferRelease(dst);
}

static void ds4_probe_videotoolbox(void) {
    const char *vt_classes[] = {
        "VTFrameProcessor",
        "VTOpticalFlowConfiguration",
        "VTLowLatencySuperResolutionScalerConfiguration",
        "VTLowLatencyFrameInterpolationConfiguration",
    };
    for (size_t i = 0; i < sizeof(vt_classes) / sizeof(vt_classes[0]); i++) {
        Class cls = NSClassFromString([NSString stringWithUTF8String:vt_classes[i]]);
        printf("vt_class name=%s present=%d\n", vt_classes[i], cls ? 1 : 0);
    }
    struct Codec { const char *name; CMVideoCodecType type; } codecs[] = {
        {"h264", kCMVideoCodecType_H264},
        {"hevc", kCMVideoCodecType_HEVC},
        {"hevc_alpha", kCMVideoCodecType_HEVCWithAlpha},
        {"vp9", kCMVideoCodecType_VP9},
        {"av1", kCMVideoCodecType_AV1},
        {"prores422", kCMVideoCodecType_AppleProRes422},
        {"prores4444", kCMVideoCodecType_AppleProRes4444},
    };
    printf("videotoolbox_probe codecs=%lu\n", (unsigned long)(sizeof(codecs) / sizeof(codecs[0])));
    for (size_t i = 0; i < sizeof(codecs) / sizeof(codecs[0]); i++) {
        char fcc[5];
        ds4_fourcc(fcc, codecs[i].type);
        Boolean decode_hw = VTIsHardwareDecodeSupported(codecs[i].type);
        printf("vt_decode name=%s fourcc=%s hardware=%d\n", codecs[i].name, fcc, decode_hw ? 1 : 0);
    }

    ds4_probe_vt_h264_encode_size(128, 128, "tiny");
    ds4_probe_vt_h264_encode_size(1024, 1024, "megapixel");
    ds4_probe_vt_pixel_transfer_size(1024, 1024, "transfer_4mb");
    ds4_probe_vt_pixel_transfer_size(4096, 2048, "transfer_32mb");
}

static BOOL ds4_supports_family(id<MTLDevice> device, MTLGPUFamily family) {
    if (![device respondsToSelector:@selector(supportsFamily:)]) return NO;
    return [device supportsFamily:family];
}

static MTLAccelerationStructureUsage ds4_fast_intersection_usage(void) {
    if (@available(macOS 26.0, *)) return MTLAccelerationStructureUsagePreferFastIntersection;
    return MTLAccelerationStructureUsagePreferFastBuild;
}

static MTLPrimitiveAccelerationStructureDescriptor *ds4_make_triangle_as_desc(id<MTLDevice> device,
                                                                              NSUInteger triangle_count,
                                                                              MTLAccelerationStructureUsage usage,
                                                                              id<MTLBuffer> __strong *vertex_buffer_out) {
    const NSUInteger vertex_count = triangle_count * 3;
    simd_float3 *vertices = (simd_float3 *)malloc(sizeof(simd_float3) * vertex_count);
    if (!vertices) return nil;
    for (NSUInteger i = 0; i < triangle_count; i++) {
        const float x = (float)(i & 255u) * 2.0f;
        const float y = (float)((i >> 8) & 255u) * 2.0f;
        vertices[i * 3 + 0] = (simd_float3){x + 0.0f, y + 0.0f, 0.0f};
        vertices[i * 3 + 1] = (simd_float3){x + 1.0f, y + 0.0f, 0.0f};
        vertices[i * 3 + 2] = (simd_float3){x + 0.0f, y + 1.0f, 0.0f};
    }
    id<MTLBuffer> vb = [device newBufferWithBytes:vertices
                                           length:sizeof(simd_float3) * vertex_count
                                          options:MTLResourceStorageModeShared];
    free(vertices);
    if (!vb) return nil;
    MTLAccelerationStructureTriangleGeometryDescriptor *geom = [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
    geom.vertexBuffer = vb;
    geom.vertexFormat = MTLAttributeFormatFloat3;
    geom.vertexStride = sizeof(simd_float3);
    geom.triangleCount = triangle_count;
    MTLPrimitiveAccelerationStructureDescriptor *desc = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    desc.geometryDescriptors = @[geom];
    desc.usage = usage;
    *vertex_buffer_out = vb;
    return desc;
}

static id<MTLAccelerationStructure> ds4_probe_ray_build_size(id<MTLDevice> device,
                                                             id<MTLCommandQueue> queue,
                                                             NSUInteger triangle_count,
                                                             MTLAccelerationStructureUsage usage,
                                                             const char *label) {
    id<MTLBuffer> vb = nil;
    MTLPrimitiveAccelerationStructureDescriptor *desc = ds4_make_triangle_as_desc(device, triangle_count, usage, &vb);
    if (!desc || !vb) {
        printf("metal_ray_build label=%s triangles=%llu descriptor=0\n", label, (unsigned long long)triangle_count);
        return nil;
    }
    MTLAccelerationStructureSizes sizes = [device accelerationStructureSizesWithDescriptor:desc];
    double first_ms = 0.0;
    double best_ms = 1.0e30;
    double sum_ms = 0.0;
    NSUInteger final_status = 0;
    int final_encoder = 0;
    int final_accel = 0;
    int final_scratch = 0;
    id<MTLAccelerationStructure> final_as = nil;
    for (int round = 0; round < 8; round++) {
        id<MTLAccelerationStructure> as = [device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
        id<MTLBuffer> scratch = [device newBufferWithLength:sizes.buildScratchBufferSize
                                                    options:MTLResourceStorageModePrivate];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> enc = [cb accelerationStructureCommandEncoder];
        const double t0 = ds4_now_ms();
        if (enc && as && scratch) {
            [enc buildAccelerationStructure:as descriptor:desc scratchBuffer:scratch scratchBufferOffset:0];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        const double t1 = ds4_now_ms();
        const double elapsed = t1 - t0;
        if (round == 0) first_ms = elapsed;
        else {
            if (elapsed < best_ms) best_ms = elapsed;
            sum_ms += elapsed;
        }
        final_status = cb.status;
        final_encoder = enc ? 1 : 0;
        final_accel = as ? 1 : 0;
        final_scratch = scratch ? 1 : 0;
        final_as = as;
    }
    const double vertex_bytes = (double)triangle_count * 3.0 * (double)sizeof(simd_float3);
    printf("metal_ray_build label=%s usage=0x%lx encoder=%d accel=%d scratch=%d triangles=%llu vertex_bytes=%.0f accel_bytes=%llu scratch_bytes=%llu status=%lu first_ms=%.3f warm_best_ms=%.3f warm_mean_ms=%.3f warm_best_vertex_gbs=%.3f\n",
           label,
           (unsigned long)usage,
           final_encoder,
           final_accel,
           final_scratch,
           (unsigned long long)triangle_count,
           vertex_bytes,
           (unsigned long long)sizes.accelerationStructureSize,
           (unsigned long long)sizes.buildScratchBufferSize,
           (unsigned long)final_status,
           first_ms,
           best_ms,
           sum_ms / 7.0,
           vertex_bytes / best_ms / 1.0e6);
    return final_as;
}

static void ds4_probe_ray_traversal(id<MTLDevice> device,
                                    id<MTLCommandQueue> queue,
                                    id<MTLAccelerationStructure> as) {
    if (!as) {
        printf("metal_ray_traverse skipped=no_acceleration_structure\n");
        return;
    }
    NSString *source =
        @"#include <metal_stdlib>\n"
         "#include <metal_raytracing>\n"
         "using namespace metal;\n"
         "using namespace raytracing;\n"
         "kernel void ray_probe(primitive_acceleration_structure accel [[buffer(0)]],\n"
         "                      device uint *hits [[buffer(1)]],\n"
         "                      uint tid [[thread_position_in_grid]]) {\n"
         "    ray r;\n"
         "    float fx = float(tid & 1023u) * 0.00048828125f + 0.125f;\n"
         "    float fy = float((tid >> 10) & 1023u) * 0.00048828125f + 0.125f;\n"
         "    r.origin = float3(fx, fy, -1.0f);\n"
         "    r.direction = float3(0.0f, 0.0f, 1.0f);\n"
         "    r.min_distance = 0.0f;\n"
         "    r.max_distance = 10.0f;\n"
         "    intersector<triangle_data> isect;\n"
         "    auto result = isect.intersect(r, accel);\n"
         "    hits[tid] = result.type == intersection_type::triangle ? result.primitive_id : 0xffffffffu;\n"
         "}\n"
         "kernel void store_probe(device uint *hits [[buffer(0)]],\n"
         "                        uint tid [[thread_position_in_grid]]) {\n"
         "    hits[tid] = tid * 1664525u + 1013904223u;\n"
         "}\n";
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (!library) {
        printf("metal_ray_shader library=0 error=\"%s\"\n", error.localizedDescription.UTF8String);
        return;
    }
    id<MTLFunction> ray_fn = [library newFunctionWithName:@"ray_probe"];
    id<MTLFunction> store_fn = [library newFunctionWithName:@"store_probe"];
    id<MTLComputePipelineState> ray_pso = [device newComputePipelineStateWithFunction:ray_fn error:&error];
    if (!ray_pso) {
        printf("metal_ray_shader ray_pipeline=0 error=\"%s\"\n", error.localizedDescription.UTF8String);
        return;
    }
    id<MTLComputePipelineState> store_pso = [device newComputePipelineStateWithFunction:store_fn error:&error];
    if (!store_pso) {
        printf("metal_ray_shader store_pipeline=0 error=\"%s\"\n", error.localizedDescription.UTF8String);
        return;
    }
    const NSUInteger ray_count = 1u << 20;
    id<MTLBuffer> hits = [device newBufferWithLength:ray_count * sizeof(uint32_t)
                                             options:MTLResourceStorageModeShared];
    if (!hits) {
        printf("metal_ray_traverse skipped=no_hits_buffer\n");
        return;
    }
    const NSUInteger ray_threads = MIN((NSUInteger)256, ray_pso.maxTotalThreadsPerThreadgroup);
    const NSUInteger store_threads = MIN((NSUInteger)256, store_pso.maxTotalThreadsPerThreadgroup);
    double store_first_ms = 0.0;
    double store_best_ms = 1.0e30;
    double store_sum_ms = 0.0;
    NSUInteger store_status = 0;
    for (int round = 0; round < 8; round++) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        const double t0 = ds4_now_ms();
        [enc setComputePipelineState:store_pso];
        [enc setBuffer:hits offset:0 atIndex:0];
        [enc dispatchThreads:MTLSizeMake(ray_count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(store_threads, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const double t1 = ds4_now_ms();
        const double elapsed = t1 - t0;
        if (round == 0) store_first_ms = elapsed;
        else {
            if (elapsed < store_best_ms) store_best_ms = elapsed;
            store_sum_ms += elapsed;
        }
        store_status = cb.status;
    }
    double ray_first_ms = 0.0;
    double ray_best_ms = 1.0e30;
    double ray_sum_ms = 0.0;
    NSUInteger ray_status = 0;
    for (int round = 0; round < 8; round++) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        const double t0 = ds4_now_ms();
        [enc setComputePipelineState:ray_pso];
        [enc setAccelerationStructure:as atBufferIndex:0];
        [enc setBuffer:hits offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(ray_count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(ray_threads, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const double t1 = ds4_now_ms();
        const double elapsed = t1 - t0;
        if (round == 0) ray_first_ms = elapsed;
        else {
            if (elapsed < ray_best_ms) ray_best_ms = elapsed;
            ray_sum_ms += elapsed;
        }
        ray_status = cb.status;
    }
    const uint32_t *hit_words = (const uint32_t *)hits.contents;
    NSUInteger hit_count = 0;
    for (NSUInteger i = 0; i < ray_count; i++) hit_count += hit_words[i] != 0xffffffffu ? 1u : 0u;
    printf("metal_ray_traverse rays=%llu hit_count=%llu store_status=%lu ray_status=%lu store_first_ms=%.3f store_warm_best_ms=%.3f store_warm_mean_ms=%.3f ray_first_ms=%.3f ray_warm_best_ms=%.3f ray_warm_mean_ms=%.3f ray_warm_best_mrays_s=%.3f net_over_store_ms=%.3f\n",
           (unsigned long long)ray_count,
           (unsigned long long)hit_count,
           (unsigned long)store_status,
           (unsigned long)ray_status,
           store_first_ms,
           store_best_ms,
           store_sum_ms / 7.0,
           ray_first_ms,
           ray_best_ms,
           ray_sum_ms / 7.0,
           (double)ray_count / ray_best_ms / 1000.0,
           ray_best_ms - store_best_ms);
}

static void ds4_probe_raytracing(void) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        printf("metal_probe device=none\n");
        return;
    }
    printf("metal_probe device=\"%s\" registry_id=%llu unified_memory=%d\n",
           device.name.UTF8String,
           (unsigned long long)device.registryID,
           device.hasUnifiedMemory ? 1 : 0);
    printf("metal_families apple7=%d apple8=%d apple9=%d apple10=%d mac2=%d\n",
           ds4_supports_family(device, MTLGPUFamilyApple7) ? 1 : 0,
           ds4_supports_family(device, MTLGPUFamilyApple8) ? 1 : 0,
           ds4_supports_family(device, MTLGPUFamilyApple9) ? 1 : 0,
           ds4_supports_family(device, MTLGPUFamilyApple10) ? 1 : 0,
           ds4_supports_family(device, MTLGPUFamilyMac2) ? 1 : 0);
    const BOOL has_supports_ray = [device respondsToSelector:@selector(supportsRaytracing)];
    const BOOL ray = has_supports_ray ? device.supportsRaytracing : NO;
    const BOOL ray_render = [device respondsToSelector:@selector(supportsRaytracingFromRender)] ? device.supportsRaytracingFromRender : NO;
    const BOOL motion = [device respondsToSelector:@selector(supportsPrimitiveMotionBlur)] ? device.supportsPrimitiveMotionBlur : NO;
    printf("metal_ray_support selector=%d supports_raytracing=%d render=%d primitive_motion_blur=%d apple9_or_newer=%d\n",
           has_supports_ray ? 1 : 0,
           ray ? 1 : 0,
           ray_render ? 1 : 0,
           motion ? 1 : 0,
           (ds4_supports_family(device, MTLGPUFamilyApple9) || ds4_supports_family(device, MTLGPUFamilyApple10)) ? 1 : 0);
    if (!ray) {
        printf("metal_ray_build skipped=unsupported\n");
        return;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLAccelerationStructure> tiny_as = ds4_probe_ray_build_size(device, queue, 1, MTLAccelerationStructureUsagePreferFastBuild, "tiny_fast_build");
    ds4_probe_ray_build_size(device, queue, 4096, MTLAccelerationStructureUsagePreferFastBuild, "4k_fast_build");
    ds4_probe_ray_build_size(device, queue, 65536, MTLAccelerationStructureUsagePreferFastBuild, "64k_fast_build");
    id<MTLAccelerationStructure> trace_as = ds4_probe_ray_build_size(device, queue, 1, ds4_fast_intersection_usage(), "tiny_fast_intersection");
    ds4_probe_ray_traversal(device, queue, trace_as ? trace_as : tiny_as);
}

int main(void) {
    @autoreleasepool {
        ds4_probe_videotoolbox();
        ds4_probe_raytracing();
    }
    return 0;
}
