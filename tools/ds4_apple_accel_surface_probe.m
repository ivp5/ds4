#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <VideoToolbox/VideoToolbox.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreMedia/CoreMedia.h>
#import <objc/runtime.h>
#import <simd/simd.h>
#include <mach/mach_time.h>
#include <stdio.h>
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

    const int width = 128;
    const int height = 128;
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
    printf("vt_h264_hw_session_create status=%d session=%d\n", (int)status, session ? 1 : 0);
    if (status != noErr || !session) return;
    VTSessionSetProperty(session, kVTCompressionPropertyKey_RealTime, kCFBooleanTrue);
    VTSessionSetProperty(session, kVTCompressionPropertyKey_AllowFrameReordering, kCFBooleanFalse);
    VTSessionSetProperty(session, kVTCompressionPropertyKey_MaxKeyFrameInterval, (__bridge CFTypeRef)@(1));
    status = VTCompressionSessionPrepareToEncodeFrames(session);
    printf("vt_h264_prepare status=%d\n", (int)status);

    CVPixelBufferRef pixel = NULL;
    status = CVPixelBufferCreate(kCFAllocatorDefault,
                                 width,
                                 height,
                                 kCVPixelFormatType_32BGRA,
                                 NULL,
                                 &pixel);
    printf("vt_pixel_create status=%d pixel=%d\n", (int)status, pixel ? 1 : 0);
    if (pixel) {
        CVPixelBufferLockBaseAddress(pixel, 0);
        uint8_t *base = (uint8_t *)CVPixelBufferGetBaseAddress(pixel);
        const size_t row_bytes = CVPixelBufferGetBytesPerRow(pixel);
        for (int y = 0; y < height; y++) {
            uint8_t *row = base + (size_t)y * row_bytes;
            for (int x = 0; x < width; x++) {
                row[x * 4 + 0] = (uint8_t)(x * 2);
                row[x * 4 + 1] = (uint8_t)(y * 2);
                row[x * 4 + 2] = (uint8_t)(x ^ y);
                row[x * 4 + 3] = 255;
            }
        }
        CVPixelBufferUnlockBaseAddress(pixel, 0);
        double first_ms = 0.0;
        double best_ms = 1.0e30;
        double sum_ms = 0.0;
        VTEncodeInfoFlags flags = 0;
        status = noErr;
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
        printf("vt_h264_encode status=%d complete=%d flags=%u callback_frames=%d callback_bytes=%zu callback_status=%d callback_flags=%u first_ms=%.3f warm_best_ms=%.3f warm_mean_ms=%.3f using_hw_status=%d using_hw=%d\n",
               (int)status,
               (int)complete,
               (unsigned)flags,
               ctx.frames,
               ctx.bytes,
               (int)ctx.last_status,
               (unsigned)ctx.last_flags,
               first_ms,
               best_ms,
               sum_ms / 7.0,
               (int)prop_status,
               ds4_cfbool(using_hw));
        if (using_hw) CFRelease(using_hw);
        CVPixelBufferRelease(pixel);
    }
    VTCompressionSessionInvalidate(session);
    CFRelease(session);
}

static BOOL ds4_supports_family(id<MTLDevice> device, MTLGPUFamily family) {
    if (![device respondsToSelector:@selector(supportsFamily:)]) return NO;
    return [device supportsFamily:family];
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

    simd_float3 vertices[3] = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
    };
    id<MTLBuffer> vb = [device newBufferWithBytes:vertices
                                           length:sizeof(vertices)
                                          options:MTLResourceStorageModeShared];
    MTLAccelerationStructureTriangleGeometryDescriptor *geom = [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
    geom.vertexBuffer = vb;
    geom.vertexFormat = MTLAttributeFormatFloat3;
    geom.vertexStride = sizeof(simd_float3);
    geom.triangleCount = 1;
    MTLPrimitiveAccelerationStructureDescriptor *desc = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    desc.geometryDescriptors = @[geom];
    desc.usage = MTLAccelerationStructureUsagePreferFastBuild;
    MTLAccelerationStructureSizes sizes = [device accelerationStructureSizesWithDescriptor:desc];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    double first_ms = 0.0;
    double best_ms = 1.0e30;
    double sum_ms = 0.0;
    NSUInteger final_status = 0;
    int final_encoder = 0;
    int final_accel = 0;
    int final_scratch = 0;
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
    }
    printf("metal_ray_build encoder=%d accel=%d scratch=%d accel_bytes=%llu scratch_bytes=%llu status=%lu first_ms=%.3f warm_best_ms=%.3f warm_mean_ms=%.3f\n",
           final_encoder,
           final_accel,
           final_scratch,
           (unsigned long long)sizes.accelerationStructureSize,
           (unsigned long long)sizes.buildScratchBufferSize,
           (unsigned long)final_status,
           first_ms,
           best_ms,
           sum_ms / 7.0);
}

int main(void) {
    @autoreleasepool {
        ds4_probe_videotoolbox();
        ds4_probe_raytracing();
    }
    return 0;
}
