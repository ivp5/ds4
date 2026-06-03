#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreVideo/CVPixelBufferIOSurface.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

static double now_seconds(void) {
    return [NSDate timeIntervalSinceReferenceDate];
}

static void die(NSString *message, NSError *error) {
    if (error) {
        fprintf(stderr, "%s: %s\n", message.UTF8String, error.localizedDescription.UTF8String);
    } else {
        fprintf(stderr, "%s\n", message.UTF8String);
    }
    exit(1);
}

static NSArray<NSNumber *> *first_shape(NSDictionary<NSString *, MLFeatureDescription *> *descriptions,
                                        NSString **name_out) {
    NSString *name = descriptions.allKeys.firstObject;
    if (!name) {
        die(@"missing feature description", nil);
    }
    MLFeatureDescription *description = descriptions[name];
    NSArray<NSNumber *> *shape = description.multiArrayConstraint.shape;
    if (shape.count != 2) {
        die([NSString stringWithFormat:@"expected rank-2 MLMultiArray for %@", name], nil);
    }
    *name_out = name;
    return shape;
}

static MLMultiArray *make_iosurface_multiarray(NSArray<NSNumber *> *shape,
                                               CVPixelBufferRef *pixel_buffer_out,
                                               id<MTLDevice> device,
                                               id<MTLTexture> *texture_out) {
    size_t height = (size_t)shape[0].unsignedLongLongValue;
    size_t width = (size_t)shape[1].unsignedLongLongValue;
    NSDictionary *attrs = @{
        (NSString *)kCVPixelBufferIOSurfacePropertiesKey : @{},
        (NSString *)kCVPixelBufferMetalCompatibilityKey : @YES,
    };
    CVPixelBufferRef pixel_buffer = NULL;
    CVReturn code = CVPixelBufferCreate(kCFAllocatorDefault,
                                        width,
                                        height,
                                        kCVPixelFormatType_OneComponent16Half,
                                        (__bridge CFDictionaryRef)attrs,
                                        &pixel_buffer);
    if (code != kCVReturnSuccess || !pixel_buffer) {
        die([NSString stringWithFormat:@"CVPixelBufferCreate failed: %d", code], nil);
    }
    MLMultiArray *array = [[MLMultiArray alloc] initWithPixelBuffer:pixel_buffer shape:shape];
    if (!array) {
        die(@"MLMultiArray initWithPixelBuffer failed", nil);
    }
    IOSurfaceRef surface = CVPixelBufferGetIOSurface(pixel_buffer);
    if (!surface) {
        die(@"CVPixelBuffer has no IOSurface", nil);
    }
    MTLTextureDescriptor *descriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor iosurface:surface plane:0];
    if (!texture) {
        die(@"newTextureWithDescriptor:iosurface failed", nil);
    }
    *pixel_buffer_out = pixel_buffer;
    *texture_out = texture;
    return array;
}

static id<MTLLibrary> make_library(id<MTLDevice> device) {
    NSString *source =
        @"#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "kernel void fill(texture2d<half, access::write> out [[texture(0)]],\n"
         "                 uint2 gid [[thread_position_in_grid]]) {\n"
         "  if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;\n"
         "  out.write(half(1.0), gid);\n"
         "}\n"
         "kernel void copy_buf_to_tex(const device half *in [[buffer(0)]],\n"
         "                            texture2d<half, access::write> out [[texture(0)]],\n"
         "                            uint2 gid [[thread_position_in_grid]]) {\n"
         "  if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;\n"
         "  out.write(in[gid.y * out.get_width() + gid.x], gid);\n"
         "}\n";
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (!library) {
        die(@"newLibraryWithSource failed", error);
    }
    return library;
}

static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device,
                                                 id<MTLLibrary> library,
                                                 NSString *name) {
    NSError *error = nil;
    id<MTLFunction> fn = [library newFunctionWithName:name];
    if (!fn) {
        die([NSString stringWithFormat:@"newFunctionWithName %@ failed", name], nil);
    }
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        die(@"newComputePipelineState failed", error);
    }
    return pipeline;
}

static void fill_texture(id<MTLCommandQueue> queue,
                         id<MTLComputePipelineState> pipeline,
                         id<MTLTexture> texture) {
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setTexture:texture atIndex:0];
    MTLSize threads_per_group = MTLSizeMake(16, 16, 1);
    MTLSize groups = MTLSizeMake((texture.width + 15) / 16, (texture.height + 15) / 16, 1);
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads_per_group];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
}

static void copy_buffer_to_texture(id<MTLCommandQueue> queue,
                                   id<MTLComputePipelineState> pipeline,
                                   id<MTLBuffer> source_buffer,
                                   id<MTLTexture> texture) {
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:source_buffer offset:0 atIndex:0];
    [encoder setTexture:texture atIndex:0];
    MTLSize threads_per_group = MTLSizeMake(16, 16, 1);
    MTLSize groups = MTLSizeMake((texture.width + 15) / 16, (texture.height + 15) / 16, 1);
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads_per_group];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
}

static id<MLFeatureProvider> run_predict(MLModel *model,
                                         NSString *input_name,
                                         MLMultiArray *input_array,
                                         NSString *output_name,
                                         MLMultiArray *output_array) {
    NSError *error = nil;
    MLFeatureValue *input_value = [MLFeatureValue featureValueWithMultiArray:input_array];
    MLDictionaryFeatureProvider *provider =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{input_name : input_value}
                                                          error:&error];
    if (!provider) {
        die(@"input provider creation failed", error);
    }
    MLPredictionOptions *options = [[MLPredictionOptions alloc] init];
    options.outputBackings = @{output_name : output_array};
    id<MLFeatureProvider> prediction = [model predictionFromFeatures:provider options:options error:&error];
    if (!prediction) {
        die(@"prediction failed", error);
    }
    return prediction;
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "usage: %s MODEL.mlpackage [rounds]\n", argv[0]);
            return 2;
        }
        NSString *path = [NSString stringWithUTF8String:argv[1]];
        int rounds = argc >= 3 ? atoi(argv[2]) : 10;
        NSURL *model_url = [NSURL fileURLWithPath:path];
        NSError *error = nil;
        NSLog(@"[compile] start path=%@", path);
        NSURL *compiled_url = [MLModel compileModelAtURL:model_url error:&error];
        if (!compiled_url) {
            die(@"compileModelAtURL failed", error);
        }
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            die(@"MTLCreateSystemDefaultDevice failed", nil);
        }
        config.preferredMetalDevice = device;
        MLModel *model = [MLModel modelWithContentsOfURL:compiled_url configuration:config error:&error];
        if (!model) {
            die(@"modelWithContentsOfURL failed", error);
        }

        NSString *input_name = nil;
        NSString *output_name = nil;
        NSArray<NSNumber *> *input_shape =
            first_shape(model.modelDescription.inputDescriptionsByName, &input_name);
        NSArray<NSNumber *> *output_shape =
            first_shape(model.modelDescription.outputDescriptionsByName, &output_name);
        NSLog(@"[shape] input=%@ %@ output=%@ %@",
              input_name,
              input_shape,
              output_name,
              output_shape);

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLLibrary> library = make_library(device);
        id<MTLComputePipelineState> fill_pipeline = make_pipeline(device, library, @"fill");
        id<MTLComputePipelineState> copy_pipeline = make_pipeline(device, library, @"copy_buf_to_tex");
        CVPixelBufferRef input_pixel_buffer = NULL;
        CVPixelBufferRef output_pixel_buffer = NULL;
        id<MTLTexture> input_texture = nil;
        id<MTLTexture> output_texture = nil;
        MLMultiArray *input_array =
            make_iosurface_multiarray(input_shape, &input_pixel_buffer, device, &input_texture);
        MLMultiArray *output_array =
            make_iosurface_multiarray(output_shape, &output_pixel_buffer, device, &output_texture);
        NSUInteger input_words = input_texture.width * input_texture.height;
        id<MTLBuffer> source_buffer = [device newBufferWithLength:input_words * sizeof(uint16_t)
                                                          options:MTLResourceStorageModeShared];
        if (!source_buffer) {
            die(@"source buffer allocation failed", nil);
        }
        uint16_t *source_words = (uint16_t *)source_buffer.contents;
        for (NSUInteger i = 0; i < input_words; ++i) {
            source_words[i] = 0x3c00;
        }

        for (int i = 0; i < 3; ++i) {
            copy_buffer_to_texture(queue, copy_pipeline, source_buffer, input_texture);
            id<MLFeatureProvider> prediction =
                run_predict(model, input_name, input_array, output_name, output_array);
            MLMultiArray *actual_output = [prediction featureValueForName:output_name].multiArrayValue;
            NSLog(@"[warmup] %d output_backing_used=%@", i + 1, actual_output == output_array ? @"yes" : @"no");
        }

        double predict_t0 = now_seconds();
        for (int i = 0; i < rounds; ++i) {
            run_predict(model, input_name, input_array, output_name, output_array);
        }
        double predict_elapsed = now_seconds() - predict_t0;

        double fill_predict_t0 = now_seconds();
        for (int i = 0; i < rounds; ++i) {
            fill_texture(queue, fill_pipeline, input_texture);
            run_predict(model, input_name, input_array, output_name, output_array);
        }
        double fill_predict_elapsed = now_seconds() - fill_predict_t0;

        double copy_predict_t0 = now_seconds();
        for (int i = 0; i < rounds; ++i) {
            copy_buffer_to_texture(queue, copy_pipeline, source_buffer, input_texture);
            run_predict(model, input_name, input_array, output_name, output_array);
        }
        double copy_predict_elapsed = now_seconds() - copy_predict_t0;

        NSLog(@"[result] rounds=%d predict_ms=%.3f fill_predict_ms=%.3f fill_delta_ms=%.3f copy_predict_ms=%.3f copy_delta_ms=%.3f",
              rounds,
              predict_elapsed * 1e3 / rounds,
              fill_predict_elapsed * 1e3 / rounds,
              (fill_predict_elapsed - predict_elapsed) * 1e3 / rounds,
              copy_predict_elapsed * 1e3 / rounds,
              (copy_predict_elapsed - predict_elapsed) * 1e3 / rounds);

        [output_array getBytesWithHandler:^(const void *bytes, NSInteger size) {
            const uint16_t *half_words = (const uint16_t *)bytes;
            NSInteger words = size / (NSInteger)sizeof(uint16_t);
            NSInteger n = words < 4 ? words : 4;
            NSMutableString *sample = [NSMutableString string];
            for (NSInteger i = 0; i < n; ++i) {
                [sample appendFormat:@"%s0x%04x", i == 0 ? "" : " ", half_words[i]];
            }
            NSLog(@"[output_sample] words=%ld first=%@", (long)words, sample);
        }];

        CFRelease(input_pixel_buffer);
        CFRelease(output_pixel_buffer);
    }
    return 0;
}
