#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ds4_d8f_reader.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned long ulong;

static const char *kMetalSource =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"static inline float dot8_device(device const half *codebook, float mid0, float mid1, float mid2, float mid3, float mid4, float mid5, float mid6, float mid7) {\n"
"  return float(codebook[0]) * mid0 + float(codebook[1]) * mid1 +\n"
"         float(codebook[2]) * mid2 + float(codebook[3]) * mid3 +\n"
"         float(codebook[4]) * mid4 + float(codebook[5]) * mid5 +\n"
"         float(codebook[6]) * mid6 + float(codebook[7]) * mid7;\n"
"}\n"
"\n"
"static inline float dot8_texture(texture2d<half, access::read> codebooks, uint slot, uint code, float mid0, float mid1, float mid2, float mid3, float mid4, float mid5, float mid6, float mid7) {\n"
"  const half4 lo = codebooks.read(uint2(code * 2u, slot));\n"
"  const half4 hi = codebooks.read(uint2(code * 2u + 1u, slot));\n"
"  return float(lo.x) * mid0 + float(lo.y) * mid1 + float(lo.z) * mid2 + float(lo.w) * mid3 +\n"
"         float(hi.x) * mid4 + float(hi.y) * mid5 + float(hi.z) * mid6 + float(hi.w) * mid7;\n"
"}\n"
"\n"
"static inline float dot8_texture_buffer(texture_buffer<half, access::read> codebooks, uint max_k, uint slot, uint code, float mid0, float mid1, float mid2, float mid3, float mid4, float mid5, float mid6, float mid7) {\n"
"  const uint base = (slot * max_k + code) * 2u;\n"
"  const half4 lo = codebooks.read(base);\n"
"  const half4 hi = codebooks.read(base + 1u);\n"
"  return float(lo.x) * mid0 + float(lo.y) * mid1 + float(lo.z) * mid2 + float(lo.w) * mid3 +\n"
"         float(hi.x) * mid4 + float(hi.y) * mid5 + float(hi.z) * mid6 + float(hi.w) * mid7;\n"
"}\n"
"\n"
"static inline float dot8_texture_sample(texture2d<float, access::sample> codebooks, uint slot, uint code, float mid0, float mid1, float mid2, float mid3, float mid4, float mid5, float mid6, float mid7) {\n"
"  constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);\n"
"  const float y = float(slot) + 0.5f;\n"
"  const float4 lo = codebooks.sample(s, float2(float(code * 2u) + 0.5f, y));\n"
"  const float4 hi = codebooks.sample(s, float2(float(code * 2u + 1u) + 0.5f, y));\n"
"  return lo.x * mid0 + lo.y * mid1 + lo.z * mid2 + lo.w * mid3 +\n"
"         hi.x * mid4 + hi.y * mid5 + hi.z * mid6 + hi.w * mid7;\n"
"}\n"
"\n"
"static inline float dot8_texture_gather(texture2d<float, access::sample> codebooks, uint slot, uint code, float mid0, float mid1, float mid2, float mid3, float mid4, float mid5, float mid6, float mid7) {\n"
"  constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);\n"
"  const float x = float(code * 2u) + 0.5f;\n"
"  const float y = float(slot * 4u) + 0.5f;\n"
"  const float4 lo = codebooks.gather(s, float2(x, y));\n"
"  const float4 hi = codebooks.gather(s, float2(x, y + 2.0f));\n"
"  return lo.x * mid0 + lo.y * mid1 + lo.z * mid2 + lo.w * mid3 +\n"
"         hi.x * mid4 + hi.y * mid5 + hi.z * mid6 + hi.w * mid7;\n"
"}\n"
"\n"
"static inline uint2 pack2d_coord(uint texel, uint width) {\n"
"  const uint y = texel / width;\n"
"  return uint2(texel - y * width, y);\n"
"}\n"
"\n"
"static inline float dot8_pack2d_read(texture2d<half, access::read> codebooks, uint width, uint base_texel, uint code, float mid0, float mid1, float mid2, float mid3, float mid4, float mid5, float mid6, float mid7) {\n"
"  const uint texel = base_texel + code * 2u;\n"
"  const half4 lo = codebooks.read(pack2d_coord(texel, width));\n"
"  const half4 hi = codebooks.read(pack2d_coord(texel + 1u, width));\n"
"  return float(lo.x) * mid0 + float(lo.y) * mid1 + float(lo.z) * mid2 + float(lo.w) * mid3 +\n"
"         float(hi.x) * mid4 + float(hi.y) * mid5 + float(hi.z) * mid6 + float(hi.w) * mid7;\n"
"}\n"
"\n"
"static inline float dot8_pack2d_sample(texture2d<float, access::sample> codebooks, uint width, uint base_texel, uint code, float mid0, float mid1, float mid2, float mid3, float mid4, float mid5, float mid6, float mid7) {\n"
"  constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);\n"
"  const uint texel = base_texel + code * 2u;\n"
"  const uint2 lo_coord = pack2d_coord(texel, width);\n"
"  const uint2 hi_coord = pack2d_coord(texel + 1u, width);\n"
"  const float4 lo = codebooks.sample(s, float2(float(lo_coord.x) + 0.5f, float(lo_coord.y) + 0.5f));\n"
"  const float4 hi = codebooks.sample(s, float2(float(hi_coord.x) + 0.5f, float(hi_coord.y) + 0.5f));\n"
"  return lo.x * mid0 + lo.y * mid1 + lo.z * mid2 + lo.w * mid3 +\n"
"         hi.x * mid4 + hi.y * mid5 + hi.z * mid6 + hi.w * mid7;\n"
"}\n"
"\n"
"kernel void selected_buffer(\n"
"  device const half *codebooks [[buffer(0)]],\n"
"  device const ushort *codes [[buffer(1)]],\n"
"  device const float *mid [[buffer(2)]],\n"
"  device float *out [[buffer(3)]],\n"
"  constant uint &max_k [[buffer(4)]],\n"
"  constant uint &rows [[buffer(5)]],\n"
"  constant uint &groups [[buffer(6)]],\n"
"  constant uint &slots [[buffer(7)]],\n"
"  threadgroup float *partial [[threadgroup(0)]],\n"
"  uint tid [[thread_index_in_threadgroup]],\n"
"  ushort lane [[thread_index_in_simdgroup]],\n"
"  ushort simdgroup [[simdgroup_index_in_threadgroup]],\n"
"  uint row_tile [[threadgroup_position_in_grid]]) {\n"
"  const uint row_base = row_tile << 4;\n"
"  float acc[16];\n"
"  for (uint row_offset = 0; row_offset < 16u; row_offset++) acc[row_offset] = 0.0f;\n"
"  for (uint slot = 0u; slot < slots; slot++) {\n"
"    for (uint group = tid; group < groups; group += 256u) {\n"
"      const uint mid_base = slot * groups * 8u + group * 8u;\n"
"      const float mid0 = mid[mid_base + 0u];\n"
"      const float mid1 = mid[mid_base + 1u];\n"
"      const float mid2 = mid[mid_base + 2u];\n"
"      const float mid3 = mid[mid_base + 3u];\n"
"      const float mid4 = mid[mid_base + 4u];\n"
"      const float mid5 = mid[mid_base + 5u];\n"
"      const float mid6 = mid[mid_base + 6u];\n"
"      const float mid7 = mid[mid_base + 7u];\n"
"      for (uint row_offset = 0; row_offset < 16u; row_offset++) {\n"
"        const uint row = row_base + row_offset;\n"
"        if (row >= rows) continue;\n"
"        const ushort code = codes[(ulong(slot) * ulong(rows) + ulong(row)) * ulong(groups) + ulong(group)];\n"
"        if (code >= max_k) continue;\n"
"        device const half *codebook = codebooks + (ulong(slot) * ulong(max_k) + ulong(code)) * 8ul;\n"
"        acc[row_offset] += dot8_device(codebook, mid0, mid1, mid2, mid3, mid4, mid5, mid6, mid7);\n"
"      }\n"
"    }\n"
"  }\n"
"  for (uint row_offset = 0; row_offset < 16u; row_offset++) {\n"
"    const float subtotal = simd_sum(acc[row_offset]);\n"
"    if (lane == 0u) partial[row_offset * 8u + uint(simdgroup)] = subtotal;\n"
"  }\n"
"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  if (tid == 0u) {\n"
"    for (uint row_offset = 0; row_offset < 16u; row_offset++) {\n"
"      const uint row = row_base + row_offset;\n"
"      if (row >= rows) continue;\n"
"      float total = 0.0f;\n"
"      for (uint sg = 0u; sg < 8u; sg++) total += partial[row_offset * 8u + sg];\n"
"      out[row] = total;\n"
"    }\n"
"  }\n"
"}\n"
"\n"
"kernel void selected_texture(\n"
"  texture2d<half, access::read> codebooks [[texture(0)]],\n"
"  device const ushort *codes [[buffer(0)]],\n"
"  device const float *mid [[buffer(1)]],\n"
"  device float *out [[buffer(2)]],\n"
"  constant uint &max_k [[buffer(3)]],\n"
"  constant uint &rows [[buffer(4)]],\n"
"  constant uint &groups [[buffer(5)]],\n"
"  constant uint &slots [[buffer(6)]],\n"
"  threadgroup float *partial [[threadgroup(0)]],\n"
"  uint tid [[thread_index_in_threadgroup]],\n"
"  ushort lane [[thread_index_in_simdgroup]],\n"
"  ushort simdgroup [[simdgroup_index_in_threadgroup]],\n"
"  uint row_tile [[threadgroup_position_in_grid]]) {\n"
"  const uint row_base = row_tile << 4;\n"
"  float acc[16];\n"
"  for (uint row_offset = 0; row_offset < 16u; row_offset++) acc[row_offset] = 0.0f;\n"
"  for (uint slot = 0u; slot < slots; slot++) {\n"
"    for (uint group = tid; group < groups; group += 256u) {\n"
"      const uint mid_base = slot * groups * 8u + group * 8u;\n"
"      const float mid0 = mid[mid_base + 0u];\n"
"      const float mid1 = mid[mid_base + 1u];\n"
"      const float mid2 = mid[mid_base + 2u];\n"
"      const float mid3 = mid[mid_base + 3u];\n"
"      const float mid4 = mid[mid_base + 4u];\n"
"      const float mid5 = mid[mid_base + 5u];\n"
"      const float mid6 = mid[mid_base + 6u];\n"
"      const float mid7 = mid[mid_base + 7u];\n"
"      for (uint row_offset = 0; row_offset < 16u; row_offset++) {\n"
"        const uint row = row_base + row_offset;\n"
"        if (row >= rows) continue;\n"
"        const ushort code = codes[(ulong(slot) * ulong(rows) + ulong(row)) * ulong(groups) + ulong(group)];\n"
"        if (code >= max_k) continue;\n"
"        acc[row_offset] += dot8_texture(codebooks, slot, uint(code), mid0, mid1, mid2, mid3, mid4, mid5, mid6, mid7);\n"
"      }\n"
"    }\n"
"  }\n"
"  for (uint row_offset = 0; row_offset < 16u; row_offset++) {\n"
"    const float subtotal = simd_sum(acc[row_offset]);\n"
"    if (lane == 0u) partial[row_offset * 8u + uint(simdgroup)] = subtotal;\n"
"  }\n"
"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  if (tid == 0u) {\n"
"    for (uint row_offset = 0; row_offset < 16u; row_offset++) {\n"
"      const uint row = row_base + row_offset;\n"
"      if (row >= rows) continue;\n"
"      float total = 0.0f;\n"
"      for (uint sg = 0u; sg < 8u; sg++) total += partial[row_offset * 8u + sg];\n"
"      out[row] = total;\n"
"    }\n"
"  }\n"
"}\n"
"\n"
"kernel void selected_texture_buffer(\n"
"  texture_buffer<half, access::read> codebooks [[texture(0)]],\n"
"  device const ushort *codes [[buffer(0)]],\n"
"  device const float *mid [[buffer(1)]],\n"
"  device float *out [[buffer(2)]],\n"
"  constant uint &max_k [[buffer(3)]],\n"
"  constant uint &rows [[buffer(4)]],\n"
"  constant uint &groups [[buffer(5)]],\n"
"  constant uint &slots [[buffer(6)]],\n"
"  threadgroup float *partial [[threadgroup(0)]],\n"
"  uint tid [[thread_index_in_threadgroup]],\n"
"  ushort lane [[thread_index_in_simdgroup]],\n"
"  ushort simdgroup [[simdgroup_index_in_threadgroup]],\n"
"  uint row_tile [[threadgroup_position_in_grid]]) {\n"
"  const uint row_base = row_tile << 4;\n"
"  float acc[16];\n"
"  for (uint row_offset = 0; row_offset < 16u; row_offset++) acc[row_offset] = 0.0f;\n"
"  for (uint slot = 0u; slot < slots; slot++) {\n"
"    for (uint group = tid; group < groups; group += 256u) {\n"
"      const uint mid_base = slot * groups * 8u + group * 8u;\n"
"      const float mid0 = mid[mid_base + 0u];\n"
"      const float mid1 = mid[mid_base + 1u];\n"
"      const float mid2 = mid[mid_base + 2u];\n"
"      const float mid3 = mid[mid_base + 3u];\n"
"      const float mid4 = mid[mid_base + 4u];\n"
"      const float mid5 = mid[mid_base + 5u];\n"
"      const float mid6 = mid[mid_base + 6u];\n"
"      const float mid7 = mid[mid_base + 7u];\n"
"      for (uint row_offset = 0; row_offset < 16u; row_offset++) {\n"
"        const uint row = row_base + row_offset;\n"
"        if (row >= rows) continue;\n"
"        const ushort code = codes[(ulong(slot) * ulong(rows) + ulong(row)) * ulong(groups) + ulong(group)];\n"
"        if (code >= max_k) continue;\n"
"        acc[row_offset] += dot8_texture_buffer(codebooks, max_k, slot, uint(code), mid0, mid1, mid2, mid3, mid4, mid5, mid6, mid7);\n"
"      }\n"
"    }\n"
"  }\n"
"  for (uint row_offset = 0; row_offset < 16u; row_offset++) {\n"
"    const float subtotal = simd_sum(acc[row_offset]);\n"
"    if (lane == 0u) partial[row_offset * 8u + uint(simdgroup)] = subtotal;\n"
"  }\n"
"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  if (tid == 0u) {\n"
"    for (uint row_offset = 0; row_offset < 16u; row_offset++) {\n"
"      const uint row = row_base + row_offset;\n"
"      if (row >= rows) continue;\n"
"      float total = 0.0f;\n"
"      for (uint sg = 0u; sg < 8u; sg++) total += partial[row_offset * 8u + sg];\n"
"      out[row] = total;\n"
"    }\n"
"  }\n"
"}\n"
"\n"
"kernel void selected_texture_sample(\n"
"  texture2d<float, access::sample> codebooks [[texture(0)]],\n"
"  device const ushort *codes [[buffer(0)]],\n"
"  device const float *mid [[buffer(1)]],\n"
"  device float *out [[buffer(2)]],\n"
"  constant uint &max_k [[buffer(3)]],\n"
"  constant uint &rows [[buffer(4)]],\n"
"  constant uint &groups [[buffer(5)]],\n"
"  constant uint &slots [[buffer(6)]],\n"
"  threadgroup float *partial [[threadgroup(0)]],\n"
"  uint tid [[thread_index_in_threadgroup]],\n"
"  ushort lane [[thread_index_in_simdgroup]],\n"
"  ushort simdgroup [[simdgroup_index_in_threadgroup]],\n"
"  uint row_tile [[threadgroup_position_in_grid]]) {\n"
"  const uint row_base = row_tile << 4;\n"
"  float acc[16];\n"
"  for (uint row_offset = 0; row_offset < 16u; row_offset++) acc[row_offset] = 0.0f;\n"
"  for (uint slot = 0u; slot < slots; slot++) {\n"
"    for (uint group = tid; group < groups; group += 256u) {\n"
"      const uint mid_base = slot * groups * 8u + group * 8u;\n"
"      const float mid0 = mid[mid_base + 0u];\n"
"      const float mid1 = mid[mid_base + 1u];\n"
"      const float mid2 = mid[mid_base + 2u];\n"
"      const float mid3 = mid[mid_base + 3u];\n"
"      const float mid4 = mid[mid_base + 4u];\n"
"      const float mid5 = mid[mid_base + 5u];\n"
"      const float mid6 = mid[mid_base + 6u];\n"
"      const float mid7 = mid[mid_base + 7u];\n"
"      for (uint row_offset = 0; row_offset < 16u; row_offset++) {\n"
"        const uint row = row_base + row_offset;\n"
"        if (row >= rows) continue;\n"
"        const ushort code = codes[(ulong(slot) * ulong(rows) + ulong(row)) * ulong(groups) + ulong(group)];\n"
"        if (code >= max_k) continue;\n"
"        acc[row_offset] += dot8_texture_sample(codebooks, slot, uint(code), mid0, mid1, mid2, mid3, mid4, mid5, mid6, mid7);\n"
"      }\n"
"    }\n"
"  }\n"
"  for (uint row_offset = 0; row_offset < 16u; row_offset++) {\n"
"    const float subtotal = simd_sum(acc[row_offset]);\n"
"    if (lane == 0u) partial[row_offset * 8u + uint(simdgroup)] = subtotal;\n"
"  }\n"
"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  if (tid == 0u) {\n"
"    for (uint row_offset = 0; row_offset < 16u; row_offset++) {\n"
"      const uint row = row_base + row_offset;\n"
"      if (row >= rows) continue;\n"
"      float total = 0.0f;\n"
"      for (uint sg = 0u; sg < 8u; sg++) total += partial[row_offset * 8u + sg];\n"
"      out[row] = total;\n"
"    }\n"
"  }\n"
"}\n"
"\n"
"kernel void selected_texture_gather(\n"
"  texture2d<float, access::sample> codebooks [[texture(0)]],\n"
"  device const ushort *codes [[buffer(0)]],\n"
"  device const float *mid [[buffer(1)]],\n"
"  device float *out [[buffer(2)]],\n"
"  constant uint &max_k [[buffer(3)]],\n"
"  constant uint &rows [[buffer(4)]],\n"
"  constant uint &groups [[buffer(5)]],\n"
"  constant uint &slots [[buffer(6)]],\n"
"  threadgroup float *partial [[threadgroup(0)]],\n"
"  uint tid [[thread_index_in_threadgroup]],\n"
"  ushort lane [[thread_index_in_simdgroup]],\n"
"  ushort simdgroup [[simdgroup_index_in_threadgroup]],\n"
"  uint row_tile [[threadgroup_position_in_grid]]) {\n"
"  const uint row_base = row_tile << 4;\n"
"  float acc[16];\n"
"  for (uint row_offset = 0; row_offset < 16u; row_offset++) acc[row_offset] = 0.0f;\n"
"  for (uint slot = 0u; slot < slots; slot++) {\n"
"    for (uint group = tid; group < groups; group += 256u) {\n"
"      const uint mid_base = slot * groups * 8u + group * 8u;\n"
"      const float mid0 = mid[mid_base + 0u];\n"
"      const float mid1 = mid[mid_base + 1u];\n"
"      const float mid2 = mid[mid_base + 2u];\n"
"      const float mid3 = mid[mid_base + 3u];\n"
"      const float mid4 = mid[mid_base + 4u];\n"
"      const float mid5 = mid[mid_base + 5u];\n"
"      const float mid6 = mid[mid_base + 6u];\n"
"      const float mid7 = mid[mid_base + 7u];\n"
"      for (uint row_offset = 0; row_offset < 16u; row_offset++) {\n"
"        const uint row = row_base + row_offset;\n"
"        if (row >= rows) continue;\n"
"        const ushort code = codes[(ulong(slot) * ulong(rows) + ulong(row)) * ulong(groups) + ulong(group)];\n"
"        if (code >= max_k) continue;\n"
"        acc[row_offset] += dot8_texture_gather(codebooks, slot, uint(code), mid0, mid1, mid2, mid3, mid4, mid5, mid6, mid7);\n"
"      }\n"
"    }\n"
"  }\n"
"  for (uint row_offset = 0; row_offset < 16u; row_offset++) {\n"
"    const float subtotal = simd_sum(acc[row_offset]);\n"
"    if (lane == 0u) partial[row_offset * 8u + uint(simdgroup)] = subtotal;\n"
"  }\n"
"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  if (tid == 0u) {\n"
"    for (uint row_offset = 0; row_offset < 16u; row_offset++) {\n"
"      const uint row = row_base + row_offset;\n"
"      if (row >= rows) continue;\n"
"      float total = 0.0f;\n"
"      for (uint sg = 0u; sg < 8u; sg++) total += partial[row_offset * 8u + sg];\n"
"      out[row] = total;\n"
"    }\n"
"  }\n"
"}\n"
"\n"
"kernel void warm_pack2d_codebooks(\n"
"  texture2d<half, access::read> codebooks [[texture(0)]],\n"
"  device const uint *base_texels [[buffer(0)]],\n"
"  device float *sink [[buffer(1)]],\n"
"  constant uint &pack_width [[buffer(2)]],\n"
"  constant uint &max_k [[buffer(3)]],\n"
"  constant uint &slots [[buffer(4)]],\n"
"  uint index [[thread_position_in_grid]]) {\n"
"  const uint texels_per_slot = max_k * 2u;\n"
"  const uint total = slots * texels_per_slot;\n"
"  if (index >= total) return;\n"
"  const uint slot = index / texels_per_slot;\n"
"  const uint inner = index - slot * texels_per_slot;\n"
"  const uint texel = base_texels[slot] + inner;\n"
"  const half4 v = codebooks.read(pack2d_coord(texel, pack_width));\n"
"  sink[index] = float(v.x) + float(v.y) + float(v.z) + float(v.w);\n"
"}\n"
"\n"
"kernel void selected_pack2d_read(\n"
"  texture2d<half, access::read> codebooks [[texture(0)]],\n"
"  device const ushort *codes [[buffer(0)]],\n"
"  device const float *mid [[buffer(1)]],\n"
"  device float *out [[buffer(2)]],\n"
"  constant uint &max_k [[buffer(3)]],\n"
"  constant uint &rows [[buffer(4)]],\n"
"  constant uint &groups [[buffer(5)]],\n"
"  constant uint &slots [[buffer(6)]],\n"
"  device const uint *base_texels [[buffer(7)]],\n"
"  constant uint &pack_width [[buffer(8)]],\n"
"  threadgroup float *partial [[threadgroup(0)]],\n"
"  uint tid [[thread_index_in_threadgroup]],\n"
"  ushort lane [[thread_index_in_simdgroup]],\n"
"  ushort simdgroup [[simdgroup_index_in_threadgroup]],\n"
"  uint row_tile [[threadgroup_position_in_grid]]) {\n"
"  const uint row_base = row_tile << 4;\n"
"  float acc[16];\n"
"  for (uint row_offset = 0; row_offset < 16u; row_offset++) acc[row_offset] = 0.0f;\n"
"  for (uint slot = 0u; slot < slots; slot++) {\n"
"    const uint base_texel = base_texels[slot];\n"
"    for (uint group = tid; group < groups; group += 256u) {\n"
"      const uint mid_base = slot * groups * 8u + group * 8u;\n"
"      const float mid0 = mid[mid_base + 0u];\n"
"      const float mid1 = mid[mid_base + 1u];\n"
"      const float mid2 = mid[mid_base + 2u];\n"
"      const float mid3 = mid[mid_base + 3u];\n"
"      const float mid4 = mid[mid_base + 4u];\n"
"      const float mid5 = mid[mid_base + 5u];\n"
"      const float mid6 = mid[mid_base + 6u];\n"
"      const float mid7 = mid[mid_base + 7u];\n"
"      for (uint row_offset = 0; row_offset < 16u; row_offset++) {\n"
"        const uint row = row_base + row_offset;\n"
"        if (row >= rows) continue;\n"
"        const ushort code = codes[(ulong(slot) * ulong(rows) + ulong(row)) * ulong(groups) + ulong(group)];\n"
"        if (code >= max_k) continue;\n"
"        acc[row_offset] += dot8_pack2d_read(codebooks, pack_width, base_texel, uint(code), mid0, mid1, mid2, mid3, mid4, mid5, mid6, mid7);\n"
"      }\n"
"    }\n"
"  }\n"
"  for (uint row_offset = 0; row_offset < 16u; row_offset++) {\n"
"    const float subtotal = simd_sum(acc[row_offset]);\n"
"    if (lane == 0u) partial[row_offset * 8u + uint(simdgroup)] = subtotal;\n"
"  }\n"
"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  if (tid == 0u) {\n"
"    for (uint row_offset = 0; row_offset < 16u; row_offset++) {\n"
"      const uint row = row_base + row_offset;\n"
"      if (row >= rows) continue;\n"
"      float total = 0.0f;\n"
"      for (uint sg = 0u; sg < 8u; sg++) total += partial[row_offset * 8u + sg];\n"
"      out[row] = total;\n"
"    }\n"
"  }\n"
"}\n"
"\n"
"kernel void selected_pack2d_sample(\n"
"  texture2d<float, access::sample> codebooks [[texture(0)]],\n"
"  device const ushort *codes [[buffer(0)]],\n"
"  device const float *mid [[buffer(1)]],\n"
"  device float *out [[buffer(2)]],\n"
"  constant uint &max_k [[buffer(3)]],\n"
"  constant uint &rows [[buffer(4)]],\n"
"  constant uint &groups [[buffer(5)]],\n"
"  constant uint &slots [[buffer(6)]],\n"
"  device const uint *base_texels [[buffer(7)]],\n"
"  constant uint &pack_width [[buffer(8)]],\n"
"  threadgroup float *partial [[threadgroup(0)]],\n"
"  uint tid [[thread_index_in_threadgroup]],\n"
"  ushort lane [[thread_index_in_simdgroup]],\n"
"  ushort simdgroup [[simdgroup_index_in_threadgroup]],\n"
"  uint row_tile [[threadgroup_position_in_grid]]) {\n"
"  const uint row_base = row_tile << 4;\n"
"  float acc[16];\n"
"  for (uint row_offset = 0; row_offset < 16u; row_offset++) acc[row_offset] = 0.0f;\n"
"  for (uint slot = 0u; slot < slots; slot++) {\n"
"    const uint base_texel = base_texels[slot];\n"
"    for (uint group = tid; group < groups; group += 256u) {\n"
"      const uint mid_base = slot * groups * 8u + group * 8u;\n"
"      const float mid0 = mid[mid_base + 0u];\n"
"      const float mid1 = mid[mid_base + 1u];\n"
"      const float mid2 = mid[mid_base + 2u];\n"
"      const float mid3 = mid[mid_base + 3u];\n"
"      const float mid4 = mid[mid_base + 4u];\n"
"      const float mid5 = mid[mid_base + 5u];\n"
"      const float mid6 = mid[mid_base + 6u];\n"
"      const float mid7 = mid[mid_base + 7u];\n"
"      for (uint row_offset = 0; row_offset < 16u; row_offset++) {\n"
"        const uint row = row_base + row_offset;\n"
"        if (row >= rows) continue;\n"
"        const ushort code = codes[(ulong(slot) * ulong(rows) + ulong(row)) * ulong(groups) + ulong(group)];\n"
"        if (code >= max_k) continue;\n"
"        acc[row_offset] += dot8_pack2d_sample(codebooks, pack_width, base_texel, uint(code), mid0, mid1, mid2, mid3, mid4, mid5, mid6, mid7);\n"
"      }\n"
"    }\n"
"  }\n"
"  for (uint row_offset = 0; row_offset < 16u; row_offset++) {\n"
"    const float subtotal = simd_sum(acc[row_offset]);\n"
"    if (lane == 0u) partial[row_offset * 8u + uint(simdgroup)] = subtotal;\n"
"  }\n"
"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  if (tid == 0u) {\n"
"    for (uint row_offset = 0; row_offset < 16u; row_offset++) {\n"
"      const uint row = row_base + row_offset;\n"
"      if (row >= rows) continue;\n"
"      float total = 0.0f;\n"
"      for (uint sg = 0u; sg < 8u; sg++) total += partial[row_offset * 8u + sg];\n"
"      out[row] = total;\n"
"    }\n"
"  }\n"
"}\n";

static float f16_to_f32(uint16_t bits) {
    __fp16 value;
    memcpy(&value, &bits, sizeof(value));
    return (float)value;
}

static uint16_t load_u16(const uint8_t *bytes) {
    return (uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8);
}

static int parse_experts(const char *text, uint32_t *experts, uint32_t *count) {
    *count = 0;
    const char *cursor = text;
    while (cursor && *cursor) {
        char *end = NULL;
        const unsigned long value = strtoul(cursor, &end, 10);
        if (end == cursor || value >= 256ul || *count >= 6u) return 0;
        experts[(*count)++] = (uint32_t)value;
        if (*end == '\0') break;
        if (*end != ',') return 0;
        cursor = end + 1;
    }
    return *count > 0;
}

static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString *name) {
    NSError *error = nil;
    id<MTLFunction> function = [library newFunctionWithName:name];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) NSLog(@"pipeline %@ failed: %@", name, error);
    return pipeline;
}

static id<MTLTexture> make_buffer_backed_texture(id<MTLBuffer> texel_buffer, uint32_t width, uint32_t height) {
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                          width:width
                                                                                         height:height
                                                                                      mipmapped:NO];
    descriptor.usage = MTLTextureUsageShaderRead;
    return [texel_buffer newTextureWithDescriptor:descriptor
                                           offset:0
                                      bytesPerRow:(NSUInteger)width * 4u * sizeof(uint16_t)];
}

static id<MTLTexture> make_r16_buffer_backed_texture(id<MTLBuffer> texel_buffer, uint32_t width, uint32_t height) {
    const NSUInteger bytes_per_row = (NSUInteger)width * sizeof(uint16_t);
    if ((bytes_per_row & 15u) != 0u) return nil;
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                                                                          width:width
                                                                                         height:height
                                                                                      mipmapped:NO];
    descriptor.usage = MTLTextureUsageShaderRead;
    return [texel_buffer newTextureWithDescriptor:descriptor
                                           offset:0
                                      bytesPerRow:bytes_per_row];
}

static id<MTLTexture> make_texture_buffer(id<MTLBuffer> texel_buffer, uint32_t texel_count) {
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                              width:texel_count
                                                                                    resourceOptions:MTLResourceStorageModeShared
                                                                                              usage:MTLTextureUsageShaderRead];
    return [texel_buffer newTextureWithDescriptor:descriptor offset:0 bytesPerRow:(NSUInteger)texel_count * 4u * sizeof(uint16_t)];
}

static double run_buffer(id<MTLCommandQueue> queue,
                         id<MTLComputePipelineState> pipeline,
                         id<MTLBuffer> codebooks,
                         id<MTLBuffer> codes,
                         id<MTLBuffer> mid,
                         id<MTLBuffer> out,
                         uint32_t max_k,
                         uint32_t rows,
                         uint32_t groups,
                         uint32_t slots,
                         uint32_t rounds) {
    double best = 1.0e30;
    for (uint32_t round = 0; round < rounds; round++) {
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:codebooks offset:0 atIndex:0];
        [encoder setBuffer:codes offset:0 atIndex:1];
        [encoder setBuffer:mid offset:0 atIndex:2];
        [encoder setBuffer:out offset:0 atIndex:3];
        [encoder setBytes:&max_k length:sizeof(max_k) atIndex:4];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:5];
        [encoder setBytes:&groups length:sizeof(groups) atIndex:6];
        [encoder setBytes:&slots length:sizeof(slots) atIndex:7];
        [encoder setThreadgroupMemoryLength:16u * 8u * sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake((rows + 15u) >> 4, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        const double ms = (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0;
        if (ms > 0.0 && ms < best) best = ms;
    }
    return best;
}

static double run_texture(id<MTLCommandQueue> queue,
                          id<MTLComputePipelineState> pipeline,
                          id<MTLTexture> codebooks,
                          id<MTLBuffer> codes,
                          id<MTLBuffer> mid,
                          id<MTLBuffer> out,
                          uint32_t max_k,
                          uint32_t rows,
                          uint32_t groups,
                          uint32_t slots,
                          uint32_t rounds) {
    double best = 1.0e30;
    for (uint32_t round = 0; round < rounds; round++) {
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setTexture:codebooks atIndex:0];
        [encoder setBuffer:codes offset:0 atIndex:0];
        [encoder setBuffer:mid offset:0 atIndex:1];
        [encoder setBuffer:out offset:0 atIndex:2];
        [encoder setBytes:&max_k length:sizeof(max_k) atIndex:3];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
        [encoder setBytes:&groups length:sizeof(groups) atIndex:5];
        [encoder setBytes:&slots length:sizeof(slots) atIndex:6];
        [encoder setThreadgroupMemoryLength:16u * 8u * sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake((rows + 15u) >> 4, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        const double ms = (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0;
        if (ms > 0.0 && ms < best) best = ms;
    }
    return best;
}

static double run_pack2d(id<MTLCommandQueue> queue,
                         id<MTLComputePipelineState> pipeline,
                         id<MTLTexture> codebooks,
                         id<MTLBuffer> codes,
                         id<MTLBuffer> mid,
                         id<MTLBuffer> out,
                         id<MTLBuffer> base_texels,
                         uint32_t pack_width,
                         uint32_t max_k,
                         uint32_t rows,
                         uint32_t groups,
                         uint32_t slots,
                         uint32_t rounds) {
    double best = 1.0e30;
    for (uint32_t round = 0; round < rounds; round++) {
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setTexture:codebooks atIndex:0];
        [encoder setBuffer:codes offset:0 atIndex:0];
        [encoder setBuffer:mid offset:0 atIndex:1];
        [encoder setBuffer:out offset:0 atIndex:2];
        [encoder setBytes:&max_k length:sizeof(max_k) atIndex:3];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
        [encoder setBytes:&groups length:sizeof(groups) atIndex:5];
        [encoder setBytes:&slots length:sizeof(slots) atIndex:6];
        [encoder setBuffer:base_texels offset:0 atIndex:7];
        [encoder setBytes:&pack_width length:sizeof(pack_width) atIndex:8];
        [encoder setThreadgroupMemoryLength:16u * 8u * sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake((rows + 15u) >> 4, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        const double ms = (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0;
        if (ms > 0.0 && ms < best) best = ms;
    }
    return best;
}

static double run_warm_pack2d(id<MTLCommandQueue> queue,
                              id<MTLComputePipelineState> pipeline,
                              id<MTLTexture> codebooks,
                              id<MTLBuffer> base_texels,
                              id<MTLBuffer> sink,
                              uint32_t pack_width,
                              uint32_t max_k,
                              uint32_t slots,
                              uint32_t rounds) {
    double best = 1.0e30;
    const uint32_t total = slots * max_k * 2u;
    for (uint32_t round = 0; round < rounds; round++) {
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setTexture:codebooks atIndex:0];
        [encoder setBuffer:base_texels offset:0 atIndex:0];
        [encoder setBuffer:sink offset:0 atIndex:1];
        [encoder setBytes:&pack_width length:sizeof(pack_width) atIndex:2];
        [encoder setBytes:&max_k length:sizeof(max_k) atIndex:3];
        [encoder setBytes:&slots length:sizeof(slots) atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        const double ms = (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0;
        if (ms > 0.0 && ms < best) best = ms;
    }
    return best;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "usage: %s layer.d8f expert_csv [rows=4096] [rounds=80]\n", argv[0]);
            return 2;
        }
        const char *path = argv[1];
        uint32_t experts[6] = {0};
        uint32_t slots = 0;
        if (!parse_experts(argv[2], experts, &slots)) {
            fprintf(stderr, "bad expert csv: %s\n", argv[2]);
            return 2;
        }
        uint32_t rows = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10) : 4096u;
        uint32_t rounds = argc > 4 ? (uint32_t)strtoul(argv[4], NULL, 10) : 80u;
        if (rows == 0 || rows > 4096u) rows = 4096u;
        if (rounds == 0) rounds = 1u;

        ds4_d8f_file file;
        if (!ds4_d8f_open(path, &file)) return 1;
        const uint32_t groups = 256u;
        const uint32_t in_dim = groups * 8u;
        ds4_d8f_record down[6];
        ds4_d8f_native_code_record native[6];
        uint32_t max_k = 0;
        uint32_t base_texels_host[6] = {0};
        uint64_t max_codebook_end = 0;
        int pack2d_available = 1;
        for (uint32_t slot = 0; slot < slots; slot++) {
            if (!ds4_d8f_get_record(&file, DS4_D8F_DOWN, experts[slot], &down[slot]) ||
                !ds4_d8f_get_down_native_codes(&file, experts[slot], &native[slot])) {
                fprintf(stderr, "missing down/native expert=%u\n", experts[slot]);
                ds4_d8f_close(&file);
                return 1;
            }
            if (native[slot].rows < rows || native[slot].groups != groups) {
                fprintf(stderr, "bad native dims expert=%u rows=%u groups=%u\n",
                        experts[slot], native[slot].rows, native[slot].groups);
                ds4_d8f_close(&file);
                return 1;
            }
            if ((down[slot].codebook_offset & 7ull) != 0ull ||
                (down[slot].codebook_offset >> 3) > (uint64_t)UINT32_MAX) {
                pack2d_available = 0;
            } else {
                base_texels_host[slot] = (uint32_t)(down[slot].codebook_offset >> 3);
            }
            const uint64_t codebook_end = down[slot].codebook_offset + down[slot].codebook_bytes;
            if (codebook_end > max_codebook_end) max_codebook_end = codebook_end;
            if (down[slot].k > max_k) max_k = down[slot].k;
        }
        if (max_k == 0u) {
            ds4_d8f_close(&file);
            return 1;
        }
        const NSUInteger codebook_bytes = (NSUInteger)slots * max_k * 8u * sizeof(uint16_t);
        const NSUInteger texel_bytes = (NSUInteger)slots * max_k * 2u * 4u * sizeof(uint16_t);
        const NSUInteger gather_texel_bytes = (NSUInteger)slots * 4u * max_k * 2u * sizeof(uint16_t);
        const NSUInteger code_bytes = (NSUInteger)slots * rows * groups * sizeof(uint16_t);
        const NSUInteger mid_bytes = (NSUInteger)slots * in_dim * sizeof(float);
        const NSUInteger out_bytes = (NSUInteger)rows * sizeof(float);
        uint16_t *codebook_host = (uint16_t *)calloc(1, codebook_bytes);
        uint16_t *texel_host = (uint16_t *)calloc(1, texel_bytes);
        uint16_t *gather_texel_host = (uint16_t *)calloc(1, gather_texel_bytes);
        uint16_t *code_host = (uint16_t *)malloc(code_bytes);
        float *mid_host = (float *)malloc(mid_bytes);
        float *ref = (float *)calloc(rows, sizeof(float));
        if (!codebook_host || !texel_host || !gather_texel_host || !code_host || !mid_host || !ref) {
            free(codebook_host); free(texel_host); free(gather_texel_host); free(code_host); free(mid_host); free(ref);
            ds4_d8f_close(&file);
            return 1;
        }
        for (uint32_t slot = 0; slot < slots; slot++) {
            for (uint32_t code = 0; code < down[slot].k; code++) {
                const uint8_t *source = file.map + down[slot].codebook_offset + (uint64_t)code * 16u;
                for (uint32_t lane = 0; lane < 8u; lane++) {
                    const uint16_t bits = load_u16(source + (uint64_t)lane * 2u);
                    codebook_host[((uint64_t)slot * max_k + code) * 8u + lane] = bits;
                    texel_host[((uint64_t)slot * max_k * 2u + code * 2u + lane / 4u) * 4u + lane % 4u] = bits;
                }
                const uint64_t gather_base = ((uint64_t)slot * 4u * max_k + code) * 2u;
                gather_texel_host[gather_base + 0u] = load_u16(source + 3u * 2u);
                gather_texel_host[gather_base + 1u] = load_u16(source + 2u * 2u);
                gather_texel_host[gather_base + (uint64_t)max_k * 2u + 0u] = load_u16(source + 0u * 2u);
                gather_texel_host[gather_base + (uint64_t)max_k * 2u + 1u] = load_u16(source + 1u * 2u);
                gather_texel_host[gather_base + (uint64_t)max_k * 4u + 0u] = load_u16(source + 7u * 2u);
                gather_texel_host[gather_base + (uint64_t)max_k * 4u + 1u] = load_u16(source + 6u * 2u);
                gather_texel_host[gather_base + (uint64_t)max_k * 6u + 0u] = load_u16(source + 4u * 2u);
                gather_texel_host[gather_base + (uint64_t)max_k * 6u + 1u] = load_u16(source + 5u * 2u);
            }
            memcpy(code_host + (uint64_t)slot * rows * groups,
                   file.map + native[slot].offset,
                   (size_t)rows * groups * sizeof(uint16_t));
            for (uint32_t index = 0; index < in_dim; index++) {
                float value = 0.50f * sinf((float)(index + slot * 17u) * 0.011f) +
                              0.25f * cosf((float)(index + slot * 29u) * 0.023f);
                if ((down[slot].flags & 1u) && down[slot].scale_offset && down[slot].scale_bytes >= (index + 1u) * 2u) {
                    value *= f16_to_f32(load_u16(file.map + down[slot].scale_offset + (uint64_t)index * 2u));
                }
                mid_host[(uint64_t)slot * in_dim + index] = value;
            }
        }
        for (uint32_t row = 0; row < rows; row++) {
            double total = 0.0;
            for (uint32_t slot = 0; slot < slots; slot++) {
                const uint16_t *codes = code_host + (uint64_t)slot * rows * groups;
                const float *mid = mid_host + (uint64_t)slot * in_dim;
                for (uint32_t group = 0; group < groups; group++) {
                    const uint32_t code = codes[(uint64_t)row * groups + group];
                    const uint16_t *cb = codebook_host + ((uint64_t)slot * max_k + code) * 8u;
                    const uint32_t mid_base = group << 3;
                    for (uint32_t lane = 0; lane < 8u; lane++) {
                        total += (double)f16_to_f32(cb[lane]) * (double)mid[mid_base + lane];
                    }
                }
            }
            ref[row] = (float)total;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kMetalSource]
                                                       options:nil
                                                         error:&error];
        if (!library) {
            NSLog(@"library creation failed: %@", error);
            free(codebook_host); free(texel_host); free(gather_texel_host); free(code_host); free(mid_host); free(ref);
            ds4_d8f_close(&file);
            return 1;
        }
        id<MTLComputePipelineState> buffer_pipeline = make_pipeline(device, library, @"selected_buffer");
        id<MTLComputePipelineState> texture_pipeline = make_pipeline(device, library, @"selected_texture");
        id<MTLComputePipelineState> texture_buffer_pipeline = make_pipeline(device, library, @"selected_texture_buffer");
        id<MTLComputePipelineState> texture_sample_pipeline = make_pipeline(device, library, @"selected_texture_sample");
        id<MTLComputePipelineState> texture_gather_pipeline = make_pipeline(device, library, @"selected_texture_gather");
        id<MTLComputePipelineState> warm_pack2d_pipeline = make_pipeline(device, library, @"warm_pack2d_codebooks");
        id<MTLComputePipelineState> pack2d_read_pipeline = make_pipeline(device, library, @"selected_pack2d_read");
        id<MTLComputePipelineState> pack2d_sample_pipeline = make_pipeline(device, library, @"selected_pack2d_sample");
        if (!buffer_pipeline || !texture_pipeline || !texture_buffer_pipeline || !texture_sample_pipeline ||
            !texture_gather_pipeline || !warm_pack2d_pipeline || !pack2d_read_pipeline || !pack2d_sample_pipeline) {
            free(codebook_host); free(texel_host); free(gather_texel_host); free(code_host); free(mid_host); free(ref);
            ds4_d8f_close(&file);
            return 1;
        }
        id<MTLBuffer> codebook_buf = [device newBufferWithBytes:codebook_host length:codebook_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> texel_buf = [device newBufferWithBytes:texel_host length:texel_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> gather_texel_buf = [device newBufferWithBytes:gather_texel_host length:gather_texel_bytes options:MTLResourceStorageModeShared];
        id<MTLTexture> texture = make_buffer_backed_texture(texel_buf, max_k * 2u, slots);
        id<MTLTexture> texture_buffer = make_texture_buffer(texel_buf, slots * max_k * 2u);
        id<MTLTexture> texture_gather = make_r16_buffer_backed_texture(gather_texel_buf, max_k * 2u, slots * 4u);
        const char *pack_width_env = getenv("DS4_TEXTURE_CANARY_PACK2D_WIDTH");
        uint32_t pack2d_width = pack_width_env && pack_width_env[0]
                              ? (uint32_t)strtoul(pack_width_env, NULL, 10)
                              : 16384u;
        if (pack2d_width < 2u) pack2d_width = 16384u;
        const NSUInteger pack2d_bytes_per_row = (NSUInteger)pack2d_width * 4u * sizeof(uint16_t);
        const NSUInteger pack2d_height = max_codebook_end
            ? (NSUInteger)((max_codebook_end + (uint64_t)pack2d_bytes_per_row - 1ull) / (uint64_t)pack2d_bytes_per_row)
            : 0u;
        const NSUInteger pack2d_view_bytes = pack2d_height * pack2d_bytes_per_row;
        if (pack2d_height == 0u ||
            pack2d_width > 16384u ||
            pack2d_height > 16384u ||
            pack2d_view_bytes > (NSUInteger)file.size) {
            pack2d_available = 0;
        }
        id<MTLBuffer> base_texels_buf = [device newBufferWithBytes:base_texels_host
                                                            length:(NSUInteger)slots * sizeof(uint32_t)
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> pack_buf = pack2d_available
            ? [device newBufferWithBytesNoCopy:(void *)file.map
                                        length:pack2d_view_bytes
                                       options:MTLResourceStorageModeShared
                                   deallocator:nil]
            : nil;
        id<MTLTexture> pack2d_texture = pack_buf
            ? make_buffer_backed_texture(pack_buf, pack2d_width, (uint32_t)pack2d_height)
            : nil;
        if (!pack2d_texture || !base_texels_buf) pack2d_available = 0;
        id<MTLBuffer> code_buf = [device newBufferWithBytes:code_host length:code_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> mid_buf = [device newBufferWithBytes:mid_host length:mid_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buffer = [device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_texture = [device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_texture_buffer = [device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_texture_sample = [device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_texture_gather = [device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_pack2d_read = [device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_pack2d_sample = [device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> warm_pack2d_sink = [device newBufferWithLength:(NSUInteger)slots * max_k * 2u * sizeof(float)
                                                              options:MTLResourceStorageModeShared];
        if (!codebook_buf || !texel_buf || !gather_texel_buf || !texture || !texture_buffer || !texture_gather || !code_buf || !mid_buf ||
            !out_buffer || !out_texture || !out_texture_buffer || !out_texture_sample ||
            !out_texture_gather || !out_pack2d_read || !out_pack2d_sample || !warm_pack2d_sink) {
            fprintf(stderr, "Metal allocation failed\n");
            free(codebook_host); free(texel_host); free(gather_texel_host); free(code_host); free(mid_host); free(ref);
            ds4_d8f_close(&file);
            return 1;
        }
        (void)run_buffer(queue, buffer_pipeline, codebook_buf, code_buf, mid_buf, out_buffer, max_k, rows, groups, slots, 1);
        (void)run_texture(queue, texture_pipeline, texture, code_buf, mid_buf, out_texture, max_k, rows, groups, slots, 1);
        (void)run_texture(queue, texture_buffer_pipeline, texture_buffer, code_buf, mid_buf, out_texture_buffer, max_k, rows, groups, slots, 1);
        (void)run_texture(queue, texture_sample_pipeline, texture, code_buf, mid_buf, out_texture_sample, max_k, rows, groups, slots, 1);
        (void)run_texture(queue, texture_gather_pipeline, texture_gather, code_buf, mid_buf, out_texture_gather, max_k, rows, groups, slots, 1);
        if (pack2d_available) {
            (void)run_warm_pack2d(queue, warm_pack2d_pipeline, pack2d_texture, base_texels_buf, warm_pack2d_sink,
                                  pack2d_width, max_k, slots, 1);
            (void)run_pack2d(queue, pack2d_read_pipeline, pack2d_texture, code_buf, mid_buf, out_pack2d_read,
                             base_texels_buf, pack2d_width, max_k, rows, groups, slots, 1);
            (void)run_pack2d(queue, pack2d_sample_pipeline, pack2d_texture, code_buf, mid_buf, out_pack2d_sample,
                             base_texels_buf, pack2d_width, max_k, rows, groups, slots, 1);
        }
        const float *buffer_values = out_buffer.contents;
        const float *texture_values = out_texture.contents;
        const float *texture_buffer_values = out_texture_buffer.contents;
        const float *texture_sample_values = out_texture_sample.contents;
        const float *texture_gather_values = out_texture_gather.contents;
        const float *pack2d_read_values = out_pack2d_read.contents;
        const float *pack2d_sample_values = out_pack2d_sample.contents;
        float max_abs_buffer = 0.0f;
        float max_abs_texture = 0.0f;
        float max_abs_texture_buffer = 0.0f;
        float max_abs_texture_sample = 0.0f;
        float max_abs_texture_gather = 0.0f;
        float max_abs_pack2d_read = pack2d_available ? 0.0f : NAN;
        float max_abs_pack2d_sample = pack2d_available ? 0.0f : NAN;
        float max_abs_buffer_texture = 0.0f;
        float max_abs_buffer_texture_buffer = 0.0f;
        float max_abs_buffer_texture_sample = 0.0f;
        float max_abs_buffer_texture_gather = 0.0f;
        float max_abs_buffer_pack2d_read = pack2d_available ? 0.0f : NAN;
        float max_abs_buffer_pack2d_sample = pack2d_available ? 0.0f : NAN;
        for (uint32_t row = 0; row < rows; row++) {
            const float db = fabsf(buffer_values[row] - ref[row]);
            const float dt = fabsf(texture_values[row] - ref[row]);
            const float dtb = fabsf(texture_buffer_values[row] - ref[row]);
            const float dts = fabsf(texture_sample_values[row] - ref[row]);
            const float dtg = fabsf(texture_gather_values[row] - ref[row]);
            const float dpr = pack2d_available ? fabsf(pack2d_read_values[row] - ref[row]) : NAN;
            const float dps = pack2d_available ? fabsf(pack2d_sample_values[row] - ref[row]) : NAN;
            const float dbt = fabsf(buffer_values[row] - texture_values[row]);
            const float dbtb = fabsf(buffer_values[row] - texture_buffer_values[row]);
            const float dbts = fabsf(buffer_values[row] - texture_sample_values[row]);
            const float dbtg = fabsf(buffer_values[row] - texture_gather_values[row]);
            const float dbpr = pack2d_available ? fabsf(buffer_values[row] - pack2d_read_values[row]) : NAN;
            const float dbps = pack2d_available ? fabsf(buffer_values[row] - pack2d_sample_values[row]) : NAN;
            if (db > max_abs_buffer) max_abs_buffer = db;
            if (dt > max_abs_texture) max_abs_texture = dt;
            if (dtb > max_abs_texture_buffer) max_abs_texture_buffer = dtb;
            if (dts > max_abs_texture_sample) max_abs_texture_sample = dts;
            if (dtg > max_abs_texture_gather) max_abs_texture_gather = dtg;
            if (pack2d_available && dpr > max_abs_pack2d_read) max_abs_pack2d_read = dpr;
            if (pack2d_available && dps > max_abs_pack2d_sample) max_abs_pack2d_sample = dps;
            if (dbt > max_abs_buffer_texture) max_abs_buffer_texture = dbt;
            if (dbtb > max_abs_buffer_texture_buffer) max_abs_buffer_texture_buffer = dbtb;
            if (dbts > max_abs_buffer_texture_sample) max_abs_buffer_texture_sample = dbts;
            if (dbtg > max_abs_buffer_texture_gather) max_abs_buffer_texture_gather = dbtg;
            if (pack2d_available && dbpr > max_abs_buffer_pack2d_read) max_abs_buffer_pack2d_read = dbpr;
            if (pack2d_available && dbps > max_abs_buffer_pack2d_sample) max_abs_buffer_pack2d_sample = dbps;
        }
        const char *measure_order = getenv("DS4_TEXTURE_CANARY_ORDER");
        if (!measure_order || !measure_order[0]) measure_order = "B2TSGPQ";
        double buffer_ms = 0.0;
        double texture_ms = 0.0;
        double texture_buffer_ms = 0.0;
        double texture_sample_ms = 0.0;
        double texture_gather_ms = 0.0;
        double pack2d_warm_ms = NAN;
        double pack2d_read_ms = NAN;
        double pack2d_sample_ms = NAN;
        for (const char *cursor = measure_order; *cursor; cursor++) {
            if (*cursor == 'B' && buffer_ms == 0.0) {
                buffer_ms = run_buffer(queue, buffer_pipeline, codebook_buf, code_buf, mid_buf, out_buffer, max_k, rows, groups, slots, rounds);
            } else if (*cursor == '2' && texture_ms == 0.0) {
                texture_ms = run_texture(queue, texture_pipeline, texture, code_buf, mid_buf, out_texture, max_k, rows, groups, slots, rounds);
            } else if (*cursor == 'T' && texture_buffer_ms == 0.0) {
                texture_buffer_ms = run_texture(queue, texture_buffer_pipeline, texture_buffer, code_buf, mid_buf, out_texture_buffer, max_k, rows, groups, slots, rounds);
            } else if (*cursor == 'S' && texture_sample_ms == 0.0) {
                texture_sample_ms = run_texture(queue, texture_sample_pipeline, texture, code_buf, mid_buf, out_texture_sample, max_k, rows, groups, slots, rounds);
            } else if (*cursor == 'G' && texture_gather_ms == 0.0) {
                texture_gather_ms = run_texture(queue, texture_gather_pipeline, texture_gather, code_buf, mid_buf, out_texture_gather, max_k, rows, groups, slots, rounds);
            } else if (*cursor == 'W' && pack2d_available && isnan(pack2d_warm_ms)) {
                pack2d_warm_ms = run_warm_pack2d(queue, warm_pack2d_pipeline, pack2d_texture, base_texels_buf, warm_pack2d_sink,
                                                 pack2d_width, max_k, slots, rounds);
            } else if (*cursor == 'P' && pack2d_available && isnan(pack2d_read_ms)) {
                pack2d_read_ms = run_pack2d(queue, pack2d_read_pipeline, pack2d_texture, code_buf, mid_buf, out_pack2d_read,
                                            base_texels_buf, pack2d_width, max_k, rows, groups, slots, rounds);
            } else if (*cursor == 'Q' && pack2d_available && isnan(pack2d_sample_ms)) {
                pack2d_sample_ms = run_pack2d(queue, pack2d_sample_pipeline, pack2d_texture, code_buf, mid_buf, out_pack2d_sample,
                                              base_texels_buf, pack2d_width, max_k, rows, groups, slots, rounds);
            }
        }
        if (buffer_ms == 0.0) buffer_ms = run_buffer(queue, buffer_pipeline, codebook_buf, code_buf, mid_buf, out_buffer, max_k, rows, groups, slots, rounds);
        if (texture_ms == 0.0) texture_ms = run_texture(queue, texture_pipeline, texture, code_buf, mid_buf, out_texture, max_k, rows, groups, slots, rounds);
        if (texture_buffer_ms == 0.0) texture_buffer_ms = run_texture(queue, texture_buffer_pipeline, texture_buffer, code_buf, mid_buf, out_texture_buffer, max_k, rows, groups, slots, rounds);
        if (texture_sample_ms == 0.0) texture_sample_ms = run_texture(queue, texture_sample_pipeline, texture, code_buf, mid_buf, out_texture_sample, max_k, rows, groups, slots, rounds);
        if (texture_gather_ms == 0.0) texture_gather_ms = run_texture(queue, texture_gather_pipeline, texture_gather, code_buf, mid_buf, out_texture_gather, max_k, rows, groups, slots, rounds);
        if (pack2d_available && isnan(pack2d_warm_ms)) {
            pack2d_warm_ms = run_warm_pack2d(queue, warm_pack2d_pipeline, pack2d_texture, base_texels_buf, warm_pack2d_sink,
                                             pack2d_width, max_k, slots, rounds);
        }
        if (pack2d_available && isnan(pack2d_read_ms)) {
            pack2d_read_ms = run_pack2d(queue, pack2d_read_pipeline, pack2d_texture, code_buf, mid_buf, out_pack2d_read,
                                        base_texels_buf, pack2d_width, max_k, rows, groups, slots, rounds);
        }
        if (pack2d_available && isnan(pack2d_sample_ms)) {
            pack2d_sample_ms = run_pack2d(queue, pack2d_sample_pipeline, pack2d_texture, code_buf, mid_buf, out_pack2d_sample,
                                          base_texels_buf, pack2d_width, max_k, rows, groups, slots, rounds);
        }
        printf("real_texture_selected file=%s experts=%s rows=%u slots=%u max_k=%u rounds=%u "
               "order=%s pack2d=%d pack2d_width=%u pack2d_height=%lu pack2d_view=%.3fMiB "
               "padded_codebook=%.3fMiB gather_codebook=%.3fMiB codes=%.3fMiB buffer=%.4fms tex_linear2d=%.4fms tex_buffer=%.4fms tex_sample=%.4fms tex_gather=%.4fms "
               "tex_pack2d_warm=%.4fms "
               "tex_pack2d_read=%.4fms tex_pack2d_sample=%.4fms "
               "speedup_2d=%.3fx speedup_tb=%.3fx speedup_sample=%.3fx speedup_gather=%.3fx speedup_pack2d_read=%.3fx speedup_pack2d_sample=%.3fx "
               "max_abs_buffer=%.6g max_abs_texture=%.6g max_abs_texture_buffer=%.6g max_abs_texture_sample=%.6g max_abs_texture_gather=%.6g "
               "max_abs_pack2d_read=%.6g max_abs_pack2d_sample=%.6g "
               "max_abs_buf_tex=%.6g max_abs_buf_tb=%.6g max_abs_buf_sample=%.6g max_abs_buf_gather=%.6g max_abs_buf_pack2d_read=%.6g max_abs_buf_pack2d_sample=%.6g\n",
               path, argv[2], rows, slots, max_k, rounds, measure_order,
               pack2d_available, pack2d_width, (unsigned long)pack2d_height,
               (double)pack2d_view_bytes / 1048576.0,
               (double)codebook_bytes / 1048576.0,
               (double)gather_texel_bytes / 1048576.0,
               (double)code_bytes / 1048576.0,
               buffer_ms, texture_ms, texture_buffer_ms, texture_sample_ms, texture_gather_ms,
               pack2d_warm_ms,
               pack2d_read_ms, pack2d_sample_ms,
               buffer_ms / texture_ms, buffer_ms / texture_buffer_ms, buffer_ms / texture_sample_ms,
               buffer_ms / texture_gather_ms,
               buffer_ms / pack2d_read_ms, buffer_ms / pack2d_sample_ms,
               max_abs_buffer, max_abs_texture, max_abs_texture_buffer, max_abs_texture_sample, max_abs_texture_gather,
               max_abs_pack2d_read, max_abs_pack2d_sample,
               max_abs_buffer_texture, max_abs_buffer_texture_buffer, max_abs_buffer_texture_sample, max_abs_buffer_texture_gather,
               max_abs_buffer_pack2d_read, max_abs_buffer_pack2d_sample);
        free(codebook_host); free(texel_host); free(gather_texel_host); free(code_host); free(mid_host); free(ref);
        ds4_d8f_close(&file);
    }
    return 0;
}
