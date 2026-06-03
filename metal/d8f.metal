// Generated from ds4_metal.m D8F embedded strings; edit source strings or regenerate.

#include <metal_stdlib>
using namespace metal;
static uint d8f_u32(device const uchar *p) {
  return uint(p[0]) | (uint(p[1]) << 8u) | (uint(p[2]) << 16u) | (uint(p[3]) << 24u);
}
static ulong d8f_u64(device const uchar *p) {
  return ulong(d8f_u32(p)) | (ulong(d8f_u32(p + 4)) << 32ul);
}

struct D8FGateUpArgs { uint rows; uint in_dim; uint n_selected; uint table_offset; uint record_bytes; uint mid_slot_stride; float clamp; };
kernel void d8f_gateup_swiglu_selected(
  device const uchar *pack     [[buffer(0)]],
  device const float *x        [[buffer(1)]],
  device const uint  *selected [[buffer(2)]],
  device float       *mid      [[buffer(3)]],
  constant D8FGateUpArgs &args [[buffer(4)]],
  threadgroup float *partial   [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  uint2 pos [[threadgroup_position_in_grid]]) {
  const uint row = pos.x;
  const uint slot = pos.y;
  if (row >= args.rows || slot >= args.n_selected) return;
  const uint expert = selected[slot];
  const uint blocks_per_row = args.in_dim >> 3;
  device const uchar *gate_rec = pack + args.table_offset + (expert) * args.record_bytes;
  device const uchar *up_rec = pack + args.table_offset + (256u + expert) * args.record_bytes;
  const uint gate_k = d8f_u32(gate_rec + 8);
  const uint gate_bits = d8f_u32(gate_rec + 12);
  const ulong gate_cb_off = d8f_u64(gate_rec + 24);
  const ulong gate_ix_off = d8f_u64(gate_rec + 32);
  const uint up_k = d8f_u32(up_rec + 8);
  const uint up_bits = d8f_u32(up_rec + 12);
  const ulong up_cb_off = d8f_u64(up_rec + 24);
  const ulong up_ix_off = d8f_u64(up_rec + 32);
  if (gate_k == 0u || up_k == 0u || gate_bits == 0u || up_bits == 0u) return;
  const uint gate_mask = (1u << gate_bits) - 1u;
  const uint up_mask = (1u << up_bits) - 1u;
  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  for (uint block_col = tid; block_col < blocks_per_row; block_col += 256u) {
    const ulong block_index = ulong(row) * ulong(blocks_per_row) + ulong(block_col);
    const uint x_base = block_col << 3;
    const float x0 = x[x_base + 0u];
    const float x1 = x[x_base + 1u];
    const float x2 = x[x_base + 2u];
    const float x3 = x[x_base + 3u];
    const float x4 = x[x_base + 4u];
    const float x5 = x[x_base + 5u];
    const float x6 = x[x_base + 6u];
    const float x7 = x[x_base + 7u];
    const ulong gate_bit_off = block_index * ulong(gate_bits);
    const ulong gate_byte_off = gate_bit_off >> 3;
    const uint gate_shift = uint(gate_bit_off & 7ul);
    device const uchar *gate_ix = pack + gate_ix_off + gate_byte_off;
    const uint gate_w = uint(gate_ix[0]) | (uint(gate_ix[1]) << 8u) | (uint(gate_ix[2]) << 16u) | (uint(gate_ix[3]) << 24u);
    const uint gate_code = (gate_w >> gate_shift) & gate_mask;
    if (gate_code < gate_k) {
      const device half *cb = (const device half *)(pack + gate_cb_off + ulong(gate_code) * 16ul);
      gate_acc += float(cb[0]) * x0 + float(cb[1]) * x1 + float(cb[2]) * x2 + float(cb[3]) * x3 +
                  float(cb[4]) * x4 + float(cb[5]) * x5 + float(cb[6]) * x6 + float(cb[7]) * x7;
    }
    const ulong up_bit_off = block_index * ulong(up_bits);
    const ulong up_byte_off = up_bit_off >> 3;
    const uint up_shift = uint(up_bit_off & 7ul);
    device const uchar *up_ix = pack + up_ix_off + up_byte_off;
    const uint up_w = uint(up_ix[0]) | (uint(up_ix[1]) << 8u) | (uint(up_ix[2]) << 16u) | (uint(up_ix[3]) << 24u);
    const uint up_code = (up_w >> up_shift) & up_mask;
    if (up_code < up_k) {
      const device half *cb = (const device half *)(pack + up_cb_off + ulong(up_code) * 16ul);
      up_acc += float(cb[0]) * x0 + float(cb[1]) * x1 + float(cb[2]) * x2 + float(cb[3]) * x3 +
                float(cb[4]) * x4 + float(cb[5]) * x5 + float(cb[6]) * x6 + float(cb[7]) * x7;
    }
  }
  gate_acc = simd_sum(gate_acc);
  up_acc = simd_sum(up_acc);
  if (tiisg == 0u) {
    partial[sgitg] = gate_acc;
    partial[8u + sgitg] = up_acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float gate_total = 0.0f;
    float up_total = 0.0f;
    for (uint sg = 0u; sg < 8u; sg++) {
      gate_total += partial[sg];
      up_total += partial[8u + sg];
    }
    gate_total = min(gate_total, args.clamp);
    up_total = clamp(up_total, -args.clamp, args.clamp);
    mid[ulong(slot) * ulong(args.mid_slot_stride) + ulong(row)] = (gate_total / (1.0f + exp(-gate_total))) * up_total;
  }
}
struct D8FGateUpBatchArgs { uint rows; uint in_dim; uint n_selected; uint n_tokens; uint table_offset; uint record_bytes; uint mid_slot_stride; uint mid_token_stride; uint x_token_stride; uint selected_token_stride; float clamp; };
kernel void d8f_gateup_swiglu_selected_batch(
  device const uchar *pack          [[buffer(0)]],
  device const float *x_base_all    [[buffer(1)]],
  device const uint  *selected      [[buffer(2)]],
  device float       *mid           [[buffer(3)]],
  constant D8FGateUpBatchArgs &args [[buffer(4)]],
  threadgroup float *partial        [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  uint3 pos [[threadgroup_position_in_grid]]) {
  const uint row = pos.x;
  const uint slot = pos.y;
  const uint token = pos.z;
  if (row >= args.rows || slot >= args.n_selected || token >= args.n_tokens) return;
  device const float *x = x_base_all + ulong(token) * ulong(args.x_token_stride);
  const uint expert = selected[ulong(token) * ulong(args.selected_token_stride) + ulong(slot)];
  const uint blocks_per_row = args.in_dim >> 3;
  device const uchar *gate_rec = pack + args.table_offset + (expert) * args.record_bytes;
  device const uchar *up_rec = pack + args.table_offset + (256u + expert) * args.record_bytes;
  const uint gate_k = d8f_u32(gate_rec + 8);
  const uint gate_bits = d8f_u32(gate_rec + 12);
  const ulong gate_cb_off = d8f_u64(gate_rec + 24);
  const ulong gate_ix_off = d8f_u64(gate_rec + 32);
  const uint up_k = d8f_u32(up_rec + 8);
  const uint up_bits = d8f_u32(up_rec + 12);
  const ulong up_cb_off = d8f_u64(up_rec + 24);
  const ulong up_ix_off = d8f_u64(up_rec + 32);
  if (gate_k == 0u || up_k == 0u || gate_bits == 0u || up_bits == 0u) return;
  const uint gate_mask = (1u << gate_bits) - 1u;
  const uint up_mask = (1u << up_bits) - 1u;
  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  for (uint block_col = tid; block_col < blocks_per_row; block_col += 256u) {
    const ulong block_index = ulong(row) * ulong(blocks_per_row) + ulong(block_col);
    const uint x_base = block_col << 3;
    const float x0 = x[x_base + 0u];
    const float x1 = x[x_base + 1u];
    const float x2 = x[x_base + 2u];
    const float x3 = x[x_base + 3u];
    const float x4 = x[x_base + 4u];
    const float x5 = x[x_base + 5u];
    const float x6 = x[x_base + 6u];
    const float x7 = x[x_base + 7u];
    const ulong gate_bit_off = block_index * ulong(gate_bits);
    const ulong gate_byte_off = gate_bit_off >> 3;
    const uint gate_shift = uint(gate_bit_off & 7ul);
    device const uchar *gate_ix = pack + gate_ix_off + gate_byte_off;
    const uint gate_w = uint(gate_ix[0]) | (uint(gate_ix[1]) << 8u) | (uint(gate_ix[2]) << 16u) | (uint(gate_ix[3]) << 24u);
    const uint gate_code = (gate_w >> gate_shift) & gate_mask;
    if (gate_code < gate_k) {
      const device half *cb = (const device half *)(pack + gate_cb_off + ulong(gate_code) * 16ul);
      gate_acc += float(cb[0]) * x0 + float(cb[1]) * x1 + float(cb[2]) * x2 + float(cb[3]) * x3 +
                  float(cb[4]) * x4 + float(cb[5]) * x5 + float(cb[6]) * x6 + float(cb[7]) * x7;
    }
    const ulong up_bit_off = block_index * ulong(up_bits);
    const ulong up_byte_off = up_bit_off >> 3;
    const uint up_shift = uint(up_bit_off & 7ul);
    device const uchar *up_ix = pack + up_ix_off + up_byte_off;
    const uint up_w = uint(up_ix[0]) | (uint(up_ix[1]) << 8u) | (uint(up_ix[2]) << 16u) | (uint(up_ix[3]) << 24u);
    const uint up_code = (up_w >> up_shift) & up_mask;
    if (up_code < up_k) {
      const device half *cb = (const device half *)(pack + up_cb_off + ulong(up_code) * 16ul);
      up_acc += float(cb[0]) * x0 + float(cb[1]) * x1 + float(cb[2]) * x2 + float(cb[3]) * x3 +
                float(cb[4]) * x4 + float(cb[5]) * x5 + float(cb[6]) * x6 + float(cb[7]) * x7;
    }
  }
  gate_acc = simd_sum(gate_acc);
  up_acc = simd_sum(up_acc);
  if (tiisg == 0u) { partial[sgitg] = gate_acc; partial[8u + sgitg] = up_acc; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float gate_total = 0.0f;
    float up_total = 0.0f;
    for (uint sg = 0u; sg < 8u; sg++) { gate_total += partial[sg]; up_total += partial[8u + sg]; }
    gate_total = min(gate_total, args.clamp);
    up_total = clamp(up_total, -args.clamp, args.clamp);
    mid[ulong(token) * ulong(args.mid_token_stride) + ulong(slot) * ulong(args.mid_slot_stride) + ulong(row)] = (gate_total / (1.0f + exp(-gate_total))) * up_total;
  }
}
kernel void d8f_gateup_swiglu_selected_batch_tile4(
  device const uchar *pack          [[buffer(0)]],
  device const float *x_base_all    [[buffer(1)]],
  device const uint  *selected      [[buffer(2)]],
  device float       *mid           [[buffer(3)]],
  constant D8FGateUpBatchArgs &args [[buffer(4)]],
  threadgroup float *partial        [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  uint3 pos [[threadgroup_position_in_grid]]) {
  const uint row_base = pos.x << 2;
  const uint slot = pos.y;
  const uint token = pos.z;
  if (row_base >= args.rows || slot >= args.n_selected || token >= args.n_tokens) return;
  device const float *x = x_base_all + ulong(token) * ulong(args.x_token_stride);
  const uint expert = selected[ulong(token) * ulong(args.selected_token_stride) + ulong(slot)];
  const uint blocks_per_row = args.in_dim >> 3;
  device const uchar *gate_rec = pack + args.table_offset + (expert) * args.record_bytes;
  device const uchar *up_rec = pack + args.table_offset + (256u + expert) * args.record_bytes;
  const uint gate_k = d8f_u32(gate_rec + 8);
  const uint gate_bits = d8f_u32(gate_rec + 12);
  const ulong gate_cb_off = d8f_u64(gate_rec + 24);
  const ulong gate_ix_off = d8f_u64(gate_rec + 32);
  const uint up_k = d8f_u32(up_rec + 8);
  const uint up_bits = d8f_u32(up_rec + 12);
  const ulong up_cb_off = d8f_u64(up_rec + 24);
  const ulong up_ix_off = d8f_u64(up_rec + 32);
  if (gate_k == 0u || up_k == 0u || gate_bits == 0u || up_bits == 0u) return;
  const uint gate_mask = (1u << gate_bits) - 1u;
  const uint up_mask = (1u << up_bits) - 1u;
  float gate_acc[4];
  float up_acc[4];
  for (uint rr = 0u; rr < 4u; rr++) { gate_acc[rr] = 0.0f; up_acc[rr] = 0.0f; }
  for (uint block_col = tid; block_col < blocks_per_row; block_col += 256u) {
    const uint x_base = block_col << 3;
    const float x0 = x[x_base + 0u];
    const float x1 = x[x_base + 1u];
    const float x2 = x[x_base + 2u];
    const float x3 = x[x_base + 3u];
    const float x4 = x[x_base + 4u];
    const float x5 = x[x_base + 5u];
    const float x6 = x[x_base + 6u];
    const float x7 = x[x_base + 7u];
    for (uint rr = 0u; rr < 4u; rr++) {
      const uint row = row_base + rr;
      if (row >= args.rows) continue;
      const ulong block_index = ulong(row) * ulong(blocks_per_row) + ulong(block_col);
      const ulong gate_bit_off = block_index * ulong(gate_bits);
      const ulong gate_byte_off = gate_bit_off >> 3;
      const uint gate_shift = uint(gate_bit_off & 7ul);
      device const uchar *gate_ix = pack + gate_ix_off + gate_byte_off;
      const uint gate_w = uint(gate_ix[0]) | (uint(gate_ix[1]) << 8u) | (uint(gate_ix[2]) << 16u) | (uint(gate_ix[3]) << 24u);
      const uint gate_code = (gate_w >> gate_shift) & gate_mask;
      if (gate_code < gate_k) {
        const device half *cb = (const device half *)(pack + gate_cb_off + ulong(gate_code) * 16ul);
        gate_acc[rr] += float(cb[0]) * x0 + float(cb[1]) * x1 + float(cb[2]) * x2 + float(cb[3]) * x3 +
                        float(cb[4]) * x4 + float(cb[5]) * x5 + float(cb[6]) * x6 + float(cb[7]) * x7;
      }
      const ulong up_bit_off = block_index * ulong(up_bits);
      const ulong up_byte_off = up_bit_off >> 3;
      const uint up_shift = uint(up_bit_off & 7ul);
      device const uchar *up_ix = pack + up_ix_off + up_byte_off;
      const uint up_w = uint(up_ix[0]) | (uint(up_ix[1]) << 8u) | (uint(up_ix[2]) << 16u) | (uint(up_ix[3]) << 24u);
      const uint up_code = (up_w >> up_shift) & up_mask;
      if (up_code < up_k) {
        const device half *cb = (const device half *)(pack + up_cb_off + ulong(up_code) * 16ul);
        up_acc[rr] += float(cb[0]) * x0 + float(cb[1]) * x1 + float(cb[2]) * x2 + float(cb[3]) * x3 +
                      float(cb[4]) * x4 + float(cb[5]) * x5 + float(cb[6]) * x6 + float(cb[7]) * x7;
      }
    }
  }
  for (uint rr = 0u; rr < 4u; rr++) {
    gate_acc[rr] = simd_sum(gate_acc[rr]);
    up_acc[rr] = simd_sum(up_acc[rr]);
    if (tiisg == 0u) { partial[rr * 16u + sgitg] = gate_acc[rr]; partial[rr * 16u + 8u + sgitg] = up_acc[rr]; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    for (uint rr = 0u; rr < 4u; rr++) {
      const uint row = row_base + rr;
      if (row >= args.rows) continue;
      float gate_total = 0.0f;
      float up_total = 0.0f;
      for (uint sg = 0u; sg < 8u; sg++) { gate_total += partial[rr * 16u + sg]; up_total += partial[rr * 16u + 8u + sg]; }
      gate_total = min(gate_total, args.clamp);
      up_total = clamp(up_total, -args.clamp, args.clamp);
      mid[ulong(token) * ulong(args.mid_token_stride) + ulong(slot) * ulong(args.mid_slot_stride) + ulong(row)] = (gate_total / (1.0f + exp(-gate_total))) * up_total;
    }
  }
}
kernel void d8f_gateup_swiglu_selected_batch_weighted(
  device const uchar *pack          [[buffer(0)]],
  device const float *x_base_all    [[buffer(1)]],
  device const uint  *selected      [[buffer(2)]],
  device float       *mid           [[buffer(3)]],
  constant D8FGateUpBatchArgs &args [[buffer(4)]],
  device const float *route_weights [[buffer(5)]],
  threadgroup float *partial        [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  uint3 pos [[threadgroup_position_in_grid]]) {
  const uint row = pos.x;
  const uint slot = pos.y;
  const uint token = pos.z;
  if (row >= args.rows || slot >= args.n_selected || token >= args.n_tokens) return;
  device const float *x = x_base_all + ulong(token) * ulong(args.x_token_stride);
  const uint expert = selected[ulong(token) * ulong(args.selected_token_stride) + ulong(slot)];
  const float route_weight = route_weights[ulong(token) * ulong(args.selected_token_stride) + ulong(slot)];
  const uint blocks_per_row = args.in_dim >> 3;
  device const uchar *gate_rec = pack + args.table_offset + (expert) * args.record_bytes;
  device const uchar *up_rec = pack + args.table_offset + (256u + expert) * args.record_bytes;
  const uint gate_k = d8f_u32(gate_rec + 8);
  const uint gate_bits = d8f_u32(gate_rec + 12);
  const ulong gate_cb_off = d8f_u64(gate_rec + 24);
  const ulong gate_ix_off = d8f_u64(gate_rec + 32);
  const uint up_k = d8f_u32(up_rec + 8);
  const uint up_bits = d8f_u32(up_rec + 12);
  const ulong up_cb_off = d8f_u64(up_rec + 24);
  const ulong up_ix_off = d8f_u64(up_rec + 32);
  if (route_weight == 0.0f || gate_k == 0u || up_k == 0u || gate_bits == 0u || up_bits == 0u) { mid[ulong(token) * ulong(args.mid_token_stride) + ulong(slot) * ulong(args.mid_slot_stride) + ulong(row)] = 0.0f; return; }
  const uint gate_mask = (1u << gate_bits) - 1u;
  const uint up_mask = (1u << up_bits) - 1u;
  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  for (uint block_col = tid; block_col < blocks_per_row; block_col += 256u) {
    const ulong block_index = ulong(row) * ulong(blocks_per_row) + ulong(block_col);
    const uint x_base = block_col << 3;
    const float x0 = x[x_base + 0u]; const float x1 = x[x_base + 1u]; const float x2 = x[x_base + 2u]; const float x3 = x[x_base + 3u];
    const float x4 = x[x_base + 4u]; const float x5 = x[x_base + 5u]; const float x6 = x[x_base + 6u]; const float x7 = x[x_base + 7u];
    const ulong gate_bit_off = block_index * ulong(gate_bits); const ulong gate_byte_off = gate_bit_off >> 3; const uint gate_shift = uint(gate_bit_off & 7ul);
    device const uchar *gate_ix = pack + gate_ix_off + gate_byte_off;
    const uint gate_w = uint(gate_ix[0]) | (uint(gate_ix[1]) << 8u) | (uint(gate_ix[2]) << 16u) | (uint(gate_ix[3]) << 24u);
    const uint gate_code = (gate_w >> gate_shift) & gate_mask;
    if (gate_code < gate_k) { const device half *cb = (const device half *)(pack + gate_cb_off + ulong(gate_code) * 16ul); gate_acc += float(cb[0]) * x0 + float(cb[1]) * x1 + float(cb[2]) * x2 + float(cb[3]) * x3 + float(cb[4]) * x4 + float(cb[5]) * x5 + float(cb[6]) * x6 + float(cb[7]) * x7; }
    const ulong up_bit_off = block_index * ulong(up_bits); const ulong up_byte_off = up_bit_off >> 3; const uint up_shift = uint(up_bit_off & 7ul);
    device const uchar *up_ix = pack + up_ix_off + up_byte_off;
    const uint up_w = uint(up_ix[0]) | (uint(up_ix[1]) << 8u) | (uint(up_ix[2]) << 16u) | (uint(up_ix[3]) << 24u);
    const uint up_code = (up_w >> up_shift) & up_mask;
    if (up_code < up_k) { const device half *cb = (const device half *)(pack + up_cb_off + ulong(up_code) * 16ul); up_acc += float(cb[0]) * x0 + float(cb[1]) * x1 + float(cb[2]) * x2 + float(cb[3]) * x3 + float(cb[4]) * x4 + float(cb[5]) * x5 + float(cb[6]) * x6 + float(cb[7]) * x7; }
  }
  gate_acc = simd_sum(gate_acc); up_acc = simd_sum(up_acc);
  if (tiisg == 0u) { partial[sgitg] = gate_acc; partial[8u + sgitg] = up_acc; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) { float gate_total = 0.0f; float up_total = 0.0f; for (uint sg = 0u; sg < 8u; sg++) { gate_total += partial[sg]; up_total += partial[8u + sg]; } gate_total = min(gate_total, args.clamp); up_total = clamp(up_total, -args.clamp, args.clamp); mid[ulong(token) * ulong(args.mid_token_stride) + ulong(slot) * ulong(args.mid_slot_stride) + ulong(row)] = route_weight * (gate_total / (1.0f + exp(-gate_total))) * up_total; }
}


static ulong d8f_scale_offset(device const uchar *rec) {
  const uint flags = d8f_u32(rec + 60);
  if ((flags & 1u) == 0u) return 0ul;
  return d8f_u64(rec + 40);
}
static float d8f_mid_value(device const uchar *pack, ulong scale_off, device const float *slot_mid, uint idx) {
  if (scale_off == 0ul) return slot_mid[idx];
  const device half *inv_scale = (const device half *)(pack + scale_off);
  return slot_mid[idx] * float(inv_scale[idx]);
}
static float d8f_mid_value_range(device const uchar *pack, ulong scale_off, device const float *slot_mid, uint local_idx, uint global_idx) {
  if (scale_off == 0ul) return slot_mid[local_idx];
  const device half *inv_scale = (const device half *)(pack + scale_off);
  return slot_mid[local_idx] * float(inv_scale[global_idx]);
}
static float d8f_mid_value_h(device const uchar *pack, ulong scale_off, device const half *slot_mid, uint idx) {
  const float v = float(slot_mid[idx]);
  if (scale_off == 0ul) return v;
  const device half *inv_scale = (const device half *)(pack + scale_off);
  return v * float(inv_scale[idx]);
}
struct D8FDownArgs { uint rows; uint in_dim; uint n_selected; uint table_offset; uint record_bytes; uint mid_slot_stride; };
kernel void d8f_down_sum_selected(
  device const uchar *pack     [[buffer(0)]],
  device const float *mid      [[buffer(1)]],
  device const uint  *selected [[buffer(2)]],
  device float       *out      [[buffer(3)]],
  constant D8FDownArgs &args   [[buffer(4)]],
  threadgroup float *partial   [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  uint row [[threadgroup_position_in_grid]]) {
  if (row >= args.rows) return;
  const uint blocks_per_row = args.in_dim >> 3;
  float acc = 0.0f;
  for (uint slot = 0u; slot < args.n_selected; slot++) {
    const uint expert = selected[slot];
    device const uchar *rec = pack + args.table_offset + (512u + expert) * args.record_bytes;
    const uint k = d8f_u32(rec + 8);
    const uint bits = d8f_u32(rec + 12);
    const ulong cb_off = d8f_u64(rec + 24);
    const ulong ix_off = d8f_u64(rec + 32);
    const ulong scale_off = d8f_scale_offset(rec);
    if (k == 0u || bits == 0u) continue;
    const uint mask = (1u << bits) - 1u;
    const device float *slot_mid = mid + ulong(slot) * ulong(args.mid_slot_stride);
    for (uint block_col = tid; block_col < blocks_per_row; block_col += 256u) {
      const ulong block_index = ulong(row) * ulong(blocks_per_row) + ulong(block_col);
      const ulong bit_off = block_index * ulong(bits);
      const ulong byte_off = bit_off >> 3;
      const uint shift = uint(bit_off & 7ul);
      device const uchar *ix = pack + ix_off + byte_off;
      const uint w = uint(ix[0]) | (uint(ix[1]) << 8u) | (uint(ix[2]) << 16u) | (uint(ix[3]) << 24u);
      const uint code = (w >> shift) & mask;
      if (code >= k) continue;
      const device half *cb = (const device half *)(pack + cb_off + ulong(code) * 16ul);
      const uint x_base = block_col << 3;
      acc += float(cb[0]) * d8f_mid_value(pack, scale_off, slot_mid, x_base + 0u) +
             float(cb[1]) * d8f_mid_value(pack, scale_off, slot_mid, x_base + 1u) +
             float(cb[2]) * d8f_mid_value(pack, scale_off, slot_mid, x_base + 2u) +
             float(cb[3]) * d8f_mid_value(pack, scale_off, slot_mid, x_base + 3u) +
             float(cb[4]) * d8f_mid_value(pack, scale_off, slot_mid, x_base + 4u) +
             float(cb[5]) * d8f_mid_value(pack, scale_off, slot_mid, x_base + 5u) +
             float(cb[6]) * d8f_mid_value(pack, scale_off, slot_mid, x_base + 6u) +
             float(cb[7]) * d8f_mid_value(pack, scale_off, slot_mid, x_base + 7u);
    }
  }
  acc = simd_sum(acc);
  if (tiisg == 0u) partial[sgitg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float total = 0.0f;
    for (uint sg = 0u; sg < 8u; sg++) total += partial[sg];
    out[row] = total;
  }
}
kernel void d8f_down_sum_selected_weighted(
  device const uchar *pack          [[buffer(0)]],
  device const float *mid           [[buffer(1)]],
  device const uint  *selected      [[buffer(2)]],
  device float       *out           [[buffer(3)]],
  constant D8FDownArgs &args        [[buffer(4)]],
  device const float *route_weights [[buffer(5)]],
  threadgroup float *partial        [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  uint row [[threadgroup_position_in_grid]]) {
  if (row >= args.rows) return;
  const uint blocks_per_row = args.in_dim >> 3;
  float acc = 0.0f;
  for (uint slot = 0u; slot < args.n_selected; slot++) {
    const float rw = route_weights[slot];
    const uint expert = selected[slot];
    device const uchar *rec = pack + args.table_offset + (512u + expert) * args.record_bytes;
    const uint k = d8f_u32(rec + 8);
    const uint bits = d8f_u32(rec + 12);
    const ulong cb_off = d8f_u64(rec + 24);
    const ulong ix_off = d8f_u64(rec + 32);
    const ulong scale_off = d8f_scale_offset(rec);
    if (rw == 0.0f || k == 0u || bits == 0u) continue;
    const uint mask = (1u << bits) - 1u;
    const device float *slot_mid = mid + ulong(slot) * ulong(args.mid_slot_stride);
    for (uint block_col = tid; block_col < blocks_per_row; block_col += 256u) {
      const ulong block_index = ulong(row) * ulong(blocks_per_row) + ulong(block_col);
      const ulong bit_off = block_index * ulong(bits);
      const ulong byte_off = bit_off >> 3;
      const uint shift = uint(bit_off & 7ul);
      device const uchar *ix = pack + ix_off + byte_off;
      const uint w = uint(ix[0]) | (uint(ix[1]) << 8u) | (uint(ix[2]) << 16u) | (uint(ix[3]) << 24u);
      const uint code = (w >> shift) & mask;
      if (code >= k) continue;
      const device half *cb = (const device half *)(pack + cb_off + ulong(code) * 16ul);
      const uint x_base = block_col << 3;
      const float block_acc = float(cb[0]) * d8f_mid_value(pack, scale_off, slot_mid, x_base + 0u) +
                              float(cb[1]) * d8f_mid_value(pack, scale_off, slot_mid, x_base + 1u) +
                              float(cb[2]) * d8f_mid_value(pack, scale_off, slot_mid, x_base + 2u) +
                              float(cb[3]) * d8f_mid_value(pack, scale_off, slot_mid, x_base + 3u) +
                              float(cb[4]) * d8f_mid_value(pack, scale_off, slot_mid, x_base + 4u) +
                              float(cb[5]) * d8f_mid_value(pack, scale_off, slot_mid, x_base + 5u) +
                              float(cb[6]) * d8f_mid_value(pack, scale_off, slot_mid, x_base + 6u) +
                              float(cb[7]) * d8f_mid_value(pack, scale_off, slot_mid, x_base + 7u);
      acc += rw * block_acc;
    }
  }
  acc = simd_sum(acc);
  if (tiisg == 0u) partial[sgitg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float total = 0.0f;
    for (uint sg = 0u; sg < 8u; sg++) total += partial[sg];
    out[row] = total;
  }
}
struct D8FDownBatchArgs {
  uint rows;
  uint in_dim;
  uint n_selected;
  uint n_tokens;
  uint table_offset;
  uint record_bytes;
  uint mid_slot_stride;
  uint mid_token_stride;
  uint out_token_stride;
  uint route_token_stride;
  uint selected_token_stride;
  uint sidecar_table_offset_lo;
  uint sidecar_table_offset_hi;
  uint sidecar_record_bytes;
  uint sidecar_rank_max;
  uint sidecar_dot_stride;
  uint cbsram_k_cap;
};

inline ulong d8f_sidecar_table_offset(constant D8FDownBatchArgs &args) {
  return ulong(args.sidecar_table_offset_lo) | (ulong(args.sidecar_table_offset_hi) << 32);
}

inline device const uchar *d8f_down_sidecar_record(device const uchar *pack,
                                                   constant D8FDownBatchArgs &args,
                                                   uint expert) {
  const ulong table = d8f_sidecar_table_offset(args);
  if (table == 0ul || args.sidecar_record_bytes == 0u || expert >= 256u) return nullptr;
  return pack + table + ulong(expert) * ulong(args.sidecar_record_bytes);
}

kernel void d8f_down_sum_selected_weighted_batch(
  device const uchar *pack               [[buffer(0)]],
  device const float *mid                [[buffer(1)]],
  device const uint  *selected           [[buffer(2)]],
  device float       *out                [[buffer(3)]],
  constant D8FDownBatchArgs &args        [[buffer(4)]],
  device const float *route_weights      [[buffer(5)]],
  threadgroup float *partial             [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  uint2 pos [[threadgroup_position_in_grid]]) {
  const uint row = pos.x;
  const uint token = pos.y;
  if (row >= args.rows || token >= args.n_tokens) return;
  const uint blocks_per_row = args.in_dim >> 3;
  float acc = 0.0f;
  device const float *token_mid = mid + ulong(token) * ulong(args.mid_token_stride);
  device const float *token_weights = route_weights + ulong(token) * ulong(args.route_token_stride);
  for (uint slot = 0u; slot < args.n_selected; slot++) {
    const float rw = token_weights[slot];
    const uint expert = selected[ulong(token) * ulong(args.selected_token_stride) + ulong(slot)];
    device const uchar *rec = pack + args.table_offset + (512u + expert) * args.record_bytes;
    const uint k = d8f_u32(rec + 8);
    const uint bits = d8f_u32(rec + 12);
    const ulong cb_off = d8f_u64(rec + 24);
    const ulong ix_off = d8f_u64(rec + 32);
    const ulong scale_off = d8f_scale_offset(rec);
    if (rw == 0.0f || k == 0u || bits == 0u) continue;
    const uint mask = (1u << bits) - 1u;
    const device float *slot_mid = token_mid + ulong(slot) * ulong(args.mid_slot_stride);
    for (uint block_col = tid; block_col < blocks_per_row; block_col += 256u) {
      const ulong block_index = ulong(row) * ulong(blocks_per_row) + ulong(block_col);
      const ulong bit_off = block_index * ulong(bits);
      const ulong byte_off = bit_off >> 3;
      const uint shift = uint(bit_off & 7ul);
      device const uchar *ix = pack + ix_off + byte_off;
      const uint w = uint(ix[0]) | (uint(ix[1]) << 8u) | (uint(ix[2]) << 16u) | (uint(ix[3]) << 24u);
      const uint code = (w >> shift) & mask;
      if (code >= k) continue;
      const device half *cb = (const device half *)(pack + cb_off + ulong(code) * 16ul);
      const uint x_base = block_col << 3;
      const float block_acc = float(cb[0]) * d8f_mid_value(pack, scale_off, slot_mid, x_base + 0u) +
                              float(cb[1]) * d8f_mid_value(pack, scale_off, slot_mid, x_base + 1u) +
                              float(cb[2]) * d8f_mid_value(pack, scale_off, slot_mid, x_base + 2u) +
                              float(cb[3]) * d8f_mid_value(pack, scale_off, slot_mid, x_base + 3u) +
                              float(cb[4]) * d8f_mid_value(pack, scale_off, slot_mid, x_base + 4u) +
                              float(cb[5]) * d8f_mid_value(pack, scale_off, slot_mid, x_base + 5u) +
                              float(cb[6]) * d8f_mid_value(pack, scale_off, slot_mid, x_base + 6u) +
                              float(cb[7]) * d8f_mid_value(pack, scale_off, slot_mid, x_base + 7u);
      acc += rw * block_acc;
    }
  }
  acc = simd_sum(acc);
  if (tiisg == 0u) partial[sgitg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float total = 0.0f;
    for (uint sg = 0u; sg < 8u; sg++) total += partial[sg];
    out[ulong(token) * ulong(args.out_token_stride) + ulong(row)] = total;
  }
}

kernel void d8f_down_rank1_dot_batch(
  device const uchar *pack               [[buffer(0)]],
  device const float *mid                [[buffer(1)]],
  device const uint  *selected           [[buffer(2)]],
  device float       *sidecar_dot        [[buffer(3)]],
  constant D8FDownBatchArgs &args        [[buffer(4)]],
  device const float *route_weights      [[buffer(5)]],
  threadgroup float *partial             [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  uint2 pos [[threadgroup_position_in_grid]]) {
  const uint slot = pos.x;
  const uint token = pos.y;
  if (slot >= args.n_selected || token >= args.n_tokens) return;
  device const float *token_weights = route_weights + ulong(token) * ulong(args.route_token_stride);
  const float rw = token_weights[slot];
  const ulong dot_index = ulong(token) * ulong(args.sidecar_dot_stride) + ulong(slot);
  if (rw == 0.0f || d8f_sidecar_table_offset(args) == 0ul) {
    if (tid == 0u) sidecar_dot[dot_index] = 0.0f;
    return;
  }
  const uint expert = selected[ulong(token) * ulong(args.selected_token_stride) + ulong(slot)];
  device const uchar *rec = d8f_down_sidecar_record(pack, args, expert);
  const uint rank = rec ? d8f_u32(rec + 4) : 0u;
  const uint in_dim = rec ? d8f_u32(rec + 8) : 0u;
  const ulong u_off = rec ? d8f_u64(rec + 16) : 0ul;
  device const uchar *down_rec = pack + ulong(args.table_offset) + ulong(512u + expert) * ulong(args.record_bytes);
  const ulong scale_off = d8f_u64(down_rec + 40);
  if (rank != 1u || in_dim != args.in_dim || u_off == 0ul) {
    if (tid == 0u) sidecar_dot[dot_index] = 0.0f;
    return;
  }
  device const half *u = (const device half *)(pack + u_off);
  device const float *slot_mid = mid + ulong(token) * ulong(args.mid_token_stride) +
                                 ulong(slot) * ulong(args.mid_slot_stride);
  float acc = 0.0f;
  for (uint i = tid; i < args.in_dim; i += 256u) {
    acc += float(u[i]) * d8f_mid_value(pack, scale_off, slot_mid, i);
  }
  acc = simd_sum(acc);
  if (tiisg == 0u) partial[sgitg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float total = 0.0f;
    for (uint sg = 0u; sg < 8u; sg++) total += partial[sg];
    sidecar_dot[dot_index] = total;
  }
}

kernel void d8f_down_rank1_axpy_batch(
  device const uchar *pack               [[buffer(0)]],
  device const uint  *selected           [[buffer(1)]],
  device float       *out                [[buffer(2)]],
  device const float *sidecar_dot        [[buffer(3)]],
  constant D8FDownBatchArgs &args        [[buffer(4)]],
  device const float *route_weights      [[buffer(5)]],
  uint tid [[thread_index_in_threadgroup]],
  uint2 pos [[threadgroup_position_in_grid]]) {
  const uint row = (pos.x << 8) + tid;
  const uint token = pos.y;
  if (row >= args.rows || token >= args.n_tokens || d8f_sidecar_table_offset(args) == 0ul) return;
  device const float *token_weights = route_weights + ulong(token) * ulong(args.route_token_stride);
  float delta = 0.0f;
  for (uint slot = 0u; slot < args.n_selected; slot++) {
    const float rw = token_weights[slot];
    if (rw == 0.0f) continue;
    const uint expert = selected[ulong(token) * ulong(args.selected_token_stride) + ulong(slot)];
    device const uchar *rec = d8f_down_sidecar_record(pack, args, expert);
    const uint rank = rec ? d8f_u32(rec + 4) : 0u;
    const uint out_dim = rec ? d8f_u32(rec + 12) : 0u;
    const ulong a_off = rec ? d8f_u64(rec + 24) : 0ul;
    if (rank != 1u || out_dim != args.rows || a_off == 0ul) continue;
    device const half *a = (const device half *)(pack + a_off);
    const float dot = sidecar_dot[ulong(token) * ulong(args.sidecar_dot_stride) + ulong(slot)];
    delta += rw * float(a[row]) * dot;
  }
  out[ulong(token) * ulong(args.out_token_stride) + ulong(row)] += delta;
}
kernel void d8f_down_sum_selected_weighted_batch_tile8(
  device const uchar *pack               [[buffer(0)]],
  device const float *mid                [[buffer(1)]],
  device const uint  *selected           [[buffer(2)]],
  device float       *out                [[buffer(3)]],
  constant D8FDownBatchArgs &args        [[buffer(4)]],
  device const float *route_weights      [[buffer(5)]],
  threadgroup float *partial             [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  uint2 pos [[threadgroup_position_in_grid]]) {
  const uint row_base = pos.x << 3;
  const uint token = pos.y;
  if (token >= args.n_tokens) return;
  const uint blocks_per_row = args.in_dim >> 3;
  device const float *token_mid = mid + ulong(token) * ulong(args.mid_token_stride);
  device const float *token_weights = route_weights + ulong(token) * ulong(args.route_token_stride);
  float acc[8];
  for (uint rr = 0u; rr < 8u; rr++) acc[rr] = 0.0f;
  for (uint slot = 0u; slot < args.n_selected; slot++) {
    const float rw = token_weights[slot];
    const uint expert = selected[ulong(token) * ulong(args.selected_token_stride) + ulong(slot)];
    device const uchar *rec = pack + args.table_offset + (512u + expert) * args.record_bytes;
    const uint k = d8f_u32(rec + 8);
    const uint bits = d8f_u32(rec + 12);
    const ulong cb_off = d8f_u64(rec + 24);
    const ulong ix_off = d8f_u64(rec + 32);
    const ulong scale_off = d8f_scale_offset(rec);
    if (rw == 0.0f || k == 0u || bits == 0u) continue;
    const uint mask = (1u << bits) - 1u;
    const device float *slot_mid = token_mid + ulong(slot) * ulong(args.mid_slot_stride);
    for (uint block_col = tid; block_col < blocks_per_row; block_col += 256u) {
      const uint x_base = block_col << 3;
      const float m0 = d8f_mid_value(pack, scale_off, slot_mid, x_base + 0u);
      const float m1 = d8f_mid_value(pack, scale_off, slot_mid, x_base + 1u);
      const float m2 = d8f_mid_value(pack, scale_off, slot_mid, x_base + 2u);
      const float m3 = d8f_mid_value(pack, scale_off, slot_mid, x_base + 3u);
      const float m4 = d8f_mid_value(pack, scale_off, slot_mid, x_base + 4u);
      const float m5 = d8f_mid_value(pack, scale_off, slot_mid, x_base + 5u);
      const float m6 = d8f_mid_value(pack, scale_off, slot_mid, x_base + 6u);
      const float m7 = d8f_mid_value(pack, scale_off, slot_mid, x_base + 7u);
      for (uint rr = 0u; rr < 8u; rr++) {
        const uint row = row_base + rr;
        if (row >= args.rows) continue;
        const ulong block_index = ulong(row) * ulong(blocks_per_row) + ulong(block_col);
        const ulong bit_off = block_index * ulong(bits);
        const ulong byte_off = bit_off >> 3;
        const uint shift = uint(bit_off & 7ul);
        device const uchar *ix = pack + ix_off + byte_off;
        const uint w = uint(ix[0]) | (uint(ix[1]) << 8u) | (uint(ix[2]) << 16u) | (uint(ix[3]) << 24u);
        const uint code = (w >> shift) & mask;
        if (code >= k) continue;
        const device half *cb = (const device half *)(pack + cb_off + ulong(code) * 16ul);
        acc[rr] += rw * (float(cb[0]) * m0 + float(cb[1]) * m1 +
                         float(cb[2]) * m2 + float(cb[3]) * m3 +
                         float(cb[4]) * m4 + float(cb[5]) * m5 +
                         float(cb[6]) * m6 + float(cb[7]) * m7);
      }
    }
  }
  for (uint rr = 0u; rr < 8u; rr++) {
    acc[rr] = simd_sum(acc[rr]);
    if (tiisg == 0u) partial[rr * 8u + sgitg] = acc[rr];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    for (uint rr = 0u; rr < 8u; rr++) {
      const uint row = row_base + rr;
      if (row < args.rows) {
        float total = 0.0f;
        for (uint sg = 0u; sg < 8u; sg++) total += partial[rr * 8u + sg];
        out[ulong(token) * ulong(args.out_token_stride) + ulong(row)] = total;
      }
    }
  }
}
kernel void d8f_down_sum_selected_weighted_batch_tile16(
  device const uchar *pack               [[buffer(0)]],
  device const float *mid                [[buffer(1)]],
  device const uint  *selected           [[buffer(2)]],
  device float       *out                [[buffer(3)]],
  constant D8FDownBatchArgs &args        [[buffer(4)]],
  device const float *route_weights      [[buffer(5)]],
  threadgroup float *partial             [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  uint2 pos [[threadgroup_position_in_grid]]) {
  const uint row_base = pos.x << 4;
  const uint token = pos.y;
  if (token >= args.n_tokens) return;
  const uint blocks_per_row = args.in_dim >> 3;
  device const float *token_mid = mid + ulong(token) * ulong(args.mid_token_stride);
  device const float *token_weights = route_weights + ulong(token) * ulong(args.route_token_stride);
  float acc[16];
  for (uint rr = 0u; rr < 16u; rr++) acc[rr] = 0.0f;
  for (uint slot = 0u; slot < args.n_selected; slot++) {
    const float rw = token_weights[slot];
    const uint expert = selected[ulong(token) * ulong(args.selected_token_stride) + ulong(slot)];
    device const uchar *rec = pack + args.table_offset + (512u + expert) * args.record_bytes;
    const uint k = d8f_u32(rec + 8);
    const uint bits = d8f_u32(rec + 12);
    const ulong cb_off = d8f_u64(rec + 24);
    const ulong ix_off = d8f_u64(rec + 32);
    const ulong scale_off = d8f_scale_offset(rec);
    if (rw == 0.0f || k == 0u || bits == 0u) continue;
    const uint mask = (1u << bits) - 1u;
    const device float *slot_mid = token_mid + ulong(slot) * ulong(args.mid_slot_stride);
    for (uint block_col = tid; block_col < blocks_per_row; block_col += 256u) {
      const uint x_base = block_col << 3;
      const float m0 = d8f_mid_value(pack, scale_off, slot_mid, x_base + 0u);
      const float m1 = d8f_mid_value(pack, scale_off, slot_mid, x_base + 1u);
      const float m2 = d8f_mid_value(pack, scale_off, slot_mid, x_base + 2u);
      const float m3 = d8f_mid_value(pack, scale_off, slot_mid, x_base + 3u);
      const float m4 = d8f_mid_value(pack, scale_off, slot_mid, x_base + 4u);
      const float m5 = d8f_mid_value(pack, scale_off, slot_mid, x_base + 5u);
      const float m6 = d8f_mid_value(pack, scale_off, slot_mid, x_base + 6u);
      const float m7 = d8f_mid_value(pack, scale_off, slot_mid, x_base + 7u);
      for (uint rr = 0u; rr < 16u; rr++) {
        const uint row = row_base + rr;
        if (row >= args.rows) continue;
        const ulong block_index = ulong(row) * ulong(blocks_per_row) + ulong(block_col);
        const ulong bit_off = block_index * ulong(bits);
        const ulong byte_off = bit_off >> 3;
        const uint shift = uint(bit_off & 7ul);
        device const uchar *ix = pack + ix_off + byte_off;
        const uint w = uint(ix[0]) | (uint(ix[1]) << 8u) | (uint(ix[2]) << 16u) | (uint(ix[3]) << 24u);
        const uint code = (w >> shift) & mask;
        if (code >= k) continue;
        const device half *cb = (const device half *)(pack + cb_off + ulong(code) * 16ul);
        acc[rr] += rw * (float(cb[0]) * m0 + float(cb[1]) * m1 +
                         float(cb[2]) * m2 + float(cb[3]) * m3 +
                         float(cb[4]) * m4 + float(cb[5]) * m5 +
                         float(cb[6]) * m6 + float(cb[7]) * m7);
      }
    }
  }
  for (uint rr = 0u; rr < 16u; rr++) {
    acc[rr] = simd_sum(acc[rr]);
    if (tiisg == 0u) partial[rr * 8u + sgitg] = acc[rr];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    for (uint rr = 0u; rr < 16u; rr++) {
      const uint row = row_base + rr;
      if (row < args.rows) {
        float total = 0.0f;
        for (uint sg = 0u; sg < 8u; sg++) total += partial[rr * 8u + sg];
        out[ulong(token) * ulong(args.out_token_stride) + ulong(row)] = total;
      }
    }
  }
}
kernel void d8f_down_sum_selected_preweighted_batch_tile16(
  device const uchar *pack               [[buffer(0)]],
  device const float *mid                [[buffer(1)]],
  device const uint  *selected           [[buffer(2)]],
  device float       *out                [[buffer(3)]],
  constant D8FDownBatchArgs &args        [[buffer(4)]],
  threadgroup float *partial             [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  uint2 pos [[threadgroup_position_in_grid]]) {
  const uint row_base = pos.x << 4;
  const uint token = pos.y;
  if (token >= args.n_tokens) return;
  const uint blocks_per_row = args.in_dim >> 3;
  device const float *token_mid = mid + ulong(token) * ulong(args.mid_token_stride);
  float acc[16];
  for (uint rr = 0u; rr < 16u; rr++) acc[rr] = 0.0f;
  for (uint slot = 0u; slot < args.n_selected; slot++) {
    const uint expert = selected[ulong(token) * ulong(args.selected_token_stride) + ulong(slot)];
    device const uchar *rec = pack + args.table_offset + (512u + expert) * args.record_bytes;
    const uint k = d8f_u32(rec + 8); const uint bits = d8f_u32(rec + 12); const ulong cb_off = d8f_u64(rec + 24); const ulong ix_off = d8f_u64(rec + 32); const ulong scale_off = d8f_scale_offset(rec);
    if (k == 0u || bits == 0u) continue;
    const uint mask = (1u << bits) - 1u;
    const device float *slot_mid = token_mid + ulong(slot) * ulong(args.mid_slot_stride);
    for (uint block_col = tid; block_col < blocks_per_row; block_col += 256u) {
      const uint x_base = block_col << 3;
      const float m0 = d8f_mid_value(pack, scale_off, slot_mid, x_base + 0u); const float m1 = d8f_mid_value(pack, scale_off, slot_mid, x_base + 1u); const float m2 = d8f_mid_value(pack, scale_off, slot_mid, x_base + 2u); const float m3 = d8f_mid_value(pack, scale_off, slot_mid, x_base + 3u);
      const float m4 = d8f_mid_value(pack, scale_off, slot_mid, x_base + 4u); const float m5 = d8f_mid_value(pack, scale_off, slot_mid, x_base + 5u); const float m6 = d8f_mid_value(pack, scale_off, slot_mid, x_base + 6u); const float m7 = d8f_mid_value(pack, scale_off, slot_mid, x_base + 7u);
      for (uint rr = 0u; rr < 16u; rr++) {
        const uint row = row_base + rr; if (row >= args.rows) continue;
        const ulong block_index = ulong(row) * ulong(blocks_per_row) + ulong(block_col); const ulong bit_off = block_index * ulong(bits); const ulong byte_off = bit_off >> 3; const uint shift = uint(bit_off & 7ul);
        device const uchar *ix = pack + ix_off + byte_off; const uint w = uint(ix[0]) | (uint(ix[1]) << 8u) | (uint(ix[2]) << 16u) | (uint(ix[3]) << 24u); const uint code = (w >> shift) & mask;
        if (code >= k) continue; const device half *cb = (const device half *)(pack + cb_off + ulong(code) * 16ul);
        acc[rr] += float(cb[0]) * m0 + float(cb[1]) * m1 + float(cb[2]) * m2 + float(cb[3]) * m3 + float(cb[4]) * m4 + float(cb[5]) * m5 + float(cb[6]) * m6 + float(cb[7]) * m7;
      }
    }
  }
  for (uint rr = 0u; rr < 16u; rr++) { acc[rr] = simd_sum(acc[rr]); if (tiisg == 0u) partial[rr * 8u + sgitg] = acc[rr]; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) { for (uint rr = 0u; rr < 16u; rr++) { const uint row = row_base + rr; if (row < args.rows) { float total = 0.0f; for (uint sg = 0u; sg < 8u; sg++) total += partial[rr * 8u + sg]; out[ulong(token) * ulong(args.out_token_stride) + ulong(row)] = total; } } }
}

struct D8FRecordLite {
  uint k;
  uint bits;
  uint mask;
  uint flags;
  ulong codebook_offset;
  ulong index_offset;
  ulong scale_offset;
  uint scale_bytes;
  uint row_base;
};

struct D8FRowBlockRangeArgs {
  uint row_offset;
  uint row_count;
  uint reserved0;
  uint reserved1;
};

struct D8FDownLutArgs {
  uint rows;
  uint in_dim;
  uint n_selected;
  uint n_tokens;
  uint table_offset;
  uint record_bytes;
  uint mid_slot_stride;
  uint mid_token_stride;
  uint selected_token_stride;
  uint max_k;
  uint score_group_stride;
  uint score_slot_stride;
  uint score_token_stride;
  uint out_token_stride;
  uint reserved0;
  uint reserved1;
};

struct D8FDownLutSidecarLite {
  uint rank;
  uint out_dim;
  uint a_offset_lo;
  uint a_offset_hi;
};

inline ulong d8f_down_lut_sidecar_a_offset(D8FDownLutSidecarLite sidecar) {
  return ulong(sidecar.a_offset_lo) | (ulong(sidecar.a_offset_hi) << 32);
}

inline float d8f_down_lut_rank1_delta(
    device const uchar *pack,
    device const D8FDownLutSidecarLite *sidecars,
    device const float *sidecar_dot,
    constant D8FDownLutArgs &args,
    uint token,
    uint slot,
    uint row) {
  if (args.reserved0 == 0u) return 0.0f;
  const D8FDownLutSidecarLite sidecar = sidecars[slot];
  if (sidecar.rank != 1u || row >= sidecar.out_dim) return 0.0f;
  const ulong a_offset = d8f_down_lut_sidecar_a_offset(sidecar);
  if (a_offset == 0ul) return 0.0f;
  const device half *a = (const device half *)(pack + a_offset);
  return float(a[row]) * sidecar_dot[ulong(token) * ulong(args.n_selected) + ulong(slot)];
}

kernel void d8f_down_lut_score_selected_batch(
  device const uchar *pack          [[buffer(0)]],
  device const float *mid           [[buffer(1)]],
  device const uint  *selected      [[buffer(2)]],
  device float       *score         [[buffer(3)]],
  constant D8FDownLutArgs &args     [[buffer(4)]],
  device const D8FRecordLite *recs  [[buffer(5)]],
  uint3 gid [[thread_position_in_grid]]) {
  const uint code = gid.x;
  const uint group = gid.y;
  const uint token_slot = gid.z;
  if (group >= (args.in_dim >> 3) || token_slot >= args.n_tokens * args.n_selected) return;
  const uint token = token_slot / args.n_selected;
  const uint slot = token_slot - token * args.n_selected;
  const uint expert = selected[ulong(token) * ulong(args.selected_token_stride) + ulong(slot)];
  const D8FRecordLite rec = recs[8192u + expert];
  const ulong out_index =
      ulong(token) * ulong(args.score_token_stride) +
      ulong(slot) * ulong(args.score_slot_stride) +
      ulong(group) * ulong(args.score_group_stride) +
      ulong(code);
  if (code >= args.max_k) return;
  if (code >= rec.k) {
    score[out_index] = 0.0f;
    return;
  }
  const uint x_base = group << 3;
  const device float *slot_mid =
      mid + ulong(token) * ulong(args.mid_token_stride) +
      ulong(slot) * ulong(args.mid_slot_stride);
  const device half *cb = (const device half *)(pack + rec.codebook_offset + ulong(code) * 16ul);
  score[out_index] =
      float(cb[0]) * d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 0u) +
      float(cb[1]) * d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 1u) +
      float(cb[2]) * d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 2u) +
      float(cb[3]) * d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 3u) +
      float(cb[4]) * d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 4u) +
      float(cb[5]) * d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 5u) +
      float(cb[6]) * d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 6u) +
      float(cb[7]) * d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 7u);
}

kernel void d8f_down_lut_gather_selected_batch(
  device const uchar *pack          [[buffer(0)]],
  device const uint  *selected      [[buffer(1)]],
  device const float *score         [[buffer(2)]],
  device float       *out           [[buffer(3)]],
  constant D8FDownLutArgs &args     [[buffer(4)]],
  device const D8FRecordLite *recs  [[buffer(5)]],
  threadgroup float *partial        [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  uint2 pos [[threadgroup_position_in_grid]]) {
  const uint row = pos.x;
  const uint token = pos.y;
  if (row >= args.rows || token >= args.n_tokens) return;
  const uint groups = args.in_dim >> 3;
  float acc = 0.0f;
  for (uint slot = 0u; slot < args.n_selected; slot++) {
    const uint expert = selected[ulong(token) * ulong(args.selected_token_stride) + ulong(slot)];
    const D8FRecordLite rec = recs[8192u + expert];
    for (uint group = tid; group < groups; group += 256u) {
      const ulong block_index = ulong(row) * ulong(groups) + ulong(group);
      const ulong bit_off = block_index * ulong(rec.bits);
      const ulong byte_off = bit_off >> 3;
      const uint shift = uint(bit_off & 7ul);
      device const uchar *ix = pack + rec.index_offset + byte_off;
      const uint w = uint(ix[0]) | (uint(ix[1]) << 8u) | (uint(ix[2]) << 16u) | (uint(ix[3]) << 24u);
      const uint code = (w >> shift) & rec.mask;
      acc += score[
          ulong(token) * ulong(args.score_token_stride) +
          ulong(slot) * ulong(args.score_slot_stride) +
          ulong(group) * ulong(args.score_group_stride) +
          ulong(code)];
    }
  }
  acc = simd_sum(acc);
  if (tiisg == 0u) partial[sgitg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float total = 0.0f;
    for (uint sg = 0u; sg < 8u; sg++) total += partial[sg];
    out[ulong(token) * ulong(args.out_token_stride) + ulong(row)] = total;
  }
}

kernel void d8f_down_lut_gather_selected_batch_tile8(
  device const uchar *pack          [[buffer(0)]],
  device const uint  *selected      [[buffer(1)]],
  device const float *score         [[buffer(2)]],
  device float       *out           [[buffer(3)]],
  constant D8FDownLutArgs &args     [[buffer(4)]],
  device const D8FRecordLite *recs  [[buffer(5)]],
  threadgroup float *partial        [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  uint2 pos [[threadgroup_position_in_grid]]) {
  const uint row_base = pos.x << 3;
  const uint token = pos.y;
  if (token >= args.n_tokens) return;
  const uint groups = args.in_dim >> 3;
  float acc[8];
  for (uint rr = 0u; rr < 8u; rr++) acc[rr] = 0.0f;
  for (uint slot = 0u; slot < args.n_selected; slot++) {
    const uint expert = selected[ulong(token) * ulong(args.selected_token_stride) + ulong(slot)];
    const D8FRecordLite rec = recs[8192u + expert];
    for (uint group = tid; group < groups; group += 256u) {
      const ulong score_base =
          ulong(token) * ulong(args.score_token_stride) +
          ulong(slot) * ulong(args.score_slot_stride) +
          ulong(group) * ulong(args.score_group_stride);
      for (uint rr = 0u; rr < 8u; rr++) {
        const uint row = row_base + rr;
        if (row >= args.rows) continue;
        const ulong block_index = ulong(row) * ulong(groups) + ulong(group);
        const ulong bit_off = block_index * ulong(rec.bits);
        const ulong byte_off = bit_off >> 3;
        const uint shift = uint(bit_off & 7ul);
        device const uchar *ix = pack + rec.index_offset + byte_off;
        const uint word = uint(ix[0]) | (uint(ix[1]) << 8u) | (uint(ix[2]) << 16u) | (uint(ix[3]) << 24u);
        const uint code = (word >> shift) & rec.mask;
        acc[rr] += score[score_base + ulong(code)];
      }
    }
  }
  for (uint rr = 0u; rr < 8u; rr++) {
    acc[rr] = simd_sum(acc[rr]);
    if (tiisg == 0u) partial[rr * 8u + sgitg] = acc[rr];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    for (uint rr = 0u; rr < 8u; rr++) {
      const uint row = row_base + rr;
      if (row < args.rows) {
        float total = 0.0f;
        for (uint sg = 0u; sg < 8u; sg++) total += partial[rr * 8u + sg];
        out[ulong(token) * ulong(args.out_token_stride) + ulong(row)] = total;
      }
    }
  }
}

kernel void d8f_down_lut_gather_selected_batch_tile16(
  device const uchar *pack          [[buffer(0)]],
  device const uint  *selected      [[buffer(1)]],
  device const float *score         [[buffer(2)]],
  device float       *out           [[buffer(3)]],
  constant D8FDownLutArgs &args     [[buffer(4)]],
  device const D8FRecordLite *recs  [[buffer(5)]],
  threadgroup float *partial        [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  uint2 pos [[threadgroup_position_in_grid]]) {
  const uint row_base = pos.x << 4;
  const uint token = pos.y;
  if (token >= args.n_tokens) return;
  const uint groups = args.in_dim >> 3;
  float acc[16];
  for (uint rr = 0u; rr < 16u; rr++) acc[rr] = 0.0f;
  for (uint slot = 0u; slot < args.n_selected; slot++) {
    const uint expert = selected[ulong(token) * ulong(args.selected_token_stride) + ulong(slot)];
    const D8FRecordLite rec = recs[8192u + expert];
    for (uint group = tid; group < groups; group += 256u) {
      const ulong score_base =
          ulong(token) * ulong(args.score_token_stride) +
          ulong(slot) * ulong(args.score_slot_stride) +
          ulong(group) * ulong(args.score_group_stride);
      for (uint rr = 0u; rr < 16u; rr++) {
        const uint row = row_base + rr;
        if (row >= args.rows) continue;
        const ulong block_index = ulong(row) * ulong(groups) + ulong(group);
        const ulong bit_off = block_index * ulong(rec.bits);
        const ulong byte_off = bit_off >> 3;
        const uint shift = uint(bit_off & 7ul);
        device const uchar *ix = pack + rec.index_offset + byte_off;
        const uint word = uint(ix[0]) | (uint(ix[1]) << 8u) | (uint(ix[2]) << 16u) | (uint(ix[3]) << 24u);
        const uint code = (word >> shift) & rec.mask;
        acc[rr] += score[score_base + ulong(code)];
      }
    }
  }
  for (uint rr = 0u; rr < 16u; rr++) {
    acc[rr] = simd_sum(acc[rr]);
    if (tiisg == 0u) partial[rr * 8u + sgitg] = acc[rr];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    for (uint rr = 0u; rr < 16u; rr++) {
      const uint row = row_base + rr;
      if (row < args.rows) {
        float total = 0.0f;
        for (uint sg = 0u; sg < 8u; sg++) total += partial[rr * 8u + sg];
        out[ulong(token) * ulong(args.out_token_stride) + ulong(row)] = total;
      }
    }
  }
}

kernel void d8f_down_lut_gather_codes_selected_batch(
  device const ushort *codes        [[buffer(0)]],
  device const float  *score        [[buffer(1)]],
  device float        *out          [[buffer(2)]],
  constant D8FDownLutArgs &args     [[buffer(3)]],
  device const D8FDownLutSidecarLite *sidecars [[buffer(4)]],
  device const float *sidecar_dot    [[buffer(5)]],
  device const uchar *pack           [[buffer(6)]],
  threadgroup float *partial        [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  uint2 pos [[threadgroup_position_in_grid]]) {
  const uint row = pos.x;
  const uint token = pos.y;
  if (row >= args.rows || token >= args.n_tokens) return;
  const uint groups = args.in_dim >> 3;
  float acc = 0.0f;
  for (uint slot = 0u; slot < args.n_selected; slot++) {
    const ulong code_base = (ulong(slot) * ulong(args.rows) + ulong(row)) * ulong(groups);
    for (uint group = tid; group < groups; group += 256u) {
      const uint code = uint(codes[code_base + ulong(group)]);
      if (code >= args.max_k) continue;
      acc += score[
          ulong(token) * ulong(args.score_token_stride) +
          ulong(slot) * ulong(args.score_slot_stride) +
          ulong(group) * ulong(args.score_group_stride) +
          ulong(code)];
    }
    if (tid == 0u) {
      acc += d8f_down_lut_rank1_delta(pack, sidecars, sidecar_dot, args, token, slot, row);
    }
  }
  acc = simd_sum(acc);
  if (tiisg == 0u) partial[sgitg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float total = 0.0f;
    for (uint sg = 0u; sg < 8u; sg++) total += partial[sg];
    out[ulong(token) * ulong(args.out_token_stride) + ulong(row)] = total;
  }
}

kernel void d8f_down_lut_gather_codes_selected_batch_tile8(
  device const ushort *codes        [[buffer(0)]],
  device const float  *score        [[buffer(1)]],
  device float        *out          [[buffer(2)]],
  constant D8FDownLutArgs &args     [[buffer(3)]],
  device const D8FDownLutSidecarLite *sidecars [[buffer(4)]],
  device const float *sidecar_dot    [[buffer(5)]],
  device const uchar *pack           [[buffer(6)]],
  threadgroup float *partial        [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  uint2 pos [[threadgroup_position_in_grid]]) {
  const uint row_base = pos.x << 3;
  const uint token = pos.y;
  if (token >= args.n_tokens) return;
  const uint groups = args.in_dim >> 3;
  float acc[8];
  for (uint rr = 0u; rr < 8u; rr++) acc[rr] = 0.0f;
  for (uint slot = 0u; slot < args.n_selected; slot++) {
    for (uint group = tid; group < groups; group += 256u) {
      const ulong score_base =
          ulong(token) * ulong(args.score_token_stride) +
          ulong(slot) * ulong(args.score_slot_stride) +
          ulong(group) * ulong(args.score_group_stride);
      for (uint rr = 0u; rr < 8u; rr++) {
        const uint row = row_base + rr;
        if (row >= args.rows) continue;
        const uint code = uint(codes[(ulong(slot) * ulong(args.rows) + ulong(row)) * ulong(groups) + ulong(group)]);
        if (code < args.max_k) acc[rr] += score[score_base + ulong(code)];
      }
    }
    if (tid == 0u) {
      for (uint rr = 0u; rr < 8u; rr++) {
        const uint row = row_base + rr;
        if (row < args.rows) {
          acc[rr] += d8f_down_lut_rank1_delta(pack, sidecars, sidecar_dot, args, token, slot, row);
        }
      }
    }
  }
  for (uint rr = 0u; rr < 8u; rr++) {
    acc[rr] = simd_sum(acc[rr]);
    if (tiisg == 0u) partial[rr * 8u + sgitg] = acc[rr];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    for (uint rr = 0u; rr < 8u; rr++) {
      const uint row = row_base + rr;
      if (row < args.rows) {
        float total = 0.0f;
        for (uint sg = 0u; sg < 8u; sg++) total += partial[rr * 8u + sg];
        out[ulong(token) * ulong(args.out_token_stride) + ulong(row)] = total;
      }
    }
  }
}

kernel void d8f_down_lut_gather_codes_selected_batch_tile16(
  device const ushort *codes        [[buffer(0)]],
  device const float  *score        [[buffer(1)]],
  device float        *out          [[buffer(2)]],
  constant D8FDownLutArgs &args     [[buffer(3)]],
  device const D8FDownLutSidecarLite *sidecars [[buffer(4)]],
  device const float *sidecar_dot    [[buffer(5)]],
  device const uchar *pack           [[buffer(6)]],
  threadgroup float *partial        [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  uint2 pos [[threadgroup_position_in_grid]]) {
  const uint row_base = pos.x << 4;
  const uint token = pos.y;
  if (token >= args.n_tokens) return;
  const uint groups = args.in_dim >> 3;
  float acc[16];
  for (uint rr = 0u; rr < 16u; rr++) acc[rr] = 0.0f;
  for (uint slot = 0u; slot < args.n_selected; slot++) {
    for (uint group = tid; group < groups; group += 256u) {
      const ulong score_base =
          ulong(token) * ulong(args.score_token_stride) +
          ulong(slot) * ulong(args.score_slot_stride) +
          ulong(group) * ulong(args.score_group_stride);
      for (uint rr = 0u; rr < 16u; rr++) {
        const uint row = row_base + rr;
        if (row >= args.rows) continue;
        const uint code = uint(codes[(ulong(slot) * ulong(args.rows) + ulong(row)) * ulong(groups) + ulong(group)]);
        if (code < args.max_k) acc[rr] += score[score_base + ulong(code)];
      }
    }
    if (tid == 0u) {
      for (uint rr = 0u; rr < 16u; rr++) {
        const uint row = row_base + rr;
        if (row < args.rows) {
          acc[rr] += d8f_down_lut_rank1_delta(pack, sidecars, sidecar_dot, args, token, slot, row);
        }
      }
    }
  }
  for (uint rr = 0u; rr < 16u; rr++) {
    acc[rr] = simd_sum(acc[rr]);
    if (tiisg == 0u) partial[rr * 8u + sgitg] = acc[rr];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    for (uint rr = 0u; rr < 16u; rr++) {
      const uint row = row_base + rr;
      if (row < args.rows) {
        float total = 0.0f;
        for (uint sg = 0u; sg < 8u; sg++) total += partial[rr * 8u + sg];
        out[ulong(token) * ulong(args.out_token_stride) + ulong(row)] = total;
      }
    }
  }
}

kernel void d8f_down_lut_scoreh_selected_batch(
  device const uchar *pack          [[buffer(0)]],
  device const float *mid           [[buffer(1)]],
  device const uint  *selected      [[buffer(2)]],
  device half        *score         [[buffer(3)]],
  constant D8FDownLutArgs &args     [[buffer(4)]],
  device const D8FRecordLite *recs  [[buffer(5)]],
  uint3 gid [[thread_position_in_grid]]) {
  const uint code = gid.x;
  const uint group = gid.y;
  const uint token_slot = gid.z;
  if (group >= (args.in_dim >> 3) || token_slot >= args.n_tokens * args.n_selected) return;
  const uint token = token_slot / args.n_selected;
  const uint slot = token_slot - token * args.n_selected;
  const uint expert = selected[ulong(token) * ulong(args.selected_token_stride) + ulong(slot)];
  const D8FRecordLite rec = recs[8192u + expert];
  const ulong out_index =
      ulong(token) * ulong(args.score_token_stride) +
      ulong(slot) * ulong(args.score_slot_stride) +
      ulong(group) * ulong(args.score_group_stride) +
      ulong(code);
  if (code >= args.max_k) return;
  if (code >= rec.k) {
    score[out_index] = half(0.0h);
    return;
  }
  const uint x_base = group << 3;
  const device float *slot_mid =
      mid + ulong(token) * ulong(args.mid_token_stride) +
      ulong(slot) * ulong(args.mid_slot_stride);
  const device half *cb = (const device half *)(pack + rec.codebook_offset + ulong(code) * 16ul);
  const float v =
      float(cb[0]) * d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 0u) +
      float(cb[1]) * d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 1u) +
      float(cb[2]) * d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 2u) +
      float(cb[3]) * d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 3u) +
      float(cb[4]) * d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 4u) +
      float(cb[5]) * d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 5u) +
      float(cb[6]) * d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 6u) +
      float(cb[7]) * d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 7u);
  score[out_index] = half(v);
}

kernel void d8f_down_lut_gatherh_selected_batch(
  device const uchar *pack          [[buffer(0)]],
  device const uint  *selected      [[buffer(1)]],
  device const half  *score         [[buffer(2)]],
  device float       *out           [[buffer(3)]],
  constant D8FDownLutArgs &args     [[buffer(4)]],
  device const D8FRecordLite *recs  [[buffer(5)]],
  threadgroup float *partial        [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  uint2 pos [[threadgroup_position_in_grid]]) {
  const uint row = pos.x;
  const uint token = pos.y;
  if (row >= args.rows || token >= args.n_tokens) return;
  const uint groups = args.in_dim >> 3;
  float acc = 0.0f;
  for (uint slot = 0u; slot < args.n_selected; slot++) {
    const uint expert = selected[ulong(token) * ulong(args.selected_token_stride) + ulong(slot)];
    const D8FRecordLite rec = recs[8192u + expert];
    for (uint group = tid; group < groups; group += 256u) {
      const ulong block_index = ulong(row) * ulong(groups) + ulong(group);
      const ulong bit_off = block_index * ulong(rec.bits);
      const ulong byte_off = bit_off >> 3;
      const uint shift = uint(bit_off & 7ul);
      device const uchar *ix = pack + rec.index_offset + byte_off;
      const uint w = uint(ix[0]) | (uint(ix[1]) << 8u) | (uint(ix[2]) << 16u) | (uint(ix[3]) << 24u);
      const uint code = (w >> shift) & rec.mask;
      acc += float(score[
          ulong(token) * ulong(args.score_token_stride) +
          ulong(slot) * ulong(args.score_slot_stride) +
          ulong(group) * ulong(args.score_group_stride) +
          ulong(code)]);
    }
  }
  acc = simd_sum(acc);
  if (tiisg == 0u) partial[sgitg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float total = 0.0f;
    for (uint sg = 0u; sg < 8u; sg++) total += partial[sg];
    out[ulong(token) * ulong(args.out_token_stride) + ulong(row)] = total;
  }
}

kernel void d8f_gateup_swiglu_selected_batch_recbuf(
  device const uchar *pack          [[buffer(0)]],
  device const float *x_base_all    [[buffer(1)]],
  device const uint  *selected      [[buffer(2)]],
  device float       *mid           [[buffer(3)]],
  constant D8FGateUpBatchArgs &args [[buffer(4)]],
  device const D8FRecordLite *recs  [[buffer(5)]],
  threadgroup float *partial        [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  uint3 pos [[threadgroup_position_in_grid]]) {
  const uint row = pos.x;
  const uint slot = pos.y;
  const uint token = pos.z;
  if (row >= args.rows || slot >= args.n_selected || token >= args.n_tokens) return;
  device const float *x = x_base_all + ulong(token) * ulong(args.x_token_stride);
  const uint expert = selected[ulong(token) * ulong(args.selected_token_stride) + ulong(slot)];
  const uint row_block = row >> 7;
  const D8FRecordLite gate_rec = recs[expert * 16u + row_block];
  const D8FRecordLite up_rec = recs[4096u + expert * 16u + row_block];
  const uint blocks_per_row = args.in_dim >> 3;
  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  for (uint block_col = tid; block_col < blocks_per_row; block_col += 256u) {
    const ulong gate_block_index = ulong(row - gate_rec.row_base) * ulong(blocks_per_row) + ulong(block_col);
    const ulong up_block_index = ulong(row - up_rec.row_base) * ulong(blocks_per_row) + ulong(block_col);
    const uint x_base = block_col << 3;
    const float x0 = x[x_base + 0u];
    const float x1 = x[x_base + 1u];
    const float x2 = x[x_base + 2u];
    const float x3 = x[x_base + 3u];
    const float x4 = x[x_base + 4u];
    const float x5 = x[x_base + 5u];
    const float x6 = x[x_base + 6u];
    const float x7 = x[x_base + 7u];
    const ulong gate_bit_off = gate_block_index * ulong(gate_rec.bits);
    const ulong gate_byte_off = gate_bit_off >> 3;
    const uint gate_shift = uint(gate_bit_off & 7ul);
    device const uchar *gate_ix = pack + gate_rec.index_offset + gate_byte_off;
    const uint gate_w = uint(gate_ix[0]) | (uint(gate_ix[1]) << 8u) | (uint(gate_ix[2]) << 16u) | (uint(gate_ix[3]) << 24u);
    const uint gate_code = (gate_w >> gate_shift) & gate_rec.mask;
    const device half *gate_cb = (const device half *)(pack + gate_rec.codebook_offset + ulong(gate_code) * 16ul);
    gate_acc += float(gate_cb[0]) * x0 + float(gate_cb[1]) * x1 + float(gate_cb[2]) * x2 + float(gate_cb[3]) * x3 +
                float(gate_cb[4]) * x4 + float(gate_cb[5]) * x5 + float(gate_cb[6]) * x6 + float(gate_cb[7]) * x7;
    const ulong up_bit_off = up_block_index * ulong(up_rec.bits);
    const ulong up_byte_off = up_bit_off >> 3;
    const uint up_shift = uint(up_bit_off & 7ul);
    device const uchar *up_ix = pack + up_rec.index_offset + up_byte_off;
    const uint up_w = uint(up_ix[0]) | (uint(up_ix[1]) << 8u) | (uint(up_ix[2]) << 16u) | (uint(up_ix[3]) << 24u);
    const uint up_code = (up_w >> up_shift) & up_rec.mask;
    const device half *up_cb = (const device half *)(pack + up_rec.codebook_offset + ulong(up_code) * 16ul);
    up_acc += float(up_cb[0]) * x0 + float(up_cb[1]) * x1 + float(up_cb[2]) * x2 + float(up_cb[3]) * x3 +
              float(up_cb[4]) * x4 + float(up_cb[5]) * x5 + float(up_cb[6]) * x6 + float(up_cb[7]) * x7;
  }
  gate_acc = simd_sum(gate_acc);
  up_acc = simd_sum(up_acc);
  if (tiisg == 0u) { partial[sgitg] = gate_acc; partial[8u + sgitg] = up_acc; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float gate_total = 0.0f;
    float up_total = 0.0f;
    for (uint sg = 0u; sg < 8u; sg++) { gate_total += partial[sg]; up_total += partial[8u + sg]; }
    gate_total = min(gate_total, args.clamp);
    up_total = clamp(up_total, -args.clamp, args.clamp);
    mid[ulong(token) * ulong(args.mid_token_stride) + ulong(slot) * ulong(args.mid_slot_stride) + ulong(row)] = (gate_total / (1.0f + exp(-gate_total))) * up_total;
  }
}

kernel void d8f_gateup_swiglu_selected_batch_recbuf_range(
  device const uchar *pack          [[buffer(0)]],
  device const float *x_base_all    [[buffer(1)]],
  device const uint  *selected      [[buffer(2)]],
  device float       *mid           [[buffer(3)]],
  constant D8FGateUpBatchArgs &args [[buffer(4)]],
  device const D8FRecordLite *recs  [[buffer(5)]],
  constant D8FRowBlockRangeArgs &range [[buffer(6)]],
  threadgroup float *partial        [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  uint3 pos [[threadgroup_position_in_grid]]) {
  const uint local_row = pos.x;
  const uint slot = pos.y;
  const uint token = pos.z;
  if (local_row >= range.row_count || slot >= args.n_selected || token >= args.n_tokens) return;
  const uint row = range.row_offset + local_row;
  if (row >= args.rows) return;
  device const float *x = x_base_all + ulong(token) * ulong(args.x_token_stride);
  const uint expert = selected[ulong(token) * ulong(args.selected_token_stride) + ulong(slot)];
  const uint row_block = row >> 7;
  const D8FRecordLite gate_rec = recs[expert * 16u + row_block];
  const D8FRecordLite up_rec = recs[4096u + expert * 16u + row_block];
  const uint blocks_per_row = args.in_dim >> 3;
  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  for (uint block_col = tid; block_col < blocks_per_row; block_col += 256u) {
    const ulong gate_block_index = ulong(row - gate_rec.row_base) * ulong(blocks_per_row) + ulong(block_col);
    const ulong up_block_index = ulong(row - up_rec.row_base) * ulong(blocks_per_row) + ulong(block_col);
    const uint x_base = block_col << 3;
    const float x0 = x[x_base + 0u];
    const float x1 = x[x_base + 1u];
    const float x2 = x[x_base + 2u];
    const float x3 = x[x_base + 3u];
    const float x4 = x[x_base + 4u];
    const float x5 = x[x_base + 5u];
    const float x6 = x[x_base + 6u];
    const float x7 = x[x_base + 7u];
    const ulong gate_bit_off = gate_block_index * ulong(gate_rec.bits);
    const ulong gate_byte_off = gate_bit_off >> 3;
    const uint gate_shift = uint(gate_bit_off & 7ul);
    device const uchar *gate_ix = pack + gate_rec.index_offset + gate_byte_off;
    const uint gate_w = uint(gate_ix[0]) | (uint(gate_ix[1]) << 8u) | (uint(gate_ix[2]) << 16u) | (uint(gate_ix[3]) << 24u);
    const uint gate_code = (gate_w >> gate_shift) & gate_rec.mask;
    const device half *gate_cb = (const device half *)(pack + gate_rec.codebook_offset + ulong(gate_code) * 16ul);
    gate_acc += float(gate_cb[0]) * x0 + float(gate_cb[1]) * x1 + float(gate_cb[2]) * x2 + float(gate_cb[3]) * x3 +
                float(gate_cb[4]) * x4 + float(gate_cb[5]) * x5 + float(gate_cb[6]) * x6 + float(gate_cb[7]) * x7;
    const ulong up_bit_off = up_block_index * ulong(up_rec.bits);
    const ulong up_byte_off = up_bit_off >> 3;
    const uint up_shift = uint(up_bit_off & 7ul);
    device const uchar *up_ix = pack + up_rec.index_offset + up_byte_off;
    const uint up_w = uint(up_ix[0]) | (uint(up_ix[1]) << 8u) | (uint(up_ix[2]) << 16u) | (uint(up_ix[3]) << 24u);
    const uint up_code = (up_w >> up_shift) & up_rec.mask;
    const device half *up_cb = (const device half *)(pack + up_rec.codebook_offset + ulong(up_code) * 16ul);
    up_acc += float(up_cb[0]) * x0 + float(up_cb[1]) * x1 + float(up_cb[2]) * x2 + float(up_cb[3]) * x3 +
              float(up_cb[4]) * x4 + float(up_cb[5]) * x5 + float(up_cb[6]) * x6 + float(up_cb[7]) * x7;
  }
  gate_acc = simd_sum(gate_acc);
  up_acc = simd_sum(up_acc);
  if (tiisg == 0u) { partial[sgitg] = gate_acc; partial[8u + sgitg] = up_acc; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float gate_total = 0.0f;
    float up_total = 0.0f;
    for (uint sg = 0u; sg < 8u; sg++) { gate_total += partial[sg]; up_total += partial[8u + sg]; }
    gate_total = min(gate_total, args.clamp);
    up_total = clamp(up_total, -args.clamp, args.clamp);
    mid[ulong(token) * ulong(args.mid_token_stride) + ulong(slot) * ulong(args.mid_slot_stride) + ulong(local_row)] =
      (gate_total / (1.0f + exp(-gate_total))) * up_total;
  }
}

kernel void d8f_down_sum_selected_weighted_batch_tile8_recbuf(
  device const uchar *pack               [[buffer(0)]],
  device const float *mid                [[buffer(1)]],
  device const uint  *selected           [[buffer(2)]],
  device float       *out                [[buffer(3)]],
  constant D8FDownBatchArgs &args        [[buffer(4)]],
  device const float *route_weights      [[buffer(5)]],
  device const D8FRecordLite *recs       [[buffer(6)]],
  device const float *sidecar_dot        [[buffer(7)]],
  threadgroup float *partial             [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  uint2 pos [[threadgroup_position_in_grid]]) {
  (void)sidecar_dot;
  const uint row_base = pos.x << 3;
  const uint token = pos.y;
  if (token >= args.n_tokens) return;
  const uint blocks_per_row = args.in_dim >> 3;
  device const float *token_mid = mid + ulong(token) * ulong(args.mid_token_stride);
  device const float *token_weights = route_weights + ulong(token) * ulong(args.route_token_stride);
  float acc[8];
  for (uint rr = 0u; rr < 8u; rr++) acc[rr] = 0.0f;
  for (uint slot = 0u; slot < args.n_selected; slot++) {
    const float rw = token_weights[slot];
    if (rw == 0.0f) continue;
    const uint expert = selected[ulong(token) * ulong(args.selected_token_stride) + ulong(slot)];
    const D8FRecordLite rec = recs[8192u + expert];
    device const uchar *side_rec = d8f_down_sidecar_record(pack, args, expert);
    const uint side_rank = side_rec ? d8f_u32(side_rec + 4) : 0u;
    const uint side_in_dim = side_rec ? d8f_u32(side_rec + 8) : 0u;
    const uint side_out_dim = side_rec ? d8f_u32(side_rec + 12) : 0u;
    const ulong side_u_off = side_rec ? d8f_u64(side_rec + 16) : 0ul;
    const ulong side_a_off = side_rec ? d8f_u64(side_rec + 24) : 0ul;
    const bool side_rank1 = side_rank == 1u && side_in_dim == args.in_dim &&
                            side_out_dim == args.rows && side_u_off != 0ul &&
                            side_a_off != 0ul;
    device const half *side_u = side_rank1 ? (const device half *)(pack + side_u_off) : nullptr;
    device const half *side_a = side_rank1 ? (const device half *)(pack + side_a_off) : nullptr;
    const device float *slot_mid = token_mid + ulong(slot) * ulong(args.mid_slot_stride);
    float side_dot = 0.0f;
    for (uint block_col = tid; block_col < blocks_per_row; block_col += 256u) {
      const uint x_base = block_col << 3;
      const float m0 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 0u);
      const float m1 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 1u);
      const float m2 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 2u);
      const float m3 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 3u);
      const float m4 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 4u);
      const float m5 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 5u);
      const float m6 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 6u);
      const float m7 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 7u);
      if (side_rank1) {
        side_dot += float(side_u[x_base + 0u]) * m0 + float(side_u[x_base + 1u]) * m1 +
                    float(side_u[x_base + 2u]) * m2 + float(side_u[x_base + 3u]) * m3 +
                    float(side_u[x_base + 4u]) * m4 + float(side_u[x_base + 5u]) * m5 +
                    float(side_u[x_base + 6u]) * m6 + float(side_u[x_base + 7u]) * m7;
      }
      for (uint rr = 0u; rr < 8u; rr++) {
        const uint row = row_base + rr;
        if (row >= args.rows) continue;
        const ulong block_index = ulong(row) * ulong(blocks_per_row) + ulong(block_col);
        const ulong bit_off = block_index * ulong(rec.bits);
        const ulong byte_off = bit_off >> 3;
        const uint shift = uint(bit_off & 7ul);
        device const uchar *ix = pack + rec.index_offset + byte_off;
        const uint w = uint(ix[0]) | (uint(ix[1]) << 8u) | (uint(ix[2]) << 16u) | (uint(ix[3]) << 24u);
        const uint code = (w >> shift) & rec.mask;
        const device half *cb = (const device half *)(pack + rec.codebook_offset + ulong(code) * 16ul);
        acc[rr] += rw * (float(cb[0]) * m0 + float(cb[1]) * m1 +
                         float(cb[2]) * m2 + float(cb[3]) * m3 +
                         float(cb[4]) * m4 + float(cb[5]) * m5 +
                         float(cb[6]) * m6 + float(cb[7]) * m7);
      }
    }
    if (side_rank1) {
      side_dot = simd_sum(side_dot);
      if (tiisg == 0u) partial[sgitg] = side_dot;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid == 0u) {
        float dot_total = 0.0f;
        for (uint sg = 0u; sg < 8u; sg++) dot_total += partial[sg];
        for (uint rr = 0u; rr < 8u; rr++) {
          const uint row = row_base + rr;
          if (row < args.rows) acc[rr] += rw * float(side_a[row]) * dot_total;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  for (uint rr = 0u; rr < 8u; rr++) {
    acc[rr] = simd_sum(acc[rr]);
    if (tiisg == 0u) partial[rr * 8u + sgitg] = acc[rr];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    for (uint rr = 0u; rr < 8u; rr++) {
      const uint row = row_base + rr;
      if (row < args.rows) {
        float total = 0.0f;
        for (uint sg = 0u; sg < 8u; sg++) total += partial[rr * 8u + sg];
        out[ulong(token) * ulong(args.out_token_stride) + ulong(row)] = total;
      }
    }
  }
}


kernel void d8f_down_sum_selected_weighted_batch_tile16_recbuf(
  device const uchar *pack               [[buffer(0)]],
  device const float *mid                [[buffer(1)]],
  device const uint  *selected           [[buffer(2)]],
  device float       *out                [[buffer(3)]],
  constant D8FDownBatchArgs &args        [[buffer(4)]],
  device const float *route_weights      [[buffer(5)]],
  device const D8FRecordLite *recs       [[buffer(6)]],
  device const float *sidecar_dot        [[buffer(7)]],
  threadgroup float *partial             [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  uint2 pos [[threadgroup_position_in_grid]]) {
  (void)sidecar_dot;
  const uint row_base = pos.x << 4;
  const uint token = pos.y;
  if (token >= args.n_tokens) return;
  const uint blocks_per_row = args.in_dim >> 3;
  device const float *token_mid = mid + ulong(token) * ulong(args.mid_token_stride);
  device const float *token_weights = route_weights + ulong(token) * ulong(args.route_token_stride);
  float acc[16];
  for (uint rr = 0u; rr < 16u; rr++) acc[rr] = 0.0f;
  for (uint slot = 0u; slot < args.n_selected; slot++) {
    const float rw = token_weights[slot];
    if (rw == 0.0f) continue;
    const uint expert = selected[ulong(token) * ulong(args.selected_token_stride) + ulong(slot)];
    const D8FRecordLite rec = recs[8192u + expert];
    device const uchar *side_rec = d8f_down_sidecar_record(pack, args, expert);
    const uint side_rank = side_rec ? d8f_u32(side_rec + 4) : 0u;
    const uint side_in_dim = side_rec ? d8f_u32(side_rec + 8) : 0u;
    const uint side_out_dim = side_rec ? d8f_u32(side_rec + 12) : 0u;
    const ulong side_u_off = side_rec ? d8f_u64(side_rec + 16) : 0ul;
    const ulong side_a_off = side_rec ? d8f_u64(side_rec + 24) : 0ul;
    const bool side_rank1 = side_rank == 1u && side_in_dim == args.in_dim &&
                            side_out_dim == args.rows && side_u_off != 0ul &&
                            side_a_off != 0ul;
    device const half *side_u = side_rank1 ? (const device half *)(pack + side_u_off) : nullptr;
    device const half *side_a = side_rank1 ? (const device half *)(pack + side_a_off) : nullptr;
    const device float *slot_mid = token_mid + ulong(slot) * ulong(args.mid_slot_stride);
    float side_dot = 0.0f;
    for (uint block_col = tid; block_col < blocks_per_row; block_col += 256u) {
      const uint x_base = block_col << 3;
      const float m0 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 0u);
      const float m1 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 1u);
      const float m2 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 2u);
      const float m3 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 3u);
      const float m4 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 4u);
      const float m5 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 5u);
      const float m6 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 6u);
      const float m7 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 7u);
      if (side_rank1) {
        side_dot += float(side_u[x_base + 0u]) * m0 + float(side_u[x_base + 1u]) * m1 +
                    float(side_u[x_base + 2u]) * m2 + float(side_u[x_base + 3u]) * m3 +
                    float(side_u[x_base + 4u]) * m4 + float(side_u[x_base + 5u]) * m5 +
                    float(side_u[x_base + 6u]) * m6 + float(side_u[x_base + 7u]) * m7;
      }
      for (uint rr = 0u; rr < 16u; rr++) {
        const uint row = row_base + rr;
        if (row >= args.rows) continue;
        const ulong block_index = ulong(row) * ulong(blocks_per_row) + ulong(block_col);
        const ulong bit_off = block_index * ulong(rec.bits);
        const ulong byte_off = bit_off >> 3;
        const uint shift = uint(bit_off & 7ul);
        device const uchar *ix = pack + rec.index_offset + byte_off;
        const uint w = uint(ix[0]) | (uint(ix[1]) << 8u) | (uint(ix[2]) << 16u) | (uint(ix[3]) << 24u);
        const uint code = (w >> shift) & rec.mask;
        const device half *cb = (const device half *)(pack + rec.codebook_offset + ulong(code) * 16ul);
        acc[rr] += rw * (float(cb[0]) * m0 + float(cb[1]) * m1 +
                         float(cb[2]) * m2 + float(cb[3]) * m3 +
                         float(cb[4]) * m4 + float(cb[5]) * m5 +
                         float(cb[6]) * m6 + float(cb[7]) * m7);
      }
    }
    if (side_rank1) {
      side_dot = simd_sum(side_dot);
      if (tiisg == 0u) partial[sgitg] = side_dot;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid == 0u) {
        float dot_total = 0.0f;
        for (uint sg = 0u; sg < 8u; sg++) dot_total += partial[sg];
        for (uint rr = 0u; rr < 16u; rr++) {
          const uint row = row_base + rr;
          if (row < args.rows) acc[rr] += rw * float(side_a[row]) * dot_total;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  for (uint rr = 0u; rr < 16u; rr++) {
    acc[rr] = simd_sum(acc[rr]);
    if (tiisg == 0u) partial[rr * 8u + sgitg] = acc[rr];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    for (uint rr = 0u; rr < 16u; rr++) {
      const uint row = row_base + rr;
      if (row < args.rows) {
        float total = 0.0f;
        for (uint sg = 0u; sg < 8u; sg++) total += partial[rr * 8u + sg];
        out[ulong(token) * ulong(args.out_token_stride) + ulong(row)] = total;
      }
    }
  }
}

kernel void d8f_down_sum_selected_weighted_batch_tile16_recbuf_range_accum(
  device const uchar *pack               [[buffer(0)]],
  device const float *mid                [[buffer(1)]],
  device const uint  *selected           [[buffer(2)]],
  device float       *out                [[buffer(3)]],
  constant D8FDownBatchArgs &args        [[buffer(4)]],
  device const float *route_weights      [[buffer(5)]],
  device const D8FRecordLite *recs       [[buffer(6)]],
  constant D8FRowBlockRangeArgs &range   [[buffer(7)]],
  threadgroup float *partial             [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  uint2 pos [[threadgroup_position_in_grid]]) {
  const uint row_base = pos.x << 4;
  const uint token = pos.y;
  if (token >= args.n_tokens || range.row_count == 0u) return;
  const uint blocks_per_row = args.in_dim >> 3;
  const uint block_begin = range.row_offset >> 3;
  const uint block_end = (range.row_offset + range.row_count) >> 3;
  device const float *token_mid = mid + ulong(token) * ulong(args.mid_token_stride);
  device const float *token_weights = route_weights + ulong(token) * ulong(args.route_token_stride);
  float acc[16];
  for (uint rr = 0u; rr < 16u; rr++) acc[rr] = 0.0f;
  for (uint slot = 0u; slot < args.n_selected; slot++) {
    const float rw = token_weights[slot];
    if (rw == 0.0f) continue;
    const uint expert = selected[ulong(token) * ulong(args.selected_token_stride) + ulong(slot)];
    const D8FRecordLite rec = recs[8192u + expert];
    const device float *slot_mid = token_mid + ulong(slot) * ulong(args.mid_slot_stride);
    for (uint block_col = block_begin + tid; block_col < block_end && block_col < blocks_per_row; block_col += 256u) {
      const uint global_x_base = block_col << 3;
      const uint local_x_base = global_x_base - range.row_offset;
      const float m0 = d8f_mid_value_range(pack, rec.scale_offset, slot_mid, local_x_base + 0u, global_x_base + 0u);
      const float m1 = d8f_mid_value_range(pack, rec.scale_offset, slot_mid, local_x_base + 1u, global_x_base + 1u);
      const float m2 = d8f_mid_value_range(pack, rec.scale_offset, slot_mid, local_x_base + 2u, global_x_base + 2u);
      const float m3 = d8f_mid_value_range(pack, rec.scale_offset, slot_mid, local_x_base + 3u, global_x_base + 3u);
      const float m4 = d8f_mid_value_range(pack, rec.scale_offset, slot_mid, local_x_base + 4u, global_x_base + 4u);
      const float m5 = d8f_mid_value_range(pack, rec.scale_offset, slot_mid, local_x_base + 5u, global_x_base + 5u);
      const float m6 = d8f_mid_value_range(pack, rec.scale_offset, slot_mid, local_x_base + 6u, global_x_base + 6u);
      const float m7 = d8f_mid_value_range(pack, rec.scale_offset, slot_mid, local_x_base + 7u, global_x_base + 7u);
      for (uint rr = 0u; rr < 16u; rr++) {
        const uint row = row_base + rr;
        if (row >= args.rows) continue;
        const ulong block_index = ulong(row) * ulong(blocks_per_row) + ulong(block_col);
        const ulong bit_off = block_index * ulong(rec.bits);
        const ulong byte_off = bit_off >> 3;
        const uint shift = uint(bit_off & 7ul);
        device const uchar *ix = pack + rec.index_offset + byte_off;
        const uint w = uint(ix[0]) | (uint(ix[1]) << 8u) | (uint(ix[2]) << 16u) | (uint(ix[3]) << 24u);
        const uint code = (w >> shift) & rec.mask;
        const device half *cb = (const device half *)(pack + rec.codebook_offset + ulong(code) * 16ul);
        acc[rr] += rw * (float(cb[0]) * m0 + float(cb[1]) * m1 +
                         float(cb[2]) * m2 + float(cb[3]) * m3 +
                         float(cb[4]) * m4 + float(cb[5]) * m5 +
                         float(cb[6]) * m6 + float(cb[7]) * m7);
      }
    }
  }
  for (uint rr = 0u; rr < 16u; rr++) {
    acc[rr] = simd_sum(acc[rr]);
    if (tiisg == 0u) partial[rr * 8u + sgitg] = acc[rr];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    for (uint rr = 0u; rr < 16u; rr++) {
      const uint row = row_base + rr;
      if (row < args.rows) {
        float total = 0.0f;
        for (uint sg = 0u; sg < 8u; sg++) total += partial[rr * 8u + sg];
        out[ulong(token) * ulong(args.out_token_stride) + ulong(row)] += total;
      }
    }
  }
}

kernel void d8f_down_sum_selected_weighted_batch_tile16_recbuf_cbsram(
  device const uchar *pack               [[buffer(0)]],
  device const float *mid                [[buffer(1)]],
  device const uint  *selected           [[buffer(2)]],
  device float       *out                [[buffer(3)]],
  constant D8FDownBatchArgs &args        [[buffer(4)]],
  device const float *route_weights      [[buffer(5)]],
  device const D8FRecordLite *recs       [[buffer(6)]],
  device const float *sidecar_dot        [[buffer(7)]],
  threadgroup float *partial             [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  uint2 pos [[threadgroup_position_in_grid]]) {
  (void)sidecar_dot;
  const uint row_base = pos.x << 4;
  const uint token = pos.y;
  if (token >= args.n_tokens) return;
  const uint blocks_per_row = args.in_dim >> 3;
  device const float *token_mid = mid + ulong(token) * ulong(args.mid_token_stride);
  device const float *token_weights = route_weights + ulong(token) * ulong(args.route_token_stride);
  threadgroup half *cb_cache = (threadgroup half *)(partial + 128u);
  float acc[16];
  for (uint rr = 0u; rr < 16u; rr++) acc[rr] = 0.0f;
  for (uint slot = 0u; slot < args.n_selected; slot++) {
    const float rw = token_weights[slot];
    if (rw == 0.0f) continue;
    const uint expert = selected[ulong(token) * ulong(args.selected_token_stride) + ulong(slot)];
    const D8FRecordLite rec = recs[8192u + expert];
    device const uchar *side_rec = d8f_down_sidecar_record(pack, args, expert);
    const uint side_rank = side_rec ? d8f_u32(side_rec + 4) : 0u;
    const uint side_in_dim = side_rec ? d8f_u32(side_rec + 8) : 0u;
    const uint side_out_dim = side_rec ? d8f_u32(side_rec + 12) : 0u;
    const ulong side_u_off = side_rec ? d8f_u64(side_rec + 16) : 0ul;
    const ulong side_a_off = side_rec ? d8f_u64(side_rec + 24) : 0ul;
    const bool side_rank1 = side_rank == 1u && side_in_dim == args.in_dim &&
                            side_out_dim == args.rows && side_u_off != 0ul &&
                            side_a_off != 0ul;
    const uint cbsram_k_cap = min(args.cbsram_k_cap, 1024u);
    const bool use_cb_cache = cbsram_k_cap > 0u && rec.k > 0u && rec.k <= cbsram_k_cap;
    if (use_cb_cache) {
      device const half *cb_src = (const device half *)(pack + rec.codebook_offset);
      const uint cb_vals = rec.k << 3;
      for (uint ci = tid; ci < cb_vals; ci += 256u) cb_cache[ci] = cb_src[ci];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device const half *side_u = side_rank1 ? (const device half *)(pack + side_u_off) : nullptr;
    device const half *side_a = side_rank1 ? (const device half *)(pack + side_a_off) : nullptr;
    const device float *slot_mid = token_mid + ulong(slot) * ulong(args.mid_slot_stride);
    float side_dot = 0.0f;
    for (uint block_col = tid; block_col < blocks_per_row; block_col += 256u) {
      const uint x_base = block_col << 3;
      const float m0 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 0u);
      const float m1 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 1u);
      const float m2 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 2u);
      const float m3 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 3u);
      const float m4 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 4u);
      const float m5 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 5u);
      const float m6 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 6u);
      const float m7 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 7u);
      if (side_rank1) {
        side_dot += float(side_u[x_base + 0u]) * m0 + float(side_u[x_base + 1u]) * m1 +
                    float(side_u[x_base + 2u]) * m2 + float(side_u[x_base + 3u]) * m3 +
                    float(side_u[x_base + 4u]) * m4 + float(side_u[x_base + 5u]) * m5 +
                    float(side_u[x_base + 6u]) * m6 + float(side_u[x_base + 7u]) * m7;
      }
      for (uint rr = 0u; rr < 16u; rr++) {
        const uint row = row_base + rr;
        if (row >= args.rows) continue;
        const ulong block_index = ulong(row) * ulong(blocks_per_row) + ulong(block_col);
        const ulong bit_off = block_index * ulong(rec.bits);
        const ulong byte_off = bit_off >> 3;
        const uint shift = uint(bit_off & 7ul);
        device const uchar *ix = pack + rec.index_offset + byte_off;
        const uint w = uint(ix[0]) | (uint(ix[1]) << 8u) | (uint(ix[2]) << 16u) | (uint(ix[3]) << 24u);
        const uint code = (w >> shift) & rec.mask;
        if (code >= rec.k) continue;
        if (use_cb_cache) {
          threadgroup const half *cb = cb_cache + ulong(code) * 8ul;
          acc[rr] += rw * (float(cb[0]) * m0 + float(cb[1]) * m1 +
                           float(cb[2]) * m2 + float(cb[3]) * m3 +
                           float(cb[4]) * m4 + float(cb[5]) * m5 +
                           float(cb[6]) * m6 + float(cb[7]) * m7);
        } else {
          const device half *cb = (const device half *)(pack + rec.codebook_offset + ulong(code) * 16ul);
          acc[rr] += rw * (float(cb[0]) * m0 + float(cb[1]) * m1 +
                           float(cb[2]) * m2 + float(cb[3]) * m3 +
                           float(cb[4]) * m4 + float(cb[5]) * m5 +
                           float(cb[6]) * m6 + float(cb[7]) * m7);
        }
      }
    }
    if (side_rank1) {
      side_dot = simd_sum(side_dot);
      if (tiisg == 0u) partial[sgitg] = side_dot;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid == 0u) {
        float dot_total = 0.0f;
        for (uint sg = 0u; sg < 8u; sg++) dot_total += partial[sg];
        for (uint rr = 0u; rr < 16u; rr++) {
          const uint row = row_base + rr;
          if (row < args.rows) acc[rr] += rw * float(side_a[row]) * dot_total;
        }
      }
    }
    if (use_cb_cache || side_rank1) threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint rr = 0u; rr < 16u; rr++) {
    acc[rr] = simd_sum(acc[rr]);
    if (tiisg == 0u) partial[rr * 8u + sgitg] = acc[rr];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    for (uint rr = 0u; rr < 16u; rr++) {
      const uint row = row_base + rr;
      if (row < args.rows) {
        float total = 0.0f;
        for (uint sg = 0u; sg < 8u; sg++) total += partial[rr * 8u + sg];
        out[ulong(token) * ulong(args.out_token_stride) + ulong(row)] = total;
      }
    }
  }
}

kernel void d8f_down_sum_selected_weighted_batch_tile32_recbuf(
  device const uchar *pack               [[buffer(0)]],
  device const float *mid                [[buffer(1)]],
  device const uint  *selected           [[buffer(2)]],
  device float       *out                [[buffer(3)]],
  constant D8FDownBatchArgs &args        [[buffer(4)]],
  device const float *route_weights      [[buffer(5)]],
  device const D8FRecordLite *recs       [[buffer(6)]],
  device const float *sidecar_dot        [[buffer(7)]],
  threadgroup float *partial             [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  uint2 pos [[threadgroup_position_in_grid]]) {
  (void)sidecar_dot;
  const uint row_base = pos.x << 5;
  const uint token = pos.y;
  if (token >= args.n_tokens) return;
  const uint blocks_per_row = args.in_dim >> 3;
  device const float *token_mid = mid + ulong(token) * ulong(args.mid_token_stride);
  device const float *token_weights = route_weights + ulong(token) * ulong(args.route_token_stride);
  float acc[32];
  for (uint rr = 0u; rr < 32u; rr++) acc[rr] = 0.0f;
  for (uint slot = 0u; slot < args.n_selected; slot++) {
    const float rw = token_weights[slot];
    if (rw == 0.0f) continue;
    const uint expert = selected[ulong(token) * ulong(args.selected_token_stride) + ulong(slot)];
    const D8FRecordLite rec = recs[8192u + expert];
    device const uchar *side_rec = d8f_down_sidecar_record(pack, args, expert);
    const uint side_rank = side_rec ? d8f_u32(side_rec + 4) : 0u;
    const uint side_in_dim = side_rec ? d8f_u32(side_rec + 8) : 0u;
    const uint side_out_dim = side_rec ? d8f_u32(side_rec + 12) : 0u;
    const ulong side_u_off = side_rec ? d8f_u64(side_rec + 16) : 0ul;
    const ulong side_a_off = side_rec ? d8f_u64(side_rec + 24) : 0ul;
    const bool side_rank1 = side_rank == 1u && side_in_dim == args.in_dim &&
                            side_out_dim == args.rows && side_u_off != 0ul &&
                            side_a_off != 0ul;
    device const half *side_u = side_rank1 ? (const device half *)(pack + side_u_off) : nullptr;
    device const half *side_a = side_rank1 ? (const device half *)(pack + side_a_off) : nullptr;
    const device float *slot_mid = token_mid + ulong(slot) * ulong(args.mid_slot_stride);
    float side_dot = 0.0f;
    for (uint block_col = tid; block_col < blocks_per_row; block_col += 256u) {
      const uint x_base = block_col << 3;
      const float m0 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 0u);
      const float m1 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 1u);
      const float m2 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 2u);
      const float m3 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 3u);
      const float m4 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 4u);
      const float m5 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 5u);
      const float m6 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 6u);
      const float m7 = d8f_mid_value(pack, rec.scale_offset, slot_mid, x_base + 7u);
      if (side_rank1) {
        side_dot += float(side_u[x_base + 0u]) * m0 + float(side_u[x_base + 1u]) * m1 +
                    float(side_u[x_base + 2u]) * m2 + float(side_u[x_base + 3u]) * m3 +
                    float(side_u[x_base + 4u]) * m4 + float(side_u[x_base + 5u]) * m5 +
                    float(side_u[x_base + 6u]) * m6 + float(side_u[x_base + 7u]) * m7;
      }
      for (uint rr = 0u; rr < 32u; rr++) {
        const uint row = row_base + rr;
        if (row >= args.rows) continue;
        const ulong block_index = ulong(row) * ulong(blocks_per_row) + ulong(block_col);
        const ulong bit_off = block_index * ulong(rec.bits);
        const ulong byte_off = bit_off >> 3;
        const uint shift = uint(bit_off & 7ul);
        device const uchar *ix = pack + rec.index_offset + byte_off;
        const uint w = uint(ix[0]) | (uint(ix[1]) << 8u) | (uint(ix[2]) << 16u) | (uint(ix[3]) << 24u);
        const uint code = (w >> shift) & rec.mask;
        const device half *cb = (const device half *)(pack + rec.codebook_offset + ulong(code) * 16ul);
        acc[rr] += rw * (float(cb[0]) * m0 + float(cb[1]) * m1 +
                         float(cb[2]) * m2 + float(cb[3]) * m3 +
                         float(cb[4]) * m4 + float(cb[5]) * m5 +
                         float(cb[6]) * m6 + float(cb[7]) * m7);
      }
    }
    if (side_rank1) {
      side_dot = simd_sum(side_dot);
      if (tiisg == 0u) partial[sgitg] = side_dot;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid == 0u) {
        float dot_total = 0.0f;
        for (uint sg = 0u; sg < 8u; sg++) dot_total += partial[sg];
        for (uint rr = 0u; rr < 32u; rr++) {
          const uint row = row_base + rr;
          if (row < args.rows) acc[rr] += rw * float(side_a[row]) * dot_total;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  for (uint rr = 0u; rr < 32u; rr++) {
    acc[rr] = simd_sum(acc[rr]);
    if (tiisg == 0u) partial[rr * 8u + sgitg] = acc[rr];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    for (uint rr = 0u; rr < 32u; rr++) {
      const uint row = row_base + rr;
      if (row < args.rows) {
        float total = 0.0f;
        for (uint sg = 0u; sg < 8u; sg++) total += partial[rr * 8u + sg];
        out[ulong(token) * ulong(args.out_token_stride) + ulong(row)] = total;
      }
    }
  }
}


kernel void d8f_gateup_swiglu_selected_batch_recbuf_hmid(
  device const uchar *pack          [[buffer(0)]],
  device const float *x_base_all    [[buffer(1)]],
  device const uint  *selected      [[buffer(2)]],
  device half        *mid           [[buffer(3)]],
  constant D8FGateUpBatchArgs &args [[buffer(4)]],
  device const D8FRecordLite *recs  [[buffer(5)]],
  threadgroup float *partial        [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  uint3 pos [[threadgroup_position_in_grid]]) {
  const uint row = pos.x;
  const uint slot = pos.y;
  const uint token = pos.z;
  if (row >= args.rows || slot >= args.n_selected || token >= args.n_tokens) return;
  device const float *x = x_base_all + ulong(token) * ulong(args.x_token_stride);
  const uint expert = selected[ulong(token) * ulong(args.selected_token_stride) + ulong(slot)];
  const uint row_block = row >> 7;
  const D8FRecordLite gate_rec = recs[expert * 16u + row_block];
  const D8FRecordLite up_rec = recs[4096u + expert * 16u + row_block];
  const uint blocks_per_row = args.in_dim >> 3;
  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  for (uint block_col = tid; block_col < blocks_per_row; block_col += 256u) {
    const ulong gate_block_index = ulong(row - gate_rec.row_base) * ulong(blocks_per_row) + ulong(block_col);
    const ulong up_block_index = ulong(row - up_rec.row_base) * ulong(blocks_per_row) + ulong(block_col);
    const uint x_base = block_col << 3;
    const float x0 = x[x_base + 0u];
    const float x1 = x[x_base + 1u];
    const float x2 = x[x_base + 2u];
    const float x3 = x[x_base + 3u];
    const float x4 = x[x_base + 4u];
    const float x5 = x[x_base + 5u];
    const float x6 = x[x_base + 6u];
    const float x7 = x[x_base + 7u];
    const ulong gate_bit_off = gate_block_index * ulong(gate_rec.bits);
    const ulong gate_byte_off = gate_bit_off >> 3;
    const uint gate_shift = uint(gate_bit_off & 7ul);
    device const uchar *gate_ix = pack + gate_rec.index_offset + gate_byte_off;
    const uint gate_w = uint(gate_ix[0]) | (uint(gate_ix[1]) << 8u) | (uint(gate_ix[2]) << 16u) | (uint(gate_ix[3]) << 24u);
    const uint gate_code = (gate_w >> gate_shift) & gate_rec.mask;
    const device half *gate_cb = (const device half *)(pack + gate_rec.codebook_offset + ulong(gate_code) * 16ul);
    gate_acc += float(gate_cb[0]) * x0 + float(gate_cb[1]) * x1 + float(gate_cb[2]) * x2 + float(gate_cb[3]) * x3 +
                float(gate_cb[4]) * x4 + float(gate_cb[5]) * x5 + float(gate_cb[6]) * x6 + float(gate_cb[7]) * x7;
    const ulong up_bit_off = up_block_index * ulong(up_rec.bits);
    const ulong up_byte_off = up_bit_off >> 3;
    const uint up_shift = uint(up_bit_off & 7ul);
    device const uchar *up_ix = pack + up_rec.index_offset + up_byte_off;
    const uint up_w = uint(up_ix[0]) | (uint(up_ix[1]) << 8u) | (uint(up_ix[2]) << 16u) | (uint(up_ix[3]) << 24u);
    const uint up_code = (up_w >> up_shift) & up_rec.mask;
    const device half *up_cb = (const device half *)(pack + up_rec.codebook_offset + ulong(up_code) * 16ul);
    up_acc += float(up_cb[0]) * x0 + float(up_cb[1]) * x1 + float(up_cb[2]) * x2 + float(up_cb[3]) * x3 +
              float(up_cb[4]) * x4 + float(up_cb[5]) * x5 + float(up_cb[6]) * x6 + float(up_cb[7]) * x7;
  }
  gate_acc = simd_sum(gate_acc);
  up_acc = simd_sum(up_acc);
  if (tiisg == 0u) { partial[sgitg] = gate_acc; partial[8u + sgitg] = up_acc; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float gate_total = 0.0f;
    float up_total = 0.0f;
    for (uint sg = 0u; sg < 8u; sg++) { gate_total += partial[sg]; up_total += partial[8u + sg]; }
    gate_total = min(gate_total, args.clamp);
    up_total = clamp(up_total, -args.clamp, args.clamp);
    mid[ulong(token) * ulong(args.mid_token_stride) + ulong(slot) * ulong(args.mid_slot_stride) + ulong(row)] =
      half((gate_total / (1.0f + exp(-gate_total))) * up_total);
  }
}

kernel void d8f_down_sum_selected_weighted_batch_tile16_recbuf_hmid(
  device const uchar *pack               [[buffer(0)]],
  device const half  *mid                [[buffer(1)]],
  device const uint  *selected           [[buffer(2)]],
  device float       *out                [[buffer(3)]],
  constant D8FDownBatchArgs &args        [[buffer(4)]],
  device const float *route_weights      [[buffer(5)]],
  device const D8FRecordLite *recs       [[buffer(6)]],
  threadgroup float *partial             [[threadgroup(0)]],
  uint tid [[thread_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  uint2 pos [[threadgroup_position_in_grid]]) {
  const uint row_base = pos.x << 4;
  const uint token = pos.y;
  if (token >= args.n_tokens) return;
  const uint blocks_per_row = args.in_dim >> 3;
  device const half *token_mid = mid + ulong(token) * ulong(args.mid_token_stride);
  device const float *token_weights = route_weights + ulong(token) * ulong(args.route_token_stride);
  float acc[16];
  for (uint rr = 0u; rr < 16u; rr++) acc[rr] = 0.0f;
  for (uint slot = 0u; slot < args.n_selected; slot++) {
    const float rw = token_weights[slot];
    if (rw == 0.0f) continue;
    const uint expert = selected[ulong(token) * ulong(args.selected_token_stride) + ulong(slot)];
    const D8FRecordLite rec = recs[8192u + expert];
    const device half *slot_mid = token_mid + ulong(slot) * ulong(args.mid_slot_stride);
    for (uint block_col = tid; block_col < blocks_per_row; block_col += 256u) {
      const uint x_base = block_col << 3;
      const float m0 = d8f_mid_value_h(pack, rec.scale_offset, slot_mid, x_base + 0u);
      const float m1 = d8f_mid_value_h(pack, rec.scale_offset, slot_mid, x_base + 1u);
      const float m2 = d8f_mid_value_h(pack, rec.scale_offset, slot_mid, x_base + 2u);
      const float m3 = d8f_mid_value_h(pack, rec.scale_offset, slot_mid, x_base + 3u);
      const float m4 = d8f_mid_value_h(pack, rec.scale_offset, slot_mid, x_base + 4u);
      const float m5 = d8f_mid_value_h(pack, rec.scale_offset, slot_mid, x_base + 5u);
      const float m6 = d8f_mid_value_h(pack, rec.scale_offset, slot_mid, x_base + 6u);
      const float m7 = d8f_mid_value_h(pack, rec.scale_offset, slot_mid, x_base + 7u);
      for (uint rr = 0u; rr < 16u; rr++) {
        const uint row = row_base + rr;
        if (row >= args.rows) continue;
        const ulong block_index = ulong(row) * ulong(blocks_per_row) + ulong(block_col);
        const ulong bit_off = block_index * ulong(rec.bits);
        const ulong byte_off = bit_off >> 3;
        const uint shift = uint(bit_off & 7ul);
        device const uchar *ix = pack + rec.index_offset + byte_off;
        const uint w = uint(ix[0]) | (uint(ix[1]) << 8u) | (uint(ix[2]) << 16u) | (uint(ix[3]) << 24u);
        const uint code = (w >> shift) & rec.mask;
        const device half *cb = (const device half *)(pack + rec.codebook_offset + ulong(code) * 16ul);
        acc[rr] += rw * (float(cb[0]) * m0 + float(cb[1]) * m1 +
                         float(cb[2]) * m2 + float(cb[3]) * m3 +
                         float(cb[4]) * m4 + float(cb[5]) * m5 +
                         float(cb[6]) * m6 + float(cb[7]) * m7);
      }
    }
  }
  for (uint rr = 0u; rr < 16u; rr++) {
    acc[rr] = simd_sum(acc[rr]);
    if (tiisg == 0u) partial[rr * 8u + sgitg] = acc[rr];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    for (uint rr = 0u; rr < 16u; rr++) {
      const uint row = row_base + rr;
      if (row < args.rows) {
        float total = 0.0f;
        for (uint sg = 0u; sg < 8u; sg++) total += partial[rr * 8u + sg];
        out[ulong(token) * ulong(args.out_token_stride) + ulong(row)] = total;
      }
    }
  }
}
