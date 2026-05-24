#ifndef DS4_GPU_H
#define DS4_GPU_H

#include <stdbool.h>
#include <stdint.h>

/* =========================================================================
 * GPU Tensor and Command Lifetime.
 * =========================================================================
 *
 * Opaque device tensor used by the DS4-specific GPU executor.
 *
 * The public GPU API is tensor-resident: activations, KV state, and scratch
 * buffers stay device-owned across the whole prefill/decode command sequence.
 */
typedef struct ds4_gpu_tensor ds4_gpu_tensor;

int ds4_gpu_init(void);
void ds4_gpu_cleanup(void);

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes);
ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes);
ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes);
void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor);
uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor);
void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor);
int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count);
int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);
int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);
int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
 const ds4_gpu_tensor *src, uint64_t src_offset,
 uint64_t bytes);
int ds4_gpu_tensor_copy_f32_to_f16(ds4_gpu_tensor *dst, uint64_t dst_offset,
 const ds4_gpu_tensor *src, uint64_t src_offset,
 uint64_t count);

int ds4_gpu_begin_commands(void);
int ds4_gpu_flush_commands(void);
int ds4_gpu_end_commands(void);
int ds4_gpu_synchronize(void);

int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size);
int ds4_gpu_set_model_fd(int fd);
int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size);
int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label);
int ds4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, uint64_t in_dim, uint64_t out_dim, const char *label);
int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes);
void ds4_gpu_set_quality(bool quality);
void ds4_gpu_print_memory_report(const char *label);

/* =========================================================================
 * Embeddings and Indexer Helpers.
 * =========================================================================
 *
 * These kernels seed HC state from token embeddings and implement the ratio-4
 * compressed-attention indexer that chooses visible compressed rows.
 */

int ds4_gpu_embed_token_hc_tensor(
 ds4_gpu_tensor *out_hc,
 const void *model_map,
 uint64_t model_size,
 uint64_t weight_offset,
 uint32_t n_vocab,
 uint32_t token,
 uint32_t n_embd,
 uint32_t n_hc);

int ds4_gpu_embed_tokens_hc_tensor(
 ds4_gpu_tensor *out_hc,
 const ds4_gpu_tensor *tokens,
 const void *model_map,
 uint64_t model_size,
 uint64_t weight_offset,
 uint32_t n_vocab,
 uint32_t n_tokens,
 uint32_t n_embd,
 uint32_t n_hc);

int ds4_gpu_indexer_score_one_tensor(
 ds4_gpu_tensor *scores,
 const ds4_gpu_tensor *q,
 const ds4_gpu_tensor *weights,
 const ds4_gpu_tensor *index_comp,
 uint32_t n_comp,
 uint32_t n_head,
 uint32_t head_dim,
 float scale);

int ds4_gpu_indexer_scores_prefill_tensor(
 ds4_gpu_tensor *scores,
 const ds4_gpu_tensor *q,
 const ds4_gpu_tensor *weights,
 const ds4_gpu_tensor *index_comp,
 uint32_t n_comp,
 uint32_t n_tokens,
 uint32_t n_head,
 uint32_t head_dim,
 uint32_t ratio,
 float scale);

int ds4_gpu_indexer_scores_decode_batch_tensor(
 ds4_gpu_tensor *scores,
 const ds4_gpu_tensor *q,
 const ds4_gpu_tensor *weights,
 const ds4_gpu_tensor *index_comp,
 uint32_t n_comp,
 uint32_t n_tokens,
 uint32_t pos0,
 uint32_t n_head,
 uint32_t head_dim,
 uint32_t ratio,
 float scale);

int ds4_gpu_indexer_topk_tensor(
 ds4_gpu_tensor *selected,
 const ds4_gpu_tensor *scores,
 uint32_t n_comp,
 uint32_t n_tokens,
 uint32_t top_k);

int ds4_gpu_dsv4_topk_mask_tensor(
 ds4_gpu_tensor *mask,
 const ds4_gpu_tensor *topk,
 uint32_t n_comp,
 uint32_t n_tokens,
 uint32_t top_k);

/* =========================================================================
 * Dense Projections, Norms, RoPE, and KV Rounding.
 * =========================================================================
 *
 * The graph uses these primitives for Q/KV projections, HC/output projections,
 * attention output projections, and DS4's tail-only RoPE.
 */

int ds4_gpu_matmul_q8_0_tensor(
 ds4_gpu_tensor *out,
 const void *model_map,
 uint64_t model_size,
 uint64_t weight_offset,
 uint64_t in_dim,
 uint64_t out_dim,
 const ds4_gpu_tensor *x,
 uint64_t n_tok);

int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
 ds4_gpu_tensor *gate,
 ds4_gpu_tensor *up,
 ds4_gpu_tensor *mid,
 const void *model_map,
 uint64_t model_size,
 uint64_t gate_offset,
 uint64_t up_offset,
 uint64_t in_dim,
 uint64_t out_dim,
 const ds4_gpu_tensor *x,
 float clamp);

int ds4_gpu_matmul_f16_tensor(
 ds4_gpu_tensor *out,
 const void *model_map,
 uint64_t model_size,
 uint64_t weight_offset,
 uint64_t in_dim,
 uint64_t out_dim,
 const ds4_gpu_tensor *x,
 uint64_t n_tok);

int ds4_gpu_matmul_f16_pair_tensor(
 ds4_gpu_tensor *out_a,
 ds4_gpu_tensor *out_b,
 const void *model_map,
 uint64_t model_size,
 uint64_t weight_a_offset,
 uint64_t weight_b_offset,
 uint64_t in_dim,
 uint64_t out_dim,
 const ds4_gpu_tensor *x,
 uint64_t n_tok);

int ds4_gpu_matmul_f32_tensor(
 ds4_gpu_tensor *out,
 const void *model_map,
 uint64_t model_size,
 uint64_t weight_offset,
 uint64_t in_dim,
 uint64_t out_dim,
 const ds4_gpu_tensor *x,
 uint64_t n_tok);

int ds4_gpu_repeat_hc_tensor(
 ds4_gpu_tensor *out,
 const ds4_gpu_tensor *row,
 uint32_t n_embd,
 uint32_t n_hc);

int ds4_gpu_rms_norm_plain_tensor(
 ds4_gpu_tensor *out,
 const ds4_gpu_tensor *x,
 uint32_t n,
 float eps);

int ds4_gpu_rms_norm_plain_rows_tensor(
 ds4_gpu_tensor *out,
 const ds4_gpu_tensor *x,
 uint32_t n,
 uint32_t rows,
 float eps);

int ds4_gpu_rms_norm_weight_tensor(
 ds4_gpu_tensor *out,
 const ds4_gpu_tensor *x,
 const void *model_map,
 uint64_t model_size,
 uint64_t weight_offset,
 uint32_t n,
 float eps);

int ds4_gpu_rms_norm_weight_rows_tensor(
 ds4_gpu_tensor *out,
 const ds4_gpu_tensor *x,
 const void *model_map,
 uint64_t model_size,
 uint64_t weight_offset,
 uint32_t n,
 uint32_t rows,
 float eps);

int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
 ds4_gpu_tensor *q_out,
 const ds4_gpu_tensor *q,
 const void *model_map,
 uint64_t model_size,
 uint64_t q_weight_offset,
 uint32_t q_n,
 ds4_gpu_tensor *kv_out,
 const ds4_gpu_tensor *kv,
 uint64_t kv_weight_offset,
 uint32_t kv_n,
 uint32_t rows,
 float eps);

int ds4_gpu_head_rms_norm_tensor(
 ds4_gpu_tensor *x,
 uint32_t n_tok,
 uint32_t n_head,
 uint32_t head_dim,
 float eps);

int ds4_gpu_dsv4_fp8_kv_quantize_tensor(
 ds4_gpu_tensor *x,
 uint32_t n_tok,
 uint32_t head_dim,
 uint32_t n_rot);

int ds4_gpu_dsv4_indexer_qat_tensor(
 ds4_gpu_tensor *x,
 uint32_t n_rows,
 uint32_t head_dim);

int ds4_gpu_rope_tail_tensor(
 ds4_gpu_tensor *x,
 uint32_t n_tok,
 uint32_t n_head,
 uint32_t head_dim,
 uint32_t n_rot,
 uint32_t pos0,
 uint32_t n_ctx_orig,
 bool inverse,
 float freq_base,
 float freq_scale,
 float ext_factor,
 float attn_factor,
 float beta_fast,
 float beta_slow);

/* Release decode fused KV finalizer: after the standalone RoPE kernel, this
 * performs DS4's FP8 non-RoPE KV round trip and writes the F16-rounded raw
 * attention cache row in one dispatch. */
int ds4_gpu_kv_fp8_store_raw_tensor(
 ds4_gpu_tensor *kv,
 ds4_gpu_tensor *raw_cache,
 uint32_t raw_cap,
 uint32_t row,
 uint32_t head_dim,
 uint32_t n_rot);

/* Reference/raw-cache primitive kept for prefill and diagnostics. Decode uses
 * ds4_gpu_kv_fp8_store_raw_tensor unless a diagnostic reference path is
 * explicitly selected by the graph driver. */
int ds4_gpu_store_raw_kv_tensor(
 ds4_gpu_tensor *raw_cache,
 const ds4_gpu_tensor *kv,
 uint32_t raw_cap,
 uint32_t row,
 uint32_t head_dim);

int ds4_gpu_store_raw_kv_batch_tensor(
 ds4_gpu_tensor *raw_cache,
 const ds4_gpu_tensor *kv,
 uint32_t raw_cap,
 uint32_t pos0,
 uint32_t n_tokens,
 uint32_t head_dim);

/* =========================================================================
 * KV Compression and Attention.
 * =========================================================================
 *
 * Compressed layers maintain rolling score/KV state and append pooled rows at
 * ratio boundaries. Attention kernels consume raw SWA rows, compressed rows,
 * and optional indexer masks.
 */

int ds4_gpu_compressor_update_tensor(
 const ds4_gpu_tensor *kv_cur,
 const ds4_gpu_tensor *sc_cur,
 ds4_gpu_tensor *state_kv,
 ds4_gpu_tensor *state_score,
 ds4_gpu_tensor *comp_cache,
 const void *model_map,
 uint64_t model_size,
 uint64_t ape_offset,
 uint32_t ape_type,
 uint64_t norm_offset,
 uint32_t norm_type,
 uint32_t head_dim,
 uint32_t ratio,
 uint32_t pos,
 uint32_t comp_row,
 uint32_t n_rot,
 uint32_t n_ctx_orig,
 float freq_base,
 float freq_scale,
 float ext_factor,
 float attn_factor,
 float beta_fast,
 float beta_slow,
 float rms_eps);

int ds4_gpu_compressor_store_batch_tensor(
 const ds4_gpu_tensor *kv,
 const ds4_gpu_tensor *sc,
 ds4_gpu_tensor *state_kv,
 ds4_gpu_tensor *state_score,
 const void *model_map,
 uint64_t model_size,
 uint64_t ape_offset,
 uint32_t ape_type,
 uint32_t head_dim,
 uint32_t ratio,
 uint32_t pos0,
 uint32_t n_tokens);

int ds4_gpu_compressor_prefill_tensor(
 ds4_gpu_tensor *comp_cache,
 ds4_gpu_tensor *state_kv,
 ds4_gpu_tensor *state_score,
 const ds4_gpu_tensor *kv,
 const ds4_gpu_tensor *sc,
 const void *model_map,
 uint64_t model_size,
 uint64_t ape_offset,
 uint32_t ape_type,
 uint64_t norm_offset,
 uint32_t norm_type,
 uint32_t head_dim,
 uint32_t ratio,
 uint32_t pos0,
 uint32_t n_tokens,
 uint32_t n_rot,
 uint32_t n_ctx_orig,
 bool quantize_fp8,
 float freq_base,
 float freq_scale,
 float ext_factor,
 float attn_factor,
 float beta_fast,
 float beta_slow,
 float rms_eps);

int ds4_gpu_compressor_prefill_ratio4_replay_tensor(
 ds4_gpu_tensor *comp_cache,
 ds4_gpu_tensor *state_kv,
 ds4_gpu_tensor *state_score,
 const ds4_gpu_tensor *kv,
 const ds4_gpu_tensor *sc,
 const void *model_map,
 uint64_t model_size,
 uint64_t ape_offset,
 uint32_t ape_type,
 uint64_t norm_offset,
 uint32_t norm_type,
 uint32_t head_dim,
 uint32_t pos0,
 uint32_t n_tokens,
 uint32_t n_rot,
 uint32_t n_ctx_orig,
 bool quantize_fp8,
 float freq_base,
 float freq_scale,
 float ext_factor,
 float attn_factor,
 float beta_fast,
 float beta_slow,
 float rms_eps);

int ds4_gpu_compressor_prefill_state_ratio4_tensor(
 ds4_gpu_tensor *state_kv,
 ds4_gpu_tensor *state_score,
 const ds4_gpu_tensor *kv_tail,
 const ds4_gpu_tensor *sc_tail,
 const void *model_map,
 uint64_t model_size,
 uint64_t ape_offset,
 uint32_t ape_type,
 uint32_t head_dim,
 uint32_t pos0);

int ds4_gpu_attention_decode_heads_tensor(
 ds4_gpu_tensor *heads,
 const void *model_map,
 uint64_t model_size,
 uint64_t sinks_offset,
 const ds4_gpu_tensor *q,
 const ds4_gpu_tensor *raw_kv,
 uint32_t n_raw,
 uint32_t raw_cap,
 uint32_t raw_start,
 const ds4_gpu_tensor *comp_kv,
 uint32_t comp_kv_f16,
 uint32_t n_comp,
 const ds4_gpu_tensor *comp_mask,
 uint32_t use_mask,
 uint32_t n_head,
 uint32_t head_dim);

int ds4_gpu_attention_prefill_raw_heads_tensor(
 ds4_gpu_tensor *heads,
 const void *model_map,
 uint64_t model_size,
 uint64_t sinks_offset,
 const ds4_gpu_tensor *q,
 const ds4_gpu_tensor *raw_kv,
 uint32_t n_tokens,
 uint32_t window,
 uint32_t n_head,
 uint32_t head_dim);

int ds4_gpu_attention_decode_raw_batch_heads_tensor(
 ds4_gpu_tensor *heads,
 const void *model_map,
 uint64_t model_size,
 uint64_t sinks_offset,
 const ds4_gpu_tensor *q,
 const ds4_gpu_tensor *raw_kv,
 uint32_t n_tokens,
 uint32_t pos0,
 uint32_t n_raw,
 uint32_t raw_cap,
 uint32_t raw_start,
 uint32_t window,
 uint32_t n_head,
 uint32_t head_dim);

int ds4_gpu_attention_decode_mixed_batch_heads_tensor(
 ds4_gpu_tensor *heads,
 const void *model_map,
 uint64_t model_size,
 uint64_t sinks_offset,
 const ds4_gpu_tensor *q,
 const ds4_gpu_tensor *raw_kv,
 const ds4_gpu_tensor *comp_kv,
 uint32_t comp_kv_f16,
 const ds4_gpu_tensor *comp_mask,
 uint32_t use_comp_mask,
 uint32_t n_tokens,
 uint32_t pos0,
 uint32_t n_raw,
 uint32_t raw_cap,
 uint32_t raw_start,
 uint32_t n_comp,
 uint32_t window,
 uint32_t ratio,
 uint32_t n_head,
 uint32_t head_dim);

int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
 ds4_gpu_tensor *heads,
 const void *model_map,
 uint64_t model_size,
 uint64_t sinks_offset,
 const ds4_gpu_tensor *q,
 const ds4_gpu_tensor *raw_kv,
 const ds4_gpu_tensor *comp_kv,
 uint32_t comp_kv_f16,
 const ds4_gpu_tensor *topk,
 uint32_t n_tokens,
 uint32_t pos0,
 uint32_t n_raw,
 uint32_t raw_cap,
 uint32_t raw_start,
 uint32_t n_comp,
 uint32_t top_k,
 uint32_t window,
 uint32_t ratio,
 uint32_t n_head,
 uint32_t head_dim);

int ds4_gpu_attention_prefill_static_mixed_heads_tensor(
 ds4_gpu_tensor *heads,
 const void *model_map,
 uint64_t model_size,
 uint64_t sinks_offset,
 const ds4_gpu_tensor *q,
 const ds4_gpu_tensor *raw_kv,
 const ds4_gpu_tensor *comp_kv,
 uint32_t comp_kv_f16,
 uint32_t n_tokens,
 uint32_t n_comp,
 uint32_t window,
 uint32_t ratio,
 uint32_t n_head,
 uint32_t head_dim);

int ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
 ds4_gpu_tensor *heads,
 const void *model_map,
 uint64_t model_size,
 uint64_t sinks_offset,
 const ds4_gpu_tensor *q,
 const ds4_gpu_tensor *raw_kv,
 const ds4_gpu_tensor *comp_kv,
 uint32_t comp_kv_f16,
 const ds4_gpu_tensor *comp_mask,
 uint32_t n_tokens,
 uint32_t n_comp,
 uint32_t window,
 uint32_t ratio,
 uint32_t n_head,
 uint32_t head_dim);

int ds4_gpu_attention_output_q8_batch_tensor(
 ds4_gpu_tensor *out,
 ds4_gpu_tensor *low,
 ds4_gpu_tensor *group_tmp,
 ds4_gpu_tensor *low_tmp,
 const void *model_map,
 uint64_t model_size,
 uint64_t out_a_offset,
 uint64_t out_b_offset,
 uint64_t group_dim,
 uint64_t rank,
 uint32_t n_groups,
 uint64_t out_dim,
 const ds4_gpu_tensor *heads,
 uint32_t n_tokens);

int ds4_gpu_attention_output_low_q8_tensor(
 ds4_gpu_tensor *low,
 const void *model_map,
 uint64_t model_size,
 uint64_t out_a_offset,
 uint64_t group_dim,
 uint64_t rank,
 uint32_t n_groups,
 const ds4_gpu_tensor *heads);

/* =========================================================================
 * Router, Shared Expert, and Routed MoE.
 * =========================================================================
 *
 * These kernels implement the FFN body: router probabilities/top-k or hash
 * routing, shared SwiGLU, and the IQ2_XXS/Q2_K/Q4_K routed experts.
 */

int ds4_gpu_swiglu_tensor(
 ds4_gpu_tensor *out,
 const ds4_gpu_tensor *gate,
 const ds4_gpu_tensor *up,
 uint32_t n,
 float clamp,
 float weight);

int ds4_gpu_add_tensor(
 ds4_gpu_tensor *out,
 const ds4_gpu_tensor *a,
 const ds4_gpu_tensor *b,
 uint32_t n);

int ds4_gpu_directional_steering_project_tensor(
 ds4_gpu_tensor *x,
 const ds4_gpu_tensor *directions,
 uint32_t layer,
 uint32_t width,
 uint32_t rows,
 float scale);

int ds4_gpu_router_select_tensor(
 ds4_gpu_tensor *selected,
 ds4_gpu_tensor *weights,
 ds4_gpu_tensor *probs,
 const void *model_map,
 uint64_t model_size,
 uint64_t bias_offset,
 uint64_t hash_offset,
 uint32_t hash_rows,
 uint32_t token,
 uint32_t n_expert_groups,
 uint32_t n_group_used,
 bool has_bias,
 bool hash_mode,
 const ds4_gpu_tensor *logits);

int ds4_gpu_router_select_batch_tensor(
 ds4_gpu_tensor *selected,
 ds4_gpu_tensor *weights,
 ds4_gpu_tensor *probs,
 const void *model_map,
 uint64_t model_size,
 uint64_t bias_offset,
 uint64_t hash_offset,
 uint32_t hash_rows,
 uint32_t n_expert_groups,
 uint32_t n_group_used,
 bool has_bias,
 bool hash_mode,
 const ds4_gpu_tensor *logits,
 const ds4_gpu_tensor *tokens,
 uint32_t n_tokens);

int ds4_gpu_routed_moe_one_tensor(
 ds4_gpu_tensor *out,
 ds4_gpu_tensor *gate,
 ds4_gpu_tensor *up,
 ds4_gpu_tensor *mid,
 ds4_gpu_tensor *experts,
 const void *model_map,
 uint64_t model_size,
 uint64_t gate_offset,
 uint64_t up_offset,
 uint64_t down_offset,
 uint32_t gate_type,
 uint32_t down_type,
 uint64_t gate_expert_bytes,
 uint64_t gate_row_bytes,
 uint64_t down_expert_bytes,
 uint64_t down_row_bytes,
 uint32_t expert_in_dim,
 uint32_t expert_mid_dim,
 uint32_t out_dim,
 const ds4_gpu_tensor *selected,
 const ds4_gpu_tensor *weights,
 uint32_t n_expert,
 float clamp,
 const ds4_gpu_tensor *x);

int ds4_gpu_routed_moe_batch_tensor(
 ds4_gpu_tensor *out,
 ds4_gpu_tensor *gate,
 ds4_gpu_tensor *up,
 ds4_gpu_tensor *mid,
 ds4_gpu_tensor *experts,
 const void *model_map,
 uint64_t model_size,
 uint64_t gate_offset,
 uint64_t up_offset,
 uint64_t down_offset,
 uint32_t gate_type,
 uint32_t down_type,
 uint64_t gate_expert_bytes,
 uint64_t gate_row_bytes,
 uint64_t down_expert_bytes,
 uint64_t down_row_bytes,
 uint32_t expert_in_dim,
 uint32_t expert_mid_dim,
 uint32_t out_dim,
 const ds4_gpu_tensor *selected,
 const ds4_gpu_tensor *weights,
 uint32_t n_expert,
 float clamp,
 const ds4_gpu_tensor *x,
 uint32_t layer_index,
 uint32_t n_tokens,
 bool *mid_is_f16);

/* MTL4 ML packed-MoE entry point (group=6 for DS4 active-experts).
 * Same signature as ds4_gpu_routed_moe_batch_tensor + one out-param:
 * mtl4_path_taken: written with true if MTL4 ML packed path was used;
 * false if fallback to legacy. May be NULL.
 * Env-gated via DS4_MTL4_MOE_ENABLE. Preflight: n_expert==6, n_tokens≤16. */
int ds4_gpu_routed_moe_batch_tensor_mtl4(
 ds4_gpu_tensor *out,
 ds4_gpu_tensor *gate,
 ds4_gpu_tensor *up,
 ds4_gpu_tensor *mid,
 ds4_gpu_tensor *experts,
 const void *model_map,
 uint64_t model_size,
 uint64_t gate_offset,
 uint64_t up_offset,
 uint64_t down_offset,
 uint32_t gate_type,
 uint32_t down_type,
 uint64_t gate_expert_bytes,
 uint64_t gate_row_bytes,
 uint64_t down_expert_bytes,
 uint64_t down_row_bytes,
 uint32_t expert_in_dim,
 uint32_t expert_mid_dim,
 uint32_t out_dim,
 const ds4_gpu_tensor *selected,
 const ds4_gpu_tensor *weights,
 uint32_t n_expert,
 float clamp,
 const ds4_gpu_tensor *x,
 uint32_t layer_index,
 uint32_t n_tokens,
 bool *mid_is_f16,
 bool *mtl4_path_taken);

/* =========================================================================
 * Hyper-Connection Kernels.
 * =========================================================================
 *
 * HC kernels reduce four residual streams before a sublayer and expand the
 * sublayer output back into four streams afterward.
 */

int ds4_gpu_hc_split_sinkhorn_tensor(
 ds4_gpu_tensor *out,
 const ds4_gpu_tensor *mix,
 const void *model_map,
 uint64_t model_size,
 uint64_t scale_offset,
 uint64_t base_offset,
 uint32_t n_hc,
 uint32_t sinkhorn_iters,
 float eps);

int ds4_gpu_hc_weighted_sum_tensor(
 ds4_gpu_tensor *out,
 const ds4_gpu_tensor *residual_hc,
 const ds4_gpu_tensor *weights,
 uint32_t n_embd,
 uint32_t n_hc);

int ds4_gpu_hc_weighted_sum_split_tensor(
 ds4_gpu_tensor *out,
 const ds4_gpu_tensor *residual_hc,
 const ds4_gpu_tensor *split,
 uint32_t n_embd,
 uint32_t n_hc);

/* Release decode fused HC pre-sublayer operation: split the HC mixer and
 * immediately reduce four HC streams into the active 4096-wide sublayer row. */
int ds4_gpu_hc_split_weighted_sum_tensor(
 ds4_gpu_tensor *out,
 ds4_gpu_tensor *split,
 const ds4_gpu_tensor *mix,
 const ds4_gpu_tensor *residual_hc,
 const void *model_map,
 uint64_t model_size,
 uint64_t scale_offset,
 uint64_t base_offset,
 uint32_t n_embd,
 uint32_t n_hc,
 uint32_t sinkhorn_iters,
 float eps);

int ds4_gpu_hc_split_weighted_sum_norm_tensor(
 ds4_gpu_tensor *out,
 ds4_gpu_tensor *norm_out,
 ds4_gpu_tensor *split,
 const ds4_gpu_tensor *mix,
 const ds4_gpu_tensor *residual_hc,
 const void *model_map,
 uint64_t model_size,
 uint64_t scale_offset,
 uint64_t base_offset,
 uint64_t norm_weight_offset,
 uint32_t n_embd,
 uint32_t n_hc,
 uint32_t sinkhorn_iters,
 float eps,
 float norm_eps);

int ds4_gpu_output_hc_weights_tensor(
 ds4_gpu_tensor *out,
 const ds4_gpu_tensor *pre,
 const void *model_map,
 uint64_t model_size,
 uint64_t scale_offset,
 uint64_t base_offset,
 uint32_t n_hc,
 float eps);

int ds4_gpu_hc_expand_tensor(
 ds4_gpu_tensor *out_hc,
 const ds4_gpu_tensor *block_out,
 const ds4_gpu_tensor *residual_hc,
 const ds4_gpu_tensor *post,
 const ds4_gpu_tensor *comb,
 uint32_t n_embd,
 uint32_t n_hc);

int ds4_gpu_hc_expand_split_tensor(
 ds4_gpu_tensor *out_hc,
 const ds4_gpu_tensor *block_out,
 const ds4_gpu_tensor *residual_hc,
 const ds4_gpu_tensor *split,
 uint32_t n_embd,
 uint32_t n_hc);

int ds4_gpu_hc_expand_add_split_tensor(
 ds4_gpu_tensor *out_hc,
 const ds4_gpu_tensor *block_out,
 const ds4_gpu_tensor *block_add,
 const ds4_gpu_tensor *residual_hc,
 const ds4_gpu_tensor *split,
 uint32_t n_embd,
 uint32_t n_hc);

int ds4_gpu_shared_down_hc_expand_q8_0_tensor(
 ds4_gpu_tensor *out_hc,
 ds4_gpu_tensor *shared_out,
 const void *model_map,
 uint64_t model_size,
 uint64_t weight_offset,
 uint64_t in_dim,
 uint64_t out_dim,
 const ds4_gpu_tensor *shared_mid,
 const ds4_gpu_tensor *routed_out,
 const ds4_gpu_tensor *residual_hc,
 const ds4_gpu_tensor *split,
 uint32_t n_embd,
 uint32_t n_hc);

int ds4_gpu_matmul_q8_0_hc_expand_tensor(
 ds4_gpu_tensor *out_hc,
 ds4_gpu_tensor *block_out,
 const void *model_map,
 uint64_t model_size,
 uint64_t weight_offset,
 uint64_t in_dim,
 uint64_t out_dim,
 const ds4_gpu_tensor *x,
 const ds4_gpu_tensor *residual_hc,
 const ds4_gpu_tensor *split,
 uint32_t n_embd,
 uint32_t n_hc);

/* Journal accessors (append-only logging): lazy-init shared
 * journal handle + session id. Defined in ds4.c; called from any
 * subsystem that needs to emit append-only audit rows. Returns NULL /
 * 0 when DS4_JOURNAL_DB env unset OR journal failed to open. */
struct ds4_journal;
struct ds4_journal *ds4_get_journal(void);
int64_t ds4_get_journal_session(void);

/* Inference plumbing port for trim50 files (parallel to
 * pending integration. Rewrites the router-emitted selected_ids tensor in-place from
 * logical IDs to file-storage positions, zeroing weights for experts that
 * were trimmed out. Env-gated via DS4_EXPERT_REMAP_ACTIVE; when env is set
 * AND the model has ds4.expert_remap.<L> metadata, the kernel runs between
 * router_select_batch_tensor and the routed MoE compute. No-op when no trim
 * metadata is loaded (identity remap table).
 *
 * selected, weights: GPU tensors from router_select_batch_tensor
 * layer_index: current decoder layer (0..42)
 * n_expert: top-K count (6 for DS4)
 * n_tokens: batch size
 * renormalize: when true, also dispatches renormalize-weights kernel
 * to preserve the per-token total contribution after
 * zeroing trimmed weights. Default: true.
 * Returns: 1 on success, 0 on failure. */
int ds4_gpu_remap_routed_for_trim(
 ds4_gpu_tensor *selected,
 ds4_gpu_tensor *weights,
 uint32_t layer_index,
 uint32_t n_expert,
 uint32_t n_tokens,
 bool renormalize);

#endif

/* project-side helpers (preserved across upstream merge): */
int ds4_gpu_set_model_map_ranges(const void *model_map, uint64_t model_size,
 const uint64_t *map_offsets, const uint64_t *map_sizes,
 uint32_t n_ranges);
int ds4_gpu_add_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size);
int ds4_gpu_set_skip_next_warmup(int skip);

/* Polar p8_m2 MTL4 dot canary (port of codex H1725). Dispatches a synthetic
 * polar-dot kernel through the M1 Max Metal 4 path: compile MSL via
 * MTL4Compiler, bind buffers via MTL4ArgumentTable .gpuAddress, residency via
 * MTLResidencySet attached to cb/queue, dispatch via MTL4ComputeCommandEncoder,
 * complete via MTL4CommitFeedback. Reports GPU elapsed + max abs error vs the
 * deterministic expected output (pairs × 1.0).
 *
 * Smoke-test surface for task #563 (polar MTL4 inference integration).
 * Returns 1 on success, 0 on failure. */
int ds4_gpu_mtl4_polar_dot_canary(uint32_t packets, uint32_t pairs);

/* H1729 tile×row×batch variant of the polar canary. Stores mag/phase/levels
 * ONCE across all (tile, row) packets; streams `batches` hidden vectors
 * through them. Hidden indexing is (batch, tile), so the same token's hidden
 * state fans out to all `rows` weight rows of a tile. Codex measured b8 =
 * 46.70 ns/output (1.45× over per-packet), b32 = 31.81× layout amortization.
 *
 * Returns 1 on success, 0 on failure. */
int ds4_gpu_mtl4_polar_tile_canary(uint32_t tiles, uint32_t rows,
                                    uint32_t batches, uint32_t pairs);

/* Real-data variant: load mag/phase/levels/hidden/cos_lut/sin_lut from
 * <prefix>.{mag,phase,levels,hidden,cos_lut,sin_lut}.bin files emitted by
 * analyzers/polar_encode_safetensors.py, then dispatch the H1729 tile×row×batch
 * kernel and compare GPU output to <prefix>.expected_polar.bin.
 *
 * Closes the loop on the polar p8_m2 path: real DS4 V4 Flash BF16/FP4 weights
 * → Python polar encoder → MTL4 GPU dot → numerical match against Python ref.
 *
 * Returns 1 on success, 0 on failure. */
int ds4_gpu_mtl4_polar_tile_real(const char *prefix,
                                  uint32_t tiles, uint32_t rows,
                                  uint32_t batches, uint32_t pairs);

/* H1733 fused gate*silu*up*route_weight kernel canary. Computes the full
 * routed-MoE MLP organ in one dispatch over the routed expert subset:
 *   out[batch, route_pair, row] = silu(gate_dot) * up_dot * route_weight[r]
 * where gate_dot/up_dot are polar dots between the (gate_code[r], row) and
 * (up_code[r], row) tiles and the hidden vector for (batch, route_pair).
 *
 * H1731 indexed-code-tile dedup applied: mag/phase/levels are one shared
 * pool of n_codes code tiles; gate_code[r] / up_code[r] select which tile
 * each route_pair uses.
 *
 * Codex measured: b32 = 21.28 ns/equiv row-dot (2.20× vs single row-dots),
 * max_abs vs CPU decoded reference = 7.55e-9.
 *
 * Returns 1 on success, 0 on failure. */
int ds4_gpu_mtl4_polar_fused_canary(uint32_t n_codes, uint32_t route_pairs,
                                     uint32_t rows, uint32_t batches,
                                     uint32_t pairs);

/* H1735 down-projection extension of the H1733 fused gate*silu*up packet.
 *
 * Adds a Q2_K-style down slice multiply on top: for each (batch, route_pair),
 * the threadgroup first computes act[r] = silu(gate)*up*route_weight for
 * r ∈ [0, act_rows), then for tid < down_rows it computes
 *   out[batch, route_pair, tid] = sum_k act[k] * down[route_pair, tid, k]
 * in a single dispatch.
 *
 * H1736/H1738 tile policy: at b8, 16×8 (act × down) hits 22.31 ns/equiv
 * row-dot; at b32, 32×64 hits 21.03 ns. H1737 refuted two-stage
 * materialization; keep this monolithic tiled fusion as-is.
 *
 * Synthetic canary uses mag=0, phase=4 (angle 0), level[0]=1, hidden=1,
 * down=1/act_rows uniform → expected output = pairs² for all out cells.
 * Pass criterion: relative error vs expected < 1e-3.
 *
 * Constraints: act_rows ≤ 64 (MAX_ACT in shared-memory tile), act_rows ≤ rows,
 * down_rows ≤ 256 (must fit in one threadgroup).
 *
 * Returns 1 on success, 0 on failure. */
int ds4_gpu_mtl4_polar_gate_up_down_canary(uint32_t n_codes, uint32_t route_pairs,
                                            uint32_t rows, uint32_t batches,
                                            uint32_t pairs, uint32_t down_rows,
                                            uint32_t act_rows);

/* #563 Phase B-2.2 real-data canary. Loads PLR2 files from <polar_dir> for
 * (layer) gate+up+down, copies expert <expert> rows into MTL buffers, runs
 * the H1735 kernel with hidden=1, route_weight=1, down=1/act_rows.
 *
 * Validates the PLR2 byte format → MTL4 GPU binding pipeline end-to-end:
 *   - byte alignment of mag/phase/levels arrays
 *   - per-expert stride arithmetic
 *   - cos_lut/sin_lut indexing consistency between encoder and kernel
 *
 * Computes the expected output on CPU via ds4_polar_decode_pair_re/im (the
 * same decode formula the encoder uses) and reports max_abs_err vs GPU.
 *
 * Returns 1 if relative error < 1e-3, 0 on any failure (file open, dispatch,
 * tolerance). */
int ds4_gpu_mtl4_polar_real_canary(const char *polar_dir,
                                    uint32_t layer, uint32_t expert,
                                    uint32_t down_rows, uint32_t act_rows);
