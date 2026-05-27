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
int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes);
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
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                token,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *logits);

int ds4_gpu_router_select_batch_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *logits,
        const ds4_gpu_tensor *tokens,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens);

int ds4_gpu_routed_moe_one_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x);

int ds4_gpu_routed_moe_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x,
        uint32_t                layer_index,
        uint32_t                n_tokens,
        bool                   *mid_is_f16);

/* MTL4 ML packed-MoE entry point (group=6 for DS4 active-experts).
 * Same signature as ds4_gpu_routed_moe_batch_tensor + one out-param:
 * mtl4_path_taken: written with true if MTL4 ML packed path was used;
 * false if fallback to legacy. May be NULL.
 * Env-gated via DS4_MTL4_MOE_ENABLE. Preflight: n_expert==6, n_tokens≤16.
 * silv-local; antirez/main has no MTL4 path. */
int ds4_gpu_routed_moe_batch_tensor_mtl4(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x,
        uint32_t                layer_index,
        uint32_t                n_tokens,
        bool                   *mid_is_f16,
        bool                   *mtl4_path_taken);

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

/* silv 2026-05-27 task #654 — MTL4 Hadamard-16 widened apply (deployable
 * fast path). Applies one orthogonal H_16 transform (with 1/sqrt(16)
 * normalization) to an n_rows × n_in FP16 buffer using the widened
 * 256-thread kernel via the MTL4 dispatch path. n_in must be divisible
 * by 16.
 *
 * Synchronous: host_buf is read once at entry, written once at exit.
 * For multi-apply sequences (encode pipeline with multiple transforms)
 * or runtime dispatch-site pre-pass integration, drive the encoder
 * directly via the internal helper.
 *
 * Returns 1 on success, 0 on failure. */
int ds4_gpu_mtl4_hadamard16_apply(_Float16 *host_buf, uint32_t n_rows, uint32_t n_in);

/* silv 2026-05-27 task #653 — MTL4 Hadamard-16 widened-dispatch canary.
 * Allocates an n_rows × n_in FP16 buffer, fills with deterministic random
 * values, applies the widened Hadamard kernel twice (H×H=I), checks
 * recovery within FP16 precision. n_in must be divisible by 16.
 *
 * The "widened" kernel processes 16 blocks of 16 halves per threadgroup
 * (256 threads) — 16× fewer threadgroups than the per-block variant. Plus
 * MTL4 dispatch overhead. Reports max|err|, rel_L2, and dispatch reduction
 * ratio vs the per-block grid.
 *
 * Returns 1 if rel_L2 < 1e-2 (FP16 precision floor for round-trip), 0
 * otherwise. */
int ds4_gpu_mtl4_hadamard16_canary(uint32_t n_rows, uint32_t n_in);

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

/* VQ-2D codec canary: mirror of polar_real_canary but reads VQB1
 * files from <vqb1_dir>/L{LL}_{gate,up,down}.vqb1. Validates the
 * VQ-2D codec + new gate_up_down_vq MTL4 kernel at fp32 noise floor
 * on real DS4 V4 weights.
 *
 * Use:
 *   ./ds4 --vq-real-canary <vqb1_dir> <layer> <expert> [down_rows [act_rows]]
 *
 * Returns 1 on success (rel_err < 1e-3), 0 on any failure.
 */
int ds4_gpu_mtl4_vq_real_canary(const char *vqb1_dir,
                                  uint32_t layer, uint32_t expert,
                                  uint32_t down_rows, uint32_t act_rows);

/* Phase B-2.3c stub: polar hot-path dispatcher entry. Validates pool
 * has gate/up/down PLR2 files for the layer; emits diagnostic; always
 * returns 0 (fallback to FP4 path). Body implementation pending silv
 * decision on row-coverage sub-strategy (A.1 full-row corpus / A.2
 * VQ / A.3 hybrid). See BRANCH_A_PREFLIGHT.md.
 *
 * Hot-path integration: called from metal_graph_encode_layer_ffn_batch
 * when g->polar_pool_ref != NULL && g->polar_layer_enabled_ref[il].
 * Returns 1 if polar substitution claimed the FFN output; 0 to
 * fall through to existing FP4 path.
 */
/* Forward declaration via void* — caller (ds4.c) passes g->polar_pool_ref
 * which is const ds4_polar_pool*. Stub casts back internally. This avoids
 * pulling ds4_polar_reader.h into ds4_gpu.h's already-deep dependency tree. */
int ds4_gpu_mtl4_polar_routed_moe_batch_stub(const void *pool,
                                              uint32_t layer,
                                              uint32_t n_tokens);

/* silv 2026-05-27 task #643 — Hadamard-16 batched FP16 transform.
 *
 * Applies the 16-point Walsh-Hadamard transform (scaled by 1/sqrt(16) for
 * orthogonal normalization) to each 16-element block of an FP16 activation
 * buffer. Used by basis-aware codec paths to transform inputs into the
 * same basis as the pre-transformed quantized weights, so the inner
 * product matches the original-basis matmul.
 *
 * `tensor` must be FP16 storage with `n_rows × blocks_per_row × 16` halves,
 * stride between rows = `row_stride_bytes`. Self-inverse: calling twice on
 * the same tensor returns the original up to FP16 precision (because
 * H × H = 16 I, scaled by (1/sqrt(16))^2 = 1/16 → identity).
 *
 * Returns 1 on success, 0 if the pipeline isn't registered (older builds
 * without metal/hadamard.metal compiled in) or args are invalid. */
int ds4_gpu_hadamard16_fp16_batched_tensor(ds4_gpu_tensor *tensor,
                                            uint32_t n_rows,
                                            uint32_t blocks_per_row,
                                            uint64_t row_stride_bytes);

/* silv 2026-05-27 task #667/#668 — per-layer attention scale + loop rescue.
 *
 * Per-layer scale: env DS4_ATTN_SCALE_MULT_PER_LAYER="L=X,L=X,..." sets
 * different multipliers per layer; layers without overrides fall back to
 * the global DS4_ATTN_SCALE_MULT. To use per-layer overrides, ds4.c must
 * call ds4_set_current_layer_idx(il) BEFORE each layer's flash-attn
 * dispatch. Without that wire, the global mult applies uniformly.
 *
 * Runtime rescue: ds4_set_attn_scale_mult_runtime(v) updates the GLOBAL
 * multiplier mid-generation. Use case: cache_lock_detector fires → loop
 * detected → call this with elevated v (e.g. 1.5) to sharpen attention
 * and break the loop, then restore to 1.0f after N tokens. Per-layer
 * overrides remain in effect for layers that have them set. */
void ds4_set_current_layer_idx(int il);
void ds4_set_attn_scale_mult_runtime(float v);

/* silv 2026-05-27 task #670 — first of 74-kernel MTL4 sweep.
 *
 * kernel_dsv4_softplus_sqrt_f32_4 ported to MTL4 path. Public canary
 * tests the pipeline against a CPU reference; returns 1 if max abs
 * diff < 1e-4. Used by the MTL4-port test harness to validate the
 * port pattern before scaling to MoE-matmul. */
int ds4_gpu_mtl4_softplus_sqrt_canary(uint32_t n_rows, uint32_t n_cols);

/* silv 2026-05-27 tasks #671, #672 — second/third of 74-kernel MTL4 sweep.
 *
 * router_weights_one_canary: validates the 6-element weight normalization
 * from the router cycle (already ICB-captured at classic-MTL level via
 * Phase 6 #560). MTL4 port enables future MTL4-specific optimizations
 * (e.g. ArgumentTable amortization across consecutive router dispatches).
 *
 * topk_mask_canary: validates the -INFINITY mask fill from the router
 * cycle (already ICB-captured at classic-MTL level via Phase 3 #557).
 * MTL4 port mirrors the storage-class migration pattern. */
int ds4_gpu_mtl4_router_weights_one_canary(void);
int ds4_gpu_mtl4_topk_mask_canary(uint32_t ne0, uint32_t ne1);
int ds4_gpu_mtl4_topk_mask_scatter_canary(uint32_t n_topk, uint32_t n_tokens, uint32_t n_comp);

/* silv 2026-05-27 tasks #674, #675 — batch 3 of MTL4 ports.
 *
 * indexer_weighted_sum: weighted sum over indexer head scores with neg-clip
 * and per-head weights × scale. Per-(token, comp_row) dispatch.
 *
 * dir_steering_project: project x onto a steering direction and subtract
 * scaled projection. Uses threadgroup scratch for parallel reduction. */
int ds4_gpu_mtl4_indexer_weighted_sum_canary(uint32_t ne0, uint32_t ne1, uint32_t n_heads);
int ds4_gpu_mtl4_dir_steering_canary(uint32_t width, uint32_t rows);

/* silv 2026-05-27 task #676. sort_i32_rows_asc — bitonic-sort each row of
 * top-K int32 indices into ascending order. top_k must be power-of-2,
 * ≤ 256 (single-threadgroup constraint of bitonic sort). */
int ds4_gpu_mtl4_sort_i32_rows_canary(uint32_t top_k, uint32_t n_rows);

/* silv 2026-05-27 task #677. router_weights_with_remap — applies the
 * inverse-remap table to selected expert IDs and renormalizes the
 * surviving weights. Completes the 4-kernel router-cycle MTL4 set
 * (router_weights_one + router_remap + topk_mask + topk_mask_scatter). */
int ds4_gpu_mtl4_router_remap_canary(uint32_t n_tokens);

/* silv 2026-05-27 task #678. ratio4_shift_f32 — tiny KV ratio-4 state
 * shift, copies state[n + gid] into state[gid] for both kv and score
 * buffers. Used at decode-time to drop the oldest row when the
 * compressor reaches capacity. */
int ds4_gpu_mtl4_ratio4_shift_canary(uint32_t width);

/* silv 2026-05-27 task #679 — MTL4 ArgumentTable amortization pool.
 *
 * Per-canary allocation creates a fresh argument table on every call,
 * costing ~10-50µs each. This pool maintains a free list keyed by
 * max_buffer_bind_count so consecutive dispatches reuse tables.
 *
 * Use:
 *   id<MTL4ArgumentTable> at = ds4_mtl4_pool_acquire(n_bindings);
 *   [at setAddress:bufA.gpuAddress atIndex:0];
 *   ... (set bindings that changed) ...
 *   [enc setArgumentTable:at];
 *   [enc dispatchThreadgroups:...];
 *   ...
 *   ds4_mtl4_pool_release(at, n_bindings);
 *
 * Bindings PERSIST across acquire/release pairs — this is the
 * amortization. Caller only re-sets buffers that change per-dispatch.
 *
 * Stats query for telemetry / canary verification. After warm-up,
 * acquire_count grows but alloc_count plateaus → pool hits dominate. */
void ds4_gpu_mtl4_pool_stats(uint64_t *out_acquire,
                              uint64_t *out_alloc,
                              uint64_t *out_release);

/* Pool-amortized version of router_weights_one canary. Runs N back-to-back
 * dispatches and reports pool acquire / alloc counts. Demonstrates the
 * amortization: alloc stays at 1 (or pool warm-up size); acquire scales N. */
int ds4_gpu_mtl4_router_weights_one_amortized_canary(uint32_t n_iterations);

/* silv 2026-05-27 tasks #680, #681 — batch 7 ports.
 * compressor_store_one: one-token KV-state append (kv, score, ape → state).
 * softmax_pool: fused softmax-weighted pool of KV rows. */
int ds4_gpu_mtl4_compressor_store_one_canary(uint32_t width);
int ds4_gpu_mtl4_softmax_pool_canary(uint32_t ne0, uint32_t ne1, uint32_t n_rows);

/* silv 2026-05-27 task #682 — MTL4 port of kernel_mul_mm_id_fp16_pair_swiglu_f32
 * (THE per-token decode bottleneck — 774 dispatches/token). This entry point
 * just initializes the MTL4 pipeline; full output canary deferred to next-
 * turn ship (synthetic 8-buffer test bench is multi-turn work). The pipeline
 * existence is the foundation for ICB Phase 7 capture and hot-path integration.
 *
 * Returns 1 if pipeline compiles + initializes; 0 on failure. */
int ds4_gpu_mtl4_moe_matmul_init_canary(void);

/* Full output canary for MoE matmul MTL4. Synthetic 1-expert × 8-token × 16K
 * test bench with identity weights. Validates the MTL4 pipeline produces
 * silu(x[t,oc]) * x[t,oc] * route_w[t] correctly. */
int ds4_gpu_mtl4_moe_matmul_full_canary(void);

/* ICB Phase 7 for MoE matmul (task #664).
 *
 * Records ONE 6-slot dispatch into a classic-MTL ICB at first call. Per
 * layer/per token replay via executeCommandsInBuffer skips ~45µs of
 * encoder overhead per dispatch. At 6 experts × 43 layers = 258 calls
 * per token, target savings ~12ms/token at baseline ~53ms/token decode.
 *
 * Opt-in via DS4_MOE_ICB=1 env var. Caller MUST pass the classic-MTL
 * pipeline (g_moe_mul_mm_id_fp16_pair_swiglu_pipeline) — classic ICB
 * cannot encode MTL4 pipelines.
 *
 * Returns 1 if dispatched via ICB (or no-op fallthrough when opt-out
 * is the env state); 0 only on hard error.
 *
 * Obj-C-only signature (the actual dispatch uses MTL id<MTL...> types
 * which aren't recognized in plain C). Wrapped behind __OBJC__ so this
 * header remains C-includable. */
#ifdef __OBJC__
int ds4_gpu_moe_matmul_icb_dispatch(
        id<MTLCommandBuffer> cb,
        id<MTLComputePipelineState> pipeline,
        id<MTLBuffer> argsbuf, NSUInteger args_off,
        id<MTLBuffer> actbuf, NSUInteger act_off,
        id<MTLBuffer> gatebuf, NSUInteger gate_off,
        id<MTLBuffer> upbuf, NSUInteger up_off,
        id<MTLBuffer> xbuf, NSUInteger x_off,
        id<MTLBuffer> midbuf, NSUInteger mid_off,
        id<MTLBuffer> idsbuf, NSUInteger ids_off,
        id<MTLBuffer> weightsbuf, NSUInteger weights_off,
        NSUInteger n_out_tiles,
        NSUInteger n_tok_tiles,
        NSUInteger n_experts);
#endif

/* Diagnostic — used by Phase 7 canary to verify ICB recording state. */
int ds4_gpu_moe_matmul_icb_status(uint32_t *out_recorded, uint32_t *out_slots);

/* silv 2026-05-27 task #683 — hc_weighted_sum: HC weighted sum
 * dst[d,t] = sum_h x[d,h,t] * weights[h,t]. First port from dsv4_hc.metal. */
int ds4_gpu_mtl4_hc_weighted_sum_canary(uint32_t n_embd, uint32_t n_hc, uint32_t n_tokens);

/* silv 2026-05-27 task #684 — router_finalize_one MTL4. Completes the
 * 5-kernel router cycle MTL4 set. 256-thread bitonic sort with optional
 * bias + hash-mode bypass. */
int ds4_gpu_mtl4_router_finalize_one_canary(int has_bias_flag);

/* silv 2026-05-27 task #685 — qkv_rms_norm_f32_4 MTL4. Per-layer fused
 * Q+KV RMSNorm. Fires 43× per token. High call-density target. */
int ds4_gpu_mtl4_qkv_rms_norm_canary(uint32_t q_n, uint32_t kv_n);

/* silv 2026-05-27 task #678 — soft_max_f32_4 MTL4. Row-softmax over
 * float4 vectorized rows, used in compressor/indexer attention paths.
 * Per-attention dispatch density. Tests simd reductions + threadgroup
 * scratch over vectorized inputs. */
int ds4_gpu_mtl4_soft_max_4_canary(uint32_t n);

/* silv 2026-05-27 task #679 — dsv4_hc_expand4 MTL4. HC=4 specialization
 * of per-layer block-expansion. 5 input buffers (block_out, residual,
 * post, comb, block_add) + dst. One thread per (d, t) → 4 outputs.
 * Per-layer dispatch density (43× per token). */
int ds4_gpu_mtl4_hc_expand4_canary(uint32_t n_embd, uint32_t n_tokens);

/* silv 2026-05-27 task #680 — dsv4_indexer_score_one_direct MTL4.
 * Decode-only DS4 ratio-4 indexer score builder. Hardcoded n_head=64,
 * head_dim=128. 128-thread threadgroup × 4 simdgroups parallel over
 * head groups. Per-attention dispatch (fires once per decode call).
 * Exercises threadgroup shared mem + simd_sum reductions + 2-stage
 * tg accumulation. */
int ds4_gpu_mtl4_indexer_score_one_direct_canary(uint32_t n_comp);

/* silv 2026-05-27 task #683 — soft_max_f32 MTL4 (non-vectorized variant
 * of soft_max_f32_4). Handles arbitrary row widths via simd_max +
 * simd_sum reductions; used by compressor pool when width%4 != 0. */
int ds4_gpu_mtl4_soft_max_canary(uint32_t n);

/* silv 2026-05-27 task #684 — dsv4_fp8_kv_quantize_f32 MTL4. Per-decode
 * KV-cache write path: FP8 E4M3FN round-trip on non-RoPE region (block
 * size 64, threadgroup-wide max reduction + LUT-driven dequant), passes
 * RoPE tail unchanged. Exercises kernel-source-embedded LUT tables,
 * binary-search FP8 dequant, halving-reduction amax. */
int ds4_gpu_mtl4_fp8_kv_quantize_canary(uint32_t n_rows, uint32_t n_full, uint32_t n_rot);

/* silv 2026-05-27 task #685b — dsv4_indexer_hadamard_fp4_f32 MTL4.
 * 128-wide Walsh-Hadamard butterfly (7-stage stride-doubling) + FP4
 * E2M1FN inplace quant on indexer Q/KV rows. Per-32-block scaling.
 * Per-row dispatch (128 threads/row). Hardcoded head_dim=128. */
int ds4_gpu_mtl4_indexer_hadamard_fp4_canary(uint32_t n_rows);

/* silv 2026-05-27 task #686 — dsv4_kv_fp8_store_f32 MTL4. Decode-time
 * KV finalizer: same FP8 E4M3FN round-trip as fp8_kv_quantize on the
 * non-RoPE region, plus FP16-rounded raw_cache mirror used by
 * FlashAttention. Per-token dispatch (64 threads). */
int ds4_gpu_mtl4_kv_fp8_store_canary(uint32_t head_dim, uint32_t n_rot);

/* silv 2026-05-27 task #687 — dsv4_moe_swiglu_weight MTL4. Routed-MoE
 * activation: mid[i] = SiLU(g) × u × route_weight with optional clamp.
 * Per (expert, token) row dispatch. */
int ds4_gpu_mtl4_moe_swiglu_weight_canary(uint32_t rows, uint32_t width);

/* silv 2026-05-27 task #688 — dsv4_moe_sum6_f32 MTL4. Routed-MoE
 * finalize: sum of 6 contiguous expert outputs per token. */
int ds4_gpu_mtl4_moe_sum6_canary(uint32_t tokens, uint32_t width);

/* silv 2026-05-27 task #689 — dsv4_moe_swiglu_weight_f16 MTL4. FP16 mid
 * variant of #687: SiLU(g)×u×route_weight stored as half. */
int ds4_gpu_mtl4_moe_swiglu_weight_f16_canary(uint32_t rows, uint32_t width);

/* silv 2026-05-27 task #690 — dsv4_hc_split_sinkhorn MTL4 (HC=4).
 * Splits HC mixer row into pre/post sigmoids + Sinkhorn-normalized
 * 4×4 combination matrix. NEW patterns: stable row-softmax (max-shift),
 * iterative col/row normalization for sinkhorn_iters steps. */
int ds4_gpu_mtl4_hc_split_sinkhorn_canary(uint32_t n_rows, uint32_t sinkhorn_iters);

/* silv 2026-05-27 task #691 — get_rows_f32 MTL4. Generic embedding /
 * table lookup: gather rows by int32 ids. Used for token embeddings
 * + router/hash lookup outputs. */
int ds4_gpu_mtl4_get_rows_f32_canary(uint32_t n_table_rows, uint32_t row_width, uint32_t n_ids);

/* silv 2026-05-27 task #692 — argsort_f32_i32_desc MTL4. Bitonic sort
 * of indices keyed by float values, descending order. Returns top_k
 * largest-value indices per row. Used by router top-K selection. */
int ds4_gpu_mtl4_argsort_f32_i32_desc_canary(uint32_t row_n, uint32_t top_k);

/* silv 2026-05-27 task #693 — cpy_f32_f32 MTL4. Generic f32→f32 typed
 * copy/conversion between graph tensors. Used at graph boundaries for
 * layout materialization. The templated MSL form also supports f32↔f16
 * conversions which would be added as paired specializations. */
int ds4_gpu_mtl4_cpy_f32_f32_canary(uint32_t n_rows, uint32_t row_width);

/* silv 2026-05-27 task #694 — dsv4_hc_split_weighted_sum MTL4 (HC=4).
 * Fused HC=4 split + HC-weighted embedding reduction per token row.
 * tid 0 runs the Sinkhorn split path (#690), stashes pre[0..3] in
 * shared mem; all threads then compute dst[d] = sum_h x[d,h] × pre[h]. */
int ds4_gpu_mtl4_hc_split_weighted_sum_canary(uint32_t n_rows, uint32_t n_embd);

/* silv 2026-05-27 task #695 — dsv4_rope_tail_f32 MTL4. DeepSeek V4
 * partial RoPE with YaRN scaling. First n_nope = ne00 - n_dims
 * elements copied unchanged; rotated tail transformed via YaRN-scaled
 * cos/sin. Two variants: interleaved (mode != 2) + NeoX (mode == 2).
 * NEW patterns: inlined YaRN helpers (ramp, corr_dims, yarn), optional
 * freq_factor table (src2), inverse rotation support. */
int ds4_gpu_mtl4_rope_tail_f32_canary(uint32_t head_dim, uint32_t n_rot, uint32_t mode);

/* silv 2026-05-27 task #696 — concat MTL4. Graph utility: concatenates
 * two float tensors along a chosen dim. Used to assemble attention
 * inputs with the exact tensor layout downstream kernels expect. */
int ds4_gpu_mtl4_concat_canary(uint32_t n0, uint32_t n1, uint32_t n_rows);

/* silv 2026-05-27 task #697 — dsv4_hc_split_weighted_sum_norm4 MTL4.
 * Three-stage fused kernel: HC=4 Sinkhorn split + HC-weighted sum
 * into 4096-wide row + RMSNorm with weight × row scaling. Hardcoded
 * n_hc=4, n_embd=4096 (DS4 production shape). 16.5KB threadgroup memory.
 * Saves 2 dispatches per token vs separate split + sum + RMSNorm. */
int ds4_gpu_mtl4_hc_split_weighted_sum_norm4_canary(uint32_t n_rows);

/* silv 2026-05-27 task #698 — dsv4_hc_expand MTL4. Generic-HC variant
 * of #679 hc_expand4. One thread per (d, dst_hc, t) → 1 output. Inner
 * loop over n_hc src_hc values. Pairs with the HC=4 unrolled #679
 * specialization for the production hot path. */
int ds4_gpu_mtl4_hc_expand_canary(uint32_t n_embd, uint32_t n_hc, uint32_t n_tokens);

/* silv 2026-05-27 task #699 — hadamard16_fp16_batched_wide MTL4.
 * 16 Hadamard-16 blocks per threadgroup (256 threads = 16 × 16).
 * All blocks' butterfly stages run in lockstep with tg-barriers
 * between stride doublings. Per-block early-out for tail groups. */
int ds4_gpu_mtl4_hadamard16_wide_canary(uint32_t n_rows, uint32_t blocks_per_row);

/* silv 2026-05-27 task #700 — argsort_merge_f32_i32_desc MTL4.
 * Merges two pre-sorted index runs (descending). Pairs with #692.
 * NEW patterns: binary-search partition + per-thread merge-front
 * advance + sentinel tail copy. */
int ds4_gpu_mtl4_argsort_merge_f32_i32_desc_canary(uint32_t len, uint32_t top_k);

/* silv 2026-05-27 tasks #702/#703 — cpy_f32_f16 + cpy_f16_f32 MTL4.
 * Type-conversion variants of #693 cpy_f32_f32. Used at graph
 * boundaries for f32↔f16 layout conversion. Non-MMA ports. */
int ds4_gpu_mtl4_cpy_f32_f16_canary(uint32_t n_rows, uint32_t row_width);
int ds4_gpu_mtl4_cpy_f16_f32_canary(uint32_t n_rows, uint32_t row_width);

/* silv 2026-05-27 task #704 — sum_rows_f32_f32 MTL4.
 * Per-row sum reduction with simd_sum + threadgroup combine.
 * Used at compressor-pooling graph boundary. Non-MMA. */
int ds4_gpu_mtl4_sum_rows_f32_f32_canary(uint32_t n_rows, uint32_t row_width);

/* silv 2026-05-27 task #705 — set_rows_f32_i32 MTL4.
 * KV-cache scatter: dst[idx[i]] = src[i]. Integer-indexed gather/
 * scatter, no MMA. Used during KV writes. */
int ds4_gpu_mtl4_set_rows_f32_i32_canary(uint32_t n_src_rows, uint32_t row_width);

/* silv 2026-05-27 task #706 — mul_mm_id_map0_ne20_8 MTL4.
 * MoE routing-table inverter: per-token expert list →
 * per-expert (token, slot) list. NE20=8 hardcoded (DS4 production
 * top-K). Non-MMA, threadgroup-memory cooperative. */
int ds4_gpu_mtl4_mul_mm_id_map0_ne20_8_canary(uint32_t n_experts, uint32_t n_tokens);

/* silv 2026-05-27 task #707 — repeat_f32 MTL4.
 * Broadcast/repeat with modulo addressing. Used at layer-0 input
 * for HC-channel expansion of token embeddings. Non-MMA. */
int ds4_gpu_mtl4_repeat_f32_canary(uint32_t src_rows, uint32_t src_cols, uint32_t row_factor, uint32_t col_factor);

/* silv 2026-05-27 task #708 — swiglu_f32 MTL4.
 * SiLU(x0) * x1 * alpha with optional clamp limit. Shared-expert
 * activation. Non-MMA elementwise. */
int ds4_gpu_mtl4_swiglu_f32_canary(uint32_t n_rows, uint32_t row_width, float alpha, float limit);

/* silv 2026-05-27 task #709 — rms_norm_mul_f32_4 MTL4.
 * RMSNorm with fused weight multiply (F=2 variant of fuse_impl). Used
 * before attention + before FFN at every layer. float4-vectorized
 * with simd_sum + threadgroup combine. Non-MMA. */
int ds4_gpu_mtl4_rms_norm_mul_f32_4_canary(uint32_t n_rows, uint32_t row_width, float eps);

/* silv 2026-05-27 task #710 — rms_norm_f32_4 MTL4.
 * F=1 variant: norm only, no weight multiply. Diagnostic variant. */
int ds4_gpu_mtl4_rms_norm_f32_4_canary(uint32_t n_rows, uint32_t row_width, float eps);

/* silv 2026-05-27 tasks #711/#712 — get_rows_f16 + get_rows_i32 MTL4.
 * Type-variant clones of #691 get_rows_f32. f16 widens to f32 on load;
 * i32 preserves type for embedding-table hash lookups. */
int ds4_gpu_mtl4_get_rows_f16_canary(uint32_t n_table_rows, uint32_t row_width, uint32_t n_ids);
int ds4_gpu_mtl4_get_rows_i32_canary(uint32_t n_table_rows, uint32_t row_width, uint32_t n_ids);
