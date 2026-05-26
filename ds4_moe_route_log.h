#ifndef DS4_MOE_ROUTE_LOG_H
#define DS4_MOE_ROUTE_LOG_H

/* ds4_moe_route_log — env-gated JSONL route-key logger for DS4 MoE FFN calls.
 *
 * Emits one SHAPE-level record per MoE call without materializing tensors.
 * Each record contains: x_shape [n_tokens, expert_in_dim], total_experts
 * (256), active_experts (6), hidden, out, tokens_per_expert, consumer_cols,
 * objective, layer.
 *
 * Env vars:
 *   DS4_MOE_ROUTE_LOG           = /path/to/file.jsonl  (enable)
 *   DS4_MOE_ROUTE_OBJECTIVE     = speed | latency      (default speed)
 *   DS4_MOE_ROUTE_CONSUMER_COLS = 32                   (default heuristic)
 *
 * Cost when disabled: enabled() returns false immediately; no syscalls.
 */

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Returns true iff DS4_MOE_ROUTE_LOG env var is set + log file opened.
 * Cached after first call (idempotent). */
bool ds4_moe_route_log_enabled(void);

/* Emit one route-key row. Cheap (single fprintf). No tensor inspection.
 * Schema:
 *   {"event":"ds4_moe_route_key","layer":N,"x_shape":[T,H],"total_experts":256,
 *    "active_experts":6,"hidden":H,"out":O,"tokens_per_expert":T,
 *    "consumer_cols":C,"objective":"speed"}
 */
void ds4_moe_route_log_emit(
    uint32_t layer,
    uint32_t n_tokens,
    uint64_t expert_in_dim,
    uint64_t expert_mid_dim,
    uint64_t down_in_dim);

/* Close + flush the log file. Called from atexit hook on first open. */
void ds4_moe_route_log_close(void);

#ifdef __cplusplus
}
#endif

#endif /* DS4_MOE_ROUTE_LOG_H */
