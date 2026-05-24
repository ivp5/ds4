#ifndef DS4_MOE_ROUTE_LOG_H
#define DS4_MOE_ROUTE_LOG_H

/*
 * ds4_moe_route_log — env-gated JSONL route-key logger for DS4 MoE FFN calls.
 *
 * Mirrors codex's H1681 olmoe.py route-key logger pattern (see
 * /Users/silv/cl/tlp_codex/research/llm_fallacy_deconstruction/framework_deconstruction/H1681_TINYGRAD_OLMOE_ROUTE_LOGGER_PATCH_20260525.md)
 * applied to DS4. Goal: emit a SHAPE-level record per MoE FFN call without
 * materializing tensors, so codex's H1679 multishape ledger can be queried
 * read-only against live DS4 traffic.
 *
 * Each entry records only shape/route metadata:
 *   - x_shape          : [n_tokens, expert_in_dim]
 *   - total_experts    : DS4 pool size (256)
 *   - active_experts   : DS4 top-k = DS4_N_EXPERT_USED = 6
 *   - hidden           : expert_in_dim
 *   - out              : expert_mid_dim (gate/up output) — caller picks orientation
 *   - tokens_per_expert: derived from n_tokens / active (approximate)
 *   - consumer_cols    : approximated from downstream attention proj dim
 *   - objective        : "speed" | "latency" (from env)
 *   - layer            : layer index (additional vs OLMoE since DS4 is 43-layer)
 *
 * Env vars (mirror H1681 conventions):
 *   DS4_MOE_ROUTE_LOG           = /path/to/file.jsonl  (enable)
 *   DS4_MOE_ROUTE_OBJECTIVE     = speed | latency      (default speed)
 *   DS4_MOE_ROUTE_CONSUMER_COLS = 32                   (default heuristic)
 *
 * Cost when enabled: 1 fprintf per MoE call = ~43 lines per token forward.
 * Negligible vs MoE compute.
 *
 * Cost when disabled: ds4_moe_route_log_enabled() returns false immediately,
 * no fprintf, no syscalls. Safe to leave in hot paths.
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
