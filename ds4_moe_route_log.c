/*
 * ds4_moe_route_log.c — env-gated MoE route-key JSONL logger.
 *
 * See ds4_moe_route_log.h for the H1681 analog rationale.
 */

#include "ds4_moe_route_log.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static _Atomic int g_route_log_state = 0;  /* 0 unchecked, 1 enabled, -1 disabled */
static FILE *g_route_log_fp = NULL;
static const char *g_route_log_objective = "speed";
static unsigned long g_route_log_consumer_cols = 32;
static pthread_mutex_t g_route_log_mu = PTHREAD_MUTEX_INITIALIZER;

void ds4_moe_route_log_close(void) {
    pthread_mutex_lock(&g_route_log_mu);
    if (g_route_log_fp) {
        fflush(g_route_log_fp);
        fclose(g_route_log_fp);
        g_route_log_fp = NULL;
    }
    pthread_mutex_unlock(&g_route_log_mu);
}

bool ds4_moe_route_log_enabled(void) {
    int state = atomic_load_explicit(&g_route_log_state, memory_order_acquire);
    if (state != 0) return state > 0;

    pthread_mutex_lock(&g_route_log_mu);
    /* Re-check inside lock. */
    state = atomic_load_explicit(&g_route_log_state, memory_order_acquire);
    if (state != 0) {
        pthread_mutex_unlock(&g_route_log_mu);
        return state > 0;
    }

    const char *path = getenv("DS4_MOE_ROUTE_LOG");
    if (!path || !path[0]) {
        atomic_store_explicit(&g_route_log_state, -1, memory_order_release);
        pthread_mutex_unlock(&g_route_log_mu);
        return false;
    }

    g_route_log_fp = fopen(path, "a");
    if (!g_route_log_fp) {
        fprintf(stderr, "ds4_moe_route_log: cannot open %s\n", path);
        atomic_store_explicit(&g_route_log_state, -1, memory_order_release);
        pthread_mutex_unlock(&g_route_log_mu);
        return false;
    }

    const char *obj = getenv("DS4_MOE_ROUTE_OBJECTIVE");
    if (obj && obj[0]) g_route_log_objective = obj;
    const char *cc = getenv("DS4_MOE_ROUTE_CONSUMER_COLS");
    if (cc && cc[0]) {
        char *end = NULL;
        unsigned long v = strtoul(cc, &end, 10);
        if (v > 0 && (v & (v - 1)) == 0 && v <= 8192) g_route_log_consumer_cols = v;
    }

    fprintf(stderr, "ds4_moe_route_log: enabled path=%s objective=%s consumer_cols=%lu\n",
            path, g_route_log_objective, g_route_log_consumer_cols);
    atexit(ds4_moe_route_log_close);
    atomic_store_explicit(&g_route_log_state, 1, memory_order_release);
    pthread_mutex_unlock(&g_route_log_mu);
    return true;
}

void ds4_moe_route_log_emit(uint32_t layer,
                              uint32_t n_tokens,
                              uint64_t expert_in_dim,
                              uint64_t expert_mid_dim,
                              uint64_t down_in_dim) {
    if (!ds4_moe_route_log_enabled()) return;
    /* DS4 constants — keep in sync with ds4.c shape header. */
    enum { DS4_TOTAL_EXPERTS = 256, DS4_ACTIVE_EXPERTS = 6 };
    pthread_mutex_lock(&g_route_log_mu);
    if (g_route_log_fp) {
        /* JSONL line. Two route-key emits per layer: gate/up (hidden=expert_in_dim,
         * out=expert_mid_dim) AND down (hidden=down_in_dim, out=expert_in_dim).
         * Codex's H1679 ledger keys by (active, hidden, out), so we emit BOTH
         * shapes so the bridge tool sees them independently. */
        fprintf(g_route_log_fp,
                "{\"event\":\"ds4_moe_route_key\",\"layer\":%u,\"phase\":\"gate_up\","
                "\"x_shape\":[%u,%llu],\"total_experts\":%d,\"active_experts\":%d,"
                "\"hidden\":%llu,\"out\":%llu,\"tokens_per_expert\":%u,"
                "\"consumer_cols\":%lu,\"objective\":\"%s\"}\n",
                layer, n_tokens, (unsigned long long)expert_in_dim,
                DS4_TOTAL_EXPERTS, DS4_ACTIVE_EXPERTS,
                (unsigned long long)expert_in_dim, (unsigned long long)expert_mid_dim,
                n_tokens, g_route_log_consumer_cols, g_route_log_objective);
        fprintf(g_route_log_fp,
                "{\"event\":\"ds4_moe_route_key\",\"layer\":%u,\"phase\":\"down\","
                "\"x_shape\":[%u,%llu],\"total_experts\":%d,\"active_experts\":%d,"
                "\"hidden\":%llu,\"out\":%llu,\"tokens_per_expert\":%u,"
                "\"consumer_cols\":%lu,\"objective\":\"%s\"}\n",
                layer, n_tokens, (unsigned long long)down_in_dim,
                DS4_TOTAL_EXPERTS, DS4_ACTIVE_EXPERTS,
                (unsigned long long)down_in_dim, (unsigned long long)expert_in_dim,
                n_tokens, g_route_log_consumer_cols, g_route_log_objective);
    }
    pthread_mutex_unlock(&g_route_log_mu);
}
