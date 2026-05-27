/* silv 2026-05-27 Cached prefix activations Phase 1 (DEEP_OPS_DESIGN_2026_05_27.md).
 *
 * Public API for the prefix-activation cache. The goal: cache the full
 * per-layer prefill state (KV cache, residual, router state) keyed by
 * tokenized prefix hash, so that repeated calls with the same system
 * prompt + variable user message can skip prefill entirely on cache hit.
 *
 * Hit-case wall-time win: cached restore ~1s vs prefill ~590s (1k tokens
 * at 1.7 t/s) = ~54× speedup on the cached prefix workloads.
 *
 * Phase 1 (THIS FILE): scaffolding only.
 *   - Data structure declarations
 *   - FNV-1a 64-bit hash over the token sequence
 *   - In-memory LRU of K=8 entries
 *   - All store/restore paths are TODOs
 *
 * Phase 2: integration with ds4_engine state — populate cache entries
 *   from ds4_session after prefill completes; restore into ds4_session
 *   before generation starts on hit.
 *
 * Phase 3: disk-backed LRU at ~/Library/Caches/ds4/prefix_cache/.
 *   Memory-mapped restore for zero-copy reload.
 *
 * Phase 4-5: prefill bypass at ds4_session_create + AIME corpus A/B.
 *
 * Phase 1 deliverable is structural-only; no live cache hits yet. */

#ifndef DS4_PREFIX_CACHE_H
#define DS4_PREFIX_CACHE_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#define DS4_PREFIX_CACHE_LRU_CAPACITY 8

/* A single cache entry. Storage of per-layer GPU state is deferred to
 * Phase 2 — Phase 1 stores only the prefix hash + token count so the
 * lookup path can be exercised end-to-end. */
typedef struct ds4_prefix_cache_entry {
    uint64_t prefix_hash;    /* FNV-1a 64-bit of int32 token sequence */
    uint32_t n_tokens;        /* prefix length in tokens */
    uint64_t last_used_seq;   /* monotonic LRU counter */
    int valid;                /* 1 if populated, 0 if free slot */

    /* Phase 2 will add per-layer GPU tensor pointers here. Names follow
     * the engine state's kv_compressed / raw_kv / residual / router
     * shape. Held as opaque void* until Phase 2 wires the actual ds4_gpu
     * dependency.
     *
     * void *kv_state_per_layer[DS4_N_LAYER];
     * void *raw_kv_state_per_layer[DS4_N_LAYER];
     * void *residual_final;
     * void *router_state_per_layer[DS4_N_LAYER];
     */
} ds4_prefix_cache_entry;

/* The cache itself. Held as a static singleton inside ds4_prefix_cache.c
 * to keep the engine state lean. Init/free are explicit so the engine
 * lifecycle can manage it. */
typedef struct ds4_prefix_cache {
    ds4_prefix_cache_entry entries[DS4_PREFIX_CACHE_LRU_CAPACITY];
    uint64_t next_seq;              /* monotonic counter for LRU */
    uint64_t stat_lookups;          /* total lookup() calls */
    uint64_t stat_hits;             /* cache hits */
    uint64_t stat_stores;           /* successful store() calls */
    uint64_t stat_evictions;        /* LRU evictions */
} ds4_prefix_cache;

/* Initialize the cache. Returns 1 on success, 0 on failure. Idempotent. */
int ds4_prefix_cache_init(ds4_prefix_cache *cache);

/* Free all cache resources. Phase 2+ will release per-layer GPU buffers. */
void ds4_prefix_cache_free(ds4_prefix_cache *cache);

/* Compute the FNV-1a 64-bit hash over a token sequence. Exposed for
 * testing + use by the engine before it calls lookup(). */
uint64_t ds4_prefix_cache_hash_tokens(const int *tokens, uint32_t n_tokens);

/* Lookup a prefix in the cache. Returns the entry pointer if found
 * (with LRU touched), or NULL if not found. The returned pointer is
 * owned by the cache and remains valid until the next lookup() that
 * triggers eviction. Phase 2 will widen the API to also return the
 * per-layer state restoration handles. */
const ds4_prefix_cache_entry *ds4_prefix_cache_lookup(
    ds4_prefix_cache *cache,
    uint64_t prefix_hash);

/* Store a new prefix entry. If the cache is full, the least-recently-used
 * entry is evicted. Returns 1 on success, 0 on failure (e.g. when Phase 2+
 * GPU resource allocation fails). Phase 1 only stores hash + n_tokens. */
int ds4_prefix_cache_store(
    ds4_prefix_cache *cache,
    uint64_t prefix_hash,
    uint32_t n_tokens);

/* Invalidate all entries (e.g. when the model changes). */
void ds4_prefix_cache_invalidate_all(ds4_prefix_cache *cache);

/* Get human-readable stats for reporting. Caller passes a buffer; the
 * function fills it with a one-line summary. Returns bytes written. */
int ds4_prefix_cache_stats(
    const ds4_prefix_cache *cache,
    char *buf, size_t buflen);

#endif /* DS4_PREFIX_CACHE_H */
