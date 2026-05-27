/* silv 2026-05-27 Cached prefix activations Phase 1 implementation
 * (DEEP_OPS_DESIGN_2026_05_27.md §Cached prefix activations).
 *
 * Phase 1: structural scaffolding — hash + in-memory LRU + stats.
 * Phase 2: GPU state population (deferred — needs ds4_engine integration).
 * Phase 3: disk-backed LRU at ~/Library/Caches/ds4/prefix_cache/.
 *
 * This file has zero ds4_engine dependencies on purpose — the cache is
 * a pure data structure that the engine code will wire up in Phase 2. */

#include "ds4_prefix_cache.h"
#include <stdio.h>
#include <string.h>

/* FNV-1a 64-bit hash. Deterministic across runs; identical input
 * (tokens + length) always produces the same hash. */
uint64_t ds4_prefix_cache_hash_tokens(const int *tokens, uint32_t n_tokens) {
    /* FNV-1a 64 constants */
    const uint64_t fnv_offset = 0xcbf29ce484222325ULL;
    const uint64_t fnv_prime  = 0x100000001b3ULL;
    if (!tokens || n_tokens == 0) return fnv_offset;
    uint64_t h = fnv_offset;
    /* Hash byte-by-byte over the 32-bit token sequence. Treats negative
     * tokens (which shouldn't appear in well-formed prompts) as their
     * two's-complement bit pattern — preserves identity. */
    const unsigned char *bytes = (const unsigned char *)tokens;
    const size_t nb = (size_t)n_tokens * sizeof(int);
    for (size_t i = 0; i < nb; i++) {
        h ^= (uint64_t)bytes[i];
        h *= fnv_prime;
    }
    return h;
}

int ds4_prefix_cache_init(ds4_prefix_cache *cache) {
    if (!cache) return 0;
    memset(cache, 0, sizeof(*cache));
    /* Mark all entries invalid */
    for (int i = 0; i < DS4_PREFIX_CACHE_LRU_CAPACITY; i++) {
        cache->entries[i].valid = 0;
    }
    cache->next_seq = 1; /* skip 0 so a valid entry never has last_used_seq=0 ambiguity */
    return 1;
}

void ds4_prefix_cache_free(ds4_prefix_cache *cache) {
    if (!cache) return;
    /* Phase 2+ will release per-layer GPU buffers here. Phase 1 has
     * no resources to free — the entries are POD. */
    memset(cache, 0, sizeof(*cache));
}

const ds4_prefix_cache_entry *ds4_prefix_cache_lookup(
    ds4_prefix_cache *cache,
    uint64_t prefix_hash) {
    if (!cache) return NULL;
    cache->stat_lookups++;
    for (int i = 0; i < DS4_PREFIX_CACHE_LRU_CAPACITY; i++) {
        ds4_prefix_cache_entry *e = &cache->entries[i];
        if (e->valid && e->prefix_hash == prefix_hash) {
            /* Touch LRU */
            e->last_used_seq = ++cache->next_seq;
            cache->stat_hits++;
            return e;
        }
    }
    return NULL;
}

/* Find the LRU victim (entry with the smallest last_used_seq, including
 * any invalid slot). Returns the index of the slot to overwrite. */
static int ds4_prefix_cache_find_lru_slot(const ds4_prefix_cache *cache) {
    int best = 0;
    uint64_t best_seq = cache->entries[0].valid ? cache->entries[0].last_used_seq : 0;
    if (!cache->entries[0].valid) return 0;
    for (int i = 1; i < DS4_PREFIX_CACHE_LRU_CAPACITY; i++) {
        const ds4_prefix_cache_entry *e = &cache->entries[i];
        if (!e->valid) return i; /* free slot wins */
        if (e->last_used_seq < best_seq) {
            best = i;
            best_seq = e->last_used_seq;
        }
    }
    return best;
}

int ds4_prefix_cache_store(
    ds4_prefix_cache *cache,
    uint64_t prefix_hash,
    uint32_t n_tokens) {
    if (!cache) return 0;
    /* Check if already present — update LRU + n_tokens instead of duplicating. */
    for (int i = 0; i < DS4_PREFIX_CACHE_LRU_CAPACITY; i++) {
        ds4_prefix_cache_entry *e = &cache->entries[i];
        if (e->valid && e->prefix_hash == prefix_hash) {
            e->n_tokens = n_tokens;
            e->last_used_seq = ++cache->next_seq;
            return 1;
        }
    }
    /* Find victim slot */
    const int slot = ds4_prefix_cache_find_lru_slot(cache);
    ds4_prefix_cache_entry *e = &cache->entries[slot];
    const int was_valid = e->valid;
    e->prefix_hash = prefix_hash;
    e->n_tokens = n_tokens;
    e->last_used_seq = ++cache->next_seq;
    e->valid = 1;
    cache->stat_stores++;
    if (was_valid) cache->stat_evictions++;
    /* Phase 2+ will copy per-layer GPU state into the entry here. */
    return 1;
}

void ds4_prefix_cache_invalidate_all(ds4_prefix_cache *cache) {
    if (!cache) return;
    for (int i = 0; i < DS4_PREFIX_CACHE_LRU_CAPACITY; i++) {
        cache->entries[i].valid = 0;
    }
    /* Phase 2+ will release per-layer GPU buffers attached to invalidated
     * entries here. */
}

int ds4_prefix_cache_stats(
    const ds4_prefix_cache *cache,
    char *buf, size_t buflen) {
    if (!cache || !buf || buflen == 0) return 0;
    int n_valid = 0;
    for (int i = 0; i < DS4_PREFIX_CACHE_LRU_CAPACITY; i++) {
        if (cache->entries[i].valid) n_valid++;
    }
    const double hit_rate = (cache->stat_lookups > 0)
        ? (double)cache->stat_hits / (double)cache->stat_lookups : 0.0;
    return snprintf(buf, buflen,
        "ds4: prefix_cache lookups=%llu hits=%llu (%.1f%%) "
        "stores=%llu evictions=%llu valid_slots=%d/%d",
        (unsigned long long)cache->stat_lookups,
        (unsigned long long)cache->stat_hits,
        hit_rate * 100.0,
        (unsigned long long)cache->stat_stores,
        (unsigned long long)cache->stat_evictions,
        n_valid, DS4_PREFIX_CACHE_LRU_CAPACITY);
}

/* Phase 1 self-test: validates the in-memory LRU + hash properties
 * before any GPU integration. Exits 0 on success, 1 on failure. */
int ds4_prefix_cache_phase1_self_test(void) {
    ds4_prefix_cache cache;
    if (!ds4_prefix_cache_init(&cache)) {
        fprintf(stderr, "prefix_cache: init failed\n");
        return 0;
    }

    /* 1. Hash determinism: same input → same output, different input → different. */
    int toks_a[5] = {100, 200, 300, 400, 500};
    int toks_b[5] = {100, 200, 300, 400, 501};
    int toks_c[5] = {100, 200, 300, 400, 500};
    const uint64_t h_a = ds4_prefix_cache_hash_tokens(toks_a, 5);
    const uint64_t h_b = ds4_prefix_cache_hash_tokens(toks_b, 5);
    const uint64_t h_c = ds4_prefix_cache_hash_tokens(toks_c, 5);
    if (h_a != h_c) {
        fprintf(stderr, "prefix_cache: hash determinism FAILED (a=0x%llx c=0x%llx)\n",
                (unsigned long long)h_a, (unsigned long long)h_c);
        return 0;
    }
    if (h_a == h_b) {
        fprintf(stderr, "prefix_cache: hash sensitivity FAILED (a=0x%llx b=0x%llx)\n",
                (unsigned long long)h_a, (unsigned long long)h_b);
        return 0;
    }

    /* 2. Empty-input + null safety */
    if (ds4_prefix_cache_hash_tokens(NULL, 0) != 0xcbf29ce484222325ULL) {
        fprintf(stderr, "prefix_cache: empty-hash mismatch\n");
        return 0;
    }
    if (ds4_prefix_cache_lookup(&cache, h_a) != NULL) {
        fprintf(stderr, "prefix_cache: pre-store lookup should miss\n");
        return 0;
    }

    /* 3. Store + lookup round-trip */
    if (!ds4_prefix_cache_store(&cache, h_a, 5)) {
        fprintf(stderr, "prefix_cache: store failed\n");
        return 0;
    }
    const ds4_prefix_cache_entry *e = ds4_prefix_cache_lookup(&cache, h_a);
    if (!e || e->prefix_hash != h_a || e->n_tokens != 5) {
        fprintf(stderr, "prefix_cache: round-trip failed\n");
        return 0;
    }

    /* 4. Update existing entry (same hash): should not duplicate or evict */
    const uint64_t lookups_before = cache.stat_lookups;
    if (!ds4_prefix_cache_store(&cache, h_a, 10)) {
        fprintf(stderr, "prefix_cache: re-store failed\n");
        return 0;
    }
    e = ds4_prefix_cache_lookup(&cache, h_a);
    if (!e || e->n_tokens != 10) {
        fprintf(stderr, "prefix_cache: re-store update n_tokens failed\n");
        return 0;
    }
    if (cache.stat_lookups != lookups_before + 1) {
        fprintf(stderr, "prefix_cache: lookup stat tracking wrong\n");
        return 0;
    }

    /* 5. LRU eviction: store 9 distinct entries, the first should be evicted */
    ds4_prefix_cache_init(&cache); /* reset */
    uint64_t hashes[DS4_PREFIX_CACHE_LRU_CAPACITY + 1];
    for (int i = 0; i <= DS4_PREFIX_CACHE_LRU_CAPACITY; i++) {
        int t = 1000 + i;
        hashes[i] = ds4_prefix_cache_hash_tokens(&t, 1);
        if (!ds4_prefix_cache_store(&cache, hashes[i], (uint32_t)(i + 1))) {
            fprintf(stderr, "prefix_cache: LRU fill store %d failed\n", i);
            return 0;
        }
    }
    if (cache.stat_evictions != 1) {
        fprintf(stderr, "prefix_cache: expected 1 eviction, got %llu\n",
                (unsigned long long)cache.stat_evictions);
        return 0;
    }
    if (ds4_prefix_cache_lookup(&cache, hashes[0]) != NULL) {
        fprintf(stderr, "prefix_cache: hashes[0] should have been evicted\n");
        return 0;
    }
    for (int i = 1; i <= DS4_PREFIX_CACHE_LRU_CAPACITY; i++) {
        if (ds4_prefix_cache_lookup(&cache, hashes[i]) == NULL) {
            fprintf(stderr, "prefix_cache: hashes[%d] should be present\n", i);
            return 0;
        }
    }

    /* 6. Stats reporting */
    char statbuf[256];
    const int n = ds4_prefix_cache_stats(&cache, statbuf, sizeof(statbuf));
    if (n <= 0 || n >= (int)sizeof(statbuf)) {
        fprintf(stderr, "prefix_cache: stats truncation\n");
        return 0;
    }
    fprintf(stderr, "%s\n", statbuf);

    /* 7. Invalidate all */
    ds4_prefix_cache_invalidate_all(&cache);
    for (int i = 0; i <= DS4_PREFIX_CACHE_LRU_CAPACITY; i++) {
        if (ds4_prefix_cache_lookup(&cache, hashes[i]) != NULL) {
            fprintf(stderr, "prefix_cache: invalidate_all left entry %d\n", i);
            return 0;
        }
    }

    ds4_prefix_cache_free(&cache);
    fprintf(stderr, "ds4: prefix_cache Phase 1 self-test PASSED "
            "(hash determinism + sensitivity + LRU eviction + invalidate)\n");
    return 1;
}
