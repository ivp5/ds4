/* ds4_cache_lock_detector.h — legacy compat shim over ds4_inflight.h.
 *
 * silv 2026-05-26 spaghetti consolidation: the implementation moved to
 * ds4_inflight.c with self-explanatory names. This header maps the
 * legacy API (ds4_cache_lock_*) onto the new (watch_token_emerging_from_model
 * + guess_next_token_assuming_loop_continues) so ds4.c gen-loop call
 * sites compile unchanged. New code should use ds4_inflight.h directly.
 */
#ifndef DS4_CACHE_LOCK_DETECTOR_H
#define DS4_CACHE_LOCK_DETECTOR_H

#include "ds4_inflight.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_CACHE_LOCK_WINDOW_DEFAULT    MAX_TOKENS_WATCHED_IN_LOOP_WINDOW
#define DS4_CACHE_LOCK_N_DEFAULT         LOOP_PATTERN_NGRAM_LENGTH
#define DS4_CACHE_LOCK_THRESHOLD_DEFAULT LOOP_REPEAT_FACTOR_TO_DECLARE_STUCK

typedef loop_detector ds4_cache_lock_detector;

typedef struct {
    bool locked;
    uint32_t n_distinct;
    uint32_t n_total;
    float repeat_factor;
    uint64_t top_ngram_hash;
    uint16_t top_count;
} ds4_cache_lock_state;

/* INOCULATION F1: compat shim silently drops args. Warn loudly if caller
 * passes non-default values; they're being ignored by the spaghetti.
 * INOCULATION F2: even with default n, autocorrelation uses ALL lags for
 * prediction; n only affects lock-onset detection's n-gram length. Half-
 * migrated. */
static inline ds4_cache_lock_detector *ds4_cache_lock_alloc(
        uint32_t window_size, uint32_t n, float threshold) {
    if (window_size != DS4_CACHE_LOCK_WINDOW_DEFAULT
     || n != DS4_CACHE_LOCK_N_DEFAULT
     || threshold != DS4_CACHE_LOCK_THRESHOLD_DEFAULT) {
        fprintf(stderr,
                "ds4_cache_lock_alloc: WARNING — args (window=%u n=%u threshold=%.2f) "
                "are IGNORED by spaghetti compat shim; using fixed defaults "
                "(window=%d n=%d threshold=%.2f). Migrate caller to "
                "ds4_inflight.h direct API if non-default tuning is needed.\n",
                (unsigned)window_size, (unsigned)n, (double)threshold,
                (int)DS4_CACHE_LOCK_WINDOW_DEFAULT, (int)DS4_CACHE_LOCK_N_DEFAULT,
                (double)DS4_CACHE_LOCK_THRESHOLD_DEFAULT);
    }
    ds4_cache_lock_detector *d = (ds4_cache_lock_detector *)calloc(1, sizeof(*d));
    return d;
}

static inline void ds4_cache_lock_free(ds4_cache_lock_detector *d) {
    if (d) free(d);
}

static inline void ds4_cache_lock_reset(ds4_cache_lock_detector *d) {
    if (d) memset(d, 0, sizeof(*d));
}

static inline int ds4_cache_lock_push(ds4_cache_lock_detector *d, int32_t token_id) {
    return watch_token_emerging_from_model(d, token_id) ? 1 : 0;
}

static inline int32_t ds4_cache_lock_predict_next(const ds4_cache_lock_detector *d) {
    return guess_next_token_assuming_loop_continues(d);
}

/* INOCULATION F3 — TRIPWIRE: state-mapping depends on loop_detector field
 * names below. If you rename these in ds4_inflight.h, update here too.
 * Mapping:
 *   ds4_cache_lock_state.locked         <- model_is_stuck_in_a_loop
 *   ds4_cache_lock_state.n_distinct     <- how_many_distinct_ngrams_currently
 *   ds4_cache_lock_state.n_total        <- how_many_ngram_occurrences_total
 *   ds4_cache_lock_state.repeat_factor  <- computed = n_total / n_distinct
 *   ds4_cache_lock_state.top_ngram_hash <- DEPRECATED 2026-05-26: source
 *     fields cut as stale-cache (silv cache-coherence directive); kept
 *     in struct ABI for legacy compat but reported as 0.
 *   ds4_cache_lock_state.top_count      <- DEPRECATED 2026-05-26: same. */
static inline void ds4_cache_lock_get_state(const ds4_cache_lock_detector *d,
                                             ds4_cache_lock_state *out) {
    if (!d || !out) return;
    out->locked         = d->model_is_stuck_in_a_loop;
    out->n_distinct     = d->how_many_distinct_ngrams_currently;
    out->n_total        = d->how_many_ngram_occurrences_total;
    out->repeat_factor  = d->how_many_distinct_ngrams_currently > 0
                       ? (float)d->how_many_ngram_occurrences_total
                         / (float)d->how_many_distinct_ngrams_currently
                       : 1.0f;
    /* Stale-cache fields cut from loop_detector; legacy callers see 0. */
    out->top_ngram_hash = 0;
    out->top_count      = 0;
}

#ifdef __cplusplus
}
#endif

#endif