/* ds4_cache_lock_detector.h — legacy compat shim over ds4_inflight.h.
 *
 * The implementation moved to ds4_inflight.c with self-explanatory names.
 * This header maps the legacy API (ds4_cache_lock_*) onto the new
 * (watch_token_emerging_from_model + guess_next_token_assuming_loop_continues)
 * so existing call sites compile unchanged. New code should use
 * ds4_inflight.h directly.
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

/* No snapshot/state struct. Derived values are read IN-BAND via the accessors
 * below (ds4_cache_lock_is_locked / _repeat_factor), computed live from the
 * detector — the single source of truth. The former ds4_cache_lock_state +
 * get_state copy-out carried fields (top_count/top_ngram_hash) that outlived
 * their source after the loop_detector dropped them, so callers read always-0
 * phantoms. Eliminated 2026-06-03 (one-level-up bug-class kill): a value that
 * isn't stored can't drift; a field that doesn't exist can't be misread. */

/* The compat shim silently drops tuning args. Warn loudly if caller passes
 * non-default values so the discrepancy is visible at runtime. Note also
 * that even with default n, autocorrelation prediction uses ALL lags;
 * n only affects the lock-onset detection n-gram length. */
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

/* In-band accessors — compute derived state live from the detector (the single
 * source of truth). O(1), no allocation, no parallel representation to drift.
 * TRIPWIRE: depends on loop_detector field names in ds4_inflight.h. */
static inline bool ds4_cache_lock_is_locked(const ds4_cache_lock_detector *d) {
    return d ? d->model_is_stuck_in_a_loop : false;
}
static inline float ds4_cache_lock_repeat_factor(const ds4_cache_lock_detector *d) {
    if (!d || d->how_many_distinct_ngrams_currently == 0) return 1.0f;
    return (float)d->how_many_ngram_occurrences_total
         / (float)d->how_many_distinct_ngrams_currently;
}

#ifdef __cplusplus
}
#endif

#endif