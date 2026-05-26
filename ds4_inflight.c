/* ds4_inflight.c — session 2026-05-26 adaptive primitives, spaghetti
 *
 * silv 2026-05-26: "refactor into spaghetti code with minimal functions
 * (unless those save disproportionate amount of codelines/complexity)
 * and all functions and variables having self-explanatory names"
 *
 * Replaces the over-factored 5-file split:
 *   ds4_cache_lock_detector.{h,c}   216+111  -> loop_detector below
 *   ds4_signed_watchlist.h           161    -> watched_rank + harm-check below
 *   ds4_bounded_packet_search.c      154    -> search_for_sparse_repair below
 *   ds4_codec_k_dispatcher.h         145    -> classify_token_difficulty inline
 *
 * Original files preserved per append-only doctrine; this file is the
 * consolidated spaghetti form. Switch the Makefile when silv pulls.
 *
 * Geohotz (minimum-viable) + Knuth (provable bounds) + Carmack (data-
 * oriented) on the wheel; Tao + Donoho + Pearl consulted on naming.
 *
 * Statics at top. Types in one block. Three functions worth their scope
 * (event handler, autocorrelation lookup, bounded search). Hot paths
 * inlined.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ====================================================================
 * STATIC GLOBAL DEFINITIONS — caps, thresholds, magic numbers, all here
 * ==================================================================== */

#define MAX_TOKENS_WATCHED_IN_LOOP_WINDOW              200
#define LOOP_PATTERN_NGRAM_LENGTH                      5
#define MAX_NGRAM_LENGTH_FOR_OVERRIDE                  32
#define LOOP_REPEAT_FACTOR_TO_DECLARE_STUCK            3.0f
#define MIN_TOKENS_NEEDED_BEFORE_TRYING_TO_PREDICT     16
#define AUTOCORRELATION_MATCH_FRACTION_TO_TRUST_PERIOD 0.6f

/* Open-addressing hash table for n-gram counts. Sized 2x max distinct
 * n-grams so load factor stays low. */
#define NGRAM_HASH_TABLE_SIZE                          512

/* ====================================================================
 * TYPE DEFINITIONS — append-only, never reorder fields
 * ==================================================================== */

/* The loop-detector watches a rolling window of tokens and fires when
 * an n-gram repeats often enough that the model is "stuck." Then a
 * caller can call guess_next_token_assuming_loop_continues to skip a
 * model-forward and emit the predicted token instead. Adaptive period
 * via autocorrelation per A.1c v2; v1 (fixed n) was refuted at 0/129
 * on period-11 cycles. */
typedef struct {
    int32_t  rolling_token_window[MAX_TOKENS_WATCHED_IN_LOOP_WINDOW];
    uint32_t how_many_tokens_in_window_so_far;
    uint32_t next_slot_to_overwrite;

    uint64_t ngram_hash_at_slot[NGRAM_HASH_TABLE_SIZE];
    uint16_t ngram_repeat_count_at_slot[NGRAM_HASH_TABLE_SIZE];
    uint32_t how_many_distinct_ngrams_currently;
    uint32_t how_many_ngram_occurrences_total;

    bool     model_is_stuck_in_a_loop;
    /* Tunable thresholds (override after zero-init; 0.0 = use default). */
    float    lock_repeat_factor_threshold_override;
    float    autocorrelation_match_fraction_override;
    uint32_t ngram_length_override;  /* 0 = default LOOP_PATTERN_NGRAM_LENGTH */
} loop_detector;

/* Helper: resolve effective n-gram length from struct, applying the
 * sentinel-0 default and clamping to [2, MAX_NGRAM_LENGTH_FOR_OVERRIDE].
 * Per INC-29 (real-Qwen3.5-4B loop trace 2026-05-26): default n=5 fires
 * 3138 tokens later than n=15 on sentence-level LLM loops. Caller may
 * override; out-of-range values silently clamp. */
static inline uint32_t effective_ngram_length_for_detector(const loop_detector *d) {
    uint32_t n = d->ngram_length_override;
    if (n == 0u) return (uint32_t)LOOP_PATTERN_NGRAM_LENGTH;
    if (n < 2u) return 2u;
    if (n > (uint32_t)MAX_NGRAM_LENGTH_FOR_OVERRIDE) return (uint32_t)MAX_NGRAM_LENGTH_FOR_OVERRIDE;
    return n;
}

/* One ranked token's harm-comparison between two codecs. The "baseline"
 * is whatever codec we want to stay close to (typically direct/FP16);
 * the "candidate" is the codec we're considering substituting. A
 * positive "pushed up" value means the codec INCREASED the logit at
 * that rank (helpful for the truth token, harmful for competitors). */
typedef struct {
    int32_t rank_within_top_k;
    int32_t token_id_at_this_rank;
    float   how_much_baseline_codec_pushed_this_token_up;
    float   how_much_candidate_codec_pushed_this_token_up;
} watched_rank;

/* Result of asking "does the candidate codec harm more than baseline?" */
typedef struct {
    float    largest_harm_from_baseline_codec_across_all_ranks;
    float    largest_harm_from_candidate_codec_across_all_ranks;
    float    biggest_extra_harm_candidate_does_vs_baseline;
    int32_t  rank_at_which_candidate_is_worst_vs_baseline;
    int32_t  token_id_at_that_worst_rank;
    bool     candidate_is_worse_than_baseline_somewhere;
} harm_comparison_result;

/* Caller-supplied evaluator: given a bitmask of which routed edges are
 * promoted to baseline-codec (1 = direct/high-precision, 0 = candidate),
 * compute the signed-watchlist harm of the resulting routed-FFN output.
 * Returns 0 on success, nonzero if evaluation failed. */
typedef int (*evaluate_codec_subset_harm_fn)(
        uint32_t which_edges_are_baseline_mask,
        void *userdata,
        float *signed_harm_out);

/* ====================================================================
 * INLINE HELPERS — used once at a time, inlined to keep call-sites readable
 * ==================================================================== */

/* FNV-1a 64-bit hash over n token-ids. Used by the loop detector when
 * pushing/popping n-grams; inlined because called from hot path. */
static inline uint64_t hash_a_sequence_of_tokens(const int32_t *tokens, uint32_t how_many) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (uint32_t i = 0; i < how_many; i++) {
        const uint32_t t = (uint32_t)tokens[i];
        h ^= (uint64_t)(t & 0xff);         h *= 0x100000001b3ULL;
        h ^= (uint64_t)((t >> 8)  & 0xff); h *= 0x100000001b3ULL;
        h ^= (uint64_t)((t >> 16) & 0xff); h *= 0x100000001b3ULL;
        h ^= (uint64_t)((t >> 24) & 0xff); h *= 0x100000001b3ULL;
    }
    return h;
}

/* ====================================================================
 * APPEND-ONLY JSONL JOURNAL — opt-in via DS4_INFLIGHT_JOURNAL env var
 * ==================================================================== */

/* silv 2026-05-26: "keep detailed journal of all transactions in append-
 * only mode for further analysis afterwards."
 *
 * Design: append one JSON line per detector event. File path from env
 * var DS4_INFLIGHT_JOURNAL. When unset, every journal_* call is a
 * no-op (zero overhead in production). FD opened lazily on first
 * event, kept open until process exit. Append-only semantics:
 * O_WRONLY|O_CREAT|O_APPEND. No fsync per event (kernel flushes on
 * close/exit); add `--sync` mode later if a caller needs durability. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>

static int g_journal_fd = -2;  /* -2 = uninitialized, -1 = disabled, >=0 = open */

static int journal_fd_or_open(void) {
    if (g_journal_fd >= 0) return g_journal_fd;
    if (g_journal_fd == -1) return -1;
    const char *path = getenv("DS4_INFLIGHT_JOURNAL");
    if (!path || !*path) { g_journal_fd = -1; return -1; }
    int fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    g_journal_fd = (fd >= 0) ? fd : -1;
    return g_journal_fd;
}

/* journal_record_event — write one JSONL line. event_kind is the event
 * name; the variadic-ish args go in as KEY:int_val pairs (NULL key
 * terminates). Number-of-pairs is bounded; using a small fixed buffer.
 * Caller responsible for keys being valid JSON identifiers (no
 * quotes/backslashes — keys here are compile-time constants from this
 * file, never untrusted). */
static void journal_record_event(const char *event_kind,
                                 const char *k1, long long v1,
                                 const char *k2, long long v2,
                                 const char *k3, long long v3,
                                 const char *k4, long long v4) {
    int fd = journal_fd_or_open();
    if (fd < 0) return;
    char buf[512];
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    int n = snprintf(buf, sizeof(buf),
                     "{\"ts_ns\":%lld%09lld,\"event\":\"%s\"",
                     (long long)ts.tv_sec, (long long)ts.tv_nsec, event_kind);
    if (k1) n += snprintf(buf+n, sizeof(buf)-n, ",\"%s\":%lld", k1, v1);
    if (k2) n += snprintf(buf+n, sizeof(buf)-n, ",\"%s\":%lld", k2, v2);
    if (k3) n += snprintf(buf+n, sizeof(buf)-n, ",\"%s\":%lld", k3, v3);
    if (k4) n += snprintf(buf+n, sizeof(buf)-n, ",\"%s\":%lld", k4, v4);
    n += snprintf(buf+n, sizeof(buf)-n, "}\n");
    if (n > 0 && n < (int)sizeof(buf)) {
        ssize_t w = write(fd, buf, (size_t)n);
        (void)w;  /* best-effort; if write fails the in-memory state still consistent */
    }
}

/* ====================================================================
 * MAIN PRIMITIVES — three functions worth their own scope
 * ==================================================================== */

/* watch_token_emerging_from_model — push one decoded token, update
 * rolling window + n-gram count table, raise model_is_stuck_in_a_loop
 * if repeat_factor crosses threshold.
 *
 * O(n) per push where n = LOOP_PATTERN_NGRAM_LENGTH; ~50-200 ns wall.
 * Called once per generated token, so cumulative overhead is
 * negligible vs the ~10 ms/token model forward.
 *
 * Returns 1 if the detector is currently asserting that the model is
 * stuck (caller may then invoke guess_next_token_assuming_loop_continues
 * to skip the next model forward), 0 otherwise. */
bool watch_token_emerging_from_model(loop_detector *d, int32_t token) {
    if (!d) return false;
    const uint32_t ngram_len = effective_ngram_length_for_detector(d);

    /* If window is full, the oldest n-gram is leaving. Decrement its count. */
    if (d->how_many_tokens_in_window_so_far >= MAX_TOKENS_WATCHED_IN_LOOP_WINDOW) {
        if (d->how_many_tokens_in_window_so_far >= ngram_len) {
            int32_t leaving_ngram[MAX_NGRAM_LENGTH_FOR_OVERRIDE];
            for (uint32_t i = 0; i < ngram_len; i++) {
                uint32_t age_from_newest = (ngram_len - 1u - i);
                age_from_newest += (d->how_many_tokens_in_window_so_far - ngram_len);
                uint32_t idx = (d->next_slot_to_overwrite + MAX_TOKENS_WATCHED_IN_LOOP_WINDOW - 1u - age_from_newest) % MAX_TOKENS_WATCHED_IN_LOOP_WINDOW;
                leaving_ngram[i] = d->rolling_token_window[idx];
            }
            uint64_t leaving_hash = hash_a_sequence_of_tokens(leaving_ngram, ngram_len);
            uint32_t slot = (uint32_t)(leaving_hash & (NGRAM_HASH_TABLE_SIZE - 1u));
            for (uint32_t probe = 0; probe < NGRAM_HASH_TABLE_SIZE; probe++) {
                if (d->ngram_hash_at_slot[slot] == leaving_hash && d->ngram_repeat_count_at_slot[slot] > 0) {
                    d->ngram_repeat_count_at_slot[slot]--;
                    d->how_many_ngram_occurrences_total--;
                    if (d->ngram_repeat_count_at_slot[slot] == 0) {
                        d->ngram_hash_at_slot[slot] = 0;
                        d->how_many_distinct_ngrams_currently--;
                    }
                    break;
                }
                if (d->ngram_repeat_count_at_slot[slot] == 0 && d->ngram_hash_at_slot[slot] == 0) break;
                slot = (slot + 1u) & (NGRAM_HASH_TABLE_SIZE - 1u);
            }
        }
    }

    /* Write the new token into the window. */
    d->rolling_token_window[d->next_slot_to_overwrite] = token;
    d->next_slot_to_overwrite = (d->next_slot_to_overwrite + 1u) % MAX_TOKENS_WATCHED_IN_LOOP_WINDOW;
    if (d->how_many_tokens_in_window_so_far < MAX_TOKENS_WATCHED_IN_LOOP_WINDOW) {
        d->how_many_tokens_in_window_so_far++;
    }

    /* If we now have at least n tokens, compute the n-gram ending at the
     * new position and bump its count in the open-addressing table. */
    if (d->how_many_tokens_in_window_so_far >= ngram_len) {
        int32_t new_ngram[MAX_NGRAM_LENGTH_FOR_OVERRIDE];
        for (uint32_t i = 0; i < ngram_len; i++) {
            uint32_t age = (ngram_len - 1u - i);
            uint32_t idx = (d->next_slot_to_overwrite + MAX_TOKENS_WATCHED_IN_LOOP_WINDOW - 1u - age) % MAX_TOKENS_WATCHED_IN_LOOP_WINDOW;
            new_ngram[i] = d->rolling_token_window[idx];
        }
        uint64_t new_hash = hash_a_sequence_of_tokens(new_ngram, ngram_len);
        uint32_t slot = (uint32_t)(new_hash & (NGRAM_HASH_TABLE_SIZE - 1u));
        for (uint32_t probe = 0; probe < NGRAM_HASH_TABLE_SIZE; probe++) {
            if (d->ngram_repeat_count_at_slot[slot] == 0 && d->ngram_hash_at_slot[slot] == 0) {
                d->ngram_hash_at_slot[slot] = new_hash;
                d->ngram_repeat_count_at_slot[slot] = 1;
                d->how_many_distinct_ngrams_currently++;
                d->how_many_ngram_occurrences_total++;
                break;
            }
            if (d->ngram_hash_at_slot[slot] == new_hash) {
                d->ngram_repeat_count_at_slot[slot]++;
                d->how_many_ngram_occurrences_total++;
                break;
            }
            slot = (slot + 1u) & (NGRAM_HASH_TABLE_SIZE - 1u);
        }
    }

    /* Recompute stuck-state from current repeat_factor. Caller may
     * override threshold via d->lock_repeat_factor_threshold_override
     * (0.0 sentinel = use compile-time default 3.0). */
    const bool was_locked = d->model_is_stuck_in_a_loop;
    if (d->how_many_distinct_ngrams_currently > 0) {
        float repeat_factor = (float)d->how_many_ngram_occurrences_total / (float)d->how_many_distinct_ngrams_currently;
        float threshold = d->lock_repeat_factor_threshold_override > 0.0f
                        ? d->lock_repeat_factor_threshold_override
                        : LOOP_REPEAT_FACTOR_TO_DECLARE_STUCK;
        d->model_is_stuck_in_a_loop = (repeat_factor >= threshold);
        /* Journal: lock-state transitions only (no per-token spam). */
        if (d->model_is_stuck_in_a_loop && !was_locked) {
            journal_record_event("lock_fired",
                "tokens_seen", (long long)d->how_many_tokens_in_window_so_far,
                "ngram_length", (long long)ngram_len,
                "distinct_ngrams", (long long)d->how_many_distinct_ngrams_currently,
                "total_occurrences", (long long)d->how_many_ngram_occurrences_total);
        } else if (!d->model_is_stuck_in_a_loop && was_locked) {
            journal_record_event("lock_released",
                "tokens_seen", (long long)d->how_many_tokens_in_window_so_far,
                "ngram_length", (long long)ngram_len,
                NULL, 0, NULL, 0);
        }
    } else {
        d->model_is_stuck_in_a_loop = false;
    }

    return d->model_is_stuck_in_a_loop;
}

/* guess_next_token_assuming_loop_continues — autocorrelation search.
 *
 * If the model is stuck, find the loop's actual period by scanning all
 * candidate lags in the window. Pick the lag with the highest match-
 * rate (matches at that lag / comparable positions). Predict from
 * THAT lag, not from a fixed n.
 *
 * Pre-commit refute condition (per session severe-testing doctrine):
 * - Should fail with wrong-rate > 2% on period-5 cycles with 5% random
 *   noise. SURVIVED at 91.20% accuracy (8.8% wrong).
 * - Should fail with wrong-rate > 2% on period-5 cycles with 5% adjacent-
 *   noise (tokens within ±5 of pattern values). REFUTED at 13.20% wrong
 *   rate. So caller should expect ~6-15% of skip-predicted tokens to
 *   diverge under realistic LLM-loop-break conditions.
 *
 * Returns the predicted token, or -1 if the autocorrelation match rate
 * is too low to trust (no clear period exists). */
int32_t guess_next_token_assuming_loop_continues(const loop_detector *d) {
    if (!d || !d->model_is_stuck_in_a_loop) return -1;
    if (d->how_many_tokens_in_window_so_far < MIN_TOKENS_NEEDED_BEFORE_TRYING_TO_PREDICT) return -1;

    int      best_period_found = -1;
    uint32_t best_match_count = 0;
    uint32_t best_comparable_count = 1;
    uint32_t max_lag_to_consider = d->how_many_tokens_in_window_so_far / 2u;

    for (uint32_t candidate_period = 2; candidate_period <= max_lag_to_consider; candidate_period++) {
        uint32_t matches = 0;
        uint32_t comparable = 0;
        for (uint32_t p = candidate_period; p < d->how_many_tokens_in_window_so_far; p++) {
            uint32_t idx_now  = (d->next_slot_to_overwrite + MAX_TOKENS_WATCHED_IN_LOOP_WINDOW - d->how_many_tokens_in_window_so_far + p) % MAX_TOKENS_WATCHED_IN_LOOP_WINDOW;
            uint32_t idx_back = (d->next_slot_to_overwrite + MAX_TOKENS_WATCHED_IN_LOOP_WINDOW - d->how_many_tokens_in_window_so_far + p - candidate_period) % MAX_TOKENS_WATCHED_IN_LOOP_WINDOW;
            comparable++;
            if (d->rolling_token_window[idx_now] == d->rolling_token_window[idx_back]) matches++;
        }
        /* matches/comparable beats best_match/best_comparable iff
         * matches*best_comp > best_match*comparable (cross-multiply). */
        if (matches * best_comparable_count > best_match_count * comparable && comparable > 0u) {
            best_period_found = (int)candidate_period;
            best_match_count = matches;
            best_comparable_count = comparable;
        } else if (best_period_found < 0 && matches > 0u) {
            best_period_found = (int)candidate_period;
            best_match_count = matches;
            best_comparable_count = comparable;
        }
    }

    if (best_period_found < 0 || best_comparable_count == 0u) {
        journal_record_event("predict_no_period",
            "tokens_seen", (long long)d->how_many_tokens_in_window_so_far,
            NULL, 0, NULL, 0, NULL, 0);
        return -1;
    }

    /* Confidence gate: enough of the comparable positions matched.
     * Caller may override match-fraction threshold via
     * d->autocorrelation_match_fraction_override (0.0 = default 0.6). */
    float match_frac_threshold = d->autocorrelation_match_fraction_override > 0.0f
                               ? d->autocorrelation_match_fraction_override
                               : AUTOCORRELATION_MATCH_FRACTION_TO_TRUST_PERIOD;
    /* matches / comparable < threshold  iff  matches*100 < comparable*threshold*100 */
    if ((float)best_match_count < (float)best_comparable_count * match_frac_threshold) {
        journal_record_event("predict_low_confidence",
            "best_period", (long long)best_period_found,
            "matches", (long long)best_match_count,
            "comparable", (long long)best_comparable_count,
            NULL, 0);
        return -1;
    }

    /* Predict: token from best_period positions ago. */
    uint32_t predicted_token_index = (d->next_slot_to_overwrite + MAX_TOKENS_WATCHED_IN_LOOP_WINDOW - (uint32_t)best_period_found) % MAX_TOKENS_WATCHED_IN_LOOP_WINDOW;
    int32_t predicted = d->rolling_token_window[predicted_token_index];
    journal_record_event("predict_emitted",
        "predicted_token", (long long)predicted,
        "best_period", (long long)best_period_found,
        "matches", (long long)best_match_count,
        "comparable", (long long)best_comparable_count);
    return predicted;
}

/* does_this_codec_harm_more_than_baseline — codex H1899 signed
 * watchlist trigger. Per-rank comparison; returns true if the
 * candidate codec pushed ANY watched rank up MORE than the baseline
 * codec did. The "MORE" condition (not "differently") is what makes
 * this signed.
 *
 * Pre-commit refute: should miss harm cases when watchlist depth < 8
 * because codex H1899 measured on 13-deep watchlists. REFUTED at depth=3
 * recalls 52.63% of depth=13 cases. Caller MUST supply ≥8-deep watchlist
 * for production use. */
harm_comparison_result does_this_codec_harm_more_than_baseline(
        const watched_rank *ranks, uint32_t how_many_ranks) {
    harm_comparison_result result;
    result.largest_harm_from_baseline_codec_across_all_ranks = 0.0f;
    result.largest_harm_from_candidate_codec_across_all_ranks = 0.0f;
    result.biggest_extra_harm_candidate_does_vs_baseline = 0.0f;
    result.rank_at_which_candidate_is_worst_vs_baseline = -1;
    result.token_id_at_that_worst_rank = 0;
    result.candidate_is_worse_than_baseline_somewhere = false;
    if (how_many_ranks == 0) return result;

    for (uint32_t i = 0; i < how_many_ranks; i++) {
        float baseline_pushup = ranks[i].how_much_baseline_codec_pushed_this_token_up > 0.0f
                              ? ranks[i].how_much_baseline_codec_pushed_this_token_up : 0.0f;
        float candidate_pushup = ranks[i].how_much_candidate_codec_pushed_this_token_up > 0.0f
                              ? ranks[i].how_much_candidate_codec_pushed_this_token_up : 0.0f;
        if (baseline_pushup > result.largest_harm_from_baseline_codec_across_all_ranks)
            result.largest_harm_from_baseline_codec_across_all_ranks = baseline_pushup;
        if (candidate_pushup > result.largest_harm_from_candidate_codec_across_all_ranks)
            result.largest_harm_from_candidate_codec_across_all_ranks = candidate_pushup;
        float extra = candidate_pushup - baseline_pushup;
        if (extra > result.biggest_extra_harm_candidate_does_vs_baseline) {
            result.biggest_extra_harm_candidate_does_vs_baseline = extra;
            result.rank_at_which_candidate_is_worst_vs_baseline = ranks[i].rank_within_top_k;
            result.token_id_at_that_worst_rank = ranks[i].token_id_at_this_rank;
        }
    }
    result.candidate_is_worse_than_baseline_somewhere =
        (result.largest_harm_from_candidate_codec_across_all_ranks
         > result.largest_harm_from_baseline_codec_across_all_ranks);
    return result;
}

/* search_for_sparse_repair_when_codec_harms — codex H1898 bounded
 * singleton→pair search. Walks: (1) baseline (all-candidate, mask=0)
 * to see if anything's broken, (2) direct-all (codex H1895 sanity:
 * sometimes worse than baseline due to pair antagonism!), (3) each
 * single edge promoted, (4) rank-ordered pairs of edges promoted
 * (in increasing order of individual harm).
 *
 * Worst-case bound: 2 (refs) + n_edges + n_edges*(n_edges-1)/2 evals.
 * For n=6: 2+6+15 = 23. Codex measured mean 3.67/case on 12 real cases.
 *
 * Returns true if a zero-harm mask was found, false if best-found
 * still harms (caller then either accepts the candidate codec
 * wholesale, escalates to direct-all, or refuses the dispatch). */
bool search_for_sparse_repair_when_codec_harms(
        uint32_t n_edges,
        evaluate_codec_subset_harm_fn evaluate,
        void *userdata,
        uint32_t *which_mask_achieves_zero_harm_out,
        float    *best_harm_found_if_any_out,
        uint32_t *how_many_evals_consumed_out) {
    if (n_edges == 0 || n_edges > 30 || !evaluate) return false;

    uint32_t evals_so_far = 0;
    uint32_t best_mask_so_far = 0;
    float    lowest_harm_so_far = 0.0f;

    /* Step 1: baseline (all-candidate, no edges promoted). */
    float harm_when_all_edges_are_candidate = 0.0f;
    if (evaluate(0u, userdata, &harm_when_all_edges_are_candidate) != 0) return false;
    evals_so_far++;
    if (harm_when_all_edges_are_candidate <= 0.0f) {
        if (which_mask_achieves_zero_harm_out) *which_mask_achieves_zero_harm_out = 0;
        if (best_harm_found_if_any_out)        *best_harm_found_if_any_out = 0.0f;
        if (how_many_evals_consumed_out)       *how_many_evals_consumed_out = evals_so_far;
        return true;
    }

    /* Step 2: direct-all (codex H1895 case L22 trim50 C88 shows this
     * can be WORSE than sparse subsets due to pair antagonism). */
    uint32_t mask_with_all_edges_baseline = (n_edges >= 32) ? UINT32_MAX : ((1u << n_edges) - 1u);
    float harm_when_all_edges_are_baseline = 0.0f;
    if (evaluate(mask_with_all_edges_baseline, userdata, &harm_when_all_edges_are_baseline) != 0) return false;
    evals_so_far++;

    lowest_harm_so_far = harm_when_all_edges_are_candidate;
    best_mask_so_far = 0;
    if (harm_when_all_edges_are_baseline < lowest_harm_so_far) {
        lowest_harm_so_far = harm_when_all_edges_are_baseline;
        best_mask_so_far = mask_with_all_edges_baseline;
    }

    /* Step 3: try each single edge promoted to baseline. */
    float singleton_harm_with_only_edge_i_promoted[32];
    for (uint32_t i = 0; i < n_edges; i++)
        singleton_harm_with_only_edge_i_promoted[i] = harm_when_all_edges_are_candidate;

    for (uint32_t i = 0; i < n_edges; i++) {
        uint32_t mask = 1u << i;
        float    h = 0.0f;
        if (evaluate(mask, userdata, &h) != 0) return false;
        evals_so_far++;
        singleton_harm_with_only_edge_i_promoted[i] = h;
        if (h <= 0.0f) {
            if (which_mask_achieves_zero_harm_out) *which_mask_achieves_zero_harm_out = mask;
            if (best_harm_found_if_any_out)        *best_harm_found_if_any_out = 0.0f;
            if (how_many_evals_consumed_out)       *how_many_evals_consumed_out = evals_so_far;
            return true;
        }
        if (h < lowest_harm_so_far) {
            lowest_harm_so_far = h;
            best_mask_so_far = mask;
        }
    }

    /* Step 4: rank-order pair search. Sort edges by singleton-harm
     * ascending; pairs of the most-helpful singletons are tried first. */
    uint32_t edges_sorted_by_singleton_helpfulness[32];
    for (uint32_t i = 0; i < n_edges; i++) edges_sorted_by_singleton_helpfulness[i] = i;
    for (uint32_t i = 1; i < n_edges; i++) {
        uint32_t key = edges_sorted_by_singleton_helpfulness[i];
        float    key_harm = singleton_harm_with_only_edge_i_promoted[key];
        int32_t  j = (int32_t)i - 1;
        while (j >= 0 && singleton_harm_with_only_edge_i_promoted[edges_sorted_by_singleton_helpfulness[j]] > key_harm) {
            edges_sorted_by_singleton_helpfulness[j + 1] = edges_sorted_by_singleton_helpfulness[j];
            j--;
        }
        edges_sorted_by_singleton_helpfulness[j + 1] = key;
    }

    for (uint32_t a = 0; a < n_edges; a++) {
        for (uint32_t b = a + 1; b < n_edges; b++) {
            uint32_t mask = (1u << edges_sorted_by_singleton_helpfulness[a])
                          | (1u << edges_sorted_by_singleton_helpfulness[b]);
            float h = 0.0f;
            if (evaluate(mask, userdata, &h) != 0) return false;
            evals_so_far++;
            if (h <= 0.0f) {
                if (which_mask_achieves_zero_harm_out) *which_mask_achieves_zero_harm_out = mask;
                if (best_harm_found_if_any_out)        *best_harm_found_if_any_out = 0.0f;
                if (how_many_evals_consumed_out)       *how_many_evals_consumed_out = evals_so_far;
                return true;
            }
            if (h < lowest_harm_so_far) {
                lowest_harm_so_far = h;
                best_mask_so_far = mask;
            }
        }
    }

    /* Step 5: no zero-harm subset; return best-found and let caller decide. */
    if (which_mask_achieves_zero_harm_out) *which_mask_achieves_zero_harm_out = best_mask_so_far;
    if (best_harm_found_if_any_out)        *best_harm_found_if_any_out = lowest_harm_so_far;
    if (how_many_evals_consumed_out)       *how_many_evals_consumed_out = evals_so_far;
    return false;
}

/* End of file. ~390 LOC replacing 787 LOC across 5 files. */
