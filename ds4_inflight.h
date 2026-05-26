/* ds4_inflight.h — public declarations for ds4_inflight.c.
 *
 * Consolidated module replacing ds4_cache_lock_detector, ds4_signed_watchlist,
 * ds4_bounded_packet_search, ds4_codec_k_dispatcher. Self-explanatory names.
 */
#ifndef DS4_INFLIGHT_H
#define DS4_INFLIGHT_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_TOKENS_WATCHED_IN_LOOP_WINDOW              200
#define LOOP_PATTERN_NGRAM_LENGTH                      5
#define MAX_NGRAM_LENGTH_FOR_OVERRIDE                  32
#define LOOP_REPEAT_FACTOR_TO_DECLARE_STUCK            3.0f
#define NGRAM_HASH_TABLE_SIZE                          512

/* Tripwires: link error on symbol shadowing (loud + immediate); static
 * assertion below catches pre-C99 builds that lack inline+stdbool. */
#if defined(__STDC_VERSION__)
_Static_assert(__STDC_VERSION__ >= 199901L,
               "ds4_inflight requires C99 or later for stdbool/inline");
#endif

typedef struct {
    int32_t  rolling_token_window[MAX_TOKENS_WATCHED_IN_LOOP_WINDOW];
    uint32_t how_many_tokens_in_window_so_far;
    uint32_t next_slot_to_overwrite;
    uint64_t ngram_hash_at_slot[NGRAM_HASH_TABLE_SIZE];
    uint16_t ngram_repeat_count_at_slot[NGRAM_HASH_TABLE_SIZE];
    uint32_t how_many_distinct_ngrams_currently;
    uint32_t how_many_ngram_occurrences_total;
    bool     model_is_stuck_in_a_loop;
    /* Tunable thresholds (override after zero-init to change behavior).
     * Sentinel 0.0 = use compile-time default. */
    float    lock_repeat_factor_threshold_override;       /* default 3.0 */
    float    autocorrelation_match_fraction_override;     /* default 0.6 */
    /* n-gram length used for repetition tracking. Real-LLM measurement
     * (far_end_tokenize_saved_trace.py 2026-05-26 on Qwen3.5-4B P01
     * 1801-line repetition trace): n=15 fires 3138 tokens earlier than
     * default n=5. Caller may override; sentinel 0 = use compile-time
     * default (5). Clamped to [2, MAX_NGRAM_LENGTH_FOR_OVERRIDE]. */
    uint32_t ngram_length_override;                       /* default 5  */
} loop_detector;

typedef struct {
    int32_t rank_within_top_k;
    int32_t token_id_at_this_rank;
    float   how_much_baseline_codec_pushed_this_token_up;
    float   how_much_candidate_codec_pushed_this_token_up;
} watched_rank;

typedef struct {
    float    largest_harm_from_baseline_codec_across_all_ranks;
    float    largest_harm_from_candidate_codec_across_all_ranks;
    float    biggest_extra_harm_candidate_does_vs_baseline;
    int32_t  rank_at_which_candidate_is_worst_vs_baseline;
    int32_t  token_id_at_that_worst_rank;
    bool     candidate_is_worse_than_baseline_somewhere;
} harm_comparison_result;

typedef int (*evaluate_codec_subset_harm_fn)(uint32_t which_edges_are_baseline_mask,
                                              void *userdata, float *signed_harm_out);

bool watch_token_emerging_from_model(loop_detector *d, int32_t token);
int32_t guess_next_token_assuming_loop_continues(const loop_detector *d);
/* DEPTH WARNING: at uniformly-distributed harm-rank scale (10K trials), this
 * function recalls only 10% of true harms at how_many_ranks=3, 20% at 5,
 * 40% at 8, 60% at 13. Caller MUST supply how_many_ranks >= 8 for production
 * use; >= 13 for the reference 23/23 recall measurement. */
harm_comparison_result does_this_codec_harm_more_than_baseline(
        const watched_rank *ranks, uint32_t how_many_ranks);
bool search_for_sparse_repair_when_codec_harms(
        uint32_t n_edges, evaluate_codec_subset_harm_fn evaluate, void *userdata,
        uint32_t *which_mask_achieves_zero_harm_out,
        float    *best_harm_found_if_any_out,
        uint32_t *how_many_evals_consumed_out);

#ifdef __cplusplus
}
#endif

#endif