/* ds4_pillars.c — ICB + hot-expert + spec-decode scaffolds (task/#547).
 * All pillar state as static globals at file top. Env-gated; safe defaults
 * when inactive. Body implementations pending per ICB_INTEGRATION_PLAN.md +
 * design memos. */

#include "ds4_pillars.h"
#include <ctype.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DS4_PILLARS_MAX_LAYER 43u
#define DS4_PILLARS_MAX_EXPERT 256u

/* ============================== PILLAR STATE ============================== */

/* Pillar A — ICB */
static _Atomic int g_icb_initialized = 0;
static _Atomic uint64_t g_icb_recorded_count = 0;
static _Atomic uint64_t g_icb_replayed_count = 0;

/* Pillar B — Hot-expert F16 cache (margin-gated) */
static uint8_t g_hot_expert_bitmap[DS4_PILLARS_MAX_LAYER][DS4_PILLARS_MAX_EXPERT] = {{0}};
static uint32_t g_hot_expert_per_layer_count[DS4_PILLARS_MAX_LAYER] = {0};
static _Atomic int g_hot_expert_manifest_loaded = 0;
static _Atomic uint64_t g_hot_expert_check_hits = 0;
static _Atomic uint64_t g_hot_expert_check_misses = 0;

/* Pillar C — Spec-decode */
static _Atomic int g_spec_decode_initialized = 0;
static int g_spec_decode_max_draft = 0;
static float g_spec_decode_min_margin = 0.0f;
static _Atomic uint64_t g_spec_decode_drafts_proposed = 0;
static _Atomic uint64_t g_spec_decode_drafts_accepted = 0;
static _Atomic uint64_t g_spec_decode_verify_calls = 0;

static int env_set(const char *name) {
 const char *v = getenv(name);
 return (v && *v) ? 1 : 0;
}

/* ===================== PILLAR A — MTLIndirectCommandBuffer ================ */

uint32_t ds4_icb_max_commands(void) {
 static int s_cached = 0;
 static uint32_t s_value = 4096;
 if (!s_cached) {
 const char *p = getenv("DS4_ICB_MAX_COMMANDS");
 if (p && *p) {
 unsigned long v = strtoul(p, NULL, 10);
 if (v >= 64 && v <= 65536) s_value = (uint32_t)v;
 }
 s_cached = 1;
 }
 return s_value;
}

int ds4_icb_init(void) {
 if (!env_set("DS4_ICB_ACTIVE")) return 1;
 atomic_store_explicit(&g_icb_initialized, 1, memory_order_relaxed);
 fprintf(stderr, "ds4: ICB scaffold: DS4_ICB_ACTIVE — full impl pending\n");
 return 1;
}

int ds4_icb_record_decode(uint32_t n_tokens, uint32_t shape_hash) {
 (void)n_tokens; (void)shape_hash;
 return env_set("DS4_ICB_ACTIVE") ? 0 : 0;
}

int ds4_icb_record_prefill(uint32_t n_tokens, uint32_t shape_hash) {
 (void)n_tokens; (void)shape_hash;
 return env_set("DS4_ICB_ACTIVE") ? 0 : 0;
}

int ds4_icb_execute(uint32_t n_tokens, uint32_t shape_hash) {
 (void)n_tokens; (void)shape_hash;
 if (!env_set("DS4_ICB_ACTIVE")) return 0;
 atomic_fetch_add_explicit(&g_icb_replayed_count, 1, memory_order_relaxed);
 return 0;
}

void ds4_icb_print_stats(void) {
 if (!env_set("DS4_ICB_COUNT") && !env_set("DS4_ICB_ACTIVE")) return;
 fprintf(stderr, "ds4: ICB stats: recorded=%llu replayed=%llu (scaffold)\n",
 (unsigned long long)atomic_load_explicit(&g_icb_recorded_count, memory_order_relaxed),
 (unsigned long long)atomic_load_explicit(&g_icb_replayed_count, memory_order_relaxed));
}

void ds4_icb_cleanup(void) {
 if (!env_set("DS4_ICB_ACTIVE")) return;
 atomic_store_explicit(&g_icb_initialized, 0, memory_order_relaxed);
}

/* ===================== PILLAR B — Hot-expert F16 cache ==================== */
/* Design note: 8-bit route + kth-margin/row-std >= 0.04 + exact fallback keeps
 * exact active sets, rel-L2 p95 0.00335. Cache decisions MUST be margin-gated. */

float ds4_hot_expert_margin_threshold(void) {
 static int s_cached = 0;
 static float s_value = 0.04f;
 if (!s_cached) {
 const char *p = getenv("DS4_HOT_EXPERT_MARGIN");
 if (p && *p) {
 float v = strtof(p, NULL);
 if (v > 0.0f && v < 1.0f) s_value = v;
 }
 s_cached = 1;
 }
 return s_value;
}

int ds4_hot_expert_init(const char *manifest_path) {
 if (!env_set("DS4_HOT_EXPERT_ACTIVE")) return 1;
 if (!manifest_path || !*manifest_path) {
 fprintf(stderr, "ds4: hot_expert: DS4_HOT_EXPERT_ACTIVE but no manifest path\n");
 return 1;
 }
 FILE *f = fopen(manifest_path, "r");
 if (!f) {
 fprintf(stderr, "ds4: hot_expert: cannot open manifest '%s'\n", manifest_path);
 return 0;
 }
 char line[256];
 int parsed = 0;
 while (fgets(line, sizeof(line), f)) {
 char *p = line;
 while (*p && isspace((unsigned char)*p)) p++;
 if (*p == '#' || *p == '\0') continue;
 int layer, expert;
 if (sscanf(p, "%d,%d", &layer, &expert) != 2) continue;
 if (layer < 0 || layer >= (int)DS4_PILLARS_MAX_LAYER) continue;
 if (expert < 0 || expert >= (int)DS4_PILLARS_MAX_EXPERT) continue;
 if (!g_hot_expert_bitmap[layer][expert]) {
 g_hot_expert_bitmap[layer][expert] = 1;
 g_hot_expert_per_layer_count[layer]++;
 parsed++;
 }
 }
 fclose(f);
 atomic_store_explicit(&g_hot_expert_manifest_loaded, 1, memory_order_relaxed);
 fprintf(stderr, "ds4: hot_expert: loaded %d entries from %s\n", parsed, manifest_path);
 return 1;
}

bool ds4_hot_expert_is_cached(uint32_t layer, uint32_t file_pos) {
 if (!atomic_load_explicit(&g_hot_expert_manifest_loaded, memory_order_relaxed)) return false;
 if (layer >= DS4_PILLARS_MAX_LAYER || file_pos >= DS4_PILLARS_MAX_EXPERT) return false;
 const bool hit = g_hot_expert_bitmap[layer][file_pos] != 0;
 if (hit) atomic_fetch_add_explicit(&g_hot_expert_check_hits, 1, memory_order_relaxed);
 else atomic_fetch_add_explicit(&g_hot_expert_check_misses, 1, memory_order_relaxed);
 /* F16 buffer not populated yet — return false until then. */
 return false;
}

bool ds4_hot_expert_is_cached_margin_gated(uint32_t layer, uint32_t file_pos, float margin) {
 return ds4_hot_expert_is_cached(layer, file_pos) && margin >= ds4_hot_expert_margin_threshold();
}

uint64_t ds4_hot_expert_f16_offset(uint32_t layer, uint32_t file_pos) {
 (void)layer; (void)file_pos;
 return 0;
}

void ds4_hot_expert_print_stats(void) {
 if (!env_set("DS4_HOT_EXPERT_COUNT") && !env_set("DS4_HOT_EXPERT_ACTIVE")) return;
 uint32_t total = 0;
 for (uint32_t l = 0; l < DS4_PILLARS_MAX_LAYER; l++) total += g_hot_expert_per_layer_count[l];
 fprintf(stderr, "ds4: hot_expert: total_hot=%u hits=%llu misses=%llu\n",
 total,
 (unsigned long long)atomic_load_explicit(&g_hot_expert_check_hits, memory_order_relaxed),
 (unsigned long long)atomic_load_explicit(&g_hot_expert_check_misses, memory_order_relaxed));
}

void ds4_hot_expert_cleanup(void) { /* TODO: release F16 MTLBuffer pool. */ }

/* ===================== PILLAR C — Spec-decode ============================= */

int ds4_spec_decode_init(int max_draft, float min_margin) {
 if (!env_set("DS4_SPEC_ACTIVE")) return 1;
 g_spec_decode_max_draft = max_draft;
 g_spec_decode_min_margin = min_margin;
 atomic_store_explicit(&g_spec_decode_initialized, 1, memory_order_relaxed);
 fprintf(stderr, "ds4: spec_decode: max_draft=%d margin=%.3f — orchestration pending\n",
 max_draft, min_margin);
 return 1;
}

int ds4_spec_decode_propose(uint32_t *out_drafts, int n_draft) {
 (void)out_drafts;
 if (!env_set("DS4_SPEC_ACTIVE")) return 0;
 if (!atomic_load_explicit(&g_spec_decode_initialized, memory_order_relaxed)) return 0;
 atomic_fetch_add_explicit(&g_spec_decode_drafts_proposed, (uint64_t)n_draft, memory_order_relaxed);
 return 0;
}

int ds4_spec_decode_verify(const uint32_t *drafts, int n_draft,
 uint32_t *out_accept_prefix_len, uint32_t *out_main_token) {
 (void)drafts; (void)n_draft;
 if (!env_set("DS4_SPEC_ACTIVE")) return 0;
 if (!atomic_load_explicit(&g_spec_decode_initialized, memory_order_relaxed)) return 0;
 if (out_accept_prefix_len) *out_accept_prefix_len = 0;
 if (out_main_token) *out_main_token = 0;
 atomic_fetch_add_explicit(&g_spec_decode_verify_calls, 1, memory_order_relaxed);
 return 0;
}

void ds4_spec_decode_print_stats(void) {
 if (!env_set("DS4_SPEC_COUNT") && !env_set("DS4_SPEC_ACTIVE")) return;
 fprintf(stderr, "ds4: spec_decode: proposed=%llu accepted=%llu verifies=%llu (scaffold)\n",
 (unsigned long long)atomic_load_explicit(&g_spec_decode_drafts_proposed, memory_order_relaxed),
 (unsigned long long)atomic_load_explicit(&g_spec_decode_drafts_accepted, memory_order_relaxed),
 (unsigned long long)atomic_load_explicit(&g_spec_decode_verify_calls, memory_order_relaxed));
}

void ds4_spec_decode_cleanup(void) {
 if (!env_set("DS4_SPEC_ACTIVE")) return;
 atomic_store_explicit(&g_spec_decode_initialized, 0, memory_order_relaxed);
}
