/* DS4 expert-table — runtime per-(layer, expert) tier classification.
 *
 * : implements the unified substrate spec from
 * audits/expert_table_design_20260523.md. Storage + loader + router-intercept
 * helper. Integration into ds4.c documented in ds4_expert_table.h.
 *
 * Per CLAUDE.md STICKY HAZARD: any DS4 launch using this code must include
 * --prefill-metal-phases auto. The mask infrastructure does NOT change the
 * wired-memory cap; it changes WHICH experts are dispatched.
 */

#include "ds4_expert_table.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

uint8_t g_expert_tier[DS4_N_LAYER * DS4_N_EXPERT] = {0};
int g_expert_table_initialized = 0;

/* Set by ds4.c at engine init after weights_bind. The intercept in
 * layer_topk_selected_experts_from_probs uses pointer arithmetic against
 * this base to derive the layer index. Forward-declared in ds4.c as well. */
const void *g_ds4_layer_base_for_mask = NULL;

static int load_mask_csv(const char *path) {
 FILE *f = fopen(path, "r");
 if (!f) {
 fprintf(stderr, "ds4_load_expert_table: cannot open mask file '%s'\n", path);
 return -1;
 }
 int n_masked = 0;
 int n_lines = 0;
 char line[256];
 while (fgets(line, sizeof(line), f)) {
 n_lines++;
 /* Skip header / comment / blank lines */
 if (line[0] == '#' || line[0] == '\n' || line[0] == '\r' || line[0] == '\0') continue;
 int layer = -1, expert = -1;
 if (sscanf(line, "%d,%d", &layer, &expert) != 2) {
 /* Skip any malformed line silently — manifests may have header
 * variations from build_expert_mask.py / pin_top*.csv etc. */
 continue;
 }
 if (layer < 0 || layer >= DS4_N_LAYER) continue;
 if (expert < 0 || expert >= DS4_N_EXPERT) continue;
 /* Optional safety: refuse to mask layers 0-2 (hash layers per
 * trim_ladder_to_64gb_20260523.md). Comment-out if needed for
 * testing. */
 if (layer < 3) {
 fprintf(stderr, "ds4_load_expert_table: refusing to mask hash-layer %d expert %d\n",
 layer, expert);
 continue;
 }
 g_expert_tier[layer * DS4_N_EXPERT + expert] = DS4_EXPERT_TIER_MASKED;
 n_masked++;
 }
 fclose(f);
 fprintf(stderr, "ds4_load_expert_table: %d experts MASKED from %s (%d lines parsed)\n",
 n_masked, path, n_lines);
 return n_masked;
}

static int load_hot_csv(const char *path) {
 FILE *f = fopen(path, "r");
 if (!f) {
 fprintf(stderr, "ds4_load_expert_table: cannot open hot file '%s'\n", path);
 return -1;
 }
 int n_hot = 0;
 int n_lines = 0;
 char line[256];
 while (fgets(line, sizeof(line), f)) {
 n_lines++;
 if (line[0] == '#' || line[0] == '\n' || line[0] == '\r' || line[0] == '\0') continue;
 int layer = -1, expert = -1;
 if (sscanf(line, "%d,%d", &layer, &expert) != 2) continue;
 if (layer < 0 || layer >= DS4_N_LAYER) continue;
 if (expert < 0 || expert >= DS4_N_EXPERT) continue;
 /* Mask wins over hot — leave masked entries alone */
 if (g_expert_tier[layer * DS4_N_EXPERT + expert] != DS4_EXPERT_TIER_MASKED) {
 g_expert_tier[layer * DS4_N_EXPERT + expert] = DS4_EXPERT_TIER_HOT;
 n_hot++;
 }
 }
 fclose(f);
 fprintf(stderr, "ds4_load_expert_table: %d experts marked HOT from %s (%d lines parsed)\n",
 n_hot, path, n_lines);
 return n_hot;
}

int ds4_load_expert_table(const char *mask_path, const char *hot_path) {
 memset(g_expert_tier, 0, sizeof(g_expert_tier));
 g_expert_table_initialized = 0;

 int err = 0;
 if (mask_path && *mask_path) {
 if (load_mask_csv(mask_path) < 0) err = -1;
 }
 if (hot_path && *hot_path) {
 if (load_hot_csv(hot_path) < 0) err = -1;
 }

 if (err) return -1;

 /* Mark initialized only if at least one file was loaded; otherwise
 * the table is logically inactive (all NORMAL). */
 if ((mask_path && *mask_path) || (hot_path && *hot_path)) {
 g_expert_table_initialized = 1;
 }
 return 0;
}

void ds4_apply_expert_mask_to_selection(uint32_t layer, float *selection) {
 if (!g_expert_table_initialized) return;
 if (layer >= DS4_N_LAYER) return;
 const uint8_t *tier_row = &g_expert_tier[layer * DS4_N_EXPERT];
 for (int i = 0; i < DS4_N_EXPERT; i++) {
 if (tier_row[i] == DS4_EXPERT_TIER_MASKED) {
 selection[i] = -INFINITY;
 }
 }
}

int ds4_expert_table_print_stats(void) {
 if (!g_expert_table_initialized) {
 fprintf(stderr, "ds4_expert_table: not initialized (no manifests loaded)\n");
 return 0;
 }
 int total_masked = 0, total_hot = 0;
 fprintf(stderr, "ds4_expert_table per-layer stats:\n");
 fprintf(stderr, " L masked hot normal\n");
 for (int L = 0; L < DS4_N_LAYER; L++) {
 int n_m = 0, n_h = 0;
 for (int e = 0; e < DS4_N_EXPERT; e++) {
 uint8_t t = g_expert_tier[L * DS4_N_EXPERT + e];
 if (t == DS4_EXPERT_TIER_MASKED) n_m++;
 else if (t == DS4_EXPERT_TIER_HOT) n_h++;
 }
 total_masked += n_m;
 total_hot += n_h;
 if (n_m > 0 || n_h > 0) {
 fprintf(stderr, " %2d %5d %4d %5d\n", L, n_m, n_h, DS4_N_EXPERT - n_m - n_h);
 }
 }
 fprintf(stderr, " TOTAL: %d masked, %d hot, %d normal (of %d total)\n",
 total_masked, total_hot, DS4_N_LAYER * DS4_N_EXPERT - total_masked - total_hot,
 DS4_N_LAYER * DS4_N_EXPERT);
 return total_masked;
}
