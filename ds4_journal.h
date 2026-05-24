/* ds4_journal.h — append-only SQLite-WAL transaction journal.
 * Opt-in via JOURNAL=1 make (defines DS4_JOURNAL_ENABLE). Default build
 * inlines all functions as no-ops → zero perf impact. */
#ifndef DS4_JOURNAL_H
#define DS4_JOURNAL_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ds4_journal ds4_journal;

#ifdef DS4_JOURNAL_ENABLE

ds4_journal *ds4_journal_open(const char *db_path,
 uint32_t flush_threshold,
 uint32_t flush_interval_ms);
int64_t ds4_journal_begin_session(ds4_journal *j, const char *model_path,
 int ctx_size, const char *backend,
 const char *options_json);
void ds4_journal_emit_token(ds4_journal *j, int64_t session_id,
 int32_t seq_pos, int32_t token_id,
 float logp, int64_t wall_us);
void ds4_journal_emit_routing(ds4_journal *j, int64_t session_id,
 int32_t seq_pos, int32_t layer_id,
 const uint16_t *selected, const float *weights,
 int k, float margin);
void ds4_journal_emit_event(ds4_journal *j, int64_t session_id,
 const char *kind, const char *payload_json);
void ds4_journal_flush(ds4_journal *j);
void ds4_journal_close(ds4_journal *j);

#else /* journal compiled out — inline no-ops, zero overhead */

static inline ds4_journal *ds4_journal_open(const char *a, uint32_t b, uint32_t c)
 { (void)a; (void)b; (void)c; return NULL; }
static inline int64_t ds4_journal_begin_session(ds4_journal *a, const char *b,
 int c, const char *d, const char *e)
 { (void)a; (void)b; (void)c; (void)d; (void)e; return 0; }
static inline void ds4_journal_emit_token(ds4_journal *a, int64_t b, int32_t c,
 int32_t d, float e, int64_t f)
 { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; }
static inline void ds4_journal_emit_routing(ds4_journal *a, int64_t b, int32_t c,
 int32_t d, const uint16_t *e,
 const float *f, int g, float h)
 { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; (void)h; }
static inline void ds4_journal_emit_event(ds4_journal *a, int64_t b,
 const char *c, const char *d)
 { (void)a; (void)b; (void)c; (void)d; }
static inline void ds4_journal_flush(ds4_journal *a) { (void)a; }
static inline void ds4_journal_close(ds4_journal *a) { (void)a; }

#endif

#ifdef __cplusplus
}
#endif

#endif
