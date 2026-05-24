/* ds4_journal.c — append-only SQLite-WAL transaction journal (opt-in via
 * JOURNAL=1 make). Synchronous prepared-statement emits per call; SQLite WAL
 * handles its own batching natively. No background thread, no usleep
 * polling, no ring buffer, no mutex. */
#ifndef DS4_JOURNAL_ENABLE
#else

#include "ds4_journal.h"
#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
struct ds4_journal {
 sqlite3 *db;
 sqlite3_stmt *insert_session;
 sqlite3_stmt *insert_token;
 sqlite3_stmt *insert_routing;
 sqlite3_stmt *insert_event;
};
static struct ds4_journal g_journal_handle = {0};
static int g_journal_is_open = 0;

/* Append-only triggers at the SQL layer reject UPDATE/DELETE — verified live. */
static const char SCHEMA_DDL[] =
 "PRAGMA journal_mode=WAL;"
 "PRAGMA synchronous=NORMAL;"
 "PRAGMA temp_store=MEMORY;"
 "CREATE TABLE IF NOT EXISTS session ("
 " id INTEGER PRIMARY KEY AUTOINCREMENT,"
 " started_unix INTEGER NOT NULL,"
 " model_path TEXT NOT NULL,"
 " ctx_size INTEGER NOT NULL,"
 " backend TEXT NOT NULL,"
 " options_json TEXT);"
 "CREATE TABLE IF NOT EXISTS token ("
 " session_id INTEGER NOT NULL,"
 " seq_pos INTEGER NOT NULL,"
 " token_id INTEGER NOT NULL,"
 " logp REAL,"
 " wall_us INTEGER,"
 " PRIMARY KEY (session_id, seq_pos)) WITHOUT ROWID;"
 "CREATE TABLE IF NOT EXISTS routing ("
 " session_id INTEGER NOT NULL,"
 " seq_pos INTEGER NOT NULL,"
 " layer_id INTEGER NOT NULL,"
 " selected BLOB NOT NULL,"
 " weights BLOB NOT NULL,"
 " margin REAL,"
 " PRIMARY KEY (session_id, seq_pos, layer_id)) WITHOUT ROWID;"
 "CREATE TABLE IF NOT EXISTS event ("
 " id INTEGER PRIMARY KEY AUTOINCREMENT,"
 " session_id INTEGER NOT NULL,"
 " unix_us INTEGER NOT NULL,"
 " kind TEXT NOT NULL,"
 " payload_json TEXT);"
 "CREATE INDEX IF NOT EXISTS idx_token_session ON token(session_id);"
 "CREATE INDEX IF NOT EXISTS idx_routing_layer ON routing(layer_id);"
 "CREATE INDEX IF NOT EXISTS idx_event_kind ON event(kind);"
 "CREATE TRIGGER IF NOT EXISTS jno_upd_token BEFORE UPDATE ON token BEGIN SELECT RAISE(ABORT,'journal is append-only'); END;"
 "CREATE TRIGGER IF NOT EXISTS jno_del_token BEFORE DELETE ON token BEGIN SELECT RAISE(ABORT,'journal is append-only'); END;"
 "CREATE TRIGGER IF NOT EXISTS jno_upd_route BEFORE UPDATE ON routing BEGIN SELECT RAISE(ABORT,'journal is append-only'); END;"
 "CREATE TRIGGER IF NOT EXISTS jno_del_route BEFORE DELETE ON routing BEGIN SELECT RAISE(ABORT,'journal is append-only'); END;"
 "CREATE TRIGGER IF NOT EXISTS jno_upd_event BEFORE UPDATE ON event BEGIN SELECT RAISE(ABORT,'journal is append-only'); END;"
 "CREATE TRIGGER IF NOT EXISTS jno_del_event BEFORE DELETE ON event BEGIN SELECT RAISE(ABORT,'journal is append-only'); END;";

ds4_journal *ds4_journal_open(const char *db_path, uint32_t a, uint32_t b) {
 (void)a; (void)b; /* legacy flush_threshold + flush_interval_ms — ignored now */
 if (!db_path || g_journal_is_open) return g_journal_is_open ? &g_journal_handle : NULL;
 if (sqlite3_open(db_path, &g_journal_handle.db) != SQLITE_OK) {
 fprintf(stderr, "ds4_journal: cannot open %s\n", db_path);
 return NULL;
 }
 char *err = NULL;
 if (sqlite3_exec(g_journal_handle.db, SCHEMA_DDL, NULL, NULL, &err) != SQLITE_OK) {
 fprintf(stderr, "ds4_journal: schema failed: %s\n", err ? err : "?");
 sqlite3_free(err);
 sqlite3_close(g_journal_handle.db);
 g_journal_handle.db = NULL;
 return NULL;
 }
 sqlite3_prepare_v2(g_journal_handle.db,
 "INSERT INTO session (started_unix, model_path, ctx_size, backend, options_json) VALUES (?1, ?2, ?3, ?4, ?5)",
 -1, &g_journal_handle.insert_session, NULL);
 sqlite3_prepare_v2(g_journal_handle.db,
 "INSERT INTO token (session_id, seq_pos, token_id, logp, wall_us) VALUES (?1, ?2, ?3, ?4, ?5)",
 -1, &g_journal_handle.insert_token, NULL);
 sqlite3_prepare_v2(g_journal_handle.db,
 "INSERT INTO routing (session_id, seq_pos, layer_id, selected, weights, margin) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
 -1, &g_journal_handle.insert_routing, NULL);
 sqlite3_prepare_v2(g_journal_handle.db,
 "INSERT INTO event (session_id, unix_us, kind, payload_json) VALUES (?1, ?2, ?3, ?4)",
 -1, &g_journal_handle.insert_event, NULL);
 g_journal_is_open = 1;
 return &g_journal_handle;
}

int64_t ds4_journal_begin_session(ds4_journal *j, const char *model_path, int ctx_size, const char *backend, const char *opts) {
 if (!j || !g_journal_is_open) return 0;
 sqlite3_reset(j->insert_session);
 sqlite3_bind_int64(j->insert_session, 1, (int64_t)time(NULL));
 sqlite3_bind_text (j->insert_session, 2, model_path ? model_path : "", -1, SQLITE_TRANSIENT);
 sqlite3_bind_int (j->insert_session, 3, ctx_size);
 sqlite3_bind_text (j->insert_session, 4, backend ? backend : "", -1, SQLITE_TRANSIENT);
 sqlite3_bind_text (j->insert_session, 5, opts ? opts : "", -1, SQLITE_TRANSIENT);
 if (sqlite3_step(j->insert_session) != SQLITE_DONE) return 0;
 return sqlite3_last_insert_rowid(j->db);
}

void ds4_journal_emit_token(ds4_journal *j, int64_t sid, int32_t seq_pos, int32_t tok, float logp, int64_t wall_us) {
 if (!j || !g_journal_is_open) return;
 sqlite3_reset(j->insert_token);
 sqlite3_bind_int64(j->insert_token, 1, sid);
 sqlite3_bind_int (j->insert_token, 2, seq_pos);
 sqlite3_bind_int (j->insert_token, 3, tok);
 sqlite3_bind_double(j->insert_token, 4, (double)logp);
 sqlite3_bind_int64(j->insert_token, 5, wall_us);
 sqlite3_step(j->insert_token);
}

void ds4_journal_emit_routing(ds4_journal *j, int64_t sid, int32_t seq_pos, int32_t layer_id,
 const uint16_t *selected, const float *weights, int k, float margin) {
 if (!j || !g_journal_is_open || !selected || !weights || k <= 0 || k > 16) return;
 sqlite3_reset(j->insert_routing);
 sqlite3_bind_int64(j->insert_routing, 1, sid);
 sqlite3_bind_int (j->insert_routing, 2, seq_pos);
 sqlite3_bind_int (j->insert_routing, 3, layer_id);
 sqlite3_bind_blob (j->insert_routing, 4, selected, (int)(k * sizeof(uint16_t)), SQLITE_TRANSIENT);
 sqlite3_bind_blob (j->insert_routing, 5, weights, (int)(k * sizeof(float)), SQLITE_TRANSIENT);
 sqlite3_bind_double(j->insert_routing, 6, (double)margin);
 sqlite3_step(j->insert_routing);
}

void ds4_journal_emit_event(ds4_journal *j, int64_t sid, const char *kind, const char *payload_json) {
 if (!j || !g_journal_is_open || !kind) return;
 struct timeval tv;
 gettimeofday(&tv, NULL);
 sqlite3_reset(j->insert_event);
 sqlite3_bind_int64(j->insert_event, 1, sid);
 sqlite3_bind_int64(j->insert_event, 2, (int64_t)tv.tv_sec * 1000000LL + (int64_t)tv.tv_usec);
 sqlite3_bind_text (j->insert_event, 3, kind, -1, SQLITE_TRANSIENT);
 sqlite3_bind_text (j->insert_event, 4, payload_json ? payload_json : "", -1, SQLITE_TRANSIENT);
 sqlite3_step(j->insert_event);
}

void ds4_journal_flush(ds4_journal *j) { (void)j; /* synchronous emits — no-op */ }

void ds4_journal_close(ds4_journal *j) {
 if (!j || !g_journal_is_open) return;
 if (j->insert_session) sqlite3_finalize(j->insert_session);
 if (j->insert_token) sqlite3_finalize(j->insert_token);
 if (j->insert_routing) sqlite3_finalize(j->insert_routing);
 if (j->insert_event) sqlite3_finalize(j->insert_event);
 if (j->db) sqlite3_close(j->db);
 memset(&g_journal_handle, 0, sizeof(g_journal_handle));
 g_journal_is_open = 0;
}

#endif /* DS4_JOURNAL_ENABLE */
