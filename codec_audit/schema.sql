-- codec_audit/schema.sql
--
-- Append-only audit database for DS4 codec quality measurements.
--
-- Design lenses applied:
--   Page/Brin (BigTable): structured key-value with append-only commit log
--   Vitalik (provenance): every row carries content-hash + parent-pointers
--   Knuth (mature data structures): one schema, indexed for every common query
--   Tao (right abstraction): 5-tuple primary key (layer, kind, expert, codec, hidden)
--                            surfaces the structure the codex arc discovered
--   Sherwood (field+seed): hidden_sha256 records the activation context
--                          H1800/H1801 proved codec quality is hidden-conditioned
--   Donoho (rigor): n_subsample + seed_int always recorded so CI is computable
--   Carmack/Geohotz (MVP): five tables, no joins required for primary queries
--
-- Invariants enforced via triggers:
--   - journal is append-only (no UPDATE / DELETE)
--   - measurement is immutable (no UPDATE / DELETE)
--   - codec / source_tensor are immutable after insert
--
-- Complexity targets:
--   - load FP4 source for (layer, kind, expert) → O(1) lookup via UNIQUE index
--   - insert measurement → O(log n) via B-tree index, but amortized O(1) for batch
--   - query "best codec per expert" → O(n_experts) GROUP BY with covering index
--   - cross-codec quality comparison → O(n_codecs) GROUP BY

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

-- =========================================================================
-- Journal: append-only record of every action.
-- =========================================================================
CREATE TABLE IF NOT EXISTS journal (
    entry_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    action     TEXT NOT NULL,
    target     TEXT,
    payload    TEXT,
    git_commit TEXT,
    pid        INTEGER
);
CREATE INDEX IF NOT EXISTS idx_journal_ts ON journal(ts);
CREATE INDEX IF NOT EXISTS idx_journal_action ON journal(action);

CREATE TRIGGER IF NOT EXISTS journal_no_update
  BEFORE UPDATE ON journal
  BEGIN SELECT RAISE(ABORT, 'journal is append-only'); END;

CREATE TRIGGER IF NOT EXISTS journal_no_delete
  BEFORE DELETE ON journal
  BEGIN SELECT RAISE(ABORT, 'journal is append-only'); END;

-- =========================================================================
-- Codec registry: each row is an immutable codec spec.
-- =========================================================================
CREATE TABLE IF NOT EXISTS codec (
    codec_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name                  TEXT NOT NULL UNIQUE,
    family                TEXT NOT NULL,
    bits_per_pair         REAL NOT NULL,
    storage_byte_per_pair REAL NOT NULL,
    params_json           TEXT,
    spec_sha256           TEXT NOT NULL,
    created_at            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_codec_family ON codec(family);

CREATE TRIGGER IF NOT EXISTS codec_no_update
  BEFORE UPDATE ON codec
  BEGIN SELECT RAISE(ABORT, 'codec specs are immutable'); END;

CREATE TRIGGER IF NOT EXISTS codec_no_delete
  BEFORE DELETE ON codec
  BEGIN SELECT RAISE(ABORT, 'codec specs are immutable'); END;

-- =========================================================================
-- Source tensor registry: one row per (layer, kind, expert).
-- sha256 of FP4 source confirms identity across runs.
-- =========================================================================
CREATE TABLE IF NOT EXISTS source_tensor (
    source_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    layer       INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    expert      INTEGER NOT NULL,
    shape_json  TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    n_weights   INTEGER NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(layer, kind, expert)
);
CREATE INDEX IF NOT EXISTS idx_source_layer_kind ON source_tensor(layer, kind);
CREATE INDEX IF NOT EXISTS idx_source_sha256 ON source_tensor(sha256);

CREATE TRIGGER IF NOT EXISTS source_tensor_no_update
  BEFORE UPDATE ON source_tensor
  BEGIN SELECT RAISE(ABORT, 'source_tensor is immutable'); END;

-- =========================================================================
-- Hidden vector registry: H1800/H1801 — codec quality is hidden-conditioned.
-- =========================================================================
CREATE TABLE IF NOT EXISTS hidden_vector (
    hidden_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,       -- 'gaussian', 'router_attractor', 'router_row_mix', 'static_zero'
    seed_int    INTEGER,             -- RNG seed when applicable
    dim         INTEGER NOT NULL,
    sha256      TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_hidden_kind ON hidden_vector(kind);

-- =========================================================================
-- Run: groups measurements from one session/sweep.
-- =========================================================================
CREATE TABLE IF NOT EXISTS run (
    run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    started_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    ended_at     TEXT,
    host         TEXT,
    git_commit   TEXT,
    params_json  TEXT
);

-- =========================================================================
-- Measurement: 6-aberration vector per (source, codec, hidden) cell.
-- This is the row-level RECONSTRUCTION measurement (H1748-H1789).
-- =========================================================================
CREATE TABLE IF NOT EXISTS measurement (
    measure_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id     INTEGER NOT NULL REFERENCES source_tensor(source_id),
    codec_id      INTEGER NOT NULL REFERENCES codec(codec_id),
    hidden_id     INTEGER REFERENCES hidden_vector(hidden_id),
    n_subsample   INTEGER NOT NULL,
    seed_int      INTEGER NOT NULL,
    -- Six aberration metrics (per error_distribution_analyzer.py)
    rel_l2        REAL NOT NULL,
    sign_flip     REAL,
    exactly_zero  REAL,
    max_abs       REAL,
    mean_abs      REAL,
    angle_rms_rad REAL,
    -- Optional richer detail
    strata_json   TEXT,            -- magnitude-decile rel_L2 (10 floats)
    outliers_json TEXT,            -- worst-row indices + rel_L2
    notes         TEXT,
    run_id        INTEGER REFERENCES run(run_id),
    ts            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_measure_src_codec
  ON measurement(source_id, codec_id, hidden_id);
CREATE INDEX IF NOT EXISTS idx_measure_codec
  ON measurement(codec_id);
CREATE INDEX IF NOT EXISTS idx_measure_run
  ON measurement(run_id);

CREATE TRIGGER IF NOT EXISTS measurement_no_update
  BEFORE UPDATE ON measurement
  BEGIN SELECT RAISE(ABORT, 'measurements are immutable'); END;

CREATE TRIGGER IF NOT EXISTS measurement_no_delete
  BEFORE DELETE ON measurement
  BEGIN SELECT RAISE(ABORT, 'measurements are immutable'); END;

-- =========================================================================
-- Composition measurement: gate/up/activation/down errors (H1795-H1801).
-- This is the COMPOSITION-LEVEL quality (the load-bearing organ).
-- A separate table because the metrics are different from row reconstruction.
-- =========================================================================
CREATE TABLE IF NOT EXISTS composition_measurement (
    comp_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id            INTEGER NOT NULL REFERENCES source_tensor(source_id),
    codec_id             INTEGER NOT NULL REFERENCES codec(codec_id),
    hidden_id            INTEGER NOT NULL REFERENCES hidden_vector(hidden_id),
    gate_rel_l2          REAL,
    up_rel_l2            REAL,
    down_row_rel_l2      REAL,
    activation_rel_l2    REAL,
    activation_cos       REAL,
    selected_down_rel_l2 REAL,
    selected_rows_json   TEXT,         -- which down rows were measured
    run_id               INTEGER REFERENCES run(run_id),
    ts                   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_comp_src_codec_hidden
  ON composition_measurement(source_id, codec_id, hidden_id);

CREATE TRIGGER IF NOT EXISTS composition_no_update
  BEFORE UPDATE ON composition_measurement
  BEGIN SELECT RAISE(ABORT, 'composition measurements are immutable'); END;

CREATE TRIGGER IF NOT EXISTS composition_no_delete
  BEFORE DELETE ON composition_measurement
  BEGIN SELECT RAISE(ABORT, 'composition measurements are immutable'); END;

-- =========================================================================
-- Common views (no joins needed by caller).
-- =========================================================================
CREATE VIEW IF NOT EXISTS v_measurement_full AS
SELECT
    m.measure_id, m.ts,
    s.layer, s.kind, s.expert, s.sha256 AS source_sha256,
    c.name AS codec_name, c.family AS codec_family,
    c.bits_per_pair, c.storage_byte_per_pair,
    h.kind AS hidden_kind, h.seed_int AS hidden_seed,
    m.n_subsample, m.seed_int,
    m.rel_l2, m.sign_flip, m.exactly_zero,
    m.max_abs, m.mean_abs, m.angle_rms_rad,
    m.run_id
FROM measurement m
JOIN source_tensor s ON s.source_id = m.source_id
JOIN codec c         ON c.codec_id  = m.codec_id
LEFT JOIN hidden_vector h ON h.hidden_id = m.hidden_id;

CREATE VIEW IF NOT EXISTS v_composition_full AS
SELECT
    cm.comp_id, cm.ts,
    s.layer, s.kind, s.expert,
    c.name AS codec_name, c.bits_per_pair,
    h.kind AS hidden_kind,
    cm.gate_rel_l2, cm.up_rel_l2, cm.down_row_rel_l2,
    cm.activation_rel_l2, cm.activation_cos, cm.selected_down_rel_l2,
    cm.run_id
FROM composition_measurement cm
JOIN source_tensor s ON s.source_id = cm.source_id
JOIN codec c         ON c.codec_id  = cm.codec_id
JOIN hidden_vector h ON h.hidden_id = cm.hidden_id;
