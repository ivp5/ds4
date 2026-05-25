# codec_audit — consolidated DS4 codec quality framework

Replaces 7 ad-hoc scripts from `tmp/20260525_codec_vs_iq2_audit/` (~2424 LOC)
with a SQLite-backed module (~900 LOC) + append-only journal.

## Architecture

```
codec_audit/
├── schema.sql       # Append-only DB schema (codec/source/hidden/measurement/journal)
├── db.py            # SQLite plumbing (WAL, foreign_keys, journal helper)
├── registry.py      # codec/source/hidden registration (idempotent)
├── measure.py       # aberration_vector + composition_metrics (pure funcs)
├── decoders.py      # IQ2_XXS / polar PLR2 / VQ K=256 decoders + FP4 source loader
├── sweep.py         # scan_layer_kind + scan_cross_codec orchestrators
├── cli.py           # python3 -m codec_audit ...
├── __main__.py      # entry point
└── README.md
```

## Design lenses applied

- **Larry Page / Sergey Brin** (BigTable): structured key-value store with
  append-only commit log; WAL mode for concurrent reads during writes.
- **Vitalik Buterin** (provenance): every row carries `sha256` content-hash
  + immutable foreign-key references; codec / source / measurement rows
  are NEVER updated (triggers enforce).
- **Knuth** (mature data structures): one schema, indexed for every common
  query. B-tree primary keys → O(log n) inserts; UNIQUE indexes →
  O(1) lookups for registration.
- **Tao** (right abstraction): 5-tuple primary key
  (layer, kind, expert, codec, hidden) reveals the structure the codex arc
  discovered. All aggregations (by layer, by codec, by hidden_kind) are
  trivial SQL GROUP BY.
- **Sherwood field+seed** (per H1800/H1801): `hidden_id` first-class
  because codec quality is hidden-conditioned. Static reconstruction
  ≠ MoE composition quality.
- **Donoho** (rigor): `n_subsample` + `seed_int` ALWAYS recorded so CI
  is computable from row data alone.
- **Carmack / Geohotz** (MVP): five tables, no joins required for primary
  queries; SQL views handle the rest. Pure functions for all measurement.
- **Thiel** (monopoly): the DB becomes canonical source-of-truth for codec
  quality on DS4. All future codec experiments query this DB; no parallel
  measurement files.
- **Krzakala / Schniter** (AMP): the codec problem IS compressed sensing
  recovery. Per-cell measurement noise level is explicit via `n_subsample`.

## Append-only invariants

Schema-level triggers enforce these:
- `journal_no_update`, `journal_no_delete`: journal is forever append-only
- `codec_no_update`, `codec_no_delete`: codec specs are immutable
- `source_tensor_no_update`: source tensors are immutable
- `measurement_no_update`, `measurement_no_delete`: measurements are immutable
- `composition_no_update`, `composition_no_delete`: composition measurements
  are immutable

The only mutable rows are `run.ended_at` (timestamp closing a run) — and
this is the documented exception, recorded in the journal as `run_end`.

## Quick-start

```bash
# Initialize DB + register default codecs
python3 -m codec_audit init
python3 -m codec_audit register-defaults

# Scan one (layer, kind), all 256 experts × 3 codecs
python3 -m codec_audit scan --layer 0 --kind gate --expert-end 256

# Query best codec per cell by mean rel_l2
python3 -m codec_audit query --limit 30

# Recent journal activity
python3 -m codec_audit journal --tail 20

# List registered codecs
python3 -m codec_audit codecs
```

## Programmatic use

```python
from codec_audit import (
    open_db, init_db, register_codec, ensure_source, ensure_hidden,
    aberration_vector, composition_metrics,
    insert_measurement, insert_composition,
    scan_layer_kind, new_run, end_run,
    load_fp4_source,
    decode_iq2_xxs, decode_polar_plr2, decode_vq_vqb1,
    DECODERS, DB_PATH,
)

with open_db() as conn:
    rid = new_run(conn, "manual_probe")
    src = load_fp4_source(0, "gate", 243)  # the VQ-worst expert
    src_id = ensure_source(conn, 0, "gate", 243, src)
    rec = decode_iq2_xxs(0, "gate", 243).flatten()
    m = aberration_vector(rec, src.flatten())
    insert_measurement(conn, src_id, codec_id=1, metrics=m,
                        n_subsample=src.size, seed_int=42, run_id=rid)
    end_run(conn, rid, {"cells": 1})
```

## What replaces what

| Old script (tmp/20260525_codec_vs_iq2_audit/) | Replaced by |
|------|-------------|
| `dequant_iq2_xxs.py` (158 LOC) | `decoders.py::_dequant_iq2_xxs` |
| `codec_quality_ab.py` (386 LOC) | `cli.py::cmd_query` + `sweep.py::scan_cross_codec` |
| `cross_layer_kind_ab.py` (187 LOC) | `sweep.py::scan_layer_kind` (with multi-layer loop) |
| `error_distribution_analyzer.py` (298 LOC) | `measure.py::aberration_vector` + `magnitude_strata` |
| `fast_aberration_scanner.py` (307 LOC) | `sweep.py::scan_layer_kind` |
| `vq_k_sweep_for_52gb.py` (230 LOC) | (will be ported to use this module) |
| `block_vq_sweep.py` (281 LOC) | (will be ported to use this module) |

Total: ~1847 LOC of scripts → ~900 LOC of module + persistent DB.

## H1800/H1801 lesson encoded in schema

Per silv's "include reading H1801" directive: H1801 REFUTED H1800's
static row-offset calibration. Static calibration didn't generalize
across held-out hidden vectors. The signal that survives:
**codec quality is hidden-conditioned at runtime**.

This is why `hidden_id` is a first-class dimension in both `measurement`
and `composition_measurement`. A cell's quality vs IQ2 isn't one number —
it's a column indexed by `hidden_kind`. Future composition runs against
multiple hidden families (gaussian, router_attractor, router_row_mix)
will populate the matrix and surface where codec choice actually matters.

## Files retired (in next session)

After this consolidation lands, the original 7 scripts can be moved to
`tmp/20260525_codec_vs_iq2_audit/.archive_consolidated/`. The data they
generated (in `*.json` files) can be one-time-imported into the SQLite
DB via a separate import script.
