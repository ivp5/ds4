"""codec_audit — DS4 codec quality audit framework.

Append-only SQLite-backed measurement database with consolidated decoders.
Replaces 7 ad-hoc scripts from tmp/20260525_codec_vs_iq2_audit/.

Design lenses (per silv 2026-05-25):
  Larry Page / Sergey Brin: structured key-value, append-only commit log.
  Vitalik Buterin: every measurement carries content-hash provenance.
  Knuth: one mature schema, indexed for every common query.
  Tao: 5-tuple primary key (layer, kind, expert, codec, hidden) reveals structure.
  Sherwood (field+seed): hidden_id is first-class because codec quality is
                          hidden-conditioned (H1800/H1801).
  Donoho: n_subsample + seed_int always recorded so CI is computable.
  Carmack/Geohotz: minimum-viable-code. Five tables, no joins required for primary
                   queries. SQL views handle the rest.
  Thiel: the DB becomes monopoly source-of-truth for codec quality on DS4.

Quick-start:
    from codec_audit import open_db, scan_layer_kind, journal, register_codec

    with open_db() as conn:
        register_codec(conn, "vq_k256", "vq", bits_per_pair=8.0)
        scan_layer_kind(conn, layer=0, kind="gate", codecs=["iq2_xxs", "vq_k256"])

Append-only invariants: ALL measurement/codec/source_tensor rows are immutable
after insert. Triggers in schema.sql enforce this at the database level.
"""

from .db import (  # noqa: F401
    open_db,
    init_db,
    journal,
    DB_PATH,
)
from .registry import (  # noqa: F401
    register_codec,
    ensure_source,
    ensure_hidden,
    list_codecs,
)
from .measure import (  # noqa: F401
    aberration_vector,
    composition_metrics,
    insert_measurement,
    insert_composition,
)
from .decoders import (  # noqa: F401
    decode_iq2_xxs,
    decode_polar_plr2,
    decode_vq_vqb1,
    load_fp4_source,
    DECODERS,
)
from .sweep import (  # noqa: F401
    scan_layer_kind,
    scan_cross_codec,
    new_run,
    end_run,
)

__version__ = "0.1.0"
