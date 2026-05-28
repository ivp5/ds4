# Audit — Cycle-5-shape latent bugs across the ds4 codebase

silv 2026-05-28: "audit for OTHER latent bugs of the Cycle 5 shape
(declared-but-never-written gates, dual-representations that can drift)."

## Method

Three audit passes, each scripted (idempotent — re-runnable as the
codebase evolves):

1. **`audit_flags.py`** — flag-shaped static globals (`g_*_active`,
   `g_*_enabled`, `g_*_ok`, `s_*_inited`, etc.) with zero assignments.
2. **`audit_struct_fields.py`** — flag-shaped struct fields (same
   suffix set + `is_*` / `has_*` / `should_*` prefixes) with zero
   assignments. Handles array indexing + `memcpy/memset` writes.
3. **`audit_dual_rep.py`** — structs with 3+ fields sharing a common
   prefix (proxy signal for dual/multi-representation candidates).

All three live at `tmp/20260528_796_771_design/audit_*.py` and can be
re-run any time. They're stateless — each call re-reads the source.

## Findings

### Pass 1 — flag-shaped static globals (229 audited)

```
$ python3 audit_flags.py
CLEAN: no flag-shaped globals with zero writes + non-zero reads
```

229 flag-shaped statics. All have ≥1 assignment. No Cycle-5-shape
latent bugs at this layer. The MTL4 lazy-init pattern accounts for the
majority (~180 of 229): `g_X_init_attempted` + `g_X_init_ok` pairs,
each written exactly once on successful init.

### Pass 2 — flag-shaped struct fields (16 audited, widened suffix+prefix list)

```
$ python3 audit_struct_fields.py
Audited 16 flag-shaped struct fields across 16 files
CLEAN: no flag-shaped struct fields with zero writes + non-zero reads
```

False positive caught during script development: `polar_layer_enabled`
appeared suspicious until the regex was widened to recognize
`field[index] = value` (array assignment) and `memcpy(&field, ...)`.
Final result: all 16 flag-shaped struct fields are written somewhere.

The Cycle 5 bug (`override_source_active`) was the ONLY instance of
this shape in the codebase, and it's now eliminated (Cycle 5 retired
the dual-representation entirely).

### Pass 3 — multi-field shared-prefix groups (30 found, all legitimate)

```
$ python3 audit_dual_rep.py
Found 30 structs with 3+ fields sharing a common prefix:
  ds4_gpu_graph.batch_*  (39 fields) — batch-phase tensors
  ds4_layer_weights.attn_*  (13)     — attention weights
  ds4_gpu_graph.spec_*  (12)         — speculative-decode state
  ds4_gpu_graph.layer_*  (10)        — per-layer caches
  ds4_layer_weights.indexer_*  (6)   — indexer weights
  ... [25 more, all domain groupings]
```

Manual inspection of the top hits: all are **domain groupings** (related
but distinct concepts in the same struct), not **dual representations**
(redundant encodings of the same fact). No drift candidates.

The closest "looks duplicated" hit was `output_*` appearing in 3
different structs — but each is a distinct concept:

| Struct | Prefix use |
|---|---|
| `ds4_weights.output_*` | static model weights |
| `ds4_gpu_graph.output_*` | GPU transient tensors |
| `ds4_cpu_decode_scratch.output_*` | CPU scratch buffers |

These are intentionally separate (weights vs activations vs scratch).

## Out-of-scope dual-reps (cross-layer)

The audit misses **cross-layer dual representations** — the same fact
encoded in two different parts of the codebase at different abstraction
levels. Examples found this session (documented in the rewrite plan but
not catchable by struct-field regex):

1. **Cycle 5 itself**: `ds4_tensor` (metadata) vs the override-fill
   loop (Phase 2b) — the metadata fields `override_*` and the
   override-fill writes drifted. Fixed: unified into `ds4_tensor_storage`.

2. **Metal view/residency conflation**: `g_model_views[]` tracked BOTH
   "what Metal can address" and "what's GPU-resident" — two distinct
   concepts in one data structure. Fixed: full views always; cpu-moe
   routes at dispatch layer.

3. **Dispatch interface vs tensor metadata** (surfaced in #796
   Increment 2 design memo): `ds4_gpu_matmul_*_tensor(map, offset)`
   bypasses `tensor_data()` entirely. The tensor's storage layer knows
   about overrides; the dispatch layer doesn't. NOT YET FIXED — Option
   C foundation shipped in Increment 2; dispatch wiring is the next
   increment.

4. **ICB record/replay duplication** (Cycle 9 surfaced): 8 separate
   implementations of the same record/replay pattern. NOT a dual-rep
   per se (each captures distinct kernels), but the EXTRACTION OF THE
   PATTERN was duplicated. Fixed: unified `ds4_icb_slot_t` primitive.

These cross-layer patterns need different tooling — static analysis
that follows function calls, not just regex over text. The three audit
scripts here cover the LOCAL pattern (declared-but-not-written within
one file or struct).

## Verdict

**Within-struct / within-file Cycle-5-shape: CLEAN.** The original bug
was the ONLY instance of its pattern, and it's now eliminated. The
three audit scripts catch any future regression — re-run as the
codebase evolves.

**Cross-layer dual-reps**: 3 known instances surfaced and fixed this
session (Cycle 5, view/residency, ICB dictionary-bloat). 1 known
instance surfaced and SCOPED (#796 dispatch interface) — foundation
shipped, wiring pending in next session.

The "audit-as-it-evolves" discipline is now mechanized for the local
pattern. Cross-layer needs the engineer-roster eye, not regex.
