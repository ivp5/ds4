# Cache audit — omitted-field bug class & cascade accuracy

silv 2026-05-28. Triggered by the Conjecture #12 inferguard finding (prose-summary
omitting BLOCK evidence via `lines.filter(ALLOW_RX).slice(-8)`). Same bug-class
in the engine: caches that project source state through an incomplete key.

## A1 — `ds4_hot_pin_expert_from_vqb2`: row_block omitted (CRITICAL, latent)

`ds4_expert_table.c:910-971`. The function takes (layer, kind, expert) but
ignores the VQB2 packet's row block. The comment says "Current VQB2 format
carries no row_start, and the encoder slices full[:rows_per_expert], so
this packet is row block 0 only" — TRUE for the pre-H2116 format (one .vqb2
file per expert with `n_rows = 2048`/`4096`), FALSE for the H2116 artifact
where packets are keyed by `(layer, kind, row_start)` with `n_rows = 128`.

Cascade:
  1. unconditionally writes `arr[idx] |= 1ULL` (bit 0) regardless of which
     row block is being pinned
  2. multiple calls overwrite `offsets[idx]` — only the LAST pin survives
  3. earlier tiles become orphan heap allocations
  4. `ds4_hot_layer_fully_pinned` checks full mask (`0xFFFF`/`0xFFFFFFFF`) —
     bitmask = `0x1` after pinning all blocks → check fails
  5. dispatch silently falls back to IQ2_XXS path

Bug is **LATENT**: no production caller routes H2116 packets through this
path yet. Fixing pre-wire avoids debugging a silent regression later.

Fix: add `row_block` parameter; place tile at
`base_offset + row_block × per_block_bytes`; mark
`arr[idx] |= (1ULL << row_block)`. Backwards-compat wrapper calls with
row_block=0.

## A2 — `ds4_vqb2_pack_view_from_entry`: CSV vs header drift (MEDIUM)

`ds4_vqb2_pack.c` (new this session). Materializes views from the index CSV
entry without re-verifying the packet's 40-byte header at `pack_offset`.
If CSV gets out of sync with the pack file (e.g., pack rebuilt with
different layout, CSV stale), views silently decode wrong data.

Fix: re-parse the header from `base` bytes and assert the per-entry fields
(n_experts, n_rows, n_pairs, layer, kind_id, k, bit_width, n_codes) match
the CSV. Cheap — one memcmp + 8 comparisons per view materialization,
once per (layer, kind, row_start) lookup.

## A3 — `ds4_prefix_cache`: Phase 1 documented-incomplete (LOW)

`ds4_prefix_cache.h`. Phase 1 stores hash + n_tokens; defers per-layer GPU
state to Phase 2. The `valid=1` flag is set on Phase 1 store but the entry
has no GPU state. A caller treating `valid=1` as "full state available"
would be wrong.

Mitigation: existing comment is explicit about Phase 1 limits. No fix
needed; Phase 2 will add the GPU state fields.

## A4 — Other caches: clean

- `ds4_expert_table` tier loading: byte-keyed per (layer, expert)
- `ds4_polar_pool`: validated at open; per-(layer, kind) only
- `ds4_moe_route_log`: append-only logger, no projection
- `ds4_inflight`: rolling window, no cache shape
- `dsv4_compressor_store_one` kernel: Liger convention, intentional ratio behavior
- Hot basis-Hadamard accessors: gated by `calibration_domain_id` (defense in depth)

## Pattern summary

The omitted-field bug class generalizes: **any cache key that's too coarse
relative to the new source-format's variation will silently overwrite
earlier-pinned cells while passing fullness checks based on the stale
assumption.** The Conjecture #12 prose-summary catch was the same shape
(filter+truncate where truncation drops critical evidence). The DS4 instance
is in the engine, not in the prose layer — but the bug-detection method
that found #12 (predict the omission point, build the test that exercises
it, watch for silent zero-mismatch where there should be a failure) applies
equally.
