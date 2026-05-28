# Metal view/residency conflation — engineer-roster diagnosis + fix

silv 2026-05-28 evening: "for metal residency bug at 8.37gb — one of the ways
to fix bugs is to simplify and do heavy rewrites of codebase, until either the
bug is found, or enough of the codebase is rewritten that there is high chance
that that includes the section where the bug was hiding. ... advised by the
following engineer roster ... allocate ample time and budget."

## The bug as observed

Live-fire smoke output:
```
ds4: Metal mapped mmaped model as 2 overlapping shared buffers   ← whole-file views
ds4: --cpu-moe: registered 2 non-routed Metal segments,
     excluded 1 routed-expert ranges (0.66 GiB) from the Metal residency set
ds4: --prefill-metal-phases: activated phase 0/3 (layers 0..13)
ds4: Metal model range 8.37..8.89 GiB is not covered by mapped model views
ds4: prompt processing failed: metal prefill failed
```

Three sub-systems each modifying the same data structure (`g_model_views[]`):

1. **`ds4_gpu_set_model_map_range`** — initial whole-file view (2 buffers, ~8 GiB each, overlapping).
2. **`ds4_gpu_set_model_map_ranges`** (cpu-moe variant) — REPLACES the view set with a NON-CONTIGUOUS subset (excludes routed-expert ranges).
3. **prefill-metal-phases** — activates a SUBSET OF LAYERS that should be on Metal.

The bug: subsystems (2) and (3) disagree. cpu-moe assumes "ALL routed → CPU"; prefill-metal-phases assumes "some routed (phase-activated layers) stays on Metal". When a phase-0 Metal kernel needs a routed-expert tensor at 8.37 GiB, cpu-moe already removed that view.

## Engineer-roster diagnosis

The data-structure debt is the same shape as the Cycle 5 dual-representation:

- **Knuth / DJB**: views and residency are two truths about the same model. They CAN drift. They DO drift (this bug).
- **Carmack / Hotz**: one always-correct invariant — views cover ALL of `[0, model_size)`. Residency is a separate optimization that doesn't change coverage.
- **Linus**: the bug is the conflation. Don't fix the symptom (re-add the excluded range); fix the cause (separate the concerns).
- **Pearl**: the dual-representation creates causal confusion. The CPU-moe exclusion is logically about RESIDENCY (don't evict from disk if we won't use it), not about COVERAGE (can Metal address these bytes if asked). Two distinct questions answered by one data structure.
- **aphyr**: under load (prefill phase activation crosses the boundary between cpu-moe-assumed and phase-activated layers), the conflation invariant breaks. Fail-stop on miss is correct; the design is wrong.
- **Theo Tso**: this is the equivalent of mixing up "block in filesystem extent" and "block in page cache". They look similar; they're orthogonal.

## Existing infrastructure already supports the separation

`ds4_gpu_metal.m:889-918` already maintains `g_model_residency_set` (MTLResidencySet) as a separate concern from views:

```objc
MTLResidencySetDescriptor *rsDesc = [MTLResidencySetDescriptor new];
rsDesc.initialCapacity = g_model_view_count;
...
for (uint32_t i = 0; i < g_model_view_count; i++) {
    [g_model_residency_set addAllocation:g_model_views[i].buffer];
}
```

But the residency set currently adds ALL views. The cpu-moe exclusion happens at the wrong layer (view creation) instead of the right layer (residency set membership).

## The rewrite (engineer-roster simplified)

**Single invariant**: `g_model_views[]` covers `[0, model_size)` fully. ALWAYS.

**Separate concern**: `g_model_residency_set` includes only the subset of views that should be GPU-resident at this moment. cpu-moe trims the residency set, not the views. Prefill-phase activation grows/shrinks the residency set per-phase.

### Code changes

1. **`ds4_gpu_set_model_map_ranges`** → DELETE (or rename to `ds4_gpu_set_model_residency_ranges` if its callers still want range-based residency management). The function currently REPLACES the view set; that's the bug.

2. **cpu-moe segment-emission logic in `ds4.c:20917-21025`** → simplify to: don't call `set_model_map_ranges` at all. Views stay whole-file. cpu-moe's role becomes: when running a routed-MoE layer, route to CPU (the apply_full logic already does this). The Metal residency for routed-expert bytes is a separate hint that can be attached to the residency set lifecycle.

3. **`ds4_gpu_wrap_model_range`** → unchanged. Now its linear search will ALWAYS find a view (because views are total). The error path becomes dead code (assertion territory).

### Engineer-roster perspectives on the rewrite

- **Carmack**: one path. No special case for cpu-moe at the view layer.
- **Linus**: the bug surface (range-not-covered) disappears because the case can't arise.
- **Knuth**: O(1) wrap lookup possible if views are contiguous-then-overlap (already true). Or keep the linear scan — only 2-3 views, doesn't matter.
- **DJB**: configuration via const tables (residency policy), not via mutation of the coverage layer.
- **aphyr**: the "exception case" the original code was meant to handle (massive models > maxBufferLength) is already handled by overlapping views. cpu-moe doesn't help with that.

### Scope estimate

- Delete `ds4_gpu_set_model_map_ranges` body (~80 lines)
- Inline its 2 callers in `ds4.c` to use `set_model_map_range` with full range (~40 lines deleted)
- Move the cpu-moe diagnostic (`registered N non-routed Metal segments`) to a residency-side hint
- Net: -100 lines, -1 entire data-path

## Increment 1 (this turn): apply the simplification

Delete the cpu-moe view-restriction path. Replace with whole-file view + cpu-moe-only-affects-dispatch.

Risk: cpu-moe's residency exclusion claim (0.66 GiB saved) is forfeited. This was likely never load-bearing on a 9 GB model (the OS page cache handles eviction). On the IQ2_XXS path (86.7 GB, OFF-LIMITS), this might have mattered more — but that's silv-permitted territory only.

**Test**: same smoke as before. Expected: "Metal model range ... not covered" disappears; prefill proceeds.
