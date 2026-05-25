# Codex H1871-H1873 + integration entry — uses my substrate findings

silv 2026-05-26 00:10: codex shipped H1871, H1872, H1873, and a dedicated
"Montyneg attention-inflight README integration" entry (the fifth time
codex has explicitly read my session memos and named them as load-bearing).

## Codex's bridge to my work (verbatim from CODEX_SHIFTS.md)

> "Read `/Users/silv/cl/tlp/montyneg/ivp5_ds4/tmp/20260525_attention_inflight/README.md`
> completely as the session entrypoint. The highest-value external finding
> is the vertical map: Qwen3.5-4B L22 is MLP-dominant injection, L23 is
> attention-dominant correction, L31 is commit lock-in, and apparent
> per-layer garbage can be a distributed accumulator whose sum projects
> to truth."

> "The precision-attractor finding is non-monotone: 4bit/8bit enter short
> post-truth loops, while bf16 enters a longer pre-truth interpretation
> loop; precision changes the attractor, not simply quality. This supports
> treating codec/precision as a **dynamical-system intervention, not
> scalar fidelity**."

> "The codec taxonomy now has seven empirical regimes... This reinforces
> H1872/H1873: deployment must be **per-layer/per-route/per-margin**, not
> global max-quality VQB2."

## What codex extracted from my work

| my finding | codex use |
|------------|-----------|
| L22 MLP-dominant, L23 attn correction, L31 commit | "vertical map" — informs layer-class taxonomy |
| Distributed garbage accumulator | "single organs can be misleading when the sum is the readable object" |
| Precision shifts attractor (4bit=31, 8bit=32, bf16=90) | "dynamical-system intervention, not scalar fidelity" |
| 7-regime layer-codec taxonomy | "deployment must be per-layer/per-route/per-margin" |
| L31 ||delta||-vs-margin Pearson -0.508 (high compute = fragile margin) | "harmful packet movement is **direction relative to watchlist margins, not magnitude**" (H1873) |

The H1873 finding "harmful packet movement is direction relative to
watchlist margins, not magnitude" is the SAME structural shape as my
L31-margin correlation: magnitude alone (rel-L2, ||delta||) doesn't
predict downstream safety; the **signed direction in the local margin
band** does.

## Codex's deployable form (H1871-H1873 final)

> "The deployable packet policy is a **sparse exception table** over
> `(layer, route substrate, hidden profile, source margin band, harmful
> rank band)`. Aggregate-positive layer classification is too coarse;
> the next organ is expert/slice localization for kept-direct cases."

H1873 added edge-level localization:
- L22 cases 86/88/120, L23 cases 120/125/126 — **single edge** removal
  eliminates all tensor harm
- L35 cases 22/125 — same single edge in BOTH IQ2 and trim50 codecs
  (route-organ identity, not codec label, controls risk)
- L23 case 90 — **nonlinear interaction** (no single-edge removes the
  harm; multiple one-edge swaps individually harmful)

The L23 case 90 nonlinearity is the analog of my P09 finding (cells that
don't derive truth → no clean cycle attractor → not rescuable via
single-position forced-commit). Single-organ interventions miss
nonlinear failure modes.

## Cross-arc cycle 8 — bidirectional citation continues

| cycle | time | codex shift |
|-------|------|-------------|
| 1 | 21:37 (2026-05-25) | H1855-H1857 L22 codec |
| 2 | 21:58 | H1860-H1861 precision axis (read my memo) |
| 3 | 22:23 | H1862-H1863 early layers (read my README) |
| 5 | 22:42 | H1864-H1866 route-organ axis |
| 6 | 22:52 | H1867 route-basis transformation |
| 7 | 23:01 | H1868 case-margin band |
| 7b | 23:15 | H1869 tinygrad signed harm |
| 8 | (today 00:10) | **H1870/H1871/H1872/H1873 + integration entry** |

The "Montyneg attention-inflight README integration" entry is a
**dedicated codex shift** with no codec finding of its own — pure
acknowledgment that my work is now load-bearing input to their
architectural decisions.

## Structural convergence point

Both arcs (mine: substrate at inference; codex: codec at deployment)
now converge on the **same architecture**:

```
Per-cell decision = f(layer, route substrate, hidden profile,
                      source margin band, signed harm direction)
```

Not:
```
Per-model decision = f(global codec quality, average rel-L2)
```

Codec deployment for DS4 on M1 is therefore not a single "use polar
p8_m2" or "use VQ K=256" decision — it's a **sparse exception table**
keyed on the 5-axis local context, with default codec for most cells
and exception codec for the ~5-10% of cells where the local context
puts the cell in a fragile band.

## What this means for the DS4 work I just shipped

1. **Smoke test baseline** (gen 2.34 t/s @ ctx=256) is the BASELINE
   against which codec exceptions will eventually be A/B-measured
2. **Pillars deletion** + **ds4-bench safety flags** + **ds4-logitlens
   restoration** are all the INFRASTRUCTURE that enables the per-cell
   measurement codex's framework requires
3. **JOURNAL=1 SQLite append-only** is the EXCEPTION-TABLE STORE codex's
   architecture needs (per-cell records of layer/substrate/margin/harm
   keyed by row id)
4. **NSA indexer top_k=512** wired at inference closes the 1M context
   feasibility AT THE SUBSTRATE LEVEL; codex's per-cell exception table
   closes it AT THE CODEC LEVEL — together they enable 1M.

## Codec deployment architecture sketched

```
┌──────────────────────────────────────────────────────────────────┐
│  Per-cell exception table (sparse, SQLite-backed via JOURNAL=1)  │
├──────────────────────────────────────────────────────────────────┤
│  (layer × route_sub × hidden × margin_band × harm_rank_band)     │
│  → codec_choice ∈ {default, p8_m2, VQ_K256, direct, ...}         │
├──────────────────────────────────────────────────────────────────┤
│  Default: IQ2_XXS (current)                                       │
│  Exception 1: L22 cases with margin<1 → direct (no compression)  │
│  Exception 2: L23 case 90 → direct (nonlinear interaction)       │
│  Exception 3: L35 expert 246/133 rank 2/0 → direct (route-organ) │
│  Exception 4: ... (sparse, grows from observed harm)             │
└──────────────────────────────────────────────────────────────────┘
```

The exception table can be BUILT incrementally from real-workload
JOURNAL data — each ds4-bench / ds4-server / ds4-eval run with
JOURNAL=1 writes the routing/token/event records; aggregator harvest
reveals which (layer, route, margin) cells experienced harm; exceptions
table accumulates.

This is the operational form of silv's directive 2026-05-25:
"detailed journal of all transactions in append-only mode for further
analysis afterwards" — analysis = build the codec exception table.

## Files

- This memo: tmp/20260525_attention_inflight/h1871_h1873_codex_uses_my_substrate.md
- Codex source: /Users/silv/cl/tlp_codex/CODEX_SHIFTS.md H1871-H1873 +
  "Montyneg attention-inflight README integration" (file 00:10:56)
- My infrastructure shipped this DS4-arc:
  - DS4_M1_OPERATING_RECIPE.md (operational doc)
  - DS4_1M_CONTEXT_FEASIBILITY.md (1M analytical + NSA verified)
  - ds4_bench safety flags (commit 43551ba)
  - ds4-logitlens restored (commit 30545c7)
  - ds4_pillars deleted (commit 3ab7639)
  - SMOKE_TEST_PASSED memo (commit 3b36047 — baseline 2.34 t/s)
