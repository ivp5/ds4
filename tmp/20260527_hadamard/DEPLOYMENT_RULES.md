# Hadamard codec — deployment rules (post-codex-H1967..H2068 read)

silv 2026-05-27. Sequential read of codex CODEX_SHIFTS.md H1967-H2068.

## What the reading changes about #643/#645

The Hadamard-16 kernel + encode-side tool I just shipped are a **codec
primitive**, not a deployment policy. The kernel does one thing well:
apply an orthogonal basis transform to FP16 tensors. The encode tool's
Python implementation matches the GPU kernel to `rel_L2=4.87e-4`.

The mistake to avoid is treating "rotation is good, ship everywhere."
Codex H1967-H2068 shows that across ~60 hypotheses, **codec primitive
choice is downstream of selector quality**. The same primitive can win
or lose depending on what slice it's applied to and what objective scores
the candidate. The actionable rule is structural, not formulaic.

## The six rules that gate Hadamard application

These are extracted from the codex shifts in their numbering. Each is a
constraint that must hold before any Hadamard sidecar is applied at
dispatch time. Violating any of them means the basis transform is dead
weight or actively harmful.

### Rule 1 — Selector before primitive (H2007, H2011)

Reconstruction loss (rel-L2) is a diagnostic only. The selector must run
**replay strata**: ordinary rows, failure rows (wrong-source), and
route-margin tails. A candidate that wins on rel-L2 but loses on the
signed source-top-k harmful loss is dropped.

Implication for #643/#645: the encode tool produces FP16 sidecars that
score equally on rel-L2 (orthogonal H ⇒ exact reconstruction in float
arithmetic). That makes rel-L2 useless for choosing WHICH layer/expert
gets Hadamard. The encode tool must be paired with a candidate-replay
selector before any layer/expert gets rotated.

### Rule 2 — Per-organ, not uniform (H2017, H2020, H2024, H2065)

Codex tested three policies on DS4 L09 high-pressure cases:

| policy            | output rel-L2 | top2 p95   | topk p95   |
|-------------------|---------------|------------|------------|
| gate/up K64, down K16 (`g64_u64_d16`) | 0.385       | 0.087      | 0.087      |
| balanced K32                          | 0.383       | 0.051      | 0.054      |
| gate/up K32, down K64 (`g32_u32_d64`) | 0.343       | 0.050      | 0.054      |
| all-kind K64                          | best         | 0.020      | 0.038      |

**Promoting gate/up while leaving down at K16 is WORSE than balanced
K32 on readout margin.** Down precision is a readout organ, not a
sink. The "rotate gate/up only" intuition that's tempting because the
gate/up matmul is fused is exactly wrong.

Implication for #643/#645: do NOT default to Hadamard-rotating only the
fused gate/up tile. Either rotate all three organs (gate, up, down) or
none. If a sparse promotion plan is wanted, the selector must score
per-organ on signed source-top-k harm — not assign by organ role.

### Rule 3 — Route-resident, not full-manifest (H2066)

The 52GiB DS4 fit constraint blocks any full-manifest sidecar that
promotes every selected expert across every routed layer. Codex H2066:

- `K4_shared` at 51.916 GiB → 84 MiB headroom under 52 GiB
- One 128-row gate/up promotion block per expert = 65,536 bytes
- Full-manifest layer-09 (164 experts × 1 block) = 10.5 MiB ✓
- Full-manifest **across 61 routed layers × 1 block** = 53 GiB ✗

Implication for #643/#645: when the GGUF-rewrite extension lands
(deferred from ENCODE_FINAL.md), it must rewrite **only the
route-resident slice** for a given calibration prompt domain. A
full-tensor-rewrite tool that processes every routed expert is
mathematically unable to fit under 52 GiB.

The legal shape is `(route-domain, expert, row-block, organ)`. The
encode tool's `--gguf --tensor` single-tensor mode is already aligned
with this — it doesn't try to rewrite the whole file.

### Rule 4 — Per-expert pruning is the 52GiB lever (H2067, H2068)

When a sidecar overshoots the budget by ~3 MiB, the right move is
NOT to drop row-blocks. Codex H2068 showed that pruning 5 of 130
selected route experts saves 3.4 MiB — 2.36× more than dropping one
row of a 32-case d11 sidecar would save (8.1 MiB) but the row-drop
also destroys quality. Pruning the lowest-rank/lowest-weight experts
preserves quality of the remaining sidecar.

Codex's prune candidates for layer-09 were `[52, 231, 87, 55, 71]`
selected by `greedy_minimax_case_weight`. This is policy-specific
to codex's calibration corpus and routed-expert space; it does not
port to DS4-V4-Flash IQ2_XXS directly (DS4 has 164 routed experts,
not 240+).

Implication for #643/#645: when a Hadamard sidecar exceeds budget,
ship a per-expert pruner that uses the same route-weight signal,
not a row-block dropper. The pruner runs on the OUTPUT of the
encode tool, not the input.

### Rule 5 — Signed harm, not absolute drift (H2022)

Codex H2022 found that ~3 of the "largest-looking K32 tails" were
not actually harmful — they INCREASED the source-top1 margin over
the wrong competitor. Absolute watchlist drift treated them as
"consumption" but signed analysis showed `-0.044` (helpful).

Implication for #643/#645: any validator script that ranks Hadamard
sidecars by `|max_err|` or `|rel_L2|` is using absolute drift. The
correct ranker is `signed source-top-k harm`, computed by running
the lens at the readout position and comparing token margins
before and after. The encode tool's current `verify_matmul` is fine
for self-consistency but inappropriate for selection.

### Rule 6 — Domain certificate, not packet identity (H2042, H2043, H2044)

Codex H2042-H2044: a "packet that wins on slice X" is not a packet
that wins globally. The artifact boundary must carry `(generator,
domain_cases, out_of_domain_cases, own_slice_score, global_score,
out_domain_score)` — a packet without its scoring domain is
"misinformation."

Implication for #643/#645: the GGUF-rewrite extension must emit
sidecar metadata that includes the calibration domain (prompt
identifiers, route-domain ID). At load time, the runtime should
refuse to apply a sidecar whose calibration domain doesn't match
the current route-domain hint, OR should treat it as a fallback
rather than a primary path.

## What this means concretely for the next code move

Three deferred items in `ENCODE_FINAL.md`:

1. **GGUF rewrite tool**: ✗ DO NOT BUILD as a full-tensor rewrite.
   Must be route-resident-sidecar-shaped per Rule 3. Build the
   single-expert single-row-block emit path first, then the
   per-(route-domain, expert, row-block, organ) walker.

2. **Hot-store loader basis-aware marker**: build with Rule 2 in
   mind — the `is_basis_transformed` flag must be per-organ
   (`gate_h`, `up_h`, `down_h`), not per-expert.

3. **Dispatch-site pre-pass integration**: build with Rule 1 +
   Rule 6 in mind — the runtime must check that the current
   route-domain matches the sidecar's calibration domain before
   applying the activation-side Hadamard transform. Mismatch =
   skip the transform, use the original-basis path.

## What the codex reading does NOT change

- The Hadamard primitive itself (kernel + encode tool) is sound.
  Cross-validation confirmed Python↔GPU agreement at 4.87e-4 rel_L2.
- The #631 simdgroup mat-mat dispatch wire is selector-agnostic;
  the codec axis sits ABOVE it and doesn't break the dispatch.
- The codec primitive can still be deployed selectively — the rules
  just constrain WHERE and HOW, not WHETHER.

## What the codex reading reveals about my prior intuition

Three corrections to my mental model going in:

1. I had been thinking "rotate gate/up because it's the fused tile."
   Rule 2 refutes this. Down precision matters more for readout
   margin than gate/up does.

2. I had been thinking "Hadamard is a general activation-side
   transform; apply it broadly." Rule 3 refutes this. The 52GiB
   constraint forces route-residency, not breadth.

3. I had been thinking "any orthogonal basis that preserves matmul
   is interchangeable." Rule 5 refutes this — the matmul is the
   downstream observable; what gets selected depends on signed
   readout harm at the lens position, not on basis preservation.

## Status

Memo only. No code shipped this turn. The deferred items in
`ENCODE_FINAL.md` now have explicit selector-design constraints
that any future ship must satisfy.

The deployment_rules.md should be re-read at the start of any
session that touches the Hadamard codec, the GGUF rewrite tool,
or the route-resident sidecar plumbing.
