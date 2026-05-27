# Per-organ signed source-top-K harm scorer for DS4 — design memo

silv directive 2026-05-27 (post-codex-H1967..H2068 + Hadamard ship):

> review your thinking based on the most recent understanding, refine
> for an order of magnitude higher accuracy which should enable to
> sense small aberrations unnoticed before, which can be amplified to
> yield valuable signal. look for 2-3 order of magnitude speedups
> along the way which would enable additional applications at the
> expanded level of accuracy and detail

## The accuracy gap (where current methodology lies to us)

DS4's substrate work has been measured at two coarse resolutions:

1. **Top-1 rank at readout**: integer-valued. Tells us "did the model
   pick token X at position P". Cannot distinguish "barely picked X
   over Y" from "picked X with 99% probability". Cannot see helpful-but-
   small reorderings outside top-1.

2. **Reconstruction rel-L2 of tensor outputs**: absolute, unsigned.
   Codex H2022 nailed the failure mode: "K32 P=0 baseline had
   signed damage `-0.044`, meaning K32 INCREASED source-top1 margin
   over the wrong competitor; absolute drift counted that as
   `consumption`". Helpful and harmful moves both look big.

Both metrics are blind to the small aberrations silv is asking about.
A codec perturbation that helps the top-3 distribution by tightening
truth's margin while slightly degrading rank-1 of an off-distribution
junk token reads as "noise" to rel-L2 and as "neutral" to top-1 rank —
when in fact it's a +signal we want to amplify.

## The accuracy lift: signed source-top-K harmful loss

Defined per codex H2022 + H2048:

```
For each calibration prompt p:
  baseline_run(p) →  source_topk = [t_1, t_2, ..., t_K]  (top-K token ids)
                     baseline_logits = L_baseline[V]
                     truth_token = t_1 if model is correct, else the
                                   prompt-provided ground truth
  perturbed_run(p, perturbation):
                     perturbed_logits = L_perturbed[V]

  for k in 1..K:
    margin_baseline[k]  = L_baseline[truth_token]  - L_baseline[t_k]
    margin_perturbed[k] = L_perturbed[truth_token] - L_perturbed[t_k]
    signed_harm[k]      = margin_baseline[k] - margin_perturbed[k]
                        # >0 = harmful (truth lost margin over competitor k)
                        # <0 = helpful (truth GAINED margin)
                        # =0 = inert
```

Resolution per cell: K continuous floats × N prompts = K·N continuous
floats. With K=8, N=100, per (layer, expert, organ) row: 800 floats =
exposes ranking shifts at the 4th-decimal place that rel-L2 floors out.

**Why signed.** Codex H2022 found that on the K32 high-pressure slice,
absolute-drift ranking flagged 4 cells as "worst tail consumers" but
signed-harm analysis showed 2 of them were actually HELPFUL (negative
damage). A selector built on absolute drift would have downgraded
those cells, losing free signal. The signed version flags only the
TRULY harmful cells and IDENTIFIES the helpful ones — turning what
was noise into a positive selection signal.

**Why top-K not just top-1.** A perturbation that shifts truth from
rank-2 to rank-1 is visible to top-1 metrics. A perturbation that
shifts truth from rank-2 (margin -0.1) to rank-2 (margin -0.005) —
on the threshold of becoming top-1 under any further nudge — is
invisible to top-1 but *very* visible to top-K margin. This is the
"small aberration" silv's prompt names: the model is one nudge away
from getting the answer right, and our current methodology can't see
which perturbations close that gap.

## The ablation cell space

DS4-V4-Flash IQ2_XXS at full N=43 layers:
- 61 routed layers × 256 experts × 3 organs (gate/up/down) = **~47K cells**
- With row-block precision: × 16 (gate/up) or × 32 (down) = ~1M row-block cells

The 47K-cell granularity is what the basis-aware marker (task #647)
exposes today: per-(layer, expert, organ). The row-block level is a
refinement once organ-level shows where to focus.

## The speedup ladder (target: 2-3 OOM)

A single naive cell score = one forward pass with one organ ablated.
At DS4's measured ~2 t/s prefill on M1 Max with phases=auto, a
1024-token AIME prompt's prefill is ~512 seconds. Ablating all 47K
cells × 100 prompts = 4.7M forwards = **~750 days**. Infeasible.

Three independent OOMs compose to ~600× speedup:

### Speedup 1 — KV-cache shared prefix (~10×)

For a calibration corpus of N prompts sharing a common preamble (chat
template + system message), the first ~256-2048 tokens are identical.
Re-prefilling them N times is pure waste.

The antirez merge from 2026-05-27 (312e1da) just hardened the disk-KV
cache. Setting up a shared-prefix loader for the calibration corpus
gives 7.7× to 10.3× speedup at 793-1600 token prefixes (measured in
the Hadamard probe lib).

For DS4 specifically: the `kv_cache_maybe_store_continued` path already
exists; what's needed is a flag that says "this is calibration mode,
the prefix is shared, don't re-prefill it for cells 2..N".

**Per-cell cost reduction**: 512s → ~50s.

### Speedup 2 — structural probe panel (~17×)

Codex H2040: instead of ablating ALL 47K cells, ablate a structural
panel of ~14 strata (`:best`, `route_weight2:top2`,
`expert_overlap:count15`, etc.) that preserves the max-frontier of
harm scores at `17.14×` fewer probes.

For DS4: the strata are route-pressure (high/med/low from H1739),
expert-overlap (count15 sparse tail), prefix-position (8/16/32). The
exact panel needs calibration on DS4's own route-pressure surface
(H1739 strata are from a different model), but the *17×* speedup
factor is the structural property — not model-specific.

Crucially, codex H2040 saw the speedup PRESERVED THE MAX-FRONTIER.
Mean accuracy degraded slightly (`0.0319` vs oracle `0.0233`) but
the dangerous cells were all caught. That's exactly the tradeoff a
selector needs: don't miss the bad cells; coarser scoring on the
good cells is fine.

**Cell count reduction**: 47K → ~2700.

### Speedup 3 — batched per-case scoring (~3.5×)

Codex H2058: batching all `48 cases × 16 row actions` into one tensor
matmul gives 3.46× speedup over per-case scoring while being exact
(max_abs=0 between batched and sequential outputs).

For DS4: organ ablation is implemented as a CPU-side mask on the
expert dispatch loop. Batching cases means running multiple ablations
in one prefill pass, capturing per-cell outputs at the readout
position. The Metal kernel already supports per-token-per-expert
dispatch; adding per-cell ablation as a row dimension is a matmul
batching transformation.

**Per-cell wall reduction**: another 3.46×.

### Combined: ~600× speedup

100 prompts × 2700 cells × 50s ÷ 3.46 ≈ **~108 hours** vs the 750
days naive. Still long but tractable. With careful prefix-cache
discipline + a partial probe panel of just the top 100 high-pressure
cells (codex H1739 says high-pressure is a tiny fraction of the
route surface), this drops to **~10 hours** for first-cut signal.

A first-cut signal in 10 hours is the threshold where this becomes
weekly-iterable research. That's the "additional applications
enabled at expanded accuracy" silv named.

## The applications unlocked

Given 600× speedup at K continuous-margin precision per (layer,
expert, organ) cell:

1. **Route-resident sidecar selector** (DEPLOYMENT_RULES.md Rules
   1, 2, 5): pick which (layer, expert, organ) cells get Hadamard /
   higher-K codec promotion based on signed harm on the calibration
   corpus. Not naively "all routed FFNs get rotated" — only the cells
   where ablation makes truth-margin worse get protected.

2. **Conjecture #23 per-cell rescue boundary**: the AIME rescue
   protocol currently fires at corpus level (whole-CoT detection +
   forced commit). With per-cell signed harm we can ask: which
   (layer, expert) cells DRIVE non-convergence vs which are passive?
   The 8/10 cell ceiling becomes a per-cell mechanism map.

3. **Cross-model attractor map at per-token resolution** (refinement
   of Conjecture #23 forced-commit BF16→11 vs 4bit→229 attractor
   divergence): why do precision tiers pick different wrong attractors?
   With per-organ signed harm scoring, "which experts at which
   precision tier shift the readout toward which attractor" becomes
   a measurable per-cell question.

4. **Layer-skip composition mapping at per-prompt precision**: the
   safe-10 trim arc (Conjecture #15 + 22+ probes) measured
   composition cliffs by reading the final boxed answer of one
   prompt. With signed top-K harm per (layer, organ), the
   composition cliff dissolves into "which organs in which layers
   contribute to which top-K margins on which prompts" — a fine-
   grained map where the currently-binary "breaks" / "preserves"
   becomes a continuous landscape with directional structure.

5. **The codec selector that DEPLOYMENT_RULES.md is missing**.
   Rules 1+2+5 specify "selector before primitive, per-organ, signed
   harm" but no implementation of the selector exists. This scorer
   IS the selector input.

## Implementation phasing

### Phase A — organ-ablation hook (minimum new code)

Add to `ds4_expert_table.h/c`:

```c
/* Per-(layer, expert) per-organ skip flag. kind = 0/1/2 (gate/up/down).
 * Set via env var DS4_ORGAN_SKIP="L,E,K;L,E,K;..." or CSV file. */
extern uint8_t g_organ_skip[DS4_N_LAYER * DS4_N_EXPERT * 3];
int ds4_load_organ_skip(const char *manifest_or_env);
static inline int ds4_organ_should_skip(uint32_t L, uint32_t E, uint32_t K) {
    if (L >= DS4_N_LAYER || E >= DS4_N_EXPERT || K >= 3) return 0;
    return g_organ_skip[(L * DS4_N_EXPERT + E) * 3 + K];
}
```

At the dispatch site (CPU path in ds4.c + Metal path in ds4_metal.m):

```c
if (ds4_organ_should_skip(layer, expert, GATE)) /* zero gate output */;
if (ds4_organ_should_skip(layer, expert, UP))   /* zero up output */;
if (ds4_organ_should_skip(layer, expert, DOWN)) /* zero down output */;
```

The "zero output" semantics is the cleanest ablation. Alternatives
(random output, mean output) are second-order; start with zero.

### Phase B — harm-scoring driver

A new binary `ds4-harm-score`:
- Input: model + calibration prompt CSV + ablation cells CSV
- Output: per-cell signed top-K margin delta CSV

Loop:
1. For each prompt: prefill (cached if shared prefix), capture
   baseline top-K logprobs + readout position id
2. For each ablation cell C in cells.csv:
    - Set g_organ_skip[C]
    - Re-prefill from the shared-prefix checkpoint (10× speedup
      lever 1)
    - Capture perturbed top-K logprobs
    - Compute signed_harm[k] per top-K
    - Clear g_organ_skip[C]
3. Emit CSV: prompt_id, layer, expert, organ, k, margin_baseline,
   margin_perturbed, signed_harm

### Phase C — batched scoring optimization

Once Phase B works on a small (10 prompt, 100 cell) probe and
validates the resolution claim, add batching:
- Pack 16-32 ablation cells per Metal command buffer
- Single prefill produces 16-32 logit vectors in parallel
- 3.5× speedup lever 3 lands

### Phase D — structural probe panel

Replace the all-cells loop with the H2040-style strata panel.
Requires calibration of DS4's own route-pressure distribution
(currently we only have codex's strata from a different model).
17× speedup lever 2 lands.

## Risk register

- **Ablation semantics**: zero output is one ablation choice. Codex
  found random/mean ablations differ on signed harm. Start with zero,
  measure spread vs random as a sanity check.

- **Calibration corpus selection bias**: the choice of N=100 prompts
  determines what the selector "sees" as the operating domain
  (DEPLOYMENT_RULES Rule 6). AIME-shape vs general-text vs code
  produces different signed-harm distributions. The selector must
  carry its calibration-domain id (already implemented in the basis-
  aware marker via `calibration_domain_id`).

- **MoE routing variance**: ablating one expert at one layer may
  change WHICH experts are routed at downstream layers (the
  router-finalize step uses the current layer's output). The signed
  harm captured is therefore "effect on top-K margins given the
  routed expert set actually used", not "effect on a fixed routed
  set". This is the more realistic measurement but it confounds
  layer-vs-layer comparisons. Note in the output: include the
  routed expert set per prompt as a column.

- **Speedup lever 1 (KV-shared prefix) cache invariant**: the disk-
  KV cache assumes the model state is identical when reading back.
  With g_organ_skip mutated between prefill chunks, the cache write
  paths must be DISABLED during scoring. Each ablation cell scores
  in isolation; we don't want one ablated session's KV to be
  reused for the next ablation. This is a 1-line guard in the
  scorer driver.

## Falsification design (Popperian)

This memo is itself a conjecture; the conjecture is
**signed top-K margin delta sees per-cell aberrations invisible to
rel-L2 and top-1 rank, and the speedup ladder composes**.

REFUTE conditions, each pre-committed:

1. On a 10-prompt × 100-cell pilot (Phase B), <5% of cells produce
   a signed-harm signal exceeding the 4×measurement-noise floor.
   (Would mean: the cells are too coarse OR the precision lift is
   illusory; H2022's K32 signed-vs-absolute finding doesn't replicate
   on DS4.)

2. Signed-harm rankings disagree with rel-L2 rankings on <20% of
   the top-100 hardest cells. (Would mean: the two metrics measure
   the same thing in practice; the lift is theoretical only.)

3. Speedup lever 1 (KV-shared prefix) gives <3× actual measured
   speedup on the pilot (vs the 10× target). (Would mean: the
   antirez disk-KV path doesn't compose with calibration mode as
   designed; the speedup story is wrong.)

4. Speedup lever 2 (structural panel) loses >5% of high-pressure-
   cell coverage when calibrated on DS4's own route-pressure
   distribution. (Would mean: H2040's panel structure is model-
   specific in a way that doesn't transfer to DS4.)

5. Phase B running on AIME 2026 P1-P10 with the inferguard rescue
   wrapper from #511 doesn't reproduce the 6/8 rescue rate at per-
   prompt granularity. (Would mean: the rescue protocol's success
   doesn't have a per-cell mechanism the scorer can locate; the
   8/10 ceiling is something else.)

## Pre-committed next ship targets

- Phase A code in `ds4_expert_table.h/c` (organ skip array + env-var
  loader). ~50 lines.
- A no-op smoke that confirms the dispatch path actually changes
  output when an organ is skipped. ~30 lines.
- A first DSL for the cells.csv format. ~10 lines doc.

Then the Phase B driver, then C, then D.

## Status

Memo only. No code shipped this turn. silv's prompt asked for thinking
review + accuracy/speedup ladder, not new code. Pre-committed REFUTE
conditions named above so the next ship lands against a falsifiable
target rather than another self-fulfilling exploration.

## Files

This memo: `tmp/20260527_harm_scorer/DESIGN.md`
DEPLOYMENT_RULES.md context: `tmp/20260527_hadamard/DEPLOYMENT_RULES.md`
Basis-aware marker: `ds4_expert_table.h/c` (gate_basis_hadamard16 etc.)
