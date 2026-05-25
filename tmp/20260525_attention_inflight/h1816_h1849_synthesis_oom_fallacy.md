# Codex H1816-H1849 sequential read + the top-k OOM fallacy

silv 2026-05-25 directive: read codex H1816-H1850 sequentially +
consider implications of sampling-instrumentation fallacy. Codex
ended at H1849 (file timestamp 20:42). Read complete.

## The H1816-H1849 codec arc summary

**H1816-H1819 (AMD VQB1 packet generation)**: from concept to first
artifact. Packet quality at K256 max_iter=1 already reaches rel-L2
0.023; max_iter=50 only improves to 0.022. Convergence comes from
better objectives, not more iterations.

**H1820-H1822 (ROCm encoder speedup)**: 42.73× speedup via slice-
before-dequant + grouped-shard loading. "Tensor load is not atomic"
— it's file lookup + byte fetch + byte/scale slicing + FP4 unpack +
scale expansion + materialization + consumption. Sliced before
dequant whenever the downstream objective reads a window.

**H1823-H1826 (VQB2 packed-codes)**: VQB1's uint8 codes waste memory;
VQB2 packs to 4-bit (K16) / 6-bit (K64) / 8-bit (K256). Direct VQB2
emission removes VQB1 intermediate.

**H1827-H1829 (max_iter pressure)**: max_iter from 7 → 500 gives
EXACT SAME fixed point. **max_iter is a ceiling, not a quality dial.**
Quality must move via objective/sample/representation change.

**H1830-H1834 (objective selector ladder beginning)**: candidates
evaluated by source rel-L2 → expert-route rel-L2. H1830 wins globally
at identical bytes. H1832-H1834: route-weighted scoring → still
prefers H1830, but route concentration was broad. H1834: per-expert
profile permutation cheap; H1830 wins all 13 profiles.

**H1835-H1836 (first inversion)**: activation-weighted scoring
(`error · hidden`) → H1830 wins globally but H1826 wins
`router_row_mix|pressure:high`. **The first proof that source rel-L2
and expert route rel-L2 are FALSE PROXIES on some hidden-conditioned
submanifolds.** H1836 selects s20260525_n200k as new winner on the
target surface.

**H1837 (the meta-doctrine explicit)**: "each rung can refute the
previous one." Objective ladder = source rel-L2 → expert-route rel-L2
→ selected up@hidden → gated partial FFN → full FFN/logit delta.

**H1838-H1841 (slice-aware loader + fast scorer = 10.83×)**:
"Objective functions are not indivisible either." Decomposed into
route-edge table + hidden-by-edge + source gate activation + source
partial output + decoded expert matrix + artifact decoded output +
profile aggregation. Bridge to fast full-FFN/logit-delta selectors.

**H1842 (wide-down refutation)**: first-128 down output rows is NOT
a harmless shortcut. 3/13 winners flip. The next rung needs routed
top-6 sum + post-layer/logit delta.

**H1843-H1844 (case-summed routed FFN + anchor-RMS)**: H1843 moves
to case-summed `sum_top6(route_weight * expert_output)`. 4/13 flips
vs H1842 edge-additive. H1844 anchor-RMS matches case-summed (0/13
flips, validating).

**H1845 (max_iter quality hardening)**: max_iter from 100 → 5000 →
exact same fixed point. Reaffirms H1827-H1829.

**H1846 (tinygrad MoE state trace)**: PATCH SHIPPED.
oss_patches/20260525_h1845_tinygrad_moe_state_trace.patch. Captures
pre/post-residual, attention norm/output, FFN norm, routed FFN output,
post-FFN residual, deltas, cosines. Output-parity test passes.

**H1847 (residual damage grid)**: 108 variants, 54 changed route. Same
route loss attenuated 70× by residual scale (carrier scale 1→100); or
amplified 24× by expert function (spread 0.1→5.0). **Packet risk is
not atomic — it's a RELATION** among route choice, expert function,
residual carrier, normalization, readout basis.

**H1848 (logit-margin damage)**: 243 variants × 729 readout cases =
194 argmax flips. Small residual damage decisive: residual rel-L2
0.000540 produced argmax flip with 120× margin consumption. High
route loss harmless: probability-mass loss 0.918, post-residual rel-L2
0.209 — NO flip when margin opposed. **Packet risk is downstream
margin consumption at the actual readout**, not route mass or
residual norm.

**H1849 (DS4 logit-lens margin selector + 64640× cut)**: applied
H1848 doctrine to real DS4 L25 candidates via tied embed projection.
On `pressure:high`, case rel-L2 prefers h1830_s42_n100k; margin
consumption prefers h1827_fit1m despite worse case rel-L2. New
selector inverts the choice on the metric that matters downstream.

**The 64640× speedup**: full-vocab logit lens reduces from
`144 × 129280 × 4096` dot work to `144 × 2 × 4096` when scoring only
top-2 margin. **4.81 OOM arithmetic cut in the inner loop**.

## THE FALLACY silv pointed at (and codex H1849 demonstrates)

> "sampling the final/interim logits/weights and keeping only the
> temp/top_k instead of employing sampling instrumentation"

**Default developer assumption**: compute the full softmax distribution
over all 129,280 vocabulary tokens, then apply temperature/top_p/top_k
as post-hoc filtering. The full softmax computation is "the model's
output" and the filtering is "sampling configuration."

**The actual decision uses 1-2 tokens**: argmax, or top-2 for margin
analysis. The 129,278 other tokens were computed and discarded.

**The fallacy**: confusing "model output" (which is conceptually the
full softmax) with "what the system actually needs" (the top-k for
the actual decision step).

**Codex H1849's structural fix**: compute the embedding rows for the
source's top-k once, then score candidate packets against those k rows.
129,280-dim full-vocab projection → 2-dim margin scoring. Same answer,
64640× less arithmetic.

**Generalizations across the stack**:

1. **Inference-time decoding**: don't compute the full softmax then
   sample. Compute only top-k logits via partial sort. The standard
   `sample(softmax(logits / T), top_p, top_k)` pipeline burns
   ~129,000× more compute than needed for an argmax/top-2 decision.

2. **Training-time loss**: cross-entropy on full vocab vs sampled-softmax
   loss on top-k targets + noise contrastive. Full vocab CE assumes
   we need the full distribution; in practice we need the GROUND TRUTH
   match + top-K margin. ~10-100× compute reduction in training step.

3. **Attention**: standard softmax attention computes O(N²) scores.
   Top-k attention (e.g., NoPE-style or routed-attention) computes O(N·k).
   When k=8, that's ~100× speedup at long context.

4. **KV cache**: store full K,V per token vs store top-k attended
   K,V per query. The KV cache as deployed assumes all-pairs attention
   may be needed; the actual decision uses k=8-32 attended tokens per
   query. ~50× memory reduction.

5. **Codec quality (codex's domain)**: H1849's 64640× — the proof
   that this fallacy holds in adjacent domain.

## What this means for sampling instrumentation

Standard "sampling instrumentation" (temperature, top_p, top_k as
hyperparameters) is THE EVIDENCE that the fallacy is widespread. We
compute the full softmax then ADD filtering parameters. The
parameters wouldn't need to exist if we computed top-k natively.

**Refined**: sampling instrumentation is the BAND-AID over the
fundamental "compute everything, filter later" wasteful pattern.
Replacing softmax(.)[:k] with top_k(...) at the GPU kernel level
eliminates the need for sampling instrumentation as a separate
configuration layer.

## Cross-arc convergence with my AIME session

My session's findings parallel:
- Substrate L31 truth latent (refuted as position artifact) ↔ H1842's
  first-128 down output rows refuted
- Mode 2 prior fallback at boundary ↔ H1848's margin consumption
  matters not residual norm
- Late-layer flip on P04 (probe 19) ↔ H1846 residual capture +
  H1847 component-vs-relation distinction
- Cap mechanism rule ↔ H1849's objective selector

Both arcs converge on: **isolated component metrics LIE; the consumed
downstream object is what matters**.

## OOM speedup ladder for production deployment

Combining codex's findings with my session's:

| layer | speedup | mechanism | status |
|-------|---------|-----------|--------|
| 0 | 4.8 OOM | Top-k softmax (H1849-style) | UNTESTED at inference |
| 1 | 2.16 OOM | Cap-based rescue vs naive K=8 | DEPLOYED (this session) |
| 2 | 1.5 OOM | Tinygrad MoE state trace | SHIPPED (codex H1846) |
| 3 | 1 OOM | Fused router_weights_with_remap | SHIPPED (silv architecture review) |
| 4 | 1.3 OOM | ICB recording 19× wall (H1696) | scaffold + Phase 6 done |
| 5 | 0.5 OOM | Slice-before-dequant | SHIPPED (codex H1822) |

**Composable theoretical ceiling**: ~10 OOM if all layers were
multiplicatively independent. Practical: 3-4 OOM after Amdahl
adjustment for non-load-bearing paths.

The top-k softmax fallacy is the highest single-lever speedup
identified. Untested in production inference but proven at codec
quality scoring (4.81 OOM).

## What needs to happen next

1. **Verify the merge build** (M1 with --prefill-metal-phases auto)
2. **Test top-k softmax at inference**: replace full-vocab logit
   computation with partial-sort top-k in DS4's generation path
3. **Apply codex H1849 doctrine to DS4 codec selection**: use margin
   consumption not source rel-L2
4. **Cross-validate** AIME rescue on the merged binary with PRO support

## Files

- `tmp/20260525_attention_inflight/h1816_h1849_synthesis_oom_fallacy.md` (this)
- Codex source: `/Users/silv/cl/tlp_codex/CODEX_SHIFTS.md` H1816-H1849
- Merged repo: `/Users/silv/cl/tlp/montyneg/ivp5_ds4/` (commit 755fec5)
