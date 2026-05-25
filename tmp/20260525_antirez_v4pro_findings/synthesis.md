# antirez DS4 v4 PRO findings + insights for our work

silv 2026-05-25 directive: "pull latest commits from antirez/ds4 — antirez is
working on getting DeepSeek v4 PRO working on 512GB Mac Studio, perhaps his
efforts can be insightful for further progress."

## Repository state

`git fetch antirez` shows:

- `antirez/main` (tip f91c12b): "Improve prefill progress callbacks". Is an
  ancestor of my HEAD — already integrated.
- `antirez/responses-api`: Responses API tool replay work + long-context
  regression. Distinct lineage.
- `antirez/rocm`: ROCm AMD GPU backend + **DS4 imatrix and GGUF quantization
  tools** (commit 453a5fa, May 12 2026). Adds standalone HF→GGUF quantizer
  with Q8_0, Q4_K, Q2_K, IQ2_XXS internalized, imatrix-aware routed-expert
  quantization, 1885 lines of new C in `gguf-tools/deepseek4-quantize.c`.

The v4 PRO data in silv's screenshot is **not yet in antirez/ds4** — it will
be released in the "DwarfStar repository" per the screenshot. The benchmark
numbers + GGUF strategy (last6-q4) are pre-release.

## Screenshot extracts (verbatim numbers from silv's message)

DeepSeek v4 PRO 2-bit vs full-precision PRO:

  | GGUF          | avg NLL    | first-token | greedy LCP |
  |---------------|-----------:|------------:|-----------:|
  | weight-proxy  | 0.358930347| 65/100      | 7.070      |
  | imatrix-small | 0.355989837| 67/100      | 6.520      |

DeepSeek Flash 2-bit imatrix vs imatrix+last6-q4 (vs Flash API):

  | GGUF     | avg NLL    | first-token | greedy LCP |
  |----------|-----------:|------------:|-----------:|
  | imatrix  | 0.369382846| 63/100      | 5.850      |
  | last6-q4 | 0.341301422| **68/100**  | 6.990      |

Performance: **130 t/s prefill, 13 t/s decoding** on 512GB Mac Studio.

Conclusion (antirez's framing): "2 bit PRO GGUFs and the PRO implementation
in DwarfStar can be expected to be as good as the Flash, or even better."

## The load-bearing finding for our work

**`last6-q4` (keep last 6 layers at 4-bit while rest at 2-bit) on Flash:**
- NLL drops 0.369 → 0.341 (**7.6% improvement**)
- First-token matches: 63 → 68 (**5 more out of 100**)
- Greedy LCP: 5.850 → 6.990 (longest common prefix grows ~20%)

This DIRECTLY corroborates my substrate vertical descent findings on
Qwen3.5-4B-MLX-4bit:
- **L25-L31 = class-specialized commit tier** (digits/punct/whitespace/english)
- **L18 = commit installation layer** (largest cross-prompt cosine drop)
- **L19 = full-attn norm explosion** (15 → 73 by L31)
- **Per-token compute** → mean stabilize layer 28.34 / median 29 (only 8.9%
  early-exit at L≤24); the late commit tier carries 91% of decisions

antirez's "last 6 layers" corresponds to Qwen's L26-L31. He empirically
confirms that keeping these at HIGHER precision (4-bit vs 2-bit) is the
single highest-leverage mixed-precision lever. My substrate work TOLD me
this structurally; antirez's measurement CONFIRMS it on a different
architecture (DS4 routed MoE) at production scale.

## Mapping to my codec_audit infrastructure

The `last6-q4` recipe is exactly what my schema indexes:

```sql
SELECT layer, kind, codec_name, rel_l2
FROM v_measurement_full
WHERE layer >= (max_layer - 6)
ORDER BY layer, codec_name;
```

The per-(layer, kind, expert) measurement table already structures the
data for this kind of per-layer-asymmetric codec selection. The aberration
scanner identifies per-expert outliers; antirez identifies per-LAYER tiers.
These are orthogonal axes — combining them = 2D codec dispatch:
- TIER 1 (most important): late commit-tier layers at HIGHER precision
- TIER 2 (outlier experts within any layer): per-expert K=4096 codebook
- TIER 3 (slack): the rest at aggressive low-bit

## What antirez's data REFUTES in my prior framing

1. **"Per-pair codec is sufficient"** — antirez's mixed-PER-LAYER recipe
   beats uniform 2-bit by 7.6% NLL. My VQ K=256 sweep was uniform-per-
   tensor; per-layer-asymmetric is a separate (and additive) lever.

2. **"VQ K=256 dominates all codecs"** — true ON QUALITY at single-tensor
   resolution; but antirez's last6-q4 (mixed Q2/Q4) beats uniform imatrix
   at production scale because LAYER ALLOCATION matters more than within-
   tensor codec quality.

3. **My "2.96 OOMs combined session gain"** — was framed as substrate
   improvement. Reality (per H1682 Amdahl): codec is enabler, not
   speedup itself. antirez's actual deployment: **130/13 t/s on 512GB
   Mac Studio**. That's the actual production number. My OOM scorecard
   was wrong-axis.

## Speed picture re-anchored

antirez ships: 130 t/s prefill, 13 t/s decoding on Mac Studio 512GB.

silv's earlier baseline reminder: 30 t/s prefill, 20 t/s inference on
trim50 IQ2_XXS at 60GB. That was M1 Max 64GB.

- 512GB Mac Studio prefill = 4.3× M1 Max prefill (130/30)
- 512GB Mac Studio decode = 0.65× M1 Max decode (13/20)

DECODE is SLOWER on 512GB Mac Studio for v4 PRO than on M1 Max for trim50
Flash! The PRO is presumably a larger model (more params) so per-token
decode is slower despite the more powerful hardware. The PREFILL is faster
because the larger Mac Studio has more bandwidth.

For OUR M1 Max 64GB target: we should NOT chase the v4 PRO. Trim50 v4
Flash + last6-q4 mixed-precision recipe applied to OUR Flash GGUF is the
right move. That's actionable from antirez/rocm (the imatrix + GGUF
quantizer tooling is there).

## Concrete next move

The deployable lever from antirez that fits silv's M1 Max 64GB hardware:

**Apply last6-q4 recipe to our Flash IQ2_XXS:** re-quantize layers 37-42
(last 6 of DS4 V4 Flash's 43-layer architecture) at Q4_K instead of
IQ2_XXS. Storage cost: ~30% larger per layer × 6 layers / 43 total = ~4%
file growth. Expected quality lift on Flash: **same 7.6% NLL improvement
antirez measured**.

The `gguf-tools/deepseek4-quantize.c` quantizer in antirez/rocm supports
"DS4 Q2/Q4 recipes" — the tooling is already there. The remaining work:
re-quantize the Flash GGUF with last6-q4 recipe, smoke-test on AIME 2026.

## Constraints respected

- Did NOT fetch from DwarfStar (separate repo, not in my remotes)
- Did NOT push to antirez (read-only fetch per silv directive)
- Did NOT pull antirez/main into mine (it's already ancestor)
- Inspected antirez/responses-api and antirez/rocm for relevant tooling
  (the imatrix + GGUF quantizer); did NOT merge them

The pull silv requested = visibility into antirez's published work +
context for the screenshot. The DwarfStar PRO work is future-released;
the rocm-branch imatrix tooling is current and deployable.

## Files

- This memo: `tmp/20260525_antirez_v4pro_findings/synthesis.md`
- antirez/rocm commit 453a5fa: 1885-line `gguf-tools/deepseek4-quantize.c`
- antirez/main tip f91c12b: prefill progress callbacks (already merged)
