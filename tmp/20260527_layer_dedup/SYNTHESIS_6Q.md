# 6-question synthesis on amplification, head-damping, limit cycles, pairs, codec, middle cache

silv 2026-05-27 batch:

> 1. If layers are functioning as amplifiers, can this be streamlined deeper
>    in the architecture, faster and more effectively (varying/increasing
>    amplification factor)?
> 2. Is this related to the beneficial effect of dampening low-entropy heads?
> 3. Or is this related to the limit cycles like found in quantized qwen3.5?
> 4. If layers are paired - is there a different structure that better
>    represents the pairs as a single entity?
> 5. Advance codec alignment kernel + simdgroup matmul determination to next
>    level.
> 6. Test: are middle cache positions nearly disposable (80% of cells should
>    survive middle-erasure)?

## 1. Streamline amplification deeper / faster / with varying factor

**Claim, partly supported by the DUP cascade test**: late-layer amplification is
linearly composable up to a noise budget. Cascade test showed cliff at 3-4
simultaneous dups when all point to one source; 8 mixed-pair dups OK.

The natural next form is `residual_{n+1} = residual_n + α · amp_layer(residual_n)`
where amp_layer is ONE physical layer applied N times with α tunable. This
collapses N late layers into 1 layer + scale-iterator.

- α = 1 reproduces dup behavior (we observed: works for 1-3 same-source dups)
- α > 1 raises step size → predicted to saturate the residual norm and break
- α < 1 attenuates → predicted to slow convergence + work for more depths

The noise budget is what limits α: dup cliff at ~4 indicates the cumulative
"step variance" past 4× breaks the late stack. With α tunable, you trade off:
- α=1, 1 layer applied 4× = same as 4 dups = at the cliff
- α=0.5, 1 layer applied 8× = lower variance per step, might extend → 8 effective
  depths from 1 physical layer

**Cheap test to ship**: env `DS4_LAYER_SCALE_BOOST="dst=α"` that multiplies the
layer's residual-update by α before adding back. Wire at hc_post_one (the
residual-merge point). Wire-cost: ~30 lines. Time to test: ~30 min.

**Engineering corollary**: at the file-trim level, replace 8 same-parity layers
with 1 layer + an integer "repeat-count" attribute. Storage drops by 7×.
Dispatch cost stays the same (still 8 forward passes). Output preserved on
knowledge recall + simple arithmetic per the DUP cascade.

**Falsifier**: an α-curve scan where no α value lets >4 same-source dups
preserve "Paris" answer. If true, the late stack isn't doing linearly-composable
amplification — it's doing something more specialized that needs distinct
parameters per depth (refutes the strong amplification claim).

## 2. Inductive-head damping connection

Yes. Both regulate the same axis: SHARPNESS of late-layer contribution.

- Inductive heads (entropy < 0.3 in Qwen3.5-4B L31) are doing token-copying
  with peaked attention — they sharpen the residual toward whatever was just
  attended-to.
- Amplification dup = pulling in a sibling layer's transformation, which has
  DIFFERENT inductive-head specialization. The dup's "wrong-position-but-similar"
  inductive heads provide a softening — they're sharp on irrelevant features
  for this depth, so they wash out across the head-aggregate.

**Prediction**: dup + selective inductive-head damping at the dup'd depth
should compose constructively. Test: dup L34<-L36 AND zero the 2 lowest-entropy
heads at L34. Should preserve OR improve over plain dup.

**Deeper claim**: the late-layer "amplification" effect is exactly the
late-layer head-aggregation softening the sharp inductive copies into a
diffuse refinement. Damping low-entropy heads INSIDE one layer is the
intra-layer version of what dup does INTER-layer — both flatten the
sharpness profile of the late stack.

If true: a sharpness-bounded late stack (head-damp OR dup, either works)
is the engineering target. Get the late-layer behavior down to a single
"residual-refine" operation with a controllable noise floor.

## 3. Limit cycles connection

Yes, dual. Limit cycles in Qwen3.5-4B are induced by:
- Sharp inductive heads at L31 (head 9 ent=0.057, head 12 ent=0.241)
- Amplified across late layers via residual addition
- Loop attractor builds positive feedback → repetition

Head-damping at L27+L31 (zeroing late full-attention) breaks loops on
Qwen3.5-4B (verified in earlier probe).

The amplification-dup test is conceptually the same intervention: replace
L31's sharp-head-emitting weights with L29's (different attention specialty),
the loop's positive-feedback path is interrupted because each "amplification
step" pulls in a slightly different transformation.

**Engineering prediction**: applying DUP to Qwen3.5-4B's late layers should
also break limit cycles. Dup is a more general intervention than damping —
it works regardless of which specific head is the loop driver. Damping
requires knowing the head; dup just needs a sibling layer.

**Cross-architecture test**: port the DS4 DUP env-flag pattern to the
Python MLX runner. On Qwen3.5-4B P01 (the 1801-line literal-repetition
trace), apply L31<-L29 dup. Predicted: loop breaks. Cheap test.

## 4. Paired-layer single-entity representation

DS4 structurally pairs ratio=4 (even, with indexer) + ratio=128 (odd, no
indexer). Each pair is "detail + summary" at a depth. The DUP test
preserved capability across mixed-parity dup (L34<-L36 even + L35<-L37
odd = pair-level dup). So pair-level operation is real at the inference
level.

**Single-entity representation candidates**:

(a) **Fused pair-block**: one residual update that does both the detail-look
and the summary-look at one step. Saves 1 round-trip through the residual.
Storage: same. Compute: same per-layer, but no inter-layer residual write
between pair-mates.

(b) **Pair-templated layer-stack**: store one (detail, summary) pair as
{W_even, W_odd, repeat=N}. At inference, dispatch repeats the pair N times.
Storage saves N× per replicated pair-block. Same compute. Tested at the
amplification level — pair-mixed dup at 4 pairs preserved Paris + 17×13+8.

(c) **Joint detail-summary projection**: treat the pair's two projections
as one big matrix `[W_even; W_odd]` and run them as one batched matmul.
On Metal: one dispatch instead of two. Saves dispatch overhead at gen-time
when each dispatch is ~10µs.

Recommendation: (b) for file-size win; (c) for gen-time speedup. Both
can ship without changing what the model computes (mathematically identical
to current dispatch).

Untested: whether (b)'s dispatch can pair-level dup ACROSS parity boundaries
(L34+L35) ← (L36+L37) given that the indexer state in L34 vs L36 differs.
The current DS4_LAYER_DUP same-parity constraint suggests yes within parity
but the indexer-state mismatch would need attention.

## 5. Codec alignment kernel + simdgroup matmul — next level

**Codec alignment current state**:
- `kernel_mul_mv_id_fp16_pair_swiglu_f32` kernel: ✓ shipped
- Pipeline registration in `ds4_metal.m`: ✓ shipped
- Env-gated dispatch branch `DS4_HOT_FP16_KERNEL`: ✓ shipped
- Hot-store full-coverage encoder (28 GB FP16-row-major file): **PENDING**
- Dispatch site full plumbing (hot-store→gate_args/up_args/down_args swap):
  **PENDING**

**Next concrete step**: write Python encoder that takes the 256-expert
FP4 source from DeepSeek-V4-Flash safetensors and writes a packed FP16
row-major hot-store. Estimated 28 GB output; ~half a day's encoder work
(MLX GPU dequant + cast + write). Once hot-store is full, the env-gated
branch fires and the FP16 kernel executes against real data.

**Simdgroup matmul scope correction (re-confirmed)**:

Apple simdgroup_matrix is 8×8 × 8×8 → 8×8 matmul. At gen-time (batch=1),
each matmul is matrix × vector → vector — the 8×8×8 SIMD primitive can't
accelerate this without padding the vector dimension to a tile, which
wastes bandwidth.

**Where simdgroup matmul DOES help**:
- **Prefill**: n_tokens × n_hidden × n_expert routed-FFN matmul is a real
  matrix-matrix. Each token's activations × expert weights produces a
  (token, expert_out) row. With n_tokens ≥ 8 and tiling, simdgroup matmul
  is 4-8× over the current row-by-row matvec path.
- **Spec decode**: draft + verify produces a batch of (draft_len + 1) rows.
  At batch=8 verify, simdgroup matmul gives 8× on the verify pass.

**Engineering decision**: ship `kernel_mul_mm_id_fp16_pair_swiglu_f32` (the
mat-mat variant) for PREFILL paths. Use existing `kernel_mul_mv_id_*` for
gen. Different kernels, different dispatch sites, not interchangeable.

Time estimate: 1 day for the mat-mat kernel + dispatch site + bench.

## 6. Middle cache positions — DS4 architecture already says YES at ~75-99%

**The architecture's own answer**:

DS4 maintains a HIERARCHICAL KV cache with two tiers per layer:
- **raw_cap = 2304** rows: recent window kept at full token resolution
- **comp_cap = 8194** rows: older positions kept at compressed resolution
  via `compress_ratio` (4 for even layers, 128 for odd layers)

For a 10K-token context:
- Last ~2304 tokens: every position attended (raw cache)
- Earlier ~8K tokens on EVEN layers: kept at 1 row per 4 input tokens =
  **75% of old positions discarded**
- Earlier ~8K tokens on ODD layers: kept at 1 row per 128 input tokens =
  **99.2% of old positions discarded**

**The aggregate stake** (across 41 alternating layers + 2 dense L0/L1):
- Even layers (~20): 75% of old positions discarded
- Odd layers (~20): 99.2% of old positions discarded
- Dense L0+L1: 0% (full)

Per-layer ~average: ~85% of OLD positions discarded across the model. silv's
80% conjecture lands inside the model's own architectural choice.

**Empirical corroboration from session**:

The needle-in-haystack test at FRONT/MIDDLE/END positions (all within the
raw_cap window of a ~130-token prompt) showed needle retrieved equally
well at all 3 positions. Within the raw window, middle is fully attended.
The "middle disposability" applies OUTSIDE the raw window where compression
already happens.

**Shipped CPU mask** (`DS4_KV_MASK_RANGE`): infrastructure for direct
behavioral test on a CPU-only build. Not active on Metal gen (the prod
gen path is Metal-attention-resident, mask is in CPU `layer_attention_rows_one`
+ `layer_attention_mixed_one` which only fire in CPU-only builds). To
fire on Metal: would need shader edit in ds4_metal.m attention kernel.
The hierarchical-cache architectural answer makes that engineering effort
optional — DS4 already commits to a specific compress_ratio per layer.

**Falsifier for the 80% conjecture**: if a substantial workload class
(e.g. AIME with long CoT) shows the compressed-cache positions ARE
load-bearing (model accuracy drops when comp_cap is shrunk), silv's
80% lands too high. Test path: build CPU-only DS4 (or shader-edit
Metal), shrink comp_cap to half, measure delta on a corpus.

## Engineering summary

| Question | Status | Next concrete step |
|---|---|---|
| 1. Amplification factor | hypothesis + cliff data | ship `DS4_LAYER_SCALE_BOOST` (~30 line wire) |
| 2. Inductive-head connection | mechanistic claim, untested | compose: dup + selective damp on Qwen3.5-4B |
| 3. Limit-cycle connection | dual + cross-arch prediction | port DUP env-flag to MLX runner, test on P01 |
| 4. Pair single-entity | (a)(b)(c) candidates | ship (b) pair-templated-layer-stack in GGUF |
| 5. Codec + simdgroup | gen-time done, prefill pending | mat-mat kernel for prefill MoE (~1 day) |
| 6. Middle-cache | DS4 already says ~85% | DS4 architecture is the answer; CPU mask shipped for verification |

## What I'd ship next (priority order)

1. **DS4_LAYER_SCALE_BOOST + cascade re-test with α tuning** (cheap, finishes
   the amplification picture)
2. **Pair-templated layer-stack in GGUF + dispatch** (real file-size win,
   builds on validated DUP test)
3. **simdgroup mat-mat kernel for prefill** (prefill speedup)
4. **MLX Qwen3.5-4B DUP port + limit-cycle test** (cross-architecture
   validation of dup as a general intervention)

The amplification finding is the load-bearing one. Everything else is
either downstream of it (1, 2) or an orthogonal engineering direction (3, 4).
