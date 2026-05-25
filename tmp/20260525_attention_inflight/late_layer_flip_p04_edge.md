# 19th probe: late-layer flip mechanism in P04 edge case

silv 2026-05-25 post-18-probe-closure-attempt: deep-mined per_layer
dict in forced_p*.json. The 32-layer trajectory reveals LATE-LAYER
INSTABILITY on edge cases.

## P05 (clean success, truth=65, success at cap=12k)

| layer | top1_token | top1_prob | rank_truth | p_truth |
|-------|-----------|-----------|------------|---------|
| 0-22  | garbage (CJK, weird) | varies | huge | ~1e-6 |
| 23    | **'6'**   | 0.006     | 0          | 0.006   |
| 24-28 | noise mixed |         | 0-9       | varies  |
| 29    | **'6'**   | 0.572     | 0          | 0.572   |
| 30    | **'6'**   | 0.9995    | 0          | 0.9995  |
| 31    | **'6'**   | 0.9697    | 0          | 0.9697  |

Stable truth at layers 29-31. Emission emits '65' (correct).

## P04 (borderline fail, truth=70, fail at cap=12k)

| layer | top1_token | top1_prob | rank_truth | p_truth |
|-------|-----------|-----------|------------|---------|
| 0-22  | garbage | varies | huge | ~1e-5 |
| 23    | ' prime'  | 0.268     | 2126       | 3.2e-5  |
| 24    | ' prime'  | 0.110     | 1424       | 5.8e-5  |
| 27    | ' seventy' | 0.004    | 16         | 6.6e-4  |
| 28    | ' something' | 0.010 | 2          | 0.007   |
| 29    | **'7'**   | 0.086     | 0          | 0.086   |
| 30    | **'7'**   | 0.760     | 0          | 0.760   |
| **31** | **'6'**  | 0.252     | 2          | 0.200   |

**Layer 30 top-1='7' (truth)** with prob 0.76. **Layer 31 FLIPS to '6'**,
truth drops to rank-2.

## The mechanism

P04 is the cap=12k edge case where my rule predicted FAIL (truth_at=20149
> 12000). The 18th probe noted `final3_agreement=3` suggesting substrate
had truth at final 3 layers. The deeper data refines this:

- Layer 30: truth IS top-1 (prob 0.76 — strong)
- Layer 31: lm_head FLIPS away from truth to '6' (no obvious connection
  to context content)

The model's substrate at layer 30 had the right intuition (perhaps from
problem-structure knowledge that '70' is plausible). Layer 31's lm_head
output overrode that intuition.

The override toward '6' is interesting because:
- '6' isn't in the last 500 chars of the prefix (per 15th probe finding)
- '6' isn't truth
- '6' isn't related to ' seventy' (layer 27) or ' something' (layer 28)
  semantically

This is a clean instance of **late-layer instability where the substrate
"holds" truth at layer 30 but the final layer flips to a Mode-2 prior
digit at layer 31**.

## Implication

The "final3_agreement" metric from the 18th probe was MISLEADING — it
suggested all final 3 layers agreed on truth, but actually only layers
29-30 agreed; layer 31 flipped. The metric must have counted layers
where TRUTH DIGIT is rank-1 individually, regardless of final-layer
consistency.

Refined picture of P04 cap=12k:
- Substrate has truth latent at layers 29-30 (rank-1)
- Final lm_head layer flips away (rank-2)
- Emit = whatever layer 31 says (Mode 2 prior)
- Rescue fails

The cap mechanism rule (rescuable ⇔ truth_at < cap) predicted FAIL at
cap=12k for P04. The rule is correct on emit-level. The substrate
analysis reveals the FAILURE MECHANISM is "late-layer flip from
latent-truth to Mode-2 prior."

## What about the success cells?

P05 (and presumably P03, P06, P07, P08) have STABLE truth across layers
29-31. The model's substrate progressively converges to truth and the
lm_head emits it.

When truth is in the prefix (cap > truth_at):
- Mid-layers: noise
- Late layers: substrate converges to truth via attention to prefix
- lm_head: emits truth from layer 31

When truth is NOT in prefix but substrate has problem-structure-derived
intuition (P04 edge):
- Substrate at layer 30 reaches truth (intuition fires)
- lm_head at layer 31 overrides toward Mode-2 prior

The lm_head's "override" mechanism is interesting. The model has
TRAINED to use lm_head as the final extraction layer; intuition-from-
substrate at intermediate layers may not survive the final lm_head
transform when context doesn't reinforce.

## Possible third lever (untested without live model)

If we could capture LAYER-30 logits during forced-commit and EMIT
from layer-30 rank-1 instead of layer-31 rank-1, we might rescue P04
at cap=12k. The substrate has truth latent; the lm_head is what
loses it.

This would be a substrate-bypass rescue: skip the final lm_head, emit
from the last intermediate layer that has truth at rank-1.

Untested without live MLX access. Just a candidate mechanism.

## 19-probe arc

The 17th was "closure"; the 18th found summary metrics in un-mined
forced_p*.json; the 19th deep-mined the per_layer data to find the
specific FLIP MECHANISM in P04. Each "closure" continues to surface
deeper structure when actually examining the data.

The pattern is: **closure is provisional until the next layer of
data is examined**. Each session has effectively unbounded cache to
mine; the question is which mining yields informative findings vs
which mines diminishing-returns.

## Files

- `tmp/20260525_attention_inflight/late_layer_flip_p04_edge.md` (this)
- Data: `tmp/20260525_attention_inflight/forced_p{04,05}.json`
  per_layer dict with 32-layer logit-lens trajectory
