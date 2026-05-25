# VQ cross-tile codebook generalization (partial)

Extension of the row-coverage audit (commit 3bef5a7) to test
cross-(layer, kind) codebook generalization. Hypothesis: FP4 (re, im)
structure is uniform enough that ONE codebook serves multiple tiles.

## Setup

Fit VQ K=256 codebook on L0.down (256 experts × 128 rows × 1024
pairs = 33M pairs). Apply that single codebook to other (layer, kind)
tiles. Compare rel_L2 vs per-tile codebook.

## Partial results (L0 cross-kind)

| Layer | Kind | per-tile rel_L2 | shared rel_L2 | ratio |
|-------|------|-----------------|---------------|-------|
| 0     | gate | 0.0192          | 0.0204        | 1.061 |
| 0     | up   | 0.0190          | 0.0197        | 1.037 |
| 0     | down | 0.0195          | 0.0195        | 1.000 (identity) |

L22 and L42 tests didn't complete in time budget (~300s).

## Finding (within scope)

**Within L0**: the down-fit codebook generalizes to gate and up at
~4-6% quality penalty. Strong evidence for the K=256 saturation
hypothesis — FP4 dequant produces only 16² = 256 distinct (re, im)
pairs per scale block, and the dominant pairs are similar across
kinds within a layer.

## Implication for deployment

If cross-tile generalization holds across LAYERS too (untested here),
the full DS4 V4 corpus could use:
- **One global codebook** (256 × 2 floats = 2 KB total)
- 1-byte codes per pair (139 GB at full row coverage)
- Codebook storage overhead approaches zero (2 KB vs 258 KB for
  per-tile codebooks)
- Quality penalty: ≤6% rel_L2 increase (per the L0 measurement)

Combined with row-coverage audit (commit 3bef5a7) which showed
codebook generalizes across rows within a tile, this would mean
the ENTIRE DS4 V4 model substrate codec needs only 2 KB of codebook
metadata.

## Open question (next session)

Does the codebook generalize across LAYERS too? Need to confirm with
L22 and L42 tests (which timed out at 300s in this run).
Approximate budget for full 3×3 cross-tile test: ~10 min if using
MLX-accelerated K-means + assignment.

## How this composes with prior session findings

Row-coverage audit (commit 3bef5a7):
- Within (layer, kind): codebook fit on rows[0:128] gives rel_L2
  0.0203 on full 4096 rows vs full-fit 0.0202 (ratio 1.006×)

This audit (partial):
- Within layer, across kinds: L0.down codebook on L0.gate gives
  ratio 1.061×; on L0.up gives 1.037×

Combined: within a single layer, ONE codebook serves all 3 kinds
across all rows at ≤6% quality cost. Cross-layer remains untested.

## Why this matters

Deploys VQ-2D with substantially smaller per-layer codebook
overhead. The full DS4 V4 corpus codebook footprint:
- Per-(layer, kind): 129 × 2 KB = 258 KB total
- Per-layer (shared across 3 kinds): 43 × 2 KB = 86 KB total
- Global (shared across all): 1 × 2 KB = 2 KB total

Negligible savings vs the 138 GB codes payload, but architectural
simplicity matters for the kernel (single codebook fetch vs
per-(layer, kind) lookup).
