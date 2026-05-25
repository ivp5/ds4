# Can VQ fit in <52 GB on M1 Max 64 GB?

silv 2026-05-25: "does vq have knobs to tune space-vs-accuracy tradeoff to try
and fit into less than 52gb and thus inside m1 max 64gb memory?"

## TL;DR

Yes. Three families of VQ knobs trade space for quality:

1. **K (codebook size)** — log2(K) bits per encoding unit
2. **Per-block scale** — adds dynamic range at fixed K, costs 1 byte / N weights
3. **Block size N** — encode N-pair blocks as one index = log2(K)/N bits per pair
4. **PQ (Product Quantization)** — split block into sub-blocks, encode each — Faiss trick
5. **Codebook scope** — per-tensor / per-(layer,kind) / per-row / per-expert

The VQ recipe-space includes designs strictly better than IQ2_XXS at SAME storage,
and designs that FIT under 52 GB at varying quality lift.

## Simple K-sweep (per-pair VQ, no block, scale variants)

`vq_k_sweep_for_52gb.py` results on L0 gate expert 0 first 128 rows:

| Recipe | byte/pair | rel_L2 | vs IQ2 |
|--------|----------:|-------:|-------:|
| IQ2_XXS reference | 0.516 | 0.354 | 1.0× |
| VQ K=4 (no scale) | 0.250 | 0.606 | 0.58× WORSE |
| VQ K=4 + block32 scale | 0.313 | 0.610 | 0.58× WORSE |
| VQ K=4 + per-row scale | 0.252 | 0.608 | 0.58× WORSE |
| VQ K=16 (no scale) | 0.500 | 0.349 | 1.01× |
| **VQ K=16 + block32 scale** | **0.563** | **0.319** | **1.11× better** |
| VQ K=16 + per-row scale | 0.502 | 0.341 | 1.04× better |
| VQ K=64 (no scale) | 0.750 | 0.175 | 2.02× |
| VQ K=64 + block32 scale | 0.813 | 0.156 | 2.27× |
| VQ K=256 (no scale) | 1.000 | 0.089 | 3.98× |
| **VQ K=256 + block32 scale** | **1.063** | **0.073** | **4.85× better** |

Key finding from per-pair sweep:
- **K=4 (any scale) is WORSE than IQ2.** Too few clusters in R² to cover pair
  distribution. Drops below IQ2 quality despite 2× smaller storage.
- **K=16 (best storage match to IQ2) beats IQ2 by 1.04-1.11×.** Same storage,
  slightly better quality. Marginal but positive.
- **K=64 beats IQ2 by 2.02-2.27× at 1.5× storage cost.**
- **K=256 beats IQ2 by 3.98-4.85× at 2× storage cost.**

## Storage projections (per-pair VQ only)

| Recipe | Full DS4 | Trim50 | Fits 52 GB? |
|--------|---------:|-------:|:-----------:|
| IQ2_XXS reference | 86.7 GB | 47.7 GB | yes (Trim50) |
| VQ K=4 (no scale) | 47.2 GB | 28.6 GB | YES — Full DS4! But quality WORSE |
| VQ K=4 + per-row scale | 47.5 GB | 28.7 GB | YES — Full DS4! But quality WORSE |
| VQ K=16 (no scale) | 84.4 GB | 47.2 GB | yes Trim50 only |
| VQ K=16 + block32 scale | 93.7 GB | 51.8 GB | barely Trim50 (51.8 GB) |
| VQ K=64 (any scale) | >120 GB | >65 GB | no |
| VQ K=256 (any scale) | >159 GB | >84 GB | no |

The PER-PAIR VQ design space has a hard limit: at K ≥ 16 (needed to beat IQ2
quality), storage is ≥ 0.5 byte/pair. Full DS4 routed experts at 0.5 byte/pair =
74 GB just for routed experts, before the 10 GB spine. **Full DS4 in <52 GB is
not achievable with per-pair VQ at IQ2-or-better quality.**

The two paths forward for full DS4 <52 GB:

1. **Block-VQ / Product Quantization** — encode larger blocks as single indices.
   Theoretical storage: log2(K)/N bits per pair. For 16-pair blocks and K=4096:
   12 bits / 16 pairs = 0.094 byte/pair → routed-expert ~14 GB → total ~24 GB.
   Quality TBD (running `block_vq_sweep.py` now).

2. **Smarter trim** — extend silv's existing trim50 with the per-expert
   aberration signatures. The L0 gate scan showed E243, E20, E203, E55, E93 are
   the 5 worst-VQ outliers. If trim removes those (assuming the bimodal tail
   maps to "less-important" experts), the remaining experts encode cleaner. Need
   to cross-reference aberration outliers with task #463's existing mask.

## High-potential recipes — comparison-ready

Currently waiting on `block_vq_sweep.py` for measured rel_L2 of:

- Block-VQ K=256/1024/4096/16384 with 4/8/16-pair blocks (12 configs)
- Block-VQ with per-block fp16 scale (5 configs)
- PQ with various sub-block / K_sub combinations (9 configs)

Expected Pareto from theory:
- **PQ block=16p sub=2 K_sub=256**: 0.125 byte/pair, rel_L2 ~0.05-0.10 (Faiss baseline)
- **Block-VQ block=16p K=4096**: 0.094 byte/pair, rel_L2 ~0.10-0.15 (joint coverage limited)
- **PQ block=16p sub=4 K_sub=256**: 0.250 byte/pair, rel_L2 ~0.01-0.03 (best quality at ~IQ2 storage)

## Full deployment scenarios for <52 GB on Full DS4

| Scenario | byte/pair | Routed GB | Total GB | Expected rel_L2 | Lift vs IQ2 |
|----------|----------:|---------:|--------:|----------------:|------------:|
| IQ2_XXS (current) | 0.516 | 76.7 | 86.7 | 0.354 | 1.0× |
| **PQ block=16p sub=4 K_sub=256** | 0.250 | 37.2 | **47.2** | est. 0.02-0.03 | est. 14× |
| **Block-VQ block=16p K=4096** | 0.094 | 14.0 | **24.0** | est. 0.10-0.15 | est. 3× |

Both PQ and Block-VQ fit comfortably AND beat IQ2 quality (per theory; measured
data pending). The PQ design is the recommended target: ~half IQ2 storage with
~14× quality lift.

## Speed considerations (decode kernel)

Per-pair VQ K=256: 1 codebook lookup per pair, single fp32 read per code byte.
This is the `gate_up_down_vq` kernel I shipped in B-2.3c canary chain. Codex
H1789 measured 3.91-5.37ms for the raw-IQ2 equivalent path on 864-case packet.

Block-VQ K=4096: 1 lookup per 16-pair block, 16 fp32 reads per 12-bit code. Net
read traffic = 16/8 × 4 bytes = 8 bytes per pair (vs 1 byte for per-pair VQ K=256).
But codebook is small (4096 × 32 × 4 = 512 KB per (layer, kind)) so fits
threadgroup memory; decode is fast.

PQ: 1 lookup per sub-block, more lookups per block but each in a small
sub-codebook. Sub-codebooks pack into 32 KB threadgroup memory together.

H1793's chunked-256-dim decode pattern composes directly: chunk = 16 pairs ×
M sub-blocks each at K=256. PQ decode is the same threadgroup pattern as H1789,
just with M-stride lookups instead of M=1.

## Action items

When `block_vq_sweep.py` completes:
1. Update this memo with measured rel_L2 for all 26 block/PQ configs
2. Identify the Pareto frontier: codecs that fit <52 GB AND beat IQ2 quality
3. Pick deployment candidate (likely PQ block=16 sub=4 K_sub=256 or similar)
4. Sketch the MTL4 decode kernel
5. (silv approval required) encode full DS4 with chosen recipe → measure on hardware

## The smaller knob silv didn't explicitly ask about

There's a 6th knob worth flagging: **per-layer / per-kind codec selection.**
The aberration scan showed L0 gate IQ2_XXS rel_L2 was uniformly 0.353. But
the codec_quality_ab cross-layer sweep tested only gate kind on L0/L5/L10/L20/L42.
**Different layers and kinds may have different optimal codecs.**

For example: if `gate` is high-magnitude dense + `up` is low-magnitude sparse,
they could use different K. Per-layer-kind hybrid storage:

```
Layers 0-5 (early/hash):     IQ2_XXS (current, antirez-validated)
Layers 6-26 (mid, MoE bulk): PQ K_sub=256 block=16p sub=4 (0.25 byte/pair)
Layers 27-42 (late):         VQ K=256 (1.0 byte/pair, highest quality)
```

This adds a top-down "where does quality matter most" question on top of the
per-pair codec choice. Worth exploring if the aberration scan reveals layer-
specific patterns.

(Caveat: the cross-layer audit only tested IQ2 and showed it uniform across
L0-L42 gate; need to extend to polar/VQ to find layer-specific aberrations.)
