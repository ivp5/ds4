# IQ2_XXS is NOT source quality — codec quality A/B vs FP4 source

silv 2026-05-25 correction: "IQ2_XXS is not source quantization - source is at fp8 - check it yourself"
+ "iq2_xxs was antirez chosen quantization, but the source is fp8/fp4, so iq_xxs loses data. do proper research on what preserves better - vq, polar etc"

## Source-quantization facts (per DeepSeek-V4-Flash config.json)

```json
"expert_dtype": "fp4",
"quantization_config": {
  "activation_scheme": "dynamic",
  "fmt": "e4m3",            // FP8 E4M3 for non-expert weights
  "quant_method": "fp8",
  "scale_fmt": "ue8m0",     // E8M0 unsigned scales
  "weight_block_size": [128, 128]
}
```

So:
- **Source = FP4 routed experts + FP8 E4M3 non-expert weights** (DeepSeek's release).
- **IQ2_XXS = antirez's downstream lossy ~2-bit re-quantization of FP4 source.**
- My codec arc (polar p32_m8, VQ K=256) also encodes FROM FP4 source.

All three are LOSSY CODECS of the FP4 source, on different storage/quality Pareto points.

## A/B measurement (L0 gate expert 0, first 128 rows)

Direct rel_L2 comparison: each codec's reconstruction vs the FP4 source dequant.

| Codec | bytes/pair | rel_L2 vs FP4 source | quality rank |
|-------|-----------:|---------------------:|--------------|
| **IQ2_XXS (antirez)** | 0.5156 | 0.3543 | WORST |
| **polar p32_m8** | 2.0000 | 0.1218 | middle |
| **VQ K=256** | 1.0000 | **0.0212** | **BEST** |

## Pareto interpretation

**VQ K=256 strictly dominates IQ2_XXS on quality** (16.71× lower error) at only **1.94× larger storage** (1.0 vs 0.5156 byte/pair).

For a 86 GB DS4 IQ2_XXS file with experts dominating ~75 GB of that, switching all routed experts to VQ K=256 would:
- ~2× storage: ~150 GB total file (still under 192 GB M1 Max unified memory)
- ~16.7× lower codec error per expert weight
- Existing `gate_up_down_vq` MTL4 kernel already shipped (B-2.3c canary chain)

This is the deployable shape my codec arc actually delivers — NOT a duplicate of IQ2_XXS, but a higher-quality alternative.

## Why my earlier "Chesterton fence" framing was wrong

I claimed `kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32` at metal/moe.metal:991 was "source quality" because it consumed raw IQ2_XXS without f32 staging. The "raw IQ2 in-kernel" pattern IS shipped in DS4 — but the data going in is already 16.7× more lossy than what VQ K=256 could provide.

The architectural pattern (compressed-as-compute-boundary, in-kernel decode) is the same across IQ2, polar, and VQ. The DIFFERENCE is the codec quality at the input. silv's catch: IQ2_XXS was antirez's STORAGE choice; quality wasn't audited against alternatives.

## What this changes for the codec arc

Pre-correction framing: "codec arc duplicates DS4 production with weaker numbers."
Post-correction framing: **"codec arc provides a 16.7× lower-error alternative to the antirez IQ2_XXS quantization at <2× storage cost."**

Branch A sub-decision becomes:
- **A.1 / A.2 (full-row codec corpora)**: now well-motivated. A.2 (VQ K=256 full-row) would deliver 16.7× lower codec error than current IQ2_XXS path.
- **A.4 (raw IQ2 hot-path)**: NOT a quality improvement over current production — it's the same lossy data.
- **The expert_table HOT tier** (task #455, fp16 pre-dequant for hot experts) would deliver "source-as-FP4-dequant" quality — equivalent to VQ at K→∞ but 4× storage cost.

## Storage Pareto (per-pair, for a single L0 gate expert 0)

```
bytes/pair  rel_L2   codec
---------   ------   -----
   0.5      0.35     IQ2_XXS  (antirez choice — worst quality)
   1.0      0.02     VQ K=256 (16.7× lower error at 1.94× storage)
   2.0      0.12     polar p32_m8
   4.0      0.00     fp32 reconstruction of FP4 source (literal)
```

The deployable lever for higher-quality DS4 inference: substitute IQ2_XXS expert weights with VQ K=256 encoded corpus, using the existing `gate_up_down_vq` kernel. Storage budget: 86 GB IQ2 file × (1.0 / 0.5156) ≈ 167 GB for full-row VQ corpus on disk. Still fits M1 Max unified memory.

## Files

- `dequant_iq2_xxs.py` — Python IQ2_XXS dequantizer (vectorized, self-test passes)
- `codec_quality_ab.py` — the A/B harness
- `iq2xxs_grid.npy` + `ksigns_iq2xs.npy` — lookup tables extracted from metal/moe.metal
- `result_20260525T040140Z.json` — final A/B numbers
- `out_*.log` — execution traces

## Caveats

- Single layer, single expert tested (L0 gate expert 0). Need cross-layer + cross-kind verification before universal claim.
- The polar/VQ corpora encode only 128 of 2048 rows (codec-arc scope-caveat from earlier session). The IQ2_XXS rel_L2 = 0.3543 reproduces identically on the full 2048 rows AND on the first 128 — suggesting the codec error is roughly homogeneous; should test other tensors to confirm.
- IQ2_XXS quality may vary across (layer, kind) — should expect worse on outliers, better on uniform tensors.

## Next severe-test conditions

REFUTE: a tensor where IQ2_XXS has LOWER rel_L2 than VQ K=256. (Would mean VQ doesn't universally dominate; might be true for very sparse / very low-magnitude tensors.)
REFUTE: VQ K=256 inference produces noticeably worse AIME hold-rate than IQ2_XXS even though codec error is 16.7× lower. (Would mean codec-error isn't the right metric for downstream task performance.)
CORROBORATE: cross-layer (L5/L10/L20/L42) cross-kind (up, down) all show VQ K=256 < IQ2_XXS rel_L2.
