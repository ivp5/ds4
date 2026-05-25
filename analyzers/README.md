# analyzers/ — codec encoders + quality A/B harnesses

Two codec families, each with encoder + quality-validation analyzers:

## Polar codec (`polar_*.py`)

Splits each complex pair into magnitude + phase, each 1 byte. PLR2
file format (see `../ds4_polar_reader.h` for spec).

| File | Purpose |
|------|---------|
| `polar_encode_safetensors.py` | Per-expert legacy encoder. Single-shot. |
| `polar_encode_bulk.py` | Vectorized bulk encoder + FP4 dequant helpers (imported by others). |
| `polar_encode_expert.py` | Per-expert wrapper used by the bulk path. |
| `polar_encode_mlx.py` | **Production encoder.** MLX-GPU shard-grouped. 1.86 OOM speedup vs numpy bulk. Parameterized for arbitrary (phase_levels, mag_levels). |
| `polar_down_decode_validator.py` | Validates polar-decoded down weights match FP4 source at rel_L2. |
| `polar_vs_fp4_kernel_output_ab.py` | A/B at canary geometry: polar vs FP4 source through kernel-output computation. |
| `polar_down_real_ffn_scale_ab.py` | A/B at real-FFN scale (act_rows=2048). Mirror of `vq2d_real_ffn_scale_ab.py`. |
| `polar_resolution_sweep.py` | Pareto sweep across (phase_levels, mag_levels). |
| `polar_codebook_ab.py` | Mean codebook vs percentile codebook A/B (refuted percentile by 0.88%). |
| `polar_phase_histogram.py` | Phase distribution analysis (found 21% FP4-induced non-uniformity). |
| `polar_learned_phase_codebook.py` | Learned codebook test (refuted naïve C amplification — gain only 1.11%). |

**Production recommendation**: `polar_encode_mlx.py --phase-levels 32 --mag-levels 8 ...`. See `tmp/20260525_codex_h1724_h1741_synthesis/codec_pareto_v2_m8m16.md` for the Pareto justification.

## VQ-2D codec (`vq2d_*.py`)

K-means in R² on (re, im) pairs. K=256 → 1 byte per pair. VQB1 file
format (see `../ds4_vqb1_reader.h` for spec).

| File | Purpose |
|------|---------|
| `vq2d_codec_explore.py` | Base K-means encoder + load helpers (imported by others). |
| `vq2d_encode_vqb1.py` | **CPU encoder.** Writes VQB1 files. 17s/tile, K=256. |
| `vq2d_encode_mlx_vqb1.py` | MLX-accelerated encoder. 13s/tile (24% over CPU). |
| `vq2d_real_ffn_scale_ab.py` | A/B at real-FFN scale (act_rows=2048). Mirror of `polar_down_real_ffn_scale_ab.py`. |
| `vq2d_row_coverage_audit.py` | Tests codebook generalization across rows (refuted scope-caveat — codebook fit on first 128 rows works on full 4096). |

**Production recommendation (when shipped)**: `vq2d_encode_mlx_vqb1.py --k 256 ...`. 6× lower codec error than polar p32_m8 at half the storage. See `tmp/20260525_codex_h1724_h1741_synthesis/vq2d_codec_breakthrough.md`.

## Other (`trim_experts_gguf.py`)

GGUF-format expert trim tool (silv's path-A trim work; not codec-related).

## How they relate

```
                  FP4 weights (safetensors)
                       │
              ┌────────┼────────┐
              ▼        ▼        ▼
     polar_encode_*   vq2d_encode_*   trim_experts_gguf
              │        │
              ▼        ▼
            PLR2     VQB1
              │        │
              └────┬───┘
                   │
                   ▼
     Validation A/B harnesses (*_real_ffn_scale_ab.py)
                   │
                   ▼
              Quality report
                   │
                   ▼
     C-side reader (ds4_polar_reader / ds4_vqb1_reader)
                   │
                   ▼
     MTL4 kernel (H1735 gate_up_down_partial /
                  gate_up_down_vq)
                   │
                   ▼
     Canary entry (--polar-real-canary /
                   --vq-real-canary)
```

## Quick-start commands

```bash
# Encode full DS4 V4 corpus at p32_m8 (~45 sec, ~14 GB):
python3 analyzers/polar_encode_mlx.py \
    --layers all --kinds gate,up,down \
    --phase-levels 32 --mag-levels 8 \
    --out-dir tmp/polar_full_p32m8

# Validate against FP4 source at real FFN scale:
python3 analyzers/polar_down_real_ffn_scale_ab.py \
    --polar-dir tmp/polar_full_p32m8 \
    --layers 0,22,42 --experts 0,100,255

# GPU canary on the encoded corpus:
DS4_POLAR_REAL_DOWN=1 ./ds4 --polar-real-canary \
    tmp/polar_full_p32m8 0 0 8 16

# VQ-2D K=256 encode (single layer reference, 160 MB):
python3 analyzers/vq2d_encode_mlx_vqb1.py \
    --layers 0 --kinds gate,up,down \
    --k 256 --out-dir tmp/vqb1_L0_k256

# VQ canary:
./ds4 --vq-real-canary tmp/vqb1_L0_k256 0 0 8 16
```

## Inheritance map (cross-session)

- All polar/VQ session memos: `tmp/20260525_codex_h1724_h1741_synthesis/`
- Session-end navigation: `tmp/20260525_codex_h1724_h1741_synthesis/SESSION_END_INDEX.md`
- Decision branches pending silv: `tmp/20260525_codex_h1724_h1741_synthesis/BRANCH_A_PREFLIGHT.md`
- Disk audit: `tmp/20260525_codex_h1724_h1741_synthesis/DISK_AUDIT.md`
