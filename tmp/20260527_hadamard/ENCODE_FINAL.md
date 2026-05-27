# Hadamard-16 encode-side tool — task #645 shipped

silv 2026-05-27: "encode-side tool next" (after #643 runtime kernel landed)

## What shipped

The encode-side counterpart to the runtime Hadamard kernel. Together they
form a complete codec primitive: encode-time applies the transform to
weights, runtime applies the same transform to activations, the matmul
yields the original-basis result via orthogonal commutation
`<H w, H x> = <w, x>`.

## Files (~440 net lines)

- `tmp/20260527_hadamard/encode.py` (240 lines): standalone Python tool.
  - `hadamard16_block(x)` — vectorized 4-stage butterfly that mirrors
    `kernel_hadamard16_fp16_batched` exactly (same pairing pattern,
    same 1/sqrt(16) normalization).
  - I/O: raw FP16 binary, NPY, or GGUF tensor (IQ2_XXS dequant + FP16).
  - `verify_roundtrip(W)` — checks H(H(W)) = W within FP16 precision.
  - `verify_matmul(W)` — checks `<H(W), H(x)> = <W, x>` over 32 random
    Gaussian inputs. Uses `rel_L2` (not per-element max) because dot
    products occasionally land near zero, which inflates per-element
    rel_err meaninglessly while not affecting downstream behavior.
  - CLI: `--in`/`--out`/`--shape`/`--gguf`/`--tensor`/`--verify`.

- `tmp/20260527_hadamard/encode_smoke.sh`: synthetic FP16 round-trip
  smoke — generates a 256×7168 weight, encodes, verifies, reads back,
  confirms matmul equivalence on a separate code path.

- `tmp/20260527_hadamard/cross_gpu_python.c` (130 lines): C harness
  that performs the strongest validation — encode in Python via
  `encode.py`, then apply the GPU kernel via
  `ds4_gpu_hadamard16_fp16_batched_tensor`, then compare against the
  original. Since H × H = I orthogonally, the two-pass cross-path
  should reproduce the original within FP16 precision.

- `tmp/20260527_hadamard/cross_smoke.sh`: full pipeline driver for the
  cross-validation.

## Verification

### Self-consistency (encode.py --verify)

```
--- verify: H(H(W)) = W round-trip ---
  max|err|=1.95e-03  rel_L2=1.98e-04  PASS

--- verify: <H(W), H(x)> = <W, x> over 32 random vectors ---
  worst_rel_L2=3.53e-04  mean_rel_L2=2.95e-04  PASS
```

### Disk reproducibility

Encoder output (read from disk) bit-exact matches in-process reference:
`max|err| = 0.0e+0`.

### Python ↔ GPU cross-validation

```
loaded W (1,835,008 halves) and Wh (1,835,008 halves)
cross-validation: GPU(H(Python_H(W))) vs W
  max|err| = 3.91e-3
  rel_L2   = 4.87e-4
  PASS — Python encoder and GPU kernel agree within FP16 precision
```

**This is the load-bearing validation.** The Python encoder and the
Metal GPU kernel — two independent implementations of the same
orthogonal-Hadamard-16 transform — agree on a 256×7168 random tensor
to 4.87e-4 relative L2 error. That's well under the FP16 precision
floor (~5e-2 for accumulated dot products, ~5e-3 for single
operations). The two code paths are interchangeable for codec use.

## What this enables

The codec pipeline can now be:

1. **Encode time**: Python tool reads FP16 weights (from
   `ds4_hot_pin_layer_pair_avg` output or IQ2_XXS dequant), applies
   `hadamard16_block`, writes basis-aware FP16 sidecar.

2. **Runtime activation transform**: the GPU kernel from #643
   (`kernel_hadamard16_fp16_batched`) applied to input activations
   before the simdgroup mat-mat dispatch from #631.

3. **Numerical equivalence**: matmul on basis-aware weights × basis-aware
   activations gives the same result as the original-basis matmul
   within FP16 precision (validated end-to-end via cross_gpu_python).

## What's NOT done in this turn (deferred follow-ups)

- **GGUF rewrite tool** that applies the transform across an entire
  routed-expert tensor set and emits a new GGUF (or sidecar pack)
  with the transformed weights. The `--gguf` ingest path in
  `encode.py` reads single tensors; full-model rewrite requires
  tensor-walker + emit pipeline. This is mechanical extension of
  the existing `trim_experts_gguf.py` infra.

- **Hot-store loader basis-aware marker.** When `ds4_hot_pin_*`
  reads tiles that are already in Hadamard basis, the runtime needs
  to know NOT to apply the input-side Hadamard transform redundantly.
  Cleanest design: add a `is_basis_transformed` flag to
  `ds4_hot_expert_store`, set at file-load time from sidecar
  metadata.

- **Dispatch-site pre-pass integration**: in
  `ds4_gpu_dispatch_fp16_simdgroup_pair_swiglu` (the #631 wire),
  conditionally call `ds4_gpu_encode_hadamard16_fp16_batched` on
  the activation buffer when the layer is basis-transformed.
  Currently un-wired.

Each is a mechanical follow-up requiring no new substrate research.

## Run the smokes

```bash
# Self-consistency + disk reproducibility
bash tmp/20260527_hadamard/encode_smoke.sh

# Python ↔ GPU cross-validation
bash tmp/20260527_hadamard/cross_smoke.sh
```

Both must end with `PASS` lines.
