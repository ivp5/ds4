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

**IMPORTANT**: these three items now carry the explicit selector-design
constraints in `DEPLOYMENT_RULES.md` (post-codex-H1967..H2068 read).
Re-read that memo before shipping any of the three.

- **GGUF rewrite tool**: BLOCKED on selector design. Per Rule 3 must NOT
  be a full-tensor walk; the legal shape is route-resident sidecar
  keyed on `(route-domain, expert, row-block, organ)`. Per Rule 5 the
  encode-time validator must score signed source-top-k harm, not
  `|max_err|` or `|rel_L2|`.

- **Hot-store loader basis-aware marker.** ✅ SHIPPED 2026-05-27 as
  task #647. `ds4_hot_expert_store` now carries:
  - `gate_basis_hadamard16` / `up_basis_hadamard16` / `down_basis_hadamard16`
    (uint64_t per (layer, expert), bit i ⇔ row-block i in Hadamard basis)
  - `calibration_domain_id` (Rule 6 gate)
  Accessors `ds4_hot_get_{gate,up,down}_basis_hadamard16()` enforce
  Rule 6 inside the getter: when domain_id == 0 they return 0
  regardless of the underlying bitmask. Setter
  `ds4_hot_mark_basis_hadamard16()` OR's bits into the bitmask;
  `ds4_hot_store_set_calibration_domain()` flips the gate.
  Smoke test at `tmp/20260527_hadamard/basis_marker_smoke.c` (28/28).

- **Dispatch-site pre-pass integration**: in
  `ds4_gpu_dispatch_fp16_simdgroup_pair_swiglu` (the #631 wire),
  conditionally call `ds4_gpu_encode_hadamard16_fp16_batched` on
  the activation buffer when the layer is basis-transformed.
  Now UNBLOCKED by the marker above — the dispatch site can call
  `ds4_hot_get_gate_basis_hadamard16(store, layer, expert)` and
  apply the Hadamard transform when the relevant row-block bit is
  set. Per Rule 2 the check is PER-ORGAN, never uniform across
  gate/up/down. Per Rule 6 the getter already enforces the
  calibration-domain gate, so the dispatch site doesn't need to
  duplicate that check.

## Run the smokes

```bash
# Self-consistency + disk reproducibility
bash tmp/20260527_hadamard/encode_smoke.sh

# Python ↔ GPU cross-validation
bash tmp/20260527_hadamard/cross_smoke.sh
```

Both must end with `PASS` lines.
