# Hadamard-16 batched FP16 kernel — task #643 shipped

silv 2026-05-27: "start on #643"

## What shipped

The MTL4-Hadamard-FP16-batched primitive that unblocks the basis-aware
GGUF rewrite. Codex H1953 / H1955 found that DCT-16 / Hadamard-16
basis transforms applied before adjacent-pair VQB2 encoding improve
DS4 routed-FFN reconstruction quality by 1-3% at K=16 (the active
codec point). To deploy basis-aware quantized weights, the runtime
needs to apply the same orthogonal transform to the matmul INPUT
before the (pre-transformed) weights are consumed; orthogonal
transforms commute through inner products:

  `<w, x> = <H_normalized w, H_normalized x>`

where `H_16` is the 16-point Walsh-Hadamard matrix and
`H_normalized = H_16 / sqrt(16) = H_16 * 0.25`.

## Files (~280 net lines)

- `metal/hadamard.metal` (89 lines): `kernel_hadamard16_fp16_batched`.
  Each threadgroup processes one 16-element block via a 4-stage
  butterfly in threadgroup memory. Threadgroup = (16, 1, 1); grid =
  (n_rows, blocks_per_row, 1). Self-inverse property: H × H = I after
  the 1/sqrt(16) normalization, so the same kernel implements both
  forward (encode-side weights) and inverse (decode-side activations).

- `ds4_metal.m` (+58 lines): args struct, pipeline registration
  (soft-fail when source absent so older binaries still link), encoder
  helper `ds4_gpu_encode_hadamard16_fp16_batched`, public API
  `ds4_gpu_hadamard16_fp16_batched_tensor`, source list entry.

- `ds4_gpu.h` (+20 lines): public API declaration with full doc.

- `tmp/20260527_hadamard/smoke.c` (~220 lines): 3-test correctness
  smoke. Built as standalone C linked against ds4 .o files.

## Verification — all 3 tests PASS

```
=== test 1: H_16(e_0) should be [0.25, 0.25, ..., 0.25] ===
  PASS: all 16 cells = 0.25 (within 1e-3)

=== test 2: H(H(x)) = x within FP16 precision ===
  n_halves=512, max|err|=3.8624e-03, rel_L2=7.9835e-04
  PASS: rel_L2 within FP16 noise floor

=== test 3: GPU H(x) matches CPU reference ===
  n_halves=128, max|err|=3.3417e-03 (FP16 precision floor ~5e-3)
  PASS: GPU matches CPU reference within FP16 precision

ALL PASS
```

The CPU reference uses the same 4-stage butterfly with 1/sqrt(16)
normalization. Max relative error against CPU reference: 3.3e-3, well
under FP16's ~5e-3 precision floor.

## What this unblocks

Two downstream paths now have a runtime decode primitive available:

1. **Basis-aware VQB3 packet format** (codex H1953 next-step #4):
   store a transform id in VQB3 metadata so generated packets are
   executable without filename or sidecar assumptions. The decoder
   applies the matching inverse Hadamard at runtime.

2. **Pair-AVG GGUF rewrite + simdgroup mat-mat composite** (#631
   follow-up): pre-transform the pair-AVG'd FP16 hot-store tiles into
   Hadamard basis at encode time, then apply the inverse transform to
   activations before the simdgroup mat-mat dispatch. This unlocks
   the 1-3% reconstruction-quality gain on top of the simdgroup-fused
   gate+up+SwiGLU path that landed in 327e937.

## Engineering bridge — what's NOT done in this turn

The kernel + smoke is the BRIDGE. The actual deployment requires:

- Encode-time tool that applies Hadamard-16 to the FP16 weight tiles
  during GGUF rewrite (not yet written; codec_audit infra at
  `tmp/20260525_codec_vs_iq2_audit/` is the closest existing
  scaffolding).
- Hot-store loader that, when reading basis-aware tiles, marks them
  as "Hadamard-basis" so the runtime knows to transform inputs.
- Dispatch-site integration: where the simdgroup mat-mat wire calls
  `ds4_gpu_dispatch_fp16_simdgroup_pair_swiglu`, add an optional
  pre-pass that applies `ds4_gpu_hadamard16_fp16_batched_tensor` to
  the activation buffer when the hot-store layer is basis-aware.

Each of those is a follow-up multi-hour engineering item but does
not need new substrate research — the basis-transform improvement is
empirically established by codex H1953/H1955/H1956, and the runtime
primitive is now shipped.

## Run the smoke

```bash
cc -O3 -fobjc-arc -I. tmp/20260527_hadamard/smoke.c \
   ds4.o ds4_metal.o ds4_metal_vqb2_fp16.o ds4_inflight.o ds4_expert_table.o \
   ds4_polar_reader.o ds4_vqb1_reader.o ds4_vqb2_reader.o ds4_moe_route_log.o \
   ds4_neon_i8mm.o \
   -lm -pthread -framework Foundation -framework Metal \
   -o tmp/20260527_hadamard/smoke
./tmp/20260527_hadamard/smoke
```

Expected output: `ALL PASS` on the last line.
