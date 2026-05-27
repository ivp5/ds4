# OOM-speedup analysis — VQB2 GPU decode (silv 2026-05-28)

The stacked-speedup bench separated 4 axes and produced a surprising convergence
which revealed the actual bottleneck.

## Measurement (64 packets × 6/256 experts × 128 rows × 1024 pairs × K=16, 20 rounds)

  A. full   + per-pkt encoder: 86.6 ms/round  (baseline)
  B. select + per-pkt encoder: 96.1 ms/round  vs A = 0.90× (compute axis)
  C. select + batched encoder: 90.1 ms/round  vs B = 1.07× (encoder axis)
  D. select + fused 3D grid:   88.8 ms/round  vs C = 1.01× (dispatch axis)
  E. all rounds in 1 cmdbuf:   87.3 ms/round  vs D = 1.02× (sync axis)
  combined E vs A = 0.99×  E throughput = 2.3 GB/s output

## What this means — refining 10× accuracy

The naive expectation was 42× speedup at the compute axis (256 → 6 experts).
That's invisible. Reasons, in order of magnitude:

1. The full kernel and selected kernel SHARE per-code instruction cost.
   Bit extraction is ~8 instructions/code (load, branch, shift, mask, LUT,
   cast, store). At GPU clock ~1 GHz × 32 SIMD lanes ÷ 8 inst = ~4 G
   outputs/s. Bench shows 570 M codes/s × 4 bytes/code output = 2.3 GB/s.
   Match within factor of 3 (factor of 3 = SIMD divergence overhead).

2. The full vs selected kernel sees 42× less WORK but pays the same
   per-code instruction cost. Throughput-bound at the SAME rate; total
   time differs only because n_codes differs. The compute-reduction axis
   IS real, but at this granularity (one layer event), the absolute time
   is small enough that overhead noise swamps it.

3. Encoder + dispatch fusion (axes C, D, E) move at single-digit %
   because the bench is NOT sync-bound. The waitUntilCompleted overhead
   per command buffer is much less than the per-round kernel time.

## OOM-speedup target — vectorize per-thread work

At memory-bandwidth peak (400 GB/s) the decoder could produce ~100 G output
half2 values/sec = 50× the measured throughput. The path:

  - decode MULTIPLE codes per thread. K=4 packs 4 codes per byte; K=16
    packs 2 codes per byte. One uchar4 load = 16 K=4 codes or 8 K=16 codes.
  - emit half2[4] or half2[8] per thread via vectorized store
  - amortize the per-thread setup (linear computation, byte_off) across
    16-32× more output
  - SIMD-friendly: no per-lane branch (the (bw+shift>8) condition can be
    pre-computed at compile time per bit_width specialization)

Expected speedup: 8-16× at the kernel level. Combined with selected-expert
compute reduction (which DOES help at production scale where many layer
events stack), expect 50-100× total over the baseline full+per-pkt+sync
path.

## What got missed earlier

The earlier ICB Phase 8 bench (1.35-2.33× speedup) was correct AT ITS GRANULARITY
— there the comparison was direct-dispatch vs ICB-replay with multiple encoders
per round. ICB shifted encoder count from N to 1 per round, which was a real
~50% encoder cost reduction. The stacked bench then asks "do further axes
matter?" and the answer is "not at this granularity; the per-code instruction
cost is the wall." Without the stacked-axes isolation, the next optimization
candidate would be invisible.

## Production implications

For routed-MoE dispatch in the engine's per-token loop:
  - Each layer event = 64 packets × 6 selected × ~128K codes = ~50M codes
  - At 2.3 GB/s effective: ~85 ms per layer event
  - At 100 GB/s effective (after vectorization): ~2 ms per layer event
  - Across 43 layers per token: 85 → 2 ms × 43 = ~90 ms saved per token
  - Sub-1s token latency for 64 GB models becomes plausible without hot-store

This is the 10×-accuracy refinement that justifies "wire the GPU decode into
production" — but only AFTER the per-code vectorization lands. Wiring the
current 2.3 GB/s kernel into production gives ~3× over CPU dequant; wiring
after vectorization gives 50-100×.
