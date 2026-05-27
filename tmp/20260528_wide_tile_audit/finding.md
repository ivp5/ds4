# Wide-tile MoE matmul: structural mc[]-bound bug + speedup ladder

## What the audit found

My ne21=1 canaries (#735/#739/#740 et al.) reported all 3 widths
(n32, n64, n128) as bit-identical. That claim was insufficient —
the canaries dispatched only one threadgroup with nr1=1 in every
case, never exercising the rows that distinguish narrow from wide.

Routing R tokens to a single expert (htpe[0]=R, hids[0..R-1]={0..R-1})
gives nr1=R inside the kernel and exposes the wide-tile output write
loop end-to-end. Audit results (M=64, K=256, IQ2_XXS, single expert):

| R  | n32  | n64                     | n128                    |
|----|------|-------------------------|-------------------------|
| 1  | PASS | PASS                    | PASS                    |
| 32 | PASS | PASS                    | PASS                    |
| 33 | PASS | FAIL token-32 = 0       | FAIL token-32 = 0       |
| 64 | PASS | FAIL 2048/4096 = 50%    | FAIL 2048/4096 = 50%    |
| 128| PASS | FAIL 4096/8192 = 50%    | FAIL 6144/8192 = 75%    |

Boundary is exactly at token 32 in every failing case.

## Mechanism

`kernel_mul_mm_id` template body in `metal/moe.metal` declares:

    simdgroup_float8x8 mc[8];

This 8-tile accumulator covers a 32-col × 16-row region per simdgroup.
With 4 simdgroups laid in a 2×2 grid (sgitg%2 col-group × sgitg>>1
row-group), total output coverage is **64 cols × 32 rows**, regardless
of the `NR1` template parameter.

For NR1=32: kernel produces 64×32 output → exactly fills the tile.
For NR1=64: kernel produces 64×32 output → rows 32-63 read garbage
            (uninitialized shmem; Apple GPU returns zeros).
For NR1=128: same, rows 32-127 are garbage.

The output-write second loop reads `shmem + j*NR0` for j in [0, nr1).
For nr1=64, j=32 indexes past the 8 KB shmem allocation; the load
returns 0.0.

## Provenance

Antirez PR #264 added `_n64` and `_n128` template instantiations as
pure template-parameter variants. The PR did not extend `mc[]`, `mb[]`,
`temp_str` offset, or `shmem_bytes` to scale with NR1. My MTL4 ports
mirrored the source faithfully, so the bug transfers.

PR was merged without exercising neh1 > 32 per expert. Production
routing has not surfaced the bug — DS4's per-expert routing
distribution likely keeps neh1 in the safe band.

## Speedup ladder

### Speedup A: fix wide-tile correctness (~2-4× prefill)
Extend kernel structure to actually cover NR1 rows:
- mc[16] for NR1=64; mc[32] for NR1=128
- mb[4] for NR1=64; mb[8] for NR1=128
- Inner mma loop bound = 8 * NR1/32
- temp_str row offset = (NR1/2) * (sgitg>>1) * NR0
- shmem_bytes = NR0 * NR1 * sizeof(float) (16 KB for n64, 32 KB for n128)
- sb storage: either threadgroup_size = 256 (n64) / 512 (n128), OR
  unroll sb-write loop 2×/4× per thread

Currently the wide-tile speedup is **advertised but undelivered** —
the kernel does the work of NR1=32 even when dispatched as n64/n128
because mc[] limits actual output to 32 rows per threadgroup. Fixing
this delivers the true 2-4× prefill matmul speedup.

### Speedup B: persistent threadgroup MoE matmul (~10-100× on overhead)
Each dispatch costs ~10-100 µs of Metal command-buffer setup. For
small-batch decode, this dominates over the matmul itself. A
persistent threadgroup pattern — keep one kernel resident, feed
inputs via argument table updates — could eliminate per-token
dispatch overhead.

### Speedup C: ICB consolidation across layers (~4-10×)
DS4 already has Phase 1-7 ICB scaffolding. Phase 8+ extends to MoE
matmul dispatches, bundling 60+ layers' matmul commands into one
encode → submit cycle.

### Speedup D: dequantized expert hot-cache (~5-10× on small-batch)
For repeated decode token paths, the same experts are dequantized
many times. Caching dequantized FP16 weights for the top-K hot
experts avoids the dequant chain on each dispatch.

## Stacked theoretical speedup

A × B × C × D = 2 × 10 × 5 × 5 = **500× = 2.7 OOM** on overhead-
dominated small-batch paths. Each step is a separate ship; the audit
unblocks the path by surfacing the wide-tile correctness gap.

## Audit infrastructure shipped

- `ds4_mtl4_wide_tile_audit_run(pipeline, M, K, R, tile_n, label)` —
  reusable harness for any routed mm pipeline
- `ds4_gpu_mtl4_wide_tile_audit_iq2_xxs(M, K, R)` — runs all 3 widths
- `--wide-tile-audit M K R` CLI flag
- Per-token mismatch reporting + boundary-token detection
- Same pattern can extend to Q8_0 / Q4_K / Q2_K with minimal code

## Why my prior canaries missed this

`ne21=1` meant each canary dispatched a single threadgroup with nr1=1.
The output-write second loop ran once for j=0, reading shmem+0*NR0 =
shmem+0, which is INSIDE the always-written region (sgitg 0's 32-col
× 16-row band). Token 0 output always works. Tokens 32+ never tested.

The "bit-identical across widths" finding is a tautology when nr1=1:
of course they match — they all execute the same one-tile path with
NR1-related code dead-stripped by the dispatch grid.

## Reproducer

    ./ds4 --wide-tile-audit 64 256 64
    
    # Output:
    # wide-tile audit iq2_xxs_n32 ... mismatch=0/4096 PASS
    # wide-tile audit iq2_xxs_n64 ... mismatch=2048/4096 max_rel=1.00 first_bad_token=32
    # wide-tile audit iq2_xxs_n128 ... mismatch=2048/4096 max_rel=1.00 first_bad_token=32

