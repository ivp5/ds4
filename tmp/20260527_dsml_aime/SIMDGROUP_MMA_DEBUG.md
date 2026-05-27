# Simdgroup MMA MTL4 port attempt — FAILED canary, deferred

silv 2026-05-27: continue MTL4 ports.

## Attempted

`kernel_dsv4_indexer_scores_tiled_f32` MTL4 port. Was meant to unlock
the simdgroup MMA pattern (`simdgroup_float8x8`, `simdgroup_load`,
`simdgroup_multiply_accumulate`, `simdgroup_store`) for future Q8_0
matvec / FlashAttention inner / mul_mm variant ports.

## Failure mode

Canary returns all zeros (with sporadic `-inf` for causal-masked
tail cells). Expected score = head_dim × n_head = 128 × n_head for
all-ones inputs. Got 0.0 across every cell at every dispatch size:

```
n_tokens=8  n_comp=32  n_head=64 → score[0,0]=0.00 (ref=8192.00) mismatch=256
n_tokens=8  n_comp=32  n_head=1  → score[0,0]=0.00 (ref=128.00)  mismatch=256
```

Pipeline COMPILED successfully — the failure is at runtime.

## Hypotheses (untested)

1. **Default origin mismatch**. The original kernel passes `0` as the
   third arg to `simdgroup_load`. MSL spec says origin is `ulong2`.
   Implicit conversion `0 → ulong2(0, 0)` may differ between Metal
   versions. Try explicit `ulong2(0, 0)`.

2. **Threadgroup memory pointer space**. `simdgroup_load(mq,
   threadgroup_ptr, ...)` should be a valid overload, but if MTL4's
   library compilation picks the device overload by mistake, the
   read would hit garbage.

3. **maxTotalThreadsPerThreadgroup mismatch**. I set 128 in the
   pipeline descriptor. The kernel needs exactly 128 threads (4
   simdgroups × 32 lanes) for the lane/sg arithmetic to work. If
   the runtime dispatched a different thread count, the per-thread
   acc0/acc1 indexing breaks. Verified my dispatch passes 128 to
   `threadsPerThreadgroup`.

4. **`make_filled_simdgroup_matrix<float, 8>(0.0f)` semantics**. The
   accumulator may not be initialized as expected — perhaps each
   thread's `mdot` is per-lane and the matrix-multiply semantics
   require explicit zero from all participating lanes.

5. **Library version**. The MTL4 library compiler may target a
   different MSL version than the classic library. The simdgroup MMA
   API surface may differ between versions (e.g. 3.0 vs 3.1).
   `MTL4LibraryDescriptor.options.languageVersion` is set to default;
   may need explicit setting.

## Debug-next steps

Order from cheapest to most expensive:
1. Add explicit `ulong2(0, 0)` to all `simdgroup_load` calls
2. Set MSL language version explicitly on `MTL4CompilerOptions`
3. Dump the dispatched threadgroup size at runtime — verify 128
4. Replace `make_filled_simdgroup_matrix` with explicit zero-write
   loop into a threadgroup buffer + simdgroup_load
5. Compare classic-MTL dispatch byte-by-byte against MTL4 dispatch

## What's deferred

- `kernel_dsv4_indexer_scores_tiled_f32` (this attempt)
- `kernel_dsv4_indexer_scores_tiled` (half-precision variant)
- `kernel_dsv4_indexer_scores_nax` (NAX variant)
- All Q8_0 / FlashAttention / mul_mm matmul ports — they depend on
  simdgroup MMA working first.

## Status

**Code reverted.** Build passes at the pre-attempt state. Coverage
remains 42/81 (51.9%). No broken canary shipped.

This is the first port-attempt failure in 36 ports across the session.
The simdgroup MMA pattern is the FIRST genuinely hard pattern that
requires investigation beyond template-mechanical porting.

## 2nd attempt: ulong2(0, 0) — REFUTED (turn N+1)

Applied hypothesis #1 (explicit `ulong2(0, 0)` origin on all simdgroup_load
and simdgroup_store). Same failure pattern: score[0,0] = 0.

Then progressively diagnosed:

**Step 1**: With `pos0=1024 ratio=1` (causal masking turned off):
- All 256 cells = 0 regardless of n_head. neg_inf_count = 0.

**Step 2**: Replaced MMA with constant `mdot = 42.0`, simdgroup_store:
- score[0, 0..15] = 42 (sg=0 and sg=1 both wrote row 0 correctly)
- score[1, 0] = 0 (row 1 missing)

**Step 3**: Replaced simdgroup_store with manual per-lane fill loop
(`if (sg==0) for (i=lane; i<TM*TN; i+=32) dot[i]=42;`):
- Same pattern: row 0 = 42, row 1 = 0.

**Step 4**: Bypassed dot[] entirely, direct write `acc0=(float)(token0*100+comp0)`:
- score[0, 0..15] = [0, 1, 2, ..., 15] (row 0 correct: token=0)
- score[1, 0] = 0 (row 1 missing: should be 100)

## Deeper finding — NOT a simdgroup_store / threadgroup_mem issue

The failure is at a MORE FUNDAMENTAL layer: **lanes 8-31 of every
simdgroup silently skip the per-thread write path**. Only lanes 0-7
(which map to `row0 = lane >> 3 = 0`, i.e., token=0) actually execute
the `*dst = acc0` store.

This is a **lane masking** symptom. Possible causes:
- MTL4 dispatch shape misinterpretation: maybe `threadsPerThreadgroup`
  with (128, 1, 1) is being clamped to (32, 1, 1) silently.
- A control-flow predicate higher in the kernel (the `c0 >=
  max_visible` early-return) executes lane-divergent and only lanes
  0-7 reach the main path.
- MSL `[[thread_index_in_simdgroup]]` returns a value different from
  what classic-MTL returns under MTL4 dispatch.

**Most actionable next debug**: instrument the kernel with an atomic
counter to verify HOW MANY threads enter the main path. If only 32
threads (not 128), the dispatch shape isn't reaching the runtime.

## Status (2nd attempt)

Code reverted again. Build clean. Coverage 42/81 (51.9%).

Conclusion from 2-turn investigation: the simdgroup MMA port blocks
on a deeper MTL4 dispatch-shape or lane-execution issue that requires
multi-turn debug (atomic instrumentation + lane-by-lane analysis +
possibly cross-reference against codex's MTL4 ML pipeline findings
which discussed dispatch semantics in H1779/H1780).

Recommend deferring all simdgroup-MMA / Q8_0 / FlashAttention / mul_mm
ports until this is resolved, OR moving to non-MMA ports (Q4_K
dispatch, generic copy/conversion variants, FA's pad/blk subkernels
that don't use MMA).
