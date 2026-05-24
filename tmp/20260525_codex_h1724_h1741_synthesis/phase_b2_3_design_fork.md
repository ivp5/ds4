# Phase B-2.3 design fork — polar-down vs fp32-down

Per silv continuation directive. Phase B-2.3 (hot-path dispatch
substitution in `metal_graph_encode_layer_ffn_batch`) sits at an
architectural fork that needs silv's input before code lands.

## The substitution surface

The DS4 routed-MoE FFN MLP at line ~14375 calls:

```c
ds4_gpu_routed_moe_batch_tensor_mtl4(
    batch_routed_out,                  /* output */
    batch_routed_gate,                 /* intermediate */
    batch_routed_up,                   /* intermediate */
    batch_routed_mid,                  /* intermediate (gate*silu*up*weight) */
    batch_routed_down,                 /* intermediate */
    model->map, model->size,           /* mmap'd model file */
    ffn_gate_exps->abs_offset,         /* FP4 gate location */
    ffn_up_exps->abs_offset,           /* FP4 up location */
    ffn_down_exps->abs_offset,         /* Q2_K down location */
    IQ2_XXS, Q2_K,                     /* element types */
    gate_expert_bytes, gate_row_bytes, /* gate layout */
    down_expert_bytes, down_row_bytes, /* down layout */
    expert_in_dim, down_in_dim,        /* shapes */
    batch_router_selected,             /* per-token selected expert IDs */
    batch_router_weights,              /* per-token expert weights */
    DS4_N_EXPERT_USED, ...
    batch_ffn_norm,                    /* per-token input hidden */
    il, n_tokens, ...);
```

This is the call to replace. H1735 (my polar kernel, shipped 4b59348 +
validated on real data at d74a7d5/278db99) computes the equivalent
gate*silu*up*route_weight*down in ONE dispatch — BUT only the gate+up
side reads polar data; the down side expects fp32 slices.

## The down-projection problem

**H1735 reads `down` as `device const float*`** — pre-dequantized fp32
slices, shape [route_pairs, down_rows, act_rows]. For per-token DS4
inference, the natural mapping is:

- `mag/phase/levels` for gate + up → from polar pool (mmap'd PLR2 files)
- `down` slices → ??? 

Three options:

### Option A — fp32-down pre-cache at session-open

Decode polar down → fp32, hold all `down_in × routed_out × n_experts`
floats resident per layer. For DS4 V4 IQ2_XXS:
- Per layer: 4096 × 2048 × 256 × 4 bytes = **8.6 GB**
- 43 layers: **370 GB** — NOT VIABLE

### Option B — fp32-down on-the-fly per dispatch

GPU kernel decodes the polar-down for ONLY the routed experts per
token, on each FFN call. Adds a polar→fp32 prologue kernel before
H1735. Cost: ~ small ms per layer. Complexity: new kernel + dispatch
chain.

### Option C — mixed: polar-gate-up + FP4-down

Keep the existing FP4 down dispatch; substitute only gate+up via
polar. The H1735 kernel already does gate+up+down in one shot, so we
can't use it directly — would need a new kernel (call it H1735-half)
that takes polar gate+up + fp32-pre-decoded-empty-down. Reverts to
H1733 (which the codex shipped pre-1735) semantically.

### Option D — extend H1735 to read polar-down natively

New MSL kernel (call it H1735-full-polar) that takes mag/phase/levels
for all THREE of gate/up/down. The down dot loop is similar shape to
gate/up but with different (in_dim, out_dim). Maybe 50 extra MSL
lines.

## The architectural choice

```
Option   Memory   Compute    MSL complexity   Storage
A        +370GB   none       same             same
B        +0       +1 kernel  +medium          same
C        +0       same       +half-kernel     same
D        +0       same       +50 MSL lines    +5GB (need polar-down)
```

Wait — option D already has the polar-down: my MLX encoder shipped
gate+up+DOWN at p8_m4 and p16_m4 (the full_p16m4 corpus is 14 GB,
includes down). The "+5GB" is wrong; it's ALREADY there.

So Option D is the right choice:
- Polar pool ALREADY has down (no extra encoding)
- Need ONE NEW MSL kernel that reads polar-down (~50 lines)
- Replace the dispatcher call in the FFN encode

**Recommended path: Option D.**

## What's needed for Option D

1. **NEW MSL kernel** `polar_gate_up_down_full` (extends H1735):
   - Takes mag/phase/levels for gate, up, down separately (or in one
     packed format)
   - Reads `down_mag`, `down_phase`, `down_levels` from new args
   - Same threadgroup structure as H1735
   - Returns same output

2. **NEW C entry point** `ds4_gpu_mtl4_polar_full_dispatch(...)`:
   - Takes pool + (layer, n_tokens, batch_router_selected,
     batch_router_weights, batch_ffn_norm)
   - Looks up gate/up/down PLR2 views from pool
   - Builds gate_code/up_code/route_weight arrays from routing state
   - Hidden buffer = batch_ffn_norm reinterpret
   - Allocates output buffer
   - Dispatches the new kernel
   - Copies output back to batch_routed_out

3. **Hot-path gate** in `metal_graph_encode_layer_ffn_batch`:
   ```c
   if (polar_layer_enabled[il] && pool_has_full_gud(il)) {
       ok = ds4_gpu_mtl4_polar_full_dispatch(...);
       if (!ok && DS4_POLAR_FALLBACK_FP4) {
           /* fallback to existing FP4 path */
       }
   } else {
       /* existing FP4 routed_moe_batch_tensor_mtl4 dispatch */
   }
   ```

4. **First-call bit-equivalence check** (optional but recommended):
   - On first call per layer, ALSO run FP4 path
   - Compare batch_routed_out vs polar output
   - Refuse substitution if relative error > 1e-3
   - Cache the decision per (layer, expert_signature)

## Estimated scope

- NEW MSL kernel: ~150 lines (mirror H1735 + add down polar decode loop)
- NEW C dispatcher: ~250 lines (mirror real-canary structure + route extraction)
- Hot-path gate: ~30 lines
- Bit-equivalence checker: ~80 lines
- Total: ~500 lines, 1 focused multi-hour turn

## What silv needs to decide

1. **Option D vs C**: I recommend D (extends to full polar). silv may
   prefer C (keep FP4 down) if there's a reason to preserve the down
   compute path (e.g., for ICB recording, for less risk on first
   landing).

2. **Validation threshold**: 1e-3 relative is the canary tolerance.
   On real DS4 V4 weights at p16_m4 we expect rel_L2 = 0.205 codec
   noise — vastly above 1e-3. So per-layer rel-equivalence between
   polar and FP4 will be ~0.20, not ~0.001. The "validation" needs to
   be a different metric — perhaps "AIME hold-rate matches FP4 within
   N%" rather than per-call bit-equivalence.

3. **Per-token vs per-layer enable**: current env-var
   `DS4_POLAR_LAYERS` is per-layer. The H1741 pressure-aware routing
   would extend this to per-token (low-pressure = polar, high-pressure
   = FP4). silv may want per-token enable from day one or as a future
   extension.

4. **Bit-equivalence check infrastructure**: silv decides whether the
   first-call comparison runs at startup (slows session-open) or per
   inference (runtime cost).

## Recommended Phase B-2.3 sequence

1. **B-2.3a**: ship the NEW MSL kernel as a canary entry (similar to
   H1735 canary). Verify it produces sensible output on synthetic + real
   data. ~200 lines. Low risk.

2. **B-2.3b**: ship `ds4_gpu_mtl4_polar_full_dispatch` as a CLI canary
   entry that takes a model + polar dir + prompt and runs ONE layer in
   isolation. Compare numeric output to FP4 path on same input. ~250
   lines. Medium risk.

3. **B-2.3c**: integrate into hot-path with env-var gate. Ship as
   opt-in with fallback. ~80 lines. Medium-high risk (hot-path).

4. **B-2.3d**: AIME hold-rate measurement at polar_layers=ALL vs
   polar_layers=NONE. Confirms inference correctness end-to-end.
   silv's runtime step.

## Cumulative session work since session 22

This memo represents ~30 commits' worth of substrate establishment now
ready to consume. The polar codec is parameterized end-to-end, full
p16_m4 corpus is on disk (14 GB at `tmp/polar_full_p16m4/`), H1735
GPU kernel is validated on real data at fp32 noise floor, and Phase
B-2.1 instrumentation prints diagnostic when env vars are set. The
ONE remaining gap is Phase B-2.3 (this design fork).

Direction request: confirm Option D vs C, then I'll ship B-2.3a in
the next focused turn.
