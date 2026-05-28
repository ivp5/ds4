# #796 + #771 design memo

silv 2026-05-28 evening: "796 and 771" — both unlocked. Engineer-roster work.

## The chain

The chain that gets us to a GGUF-free DS4 boot:

```
Phase 2b override-fill (shipped) — pack bytes land in t->storage.bytes
   ↓
Cycle 5 unified storage (shipped) — t->storage tracks dtype authoritatively
   ↓
#796 kernel wiring (THIS DESIGN) — kernels dispatch on tensor_effective_type
   ↓
ground rule lifted — source-exact pack tensors actually feed kernels
   ↓
#771 pack-direct loader — boot WITHOUT any GGUF, using config.json + packs
   ↓
Cycle 7 minimal-GGUF retirement complete
```

## H2160 source-exact inventory (~4.58 GB total)

From `H2160_nonrouted_policy.json`, 18 source_exact groups + 2 rowwise_int8:

| Group | Per-tensor bytes | Samples | dtype origin |
|-------|-----------------:|---------|--------------|
| attention_compressor:compressor_wgate | 8.39 MB | 1 | BF16 |
| attention_indexer:indexer_wgate | 2.10 MB | 1 | BF16 |
| attention_kv:wkv | 2.10 MB | 2/layer | BF16 |
| attention_o:wo_a | 33.55 MB | 2/layer | BF16 |
| attention_o:wo_b | 33.55 MB | 2/layer | BF16 |
| attention_q:wq_a | 4.19 MB | 2/layer | BF16 |
| attention_q:wq_b | 33.55 MB | 2/layer | BF16 |
| **embed_lm_head:embed** | **1059.06 MB** | 2 | BF16 |
| **embed_lm_head:lm_head** | **1059.06 MB** | 2 | BF16 |
| mtp:e_proj | 16.78 MB | 1 | BF16 |
| mtp:h_proj | 16.78 MB | 1 | BF16 (worst lossy at 0.135 rel-L2) |
| mtp:w1 / w2 / w3 | 8.39 MB | 1 each | BF16 |
| router:gate_weight | 2.10 MB | 3/layer | F32 (identity-fill safe) |
| shared_experts:w1 / w2 / w3 | 8.39 MB | 2 each | BF16 |

**Key observation**: 95%+ of source-exact bytes are **BF16** weights. FP8 paired (E4M3 + E8M0) is rare in this corpus. Router is F32 (identity-fill, no kernel work needed).

**The big-bytes targets**: embed + lm_head (~2 GB combined) and per-layer Q/K/V/O matmuls.

## Kernels that need BF16 read paths

The dense matmul / matvec kernels in `ds4_metal.m`:

| Kernel | Reads | Use case | New variant needed |
|--------|-------|----------|-------------------|
| `mul_mm_f16_f32` | f16 W, f32 X | dense matmul (multi-token) | `mul_mm_bf16_f32` |
| `mul_mv_f16_f32` | f16 W, f32 X | dense matvec (single-token decode) | `mul_mv_bf16_f32` |
| `mul_mm_q8_0_f32` | Q8_0 W, f32 X | quantized dense matmul | (BF16 doesn't need quantized path) |
| `get_rows_f16` | f16 W, i32 idx | embedding lookup | `get_rows_bf16` |

**Net**: 3 new BF16-input variants needed. All structurally identical to the f16 versions except the read-and-convert at the inner loop:

```metal
// Existing f16 read:
const float w_val = (float)w_buf[idx];

// New BF16 read:
const ushort w_bits = w_buf[idx];                       // bf16 = ushort
const uint w_f32_bits = (uint)w_bits << 16u;            // upper 16 bits of f32
const float w_val = as_type<float>(w_f32_bits);
```

That's a 2-line conversion per weight read. Cost: 1 shift + 1 reinterpret. Negligible.

## Dispatch wiring at the C side

At the call site for each affected kernel:

```c
const uint32_t etype = tensor_effective_type(weight_tensor);
switch (etype) {
case DS4_TENSOR_F16:
    ds4_gpu_mul_mm_f16_f32(...);
    break;
case DS4_TENSOR_BF16:
    ds4_gpu_mul_mm_bf16_f32(...);
    break;
case DS4_TENSOR_Q8_0:
    ds4_gpu_mul_mm_q8_0_f32(...);
    break;
default:
    /* unsupported dtype — fail loudly */
    ds4_die("kernel needs BF16 path for tensor X");
}
```

## Scope estimate

- **3 new Metal kernels**: ~80 lines MSL each = ~240 lines
- **3 new MTL4 ports**: ~120 lines each = ~360 lines
- **Dispatch wiring**: ~50 lines at ~10 call sites = ~500 lines
- **Canary**: BF16 vs F16-converted reference equality test, ~150 lines

**Total: ~1250 lines, 2-3 sessions of work.**

## Incremental shipping plan

Rather than 1250 lines in one shot, ship in 4 increments:

### Increment 1 (THIS TURN): one BF16 kernel + one call-site
- Pick the highest-impact path: `mul_mm_f16_f32` → `mul_mm_bf16_f32` (used by attn.q/k/v/o + ffn)
- Add the MSL kernel
- Wire ONE attention call site to dispatch on tensor_effective_type
- Add a canary that compares BF16 path output vs F16-converted reference
- Source-exact gate STAYS closed (this is the scaffold)

### Increment 2 (next session): wire all attention matmul call sites
- Q/K/V/O paths all use mul_mm_f16_f32 → switch all to dispatch
- Per-layer × 43 layers
- Verify canary still passes

### Increment 3: wire embed/lm_head + shared experts + MTP
- get_rows_bf16 for embed/lm_head
- mul_mm_bf16_f32 for shared experts and MTP

### Increment 4: lift the ground rule
- Override-fill loop: stop skipping source-exact
- tensor_dtype_can_substitute: return 1 when storage.dtype != t->type
- End-to-end test: load nonrouted pack with BF16 source-exact tensors, run inference, compare against minimal-GGUF lossy-converted baseline

## #771 dependency analysis

The minimal GGUF currently provides:
1. **Tokenizer**: vocab/merges/BOS/EOS tokens in `kv` section
2. **Structural metadata**: 38 KV keys (n_layers, n_heads, n_kv_heads, n_embd, rope_freq_base, etc.)
3. **Routed-FFN tensors**: lossy-converted Q8_0/F16 (replaced by VQB2 pack via PATH_FUSED)
4. **Non-routed tensors**: lossy-converted (replaced by nonrouted pack once #796 lands)

Once #796 lands, the GGUF is just (1) tokenizer + (2) structural metadata + ~zero useful tensor data (the routed-FFN it carries is lossy and superseded by VQB2 pack; the non-routed it carries is lossy and superseded by nonrouted pack).

**#771 pack-direct loader design** (post-#796):
1. New CLI flag: `--config-json /path/to/config.json` — provides structural metadata
2. New CLI flag: `--tokenizer-json /path/to/tokenizer.json` — provides vocab/BOS/EOS
3. `--nonrouted-pack` (existing) — non-routed tensors
4. `--vqb2-pack` (existing) — routed-FFN tensors
5. `-m` (model GGUF) becomes OPTIONAL — when omitted, the above 4 flags drive boot

The HF DeepSeek-V4-Flash repo provides `config.json` + `tokenizer.json` in standard form. Mapping them to ds4's internal metadata is a one-time porting task.

**Scope estimate for #771 (post-#796)**: ~300 lines (config.json parser + tokenizer.json parser + boot-from-packs entry point).

## Verdict

- **#796 is the load-bearing prerequisite.** Without it, the nonrouted pack's source-exact tensors are unusable, so #771's boot-from-packs would still need a lossy minimal-GGUF — defeating the point.
- **Order: #796 increment 1 this turn → silv approval → #796 increments 2-4 → #771 pack-direct loader.**
- **Live-fire validation gate** (engineer-roster work from this session) requires either (a) silv permitting minimal-GGUF use for smoke testing, or (b) #771 fully landing. Without one of these, all this session's shipped code is correct-by-construction but unverified end-to-end.

## Open question for silv

Can I use `ds4v4_minimal.gguf` (9.0 GB, lossy diagnostic probe, NOT the IQ2_XXS 86.7 GB off-limits model) for smoke-testing this session's engineer-roster work (Cycles 1, 2, 3, 5, 6', 9)? It would validate that the rewrites haven't broken the existing lossy-baseline boot path before we add #796 + #771 complexity.

If yes: 5-token smoke against minimal-GGUF + VQB2 pack proves the path. If no: continue with structural changes and accept the unverified state.
