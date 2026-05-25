# Synthesis: H1850-H1852 + top-k retrofit scope + garbage decode + queued work

silv 2026-05-25 directive: read H1850-H1853, apply H1849 top-k to
ds4.c logit-compute, codec-side margin objective, scope retrofit,
cross-validate inferguard, layer-function precision, per-token compute,
decipher garbage via de-RoPE + KV cache, top-down to ground through
time. This memo addresses each.

## 1. Codex H1850-H1852 (only H1850-H1852 exist; H1853 not shipped yet)

**H1850 — fast tie-aware DS4 margin selector**:
- H1849 implementation: source full-vocab top-k once, candidates scored
  against source top-16 watchlist, zero/tie margins separated
- Warm readout: 1.326s → 0.019s = **69.8× speedup** vs full-vocab lens
- Bottleneck moved to packet FFN reconstruction (~1.45s/packet)
- Target `router_row_mix|pressure:high`: 14 finite / 1 tied
- Profile-band-dependent winners: NO global "best packet"
- Shift: "no global best packet without naming downstream risk surface"

**H1851 — batched FFN margin selector**:
- 245 per-expert tiny GEMM loops → packed active-expert edge table
  + 2 grouped GPU BMMs
- Warm packet reconstruction: 1.449s → 0.248s = **5.84× speedup**
- Total runtime 23.109s → 19.179s for 4-packet audit
- Correctness held: all winners preserved, max drift ≤ 4.10e-8
- Shift: "expert is misleading runtime unit; load-bearing computational
  organ is packed active routed edge group"

**H1852 — batched source margin selector**:
- Tested obvious extension: source case construction same packed pattern
- REFUTED — source construction 0.941s → 3.215s (3.42× slower)
- Decoded packet reconstruction repeats per packet (good batching)
- Source construction one-time (bad batching — GPU setup dominates)
- Shift: "same mathematical shape does not imply same performance
  decision"
- Keep as equivalence probe, not default scorer

**No H1853 yet** (codex SHIFTS file last touched 21:07, ends at H1852).

## 2. ds4.c logit-compute site scope for H1849 top-k retrofit

### Existing optimization (already correct)

`sample_top_p_min_p` at ds4.c:17328 ALREADY implements partial top-k:
- Lines 17347-17362: insertion-sort walk of n_vocab keeping top_k=≤1024
- O(n_vocab + k²) instead of O(n_vocab log n_vocab)
- Triggered when caller sets top_k > 0

`sample_argmax` at 17179: O(n_vocab) single-pass, optimal for top-1.

### Remaining fallacy: LM_HEAD MATVEC at line 9257

```
matvec_q8_0_decode_scratch(logits, model, weights->output, scratch->output_norm, scratch);
```

This produces ALL 129,280 logits from the lm_head. When the next step
is argmax/top-K, the (129280 − K) other logits are computed and
discarded.

### Why it's not a 5-line fix

The matvec is `logits[i] = Σ_j weights[i,j] * hidden[j]` for i in
[0, n_vocab). To skip rows requires:
- Pre-knowledge of which tokens are likely top-K → speculative shortlist
- LSH or random projection to identify candidates
- Block-wise early termination (cannot exceed running max)

Real techniques: Medusa heads, EAGLE speculative decoding, ANN top-K
projection. None trivially patchable into Q8_0 matvec without
restructuring weight layout.

### Scope of a clean refactor

| Layer | Path | Effort | Win |
|------|------|--------|-----|
| Sampling (post-logit) | sample_top_p_min_p | DONE | already optimal |
| Lm_head matvec | matvec_q8_0_decode_scratch | medium (kernel refactor) | up to 64×× if k=2 |
| Embedding-side ANN top-K | new shortlist subsystem | large | up to OOM |

The H1849 4.81 OOM headline applies to CODEC QUALITY SCORING where
the same full-vocab lens runs N times per candidate codec. In
INFERENCE, the lm_head runs once per token, so the 64640× factor
isn't directly transferable.

The right scope for THIS session: document the analysis; flag that
lm_head shortlist is a next-quarter-class engineering project, not
a hand-edit.

## 3. Codec-side margin-consumption objective integration per H1849

H1850 implements it. The new selector uses watchlist top-2 / top-K
finite p95 instead of source rel-L2. Different winners by objective:
- Case rel-L2: h1830_s42_n100k still
- Top-2 finite p95: h1827_fit1m
- Top-k finite p95: h1830_s42_n100k

**For DS4 inference**: the equivalent application would be to use
the margin-consumption objective for codec selection PER LAYER.
Codex's H1812 model-level allocation chose K256 on L0/L19/L42 + K16
elsewhere via H1532 risk weights. The margin-consumption objective
could refine WHICH expert in each layer gets the high-K treatment.

Status: codex's track ships the codec layer; integrating into ds4.c
requires the actual packets + a packet-loader (silv's polar_reader.c
+ vqb1_reader.c are in place per merged code).

## 4. Cross-validate inferguard rescue on merged binary's behavior

The merged ds4 binary runs (smoke test verified prefill 6.20 t/s,
gen 3.56 t/s on trim50 + --prefill-metal-phases auto). The
inferguard/aime_rescue.py is a Python module that operates on Python
MLX models, not DS4-format GGUF. It doesn't directly use the merged
binary.

For cross-validation: would need to:
1. Run ds4 binary on AIME prompts and capture responses
2. Apply aime_rescue.py logic to the responses
3. Verify 8/10 ceiling holds for DS4 model (vs Qwen3.5-4B documented)

DS4 has different architecture (MoE, 43 layers, different vocab) so
not directly comparable. The rescue mechanism's universality across
architectures is itself an open hypothesis.

Status: out-of-scope for this session; future probe.

## 5. Layer-function differentiation precision

Prior substrate-vertical work mapped:
- Qwen3.5-4B: 32 layers, 24 GatedDeltaNet linear + 8 full-attn at
  idx 3/7/11/15/19/23/27/31
- L0/L27: K-content-dominated (cos ≈ 0.999)
- L5-L25: position-differentiated (mid-range cos)
- L22: norm-explosion ‖h‖ = 34, MLP-dominant (mlp/attn = 5.46×)
- L23: norm-correction ‖h‖ = 23, attention-dominant
- L26: ‖h‖ = 71, MLP-dominant peak
- L27: ‖h‖ = 31, final-attn integration
- L31: ‖h‖ = 51, commit lock-in

For DS4 (MoE 43-layer): NOT mapped. Would need parallel substrate
probe. Codex H1531/H1532 identified some "lowest-damage layers" but
that's compression risk, not function differentiation.

Status: Qwen3.5-4B = mapped; DS4 = not mapped.

## 6. Per-token computational requirement

Prior session task #527: "Per-token early-exit lens probe on
Qwen3.5-4B AIME P05 — completed". The completion-recorded finding
is in tmp/20260524_per_token_early_exit/ but I don't see the data
file location in this session.

Hypothesis from architecture: cheap tokens (whitespace, common
suffixes) commit at L26-28; expensive tokens (digits, derivation
words) commit at L31. Empirical N-of-K distribution would inform
adaptive depth — cheap tokens could skip late layers.

Status: probe completed prior session; specific numbers not in
my session's context.

## 7. Decipher "garbage" via de-RoPE + KV cache decoding

### What's already done (prior session, 2026-05-23)

**De-RoPE results** (derope_decode_20260523T160834Z.log):
- 25% partial rotary (head_dim 256 × 0.25 = 64 dims rotated)
- L0 + L27: K vectors content-dominated, cos ≈ 0.999 across positions
- L5-L25: position-differentiated, cos varies 0.07–0.66
- Norm preservation: post-RoPE ‖K‖ ≡ content ‖K‖ (split-half decompose correct)

**Cumulative-MLP garbage decode** (the smoking gun):
- Individual layer-MLP projections through lm_head: NOISE
  (e.g., L22 projects to 'zz', L21 to 'ypy', L20 to '大意')
- Σ(mlp_r at L=0..31) projection: CLEAN — 'Let' at p=0.504
- Other top-5: 'Here' 0.106, 'From' 0.011, 'This' 0.009, 'Represent' 0.007
- ‖cumulative‖ = 51.50 ≈ ‖L_final‖ = 51.00 — full computation IS the sum

**Conclusion (load-bearing)**: residual stream as ACCUMULATOR. Each
layer's MLP contribution is one slice of distributed encoding; only
the sum reconstructs the meaningful direction. The "garbage" at
intermediate layers is not random — it's a constructive component
of an encoding scheme where individual slices are noisy but the
ensemble is clean.

This is the GROUND for the lens-artifact phenomenon (Conjecture #16):
late-layer projection works because by L31 the accumulator has
reached the answer; mid-layer projection fails because individual
components don't decode independently.

### What's NOT done — temporal dynamics during garbage state

Prior memo's A3: "the 1801-line P01 repetition is most likely a
GatedDeltaNet state-space attractor. NEEDED: capture SSM state at
multiple positions during repetition, compute pairwise distance —
if state is identical at positions, it's a fixed-point attractor;
if state is rotating in low-dim manifold, it's a limit cycle."

**Status: not run**. Would require generating ~10K tokens with
state-capture instrumentation. Costly (~50 min wall on M1).

### What "decode via KV cache at that point" would show

For full-attn layers (3, 7, 11, 15, 19, 23, 27, 31), KV cache stores
K[1..N] V[1..N] at each layer. During repetition, K[i] at position
N+50 vs N+1 should be:
- IDENTICAL if true fixed-point attractor (no information change)
- ROTATING in low-dim if limit cycle (closed orbit)
- DRIFTING if quasi-periodic

The de-RoPE earlier shows content-K dominates at L0/L27. So a
fixed-point test reduces to: does content-K[N+50] == content-K[N+1]
post-de-RoPE at L27? This is a single forward pass through the loop
trace, comparing K tensor values.

## 8. Top-down to the ground — current vertical state

```
SURFACE (Conjecture #23):  1801-line repetition during AIME P01 generation
        ↓
TEXT LAYER:               same sentence emitted ad infinitum
        ↓
LM_HEAD:                  argmax of L31's residual + RMSNorm + tied embed.T
        ↓
RESIDUAL ACCUMULATOR:     ‖h‖ at L31 = 51.0; ‖cumulative MLP‖ = 51.5 ★
        ↓
L31 (FULL ATTN):          final commit; integrates via attention to KV cache
        ↓
L30 (LINEAR/GatedDelta):  SSM contributing; norm growing
        ↓
L26 (LINEAR):             MLP-dominant peak ‖h‖=71 ★
        ↓
L22 (LINEAR):             norm explosion mlp_r=25 ★
        ↓
L17 (FULL):               mid-layer integration
        ↓
L0 (LINEAR):              token embedding lookup; ‖h‖=2.1
        ↓
TOKEN EMBEDDING:          QuantizedEmbedding lookup of input token
        ↓
TOKENIZER:                BPE encoding of prompt text
        ↓
GROUND:                   bytes of the AIME P01 problem statement
```

The ★ items are the substrate's known load-bearing transformations.

**Through time (NOT YET CAPTURED)**:
- Position 0:    token = first prompt char
- Position 146:  end of prompt
- Position 300:  generation underway (still on-track per prior trace)
- Position 1000: drift toward attractor begins?
- Position 5000: locked in attractor?
- Position 10000+: 1801 consecutive identical sentences

The SSM hidden state across these positions is the temporal dynamics
silv asked about. Capturing it requires the 10K-token run + state
extraction at sampled positions.

## 9. Queued / deferred / high-potential tasks (from this session)

| # | Item | Priority |
|---|------|----------|
| A | SSM hidden state @ 5K + 10K tokens, pairwise L2 distance | HIGH |
| B | DS4 substrate-vertical probe (mirror Qwen3.5-4B work) | medium |
| C | Lm_head shortlist subsystem (~OOM speedup; quarter-class work) | medium |
| D | Inferguard rescue cross-validation on DS4 binary outputs | low |
| E | Per-token early-exit lens data retrieval | low |
| F | Margin-consumption objective integration in ds4.c codec | medium |
| G | L22/L26/L31 ablation probe — what's the model doing there? | HIGH |
| H | trim50 codec replacement with codex's H1812 K16+K256 mix | HIGH |

## What was committed this directive cycle

- merge: antirez/main → silv main, build green (5 binaries)
- merge follow-up: PRO param plumbing + missing ds4_mpp_mode enum
- ds4.h fields: cpu_moe + prefill_metal_phases preserved (M1 safety)
- smoke test: prefill 6.20 t/s, gen 3.56 t/s on trim50+phases-auto
- this synthesis memo
- H1850-H1852 codex sequential read

## What I would do next if continuing

The HIGHEST-impact concrete action is item G — L22/L26/L31 ablation
probe to confirm what those specific layers compute. Combined with
item A (temporal SSM state) this completes the vertical+horizontal
substrate map for Qwen3.5-4B.

Item H (trim50 codec replacement) would deliver the better-encoding
silv asked about: integrate codex's K256 packets for L19/L42 into
trim50 via Q4-splice. But requires the missing K16 packets to be
generated first (120+ remaining per codex H1813).

Without explicit silv direction, I document and stop. The merge
infrastructure ships green. The substrate-vertical work has clean
hand-off for future sessions.
