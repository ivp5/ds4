# Mechanism head-to-head: attention damping vs Conjecture #23 forced-commit

silv 2026-05-27: "check which mechanism is more effective and efficient,
as the attention damping seems less intrusive then conjecture 23
forced commits."

## What each mechanism does

### Attention damping (`DS4_ATTN_SCALE_MULT`)
- Modifies flash-attn softmax scale (`1/sqrt(d) → 1/sqrt(d) × MULT`)
- Applied to EVERY token of EVERY layer (currently global)
- Changes WHICH tokens win softmax mass during attention
- Does NOT inject content — model's own distribution shifts
- Cost: ZERO additional forward passes (changes existing compute)
- Status: env var only; not auto-triggered

### Conjecture #23 forced-commit (`inferguard/aime_rescue.py`)
- After natural CoT terminates without `\boxed{N}`, appends a suffix:
  `"\n\nAfter careful analysis, the final answer is \\boxed{"`
- Runs ONE additional forward pass with that suffix
- Extracts the next ~16 tokens as the committed answer
- Cost: O(suffix_len + 16) ≈ ~50 tokens extra
- Status: deployed in inferguard; 8/10 on AIME 2026 I

## Intrusion comparison (silv's hypothesis check)

| dimension | attention damping | forced-commit | less intrusive? |
|-----------|------------------|---------------|-----------------|
| Adds prompt content | No | Yes (~30 chars suffix) | **damping** |
| Modifies internal computation | Yes (every layer) | No (model unchanged) | **forced-commit** |
| Per-token effect | Affects ALL outputs | Affects only post-suffix | **forced-commit** |
| Reversibility within run | Stateful (until env reset) | Per-call (independent) | **forced-commit** |
| Fabricates conclusion | No (model still chooses) | Partial (suffix steers) | **damping** |
| User-visible artifact | None (silent) | Suffix in trace | **damping** |

**silv's framing partially correct**: damping doesn't ADD content
(less prompt-injection-shape). But damping is MORE INVASIVE on the
computation (every token, every layer). The "less intrusive" axis
depends on what you're protecting:
- Protecting the **MODEL'S OWN OUTPUT** → forced-commit more intrusive
  (injects deciding suffix)
- Protecting the **MODEL'S COMPUTATION FIDELITY** → damping more
  intrusive (changes how every layer attends)

## Effectiveness: where each mechanism applies

### Attention damping
What it can fix:
- In-flight attention concentration / cache-lock loops (PROBE-LEVEL
  evidence — task #471 on Qwen3.5-4B; NOT confirmed on DS4 yet)
- Token-fragment vs full-token confidence rebalancing (NEW today —
  AIME P03 mult=2.5 caused "7" fragment to overtake "79")

What it CAN'T fix:
- Capability-limit cells (truth not in CoT — e.g. AIME P09 truth=29
  never derived in CoT regardless of temperature)
- Sequential reasoning errors (the wrong derivation isn't a
  confidence problem)
- Anchoring-bias in verifier roles (RLHF-trained priors override)

### Forced-commit
What it can fix:
- No-commit failures where truth IS in CoT (P01-P08 on AIME 2026 I,
  Qwen3.5-4B) — 7/8 success
- Reasoning-RL "uncertainty refusal" posture (model knows but won't commit)

What it CAN'T fix:
- Capability-limit cells where truth was never derived (P09 truth=29:
  0/15 rescues across 8 tested models)
- Loop-while-correct: if model is in a loop with wrong answer, forcing
  commit at \boxed{ extracts the wrong answer
- Pre-commit text-degradation (if CoT itself drifted)

## Efficiency: cost per cell

| metric | attention damping | forced-commit |
|--------|------------------|---------------|
| Additional forward passes | 0 (changes existing) | 1 |
| Compute cost | Same as baseline gen | +~50 tokens |
| Wall time delta | ≈ 0 (no extra compute) | +0.5-2s per cell |
| Memory delta | 0 | 0 |
| Code path complexity | trivial (env var set) | suffix construction + parse |

**Damping is cheaper at runtime** (zero added compute) IF it works.
Forced-commit adds a small linear cost per rescue.

## When the comparison breaks down

The two mechanisms target DIFFERENT FAILURE MODES:
- Damping targets **how attention concentrates** during generation
- Forced-commit targets **whether the model commits** at the answer position

Asking which is "more effective" is like asking whether a brake or a
steering wheel is more effective — they handle different failure
classes. The fair comparison: when BOTH could plausibly apply
(loop with truth-in-CoT), which works better?

### Empirical test: AIME P01 Qwen3.5-4B (truth=277, loops at default temp)

| mechanism | applied at | outcome | wall |
|-----------|-----------|---------|------|
| baseline (no intervention) | — | 1801 lines literal repetition, no commit | 50 min |
| forced-commit at truth-shape position | post-CoT | 277 extracted ✓ | ~13s |
| attention damping (untested on Qwen3.5-4B) | every token | UNKNOWN — needs probe |

Forced-commit measured. Attention damping on Qwen3.5-4B AIME P01 NOT
tested. Would need to port DS4_ATTN_SCALE_MULT-equivalent into the
mlx-lm forward path (modify the scaled_dot_product_attention call).
That port is ~30 lines of MLX code.

### Cross-domain test: DS4 capital-of-France loop on trim50 (today's data)

| mechanism | mult=1.0 baseline | mult=0.5 | mult=0.2 | mult=0.1 | mult=4.0 |
|-----------|------------------|----------|----------|----------|----------|
| trim50 substrate | LOOP | "You. You. You." | "??? ???" | ". . . ?" | (not tested) |
| full IQ2_XXS | "Paris" ✓ | (not tested) | (not tested) | (not tested) | "Paris" ✓ |

Attention damping DESTROYS the trim50-substrate output but the trim50
output was already loop-broken. On full IQ2_XXS, baseline doesn't
loop on this prompt — no rescue needed.

### Cross-prompt-class today (4 prompts × 3 temperatures, completed in 69s wall)

| prompt | mult=0.5 | mult=1.0 | mult=2.0 | top-1 change at 2.0 |
|--------|----------|----------|----------|---------------------|
| math_easy (5+3) | "8" -0.08 ✓ | "8" -0.05 ✓ | "8" -0.16 ✓ | none |
| know_rote (Japan capital) | "東京" ✓ | "東京" ✓ | "**" ✗ | **FLIPS to formatting** |
| code_python (def add) | "\\" | "\\" | " return" | flips to keyword |
| dsml_json ("age":) | "30" | "30" | "10" | **VALUE CHANGES** |

**Sharper temperature (2.0) destabilizes rote-knowledge and
structured-fill, but math arithmetic survives.** This is the first
cross-class measurement of attention-temperature sensitivity on DS4.

## Verdict

silv's hypothesis that damping is "less intrusive" depends on what
"intrusive" means:
- **Less prompt-injection-shape** ✓ damping doesn't add content
- **More computationally invasive** ✗ damping modifies every forward

For DEPLOYMENT effectiveness on TODAY'S workload (H2074 prune A/B,
DSML, AIME prefill):
- All candidates preserve top-1 → no rescue needed for either mechanism
- Attention damping NOT useful for prune A/B (nothing failing)
- Forced-commit NOT useful for prune A/B (no commit position to force)

For AIME GENERATION rescue specifically:
- Forced-commit is DEPLOYED, works on 7/8 P01-P08 cells, fast (~13s/cell)
- Attention damping is UNTESTED on this workload; would need MLX port

**Recommendation**: the head-to-head requires porting attention damping
to the mlx-lm path. Until that ports, forced-commit remains the
deployed rescue. The non-monotonicity finding (mult=4.0 cleaner than
mult=2.5) suggests attention damping has its own complexity profile;
"set temperature once" isn't a deployable policy — per-task
calibration is required.

## Next-turn experiments to land the comparison

1. **Port `DS4_ATTN_SCALE_MULT`-equivalent to mlx-lm Qwen3.5-4B**
   (~30 lines, class-patch `Qwen3NextAttention.__call__`)
2. **Run on AIME P01 loop case**: does damping at mult=2.0 or 4.0
   break the 1801-line repetition?
3. **If yes**: compare wall + correctness vs forced-commit (currently
   13s vs unknown)
4. **If no**: damping is empirically inferior on Qwen3.5-4B for this
   failure class

Until those land, the deployable rescue answer is: forced-commit
remains primary; attention damping is exploration-stage on DS4.
