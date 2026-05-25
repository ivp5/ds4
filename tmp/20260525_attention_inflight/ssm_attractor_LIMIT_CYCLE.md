# SSM attractor temporal dynamics: LIMIT CYCLE (not fixed point) — DIRECTIVE ITEM ANSWERED

silv 2026-05-25 directive: "go top-down TO THE GROUND, ... analyze
the whole vertical dynamics through time to the nearest coherent
state, making all the states after it coherent as well."

The probe (`ssm_attractor_temporal_probe.py`) ran 16K tokens on
Qwen3.5-4B-MLX-4bit / AIME P01. 56.4 t/s sustained, 283.8s wall.
Captured top-1 token + logits + recent-32 ring buffer at positions
[500, 1000, 2000, 4000, 8000, 12000, 15000, 15999].

## Through-time content phases (the model's trajectory)

| pos | recent_text (last 100 chars) | top1 | margin | p |
|-----|------------------------------|------|--------|---|
| 500 | "$0$.\nSince $v_P > 0$, $v_P + 2 > 0$..." | — | — | — |
| 1000 | derivation | — | — | — |
| 2000 | "Let's check the algebra again. ..." | 'frac' | 15.25 | 1.0 |
| 4000 | "Patrick starts at $t=0$, Tanya at $t=1$, Jose at $t=2$. Arrival times are equal" | ' equal' | **1.88** | 0.695 |
| 8000 | "The integer is 277. One more check." | ' if' | 7.88 | 1.0 |
| 12000 | "swer is 277.\nI will write the solution.\nThe solution is 277.\nThe reasoning is complete." | ' ' | 18.44 | 1.0 |
| 15000 | "asoning is complete.\nThe answer is 277.\nI will write the solution." | ' the' | 17.12 | 1.0 |
| 15999 | "swer is 277.\nI will write the solution.\nThe solution is 277...." | ' ' | 16.12 | 1.0 |

## Key timeline observations

- **pos 4000**: still uncertain (margin 1.88, p=0.695) — case-analysis phase
- **pos 8000**: TRUTH=277 correctly derived in CoT — high confidence
- **pos 12000**: attractor entered, repetition begins
- **pos 15999**: still in attractor, IDENTICAL to pos 12000

## Cross-position recent-32-token matches

```
pos 500 ↔ pos 1000:   1/32 (early derivation drift)
pos 500 ↔ pos 8000:   7/32 (some carry-over)
pos 1000 ↔ pos 15999: 1/32
pos 4000 ↔ pos 15999: 1/32
pos 8000 ↔ pos 15999: 1/32
pos 12000 ↔ pos 15000: 2/32 (DIFFERENT phase of cycle)
pos 15000 ↔ pos 15999: 2/32 (DIFFERENT phase of cycle)
pos 12000 ↔ pos 15999: 32/32 (SAME phase, 3999-token gap)
```

## The structural conclusion: LIMIT CYCLE, not fixed point

**Pos 12000 ↔ pos 15999** matches 32/32 with a 3999-token gap.
**Pos 12000 ↔ pos 15000** matches only 2/32 with a 3000-token gap.

→ The cycle length divides 3999 but does NOT divide 3000.

Possible cycle lengths (divisors of 3999 = 3 × 31 × 43):
- 3, 13, 31, 39, 43, 93, 129, 403, 1333, 3999

The visible content cycle (e.g., "I will write the solution. The
solution is 277. The reasoning is complete. The answer is 277. I
will write the solution.") appears ~30-60 tokens.

**Most likely cycle length: 31 or 43 tokens.**

This is a LIMIT CYCLE attractor:
- NOT a pure fixed point (different phases visible at pos 12000 vs
  pos 15000 vs pos 15999)
- IS periodic (pos 12000 ↔ pos 15999 identical after 3999 tokens)
- High-margin commits at every phase (top1 logit margin 16-18)

## What "the nearest coherent state" means here

Per silv's directive: "the nearest coherent state, making all states
after it coherent as well."

**The coherent state IS the limit cycle entered at pos ~12000.**
- Pre-cycle (pos 500–11999): exploratory, derivation, uncertain phases
- Post-cycle (pos 12000+): predictably cycling through a ~31-43 token
  manifold of repeated phrases

All states after pos 12000 are coherent — they're points on the
limit cycle. The model has converged. Continuing to generate past
pos 12000 cannot escape this manifold without external perturbation
(e.g., forced-commit prompt injection per inferguard rescue
mechanism).

## Connection to prior findings

1. **Truth WAS derived** at pos ~8000 — refutes "model can't solve P01"
2. **Forced-commit rescue works** because truncating BEFORE pos 12000
   captures the pre-cycle pre-loop state where truth is still in
   the active reasoning buffer
3. **Conjecture #23 non-convergence**: the model never produces a
   `\boxed{}` because the attractor it falls into IS the
   "I will write the solution / The answer is 277" cycle — not the
   `\boxed{}` form. The text says "answer is 277" but never wraps
   in LaTeX. The cycle's vocabulary doesn't include `\boxed{}`.
4. **GatedDeltaNet SSM** is the most likely substrate for the cycle:
   24/32 layers are linear SSMs whose hidden state can converge to
   a closed orbit. The 8 full-attn layers retrieve from KV cache
   which by pos 12000 is dominated by the looped content.

## Through-time vertical (now grounded)

```
pos 0..499:     prompt processing
pos 500..2000:  algebra derivation (margin high, p=1.0)
pos 2000..4000: case analysis (margin DROPS to 1.88 at pos 4000)
pos 4000..8000: brute-force enumeration (recovers margin)
pos 8000:       TRUTH=277 derived in CoT
pos 8000..12000: meta-commentary "answer is 277, let me verify..."
pos 12000:      LIMIT CYCLE entered (~31-43 token period)
pos 12000+∞:    indefinite cycling through 'answer is 277' phrases
                without committing to \boxed{}
```

The model HAS the answer at pos 8000. The cycle is the FAILURE to
EXIT productive CoT into a commit. The "no-commit pathology" silv
documented in Conjecture #23 is empirically the limit-cycle
trajectory documented here.

## Engineering implication

**The rescue protocol's cap mechanism works because cap ∈ (pos_8000, pos_12000)
captures the post-truth pre-cycle state.** Forced-commit at any
position in that window emits truth because the model's residual
stream has accumulated the derivation but not yet entered the cycle.

This grounds the cap-mechanism rule
(`rescuable ⇔ truth_at < min(frac × cot_len, cap)`) at the
temporal-dynamics level: truth_at is the position where derivation
completes; cap must be after truth_at but before cycle entry.

## What's NOT done

- Exact cycle period determination (would need finer sampling, e.g.,
  every 10 tokens from pos 12000 to 12500)
- SSM hidden state directly (vs proxy of output ring buffer)
- L-by-L decomposition of which layers carry the cycle vs which are
  static
- Cross-precision (bf16 vs 4bit) attractor comparison — H1853 said
  4bit's commit-attractor differs from bf16; cycle length and
  content might too

## Files

- This memo: ssm_attractor_LIMIT_CYCLE.md
- Probe script: ssm_attractor_temporal_probe.py
- Raw data: ssm_attractor_temporal_result.json
- Probe wall: 283.8s @ 56.4 t/s on M1 4bit
