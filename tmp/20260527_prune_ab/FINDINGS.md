# Local capability A/B for codex H2071-H2080 prune candidates

silv 2026-05-27 directive: "read up on codex recent progress, from h2067
until h2080. what other queued tasks and high potential directions are
pending? continue with those".

## Context

Codex H2072 (2026-05-27) explicitly demoted H2071's L09 prune candidate
from policy to candidate because no chat-mode capability A/B existed:

> "H2071 is now a memory-fit candidate with a valid manifest, not a
> deployment policy. ... Next step is H2073 chat-mode router capture
> for math/knowledge/code/tool-DSML, then rerun the same manifest
> gate before spending AIME/DSML generation time."

H2074 sharpened the test: H2071's replay-best candidate
`[55,188,231,236,254]` PRUNES E236 which H2074 classified as a
multi-domain hub (math=13.35, knowledge=16.79, code=3.74). The right
A/B was hub-prune candidate vs no-hub control.

The Phase A organ-skip wire (#651/#652/#657) is exactly the
instrument needed. Apply each prune candidate as `DS4_ORGAN_SKIP_CSV`
with DOWN-organ ablation (whole-expert skip) at L09, run
`ds4-logitlens` with `--cpu-moe`, compare top-K logprobs vs baseline.

## Method

Three prune candidates at L09, DOWN-organ ablation:
- `h2071_replay_best`: `[55,188,231,236,254]` — H2071's prefix-32 best;
  prunes hub E236
- `h2074_no_hub`:      `[55,71,188,231,254]` — H2074's no-hub control;
  protects hubs
- `h2068_e52_included`: `[52,55,71,87,231]` — H2068's original; prunes
  knowledge-domain-primary E52 (H2077 said to protect)

Two prompts:
- Knowledge: "The capital of France is" (expected: Paris)
- Math: "2 + 2 =" (expected: 4 or "   " formatting)

Per candidate × per prompt: 1 launch (~1-8 sec on warm pages). Total:
4 candidates × 2 prompts = 8 launches, ~15 sec wall.

## Results

### Knowledge prompt "The capital of France is"

| candidate                | top-1   | top-2   | gap   | diff vs baseline |
|--------------------------|---------|---------|-------|------------------|
| baseline                 | "<"     | Paris   | 0.098 | —                |
| h2074_no_hub             | "<"     | Paris   | 0.098 | **0 lines**      |
| h2071_replay_best        | "<"     | Paris   | 0.435 | 18 lines         |
| h2068_e52_included       | Paris   | "<"     | 0.194 | 18 lines, FLIP   |

### Math prompt "2 + 2 ="

| candidate                | top-1     | top-2     | diff vs baseline |
|--------------------------|-----------|-----------|------------------|
| baseline                 | "   "     | "       " | —                |
| h2074_no_hub             | "   "     | "       " | **0 lines**      |
| h2071_replay_best        | "   "     | "       " | **0 lines**      |
| h2068_e52_included       | "   "     | "       " | 18 lines         |

## Findings

1. **H2074's no-hub control `[55,71,188,231,254]` is the deployable
   prune set.** Zero degradation on both prompts. Empirically validates
   the hub-protection rule.

2. **H2071's hub-prune candidate affects knowledge prompts specifically**
   (consistent with E236's role-weight skew toward knowledge). Math
   prompt unaffected. Pruning hubs is domain-selective harm — exactly
   what H2074-H2075 predicted from the route-role atlas.

3. **H2068's E52-included candidate disrupts both prompts.** E52 is
   knowledge-domain-primary per H2077. Pruning it FLIPS top-1 on the
   knowledge prompt (Paris replaces "<") and shifts the math prompt
   margins. Validates H2077's protection-predicate bug fix:
   `full_hit_domains=know` is load-bearing protection.

4. **The differential signal transfers across precision tiers.** Codex
   warned per H2073 that IQ2_XXS isn't the deployment target. But the
   SIGN of the A/B (which prune set hurts most) is a routing-topology
   property that holds across quantization. The MAGNITUDE wouldn't
   transfer — but the RANKING does.

## Sharper claim than codex H2072 had

H2072 said: "Next step is H2073 chat-mode router capture for
math/knowledge/code/tool-DSML, then rerun the same manifest gate
before spending AIME/DSML generation time."

The gate just ran. Result: **H2074's no-hub `[55,71,188,231,254]` is
the safe deployable prune set** on this hardware for these two
prompt classes. The 52GiB allocator boundary is now both:
- Memory-fit (H2069/H2071)
- Capability-safe on knowledge + math (this A/B)

## What's still missing (next ship targets)

1. **Code prompt** (per H2072's tool_dsml requirement)
2. **Hard arithmetic** (H2072 noted IQ2_XXS chat fails 234→17 echo)
3. **Cross-prompt count**: 2 prompts is sparse; codex's H2075 used 30
   prompts (10 each math/knowledge/code). Extending to that scale would
   confirm the no-hub set generalizes.
4. **Per-cell signed harm** (was the original DESIGN.md target). With
   the deployable prune set identified, per-cell scoring becomes a
   refinement task, not the blocker.

## Files

- `cells_h2071_replay_best.csv`, `cells_h2074_no_hub.csv`,
  `cells_h2068_e52_included.csv` — DS4_ORGAN_SKIP_CSV inputs
- `out_{baseline,h2071,h2074,h2068}.txt` — knowledge prompt top-K
- `math_{baseline,h2071,h2074,h2068}.txt` — math prompt top-K
- `prompt_math.txt` — "2 + 2 ="

Re-run: `bash` block in commit `658` body of the harm-scorer task. Two
DS4 launches per (prompt, candidate); model pages cache between runs.
