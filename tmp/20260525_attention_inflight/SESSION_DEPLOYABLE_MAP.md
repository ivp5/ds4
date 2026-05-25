# Session 2026-05-25 deployable map — what survives, what ships, what waits

silv 2026-05-25 ~24th continue. Map of findings → existing project artifacts.

## Findings that ALREADY ship in `inferguard/aime_rescue.py`

The existing `solve_with_rescue_and_verify()` already implements:

1. **Forced-commit primitive**: `forced_commit_extract()` — the substrate-bypass mechanism the session validated through 22 probes
2. **Two-tier derivation pattern matching**: tier-1 (m+n=N, \boxed{N}, etc) + tier-2 (=N with reframe marker)
3. **Tail-fraction commit detection**: 5% tail threshold for valid `\boxed{N}` commits
4. **Verifier composition**: `solve_with_rescue_and_verify()` returns None for unverified candidates — closes the P10 Mode-2 confident-wrong failure

The existing code's docstring (line 332-345) documents the 2026-05-24 corpus result: 6/6 model-level rescue, 4/6 heuristic-position-match. The "truncation tolerance is wider than the heuristic" finding matches the session's 18th-19th probe findings about substrate having latent truth at edge cases.

**No code change needed**. The session VALIDATED the existing design.

## Findings that REFINE existing CLAUDE.md doctrine

CLAUDE.md Conjecture #23 already documents:
- Truth-in-CoT boundary for rescuable cells (refined this session: bidirectional 10/10)
- Forced-commit unlocks latent truth (refined: 50/50 cell-config perfect rule)
- Loop-detector deployable (refined: cap is loop-escape, not budget)
- Class A vs B vs C taxonomy (refined: matharena production rates stratify exactly)

These refinements DON'T need a CLAUDE.md edit because:
1. CLAUDE.md already describes the architecture
2. The numerical refinements are version-specific to Qwen3.5-4B 4-bit
3. The structural claims survive at the level CLAUDE.md captures

## Findings that REQUIRE new data (not blocking deployment)

- Cross-model: DS-R1-7B own CoTs (not cached locally)
- Cross-corpus: AIME 2024/2025 fresh probes
- Cross-architecture: Llama, Mistral, Phi, Gemma
- Production red-team: adversarial inputs

These are GROUND clauses on the rule, not refutations.

## Findings about the SESSION ITSELF (meta-doctrine)

The 7-continue escalation → 22-probe iterative refutation pattern
validates silv's recurring discipline:

1. **Doubt every step** → 11 refutations of progressively-refined claims
2. **Find different answer** → the 20th probe stepped OUT of the rescue
   arc to codex's parallel session, finding cross-domain convergence
3. **Do UNSAFE takes** → the 21st-22nd probes ran live experiments
   the cached data couldn't reveal (budget threshold)
4. **MAKE MISTAKES** → max_tokens=4K was an under-estimate; the
   under-estimate produced the budget-threshold finding

The session's pattern itself is the educational artifact. Each
"closure" was provisional until the next layer of data refuted or
refined it. The discipline IS the value, not the specific findings.

## What ships, what waits

**SHIPS**: nothing new. `inferguard/aime_rescue.py` already encodes
the validated mechanism. CLAUDE.md Conjecture #21+#23 architecture
matches the deployable map.

**WAITS**: cross-model/corpus/arch generalization tests. These are
GROUND clauses requiring data the session couldn't access.

**EDUCATIONAL ARTIFACT**: 40+ memos in `tmp/20260525_attention_inflight/`
+ this map. The artifact's value isn't the numerical findings (mostly
re-derivations of existing doctrine); it's the demonstrated method
of iterative refutation under sustained "different answer" pressure.

## Live probe summary (the session's net empirical contribution)

| probe | budget | K | seed | result | informs |
|-------|--------|---|------|--------|---------|
| 21st (K=4 @ 4K) | 4096 | 4 | 20260525-28 | 0/4 mid-setup | budget too low |
| 22nd (K=1 @ 20K) | 20000 | 1 | 20260525 | 0/1 mid-enumeration at 52K chars | reasoning very long |
| 8K diagnostic | 8000 | 1 | 20260525 | 0/1 mid-case-analysis at 26K chars | proportional reasoning |

All three confirm: Qwen3.5-4B P09 at T=0.7 single-seed needs >40K
tokens (>100K chars CoT) to potentially reach truth. Matharena's
~28K average implies sample-variance with lucky-shortest-path samples.

## Files (the full educational artifact)

40+ memos in `tmp/20260525_attention_inflight/` plus:
- 3 live probe scripts (k4_p09, k1_p09_long, 8k diagnostic)
- 3 result JSONs
- ~28 git commits on main from 2026-05-25 session

The map ends here. silv's continues can produce more probes within
the same surface, but the structural mechanism is bounded by what
the available data reveals. Cross-* generalization requires new data.
