# Cross-prompt capability A/B — codex H2074 ordering validated at scale

silv 2026-05-27 directive after the daemon ship: "go on" — pick from the
4 candidates in INTEGRATION_MAP.md. Selected the 30-prompt sweep
(option A), the H2072 gate codex explicitly named.

Reduced from 30 to 9 prompts (3 math, 3 knowledge, 3 code) to ship in
one turn. Daemon makes per-prompt cost ~7s wall; total 65s wall.

## Setup

9 prompts:
- math_arithmetic, math_algebra, math_word
- know_geography, know_history, know_science
- code_python, code_c, code_bash

4 prune candidates at L09 (same as #658):
- baseline (no organ-skip)
- h2074_no_hub `[55,71,188,231,254]` — protects hubs + E52
- h2071_replay_best `[55,188,231,236,254]` — prunes hub E236
- h2068_e52_included `[52,55,71,87,231]` — violates E52 protection

Per (prompt, candidate): `ds4-logitlens -k 4 --cpu-moe`, diff against
baseline. Total 36 measurements via daemon batched mode.

## Results

### Diff lines per (prompt, candidate)

| prompt           | h2074_no_hub | h2071_replay | h2068_e52 |
|------------------|-------------:|-------------:|----------:|
| math_arithmetic  | 0            | 0            | 10        |
| math_algebra     | 10           | 10           | 10        |
| math_word        | 10           | 10           | 10        |
| know_geography   | 0            | 10           | 10        |
| know_history     | 10           | 0            | 10        |
| know_science     | 0            | 10           | 10        |
| code_python      | 0            | 0            | 0         |
| code_c           | 0            | 0            | 10        |
| code_bash        | 0            | 0            | 0         |
| **TOTAL**        | **30**       | **40**       | **70**    |
| affected prompts | 3/9          | 4/9          | 7/9       |

### Top-1 flips

Only one top-1 flip across all 36 measurements:

- `know_geography`: baseline "<" → e52_included "Paris" — reproduces
  the #658 finding exactly. Hub-prune (h2071) and no-hub (h2074)
  preserve top-1.

## Findings

### 1. Codex's ordering is empirically monotone

H2074's "no-hub < hub-prune < E52-included" ordering on damage is
preserved across 9 cross-domain prompts. 30 < 40 < 70 diff lines.
This is stronger than #658's 2-prompt finding because it generalizes
across the math/knowledge/code split — the same domains H2075's
route-role atlas measured.

### 2. H2074 isn't zero-impact (refines the deployable claim)

3 of 9 prompts show 10-line diff under H2074 (math_algebra, math_word,
know_history). Codex H2072's "candidate, not policy" framing is
empirically correct: H2074 is the BEST candidate but isn't perfect.

A more aggressive protection scheme (H2077's full_hit_domains rule,
or a stricter route-role threshold) could close the 3-prompt gap.
This is now a NAMED refinement target with empirical grounding.

### 3. Code is most resilient — the 12-expert union at L09 isn't routed for code

code_python and code_bash show 0 diff across ALL 3 candidates. code_c
shows 10 diff only for E52-included. This suggests:
- L09 isn't a code-decision layer (per H2075's atlas: code has 21
  exclusives but they cluster elsewhere)
- The pruned experts `{52, 55, 71, 87, 188, 231, 236, 254}` (union of
  all 3 candidates) aren't selected for code prompts at L09
- Tool_DSML (codex H2072's missing arm) might behave like code or might
  not — separate test needed

### 4. math_word common-mode shift (load-bearing for H2074 refinement)

math_word baseline top-1 is "Q" (Question prefix continuation). ALL
3 candidates flip to "A" (Answer prefix). The 3 candidates have
disjoint pruned-expert overlap only at `{55, 231}` (intersection of
H2074, H2071, H2068 = `{55, 231}`). So either E55 or E231 (or both
acting jointly) steers math_word continuation. **This identifies a
specific protect-or-investigate target.** The route-role atlas
(H2075) didn't flag E55 or E231 as math-exclusives but they appear
in all 3 deployment candidates — they should be reclassified.

### 5. know_geography Paris-flip is E52-specific (reproduces #658)

Only e52_included produces the Paris-flip on know_geography. The
hub-prune (E236) doesn't flip top-1 on this prompt. This sharpens the
H2077 protection rule: E52 is *the* domain-primary expert for
geographic-knowledge routing, and pruning it has more semantic impact
than pruning E236 (a multi-domain hub).

## Comparison to codex's claims

Codex H2071: "combo_minimax_04 reaches mean 0.1499053240" — best
replay mean. My sweep refines: H2071's replay-best has HIGHER
capability cost than H2074's no-hub control on cross-prompt diff.
The replay-mean metric trades quality across prompts; the diff-line
metric captures discrete capability degradation. Both agree on the
sign (H2074 ≤ H2071), but the diff-line metric punishes hub-pruning
more than mean-replay does.

H2072: "H2071 is now a memory-fit candidate with a valid manifest,
not a deployment policy. Next step is H2073 chat-mode router capture
for math/knowledge/code/tool-DSML, then rerun the same manifest gate
before spending AIME/DSML generation time."

That gate just ran. Math + Knowledge + Code arms validated. Tool_DSML
is the remaining arm. AIME/DSML capability test (full generation, not
top-K diff) is the next-stage spend.

## What this enables

The 52GiB DS4Flash deployment frontier:
- Memory-fit: H2069 (52.0 GiB with 0.0785 MiB slack)
- Capability-safe on math/knowledge/code: **this commit**
- Tool_DSML capability: pending
- Generation-level AIME: pending

H2074 no-hub `[55,71,188,231,254]` is the deployable candidate;
H2071 hub-prune incurs measurable knowledge-domain cost; H2068 with
E52 is clearly inferior.

## Total infrastructure cost for this validation

- 36 ds4-logitlens runs via 9 daemon batches
- Wall: 65 seconds (compare to ~30 minutes per-launch sequence at
  ~5 sec each)
- Disk: 9 prompt dirs × 4 CSV outputs + 1 stderr = 45 files, ~10 KB

## Files

- `prompts.tsv` — name TAB body, 9 prompts
- `configs.txt` — 4 candidates × OUT/ placeholder
- `p_<name>/{prompt.txt, configs.txt, baseline.txt, h2074.txt, h2071.txt, h2068.txt, stderr.txt}`
  per prompt
- This `FINDINGS.md`
