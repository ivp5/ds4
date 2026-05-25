# Severe audit of 2026-05-25 in-flight attention findings

silv directive: "doubt EVERYTHING, doubt EVERY STEP".

## What was claimed earlier this session

1. **L19 attention-entropy growth signal** catches commit-concentration
   (lock OR confident-wrong) at first 4000 chars in Qwen3.5-4B-MLX-4bit
   AIME 2026 cells. Threshold T=0.55: 2 TP (P01, P10) / 0 FP / 8 TN.
2. **Substrate at L31 of P01 lock contains truth digit '2' at P=0.9998** —
   originally interpreted as "the lock preserves truth latent at L31."
3. **Cross-precision P01** Pearson r(growth, HC_LOCK pers) = -0.85 (n=3).
4. **DSP replica exchange** (5 temperatures from chunk_2 cache) FAILS to
   escape lock on P01 — refutes "high-T saves us" hypothesis.

## What survived audit

**L19 attention growth signal** holds up under proper categorization. The
ORIGINAL cell categorization was wrong:
- P02, P03, P04, P06, P07, P08 were NOT "rescue successes" in native 4bit
  cache — they're native EXPLORE-NO-COMMIT cells. The rescue was a
  post-hoc forced-commit applied OFFLINE to extract truth.
- P10 is NOT "confident-wrong 100" in native — it's a LOCK cell with
  1801-line literal repetition of "Let's assume the condition..."
- Native cell categories actually in 4bit AIME 2026 cache:
  - LOCK: P01, P10 (cyclic/literal repetition, no `\boxed{}` emitted)
  - EXPLORE-NO-COMMIT: P02, P03, P04, P06, P07, P08, P09 (`\boxed{}` count=0)
  - NATIVE SUCCESS: P05 (committed `\boxed{65}` at char 33500 of 33699)

With THIS categorization, the L19 detector at first 4000 chars:
- P01 lock: growth=0.521 → flagged ✓
- P10 lock: growth=0.525 → flagged ✓
- P02-P09 explore: growth 0.609-1.281 → not flagged ✓ (8/8 correct)
- P05 native success: growth=0.872 → not flagged ✓ (commit at char 33500
  not in first 4000-char window)

**Recall on lock cells: 2/2 = 100%. Precision: 2/2 = 100%. Specificity on
non-lock cells in window: 8/8 = 100%.**

## What was REFUTED

1. **"P01 lock substrate preserves truth at L31 with P=0.9998"** — REFUTED.
   The earlier perlayer_logit_lens.py averaged over last 32 positions; some
   of those positions had '2' as natural top-1 (because P01's locked text
   contains the literal character '2' frequently — fraction notation,
   equation numbers, etc.). It was substring-emission, not
   truth-encoding.

   At 6 sampled positions in P01 (5%, 20%, 40%, 60%, 80%, 95%), max
   P('2' = first digit of truth=277) across ALL LAYERS = 0.000029
   (essentially zero). **P01 substrate does NOT contain truth=277.**
   Rank of '2' ranges 19 → 248000 across the layers.

2. **"P05 native success substrate at LAST POSITION encodes truth"** —
   REFUTED at "last position of first 4000 chars". P('7' or '0' or '6')
   = 0 at every layer. P05's success comes from emitting `\boxed{65}` at
   char 33500+; the first 4000 chars are early CoT.

3. **"Truth is latent at L19"** — REFUTED. At the actual `\boxed{`
   commit position in P05 (after derivation completed "m+n = 29+36 = 65"),
   per-layer P('6'):
   - L19: 0.000034 (rank 3768) — NOT encoded
   - L23: 0.006 (rank 0, top-1 — first emergence)
   - L27: 0.019
   - L29: 0.554
   - L30: 0.999
   - L31: 0.9998

   Truth installs at L23-L31 commit tier, NOT at L19. My session's
   "L19 carries truth" interpretation was conflating the L19 attention
   signal (real) with truth-encoding (not at L19).

## What this means for the deployable

The L19 attention-entropy GROWTH signal correctly catches commit-concentration
("about to commit to SOMETHING"), but commit-content (whether the committed
value is truth) lives in the L23-L31 install tier. Two-stage detector:

1. **L19 attention growth < 0.55** at window → model is concentrating on
   commit attempt
2. **L23-L31 logit-lens at the imminent-commit position** → check whether
   substrate has the answer being committed; if truth is at high P at L31,
   the cell will emit truth; if NOT, the cell is committing wrong or
   into-lock

Combined: predict whether to TRUNCATE+RESCUE (lock case) vs LET IT FINISH
(success case) vs FLAG-AS-WRONG (confident-wrong case).

## The deeper sampling-instrumentation critique that silv named

"keeping only the temp/top_k instead of employing sampling instrumentation"
— in retrospect this is what `logitlens_at_boxed.py` accidentally measured:
the L31 logits ARE the model's actual distribution at commit position;
top_k of those logits is what argmax/sample would emit. The model's
internal "thinking" before the commit (L0-L22) is irrelevant to the
commit DECISION — only the lm_head projection of the final residual
matters.

**Implication**: argmax/sample is not a wrapper around the model — it IS
the model's output. The L0-L22 computation is feature processing; L23-L31
is the commit-installation tier where the answer is selected.

Top_k retention at every position would let downstream backtrack through
the commit tier; my test on P01 found NO truth in any retained top_16 at
any position because P01's substrate genuinely doesn't have truth at any
layer. Top_k retention saves what was COMPUTED; it doesn't manifest what
was never computed.

## Files

- `codec_audit/perlayer_logit_lens.py` — last N positions
- `codec_audit/logitlens_at_position.py` — at fractional positions
- `codec_audit/logitlens_at_boxed.py` — at ACTUAL `\boxed{` emission position
- `tmp/20260525_attention_inflight/lens_at_pos_p01.json`,
  `lens_at_pos_p02.json`, `lens_at_pos_p05.json`,
  `lens_at_boxed_p05.json`

## Next severe falsifier (proposed)

Heavy AMD sweep: same per-layer logit-lens at every `\boxed{` position
across ALL 30 cells (10 problems × 3 precisions = 30) on AMD-cached
Qwen3.5-4B BF16. Measure: at every commit position, is truth installed
at L23-L31? Does precision affect WHEN (which layer) truth installs?

If 4bit installs truth at L29-L31 only while BF16 installs at L25-L31,
that's the substrate cost of low precision — late layers compress more
truth-uncertainty into the same install tier.
