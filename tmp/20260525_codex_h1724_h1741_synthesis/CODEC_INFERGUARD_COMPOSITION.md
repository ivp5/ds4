# Codec arc + inferguard rescue: composition for Branch A deployment

silv's inferguard package is actively maintained (recent calibration
work + cross-model F1 verification). The aime_rescue.py "fire station"
catches "model derived truth then talked itself out" failures.

The codec arc (this session) introduces ~21% rel_L2 noise at the
FFN-output level per call (polar p32_m8) or ~2% (VQ K=256). The
question for Branch A deployment is: does this noise propagate to
AIME hold-rate as outright wrong answers, OR as near-miss "talked
itself out" answers that inferguard already rescues?

## The composition

```
   Layer                       Codec substitution        Inferguard
   -------------------         ---------------------    ----------
   FFN output (per layer)      polar/VQ noise injected   —
   Logits (final layer)        cumulative drift          —
   Generated tokens            potentially shifted        rescue applies
   \boxed{N} extraction         possibly missing          rescue triggers
```

If codec noise produces:
- **Same output** → no impact (inferguard not triggered)
- **Confidently wrong** → inferguard CAN'T rescue (no truth in CoT
  to extract)
- **Talked-itself-out** → inferguard RESCUES (truth in CoT, no commit)
- **Different correct answer** → no impact (still correct)

The crucial question: WHICH failure mode does codec noise produce?

Hypothesis (testable): codec noise is most likely to push borderline-
correct cases into "talked-itself-out" failures rather than
confidently-wrong, because:
1. Per-FFN-call cos_sim is 0.97-1.00 (direction preserved)
2. Magnitude wobble of 21% (polar) or 2% (VQ) is unlikely to flip
   token-level decisions for confident attractors
3. It's MORE likely to perturb borderline reasoning steps, causing
   the model to second-guess and reframe

If this hypothesis holds, codec + inferguard composes well:
- Codec saves memory/compute (6×) at the cost of some borderline
  reasoning quality
- Inferguard catches the borderline cases that get pushed into
  "talked-itself-out" failure mode
- Net AIME hold-rate ≈ FP4 baseline (codec quality recovered by
  rescue layer)

## Branch A test plan (when silv runtime-tests)

Three measurements, not one:

1. **Baseline**: FP4 inference, no inferguard
   AIME 2026 P0-P9 K=8 hold-rate baseline

2. **Codec substitution**: polar (or VQ) FFN substitution, no inferguard
   AIME 2026 hold-rate with codec noise
   Expected: some degradation (1-3 problems flip to wrong/no-commit?)

3. **Codec + inferguard rescue**: polar substitution + inferguard wrapper
   AIME 2026 hold-rate with codec + rescue
   Expected: closer to baseline if hypothesis holds; equal-to-baseline
   = codec is "free" at answer-correctness layer

If (3) ≈ (1), codec deployment is fully justified — pure speed/memory win
with no quality cost.
If (3) < (1) substantially, codec noise produces confidently-wrong
answers that rescue can't catch. Codec deployment may need additional
mitigation (e.g., use codec only on layers known to be tolerant; per
codex H1740 pressure-aware policy).

## Why this matters NOW (before silv directs Branch A body)

The B-2.3c body decision shouldn't be made in isolation from
inferguard. Specifically:
- If silv plans to deploy DS4 with inferguard rescue layer ON, codec
  substitution is much lower risk (rescue absorbs near-misses)
- If silv plans bare-model inference, codec substitution risk is
  proportional to rel_L2 noise level

This memo doesn't add code; it surfaces the architectural composition
silv should consider when picking A.1/A.2/A.3 sub-decision.

## Tasks (no new commits required for this memo)

- inferguard/aime_rescue.py exists per task #511 [completed]
- Codec arc B-2.3c stub exists per task #578 [completed]
- B-2.3d AIME hold-rate measurement (silv runtime) should run as
  three-condition test (baseline, codec, codec+rescue), not just two
- The data from this measurement directly informs whether the codec
  arc's substrate gains (6× lower codec error at half storage)
  translate to inference-level gains

## What survives this session arc's last cogency

The codec arc + inferguard rescue compose into a unified deployment
story:
- Substrate: codec compression (1.10 OOMs)
- Engineering: hot-path stub (B-2.3c wired)
- Rescue: aime_rescue.py for talked-itself-out catches
- Validation: 3-condition AIME hold-rate test (when silv runs it)

The end-to-end target: ~30 MB resident DS4 substrate (with codex
chunk-streaming) + 6× lower codec error (with VQ K=256) + inferguard
rescue layer catching codec-induced near-misses + AIME hold-rate
≥ FP4 baseline.

This is the deployable picture as of 2026-05-25 evening on M1 Max.
silv decides A.1/A.2/A.3 + runtime test schedule.
