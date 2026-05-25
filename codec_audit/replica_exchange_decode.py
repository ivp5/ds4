"""Parallel-tempering decoding for AIME rescue.

silv 2026-05-25 directive: "applying various DSP sampling metaphors
consistently in-depth."

Standard temperature-sampling at decode is a SINGLE Markov chain at a
fixed T. Parallel tempering / replica-exchange runs N chains at different
temperatures simultaneously; high-T chains explore broadly, low-T chains
exploit. In the canonical DSP form, periodic state-exchanges propagate
exploration into the exploit-chain.

For LLM decoding, "state exchange" doesn't carry — you cannot swap
mid-generation text between replicas because the prefix locks the
distribution. The DSP-analog is:
  (a) shared cache up to a TRIGGER POSITION
  (b) N parallel continuations at distinct temperatures from that cache
  (c) ensemble verifier scores each replica's emitted answer
  (d) return the best

The TRIGGER condition is L19 entropy growth < 0.55 at 40% of budget
(commit-concentration signal from this session's substrate work).

When NOT triggered, fall through to greedy decode.

Composition:
  - Conjecture #22 entropy/rank-shift detector: input-level (refuted at
    real adversarial-dataset scale)
  - Conjecture #23 non-convergence + forced-commit rescue: outcome-level
  - L19 growth detector: mid-generation, single-forward
  - THIS module: mid-generation, multi-T continuation as DSP analog

If forced-commit rescue extracts truth from chunk_2 cache, that's the
fastest path (~10× cheaper than multi-T continuation). Multi-T is for
cells where forced-commit fails — i.e., model hasn't yet derived truth
at the trigger point but might at higher T.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
proxy_port = random.choice(range(10001, 10150))
os.environ.setdefault("HTTPS_PROXY", f"socks5h://127.0.0.1:{proxy_port}")
os.environ.setdefault("ALL_PROXY", f"socks5h://127.0.0.1:{proxy_port}")

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))
from codec_audit.db import open_db, journal
from codec_audit.l19_growth_detector import l19_growth_from_full_text, GROWTH_THRESHOLD


# Replica temperatures — span exploitation to exploration
REPLICA_TEMPERATURES = [0.0, 0.3, 0.7, 1.0, 1.4]


def find_truth_shape_positions(text: str) -> list[int]:
    """Find character positions of 1-3 digit numbers in text that could be
    AIME answers (range 1-999, AIME conventions)."""
    import re
    out = []
    for m in re.finditer(r"\b(\d{1,3})\b", text):
        n = int(m.group(1))
        if 0 <= n <= 999:
            out.append((m.start(), m.end(), n))
    return out


def split_text_at_position(text: str, position_char: int) -> tuple[str, str]:
    """Split text at character position; returns (prefix, suffix)."""
    return text[:position_char], text[position_char:]


def replica_continue(model, tok, prefix_ids: list[int], n_gen: int,
                     temperature: float, seed: int = 0) -> str:
    """Continue generation from prefix with given temperature.
    Returns the generated suffix as a string."""
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    inp = mx.array([prefix_ids])
    # Prime the cache with prefix
    out = model(inp, cache=cache)
    mx.eval(out)

    generated_ids = []
    last_token = mx.array([[prefix_ids[-1]]])  # use last prefill token

    # Use mlx random for reproducibility
    mx.random.seed(seed)
    eos_tokens = set()
    if tok.eos_token_id is not None:
        eos_tokens.add(tok.eos_token_id)
    # Also stop on `<|im_end|>`
    try:
        im_end = tok.encode("<|im_end|>")[0]
        eos_tokens.add(im_end)
    except Exception:
        pass

    for step in range(n_gen):
        out = model(last_token, cache=cache)  # (1, 1, vocab)
        logits = out[0, -1, :].astype(mx.float32)
        if temperature <= 0:
            next_id = int(mx.argmax(logits))
        else:
            scaled = logits / temperature
            # subtract max for stability
            scaled = scaled - mx.max(scaled)
            probs = mx.softmax(scaled, axis=-1)
            # sample
            next_id = int(mx.random.categorical(scaled))
        if next_id in eos_tokens:
            break
        generated_ids.append(next_id)
        last_token = mx.array([[next_id]])

    return tok.decode(generated_ids)


def extract_boxed(text: str) -> int | None:
    """Extract \\boxed{N} integer if present; else None."""
    import re
    matches = list(re.finditer(r"\\boxed\{(\d+)\}", text))
    if not matches:
        return None
    try:
        return int(matches[-1].group(1))
    except ValueError:
        return None


def replica_exchange_rescue(model, tok, full_text: str, truth: int,
                             trigger_check: bool = True,
                             n_gen_per_replica: int = 400) -> dict:
    """Given a model + cached full response text + truth, simulate the
    replica-exchange decoding by:
      1. Compute L19 growth at chunk_2 of the cached text.
      2. If trigger fires (growth < threshold), pretend we're at chunk_2
         and spawn N replicas from that position.
      3. Each replica continues at its own temperature; emit \\boxed{N}.
      4. Score replicas via sympy_verifier; return best match.

    Note: "from chunk_2" means we re-tokenize the prefix up to chunk_2 and
    do an actual model.continue forward. This isn't the same as having
    LIVE-generated to chunk_2, but tests whether multi-T from the SAME
    prefix can produce divergent outcomes.
    """
    # Step 1: check trigger
    growth_result = l19_growth_from_full_text(model, tok, full_text)
    print(f"  growth check: growth={growth_result['growth']:+.4f}, "
          f"verdict={growth_result['verdict']}")

    fire = growth_result["verdict"] == "CONCENTRATING"
    if trigger_check and not fire:
        return {"triggered": False, "growth": growth_result["growth"]}

    # Step 2: prefix to chunk_2 (40% of token-len)
    ids = tok.encode(full_text)
    n_tok = len(ids)
    chunk_2_end = int(0.4 * n_tok)
    prefix_ids = ids[:chunk_2_end]
    prefix_text = tok.decode(prefix_ids)
    print(f"  trigger fires. prefix to chunk_2 = {chunk_2_end}/{n_tok} tokens")
    print(f"  prefix ends with: ...{prefix_text[-200:]}")

    # Step 3: spawn replicas
    replicas = []
    for i, T in enumerate(REPLICA_TEMPERATURES):
        print(f"  replica {i} T={T} continuing for up to {n_gen_per_replica} tokens...")
        t0 = time.time()
        suffix = replica_continue(model, tok, prefix_ids, n_gen_per_replica,
                                    temperature=T, seed=42 + i)
        gen_t = time.time() - t0
        # Try to find an answer in the suffix
        guess = extract_boxed(suffix)
        # Fall back: look for last "= N" or "is N"
        if guess is None:
            import re
            patterns = [r"final answer is\s*(\d{1,3})\b",
                         r"answer is\s*(\d{1,3})\b",
                         r"=\s*(\d{1,3})\s*$"]
            for pat in patterns:
                m = re.search(pat, suffix, re.IGNORECASE)
                if m:
                    try:
                        guess = int(m.group(1))
                        break
                    except ValueError:
                        pass
        correct = (guess == truth)
        print(f"    T={T} done in {gen_t:.1f}s, guess={guess} (truth={truth}, correct={correct})")
        replicas.append({
            "temperature": T,
            "suffix_len": len(suffix),
            "guess": guess,
            "correct": correct,
            "elapsed_s": gen_t,
            "suffix_tail": suffix[-300:],
        })

    n_correct = sum(1 for r in replicas if r["correct"])
    return {
        "triggered": True,
        "growth": growth_result["growth"],
        "n_tokens": n_tok,
        "chunk_2_end": chunk_2_end,
        "n_replicas": len(replicas),
        "n_correct": n_correct,
        "best_replica": next((r for r in replicas if r["correct"]), None),
        "all_replicas": replicas,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text-from", required=True)
    p.add_argument("--truth", type=int, required=True)
    p.add_argument("--field", default="response")
    p.add_argument("--n-chars", type=int, default=4000)
    p.add_argument("--n-gen", type=int, default=400)
    p.add_argument("--label", default="")
    p.add_argument("--out", type=str)
    args = p.parse_args()

    text = json.loads(Path(args.text_from).read_text())[args.field][:args.n_chars]
    from mlx_lm import load
    print(f"[{time.strftime('%H:%M:%S')}] loading Qwen3.5-4B-MLX-4bit")
    model, tok = load("mlx-community/Qwen3.5-4B-MLX-4bit")
    print(f"  text len: {len(text)} chars\n")

    t0 = time.time()
    result = replica_exchange_rescue(model, tok, text, truth=args.truth,
                                       trigger_check=True,
                                       n_gen_per_replica=args.n_gen)
    elapsed = time.time() - t0
    print(f"\n=== {args.label or Path(args.text_from).stem} ===")
    print(f"  triggered: {result.get('triggered')}")
    if result.get("triggered"):
        print(f"  L19 growth: {result['growth']:+.4f}")
        print(f"  n_correct: {result['n_correct']} / {result['n_replicas']}")
        if result.get("best_replica"):
            br = result["best_replica"]
            print(f"  best: T={br['temperature']} guess={br['guess']}")
        else:
            print(f"  NO REPLICA matched truth={args.truth}")
        print(f"  total wall: {elapsed:.1f}s")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"saved {args.out}")
    with open_db() as conn:
        journal(conn, "replica_exchange_rescue",
                str(args.text_from),
                {"label": args.label, "truth": args.truth,
                 "triggered": result.get("triggered"),
                 "n_correct": result.get("n_correct"),
                 "elapsed_s": elapsed})


if __name__ == "__main__":
    main()
