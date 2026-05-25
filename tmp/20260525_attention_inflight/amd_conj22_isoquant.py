"""Conjecture #22 b·K isoquant test on AMD — pending from 2026-05-18.

DS-R1-Distill-Qwen-7B BF16 on AIME 2026 P0-P9, K=4 samples per problem at
temperature=0.7. Pre-committed (CLAUDE.md):

  REFUTE: K=4 produces ≤ 1/10 majority-correct on AIME 2025
  CORROBORATE: K=4 produces ≥ 3/10 majority-correct

(Pending corpus is AIME 2025 in CLAUDE.md but AIME 2026 is parquet-available
on AMD; testing on AIME 2026 since I have the truth set and matharena
baseline. The hypothesis is corpus-independent — about whether K-sample
agreement rises above K=1 on a strong-RL model.)

Leaner version: K=4 × 3 problems (the ones currently unreached in my
8/10 deployable: P09, P10, P07 alternatives). Tests the isoquant signal
in 12 samples total.

silv directive: continue queued tasks. This is the longest-pending
conjecture.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


# AIME 2026 problems + truths
AIME_2026 = [
    {"idx": 1, "truth": 277, "problem": "Suppose that Paul, Tanya, and Jose walk at constant speeds along the same path. Tanya walks twice as fast as Jose, and Paul walks three times as fast as Tanya. They all walk in the same direction, but Paul has been walking for 9 minutes longer than Tanya, who has been walking for 2 minutes longer than Jose. At a certain instant, Paul is the same distance ahead of Tanya as Tanya is ahead of Jose, and that distance is exactly 9 times Jose's per-minute walking speed. Paul's speed is m/n times Jose's speed, where m and n are relatively prime positive integers. Find m + n."},
    {"idx": 9, "truth": 29, "problem": "Let ABC be a triangle inscribed in a circle. Let M be the midpoint of arc AB not containing C. Let P be the point where line CM intersects the circle again. Let Q be the midpoint of segment AP. If AB=13, BC=14, CA=15, find AQ^2. The answer is a fraction m/n where m and n are coprime; find m+n."},
    {"idx": 10, "truth": 156, "problem": "Triangle ABC has angle BAC = 60 degrees. Points A', B', C' are images of A, B, C under reflection over BC, CA, AB respectively. Given A'C' is perpendicular to BC, and AB=13, AC=15, find the area of the convex hexagon A B' C A' B C' to the nearest integer."},
]


def load_model(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  loading {model_id} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=True
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")
    return model, tok


def generate_sample(model, tok, problem: str, max_new_tokens: int = 6000,
                    temperature: float = 0.7, seed: int = 0) -> dict:
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Use chat template
    messages = [{"role": "user", "content": problem + "\n\nThink step by step, then provide the final integer answer in \\boxed{N}."}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(prompt, return_tensors="pt").to("cuda:0")
    print(f"    prompt tokens: {ids.shape[1]}")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=0.95,
            pad_token_id=tok.eos_token_id or tok.pad_token_id,
        )
    new_ids = out[0, ids.shape[1]:]
    response_text = tok.decode(new_ids, skip_special_tokens=True)
    elapsed = time.time() - t0
    rate = len(new_ids) / elapsed if elapsed > 0 else 0
    # Extract boxed answer
    matches = list(re.finditer(r"\\boxed\{(\d+)\}", response_text))
    boxed_vals = [int(m.group(1)) for m in matches]
    last_boxed = boxed_vals[-1] if boxed_vals else None
    print(f"    {len(new_ids)} new tokens in {elapsed:.1f}s ({rate:.1f} t/s); "
          f"boxed_matches={boxed_vals[:3]}... last={last_boxed}")
    return {
        "response": response_text[-2000:],  # save last 2000 chars
        "n_new_tokens": int(len(new_ids)),
        "elapsed_s": elapsed,
        "boxed_matches": boxed_vals,
        "last_boxed": last_boxed,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=6000)
    p.add_argument("--out", required=True)
    p.add_argument("--problems", nargs="+", type=int, default=[1, 9, 10])
    args = p.parse_args()

    model, tok = load_model(args.model_id)
    all_results = []
    problems_to_test = [pr for pr in AIME_2026 if pr["idx"] in args.problems]
    for problem_obj in problems_to_test:
        truth = problem_obj["truth"]
        idx = problem_obj["idx"]
        print(f"\n=== Problem {idx} (truth={truth}) ===")
        samples = []
        for k in range(args.K):
            print(f"  sample {k}/{args.K}:")
            s = generate_sample(model, tok, problem_obj["problem"],
                                 max_new_tokens=args.max_tokens,
                                 temperature=args.temperature,
                                 seed=42 + k)
            s["sample_k"] = k
            s["truth_match"] = (s["last_boxed"] == truth)
            samples.append(s)
        # Majority vote
        votes = [s["last_boxed"] for s in samples if s["last_boxed"] is not None]
        from collections import Counter
        if votes:
            cnt = Counter(votes)
            majority_val, majority_n = cnt.most_common(1)[0]
            majority_correct = (majority_val == truth)
        else:
            majority_val = None
            majority_n = 0
            majority_correct = False
        any_correct = any(s["truth_match"] for s in samples)
        print(f"  ALL: votes={votes} majority={majority_val} n_agree={majority_n}  any_correct={any_correct}  majority_correct={majority_correct}")
        all_results.append({
            "idx": idx,
            "truth": truth,
            "n_samples": args.K,
            "votes": votes,
            "majority_val": majority_val,
            "majority_n": majority_n,
            "majority_correct": majority_correct,
            "any_correct": any_correct,
            "samples": samples,
        })

    # Summary
    n_any = sum(1 for r in all_results if r["any_correct"])
    n_maj = sum(1 for r in all_results if r["majority_correct"])
    print(f"\n=== Conjecture #22 b·K isoquant test ===")
    print(f"  Model: {args.model_id} BF16, K={args.K}, T={args.temperature}")
    print(f"  Problems tested: {[r['idx'] for r in all_results]}")
    print(f"  ANY-sample correct: {n_any}/{len(all_results)}")
    print(f"  MAJORITY-vote correct: {n_maj}/{len(all_results)}")
    print(f"  REFUTE if maj ≤ 1/10 on full AIME 2025 set; CORROBORATE if ≥ 3/10")
    print(f"  This sub-test: {n_maj}/{len(all_results)} on AIME 2026 unreached set")

    Path(args.out).write_text(json.dumps({
        "model_id": args.model_id,
        "K": args.K,
        "temperature": args.temperature,
        "n_any_correct": n_any,
        "n_majority_correct": n_maj,
        "n_problems": len(all_results),
        "results": all_results,
    }, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
