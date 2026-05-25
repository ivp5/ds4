"""AMD: forced-commit + greedy extract using Qwen3.5-4B BF16.

silv 2026-05-25 continue: cross-precision rescue ceiling.

Mirror of M1 codec_audit/forced_commit_extract.py but using HuggingFace
torch on AMD ROCm with BF16 Qwen3.5-4B model. Reads cached M1 responses
(transferred via input JSON), applies forced-commit + greedy K=8.

Tests: does BF16 substrate retrieve truth from cached CoT at the same
rate as 4bit MLX? If yes → F3 (cross-precision invariance) extends
to rescue protocol. If no → precision affects retrieval.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


FORCED_PREAMBLE = "\n\nAfter careful analysis, the final answer is \\boxed{"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="JSON: {cells: [{cell, truth, response}]}")
    p.add_argument("--output", required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--prefix-frac", type=float, default=0.60)
    p.add_argument("--n-chars-cap", type=int, default=16000)
    p.add_argument("--k-tokens", type=int, default=8)
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    inp = json.loads(Path(args.input).read_text())
    print(f"  loading {args.model_id} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=True
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    results = []
    for c in inp["cells"]:
        cell_id = c["cell"]
        truth = c["truth"]
        full = c["response"]
        prefix_len = min(int(args.prefix_frac * len(full)), args.n_chars_cap)
        prompt = full[:prefix_len] + FORCED_PREAMBLE
        ids = tok.encode(prompt, return_tensors="pt").to("cuda:0")
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=args.k_tokens,
                do_sample=False,  # greedy
                pad_token_id=tok.eos_token_id or tok.pad_token_id,
            )
        new_ids = out[0, ids.shape[1]:]
        emit = tok.decode(new_ids, skip_special_tokens=True)
        elapsed = time.time() - t0
        m = re.match(r"(\d+)", emit.strip())
        predicted = int(m.group(1)) if m else None
        correct = (predicted == truth)
        print(f"  {cell_id}: truth={truth} predicted={predicted} correct={correct} "
              f"emit={emit[:30]!r} elapsed={elapsed:.1f}s")
        results.append({
            "cell": cell_id,
            "truth": truth,
            "predicted": predicted,
            "is_correct": correct,
            "emit": emit,
            "prefix_chars": prefix_len,
            "elapsed_s": elapsed,
        })
    n = sum(1 for r in results if r["is_correct"])
    print(f"\nTotal: {n}/{len(results)} correct via BF16 forced-commit + greedy K={args.k_tokens}")
    Path(args.output).write_text(json.dumps({
        "model_id": args.model_id,
        "prefix_frac": args.prefix_frac,
        "n_chars_cap": args.n_chars_cap,
        "k_tokens": args.k_tokens,
        "n_correct": n,
        "n_total": len(results),
        "results": results,
    }, indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
