"""AMD-side: per-layer logit-lens at \\boxed{ commit position via PyTorch HF.

silv 2026-05-25 directive: offload heavy substrate sweep.

This script runs on AMD (RX 7900 XTX, 25.6 GB free) with
Qwen/Qwen3.5-4B (BF16) cached. It is the BF16 reference comparison
to the 4bit-MLX result on M1.

Input: response_text (from M1 cached 4bit responses) + truth_digit
Output: per-layer P(truth_digit) and rank at each \\boxed{ commit position.

Method: use transformers `output_hidden_states=True` to get all
hidden states from a forward pass; project each through final
RMSNorm + lm_head.weight to get logits per layer. Cheap because
HF supports this natively.

Usage (on AMD):
  python amd_perlayer_lens_bf16.py --input PATH_TO_INPUT_JSON --output OUT.json
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


def find_boxed_emission_positions(text: str) -> list[int]:
    return [m.end() for m in re.finditer(r"\\boxed\{", text)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="JSON {response, truth_digit, ...}")
    p.add_argument("--output", required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--max-commits", type=int, default=3)
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    inp = json.loads(Path(args.input).read_text())
    text = inp["response"]
    truth_digit = inp["truth_digit"]
    print(f"  full response: {len(text)} chars; truth_digit='{truth_digit}'")

    boxed_positions = find_boxed_emission_positions(text)
    print(f"  \\boxed{{ openings: {len(boxed_positions)}")
    if not boxed_positions:
        Path(args.output).write_text(json.dumps({"no_boxed": True, "label": inp.get("label", "")}, indent=2))
        print(f"  NO commits — saved {args.output}")
        return

    print(f"  loading {args.model_id} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=True
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s; device={next(model.parameters()).device}, dtype={next(model.parameters()).dtype}")
    # Resolve truth token id
    truth_id = tok.encode(truth_digit, add_special_tokens=False)[0]
    print(f"  truth_digit='{truth_digit}' -> token_id={truth_id} decode='{tok.decode([truth_id])}'")

    # Resolve final_norm + lm_head
    internal = model.model if hasattr(model, "model") else model
    final_norm = internal.norm if hasattr(internal, "norm") else None
    lm_head = model.lm_head if hasattr(model, "lm_head") else None
    if final_norm is None or lm_head is None:
        raise RuntimeError(f"Could not find norm or lm_head on {type(model).__name__}")
    n_layers = len(internal.layers)
    print(f"  n_layers: {n_layers}")

    results = []
    for ci, pos in enumerate(boxed_positions[:args.max_commits]):
        prefix = text[:pos]
        ids = tok.encode(prefix, return_tensors="pt").to("cuda:0")
        n_tok = ids.shape[1]
        print(f"\n  commit {ci} at char {pos}, prefix tokens: {n_tok}")
        print(f"    prefix ends: ...{prefix[-100:]!r}")
        with torch.no_grad():
            out = model(ids, output_hidden_states=True, return_dict=True)
        # out.hidden_states is a tuple of (n_layers+1) tensors of shape (1, n_tok, hidden)
        # [0] = embedding output, [1..n_layers] = post-layer hidden states
        hidden_states = out.hidden_states
        print(f"    n_hidden_state_tensors: {len(hidden_states)}")

        per_layer = {}
        for L in range(len(hidden_states)):
            h = hidden_states[L][:, -1:, :]  # last position, shape (1, 1, hidden)
            # Apply final_norm + lm_head
            h_norm = final_norm(h)
            logits = lm_head(h_norm)  # (1, 1, vocab)
            logits_f32 = logits[0, 0].float().cpu().numpy()
            # softmax
            import numpy as np
            z = logits_f32 - logits_f32.max()
            probs = np.exp(z) / np.exp(z).sum()
            p_truth = float(probs[truth_id])
            rank = int((logits_f32 > logits_f32[truth_id]).sum())
            top1_id = int(np.argmax(logits_f32))
            per_layer[L] = {
                "p_truth": p_truth,
                "rank_truth": rank,
                "top1_id": top1_id,
                "top1_token": tok.decode([top1_id]),
                "top1_prob": float(probs[top1_id]),
            }
        # Print compact
        print(f"    per-layer P({truth_digit}) and rank (selected layers):")
        for L in [0, 5, 10, 15, 18, 19, 23, 27, 29, 30, n_layers-1, n_layers]:
            if L in per_layer:
                r = per_layer[L]
                print(f"      L{L:>2}  P={r['p_truth']:.6f} rank={r['rank_truth']:>5d}  top1={r['top1_token'][:8]!r:>10} p_top1={r['top1_prob']:.4f}")

        results.append({
            "commit_idx": ci,
            "char_pos": pos,
            "n_tokens": n_tok,
            "context_before_50": prefix[-50:],
            "context_after_30": text[pos:pos+30],
            "per_layer": per_layer,
        })
        # Free
        del out, hidden_states
        torch.cuda.empty_cache()

    Path(args.output).write_text(json.dumps({
        "label": inp.get("label", ""),
        "model_id": args.model_id,
        "n_boxed_emissions": len(boxed_positions),
        "commits_analyzed": len(results),
        "results": results,
    }, indent=2))
    print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
