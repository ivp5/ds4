"""AMD-side recency rescue — measure deployable speedup."""
import json, os, re, sys, time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"


def find_truth_shape_last_positions(text: str) -> list[tuple[int, int]]:
    seen = {}
    for m in re.finditer(r"\b(\d{1,3})\b", text):
        try:
            val = int(m.group(1))
            if 0 <= val <= 999:
                seen[val] = m.start()
        except ValueError:
            pass
    return sorted(seen.items(), key=lambda x: -x[1])


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    inp = json.loads(Path("c:/LLMTEST/amd_forced_input_FULL.json").read_text())
    print("loading Qwen3.5-4B BF16...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", dtype=torch.bfloat16,
                                                   device_map="cuda:0", trust_remote_code=True)
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s\n")

    PREAMBLE = "\n\nAfter careful analysis, the final answer is \\boxed{"
    overall = time.time()
    n_correct = 0
    results = []
    for c in inp["cells"]:
        cell_id = c["cell"]
        truth = c["truth"]
        text = c["response"]
        candidates = find_truth_shape_last_positions(text)
        t_cell = time.time()
        found = False
        for val, char_pos in candidates[:10]:
            prefix_end = char_pos + len(str(val))
            prefix = text[:prefix_end]
            prompt = prefix + PREAMBLE
            ids = tok.encode(prompt, return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=8, do_sample=False,
                                       pad_token_id=tok.eos_token_id)
            emit = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            m = re.match(r"(\d+)", emit.strip())
            predicted = int(m.group(1)) if m else None
            if predicted == truth:
                found = True
                t_cell_s = time.time() - t_cell
                print(f"  {cell_id}: ✓ truth={truth} at candidate_pos={char_pos} after {candidates[:10].index((val, char_pos))+1} tries ({t_cell_s:.1f}s)")
                break
        if not found:
            t_cell_s = time.time() - t_cell
            print(f"  {cell_id}: ✗ truth={truth} unrescuable ({t_cell_s:.1f}s)")
        if found:
            n_correct += 1
        results.append({"cell": cell_id, "truth": truth, "found": found,
                          "elapsed_s": time.time() - t_cell})
    total = time.time() - overall
    print(f"\nDeployable rescue: {n_correct}/{len(results)} correct, total {total:.1f}s")
    print(f"  avg per cell: {total/len(results):.1f}s")
    print(f"  vs M1 MLX cap=24000 sweep: ~30s/cell × 10 cells = ~300s + 10s model load = 310s")
    Path("c:/LLMTEST/amd_recency_rescue_results.json").write_text(json.dumps({
        "n_correct": n_correct, "n_total": len(results),
        "total_elapsed_s": total, "results": results,
    }, indent=2))
    print(f"saved")


if __name__ == "__main__":
    main()
