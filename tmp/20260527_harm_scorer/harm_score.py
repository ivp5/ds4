#!/usr/bin/env python3
"""
Phase B driver — per-organ signed source-top-K harm scorer.

silv task #656 (after #649 design + #651/#652/#654 dispatch wires +
#656 engine auto-load of DS4_ORGAN_SKIP).

Reads a cells.csv (layer,expert,organ triples) and a prompts.txt
(one prompt per line), invokes ds4-logitlens per (prompt, cell) cell,
captures top-K logprobs, and emits a CSV of signed source-top-K margin
deltas per cell.

CSV input format (cells.csv):
    # comment lines OK
    layer,expert,organ
    9,52,2     # gate=0, up=1, down=2
    9,231,0
    ...

Output CSV columns:
    prompt_id, layer, expert, organ, k, baseline_logprob,
    perturbed_logprob, signed_harm
where:
    margin_baseline[k]  = L_baseline[truth]  - L_baseline[t_k]
    margin_perturbed[k] = L_perturbed[truth] - L_perturbed[t_k]
    signed_harm[k]      = margin_baseline[k] - margin_perturbed[k]
                          (>0 = harmful, <0 = helpful, =0 = inert)

PRECISION/SPEED TRADEOFF (per DESIGN.md OOM-precision + OOM-speedup):
    Current: SLOW path — one DS4 launch per cell. 81 GB model load
    every time = ~5 min/cell. 100 cells × 10 prompts = ~83 hours.

    Fast path (deferred): a long-running ds4-logitlens daemon that loads
    the model once + accepts STDIN prompts + ablation cells. Achieves
    the OOM speedup ladder named in DESIGN.md (KV shared prefix ×
    structural panel × batched case = ~600×). 100×10 → ~8 minutes.

    Phase B scaffold validates the SIGNAL on a tiny corpus (3 prompts ×
    5 cells) before the daemon is built. If signal is real here, the
    daemon investment is justified.

Usage:
    python3 harm_score.py \\
        --model    /path/to/DS4-V4-Flash-IQ2XXS-...-imatrix.gguf \\
        --prompts  tmp/20260527_harm_scorer/prompts.txt \\
        --cells    tmp/20260527_harm_scorer/cells.csv \\
        --truth-tokens 277,62,79 \\
        --topk     8 \\
        --out      tmp/20260527_harm_scorer/results.csv
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def parse_cells(cells_path: Path) -> list[tuple[int, int, int]]:
    """Parse cells.csv → list of (layer, expert, organ) tuples."""
    cells = []
    for lineno, line in enumerate(cells_path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            raise SystemExit(f"cells.csv:{lineno}: expected 'layer,expert,organ', got {line!r}")
        cells.append(tuple(int(p) for p in parts))
    return cells


def fmt_organ_skip_env(cell: tuple[int, int, int] | None) -> str:
    """Build the DS4_ORGAN_SKIP env value for one cell. None = baseline (empty)."""
    if cell is None:
        return ""
    return f"{cell[0]},{cell[1]},{cell[2]}"


def run_logitlens(
    binary: Path,
    model: Path,
    prompt_path: Path,
    organ_skip: str,
    topk: int,
    extra_args: list[str],
) -> dict:
    """Invoke ds4-logitlens once. Returns parsed top-K as {token_id: logprob}.

    The binary already accepts -k N + --prompt-file FILE. Per the engine
    auto-load wire in ds4_engine_open (commit pending #656), setting
    DS4_ORGAN_SKIP in the environment activates the organ-skip flag
    BEFORE the FFN dispatches. Combined with --cpu-moe this gives the
    honest per-organ ablation semantics from Phase A.1/A.2.

    Returns dict: {token_id: float_logprob} for top-K tokens.
    """
    env = os.environ.copy()
    if organ_skip:
        env["DS4_ORGAN_SKIP"] = organ_skip
    else:
        env.pop("DS4_ORGAN_SKIP", None)

    cmd = [
        str(binary),
        "-m", str(model),
        "--prompt-file", str(prompt_path),
        "-k", str(topk),
        "--cpu-moe",
        *extra_args,
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        raise RuntimeError(
            f"ds4-logitlens failed (rc={result.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  env DS4_ORGAN_SKIP={organ_skip or '(unset)'}\n"
            f"  stderr tail:\n{result.stderr[-500:]}"
        )

    # ds4-logitlens emits CSV header `rank,token_id,logprob,logit,text`
    # followed by K rows. Parse from stdout.
    rows = {}
    in_csv = False
    for line in result.stdout.splitlines():
        if line.startswith("rank,token_id,"):
            in_csv = True
            continue
        if not in_csv or not line.strip():
            continue
        parts = line.split(",", 4)
        if len(parts) < 3:
            continue
        try:
            tok_id = int(parts[1])
            logp = float(parts[2])
            rows[tok_id] = logp
        except ValueError:
            continue
    if not rows:
        raise RuntimeError(f"no top-K rows parsed from ds4-logitlens stdout (last 200 chars):\n{result.stdout[-200:]}")
    return rows


def signed_harm(baseline: dict, perturbed: dict, truth_id: int, topk: int) -> list[tuple[int, float, float, float]]:
    """For each top-K competitor in BASELINE, compute the signed margin delta.

    Returns list of (k_index, baseline_margin_to_truth, perturbed_margin_to_truth, signed_harm).
    """
    if truth_id not in baseline:
        # Truth token isn't even in baseline top-K — measurement degenerates.
        # We still report against the competitors but with NaN truth margins.
        return []
    base_truth = baseline[truth_id]
    # Top-K tokens ordered by descending logprob in baseline.
    competitors = sorted(baseline.keys(), key=lambda t: -baseline[t])[:topk]
    out = []
    for k, tok in enumerate(competitors):
        if tok == truth_id:
            continue
        margin_baseline = base_truth - baseline[tok]
        # The PERTURBED logprobs may not include all baseline tokens.
        # Missing → treat as -inf logprob → near-infinite margin → impl as
        # very negative perturbed_logp.
        pert_truth = perturbed.get(truth_id, -1e9)
        pert_tok   = perturbed.get(tok,       -1e9)
        margin_perturbed = pert_truth - pert_tok
        harm = margin_baseline - margin_perturbed
        out.append((k, margin_baseline, margin_perturbed, harm))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--prompts", type=Path, required=True,
                    help="One prompt per line. The truth-tokens list applies element-wise.")
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--truth-tokens", type=str, required=True,
                    help="Comma-separated token ids: the expected next token per prompt.")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--binary", type=Path, default=Path("./ds4-logitlens"))
    ap.add_argument("--extra-args", default="", help="Extra args passed through to ds4-logitlens.")
    args = ap.parse_args()

    prompts = [line.rstrip() for line in args.prompts.read_text().splitlines() if line.strip()]
    truth_tokens = [int(t.strip()) for t in args.truth_tokens.split(",")]
    if len(prompts) != len(truth_tokens):
        raise SystemExit(f"--prompts has {len(prompts)} lines but --truth-tokens has {len(truth_tokens)} ids")
    cells = parse_cells(args.cells)
    extra = args.extra_args.split() if args.extra_args else []

    print(f"harm_score: {len(prompts)} prompts × ({len(cells)}+1 cells = baseline + cells) = "
          f"{len(prompts) * (len(cells) + 1)} ds4-logitlens launches",
          file=sys.stderr)

    out_rows: list[dict] = []
    for p_idx, (prompt, truth_id) in enumerate(zip(prompts, truth_tokens)):
        # Write the prompt to a temp file (ds4-logitlens expects --prompt-file).
        tmp_prompt = Path(f"/tmp/harm_score_p{p_idx}.txt")
        tmp_prompt.write_text(prompt + "\n")

        # Baseline.
        print(f"[p={p_idx}] baseline …", file=sys.stderr)
        base = run_logitlens(args.binary, args.model, tmp_prompt,
                             organ_skip="", topk=args.topk, extra_args=extra)

        # Per-cell perturbations.
        for c_idx, cell in enumerate(cells):
            skip = fmt_organ_skip_env(cell)
            print(f"[p={p_idx}] cell {c_idx+1}/{len(cells)} = {skip} …", file=sys.stderr)
            pert = run_logitlens(args.binary, args.model, tmp_prompt,
                                 organ_skip=skip, topk=args.topk, extra_args=extra)
            for k, mb, mp, harm in signed_harm(base, pert, truth_id, args.topk):
                out_rows.append({
                    "prompt_id": p_idx,
                    "layer": cell[0],
                    "expert": cell[1],
                    "organ": cell[2],
                    "k": k,
                    "margin_baseline": mb,
                    "margin_perturbed": mp,
                    "signed_harm": harm,
                })

    # Write results CSV.
    if out_rows:
        with args.out.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"harm_score: wrote {len(out_rows)} rows to {args.out}", file=sys.stderr)
    else:
        print("harm_score: NO ROWS produced (truth-token never in baseline top-K?)", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
