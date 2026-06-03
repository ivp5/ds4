#!/usr/bin/env python3
import argparse
import ast
import json
import math
import os
import pathlib
import subprocess
import sys
import time

import numpy as np


DEFAULT_TRACE_ROOT = pathlib.Path(
    "/Users/silv/cl/tlp/montyneg/ivp5_ds4/tmp/20260531_codec_coherence/CORRECTNESS_TRACES"
)
DEFAULT_MODEL = pathlib.Path(
    "/Users/silv/cl/tlp/montyneg/ds4/DeepSeek-V4-Flash_H2597_VQD8_triple_noE8_alloccoded_rich8192_fit131k_i8_mlx_20260601/metadata/DeepSeek-V4-Flash.metadata-only.full-tensor-manifest.zero-tensor-data.pack-direct.gguf"
)
DEFAULT_NONROUTED = pathlib.Path(
    "/Users/silv/cl/tlp/montyneg/ds4/DeepSeek-V4-Flash_H2597_VQD8_triple_noE8_alloccoded_rich8192_fit131k_i8_mlx_20260601/nonrouted/ds4v4_nonrouted.i32_normf32_bf16matf16.pack"
)
DEFAULT_D8F = pathlib.Path(
    "/Users/silv/cl/tlp/montyneg/ds4/DeepSeek-V4-Flash_H2597_VQD8_triple_noE8_alloccoded_rich8192_fit131k_i8_mlx_20260601/routed_fused_d8f"
)
DEFAULT_OUT = pathlib.Path("/Users/silv/cl/tlp_codex/tmp/20260601_codec_runtime/correctness_trace_gate")


def log(msg):
    print(f"[trace-gate] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def load_token_ids(case_dir):
    payload = json.loads((case_dir / "token_ids.json").read_text())
    return [int(x) for x in payload["token_ids"]]


def parse_dump_tokens(stdout_path):
    for line in stdout_path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            parsed = ast.literal_eval(stripped)
            return [int(x) for x in parsed]
    raise RuntimeError(f"no token-id list found in {stdout_path}")


def run_logged(cmd, env, stdout_path):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    log(f"run start elapsed=0.0s log={stdout_path} cmd={' '.join(cmd)}")
    with stdout_path.open("wb") as fp:
        proc = subprocess.run(cmd, stdout=fp, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - start
    log(f"run done elapsed={elapsed:.1f}s rc={proc.returncode} log={stdout_path}")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(cmd)}")
    return elapsed


def softmax_f64(logits):
    x = logits.astype(np.float64, copy=False)
    m = float(np.max(x))
    e = np.exp(x - m)
    total = float(np.sum(e))
    return e / total


def compare_logits(engine_logits, target_logits):
    engine = engine_logits.astype(np.float64, copy=False)
    target = target_logits.astype(np.float64, copy=False)
    engine_argmax = int(np.argmax(engine))
    target_argmax = int(np.argmax(target))
    dot = float(np.dot(engine, target))
    denom = math.sqrt(float(np.dot(engine, engine)) * float(np.dot(target, target)))
    cosine = dot / denom if denom else float("nan")
    eprob = softmax_f64(engine)
    tprob = softmax_f64(target)
    eps = 1e-300
    kl_target_engine = float(np.sum(tprob * (np.log(tprob + eps) - np.log(eprob + eps))))
    max_abs = float(np.max(np.abs(engine - target)))
    rmse = float(np.sqrt(np.mean((engine - target) ** 2)))
    return {
        "engine_argmax": engine_argmax,
        "target_argmax": target_argmax,
        "argmax_match": engine_argmax == target_argmax,
        "cosine": cosine,
        "kl_target_engine": kl_target_engine,
        "max_abs": max_abs,
        "rmse": rmse,
        "engine_argmax_logit": float(engine[engine_argmax]),
        "target_argmax_logit": float(target[target_argmax]),
    }


def load_engine_logits(path):
    payload = json.loads(path.read_text())
    logits = np.array(payload["logits"], dtype=np.float32)
    return payload, logits


def case_names(trace_root, requested):
    if requested == ["all"]:
        return sorted([p.name for p in trace_root.iterdir() if p.is_dir()])
    return requested


def main(argv):
    ap = argparse.ArgumentParser(
        description="Validate ds4 engine logits against CORRECTNESS_TRACES reference artifacts."
    )
    ap.add_argument("--trace-root", type=pathlib.Path, default=DEFAULT_TRACE_ROOT)
    ap.add_argument("--engine", type=pathlib.Path, default=pathlib.Path("./ds4"))
    ap.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
    ap.add_argument("--nonrouted-pack", type=pathlib.Path, default=DEFAULT_NONROUTED)
    ap.add_argument("--d8f-pack-dir", type=pathlib.Path, default=DEFAULT_D8F)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--case", action="append", default=None)
    ap.add_argument("--target", choices=["cod", "ref"], default="cod")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--prime", action="store_true")
    ap.add_argument("--tokenize-only", action="store_true")
    ap.add_argument("--skip-tokenizer-check", action="store_true")
    args = ap.parse_args(argv)

    if not args.engine.is_absolute():
        local_engine = (pathlib.Path.cwd() / args.engine).resolve()
        if local_engine.exists():
            args.engine = local_engine

    trace_root = args.trace_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    names = case_names(trace_root, args.case or ["all"])
    env = os.environ.copy()
    env["DS4_D8F_PACK_DIR"] = str(args.d8f_pack_dir)
    if args.prime:
        env["DS4_PRIME_PATH"] = "1"

    summary = {
        "trace_root": str(trace_root),
        "engine": str(args.engine),
        "model": str(args.model),
        "nonrouted_pack": str(args.nonrouted_pack),
        "d8f_pack_dir": str(args.d8f_pack_dir),
        "target": args.target,
        "prime": bool(args.prime),
        "cases": [],
    }
    failures = 0
    start_all = time.time()

    for name in names:
        case_start = time.time()
        case_dir = trace_root / name
        if not case_dir.is_dir():
            raise RuntimeError(f"missing case dir: {case_dir}")
        log(f"case={name} phase=start")
        expected_ids = load_token_ids(case_dir)
        case_out = out_dir / name
        case_out.mkdir(parents=True, exist_ok=True)
        case_summary = {"name": name, "n_tokens": len(expected_ids)}

        if not args.skip_tokenizer_check:
            token_log = case_out / "dump_tokens.log"
            run_logged(
                [
                    str(args.engine),
                    "--model",
                    str(args.model),
                    "--raw-prompt",
                    "--prompt-file",
                    str(case_dir / "prompt.txt"),
                    "--dump-tokens",
                ],
                env,
                token_log,
            )
            actual_ids = parse_dump_tokens(token_log)
            token_match = actual_ids == expected_ids
            case_summary["tokenizer_match"] = token_match
            case_summary["tokenizer_log"] = str(token_log)
            if not token_match:
                failures += 1
                case_summary["tokenizer_first_mismatch"] = next(
                    (i for i, pair in enumerate(zip(actual_ids, expected_ids)) if pair[0] != pair[1]),
                    min(len(actual_ids), len(expected_ids)),
                )
                log(f"case={name} tokenizer_match=false")
                summary["cases"].append(case_summary)
                continue

        if not args.tokenize_only:
            logits_json = case_out / f"engine_logits_{args.target}{'_prime' if args.prime else ''}.json"
            engine_log = case_out / f"dump_logits_{args.target}{'_prime' if args.prime else ''}.log"
            run_logged(
                [
                    str(args.engine),
                    "--model",
                    str(args.model),
                    "--raw-prompt",
                    "--nonrouted-pack",
                    str(args.nonrouted_pack),
                    "--prefill-metal-phases",
                    "auto",
                    "--prompt-file",
                    str(case_dir / "prompt.txt"),
                    "--dump-logits",
                    str(logits_json),
                    "--ctx",
                    str(args.ctx),
                ],
                env,
                engine_log,
            )
            payload, engine_logits = load_engine_logits(logits_json)
            engine_prompt_tokens = int(payload.get("prompt_tokens", -1))
            case_summary["engine_prompt_tokens"] = engine_prompt_tokens
            if engine_prompt_tokens != len(expected_ids):
                failures += 1
                case_summary["prompt_token_count_match"] = False
                case_summary["prompt_token_count_expected"] = len(expected_ids)
                log(
                    f"case={name} prompt_token_count_match=false "
                    f"engine={engine_prompt_tokens} expected={len(expected_ids)}"
                )
                summary["cases"].append(case_summary)
                continue
            case_summary["prompt_token_count_match"] = True
            target_logits = np.load(case_dir / f"logits_{args.target}.npy")
            if engine_logits.shape != target_logits.shape:
                raise RuntimeError(
                    f"shape mismatch case={name}: engine={engine_logits.shape} target={target_logits.shape}"
                )
            metrics = compare_logits(engine_logits, target_logits)
            case_summary.update(metrics)
            case_summary["engine_log"] = str(engine_log)
            case_summary["engine_logits_json"] = str(logits_json)
            case_summary["engine_argmax_token"] = payload.get("argmax_token")
            if not metrics["argmax_match"]:
                failures += 1
            log(
                "case=%s argmax_match=%s cosine=%.9f kl=%.6g rmse=%.6g elapsed=%.1fs"
                % (
                    name,
                    metrics["argmax_match"],
                    metrics["cosine"],
                    metrics["kl_target_engine"],
                    metrics["rmse"],
                    time.time() - case_start,
                )
            )
        else:
            log(f"case={name} tokenize-only ok elapsed={time.time() - case_start:.1f}s")

        summary["cases"].append(case_summary)

    summary["elapsed_sec"] = time.time() - start_all
    summary["failures"] = failures
    summary_path = out_dir / f"summary_{args.target}{'_prime' if args.prime else ''}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    log(f"summary={summary_path} failures={failures} elapsed={summary['elapsed_sec']:.1f}s")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
