#!/usr/bin/env python3
"""End-to-end DS4 pack gate: traces, answer extraction, and speed in one report.

This is intentionally a wrapper, not a new evaluator.  It preserves full logs,
uses the existing correctness-trace checker for logit fidelity, and adds the
runtime-facing checks that decide whether a pack is useful on the M1 Max path.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from dataclasses import dataclass


REPO = pathlib.Path(__file__).resolve().parents[1]
TRACE_CHECKER = REPO / "tools" / "check_correctness_traces.py"
DEFAULT_TRACE_ROOT = REPO / "tmp" / "20260531_codec_coherence" / "CORRECTNESS_TRACES"
DEFAULT_MODEL_ROOT = pathlib.Path(
    "/Users/silv/cl/tlp/montyneg/ds4/"
    "DeepSeek-V4-Flash_H2597_VQD8_triple_noE8_alloccoded_rich8192_fit131k_i8_mlx_20260601"
)
DEFAULT_MODEL = DEFAULT_MODEL_ROOT / "metadata" / (
    "DeepSeek-V4-Flash.metadata-only.full-tensor-manifest.zero-tensor-data.pack-direct.gguf"
)
DEFAULT_NONROUTED = DEFAULT_MODEL_ROOT / "nonrouted" / "ds4v4_nonrouted.i32_normf32_bf16matf16.pack"
DEFAULT_D8F = DEFAULT_MODEL_ROOT / "routed_fused_d8f"
DEFAULT_ANSWER_PROMPT = REPO / "tmp" / "20260602_runtime_probe" / "h2878_popper_refutation_prompt.txt"
DEFAULT_OUT_ROOT = REPO / "tmp" / "20260602_release_gate"


TIMING_RE = re.compile(r"prefill:\s*([0-9.]+)\s*t/s,\s*generation:\s*([0-9.]+)\s*t/s")
BOXED_RE = re.compile(r"\\boxed\{\s*([^}]+?)\s*\}")
SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")
ENGINE_PREFIXES = (
    "ds4:",
    "ds4_",
    "argv:",
    "bind_store:",
    "===",
    "ggml_",
    "llama_",
    "main:",
    "system_info",
    "warning:",
    "build:",
    "Metal",
    "gpu ",
)


@dataclass
class GatePaths:
    engine: pathlib.Path
    model: pathlib.Path
    nonrouted: pathlib.Path
    d8f: pathlib.Path
    trace_root: pathlib.Path
    out_dir: pathlib.Path


def utcstamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def log(message: str) -> None:
    print(f"[release-gate] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}", flush=True)


def clean_generation_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if not stripped:
            continue
        if stripped.startswith(ENGINE_PREFIXES):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def parse_timing(text: str) -> dict[str, float | None]:
    matches = TIMING_RE.findall(text)
    if not matches:
        return {"prefill_tps": None, "generation_tps": None}
    prefill, generation = matches[-1]
    return {"prefill_tps": float(prefill), "generation_tps": float(generation)}


def extract_answer_signals(text: str) -> dict[str, object]:
    generated = clean_generation_text(text)
    boxed = BOXED_RE.findall(generated)
    think_closed = "</think>" in generated
    answer_words = re.findall(r"\b(answer|therefore|so the|final)\b", generated[-4000:], re.IGNORECASE)
    tokens = generated.split()
    grams = [tuple(tokens[i : i + 4]) for i in range(max(0, len(tokens) - 3))]
    distinct_4gram_ratio = None
    if grams:
        distinct_4gram_ratio = round(len(set(grams)) / len(grams), 4)
    sentence_counts: dict[str, int] = {}
    for match in SENTENCE_RE.finditer(generated):
        sentence = " ".join(match.group(0).strip().lower().split())
        if len(sentence) < 24:
            continue
        sentence_counts[sentence] = sentence_counts.get(sentence, 0) + 1
    repeated_sentences = [
        {"count": count, "sentence": sentence[:240]}
        for sentence, count in sorted(sentence_counts.items(), key=lambda item: (-item[1], item[0]))
        if count > 1
    ][:5]
    max_sentence_repeat = max(sentence_counts.values(), default=0)
    return {
        "generated_chars": len(generated),
        "generated_words": len(tokens),
        "think_closed": think_closed,
        "boxed_answer_seen": bool(boxed),
        "boxed_answers": boxed[-3:],
        "answer_signal_seen": bool(answer_words or boxed),
        "distinct_4gram_ratio": distinct_4gram_ratio,
        "max_sentence_repeat": max_sentence_repeat,
        "repeated_sentences": repeated_sentences,
        "repetition_seen": bool(repeated_sentences),
        "generation_tail": generated[-1200:],
    }


def run_logged(cmd: list[str], env: dict[str, str], log_path: pathlib.Path, timeout: int | None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    log(f"run start log={log_path} timeout={timeout or 0}s cmd={' '.join(cmd[:8])} ...")
    with log_path.open("wb") as out:
        out.write(("# cwd: " + str(REPO) + "\n").encode())
        out.write(("# cmd: " + " ".join(cmd) + "\n\n").encode())
        out.flush()
        try:
            proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            out.write(f"\n# release-gate-timeout: timeout={timeout or 0}s elapsed={elapsed:.1f}s\n".encode())
            out.flush()
            log(f"run timeout elapsed={elapsed:.1f}s timeout={timeout or 0}s log={log_path}")
            return 124
    log(f"run done rc={proc.returncode} elapsed={time.time() - start:.1f}s log={log_path}")
    return int(proc.returncode)


def base_env(paths: GatePaths) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("DS4_D8F_PACK_PATH", "DS4_D8F_PACK_LAYER", "DS4_D8F_PACK_TEMPLATE", "DS4_D8F_PACK_DIR"):
        env.pop(key, None)
    env["DS4_D8F_PACK_DIR"] = str(paths.d8f)
    return env


def existing_path_report(paths: GatePaths) -> tuple[dict[str, object], list[str]]:
    checks: dict[str, object] = {}
    missing: list[str] = []
    for name, path in (
        ("engine", paths.engine),
        ("model", paths.model),
        ("nonrouted", paths.nonrouted),
        ("d8f", paths.d8f),
        ("trace_root", paths.trace_root),
    ):
        exists = path.exists()
        entry: dict[str, object] = {"path": str(path), "exists": exists}
        if exists and path.is_file():
            entry["bytes"] = path.stat().st_size
        checks[name] = entry
        if not exists:
            missing.append(name)
    return checks, missing


def run_trace_gate(paths: GatePaths, args: argparse.Namespace, report: dict[str, object]) -> int:
    out_dir = paths.out_dir / "trace"
    cmd = [
        sys.executable,
        str(TRACE_CHECKER),
        "--trace-root",
        str(paths.trace_root),
        "--engine",
        str(paths.engine),
        "--model",
        str(paths.model),
        "--nonrouted-pack",
        str(paths.nonrouted),
        "--d8f-pack-dir",
        str(paths.d8f),
        "--out-dir",
        str(out_dir),
        "--target",
        args.trace_target,
        "--ctx",
        str(args.ctx),
    ]
    if args.prime:
        cmd.append("--prime")
    if args.skip_tokenizer_check:
        cmd.append("--skip-tokenizer-check")
    if args.tokenize_only:
        cmd.append("--tokenize-only")
    log_path = paths.out_dir / "trace_gate.log"
    rc = run_logged(cmd, base_env(paths), log_path, args.trace_timeout)
    summary_name = f"summary_{args.trace_target}{'_prime' if args.prime else ''}.json"
    summary_path = out_dir / summary_name
    trace_summary = {"rc": rc, "log": str(log_path), "summary": str(summary_path)}
    if summary_path.exists():
        payload = json.loads(summary_path.read_text())
        trace_summary["failures"] = payload.get("failures")
        trace_summary["cases"] = [
            {
                "name": case.get("name"),
                "argmax_match": case.get("argmax_match"),
                "kl_target_engine": case.get("kl_target_engine"),
                "cosine": case.get("cosine"),
                "rmse": case.get("rmse"),
            }
            for case in payload.get("cases", [])
        ]
    report["trace_gate"] = trace_summary
    return rc


def run_generation_gate(paths: GatePaths, args: argparse.Namespace, report: dict[str, object]) -> int:
    log_path = paths.out_dir / "answer_speed" / "generation.log"
    cmd = [
        str(paths.engine),
        "--model",
        str(paths.model),
        "--nonrouted-pack",
        str(paths.nonrouted),
        "--prefill-metal-phases",
        "auto",
        "--ctx",
        str(args.ctx),
        "-n",
        str(args.max_tokens),
        "--temp",
        str(args.temp),
        "--seed",
        str(args.seed),
    ]
    if args.raw_prompt:
        prompt_display = str(args.answer_prompt)
        cmd.extend(["-p", str(args.answer_prompt)])
    else:
        prompt_path = args.answer_prompt.resolve()
        prompt_display = str(prompt_path)
        cmd.extend(["--prompt-file", str(prompt_path)])
    if args.nothink:
        cmd.append("--nothink")
    if args.raw_prompt:
        cmd.append("--raw-prompt")
    if args.engine_stop_repeat_sentence:
        cmd.append("--stop-repeat-sentence")
    if args.presence_penalty != 0:
        cmd.extend(["--presence-penalty", str(args.presence_penalty)])
    rc = run_logged(cmd, base_env(paths), log_path, args.generation_timeout)
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    timing = parse_timing(text)
    signals = extract_answer_signals(text)
    engine_stop_triggered = "ds4: repeated sentence stop:" in text
    generation_tps = timing.get("generation_tps")
    speed_ok = generation_tps is not None and generation_tps >= args.min_generation_tps
    answer_ok = (not args.require_answer_signal) or bool(signals["answer_signal_seen"])
    clean_stop_ok = (
        (not args.require_clean_stop)
        or not bool(signals["repetition_seen"])
        or (args.engine_stop_repeat_sentence and engine_stop_triggered)
    )
    stage = {
        "rc": rc,
        "log": str(log_path),
        "prompt": prompt_display,
        "raw_prompt": bool(args.raw_prompt),
        "nothink": bool(args.nothink),
        "engine_stop_repeat_sentence": bool(args.engine_stop_repeat_sentence),
        "max_tokens": args.max_tokens,
        "timing": timing,
        "signals": signals,
        "engine_stop_triggered": bool(engine_stop_triggered),
        "min_generation_tps": args.min_generation_tps,
        "speed_ok": bool(speed_ok),
        "answer_ok": bool(answer_ok),
        "clean_stop_ok": bool(clean_stop_ok),
    }
    report["answer_speed_gate"] = stage
    if rc != 0:
        return rc
    if not speed_ok or not answer_ok or not clean_stop_ok:
        return 2
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Run DS4 codec/runtime/answer gates as one measured object.")
    ap.add_argument("--engine", type=pathlib.Path, default=REPO / "ds4")
    ap.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
    ap.add_argument("--nonrouted-pack", type=pathlib.Path, default=DEFAULT_NONROUTED)
    ap.add_argument("--d8f-pack-dir", type=pathlib.Path, default=DEFAULT_D8F)
    ap.add_argument("--trace-root", type=pathlib.Path, default=DEFAULT_TRACE_ROOT)
    ap.add_argument("--out-root", type=pathlib.Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--run-id", default=utcstamp())
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--answer-speed", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--trace-target", choices=["cod", "ref"], default="cod")
    ap.add_argument("--prime", action="store_true")
    ap.add_argument("--tokenize-only", action="store_true")
    ap.add_argument("--skip-tokenizer-check", action="store_true")
    ap.add_argument("--trace-timeout", type=int, default=1800)
    ap.add_argument("--answer-prompt", type=pathlib.Path, default=DEFAULT_ANSWER_PROMPT)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--generation-timeout", type=int, default=900)
    ap.add_argument("--min-generation-tps", type=float, default=50.0)
    ap.add_argument("--require-answer-signal", action="store_true")
    ap.add_argument("--require-clean-stop", action="store_true")
    ap.add_argument("--raw-prompt", action="store_true")
    ap.add_argument("--nothink", action="store_true")
    ap.add_argument("--engine-stop-repeat-sentence", action="store_true")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    args = ap.parse_args(argv)

    paths = GatePaths(
        engine=args.engine.resolve(),
        model=args.model.resolve(),
        nonrouted=args.nonrouted_pack.resolve(),
        d8f=args.d8f_pack_dir.resolve(),
        trace_root=args.trace_root.resolve(),
        out_dir=(args.out_root / args.run_id).resolve(),
    )
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "run_id": args.run_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "paths": {
            "engine": str(paths.engine),
            "model": str(paths.model),
            "nonrouted": str(paths.nonrouted),
            "d8f": str(paths.d8f),
            "trace_root": str(paths.trace_root),
            "out_dir": str(paths.out_dir),
        },
        "target": {
            "pack_size_gb_decimal": 52.0,
            "generation_tps": args.min_generation_tps,
            "fidelity": "trace argmax preservation + low KL + answer-bearing behavior",
        },
    }
    checks, missing = existing_path_report(paths)
    report["preflight"] = checks
    report_path = paths.out_dir / "release_gate_report.json"

    if missing:
        report["status"] = "missing-input"
        report["missing"] = missing
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        log(f"missing inputs {missing}; report={report_path}")
        return 2

    planned = []
    if args.all or args.trace:
        planned.append("trace")
    if args.all or args.answer_speed:
        planned.append("answer_speed")
    report["planned_stages"] = planned
    if args.dry_run or not planned:
        report["status"] = "dry-run" if args.dry_run else "no-stage-selected"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        log(f"{report['status']} report={report_path}")
        return 0

    failures: list[str] = []
    if "trace" in planned:
        if run_trace_gate(paths, args, report) != 0:
            failures.append("trace")
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    if "answer_speed" in planned:
        if run_generation_gate(paths, args, report) != 0:
            failures.append("answer_speed")
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    report["status"] = "failed" if failures else "passed"
    report["failed_stages"] = failures
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    log(f"status={report['status']} report={report_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
