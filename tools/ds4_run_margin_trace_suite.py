#!/usr/bin/env python3
"""Run DS4 --dump-logprobs over a diversified margin trace suite.

This is the execution half of ds4_margin_trace_suite.py.  It writes one prompt
file and one logprob trace per suite row, preserves a manifest with exact
commands, and supports --dry-run/--reuse so the suite can be extended without
clobbering prior traces.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from ds4_margin_trace_suite import build_suite  # noqa: E402


def shell_join(argv: list[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in argv)


def load_suite(path: Path | None, max_chars: int) -> list[dict[str, Any]]:
    if path is None:
        return build_suite(max_chars)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_id(value: str) -> str:
    out = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            out.append(char)
        else:
            out.append("_")
    text = "".join(out).strip("_")
    return text or "trace"


def base_command(args: argparse.Namespace, prompt_path: Path, trace_path: Path) -> list[str]:
    cmd = [
        str(args.ds4),
        "--metal" if args.backend == "metal" else "--cpu",
        "--temp",
        "0",
        "--nothink",
        "--system",
        "",
        "-c",
        str(args.ctx),
        "--prompt-file",
        str(prompt_path),
        "-n",
        str(args.tokens),
        "--dump-logprobs",
        str(trace_path),
        "--logprobs-top-k",
        str(args.top_k),
    ]
    if args.raw_prompt:
        cmd.insert(cmd.index("--prompt-file"), "--raw-prompt")
    if args.flat_pack:
        cmd[1:1] = ["--flat-pack", str(args.flat_pack)]
    elif args.model:
        cmd[1:1] = ["-m", str(args.model)]
    return cmd


def current_wired_gb() -> float | None:
    if sys.platform != "darwin":
        return None
    proc = subprocess.run(["vm_stat"], text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return None
    page_size = 16384
    wired_pages: int | None = None
    for line in proc.stdout.splitlines():
        if "page size of" in line:
            parts = line.replace(")", "").split()
            for index, part in enumerate(parts):
                if part == "of" and index + 1 < len(parts):
                    try:
                        page_size = int(parts[index + 1])
                    except ValueError:
                        pass
        if line.startswith("Pages wired down:"):
            value = line.split(":", 1)[1].strip().rstrip(".").replace(".", "")
            wired_pages = int(value)
    if wired_pages is None:
        return None
    return float(wired_pages * page_size) / 1.0e9


def enforce_wired_budget(args: argparse.Namespace) -> None:
    if args.backend != "metal" or args.dry_run:
        return
    wired = current_wired_gb()
    if wired is None:
        print("[trace-suite] wired preflight unavailable; continuing", file=sys.stderr, flush=True)
        return
    projected = wired + max(args.wired_estimate_gb, 0.0)
    print(
        f"[trace-suite] wired preflight current={wired:.2f}GB "
        f"estimate={args.wired_estimate_gb:.2f}GB projected={projected:.2f}GB "
        f"limit={args.wired_limit_gb:.2f}GB",
        flush=True,
    )
    if projected > args.wired_limit_gb:
        raise SystemExit(
            f"refusing DS4 launch: projected wired {projected:.2f}GB exceeds "
            f"--wired-limit-gb {args.wired_limit_gb:.2f}. Wait for other GPU jobs, "
            "lower --wired-estimate-gb only with evidence, or run elsewhere."
        )


def run_command(cmd: list[str], cwd: Path, env: dict[str, str], dry_run: bool) -> tuple[int, float]:
    started = time.perf_counter()
    print("+", shell_join(cmd), flush=True)
    if dry_run:
        return 0, 0.0
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True)
    elapsed = time.perf_counter() - started
    return proc.returncode, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-jsonl", type=Path, help="existing suite JSONL; defaults to generated repo suite")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ds4", type=Path, default=REPO_ROOT / "ds4")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--flat-pack", type=Path)
    parser.add_argument("--backend", choices=("metal", "cpu"), default="metal")
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument("--wired-limit-gb", type=float, default=56.0)
    parser.add_argument(
        "--wired-estimate-gb",
        type=float,
        default=47.0,
        help="expected additional wired-memory footprint for this DS4 trace run; H3384 AIME prefill measured ~47GB",
    )
    parser.add_argument(
        "--allow-uncertified-h3384",
        action="store_true",
        help="sets DS4_ALLOW_UNCERTIFIED_H3384=1; explicit --flat-pack is preferred for speed-only traces",
    )
    args = parser.parse_args()

    rows = load_suite(args.suite_jsonl, args.max_chars)
    if args.limit > 0:
        rows = rows[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir = args.out_dir / "prompts"
    trace_dir = args.out_dir / "logprobs"
    prompt_dir.mkdir(exist_ok=True)
    trace_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    if args.allow_uncertified_h3384:
        env["DS4_ALLOW_UNCERTIFIED_H3384"] = "1"

    manifest_rows: list[dict[str, Any]] = []
    failures = 0
    for index, row in enumerate(rows):
        trace_id = safe_id(str(row["id"]))
        prompt_path = prompt_dir / f"{index:03d}_{trace_id}.txt"
        trace_path = trace_dir / f"{index:03d}_{trace_id}.logprobs.json"
        prompt_path.write_text(str(row["prompt"]), encoding="utf-8")
        cmd = base_command(args, prompt_path, trace_path)
        if args.reuse and trace_path.exists():
            code = 0
            elapsed = 0.0
            print(f"[trace-suite] reuse {trace_path}", flush=True)
        else:
            enforce_wired_budget(args)
            code, elapsed = run_command(cmd, REPO_ROOT, env, args.dry_run)
        if code != 0:
            failures += 1
        manifest_rows.append(
            {
                "id": row["id"],
                "category": row["category"],
                "source": row.get("source"),
                "prompt_path": str(prompt_path),
                "logprobs_path": str(trace_path),
                "command": cmd,
                "elapsed_s": elapsed,
                "exit_code": code,
            }
        )
        print(f"[trace-suite] {index + 1}/{len(rows)} {row['id']} exit={code} elapsed={elapsed:.2f}s", flush=True)

    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ds4": str(args.ds4),
        "flat_pack": str(args.flat_pack) if args.flat_pack else None,
        "model": str(args.model) if args.model else None,
        "backend": args.backend,
        "ctx": args.ctx,
        "tokens": args.tokens,
        "top_k": args.top_k,
        "raw_prompt": args.raw_prompt,
        "dry_run": args.dry_run,
        "rows": manifest_rows,
    }
    manifest_path = args.out_dir / "trace_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[trace-suite] wrote {manifest_path} failures={failures}", flush=True)
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
