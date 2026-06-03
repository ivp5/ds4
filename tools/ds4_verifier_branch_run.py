#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ds4_verifier_branch import ScanResult, result_to_json, scan_trace


DEFAULT_STATUS_INTERVAL_S = 5.0
BOS = "<｜begin▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
THINK_START = "<think>"
THINK_END = "</think>"


@dataclass(frozen=True)
class RunOutput:
    code: int
    stdout_path: Path
    stderr_path: Path
    stdout_text: str


def timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_chat_prompt(user_text: str, system_text: str, think_mode: str) -> str:
    if think_mode not in {"none", "high"}:
        raise ValueError(f"unsupported think mode: {think_mode}")
    think_token = THINK_START if think_mode == "high" else THINK_END
    return f"{BOS}{system_text if system_text else ''}{USER}{user_text}{ASSISTANT}{think_token}"


def run_command(command: list[str], stdout_path: Path, stderr_path: Path, label: str, status_interval_s: float) -> RunOutput:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"phase=run_start label={label} cmd={shlex.join(command)} stdout={stdout_path} stderr={stderr_path}", flush=True)
    start = time.monotonic()
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        process = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file)
        while True:
            code = process.poll()
            elapsed = time.monotonic() - start
            if code is not None:
                print(f"phase=run_done label={label} code={code} elapsed={elapsed:.1f}s", flush=True)
                break
            print(f"phase=run_wait label={label} elapsed={elapsed:.1f}s", flush=True)
            time.sleep(status_interval_s)
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    return RunOutput(code=code, stdout_path=stdout_path, stderr_path=stderr_path, stdout_text=stdout_text)


def make_ds4_command(
    args: argparse.Namespace,
    prompt_file: Path,
    token_count: int,
    temperature: float,
    seed: int | None,
) -> list[str]:
    command = [args.ds4]
    if args.flat_pack:
        command.extend(["--flat-pack", str(args.flat_pack)])
    if args.model:
        command.extend(["--model", str(args.model)])
    if args.nonrouted_pack:
        command.extend(["--nonrouted-pack", str(args.nonrouted_pack)])
    for ds4_arg in args.ds4_arg:
        command.append(ds4_arg)
    command.extend(["--prompt-file", str(prompt_file), "--ctx", str(args.ctx), "-n", str(token_count), "--temp", str(temperature)])
    if args.raw_prompt:
        command.append("--raw-prompt")
    if args.nothink:
        command.append("--nothink")
    if seed is not None:
        command.extend(["--seed", str(seed)])
    return command


def first_flag_position(text: str, result: ScanResult) -> int | None:
    for flag in result.flags:
        position = text.find(flag.clause)
        if position >= 0:
            return position
    return None


def run_self_test() -> int:
    prompt = "Solve. "
    generated = "v = 18. t = 14. D = v * t = 18. final answer is 18."
    initial = scan_trace(generated)
    position = first_flag_position(generated, initial)
    if position is None:
        print("selftest FAIL no flag position")
        return 1
    prefix = generated[:position]
    candidates = [
        "D = v * t = 18.",
        "D = v * t = 252.",
    ]
    for candidate_index, candidate in enumerate(candidates):
        result = scan_trace(prefix + candidate)
        print(
            f"selftest candidate={candidate_index} flags={len(result.flags) + len(result.count_flags)} "
            f"text={json.dumps(prompt + prefix + candidate)}"
        )
        if not result.flags and not result.count_flags and candidate_index == 1:
            print("selftest OK selected=1")
            return 0
    print("selftest FAIL no clean candidate")
    return 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a verifier-guided branch experiment around ds4. This is a severe-test harness: "
            "it replays prompt+clean-prefix by full prefill for each branch, so it is slower than native KV branching "
            "but tests whether verifier selection can lift arithmetic failures."
        )
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--ds4", default="./ds4")
    parser.add_argument("--flat-pack", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--nonrouted-pack", type=Path)
    parser.add_argument("--ds4-arg", action="append", default=[])
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("tmp/20260603_fidelity_runtime/verifier_branch_runs"))
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--initial-tokens", type=int, default=512)
    parser.add_argument("--branch-tokens", type=int, default=48)
    parser.add_argument("--continue-tokens", type=int, default=512)
    parser.add_argument("--branch-count", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--branch-temperature", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--render-chat", action="store_true")
    parser.add_argument("--system", default="You are a helpful assistant")
    parser.add_argument("--think-mode", choices=["none", "high"], default=None)
    parser.add_argument("--raw-prompt", dest="raw_prompt", action="store_true", default=True)
    parser.add_argument("--chat-template", dest="raw_prompt", action="store_false")
    parser.add_argument("--nothink", action="store_true")
    parser.add_argument("--status-interval-s", type=float, default=DEFAULT_STATUS_INTERVAL_S)
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()
    if not args.prompt_file:
        parser.error("--prompt-file is required unless --self-test is used")
    if args.branch_count <= 0:
        parser.error("--branch-count must be positive")
    if args.branch_tokens <= 0 or args.initial_tokens <= 0 or args.continue_tokens < 0:
        parser.error("token counts must be positive except --continue-tokens may be zero")
    if args.render_chat:
        args.raw_prompt = False

    run_dir = args.work_dir / timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    source_prompt_text = args.prompt_file.read_text(encoding="utf-8")
    prompt_text = source_prompt_text
    if args.render_chat:
        think_mode = args.think_mode if args.think_mode else ("none" if args.nothink else "high")
        prompt_text = render_chat_prompt(source_prompt_text, args.system, think_mode)
    initial_prompt = run_dir / "initial_prompt.txt"
    write_text(initial_prompt, prompt_text)

    initial_output = run_command(
        make_ds4_command(args, initial_prompt, args.initial_tokens, args.temperature, args.seed),
        run_dir / "initial.stdout.txt",
        run_dir / "initial.stderr.txt",
        "initial",
        args.status_interval_s,
    )
    if initial_output.code != 0:
        print(f"phase=abort reason=initial_failed code={initial_output.code}", flush=True)
        return initial_output.code

    initial_result = scan_trace(initial_output.stdout_text)
    write_text(run_dir / "initial.scan.json", json.dumps(result_to_json(initial_result), indent=2, sort_keys=True))
    if not initial_result.flags and not initial_result.count_flags:
        print(f"phase=no_branch reason=no_verifier_flags run_dir={run_dir}", flush=True)
        return 0

    flag_position = first_flag_position(initial_output.stdout_text, initial_result)
    if flag_position is None:
        print(f"phase=abort reason=flag_position_not_found run_dir={run_dir}", flush=True)
        return 3
    clean_prefix = initial_output.stdout_text[:flag_position]
    flagged_suffix = initial_output.stdout_text[flag_position:]
    write_text(run_dir / "clean_prefix.txt", clean_prefix)
    write_text(run_dir / "flagged_suffix.txt", flagged_suffix)
    clean_prefix_result = scan_trace(clean_prefix)

    selected_index: int | None = None
    selected_text = ""
    selected_result: ScanResult | None = None
    for candidate_index in range(args.branch_count):
        branch_prompt = run_dir / f"branch_{candidate_index:03d}.prompt.txt"
        write_text(branch_prompt, prompt_text + clean_prefix)
        branch_output = run_command(
            make_ds4_command(
                args,
                branch_prompt,
                args.branch_tokens,
                args.branch_temperature,
                args.seed + candidate_index + 1,
            ),
            run_dir / f"branch_{candidate_index:03d}.stdout.txt",
            run_dir / f"branch_{candidate_index:03d}.stderr.txt",
            f"branch_{candidate_index:03d}",
            args.status_interval_s,
        )
        if branch_output.code != 0:
            print(f"phase=branch_skip candidate={candidate_index} code={branch_output.code}", flush=True)
            continue
        branch_local_result = scan_trace(branch_output.stdout_text, clean_prefix_result.bindings)
        candidate_text = clean_prefix + branch_output.stdout_text
        candidate_result = scan_trace(candidate_text)
        write_text(
            run_dir / f"branch_{candidate_index:03d}.scan.json",
            json.dumps(result_to_json(candidate_result), indent=2, sort_keys=True),
        )
        flag_count = len(candidate_result.flags) + len(candidate_result.count_flags)
        evidence_count = len(branch_local_result.claims) + len(branch_local_result.count_flags)
        print(
            f"phase=branch_scanned candidate={candidate_index} flags={flag_count} evidence={evidence_count}",
            flush=True,
        )
        if flag_count == 0 and evidence_count > 0:
            selected_index = candidate_index
            selected_text = branch_output.stdout_text
            selected_result = candidate_result
            break

    if selected_index is None or selected_result is None:
        print(f"phase=no_selection branch_count={args.branch_count} run_dir={run_dir}", flush=True)
        return 2

    write_text(run_dir / "selected_prefix.txt", clean_prefix + selected_text)
    write_text(run_dir / "selected.scan.json", json.dumps(result_to_json(selected_result), indent=2, sort_keys=True))
    print(f"phase=selected candidate={selected_index} run_dir={run_dir}", flush=True)

    if args.continue_tokens > 0:
        continue_prompt = run_dir / "continue.prompt.txt"
        write_text(continue_prompt, prompt_text + clean_prefix + selected_text)
        continue_output = run_command(
            make_ds4_command(args, continue_prompt, args.continue_tokens, 0.0, args.seed + 100000),
            run_dir / "continue.stdout.txt",
            run_dir / "continue.stderr.txt",
            "continue",
            args.status_interval_s,
        )
        if continue_output.code != 0:
            print(f"phase=continue_failed code={continue_output.code} run_dir={run_dir}", flush=True)
            return continue_output.code
        write_text(run_dir / "final.txt", clean_prefix + selected_text + continue_output.stdout_text)
        final_result = scan_trace(clean_prefix + selected_text + continue_output.stdout_text)
        write_text(run_dir / "final.scan.json", json.dumps(result_to_json(final_result), indent=2, sort_keys=True))
        print(f"phase=final flags={len(final_result.flags) + len(final_result.count_flags)} run_dir={run_dir}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
