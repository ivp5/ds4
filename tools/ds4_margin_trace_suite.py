#!/usr/bin/env python3
"""Build a diversified prompt suite for DS4 margin/logprob traces.

The echo trace is a useful canary, not a sufficient calibration set.  This
tool gathers existing repo prompts into a fixed suite spanning math, code,
tool/schema text, complex prose, and ordinary chat/knowledge prompts.  It does
not run the model; it emits a JSONL manifest to feed whatever trace dumper is
current.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


FIXED_PROMPTS: list[tuple[str, str, str]] = [
    ("math_short", "math", "tmp/20260531_codec_coherence/CORRECTNESS_TRACES/math/prompt.txt"),
    ("math_aime_p01", "math_aime", "tmp/20260604_ds4_rewrite_corner/aime_p01_prompt.txt"),
    ("code_c_matmul", "code", "tmp/20260531_codec_coherence/CORRECTNESS_TRACES/matmul_c/prompt.txt"),
    ("code_python", "code", "tmp/20260527_30prompt_sweep/p_code_python/prompt.txt"),
    ("code_bash", "code", "tmp/20260527_30prompt_sweep/p_code_bash/prompt.txt"),
    ("tool_function_call", "tool_schema", "tmp/20260527_dsml_aime/dsml_dsml_func_call/prompt.txt"),
    ("json_object", "tool_schema", "tmp/20260527_dsml_aime/dsml_dsml_json_obj/prompt.txt"),
    ("xml_tag", "tool_schema", "tmp/20260527_dsml_aime/dsml_dsml_xml_tag/prompt.txt"),
    ("yaml_kv", "tool_schema", "tmp/20260527_dsml_aime/dsml_dsml_yaml_kv/prompt.txt"),
    ("complex_prose", "prose", "tmp/20260531_codec_coherence/CORRECTNESS_TRACES/prose/prompt.txt"),
]


JSONL_SOURCES: list[tuple[str, str, str, int]] = [
    ("quality_btree", "complex_text", "gguf-tools/quality-testing/prompts.jsonl", 0),
    ("quality_tcp_slow_clients", "coding_design", "gguf-tools/quality-testing/prompts.jsonl", 1),
    ("quality_mmap_macos", "systems", "gguf-tools/quality-testing/prompts.jsonl", 2),
    ("quality_rmsnorm", "math_systems", "gguf-tools/quality-testing/prompts.jsonl", 3),
    ("quality_heap_push", "code_algorithm", "gguf-tools/quality-testing/prompts.jsonl", 5),
]


TEXT_LINE_SOURCES: list[tuple[str, str, str, int]] = [
    ("eval_db_indexes", "ordinary_chat", "dir-steering/examples/eval_prompts.txt", 0),
    ("eval_sorting", "ordinary_chat", "dir-steering/examples/eval_prompts.txt", 1),
    ("eval_debug_slow", "ordinary_chat", "dir-steering/examples/eval_prompts.txt", 3),
]


def read_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_jsonl_prompt(path: Path, row_index: int) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            if index == row_index:
                payload = json.loads(line)
                return str(payload.get("prompt", ""))
    raise IndexError(f"{path}: missing row {row_index}")


def load_line_prompt(path: Path, row_index: int) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if row_index >= len(lines):
        raise IndexError(f"{path}: missing line {row_index}")
    return lines[row_index]


def add_row(rows: list[dict[str, Any]], prompt_id: str, category: str, source: str, prompt: str, max_chars: int) -> None:
    if max_chars > 0:
        prompt = prompt[:max_chars]
    rows.append(
        {
            "id": prompt_id,
            "category": category,
            "source": source,
            "prompt_chars": len(prompt),
            "prompt": prompt,
        }
    )


def build_suite(max_chars: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt_id, category, rel in FIXED_PROMPTS:
        path = REPO_ROOT / rel
        if path.exists():
            add_row(rows, prompt_id, category, rel, read_prompt(path), max_chars)
    for prompt_id, category, rel, row_index in JSONL_SOURCES:
        path = REPO_ROOT / rel
        if path.exists():
            add_row(rows, prompt_id, category, f"{rel}:{row_index}", load_jsonl_prompt(path, row_index), max_chars)
    for prompt_id, category, rel, row_index in TEXT_LINE_SOURCES:
        path = REPO_ROOT / rel
        if path.exists():
            add_row(rows, prompt_id, category, f"{rel}:{row_index + 1}", load_line_prompt(path, row_index), max_chars)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-chars", type=int, default=0, help="truncate prompts for quick canary traces; 0 keeps full text")
    args = parser.parse_args()

    rows = build_suite(args.max_chars)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    print(json.dumps({"prompts": len(rows), "categories": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
