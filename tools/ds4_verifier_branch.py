#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import operator
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


NUMERIC_TOLERANCE = 1e-9
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[()+\-*/%^]")
CLAUSE_RE = re.compile(r"[;,.\n]+")
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
OPS: dict[type[ast.operator] | type[ast.unaryop], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


@dataclass(frozen=True)
class LinearForm:
    coeffs: dict[str, float]
    const: float


@dataclass(frozen=True)
class Claim:
    clause: str
    values: list[float]
    ok: bool


@dataclass(frozen=True)
class ScanResult:
    claims: list[Claim]
    flags: list[Claim]
    bindings: dict[str, float]
    count_flags: list[dict[str, Any]]


def normalize_expression(expression: str) -> str:
    normalized = expression.strip().replace("^", "**")
    normalized = re.sub(r"(\d(?:\.\d+)?)([A-Za-z_])", r"\1*\2", normalized)
    normalized = re.sub(r"(\))(\d|[A-Za-z_])", r"\1*\2", normalized)
    return normalized


def safe_eval(expression: str, bindings: dict[str, float] | None = None) -> float | None:
    normalized = normalize_expression(expression)
    if not normalized:
        return None
    try:
        parsed = ast.parse(normalized, mode="eval").body
    except SyntaxError:
        return None

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and bindings and node.id in bindings:
            return float(bindings[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return float(OPS[type(node.op)](evaluate(node.left), evaluate(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return float(OPS[type(node.op)](evaluate(node.operand)))
        raise ValueError

    try:
        value = evaluate(parsed)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def linear_add(left: LinearForm, right: LinearForm, scale: float = 1.0) -> LinearForm:
    coeffs = dict(left.coeffs)
    for name, value in right.coeffs.items():
        coeffs[name] = coeffs.get(name, 0.0) + scale * value
        if abs(coeffs[name]) <= NUMERIC_TOLERANCE:
            del coeffs[name]
    return LinearForm(coeffs=coeffs, const=left.const + scale * right.const)


def linear_scale(form: LinearForm, scale: float) -> LinearForm:
    return LinearForm(
        coeffs={name: value * scale for name, value in form.coeffs.items() if abs(value * scale) > NUMERIC_TOLERANCE},
        const=form.const * scale,
    )


def linear_form(expression: str, bindings: dict[str, float] | None = None) -> LinearForm | None:
    normalized = normalize_expression(expression)
    if not normalized:
        return None
    try:
        parsed = ast.parse(normalized, mode="eval").body
    except SyntaxError:
        return None

    def evaluate(node: ast.AST) -> LinearForm:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return LinearForm(coeffs={}, const=float(node.value))
        if isinstance(node, ast.Name):
            if bindings and node.id in bindings:
                return LinearForm(coeffs={}, const=float(bindings[node.id]))
            return LinearForm(coeffs={node.id: 1.0}, const=0.0)
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            form = evaluate(node.operand)
            if isinstance(node.op, ast.USub):
                return linear_scale(form, -1.0)
            if isinstance(node.op, ast.UAdd):
                return form
            raise ValueError
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            left = evaluate(node.left)
            right = evaluate(node.right)
            return linear_add(left, right, -1.0 if isinstance(node.op, ast.Sub) else 1.0)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if left.coeffs and right.coeffs:
                raise ValueError
            if left.coeffs:
                return linear_scale(left, right.const)
            return linear_scale(right, left.const)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if right.coeffs or abs(right.const) <= NUMERIC_TOLERANCE:
                raise ValueError
            return linear_scale(left, 1.0 / right.const)
        raise ValueError

    try:
        return evaluate(parsed)
    except Exception:
        return None


def solve_linear_equality(left_member: str, right_member: str, bindings: dict[str, float]) -> tuple[str, float] | None:
    left = linear_form(left_member, bindings)
    right = linear_form(right_member, bindings)
    if left is None or right is None:
        return None
    diff = linear_add(left, right, -1.0)
    if len(diff.coeffs) != 1:
        return None
    name, coeff = next(iter(diff.coeffs.items()))
    if abs(coeff) <= NUMERIC_TOLERANCE:
        return None
    value = -diff.const / coeff
    if not math.isfinite(value):
        return None
    return name, value


def token_is_number(token: str) -> bool:
    return re.fullmatch(r"\d+(?:\.\d+)?", token) is not None


def token_is_bound_name(token: str, bindings: dict[str, float] | None) -> bool:
    return NAME_RE.match(token) is not None and bool(bindings) and token in bindings


def has_unbound_name(expression: str, bindings: dict[str, float] | None) -> bool:
    for token_match in TOKEN_RE.finditer(expression):
        token = token_match.group(0)
        if NAME_RE.match(token) and not token_is_bound_name(token, bindings):
            return True
    return False


def arithmetic_value(member: str, bindings: dict[str, float] | None = None) -> float | None:
    tokens = list(TOKEN_RE.finditer(member))
    for token_index, token_match in enumerate(tokens):
        candidate = member[token_match.start() : tokens[-1].end()].strip()
        if not candidate:
            continue
        token = token_match.group(0)
        if token_index > 0 and token in {"+", "-", "*", "/", "%", "^"}:
            continue
        value = safe_eval(candidate, bindings)
        if value is not None:
            return value
        if token_is_number(token) or token == "(" or token_is_bound_name(token, bindings):
            if has_unbound_name(candidate, bindings):
                return None
    return None


def expand_relative_member(member: str, prior_value: float | None, bindings: dict[str, float] | None = None) -> str:
    if prior_value is None:
        return member
    operator_match = re.search(r"[+*/%]", member)
    if not operator_match:
        return member
    prefix = member[:operator_match.start()]
    for token_match in TOKEN_RE.finditer(prefix):
        token = token_match.group(0)
        if token_is_number(token):
            return member
        if NAME_RE.match(token):
            if token_is_bound_name(token, bindings) or len(token) <= 2:
                return member
    return f"{prior_value:g}{member[operator_match.start():]}"


def simple_name(member: str) -> str | None:
    tokens = [token_match.group(0) for token_match in TOKEN_RE.finditer(member)]
    if len(tokens) == 1 and NAME_RE.match(tokens[0]):
        return tokens[0]
    if len(tokens) >= 1 and NAME_RE.match(tokens[-1]):
        prefix = member[: member.rfind(tokens[-1])].strip().lower()
        if prefix in {"", "let", "so", "then", "therefore"}:
            return tokens[-1]
    return None


def verify_step(text: str, bindings: dict[str, float] | None = None, prior_value: float | None = None) -> list[Claim]:
    current_bindings = dict(bindings or {})
    claims: list[Claim] = []
    for raw_clause in CLAUSE_RE.split(text):
        clause = raw_clause.strip()
        if "=" not in clause:
            continue
        members = [member.strip() for member in clause.split("=")]
        if len(members) < 2:
            continue
        members = [expand_relative_member(member, prior_value if member_index == 0 else None, current_bindings)
                   for member_index, member in enumerate(members)]
        values = [
            value
            for value in (arithmetic_value(member, current_bindings) for member in members)
            if value is not None
        ]
        if len(values) >= 2:
            baseline = values[0]
            ok = all(abs(value - baseline) <= NUMERIC_TOLERANCE for value in values)
            claims.append(Claim(clause=clause, values=values, ok=ok))
    return claims


def infer_bindings_from_clause(clause: str, bindings: dict[str, float]) -> None:
    if "=" not in clause:
        return
    members = [member.strip() for member in clause.split("=")]
    for left_member, right_member in zip(members, members[1:]):
        left_name = simple_name(left_member)
        right_name = simple_name(right_member)
        right_value = arithmetic_value(right_member, bindings)
        left_value = arithmetic_value(left_member, bindings)
        if left_name and right_value is not None:
            bindings[left_name] = right_value
        elif right_name and left_value is not None:
            bindings[right_name] = left_value
        solved = solve_linear_equality(left_member, right_member, bindings)
        if solved:
            name, value = solved
            bindings.setdefault(name, value)


def count_claims(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for claim_match in re.finditer(r"(?:count|answer|total|number)\s+is\s+(\d+)", text, re.IGNORECASE):
        claim_prefix = text[max(0, claim_match.start() - 220) : claim_match.start()]
        listed_match = re.search(r"are\s+((?:\d+\s*,\s*)+(?:and\s+)?\d+)", claim_prefix, re.IGNORECASE)
        if not listed_match:
            continue
        items = re.findall(r"\d+", listed_match.group(1))
        stated = int(claim_match.group(1))
        actual = len(items)
        out.append(
            {
                "claim": claim_match.group(0),
                "stated": stated,
                "actual": actual,
                "ok": stated == actual,
            }
        )
    return out


def scan_trace(text: str, initial_bindings: dict[str, float] | None = None) -> ScanResult:
    bindings = dict(initial_bindings or {})
    claims: list[Claim] = []
    flags: list[Claim] = []
    prior_value: float | None = None
    for raw_clause in CLAUSE_RE.split(text):
        clause = raw_clause.strip()
        if not clause:
            continue
        clause_claims = verify_step(clause, bindings, prior_value)
        clause_flags = [claim for claim in clause_claims if not claim.ok]
        claims.extend(clause_claims)
        flags.extend(clause_flags)
        if clause_claims and clause_claims[-1].values:
            prior_value = clause_claims[-1].values[-1]
        if not clause_flags:
            infer_bindings_from_clause(clause, bindings)
    count_flags = [claim for claim in count_claims(text) if not claim["ok"]]
    return ScanResult(claims=claims, flags=flags, bindings=bindings, count_flags=count_flags)


def select_first_verified(candidates: list[str], initial_bindings: dict[str, float] | None = None) -> tuple[int, ScanResult] | None:
    for candidate_index, candidate in enumerate(candidates):
        result = scan_trace(candidate, initial_bindings)
        if not result.flags and not result.count_flags:
            return candidate_index, result
    return None


def parse_bindings(raw_bindings: list[str]) -> dict[str, float]:
    bindings: dict[str, float] = {}
    for raw_binding in raw_bindings:
        if "=" not in raw_binding:
            raise SystemExit(f"binding must be NAME=VALUE: {raw_binding}")
        name, value_text = raw_binding.split("=", 1)
        name = name.strip()
        if not NAME_RE.match(name):
            raise SystemExit(f"invalid binding name: {name}")
        value = safe_eval(value_text, bindings)
        if value is None:
            raise SystemExit(f"invalid binding value: {raw_binding}")
        bindings[name] = value
    return bindings


def claim_to_json(claim: Claim) -> dict[str, Any]:
    return {"clause": claim.clause, "values": claim.values, "ok": claim.ok}


def result_to_json(result: ScanResult) -> dict[str, Any]:
    return {
        "claims": [claim_to_json(claim) for claim in result.claims],
        "flags": [claim_to_json(claim) for claim in result.flags],
        "bindings": result.bindings,
        "count_flags": result.count_flags,
    }


def run_self_test() -> int:
    cases = [
        ("v = 18. t = 14. D = v * t = 18", 1),
        ("v = 18. t = 14. D = v * t = 252", 0),
        ("14v = 252. v = 19", 1),
        ("14v = 252. v = 18", 0),
        ("2x + 3 = 17. x = 8", 1),
        ("2x + 3 = 17. x = 7", 0),
        ("7*11=77, then +2=79", 0),
        ("7*11=77, then +2=80", 1),
        ("5 * 5 = 25, then 25 + 10 = 30", 1),
        ("The multiples of 12 that are less than 100 are 12, 24, 36, 48, 60, 72, 84, and 96, so the count is 8", 0),
        ("the multiples are 12, 24, 36, 48, 60, 72, 84, and 96, so the count is 9", 1),
    ]
    failures = 0
    for text, expected_flags in cases:
        result = scan_trace(text)
        observed_flags = len(result.flags) + len(result.count_flags)
        ok = observed_flags == expected_flags
        print(f"selftest flags={observed_flags} expected={expected_flags} {'OK' if ok else 'FAIL'} :: {text}")
        if not ok:
            failures += 1
    selected = select_first_verified(["D = 18 * 14 = 18", "D = 18 * 14 = 252"])
    if not selected or selected[0] != 1:
        print("selftest select_first_verified FAIL")
        failures += 1
    else:
        print("selftest select_first_verified OK")
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="DS4 arithmetic verifier/selector for verifier-guided AIME branch tests.")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--scan-file", type=Path)
    parser.add_argument("--select-file", type=Path, action="append", default=[])
    parser.add_argument("--binding", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    bindings = parse_bindings(args.binding)
    if args.select_file:
        candidates = [path.read_text(encoding="utf-8") for path in args.select_file]
        selected = select_first_verified(candidates, bindings)
        if args.json:
            payload = {"selected": None}
            if selected:
                payload = {"selected": selected[0], "result": result_to_json(selected[1])}
            print(json.dumps(payload, indent=2, sort_keys=True))
        elif selected:
            print(f"selected={selected[0]} path={args.select_file[selected[0]]}")
        else:
            print("selected=None")
        return 0 if selected else 2

    if args.scan_file:
        result = scan_trace(args.scan_file.read_text(encoding="utf-8"), bindings)
        if args.json:
            print(json.dumps(result_to_json(result), indent=2, sort_keys=True))
        else:
            for claim in result.claims:
                status = "OK" if claim.ok else "FLAG"
                print(f"{status}\t{claim.values}\t{claim.clause}")
            for count_flag in result.count_flags:
                print(f"FLAG\tcount stated={count_flag['stated']} actual={count_flag['actual']}\t{count_flag['claim']}")
            print(f"bindings={result.bindings}")
        return 1 if result.flags or result.count_flags else 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
