#!/usr/bin/env python3
"""
Compare imgproc benchmark CSV files and fail on significant slowdown.

CSV columns expected:
profile,op,depth,channels,shape,elements,ms_per_iter,melems_per_sec,gb_per_sec
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple


BenchKey = Tuple[str, str, str, str, str]


@dataclass
class BenchRow:
    key: BenchKey
    ms_per_iter: float


def load_csv(path: pathlib.Path) -> Dict[BenchKey, BenchRow]:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")

    rows: Dict[BenchKey, BenchRow] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row["profile"].strip(),
                row["op"].strip(),
                row["depth"].strip(),
                row["channels"].strip(),
                row["shape"].strip(),
            )
            rows[key] = BenchRow(key=key, ms_per_iter=float(row["ms_per_iter"]))
    return rows


def format_key(key: BenchKey) -> str:
    profile, op, depth, channels, shape = key
    return f"{profile:<5} {op:<20} {depth:<7} c={channels:<2} shape={shape}"


@dataclass(frozen=True)
class SlowdownRule:
    selector: str
    max_slowdown: float


def normalize_token(token: str) -> str:
    return token.strip().upper()


def parse_slowdown_rule(text: str) -> SlowdownRule:
    if "=" not in text:
        raise ValueError(f"invalid rule '{text}', expected SELECTOR=RATIO")
    selector_raw, ratio_raw = text.split("=", 1)
    selector_raw = selector_raw.strip()
    ratio_raw = ratio_raw.strip()
    if not selector_raw:
        raise ValueError(f"invalid rule '{text}', empty selector")

    try:
        ratio = float(ratio_raw)
    except ValueError as exc:
        raise ValueError(f"invalid ratio in rule '{text}'") from exc
    if ratio < 0.0:
        raise ValueError(f"ratio must be >= 0, got {ratio} in rule '{text}'")

    selector = normalize_token(selector_raw)
    if ":" not in selector:
        if selector.startswith("CV_"):
            selector = f"*:{selector}"
        else:
            selector = f"{selector}:*"

    left, right = selector.split(":", 1)
    left = left.strip() or "*"
    right = right.strip() or "*"

    if left != "*" and not re.match(r"^[A-Z0-9_]+$", left):
        raise ValueError(f"invalid op selector '{left}' in rule '{text}'")
    if right != "*" and not re.match(r"^CV_[A-Z0-9]+$", right):
        raise ValueError(f"invalid depth selector '{right}' in rule '{text}'")

    return SlowdownRule(selector=f"{left}:{right}", max_slowdown=ratio)


def resolve_threshold(
    op: str,
    depth: str,
    default_ratio: float,
    exact_rules: Dict[Tuple[str, str], float],
    op_rules: Dict[str, float],
    depth_rules: Dict[str, float],
) -> float:
    op_key = normalize_token(op)
    depth_key = normalize_token(depth)
    if (op_key, depth_key) in exact_rules:
        return exact_rules[(op_key, depth_key)]
    if op_key in op_rules:
        return op_rules[op_key]
    if depth_key in depth_rules:
        return depth_rules[depth_key]
    return default_ratio


def main() -> int:
    parser = argparse.ArgumentParser(description="Check imgproc benchmark regression against baseline CSV")
    parser.add_argument("--baseline", required=True, type=pathlib.Path, help="baseline CSV file")
    parser.add_argument("--current", required=True, type=pathlib.Path, help="current CSV file")
    parser.add_argument(
        "--max-slowdown",
        type=float,
        default=0.08,
        help="maximum allowed slowdown ratio (default: 0.08 == 8%%)",
    )
    parser.add_argument(
        "--max-slowdown-by-op-depth",
        action="append",
        default=[],
        metavar="RULE",
        help=(
            "override slowdown threshold for specific slices. "
            "Format: OP:DEPTH=RATIO, OP=RATIO, or CV_DEPTH=RATIO. "
            "Repeatable."
        ),
    )
    parser.add_argument(
        "--allow-missing-current",
        action="store_true",
        help="do not fail when baseline key is missing in current CSV",
    )
    args = parser.parse_args()

    baseline = load_csv(args.baseline)
    current = load_csv(args.current)

    rules: List[SlowdownRule] = []
    for text in args.max_slowdown_by_op_depth:
        try:
            rules.append(parse_slowdown_rule(text))
        except ValueError as exc:
            print(f"[imgproc-benchmark-regression] invalid --max-slowdown-by-op-depth: {exc}")
            return 2

    exact_rules: Dict[Tuple[str, str], float] = {}
    op_rules: Dict[str, float] = {}
    depth_rules: Dict[str, float] = {}
    for rule in rules:
        op_selector, depth_selector = rule.selector.split(":", 1)
        if op_selector != "*" and depth_selector != "*":
            exact_rules[(op_selector, depth_selector)] = rule.max_slowdown
        elif op_selector != "*" and depth_selector == "*":
            op_rules[op_selector] = rule.max_slowdown
        elif op_selector == "*" and depth_selector != "*":
            depth_rules[depth_selector] = rule.max_slowdown

    failures = []
    missing = []
    improved = 0
    compared = 0

    for key, base_row in baseline.items():
        cur_row = current.get(key)
        if cur_row is None:
            missing.append(key)
            continue

        compared += 1
        base = base_row.ms_per_iter
        cur = cur_row.ms_per_iter
        if base <= 0.0:
            continue

        _profile, op, depth, _channels, _shape = key
        max_allowed = resolve_threshold(
            op=op,
            depth=depth,
            default_ratio=args.max_slowdown,
            exact_rules=exact_rules,
            op_rules=op_rules,
            depth_rules=depth_rules,
        )
        ratio = cur / base - 1.0
        if ratio > max_allowed:
            failures.append((key, base, cur, ratio, max_allowed))
        elif ratio < 0.0:
            improved += 1

    if missing and not args.allow_missing_current:
        print("[imgproc-benchmark-regression] missing cases in current CSV:")
        for key in missing[:20]:
            print("  -", format_key(key))
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        print("Set --allow-missing-current to ignore missing cases.")
        return 2

    print(
        f"[imgproc-benchmark-regression] compared={compared}, improved_or_equal={compared - len(failures)}, "
        f"improved={improved}, default_threshold={args.max_slowdown * 100:.2f}%, "
        f"override_rules={len(rules)}"
    )

    if failures:
        failures.sort(key=lambda item: item[3], reverse=True)
        print("[imgproc-benchmark-regression] slowdown violations:")
        for key, base, cur, ratio, max_allowed in failures[:30]:
            print(
                f"  - {format_key(key)} | baseline={base:.6f} ms | current={cur:.6f} ms "
                f"| slowdown={ratio * 100:.2f}% | allowed={max_allowed * 100:.2f}%"
            )
        if len(failures) > 30:
            print(f"  ... and {len(failures) - 30} more")
        return 1

    print("[imgproc-benchmark-regression] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
