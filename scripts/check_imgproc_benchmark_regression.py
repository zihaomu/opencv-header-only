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
import sys
from dataclasses import dataclass
from typing import Dict, Tuple


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
        "--allow-missing-current",
        action="store_true",
        help="do not fail when baseline key is missing in current CSV",
    )
    args = parser.parse_args()

    baseline = load_csv(args.baseline)
    current = load_csv(args.current)

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

        ratio = cur / base - 1.0
        if ratio > args.max_slowdown:
            failures.append((key, base, cur, ratio))
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
        f"improved={improved}, threshold={args.max_slowdown * 100:.2f}%"
    )

    if failures:
        failures.sort(key=lambda item: item[3], reverse=True)
        print("[imgproc-benchmark-regression] slowdown violations:")
        for key, base, cur, ratio in failures[:30]:
            print(
                f"  - {format_key(key)} | baseline={base:.6f} ms | current={cur:.6f} ms "
                f"| slowdown={ratio * 100:.2f}%"
            )
        if len(failures) > 30:
            print(f"  ... and {len(failures) - 30} more")
        return 1

    print("[imgproc-benchmark-regression] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

