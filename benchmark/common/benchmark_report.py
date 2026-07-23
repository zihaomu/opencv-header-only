#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def row_metric_ms(row: dict) -> float:
    for key in ("median_ms", "median_ms_per_call", "ms_per_iter", "cvh_ms"):
        value = row.get(key, "")
        if value not in ("", None):
            try:
                return float(value)
            except ValueError:
                continue
    return 0.0


def row_key(row: dict) -> Tuple[str, ...]:
    preferred = [
        "suite",
        "module",
        "op",
        "variant",
        "depth",
        "channels",
        "layout",
        "shape",
        "src_width",
        "src_height",
        "dst_width",
        "dst_height",
        "width",
        "height",
        "backend",
        "entry",
        "implementation",
        "dispatch_path",
        "allocation_mode",
    ]
    parts = []
    for key in preferred:
        parts.append(str(row.get(key, "")))
    return tuple(parts)


def short_case(row: dict) -> str:
    parts = []
    for key in ("suite", "op", "variant", "depth", "channels", "layout", "shape", "backend", "entry", "allocation_mode"):
        value = row.get(key, "")
        if value:
            parts.append(f"{key}={value}")
    return ", ".join(parts) if parts else str(row_key(row))


def geomean(values: Iterable[float]) -> float:
    vals = [v for v in values if v > 0.0 and math.isfinite(v)]
    if not vals:
        return 0.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def compare_internal(args: argparse.Namespace) -> int:
    baseline_rows = read_csv(Path(args.baseline))
    current_rows = read_csv(Path(args.current))
    baseline_by_key: Dict[Tuple[str, ...], dict] = {row_key(row): row for row in baseline_rows}
    current_by_key: Dict[Tuple[str, ...], dict] = {row_key(row): row for row in current_rows}

    missing = []
    comparisons = []
    failures = []
    for key, base in baseline_by_key.items():
        cur = current_by_key.get(key)
        if cur is None:
            missing.append(base)
            continue
        base_ms = row_metric_ms(base)
        cur_ms = row_metric_ms(cur)
        speedup = base_ms / cur_ms if base_ms > 0.0 and cur_ms > 0.0 else 0.0
        slowdown = (cur_ms / base_ms - 1.0) if base_ms > 0.0 and cur_ms > 0.0 else 0.0
        item = {
            "case": short_case(base),
            "baseline_ms": base_ms,
            "current_ms": cur_ms,
            "speedup": speedup,
            "slowdown": slowdown,
        }
        comparisons.append(item)
        if slowdown > args.max_slowdown:
            failures.append(item)

    comparisons.sort(key=lambda item: item["slowdown"], reverse=True)
    failures.sort(key=lambda item: item["slowdown"], reverse=True)
    speedups = [item["speedup"] for item in comparisons if item["speedup"] > 0.0]
    summary = {
        "mode": "internal_regression",
        "suite": args.suite,
        "baseline": args.baseline,
        "current": args.current,
        "compared": len(comparisons),
        "missing": len(missing),
        "failures": len(failures),
        "max_slowdown": args.max_slowdown,
        "geomean_speedup": geomean(speedups),
        "min_speedup": min(speedups) if speedups else 0.0,
        "max_speedup": max(speedups) if speedups else 0.0,
        "top_regressions": comparisons[:20],
        "missing_cases": [short_case(row) for row in missing[:50]],
    }

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.output_md:
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Internal Benchmark Regression: {args.suite}",
            "",
            f"- Baseline CSV: `{args.baseline}`",
            f"- Current CSV: `{args.current}`",
            f"- Compared cases: `{summary['compared']}`",
            f"- Missing cases: `{summary['missing']}`",
            f"- Failures: `{summary['failures']}`",
            f"- Max slowdown gate: `{args.max_slowdown:.2%}`",
            f"- Geomean speedup: `{summary['geomean_speedup']:.4f}`",
            "",
            "## Top Regressions",
            "",
        ]
        table_rows = [
            [
                item["case"],
                f"{item['baseline_ms']:.6f}",
                f"{item['current_ms']:.6f}",
                f"{item['speedup']:.4f}",
                f"{item['slowdown']:.2%}",
            ]
            for item in comparisons[:20]
        ]
        lines.append(markdown_table(["Case", "Baseline ms", "Current ms", "Speedup", "Slowdown"], table_rows))
        if missing:
            lines.extend(["", "## Missing Cases", ""])
            lines.extend(f"- {short_case(row)}" for row in missing[:50])
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        "benchmark_report: internal "
        f"suite={args.suite} compared={summary['compared']} missing={summary['missing']} "
        f"failures={summary['failures']} geomean_speedup={summary['geomean_speedup']:.4f}"
    )
    return 1 if failures or missing else 0


def render_opencv_compare(args: argparse.Namespace) -> int:
    rows = read_csv(Path(args.compare))
    supported = [row for row in rows if row.get("status", "OK") == "OK"]
    speedups = []
    for row in supported:
        try:
            speedups.append(float(row.get("speedup", "0") or "0"))
        except ValueError:
            pass
    summary = {
        "mode": "opencv_compare",
        "suite": args.suite,
        "compare": args.compare,
        "rows": len(rows),
        "supported": len(supported),
        "unsupported": len(rows) - len(supported),
        "geomean_opencv_over_cvh": geomean(speedups),
    }
    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# OpenCV Compare: {args.suite}",
            "",
            f"- Compare CSV: `{args.compare}`",
            f"- Rows: `{summary['rows']}`",
            f"- Supported: `{summary['supported']}`",
            f"- Unsupported: `{summary['unsupported']}`",
            f"- Geomean `OpenCV/CVH`: `{summary['geomean_opencv_over_cvh']:.4f}`",
            "",
        ]
        out_md.write_text("\n".join(lines), encoding="utf-8")
    print(
        "benchmark_report: opencv_compare "
        f"suite={args.suite} rows={summary['rows']} supported={summary['supported']} "
        f"geomean={summary['geomean_opencv_over_cvh']:.4f}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render benchmark reports and enforce internal regression gates.")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_internal = sub.add_parser("internal", help="Compare baseline/current CSV files.")
    p_internal.add_argument("--suite", required=True)
    p_internal.add_argument("--baseline", required=True)
    p_internal.add_argument("--current", required=True)
    p_internal.add_argument("--output-md", default="")
    p_internal.add_argument("--output-json", default="")
    p_internal.add_argument("--max-slowdown", type=float, default=0.08)

    p_opencv = sub.add_parser("opencv-compare", help="Render an OpenCV compare CSV.")
    p_opencv.add_argument("--suite", required=True)
    p_opencv.add_argument("--compare", required=True)
    p_opencv.add_argument("--output-md", default="")
    p_opencv.add_argument("--output-json", default="")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "internal":
        return compare_internal(args)
    if args.mode == "opencv-compare":
        return render_opencv_compare(args)
    raise SystemExit(f"unsupported mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
