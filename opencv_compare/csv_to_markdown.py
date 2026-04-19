#!/usr/bin/env python3

import argparse
import csv
import datetime as dt
import statistics
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render cvh vs OpenCV compare CSV to Markdown")
    p.add_argument("--input", required=True, help="Input CSV path")
    p.add_argument("--output", required=True, help="Output Markdown path")
    p.add_argument("--meta", default="", help="Optional metadata JSON path")
    p.add_argument("--title", default="cvh vs OpenCV Benchmark Report", help="Markdown title")
    return p.parse_args()


def to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def read_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def md_table(headers, rows):
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def render_report(rows, title: str, input_path: Path, meta_path: Optional[Path] = None) -> str:
    supported = [r for r in rows if r.get("status", "") == "OK"]
    unsupported = [r for r in rows if r.get("status", "") != "OK"]

    speedups = [to_float(r.get("speedup", "0")) for r in supported if to_float(r.get("speedup", "0")) > 0.0]
    cvh_faster = sum(1 for x in speedups if x > 1.0)
    opencv_faster_or_equal = sum(1 for x in speedups if x <= 1.0)
    avg_speedup = statistics.mean(speedups) if speedups else 0.0
    median_speedup = statistics.median(speedups) if speedups else 0.0

    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Generated at (UTC): `{generated_at}`")
    lines.append("")
    lines.append(f"Source CSV: `{input_path}`")
    lines.append("")

    if meta_path and meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        lines.append("## Run Config")
        lines.append("")
        lines.append(f"- Profile: `{meta.get('profile', 'unknown')}`")
        lines.append(
            f"- Samples: `warmup={meta.get('warmup', 'n/a')}, iters={meta.get('iters', 'n/a')}, repeats={meta.get('repeats', 'n/a')}`"
        )
        lines.append(f"- Threads: `{meta.get('threads', 'n/a')}`")
        lines.append(
            f"- Runtime: `omp_dynamic={meta.get('omp_dynamic', 'n/a')}, omp_proc_bind={meta.get('omp_proc_bind', 'n/a')}`"
        )
        lines.append(f"- Host: `{meta.get('system', 'n/a')} {meta.get('arch', 'n/a')}`")
        lines.append(f"- CPU: `{meta.get('cpu_model', 'n/a')}`")
        lines.append(f"- Build type: `{meta.get('build_type', 'n/a')}`")
        lines.append(f"- Compare mode: `{meta.get('compare_mode', 'n/a')}`")
        lines.append(f"- Meta JSON: `{meta_path}`")
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total rows: `{len(rows)}`")
    lines.append(f"- Supported rows (`status=OK`): `{len(supported)}`")
    lines.append(f"- Unsupported rows: `{len(unsupported)}`")
    lines.append(f"- Mean speedup (`OpenCV/CVH`): `{avg_speedup:.4f}`")
    lines.append(f"- Median speedup (`OpenCV/CVH`): `{median_speedup:.4f}`")
    lines.append(f"- Cases where CVH is faster (`OpenCV/CVH > 1`): `{cvh_faster}`")
    lines.append(f"- Cases where OpenCV is faster or equal (`OpenCV/CVH <= 1`): `{opencv_faster_or_equal}`")
    lines.append("")

    lines.append("## Supported Cases")
    lines.append("")
    if supported:
        supported_sorted = sorted(
            supported,
            key=lambda r: (
                r.get("op", ""),
                r.get("depth", ""),
                int(r.get("channels", "0")),
                r.get("shape", ""),
            ),
        )
        table_rows = []
        for r in supported_sorted:
            table_rows.append(
                [
                    r.get("op", ""),
                    r.get("depth", ""),
                    r.get("channels", ""),
                    r.get("shape", ""),
                    f"{to_float(r.get('cvh_ms', '0')):.6f}",
                    f"{to_float(r.get('opencv_ms', '0')):.6f}",
                    f"{to_float(r.get('speedup', '0')):.6f}",
                ]
            )
        lines.append(
            md_table(
                ["Op", "Depth", "Ch", "Shape", "CVH (ms)", "OpenCV (ms)", "OpenCV/CVH"],
                table_rows,
            )
        )
    else:
        lines.append("No supported rows.")
    lines.append("")

    lines.append("## Unsupported Cases")
    lines.append("")
    if unsupported:
        unsupported_sorted = sorted(
            unsupported,
            key=lambda r: (
                r.get("op", ""),
                r.get("depth", ""),
                int(r.get("channels", "0")),
                r.get("shape", ""),
            ),
        )
        table_rows = []
        for r in unsupported_sorted:
            table_rows.append(
                [
                    r.get("op", ""),
                    r.get("depth", ""),
                    r.get("channels", ""),
                    r.get("shape", ""),
                    f"{to_float(r.get('opencv_ms', '0')):.6f}",
                    r.get("status", ""),
                    r.get("note", ""),
                ]
            )
        lines.append(
            md_table(
                ["Op", "Depth", "Ch", "Shape", "OpenCV (ms)", "Status", "Note"],
                table_rows,
            )
        )
    else:
        lines.append("No unsupported rows.")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Speedup column is `OpenCV/CVH`; values `< 1` mean OpenCV is faster for that case.")
    lines.append("- This report is generated automatically from the compare CSV.")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise SystemExit(f"input CSV not found: {in_path}")

    rows = read_rows(in_path)
    meta_path = Path(args.meta) if args.meta else None
    report = render_report(rows, args.title, in_path, meta_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"markdown_report_written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
