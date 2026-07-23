#!/usr/bin/env python3

import argparse
import csv
import datetime as dt
import json
import math
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


def row_suite(row: dict) -> str:
    suite = (row.get("suite", "") or "").strip().lower()
    if suite:
        return suite
    return "core_mat" if (row.get("op", "") or "").startswith("MAT_") else "imgproc"


def geometric_mean(values) -> float:
    positive = [x for x in values if x > 0.0]
    if not positive:
        return 0.0
    return math.exp(sum(math.log(x) for x in positive) / len(positive))


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
    meta = {}
    if meta_path and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    speedups = [to_float(r.get("speedup", "0")) for r in supported if to_float(r.get("speedup", "0")) > 0.0]
    cvh_faster = sum(1 for x in speedups if x > 1.0)
    opencv_faster_or_equal = sum(1 for x in speedups if x <= 1.0)
    geo_speedup = geometric_mean(speedups)
    median_speedup = statistics.median(speedups) if speedups else 0.0

    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"生成时间（UTC）：`{generated_at}`")
    lines.append("")
    lines.append("## 当前项目状态")
    lines.append("")
    lines.append("- `opencv-header-only` 当前公共定位是纯 header-only，不依赖项目内 `.cpp` 扩展层。")
    lines.append("- Mode B 只比较当前 `cvh::headers_fast` 与同机编译的 upstream OpenCV；`cvh::headers_fast` 表示最快 header-only 构建配置。")
    lines.append("- `cvh::headers_fast` 完整继承 `cvh::headers`。算子没有专用 fast-path 时继续执行继承的 header 实现并参与 benchmark，不因缺少 SIMD 特化而跳过。")
    lines.append("- OpenCV Universal Intrinsics 是默认 SIMD 方言，kernel 直接使用 OpenCV UI；项目已移除 xsimd 性能路径。")
    lines.append("- ARM 当前关注 NEON，本次实测平台为 Apple ARM；x86 目标是 SSE/AVX 系列，RVV 因 scalable vector 设计问题暂缓。")
    lines.append("- 当前已验证的 UI 专用路径主要包括 `BGR/RGB2GRAY` 和 `CV_8UC1` 精确 2x `INTER_LINEAR` 下采样；其余本报告算子多数仍走继承的 header baseline。")
    lines.append("")

    lines.append("## 高层优化结构")
    lines.append("")
    lines.append("| 层次 | 当前实现 | 本报告中的含义 |")
    lines.append("| --- | --- | --- |")
    lines.append("| 公共 API | OpenCV-compatible header API | 所有 case 均从 `cvh::headers_fast` 公共入口调用 |")
    lines.append("| SIMD 方言 | OpenCV Universal Intrinsics | 在 Apple ARM 上映射到 NEON |")
    lines.append("| 专用 kernel | `cvtColor`、特定 `resize` UI kernel | 记录为 `dispatch_path=opencv_ui` |")
    lines.append("| 通用实现 | `cvh::headers` 中的 header baseline | 无专用 fast-path 时自动继承，记录为 `headers_baseline` |")
    lines.append("| 对照实现 | upstream OpenCV `core` / `imgproc` | 相同输入、尺寸、border 和线程配置 |")
    lines.append("")

    if meta:
        lines.append("## 运行配置")
        lines.append("")
        lines.append(f"- Profile：`{meta.get('profile', 'unknown')}`")
        meta_impls = meta.get("impls", [])
        if isinstance(meta_impls, list) and meta_impls:
            lines.append(f"- CVH 实现：`{', '.join(meta_impls)}`")
        lines.append(
            f"- 采样：`warmup={meta.get('warmup', 'n/a')}, iters={meta.get('iters', 'n/a')}, repeats={meta.get('repeats', 'n/a')}`"
        )
        lines.append(f"- 线程数：`{meta.get('threads', 'n/a')}`")
        lines.append(
            f"- OpenMP：`dynamic={meta.get('omp_dynamic', 'n/a')}, proc_bind={meta.get('omp_proc_bind', 'n/a')}`"
        )
        lines.append(f"- 主机：`{meta.get('system', 'n/a')} {meta.get('arch', 'n/a')}`")
        lines.append(f"- CPU：`{meta.get('cpu_model', 'n/a')}`")
        lines.append(f"- 编译器：`{meta.get('compiler', 'n/a')}`")
        lines.append(f"- 构建类型：`{meta.get('build_type', 'n/a')}`")
        lines.append(
            f"- CVH commit：`{meta.get('repo_git_commit', 'unknown')}`"
            f"{' + dirty' if meta.get('repo_git_dirty', False) else ''}"
        )
        lines.append(
            f"- OpenCV：`{meta.get('opencv_version', 'unknown')}`，commit "
            f"`{meta.get('opencv_git_commit', 'unknown')}`"
            f"{' + dirty' if meta.get('opencv_git_dirty', False) else ''}"
        )
        lines.append(f"- 原始数据：`{input_path.name}`；元数据：`{meta_path.name if meta_path else 'n/a'}`")
        lines.append("")

    lines.append("## 汇总")
    lines.append("")
    lines.append(f"- 总 case：`{len(rows)}`；有效：`{len(supported)}`；不支持：`{len(unsupported)}`。")
    lines.append(f"- `OpenCV/CVH` 几何平均：`{geo_speedup:.4f}`；中位数：`{median_speedup:.4f}`。")
    lines.append(f"- CVH 更快：`{cvh_faster}` 个；OpenCV 更快或相当：`{opencv_faster_or_equal}` 个。")
    lines.append("")

    suite_summary_rows = []
    for suite in ("core_mat", "imgproc"):
        suite_speedups = [
            to_float(r.get("speedup", "0"))
            for r in supported
            if row_suite(r) == suite and to_float(r.get("speedup", "0")) > 0.0
        ]
        if suite_speedups:
            suite_summary_rows.append(
                [
                    suite,
                    str(len(suite_speedups)),
                    f"{geometric_mean(suite_speedups):.4f}",
                    f"{statistics.median(suite_speedups):.4f}",
                    str(sum(1 for x in suite_speedups if x > 1.0)),
                    str(sum(1 for x in suite_speedups if x <= 1.0)),
                ]
            )
    if suite_summary_rows:
        lines.append(
            md_table(
                ["Suite", "Cases", "几何平均 OpenCV/CVH", "中位数", "CVH 更快", "OpenCV 更快/相当"],
                suite_summary_rows,
            )
        )
        lines.append("")

    lines.append("## 算子级概览")
    lines.append("")
    for suite in ("core_mat", "imgproc"):
        suite_rows = [r for r in supported if row_suite(r) == suite]
        if not suite_rows:
            continue
        lines.append(f"### `{suite}`")
        lines.append("")
        op_names = sorted({r.get("op", "") for r in suite_rows})
        op_rows = []
        for op in op_names:
            values = [
                to_float(r.get("speedup", "0"))
                for r in suite_rows
                if r.get("op", "") == op and to_float(r.get("speedup", "0")) > 0.0
            ]
            dispatches = sorted({r.get("dispatch_path", "") or "unknown" for r in suite_rows if r.get("op", "") == op})
            ratio = geometric_mean(values)
            winner = f"CVH `{ratio:.2f}x`" if ratio > 1.0 else f"OpenCV `{(1.0 / ratio):.2f}x`"
            op_rows.append(
                [
                    op,
                    ", ".join(dispatches),
                    str(len(values)),
                    f"{ratio:.4f}",
                    winner,
                ]
            )
        lines.append(
            md_table(
                ["Op", "CVH dispatch", "Cases", "几何平均 OpenCV/CVH", "领先方"],
                op_rows,
            )
        )
        lines.append("")

    lines.append("## 详细结果")
    lines.append("")
    for suite in ("core_mat", "imgproc"):
        suite_rows = [r for r in supported if row_suite(r) == suite]
        if not suite_rows:
            continue
        lines.append(f"### `{suite}`")
        lines.append("")
        supported_sorted = sorted(
            suite_rows,
            key=lambda r: (
                r.get("op", ""),
                r.get("variant", ""),
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
                    r.get("variant", "") or "default",
                    r.get("dispatch_path", "") or "unknown",
                    r.get("depth", ""),
                    r.get("channels", ""),
                    r.get("shape", ""),
                    f"{to_float(r.get('cvh_ms', '0')):.6f}",
                    f"{to_float(r.get('opencv_ms', '0')):.6f}",
                    f"{to_float(r.get('speedup', '0')):.4f}",
                    r.get("note", ""),
                ]
            )
        lines.append(
            md_table(
                ["Op", "Variant", "CVH dispatch", "Depth", "Ch", "Shape", "CVH ms", "OpenCV ms", "OpenCV/CVH", "Note"],
                table_rows,
            )
        )
        lines.append("")

    if unsupported:
        lines.append("## 不支持用例")
        lines.append("")
        unsupported_rows = []
        for r in unsupported:
            unsupported_rows.append(
                [
                    row_suite(r),
                    r.get("op", ""),
                    r.get("variant", "") or "default",
                    r.get("shape", ""),
                    r.get("status", ""),
                    r.get("note", ""),
                ]
            )
        lines.append(md_table(["Suite", "Op", "Variant", "Shape", "Status", "Note"], unsupported_rows))
        lines.append("")

    lines.append("## 说明")
    lines.append("")
    lines.append("- 比值统一为 `OpenCV耗时 / CVH耗时`：大于 `1` 表示 CVH 更快，小于 `1` 表示 OpenCV 更快。")
    lines.append("- 表内耗时取各 repeat 的最小单次耗时，用于降低系统抖动影响；本报告不是跨机器排名。")
    lines.append("- Mat case 对比相同的分配/复用语义；imgproc case 对齐输入尺寸、类型、kernel、border 和主要参数。")
    lines.append("- `headers_baseline` 不等于跳过优化，它表示 `cvh::headers_fast` 当前继承了 `cvh::headers` 的通用实现。")
    lines.append("- 原始 CSV 和 metadata 是可再生成的运行产物，日期命名 Markdown 是阶段性快照。")

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
