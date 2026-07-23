# OpenCV Compare Workspace

This directory is dedicated to **cvh vs official OpenCV** speed comparison.

This workspace is an internal benchmark harness. Its `native` / `lite`
implementation names are legacy compare modes used by the scripts and CSV
schema; they are not public product targets. The public product targets remain
`cvh::headers` and `cvh::headers_fast`.

## Reports

- Quick report: [opencv_compare_quick.md](opencv_compare_quick.md)
- Stable report: [opencv_compare_stable.md](opencv_compare_stable.md)
- Baseline stable report: [opencv_compare_baseline_stable.md](opencv_compare_baseline_stable.md)

All reports are generated in bilingual format (English + 中文).

## Layout

- `setup_opencv_bench_slim.sh`: shallow clone/update `opencv-bench-slim` and optionally build it.
- `run_compare.sh`: one command for setup + configure + build + run compare benchmark.
- `csv_to_markdown.py`: render compare CSV into Markdown report.
- `results/`: generated CSV outputs (ignored by git).
- `opencv-bench-slim/`: shallow clone of the external OpenCV slim repo (ignored by git).
- `*.meta.json`: runtime metadata/fingerprint generated alongside compare CSV.

## External OpenCV Source

Default source repo and branch:

- Repo: `https://github.com/zihaomu/opencv.git`
- Branch: `opencv-bench-slim-v4.13`
- Clone mode: `--depth=1`

You can override with environment variables:

- `CVH_OPENCV_REPO`
- `CVH_OPENCV_BRANCH`
- `CVH_OPENCV_DIR`
- `CVH_OPENCV_BUILD_DIR`
- `CVH_COMPARE_BUILD_OPENCV` (`auto|0|1`, default `auto`)
- `CVH_COMPARE_RENDER_MD` (`0` means skip Markdown generation in `run_compare.sh`)
- `CVH_COMPARE_BUILD_TYPE` (`Release|RelWithDebInfo|Debug`, default `Release`)
- `CVH_COMPARE_OUTPUT_MD` (override Markdown output path)
- `CVH_COMPARE_OUTPUT_META` (override metadata JSON output path)
- `CVH_COMPARE_IMPLS` (`native|lite|native,lite`, default `native,lite`; internal compare modes only; `full` is a deprecated alias for `native`)

## Compare Profiles

`run_compare.sh` supports three profiles:

- `quick`: `warmup=1, iters=5, repeats=1`
- `stable`: `warmup=2, iters=20, repeats=5`
- `full`: `warmup=1, iters=10, repeats=3`

These defaults can be overridden by `CVH_COMPARE_WARMUP/ITERS/REPEATS` or CLI flags.

## Typical Workflow

1. Setup and build slim OpenCV:

```bash
./benchmark/opencv_compare/setup_opencv_bench_slim.sh --build
```

2. Run compare benchmark and output CSV:

```bash
./benchmark/opencv_compare/run_compare.sh --profile quick
```

Run only one implementation mode:

```bash
./benchmark/opencv_compare/run_compare.sh --profile quick --impls native
./benchmark/opencv_compare/run_compare.sh --profile quick --impls lite
```

These names select benchmark binaries inside this workspace. They should not be
used in README/API documentation as replacement names for `cvh::headers` or
`cvh::headers_fast`.

3. Generate/update baseline CSV + Markdown + metadata:

```bash
./benchmark/opencv_compare/run_compare.sh --profile stable --baseline
```

Default CSV path:

- `benchmark/opencv_compare/results/current_compare_quick.csv`
  - includes `impl` column to distinguish internal `native` and `lite` compare rows.

Default Markdown path:

- `benchmark/opencv_compare/opencv_compare_quick.md`

Default metadata path:

- `benchmark/opencv_compare/results/current_compare_quick.csv.meta.json`

## CMake Switches

The main repo adds:

- `CVH_ENABLE_OPENCV_COMPARE` (default `OFF`)
- `CVH_OPENCV_BENCH_DIR` (default `benchmark/opencv_compare/opencv-bench-slim`)

When compare is enabled, internal benchmark targets `cvh_benchmark_compare`
(`native` mode) and `cvh_benchmark_compare_lite` (`lite` mode) are built. These
targets are not part of the public package surface.

## Bench Scope

Current compare benchmark includes:

- `ADD`, `SUB`, `GEMM`, `GEMM_PREPACK` (fixed-`B` pack-once scenario)
- `GAUSSIAN_3X3`, `GAUSSIAN_5X5`, `GAUSSIAN_11X11`
- `BOX_3X3`, `BOX_5X5`, `BOX_11X11`
- `COPYMAKEBORDER`
- `LUT`
- `FILTER2D_3X3`
- `SEPFILTER2D_3X3`
- `WARP_AFFINE`
- `SOBEL`, `ERODE`, `DILATE`（`C1/C3/C4`）
- `CANNY_A3_L1`, `CANNY_A3_L2`, `CANNY_A5_L1`, `CANNY_A5_L2`（`CV_8UC1`）
