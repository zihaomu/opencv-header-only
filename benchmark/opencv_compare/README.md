# OpenCV Compare Mode

This directory is the current workspace for comparing `cvh` with official
OpenCV. It belongs to Mode B in [../readme.md](../readme.md): current
header-only `cvh` versus OpenCV upstream.

## Target Design

Mode B intentionally uses only the fastest header-only profile on the `cvh`
side:

| Implementation | Meaning |
|---|---|
| `cvh_headers_fast` | Current `cvh::headers_fast`, representing the fastest header-only implementation. |
| `opencv` | Official OpenCV `core` / `imgproc` built on the same machine. |

The compare report is for visibility and prioritization. It is log-only by
default and should not block every PR.

Required metadata for every run:

- `cvh` git commit
- OpenCV git commit
- compiler and build type
- OS, arch, CPU
- thread count and runtime flags
- profile, warmup, iters, repeats
- OpenCV source/build directory

## Local OpenCV Source

For this workspace, the preferred OpenCV source tree is:

```text
/Users/zmu/work/my_project/ocvh/opencv
```

From the `opencv-header-only` repository root this is:

```text
../opencv
```

The full OpenCV tree should be built separately and passed to the compare
runner by environment variables or future CLI flags. Do not point the legacy
`setup_opencv_bench_slim.sh` clone/update flow at the full local OpenCV
checkout unless you explicitly want that script to manage the repo.

## Current Harness

Existing files:

- `setup_opencv_bench_slim.sh`: historical helper for a slim OpenCV clone.
- `run_compare.sh`: one-command runner for `cvh::headers_fast` versus
  upstream OpenCV.
- `csv_to_markdown.py`: render compare CSV into Markdown.
- `opencv_compare_header_benchmark.cpp`: pure header-only `cvh` compare cases.
- `opencv_compare_opencv_backend.cpp`: OpenCV-side implementation, compiled
  without `cvh::headers` include paths.

Current caveats:

- `cvh::headers` is intentionally not a Mode B compare implementation. It is
  useful for default header-only validation and internal regression, while
  Mode B should stay easy to read: fastest header-only `cvh` versus upstream
  OpenCV.
- Raw CSV/metadata and rolling `current_*` reports are generated artifacts.
  Curated date-named `*-opencv-upstream-performance.md` snapshots may be
  committed under `benchmark/opencv_compare/results/`.
- A missing `headers_fast` specialization is not an unsupported case:
  `cvh::headers_fast` inherits the `cvh::headers` implementation and the case
  remains in the report as `dispatch_path=headers_baseline`.

## Dated Snapshots

- [2026-07-23 OpenCV upstream performance](results/2026-07-23-opencv-upstream-performance.md):
  Apple M5, single-threaded stable profile, `core_mat` plus `imgproc`.

## Current Commands

Header-only quick run:

```bash
./benchmark/opencv_compare/run_compare.sh --profile quick
```

Use an existing local OpenCV build:

```bash
CVH_COMPARE_SKIP_OPENCV_SETUP=1 \
CVH_OPENCV_DIR=../opencv \
CVH_OPENCV_CONFIG_DIR=../opencv/build-slim \
./benchmark/opencv_compare/run_compare.sh --profile quick
```

Stable baseline:

```bash
./benchmark/opencv_compare/run_compare.sh --profile stable --baseline
```

Explicit implementation:

```bash
./benchmark/opencv_compare/run_compare.sh --profile quick --impls headers_fast
```

`cvh_headers_fast` is accepted as an alias for `headers_fast`.

## Coverage Status

- Stable covers the core compute matrix plus representative imgproc U8/F32
  C1/C3/C4 cases.
- Full adds odd-width and non-contiguous ROI cases plus representative
  I420/YUY2/NV12 layouts.
- Raw CSV and metadata stay generated under
  `benchmark/opencv_compare/results/`; date-named Markdown snapshots may be
  tracked.
- Missing upstream operations remain explicit `UNSUPPORTED` rows.
