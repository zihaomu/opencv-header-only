# Benchmark Refactor Implementation Plan

This plan turns the benchmark framework in `benchmark/readme.md` into concrete
implementation steps. The project direction is pure header-only: benchmark code
must not make `native` or compiled `.cpp` paths look like part of the product.

## Goals

- Build one benchmark framework for two suites:
  - `core_mat`: `Mat` lifetime, layout, copy, fill, conversion, and accepted
    basic array operators.
  - `imgproc`: image processing operators used by CV preprocessing and
    postprocessing.
- Support two benchmark modes:
  - Internal header-only regression: old `cvh` header-only vs current `cvh`
    header-only.
  - OpenCV upstream compare: current `cvh` header-only vs official OpenCV.
- Keep benchmark artifacts out of source directories.
- Make speed data actionable: every result should say which implementation,
  dispatch path, allocation mode, shape, and profile produced it.

## Non-goals

- Do not reintroduce xsimd as a benchmark candidate.
- Do not add public `native` benchmark language.
- Do not require compiled `.cpp` code for the header-only benchmark path.
- Do not make OpenCV compare a default hard gate for every local change.

## Current Starting Point

Pure header-only benchmark targets that already exist:

| Target | Current value |
|---|---|
| `cvh_benchmark_cvtcolor_bgr2gray_header` | Useful imgproc fast-path diagnostic for `BGR2GRAY` / `RGB2GRAY`. |
| `cvh_benchmark_resize_bilinear_header` | Useful imgproc fast-path diagnostic for `CV_8UC1` exact 2x `INTER_LINEAR`. |

Legacy benchmark targets that need migration:

| Target | Current issue |
|---|---|
| `cvh_benchmark_core_ops` | Links old compiled layer. |
| `cvh_benchmark_imgproc_ops` | Links old compiled layer. |
| `cvh_benchmark_imgproc_filter` | Links old compiled layer. |
| `cvh_benchmark_compare` / `cvh_benchmark_compare_lite` | Use legacy `native` / `lite` internal modes. |

Preferred local OpenCV source:

```text
/Users/zmu/work/my_project/ocvh/opencv
```

Relative to this repository:

```text
../opencv
```

## Target Layout

```text
benchmark/
  common/
    benchmark_common.h
    benchmark_csv.h
    benchmark_metadata.h
  internal/
    run_header_regression.sh
  opencv_compare/
    run_compare.sh
    csv_to_markdown.py
  results/
    internal/
    opencv/
```

The exact source file layout can change if CMake integration suggests a simpler
shape, but generated results must stay under `benchmark/results/` or
`benchmark/opencv_compare/results/`.

## Standard Result Model

New benchmark rows should converge on these fields:

| Field | Required | Meaning |
|---|---|---|
| `mode` | yes | `internal` or `opencv_compare`. |
| `suite` | yes | `core_mat` or `imgproc`. |
| `module` | yes | `core` or `imgproc`. |
| `op` | yes | Operator name. |
| `variant` | yes | Interpolation, border, kernel size, color code, or `default`. |
| `depth` | yes | `CV_8U`, `CV_32F`, etc. |
| `channels` | yes | Channel count. |
| `layout` | yes | `continuous`, `roi`, YUV layout, or `none`. |
| `shape` | yes | Human-readable shape. |
| `elements` | yes | Logical element count. |
| `pixels` | imgproc | Output pixel count. |
| `implementation` | yes | Mode A may use `cvh_headers`, `cvh_headers_fast`, `scalar_fallback`, etc.; Mode B only uses `cvh_headers_fast` and `opencv`. |
| `dispatch_path` | yes | Actual internal path. |
| `allocation_mode` | yes | `reuse`, `recreate`, or `none`. |
| `warmup`, `iters`, `repeats`, `threads` | yes | Sampling config. |
| `min_ms`, `median_ms` | yes | Timing metrics. |
| `mpix_per_sec`, `melems_per_sec`, `gb_per_sec` | when applicable | Throughput metrics. |
| `checksum` | yes | Lightweight output-use guard. |
| `status`, `note` | yes | Support status and skip reason. |

Existing CSV schemas can be adapted incrementally. New report/gate scripts
should normalize old rows into this model before comparing.

## Implementation Steps

### P-Bench-0: Documentation And Artifact Cleanup

Status: complete.

Purpose: establish the benchmark contract before changing code.

Tasks:

- Rewrite `benchmark/readme.md` as the framework entry.
- Add this implementation plan under `doc/`.
- Add `benchmark/results/` and `benchmark/opencv_compare/results/` as artifact
  directories.
- Ignore generated benchmark outputs.
- Remove tracked historical CSV/Markdown artifacts from source locations.
- Keep existing benchmark source files in place until replacement targets exist.

DoD:

- No README or doc link points at deleted generated reports.
- `benchmark/gate_policy.json` documents both modes and suites.
- `git diff --check` passes.

Completion notes:

- `benchmark/readme.md` is now the benchmark framework entry.
- `doc/benchmark-refactor-implementation-plan.md` records the implementation
  sequence.
- `benchmark/results/` and `benchmark/opencv_compare/results/` are the tracked
  artifact roots via `.gitkeep`; generated files under them are ignored.
- Historical root-level benchmark CSV files and generated OpenCV compare
  Markdown reports have been removed from source locations.
- Existing benchmark source files and CMake targets are intentionally kept in
  place for P-Bench-1 and later migrations.

### P-Bench-1: Shared Benchmark Utilities

Status: complete.

Purpose: stop duplicating timing, CSV, metadata, checksum, and argument parsing
logic across benchmark binaries.

Tasks:

- Add `benchmark/common/benchmark_common.h`.
- Provide:
  - stable timer helper using `std::chrono::steady_clock`
  - warmup/iters/repeats measurement helper
  - min/median calculation
  - deterministic data generation
  - checksum helpers for `CV_8U` and `CV_32F`
  - profile parsing for `quick`, `stable`, `full`, `micro`
  - CSV escaping/writing helpers
- Add `benchmark/common/benchmark_metadata.h` or equivalent script support for:
  - `cvh` commit
  - compiler id/version
  - OS/arch/CPU
  - CMake build type
  - thread/runtime settings
- Keep the helpers header-only.

DoD:

- Existing header-only `cvtColor` and `resize` benchmarks can include the common
  helpers without behavior changes.
- Python/C++ compile checks pass.

Completion notes:

- Added `benchmark/common/benchmark_common.h` for basic argument parsing,
  repeated timing, sample summary, FNV-1a checksum helpers, bytewise Mat
  comparison, deterministic `CV_8U` input fill, SIMD lane reporting, and MPix/s
  helper math.
- Added `benchmark/common/benchmark_csv.h` for CSV escaping and row writing.
- Added `benchmark/common/benchmark_metadata.h` for compiler/platform/runtime
  metadata JSON helpers.
- `cvh_benchmark_cvtcolor_bgr2gray_header` now uses the common parser, timing,
  checksum, input-fill, Mat compare, SIMD backend, and SIMD lane helpers.
- `cvh_benchmark_resize_bilinear_header` now uses the same common helpers while
  keeping its table-build micro timing local.
- Existing CSV schemas and column order are intentionally unchanged.

### P-Bench-2: Internal Header-only Regression Runner

Status: complete.

Purpose: compare old header-only `cvh` with current header-only `cvh` without
OpenCV and without compiled extension paths.

Tasks:

- Add `benchmark/internal/run_header_regression.sh`.
- Support:
  - `--baseline-ref <git-ref>`
  - `--suite core_mat|imgproc|all`
  - `--profile quick|stable|full|micro`
  - `--target headers|headers_fast`
  - `--build-type Release|RelWithDebInfo`
  - `--output-dir <path>`
- Use temporary `git worktree` for the baseline checkout.
- Build baseline and current with:

```bash
-DCVH_BUILD_NATIVE_BACKEND=OFF
-DCVH_BUILD_TESTS=OFF
-DCVH_BUILD_BENCHMARKS=ON
```

- Write:
  - `baseline.csv`
  - `current.csv`
  - `report.md`
  - `meta.json`
- Normalize result rows before comparing.

DoD:

- A single command can compare current checkout against an older commit.
- Runner refuses to use benchmark binaries that require compiled extension
  targets.
- Missing cases are reported explicitly.

Completion notes:

- Added `benchmark/internal/run_header_regression.sh`.
- The runner configures both baseline and current with
  `CVH_BUILD_NATIVE_BACKEND=OFF`, `CVH_BUILD_TESTS=OFF`, and
  `CVH_BUILD_BENCHMARKS=ON`.
- Supported suites are `core_mat`, `imgproc`, and `all`.
- The runner writes per-suite baseline/current CSV, Markdown report, JSON
  summary, and run metadata under `benchmark/results/internal/...`.
- Baseline refs must contain the new benchmark targets. Full end-to-end
  comparison becomes available after the P-Bench changes are committed.

### P-Bench-3: `core_mat` Header-only Benchmark

Status: complete.

Purpose: measure `Mat` baseline costs and accepted core operators through
header-only targets.

Tasks:

- Add target:

```text
cvh_benchmark_core_mat_header
```

- Link only `cvh::headers` or `cvh::headers_fast`.
- Start with `Mat` cases:
  - `Mat::create` no-op reuse
  - `Mat::create` reallocation
  - `release`
  - `clone`
  - `copyTo` continuous
  - `copyTo` ROI/non-contiguous
  - `setTo`
  - `convertTo`
  - `reshape`
- Add basic array ops only after they are accepted into the pure header-only
  contract.
- Cover small, medium, and large shapes:
  - tiny metadata-sensitive cases
  - `480x640`
  - `1080p`
  - `4K`
  - non-contiguous ROI
- Emit standard result rows.

DoD:

- Target builds with `CVH_BUILD_NATIVE_BACKEND=OFF`.
- Results include allocation mode and layout.
- Internal regression script can gate `core_mat`.

Completion notes:

- Added `benchmark/core_mat_header_benchmark.cpp`.
- Added CMake target `cvh_benchmark_core_mat_header`, linked only to
  `cvh::headers`.
- Initial coverage includes `Mat::create`, release/create, `clone`, `copyTo`
  continuous, `copyTo` ROI, `setTo`, `convertTo`, and `reshape`.
- Output uses the standard benchmark row model.

### P-Bench-4: `imgproc` Header-only Benchmark

Status: complete.

Purpose: unify existing imgproc diagnostics into one expandable header-only
suite.

Tasks:

- Add target:

```text
cvh_benchmark_imgproc_header
```

- Link only `cvh::headers` or `cvh::headers_fast`.
- Reuse logic from:
  - `cvh_benchmark_cvtcolor_bgr2gray_header`
  - `cvh_benchmark_resize_bilinear_header`
- Initial accepted cases:
  - `resize` `CV_8U` / `CV_32F`, `C1` / `C3` / `C4`,
    `INTER_NEAREST`, `INTER_NEAREST_EXACT`, `INTER_LINEAR`
  - `cvtColor` BGR/RGB/GRAY/BGRA/RGBA families
  - accepted YUV encode/decode layouts
  - `threshold`
  - `LUT`
  - `copyMakeBorder`
  - `filter2D`
  - `sepFilter2D`
  - `boxFilter` / `blur`
  - `GaussianBlur`
  - `Sobel`
  - `Canny`
  - `erode` / `dilate` / `morphologyEx`
- Preserve diagnostic dimensions:
  - public API entry
  - scalar fallback
  - direct OpenCV UI fast path when available
  - allocation `reuse` / `recreate`
  - `tail_ratio`
- Keep `micro` rows for kernel-cost analysis, but exclude them from product
  speedup gates by default.

DoD:

- Target builds with `CVH_BUILD_NATIVE_BACKEND=OFF`.
- Existing `cvtColor` and `resize` benchmark coverage is not lost.
- Results use standard row model or a documented compatible subset.
- Internal regression script can gate `imgproc`.

Completion notes:

- Added `benchmark/imgproc_header_benchmark.cpp`.
- Added CMake target `cvh_benchmark_imgproc_header`, linked only to
  `cvh::headers_fast`.
- Initial public API coverage includes `resize`, `cvtColor`, `threshold`,
  `LUT`, `copyMakeBorder`, `boxFilter`, `GaussianBlur`, `Sobel`, `Canny`,
  `erode`, `dilate`, and `morphologyEx`.
- Existing specialized `cvtColor` and `resize` diagnostic targets remain in
  place.

### P-Bench-5: Report And Gate Normalization

Status: complete.

Purpose: make benchmark output useful without manually reading raw CSV.

Tasks:

- Add a common report script, for example:

```text
benchmark/common/benchmark_report.py
```

- Inputs:
  - baseline/current CSV for internal regression
  - compare CSV for OpenCV upstream compare
  - metadata JSON
  - `benchmark/gate_policy.json`
- Outputs:
  - Markdown summary
  - JSON summary
  - non-zero exit for internal regression gate failures
- Report:
  - geomean/median/min/max speedup
  - top regressions
  - top improvements
  - missing cases
  - unsupported cases
  - per-suite/per-op breakdown
- Gate rules:
  - internal quick default slowdown limit: `8%`
  - internal stable accepted fast-path slowdown limit: about `5%`
  - OpenCV compare: log-only by default

DoD:

- Existing regression scripts are either reused through wrappers or deprecated
  with a clear replacement.
- Report output is deterministic and reviewable in CI logs.

Completion notes:

- Added `benchmark/common/benchmark_report.py`.
- Internal mode compares baseline/current CSV files, reports missing cases,
  top regressions, geomean speedup, and fails when slowdown exceeds the
  configured threshold.
- OpenCV compare mode renders log-only Markdown/JSON summaries.

### P-Bench-6: OpenCV Upstream Compare Migration

Status: complete.

Purpose: compare the fastest pure header-only `cvh` profile directly against
official OpenCV.

Tasks:

- Add a compare path that can use the local OpenCV source `../opencv` and a user
  supplied OpenCV build directory.
- Do not let the runner mutate the full local OpenCV checkout.
- Add or refactor compare target:

```text
cvh_benchmark_opencv_compare_headers_fast
```

- Link `cvh` side only against `cvh::headers_fast`.
- Link OpenCV side against official OpenCV `core` and `imgproc`.
- Use only these report implementation labels:
  - `cvh_headers_fast`
  - `opencv`
- Do not expose `cvh::headers`, `native`, or `lite` as Mode B implementations.

DoD:

- OpenCV compare runs without enabling `CVH_BUILD_NATIVE_BACKEND`.
- Metadata records both commits.
- Reports show `OpenCV/CVH` gap and unsupported matrix.

Completion notes:

- Added `benchmark/opencv_compare_header_benchmark.cpp`.
- Added CMake target `cvh_benchmark_opencv_compare_headers_fast`.
- The OpenCV backend source is isolated in an object library so OpenCV headers
  do not accidentally include the vendored OpenCV UI headers from `cvh`.
- `benchmark/opencv_compare/run_compare.sh` now supports only `headers_fast`
  for Mode B, emits `cvh_headers_fast`, and can use an existing local OpenCV
  build through `CVH_OPENCV_CONFIG_DIR` and
  `CVH_COMPARE_SKIP_OPENCV_SETUP=1`.
- The compare matrix now covers `core_mat` lifecycle/copy/convert/view cases
  and 14 `imgproc` operators per shape. Rows distinguish direct `opencv_ui`
  paths from inherited `headers_baseline` paths; inherited paths are measured
  rather than skipped.
- Date-named curated Markdown snapshots can be tracked under
  `benchmark/opencv_compare/results/`; raw CSV and metadata remain generated
  artifacts.
- Verified against a local minimal OpenCV `core,imgproc` build from `../opencv`.

### P-Bench-7: CI Integration

Status: complete.

Purpose: make the framework usable by automation without making performance
noise block all work.

Tasks:

- Add CI command for internal quick regression.
- Keep OpenCV compare as on-demand/log-only.
- Publish artifact paths:
  - CSV
  - Markdown report
  - metadata JSON
- Ensure generated files stay ignored.

DoD:

- Header-only CI can build benchmark targets without compiled extension paths.
- On-demand compare jobs emit enough data to guide operator prioritization.

Completion notes:

- Added `scripts/ci_benchmark_headers_quick.sh`.
- Without `CVH_BENCH_BASELINE_REF`, it builds and runs a quick smoke for
  `core_mat`, aggregate `imgproc`, and the two specialized imgproc diagnostic
  benchmarks.
- With `CVH_BENCH_BASELINE_REF`, it delegates to the internal regression
  runner and applies the configured slowdown threshold.

### P-Bench-8: Core C++ Cleanup Prerequisite

Status: TODO.

Purpose: remove the compiled-core dependency before claiming Mat arithmetic
results for `cvh::headers_fast`.

Plan:

- Follow [core-cpp-cleanup-plan.md](core-cpp-cleanup-plan.md).
- Inventory and clean overlapping implementations in `src/core`.
- Migrate accepted add/sub/mul/div, transpose, and GEMM implementations into
  ODR-safe headers by reusing the existing kernels.
- Do not include `.cpp` files from headers or benchmarks.
- Do not link `cvh_native_backend` into a header-only compare target.

DoD:

- Accepted core compute APIs compile and link with `cvh::headers` only.
- A staged installed package can compile the same APIs without repository
  `src/` paths.
- Multi-translation-unit ODR tests pass.

### P-Bench-9: Core Compute Correctness Gate

Status: TODO; blocked by P-Bench-8.

Purpose: establish correctness before publishing core compute performance.

Initial scope:

- add/subtract/multiply/divide:
  - `CV_8U`, `CV_32F`
  - C1/C3
  - continuous and ROI/non-contiguous inputs
- transpose:
  - `CV_8U`, `CV_32F`
  - square/non-square
  - continuous and ROI/non-contiguous inputs
- GEMM:
  - FP32 2D NN
  - end-to-end and pack-once semantics kept separate

DoD:

- Results match upstream OpenCV within the declared integer/float contract.
- Unsupported combinations remain explicit and do not fall through to legacy
  `.cpp` symbols.
- Tests link only `cvh::headers` or `cvh::headers_fast`.

### P-Bench-10: Mode B Core Compute Expansion

Status: TODO; blocked by P-Bench-9.

Purpose: add common Mat computation to the dated OpenCV upstream report.

Tasks:

- Add Mode B rows for add/sub/mul/div, transpose, and accepted GEMM variants.
- Use existing upstream backend helpers where available; extend them for
  multiply/divide/transpose.
- Keep `cvh_headers_fast` as the only CVH implementation label.
- Record `opencv_ui`, future platform fast-path, or `headers_baseline` as the
  actual dispatch path.
- Run the stable single-thread profile and update:

```text
benchmark/opencv_compare/results/2026-07-23-opencv-upstream-performance.md
```

DoD:

- Core compute rows contain real CVH and OpenCV timings.
- Missing fast specialization falls back to the accepted `cvh::headers`
  implementation and is still benchmarked.
- A not-yet-migrated operator is `UNSUPPORTED`, never silently omitted or
  satisfied by a compiled legacy object.

## Suggested Execution Order

1. Finish P-Bench-0 and commit documentation/artifact cleanup. Done.
2. Implement P-Bench-1 common helpers. Done.
3. Implement P-Bench-2 internal runner. Done.
4. Implement P-Bench-3 `core_mat` target. Done.
5. Implement P-Bench-4 `imgproc` target. Done.
6. Implement P-Bench-5 unified report/gate. Done.
7. Implement P-Bench-6 OpenCV upstream compare. Done.
8. Add P-Bench-7 CI integration after local behavior is stable. Done.
9. Complete P-Bench-8 core C++ cleanup and header-only migration.
10. Complete P-Bench-9 core compute correctness gate.
11. Complete P-Bench-10 Mode B core compute expansion and refresh the dated
    upstream report.

## Acceptance Rule For New Fast Paths

A new fast path can be considered accepted only when:

- correctness tests pass through `cvh::headers`;
- scalar fallback remains available;
- benchmark rows show public API speed, not only direct kernel speed;
- allocation mode is visible;
- OpenCV compare gap is known or explicitly marked pending;
- no xsimd dependency is introduced.
