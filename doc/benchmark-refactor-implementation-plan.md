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

## Current State After C++ Cleanup

All accepted `core` and `imgproc` implementations have moved from
`src/core/*.cpp` and `src/imgproc/*.cpp` into ODR-safe headers. Those source
directories no longer contain operator implementations, so the old benchmark
TODO to migrate compiled targets is obsolete.

Canonical product benchmark targets:

| Target | Role |
|---|---|
| `cvh_benchmark_core_mat_header` | Mode A `core_mat`; links `cvh::headers_fast`. |
| `cvh_benchmark_imgproc_header` | Mode A `imgproc`; links `cvh::headers_fast`. |
| `cvh_benchmark_opencv_compare_headers_fast` | Mode B; compares only `cvh::headers_fast` with OpenCV. |

Header-only diagnostic targets retained outside product gates:

| Target | Role |
|---|---|
| `cvh_benchmark_cvtcolor_bgr2gray_header` | Scalar/public/direct UI/micro diagnosis for `BGR2GRAY` / `RGB2GRAY`. |
| `cvh_benchmark_resize_bilinear_header` | Scalar/public/direct UI/micro diagnosis for exact-half bilinear resize. |
| `cvh_benchmark_imgproc_coverage` | Exhaustive type/channel/color/YUV compatibility sweep. |
| `cvh_benchmark_imgproc_filter` | Forced fallback/fast-path filter diagnosis. |

The remaining work is benchmark consolidation and matrix expansion, not
`.cpp` migration.

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
| `schema_version` | yes | Canonical Mode A CSV compatibility version. |
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
| `implementation` | yes | Mode A product rows use `cvh_headers_fast`; diagnostic rows may use `scalar_fallback` / `opencv_ui_fastpath`. Mode B only uses `cvh_headers_fast` and `opencv`. |
| `dispatch_path` | yes | Actual internal path. |
| `allocation_mode` | yes | `reuse`, `recreate`, or `none`. |
| `tail_ratio` | imgproc | Per-row scalar tail ratio for the selected SIMD lane width. |
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
- P-Bench-11 removes the ineffective `--target` selector. Both suites use
  their canonical `cvh::headers_fast` product target; scalar/UI choices stay
  in diagnostic binaries.

### P-Bench-3: `core_mat` Header-only Benchmark

Status: complete.

Purpose: measure `Mat` baseline costs and accepted core operators through
header-only targets.

Tasks:

- Add target:

```text
cvh_benchmark_core_mat_header
```

- Link only `cvh::headers_fast` for the canonical Mode A product benchmark.
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
- Added CMake target `cvh_benchmark_core_mat_header`; P-Bench-11 aligns it
  with the canonical `cvh::headers_fast` Mode A profile.
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
- This phase established the aggregate target, but did not yet absorb the full
  type/channel/YUV/filter matrix from the diagnostic benchmarks. That coverage
  debt is moved to P-Bench-12.

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

Status: complete.

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

Completion notes:

- Mat/system duplicate implementations and all accepted `src/core/*.cpp`
  dependencies were removed.
- Arithmetic, transpose, GEMM, utilities, and MatExpr now have ODR-safe header
  definitions.
- `cvh_core_header_odr_smoke` covers arithmetic, transpose, GEMM, and MatExpr
  from two translation units.
- The staged install contract consumer compiles and executes the same core
  APIs without repository `src/` paths.

### P-Bench-9: Core Compute Correctness Gate

Status: complete.

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

Completion notes:

- Existing core contracts now run only against `cvh::headers`, including
  continuous and non-contiguous Mat arithmetic.
- Added non-contiguous multichannel ROI transpose coverage.
- Mode B performs an upstream correctness preflight before timing every
  arithmetic, transpose, and GEMM case.
- U8 add/subtract/multiply and transpose are byte-exact. U8 divide declares
  absolute tolerance `1` for OpenCV's optimized half-way rounding variation;
  F32 and GEMM use relative tolerance.

### P-Bench-10: Mode B Core Compute Expansion

Status: complete.

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

Completion notes:

- Added stable Mode B rows for add/subtract/multiply/divide across `CV_8U` and
  `CV_32F`, C1/C3, VGA/720p/1080p.
- Added transpose rows for the same type/channel/shape matrix.
- Added FP32 NN GEMM `128^3`, `256^3`, and `512^3`, split into end-to-end and
  pack-once variants.
- Refreshed
  `benchmark/opencv_compare/results/2026-07-23-opencv-upstream-performance.md`
  with 126 stable single-thread cases.

### P-Bench-11: Post-cleanup Benchmark Contract

Status: complete.

Purpose: align Mode A with the now fully header-only implementation and remove
benchmark controls that do not change the compiled product path.

Tasks:

- Make both canonical aggregate targets link `cvh::headers_fast`.
- Keep `cvh::headers` for compile/correctness contract tests, not product
  performance reports.
- Remove Mode A runner `--target headers|headers_fast`; it previously changed
  only output metadata and did not alter target linkage.
- Define these target roles:
  - product aggregate: `cvh_benchmark_core_mat_header`,
    `cvh_benchmark_imgproc_header`
  - kernel diagnostics: `cvh_benchmark_cvtcolor_bgr2gray_header`,
    `cvh_benchmark_resize_bilinear_header`,
    `cvh_benchmark_imgproc_filter`
  - exhaustive compatibility sweep: `cvh_benchmark_imgproc_coverage`
- Treat the first commit containing this contract as the new Mode A baseline
  floor. Older refs remain useful for manual diagnosis but are not schema-
  stable regression baselines.

DoD:

- Mode A metadata and CSV implementation labels match actual CMake linkage.
- The quick CI wrapper no longer passes an ineffective target selector.
- Both aggregate targets build and run with
  `CVH_BUILD_NATIVE_BACKEND=OFF`.
- Documentation contains no remaining claim that an imgproc/core benchmark
  still requires migrated `.cpp` operator code.

Completion notes:

- Both canonical aggregate targets now link `cvh::headers_fast` and emit the
  `cvh_headers_fast` implementation label.
- Removed the ineffective Mode A `--target` option from the runner and CI
  wrapper.
- Fixed Mode A execution on macOS Bash 3 by removing `mapfile`.
- Fixed temporary worktree reuse/cleanup by treating worktree `.git` as a file
  rather than a directory.
- Built and ran both aggregate quick profiles with
  `CVH_BUILD_NATIVE_BACKEND=OFF`; an imgproc baseline/current runner smoke
  matched all 13 rows.

### P-Bench-12: Aggregate Imgproc Matrix Consolidation

Status: complete.

Purpose: make `cvh_benchmark_imgproc_header` the canonical Mode A suite without
losing useful coverage from the older broad and specialized binaries.

Steps:

- P-Bench-12.0: produce an operator/variant/type/channel/layout inventory for
  the aggregate target and each diagnostic source.
- P-Bench-12.1: add missing accepted public operators, starting with
  `filter2D`, `sepFilter2D`, and `blur`.
- P-Bench-12.2: expand resize and color coverage:
  - resize `CV_8U` / `CV_32F`, C1/C3/C4, nearest/nearest-exact/linear
  - BGR/RGB/GRAY/BGRA/RGBA and accepted YUV layouts
- P-Bench-12.3: add threshold/filter type and channel coverage plus
  representative ROI/non-contiguous inputs.
- P-Bench-12.4: add explicit `reuse` / `recreate`, `tail_ratio`, and truthful
  dispatch-path telemetry where available.
- P-Bench-12.5: define bounded profile matrices:
  - `quick`: representative smoke and PR gate
  - `stable`: supported operator/type/channel matrix
  - `full`: ROI, odd-width, allocation, and layout expansion
  - `micro`: kernel diagnostics, excluded from product gates

P-Bench-12.0 inventory:

| Area | Current aggregate | Coverage source | Consolidation decision |
|---|---|---|---|
| resize | U8C1 linear exact-half | `imgproc_ops`: U8/F32 C1/C3/C4 nearest/linear; specialized target: exact-half scalar/UI micro | Add bounded U8/F32 C1/C3/C4 matrix and nearest-exact; keep exact-half micro target. |
| color RGB families | U8 BGR2GRAY and GRAY2BGR | `imgproc_ops`: U8/F32 RGB/BGR/GRAY/BGRA/RGBA families | Add representative quick rows and full accepted family in stable/full. |
| color YUV | none | `imgproc_ops`: packed, planar, semi-planar, 420/422/444 encode/decode | Keep quick small; add accepted layouts to full with explicit layout names. |
| threshold | U8C1 binary | `imgproc_ops`: U8 C1/C3 and F32 C1/C3/C4 | Add type/channel matrix to stable/full. |
| LUT/border | U8C1 single variant | no broader canonical source | Retain quick row; add channel/border variants only where they affect dispatch. |
| box/Gaussian | U8C1 one kernel/border | `imgproc_ops`: U8/F32 C1/C3/C4; `imgproc_filter`: U8 C1/C3/C4, ROI, kernels, borders, forced dispatch | Move public type/channel/ROI rows; keep forced fallback diagnostics separate. |
| filter2D/sepFilter2D/blur | absent | Mode B already exercises filter2D/sepFilter2D; public headers are accepted | Add to aggregate before expanding variants. |
| Sobel/Canny/morphology | U8C1 representative rows | Mode B has the same representative public cases | Retain quick rows, add bounded channel/ROI variants only when the API accepts them. |
| allocation/layout | rows claim reuse; no recreate/ROI product matrix | filter diagnostic has ROI; specialized targets split public/direct paths | Make reuse real through preallocation, add explicit recreate rows, and fix ROI-safe checksum first. |
| dispatch | mostly `public_header` constant | filter getters expose box/Gaussian paths; specialized targets expose scalar/UI paths | Record available runtime telemetry; use documented static tags only when the selected path is unambiguous. |

P-Bench-12.0 status: complete.

DoD:

- Aggregate `quick` remains short enough for CI.
- Aggregate `stable` covers every accepted imgproc operator at least once.
- Unsupported combinations emit `UNSUPPORTED` rows with a reason instead of
  disappearing.
- Every product row calls a public `cvh` API.

Completion notes:

- Added `blur`, `filter2D`, and `sepFilter2D` public API rows to the quick
  product matrix.
- Stable now covers 16 operator families across representative `CV_8U` /
  `CV_32F` and C1/C3/C4 cases without multiplying the matrix across every
  large shape.
- Full adds representative NV12/I420/NV16/YUY2/NV24/I444 encode/decode
  layouts, odd-width tails, seven non-contiguous ROI rows, and four true
  recreate rows.
- Added `tail_ratio`; changed reuse measurement to preallocate outside the
  timed samples.
- Made common checksum, equality, and deterministic U8 fill helpers safe for
  non-contiguous 2D Mat inputs; added deterministic F32 fill.
- Added an explicit `UNSUPPORTED` row for resize interpolation outside the
  accepted nearest/nearest-exact/linear contract.
- Release smoke results contain 16 quick, 102 stable, and 172 full rows.
- Specialized scalar/direct-UI micro targets remain separate by design.

### P-Bench-13: Mode A Baseline And Gate Stabilization

Status: complete.

Purpose: turn the runner into a repeatable optimization gate after the
aggregate schema is stable.

Tasks:

- Add a benchmark schema version to CSV metadata and reports.
- Reject incompatible baseline refs with a clear error rather than failing
  later during build or row matching.
- Run the first stable baseline/current comparison from the P-Bench-11
  baseline floor.
- Separate missing rows caused by schema growth from actual regressions.
- Allow per-op thresholds only after enough stable samples exist.

DoD:

- `core_mat`, `imgproc`, and `all` run end to end from one command.
- Reports identify schema mismatch, missing cases, and performance regressions
  separately.
- CI quick gates only matched canonical product rows.

Progress notes:

- Canonical core/imgproc CSV now uses benchmark schema version `2`.
- Reports reject missing or incompatible schema versions with a dedicated
  exit code and error instead of producing misleading missing rows.
- Case identity excludes implementation and dispatch metadata, so a dispatch
  improvement remains comparable to its baseline.
- Reports distinguish baseline cases removed from candidate cases newly added;
  newly added rows are log-only.
- `UNSUPPORTED` and non-canonical implementation rows are excluded from the
  product regression gate.
- Dirty candidates are labeled `<commit>-dirty` in run directories and
  metadata records `current_dirty`, avoiding attribution to a clean commit.
- A snapshot-backed `suite=all` quick run matched 36 core and 16 imgproc rows.
- A snapshot-backed `suite=all` stable run matched 63 core and 101 supported
  imgproc rows with no missing/added cases.
- Per-op thresholds remain intentionally unset until distinct optimization
  commits provide enough stable samples; the global quick/stable policy
  remains in force.

### P-Bench-14: Mode B Coverage Expansion

Status: complete.

Purpose: extend the OpenCV comparison beyond the current mostly continuous
`CV_8UC1` imgproc matrix.

Tasks:

- Reuse the P-Bench-12 case descriptors where practical so Mode A and Mode B
  do not drift.
- Add representative depth/channel cases:
  - `CV_8U` / `CV_32F`
  - C1/C3/C4 where the public operator accepts them
- Add ROI/non-contiguous rows for operators whose API contract supports them.
- Add resize interpolation and color/YUV variants in bounded stable/full
  profiles.
- Preserve only `cvh_headers_fast` and `opencv` implementation labels.
- Refresh the date-named upstream performance snapshot after correctness
  preflight passes.

DoD:

- Mode B reports performance gaps for the accepted matrix and explicit
  unsupported rows for the rest.
- OpenCV compare remains log-only.
- No benchmark side links product code from `src/core` or `src/imgproc`.

Completion notes:

- Stable Mode B now contains 176 cases: 84 `core_mat` and 92 `imgproc`.
- Added bounded U8 C3/C4 and F32 C1/C3/C4 coverage for filter, border, warp,
  resize, and RGB/YUV conversion paths.
- Full contains 229 rows, including seven non-contiguous ROI rows and
  representative I420/YUY2/NV12 layout coverage.
- Upstream's missing single-call BGR-to-NV12 encoder is emitted as an explicit
  `UNSUPPORTED` row in full rather than silently omitted.
- Added `layout` to Mode B CSV and detailed Markdown output.
- The OpenCV backend remains isolated in its object library; case descriptors
  are mirrored across the boundary instead of including OpenCV headers in the
  CVH translation unit.
- Re-ran the single-thread Apple M5 stable report and refreshed
  `benchmark/opencv_compare/results/2026-07-23-opencv-upstream-performance.md`.
- The refreshed stable report has imgproc geometric mean
  `OpenCV/CVH=0.3637`; this is a mixed-matrix summary, not a claim that every
  operator improved.

### P-Bench-15: Diagnostic Target Retirement

Status: complete.

Purpose: remove duplicate benchmark maintenance only after the aggregate suite
has absorbed the valuable cases.

Tasks:

- Compare aggregate coverage against the former `cvh_benchmark_imgproc_ops`.
- Keep specialized cvtColor/resize/filter binaries only when they still expose
  scalar/direct-UI/micro data that the aggregate suite intentionally excludes.
- Remove obsolete source/targets and update CI commands.
- Keep report compatibility wrappers only where historical data still needs
  them.

DoD:

- There is one canonical product benchmark per suite.
- Remaining diagnostics have a unique documented purpose.
- No duplicate broad product matrix is maintained in two binaries.

Completion notes:

- Renamed `cvh_benchmark_imgproc_ops` and its source to
  `cvh_benchmark_imgproc_coverage`; it is now explicitly an exhaustive
  compatibility sweep, not a second product performance suite.
- Kept `cvh_benchmark_imgproc_filter` because forced fallback and runtime
  filter dispatch remain unique diagnostics.
- Kept specialized cvtColor/resize binaries because they expose scalar,
  direct-UI, and micro rows intentionally excluded from product gates.
- `cvh_benchmark_core_mat_header` and `cvh_benchmark_imgproc_header` are the
  only canonical Mode A product targets.

### P-Bench-16: Phase 1 Full Mode B Coverage

Status: complete.

Purpose: extend Mode B after the first Core/Imgproc API phase so every newly
supported operation family has a representative upstream comparison.

Completion notes:

- Added a shared `Phase1OpId` contract and one shared case implementation
  included by isolated CVH and OpenCV translation units.
- Added 76 representative cases; together with the existing `remap`,
  `warpPerspective`, and `getRectSubPix` matrix, all 79 Phase 1 operation
  families are measured.
- Quick preflight produced 122 valid rows and confirmed Phase 1 coverage
  `79/79`.
- The single-thread Apple M5 full run uses `warmup=1`, `iters=10`,
  `repeats=3` and produced 321 rows: 320 valid and one explicit unsupported
  upstream BGR-to-NV12 case.
- The dated report now contains 92 Phase 1 performance cases and marks every
  result as `P1 新增` or `既有`.
- The report generator derives Phase 1 measured counts from the CSV instead
  of retaining the former hard-coded `3/79` snapshot.

## Suggested Execution Order

1. Finish P-Bench-0 and commit documentation/artifact cleanup. Done.
2. Implement P-Bench-1 common helpers. Done.
3. Implement P-Bench-2 internal runner. Done.
4. Implement P-Bench-3 `core_mat` target. Done.
5. Implement P-Bench-4 `imgproc` target. Done.
6. Implement P-Bench-5 unified report/gate. Done.
7. Implement P-Bench-6 OpenCV upstream compare. Done.
8. Add P-Bench-7 CI integration after local behavior is stable. Done.
9. Complete P-Bench-8 core C++ cleanup and header-only migration. Done.
10. Complete P-Bench-9 core compute correctness gate. Done.
11. Complete P-Bench-10 Mode B core compute expansion and refresh the dated
    upstream report. Done.
12. Complete P-Bench-11 post-cleanup benchmark contract. Done.
13. Complete P-Bench-12 aggregate imgproc matrix consolidation. Done.
14. Complete P-Bench-13 Mode A baseline and gate stabilization. Done.
15. Complete P-Bench-14 Mode B coverage expansion. Done.
16. Complete P-Bench-15 diagnostic target retirement. Done.
17. Complete P-Bench-16 Phase 1 full Mode B coverage. Done.

## Acceptance Rule For New Fast Paths

A new fast path can be considered accepted only when:

- correctness tests pass through `cvh::headers`;
- scalar fallback remains available;
- benchmark rows show public API speed, not only direct kernel speed;
- allocation mode is visible;
- OpenCV compare gap is known or explicitly marked pending;
- no xsimd dependency is introduced.
