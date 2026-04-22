# opencv-header-only (cvh)

**An OpenCV-style C++ library for edge-friendly common CV operators and fast AI vision preprocessing/postprocessing.**

`opencv-header-only (cvh)` is a compact C++ vision library for projects that want familiar OpenCV-style APIs without carrying the full weight of OpenCV.

## Status

- **Long-term direction:** pure header-only
- **Strategic path:** `Lite`
- **Legacy transition path:** `Full`
- **Primary focus:** edge deployment and faster CV model preprocessing/postprocessing than OpenCV on important hot paths

This project is **not** a full replacement for OpenCV. It focuses on a smaller, high-value subset for constrained deployment and CV model input/output pipelines.

## Why this project exists

Many real-world projects do not need all of OpenCV.

They often need:

- a small set of common image processing operators
- easier integration in build-unfriendly environments
- lower dependency complexity
- tighter control over memory usage
- better preprocessing/postprocessing performance for AI pipelines

`opencv-header-only` is built for those cases.

## Modes

`opencv-header-only` currently has two modes:

- **`Lite`** — the strategic path: header-only, lightweight, edge-friendly, and designed for constrained build environments, memory-sensitive deployments, and common CV workloads.
- **`Full`** — the legacy transition path: compiled implementations in `src/`, kept for broader historical coverage and migration, but no longer the long-term focus of the project.

New features should prefer the header-only path first.

## Focus

This project focuses on two main scenarios.

### 1. Edge-friendly common CV operators

`opencv-header-only` is designed for environments where:

- dependency surface should stay small
- build environments are unfriendly or heavily constrained
- memory matters
- only common CV operators are needed
- users want a simpler OpenCV-style integration story

Typical examples include:

- edge devices
- embedded-oriented applications
- portable SDKs
- deployment packages where full OpenCV is too large or too heavy

### 2. CV model preprocessing and postprocessing

A second core focus is CV model input/output processing.

This includes high-frequency operations such as:

- `resize`
- `warpAffine`
- `cvtColor`
- `convertTo`
- `copyMakeBorder` / letterbox-style padding
- normalization
- layout conversion such as HWC ↔ CHW
- tensor packing

The long-term performance goal is not just API compatibility, but to provide **faster preprocessing/postprocessing interfaces than the official OpenCV implementation on practical hot paths**.

## Performance

`cvh` keeps explicit comparison against OpenCV to track correctness and performance on supported operators.

Current performance work should be interpreted as follows:

- `Lite` prioritizes header-only usability, portability, and correctness-first evolution
- `Full` reflects legacy compiled implementations and broader historical optimization coverage
- the long-term optimization focus is **CV model preprocessing/postprocessing hot paths**, where the project aims to become faster than OpenCV on important real-world pipelines

Compare workspace:

- [OpenCV Compare README](benchmark/opencv_compare/README.md)

Available Markdown reports:

- Quick: [benchmark/opencv_compare/opencv_compare_quick.md](benchmark/opencv_compare/opencv_compare_quick.md)
- Stable: [benchmark/opencv_compare/opencv_compare_stable.md](benchmark/opencv_compare/opencv_compare_stable.md)
- Baseline Stable: [benchmark/opencv_compare/opencv_compare_baseline_stable.md](benchmark/opencv_compare/opencv_compare_baseline_stable.md)

Scripts:

- Runner: `benchmark/opencv_compare/run_compare.sh`
- CI log-only wrapper: `scripts/ci_compare_log_only.sh`

PR admins can toggle compare jobs by comment:

    /cvh-compare on
    /cvh-compare off

`/cvh-compare on` will add the compare label and trigger the dedicated `CI Compare On Demand` workflow immediately.

## Usage

### Lite mode

`Lite` is header-only.

For Lite mode, you only need to include headers from `include/`.  
No library build step is required.

Example integration:

```cpp
#include <cvh/...>
```

### Full mode

Full is the legacy compiled transition path.

Build is only required for Full mode:
#### 1. Build

```bash
cmake -S . -B build
cmake --build build -j
```

#### 2. Run tests

```bash
# Lite suite (core-lite + imgproc)
./scripts/ci_lite_all.sh

# Full suite (core-full + imgproc)
./scripts/ci_full_all.sh
```

#### 3. Run cvh / OpenCV comparison

```bash
# Prints Markdown report to logs/stdout; intermediate artifacts go to temp dir
./scripts/ci_compare_log_only.sh
```

## Repository Layout

- `include/` — public headers and the header-only / header-first implementation path
- `src/` — legacy compiled implementations used by the `Full` transition path
- `test/` — correctness and regression tests
- `benchmark/` — performance benchmarks, including `benchmark/opencv_compare/`
- `example/` — usage examples

## License

This project is licensed under the [Apache License 2.0](LICENSE).