# cvh

**A compact OpenCV-style subset for common C++ image processing workloads.**

`cvh` is a focused OpenCV-style subset library for common `core` and `imgproc` workloads.  
It provides two runtime modes:

- `Lite`: header-first, minimal dependency footprint.
- `Full`: links backend implementations in `src/` for broader coverage and higher performance.

This project is designed for users who want familiar OpenCV-like APIs in a more portable and trim package.

## Usage

1. Build

```bash
cmake -S . -B build
cmake --build build -j
```

2. Run tests

```bash
# Lite suite (core-lite + imgproc)
./scripts/ci_lite_all.sh

# Full suite (core-full + imgproc)
./scripts/ci_full_all.sh
```

3. Run cvh / OpenCV comparison

```bash
# Prints Markdown report to logs/stdout; intermediate artifacts go to temp dir
./scripts/ci_compare_log_only.sh
```

PR admins can toggle compare jobs by comment:

```text
/cvh-compare on
/cvh-compare off
```

`/cvh-compare on` will add the compare label and trigger the dedicated `CI Compare On Demand` workflow immediately.

## Performance Comparison

Compare workspace (entry):

- [OpenCV Compare README](benchmark/opencv_compare/README.md)

Available Markdown reports:

- Quick: [benchmark/opencv_compare/opencv_compare_quick.md](benchmark/opencv_compare/opencv_compare_quick.md)
- Stable: [benchmark/opencv_compare/opencv_compare_stable.md](benchmark/opencv_compare/opencv_compare_stable.md)
- Baseline Stable: [benchmark/opencv_compare/opencv_compare_baseline_stable.md](benchmark/opencv_compare/opencv_compare_baseline_stable.md)

Scripts:

- Runner: `benchmark/opencv_compare/run_compare.sh`
- CI log-only wrapper: `scripts/ci_compare_log_only.sh`

## Repository Layout

- `include/` - public headers and header-first implementation pieces
- `src/` - compiled/full-mode backend implementations
- `test/` - correctness and regression tests
- `benchmark/` - performance benchmarks (including `benchmark/opencv_compare/`)
- `example/` - usage examples

## License

This project is licensed under the  [Apache License 2.0](LICENSE).
