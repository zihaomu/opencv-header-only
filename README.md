# opencv-header-only

[中文](README.zh-CN.md) | English

`opencv-header-only` is a lightweight OpenCV-style subset focused on practical portability and easy integration.
It provides two runtime modes:

- `Lite`: header-only first, minimal dependency footprint.
- `Full`: links backend implementations in `src/` for broader coverage and higher performance.

This project is designed for users who want familiar OpenCV-like APIs in a more portable and trim package.

## Usage

### 1) Build

```bash
cmake -S . -B build
cmake --build build -j
```

### 2) Run tests

```bash
# Lite suite (core-lite + imgproc)
./scripts/ci_lite_all.sh

# Full suite (core-full + imgproc)
./scripts/ci_full_all.sh
```

### 3) Run cvh / OpenCV comparison

```bash
# Prints Markdown report to logs/stdout; intermediate artifacts go to temp dir
./scripts/ci_compare_log_only.sh
```

PR admins can toggle compare jobs by comment:

```text
/cvh-compare on
/cvh-compare off
```

`/cvh-compare on` will add the compare label and trigger the dedicated
`CI Compare On Demand` workflow immediately.

## Performance Comparison

Compare workspace (entry):

- [OpenCV Compare README](opencv_compare/README.md)

Available Markdown reports:

- Quick: [opencv_compare/opencv_compare_quick.md](opencv_compare/opencv_compare_quick.md)
- Stable: [opencv_compare/opencv_compare_stable.md](opencv_compare/opencv_compare_stable.md)
- Baseline Stable: [opencv_compare/opencv_compare_baseline_stable.md](opencv_compare/opencv_compare_baseline_stable.md)

Scripts:

- Runner: `opencv_compare/run_compare.sh`
- CI log-only wrapper: `scripts/ci_compare_log_only.sh`

## License

This project is licensed under [LICENSE](LICENSE).
