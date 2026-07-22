# opencv-header-only (cvh)

**A pure header-only OpenCV-style C++ subset for common CV operators and AI vision preprocessing/postprocessing.**

`opencv-header-only (cvh)` targets projects that want familiar OpenCV-style APIs without carrying the full OpenCV dependency. The public product direction is intentionally header-only: include headers, link an interface CMake target, and avoid a required library build step.

## Status

- **Project direction:** pure header-only
- **Default target:** `cvh::headers`
- **Fast target:** `cvh::headers_fast`
- **Scope:** a focused OpenCV-style subset, not a full OpenCV replacement
- **Performance goal:** benchmark-gated speedups on practical preprocessing/postprocessing hot paths

## Why this project exists

Many real-world deployments only need a small and predictable part of OpenCV:

- common image processing operators
- small dependency surface
- simple integration in constrained build environments
- predictable `Mat` memory/layout behavior
- fast AI vision preprocessing and postprocessing on selected hot paths

## CMake Targets

`opencv-header-only` exposes two public header-only targets:

| Target | Role | Behavior |
|---|---|---|
| `cvh::headers` | Default header-only baseline | Stable scalar fallback behavior, minimal assumptions, no experimental fast paths enabled by default. |
| `cvh::headers_fast` | Opt-in header-only fast profile | Enables validated SIMD fast paths through the vendored OpenCV Universal Intrinsics adapter and platform intrinsic toggles. It does not compile `.cpp` files and does not enable xsimd. |

New public features should land in `cvh::headers` first. `cvh::headers_fast` should only enable paths that have correctness tests and a benchmark reason to exist.

## Usage

For direct include usage, only headers from `include/` are required:

```cpp
#include <cvh/cvh.h>
```

CMake baseline integration:

```cmake
find_package(opencv_header_only CONFIG REQUIRED)
target_link_libraries(app PRIVATE cvh::headers)
```

CMake fast-profile integration:

```cmake
find_package(opencv_header_only CONFIG REQUIRED)
target_link_libraries(app PRIVATE cvh::headers_fast)
```

Users should prefer `cvh::headers_fast` over manually combining `CVH_ENABLE_OPENCV_INTRIN`, `CVH_ENABLE_PLATFORM_INTRINSICS`, or vendored OpenCV UI include paths.

## Operator Status

Legend:

- **Supported:** implemented inline in headers and covered by the header-only test path.
- **Supported + fast path:** supported by the baseline target, with an additional validated SIMD path in `cvh::headers_fast`.
- **WIP:** target API or historical implementation exists, but it is not yet accepted as part of the pure header-only contract.
- **Out of scope:** intentionally not promised by the pure header-only product.

| Module | API / operator | Status | Current header-only scope | `cvh::headers_fast` |
|---|---|---|---|---|
| `core` | `Mat`, `Scalar`, `Range`, `Point`, `Size`, type/channel macros | Supported | Core data model and OpenCV-style type helpers. | Same behavior as baseline. |
| `core` | `Mat::create`, `release`, `clone`, `copyTo`, `setTo`, `convertTo`, `reshape`, 2D ROI helpers | Supported | Covers common ownership, layout, continuous/non-contiguous, and conversion paths used by imgproc. | Same behavior as baseline. |
| `core` | `parallel_for_`, thread controls | Supported | Header-only serial and standard-thread runtime. | Same behavior as baseline. |
| `core` | `add`, `subtract`, `multiply`, `divide`, `compare`, `merge`, `split` | WIP | Public declarations exist, but these are not yet accepted as pure header-only supported operators. | No accepted fast path. |
| `core` | `transpose`, `norm`, `softmax`, `silu`, `rmsnorm`, `rope`, GEMM-related helpers | WIP | Useful historical/AI-kernel work, but outside the current header-only OpenCV-operator contract. | No accepted fast path. |
| `imgproc` | `resize` | Supported + fast path | `CV_8U` / `CV_32F`, `C1` / `C3` / `C4`, `INTER_NEAREST`, `INTER_NEAREST_EXACT`, `INTER_LINEAR`. | `CV_8UC1` exact 2x downsample with `INTER_LINEAR`. |
| `imgproc` | `cvtColor` | Supported + fast path | `CV_8U` / `CV_32F` common BGR/RGB/GRAY/BGRA/RGBA conversions; `CV_8U` YUV encode/decode families. | `CV_8UC3` `BGR2GRAY` and `RGB2GRAY`. |
| `imgproc` | `threshold` | Supported | `CV_8U` / `CV_32F` fixed thresholds; `OTSU` / `TRIANGLE` for `CV_8UC1`. | No accepted fast path. |
| `imgproc` | `LUT` | Supported | `src=CV_8U`, `lut.total()==256`, LUT channels `1` or source channel count. | No accepted fast path. |
| `imgproc` | `copyMakeBorder` | Supported | `CV_8U` / `CV_32F`, `BORDER_CONSTANT`, `REPLICATE`, `REFLECT`, `REFLECT_101`, `WRAP`. | No accepted fast path. |
| `imgproc` | `filter2D` | Supported | `CV_8U` / `CV_32F` source, `CV_32FC1` kernel, `ddepth=-1/CV_8U/CV_32F`. | No accepted fast path. |
| `imgproc` | `sepFilter2D` | Supported | `CV_8U` / `CV_32F` source, `CV_32FC1` vector kernels, `ddepth=-1/CV_8U/CV_32F`. | No accepted fast path. |
| `imgproc` | `boxFilter`, `blur` | Supported | `CV_8U` / `CV_32F`, common border modes, `blur` as `boxFilter` semantic wrapper. | No accepted fast path. |
| `imgproc` | `GaussianBlur` | Supported | `CV_8U` / `CV_32F`, odd kernel sizes and sigma-based separable path. | No accepted fast path. |
| `imgproc` | `Sobel` | Supported | `CV_8U` / `CV_16S` / `CV_32F` input, `CV_16S` / `CV_32F` output, `ksize=3/5`, first-order derivatives. | No accepted fast path. |
| `imgproc` | `Canny` | Supported | Image overload for `CV_8UC1`; derivative overload for `CV_16SC1`; `apertureSize=3/5`; L1/L2 gradient. | No accepted fast path. |
| `imgproc` | `erode`, `dilate`, `morphologyEx` | Supported | `CV_8U`; `MORPH_ERODE`, `DILATE`, `OPEN`, `CLOSE`, `GRADIENT`, `TOPHAT`, `BLACKHAT`, `HITMISS`; `HITMISS` limited to `CV_8UC1`. | No accepted fast path. |
| `imgcodecs` | `imread` | Supported | stb-backed `CV_8U` image load with `IMREAD_UNCHANGED`, `IMREAD_GRAYSCALE`, `IMREAD_COLOR`; OpenCV-style BGR/BGRA output for color reads. | Same behavior as baseline. |
| `imgcodecs` | `imwrite` | Supported | `CV_8U` 2D `C1` / `C3` / `C4`; writes `png`, `jpg/jpeg`, `bmp`. | Same behavior as baseline. |
| `highgui` | `imshow`, `waitKey` | Out of scope | Display/event-loop APIs are not part of the pure header-only product. Use `imwrite` or application-owned UI code. | Out of scope. |

## WIP / Roadmap

These are target areas, but they are not yet supported promises in the pure header-only contract:

| Area | Candidate APIs / work | Current intent |
|---|---|---|
| Core array ops | `add/subtract/multiply/divide/compare/merge/split` | Move accepted implementations into headers, then require tests through `cvh::headers`. |
| AI preprocessing | `normalize`, HWC-to-CHW / CHW-to-HWC, tensor packing | Add as focused preprocessing utilities once `Mat` and imgproc behavior stay stable. |
| SIMD expansion | general `resize`, broader `cvtColor`, YUV fast paths | Use OpenCV Universal Intrinsics first; add platform-specific paths only when benchmark data justifies them. |
| OpenCV compatibility | more flags, depths, borders, and edge cases | Expand only with explicit behavior contracts and regression tests. |

## Performance

Performance work is benchmark-driven. `cvh::headers` is the correctness-first baseline. `cvh::headers_fast` is where validated header-only SIMD paths are enabled.

Current accepted fast paths:

- `cvtColor`: `CV_8UC3` `BGR2GRAY` / `RGB2GRAY`
- `resize`: `CV_8UC1` exact 2x downsample with `INTER_LINEAR`

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

## Development

Header-only validation:

```bash
./scripts/ci_lite_all.sh
```

Benchmark targets:

```bash
cmake -S . -B build-bench -DCVH_BUILD_BENCHMARKS=ON
cmake --build build-bench -j --target \
  cvh_benchmark_cvtcolor_bgr2gray_header \
  cvh_benchmark_resize_bilinear_header
```

## Repository Layout

- `include/` - public headers and accepted header-only implementation path
- `src/` - legacy experiments and historical implementation code; not part of the public header-only contract
- `test/` - correctness and regression tests
- `benchmark/` - performance benchmarks, including `benchmark/opencv_compare/`
- `example/` - usage examples
- `doc/` - design notes and execution plans

## License

This project is licensed under the [Apache License 2.0](LICENSE).
