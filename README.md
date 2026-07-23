# opencv-header-only (cvh)

**A pure header-only OpenCV-style C++ subset for common CV operators and AI vision preprocessing/postprocessing.**

`opencv-header-only (cvh)` targets projects that want familiar OpenCV-style APIs without carrying the full OpenCV dependency. The public product direction is intentionally header-only: include headers, link an interface CMake target, and avoid a required library build step.

> **Latest performance report:** [cvh vs OpenCV upstream benchmark (2026-07-23)](benchmark/opencv_compare/results/2026-07-23-opencv-upstream-performance.md)
>
> This is the project's stable public performance checkpoint. This link will be updated to the newest dated report as optimization work lands. We commit to continuous, benchmark-backed speed improvements while keeping regressions and the remaining gap to upstream OpenCV visible.

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
| `cvh::headers` | Default header-only target | Enables the vendored OpenCV Universal Intrinsics headers by default as the internal SIMD dialect and keeps platform-specific project fast toggles off. It does not compile `.cpp` files and does not enable xsimd. |
| `cvh::headers_fast` | Fast-profile target | Inherits `cvh::headers` and enables validated platform fast-profile toggles. It does not compile `.cpp` files and does not enable xsimd. |

New public features should land in `cvh::headers` first. `cvh::headers_fast` should only add platform fast-profile toggles that have correctness tests and a benchmark reason to exist.

## Usage

For CMake users, `cvh::headers` propagates all required include roots. For non-CMake direct include usage, add both `include/` and `include/cvh/3rdparty/opencv_intrin/` to the compiler include path.

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

Users should not need to define `CVH_ENABLE_OPENCV_INTRIN`; it is enabled by default. Use `cvh::headers_fast` only when the platform fast-profile toggles are desired.

xsimd is not part of the accepted runtime path. P5.3 removed the legacy `.cpp` xsimd kernels, public adapter surface, tests, dispatch mode, and vendored xsimd tree.

## Operator Status

Legend:

- **Supported:** implemented inline in headers and covered by the header-only test path.
- **Supported + fast path:** supported by the default target, with validated OpenCV Universal Intrinsics paths enabled by default and extra platform fast-profile toggles available through `cvh::headers_fast`.
- **WIP:** target API or historical implementation exists, but it is not yet accepted as part of the pure header-only contract.
- **Out of scope:** intentionally not promised by the pure header-only product.

| Module | API / operator | Status | Current header-only scope | `cvh::headers_fast` |
|---|---|---|---|---|
| `core` | `Mat`, `Scalar`, `Range`, `Point`, `Size`, type/channel macros | Supported | Core data model and OpenCV-style type helpers. | Same behavior as baseline. |
| `core` | `Mat::create`, `release`, `clone`, `copyTo`, `setTo`, `convertTo`, `reshape`, 2D ROI helpers | Supported | Covers common ownership, layout, continuous/non-contiguous, and conversion paths used by imgproc. | Same behavior as baseline. |
| `core` | `parallel_for_`, thread controls | Supported | Header-only serial and standard-thread runtime. | Same behavior as baseline. |
| `core` | `add`, `subtract`, `multiply`, `divide`, `compare`, `merge`, `split` | Supported | Header-only Mat-Mat/Mat-Scalar implementations with continuous and ROI coverage. | Inherits the scalar header baseline; SIMD specialization is pending. |
| `core` | `transpose`, `transposeND` | Supported | Header-only blocked transpose with continuous, ROI, C1/C3/C4 and non-square coverage. | Inherits the scalar header baseline. |
| `core` | `gemm`, `gemm_pack_b` | Supported | FP32 activation with FP32/FP16 weights, 2D/broadcast NN and packed-B; INT8 scales remain limited to the existing NT path. | Inherits the scalar header baseline. |
| `core` | `norm`, `softmax`, `silu`, `rmsnorm`, `rope` | WIP | Declarations remain outside the accepted pure header-only operator contract. | No accepted fast path. |
| `imgproc` | `resize` | Supported + fast path | `CV_8U` / `CV_32F`, `C1` / `C3` / `C4`, `INTER_NEAREST`, `INTER_NEAREST_EXACT`, `INTER_LINEAR`. | `CV_8UC1` exact 2x downsample with `INTER_LINEAR`. |
| `imgproc` | `cvtColor` | Supported + fast path | `CV_8U` / `CV_32F` common BGR/RGB/GRAY/BGRA/RGBA conversions; `CV_8U` YUV encode/decode families. | `CV_8UC3` `BGR2GRAY` and `RGB2GRAY`. |
| `imgproc` | `threshold` | Supported + fast path | `CV_8U` / `CV_32F` fixed thresholds; `OTSU` / `TRIANGLE` for `CV_8UC1`. | Row-parallel `CV_32F` fixed thresholds; other modes fall back. |
| `imgproc` | `LUT` | Supported + fast path | `src=CV_8U`, `lut.total()==256`, LUT channels `1` or source channel count. | Row-parallel `CV_8U` table path. |
| `imgproc` | `copyMakeBorder` | Supported + fast path | `CV_8U` / `CV_32F`, `BORDER_CONSTANT`, `REPLICATE`, `REFLECT`, `REFLECT_101`, `WRAP`. | Row-parallel `BORDER_REPLICATE`; other borders fall back. |
| `imgproc` | `filter2D` | Supported + fast path | `CV_8U` / `CV_32F` source, `CV_32FC1` kernel, `ddepth=-1/CV_8U/CV_32F`. | Header row-parallel convolution for the accepted type matrix. |
| `imgproc` | `sepFilter2D` | Supported + fast path | `CV_8U` / `CV_32F` source, `CV_32FC1` vector kernels, `ddepth=-1/CV_8U/CV_32F`. | Header row/column fast path for the accepted type matrix. |
| `imgproc` | `boxFilter`, `blur` | Supported + fast path | `CV_8U` / `CV_32F`, common border modes, `blur` as `boxFilter` semantic wrapper. | Specialized 3x3 and generic separable header paths. |
| `imgproc` | `GaussianBlur` | Supported + fast path | `CV_8U` / `CV_32F`, odd kernel sizes and sigma-based separable path. | Specialized 3x3 and generic separable header paths. |
| `imgproc` | `Sobel` | Supported + fast path | `CV_8U` / `CV_16S` / `CV_32F` input, `CV_16S` / `CV_32F` output, `ksize=3/5`, first-order derivatives. | `CV_8U`, `ksize=3/5`, first-order header path. |
| `imgproc` | `Canny` | Supported + fast path | Image overload for `CV_8UC1`; derivative overload for `CV_16SC1`; `apertureSize=3/5`; L1/L2 gradient. | Shared header magnitude/NMS/hysteresis path. |
| `imgproc` | `erode`, `dilate`, `morphologyEx` | Supported + fast path | `CV_8U`; `MORPH_ERODE`, `DILATE`, `OPEN`, `CLOSE`, `GRADIENT`, `TOPHAT`, `BLACKHAT`, `HITMISS`; `HITMISS` limited to `CV_8UC1`. | Shared 3x3 rectangular min/max header path; generic kernels fall back. |
| `imgcodecs` | `imread` | Supported | stb-backed `CV_8U` image load with `IMREAD_UNCHANGED`, `IMREAD_GRAYSCALE`, `IMREAD_COLOR`; OpenCV-style BGR/BGRA output for color reads. | Same behavior as baseline. |
| `imgcodecs` | `imwrite` | Supported | `CV_8U` 2D `C1` / `C3` / `C4`; writes `png`, `jpg/jpeg`, `bmp`. | Same behavior as baseline. |
| `highgui` | `imshow`, `waitKey` | Out of scope | Display/event-loop APIs are not part of the pure header-only product. Use `imwrite` or application-owned UI code. | Out of scope. |

## Header-only Contract Tests

The support table above is tied to the header-only test path:

| Contract area | Test / gate |
|---|---|
| Public headers and forbidden `src/` includes | `scripts/check_public_headers.sh` |
| Installed public targets and external package consumers | `scripts/check_header_only_contract.sh` |
| `cvh::headers` macro/default behavior | `cvh_header_compile_smoke`, `cvh_include_only_smoke` |
| `cvh::headers_fast` macro/default behavior | `cvh_headers_fast_smoke` |
| Imgproc multi-TU ODR | `cvh_imgproc_header_odr_smoke` |
| `core` supported baseline | `cvh_test_core_lite` |
| Multi-translation-unit core ODR/link | `cvh_core_header_odr_smoke` |
| `imgproc` supported operators | `cvh_test_imgproc` |
| `imgcodecs` supported read/write subset | `cvh_test_imgcodecs` |
| `highgui` header-only out-of-scope behavior | `cvh_test_highgui` |

## WIP / Roadmap

These are target areas, but they are not yet supported promises in the pure header-only contract:

| Area | Candidate APIs / work | Current intent |
|---|---|---|
| Core SIMD | `add/subtract/multiply/divide/transpose/GEMM` | Add UI or platform-specific paths only after the public header baseline is measured against upstream. |
| AI preprocessing | `normalize`, HWC-to-CHW / CHW-to-HWC, tensor packing | Add as focused preprocessing utilities once `Mat` and imgproc behavior stay stable. |
| SIMD expansion | general `resize`, broader `cvtColor`, YUV fast paths | Use direct OpenCV Universal Intrinsics style first; add platform-specific paths only when benchmark data justifies them. |
| OpenCV compatibility | more flags, depths, borders, and edge cases | Expand only with explicit behavior contracts and regression tests. |

## Performance

Performance work is benchmark-driven. `cvh::headers` enables OpenCV Universal Intrinsics as the internal SIMD dialect by default while retaining scalar fallback code paths. `cvh::headers_fast` is reserved for additional platform fast-profile toggles.

Current SIMD platform work is limited to ARM NEON and the x86 AVX family. RVV support is a future TODO; SSE headers/macros exist only as x86 OpenCV UI/AVX prerequisites, not as a separate current optimization track.

Current accepted fast paths:

- `cvtColor`: `CV_8UC3` `BGR2GRAY` / `RGB2GRAY`
- `resize`: `CV_8UC1` exact 2x downsample with `INTER_LINEAR`
- general U8 resize and RGB/GRAY/YUV conversion families
- threshold FP32, U8 LUT, and replicate copyMakeBorder
- box/Gaussian/filter2D/sepFilter2D/Sobel
- Canny image/derivative and 3x3 rectangular morphology

Compare workspace:

- [Benchmark Framework](benchmark/readme.md) - internal header-only regression and OpenCV upstream compare design
- [OpenCV Compare README](benchmark/opencv_compare/README.md) - `cvh::headers_fast` versus upstream OpenCV
- [OpenCV UI Kernel Migration Checklist](doc/opencv-ui-kernel-migration-checklist.md)

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
./scripts/ci_headers_all.sh
```

`scripts/ci_lite_all.sh` remains as a deprecated compatibility wrapper for now.

Benchmark targets:

```bash
cmake -S . -B build-bench -DCVH_BUILD_BENCHMARKS=ON
cmake --build build-bench -j --target \
  cvh_benchmark_core_mat_header \
  cvh_benchmark_imgproc_header \
  cvh_benchmark_cvtcolor_bgr2gray_header \
  cvh_benchmark_resize_bilinear_header
```

Header-only benchmark quick smoke:

```bash
./scripts/ci_benchmark_headers_quick.sh
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
