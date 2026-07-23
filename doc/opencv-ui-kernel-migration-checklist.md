# OpenCV UI Kernel Migration Checklist

This checklist is the working rule for migrating OpenCV SIMD kernel fragments into
`opencv-header-only` after P6.

The goal is to keep OpenCV Universal Intrinsics as the internal SIMD dialect while
preserving the pure header-only product boundary. Do not reintroduce a project
owned SIMD facade.

## Scope

Allowed current SIMD scope:

- ARM NEON through OpenCV Universal Intrinsics.
- x86 AVX-family paths through OpenCV Universal Intrinsics.
- SSE headers/macros only as x86 OpenCV UI/AVX prerequisites.

Deferred scope:

- RVV is a future TODO. Do not vendor RVV headers, enable RVV macros, or accept an
  RVV fast path until scalable SIMD design is handled separately.
- Direct NEON/AVX intrinsics are allowed only as benchmark-gated candidates when
  OpenCV UI is not good enough for a specific accepted hot path.

## Source Selection

Before porting code from the OpenCV source tree:

- Record the source file path, upstream commit, and the relevant function/block.
- Prefer compact `*.simd.hpp` blocks that already use `cv::v_*`, `cv::VTraits`,
  `CV_SIMD`, `CV_SIMD_WIDTH`, or `vx_*`.
- Avoid starting from OpenCV runtime dispatch, IPP, OpenCL, threading, or HAL C
  dispatch glue. Port the kernel expression, not the whole execution framework.
- Keep the first port narrow: one type, one channel family, one interpolation or
  mode, and one clearly named fast path predicate.

## Required Shape

Every migrated fast path should have this structure:

- A scalar fallback already exists and remains the correctness baseline.
- A precise support predicate checks type, channels, dimensions, flags, and any
  shape constraints.
- A direct implementation function is guarded by `#if CVH_ENABLE_OPENCV_INTRIN`.
- The public API dispatches to the fast path only when the predicate is true.
- Tail and unsupported cases always fall back to scalar code.
- The fast path writes the same output layout and handles non-contiguous rows when
  the scalar path supports them.

Suggested naming:

```cpp
inline bool foo_u8c1_opencv_intrin_supported(...);
inline void foo_u8c1_opencv_intrin_impl(...);
```

## OpenCV UI Style

Preserve OpenCV UI expressions directly:

- Use `cv::v_uint8x16`, `cv::v_uint16x8`, `cv::v_uint32x4`,
  `cv::v_float32x4`, or the matching fixed-lane type when the accepted kernel is
  fixed-lane.
- Use `cv::v_*` / `cv::vx_*` operations directly.
- Use `cv::VTraits<T>::vlanes()` for loop stride.
- Use OpenCV UI helpers such as `cv::v_load`, `cv::v_store`,
  `cv::v_load_deinterleave`, `cv::v_pack`, `cv::v_rshr_pack`,
  `cv::v_pack_store`, `cv::v_expand`, `cv::v_mul_expand`, and `cv::v_reinterpret_*`.
- Include `cvh/core/simd/opencv_ui.h` from `cvh` implementation headers. Do not
  include vendored OpenCV UI headers by long path from business kernels.

Do not do these:

- Do not create `cvh::detail::simd::*` types or wrappers.
- Do not add a new project-owned SIMD facade for load/store/add/pack operations.
- Do not expose `cv::v_*` types in the public user API.
- Do not make `cvh::headers_fast` depend on `.cpp` files, native targets, or
  xsimd.
- Do not make xsimd a runtime performance candidate.

## Runtime Dependencies To Remove

When copying from OpenCV, replace or remove these dependencies:

- Replace OpenCV `Mat`, `Size`, `Range`, and scalar/type helpers with existing
  `cvh` equivalents.
- Use existing `CV_Assert`, `CV_Error_`, and `saturate_cast` behavior from `cvh`
  headers.
- Remove OpenCV dispatcher tables, CPU dispatch macros, IPP, OpenCL, parallel
  runtime, global state, and module registration dependencies.
- Avoid pulling broad OpenCV headers into public `cvh` headers. Add only the
  minimal vendored UI header through `opencv_ui.h`.

## Correctness Gate

Each migrated fast path needs focused coverage:

- A smoke test for the accepted fast path and at least one unsupported fallback
  case.
- Existing operator contract tests must still pass.
- Scalar output and direct OpenCV UI output must match byte-for-byte for integer
  kernels, or within the existing tolerance for floating-point kernels.
- Include ROI / non-contiguous row tests when the public operator supports them.
- Include tail-width tests, especially widths not divisible by the SIMD lane count.
- Keep RVV negative compile behavior intact when the code includes `opencv_ui.h`.

## Benchmark Gate

Every accepted fast path must have benchmark evidence:

- Benchmark scalar fallback, public API dispatch, and direct detail implementation
  on the same input.
- Record input shape, output shape, channels, lane count, tail pixels, time,
  MPix/s, speedup, and checksum.
- Verify public and direct detail checksums match scalar.
- On ARM, validate the NEON OpenCV UI path on representative image sizes.
- For x86 AVX changes, at minimum keep x86_64 `-mavx2` compile-only coverage from
  the ARM development machine; prefer real x86 runtime numbers before accepting
  a platform-specific claim.
- If OpenCV UI does not deliver a stable win, document the result and keep direct
  NEON/AVX specialization as a separate benchmark-gated candidate.

## Required Commands

Use a command set appropriate to the touched operator. For accepted fast paths,
run at least:

```bash
cmake -S . -B build-p6-kernel -DCVH_BUILD_NATIVE_BACKEND=OFF -DCVH_BUILD_TESTS=ON -DCVH_BUILD_BENCHMARKS=ON
cmake --build build-p6-kernel -j --target \
  cvh_header_compile_smoke \
  cvh_include_only_smoke \
  cvh_headers_fast_smoke \
  cvh_opencv_intrin_smoke
./scripts/check_header_only_contract.sh
./scripts/ci_headers_all.sh
scripts/sync_opencv_intrin.py --check
python3 -m py_compile scripts/sync_opencv_intrin.py
git diff --check
```

Add operator-specific targets, for example:

```bash
cmake --build build-p6-kernel -j --target \
  cvh_cvtcolor_opencv_intrin_smoke \
  cvh_resize_opencv_intrin_smoke \
  cvh_benchmark_cvtcolor_bgr2gray_header \
  cvh_benchmark_resize_bilinear_header
```

Keep platform gates:

```bash
/usr/bin/c++ -std=c++17 -arch x86_64 -mavx2 \
  -Iinclude -Iinclude/cvh/3rdparty/opencv_intrin \
  -DCVH_ENABLE_PLATFORM_INTRINSICS=1 \
  -c test/smoke/cvh_opencv_intrin_x86_smoke.cpp \
  -o /tmp/cvh_opencv_intrin_x86.o

if /usr/bin/c++ -std=c++17 \
  -Iinclude -Iinclude/cvh/3rdparty/opencv_intrin \
  -DCV_RVV=1 \
  -c test/smoke/cvh_opencv_intrin_smoke.cpp \
  -o /tmp/cvh_rvv_should_fail.o; then
  exit 1
fi
```

## Review Checklist

Before accepting a migrated kernel:

- The code is pure header-only and linked through `cvh::headers` or
  `cvh::headers_fast`.
- The implementation uses direct OpenCV UI, not a `cvh` SIMD facade.
- The support predicate is narrow and test-covered.
- Scalar fallback remains reachable and test-covered.
- Tail handling is explicit.
- Non-contiguous input/output behavior matches the operator contract.
- Benchmark rows include scalar/public/direct entries and matching checksums.
- RVV remains deferred and fails clearly if manually enabled.
- README/operator status is updated only if the accepted public surface changed.
