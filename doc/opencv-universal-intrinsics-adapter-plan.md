# OpenCV UI SIMD Plan

This file replaces the old OpenCV Universal Intrinsics adapter execution log.
The adapter/facade route is complete and retired; the current implementation
uses OpenCV Universal Intrinsics directly as an internal header-only SIMD
dialect.

## Current Status

P6 is complete.

- `cvh::headers` is the default pure header-only target.
- `CVH_ENABLE_OPENCV_INTRIN` is enabled by default.
- `cvh::headers_fast` only adds extra platform fast-profile toggles.
- xsimd is removed from the accepted runtime/vendor path.
- `cvh::detail::simd` is not a future route.
- `opencv_intrin_adapter.h` and `scalar_adapter.h` are deleted.
- Business fast paths include `cvh/core/simd/opencv_ui.h` and use direct
  OpenCV UI expressions.

Current SIMD platform scope:

- ARM NEON through OpenCV Universal Intrinsics.
- x86 AVX-family paths through OpenCV Universal Intrinsics.
- SSE headers/macros only as x86 OpenCV UI/AVX prerequisites.
- RVV is deferred until scalable SIMD design is handled separately.

## Current Public Layers

```text
cvh::headers
  default header-only baseline
  OpenCV Universal Intrinsics enabled by default
  scalar fallback remains the correctness baseline

cvh::headers_fast
  header-only fast profile
  inherits cvh::headers
  enables validated platform fast-profile toggles
```

No public product layer requires compiled `.cpp` code.

## Active SIMD Entry Points

- Gateway header: `include/cvh/core/simd/opencv_ui.h`
- Compatibility header: `include/cvh/core/simd/simd.h`
  - deprecated internal compatibility header
  - only includes `opencv_ui.h`
  - defines no SIMD facade namespace, types, or operations

Accepted OpenCV UI fast paths:

- `cvtColor`: `CV_8UC3` `BGR2GRAY` / `RGB2GRAY`
- `resize`: `CV_8UC1` exact 2x downsample with `INTER_LINEAR`

## Migration Rules

Use [opencv-ui-kernel-migration-checklist.md](opencv-ui-kernel-migration-checklist.md)
for new SIMD kernel work.

Rules that should not be revisited:

- Keep OpenCV UI as an internal implementation dialect, not a public `cvh` API.
- Preserve scalar fallback for every public operator.
- Do not reintroduce `cvh::detail::simd`.
- Do not reintroduce xsimd as a runtime performance candidate.
- Do not add native or compiled `.cpp` requirements to public docs or package
  targets.
- Do not vendor full OpenCV HAL, `cv_hal_*`, `CALL_HAL`, IPP, OpenCL, or runtime
  dispatch machinery.

## Verification Snapshot

P6.7.2 final validation passed.

- `git diff --check`
- `scripts/sync_opencv_intrin.py --check`
- `python3 -m py_compile scripts/sync_opencv_intrin.py`
- P6 direct OpenCV UI CMake configure/build
- P6 smoke CTest: 6/6 passed
- `./scripts/check_header_only_contract.sh`
- `./scripts/ci_headers_all.sh`
  - core 25/25
  - imgproc 141/141
  - imgcodecs 7 passed / 1 optional skipped
  - highgui 4/4
- x86_64 AVX2 compile-only gate without manually defining
  `CVH_ENABLE_OPENCV_INTRIN=1`
- `CV_RVV=1` negative compile gate
- old adapter/facade residual scan

Representative ARM quick benchmark observations:

- 4K `BGR2GRAY` public reuse: about `1.40x`
- 4K `RGB2GRAY` public reuse: about `1.19x`
- 4K exact 2x `resize` public reuse: about `18.93x`

The `cvtColor` OpenCV UI path has modest speedup. If it needs a stable `>=1.5x`
gate later, treat it as a direct NEON/AVX specialization candidate. Do not
restore a project-owned SIMD facade for that purpose.

## Commits

- `87c743f Use direct OpenCV UI for header SIMD paths`
- `658fabd Record P6 direct OpenCV UI completion`

## Next Work

Recommended P7 direction:

1. Choose a narrow `boxFilter` / `blur` fast path candidate.
2. Port a compact OpenCV `*.simd.hpp` fragment using direct OpenCV UI style.
3. Keep the predicate narrow.
4. Add correctness and benchmark gates.
5. Accept the fast path only if benchmark evidence justifies it.
