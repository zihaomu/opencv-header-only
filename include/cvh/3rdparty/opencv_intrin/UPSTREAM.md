# OpenCV Universal Intrinsics Upstream

## Source

- Repository: `/Users/zmu/work/my_project/ocvh/opencv`
- Version describe: `4.13.0-457-gd48bf69f65`
- Commit: `d48bf69f65444a13f8a34b8982b083c1b78fa0e8`
- OpenCV version header: `4.14.0-pre`

## Imported Files

The P1 import intentionally uses a small whitelist for a compile-only smoke. It does not vendor the full OpenCV HAL or full OpenCV core.

Original OpenCV files copied without manual edits:

```text
modules/core/include/opencv2/core/hal/intrin.hpp
modules/core/include/opencv2/core/hal/intrin_cpp.hpp
modules/core/include/opencv2/core/hal/intrin_forward.hpp
modules/core/include/opencv2/core/hal/intrin_math.hpp
modules/core/include/opencv2/core/hal/intrin_neon.hpp
modules/core/include/opencv2/core/hal/simd_utils.impl.hpp
LICENSE
```

Local compatibility shims:

```text
opencv2/core/cvdef.h
opencv2/core/utility.hpp
opencv2/core/saturate.hpp
```

## Local Policy

- Do not import `cv_hal_*`, `CALL_HAL`, `custom_hal.hpp`, or OpenCV module dispatch machinery in this adapter tree.
- Do not add an OpenCV binary or build-time dependency.
- Keep all business kernels behind `cvh::detail::simd` instead of including this tree directly.
- Update this file whenever the upstream commit, imported file whitelist, or local shims change.

## P1 Scope

P1 only proves that `CVH_ENABLE_OPENCV_INTRIN=1` can compile in header-only mode without linking OpenCV.

The smoke target currently forces `CV_FORCE_SIMD128_CPP=1` so the import validates the OpenCV C++ universal-intrinsics fallback path first. Native NEON/AVX selection is intentionally deferred to P2/P3.

## P3.2 Scope

P3.2 adds the NEON Universal Intrinsics header and its direct math helper so ARM header-only benchmarks can measure the platform implementation instead of the C++ fallback.

The benchmark target enables NEON explicitly with `CV_NEON=1`; regular `cvh::headers` users still default to no OpenCV intrinsics unless they opt in with project macros.
