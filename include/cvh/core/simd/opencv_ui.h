#ifndef CVH_CORE_SIMD_OPENCV_UI_H
#define CVH_CORE_SIMD_OPENCV_UI_H

#include "cvh/detail/config.h"

#ifdef CVH_ENABLE_XSIMD
#error "CVH_ENABLE_XSIMD has been removed; use OpenCV Universal Intrinsics"
#endif

#ifdef CVH_ENABLE_LEGACY_XSIMD
#error "CVH_ENABLE_LEGACY_XSIMD has been removed"
#endif

#if (defined(CV_RVV) && CV_RVV) || (defined(CV_RVV071) && CV_RVV071)
#error "CVH OpenCV Universal Intrinsics RVV is deferred; use NEON or AVX paths until a scalable design exists"
#endif

#if CVH_ENABLE_OPENCV_INTRIN
#include "cvh/3rdparty/opencv_intrin/opencv2/core/hal/intrin.hpp"
#endif

namespace cvh {
namespace detail {

inline const char* opencv_ui_backend_name()
{
#if CVH_ENABLE_OPENCV_INTRIN
    return "opencv_intrin";
#else
    return "scalar";
#endif
}

}  // namespace detail
}  // namespace cvh

#endif  // CVH_CORE_SIMD_OPENCV_UI_H
