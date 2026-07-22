#ifndef CVH_CORE_SIMD_SIMD_H
#define CVH_CORE_SIMD_SIMD_H

#include "cvh/detail/config.h"

#ifdef CVH_ENABLE_XSIMD
#error "CVH_ENABLE_XSIMD has been removed from the header-only SIMD facade; use cvh::headers_fast for accepted SIMD paths"
#endif

#ifdef CVH_ENABLE_LEGACY_XSIMD
#error "CVH_ENABLE_LEGACY_XSIMD has been removed from the header-only SIMD facade"
#endif

#if CVH_ENABLE_OPENCV_INTRIN
#include "opencv_intrin_adapter.h"
#else
#include "scalar_adapter.h"
#endif

namespace cvh {
namespace detail {
namespace simd {

#if CVH_ENABLE_OPENCV_INTRIN
using namespace opencv_intrin;
#else
using namespace scalar;
#endif

}  // namespace simd
}  // namespace detail
}  // namespace cvh

#endif  // CVH_CORE_SIMD_SIMD_H
