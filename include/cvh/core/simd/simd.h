#ifndef CVH_CORE_SIMD_SIMD_H
#define CVH_CORE_SIMD_SIMD_H

#include "cvh/detail/config.h"

#if CVH_ENABLE_XSIMD && !CVH_ENABLE_LEGACY_XSIMD
#error "CVH_ENABLE_XSIMD is legacy/experimental; define CVH_ENABLE_LEGACY_XSIMD=1 for internal xsimd checks"
#endif

#if CVH_ENABLE_OPENCV_INTRIN
#include "opencv_intrin_adapter.h"
#elif CVH_ENABLE_XSIMD && CVH_ENABLE_LEGACY_XSIMD
#include "xsimd_adapter.h"
#else
#include "scalar_adapter.h"
#endif

namespace cvh {
namespace detail {
namespace simd {

#if CVH_ENABLE_OPENCV_INTRIN
using namespace opencv_intrin;
#elif CVH_ENABLE_XSIMD && CVH_ENABLE_LEGACY_XSIMD
using namespace xsimd_adapter;
#else
using namespace scalar;
#endif

}  // namespace simd
}  // namespace detail
}  // namespace cvh

#endif  // CVH_CORE_SIMD_SIMD_H
