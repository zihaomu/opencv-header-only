#include "cvh/core/simd/simd.h"

#include <cstdint>

#if !CVH_ENABLE_OPENCV_INTRIN
#error "cvh_opencv_intrin_x86_smoke must be compiled with CVH_ENABLE_OPENCV_INTRIN=1"
#endif

#if !CV_SSE2
#error "cvh_opencv_intrin_x86_smoke requires OpenCV UI SSE2"
#endif

#if !CV_AVX2
#error "cvh_opencv_intrin_x86_smoke requires OpenCV UI AVX2"
#endif

#if !CV_SIMD128
#error "cvh_opencv_intrin_x86_smoke expects CV_SIMD128"
#endif

#if !CV_SIMD256
#error "cvh_opencv_intrin_x86_smoke expects CV_SIMD256"
#endif

int main()
{
    alignas(32) std::uint8_t src[32] = {};
    alignas(32) std::uint8_t dst[32] = {};
    for (int i = 0; i < 32; ++i)
    {
        src[i] = static_cast<std::uint8_t>(i + 1);
    }

    cvh::detail::simd::u8 value = cvh::detail::simd::load_u8(src);
    cvh::detail::simd::store_u8(dst, value);

    if (cvh::detail::simd::u8_lanes() != 16)
    {
        return 1;
    }

    for (std::size_t i = 0; i < cvh::detail::simd::u8_lanes(); ++i)
    {
        if (dst[i] != src[i])
        {
            return 2;
        }
    }

    cvh::detail::simd::u16 low;
    cvh::detail::simd::u16 high;
    cvh::detail::simd::expand_u8(value, low, high);
    const cvh::detail::simd::u16 sum = cvh::detail::simd::add(low, high);
    (void)sum;

    return 0;
}
