#include "cvh/core/simd/opencv_ui.h"

#include <cstring>

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
    if (std::strcmp(cvh::detail::opencv_ui_backend_name(), "opencv_intrin") != 0)
    {
        return 1;
    }

    alignas(32) uchar src[32] = {};
    alignas(32) uchar dst[32] = {};
    for (int i = 0; i < 32; ++i)
    {
        src[i] = static_cast<uchar>(i + 1);
    }

    const cv::v_uint8x16 value128 = cv::v_load(src);
    cv::v_store(dst, cv::v_add(value128, cv::v_setzero_u8()));
    if (cv::VTraits<cv::v_uint8x16>::vlanes() != 16)
    {
        return 2;
    }

    for (int i = 0; i < cv::VTraits<cv::v_uint8x16>::vlanes(); ++i)
    {
        if (dst[i] != src[i])
        {
            return 3;
        }
    }

    const cv::v_uint8x32 value256 = cv::v256_load(src);
    cv::v_store(dst, cv::v_add(value256, cv::v256_setzero_u8()));
    if (cv::VTraits<cv::v_uint8x32>::vlanes() != 32)
    {
        return 4;
    }

    for (int i = 0; i < cv::VTraits<cv::v_uint8x32>::vlanes(); ++i)
    {
        if (dst[i] != src[i])
        {
            return 5;
        }
    }

    return 0;
}
