#include "cvh/cvh.h"
#include "cvh/core/simd/simd.h"

#include <cstring>

#ifndef CVH_LITE
#error "cvh::headers_fast must keep the pure header-only compatibility mode"
#endif

#ifdef CVH_NATIVE
#error "cvh::headers_fast must not enable legacy .cpp mode"
#endif

#if !CVH_ENABLE_OPENCV_INTRIN
#error "cvh::headers_fast must enable OpenCV Universal Intrinsics"
#endif

#if !CVH_ENABLE_PLATFORM_INTRINSICS
#error "cvh::headers_fast must enable platform intrinsics"
#endif

int main()
{
    if (std::strcmp(cvh::detail::simd::backend_name(), "opencv_intrin") != 0)
    {
        return 1;
    }

    cvh::Mat src({2, 2}, CV_8UC1);
    src.at<uchar>(0, 0, 0) = 4;
    src.at<uchar>(0, 1, 0) = 8;
    src.at<uchar>(1, 0, 0) = 12;
    src.at<uchar>(1, 1, 0) = 16;

    cvh::Mat dst;
    cvh::resize(src, dst, cvh::Size(1, 1), 0.0, 0.0, cvh::INTER_LINEAR);

    if (dst.dims != 2 ||
        dst.type() != CV_8UC1 ||
        dst.size[0] != 1 ||
        dst.size[1] != 1)
    {
        return 2;
    }

    return dst.at<uchar>(0, 0, 0) == 10 ? 0 : 3;
}
