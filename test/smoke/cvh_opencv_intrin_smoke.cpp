#include "cvh/core/simd/opencv_ui.h"

#include <cstring>

#if !CVH_ENABLE_OPENCV_INTRIN
#error "cvh_opencv_intrin_smoke must be compiled with CVH_ENABLE_OPENCV_INTRIN=1"
#endif

int main()
{
    if (std::strcmp(cvh::detail::opencv_ui_backend_name(), "opencv_intrin") != 0)
    {
        return 1;
    }

    alignas(16) uchar values[16] {};
    const cv::v_uint8x16 zero = cv::v_setzero_u8();
    cv::v_store(values, zero);
    return cv::VTraits<cv::v_uint8x16>::vlanes() == 16 && values[0] == 0 ? 0 : 2;
}
