#include "cvh/core/simd/opencv_intrin_adapter.h"

#if !CVH_ENABLE_OPENCV_INTRIN
#error "cvh_opencv_intrin_smoke must be compiled with CVH_ENABLE_OPENCV_INTRIN=1"
#endif

int main()
{
    alignas(16) uchar values[16] {};
    const cv::v_uint8x16 zero = cv::v_setzero_u8();
    cv::v_store(values, zero);
    return cv::v_uint8x16::nlanes == 16 && values[0] == 0 ? 0 : 1;
}
