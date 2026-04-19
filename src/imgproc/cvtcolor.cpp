#include "cvtcolor_internal.h"

namespace cvh
{
namespace detail
{

void cvtColor_backend_impl(const Mat& src, Mat& dst, int code)
{
    if (src.depth() == CV_8U)
    {
        if (try_cvtcolor_fastpath_u8_rgb_gray(src, dst, code))
        {
            return;
        }
        if (try_cvtcolor_fastpath_u8_yuv420(src, dst, code))
        {
            return;
        }
        if (try_cvtcolor_fastpath_u8_yuv422(src, dst, code))
        {
            return;
        }
        if (try_cvtcolor_fastpath_u8_yuv444(src, dst, code))
        {
            return;
        }
    }
    else if (src.depth() == CV_32F)
    {
        if (try_cvtcolor_fastpath_f32_rgb_gray(src, dst, code))
        {
            return;
        }
    }

    cvtColor_fallback(src, dst, code);
}

} // namespace detail
} // namespace cvh
