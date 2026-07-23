#ifndef CVH_IMGPROC_DETAIL_CVTCOLOR_IMPL_HPP
#define CVH_IMGPROC_DETAIL_CVTCOLOR_IMPL_HPP

namespace cvh
{
namespace detail
{

inline void cvtColor_fast_impl(const Mat& src, Mat& dst, int code)
{
#if CVH_ENABLE_OPENCV_INTRIN
    if (src.depth() == CV_8U && (code == COLOR_BGR2GRAY || code == COLOR_RGB2GRAY))
    {
        cvtColor_fallback(src, dst, code);
        return;
    }
#endif

    if (src.depth() == CV_8U)
    {
        if (cvtcolor_rgb_gray_fastpath::try_cvtcolor_fastpath_u8_rgb_gray(src, dst, code))
        {
            return;
        }
        if (cvtcolor_yuv420_fastpath::try_cvtcolor_fastpath_u8_yuv420(src, dst, code))
        {
            return;
        }
        if (cvtcolor_yuv422_fastpath::try_cvtcolor_fastpath_u8_yuv422(src, dst, code))
        {
            return;
        }
        if (cvtcolor_yuv444_fastpath::try_cvtcolor_fastpath_u8_yuv444(src, dst, code))
        {
            return;
        }
    }
    else if (src.depth() == CV_32F)
    {
        if (cvtcolor_rgb_gray_fastpath::try_cvtcolor_fastpath_f32_rgb_gray(src, dst, code))
        {
            return;
        }
    }

    cvtColor_fallback(src, dst, code);
}

} // namespace detail
} // namespace cvh

#endif // CVH_IMGPROC_DETAIL_CVTCOLOR_IMPL_HPP
