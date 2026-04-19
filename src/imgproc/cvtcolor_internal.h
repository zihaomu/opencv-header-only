#ifndef CVH_IMGPROC_BACKEND_CVTCOLOR_INTERNAL_H
#define CVH_IMGPROC_BACKEND_CVTCOLOR_INTERNAL_H

#include "cvh/imgproc/imgproc.h"

namespace cvh
{
namespace detail
{

bool try_cvtcolor_fastpath_u8_rgb_gray(const Mat& src, Mat& dst, int code);
bool try_cvtcolor_fastpath_f32_rgb_gray(const Mat& src, Mat& dst, int code);
bool try_cvtcolor_fastpath_u8_yuv420(const Mat& src, Mat& dst, int code);
bool try_cvtcolor_fastpath_u8_yuv422(const Mat& src, Mat& dst, int code);
bool try_cvtcolor_fastpath_u8_yuv444(const Mat& src, Mat& dst, int code);

} // namespace detail
} // namespace cvh

#endif // CVH_IMGPROC_BACKEND_CVTCOLOR_INTERNAL_H
