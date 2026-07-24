#ifndef CVH_IMGPROC_SQR_BOX_FILTER_H
#define CVH_IMGPROC_SQR_BOX_FILTER_H

#include "detail/common.h"

#include <vector>

namespace cvh
{
namespace sqr_box_detail
{

inline double read_value(const Mat& src, int y, int x, int channel)
{
    const size_t index =
        static_cast<size_t>(x) * src.channels() +
        static_cast<size_t>(channel);
    const uchar* row = src.data + static_cast<size_t>(y) * src.step(0);
    if (src.depth() == CV_8U)
    {
        return row[index];
    }
    return reinterpret_cast<const float*>(row)[index];
}

inline void write_value(Mat& dst,
                        int y,
                        int x,
                        int channel,
                        double value)
{
    const size_t index =
        static_cast<size_t>(x) * dst.channels() +
        static_cast<size_t>(channel);
    uchar* row = dst.data + static_cast<size_t>(y) * dst.step(0);
    if (dst.depth() == CV_8U)
    {
        row[index] = saturate_cast<uchar>(value);
    }
    else if (dst.depth() == CV_32F)
    {
        reinterpret_cast<float*>(row)[index] = static_cast<float>(value);
    }
    else
    {
        reinterpret_cast<double*>(row)[index] = value;
    }
}

}  // namespace sqr_box_detail

inline void sqrBoxFilter(const Mat& src,
                         Mat& dst,
                         int ddepth,
                         Size ksize,
                         Point anchor = Point(-1, -1),
                         bool normalize = true,
                         int borderType = BORDER_DEFAULT)
{
    if (src.empty() || src.dims != 2 ||
        (src.depth() != CV_8U && src.depth() != CV_32F) ||
        (src.channels() != 1 && src.channels() != 3 && src.channels() != 4))
    {
        CV_Error(Error::StsBadArg, "sqrBoxFilter unsupported src");
    }
    if (ksize.width <= 0 || ksize.height <= 0)
    {
        CV_Error(Error::StsBadSize, "sqrBoxFilter invalid ksize");
    }
    if (anchor.x < 0) anchor.x = ksize.width / 2;
    if (anchor.y < 0) anchor.y = ksize.height / 2;
    if (anchor.x < 0 || anchor.x >= ksize.width ||
        anchor.y < 0 || anchor.y >= ksize.height)
    {
        CV_Error(Error::StsOutOfRange, "sqrBoxFilter invalid anchor");
    }
    const int border_type = detail::normalize_border_type(borderType);
    if (!detail::is_supported_filter_border(border_type))
    {
        CV_Error(Error::StsBadArg, "sqrBoxFilter unsupported border");
    }
    const int output_depth =
        ddepth < 0 ? src.depth() : CV_MAT_DEPTH(ddepth);
    if (output_depth != CV_8U && output_depth != CV_32F &&
        output_depth != CV_64F)
    {
        CV_Error(Error::StsBadArg, "sqrBoxFilter unsupported ddepth");
    }

    const Mat source = src.data == dst.data ? src.clone() : src;
    dst.create(
        source.shape(), CV_MAKETYPE(output_depth, source.channels()));
    const int area = ksize.width * ksize.height;
    const double scale = normalize ? 1.0 / area : 1.0;
    for (int y = 0; y < source.size.p[0]; ++y)
    {
        for (int x = 0; x < source.size.p[1]; ++x)
        {
            for (int ch = 0; ch < source.channels(); ++ch)
            {
                long double accumulator = 0.0L;
                for (int ky = 0; ky < ksize.height; ++ky)
                {
                    const int source_y = detail::border_interpolate(
                        y + ky - anchor.y,
                        source.size.p[0],
                        border_type);
                    if (source_y < 0)
                    {
                        continue;
                    }
                    for (int kx = 0; kx < ksize.width; ++kx)
                    {
                        const int source_x = detail::border_interpolate(
                            x + kx - anchor.x,
                            source.size.p[1],
                            border_type);
                        if (source_x < 0)
                        {
                            continue;
                        }
                        const double value = sqr_box_detail::read_value(
                            source, source_y, source_x, ch);
                        accumulator +=
                            static_cast<long double>(value) * value;
                    }
                }
                sqr_box_detail::write_value(
                    dst,
                    y,
                    x,
                    ch,
                    static_cast<double>(accumulator) * scale);
            }
        }
    }
}

}  // namespace cvh

#endif  // CVH_IMGPROC_SQR_BOX_FILTER_H
