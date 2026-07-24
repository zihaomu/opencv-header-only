#ifndef CVH_IMGPROC_DEMOSAICING_H
#define CVH_IMGPROC_DEMOSAICING_H

#include "detail/common.h"

#include <cmath>

namespace cvh
{
namespace demosaicing_detail
{

enum Channel
{
    Blue = 0,
    Green = 1,
    Red = 2,
};

inline int pattern_from_code(int code)
{
    switch (code)
    {
        case COLOR_BayerBG2BGR: return 0;  // RGGB
        case COLOR_BayerGB2BGR: return 1;  // GRBG
        case COLOR_BayerRG2BGR: return 2;  // BGGR
        case COLOR_BayerGR2BGR: return 3;  // GBRG
        default:
            CV_Error(
                Error::StsBadFlag,
                "demosaicing unsupported Bayer conversion code");
    }
    return -1;
}

inline Channel color_at(int y, int x, int pattern)
{
    const bool odd_y = (y & 1) != 0;
    const bool odd_x = (x & 1) != 0;
    switch (pattern)
    {
        case 0:
            if (!odd_y && !odd_x) return Red;
            if (odd_y && odd_x) return Blue;
            return Green;
        case 1:
            if (!odd_y && odd_x) return Red;
            if (odd_y && !odd_x) return Blue;
            return Green;
        case 2:
            if (!odd_y && !odd_x) return Blue;
            if (odd_y && odd_x) return Red;
            return Green;
        case 3:
            if (!odd_y && odd_x) return Blue;
            if (odd_y && !odd_x) return Red;
            return Green;
        default:
            return Green;
    }
}

inline uchar interpolate(const Mat& src,
                         int y,
                         int x,
                         int pattern,
                         Channel target)
{
    if (color_at(y, x, pattern) == target)
    {
        return src.at<uchar>(y, x);
    }
    int sum = 0;
    int count = 0;
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            if (dx == 0 && dy == 0)
            {
                continue;
            }
            const int source_y = detail::border_interpolate(
                y + dy, src.size.p[0], BORDER_REFLECT_101);
            const int source_x = detail::border_interpolate(
                x + dx, src.size.p[1], BORDER_REFLECT_101);
            if (color_at(source_y, source_x, pattern) == target)
            {
                sum += src.at<uchar>(source_y, source_x);
                ++count;
            }
        }
    }
    if (count == 0)
    {
        return src.at<uchar>(y, x);
    }
    return saturate_cast<uchar>(
        static_cast<int>(std::lrint(
            static_cast<double>(sum) / count)));
}

}  // namespace demosaicing_detail

inline void demosaicing(const Mat& src,
                        Mat& dst,
                        int code,
                        int dstCn = 0)
{
    if (src.empty() || src.dims != 2 || src.type() != CV_8UC1)
    {
        CV_Error(
            Error::StsBadArg,
            "demosaicing currently expects CV_8UC1 source");
    }
    if (dstCn != 0 && dstCn != 3)
    {
        CV_Error(
            Error::StsBadArg,
            "demosaicing currently supports three-channel output");
    }
    const int pattern = demosaicing_detail::pattern_from_code(code);
    const Mat source = src.data == dst.data ? src.clone() : src;
    dst.create(source.shape(), CV_8UC3);
    const int rows = source.size.p[0];
    const int cols = source.size.p[1];
    if (rows <= 2 || cols <= 2)
    {
        dst.setTo(Scalar::all(0.0));
        return;
    }
    for (int y = 1; y < rows - 1; ++y)
    {
        uchar* output =
            dst.data + static_cast<size_t>(y) * dst.step(0);
        for (int x = 1; x < cols - 1; ++x)
        {
            const size_t index = static_cast<size_t>(x) * 3;
            output[index] = demosaicing_detail::interpolate(
                source,
                y,
                x,
                pattern,
                demosaicing_detail::Blue);
            output[index + 1] = demosaicing_detail::interpolate(
                source,
                y,
                x,
                pattern,
                demosaicing_detail::Green);
            output[index + 2] = demosaicing_detail::interpolate(
                source,
                y,
                x,
                pattern,
                demosaicing_detail::Red);
        }
        std::copy_n(output + 3, 3, output);
        std::copy_n(
            output + static_cast<size_t>(cols - 2) * 3,
            3,
            output + static_cast<size_t>(cols - 1) * 3);
    }
    uchar* first_row = dst.data;
    const uchar* second_row = dst.data + dst.step(0);
    uchar* last_row =
        dst.data + static_cast<size_t>(rows - 1) * dst.step(0);
    const uchar* penultimate_row =
        dst.data + static_cast<size_t>(rows - 2) * dst.step(0);
    const size_t row_bytes = static_cast<size_t>(cols) * 3;
    std::copy_n(second_row, row_bytes, first_row);
    std::copy_n(penultimate_row, row_bytes, last_row);
}

}  // namespace cvh

#endif  // CVH_IMGPROC_DEMOSAICING_H
