#ifndef CVH_IMGPROC_COLORMAP_H
#define CVH_IMGPROC_COLORMAP_H

#include "detail/common.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace cvh
{
namespace colormap_detail
{

inline uchar channel(double value)
{
    return saturate_cast<uchar>(
        std::clamp(value, 0.0, 1.0) * 255.0);
}

inline std::array<uchar, 3> builtin_color(int value, int colormap)
{
    const double t = static_cast<double>(value) / 255.0;
    double red = 0.0;
    double green = 0.0;
    double blue = 0.0;
    switch (colormap)
    {
        case COLORMAP_AUTUMN:
            red = 1.0;
            green = t;
            break;
        case COLORMAP_JET:
            red = 1.5 - std::fabs(4.0 * t - 3.0);
            green = 1.5 - std::fabs(4.0 * t - 2.0);
            blue = 1.5 - std::fabs(4.0 * t - 1.0);
            break;
        case COLORMAP_WINTER:
            green = t;
            blue = 1.0 - 0.5 * t;
            break;
        case COLORMAP_COOL:
            red = t;
            green = 1.0 - t;
            blue = 1.0;
            break;
        case COLORMAP_HOT:
            red = (8.0 / 3.0) * t;
            green = (8.0 / 3.0) * t - 1.0;
            blue = 4.0 * t - 3.0;
            break;
        default:
            CV_Error(
                Error::StsBadArg,
                "applyColorMap unsupported built-in colormap id");
    }
    return {channel(blue), channel(green), channel(red)};
}

inline const uchar* lookup_entry(const Mat& lookup, int value)
{
    if (lookup.isContinuous())
    {
        return lookup.data +
               static_cast<size_t>(value) * lookup.elemSize();
    }
    const int columns = lookup.size.p[1];
    const int row = value / columns;
    const int column = value % columns;
    return lookup.data +
           static_cast<size_t>(row) * lookup.step(0) +
           static_cast<size_t>(column) * lookup.elemSize();
}

}  // namespace colormap_detail

inline void applyColorMap(const Mat& src, Mat& dst, const Mat& userColor)
{
    if (src.empty() || src.dims != 2 || src.type() != CV_8UC1)
    {
        CV_Error(Error::StsBadArg, "applyColorMap expects CV_8UC1 source");
    }
    if (userColor.empty() || userColor.dims != 2 ||
        userColor.total() != 256 ||
        (userColor.type() != CV_8UC1 &&
         userColor.type() != CV_8UC3))
    {
        CV_Error(
            Error::StsBadArg,
            "applyColorMap userColor must be a 256-entry CV_8UC1/CV_8UC3 table");
    }
    const Mat source = src.data == dst.data ? src.clone() : src;
    const Mat lookup =
        userColor.data == dst.data ? userColor.clone() : userColor;
    dst.create(
        source.shape(),
        CV_MAKETYPE(CV_8U, lookup.channels()));
    const size_t output_pixel_size =
        static_cast<size_t>(lookup.channels());
    for (int y = 0; y < source.size.p[0]; ++y)
    {
        const uchar* input =
            source.data + static_cast<size_t>(y) * source.step(0);
        uchar* output =
            dst.data + static_cast<size_t>(y) * dst.step(0);
        for (int x = 0; x < source.size.p[1]; ++x)
        {
            const uchar* entry =
                colormap_detail::lookup_entry(lookup, input[x]);
            std::copy_n(
                entry,
                output_pixel_size,
                output + static_cast<size_t>(x) * output_pixel_size);
        }
    }
}

inline void applyColorMap(const Mat& src, Mat& dst, int colormap)
{
    if (src.empty() || src.dims != 2 || src.type() != CV_8UC1)
    {
        CV_Error(Error::StsBadArg, "applyColorMap expects CV_8UC1 source");
    }
    if (colormap != COLORMAP_AUTUMN &&
        colormap != COLORMAP_JET &&
        colormap != COLORMAP_WINTER &&
        colormap != COLORMAP_COOL &&
        colormap != COLORMAP_HOT)
    {
        CV_Error(
            Error::StsBadArg,
            "applyColorMap unsupported built-in colormap id");
    }

    const Mat source = src.data == dst.data ? src.clone() : src;
    dst.create(source.shape(), CV_8UC3);
    for (int y = 0; y < source.size.p[0]; ++y)
    {
        const uchar* input =
            source.data + static_cast<size_t>(y) * source.step(0);
        uchar* output =
            dst.data + static_cast<size_t>(y) * dst.step(0);
        for (int x = 0; x < source.size.p[1]; ++x)
        {
            const std::array<uchar, 3> color =
                colormap_detail::builtin_color(input[x], colormap);
            std::copy(
                color.begin(),
                color.end(),
                output + static_cast<size_t>(x) * 3);
        }
    }
}

}  // namespace cvh

#endif  // CVH_IMGPROC_COLORMAP_H
