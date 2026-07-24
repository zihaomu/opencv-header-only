#ifndef CVH_IMGPROC_REMAP_H
#define CVH_IMGPROC_REMAP_H

#include "convert_maps.h"
#include "detail/geometric_sampling.hpp"

namespace cvh {
namespace detail {

template<typename T>
inline void remap_typed(const Mat& source,
                        Mat& destination,
                        const Mat& map1,
                        const Mat& map2,
                        int interpolation,
                        int border_type,
                        const Scalar& border_value)
{
    const bool fixed = map1.type() == CV_16SC2;
    for (int row = 0; row < map1.size[0]; ++row)
    {
        T* output = reinterpret_cast<T*>(
            destination.data +
            static_cast<size_t>(row) * destination.step(0));
        for (int col = 0; col < map1.size[1]; ++col)
        {
            T* pixel =
                output + static_cast<size_t>(col) * source.channels();
            if (fixed)
            {
                const int integer_x =
                    map1.at<short>(row, col, 0);
                const int integer_y =
                    map1.at<short>(row, col, 1);
                const ushort fraction = map2.empty()
                    ? 0
                    : static_cast<ushort>(
                          map2.at<ushort>(row, col) &
                          (INTER_TAB_SIZE2 - 1));
                const int fraction_x =
                    fraction & (INTER_TAB_SIZE - 1);
                const int fraction_y =
                    fraction >> INTER_BITS;
                if (interpolation == INTER_NEAREST)
                {
                    geometric_write_nearest(
                        source,
                        pixel,
                        integer_x +
                            (fraction_x >= INTER_TAB_SIZE / 2),
                        integer_y +
                            (fraction_y >= INTER_TAB_SIZE / 2),
                        border_type,
                        border_value);
                }
                else
                {
                    geometric_write_linear(
                        source,
                        pixel,
                        integer_x,
                        integer_y,
                        static_cast<double>(fraction_x) /
                            INTER_TAB_SIZE,
                        static_cast<double>(fraction_y) /
                            INTER_TAB_SIZE,
                        border_type,
                        border_value);
                }
                continue;
            }

            double source_x = 0.0;
            double source_y = 0.0;
            read_map_coordinate(
                map1,
                map2,
                row,
                col,
                source_x,
                source_y);
            geometric_write_coordinate(
                source,
                pixel,
                source_x,
                source_y,
                interpolation,
                border_type,
                border_value,
                true,
                true);
        }
    }
}

}  // namespace detail

inline void remap(const Mat& src,
                  Mat& dst,
                  const Mat& map1,
                  const Mat& map2,
                  int interpolation,
                  int borderMode = BORDER_CONSTANT,
                  const Scalar& borderValue = Scalar())
{
    if (src.empty() || src.dims != 2)
    {
        CV_Error(Error::StsBadArg, "remap expects a non-empty 2D source");
    }
    detail::validate_map_input(map1, map2);
    if (interpolation != INTER_NEAREST &&
        interpolation != INTER_LINEAR)
    {
        CV_Error(
            Error::StsBadFlag,
            "remap supports INTER_NEAREST and INTER_LINEAR only");
    }
    const int border_type = detail::normalize_border_type(borderMode);
    if (!detail::is_supported_filter_border(border_type))
    {
        CV_Error(
            Error::StsBadFlag,
            "remap supports constant, replicate, reflect, and reflect-101 borders");
    }
    if ((src.depth() != CV_8U && src.depth() != CV_32F) ||
        (src.channels() != 1 &&
         src.channels() != 3 &&
         src.channels() != 4))
    {
        CV_Error(
            Error::StsUnsupportedFormat,
            "remap supports U8/F32 C1/C3/C4 source");
    }

    const Mat source =
        src.data == dst.data ? src.clone() : src;
    const Mat first =
        map1.data == dst.data ? map1.clone() : map1;
    const Mat second =
        !map2.empty() && map2.data == dst.data
            ? map2.clone()
            : map2;
    dst.create(
        {first.size[0], first.size[1]},
        source.type());
    if (source.depth() == CV_8U)
    {
        detail::remap_typed<uchar>(
            source,
            dst,
            first,
            second,
            interpolation,
            border_type,
            borderValue);
    }
    else
    {
        detail::remap_typed<float>(
            source,
            dst,
            first,
            second,
            interpolation,
            border_type,
            borderValue);
    }
}

}  // namespace cvh

#endif  // CVH_IMGPROC_REMAP_H
