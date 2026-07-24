#ifndef CVH_IMGPROC_CONVERT_MAPS_H
#define CVH_IMGPROC_CONVERT_MAPS_H

#include "detail/common.h"

#include <cmath>
#include <limits>

namespace cvh {
namespace detail {

inline bool same_map_size(const Mat& first, const Mat& second)
{
    return first.dims == 2 && second.dims == 2 &&
        first.size[0] == second.size[0] &&
        first.size[1] == second.size[1];
}

inline int map_round_to_int(double value)
{
    if (!std::isfinite(value))
    {
        return value > 0.0
            ? std::numeric_limits<int>::max()
            : std::numeric_limits<int>::min();
    }
    if (value >= static_cast<double>(std::numeric_limits<int>::max()))
    {
        return std::numeric_limits<int>::max();
    }
    if (value <= static_cast<double>(std::numeric_limits<int>::min()))
    {
        return std::numeric_limits<int>::min();
    }
    return static_cast<int>(std::lrint(value));
}

inline void read_map_coordinate(const Mat& first,
                                const Mat& second,
                                int row,
                                int col,
                                double& x,
                                double& y)
{
    if (first.type() == CV_32FC2)
    {
        x = first.at<float>(row, col, 0);
        y = first.at<float>(row, col, 1);
        return;
    }
    if (first.type() == CV_32FC1)
    {
        x = first.at<float>(row, col);
        y = second.at<float>(row, col);
        return;
    }
    const short integer_x = first.at<short>(row, col, 0);
    const short integer_y = first.at<short>(row, col, 1);
    const ushort fraction = second.empty()
        ? 0
        : static_cast<ushort>(
              second.at<ushort>(row, col) &
              (INTER_TAB_SIZE2 - 1));
    x = static_cast<double>(integer_x) +
        static_cast<double>(fraction & (INTER_TAB_SIZE - 1)) /
            INTER_TAB_SIZE;
    y = static_cast<double>(integer_y) +
        static_cast<double>(fraction >> INTER_BITS) /
            INTER_TAB_SIZE;
}

inline void validate_map_input(const Mat& first, const Mat& second)
{
    if (first.empty() || first.dims != 2)
    {
        CV_Error(Error::StsBadArg, "convertMaps expects a non-empty 2D map1");
    }
    const bool float_pair =
        first.type() == CV_32FC1 &&
        second.type() == CV_32FC1 &&
        same_map_size(first, second);
    const bool float_interleaved =
        first.type() == CV_32FC2 && second.empty();
    const bool fixed_pair =
        first.type() == CV_16SC2 &&
        (second.empty() ||
         (second.type() == CV_16UC1 &&
          same_map_size(first, second)));
    if (!float_pair && !float_interleaved && !fixed_pair)
    {
        CV_Error(
            Error::StsBadArg,
            "convertMaps expects F32 pair, F32C2, or S16C2/U16 map");
    }
}

}  // namespace detail

inline void convertMaps(const Mat& map1,
                        const Mat& map2,
                        Mat& dstmap1,
                        Mat& dstmap2,
                        int dstmap1type,
                        bool nninterpolation = false)
{
    if (&dstmap1 == &dstmap2)
    {
        CV_Error(
            Error::StsBadArg,
            "convertMaps outputs must be distinct Mat objects");
    }
    detail::validate_map_input(map1, map2);
    const Mat first = map1.clone();
    const Mat second = map2.empty() ? Mat() : map2.clone();
    if (dstmap1type <= 0)
    {
        dstmap1type =
            first.type() == CV_16SC2 ? CV_32FC2 : CV_16SC2;
    }
    if (dstmap1type != CV_16SC2 &&
        dstmap1type != CV_32FC1 &&
        dstmap1type != CV_32FC2)
    {
        CV_Error(
            Error::StsBadArg,
            "convertMaps output map1 must be S16C2, F32C1, or F32C2");
    }

    dstmap1.create(
        {first.size[0], first.size[1]},
        dstmap1type);
    if (!nninterpolation && dstmap1type != CV_32FC2)
    {
        dstmap2.create(
            {first.size[0], first.size[1]},
            dstmap1type == CV_16SC2 ? CV_16UC1 : CV_32FC1);
    }
    else
    {
        dstmap2.release();
    }

    for (int row = 0; row < first.size[0]; ++row)
    {
        for (int col = 0; col < first.size[1]; ++col)
        {
            double x = 0.0;
            double y = 0.0;
            detail::read_map_coordinate(
                first,
                second,
                row,
                col,
                x,
                y);
            if (dstmap1type == CV_32FC2)
            {
                dstmap1.at<float>(row, col, 0) =
                    static_cast<float>(x);
                dstmap1.at<float>(row, col, 1) =
                    static_cast<float>(y);
                continue;
            }
            if (dstmap1type == CV_32FC1)
            {
                dstmap1.at<float>(row, col) =
                    static_cast<float>(x);
                dstmap2.at<float>(row, col) =
                    static_cast<float>(y);
                continue;
            }

            if (nninterpolation)
            {
                dstmap1.at<short>(row, col, 0) =
                    saturate_cast<short>(detail::map_round_to_int(x));
                dstmap1.at<short>(row, col, 1) =
                    saturate_cast<short>(detail::map_round_to_int(y));
                continue;
            }
            const int scaled_x = detail::map_round_to_int(
                x * INTER_TAB_SIZE);
            const int scaled_y = detail::map_round_to_int(
                y * INTER_TAB_SIZE);
            const int integer_x = static_cast<int>(std::floor(
                static_cast<double>(scaled_x) / INTER_TAB_SIZE));
            const int integer_y = static_cast<int>(std::floor(
                static_cast<double>(scaled_y) / INTER_TAB_SIZE));
            const int fraction_x =
                scaled_x - integer_x * INTER_TAB_SIZE;
            const int fraction_y =
                scaled_y - integer_y * INTER_TAB_SIZE;
            dstmap1.at<short>(row, col, 0) =
                saturate_cast<short>(integer_x);
            dstmap1.at<short>(row, col, 1) =
                saturate_cast<short>(integer_y);
            dstmap2.at<ushort>(row, col) = static_cast<ushort>(
                fraction_y * INTER_TAB_SIZE + fraction_x);
        }
    }
}

}  // namespace cvh

#endif  // CVH_IMGPROC_CONVERT_MAPS_H
