#ifndef CVH_IMGPROC_WARP_PERSPECTIVE_H
#define CVH_IMGPROC_WARP_PERSPECTIVE_H

#include "detail/geometric_sampling.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace cvh {
namespace detail {

inline void read_perspective_matrix(const Mat& matrix, double values[9])
{
    if (matrix.empty() || matrix.dims != 2 ||
        matrix.size[0] != 3 || matrix.size[1] != 3 ||
        (matrix.type() != CV_32FC1 && matrix.type() != CV_64FC1))
    {
        CV_Error(
            Error::StsBadArg,
            "warpPerspective expects a 3x3 F32/F64 matrix");
    }
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            const double value = matrix.depth() == CV_32F
                ? matrix.at<float>(row, col)
                : matrix.at<double>(row, col);
            if (!std::isfinite(value))
            {
                CV_Error(
                    Error::StsBadArg,
                    "warpPerspective matrix values must be finite");
            }
            values[row * 3 + col] = value;
        }
    }
}

inline void invert_perspective_matrix(double values[9])
{
    double scale = 0.0;
    for (int index = 0; index < 9; ++index)
    {
        scale = std::max(scale, std::fabs(values[index]));
    }
    if (scale == 0.0)
    {
        CV_Error(
            Error::StsBadArg,
            "warpPerspective transform matrix is singular");
    }
    double matrix[9] = {};
    for (int index = 0; index < 9; ++index)
    {
        matrix[index] = values[index] / scale;
    }
    const double determinant =
        matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
        matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
        matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
    if (std::fabs(determinant) <=
        std::numeric_limits<double>::epsilon() * 64.0)
    {
        CV_Error(
            Error::StsBadArg,
            "warpPerspective transform matrix is singular");
    }
    const double factor = 1.0 / (determinant * scale);
    const double inverse[9] = {
        (matrix[4] * matrix[8] - matrix[5] * matrix[7]) * factor,
        (matrix[2] * matrix[7] - matrix[1] * matrix[8]) * factor,
        (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * factor,
        (matrix[5] * matrix[6] - matrix[3] * matrix[8]) * factor,
        (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * factor,
        (matrix[2] * matrix[3] - matrix[0] * matrix[5]) * factor,
        (matrix[3] * matrix[7] - matrix[4] * matrix[6]) * factor,
        (matrix[1] * matrix[6] - matrix[0] * matrix[7]) * factor,
        (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * factor,
    };
    std::copy(inverse, inverse + 9, values);
}

template<typename T>
inline void warp_perspective_typed(const Mat& source,
                                   Mat& destination,
                                   const double matrix[9],
                                   int interpolation,
                                   int border_type,
                                   const Scalar& border_value)
{
    for (int row = 0; row < destination.size[0]; ++row)
    {
        T* output = reinterpret_cast<T*>(
            destination.data +
            static_cast<size_t>(row) * destination.step(0));
        for (int col = 0; col < destination.size[1]; ++col)
        {
            const double denominator =
                matrix[6] * col + matrix[7] * row + matrix[8];
            const double reciprocal =
                denominator != 0.0 ? 1.0 / denominator : 0.0;
            const double source_x =
                (matrix[0] * col + matrix[1] * row + matrix[2]) *
                reciprocal;
            const double source_y =
                (matrix[3] * col + matrix[4] * row + matrix[5]) *
                reciprocal;
            geometric_write_coordinate(
                source,
                output + static_cast<size_t>(col) * source.channels(),
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

inline void warpPerspective(const Mat& src,
                            Mat& dst,
                            const Mat& matrix,
                            Size dsize,
                            int flags = INTER_LINEAR,
                            int borderMode = BORDER_CONSTANT,
                            const Scalar& borderValue = Scalar())
{
    if (src.empty() || src.dims != 2)
    {
        CV_Error(
            Error::StsBadArg,
            "warpPerspective expects a non-empty 2D source");
    }
    if (dsize.width <= 0 || dsize.height <= 0)
    {
        CV_Error(
            Error::StsBadSize,
            "warpPerspective output size must be positive");
    }
    const int interpolation = flags & INTER_MAX;
    if ((interpolation != INTER_NEAREST &&
         interpolation != INTER_LINEAR) ||
        (flags & ~(INTER_MAX | WARP_INVERSE_MAP)) != 0)
    {
        CV_Error(
            Error::StsBadFlag,
            "warpPerspective supports nearest/linear and inverse-map only");
    }
    const int border_type = detail::normalize_border_type(borderMode);
    if (!detail::is_supported_filter_border(border_type))
    {
        CV_Error(
            Error::StsBadFlag,
            "warpPerspective supports constant, replicate, reflect, and reflect-101 borders");
    }
    if ((src.depth() != CV_8U && src.depth() != CV_32F) ||
        (src.channels() != 1 &&
         src.channels() != 3 &&
         src.channels() != 4))
    {
        CV_Error(
            Error::StsUnsupportedFormat,
            "warpPerspective supports U8/F32 C1/C3/C4 source");
    }

    double values[9] = {};
    detail::read_perspective_matrix(matrix, values);
    if ((flags & WARP_INVERSE_MAP) == 0)
    {
        detail::invert_perspective_matrix(values);
    }
    const Mat source =
        src.data == dst.data ? src.clone() : src;
    dst.create({dsize.height, dsize.width}, source.type());
    if (source.depth() == CV_8U)
    {
        detail::warp_perspective_typed<uchar>(
            source,
            dst,
            values,
            interpolation,
            border_type,
            borderValue);
    }
    else
    {
        detail::warp_perspective_typed<float>(
            source,
            dst,
            values,
            interpolation,
            border_type,
            borderValue);
    }
}

}  // namespace cvh

#endif  // CVH_IMGPROC_WARP_PERSPECTIVE_H
