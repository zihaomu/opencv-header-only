#ifndef CVH_IMGPROC_GEOMETRY_TRANSFORM_H
#define CVH_IMGPROC_GEOMETRY_TRANSFORM_H

#include "detail/common.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace cvh
{

struct AffineMatrix2x3d
{
    double val[6] = {};

    constexpr double& operator()(int row, int col)
    {
        return val[row * 3 + col];
    }

    constexpr const double& operator()(int row, int col) const
    {
        return val[row * 3 + col];
    }
};

namespace geometry_transform_detail
{

template<size_t N>
inline std::array<double, N> solve_fixed(
    std::array<std::array<double, N>, N> matrix,
    std::array<double, N> rhs,
    const char* name)
{
    std::array<double, N> column_scales = {};
    for (size_t column = 0; column < N; ++column)
    {
        for (size_t row = 0; row < N; ++row)
        {
            column_scales[column] = std::max(
                column_scales[column],
                std::fabs(matrix[row][column]));
        }
        if (column_scales[column] == 0.0)
        {
            CV_Error_(
                Error::StsBadArg,
                ("%s point set is degenerate", name));
        }
        for (size_t row = 0; row < N; ++row)
        {
            matrix[row][column] /= column_scales[column];
        }
    }
    const double epsilon =
        std::numeric_limits<double>::epsilon() * 256.0;
    for (size_t pivot = 0; pivot < N; ++pivot)
    {
        size_t best = pivot;
        for (size_t row = pivot + 1; row < N; ++row)
        {
            if (std::fabs(matrix[row][pivot]) >
                std::fabs(matrix[best][pivot]))
            {
                best = row;
            }
        }
        if (std::fabs(matrix[best][pivot]) <= epsilon)
        {
            CV_Error_(
                Error::StsBadArg,
                ("%s point set is degenerate", name));
        }
        if (best != pivot)
        {
            std::swap(matrix[best], matrix[pivot]);
            std::swap(rhs[best], rhs[pivot]);
        }
        const double diagonal = matrix[pivot][pivot];
        for (size_t column = pivot; column < N; ++column)
        {
            matrix[pivot][column] /= diagonal;
        }
        rhs[pivot] /= diagonal;
        for (size_t row = 0; row < N; ++row)
        {
            if (row == pivot)
            {
                continue;
            }
            const double factor = matrix[row][pivot];
            for (size_t column = pivot; column < N; ++column)
            {
                matrix[row][column] -=
                    factor * matrix[pivot][column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    for (size_t index = 0; index < N; ++index)
    {
        rhs[index] /= column_scales[index];
    }
    return rhs;
}

inline Mat affine_to_mat(const AffineMatrix2x3d& matrix)
{
    Mat result({2, 3}, CV_64FC1);
    std::copy(
        matrix.val,
        matrix.val + 6,
        reinterpret_cast<double*>(result.data));
    return result;
}

}  // namespace geometry_transform_detail

template<typename T>
inline AffineMatrix2x3d getRotationMatrix2D_(
    Point_<T> center,
    double angle,
    double scale)
{
    if (!std::isfinite(static_cast<double>(center.x)) ||
        !std::isfinite(static_cast<double>(center.y)) ||
        !std::isfinite(angle) || !std::isfinite(scale))
    {
        CV_Error(
            Error::StsBadArg,
            "getRotationMatrix2D_ parameters must be finite");
    }
    const double radians =
        angle * 3.141592653589793238462643383279502884 / 180.0;
    const double alpha = std::cos(radians) * scale;
    const double beta = std::sin(radians) * scale;
    const double center_x = static_cast<double>(center.x);
    const double center_y = static_cast<double>(center.y);
    return AffineMatrix2x3d{
        {alpha,
         beta,
         (1.0 - alpha) * center_x - beta * center_y,
         -beta,
         alpha,
         beta * center_x + (1.0 - alpha) * center_y}};
}

template<typename T>
inline Mat getRotationMatrix2D(Point_<T> center,
                               double angle,
                               double scale)
{
    return geometry_transform_detail::affine_to_mat(
        getRotationMatrix2D_(center, angle, scale));
}

template<typename T>
inline Mat getAffineTransform(const Point_<T> src[3],
                              const Point_<T> dst[3])
{
    std::array<std::array<double, 3>, 3> matrix = {};
    std::array<double, 3> target_x = {};
    std::array<double, 3> target_y = {};
    for (size_t index = 0; index < 3; ++index)
    {
        matrix[index] = {
            static_cast<double>(src[index].x),
            static_cast<double>(src[index].y),
            1.0};
        target_x[index] = static_cast<double>(dst[index].x);
        target_y[index] = static_cast<double>(dst[index].y);
    }
    const std::array<double, 3> first =
        geometry_transform_detail::solve_fixed<3>(
            matrix, target_x, "getAffineTransform");
    const std::array<double, 3> second =
        geometry_transform_detail::solve_fixed<3>(
            matrix, target_y, "getAffineTransform");
    Mat result({2, 3}, CV_64FC1);
    double* output = reinterpret_cast<double*>(result.data);
    std::copy(first.begin(), first.end(), output);
    std::copy(second.begin(), second.end(), output + 3);
    return result;
}

template<typename T>
inline Mat getPerspectiveTransform(const Point_<T> src[4],
                                   const Point_<T> dst[4],
                                   int solveMethod = DECOMP_LU)
{
    if (solveMethod != DECOMP_LU)
    {
        CV_Error(
            Error::StsBadFlag,
            "getPerspectiveTransform currently supports DECOMP_LU only");
    }
    std::array<std::array<double, 8>, 8> matrix = {};
    std::array<double, 8> rhs = {};
    for (size_t index = 0; index < 4; ++index)
    {
        const double x = static_cast<double>(src[index].x);
        const double y = static_cast<double>(src[index].y);
        const double u = static_cast<double>(dst[index].x);
        const double v = static_cast<double>(dst[index].y);
        matrix[index] = {x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u};
        matrix[index + 4] =
            {0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v};
        rhs[index] = u;
        rhs[index + 4] = v;
    }
    const std::array<double, 8> solution =
        geometry_transform_detail::solve_fixed<8>(
            matrix, rhs, "getPerspectiveTransform");
    Mat result({3, 3}, CV_64FC1);
    double* output = reinterpret_cast<double*>(result.data);
    std::copy(solution.begin(), solution.end(), output);
    output[8] = 1.0;
    return result;
}

inline void invertAffineTransform(const Mat& matrix, Mat& inverse)
{
    if (matrix.empty() || matrix.dims != 2 ||
        matrix.size.p[0] != 2 || matrix.size.p[1] != 3 ||
        (matrix.type() != CV_32FC1 && matrix.type() != CV_64FC1))
    {
        CV_Error(
            Error::StsBadArg,
            "invertAffineTransform expects 2x3 CV_32F/CV_64F matrix");
    }
    const Mat source =
        matrix.data == inverse.data ? matrix.clone() : matrix;
    double values[6] = {};
    for (int row = 0; row < 2; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            values[row * 3 + col] =
                source.depth() == CV_32F
                    ? source.at<float>(row, col)
                    : source.at<double>(row, col);
        }
    }
    const double determinant =
        values[0] * values[4] - values[1] * values[3];
    double output[6] = {};
    if (determinant != 0.0)
    {
        const double reciprocal = 1.0 / determinant;
        output[0] = values[4] * reciprocal;
        output[1] = -values[1] * reciprocal;
        output[3] = -values[3] * reciprocal;
        output[4] = values[0] * reciprocal;
        output[2] =
            -output[0] * values[2] - output[1] * values[5];
        output[5] =
            -output[3] * values[2] - output[4] * values[5];
    }
    inverse.create({2, 3}, source.type());
    for (int row = 0; row < 2; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            if (inverse.depth() == CV_32F)
            {
                inverse.at<float>(row, col) =
                    static_cast<float>(output[row * 3 + col]);
            }
            else
            {
                inverse.at<double>(row, col) =
                    output[row * 3 + col];
            }
        }
    }
}

}  // namespace cvh

#endif  // CVH_IMGPROC_GEOMETRY_TRANSFORM_H
