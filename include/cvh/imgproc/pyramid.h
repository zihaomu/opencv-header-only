#ifndef CVH_IMGPROC_PYRAMID_H
#define CVH_IMGPROC_PYRAMID_H

#include "detail/common.h"

#include <cmath>
#include <type_traits>
#include <vector>

namespace cvh
{
namespace pyramid_detail
{

inline void validate_source(const Mat& src, const char* name)
{
    if (src.empty() || src.dims != 2 ||
        (src.depth() != CV_8U && src.depth() != CV_32F) ||
        (src.channels() != 1 && src.channels() != 3 &&
         src.channels() != 4))
    {
        CV_Error_(Error::StsBadArg, ("%s unsupported source", name));
    }
}

inline int pyramid_border(int border_type, bool upsample)
{
    const int normalized = detail::normalize_border_type(border_type);
    if (upsample)
    {
        if (normalized != BORDER_REFLECT_101)
        {
            CV_Error(
                Error::StsBadArg,
                "pyrUp currently supports BORDER_DEFAULT only");
        }
    }
    else if (normalized == BORDER_CONSTANT ||
             (normalized != BORDER_REPLICATE &&
              normalized != BORDER_REFLECT &&
              normalized != BORDER_REFLECT_101 &&
              normalized != BORDER_WRAP))
    {
        CV_Error(Error::StsBadArg, "pyrDown unsupported border");
    }
    return normalized;
}

template<typename T>
inline T cast_value(double value)
{
    if constexpr (std::is_same<T, uchar>::value)
    {
        return saturate_cast<uchar>(
            static_cast<int>(std::lrint(value)));
    }
    return static_cast<float>(value);
}

template<typename T>
inline void downsample(const Mat& src, Mat& dst, int border_type)
{
    static constexpr int weights[5] = {1, 4, 6, 4, 1};
    const int rows = src.size.p[0];
    const int cols = src.size.p[1];
    const int channels = src.channels();
    for (int y = 0; y < dst.size.p[0]; ++y)
    {
        T* output = reinterpret_cast<T*>(
            dst.data + static_cast<size_t>(y) * dst.step(0));
        for (int x = 0; x < dst.size.p[1]; ++x)
        {
            for (int ch = 0; ch < channels; ++ch)
            {
                double sum = 0.0;
                for (int ky = -2; ky <= 2; ++ky)
                {
                    const int source_y = detail::border_interpolate(
                        2 * y + ky, rows, border_type);
                    const T* input = reinterpret_cast<const T*>(
                        src.data +
                        static_cast<size_t>(source_y) * src.step(0));
                    for (int kx = -2; kx <= 2; ++kx)
                    {
                        const int source_x = detail::border_interpolate(
                            2 * x + kx, cols, border_type);
                        sum +=
                            static_cast<double>(
                                weights[ky + 2] * weights[kx + 2]) *
                            static_cast<double>(
                                input[
                                    static_cast<size_t>(source_x) * channels +
                                    static_cast<size_t>(ch)]);
                    }
                }
                output[static_cast<size_t>(x) * channels +
                       static_cast<size_t>(ch)] =
                    cast_value<T>(sum / 256.0);
            }
        }
    }
}

template<typename T>
inline void upsample(const Mat& src, Mat& dst, int border_type)
{
    static constexpr int weights[5] = {1, 4, 6, 4, 1};
    const int rows = src.size.p[0];
    const int cols = src.size.p[1];
    const int channels = src.channels();
    for (int y = 0; y < dst.size.p[0]; ++y)
    {
        T* output = reinterpret_cast<T*>(
            dst.data + static_cast<size_t>(y) * dst.step(0));
        for (int x = 0; x < dst.size.p[1]; ++x)
        {
            for (int ch = 0; ch < channels; ++ch)
            {
                double sum = 0.0;
                for (int ky = -2; ky <= 2; ++ky)
                {
                    const int expanded_y = detail::border_interpolate(
                        y + ky, rows * 2, border_type);
                    if ((expanded_y & 1) != 0)
                    {
                        continue;
                    }
                    const int source_y = expanded_y / 2;
                    const T* input = reinterpret_cast<const T*>(
                        src.data +
                        static_cast<size_t>(source_y) * src.step(0));
                    for (int kx = -2; kx <= 2; ++kx)
                    {
                        const int expanded_x = detail::border_interpolate(
                            x + kx, cols * 2, border_type);
                        if ((expanded_x & 1) != 0)
                        {
                            continue;
                        }
                        const int source_x = expanded_x / 2;
                        sum +=
                            static_cast<double>(
                                weights[ky + 2] * weights[kx + 2]) *
                            static_cast<double>(
                                input[
                                    static_cast<size_t>(source_x) * channels +
                                    static_cast<size_t>(ch)]);
                    }
                }
                output[static_cast<size_t>(x) * channels +
                       static_cast<size_t>(ch)] =
                    cast_value<T>(sum / 64.0);
            }
        }
    }
}

}  // namespace pyramid_detail

inline void pyrDown(const Mat& src,
                    Mat& dst,
                    const Size& dstsize = Size(),
                    int borderType = BORDER_DEFAULT)
{
    pyramid_detail::validate_source(src, "pyrDown");
    const int border_type =
        pyramid_detail::pyramid_border(borderType, false);
    const int output_cols =
        dstsize.width > 0 ? dstsize.width : (src.size.p[1] + 1) / 2;
    const int output_rows =
        dstsize.height > 0 ? dstsize.height : (src.size.p[0] + 1) / 2;
    if ((dstsize.width == 0) != (dstsize.height == 0) ||
        output_cols <= 0 || output_rows <= 0 ||
        std::abs(output_cols * 2 - src.size.p[1]) > 2 ||
        std::abs(output_rows * 2 - src.size.p[0]) > 2)
    {
        CV_Error(Error::StsBadSize, "pyrDown invalid destination size");
    }
    const Mat source = src.data == dst.data ? src.clone() : src;
    dst.create(
        {output_rows, output_cols}, source.type());
    if (source.depth() == CV_8U)
    {
        pyramid_detail::downsample<uchar>(
            source, dst, border_type);
    }
    else
    {
        pyramid_detail::downsample<float>(
            source, dst, border_type);
    }
}

inline void pyrUp(const Mat& src,
                  Mat& dst,
                  const Size& dstsize = Size(),
                  int borderType = BORDER_DEFAULT)
{
    pyramid_detail::validate_source(src, "pyrUp");
    const int border_type =
        pyramid_detail::pyramid_border(borderType, true);
    const int output_cols =
        dstsize.width > 0 ? dstsize.width : src.size.p[1] * 2;
    const int output_rows =
        dstsize.height > 0 ? dstsize.height : src.size.p[0] * 2;
    if ((dstsize.width == 0) != (dstsize.height == 0) ||
        output_cols <= 0 || output_rows <= 0 ||
        std::abs(output_cols - src.size.p[1] * 2) >
            (output_cols & 1) ||
        std::abs(output_rows - src.size.p[0] * 2) >
            (output_rows & 1))
    {
        CV_Error(Error::StsBadSize, "pyrUp invalid destination size");
    }
    const Mat source = src.data == dst.data ? src.clone() : src;
    dst.create(
        {output_rows, output_cols}, source.type());
    if (source.depth() == CV_8U)
    {
        pyramid_detail::upsample<uchar>(
            source, dst, border_type);
    }
    else
    {
        pyramid_detail::upsample<float>(
            source, dst, border_type);
    }
}

inline void buildPyramid(const Mat& src,
                         std::vector<Mat>& dst,
                         int maxlevel,
                         int borderType = BORDER_DEFAULT)
{
    pyramid_detail::validate_source(src, "buildPyramid");
    if (maxlevel < 0)
    {
        CV_Error(
            Error::StsOutOfRange,
            "buildPyramid maxlevel must be non-negative");
    }
    dst.clear();
    dst.reserve(static_cast<size_t>(maxlevel + 1));
    dst.push_back(src.clone());
    for (int level = 1; level <= maxlevel; ++level)
    {
        Mat next;
        pyrDown(dst.back(), next, Size(), borderType);
        dst.push_back(next);
    }
}

}  // namespace cvh

#endif  // CVH_IMGPROC_PYRAMID_H
