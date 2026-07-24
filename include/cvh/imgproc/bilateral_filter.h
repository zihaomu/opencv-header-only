#ifndef CVH_IMGPROC_BILATERAL_FILTER_H
#define CVH_IMGPROC_BILATERAL_FILTER_H

#include "detail/common.h"

#include <cmath>
#include <type_traits>
#include <vector>

namespace cvh
{
namespace bilateral_filter_detail
{

struct OffsetWeight
{
    int x;
    int y;
    double weight;
};

inline std::vector<OffsetWeight> spatial_weights(int radius,
                                                 double sigma_space)
{
    const double coefficient =
        -0.5 / (sigma_space * sigma_space);
    std::vector<OffsetWeight> weights;
    weights.reserve(static_cast<size_t>(radius * 2 + 1) *
                    static_cast<size_t>(radius * 2 + 1));
    for (int y = -radius; y <= radius; ++y)
    {
        for (int x = -radius; x <= radius; ++x)
        {
            const int distance_squared = x * x + y * y;
            if (distance_squared > radius * radius)
            {
                continue;
            }
            weights.push_back(
                {x,
                 y,
                 std::exp(static_cast<double>(distance_squared) *
                          coefficient)});
        }
    }
    return weights;
}

template<typename T>
inline void run(const Mat& src,
                Mat& dst,
                const std::vector<OffsetWeight>& spatial,
                double sigma_color,
                int border_type)
{
    const int rows = src.size.p[0];
    const int cols = src.size.p[1];
    const int channels = src.channels();
    const double color_coefficient =
        -0.5 / (sigma_color * sigma_color);

    for (int y = 0; y < rows; ++y)
    {
        const T* center_row = reinterpret_cast<const T*>(
            src.data + static_cast<size_t>(y) * src.step(0));
        T* output = reinterpret_cast<T*>(
            dst.data + static_cast<size_t>(y) * dst.step(0));
        for (int x = 0; x < cols; ++x)
        {
            const T* center =
                center_row + static_cast<size_t>(x) * channels;
            double weighted_sum[3] = {0.0, 0.0, 0.0};
            double weight_sum = 0.0;
            for (const OffsetWeight& offset : spatial)
            {
                const int source_y = detail::border_interpolate(
                    y + offset.y, rows, border_type);
                const int source_x = detail::border_interpolate(
                    x + offset.x, cols, border_type);
                if (source_y < 0 || source_x < 0)
                {
                    continue;
                }
                const T* sample_row = reinterpret_cast<const T*>(
                    src.data +
                    static_cast<size_t>(source_y) * src.step(0));
                const T* sample =
                    sample_row + static_cast<size_t>(source_x) * channels;
                double color_distance = 0.0;
                for (int ch = 0; ch < channels; ++ch)
                {
                    color_distance += std::fabs(
                        static_cast<double>(sample[ch]) -
                        static_cast<double>(center[ch]));
                }
                const double weight =
                    offset.weight *
                    std::exp(color_distance * color_distance *
                             color_coefficient);
                weight_sum += weight;
                for (int ch = 0; ch < channels; ++ch)
                {
                    weighted_sum[ch] +=
                        weight * static_cast<double>(sample[ch]);
                }
            }
            for (int ch = 0; ch < channels; ++ch)
            {
                const double value =
                    weight_sum > 0.0
                        ? weighted_sum[ch] / weight_sum
                        : static_cast<double>(center[ch]);
                if constexpr (std::is_same<T, uchar>::value)
                {
                    output[static_cast<size_t>(x) * channels + ch] =
                        saturate_cast<uchar>(value);
                }
                else
                {
                    output[static_cast<size_t>(x) * channels + ch] =
                        static_cast<float>(value);
                }
            }
        }
    }
}

}  // namespace bilateral_filter_detail

inline void bilateralFilter(const Mat& src,
                            Mat& dst,
                            int d,
                            double sigmaColor,
                            double sigmaSpace,
                            int borderType = BORDER_DEFAULT)
{
    if (src.empty() || src.dims != 2 ||
        (src.depth() != CV_8U && src.depth() != CV_32F) ||
        (src.channels() != 1 && src.channels() != 3))
    {
        CV_Error(Error::StsBadArg, "bilateralFilter unsupported source");
    }
    if (src.data == dst.data)
    {
        CV_Error(Error::StsBadArg, "bilateralFilter does not support in-place operation");
    }
    if (!std::isfinite(sigmaColor) || !std::isfinite(sigmaSpace))
    {
        CV_Error(Error::StsBadArg, "bilateralFilter sigma values must be finite");
    }
    const int border_type = detail::normalize_border_type(borderType);
    if (!detail::is_supported_filter_border(border_type))
    {
        CV_Error(Error::StsBadArg, "bilateralFilter unsupported border");
    }

    dst.create(src.shape(), src.type());
    if (sigmaColor <= 1e-6 || sigmaSpace <= 1e-6)
    {
        src.copyTo(dst);
        return;
    }
    int radius =
        d <= 0 ? static_cast<int>(std::lround(sigmaSpace * 1.5)) : d / 2;
    radius = std::max(radius, 1);
    const std::vector<bilateral_filter_detail::OffsetWeight> spatial =
        bilateral_filter_detail::spatial_weights(radius, sigmaSpace);
    if (src.depth() == CV_8U)
    {
        bilateral_filter_detail::run<uchar>(
            src, dst, spatial, sigmaColor, border_type);
    }
    else
    {
        bilateral_filter_detail::run<float>(
            src, dst, spatial, sigmaColor, border_type);
    }
}

}  // namespace cvh

#endif  // CVH_IMGPROC_BILATERAL_FILTER_H
