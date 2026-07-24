#ifndef CVH_IMGPROC_STACK_BLUR_H
#define CVH_IMGPROC_STACK_BLUR_H

#include "detail/common.h"

#include <type_traits>
#include <vector>

namespace cvh
{
namespace stack_blur_detail
{

template<typename T>
inline T cast_value(double value)
{
    if constexpr (std::is_same<T, uchar>::value)
    {
        return saturate_cast<uchar>(value);
    }
    return static_cast<T>(value);
}

template<typename T>
inline void run(const Mat& src, Mat& dst, Size ksize)
{
    const int rows = src.size.p[0];
    const int cols = src.size.p[1];
    const int channels = src.channels();
    const int radius_x = ksize.width / 2;
    const int radius_y = ksize.height / 2;
    const double divisor_x =
        static_cast<double>((radius_x + 1) * (radius_x + 1));
    const double divisor_y =
        static_cast<double>((radius_y + 1) * (radius_y + 1));
    std::vector<double> temporary(
        static_cast<size_t>(rows) * cols * channels, 0.0);

    for (int y = 0; y < rows; ++y)
    {
        const T* input = reinterpret_cast<const T*>(
            src.data + static_cast<size_t>(y) * src.step(0));
        for (int x = 0; x < cols; ++x)
        {
            for (int ch = 0; ch < channels; ++ch)
            {
                double accumulator = 0.0;
                for (int k = -radius_x; k <= radius_x; ++k)
                {
                    const int source_x = std::clamp(x + k, 0, cols - 1);
                    const int weight = radius_x + 1 - std::abs(k);
                    accumulator +=
                        weight *
                        static_cast<double>(
                            input[static_cast<size_t>(source_x) * channels +
                                  static_cast<size_t>(ch)]);
                }
                temporary[
                    (static_cast<size_t>(y) * cols +
                     static_cast<size_t>(x)) *
                        channels +
                    static_cast<size_t>(ch)] =
                    accumulator / divisor_x;
            }
        }
    }

    for (int y = 0; y < rows; ++y)
    {
        T* output = reinterpret_cast<T*>(
            dst.data + static_cast<size_t>(y) * dst.step(0));
        for (int x = 0; x < cols; ++x)
        {
            for (int ch = 0; ch < channels; ++ch)
            {
                double accumulator = 0.0;
                for (int k = -radius_y; k <= radius_y; ++k)
                {
                    const int source_y = std::clamp(y + k, 0, rows - 1);
                    const int weight = radius_y + 1 - std::abs(k);
                    accumulator +=
                        weight *
                        temporary[
                            (static_cast<size_t>(source_y) * cols +
                             static_cast<size_t>(x)) *
                                channels +
                            static_cast<size_t>(ch)];
                }
                output[static_cast<size_t>(x) * channels +
                       static_cast<size_t>(ch)] =
                    cast_value<T>(accumulator / divisor_y);
            }
        }
    }
}

}  // namespace stack_blur_detail

inline void stackBlur(const Mat& src, Mat& dst, Size ksize)
{
    if (src.empty() || src.dims != 2 ||
        (src.depth() != CV_8U && src.depth() != CV_32F) ||
        (src.channels() != 1 && src.channels() != 3 &&
         src.channels() != 4))
    {
        CV_Error(Error::StsBadArg, "stackBlur unsupported source");
    }
    if (ksize.width <= 0 || ksize.height <= 0 ||
        (ksize.width & 1) == 0 || (ksize.height & 1) == 0)
    {
        CV_Error(Error::StsBadSize, "stackBlur ksize must be positive and odd");
    }
    const Mat source = src.data == dst.data ? src.clone() : src;
    dst.create(source.shape(), source.type());
    if (source.depth() == CV_8U)
    {
        stack_blur_detail::run<uchar>(source, dst, ksize);
    }
    else
    {
        stack_blur_detail::run<float>(source, dst, ksize);
    }
}

}  // namespace cvh

#endif  // CVH_IMGPROC_STACK_BLUR_H
