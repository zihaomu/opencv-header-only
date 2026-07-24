#ifndef CVH_IMGPROC_MEDIAN_BLUR_H
#define CVH_IMGPROC_MEDIAN_BLUR_H

#include "detail/common.h"

#include <algorithm>
#include <vector>

namespace cvh
{
namespace median_blur_detail
{

template<typename T>
inline void run(const Mat& src, Mat& dst, int ksize)
{
    const int rows = src.size.p[0];
    const int cols = src.size.p[1];
    const int channels = src.channels();
    const int radius = ksize / 2;
    std::vector<T> window(static_cast<size_t>(ksize) * ksize);

    for (int y = 0; y < rows; ++y)
    {
        T* output = reinterpret_cast<T*>(
            dst.data + static_cast<size_t>(y) * dst.step(0));
        for (int x = 0; x < cols; ++x)
        {
            for (int ch = 0; ch < channels; ++ch)
            {
                size_t index = 0;
                for (int ky = -radius; ky <= radius; ++ky)
                {
                    const int source_y = std::clamp(y + ky, 0, rows - 1);
                    const T* input = reinterpret_cast<const T*>(
                        src.data +
                        static_cast<size_t>(source_y) * src.step(0));
                    for (int kx = -radius; kx <= radius; ++kx)
                    {
                        const int source_x =
                            std::clamp(x + kx, 0, cols - 1);
                        window[index++] =
                            input[static_cast<size_t>(source_x) * channels +
                                  static_cast<size_t>(ch)];
                    }
                }
                auto middle =
                    window.begin() + static_cast<std::ptrdiff_t>(window.size() / 2);
                std::nth_element(window.begin(), middle, window.end());
                output[static_cast<size_t>(x) * channels +
                       static_cast<size_t>(ch)] = *middle;
            }
        }
    }
}

}  // namespace median_blur_detail

inline void medianBlur(const Mat& src, Mat& dst, int ksize)
{
    if (src.empty() || src.dims != 2 ||
        (src.depth() != CV_8U && src.depth() != CV_32F) ||
        (src.channels() != 1 && src.channels() != 3 &&
         src.channels() != 4))
    {
        CV_Error(Error::StsBadArg, "medianBlur unsupported source");
    }
    if (ksize <= 1 || (ksize & 1) == 0)
    {
        CV_Error(Error::StsBadSize, "medianBlur ksize must be odd and greater than 1");
    }
    if (src.depth() == CV_32F && ksize != 3 && ksize != 5)
    {
        CV_Error(Error::StsBadSize, "medianBlur CV_32F supports ksize 3 or 5");
    }

    const Mat source = src.data == dst.data ? src.clone() : src;
    dst.create(source.shape(), source.type());
    if (source.depth() == CV_8U)
    {
        median_blur_detail::run<uchar>(source, dst, ksize);
    }
    else
    {
        median_blur_detail::run<float>(source, dst, ksize);
    }
}

}  // namespace cvh

#endif  // CVH_IMGPROC_MEDIAN_BLUR_H
