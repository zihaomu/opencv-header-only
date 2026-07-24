#ifndef CVH_IMGPROC_INTEGRAL_H
#define CVH_IMGPROC_INTEGRAL_H

#include "detail/common.h"

#include <cstdint>
#include <vector>

namespace cvh
{

inline void integral(const Mat& src, Mat& sum, int sdepth = -1)
{
    if (src.empty() || src.dims != 2 || src.depth() != CV_8U)
    {
        CV_Error(Error::StsBadArg, "integral currently expects non-empty 2D CV_8U src");
    }
    const int output_depth = sdepth < 0 ? CV_32S : CV_MAT_DEPTH(sdepth);
    if (output_depth != CV_32S && output_depth != CV_64F)
    {
        CV_Error(Error::StsBadArg, "integral output depth must be CV_32S or CV_64F");
    }
    const Mat source = src.data == sum.data ? src.clone() : src;
    const int rows = source.size.p[0];
    const int cols = source.size.p[1];
    const int channels = source.channels();
    sum.create(
        {rows + 1, cols + 1}, CV_MAKETYPE(output_depth, channels));
    sum.setTo(Scalar::all(0.0));
    std::vector<std::int64_t> row_sums(static_cast<size_t>(channels), 0);
    for (int y = 0; y < rows; ++y)
    {
        std::fill(row_sums.begin(), row_sums.end(), 0);
        const uchar* source_row =
            source.data + static_cast<size_t>(y) * source.step(0);
        for (int x = 0; x < cols; ++x)
        {
            for (int ch = 0; ch < channels; ++ch)
            {
                row_sums[static_cast<size_t>(ch)] +=
                    source_row[
                        static_cast<size_t>(x) * channels +
                        static_cast<size_t>(ch)];
                if (output_depth == CV_32S)
                {
                    const int* previous_row =
                        reinterpret_cast<const int*>(
                            sum.data + static_cast<size_t>(y) * sum.step(0));
                    int* output_row = reinterpret_cast<int*>(
                        sum.data + static_cast<size_t>(y + 1) * sum.step(0));
                    const std::int64_t value =
                        row_sums[static_cast<size_t>(ch)] +
                        previous_row[
                            static_cast<size_t>(x + 1) * channels +
                            static_cast<size_t>(ch)];
                    output_row[
                        static_cast<size_t>(x + 1) * channels +
                        static_cast<size_t>(ch)] = static_cast<int>(value);
                }
                else
                {
                    const double* previous_row =
                        reinterpret_cast<const double*>(
                            sum.data + static_cast<size_t>(y) * sum.step(0));
                    double* output_row = reinterpret_cast<double*>(
                        sum.data + static_cast<size_t>(y + 1) * sum.step(0));
                    output_row[
                        static_cast<size_t>(x + 1) * channels +
                        static_cast<size_t>(ch)] =
                        static_cast<double>(
                            row_sums[static_cast<size_t>(ch)]) +
                        previous_row[
                            static_cast<size_t>(x + 1) * channels +
                            static_cast<size_t>(ch)];
                }
            }
        }
    }
}

}  // namespace cvh

#endif  // CVH_IMGPROC_INTEGRAL_H
