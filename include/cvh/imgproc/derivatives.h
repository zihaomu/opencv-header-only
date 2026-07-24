#ifndef CVH_IMGPROC_DERIVATIVES_H
#define CVH_IMGPROC_DERIVATIVES_H

#include "kernels.h"
#include "sobel.h"

#include <vector>

namespace cvh
{
namespace derivative_detail
{

inline void write_derivative(Mat& dst,
                             int y,
                             int x,
                             int channel,
                             double value)
{
    const size_t index =
        static_cast<size_t>(x) * dst.channels() +
        static_cast<size_t>(channel);
    uchar* row = dst.data + static_cast<size_t>(y) * dst.step(0);
    if (dst.depth() == CV_16S)
    {
        reinterpret_cast<short*>(row)[index] = saturate_cast<short>(value);
    }
    else
    {
        reinterpret_cast<float*>(row)[index] = static_cast<float>(value);
    }
}

inline void convolve(const Mat& src,
                     Mat& dst,
                     int ddepth,
                     const std::vector<double>& kernel,
                     int kernel_width,
                     int kernel_height,
                     double scale,
                     double delta,
                     int borderType,
                     const char* fn_name)
{
    if (src.empty() || src.dims != 2 ||
        (src.depth() != CV_8U && src.depth() != CV_16S &&
         src.depth() != CV_32F))
    {
        CV_Error_(Error::StsBadArg, ("%s unsupported src", fn_name));
    }
    const int output_depth = ddepth < 0 ? CV_32F : CV_MAT_DEPTH(ddepth);
    if (output_depth != CV_16S && output_depth != CV_32F)
    {
        CV_Error_(Error::StsBadArg, ("%s unsupported ddepth", fn_name));
    }
    const bool isolated = (borderType & BORDER_ISOLATED) != 0;
    const int border_type = detail::normalize_border_type(borderType);
    if (!detail::is_supported_filter_border(border_type))
    {
        CV_Error_(Error::StsBadArg, ("%s unsupported border", fn_name));
    }
    const Mat source = src.data == dst.data ? src.clone() : src;
    const detail::SobelSamplingWindow window =
        detail::resolve_sobel_sampling_window(source, isolated);
    dst.create(
        source.shape(), CV_MAKETYPE(output_depth, source.channels()));
    const int radius_x = kernel_width / 2;
    const int radius_y = kernel_height / 2;
    for (int y = 0; y < source.size.p[0]; ++y)
    {
        for (int x = 0; x < source.size.p[1]; ++x)
        {
            for (int ch = 0; ch < source.channels(); ++ch)
            {
                double accumulator = 0.0;
                for (int ky = 0; ky < kernel_height; ++ky)
                {
                    const int source_y = detail::border_interpolate(
                        y + window.row_offset + ky - radius_y,
                        window.rows,
                        border_type);
                    for (int kx = 0; kx < kernel_width; ++kx)
                    {
                        const int source_x = detail::border_interpolate(
                            x + window.col_offset + kx - radius_x,
                            window.cols,
                            border_type);
                        const double sample = detail::sobel_sample_window_as_f64(
                            window.base_data,
                            source.step(0),
                            source.depth(),
                            source.channels(),
                            source_y,
                            source_x,
                            ch);
                        accumulator +=
                            sample *
                            kernel[
                                static_cast<size_t>(ky * kernel_width + kx)];
                    }
                }
                write_derivative(
                    dst,
                    y,
                    x,
                    ch,
                    accumulator * scale + delta);
            }
        }
    }
}

}  // namespace derivative_detail

inline void Scharr(const Mat& src,
                   Mat& dst,
                   int ddepth,
                   int dx,
                   int dy,
                   double scale = 1.0,
                   double delta = 0.0,
                   int borderType = BORDER_DEFAULT)
{
    if (!((dx == 1 && dy == 0) || (dx == 0 && dy == 1)))
    {
        CV_Error(Error::StsBadArg, "Scharr expects first x or y derivative");
    }
    const std::vector<double> derivative = {-1.0, 0.0, 1.0};
    const std::vector<double> smoothing = {3.0, 10.0, 3.0};
    const std::vector<double>& kernel_x = dx == 1 ? derivative : smoothing;
    const std::vector<double>& kernel_y = dy == 1 ? derivative : smoothing;
    std::vector<double> kernel(9);
    for (int y = 0; y < 3; ++y)
    {
        for (int x = 0; x < 3; ++x)
        {
            kernel[static_cast<size_t>(y * 3 + x)] =
                kernel_y[static_cast<size_t>(y)] *
                kernel_x[static_cast<size_t>(x)];
        }
    }
    derivative_detail::convolve(
        src, dst, ddepth, kernel, 3, 3, scale, delta, borderType, "Scharr");
}

inline void Laplacian(const Mat& src,
                      Mat& dst,
                      int ddepth,
                      int ksize = 1,
                      double scale = 1.0,
                      double delta = 0.0,
                      int borderType = BORDER_DEFAULT)
{
    if (ksize != 1 && ksize != 3 && ksize != 5)
    {
        CV_Error(Error::StsBadArg, "Laplacian supports ksize 1, 3 or 5");
    }
    std::vector<double> kernel;
    int width = ksize == 1 ? 3 : ksize;
    if (ksize == 1)
    {
        kernel = {0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0};
    }
    else
    {
        const std::vector<double> second =
            kernel_detail::sobel_kernel(2, ksize);
        const std::vector<double> smoothing =
            kernel_detail::sobel_kernel(0, ksize);
        kernel.resize(static_cast<size_t>(ksize * ksize));
        for (int y = 0; y < ksize; ++y)
        {
            for (int x = 0; x < ksize; ++x)
            {
                kernel[static_cast<size_t>(y * ksize + x)] =
                    second[static_cast<size_t>(x)] *
                        smoothing[static_cast<size_t>(y)] +
                    smoothing[static_cast<size_t>(x)] *
                        second[static_cast<size_t>(y)];
            }
        }
    }
    derivative_detail::convolve(
        src,
        dst,
        ddepth,
        kernel,
        width,
        width,
        scale,
        delta,
        borderType,
        "Laplacian");
}

inline void spatialGradient(const Mat& src,
                            Mat& dx,
                            Mat& dy,
                            int ksize = 3,
                            int borderType = BORDER_DEFAULT)
{
    if (ksize != 3 ||
        (detail::normalize_border_type(borderType) != BORDER_DEFAULT &&
         detail::normalize_border_type(borderType) != BORDER_REPLICATE))
    {
        CV_Error(
            Error::StsBadArg,
            "spatialGradient supports ksize=3 and reflect101/replicate borders");
    }
    Sobel(src, dx, CV_16S, 1, 0, 3, 1.0, 0.0, borderType);
    Sobel(src, dy, CV_16S, 0, 1, 3, 1.0, 0.0, borderType);
}

}  // namespace cvh

#endif  // CVH_IMGPROC_DERIVATIVES_H
