#ifndef CVH_IMGPROC_FILTER2D_H
#define CVH_IMGPROC_FILTER2D_H

#include "detail/common.h"

#include <cstddef>
#include <vector>

namespace cvh {
namespace detail {

using Filter2DFn = void (*)(const Mat&, Mat&, int, const Mat&, Point, double, int);

inline double filter2d_kernel_value(const Mat& kernel, int y, int x)
{
    return static_cast<double>(kernel.at<float>(y, x));
}

inline double filter2d_sample_value(const Mat& src, int y, int x, int c)
{
    const size_t src_step = src.step(0);
    const uchar* src_row = src.data + static_cast<size_t>(y) * src_step;
    if (src.depth() == CV_8U)
    {
        return static_cast<double>(src_row[static_cast<size_t>(x) * src.channels() + c]);
    }
    const float* src_row_f32 = reinterpret_cast<const float*>(src_row);
    return static_cast<double>(src_row_f32[static_cast<size_t>(x) * src.channels() + c]);
}

inline void filter2D_fallback(const Mat& src,
                              Mat& dst,
                              int ddepth,
                              const Mat& kernel,
                              Point anchor,
                              double delta,
                              int borderType)
{
    CV_Assert(!src.empty() && "filter2D: source image can not be empty");
    CV_Assert(src.dims == 2 && "filter2D: only 2D Mat is supported");

    const int src_depth = src.depth();
    if (src_depth != CV_8U && src_depth != CV_32F)
    {
        CV_Error_(Error::StsBadArg, ("filter2D: unsupported src depth=%d", src_depth));
    }

    CV_Assert(!kernel.empty() && "filter2D: kernel can not be empty");
    CV_Assert(kernel.dims == 2 && "filter2D: kernel must be 2D");
    CV_Assert(kernel.channels() == 1 && "filter2D: kernel must be single-channel");
    if (kernel.depth() != CV_32F)
    {
        CV_Error_(Error::StsBadArg, ("filter2D: unsupported kernel depth=%d", kernel.depth()));
    }

    const int krows = kernel.size[0];
    const int kcols = kernel.size[1];
    if (krows <= 0 || kcols <= 0)
    {
        CV_Error_(Error::StsBadArg, ("filter2D: invalid kernel size=(%d,%d)", kcols, krows));
    }

    const int ax = anchor.x >= 0 ? anchor.x : (kcols / 2);
    const int ay = anchor.y >= 0 ? anchor.y : (krows / 2);
    if (ax < 0 || ax >= kcols || ay < 0 || ay >= krows)
    {
        CV_Error_(Error::StsBadArg,
                  ("filter2D: invalid anchor=(%d,%d) for kernel=(%d,%d)", anchor.x, anchor.y, kcols, krows));
    }

    int out_depth = ddepth;
    if (out_depth == -1)
    {
        out_depth = src_depth;
    }
    if (out_depth != CV_8U && out_depth != CV_32F)
    {
        CV_Error_(Error::StsBadArg, ("filter2D: unsupported ddepth=%d", ddepth));
    }

    const int border_type = normalize_border_type(borderType);
    if (!is_supported_filter_border(border_type))
    {
        CV_Error_(Error::StsBadArg, ("filter2D: unsupported borderType=%d", borderType));
    }

    Mat src_local;
    const Mat* src_ref = &src;
    if (src.data == dst.data)
    {
        src_local = src.clone();
        src_ref = &src_local;
    }

    const int rows = src_ref->size[0];
    const int cols = src_ref->size[1];
    const int channels = src_ref->channels();

    dst.create(std::vector<int>{rows, cols}, CV_MAKETYPE(out_depth, channels));
    const size_t dst_step = dst.step(0);

    for (int y = 0; y < rows; ++y)
    {
        uchar* dst_row = dst.data + static_cast<size_t>(y) * dst_step;
        float* dst_row_f32 = out_depth == CV_32F ? reinterpret_cast<float*>(dst_row) : nullptr;
        for (int x = 0; x < cols; ++x)
        {
            for (int c = 0; c < channels; ++c)
            {
                double acc = delta;
                for (int ky = 0; ky < krows; ++ky)
                {
                    const int sy = y + ky - ay;
                    const int src_y = border_interpolate(sy, rows, border_type);
                    if (src_y < 0)
                    {
                        continue;
                    }
                    for (int kx = 0; kx < kcols; ++kx)
                    {
                        const int sx = x + kx - ax;
                        const int src_x = border_interpolate(sx, cols, border_type);
                        if (src_x < 0)
                        {
                            continue;
                        }
                        const double coeff = filter2d_kernel_value(kernel, ky, kx);
                        const double sample = filter2d_sample_value(*src_ref, src_y, src_x, c);
                        acc += coeff * sample;
                    }
                }

                const size_t out_idx = static_cast<size_t>(x) * channels + c;
                if (dst_row_f32)
                {
                    dst_row_f32[out_idx] = static_cast<float>(acc);
                }
                else
                {
                    dst_row[out_idx] = saturate_cast<uchar>(acc);
                }
            }
        }
    }
}

inline Filter2DFn& filter2d_dispatch()
{
    static Filter2DFn fn = &filter2D_fallback;
    return fn;
}

inline void register_filter2d_backend(Filter2DFn fn)
{
    if (fn)
    {
        filter2d_dispatch() = fn;
    }
}

inline bool is_filter2d_backend_registered()
{
    return filter2d_dispatch() != &filter2D_fallback;
}

}  // namespace detail

inline void filter2D(const Mat& src,
                     Mat& dst,
                     int ddepth,
                     const Mat& kernel,
                     Point anchor = Point(-1, -1),
                     double delta = 0.0,
                     int borderType = BORDER_DEFAULT)
{
    detail::ensure_backends_registered_once();
    detail::filter2d_dispatch()(src, dst, ddepth, kernel, anchor, delta, borderType);
}

}  // namespace cvh

#endif  // CVH_IMGPROC_FILTER2D_H
