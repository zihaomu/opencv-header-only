#ifndef CVH_IMGPROC_SEP_FILTER2D_H
#define CVH_IMGPROC_SEP_FILTER2D_H

#include "detail/common.h"

#include <cstddef>
#include <vector>

namespace cvh {
namespace detail {

using SepFilter2DFn = void (*)(const Mat&, Mat&, int, const Mat&, const Mat&, Point, double, int);

inline void sepfilter2d_collect_kernel(const Mat& kernel, std::vector<float>& coeffs, const char* kernel_name)
{
    CV_Assert(!kernel.empty() && "sepFilter2D: kernel can not be empty");
    CV_Assert(kernel.dims == 2 && "sepFilter2D: kernel must be 2D");
    CV_Assert(kernel.channels() == 1 && "sepFilter2D: kernel must be single-channel");
    if (kernel.depth() != CV_32F)
    {
        CV_Error_(Error::StsBadArg, ("sepFilter2D: unsupported %s depth=%d", kernel_name, kernel.depth()));
    }

    const int rows = kernel.size[0];
    const int cols = kernel.size[1];
    if (rows <= 0 || cols <= 0)
    {
        CV_Error_(Error::StsBadArg, ("sepFilter2D: invalid %s size=(%d,%d)", kernel_name, cols, rows));
    }

    if (rows != 1 && cols != 1)
    {
        CV_Error_(Error::StsBadArg,
                  ("sepFilter2D: %s must be 1xN or Nx1, got (%d,%d)", kernel_name, cols, rows));
    }

    const int len = rows == 1 ? cols : rows;
    coeffs.resize(static_cast<size_t>(len));
    if (rows == 1)
    {
        for (int i = 0; i < len; ++i)
        {
            coeffs[static_cast<size_t>(i)] = kernel.at<float>(0, i);
        }
        return;
    }

    for (int i = 0; i < len; ++i)
    {
        coeffs[static_cast<size_t>(i)] = kernel.at<float>(i, 0);
    }
}

inline double sepfilter2d_sample_value(const Mat& src, int y, int x, int c)
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

inline void sepFilter2D_fallback(const Mat& src,
                                 Mat& dst,
                                 int ddepth,
                                 const Mat& kernelX,
                                 const Mat& kernelY,
                                 Point anchor,
                                 double delta,
                                 int borderType)
{
    CV_Assert(!src.empty() && "sepFilter2D: source image can not be empty");
    CV_Assert(src.dims == 2 && "sepFilter2D: only 2D Mat is supported");

    const int src_depth = src.depth();
    if (src_depth != CV_8U && src_depth != CV_32F)
    {
        CV_Error_(Error::StsBadArg, ("sepFilter2D: unsupported src depth=%d", src_depth));
    }

    std::vector<float> kx;
    std::vector<float> ky;
    sepfilter2d_collect_kernel(kernelX, kx, "kernelX");
    sepfilter2d_collect_kernel(kernelY, ky, "kernelY");

    const int kx_len = static_cast<int>(kx.size());
    const int ky_len = static_cast<int>(ky.size());
    const int ax = anchor.x >= 0 ? anchor.x : (kx_len / 2);
    const int ay = anchor.y >= 0 ? anchor.y : (ky_len / 2);
    if (ax < 0 || ax >= kx_len || ay < 0 || ay >= ky_len)
    {
        CV_Error_(Error::StsBadArg,
                  ("sepFilter2D: invalid anchor=(%d,%d) for kernel lengths (%d,%d)",
                   anchor.x,
                   anchor.y,
                   kx_len,
                   ky_len));
    }

    int out_depth = ddepth;
    if (out_depth == -1)
    {
        out_depth = src_depth;
    }
    if (out_depth != CV_8U && out_depth != CV_32F)
    {
        CV_Error_(Error::StsBadArg, ("sepFilter2D: unsupported ddepth=%d", ddepth));
    }

    const int border_type = normalize_border_type(borderType);
    if (!is_supported_filter_border(border_type))
    {
        CV_Error_(Error::StsBadArg, ("sepFilter2D: unsupported borderType=%d", borderType));
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

    std::vector<float> tmp(static_cast<size_t>(rows) * cols * channels, 0.0f);
    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            const size_t tmp_base = (static_cast<size_t>(y) * cols + x) * channels;
            for (int c = 0; c < channels; ++c)
            {
                double acc = 0.0;
                for (int k = 0; k < kx_len; ++k)
                {
                    const int sx = x + k - ax;
                    const int src_x = border_interpolate(sx, cols, border_type);
                    if (src_x < 0)
                    {
                        continue;
                    }
                    const double sample = sepfilter2d_sample_value(*src_ref, y, src_x, c);
                    acc += static_cast<double>(kx[static_cast<size_t>(k)]) * sample;
                }
                tmp[tmp_base + c] = static_cast<float>(acc);
            }
        }
    }

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
                for (int k = 0; k < ky_len; ++k)
                {
                    const int sy = y + k - ay;
                    const int src_y = border_interpolate(sy, rows, border_type);
                    if (src_y < 0)
                    {
                        continue;
                    }
                    const size_t tmp_idx = (static_cast<size_t>(src_y) * cols + x) * channels + c;
                    acc += static_cast<double>(ky[static_cast<size_t>(k)]) * static_cast<double>(tmp[tmp_idx]);
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

inline SepFilter2DFn& sepfilter2d_dispatch()
{
    static SepFilter2DFn fn = &sepFilter2D_fallback;
    return fn;
}

inline void register_sep_filter2d_backend(SepFilter2DFn fn)
{
    if (fn)
    {
        sepfilter2d_dispatch() = fn;
    }
}

inline bool is_sep_filter2d_backend_registered()
{
    return sepfilter2d_dispatch() != &sepFilter2D_fallback;
}

}  // namespace detail

inline void sepFilter2D(const Mat& src,
                        Mat& dst,
                        int ddepth,
                        const Mat& kernelX,
                        const Mat& kernelY,
                        Point anchor = Point(-1, -1),
                        double delta = 0.0,
                        int borderType = BORDER_DEFAULT)
{
    detail::ensure_backends_registered_once();
    detail::sepfilter2d_dispatch()(src, dst, ddepth, kernelX, kernelY, anchor, delta, borderType);
}

}  // namespace cvh

#endif  // CVH_IMGPROC_SEP_FILTER2D_H
