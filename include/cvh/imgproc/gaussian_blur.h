#ifndef CVH_IMGPROC_GAUSSIAN_BLUR_H
#define CVH_IMGPROC_GAUSSIAN_BLUR_H

#include "detail/common.h"

#include <type_traits>
#include <vector>

namespace cvh {
namespace detail {

using GaussianBlurFn = void (*)(const Mat&, Mat&, Size, double, double, int);

template <typename T>
inline void gaussian_blur_fallback_impl_typed(const Mat& src,
                                              Mat& dst,
                                              int kx,
                                              int ky,
                                              double sigmaX,
                                              double sigmaY,
                                              int border_type)
{
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
    const size_t src_step = src_ref->step(0);

    dst.create(std::vector<int>{rows, cols}, src_ref->type());
    const size_t dst_step = dst.step(0);

    const std::vector<float> kernel_x = build_gaussian_kernel_1d(kx, sigmaX);
    const std::vector<float> kernel_y = build_gaussian_kernel_1d(ky, sigmaY);
    const int rx = kx / 2;
    const int ry = ky / 2;

    std::vector<float> tmp(static_cast<size_t>(rows) * cols * channels, 0.0f);

    for (int y = 0; y < rows; ++y)
    {
        const T* src_row = reinterpret_cast<const T*>(src_ref->data + static_cast<size_t>(y) * src_step);
        for (int x = 0; x < cols; ++x)
        {
            const size_t tmp_base = (static_cast<size_t>(y) * cols + x) * channels;
            for (int c = 0; c < channels; ++c)
            {
                double acc = 0.0;
                for (int k = 0; k < kx; ++k)
                {
                    const int sx = x + k - rx;
                    const int src_x = border_interpolate(sx, cols, border_type);
                    if (src_x < 0)
                    {
                        continue;
                    }
                    acc += static_cast<double>(kernel_x[static_cast<size_t>(k)]) *
                           static_cast<double>(src_row[static_cast<size_t>(src_x) * channels + c]);
                }
                tmp[tmp_base + c] = static_cast<float>(acc);
            }
        }
    }

    for (int y = 0; y < rows; ++y)
    {
        T* dst_row = reinterpret_cast<T*>(dst.data + static_cast<size_t>(y) * dst_step);
        for (int x = 0; x < cols; ++x)
        {
            T* out_px = dst_row + static_cast<size_t>(x) * channels;
            for (int c = 0; c < channels; ++c)
            {
                double acc = 0.0;
                for (int k = 0; k < ky; ++k)
                {
                    const int sy = y + k - ry;
                    const int src_y = border_interpolate(sy, rows, border_type);
                    if (src_y < 0)
                    {
                        continue;
                    }
                    const size_t tmp_idx = (static_cast<size_t>(src_y) * cols + x) * channels + c;
                    acc += static_cast<double>(kernel_y[static_cast<size_t>(k)]) * static_cast<double>(tmp[tmp_idx]);
                }
                if constexpr (std::is_same<T, uchar>::value)
                {
                    out_px[c] = saturate_cast<uchar>(acc);
                }
                else
                {
                    out_px[c] = static_cast<float>(acc);
                }
            }
        }
    }
}

inline void gaussian_blur_fallback(const Mat& src,
                                   Mat& dst,
                                   Size ksize,
                                   double sigmaX,
                                   double sigmaY,
                                   int borderType)
{
    CV_Assert(!src.empty() && "GaussianBlur: source image can not be empty");
    CV_Assert(src.dims == 2 && "GaussianBlur: only 2D Mat is supported");

    const int src_depth = src.depth();
    if (src_depth != CV_8U && src_depth != CV_32F)
    {
        CV_Error_(Error::StsBadArg, ("GaussianBlur: unsupported src depth=%d", src_depth));
    }

    int kx = ksize.width;
    int ky = ksize.height;

    if (kx <= 0 && sigmaX > 0.0)
    {
        kx = auto_gaussian_ksize(sigmaX);
    }
    if (ky <= 0 && sigmaY > 0.0)
    {
        ky = auto_gaussian_ksize(sigmaY);
    }

    if (kx <= 0 && ky > 0)
    {
        kx = ky;
    }
    if (ky <= 0 && kx > 0)
    {
        ky = kx;
    }

    if (kx <= 0 || ky <= 0)
    {
        CV_Error_(Error::StsBadArg,
                  ("GaussianBlur: invalid ksize=(%d,%d), sigmaX=%.6f sigmaY=%.6f",
                   ksize.width,
                   ksize.height,
                   sigmaX,
                   sigmaY));
    }

    if ((kx & 1) == 0 || (ky & 1) == 0)
    {
        CV_Error_(Error::StsBadArg, ("GaussianBlur: ksize must be odd, got (%d,%d)", kx, ky));
    }

    if (sigmaX <= 0.0)
    {
        sigmaX = default_gaussian_sigma_for_ksize(kx);
    }
    if (sigmaY <= 0.0)
    {
        sigmaY = sigmaX;
    }

    if (sigmaX <= 0.0 || sigmaY <= 0.0)
    {
        CV_Error_(Error::StsBadArg,
                  ("GaussianBlur: invalid sigmaX/sigmaY after resolve (%.6f, %.6f)", sigmaX, sigmaY));
    }

    const int border_type = normalize_border_type(borderType);
    if (!is_supported_filter_border(border_type))
    {
        CV_Error_(Error::StsBadArg, ("GaussianBlur: unsupported borderType=%d", borderType));
    }

    if (src_depth == CV_8U)
    {
        gaussian_blur_fallback_impl_typed<uchar>(src, dst, kx, ky, sigmaX, sigmaY, border_type);
        return;
    }

    gaussian_blur_fallback_impl_typed<float>(src, dst, kx, ky, sigmaX, sigmaY, border_type);
}

inline GaussianBlurFn& gaussianblur_dispatch()
{
    static GaussianBlurFn fn = &gaussian_blur_fallback;
    return fn;
}

inline void register_gaussianblur_backend(GaussianBlurFn fn)
{
    if (fn)
    {
        gaussianblur_dispatch() = fn;
    }
}

inline bool is_gaussianblur_backend_registered()
{
    return gaussianblur_dispatch() != &gaussian_blur_fallback;
}

}  // namespace detail

inline void GaussianBlur(const Mat& src,
                         Mat& dst,
                         Size ksize,
                         double sigmaX,
                         double sigmaY = 0.0,
                         int borderType = BORDER_DEFAULT)
{
    detail::ensure_backends_registered_once();
    detail::gaussianblur_dispatch()(src, dst, ksize, sigmaX, sigmaY, borderType);
}

}  // namespace cvh

#endif  // CVH_IMGPROC_GAUSSIAN_BLUR_H
