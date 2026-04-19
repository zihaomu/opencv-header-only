#ifndef CVH_IMGPROC_WARP_AFFINE_H
#define CVH_IMGPROC_WARP_AFFINE_H

#include "detail/common.h"

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace cvh {
namespace detail {

using WarpAffineFn = void (*)(const Mat&, Mat&, const Mat&, Size, int, int, const Scalar&);

template <typename T>
inline T warp_affine_cast(double v)
{
    if constexpr (std::is_same<T, uchar>::value)
    {
        return saturate_cast<uchar>(v);
    }
    else
    {
        return static_cast<float>(v);
    }
}

template <typename T>
inline T warp_affine_scalar_value(const Scalar& value, int channel)
{
    const int idx = channel < 4 ? channel : 3;
    if constexpr (std::is_same<T, uchar>::value)
    {
        return saturate_cast<uchar>(value.val[idx]);
    }
    else
    {
        return static_cast<float>(value.val[idx]);
    }
}

inline void warp_affine_read_matrix(const Mat& M,
                                    double& m00, double& m01, double& m02,
                                    double& m10, double& m11, double& m12)
{
    CV_Assert(M.dims == 2 && "warpAffine: transform matrix must be 2D");
    CV_Assert(M.channels() == 1 && "warpAffine: transform matrix must be single-channel");
    CV_Assert(M.size[0] == 2 && M.size[1] == 3 && "warpAffine: transform matrix must be 2x3");

    if (M.depth() == CV_32F)
    {
        m00 = static_cast<double>(M.at<float>(0, 0));
        m01 = static_cast<double>(M.at<float>(0, 1));
        m02 = static_cast<double>(M.at<float>(0, 2));
        m10 = static_cast<double>(M.at<float>(1, 0));
        m11 = static_cast<double>(M.at<float>(1, 1));
        m12 = static_cast<double>(M.at<float>(1, 2));
        return;
    }

    if (M.depth() == CV_64F)
    {
        m00 = M.at<double>(0, 0);
        m01 = M.at<double>(0, 1);
        m02 = M.at<double>(0, 2);
        m10 = M.at<double>(1, 0);
        m11 = M.at<double>(1, 1);
        m12 = M.at<double>(1, 2);
        return;
    }

    CV_Error_(Error::StsBadArg, ("warpAffine: unsupported transform depth=%d", M.depth()));
}

inline void warp_affine_resolve_inverse_map(double& m00, double& m01, double& m02,
                                            double& m10, double& m11, double& m12,
                                            bool inverse_map)
{
    if (inverse_map)
    {
        return;
    }

    const double det = m00 * m11 - m01 * m10;
    if (std::abs(det) < 1e-12)
    {
        CV_Error(Error::StsBadArg, "warpAffine: singular transform matrix");
    }

    const double inv_det = 1.0 / det;
    const double a00 = m11 * inv_det;
    const double a01 = -m01 * inv_det;
    const double a10 = -m10 * inv_det;
    const double a11 = m00 * inv_det;
    const double a02 = -(a00 * m02 + a01 * m12);
    const double a12 = -(a10 * m02 + a11 * m12);

    m00 = a00;
    m01 = a01;
    m02 = a02;
    m10 = a10;
    m11 = a11;
    m12 = a12;
}

template <typename T>
inline void warpAffine_fallback_impl_typed(const Mat& src,
                                           Mat& dst,
                                           const Mat& M,
                                           Size dsize,
                                           int flags,
                                           int borderMode,
                                           const Scalar& borderValue)
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

    const int dst_rows = dsize.height > 0 ? dsize.height : rows;
    const int dst_cols = dsize.width > 0 ? dsize.width : cols;
    CV_Assert(dst_rows > 0 && dst_cols > 0 && "warpAffine: invalid output size");

    dst.create(std::vector<int>{dst_rows, dst_cols}, src_ref->type());
    const size_t dst_step = dst.step(0);

    double m00 = 0.0;
    double m01 = 0.0;
    double m02 = 0.0;
    double m10 = 0.0;
    double m11 = 0.0;
    double m12 = 0.0;
    warp_affine_read_matrix(M, m00, m01, m02, m10, m11, m12);

    const bool inverse_map = (flags & WARP_INVERSE_MAP) != 0;
    warp_affine_resolve_inverse_map(m00, m01, m02, m10, m11, m12, inverse_map);

    const int interpolation = flags & 7;
    const int border_type = normalize_border_type(borderMode);

    auto sample = [&](int sy, int sx, int c) -> T {
        if (static_cast<unsigned>(sy) < static_cast<unsigned>(rows) &&
            static_cast<unsigned>(sx) < static_cast<unsigned>(cols))
        {
            const T* row_ptr = reinterpret_cast<const T*>(src_ref->data + static_cast<size_t>(sy) * src_step);
            return row_ptr[static_cast<size_t>(sx) * channels + c];
        }

        if (border_type == BORDER_CONSTANT)
        {
            return warp_affine_scalar_value<T>(borderValue, c);
        }

        const int by = border_interpolate(sy, rows, border_type);
        const int bx = border_interpolate(sx, cols, border_type);
        if (by < 0 || bx < 0)
        {
            return warp_affine_scalar_value<T>(borderValue, c);
        }

        const T* row_ptr = reinterpret_cast<const T*>(src_ref->data + static_cast<size_t>(by) * src_step);
        return row_ptr[static_cast<size_t>(bx) * channels + c];
    };

    for (int y = 0; y < dst_rows; ++y)
    {
        T* dst_row = reinterpret_cast<T*>(dst.data + static_cast<size_t>(y) * dst_step);
        for (int x = 0; x < dst_cols; ++x)
        {
            const double sx = m00 * static_cast<double>(x) + m01 * static_cast<double>(y) + m02;
            const double sy = m10 * static_cast<double>(x) + m11 * static_cast<double>(y) + m12;
            T* dst_px = dst_row + static_cast<size_t>(x) * channels;

            if (interpolation == INTER_NEAREST)
            {
                const int isx = static_cast<int>(std::floor(sx + 0.5));
                const int isy = static_cast<int>(std::floor(sy + 0.5));
                for (int c = 0; c < channels; ++c)
                {
                    dst_px[c] = sample(isy, isx, c);
                }
                continue;
            }

            const int x0 = static_cast<int>(std::floor(sx));
            const int y0 = static_cast<int>(std::floor(sy));
            const int x1 = x0 + 1;
            const int y1 = y0 + 1;
            const double fx = sx - static_cast<double>(x0);
            const double fy = sy - static_cast<double>(y0);

            for (int c = 0; c < channels; ++c)
            {
                const double p00 = static_cast<double>(sample(y0, x0, c));
                const double p01 = static_cast<double>(sample(y0, x1, c));
                const double p10 = static_cast<double>(sample(y1, x0, c));
                const double p11 = static_cast<double>(sample(y1, x1, c));
                const double top = p00 + (p01 - p00) * fx;
                const double bot = p10 + (p11 - p10) * fx;
                dst_px[c] = warp_affine_cast<T>(top + (bot - top) * fy);
            }
        }
    }
}

inline void warpAffine_fallback(const Mat& src,
                                Mat& dst,
                                const Mat& M,
                                Size dsize,
                                int flags,
                                int borderMode,
                                const Scalar& borderValue)
{
    CV_Assert(!src.empty() && "warpAffine: source image can not be empty");
    CV_Assert(src.dims == 2 && "warpAffine: only 2D Mat is supported");

    const int interpolation = flags & 7;
    const bool interpolation_ok = interpolation == INTER_NEAREST || interpolation == INTER_LINEAR;
    if (!interpolation_ok)
    {
        CV_Error_(Error::StsBadArg, ("warpAffine: unsupported interpolation flags=%d", flags));
    }

    const int supported_flag_mask = 7 | WARP_INVERSE_MAP;
    if ((flags & ~supported_flag_mask) != 0)
    {
        CV_Error_(Error::StsBadArg, ("warpAffine: unsupported flags=%d", flags));
    }

    const int border_type = normalize_border_type(borderMode);
    if (!is_supported_filter_border(border_type))
    {
        CV_Error_(Error::StsBadArg, ("warpAffine: unsupported borderMode=%d", borderMode));
    }

    const int src_depth = src.depth();
    if (src_depth != CV_8U && src_depth != CV_32F)
    {
        CV_Error_(Error::StsBadArg, ("warpAffine: unsupported src depth=%d", src_depth));
    }

    if (src_depth == CV_8U)
    {
        warpAffine_fallback_impl_typed<uchar>(src, dst, M, dsize, flags, borderMode, borderValue);
        return;
    }

    warpAffine_fallback_impl_typed<float>(src, dst, M, dsize, flags, borderMode, borderValue);
}

inline WarpAffineFn& warp_affine_dispatch()
{
    static WarpAffineFn fn = &warpAffine_fallback;
    return fn;
}

inline void register_warp_affine_backend(WarpAffineFn fn)
{
    if (fn)
    {
        warp_affine_dispatch() = fn;
    }
}

inline bool is_warp_affine_backend_registered()
{
    return warp_affine_dispatch() != &warpAffine_fallback;
}

}  // namespace detail

inline void warpAffine(const Mat& src,
                       Mat& dst,
                       const Mat& M,
                       Size dsize,
                       int flags = INTER_LINEAR,
                       int borderMode = BORDER_CONSTANT,
                       const Scalar& borderValue = Scalar())
{
    detail::ensure_backends_registered_once();
    detail::warp_affine_dispatch()(src, dst, M, dsize, flags, borderMode, borderValue);
}

}  // namespace cvh

#endif  // CVH_IMGPROC_WARP_AFFINE_H
