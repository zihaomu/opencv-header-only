#ifndef CVH_IMGPROC_WARP_AFFINE_H
#define CVH_IMGPROC_WARP_AFFINE_H

#include "detail/geometric_sampling.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace cvh {
namespace detail {

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

    const int channels = src_ref->channels();

    const int dst_rows =
        dsize.height > 0 ? dsize.height : src_ref->size[0];
    const int dst_cols =
        dsize.width > 0 ? dsize.width : src_ref->size[1];
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

    for (int y = 0; y < dst_rows; ++y)
    {
        T* dst_row = reinterpret_cast<T*>(dst.data + static_cast<size_t>(y) * dst_step);
        for (int x = 0; x < dst_cols; ++x)
        {
            const double sx = m00 * static_cast<double>(x) + m01 * static_cast<double>(y) + m02;
            const double sy = m10 * static_cast<double>(x) + m11 * static_cast<double>(y) + m12;
            T* dst_px = dst_row + static_cast<size_t>(x) * channels;

            geometric_write_coordinate(
                *src_ref,
                dst_px,
                sx,
                sy,
                interpolation,
                border_type,
                borderValue,
                false,
                false);
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

}  // namespace detail

inline void warpAffine(const Mat& src,
                       Mat& dst,
                       const Mat& M,
                       Size dsize,
                       int flags = INTER_LINEAR,
                       int borderMode = BORDER_CONSTANT,
                       const Scalar& borderValue = Scalar())
{
    detail::warpAffine_fallback(src, dst, M, dsize, flags, borderMode, borderValue);
}

}  // namespace cvh

#endif  // CVH_IMGPROC_WARP_AFFINE_H
