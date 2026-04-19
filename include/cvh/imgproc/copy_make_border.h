#ifndef CVH_IMGPROC_COPY_MAKE_BORDER_H
#define CVH_IMGPROC_COPY_MAKE_BORDER_H

#include "detail/common.h"

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace cvh {
namespace detail {

using CopyMakeBorderFn = void (*)(const Mat&, Mat&, int, int, int, int, int, const Scalar&);

inline int copy_make_border_interpolate(int p, int len, int border_type)
{
    if (border_type == BORDER_WRAP)
    {
        if (len == 1)
        {
            return 0;
        }
        int q = p % len;
        if (q < 0)
        {
            q += len;
        }
        return q;
    }
    return border_interpolate(p, len, border_type);
}

template <typename T>
inline T copy_make_border_scalar_value(const Scalar& border_value, int channel)
{
    const int idx = channel < 4 ? channel : 3;
    if constexpr (std::is_same<T, uchar>::value)
    {
        return saturate_cast<uchar>(border_value.val[idx]);
    }
    else
    {
        return static_cast<float>(border_value.val[idx]);
    }
}

template <typename T>
inline void copyMakeBorder_fallback_impl_typed(const Mat& src,
                                               Mat& dst,
                                               int top,
                                               int bottom,
                                               int left,
                                               int right,
                                               int border_type,
                                               const Scalar& border_value)
{
    Mat src_local;
    const Mat* src_ref = &src;
    if (src.data == dst.data)
    {
        src_local = src.clone();
        src_ref = &src_local;
    }

    const int src_rows = src_ref->size[0];
    const int src_cols = src_ref->size[1];
    const int channels = src_ref->channels();
    const size_t src_step = src_ref->step(0);

    const int dst_rows = src_rows + top + bottom;
    const int dst_cols = src_cols + left + right;
    dst.create(std::vector<int>{dst_rows, dst_cols}, src_ref->type());
    const size_t dst_step = dst.step(0);

    for (int y = 0; y < dst_rows; ++y)
    {
        T* dst_row = reinterpret_cast<T*>(dst.data + static_cast<size_t>(y) * dst_step);
        const int sy_raw = y - top;
        for (int x = 0; x < dst_cols; ++x)
        {
            T* dst_px = dst_row + static_cast<size_t>(x) * channels;
            const int sx_raw = x - left;

            const bool inside = (static_cast<unsigned>(sy_raw) < static_cast<unsigned>(src_rows)) &&
                                (static_cast<unsigned>(sx_raw) < static_cast<unsigned>(src_cols));
            if (inside)
            {
                const T* src_row = reinterpret_cast<const T*>(src_ref->data + static_cast<size_t>(sy_raw) * src_step);
                const T* src_px = src_row + static_cast<size_t>(sx_raw) * channels;
                for (int c = 0; c < channels; ++c)
                {
                    dst_px[c] = src_px[c];
                }
                continue;
            }

            if (border_type == BORDER_CONSTANT)
            {
                for (int c = 0; c < channels; ++c)
                {
                    dst_px[c] = copy_make_border_scalar_value<T>(border_value, c);
                }
                continue;
            }

            const int sy = copy_make_border_interpolate(sy_raw, src_rows, border_type);
            const int sx = copy_make_border_interpolate(sx_raw, src_cols, border_type);
            const T* src_row = reinterpret_cast<const T*>(src_ref->data + static_cast<size_t>(sy) * src_step);
            const T* src_px = src_row + static_cast<size_t>(sx) * channels;
            for (int c = 0; c < channels; ++c)
            {
                dst_px[c] = src_px[c];
            }
        }
    }
}

inline void copyMakeBorder_fallback(const Mat& src,
                                    Mat& dst,
                                    int top,
                                    int bottom,
                                    int left,
                                    int right,
                                    int borderType,
                                    const Scalar& value)
{
    CV_Assert(!src.empty() && "copyMakeBorder: source image can not be empty");
    CV_Assert(src.dims == 2 && "copyMakeBorder: only 2D Mat is supported");
    CV_Assert(top >= 0 && bottom >= 0 && left >= 0 && right >= 0 &&
              "copyMakeBorder: top/bottom/left/right must be >= 0");

    const int src_depth = src.depth();
    if (src_depth != CV_8U && src_depth != CV_32F)
    {
        CV_Error_(Error::StsBadArg, ("copyMakeBorder: unsupported src depth=%d", src_depth));
    }

    const int border_type = normalize_border_type(borderType);
    const bool supported_border = border_type == BORDER_CONSTANT ||
                                  border_type == BORDER_REPLICATE ||
                                  border_type == BORDER_REFLECT ||
                                  border_type == BORDER_REFLECT_101 ||
                                  border_type == BORDER_WRAP;
    if (!supported_border)
    {
        CV_Error_(Error::StsBadArg, ("copyMakeBorder: unsupported borderType=%d", borderType));
    }

    if (src_depth == CV_8U)
    {
        copyMakeBorder_fallback_impl_typed<uchar>(src, dst, top, bottom, left, right, border_type, value);
        return;
    }
    copyMakeBorder_fallback_impl_typed<float>(src, dst, top, bottom, left, right, border_type, value);
}

inline CopyMakeBorderFn& copy_make_border_dispatch()
{
    static CopyMakeBorderFn fn = &copyMakeBorder_fallback;
    return fn;
}

inline void register_copy_make_border_backend(CopyMakeBorderFn fn)
{
    if (fn)
    {
        copy_make_border_dispatch() = fn;
    }
}

inline bool is_copy_make_border_backend_registered()
{
    return copy_make_border_dispatch() != &copyMakeBorder_fallback;
}

}  // namespace detail

inline void copyMakeBorder(const Mat& src,
                           Mat& dst,
                           int top,
                           int bottom,
                           int left,
                           int right,
                           int borderType,
                           const Scalar& value = Scalar())
{
    detail::ensure_backends_registered_once();
    detail::copy_make_border_dispatch()(src, dst, top, bottom, left, right, borderType, value);
}

}  // namespace cvh

#endif  // CVH_IMGPROC_COPY_MAKE_BORDER_H
