#ifndef CVH_IMGPROC_BOX_FILTER_H
#define CVH_IMGPROC_BOX_FILTER_H

#include "detail/common.h"

#include <cstdint>
#include <type_traits>
#include <vector>

namespace cvh {
namespace detail {

using BoxFilterFn = void (*)(const Mat&, Mat&, int, Size, Point, bool, int);

template <typename T>
inline void boxFilter_fallback_impl_typed(const Mat& src, Mat& dst, Size ksize, Point anchor, bool normalize, int border_type)
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

    const int kernel_area = ksize.width * ksize.height;
    const double inv_kernel_area = kernel_area > 0 ? (1.0 / static_cast<double>(kernel_area)) : 0.0;

    using AccumT = typename std::conditional<std::is_same<T, uchar>::value, int64, double>::type;

    for (int y = 0; y < rows; ++y)
    {
        T* dst_row = reinterpret_cast<T*>(dst.data + static_cast<size_t>(y) * dst_step);
        for (int x = 0; x < cols; ++x)
        {
            T* dst_px = dst_row + static_cast<size_t>(x) * channels;
            for (int c = 0; c < channels; ++c)
            {
                AccumT sum = static_cast<AccumT>(0);
                for (int ky = 0; ky < ksize.height; ++ky)
                {
                    const int sy = y + ky - anchor.y;
                    const int src_y = border_interpolate(sy, rows, border_type);
                    if (src_y < 0)
                    {
                        continue;
                    }

                    const T* src_row = reinterpret_cast<const T*>(src_ref->data + static_cast<size_t>(src_y) * src_step);
                    for (int kx = 0; kx < ksize.width; ++kx)
                    {
                        const int sx = x + kx - anchor.x;
                        const int src_x = border_interpolate(sx, cols, border_type);
                        if (src_x < 0)
                        {
                            continue;
                        }
                        sum += static_cast<AccumT>(src_row[static_cast<size_t>(src_x) * channels + c]);
                    }
                }

                if constexpr (std::is_same<T, uchar>::value)
                {
                    if (normalize)
                    {
                        dst_px[c] = saturate_cast<uchar>(static_cast<float>(sum) * static_cast<float>(inv_kernel_area));
                    }
                    else
                    {
                        dst_px[c] = saturate_cast<uchar>(sum);
                    }
                }
                else
                {
                    if (normalize)
                    {
                        dst_px[c] = static_cast<float>(sum * inv_kernel_area);
                    }
                    else
                    {
                        dst_px[c] = static_cast<float>(sum);
                    }
                }
            }
        }
    }
}

inline void boxFilter_fallback(const Mat& src, Mat& dst, int ddepth, Size ksize, Point anchor, bool normalize, int borderType)
{
    CV_Assert(!src.empty() && "boxFilter: source image can not be empty");
    CV_Assert(src.dims == 2 && "boxFilter: only 2D Mat is supported");

    const int src_depth = src.depth();
    if (src_depth != CV_8U && src_depth != CV_32F)
    {
        CV_Error_(Error::StsBadArg, ("boxFilter: unsupported src depth=%d", src_depth));
    }

    const int allowed_ddepth = src_depth == CV_8U ? CV_8U : CV_32F;
    if (ddepth != -1 && ddepth != allowed_ddepth)
    {
        CV_Error_(Error::StsBadArg,
                  ("boxFilter: unsupported ddepth=%d (only -1/%s)",
                   ddepth,
                   src_depth == CV_8U ? "CV_8U" : "CV_32F"));
    }

    if (ksize.width <= 0 || ksize.height <= 0)
    {
        CV_Error_(Error::StsBadArg, ("boxFilter: invalid ksize=(%d,%d)", ksize.width, ksize.height));
    }

    const int anchor_x = anchor.x >= 0 ? anchor.x : (ksize.width / 2);
    const int anchor_y = anchor.y >= 0 ? anchor.y : (ksize.height / 2);
    if (anchor_x < 0 || anchor_x >= ksize.width || anchor_y < 0 || anchor_y >= ksize.height)
    {
        CV_Error_(Error::StsBadArg,
                  ("boxFilter: invalid anchor=(%d,%d) for ksize=(%d,%d)",
                   anchor_x, anchor_y, ksize.width, ksize.height));
    }

    const int border_type = normalize_border_type(borderType);
    if (!is_supported_filter_border(border_type))
    {
        CV_Error_(Error::StsBadArg, ("boxFilter: unsupported borderType=%d", borderType));
    }

    if (src_depth == CV_8U)
    {
        boxFilter_fallback_impl_typed<uchar>(src, dst, ksize, Point(anchor_x, anchor_y), normalize, border_type);
        return;
    }

    boxFilter_fallback_impl_typed<float>(src, dst, ksize, Point(anchor_x, anchor_y), normalize, border_type);
}

inline BoxFilterFn& boxfilter_dispatch()
{
    static BoxFilterFn fn = &boxFilter_fallback;
    return fn;
}

inline void register_boxfilter_backend(BoxFilterFn fn)
{
    if (fn)
    {
        boxfilter_dispatch() = fn;
    }
}

inline bool is_boxfilter_backend_registered()
{
    return boxfilter_dispatch() != &boxFilter_fallback;
}

}  // namespace detail

inline void boxFilter(const Mat& src,
                      Mat& dst,
                      int ddepth,
                      Size ksize,
                      Point anchor = Point(-1, -1),
                      bool normalize = true,
                      int borderType = BORDER_DEFAULT)
{
    detail::ensure_backends_registered_once();
    detail::boxfilter_dispatch()(src, dst, ddepth, ksize, anchor, normalize, borderType);
}

}  // namespace cvh

#endif  // CVH_IMGPROC_BOX_FILTER_H
