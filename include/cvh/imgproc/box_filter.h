#ifndef CVH_IMGPROC_BOX_FILTER_H
#define CVH_IMGPROC_BOX_FILTER_H

#include "detail/common.h"

#include <cstdint>
#include <vector>

namespace cvh {
namespace detail {

using BoxFilterFn = void (*)(const Mat&, Mat&, int, Size, Point, bool, int);

inline void boxFilter_fallback(const Mat& src, Mat& dst, int ddepth, Size ksize, Point anchor, bool normalize, int borderType)
{
    CV_Assert(!src.empty() && "boxFilter: source image can not be empty");
    CV_Assert(src.dims == 2 && "boxFilter: only 2D Mat is supported");
    CV_Assert(src.depth() == CV_8U && "boxFilter: v1 supports CV_8U only");

    if (ddepth != -1 && ddepth != CV_8U)
    {
        CV_Error_(Error::StsBadArg, ("boxFilter: unsupported ddepth=%d (only -1/CV_8U)", ddepth));
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
    const float inv_kernel_area = kernel_area > 0 ? (1.0f / static_cast<float>(kernel_area)) : 0.0f;

    for (int y = 0; y < rows; ++y)
    {
        uchar* dst_row = dst.data + static_cast<size_t>(y) * dst_step;
        for (int x = 0; x < cols; ++x)
        {
            uchar* dst_px = dst_row + static_cast<size_t>(x) * channels;
            for (int c = 0; c < channels; ++c)
            {
                int64_t sum = 0;
                for (int ky = 0; ky < ksize.height; ++ky)
                {
                    const int sy = y + ky - anchor_y;
                    const int src_y = border_interpolate(sy, rows, border_type);
                    if (src_y < 0)
                    {
                        continue;
                    }

                    const uchar* src_row = src_ref->data + static_cast<size_t>(src_y) * src_step;
                    for (int kx = 0; kx < ksize.width; ++kx)
                    {
                        const int sx = x + kx - anchor_x;
                        const int src_x = border_interpolate(sx, cols, border_type);
                        if (src_x < 0)
                        {
                            continue;
                        }
                        sum += static_cast<int64_t>(src_row[static_cast<size_t>(src_x) * channels + c]);
                    }
                }

                if (normalize)
                {
                    dst_px[c] = saturate_cast<uchar>(static_cast<float>(sum) * inv_kernel_area);
                }
                else
                {
                    dst_px[c] = saturate_cast<uchar>(sum);
                }
            }
        }
    }
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
