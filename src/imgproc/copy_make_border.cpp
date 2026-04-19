#include "fastpath_common.h"

namespace cvh
{
namespace detail
{

namespace
{
bool try_copy_make_border_fastpath_replicate(const Mat& src,
                                             Mat& dst,
                                             int top,
                                             int bottom,
                                             int left,
                                             int right,
                                             int borderType)
{
    if (src.empty() || src.dims != 2)
    {
        return false;
    }

    if (top < 0 || bottom < 0 || left < 0 || right < 0)
    {
        return false;
    }

    const int src_depth = src.depth();
    if (src_depth != CV_8U && src_depth != CV_32F)
    {
        return false;
    }

    const int border_type = normalize_border_type(borderType);
    if (border_type != BORDER_REPLICATE)
    {
        return false;
    }

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
    if (src_rows <= 0 || src_cols <= 0 || channels <= 0)
    {
        return false;
    }

    const std::size_t pixel_bytes = src_ref->elemSize();
    const std::size_t src_step = src_ref->step(0);
    const std::size_t src_row_bytes = static_cast<std::size_t>(src_cols) * pixel_bytes;

    const int dst_rows = src_rows + top + bottom;
    const int dst_cols = src_cols + left + right;
    dst.create(std::vector<int>{dst_rows, dst_cols}, src_ref->type());
    const std::size_t dst_step = dst.step(0);

    const std::size_t left_bytes = static_cast<std::size_t>(left) * pixel_bytes;
    const bool do_parallel = should_parallelize_filter_rows(dst_rows, dst_cols, channels, 1);
    parallel_for_index_if(do_parallel, dst_rows, [&](int y) {
        const int sy = std::clamp(y - top, 0, src_rows - 1);
        const uchar* src_row = src_ref->data + static_cast<std::size_t>(sy) * src_step;
        uchar* dst_row = dst.data + static_cast<std::size_t>(y) * dst_step;

        uchar* dst_inner = dst_row + left_bytes;
        std::memcpy(dst_inner, src_row, src_row_bytes);

        if (left > 0)
        {
            const uchar* first_px = dst_inner;
            for (int x = 0; x < left; ++x)
            {
                std::memcpy(dst_row + static_cast<std::size_t>(x) * pixel_bytes, first_px, pixel_bytes);
            }
        }

        if (right > 0)
        {
            const uchar* last_px = dst_inner + static_cast<std::size_t>(src_cols - 1) * pixel_bytes;
            uchar* dst_right = dst_inner + src_row_bytes;
            for (int x = 0; x < right; ++x)
            {
                std::memcpy(dst_right + static_cast<std::size_t>(x) * pixel_bytes, last_px, pixel_bytes);
            }
        }
    });

    return true;
}


} // namespace

void copy_make_border_backend_impl(const Mat& src,
                                   Mat& dst,
                                   int top,
                                   int bottom,
                                   int left,
                                   int right,
                                   int borderType,
                                   const Scalar& value)
{
    if (try_copy_make_border_fastpath_replicate(src, dst, top, bottom, left, right, borderType))
    {
        return;
    }

    copyMakeBorder_fallback(src, dst, top, bottom, left, right, borderType, value);
}

} // namespace detail
} // namespace cvh
