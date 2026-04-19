#include "fastpath_common.h"

namespace cvh
{
namespace detail
{

namespace
{
thread_local const char* g_last_boxfilter_dispatch_path = "fallback";

inline void set_last_boxfilter_dispatch_path(const char* path)
{
    g_last_boxfilter_dispatch_path = path ? path : "fallback";
}

bool try_boxfilter_fastpath_u8(const Mat& src,
                               Mat& dst,
                               int ddepth,
                               Size ksize,
                               Point anchor,
                               bool normalize,
                               int borderType)
{
    if (src.empty() || src.dims != 2 || src.depth() != CV_8U)
    {
        return false;
    }

    if (!is_u8_fastpath_channels(src.channels()))
    {
        return false;
    }

    if (ddepth != -1 && ddepth != CV_8U)
    {
        return false;
    }

    if (ksize.width <= 0 || ksize.height <= 0)
    {
        return false;
    }

    const int anchor_x = anchor.x >= 0 ? anchor.x : (ksize.width / 2);
    const int anchor_y = anchor.y >= 0 ? anchor.y : (ksize.height / 2);
    if (anchor_x < 0 || anchor_x >= ksize.width || anchor_y < 0 || anchor_y >= ksize.height)
    {
        return false;
    }

    const int border_type = normalize_border_type(borderType);
    if (!is_supported_filter_border(border_type))
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

    const int rows = src_ref->size[0];
    const int cols = src_ref->size[1];
    const int channels = src_ref->channels();
    const std::size_t src_step = src_ref->step(0);
    const std::size_t dst_row_stride = static_cast<std::size_t>(cols) * static_cast<std::size_t>(channels);
    const int row_stride = cols * channels;

    dst.create(std::vector<int>{rows, cols}, src_ref->type());
    const std::size_t dst_step = dst.step(0);

    const int kx = ksize.width;
    const int ky = ksize.height;
    const int kernel_area = kx * ky;
    const float inv_kernel_area = kernel_area > 0 ? (1.0f / static_cast<float>(kernel_area)) : 0.0f;

    const int right = kx - anchor_x - 1;
    const int bottom = ky - anchor_y - 1;
    const std::vector<int> x_map = build_extended_index_map(cols, anchor_x, right, border_type);
    const std::vector<int> y_map = build_extended_index_map(rows, anchor_y, bottom, border_type);

    std::vector<std::int32_t> row_sums(static_cast<std::size_t>(rows) * static_cast<std::size_t>(row_stride), 0);
    const bool do_parallel_h = should_parallelize_filter_rows(rows, cols, channels, kx);
    parallel_for_index_if(do_parallel_h, rows, [&](int y) {
        const uchar* src_row = src_ref->data + static_cast<std::size_t>(y) * src_step;
        std::int32_t* sum_row = row_sums.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(row_stride);

        if (channels == 1)
        {
            std::int64_t s0 = 0;
            for (int i = 0; i < kx; ++i)
            {
                const int sx = x_map[static_cast<std::size_t>(i)];
                if (sx >= 0)
                {
                    s0 += static_cast<std::int64_t>(src_row[sx]);
                }
            }
            sum_row[0] = static_cast<std::int32_t>(s0);

            for (int x = 1; x < cols; ++x)
            {
                const int sx_add = x_map[static_cast<std::size_t>(x + kx - 1)];
                if (sx_add >= 0)
                {
                    s0 += static_cast<std::int64_t>(src_row[sx_add]);
                }

                const int sx_sub = x_map[static_cast<std::size_t>(x - 1)];
                if (sx_sub >= 0)
                {
                    s0 -= static_cast<std::int64_t>(src_row[sx_sub]);
                }

                sum_row[x] = static_cast<std::int32_t>(s0);
            }
            return;
        }

        if (channels == 3)
        {
            std::int64_t s0 = 0;
            std::int64_t s1 = 0;
            std::int64_t s2 = 0;
            for (int i = 0; i < kx; ++i)
            {
                const int sx = x_map[static_cast<std::size_t>(i)];
                if (sx < 0)
                {
                    continue;
                }
                const uchar* px = src_row + static_cast<std::size_t>(sx) * 3;
                s0 += static_cast<std::int64_t>(px[0]);
                s1 += static_cast<std::int64_t>(px[1]);
                s2 += static_cast<std::int64_t>(px[2]);
            }
            sum_row[0] = static_cast<std::int32_t>(s0);
            sum_row[1] = static_cast<std::int32_t>(s1);
            sum_row[2] = static_cast<std::int32_t>(s2);

            for (int x = 1; x < cols; ++x)
            {
                const int sx_add = x_map[static_cast<std::size_t>(x + kx - 1)];
                if (sx_add >= 0)
                {
                    const uchar* px = src_row + static_cast<std::size_t>(sx_add) * 3;
                    s0 += static_cast<std::int64_t>(px[0]);
                    s1 += static_cast<std::int64_t>(px[1]);
                    s2 += static_cast<std::int64_t>(px[2]);
                }

                const int sx_sub = x_map[static_cast<std::size_t>(x - 1)];
                if (sx_sub >= 0)
                {
                    const uchar* px = src_row + static_cast<std::size_t>(sx_sub) * 3;
                    s0 -= static_cast<std::int64_t>(px[0]);
                    s1 -= static_cast<std::int64_t>(px[1]);
                    s2 -= static_cast<std::int64_t>(px[2]);
                }

                const int dx = x * 3;
                sum_row[dx + 0] = static_cast<std::int32_t>(s0);
                sum_row[dx + 1] = static_cast<std::int32_t>(s1);
                sum_row[dx + 2] = static_cast<std::int32_t>(s2);
            }
            return;
        }

        if (channels == 4)
        {
            std::int64_t s0 = 0;
            std::int64_t s1 = 0;
            std::int64_t s2 = 0;
            std::int64_t s3 = 0;
            for (int i = 0; i < kx; ++i)
            {
                const int sx = x_map[static_cast<std::size_t>(i)];
                if (sx < 0)
                {
                    continue;
                }
                const uchar* px = src_row + static_cast<std::size_t>(sx) * 4;
                s0 += static_cast<std::int64_t>(px[0]);
                s1 += static_cast<std::int64_t>(px[1]);
                s2 += static_cast<std::int64_t>(px[2]);
                s3 += static_cast<std::int64_t>(px[3]);
            }
            sum_row[0] = static_cast<std::int32_t>(s0);
            sum_row[1] = static_cast<std::int32_t>(s1);
            sum_row[2] = static_cast<std::int32_t>(s2);
            sum_row[3] = static_cast<std::int32_t>(s3);

            for (int x = 1; x < cols; ++x)
            {
                const int sx_add = x_map[static_cast<std::size_t>(x + kx - 1)];
                if (sx_add >= 0)
                {
                    const uchar* px = src_row + static_cast<std::size_t>(sx_add) * 4;
                    s0 += static_cast<std::int64_t>(px[0]);
                    s1 += static_cast<std::int64_t>(px[1]);
                    s2 += static_cast<std::int64_t>(px[2]);
                    s3 += static_cast<std::int64_t>(px[3]);
                }

                const int sx_sub = x_map[static_cast<std::size_t>(x - 1)];
                if (sx_sub >= 0)
                {
                    const uchar* px = src_row + static_cast<std::size_t>(sx_sub) * 4;
                    s0 -= static_cast<std::int64_t>(px[0]);
                    s1 -= static_cast<std::int64_t>(px[1]);
                    s2 -= static_cast<std::int64_t>(px[2]);
                    s3 -= static_cast<std::int64_t>(px[3]);
                }

                const int dx = x * 4;
                sum_row[dx + 0] = static_cast<std::int32_t>(s0);
                sum_row[dx + 1] = static_cast<std::int32_t>(s1);
                sum_row[dx + 2] = static_cast<std::int32_t>(s2);
                sum_row[dx + 3] = static_cast<std::int32_t>(s3);
            }
            return;
        }

        std::vector<std::int64_t> sums(static_cast<std::size_t>(channels), 0);
        for (int i = 0; i < kx; ++i)
        {
            const int sx = x_map[static_cast<std::size_t>(i)];
            if (sx < 0)
            {
                continue;
            }
            const uchar* px = src_row + static_cast<std::size_t>(sx) * channels;
            for (int c = 0; c < channels; ++c)
            {
                sums[static_cast<std::size_t>(c)] += static_cast<std::int64_t>(px[c]);
            }
        }
        for (int c = 0; c < channels; ++c)
        {
            sum_row[c] = static_cast<std::int32_t>(sums[static_cast<std::size_t>(c)]);
        }

        for (int x = 1; x < cols; ++x)
        {
            const int sx_add = x_map[static_cast<std::size_t>(x + kx - 1)];
            if (sx_add >= 0)
            {
                const uchar* px_add = src_row + static_cast<std::size_t>(sx_add) * channels;
                for (int c = 0; c < channels; ++c)
                {
                    sums[static_cast<std::size_t>(c)] += static_cast<std::int64_t>(px_add[c]);
                }
            }

            const int sx_sub = x_map[static_cast<std::size_t>(x - 1)];
            if (sx_sub >= 0)
            {
                const uchar* px_sub = src_row + static_cast<std::size_t>(sx_sub) * channels;
                for (int c = 0; c < channels; ++c)
                {
                    sums[static_cast<std::size_t>(c)] -= static_cast<std::int64_t>(px_sub[c]);
                }
            }

            std::int32_t* out_px = sum_row + static_cast<std::size_t>(x) * channels;
            for (int c = 0; c < channels; ++c)
            {
                out_px[c] = static_cast<std::int32_t>(sums[static_cast<std::size_t>(c)]);
            }
        }
    });

    std::vector<std::int64_t> accum(dst_row_stride, 0);
    for (int i = 0; i < ky; ++i)
    {
        const int sy = y_map[static_cast<std::size_t>(i)];
        if (sy < 0)
        {
            continue;
        }
        const std::int32_t* row_ptr = row_sums.data() + static_cast<std::size_t>(sy) * static_cast<std::size_t>(row_stride);
        for (std::size_t idx = 0; idx < dst_row_stride; ++idx)
        {
            accum[idx] += static_cast<std::int64_t>(row_ptr[idx]);
        }
    }

    for (int y = 0; y < rows; ++y)
    {
        uchar* dst_row = dst.data + static_cast<std::size_t>(y) * dst_step;
        if (normalize)
        {
            for (std::size_t idx = 0; idx < dst_row_stride; ++idx)
            {
                dst_row[idx] = saturate_cast<uchar>(static_cast<float>(accum[idx]) * inv_kernel_area);
            }
        }
        else
        {
            for (std::size_t idx = 0; idx < dst_row_stride; ++idx)
            {
                dst_row[idx] = saturate_cast<uchar>(accum[idx]);
            }
        }

        if (y + 1 >= rows)
        {
            continue;
        }

        const int sy_sub = y_map[static_cast<std::size_t>(y)];
        if (sy_sub >= 0)
        {
            const std::int32_t* row_ptr = row_sums.data() + static_cast<std::size_t>(sy_sub) * static_cast<std::size_t>(row_stride);
            for (std::size_t idx = 0; idx < dst_row_stride; ++idx)
            {
                accum[idx] -= static_cast<std::int64_t>(row_ptr[idx]);
            }
        }

        const int sy_add = y_map[static_cast<std::size_t>(y + ky)];
        if (sy_add >= 0)
        {
            const std::int32_t* row_ptr = row_sums.data() + static_cast<std::size_t>(sy_add) * static_cast<std::size_t>(row_stride);
            for (std::size_t idx = 0; idx < dst_row_stride; ++idx)
            {
                accum[idx] += static_cast<std::int64_t>(row_ptr[idx]);
            }
        }
    }

    return true;
}

bool try_boxfilter_fastpath_f32(const Mat& src,
                                Mat& dst,
                                int ddepth,
                                Size ksize,
                                Point anchor,
                                bool normalize,
                                int borderType)
{
    if (src.empty() || src.dims != 2 || src.depth() != CV_32F)
    {
        return false;
    }

    if (!is_u8_fastpath_channels(src.channels()))
    {
        return false;
    }

    if (ddepth != -1 && ddepth != CV_32F)
    {
        return false;
    }

    if (ksize.width <= 0 || ksize.height <= 0)
    {
        return false;
    }

    const int anchor_x = anchor.x >= 0 ? anchor.x : (ksize.width / 2);
    const int anchor_y = anchor.y >= 0 ? anchor.y : (ksize.height / 2);
    if (anchor_x < 0 || anchor_x >= ksize.width || anchor_y < 0 || anchor_y >= ksize.height)
    {
        return false;
    }

    const int border_type = normalize_border_type(borderType);
    if (!is_supported_filter_border(border_type))
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

    const int rows = src_ref->size[0];
    const int cols = src_ref->size[1];
    const int channels = src_ref->channels();
    const std::size_t src_step = src_ref->step(0);
    const std::size_t dst_row_stride = static_cast<std::size_t>(cols) * static_cast<std::size_t>(channels);
    const int row_stride = cols * channels;

    dst.create(std::vector<int>{rows, cols}, src_ref->type());
    const std::size_t dst_step = dst.step(0);

    const int kx = ksize.width;
    const int ky = ksize.height;
    const int kernel_area = kx * ky;
    const float inv_kernel_area = kernel_area > 0 ? (1.0f / static_cast<float>(kernel_area)) : 0.0f;

    const int right = kx - anchor_x - 1;
    const int bottom = ky - anchor_y - 1;
    const std::vector<int> x_map = build_extended_index_map(cols, anchor_x, right, border_type);
    const std::vector<int> y_map = build_extended_index_map(rows, anchor_y, bottom, border_type);

    std::vector<float> row_sums(static_cast<std::size_t>(rows) * static_cast<std::size_t>(row_stride), 0.0f);
    const bool do_parallel_h = should_parallelize_filter_rows(rows, cols, channels, kx);
    parallel_for_index_if(do_parallel_h, rows, [&](int y) {
        const float* src_row = reinterpret_cast<const float*>(src_ref->data + static_cast<std::size_t>(y) * src_step);
        float* sum_row = row_sums.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(row_stride);

        if (channels == 1)
        {
            float s0 = 0.0f;
            for (int i = 0; i < kx; ++i)
            {
                const int sx = x_map[static_cast<std::size_t>(i)];
                if (sx >= 0)
                {
                    s0 += src_row[sx];
                }
            }
            sum_row[0] = s0;

            for (int x = 1; x < cols; ++x)
            {
                const int sx_add = x_map[static_cast<std::size_t>(x + kx - 1)];
                if (sx_add >= 0)
                {
                    s0 += src_row[sx_add];
                }

                const int sx_sub = x_map[static_cast<std::size_t>(x - 1)];
                if (sx_sub >= 0)
                {
                    s0 -= src_row[sx_sub];
                }

                sum_row[x] = s0;
            }
            return;
        }

        if (channels == 3)
        {
            float s0 = 0.0f;
            float s1 = 0.0f;
            float s2 = 0.0f;
            for (int i = 0; i < kx; ++i)
            {
                const int sx = x_map[static_cast<std::size_t>(i)];
                if (sx < 0)
                {
                    continue;
                }
                const float* px = src_row + static_cast<std::size_t>(sx) * 3;
                s0 += px[0];
                s1 += px[1];
                s2 += px[2];
            }
            sum_row[0] = s0;
            sum_row[1] = s1;
            sum_row[2] = s2;

            for (int x = 1; x < cols; ++x)
            {
                const int sx_add = x_map[static_cast<std::size_t>(x + kx - 1)];
                if (sx_add >= 0)
                {
                    const float* px = src_row + static_cast<std::size_t>(sx_add) * 3;
                    s0 += px[0];
                    s1 += px[1];
                    s2 += px[2];
                }

                const int sx_sub = x_map[static_cast<std::size_t>(x - 1)];
                if (sx_sub >= 0)
                {
                    const float* px = src_row + static_cast<std::size_t>(sx_sub) * 3;
                    s0 -= px[0];
                    s1 -= px[1];
                    s2 -= px[2];
                }

                const int dx = x * 3;
                sum_row[dx + 0] = s0;
                sum_row[dx + 1] = s1;
                sum_row[dx + 2] = s2;
            }
            return;
        }

        if (channels == 4)
        {
            float s0 = 0.0f;
            float s1 = 0.0f;
            float s2 = 0.0f;
            float s3 = 0.0f;
            for (int i = 0; i < kx; ++i)
            {
                const int sx = x_map[static_cast<std::size_t>(i)];
                if (sx < 0)
                {
                    continue;
                }
                const float* px = src_row + static_cast<std::size_t>(sx) * 4;
                s0 += px[0];
                s1 += px[1];
                s2 += px[2];
                s3 += px[3];
            }
            sum_row[0] = s0;
            sum_row[1] = s1;
            sum_row[2] = s2;
            sum_row[3] = s3;

            for (int x = 1; x < cols; ++x)
            {
                const int sx_add = x_map[static_cast<std::size_t>(x + kx - 1)];
                if (sx_add >= 0)
                {
                    const float* px = src_row + static_cast<std::size_t>(sx_add) * 4;
                    s0 += px[0];
                    s1 += px[1];
                    s2 += px[2];
                    s3 += px[3];
                }

                const int sx_sub = x_map[static_cast<std::size_t>(x - 1)];
                if (sx_sub >= 0)
                {
                    const float* px = src_row + static_cast<std::size_t>(sx_sub) * 4;
                    s0 -= px[0];
                    s1 -= px[1];
                    s2 -= px[2];
                    s3 -= px[3];
                }

                const int dx = x * 4;
                sum_row[dx + 0] = s0;
                sum_row[dx + 1] = s1;
                sum_row[dx + 2] = s2;
                sum_row[dx + 3] = s3;
            }
        }
    });

    std::vector<float> accum(dst_row_stride, 0.0f);
    for (int i = 0; i < ky; ++i)
    {
        const int sy = y_map[static_cast<std::size_t>(i)];
        if (sy < 0)
        {
            continue;
        }
        const float* row_ptr = row_sums.data() + static_cast<std::size_t>(sy) * static_cast<std::size_t>(row_stride);
        for (std::size_t idx = 0; idx < dst_row_stride; ++idx)
        {
            accum[idx] += row_ptr[idx];
        }
    }

    for (int y = 0; y < rows; ++y)
    {
        float* dst_row = reinterpret_cast<float*>(dst.data + static_cast<std::size_t>(y) * dst_step);
        if (normalize)
        {
            for (std::size_t idx = 0; idx < dst_row_stride; ++idx)
            {
                dst_row[idx] = accum[idx] * inv_kernel_area;
            }
        }
        else
        {
            std::memcpy(dst_row, accum.data(), dst_row_stride * sizeof(float));
        }

        if (y + 1 >= rows)
        {
            continue;
        }

        const int sy_sub = y_map[static_cast<std::size_t>(y)];
        if (sy_sub >= 0)
        {
            const float* row_ptr = row_sums.data() + static_cast<std::size_t>(sy_sub) * static_cast<std::size_t>(row_stride);
            for (std::size_t idx = 0; idx < dst_row_stride; ++idx)
            {
                accum[idx] -= row_ptr[idx];
            }
        }

        const int sy_add = y_map[static_cast<std::size_t>(y + ky)];
        if (sy_add >= 0)
        {
            const float* row_ptr = row_sums.data() + static_cast<std::size_t>(sy_add) * static_cast<std::size_t>(row_stride);
            for (std::size_t idx = 0; idx < dst_row_stride; ++idx)
            {
                accum[idx] += row_ptr[idx];
            }
        }
    }

    return true;
}


} // namespace

const char* last_boxfilter_dispatch_path()
{
    return g_last_boxfilter_dispatch_path;
}

void boxFilter_backend_impl(const Mat& src,
                            Mat& dst,
                            int ddepth,
                            Size ksize,
                            Point anchor,
                            bool normalize,
                            int borderType)
{
    set_last_boxfilter_dispatch_path("fallback");

    if (try_boxfilter_fastpath_u8(src, dst, ddepth, ksize, anchor, normalize, borderType))
    {
        if (is_boxfilter_3x3_candidate(ksize, anchor, normalize))
        {
            set_last_boxfilter_dispatch_path("box3x3");
        }
        else
        {
            set_last_boxfilter_dispatch_path("box_generic");
        }
        return;
    }

    if (try_boxfilter_fastpath_f32(src, dst, ddepth, ksize, anchor, normalize, borderType))
    {
        if (is_boxfilter_3x3_candidate(ksize, anchor, normalize))
        {
            set_last_boxfilter_dispatch_path("box3x3");
        }
        else
        {
            set_last_boxfilter_dispatch_path("box_generic");
        }
        return;
    }

    boxFilter_fallback(src, dst, ddepth, ksize, anchor, normalize, borderType);
}

} // namespace detail
} // namespace cvh
