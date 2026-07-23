#ifndef CVH_IMGPROC_DETAIL_RESIZE_IMPL_HPP
#define CVH_IMGPROC_DETAIL_RESIZE_IMPL_HPP

#include "fastpath_common.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace cvh
{
namespace detail
{

namespace resize_fastpath
{
inline std::vector<int> build_x_ofs_nearest(int src_cols, int dst_cols, bool exact)
{
    std::vector<int> x_ofs(static_cast<std::size_t>(dst_cols), 0);
    if (exact)
    {
        const int64_t ifx = ((static_cast<int64_t>(src_cols) << 16) + dst_cols / 2) / dst_cols;
        const int64_t ifx0 = ifx / 2 - (src_cols % 2);
        for (int x = 0; x < dst_cols; ++x)
        {
            const int sx = static_cast<int>((ifx * x + ifx0) >> 16);
            x_ofs[static_cast<std::size_t>(x)] = std::clamp(sx, 0, src_cols - 1);
        }
        return x_ofs;
    }

    for (int x = 0; x < dst_cols; ++x)
    {
        x_ofs[static_cast<std::size_t>(x)] = std::min(src_cols - 1, (x * src_cols) / dst_cols);
    }
    return x_ofs;
}

inline std::vector<int> build_y_ofs_nearest(int src_rows, int dst_rows, bool exact)
{
    std::vector<int> y_ofs(static_cast<std::size_t>(dst_rows), 0);
    if (exact)
    {
        const int64_t ify = ((static_cast<int64_t>(src_rows) << 16) + dst_rows / 2) / dst_rows;
        const int64_t ify0 = ify / 2 - (src_rows % 2);
        for (int y = 0; y < dst_rows; ++y)
        {
            const int sy = static_cast<int>((ify * y + ify0) >> 16);
            y_ofs[static_cast<std::size_t>(y)] = std::clamp(sy, 0, src_rows - 1);
        }
        return y_ofs;
    }

    for (int y = 0; y < dst_rows; ++y)
    {
        y_ofs[static_cast<std::size_t>(y)] = std::min(src_rows - 1, (y * src_rows) / dst_rows);
    }
    return y_ofs;
}

inline void resize_nearest_u8(const uchar* src_data,
                       std::size_t src_step,
                       uchar* dst_data,
                       std::size_t dst_step,
                       int src_rows,
                       int src_cols,
                       int dst_rows,
                       int dst_cols,
                       int channels,
                       bool exact)
{
    const std::vector<int> x_ofs = build_x_ofs_nearest(src_cols, dst_cols, exact);
    const std::vector<int> y_ofs = build_y_ofs_nearest(src_rows, dst_rows, exact);
    const bool do_parallel = should_parallelize_resize(dst_rows, dst_cols, channels);

    parallel_for_index_if(do_parallel, dst_rows, [&](int y) {
        const int sy = y_ofs[static_cast<std::size_t>(y)];
        const uchar* src_row = src_data + static_cast<std::size_t>(sy) * src_step;
        uchar* dst_row = dst_data + static_cast<std::size_t>(y) * dst_step;

        if (channels == 1)
        {
            for (int x = 0; x < dst_cols; ++x)
            {
                dst_row[x] = src_row[x_ofs[static_cast<std::size_t>(x)]];
            }
            return;
        }

        if (channels == 3)
        {
            for (int x = 0; x < dst_cols; ++x)
            {
                const int sx3 = x_ofs[static_cast<std::size_t>(x)] * 3;
                const int dx3 = x * 3;
                dst_row[dx3 + 0] = src_row[sx3 + 0];
                dst_row[dx3 + 1] = src_row[sx3 + 1];
                dst_row[dx3 + 2] = src_row[sx3 + 2];
            }
            return;
        }

        if (channels == 4)
        {
            for (int x = 0; x < dst_cols; ++x)
            {
                const int sx4 = x_ofs[static_cast<std::size_t>(x)] << 2;
                const int dx4 = x << 2;
                dst_row[dx4 + 0] = src_row[sx4 + 0];
                dst_row[dx4 + 1] = src_row[sx4 + 1];
                dst_row[dx4 + 2] = src_row[sx4 + 2];
                dst_row[dx4 + 3] = src_row[sx4 + 3];
            }
            return;
        }

        for (int x = 0; x < dst_cols; ++x)
        {
            const int sx = x_ofs[static_cast<std::size_t>(x)];
            const uchar* src_px = src_row + static_cast<std::size_t>(sx) * channels;
            uchar* dst_px = dst_row + static_cast<std::size_t>(x) * channels;
            std::memcpy(dst_px, src_px, static_cast<std::size_t>(channels));
        }
    });
}

inline void resize_linear_u8(const uchar* src_data,
                      std::size_t src_step,
                      uchar* dst_data,
                      std::size_t dst_step,
                      int src_rows,
                      int src_cols,
                      int dst_rows,
                      int dst_cols,
                      int channels)
{
    const float scale_x = static_cast<float>(src_cols) / static_cast<float>(dst_cols);
    const float scale_y = static_cast<float>(src_rows) / static_cast<float>(dst_rows);

    std::vector<int> x0b(static_cast<std::size_t>(dst_cols), 0);
    std::vector<int> x1b(static_cast<std::size_t>(dst_cols), 0);
    std::vector<float> wx(static_cast<std::size_t>(dst_cols), 0.0f);
    for (int x = 0; x < dst_cols; ++x)
    {
        const float fx_src = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
        const int ix0 = std::clamp(static_cast<int>(std::floor(fx_src)), 0, src_cols - 1);
        const int ix1 = std::min(ix0 + 1, src_cols - 1);
        x0b[static_cast<std::size_t>(x)] = ix0 * channels;
        x1b[static_cast<std::size_t>(x)] = ix1 * channels;
        wx[static_cast<std::size_t>(x)] = fx_src - static_cast<float>(ix0);
    }

    std::vector<int> y0(static_cast<std::size_t>(dst_rows), 0);
    std::vector<int> y1(static_cast<std::size_t>(dst_rows), 0);
    std::vector<float> wy(static_cast<std::size_t>(dst_rows), 0.0f);
    for (int y = 0; y < dst_rows; ++y)
    {
        const float fy_src = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const int iy0 = std::clamp(static_cast<int>(std::floor(fy_src)), 0, src_rows - 1);
        const int iy1 = std::min(iy0 + 1, src_rows - 1);
        y0[static_cast<std::size_t>(y)] = iy0;
        y1[static_cast<std::size_t>(y)] = iy1;
        wy[static_cast<std::size_t>(y)] = fy_src - static_cast<float>(iy0);
    }

    const bool do_parallel = should_parallelize_resize(dst_rows, dst_cols, channels);

    parallel_for_index_if(do_parallel, dst_rows, [&](int y) {
        const int iy0 = y0[static_cast<std::size_t>(y)];
        const int iy1 = y1[static_cast<std::size_t>(y)];
        const float wyv = wy[static_cast<std::size_t>(y)];

        const uchar* src_row0 = src_data + static_cast<std::size_t>(iy0) * src_step;
        const uchar* src_row1 = src_data + static_cast<std::size_t>(iy1) * src_step;
        uchar* dst_row = dst_data + static_cast<std::size_t>(y) * dst_step;

        if (channels == 1)
        {
            for (int x = 0; x < dst_cols; ++x)
            {
                const int ix0 = x0b[static_cast<std::size_t>(x)];
                const int ix1 = x1b[static_cast<std::size_t>(x)];
                const float wxv = wx[static_cast<std::size_t>(x)];
                const float top = lerp(static_cast<float>(src_row0[ix0]), static_cast<float>(src_row0[ix1]), wxv);
                const float bot = lerp(static_cast<float>(src_row1[ix0]), static_cast<float>(src_row1[ix1]), wxv);
                dst_row[x] = saturate_cast<uchar>(lerp(top, bot, wyv));
            }
            return;
        }

        if (channels == 3)
        {
            for (int x = 0; x < dst_cols; ++x)
            {
                const int ix0 = x0b[static_cast<std::size_t>(x)];
                const int ix1 = x1b[static_cast<std::size_t>(x)];
                const float wxv = wx[static_cast<std::size_t>(x)];
                const int dx3 = x * 3;

                const float top0 = lerp(static_cast<float>(src_row0[ix0 + 0]), static_cast<float>(src_row0[ix1 + 0]), wxv);
                const float top1 = lerp(static_cast<float>(src_row0[ix0 + 1]), static_cast<float>(src_row0[ix1 + 1]), wxv);
                const float top2 = lerp(static_cast<float>(src_row0[ix0 + 2]), static_cast<float>(src_row0[ix1 + 2]), wxv);
                const float bot0 = lerp(static_cast<float>(src_row1[ix0 + 0]), static_cast<float>(src_row1[ix1 + 0]), wxv);
                const float bot1 = lerp(static_cast<float>(src_row1[ix0 + 1]), static_cast<float>(src_row1[ix1 + 1]), wxv);
                const float bot2 = lerp(static_cast<float>(src_row1[ix0 + 2]), static_cast<float>(src_row1[ix1 + 2]), wxv);

                dst_row[dx3 + 0] = saturate_cast<uchar>(lerp(top0, bot0, wyv));
                dst_row[dx3 + 1] = saturate_cast<uchar>(lerp(top1, bot1, wyv));
                dst_row[dx3 + 2] = saturate_cast<uchar>(lerp(top2, bot2, wyv));
            }
            return;
        }

        if (channels == 4)
        {
            for (int x = 0; x < dst_cols; ++x)
            {
                const int ix0 = x0b[static_cast<std::size_t>(x)];
                const int ix1 = x1b[static_cast<std::size_t>(x)];
                const float wxv = wx[static_cast<std::size_t>(x)];
                const int dx4 = x << 2;

                const float top0 = lerp(static_cast<float>(src_row0[ix0 + 0]), static_cast<float>(src_row0[ix1 + 0]), wxv);
                const float top1 = lerp(static_cast<float>(src_row0[ix0 + 1]), static_cast<float>(src_row0[ix1 + 1]), wxv);
                const float top2 = lerp(static_cast<float>(src_row0[ix0 + 2]), static_cast<float>(src_row0[ix1 + 2]), wxv);
                const float top3 = lerp(static_cast<float>(src_row0[ix0 + 3]), static_cast<float>(src_row0[ix1 + 3]), wxv);
                const float bot0 = lerp(static_cast<float>(src_row1[ix0 + 0]), static_cast<float>(src_row1[ix1 + 0]), wxv);
                const float bot1 = lerp(static_cast<float>(src_row1[ix0 + 1]), static_cast<float>(src_row1[ix1 + 1]), wxv);
                const float bot2 = lerp(static_cast<float>(src_row1[ix0 + 2]), static_cast<float>(src_row1[ix1 + 2]), wxv);
                const float bot3 = lerp(static_cast<float>(src_row1[ix0 + 3]), static_cast<float>(src_row1[ix1 + 3]), wxv);

                dst_row[dx4 + 0] = saturate_cast<uchar>(lerp(top0, bot0, wyv));
                dst_row[dx4 + 1] = saturate_cast<uchar>(lerp(top1, bot1, wyv));
                dst_row[dx4 + 2] = saturate_cast<uchar>(lerp(top2, bot2, wyv));
                dst_row[dx4 + 3] = saturate_cast<uchar>(lerp(top3, bot3, wyv));
            }
            return;
        }

        for (int x = 0; x < dst_cols; ++x)
        {
            const int ix0 = x0b[static_cast<std::size_t>(x)];
            const int ix1 = x1b[static_cast<std::size_t>(x)];
            const float wxv = wx[static_cast<std::size_t>(x)];
            uchar* dst_px = dst_row + static_cast<std::size_t>(x) * channels;
            for (int c = 0; c < channels; ++c)
            {
                const float top = lerp(static_cast<float>(src_row0[ix0 + c]), static_cast<float>(src_row0[ix1 + c]), wxv);
                const float bot = lerp(static_cast<float>(src_row1[ix0 + c]), static_cast<float>(src_row1[ix1 + c]), wxv);
                dst_px[c] = saturate_cast<uchar>(lerp(top, bot, wyv));
            }
        }
    });
}

inline bool try_resize_fastpath_u8(const Mat& src, Mat& dst, Size dsize, double fx, double fy, int interpolation)
{
    if (src.empty() || src.dims != 2 || src.depth() != CV_8U)
    {
        return false;
    }

    const int channels = src.channels();
    if (!is_u8_fastpath_channels(channels))
    {
        return false;
    }

    if (interpolation != INTER_NEAREST &&
        interpolation != INTER_NEAREST_EXACT &&
        interpolation != INTER_LINEAR)
    {
        return false;
    }

    const int src_rows = src.size[0];
    const int src_cols = src.size[1];
    const int dst_cols = resolve_resize_dim(src_cols, dsize.width, fx);
    const int dst_rows = resolve_resize_dim(src_rows, dsize.height, fy);
    if (dst_cols <= 0 || dst_rows <= 0)
    {
        return false;
    }

#if CVH_ENABLE_OPENCV_INTRIN
    if (resize_linear_u8c1_downsample2_opencv_intrin_supported(src, dst_rows, dst_cols, interpolation))
    {
        return false;
    }
#endif

    dst.create(std::vector<int>{dst_rows, dst_cols}, src.type());

    const size_t src_step = src.step(0);
    const size_t dst_step = dst.step(0);
    const uchar* src_data = src.data;
    uchar* dst_data = dst.data;

    if (interpolation == INTER_NEAREST || interpolation == INTER_NEAREST_EXACT)
    {
        resize_nearest_u8(src_data, src_step, dst_data, dst_step,
                          src_rows, src_cols, dst_rows, dst_cols, channels,
                          interpolation == INTER_NEAREST_EXACT);
        return true;
    }

    resize_linear_u8(src_data, src_step, dst_data, dst_step,
                     src_rows, src_cols, dst_rows, dst_cols, channels);

    return true;
}

} // namespace resize_fastpath

inline void resize_fast_impl(const Mat& src, Mat& dst, Size dsize, double fx, double fy, int interpolation)
{
    if (resize_fastpath::try_resize_fastpath_u8(src, dst, dsize, fx, fy, interpolation))
    {
        return;
    }

    resize_fallback(src, dst, dsize, fx, fy, interpolation);
}

} // namespace detail
} // namespace cvh

#endif // CVH_IMGPROC_DETAIL_RESIZE_IMPL_HPP
