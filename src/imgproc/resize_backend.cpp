#include "cvh/imgproc/imgproc.h"
#include "cvh/core/parallel.h"

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

namespace
{

thread_local const char* g_last_boxfilter_dispatch_path = "fallback";
thread_local const char* g_last_gaussianblur_dispatch_path = "fallback";

inline void set_last_boxfilter_dispatch_path(const char* path)
{
    g_last_boxfilter_dispatch_path = path ? path : "fallback";
}

inline void set_last_gaussianblur_dispatch_path(const char* path)
{
    g_last_gaussianblur_dispatch_path = path ? path : "fallback";
}

inline bool is_u8_fastpath_channels(int cn)
{
    return cn == 1 || cn == 3 || cn == 4;
}

inline bool should_parallelize_resize(int rows, int cols, int channels)
{
    return cvh::cpu::should_parallelize_1d_loop(
        static_cast<std::size_t>(rows),
        static_cast<std::size_t>(cols) * static_cast<std::size_t>(channels),
        1LL << 16,
        2);
}

template <class Fn>
inline void parallel_for_index_if(bool do_parallel, int end, Fn&& fn)
{
    if (end <= 0)
    {
        return;
    }

    if (!do_parallel)
    {
        for (int i = 0; i < end; ++i)
        {
            fn(i);
        }
        return;
    }

    cvh::parallel_for_(
        cvh::Range(0, end),
        [&](const cvh::Range& range) {
            for (int i = range.start; i < range.end; ++i)
            {
                fn(i);
            }
        },
        static_cast<double>(end));
}

template <class Fn>
inline void parallel_for_index_if_step(bool do_parallel, int begin, int end, int step, Fn&& fn)
{
    CV_Assert(step > 0);
    if (begin >= end)
    {
        return;
    }

    const int count = (end - begin + step - 1) / step;
    parallel_for_index_if(
        do_parallel,
        count,
        [&](int idx) {
            fn(begin + idx * step);
        });
}

template <class Fn>
inline void parallel_for_count_if(bool do_parallel, std::size_t count, Fn&& fn)
{
    if (count == 0)
    {
        return;
    }

    const std::size_t max_parallel_range = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (!do_parallel || count > max_parallel_range)
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            fn(i);
        }
        return;
    }

    cvh::parallel_for_(
        cvh::Range(0, static_cast<int>(count)),
        [&](const cvh::Range& range) {
            for (int i = range.start; i < range.end; ++i)
            {
                fn(static_cast<std::size_t>(i));
            }
        },
        static_cast<double>(count));
}

inline bool is_boxfilter_3x3_candidate(Size ksize, Point anchor, bool normalize)
{
    if (!normalize || ksize.width != 3 || ksize.height != 3)
    {
        return false;
    }

    const int anchor_x = anchor.x >= 0 ? anchor.x : (ksize.width / 2);
    const int anchor_y = anchor.y >= 0 ? anchor.y : (ksize.height / 2);
    return anchor_x == 1 && anchor_y == 1;
}

inline bool resolve_gaussian_kernel_size(Size ksize, double sigmaX, double sigmaY, int& kx, int& ky)
{
    kx = ksize.width;
    ky = ksize.height;
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
    return kx > 0 && ky > 0 && (kx & 1) != 0 && (ky & 1) != 0;
}

std::vector<int> build_x_ofs_nearest(int src_cols, int dst_cols, bool exact)
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

std::vector<int> build_y_ofs_nearest(int src_rows, int dst_rows, bool exact)
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

void resize_nearest_u8(const uchar* src_data,
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

void resize_linear_u8(const uchar* src_data,
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

bool try_resize_fastpath_u8(const Mat& src, Mat& dst, Size dsize, double fx, double fy, int interpolation)
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

inline bool should_parallelize_cvtcolor(int rows, int cols, int src_channels)
{
    return cvh::cpu::should_parallelize_1d_loop(
        static_cast<std::size_t>(rows),
        static_cast<std::size_t>(cols) * static_cast<std::size_t>(src_channels),
        1LL << 15,
        2);
}

inline bool should_parallelize_filter_rows(int rows, int cols, int channels, int taps)
{
    return cvh::cpu::should_parallelize_1d_loop(
        static_cast<std::size_t>(rows),
        static_cast<std::size_t>(cols) * static_cast<std::size_t>(channels) * static_cast<std::size_t>(std::max(1, taps)),
        1LL << 16,
        2);
}

inline bool should_parallelize_threshold_contiguous(std::size_t scalar_count)
{
    return cvh::cpu::should_parallelize_1d_loop(
        scalar_count,
        1,
        1LL << 17,
        2);
}

inline bool should_parallelize_threshold_rows(int rows, int cols_scalar)
{
    return cvh::cpu::should_parallelize_1d_loop(
        static_cast<std::size_t>(rows),
        static_cast<std::size_t>(cols_scalar),
        1LL << 16,
        2);
}

std::vector<int> build_extended_index_map(int len, int left, int right, int border_type)
{
    CV_Assert(len > 0);
    CV_Assert(left >= 0 && right >= 0);

    const int ext_len = len + left + right;
    std::vector<int> map(static_cast<std::size_t>(ext_len), -1);
    for (int i = 0; i < ext_len; ++i)
    {
        map[static_cast<std::size_t>(i)] = border_interpolate(i - left, len, border_type);
    }
    return map;
}

void cvtcolor_bgr2gray_u8(const uchar* src_data,
                          std::size_t src_step,
                          uchar* dst_data,
                          std::size_t dst_step,
                          int rows,
                          int cols)
{
    // Integer coefficients matching BT.601 luminance conversion.
    constexpr int kB = 7471;
    constexpr int kG = 38470;
    constexpr int kR = 19595;
    constexpr int kRound = 1 << 15;

    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 3);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* src_row = src_data + static_cast<std::size_t>(y) * src_step;
        uchar* dst_row = dst_data + static_cast<std::size_t>(y) * dst_step;

        int x = 0;
        for (; x + 3 < cols; x += 4)
        {
            const int sx0 = x * 3;
            const int sx1 = sx0 + 3;
            const int sx2 = sx1 + 3;
            const int sx3 = sx2 + 3;

            dst_row[x + 0] = static_cast<uchar>((kB * src_row[sx0 + 0] + kG * src_row[sx0 + 1] + kR * src_row[sx0 + 2] + kRound) >> 16);
            dst_row[x + 1] = static_cast<uchar>((kB * src_row[sx1 + 0] + kG * src_row[sx1 + 1] + kR * src_row[sx1 + 2] + kRound) >> 16);
            dst_row[x + 2] = static_cast<uchar>((kB * src_row[sx2 + 0] + kG * src_row[sx2 + 1] + kR * src_row[sx2 + 2] + kRound) >> 16);
            dst_row[x + 3] = static_cast<uchar>((kB * src_row[sx3 + 0] + kG * src_row[sx3 + 1] + kR * src_row[sx3 + 2] + kRound) >> 16);
        }
        for (; x < cols; ++x)
        {
            const int sx = x * 3;
            dst_row[x] = static_cast<uchar>((kB * src_row[sx + 0] + kG * src_row[sx + 1] + kR * src_row[sx + 2] + kRound) >> 16);
        }
    });
}

void cvtcolor_gray2bgr_u8(const uchar* src_data,
                          std::size_t src_step,
                          uchar* dst_data,
                          std::size_t dst_step,
                          int rows,
                          int cols)
{
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 1);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* src_row = src_data + static_cast<std::size_t>(y) * src_step;
        uchar* dst_row = dst_data + static_cast<std::size_t>(y) * dst_step;

        int x = 0;
        for (; x + 7 < cols; x += 8)
        {
            const int dx = x * 3;
            const uchar g0 = src_row[x + 0];
            const uchar g1 = src_row[x + 1];
            const uchar g2 = src_row[x + 2];
            const uchar g3 = src_row[x + 3];
            const uchar g4 = src_row[x + 4];
            const uchar g5 = src_row[x + 5];
            const uchar g6 = src_row[x + 6];
            const uchar g7 = src_row[x + 7];

            dst_row[dx + 0] = g0; dst_row[dx + 1] = g0; dst_row[dx + 2] = g0;
            dst_row[dx + 3] = g1; dst_row[dx + 4] = g1; dst_row[dx + 5] = g1;
            dst_row[dx + 6] = g2; dst_row[dx + 7] = g2; dst_row[dx + 8] = g2;
            dst_row[dx + 9] = g3; dst_row[dx + 10] = g3; dst_row[dx + 11] = g3;
            dst_row[dx + 12] = g4; dst_row[dx + 13] = g4; dst_row[dx + 14] = g4;
            dst_row[dx + 15] = g5; dst_row[dx + 16] = g5; dst_row[dx + 17] = g5;
            dst_row[dx + 18] = g6; dst_row[dx + 19] = g6; dst_row[dx + 20] = g6;
            dst_row[dx + 21] = g7; dst_row[dx + 22] = g7; dst_row[dx + 23] = g7;
        }
        for (; x < cols; ++x)
        {
            const uchar g = src_row[x];
            const int dx = x * 3;
            dst_row[dx + 0] = g;
            dst_row[dx + 1] = g;
            dst_row[dx + 2] = g;
        }
    });
}

template <typename T>
void cvtcolor_gray_to_4ch_alpha(const uchar* src_data,
                                std::size_t src_step,
                                uchar* dst_data,
                                std::size_t dst_step,
                                int rows,
                                int cols,
                                T alpha)
{
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 1);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const T* src_row = reinterpret_cast<const T*>(src_data + static_cast<std::size_t>(y) * src_step);
        T* dst_row = reinterpret_cast<T*>(dst_data + static_cast<std::size_t>(y) * dst_step);

        for (int x = 0; x < cols; ++x)
        {
            const T g = src_row[x];
            const int dx = x * 4;
            dst_row[dx + 0] = g;
            dst_row[dx + 1] = g;
            dst_row[dx + 2] = g;
            dst_row[dx + 3] = alpha;
        }
    });
}

void cvtcolor_4ch_to_gray_u8(const uchar* src_data,
                             std::size_t src_step,
                             uchar* dst_data,
                             std::size_t dst_step,
                             int rows,
                             int cols,
                             bool rgba_order)
{
    constexpr int kB = 7471;
    constexpr int kG = 38470;
    constexpr int kR = 19595;
    constexpr int kRound = 1 << 15;

    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 4);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* src_row = src_data + static_cast<std::size_t>(y) * src_step;
        uchar* dst_row = dst_data + static_cast<std::size_t>(y) * dst_step;

        int x = 0;
        for (; x + 3 < cols; x += 4)
        {
            const int sx0 = x * 4;
            const int sx1 = sx0 + 4;
            const int sx2 = sx1 + 4;
            const int sx3 = sx2 + 4;

            dst_row[x + 0] = static_cast<uchar>((kB * src_row[sx0 + (rgba_order ? 2 : 0)] + kG * src_row[sx0 + 1] + kR * src_row[sx0 + (rgba_order ? 0 : 2)] + kRound) >> 16);
            dst_row[x + 1] = static_cast<uchar>((kB * src_row[sx1 + (rgba_order ? 2 : 0)] + kG * src_row[sx1 + 1] + kR * src_row[sx1 + (rgba_order ? 0 : 2)] + kRound) >> 16);
            dst_row[x + 2] = static_cast<uchar>((kB * src_row[sx2 + (rgba_order ? 2 : 0)] + kG * src_row[sx2 + 1] + kR * src_row[sx2 + (rgba_order ? 0 : 2)] + kRound) >> 16);
            dst_row[x + 3] = static_cast<uchar>((kB * src_row[sx3 + (rgba_order ? 2 : 0)] + kG * src_row[sx3 + 1] + kR * src_row[sx3 + (rgba_order ? 0 : 2)] + kRound) >> 16);
        }
        for (; x < cols; ++x)
        {
            const int sx = x * 4;
            dst_row[x] = static_cast<uchar>((kB * src_row[sx + (rgba_order ? 2 : 0)] + kG * src_row[sx + 1] + kR * src_row[sx + (rgba_order ? 0 : 2)] + kRound) >> 16);
        }
    });
}

void cvtcolor_4ch_to_gray_f32(const uchar* src_data,
                              std::size_t src_step,
                              uchar* dst_data,
                              std::size_t dst_step,
                              int rows,
                              int cols,
                              bool rgba_order)
{
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 4);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const float* src_row = reinterpret_cast<const float*>(src_data + static_cast<std::size_t>(y) * src_step);
        float* dst_row = reinterpret_cast<float*>(dst_data + static_cast<std::size_t>(y) * dst_step);

        int x = 0;
        for (; x + 3 < cols; x += 4)
        {
            const int sx0 = x * 4;
            const int sx1 = sx0 + 4;
            const int sx2 = sx1 + 4;
            const int sx3 = sx2 + 4;

            dst_row[x + 0] = 0.114f * src_row[sx0 + (rgba_order ? 2 : 0)] + 0.587f * src_row[sx0 + 1] + 0.299f * src_row[sx0 + (rgba_order ? 0 : 2)];
            dst_row[x + 1] = 0.114f * src_row[sx1 + (rgba_order ? 2 : 0)] + 0.587f * src_row[sx1 + 1] + 0.299f * src_row[sx1 + (rgba_order ? 0 : 2)];
            dst_row[x + 2] = 0.114f * src_row[sx2 + (rgba_order ? 2 : 0)] + 0.587f * src_row[sx2 + 1] + 0.299f * src_row[sx2 + (rgba_order ? 0 : 2)];
            dst_row[x + 3] = 0.114f * src_row[sx3 + (rgba_order ? 2 : 0)] + 0.587f * src_row[sx3 + 1] + 0.299f * src_row[sx3 + (rgba_order ? 0 : 2)];
        }
        for (; x < cols; ++x)
        {
            const int sx = x * 4;
            dst_row[x] = 0.114f * src_row[sx + (rgba_order ? 2 : 0)] + 0.587f * src_row[sx + 1] + 0.299f * src_row[sx + (rgba_order ? 0 : 2)];
        }
    });
}

template <typename T>
void cvtcolor_swap_rb_3ch(const uchar* src_data,
                          std::size_t src_step,
                          uchar* dst_data,
                          std::size_t dst_step,
                          int rows,
                          int cols)
{
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 3);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const T* src_row = reinterpret_cast<const T*>(src_data + static_cast<std::size_t>(y) * src_step);
        T* dst_row = reinterpret_cast<T*>(dst_data + static_cast<std::size_t>(y) * dst_step);

        int x = 0;
        for (; x + 3 < cols; x += 4)
        {
            const int sx0 = x * 3;
            const int sx1 = sx0 + 3;
            const int sx2 = sx1 + 3;
            const int sx3 = sx2 + 3;

            dst_row[sx0 + 0] = src_row[sx0 + 2];
            dst_row[sx0 + 1] = src_row[sx0 + 1];
            dst_row[sx0 + 2] = src_row[sx0 + 0];

            dst_row[sx1 + 0] = src_row[sx1 + 2];
            dst_row[sx1 + 1] = src_row[sx1 + 1];
            dst_row[sx1 + 2] = src_row[sx1 + 0];

            dst_row[sx2 + 0] = src_row[sx2 + 2];
            dst_row[sx2 + 1] = src_row[sx2 + 1];
            dst_row[sx2 + 2] = src_row[sx2 + 0];

            dst_row[sx3 + 0] = src_row[sx3 + 2];
            dst_row[sx3 + 1] = src_row[sx3 + 1];
            dst_row[sx3 + 2] = src_row[sx3 + 0];
        }
        for (; x < cols; ++x)
        {
            const int sx = x * 3;
            dst_row[sx + 0] = src_row[sx + 2];
            dst_row[sx + 1] = src_row[sx + 1];
            dst_row[sx + 2] = src_row[sx + 0];
        }
    });
}

template <typename T>
void cvtcolor_3ch_to_4ch_alpha(const uchar* src_data,
                               std::size_t src_step,
                               uchar* dst_data,
                               std::size_t dst_step,
                               int rows,
                               int cols,
                               T alpha,
                               bool swap_rb)
{
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 3);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const T* src_row = reinterpret_cast<const T*>(src_data + static_cast<std::size_t>(y) * src_step);
        T* dst_row = reinterpret_cast<T*>(dst_data + static_cast<std::size_t>(y) * dst_step);

        int x = 0;
        for (; x + 3 < cols; x += 4)
        {
            for (int i = 0; i < 4; ++i)
            {
                const int sx = (x + i) * 3;
                const int dx = (x + i) * 4;
                dst_row[dx + 0] = src_row[sx + (swap_rb ? 2 : 0)];
                dst_row[dx + 1] = src_row[sx + 1];
                dst_row[dx + 2] = src_row[sx + (swap_rb ? 0 : 2)];
                dst_row[dx + 3] = alpha;
            }
        }
        for (; x < cols; ++x)
        {
            const int sx = x * 3;
            const int dx = x * 4;
            dst_row[dx + 0] = src_row[sx + (swap_rb ? 2 : 0)];
            dst_row[dx + 1] = src_row[sx + 1];
            dst_row[dx + 2] = src_row[sx + (swap_rb ? 0 : 2)];
            dst_row[dx + 3] = alpha;
        }
    });
}

template <typename T>
void cvtcolor_4ch_to_3ch_drop_alpha(const uchar* src_data,
                                    std::size_t src_step,
                                    uchar* dst_data,
                                    std::size_t dst_step,
                                    int rows,
                                    int cols,
                                    bool swap_rb)
{
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 4);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const T* src_row = reinterpret_cast<const T*>(src_data + static_cast<std::size_t>(y) * src_step);
        T* dst_row = reinterpret_cast<T*>(dst_data + static_cast<std::size_t>(y) * dst_step);

        int x = 0;
        for (; x + 3 < cols; x += 4)
        {
            for (int i = 0; i < 4; ++i)
            {
                const int sx = (x + i) * 4;
                const int dx = (x + i) * 3;
                dst_row[dx + 0] = src_row[sx + (swap_rb ? 2 : 0)];
                dst_row[dx + 1] = src_row[sx + 1];
                dst_row[dx + 2] = src_row[sx + (swap_rb ? 0 : 2)];
            }
        }
        for (; x < cols; ++x)
        {
            const int sx = x * 4;
            const int dx = x * 3;
            dst_row[dx + 0] = src_row[sx + (swap_rb ? 2 : 0)];
            dst_row[dx + 1] = src_row[sx + 1];
            dst_row[dx + 2] = src_row[sx + (swap_rb ? 0 : 2)];
        }
    });
}

template <typename T>
void cvtcolor_swap_rb_4ch(const uchar* src_data,
                          std::size_t src_step,
                          uchar* dst_data,
                          std::size_t dst_step,
                          int rows,
                          int cols)
{
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 4);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const T* src_row = reinterpret_cast<const T*>(src_data + static_cast<std::size_t>(y) * src_step);
        T* dst_row = reinterpret_cast<T*>(dst_data + static_cast<std::size_t>(y) * dst_step);

        int x = 0;
        for (; x + 3 < cols; x += 4)
        {
            for (int i = 0; i < 4; ++i)
            {
                const int sx = (x + i) * 4;
                dst_row[sx + 0] = src_row[sx + 2];
                dst_row[sx + 1] = src_row[sx + 1];
                dst_row[sx + 2] = src_row[sx + 0];
                dst_row[sx + 3] = src_row[sx + 3];
            }
        }
        for (; x < cols; ++x)
        {
            const int sx = x * 4;
            dst_row[sx + 0] = src_row[sx + 2];
            dst_row[sx + 1] = src_row[sx + 1];
            dst_row[sx + 2] = src_row[sx + 0];
            dst_row[sx + 3] = src_row[sx + 3];
        }
    });
}

template <typename T>
void cvtcolor_3ch_to_yuv(const uchar* src_data,
                         std::size_t src_step,
                         uchar* dst_data,
                         std::size_t dst_step,
                         int rows,
                         int cols,
                         bool rgb_order)
{
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 3);
    const float delta = std::is_same_v<T, uchar> ? 128.0f : 0.5f;
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const T* src_row = reinterpret_cast<const T*>(src_data + static_cast<std::size_t>(y) * src_step);
        T* dst_row = reinterpret_cast<T*>(dst_data + static_cast<std::size_t>(y) * dst_step);

        int x = 0;
        for (; x + 3 < cols; x += 4)
        {
            for (int i = 0; i < 4; ++i)
            {
                const int sx = (x + i) * 3;
                const float r = static_cast<float>(src_row[sx + (rgb_order ? 0 : 2)]);
                const float g = static_cast<float>(src_row[sx + 1]);
                const float b = static_cast<float>(src_row[sx + (rgb_order ? 2 : 0)]);
                const float yy = 0.299f * r + 0.587f * g + 0.114f * b;
                const float uu = 0.492f * (b - yy) + delta;
                const float vv = 0.877f * (r - yy) + delta;

                if constexpr (std::is_same_v<T, uchar>)
                {
                    dst_row[sx + 0] = saturate_cast<uchar>(yy);
                    dst_row[sx + 1] = saturate_cast<uchar>(uu);
                    dst_row[sx + 2] = saturate_cast<uchar>(vv);
                }
                else
                {
                    dst_row[sx + 0] = yy;
                    dst_row[sx + 1] = uu;
                    dst_row[sx + 2] = vv;
                }
            }
        }
        for (; x < cols; ++x)
        {
            const int sx = x * 3;
            const float r = static_cast<float>(src_row[sx + (rgb_order ? 0 : 2)]);
            const float g = static_cast<float>(src_row[sx + 1]);
            const float b = static_cast<float>(src_row[sx + (rgb_order ? 2 : 0)]);
            const float yy = 0.299f * r + 0.587f * g + 0.114f * b;
            const float uu = 0.492f * (b - yy) + delta;
            const float vv = 0.877f * (r - yy) + delta;

            if constexpr (std::is_same_v<T, uchar>)
            {
                dst_row[sx + 0] = saturate_cast<uchar>(yy);
                dst_row[sx + 1] = saturate_cast<uchar>(uu);
                dst_row[sx + 2] = saturate_cast<uchar>(vv);
            }
            else
            {
                dst_row[sx + 0] = yy;
                dst_row[sx + 1] = uu;
                dst_row[sx + 2] = vv;
            }
        }
    });
}

template <typename T>
void cvtcolor_yuv_to_3ch(const uchar* src_data,
                         std::size_t src_step,
                         uchar* dst_data,
                         std::size_t dst_step,
                         int rows,
                         int cols,
                         bool rgb_order)
{
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 3);
    const float delta = std::is_same_v<T, uchar> ? 128.0f : 0.5f;
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const T* src_row = reinterpret_cast<const T*>(src_data + static_cast<std::size_t>(y) * src_step);
        T* dst_row = reinterpret_cast<T*>(dst_data + static_cast<std::size_t>(y) * dst_step);

        int x = 0;
        for (; x + 3 < cols; x += 4)
        {
            for (int i = 0; i < 4; ++i)
            {
                const int sx = (x + i) * 3;
                const float yy = static_cast<float>(src_row[sx + 0]);
                const float uu = static_cast<float>(src_row[sx + 1]) - delta;
                const float vv = static_cast<float>(src_row[sx + 2]) - delta;
                const float b = yy + 2.032f * uu;
                const float g = yy - 0.395f * uu - 0.581f * vv;
                const float r = yy + 1.140f * vv;

                if constexpr (std::is_same_v<T, uchar>)
                {
                    dst_row[sx + (rgb_order ? 0 : 2)] = saturate_cast<uchar>(r);
                    dst_row[sx + 1] = saturate_cast<uchar>(g);
                    dst_row[sx + (rgb_order ? 2 : 0)] = saturate_cast<uchar>(b);
                }
                else
                {
                    dst_row[sx + (rgb_order ? 0 : 2)] = r;
                    dst_row[sx + 1] = g;
                    dst_row[sx + (rgb_order ? 2 : 0)] = b;
                }
            }
        }
        for (; x < cols; ++x)
        {
            const int sx = x * 3;
            const float yy = static_cast<float>(src_row[sx + 0]);
            const float uu = static_cast<float>(src_row[sx + 1]) - delta;
            const float vv = static_cast<float>(src_row[sx + 2]) - delta;
            const float b = yy + 2.032f * uu;
            const float g = yy - 0.395f * uu - 0.581f * vv;
            const float r = yy + 1.140f * vv;

            if constexpr (std::is_same_v<T, uchar>)
            {
                dst_row[sx + (rgb_order ? 0 : 2)] = saturate_cast<uchar>(r);
                dst_row[sx + 1] = saturate_cast<uchar>(g);
                dst_row[sx + (rgb_order ? 2 : 0)] = saturate_cast<uchar>(b);
            }
            else
            {
                dst_row[sx + (rgb_order ? 0 : 2)] = r;
                dst_row[sx + 1] = g;
                dst_row[sx + (rgb_order ? 2 : 0)] = b;
            }
        }
    });
}

void cvtcolor_yuv420sp_to_3ch_u8(const uchar* src_data,
                                 std::size_t src_step,
                                 uchar* dst_data,
                                 std::size_t dst_step,
                                 int rows,
                                 int cols,
                                 bool nv21_layout,
                                 bool rgb_order)
{
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 1);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* y_row = src_data + static_cast<std::size_t>(y) * src_step;
        const uchar* uv_row = src_data + static_cast<std::size_t>(rows + y / 2) * src_step;
        uchar* dst_row = dst_data + static_cast<std::size_t>(y) * dst_step;

        for (int x = 0; x < cols; x += 2)
        {
            const int first = static_cast<int>(uv_row[x + 0]);
            const int second = static_cast<int>(uv_row[x + 1]);
            const int uu = nv21_layout ? second : first;
            const int vv = nv21_layout ? first : second;

            for (int i = 0; i < 2; ++i)
            {
                const int dx = (x + i) * 3;
                const int yy = static_cast<int>(y_row[x + i]);
                const uchar b = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 0);
                const uchar g = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 1);
                const uchar r = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 2);

                dst_row[dx + (rgb_order ? 0 : 2)] = r;
                dst_row[dx + 1] = g;
                dst_row[dx + (rgb_order ? 2 : 0)] = b;
            }
        }
    });
}

void cvtcolor_3ch_to_yuv420sp_u8(const uchar* src_data,
                                 std::size_t src_step,
                                 uchar* dst_data,
                                 std::size_t dst_step,
                                 int rows,
                                 int cols,
                                 bool rgb_order,
                                 bool nv21_layout)
{
    CV_Assert((rows % 2) == 0 && "cvtColor(BGR/RGB2YUV420sp): source height must be even");
    CV_Assert((cols % 2) == 0 && "cvtColor(BGR/RGB2YUV420sp): source width must be even");

    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 3);
    parallel_for_index_if_step(do_parallel, 0, rows, 2, [&](int y) {
        const uchar* src_row0 = src_data + static_cast<std::size_t>(y + 0) * src_step;
        const uchar* src_row1 = src_data + static_cast<std::size_t>(y + 1) * src_step;
        uchar* dst_y_row0 = dst_data + static_cast<std::size_t>(y + 0) * dst_step;
        uchar* dst_y_row1 = dst_data + static_cast<std::size_t>(y + 1) * dst_step;
        uchar* dst_uv_row = dst_data + static_cast<std::size_t>(rows + y / 2) * dst_step;

        for (int x = 0; x < cols; x += 2)
        {
            int sum_b = 0;
            int sum_g = 0;
            int sum_r = 0;

            for (int dy = 0; dy < 2; ++dy)
            {
                const uchar* src_row = (dy == 0) ? src_row0 : src_row1;
                uchar* dst_y_row = (dy == 0) ? dst_y_row0 : dst_y_row1;

                for (int dx = 0; dx < 2; ++dx)
                {
                    const int sx = (x + dx) * 3;
                    const int bb = static_cast<int>(src_row[sx + (rgb_order ? 2 : 0)]);
                    const int gg = static_cast<int>(src_row[sx + 1]);
                    const int rr = static_cast<int>(src_row[sx + (rgb_order ? 0 : 2)]);
                    const uchar yy = cvtcolor_color3_to_yuv_limited_u8(bb, gg, rr, 0);

                    dst_y_row[x + dx] = yy;
                    sum_b += bb;
                    sum_g += gg;
                    sum_r += rr;
                }
            }

            const int avg_b = (sum_b + 2) >> 2;
            const int avg_g = (sum_g + 2) >> 2;
            const int avg_r = (sum_r + 2) >> 2;
            const uchar uu = cvtcolor_color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 1);
            const uchar vv = cvtcolor_color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 2);

            dst_uv_row[x + 0] = nv21_layout ? vv : uu;
            dst_uv_row[x + 1] = nv21_layout ? uu : vv;
        }
    });
}

void cvtcolor_3ch_to_yuv420p_u8(const uchar* src_data,
                                std::size_t src_step,
                                uchar* dst_data,
                                std::size_t dst_step,
                                int rows,
                                int cols,
                                bool rgb_order,
                                bool yv12_layout)
{
    CV_Assert((rows % 2) == 0 && "cvtColor(BGR/RGB2YUV420p): source height must be even");
    CV_Assert((cols % 2) == 0 && "cvtColor(BGR/RGB2YUV420p): source width must be even");

    const int uv_size = rows * cols / 4;
    const int u_plane_offset = yv12_layout ? uv_size : 0;
    const int v_plane_offset = yv12_layout ? 0 : uv_size;
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 3);
    parallel_for_index_if_step(do_parallel, 0, rows, 2, [&](int y) {
        const uchar* src_row0 = src_data + static_cast<std::size_t>(y + 0) * src_step;
        const uchar* src_row1 = src_data + static_cast<std::size_t>(y + 1) * src_step;
        uchar* dst_y_row0 = dst_data + static_cast<std::size_t>(y + 0) * dst_step;
        uchar* dst_y_row1 = dst_data + static_cast<std::size_t>(y + 1) * dst_step;

        for (int x = 0; x < cols; x += 2)
        {
            int sum_b = 0;
            int sum_g = 0;
            int sum_r = 0;

            for (int dy = 0; dy < 2; ++dy)
            {
                const uchar* src_row = (dy == 0) ? src_row0 : src_row1;
                uchar* dst_y_row = (dy == 0) ? dst_y_row0 : dst_y_row1;

                for (int dx = 0; dx < 2; ++dx)
                {
                    const int sx = (x + dx) * 3;
                    const int bb = static_cast<int>(src_row[sx + (rgb_order ? 2 : 0)]);
                    const int gg = static_cast<int>(src_row[sx + 1]);
                    const int rr = static_cast<int>(src_row[sx + (rgb_order ? 0 : 2)]);
                    const uchar yy = cvtcolor_color3_to_yuv_limited_u8(bb, gg, rr, 0);

                    dst_y_row[x + dx] = yy;
                    sum_b += bb;
                    sum_g += gg;
                    sum_r += rr;
                }
            }

            const int avg_b = (sum_b + 2) >> 2;
            const int avg_g = (sum_g + 2) >> 2;
            const int avg_r = (sum_r + 2) >> 2;
            const uchar uu = cvtcolor_color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 1);
            const uchar vv = cvtcolor_color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 2);
            const int chroma_index = (y / 2) * (cols / 2) + (x / 2);

            *(dst_data +
              static_cast<std::size_t>(rows + (u_plane_offset + chroma_index) / cols) * dst_step +
              static_cast<std::size_t>((u_plane_offset + chroma_index) % cols)) = uu;
            *(dst_data +
              static_cast<std::size_t>(rows + (v_plane_offset + chroma_index) / cols) * dst_step +
              static_cast<std::size_t>((v_plane_offset + chroma_index) % cols)) = vv;
        }
    });
}

void cvtcolor_yuv422sp_to_3ch_u8(const uchar* src_data,
                                 std::size_t src_step,
                                 uchar* dst_data,
                                 std::size_t dst_step,
                                 int rows,
                                 int cols,
                                 bool nv61_layout,
                                 bool rgb_order)
{
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 1);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* y_row = src_data + static_cast<std::size_t>(y) * src_step;
        const uchar* uv_row = src_data + static_cast<std::size_t>(rows + y) * src_step;
        uchar* dst_row = dst_data + static_cast<std::size_t>(y) * dst_step;

        for (int x = 0; x < cols; x += 2)
        {
            const int first = static_cast<int>(uv_row[x + 0]);
            const int second = static_cast<int>(uv_row[x + 1]);
            const int uu = nv61_layout ? second : first;
            const int vv = nv61_layout ? first : second;

            for (int i = 0; i < 2; ++i)
            {
                const int dx = (x + i) * 3;
                const int yy = static_cast<int>(y_row[x + i]);
                const uchar b = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 0);
                const uchar g = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 1);
                const uchar r = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 2);

                dst_row[dx + (rgb_order ? 0 : 2)] = r;
                dst_row[dx + 1] = g;
                dst_row[dx + (rgb_order ? 2 : 0)] = b;
            }
        }
    });
}

void cvtcolor_3ch_to_yuv422sp_u8(const uchar* src_data,
                                 std::size_t src_step,
                                 uchar* dst_data,
                                 std::size_t dst_step,
                                 int rows,
                                 int cols,
                                 bool rgb_order,
                                 bool nv61_layout)
{
    CV_Assert((cols % 2) == 0 && "cvtColor(BGR/RGB2YUV422sp): source width must be even");

    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 3);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* src_row = src_data + static_cast<std::size_t>(y) * src_step;
        uchar* dst_y_row = dst_data + static_cast<std::size_t>(y) * dst_step;
        uchar* dst_uv_row = dst_data + static_cast<std::size_t>(rows + y) * dst_step;

        for (int x = 0; x < cols; x += 2)
        {
            int sum_b = 0;
            int sum_g = 0;
            int sum_r = 0;

            for (int i = 0; i < 2; ++i)
            {
                const int sx = (x + i) * 3;
                const int bb = static_cast<int>(src_row[sx + (rgb_order ? 2 : 0)]);
                const int gg = static_cast<int>(src_row[sx + 1]);
                const int rr = static_cast<int>(src_row[sx + (rgb_order ? 0 : 2)]);
                const uchar yy = cvtcolor_color3_to_yuv_limited_u8(bb, gg, rr, 0);

                dst_y_row[x + i] = yy;
                sum_b += bb;
                sum_g += gg;
                sum_r += rr;
            }

            const int avg_b = (sum_b + 1) >> 1;
            const int avg_g = (sum_g + 1) >> 1;
            const int avg_r = (sum_r + 1) >> 1;
            const uchar uu = cvtcolor_color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 1);
            const uchar vv = cvtcolor_color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 2);

            dst_uv_row[x + 0] = nv61_layout ? vv : uu;
            dst_uv_row[x + 1] = nv61_layout ? uu : vv;
        }
    });
}

void cvtcolor_3ch_to_yuv422packed_u8(const uchar* src_data,
                                     std::size_t src_step,
                                     uchar* dst_data,
                                     std::size_t dst_step,
                                     int rows,
                                     int cols,
                                     bool rgb_order,
                                     bool uyvy_layout)
{
    CV_Assert((cols % 2) == 0 && "cvtColor(BGR/RGB2YUV422packed): source width must be even");

    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 3);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* src_row = src_data + static_cast<std::size_t>(y) * src_step;
        uchar* dst_row = dst_data + static_cast<std::size_t>(y) * dst_step;

        for (int x = 0; x < cols; x += 2)
        {
            int sum_b = 0;
            int sum_g = 0;
            int sum_r = 0;
            uchar yy[2] = {0, 0};

            for (int i = 0; i < 2; ++i)
            {
                const int sx = (x + i) * 3;
                const int bb = static_cast<int>(src_row[sx + (rgb_order ? 2 : 0)]);
                const int gg = static_cast<int>(src_row[sx + 1]);
                const int rr = static_cast<int>(src_row[sx + (rgb_order ? 0 : 2)]);

                yy[i] = cvtcolor_color3_to_yuv_limited_u8(bb, gg, rr, 0);
                sum_b += bb;
                sum_g += gg;
                sum_r += rr;
            }

            const int avg_b = (sum_b + 1) >> 1;
            const int avg_g = (sum_g + 1) >> 1;
            const int avg_r = (sum_r + 1) >> 1;
            const uchar uu = cvtcolor_color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 1);
            const uchar vv = cvtcolor_color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 2);
            const int base = x * 2;

            if (uyvy_layout)
            {
                dst_row[base + 0] = uu;
                dst_row[base + 1] = yy[0];
                dst_row[base + 2] = vv;
                dst_row[base + 3] = yy[1];
            }
            else
            {
                dst_row[base + 0] = yy[0];
                dst_row[base + 1] = uu;
                dst_row[base + 2] = yy[1];
                dst_row[base + 3] = vv;
            }
        }
    });
}

inline uchar cvtcolor_yuv444sp_plane_byte_u8(const uchar* src_data,
                                             std::size_t src_step,
                                             int rows,
                                             int cols,
                                             int plane_index)
{
    return *(src_data +
             static_cast<std::size_t>(rows + plane_index / cols) * src_step +
             static_cast<std::size_t>(plane_index % cols));
}

void cvtcolor_yuv444sp_to_3ch_u8(const uchar* src_data,
                                 std::size_t src_step,
                                 uchar* dst_data,
                                 std::size_t dst_step,
                                 int rows,
                                 int cols,
                                 bool nv42_layout,
                                 bool rgb_order)
{
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 1);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* y_row = src_data + static_cast<std::size_t>(y) * src_step;
        uchar* dst_row = dst_data + static_cast<std::size_t>(y) * dst_step;

        for (int x = 0; x < cols; ++x)
        {
            const int yy = static_cast<int>(y_row[x]);
            const int base = y * (cols * 2) + x * 2;
            const int uu = static_cast<int>(cvtcolor_yuv444sp_plane_byte_u8(src_data, src_step, rows, cols, base + (nv42_layout ? 1 : 0)));
            const int vv = static_cast<int>(cvtcolor_yuv444sp_plane_byte_u8(src_data, src_step, rows, cols, base + (nv42_layout ? 0 : 1)));
            const uchar b = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 0);
            const uchar g = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 1);
            const uchar r = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 2);

            const int dx = x * 3;
            dst_row[dx + (rgb_order ? 0 : 2)] = r;
            dst_row[dx + 1] = g;
            dst_row[dx + (rgb_order ? 2 : 0)] = b;
        }
    });
}

void cvtcolor_3ch_to_yuv444sp_u8(const uchar* src_data,
                                 std::size_t src_step,
                                 uchar* dst_data,
                                 std::size_t dst_step,
                                 int rows,
                                 int cols,
                                 bool rgb_order,
                                 bool nv42_layout)
{
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 3);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* src_row = src_data + static_cast<std::size_t>(y) * src_step;
        uchar* dst_y_row = dst_data + static_cast<std::size_t>(y) * dst_step;

        for (int x = 0; x < cols; ++x)
        {
            const int sx = x * 3;
            const int bb = static_cast<int>(src_row[sx + (rgb_order ? 2 : 0)]);
            const int gg = static_cast<int>(src_row[sx + 1]);
            const int rr = static_cast<int>(src_row[sx + (rgb_order ? 0 : 2)]);
            const uchar yy = cvtcolor_color3_to_yuv_limited_u8(bb, gg, rr, 0);
            const uchar uu = cvtcolor_color3_to_yuv_limited_u8(bb, gg, rr, 1);
            const uchar vv = cvtcolor_color3_to_yuv_limited_u8(bb, gg, rr, 2);
            const int base = y * (cols * 2) + x * 2;

            dst_y_row[x] = yy;
            *(dst_data +
              static_cast<std::size_t>(rows + (base + 0) / cols) * dst_step +
              static_cast<std::size_t>((base + 0) % cols)) = nv42_layout ? vv : uu;
            *(dst_data +
              static_cast<std::size_t>(rows + (base + 1) / cols) * dst_step +
              static_cast<std::size_t>((base + 1) % cols)) = nv42_layout ? uu : vv;
        }
    });
}

void cvtcolor_3ch_to_yuv444p_u8(const uchar* src_data,
                                std::size_t src_step,
                                uchar* dst_data,
                                std::size_t dst_step,
                                int rows,
                                int cols,
                                bool rgb_order,
                                bool yv24_layout)
{
    const int plane_size = rows * cols;
    const int u_plane_offset = yv24_layout ? plane_size : 0;
    const int v_plane_offset = yv24_layout ? 0 : plane_size;
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 3);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* src_row = src_data + static_cast<std::size_t>(y) * src_step;
        uchar* dst_y_row = dst_data + static_cast<std::size_t>(y) * dst_step;

        for (int x = 0; x < cols; ++x)
        {
            const int sx = x * 3;
            const int bb = static_cast<int>(src_row[sx + (rgb_order ? 2 : 0)]);
            const int gg = static_cast<int>(src_row[sx + 1]);
            const int rr = static_cast<int>(src_row[sx + (rgb_order ? 0 : 2)]);
            const uchar yy = cvtcolor_color3_to_yuv_limited_u8(bb, gg, rr, 0);
            const uchar uu = cvtcolor_color3_to_yuv_limited_u8(bb, gg, rr, 1);
            const uchar vv = cvtcolor_color3_to_yuv_limited_u8(bb, gg, rr, 2);
            const int chroma_index = y * cols + x;

            dst_y_row[x] = yy;
            *(dst_data +
              static_cast<std::size_t>(rows + (u_plane_offset + chroma_index) / cols) * dst_step +
              static_cast<std::size_t>((u_plane_offset + chroma_index) % cols)) = uu;
            *(dst_data +
              static_cast<std::size_t>(rows + (v_plane_offset + chroma_index) / cols) * dst_step +
              static_cast<std::size_t>((v_plane_offset + chroma_index) % cols)) = vv;
        }
    });
}

inline uchar cvtcolor_yuv420p_plane_byte_u8(const uchar* src_data,
                                            std::size_t src_step,
                                            int rows,
                                            int cols,
                                            int plane_offset,
                                            int plane_index)
{
    const int logical_offset = plane_offset + plane_index;
    return *(src_data +
             static_cast<std::size_t>(rows + logical_offset / cols) * src_step +
             static_cast<std::size_t>(logical_offset % cols));
}

inline uchar cvtcolor_yuv444p_plane_byte_u8(const uchar* src_data,
                                            std::size_t src_step,
                                            int rows,
                                            int cols,
                                            int plane_offset,
                                            int plane_index)
{
    const int logical_offset = plane_offset + plane_index;
    return *(src_data +
             static_cast<std::size_t>(rows + logical_offset / cols) * src_step +
             static_cast<std::size_t>(logical_offset % cols));
}

void cvtcolor_yuv420p_to_3ch_u8(const uchar* src_data,
                                std::size_t src_step,
                                uchar* dst_data,
                                std::size_t dst_step,
                                int rows,
                                int cols,
                                bool yv12_layout,
                                bool rgb_order)
{
    const int uv_size = rows * cols / 4;
    const int u_plane_offset = yv12_layout ? uv_size : 0;
    const int v_plane_offset = yv12_layout ? 0 : uv_size;
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 1);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* y_row = src_data + static_cast<std::size_t>(y) * src_step;
        uchar* dst_row = dst_data + static_cast<std::size_t>(y) * dst_step;

        for (int x = 0; x < cols; ++x)
        {
            const int yy = static_cast<int>(y_row[x]);
            const int chroma_index = (y / 2) * (cols / 2) + (x / 2);
            const int uu = static_cast<int>(cvtcolor_yuv420p_plane_byte_u8(src_data, src_step, rows, cols, u_plane_offset, chroma_index));
            const int vv = static_cast<int>(cvtcolor_yuv420p_plane_byte_u8(src_data, src_step, rows, cols, v_plane_offset, chroma_index));
            const uchar b = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 0);
            const uchar g = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 1);
            const uchar r = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 2);

            const int dx = x * 3;
            dst_row[dx + (rgb_order ? 0 : 2)] = r;
            dst_row[dx + 1] = g;
            dst_row[dx + (rgb_order ? 2 : 0)] = b;
        }
    });
}

void cvtcolor_yuv444p_to_3ch_u8(const uchar* src_data,
                                std::size_t src_step,
                                uchar* dst_data,
                                std::size_t dst_step,
                                int rows,
                                int cols,
                                bool yv24_layout,
                                bool rgb_order)
{
    const int plane_size = rows * cols;
    const int u_plane_offset = yv24_layout ? plane_size : 0;
    const int v_plane_offset = yv24_layout ? 0 : plane_size;
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 1);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* y_row = src_data + static_cast<std::size_t>(y) * src_step;
        uchar* dst_row = dst_data + static_cast<std::size_t>(y) * dst_step;

        for (int x = 0; x < cols; ++x)
        {
            const int yy = static_cast<int>(y_row[x]);
            const int chroma_index = y * cols + x;
            const int uu = static_cast<int>(cvtcolor_yuv444p_plane_byte_u8(src_data, src_step, rows, cols, u_plane_offset, chroma_index));
            const int vv = static_cast<int>(cvtcolor_yuv444p_plane_byte_u8(src_data, src_step, rows, cols, v_plane_offset, chroma_index));
            const uchar b = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 0);
            const uchar g = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 1);
            const uchar r = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 2);

            const int dx = x * 3;
            dst_row[dx + (rgb_order ? 0 : 2)] = r;
            dst_row[dx + 1] = g;
            dst_row[dx + (rgb_order ? 2 : 0)] = b;
        }
    });
}

void cvtcolor_yuv422packed_to_3ch_u8(const uchar* src_data,
                                     std::size_t src_step,
                                     uchar* dst_data,
                                     std::size_t dst_step,
                                     int rows,
                                     int cols,
                                     bool uyvy_layout,
                                     bool rgb_order)
{
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 2);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* src_row = src_data + static_cast<std::size_t>(y) * src_step;
        uchar* dst_row = dst_data + static_cast<std::size_t>(y) * dst_step;

        for (int x = 0; x < cols; x += 2)
        {
            const int base = x * 2;
            const int first0 = static_cast<int>(src_row[base + 0]);
            const int first1 = static_cast<int>(src_row[base + 1]);
            const int second0 = static_cast<int>(src_row[base + 2]);
            const int second1 = static_cast<int>(src_row[base + 3]);

            const int yy0 = uyvy_layout ? first1 : first0;
            const int uu = uyvy_layout ? first0 : first1;
            const int yy1 = uyvy_layout ? second1 : second0;
            const int vv = uyvy_layout ? second0 : second1;

            for (int i = 0; i < 2; ++i)
            {
                const int dx = (x + i) * 3;
                const int yy = (i == 0) ? yy0 : yy1;
                const uchar b = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 0);
                const uchar g = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 1);
                const uchar r = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 2);

                dst_row[dx + (rgb_order ? 0 : 2)] = r;
                dst_row[dx + 1] = g;
                dst_row[dx + (rgb_order ? 2 : 0)] = b;
            }
        }
    });
}

void cvtcolor_bgr2gray_f32(const uchar* src_data,
                           std::size_t src_step,
                           uchar* dst_data,
                           std::size_t dst_step,
                           int rows,
                           int cols)
{
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 3);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const float* src_row = reinterpret_cast<const float*>(src_data + static_cast<std::size_t>(y) * src_step);
        float* dst_row = reinterpret_cast<float*>(dst_data + static_cast<std::size_t>(y) * dst_step);

        int x = 0;
        for (; x + 3 < cols; x += 4)
        {
            const int sx0 = x * 3;
            const int sx1 = sx0 + 3;
            const int sx2 = sx1 + 3;
            const int sx3 = sx2 + 3;

            dst_row[x + 0] = 0.114f * src_row[sx0 + 0] + 0.587f * src_row[sx0 + 1] + 0.299f * src_row[sx0 + 2];
            dst_row[x + 1] = 0.114f * src_row[sx1 + 0] + 0.587f * src_row[sx1 + 1] + 0.299f * src_row[sx1 + 2];
            dst_row[x + 2] = 0.114f * src_row[sx2 + 0] + 0.587f * src_row[sx2 + 1] + 0.299f * src_row[sx2 + 2];
            dst_row[x + 3] = 0.114f * src_row[sx3 + 0] + 0.587f * src_row[sx3 + 1] + 0.299f * src_row[sx3 + 2];
        }
        for (; x < cols; ++x)
        {
            const int sx = x * 3;
            dst_row[x] = 0.114f * src_row[sx + 0] + 0.587f * src_row[sx + 1] + 0.299f * src_row[sx + 2];
        }
    });
}

void cvtcolor_gray2bgr_f32(const uchar* src_data,
                           std::size_t src_step,
                           uchar* dst_data,
                           std::size_t dst_step,
                           int rows,
                           int cols)
{
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 1);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const float* src_row = reinterpret_cast<const float*>(src_data + static_cast<std::size_t>(y) * src_step);
        float* dst_row = reinterpret_cast<float*>(dst_data + static_cast<std::size_t>(y) * dst_step);

        int x = 0;
        for (; x + 3 < cols; x += 4)
        {
            const float g0 = src_row[x + 0];
            const float g1 = src_row[x + 1];
            const float g2 = src_row[x + 2];
            const float g3 = src_row[x + 3];
            const int dx = x * 3;

            dst_row[dx + 0] = g0; dst_row[dx + 1] = g0; dst_row[dx + 2] = g0;
            dst_row[dx + 3] = g1; dst_row[dx + 4] = g1; dst_row[dx + 5] = g1;
            dst_row[dx + 6] = g2; dst_row[dx + 7] = g2; dst_row[dx + 8] = g2;
            dst_row[dx + 9] = g3; dst_row[dx + 10] = g3; dst_row[dx + 11] = g3;
        }
        for (; x < cols; ++x)
        {
            const float g = src_row[x];
            const int dx = x * 3;
            dst_row[dx + 0] = g;
            dst_row[dx + 1] = g;
            dst_row[dx + 2] = g;
        }
    });
}

bool try_cvtcolor_fastpath_u8(const Mat& src, Mat& dst, int code)
{
    if (src.empty() || src.dims != 2 || src.depth() != CV_8U)
    {
        return false;
    }

    const int rows = src.size[0];
    const int cols = src.size[1];
    const size_t src_step = src.step(0);

    if (code == COLOR_BGR2GRAY)
    {
        if (src.channels() != 3)
        {
            return false;
        }

        dst.create(std::vector<int>{rows, cols}, CV_8UC1);
        cvtcolor_bgr2gray_u8(src.data, src_step, dst.data, dst.step(0), rows, cols);
        return true;
    }

    if (code == COLOR_GRAY2BGR)
    {
        if (src.channels() != 1)
        {
            return false;
        }

        dst.create(std::vector<int>{rows, cols}, CV_8UC3);
        cvtcolor_gray2bgr_u8(src.data, src_step, dst.data, dst.step(0), rows, cols);
        return true;
    }

    if (code == COLOR_GRAY2BGRA || code == COLOR_GRAY2RGBA)
    {
        if (src.channels() != 1)
        {
            return false;
        }

        dst.create(std::vector<int>{rows, cols}, CV_8UC4);
        cvtcolor_gray_to_4ch_alpha<uchar>(src.data, src_step, dst.data, dst.step(0), rows, cols, static_cast<uchar>(255));
        return true;
    }

    if (code == COLOR_BGR2RGB || code == COLOR_RGB2BGR)
    {
        if (src.channels() != 3)
        {
            return false;
        }

        dst.create(std::vector<int>{rows, cols}, CV_8UC3);
        cvtcolor_swap_rb_3ch<uchar>(src.data, src_step, dst.data, dst.step(0), rows, cols);
        return true;
    }

    if (code == COLOR_BGR2YUV || code == COLOR_RGB2YUV)
    {
        if (src.channels() != 3)
        {
            return false;
        }

        dst.create(std::vector<int>{rows, cols}, CV_8UC3);
        cvtcolor_3ch_to_yuv<uchar>(src.data, src_step, dst.data, dst.step(0), rows, cols, code == COLOR_RGB2YUV);
        return true;
    }

    if (code == COLOR_BGR2YUV_NV24 ||
        code == COLOR_RGB2YUV_NV24 ||
        code == COLOR_BGR2YUV_NV42 ||
        code == COLOR_RGB2YUV_NV42)
    {
        if (src.channels() != 3)
        {
            return false;
        }

        dst.create(std::vector<int>{rows * 3, cols}, CV_8UC1);
        cvtcolor_3ch_to_yuv444sp_u8(
            src.data,
            src_step,
            dst.data,
            dst.step(0),
            rows,
            cols,
            code == COLOR_RGB2YUV_NV24 || code == COLOR_RGB2YUV_NV42,
            code == COLOR_BGR2YUV_NV42 || code == COLOR_RGB2YUV_NV42);
        return true;
    }

    if (code == COLOR_BGR2YUV_NV12 ||
        code == COLOR_RGB2YUV_NV12 ||
        code == COLOR_BGR2YUV_NV21 ||
        code == COLOR_RGB2YUV_NV21)
    {
        if (src.channels() != 3)
        {
            return false;
        }

        dst.create(std::vector<int>{rows * 3 / 2, cols}, CV_8UC1);
        cvtcolor_3ch_to_yuv420sp_u8(
            src.data,
            src_step,
            dst.data,
            dst.step(0),
            rows,
            cols,
            code == COLOR_RGB2YUV_NV12 || code == COLOR_RGB2YUV_NV21,
            code == COLOR_BGR2YUV_NV21 || code == COLOR_RGB2YUV_NV21);
        return true;
    }

    if (code == COLOR_BGR2YUV_I420 ||
        code == COLOR_RGB2YUV_I420 ||
        code == COLOR_BGR2YUV_YV12 ||
        code == COLOR_RGB2YUV_YV12)
    {
        if (src.channels() != 3)
        {
            return false;
        }

        dst.create(std::vector<int>{rows * 3 / 2, cols}, CV_8UC1);
        cvtcolor_3ch_to_yuv420p_u8(
            src.data,
            src_step,
            dst.data,
            dst.step(0),
            rows,
            cols,
            code == COLOR_RGB2YUV_I420 || code == COLOR_RGB2YUV_YV12,
            code == COLOR_BGR2YUV_YV12 || code == COLOR_RGB2YUV_YV12);
        return true;
    }

    if (code == COLOR_BGR2YUV_NV16 ||
        code == COLOR_RGB2YUV_NV16 ||
        code == COLOR_BGR2YUV_NV61 ||
        code == COLOR_RGB2YUV_NV61)
    {
        if (src.channels() != 3)
        {
            return false;
        }

        dst.create(std::vector<int>{rows * 2, cols}, CV_8UC1);
        cvtcolor_3ch_to_yuv422sp_u8(
            src.data,
            src_step,
            dst.data,
            dst.step(0),
            rows,
            cols,
            code == COLOR_RGB2YUV_NV16 || code == COLOR_RGB2YUV_NV61,
            code == COLOR_BGR2YUV_NV61 || code == COLOR_RGB2YUV_NV61);
        return true;
    }

    if (code == COLOR_BGR2YUV_YUY2 ||
        code == COLOR_RGB2YUV_YUY2 ||
        code == COLOR_BGR2YUV_UYVY ||
        code == COLOR_RGB2YUV_UYVY)
    {
        if (src.channels() != 3)
        {
            return false;
        }

        dst.create(std::vector<int>{rows, cols}, CV_8UC2);
        cvtcolor_3ch_to_yuv422packed_u8(
            src.data,
            src_step,
            dst.data,
            dst.step(0),
            rows,
            cols,
            code == COLOR_RGB2YUV_YUY2 || code == COLOR_RGB2YUV_UYVY,
            code == COLOR_BGR2YUV_UYVY || code == COLOR_RGB2YUV_UYVY);
        return true;
    }

    if (code == COLOR_BGR2YUV_I444 ||
        code == COLOR_RGB2YUV_I444 ||
        code == COLOR_BGR2YUV_YV24 ||
        code == COLOR_RGB2YUV_YV24)
    {
        if (src.channels() != 3)
        {
            return false;
        }

        dst.create(std::vector<int>{rows * 3, cols}, CV_8UC1);
        cvtcolor_3ch_to_yuv444p_u8(
            src.data,
            src_step,
            dst.data,
            dst.step(0),
            rows,
            cols,
            code == COLOR_RGB2YUV_I444 || code == COLOR_RGB2YUV_YV24,
            code == COLOR_BGR2YUV_YV24 || code == COLOR_RGB2YUV_YV24);
        return true;
    }

    if (code == COLOR_YUV2BGR || code == COLOR_YUV2RGB)
    {
        if (src.channels() != 3)
        {
            return false;
        }

        dst.create(std::vector<int>{rows, cols}, CV_8UC3);
        cvtcolor_yuv_to_3ch<uchar>(src.data, src_step, dst.data, dst.step(0), rows, cols, code == COLOR_YUV2RGB);
        return true;
    }

    if (code == COLOR_YUV2BGR_NV12 ||
        code == COLOR_YUV2RGB_NV12 ||
        code == COLOR_YUV2BGR_NV21 ||
        code == COLOR_YUV2RGB_NV21)
    {
        if (src.channels() != 1)
        {
            return false;
        }

        const int y_rows = cvtcolor_validate_yuv420sp_layout_u8(src);
        dst.create(std::vector<int>{y_rows, cols}, CV_8UC3);
        cvtcolor_yuv420sp_to_3ch_u8(
            src.data,
            src_step,
            dst.data,
            dst.step(0),
            y_rows,
            cols,
            code == COLOR_YUV2BGR_NV21 || code == COLOR_YUV2RGB_NV21,
            code == COLOR_YUV2RGB_NV12 || code == COLOR_YUV2RGB_NV21);
        return true;
    }

    if (code == COLOR_YUV2BGR_NV16 ||
        code == COLOR_YUV2RGB_NV16 ||
        code == COLOR_YUV2BGR_NV61 ||
        code == COLOR_YUV2RGB_NV61)
    {
        if (src.channels() != 1)
        {
            return false;
        }

        const int y_rows = cvtcolor_validate_yuv422sp_layout_u8(src);
        dst.create(std::vector<int>{y_rows, cols}, CV_8UC3);
        cvtcolor_yuv422sp_to_3ch_u8(
            src.data,
            src_step,
            dst.data,
            dst.step(0),
            y_rows,
            cols,
            code == COLOR_YUV2BGR_NV61 || code == COLOR_YUV2RGB_NV61,
            code == COLOR_YUV2RGB_NV16 || code == COLOR_YUV2RGB_NV61);
        return true;
    }

    if (code == COLOR_YUV2BGR_NV24 ||
        code == COLOR_YUV2RGB_NV24 ||
        code == COLOR_YUV2BGR_NV42 ||
        code == COLOR_YUV2RGB_NV42)
    {
        if (src.channels() != 1)
        {
            return false;
        }

        const int y_rows = cvtcolor_validate_yuv444sp_layout_u8(src);
        dst.create(std::vector<int>{y_rows, cols}, CV_8UC3);
        cvtcolor_yuv444sp_to_3ch_u8(
            src.data,
            src_step,
            dst.data,
            dst.step(0),
            y_rows,
            cols,
            code == COLOR_YUV2BGR_NV42 || code == COLOR_YUV2RGB_NV42,
            code == COLOR_YUV2RGB_NV24 || code == COLOR_YUV2RGB_NV42);
        return true;
    }

    if (code == COLOR_YUV2BGR_I444 ||
        code == COLOR_YUV2RGB_I444 ||
        code == COLOR_YUV2BGR_YV24 ||
        code == COLOR_YUV2RGB_YV24)
    {
        if (src.channels() != 1)
        {
            return false;
        }

        const int y_rows = cvtcolor_validate_yuv444p_layout_u8(src);
        dst.create(std::vector<int>{y_rows, cols}, CV_8UC3);
        cvtcolor_yuv444p_to_3ch_u8(
            src.data,
            src_step,
            dst.data,
            dst.step(0),
            y_rows,
            cols,
            code == COLOR_YUV2BGR_YV24 || code == COLOR_YUV2RGB_YV24,
            code == COLOR_YUV2RGB_I444 || code == COLOR_YUV2RGB_YV24);
        return true;
    }

    if (code == COLOR_YUV2BGR_I420 ||
        code == COLOR_YUV2RGB_I420 ||
        code == COLOR_YUV2BGR_YV12 ||
        code == COLOR_YUV2RGB_YV12)
    {
        if (src.channels() != 1)
        {
            return false;
        }

        const int y_rows = cvtcolor_validate_yuv420sp_layout_u8(src);
        dst.create(std::vector<int>{y_rows, cols}, CV_8UC3);
        cvtcolor_yuv420p_to_3ch_u8(
            src.data,
            src_step,
            dst.data,
            dst.step(0),
            y_rows,
            cols,
            code == COLOR_YUV2BGR_YV12 || code == COLOR_YUV2RGB_YV12,
            code == COLOR_YUV2RGB_I420 || code == COLOR_YUV2RGB_YV12);
        return true;
    }

    if (code == COLOR_YUV2BGR_YUY2 ||
        code == COLOR_YUV2RGB_YUY2 ||
        code == COLOR_YUV2BGR_UYVY ||
        code == COLOR_YUV2RGB_UYVY)
    {
        if (src.channels() != 2)
        {
            return false;
        }

        cvtcolor_validate_yuv422packed_layout_u8(src);
        dst.create(std::vector<int>{rows, cols}, CV_8UC3);
        cvtcolor_yuv422packed_to_3ch_u8(
            src.data,
            src_step,
            dst.data,
            dst.step(0),
            rows,
            cols,
            code == COLOR_YUV2BGR_UYVY || code == COLOR_YUV2RGB_UYVY,
            code == COLOR_YUV2RGB_YUY2 || code == COLOR_YUV2RGB_UYVY);
        return true;
    }

    if (code == COLOR_BGRA2GRAY || code == COLOR_RGBA2GRAY)
    {
        if (src.channels() != 4)
        {
            return false;
        }

        dst.create(std::vector<int>{rows, cols}, CV_8UC1);
        cvtcolor_4ch_to_gray_u8(src.data, src_step, dst.data, dst.step(0), rows, cols, code == COLOR_RGBA2GRAY);
        return true;
    }

    if (code == COLOR_BGR2BGRA ||
        code == COLOR_RGB2RGBA ||
        code == COLOR_BGR2RGBA ||
        code == COLOR_RGB2BGRA)
    {
        const bool swap_rb = (code == COLOR_BGR2RGBA || code == COLOR_RGB2BGRA);
        if (src.channels() != 3)
        {
            return false;
        }

        dst.create(std::vector<int>{rows, cols}, CV_8UC4);
        cvtcolor_3ch_to_4ch_alpha<uchar>(src.data, src_step, dst.data, dst.step(0), rows, cols, static_cast<uchar>(255), swap_rb);
        return true;
    }

    if (code == COLOR_BGRA2BGR ||
        code == COLOR_RGBA2RGB ||
        code == COLOR_BGRA2RGB ||
        code == COLOR_RGBA2BGR)
    {
        const bool swap_rb = (code == COLOR_BGRA2RGB || code == COLOR_RGBA2BGR);
        if (src.channels() != 4)
        {
            return false;
        }

        dst.create(std::vector<int>{rows, cols}, CV_8UC3);
        cvtcolor_4ch_to_3ch_drop_alpha<uchar>(src.data, src_step, dst.data, dst.step(0), rows, cols, swap_rb);
        return true;
    }

    if (code == COLOR_BGRA2RGBA || code == COLOR_RGBA2BGRA)
    {
        if (src.channels() != 4)
        {
            return false;
        }

        dst.create(std::vector<int>{rows, cols}, CV_8UC4);
        cvtcolor_swap_rb_4ch<uchar>(src.data, src_step, dst.data, dst.step(0), rows, cols);
        return true;
    }

    return false;
}

bool try_cvtcolor_fastpath_f32(const Mat& src, Mat& dst, int code)
{
    if (src.empty() || src.dims != 2 || src.depth() != CV_32F)
    {
        return false;
    }

    const int rows = src.size[0];
    const int cols = src.size[1];
    const size_t src_step = src.step(0);

    if (code == COLOR_BGR2GRAY)
    {
        if (src.channels() != 3)
        {
            return false;
        }

        dst.create(std::vector<int>{rows, cols}, CV_32FC1);
        cvtcolor_bgr2gray_f32(src.data, src_step, dst.data, dst.step(0), rows, cols);
        return true;
    }

    if (code == COLOR_GRAY2BGR)
    {
        if (src.channels() != 1)
        {
            return false;
        }

        dst.create(std::vector<int>{rows, cols}, CV_32FC3);
        cvtcolor_gray2bgr_f32(src.data, src_step, dst.data, dst.step(0), rows, cols);
        return true;
    }

    if (code == COLOR_GRAY2BGRA || code == COLOR_GRAY2RGBA)
    {
        if (src.channels() != 1)
        {
            return false;
        }

        dst.create(std::vector<int>{rows, cols}, CV_32FC4);
        cvtcolor_gray_to_4ch_alpha<float>(src.data, src_step, dst.data, dst.step(0), rows, cols, 1.0f);
        return true;
    }

    if (code == COLOR_BGR2RGB || code == COLOR_RGB2BGR)
    {
        if (src.channels() != 3)
        {
            return false;
        }

        dst.create(std::vector<int>{rows, cols}, CV_32FC3);
        cvtcolor_swap_rb_3ch<float>(src.data, src_step, dst.data, dst.step(0), rows, cols);
        return true;
    }

    if (code == COLOR_BGR2YUV || code == COLOR_RGB2YUV)
    {
        if (src.channels() != 3)
        {
            return false;
        }

        dst.create(std::vector<int>{rows, cols}, CV_32FC3);
        cvtcolor_3ch_to_yuv<float>(src.data, src_step, dst.data, dst.step(0), rows, cols, code == COLOR_RGB2YUV);
        return true;
    }

    if (code == COLOR_YUV2BGR || code == COLOR_YUV2RGB)
    {
        if (src.channels() != 3)
        {
            return false;
        }

        dst.create(std::vector<int>{rows, cols}, CV_32FC3);
        cvtcolor_yuv_to_3ch<float>(src.data, src_step, dst.data, dst.step(0), rows, cols, code == COLOR_YUV2RGB);
        return true;
    }

    if (code == COLOR_BGRA2GRAY || code == COLOR_RGBA2GRAY)
    {
        if (src.channels() != 4)
        {
            return false;
        }

        dst.create(std::vector<int>{rows, cols}, CV_32FC1);
        cvtcolor_4ch_to_gray_f32(src.data, src_step, dst.data, dst.step(0), rows, cols, code == COLOR_RGBA2GRAY);
        return true;
    }

    if (code == COLOR_BGR2BGRA ||
        code == COLOR_RGB2RGBA ||
        code == COLOR_BGR2RGBA ||
        code == COLOR_RGB2BGRA)
    {
        const bool swap_rb = (code == COLOR_BGR2RGBA || code == COLOR_RGB2BGRA);
        if (src.channels() != 3)
        {
            return false;
        }

        dst.create(std::vector<int>{rows, cols}, CV_32FC4);
        cvtcolor_3ch_to_4ch_alpha<float>(src.data, src_step, dst.data, dst.step(0), rows, cols, 1.0f, swap_rb);
        return true;
    }

    if (code == COLOR_BGRA2BGR ||
        code == COLOR_RGBA2RGB ||
        code == COLOR_BGRA2RGB ||
        code == COLOR_RGBA2BGR)
    {
        const bool swap_rb = (code == COLOR_BGRA2RGB || code == COLOR_RGBA2BGR);
        if (src.channels() != 4)
        {
            return false;
        }

        dst.create(std::vector<int>{rows, cols}, CV_32FC3);
        cvtcolor_4ch_to_3ch_drop_alpha<float>(src.data, src_step, dst.data, dst.step(0), rows, cols, swap_rb);
        return true;
    }

    if (code == COLOR_BGRA2RGBA || code == COLOR_RGBA2BGRA)
    {
        if (src.channels() != 4)
        {
            return false;
        }

        dst.create(std::vector<int>{rows, cols}, CV_32FC4);
        cvtcolor_swap_rb_4ch<float>(src.data, src_step, dst.data, dst.step(0), rows, cols);
        return true;
    }

    return false;
}

bool try_threshold_fastpath_f32(const Mat& src, Mat& dst, double thresh, double maxval, int type, double* out_ret)
{
    if (out_ret == nullptr)
    {
        return false;
    }

    if (src.empty() || src.depth() != CV_32F)
    {
        return false;
    }

    const bool is_dryrun = (type & THRESH_DRYRUN) != 0;
    type &= ~THRESH_DRYRUN;

    const int automatic_thresh = type & (~THRESH_MASK);
    const int thresh_type = type & THRESH_MASK;

    if (automatic_thresh == THRESH_OTSU || automatic_thresh == THRESH_TRIANGLE)
    {
        CV_Error(Error::StsBadArg, "threshold: OTSU/TRIANGLE requires CV_8UC1 source");
    }

    if (automatic_thresh != 0)
    {
        CV_Error_(Error::StsBadArg, ("threshold: unsupported automatic threshold flag=%d", automatic_thresh));
    }

    if (thresh_type != THRESH_BINARY &&
        thresh_type != THRESH_BINARY_INV &&
        thresh_type != THRESH_TRUNC &&
        thresh_type != THRESH_TOZERO &&
        thresh_type != THRESH_TOZERO_INV)
    {
        CV_Error_(Error::StsBadArg, ("threshold: unsupported threshold type=%d", thresh_type));
    }

    const float thresh_f = static_cast<float>(thresh);
    const float max_f = static_cast<float>(maxval);
    *out_ret = thresh;

    if (is_dryrun)
    {
        return true;
    }

    dst.create(src.dims, src.size.p, src.type());

    const std::size_t scalar_count = src.total() * static_cast<std::size_t>(src.channels());
    const float* src_ptr = reinterpret_cast<const float*>(src.data);
    float* dst_ptr = reinterpret_cast<float*>(dst.data);

    if (src.isContinuous() && dst.isContinuous())
    {
        const bool do_parallel = should_parallelize_threshold_contiguous(scalar_count);
        parallel_for_count_if(do_parallel, scalar_count, [&](std::size_t i) {
            const float s = src_ptr[i];
            const bool cond = s > thresh_f;
            switch (thresh_type)
            {
            case THRESH_BINARY:
                dst_ptr[i] = cond ? max_f : 0.0f;
                break;
            case THRESH_BINARY_INV:
                dst_ptr[i] = cond ? 0.0f : max_f;
                break;
            case THRESH_TRUNC:
                dst_ptr[i] = cond ? thresh_f : s;
                break;
            case THRESH_TOZERO:
                dst_ptr[i] = cond ? s : 0.0f;
                break;
            case THRESH_TOZERO_INV:
                dst_ptr[i] = cond ? 0.0f : s;
                break;
            default:
                CV_Error_(Error::StsBadArg, ("threshold: unsupported threshold type=%d", thresh_type));
            }
        });
        return true;
    }

    CV_Assert(src.dims == 2 && "threshold: non-contiguous path supports 2D Mat only");
    const int rows = src.size[0];
    const int cols_scalar = src.size[1] * src.channels();
    const std::size_t src_step = src.step(0);
    const std::size_t dst_step = dst.step(0);
    const bool do_parallel = should_parallelize_threshold_rows(rows, cols_scalar);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const float* src_row = reinterpret_cast<const float*>(src.data + static_cast<std::size_t>(y) * src_step);
        float* dst_row = reinterpret_cast<float*>(dst.data + static_cast<std::size_t>(y) * dst_step);
        for (int x = 0; x < cols_scalar; ++x)
        {
            const float s = src_row[x];
            const bool cond = s > thresh_f;
            switch (thresh_type)
            {
            case THRESH_BINARY:
                dst_row[x] = cond ? max_f : 0.0f;
                break;
            case THRESH_BINARY_INV:
                dst_row[x] = cond ? 0.0f : max_f;
                break;
            case THRESH_TRUNC:
                dst_row[x] = cond ? thresh_f : s;
                break;
            case THRESH_TOZERO:
                dst_row[x] = cond ? s : 0.0f;
                break;
            case THRESH_TOZERO_INV:
                dst_row[x] = cond ? 0.0f : s;
                break;
            default:
                CV_Error_(Error::StsBadArg, ("threshold: unsupported threshold type=%d", thresh_type));
            }
        }
    });

    return true;
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

bool try_gaussian_blur_fastpath_u8(const Mat& src, Mat& dst, Size ksize, double sigmaX, double sigmaY, int borderType)
{
    if (src.empty() || src.dims != 2 || src.depth() != CV_8U)
    {
        return false;
    }

    if (!is_u8_fastpath_channels(src.channels()))
    {
        return false;
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

    if (kx <= 0 || ky <= 0 || (kx & 1) == 0 || (ky & 1) == 0)
    {
        return false;
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
    const int row_stride = cols * channels;

    dst.create(std::vector<int>{rows, cols}, src_ref->type());
    const std::size_t dst_step = dst.step(0);

    const std::vector<float> kernel_x = build_gaussian_kernel_1d(kx, sigmaX);
    const std::vector<float> kernel_y = build_gaussian_kernel_1d(ky, sigmaY);
    const int rx = kx / 2;
    const int ry = ky / 2;
    const bool has_constant_border = border_type == BORDER_CONSTANT;

    std::vector<int> x_offsets(static_cast<std::size_t>(cols) * static_cast<std::size_t>(kx), -1);
    for (int x = 0; x < cols; ++x)
    {
        int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kx);
        for (int i = 0; i < kx; ++i)
        {
            const int sx = border_interpolate(x + i - rx, cols, border_type);
            x_ofs[i] = sx >= 0 ? sx * channels : -1;
        }
    }

    std::vector<int> y_offsets(static_cast<std::size_t>(rows) * static_cast<std::size_t>(ky), -1);
    for (int y = 0; y < rows; ++y)
    {
        int* y_ofs = y_offsets.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(ky);
        for (int i = 0; i < ky; ++i)
        {
            const int sy = border_interpolate(y + i - ry, rows, border_type);
            y_ofs[i] = sy >= 0 ? sy * row_stride : -1;
        }
    }

    std::vector<float> tmp(static_cast<std::size_t>(rows) * static_cast<std::size_t>(row_stride), 0.0f);

    const bool do_parallel_h = should_parallelize_filter_rows(rows, cols, channels, kx);
    parallel_for_index_if(do_parallel_h, rows, [&](int y) {
        const uchar* src_row = src_ref->data + static_cast<std::size_t>(y) * src_step;
        float* tmp_row = tmp.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(row_stride);

        if (channels == 1)
        {
            for (int x = 0; x < cols; ++x)
            {
                const int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kx);
                float acc0 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        acc0 += kernel_x[static_cast<std::size_t>(i)] * static_cast<float>(src_row[sx]);
                    }
                }
                else
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        acc0 += kernel_x[static_cast<std::size_t>(i)] * static_cast<float>(src_row[sx]);
                    }
                }
                tmp_row[x] = acc0;
            }
            return;
        }

        if (channels == 3)
        {
            for (int x = 0; x < cols; ++x)
            {
                const int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kx);
                const int dx = x * 3;
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        const float w = kernel_x[static_cast<std::size_t>(i)];
                        const uchar* px = src_row + sx;
                        acc0 += w * static_cast<float>(px[0]);
                        acc1 += w * static_cast<float>(px[1]);
                        acc2 += w * static_cast<float>(px[2]);
                    }
                }
                else
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        const float w = kernel_x[static_cast<std::size_t>(i)];
                        const uchar* px = src_row + sx;
                        acc0 += w * static_cast<float>(px[0]);
                        acc1 += w * static_cast<float>(px[1]);
                        acc2 += w * static_cast<float>(px[2]);
                    }
                }
                tmp_row[dx + 0] = acc0;
                tmp_row[dx + 1] = acc1;
                tmp_row[dx + 2] = acc2;
            }
            return;
        }

        if (channels == 4)
        {
            for (int x = 0; x < cols; ++x)
            {
                const int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kx);
                const int dx = x * 4;
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        const float w = kernel_x[static_cast<std::size_t>(i)];
                        const uchar* px = src_row + sx;
                        acc0 += w * static_cast<float>(px[0]);
                        acc1 += w * static_cast<float>(px[1]);
                        acc2 += w * static_cast<float>(px[2]);
                        acc3 += w * static_cast<float>(px[3]);
                    }
                }
                else
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        const float w = kernel_x[static_cast<std::size_t>(i)];
                        const uchar* px = src_row + sx;
                        acc0 += w * static_cast<float>(px[0]);
                        acc1 += w * static_cast<float>(px[1]);
                        acc2 += w * static_cast<float>(px[2]);
                        acc3 += w * static_cast<float>(px[3]);
                    }
                }
                tmp_row[dx + 0] = acc0;
                tmp_row[dx + 1] = acc1;
                tmp_row[dx + 2] = acc2;
                tmp_row[dx + 3] = acc3;
            }
        }
    });

    const bool do_parallel_v = should_parallelize_filter_rows(rows, cols, channels, ky);
    parallel_for_index_if(do_parallel_v, rows, [&](int y) {
        uchar* dst_row = dst.data + static_cast<std::size_t>(y) * dst_step;
        const int* y_ofs = y_offsets.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(ky);

        if (channels == 1)
        {
            for (int x = 0; x < cols; ++x)
            {
                float acc0 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        if (sy < 0)
                        {
                            continue;
                        }
                        acc0 += kernel_y[static_cast<std::size_t>(i)] *
                                tmp[static_cast<std::size_t>(sy + x)];
                    }
                }
                else
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        acc0 += kernel_y[static_cast<std::size_t>(i)] *
                                tmp[static_cast<std::size_t>(sy + x)];
                    }
                }
                dst_row[x] = saturate_cast<uchar>(acc0);
            }
            return;
        }

        if (channels == 3)
        {
            for (int x = 0; x < cols; ++x)
            {
                const int dx = x * 3;
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        if (sy < 0)
                        {
                            continue;
                        }
                        const float w = kernel_y[static_cast<std::size_t>(i)];
                        const float* px = tmp.data() + static_cast<std::size_t>(sy + dx);
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                    }
                }
                else
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        const float w = kernel_y[static_cast<std::size_t>(i)];
                        const float* px = tmp.data() + static_cast<std::size_t>(sy + dx);
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                    }
                }
                dst_row[dx + 0] = saturate_cast<uchar>(acc0);
                dst_row[dx + 1] = saturate_cast<uchar>(acc1);
                dst_row[dx + 2] = saturate_cast<uchar>(acc2);
            }
            return;
        }

        if (channels == 4)
        {
            for (int x = 0; x < cols; ++x)
            {
                const int dx = x * 4;
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        if (sy < 0)
                        {
                            continue;
                        }
                        const float w = kernel_y[static_cast<std::size_t>(i)];
                        const float* px = tmp.data() + static_cast<std::size_t>(sy + dx);
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                        acc3 += w * px[3];
                    }
                }
                else
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        const float w = kernel_y[static_cast<std::size_t>(i)];
                        const float* px = tmp.data() + static_cast<std::size_t>(sy + dx);
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                        acc3 += w * px[3];
                    }
                }
                dst_row[dx + 0] = saturate_cast<uchar>(acc0);
                dst_row[dx + 1] = saturate_cast<uchar>(acc1);
                dst_row[dx + 2] = saturate_cast<uchar>(acc2);
                dst_row[dx + 3] = saturate_cast<uchar>(acc3);
            }
        }
    });

    return true;
}

} // namespace

const char* last_boxfilter_dispatch_path()
{
    return g_last_boxfilter_dispatch_path;
}

const char* last_gaussianblur_dispatch_path()
{
    return g_last_gaussianblur_dispatch_path;
}

void resize_backend_impl(const Mat& src, Mat& dst, Size dsize, double fx, double fy, int interpolation)
{
    if (try_resize_fastpath_u8(src, dst, dsize, fx, fy, interpolation))
    {
        return;
    }

    resize_fallback(src, dst, dsize, fx, fy, interpolation);
}

void cvtColor_backend_impl(const Mat& src, Mat& dst, int code)
{
    if (try_cvtcolor_fastpath_u8(src, dst, code))
    {
        return;
    }

    if (try_cvtcolor_fastpath_f32(src, dst, code))
    {
        return;
    }

    cvtColor_fallback(src, dst, code);
}

double threshold_backend_impl(const Mat& src, Mat& dst, double thresh, double maxval, int type)
{
    double ret_value = 0.0;
    if (try_threshold_fastpath_f32(src, dst, thresh, maxval, type, &ret_value))
    {
        return ret_value;
    }

    return threshold_fallback(src, dst, thresh, maxval, type);
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

    boxFilter_fallback(src, dst, ddepth, ksize, anchor, normalize, borderType);
}

void gaussianBlur_backend_impl(const Mat& src, Mat& dst, Size ksize, double sigmaX, double sigmaY, int borderType)
{
    set_last_gaussianblur_dispatch_path("fallback");
    if (try_gaussian_blur_fastpath_u8(src, dst, ksize, sigmaX, sigmaY, borderType))
    {
        int kx = 0;
        int ky = 0;
        if (resolve_gaussian_kernel_size(ksize, sigmaX, sigmaY, kx, ky) && kx == 3 && ky == 3)
        {
            set_last_gaussianblur_dispatch_path("gauss3x3");
        }
        else
        {
            set_last_gaussianblur_dispatch_path("gauss_separable");
        }
        return;
    }

    gaussian_blur_fallback(src, dst, ksize, sigmaX, sigmaY, borderType);
}

} // namespace detail

void register_all_backends()
{
    static bool initialized = []() {
        detail::register_resize_backend(&detail::resize_backend_impl);
        detail::register_cvtcolor_backend(&detail::cvtColor_backend_impl);
        detail::register_threshold_backend(&detail::threshold_backend_impl);
        detail::register_boxfilter_backend(&detail::boxFilter_backend_impl);
        detail::register_gaussianblur_backend(&detail::gaussianBlur_backend_impl);
        return true;
    }();
    (void)initialized;
}

} // namespace cvh
