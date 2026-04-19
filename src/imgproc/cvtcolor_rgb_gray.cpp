#include "fastpath_common.h"
#include "cvtcolor_internal.h"

namespace cvh
{
namespace detail
{

namespace
{
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


} // namespace

bool try_cvtcolor_fastpath_u8_rgb_gray(const Mat& src, Mat& dst, int code)
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

bool try_cvtcolor_fastpath_f32_rgb_gray(const Mat& src, Mat& dst, int code)
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

} // namespace detail
} // namespace cvh
