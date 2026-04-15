#ifndef CVH_IMGPROC_CVTCOLOR_H
#define CVH_IMGPROC_CVTCOLOR_H

#include "detail/common.h"

#include <type_traits>
#include <vector>

namespace cvh {
namespace detail {

using CvtColorFn = void (*)(const Mat&, Mat&, int);

template <typename T>
inline void cvtcolor_bgr2gray_fallback_impl(const Mat& src, Mat& dst)
{
    CV_Assert(src.channels() == 3 && "cvtColor(BGR2GRAY): source must have 3 channels");

    const int rows = src.size[0];
    const int cols = src.size[1];
    const size_t src_step = src.step(0);

    const int dst_type = std::is_same_v<T, uchar> ? CV_8UC1 : CV_32FC1;
    dst.create(std::vector<int>{rows, cols}, dst_type);
    const size_t dst_step = dst.step(0);

    if constexpr (std::is_same_v<T, uchar>)
    {
        constexpr int kB = 7471;
        constexpr int kG = 38470;
        constexpr int kR = 19595;
        constexpr int kRound = 1 << 15;

        for (int y = 0; y < rows; ++y)
        {
            const uchar* src_row = src.data + static_cast<size_t>(y) * src_step;
            uchar* dst_row = dst.data + static_cast<size_t>(y) * dst_step;
            for (int x = 0; x < cols; ++x)
            {
                const uchar* px = src_row + static_cast<size_t>(x) * 3;
                dst_row[x] = static_cast<uchar>(
                    (kB * static_cast<int>(px[0]) +
                     kG * static_cast<int>(px[1]) +
                     kR * static_cast<int>(px[2]) +
                     kRound) >> 16);
            }
        }
        return;
    }

    for (int y = 0; y < rows; ++y)
    {
        const T* src_row = reinterpret_cast<const T*>(src.data + static_cast<size_t>(y) * src_step);
        T* dst_row = reinterpret_cast<T*>(dst.data + static_cast<size_t>(y) * dst_step);
        for (int x = 0; x < cols; ++x)
        {
            const int sx = x * 3;
            dst_row[x] = static_cast<T>(
                0.114f * static_cast<float>(src_row[sx + 0]) +
                0.587f * static_cast<float>(src_row[sx + 1]) +
                0.299f * static_cast<float>(src_row[sx + 2]));
        }
    }
}

template <typename T>
inline void cvtcolor_gray2bgr_fallback_impl(const Mat& src, Mat& dst)
{
    CV_Assert(src.channels() == 1 && "cvtColor(GRAY2BGR): source must have 1 channel");

    const int rows = src.size[0];
    const int cols = src.size[1];
    const size_t src_step = src.step(0);
    const int dst_type = std::is_same_v<T, uchar> ? CV_8UC3 : CV_32FC3;

    dst.create(std::vector<int>{rows, cols}, dst_type);
    const size_t dst_step = dst.step(0);
    for (int y = 0; y < rows; ++y)
    {
        const T* src_row = reinterpret_cast<const T*>(src.data + static_cast<size_t>(y) * src_step);
        T* dst_row = reinterpret_cast<T*>(dst.data + static_cast<size_t>(y) * dst_step);
        for (int x = 0; x < cols; ++x)
        {
            const T g = src_row[x];
            const int dx = x * 3;
            dst_row[dx + 0] = g;
            dst_row[dx + 1] = g;
            dst_row[dx + 2] = g;
        }
    }
}

template <typename T>
inline void cvtcolor_gray2bgra_fallback_impl(const Mat& src, Mat& dst)
{
    CV_Assert(src.channels() == 1 && "cvtColor(GRAY2BGRA/RGBA): source must have 1 channel");

    const int rows = src.size[0];
    const int cols = src.size[1];
    const size_t src_step = src.step(0);
    const int dst_type = std::is_same_v<T, uchar> ? CV_8UC4 : CV_32FC4;
    const T alpha = std::is_same_v<T, uchar> ? static_cast<T>(255) : static_cast<T>(1.0f);

    dst.create(std::vector<int>{rows, cols}, dst_type);
    const size_t dst_step = dst.step(0);
    for (int y = 0; y < rows; ++y)
    {
        const T* src_row = reinterpret_cast<const T*>(src.data + static_cast<size_t>(y) * src_step);
        T* dst_row = reinterpret_cast<T*>(dst.data + static_cast<size_t>(y) * dst_step);
        for (int x = 0; x < cols; ++x)
        {
            const T g = src_row[x];
            const int dx = x * 4;
            dst_row[dx + 0] = g;
            dst_row[dx + 1] = g;
            dst_row[dx + 2] = g;
            dst_row[dx + 3] = alpha;
        }
    }
}

template <typename T>
inline void cvtcolor_swap_rb_fallback_impl(const Mat& src, Mat& dst)
{
    CV_Assert(src.channels() == 3 && "cvtColor(BGR<->RGB): source must have 3 channels");

    const int rows = src.size[0];
    const int cols = src.size[1];
    const size_t src_step = src.step(0);
    const int dst_type = std::is_same_v<T, uchar> ? CV_8UC3 : CV_32FC3;

    dst.create(std::vector<int>{rows, cols}, dst_type);
    const size_t dst_step = dst.step(0);
    for (int y = 0; y < rows; ++y)
    {
        const T* src_row = reinterpret_cast<const T*>(src.data + static_cast<size_t>(y) * src_step);
        T* dst_row = reinterpret_cast<T*>(dst.data + static_cast<size_t>(y) * dst_step);
        for (int x = 0; x < cols; ++x)
        {
            const int sx = x * 3;
            dst_row[sx + 0] = src_row[sx + 2];
            dst_row[sx + 1] = src_row[sx + 1];
            dst_row[sx + 2] = src_row[sx + 0];
        }
    }
}

template <typename T>
inline void cvtcolor_4ch2gray_fallback_impl(const Mat& src, Mat& dst, bool rgba_order)
{
    CV_Assert(src.channels() == 4 && "cvtColor(BGRA/RGBA2GRAY): source must have 4 channels");

    const int rows = src.size[0];
    const int cols = src.size[1];
    const size_t src_step = src.step(0);
    const int dst_type = std::is_same_v<T, uchar> ? CV_8UC1 : CV_32FC1;

    dst.create(std::vector<int>{rows, cols}, dst_type);
    const size_t dst_step = dst.step(0);

    if constexpr (std::is_same_v<T, uchar>)
    {
        constexpr int kB = 7471;
        constexpr int kG = 38470;
        constexpr int kR = 19595;
        constexpr int kRound = 1 << 15;

        for (int y = 0; y < rows; ++y)
        {
            const uchar* src_row = src.data + static_cast<size_t>(y) * src_step;
            uchar* dst_row = dst.data + static_cast<size_t>(y) * dst_step;
            for (int x = 0; x < cols; ++x)
            {
                const int sx = x * 4;
                const int b = src_row[sx + (rgba_order ? 2 : 0)];
                const int g = src_row[sx + 1];
                const int r = src_row[sx + (rgba_order ? 0 : 2)];
                dst_row[x] = static_cast<uchar>((kB * b + kG * g + kR * r + kRound) >> 16);
            }
        }
        return;
    }

    for (int y = 0; y < rows; ++y)
    {
        const T* src_row = reinterpret_cast<const T*>(src.data + static_cast<size_t>(y) * src_step);
        T* dst_row = reinterpret_cast<T*>(dst.data + static_cast<size_t>(y) * dst_step);
        for (int x = 0; x < cols; ++x)
        {
            const int sx = x * 4;
            const float b = static_cast<float>(src_row[sx + (rgba_order ? 2 : 0)]);
            const float g = static_cast<float>(src_row[sx + 1]);
            const float r = static_cast<float>(src_row[sx + (rgba_order ? 0 : 2)]);
            dst_row[x] = static_cast<T>(0.114f * b + 0.587f * g + 0.299f * r);
        }
    }
}

template <typename T>
inline void cvtcolor_3ch_to_4ch_alpha_fallback_impl(const Mat& src, Mat& dst, bool swap_rb)
{
    CV_Assert(src.channels() == 3 && "cvtColor(3ch->4ch): source must have 3 channels");

    const int rows = src.size[0];
    const int cols = src.size[1];
    const size_t src_step = src.step(0);
    const int dst_type = std::is_same_v<T, uchar> ? CV_8UC4 : CV_32FC4;
    const T alpha = std::is_same_v<T, uchar> ? static_cast<T>(255) : static_cast<T>(1.0f);

    dst.create(std::vector<int>{rows, cols}, dst_type);
    const size_t dst_step = dst.step(0);
    for (int y = 0; y < rows; ++y)
    {
        const T* src_row = reinterpret_cast<const T*>(src.data + static_cast<size_t>(y) * src_step);
        T* dst_row = reinterpret_cast<T*>(dst.data + static_cast<size_t>(y) * dst_step);
        for (int x = 0; x < cols; ++x)
        {
            const int sx = x * 3;
            const int dx = x * 4;
            dst_row[dx + 0] = src_row[sx + (swap_rb ? 2 : 0)];
            dst_row[dx + 1] = src_row[sx + 1];
            dst_row[dx + 2] = src_row[sx + (swap_rb ? 0 : 2)];
            dst_row[dx + 3] = alpha;
        }
    }
}

template <typename T>
inline void cvtcolor_4ch_to_3ch_drop_alpha_fallback_impl(const Mat& src, Mat& dst, bool swap_rb)
{
    CV_Assert(src.channels() == 4 && "cvtColor(4ch->3ch): source must have 4 channels");

    const int rows = src.size[0];
    const int cols = src.size[1];
    const size_t src_step = src.step(0);
    const int dst_type = std::is_same_v<T, uchar> ? CV_8UC3 : CV_32FC3;

    dst.create(std::vector<int>{rows, cols}, dst_type);
    const size_t dst_step = dst.step(0);
    for (int y = 0; y < rows; ++y)
    {
        const T* src_row = reinterpret_cast<const T*>(src.data + static_cast<size_t>(y) * src_step);
        T* dst_row = reinterpret_cast<T*>(dst.data + static_cast<size_t>(y) * dst_step);
        for (int x = 0; x < cols; ++x)
        {
            const int sx = x * 4;
            const int dx = x * 3;
            dst_row[dx + 0] = src_row[sx + (swap_rb ? 2 : 0)];
            dst_row[dx + 1] = src_row[sx + 1];
            dst_row[dx + 2] = src_row[sx + (swap_rb ? 0 : 2)];
        }
    }
}

template <typename T>
inline void cvtcolor_swap_rb_4ch_fallback_impl(const Mat& src, Mat& dst)
{
    CV_Assert(src.channels() == 4 && "cvtColor(BGRA<->RGBA): source must have 4 channels");

    const int rows = src.size[0];
    const int cols = src.size[1];
    const size_t src_step = src.step(0);
    const int dst_type = std::is_same_v<T, uchar> ? CV_8UC4 : CV_32FC4;

    dst.create(std::vector<int>{rows, cols}, dst_type);
    const size_t dst_step = dst.step(0);
    for (int y = 0; y < rows; ++y)
    {
        const T* src_row = reinterpret_cast<const T*>(src.data + static_cast<size_t>(y) * src_step);
        T* dst_row = reinterpret_cast<T*>(dst.data + static_cast<size_t>(y) * dst_step);
        for (int x = 0; x < cols; ++x)
        {
            const int sx = x * 4;
            dst_row[sx + 0] = src_row[sx + 2];
            dst_row[sx + 1] = src_row[sx + 1];
            dst_row[sx + 2] = src_row[sx + 0];
            dst_row[sx + 3] = src_row[sx + 3];
        }
    }
}

template <typename T>
inline void cvtcolor_3ch_to_yuv_fallback_impl(const Mat& src, Mat& dst, bool rgb_order)
{
    CV_Assert(src.channels() == 3 && "cvtColor(BGR/RGB2YUV): source must have 3 channels");

    const int rows = src.size[0];
    const int cols = src.size[1];
    const size_t src_step = src.step(0);
    const int dst_type = std::is_same_v<T, uchar> ? CV_8UC3 : CV_32FC3;
    const float delta = std::is_same_v<T, uchar> ? 128.0f : 0.5f;

    dst.create(std::vector<int>{rows, cols}, dst_type);
    const size_t dst_step = dst.step(0);
    for (int y = 0; y < rows; ++y)
    {
        const T* src_row = reinterpret_cast<const T*>(src.data + static_cast<size_t>(y) * src_step);
        T* dst_row = reinterpret_cast<T*>(dst.data + static_cast<size_t>(y) * dst_step);
        for (int x = 0; x < cols; ++x)
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
                dst_row[sx + 0] = static_cast<float>(yy);
                dst_row[sx + 1] = static_cast<float>(uu);
                dst_row[sx + 2] = static_cast<float>(vv);
            }
        }
    }
}

template <typename T>
inline void cvtcolor_yuv_to_3ch_fallback_impl(const Mat& src, Mat& dst, bool rgb_order)
{
    CV_Assert(src.channels() == 3 && "cvtColor(YUV2BGR/RGB): source must have 3 channels");

    const int rows = src.size[0];
    const int cols = src.size[1];
    const size_t src_step = src.step(0);
    const int dst_type = std::is_same_v<T, uchar> ? CV_8UC3 : CV_32FC3;
    const float delta = std::is_same_v<T, uchar> ? 128.0f : 0.5f;

    dst.create(std::vector<int>{rows, cols}, dst_type);
    const size_t dst_step = dst.step(0);
    for (int y = 0; y < rows; ++y)
    {
        const T* src_row = reinterpret_cast<const T*>(src.data + static_cast<size_t>(y) * src_step);
        T* dst_row = reinterpret_cast<T*>(dst.data + static_cast<size_t>(y) * dst_step);
        for (int x = 0; x < cols; ++x)
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
                dst_row[sx + (rgb_order ? 0 : 2)] = static_cast<float>(r);
                dst_row[sx + 1] = static_cast<float>(g);
                dst_row[sx + (rgb_order ? 2 : 0)] = static_cast<float>(b);
            }
        }
    }
}

inline int cvtcolor_validate_yuv420sp_layout_u8(const Mat& src)
{
    CV_Assert(src.depth() == CV_8U && "cvtColor(YUV420sp): source must be CV_8UC1");
    CV_Assert(src.channels() == 1 && "cvtColor(YUV420sp): source must have 1 channel");
    CV_Assert((src.size[0] % 3) == 0 && "cvtColor(YUV420sp): source rows must equal H*3/2");
    CV_Assert((src.size[1] % 2) == 0 && "cvtColor(YUV420sp): source width must be even");
    return src.size[0] * 2 / 3;
}

inline int cvtcolor_validate_yuv422sp_layout_u8(const Mat& src)
{
    CV_Assert(src.depth() == CV_8U && "cvtColor(YUV422sp): source must be CV_8UC1");
    CV_Assert(src.channels() == 1 && "cvtColor(YUV422sp): source must have 1 channel");
    CV_Assert((src.size[0] % 2) == 0 && "cvtColor(YUV422sp): source rows must equal H*2");
    CV_Assert((src.size[1] % 2) == 0 && "cvtColor(YUV422sp): source width must be even");
    return src.size[0] / 2;
}

inline int cvtcolor_validate_yuv444sp_layout_u8(const Mat& src)
{
    CV_Assert(src.depth() == CV_8U && "cvtColor(YUV444sp): source must be CV_8UC1");
    CV_Assert(src.channels() == 1 && "cvtColor(YUV444sp): source must have 1 channel");
    CV_Assert((src.size[0] % 3) == 0 && "cvtColor(YUV444sp): source rows must equal H*3");
    return src.size[0] / 3;
}

inline int cvtcolor_validate_yuv444p_layout_u8(const Mat& src)
{
    CV_Assert(src.depth() == CV_8U && "cvtColor(YUV444p): source must be CV_8UC1");
    CV_Assert(src.channels() == 1 && "cvtColor(YUV444p): source must have 1 channel");
    CV_Assert((src.size[0] % 3) == 0 && "cvtColor(YUV444p): source rows must equal H*3");
    return src.size[0] / 3;
}

inline uchar cvtcolor_yuv420sp_channel_u8(int yy, int uu, int vv, int channel)
{
    const int c = std::max(yy - 16, 0);
    const int d = uu - 128;
    const int e = vv - 128;

    if (channel == 0)
    {
        return saturate_cast<uchar>((298 * c + 516 * d + 128) >> 8);
    }
    if (channel == 1)
    {
        return saturate_cast<uchar>((298 * c - 100 * d - 208 * e + 128) >> 8);
    }
    return saturate_cast<uchar>((298 * c + 409 * e + 128) >> 8);
}

inline uchar cvtcolor_color3_to_yuv_limited_u8(int bb, int gg, int rr, int channel)
{
    if (channel == 0)
    {
        return saturate_cast<uchar>(((66 * rr + 129 * gg + 25 * bb + 128) >> 8) + 16);
    }
    if (channel == 1)
    {
        return saturate_cast<uchar>(((-38 * rr - 74 * gg + 112 * bb + 128) >> 8) + 128);
    }
    return saturate_cast<uchar>(((112 * rr - 94 * gg - 18 * bb + 128) >> 8) + 128);
}

inline void cvtcolor_validate_yuv422packed_layout_u8(const Mat& src)
{
    CV_Assert(src.depth() == CV_8U && "cvtColor(YUV422packed): source must be CV_8UC2");
    CV_Assert(src.channels() == 2 && "cvtColor(YUV422packed): source must have 2 channels");
    CV_Assert((src.size[1] % 2) == 0 && "cvtColor(YUV422packed): source width must be even");
}

inline void cvtcolor_yuv420sp_to_3ch_fallback_impl(const Mat& src, Mat& dst, bool nv21_layout, bool rgb_order)
{
    const int rows = cvtcolor_validate_yuv420sp_layout_u8(src);
    const int cols = src.size[1];
    const size_t src_step = src.step(0);

    dst.create(std::vector<int>{rows, cols}, CV_8UC3);
    const size_t dst_step = dst.step(0);

    for (int y = 0; y < rows; ++y)
    {
        const uchar* y_row = src.data + static_cast<size_t>(y) * src_step;
        const uchar* uv_row = src.data + static_cast<size_t>(rows + y / 2) * src_step;
        uchar* dst_row = dst.data + static_cast<size_t>(y) * dst_step;
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
    }
}

inline void cvtcolor_3ch_to_yuv420sp_fallback_impl(const Mat& src, Mat& dst, bool rgb_order, bool nv21_layout)
{
    CV_Assert(src.depth() == CV_8U && "cvtColor(BGR/RGB2YUV420sp): source must be CV_8UC3");
    CV_Assert(src.channels() == 3 && "cvtColor(BGR/RGB2YUV420sp): source must have 3 channels");
    CV_Assert((src.size[0] % 2) == 0 && "cvtColor(BGR/RGB2YUV420sp): source height must be even");
    CV_Assert((src.size[1] % 2) == 0 && "cvtColor(BGR/RGB2YUV420sp): source width must be even");

    const int rows = src.size[0];
    const int cols = src.size[1];
    const size_t src_step = src.step(0);

    dst.create(std::vector<int>{rows * 3 / 2, cols}, CV_8UC1);
    const size_t dst_step = dst.step(0);

    for (int y = 0; y < rows; y += 2)
    {
        const uchar* src_row0 = src.data + static_cast<size_t>(y + 0) * src_step;
        const uchar* src_row1 = src.data + static_cast<size_t>(y + 1) * src_step;
        uchar* dst_y_row0 = dst.data + static_cast<size_t>(y + 0) * dst_step;
        uchar* dst_y_row1 = dst.data + static_cast<size_t>(y + 1) * dst_step;
        uchar* dst_uv_row = dst.data + static_cast<size_t>(rows + y / 2) * dst_step;

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
    }
}

inline void cvtcolor_3ch_to_yuv420p_fallback_impl(const Mat& src, Mat& dst, bool rgb_order, bool yv12_layout)
{
    CV_Assert(src.depth() == CV_8U && "cvtColor(BGR/RGB2YUV420p): source must be CV_8UC3");
    CV_Assert(src.channels() == 3 && "cvtColor(BGR/RGB2YUV420p): source must have 3 channels");
    CV_Assert((src.size[0] % 2) == 0 && "cvtColor(BGR/RGB2YUV420p): source height must be even");
    CV_Assert((src.size[1] % 2) == 0 && "cvtColor(BGR/RGB2YUV420p): source width must be even");

    const int rows = src.size[0];
    const int cols = src.size[1];
    const int uv_size = rows * cols / 4;
    const int u_plane_offset = yv12_layout ? uv_size : 0;
    const int v_plane_offset = yv12_layout ? 0 : uv_size;
    const size_t src_step = src.step(0);

    dst.create(std::vector<int>{rows * 3 / 2, cols}, CV_8UC1);
    const size_t dst_step = dst.step(0);

    for (int y = 0; y < rows; y += 2)
    {
        const uchar* src_row0 = src.data + static_cast<size_t>(y + 0) * src_step;
        const uchar* src_row1 = src.data + static_cast<size_t>(y + 1) * src_step;
        uchar* dst_y_row0 = dst.data + static_cast<size_t>(y + 0) * dst_step;
        uchar* dst_y_row1 = dst.data + static_cast<size_t>(y + 1) * dst_step;

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

            *(dst.data +
              static_cast<size_t>(rows + (u_plane_offset + chroma_index) / cols) * dst_step +
              static_cast<size_t>((u_plane_offset + chroma_index) % cols)) = uu;
            *(dst.data +
              static_cast<size_t>(rows + (v_plane_offset + chroma_index) / cols) * dst_step +
              static_cast<size_t>((v_plane_offset + chroma_index) % cols)) = vv;
        }
    }
}

inline uchar cvtcolor_yuv444sp_plane_byte_u8(const Mat& src, int rows, int cols, int plane_index)
{
    return *(src.data +
             static_cast<size_t>(rows + plane_index / cols) * src.step(0) +
             static_cast<size_t>(plane_index % cols));
}

inline void cvtcolor_yuv422packed_to_3ch_fallback_impl(const Mat& src, Mat& dst, bool uyvy_layout, bool rgb_order)
{
    cvtcolor_validate_yuv422packed_layout_u8(src);
    const int rows = src.size[0];
    const int cols = src.size[1];
    const size_t src_step = src.step(0);

    dst.create(std::vector<int>{rows, cols}, CV_8UC3);
    const size_t dst_step = dst.step(0);

    for (int y = 0; y < rows; ++y)
    {
        const uchar* src_row = src.data + static_cast<size_t>(y) * src_step;
        uchar* dst_row = dst.data + static_cast<size_t>(y) * dst_step;

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
    }
}

inline void cvtcolor_yuv422sp_to_3ch_fallback_impl(const Mat& src, Mat& dst, bool nv61_layout, bool rgb_order)
{
    const int rows = cvtcolor_validate_yuv422sp_layout_u8(src);
    const int cols = src.size[1];
    const size_t src_step = src.step(0);

    dst.create(std::vector<int>{rows, cols}, CV_8UC3);
    const size_t dst_step = dst.step(0);

    for (int y = 0; y < rows; ++y)
    {
        const uchar* y_row = src.data + static_cast<size_t>(y) * src_step;
        const uchar* uv_row = src.data + static_cast<size_t>(rows + y) * src_step;
        uchar* dst_row = dst.data + static_cast<size_t>(y) * dst_step;
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
    }
}

inline void cvtcolor_3ch_to_yuv422sp_fallback_impl(const Mat& src, Mat& dst, bool rgb_order, bool nv61_layout)
{
    CV_Assert(src.depth() == CV_8U && "cvtColor(BGR/RGB2YUV422sp): source must be CV_8UC3");
    CV_Assert(src.channels() == 3 && "cvtColor(BGR/RGB2YUV422sp): source must have 3 channels");
    CV_Assert((src.size[1] % 2) == 0 && "cvtColor(BGR/RGB2YUV422sp): source width must be even");

    const int rows = src.size[0];
    const int cols = src.size[1];
    const size_t src_step = src.step(0);

    dst.create(std::vector<int>{rows * 2, cols}, CV_8UC1);
    const size_t dst_step = dst.step(0);

    for (int y = 0; y < rows; ++y)
    {
        const uchar* src_row = src.data + static_cast<size_t>(y) * src_step;
        uchar* dst_y_row = dst.data + static_cast<size_t>(y) * dst_step;

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
            uchar* dst_uv_row = dst.data + static_cast<size_t>(rows + y) * dst_step;

            dst_uv_row[x + 0] = nv61_layout ? vv : uu;
            dst_uv_row[x + 1] = nv61_layout ? uu : vv;
        }
    }
}

inline void cvtcolor_3ch_to_yuv422packed_fallback_impl(const Mat& src, Mat& dst, bool rgb_order, bool uyvy_layout)
{
    CV_Assert(src.depth() == CV_8U && "cvtColor(BGR/RGB2YUV422packed): source must be CV_8UC3");
    CV_Assert(src.channels() == 3 && "cvtColor(BGR/RGB2YUV422packed): source must have 3 channels");
    CV_Assert((src.size[1] % 2) == 0 && "cvtColor(BGR/RGB2YUV422packed): source width must be even");

    const int rows = src.size[0];
    const int cols = src.size[1];
    const size_t src_step = src.step(0);

    dst.create(std::vector<int>{rows, cols}, CV_8UC2);
    const size_t dst_step = dst.step(0);

    for (int y = 0; y < rows; ++y)
    {
        const uchar* src_row = src.data + static_cast<size_t>(y) * src_step;
        uchar* dst_row = dst.data + static_cast<size_t>(y) * dst_step;

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
    }
}

inline void cvtcolor_yuv444sp_to_3ch_fallback_impl(const Mat& src, Mat& dst, bool nv42_layout, bool rgb_order)
{
    const int rows = cvtcolor_validate_yuv444sp_layout_u8(src);
    const int cols = src.size[1];
    const size_t src_step = src.step(0);

    dst.create(std::vector<int>{rows, cols}, CV_8UC3);
    const size_t dst_step = dst.step(0);

    for (int y = 0; y < rows; ++y)
    {
        const uchar* y_row = src.data + static_cast<size_t>(y) * src_step;
        uchar* dst_row = dst.data + static_cast<size_t>(y) * dst_step;
        for (int x = 0; x < cols; ++x)
        {
            const int yy = static_cast<int>(y_row[x]);
            const int base = y * (cols * 2) + x * 2;
            const int uu = static_cast<int>(cvtcolor_yuv444sp_plane_byte_u8(src, rows, cols, base + (nv42_layout ? 1 : 0)));
            const int vv = static_cast<int>(cvtcolor_yuv444sp_plane_byte_u8(src, rows, cols, base + (nv42_layout ? 0 : 1)));
            const uchar b = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 0);
            const uchar g = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 1);
            const uchar r = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 2);

            const int dx = x * 3;
            dst_row[dx + (rgb_order ? 0 : 2)] = r;
            dst_row[dx + 1] = g;
            dst_row[dx + (rgb_order ? 2 : 0)] = b;
        }
    }
}

inline void cvtcolor_3ch_to_yuv444sp_fallback_impl(const Mat& src, Mat& dst, bool rgb_order, bool nv42_layout)
{
    CV_Assert(src.depth() == CV_8U && "cvtColor(BGR/RGB2YUV444sp): source must be CV_8UC3");
    CV_Assert(src.channels() == 3 && "cvtColor(BGR/RGB2YUV444sp): source must have 3 channels");

    const int rows = src.size[0];
    const int cols = src.size[1];
    const size_t src_step = src.step(0);

    dst.create(std::vector<int>{rows * 3, cols}, CV_8UC1);
    const size_t dst_step = dst.step(0);

    for (int y = 0; y < rows; ++y)
    {
        const uchar* src_row = src.data + static_cast<size_t>(y) * src_step;
        uchar* dst_y_row = dst.data + static_cast<size_t>(y) * dst_step;
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
            *(dst.data +
              static_cast<size_t>(rows + (base + 0) / cols) * dst_step +
              static_cast<size_t>((base + 0) % cols)) = nv42_layout ? vv : uu;
            *(dst.data +
              static_cast<size_t>(rows + (base + 1) / cols) * dst_step +
              static_cast<size_t>((base + 1) % cols)) = nv42_layout ? uu : vv;
        }
    }
}

inline void cvtcolor_3ch_to_yuv444p_fallback_impl(const Mat& src, Mat& dst, bool rgb_order, bool yv24_layout)
{
    CV_Assert(src.depth() == CV_8U && "cvtColor(BGR/RGB2YUV444p): source must be CV_8UC3");
    CV_Assert(src.channels() == 3 && "cvtColor(BGR/RGB2YUV444p): source must have 3 channels");

    const int rows = src.size[0];
    const int cols = src.size[1];
    const int plane_size = rows * cols;
    const int u_plane_offset = yv24_layout ? plane_size : 0;
    const int v_plane_offset = yv24_layout ? 0 : plane_size;
    const size_t src_step = src.step(0);

    dst.create(std::vector<int>{rows * 3, cols}, CV_8UC1);
    const size_t dst_step = dst.step(0);

    for (int y = 0; y < rows; ++y)
    {
        const uchar* src_row = src.data + static_cast<size_t>(y) * src_step;
        uchar* dst_y_row = dst.data + static_cast<size_t>(y) * dst_step;
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
            *(dst.data +
              static_cast<size_t>(rows + (u_plane_offset + chroma_index) / cols) * dst_step +
              static_cast<size_t>((u_plane_offset + chroma_index) % cols)) = uu;
            *(dst.data +
              static_cast<size_t>(rows + (v_plane_offset + chroma_index) / cols) * dst_step +
              static_cast<size_t>((v_plane_offset + chroma_index) % cols)) = vv;
        }
    }
}

inline uchar cvtcolor_yuv420p_plane_byte_u8(const Mat& src, int rows, int cols, int plane_offset, int plane_index)
{
    const int logical_offset = plane_offset + plane_index;
    return *(src.data +
             static_cast<size_t>(rows + logical_offset / cols) * src.step(0) +
             static_cast<size_t>(logical_offset % cols));
}

inline uchar cvtcolor_yuv444p_plane_byte_u8(const Mat& src, int rows, int cols, int plane_offset, int plane_index)
{
    const int logical_offset = plane_offset + plane_index;
    return *(src.data +
             static_cast<size_t>(rows + logical_offset / cols) * src.step(0) +
             static_cast<size_t>(logical_offset % cols));
}

inline void cvtcolor_yuv420p_to_3ch_fallback_impl(const Mat& src, Mat& dst, bool yv12_layout, bool rgb_order)
{
    const int rows = cvtcolor_validate_yuv420sp_layout_u8(src);
    const int cols = src.size[1];
    const int uv_size = rows * cols / 4;
    const int u_plane_offset = yv12_layout ? uv_size : 0;
    const int v_plane_offset = yv12_layout ? 0 : uv_size;
    const size_t src_step = src.step(0);

    dst.create(std::vector<int>{rows, cols}, CV_8UC3);
    const size_t dst_step = dst.step(0);

    for (int y = 0; y < rows; ++y)
    {
        const uchar* y_row = src.data + static_cast<size_t>(y) * src_step;
        uchar* dst_row = dst.data + static_cast<size_t>(y) * dst_step;
        for (int x = 0; x < cols; ++x)
        {
            const int yy = static_cast<int>(y_row[x]);
            const int chroma_index = (y / 2) * (cols / 2) + (x / 2);
            const int uu = static_cast<int>(cvtcolor_yuv420p_plane_byte_u8(src, rows, cols, u_plane_offset, chroma_index));
            const int vv = static_cast<int>(cvtcolor_yuv420p_plane_byte_u8(src, rows, cols, v_plane_offset, chroma_index));
            const uchar b = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 0);
            const uchar g = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 1);
            const uchar r = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 2);

            const int dx = x * 3;
            dst_row[dx + (rgb_order ? 0 : 2)] = r;
            dst_row[dx + 1] = g;
            dst_row[dx + (rgb_order ? 2 : 0)] = b;
        }
    }
}

inline void cvtcolor_yuv444p_to_3ch_fallback_impl(const Mat& src, Mat& dst, bool yv24_layout, bool rgb_order)
{
    const int rows = cvtcolor_validate_yuv444p_layout_u8(src);
    const int cols = src.size[1];
    const int plane_size = rows * cols;
    const int u_plane_offset = yv24_layout ? plane_size : 0;
    const int v_plane_offset = yv24_layout ? 0 : plane_size;
    const size_t src_step = src.step(0);

    dst.create(std::vector<int>{rows, cols}, CV_8UC3);
    const size_t dst_step = dst.step(0);

    for (int y = 0; y < rows; ++y)
    {
        const uchar* y_row = src.data + static_cast<size_t>(y) * src_step;
        uchar* dst_row = dst.data + static_cast<size_t>(y) * dst_step;
        for (int x = 0; x < cols; ++x)
        {
            const int yy = static_cast<int>(y_row[x]);
            const int chroma_index = y * cols + x;
            const int uu = static_cast<int>(cvtcolor_yuv444p_plane_byte_u8(src, rows, cols, u_plane_offset, chroma_index));
            const int vv = static_cast<int>(cvtcolor_yuv444p_plane_byte_u8(src, rows, cols, v_plane_offset, chroma_index));
            const uchar b = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 0);
            const uchar g = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 1);
            const uchar r = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 2);

            const int dx = x * 3;
            dst_row[dx + (rgb_order ? 0 : 2)] = r;
            dst_row[dx + 1] = g;
            dst_row[dx + (rgb_order ? 2 : 0)] = b;
        }
    }
}

inline void cvtColor_fallback(const Mat& src, Mat& dst, int code)
{
    CV_Assert(!src.empty() && "cvtColor: source image can not be empty");
    CV_Assert(src.dims == 2 && "cvtColor: only 2D Mat is supported");
    CV_Assert((src.depth() == CV_8U || src.depth() == CV_32F) && "cvtColor: supports CV_8U/CV_32F");

    if (code == COLOR_BGR2GRAY)
    {
        if (src.depth() == CV_8U)
        {
            cvtcolor_bgr2gray_fallback_impl<uchar>(src, dst);
        }
        else
        {
            cvtcolor_bgr2gray_fallback_impl<float>(src, dst);
        }
        return;
    }

    if (code == COLOR_GRAY2BGR)
    {
        if (src.depth() == CV_8U)
        {
            cvtcolor_gray2bgr_fallback_impl<uchar>(src, dst);
        }
        else
        {
            cvtcolor_gray2bgr_fallback_impl<float>(src, dst);
        }
        return;
    }

    if (code == COLOR_GRAY2BGRA || code == COLOR_GRAY2RGBA)
    {
        if (src.depth() == CV_8U)
        {
            cvtcolor_gray2bgra_fallback_impl<uchar>(src, dst);
        }
        else
        {
            cvtcolor_gray2bgra_fallback_impl<float>(src, dst);
        }
        return;
    }

    if (code == COLOR_BGR2RGB || code == COLOR_RGB2BGR)
    {
        if (src.depth() == CV_8U)
        {
            cvtcolor_swap_rb_fallback_impl<uchar>(src, dst);
        }
        else
        {
            cvtcolor_swap_rb_fallback_impl<float>(src, dst);
        }
        return;
    }

    if (code == COLOR_BGRA2GRAY || code == COLOR_RGBA2GRAY)
    {
        const bool rgba_order = (code == COLOR_RGBA2GRAY);
        if (src.depth() == CV_8U)
        {
            cvtcolor_4ch2gray_fallback_impl<uchar>(src, dst, rgba_order);
        }
        else
        {
            cvtcolor_4ch2gray_fallback_impl<float>(src, dst, rgba_order);
        }
        return;
    }

    if (code == COLOR_BGR2BGRA ||
        code == COLOR_RGB2RGBA ||
        code == COLOR_BGR2RGBA ||
        code == COLOR_RGB2BGRA)
    {
        const bool swap_rb = (code == COLOR_BGR2RGBA || code == COLOR_RGB2BGRA);
        if (src.depth() == CV_8U)
        {
            cvtcolor_3ch_to_4ch_alpha_fallback_impl<uchar>(src, dst, swap_rb);
        }
        else
        {
            cvtcolor_3ch_to_4ch_alpha_fallback_impl<float>(src, dst, swap_rb);
        }
        return;
    }

    if (code == COLOR_BGRA2BGR ||
        code == COLOR_RGBA2RGB ||
        code == COLOR_BGRA2RGB ||
        code == COLOR_RGBA2BGR)
    {
        const bool swap_rb = (code == COLOR_BGRA2RGB || code == COLOR_RGBA2BGR);
        if (src.depth() == CV_8U)
        {
            cvtcolor_4ch_to_3ch_drop_alpha_fallback_impl<uchar>(src, dst, swap_rb);
        }
        else
        {
            cvtcolor_4ch_to_3ch_drop_alpha_fallback_impl<float>(src, dst, swap_rb);
        }
        return;
    }

    if (code == COLOR_BGRA2RGBA || code == COLOR_RGBA2BGRA)
    {
        if (src.depth() == CV_8U)
        {
            cvtcolor_swap_rb_4ch_fallback_impl<uchar>(src, dst);
        }
        else
        {
            cvtcolor_swap_rb_4ch_fallback_impl<float>(src, dst);
        }
        return;
    }

    if (code == COLOR_BGR2YUV || code == COLOR_RGB2YUV)
    {
        const bool rgb_order = (code == COLOR_RGB2YUV);
        if (src.depth() == CV_8U)
        {
            cvtcolor_3ch_to_yuv_fallback_impl<uchar>(src, dst, rgb_order);
        }
        else
        {
            cvtcolor_3ch_to_yuv_fallback_impl<float>(src, dst, rgb_order);
        }
        return;
    }

    if (code == COLOR_BGR2YUV_NV24 ||
        code == COLOR_RGB2YUV_NV24 ||
        code == COLOR_BGR2YUV_NV42 ||
        code == COLOR_RGB2YUV_NV42)
    {
        cvtcolor_3ch_to_yuv444sp_fallback_impl(
            src,
            dst,
            code == COLOR_RGB2YUV_NV24 || code == COLOR_RGB2YUV_NV42,
            code == COLOR_BGR2YUV_NV42 || code == COLOR_RGB2YUV_NV42);
        return;
    }

    if (code == COLOR_BGR2YUV_NV12 ||
        code == COLOR_RGB2YUV_NV12 ||
        code == COLOR_BGR2YUV_NV21 ||
        code == COLOR_RGB2YUV_NV21)
    {
        cvtcolor_3ch_to_yuv420sp_fallback_impl(
            src,
            dst,
            code == COLOR_RGB2YUV_NV12 || code == COLOR_RGB2YUV_NV21,
            code == COLOR_BGR2YUV_NV21 || code == COLOR_RGB2YUV_NV21);
        return;
    }

    if (code == COLOR_BGR2YUV_I420 ||
        code == COLOR_RGB2YUV_I420 ||
        code == COLOR_BGR2YUV_YV12 ||
        code == COLOR_RGB2YUV_YV12)
    {
        cvtcolor_3ch_to_yuv420p_fallback_impl(
            src,
            dst,
            code == COLOR_RGB2YUV_I420 || code == COLOR_RGB2YUV_YV12,
            code == COLOR_BGR2YUV_YV12 || code == COLOR_RGB2YUV_YV12);
        return;
    }

    if (code == COLOR_BGR2YUV_NV16 ||
        code == COLOR_RGB2YUV_NV16 ||
        code == COLOR_BGR2YUV_NV61 ||
        code == COLOR_RGB2YUV_NV61)
    {
        cvtcolor_3ch_to_yuv422sp_fallback_impl(
            src,
            dst,
            code == COLOR_RGB2YUV_NV16 || code == COLOR_RGB2YUV_NV61,
            code == COLOR_BGR2YUV_NV61 || code == COLOR_RGB2YUV_NV61);
        return;
    }

    if (code == COLOR_BGR2YUV_YUY2 ||
        code == COLOR_RGB2YUV_YUY2 ||
        code == COLOR_BGR2YUV_UYVY ||
        code == COLOR_RGB2YUV_UYVY)
    {
        cvtcolor_3ch_to_yuv422packed_fallback_impl(
            src,
            dst,
            code == COLOR_RGB2YUV_YUY2 || code == COLOR_RGB2YUV_UYVY,
            code == COLOR_BGR2YUV_UYVY || code == COLOR_RGB2YUV_UYVY);
        return;
    }

    if (code == COLOR_BGR2YUV_I444 ||
        code == COLOR_RGB2YUV_I444 ||
        code == COLOR_BGR2YUV_YV24 ||
        code == COLOR_RGB2YUV_YV24)
    {
        cvtcolor_3ch_to_yuv444p_fallback_impl(
            src,
            dst,
            code == COLOR_RGB2YUV_I444 || code == COLOR_RGB2YUV_YV24,
            code == COLOR_BGR2YUV_YV24 || code == COLOR_RGB2YUV_YV24);
        return;
    }

    if (code == COLOR_YUV2BGR || code == COLOR_YUV2RGB)
    {
        const bool rgb_order = (code == COLOR_YUV2RGB);
        if (src.depth() == CV_8U)
        {
            cvtcolor_yuv_to_3ch_fallback_impl<uchar>(src, dst, rgb_order);
        }
        else
        {
            cvtcolor_yuv_to_3ch_fallback_impl<float>(src, dst, rgb_order);
        }
        return;
    }

    if (code == COLOR_YUV2BGR_NV12 ||
        code == COLOR_YUV2RGB_NV12 ||
        code == COLOR_YUV2BGR_NV21 ||
        code == COLOR_YUV2RGB_NV21)
    {
        cvtcolor_yuv420sp_to_3ch_fallback_impl(
            src,
            dst,
            code == COLOR_YUV2BGR_NV21 || code == COLOR_YUV2RGB_NV21,
            code == COLOR_YUV2RGB_NV12 || code == COLOR_YUV2RGB_NV21);
        return;
    }

    if (code == COLOR_YUV2BGR_NV16 ||
        code == COLOR_YUV2RGB_NV16 ||
        code == COLOR_YUV2BGR_NV61 ||
        code == COLOR_YUV2RGB_NV61)
    {
        cvtcolor_yuv422sp_to_3ch_fallback_impl(
            src,
            dst,
            code == COLOR_YUV2BGR_NV61 || code == COLOR_YUV2RGB_NV61,
            code == COLOR_YUV2RGB_NV16 || code == COLOR_YUV2RGB_NV61);
        return;
    }

    if (code == COLOR_YUV2BGR_NV24 ||
        code == COLOR_YUV2RGB_NV24 ||
        code == COLOR_YUV2BGR_NV42 ||
        code == COLOR_YUV2RGB_NV42)
    {
        cvtcolor_yuv444sp_to_3ch_fallback_impl(
            src,
            dst,
            code == COLOR_YUV2BGR_NV42 || code == COLOR_YUV2RGB_NV42,
            code == COLOR_YUV2RGB_NV24 || code == COLOR_YUV2RGB_NV42);
        return;
    }

    if (code == COLOR_YUV2BGR_I444 ||
        code == COLOR_YUV2RGB_I444 ||
        code == COLOR_YUV2BGR_YV24 ||
        code == COLOR_YUV2RGB_YV24)
    {
        cvtcolor_yuv444p_to_3ch_fallback_impl(
            src,
            dst,
            code == COLOR_YUV2BGR_YV24 || code == COLOR_YUV2RGB_YV24,
            code == COLOR_YUV2RGB_I444 || code == COLOR_YUV2RGB_YV24);
        return;
    }

    if (code == COLOR_YUV2BGR_I420 ||
        code == COLOR_YUV2RGB_I420 ||
        code == COLOR_YUV2BGR_YV12 ||
        code == COLOR_YUV2RGB_YV12)
    {
        cvtcolor_yuv420p_to_3ch_fallback_impl(
            src,
            dst,
            code == COLOR_YUV2BGR_YV12 || code == COLOR_YUV2RGB_YV12,
            code == COLOR_YUV2RGB_I420 || code == COLOR_YUV2RGB_YV12);
        return;
    }

    if (code == COLOR_YUV2BGR_YUY2 ||
        code == COLOR_YUV2RGB_YUY2 ||
        code == COLOR_YUV2BGR_UYVY ||
        code == COLOR_YUV2RGB_UYVY)
    {
        cvtcolor_yuv422packed_to_3ch_fallback_impl(
            src,
            dst,
            code == COLOR_YUV2BGR_UYVY || code == COLOR_YUV2RGB_UYVY,
            code == COLOR_YUV2RGB_YUY2 || code == COLOR_YUV2RGB_UYVY);
        return;
    }

    CV_Error_(Error::StsBadArg, ("cvtColor: unsupported conversion code=%d", code));
}

inline CvtColorFn& cvtcolor_dispatch()
{
    static CvtColorFn fn = &cvtColor_fallback;
    return fn;
}

inline void register_cvtcolor_backend(CvtColorFn fn)
{
    if (fn)
    {
        cvtcolor_dispatch() = fn;
    }
}

inline bool is_cvtcolor_backend_registered()
{
    return cvtcolor_dispatch() != &cvtColor_fallback;
}

}  // namespace detail

inline void cvtColor(const Mat& src, Mat& dst, int code)
{
    detail::ensure_backends_registered_once();
    detail::cvtcolor_dispatch()(src, dst, code);
}

}  // namespace cvh

#endif  // CVH_IMGPROC_CVTCOLOR_H
