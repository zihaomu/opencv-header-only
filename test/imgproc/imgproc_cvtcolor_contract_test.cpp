#include "cvh.h"
#include "gtest/gtest.h"

#include <cstdlib>
#include <type_traits>

using namespace cvh;

namespace
{

int max_abs_diff_u8(const Mat& a, const Mat& b)
{
    if (a.type() != b.type() || a.size[0] != b.size[0] || a.size[1] != b.size[1])
    {
        return 255;
    }
    const size_t count = a.total() * static_cast<size_t>(a.channels());
    int max_diff = 0;
    for (size_t i = 0; i < count; ++i)
    {
        const int diff = std::abs(static_cast<int>(a.data[i]) - static_cast<int>(b.data[i]));
        if (diff > max_diff)
        {
            max_diff = diff;
        }
    }
    return max_diff;
}

float max_abs_diff_f32(const Mat& a, const Mat& b)
{
    if (a.type() != b.type() || a.size[0] != b.size[0] || a.size[1] != b.size[1])
    {
        return 1e9f;
    }
    const size_t count = a.total() * static_cast<size_t>(a.channels());
    float max_diff = 0.0f;
    const float* ap = reinterpret_cast<const float*>(a.data);
    const float* bp = reinterpret_cast<const float*>(b.data);
    for (size_t i = 0; i < count; ++i)
    {
        const float diff = std::abs(ap[i] - bp[i]);
        if (diff > max_diff)
        {
            max_diff = diff;
        }
    }
    return max_diff;
}

Mat bgr2gray_reference_u8(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 3);

    constexpr int kB = 7471;
    constexpr int kG = 38470;
    constexpr int kR = 19595;
    constexpr int kRound = 1 << 15;

    Mat out({src.size[0], src.size[1]}, CV_8UC1);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            const int b = src.at<uchar>(y, x, 0);
            const int g = src.at<uchar>(y, x, 1);
            const int r = src.at<uchar>(y, x, 2);
            out.at<uchar>(y, x) = static_cast<uchar>((kB * b + kG * g + kR * r + kRound) >> 16);
        }
    }
    return out;
}

Mat bgr2gray_reference_f32(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_32F);
    CV_Assert(src.channels() == 3);

    Mat out({src.size[0], src.size[1]}, CV_32FC1);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            const float b = src.at<float>(y, x, 0);
            const float g = src.at<float>(y, x, 1);
            const float r = src.at<float>(y, x, 2);
            out.at<float>(y, x) = 0.114f * b + 0.587f * g + 0.299f * r;
        }
    }
    return out;
}

Mat gray2bgr_reference_u8(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 1);

    Mat out({src.size[0], src.size[1]}, CV_8UC3);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            const uchar g = src.at<uchar>(y, x);
            out.at<uchar>(y, x, 0) = g;
            out.at<uchar>(y, x, 1) = g;
            out.at<uchar>(y, x, 2) = g;
        }
    }
    return out;
}

Mat gray2bgr_reference_f32(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_32F);
    CV_Assert(src.channels() == 1);

    Mat out({src.size[0], src.size[1]}, CV_32FC3);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            const float g = src.at<float>(y, x);
            out.at<float>(y, x, 0) = g;
            out.at<float>(y, x, 1) = g;
            out.at<float>(y, x, 2) = g;
        }
    }
    return out;
}

template <typename T>
Mat bgr2rgb_reference(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 3);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC3 : CV_32FC3;
    Mat out({src.size[0], src.size[1]}, out_type);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            out.at<T>(y, x, 0) = src.at<T>(y, x, 2);
            out.at<T>(y, x, 1) = src.at<T>(y, x, 1);
            out.at<T>(y, x, 2) = src.at<T>(y, x, 0);
        }
    }
    return out;
}

template <typename T>
Mat bgr2bgra_reference(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 3);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC4 : CV_32FC4;
    const T alpha = std::is_same_v<T, uchar> ? static_cast<T>(255) : static_cast<T>(1.0f);
    Mat out({src.size[0], src.size[1]}, out_type);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            out.at<T>(y, x, 0) = src.at<T>(y, x, 0);
            out.at<T>(y, x, 1) = src.at<T>(y, x, 1);
            out.at<T>(y, x, 2) = src.at<T>(y, x, 2);
            out.at<T>(y, x, 3) = alpha;
        }
    }
    return out;
}

template <typename T>
Mat bgra2bgr_reference(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 4);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC3 : CV_32FC3;
    Mat out({src.size[0], src.size[1]}, out_type);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            out.at<T>(y, x, 0) = src.at<T>(y, x, 0);
            out.at<T>(y, x, 1) = src.at<T>(y, x, 1);
            out.at<T>(y, x, 2) = src.at<T>(y, x, 2);
        }
    }
    return out;
}

template <typename T>
Mat rgb2rgba_reference(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 3);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC4 : CV_32FC4;
    const T alpha = std::is_same_v<T, uchar> ? static_cast<T>(255) : static_cast<T>(1.0f);
    Mat out({src.size[0], src.size[1]}, out_type);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            out.at<T>(y, x, 0) = src.at<T>(y, x, 0);
            out.at<T>(y, x, 1) = src.at<T>(y, x, 1);
            out.at<T>(y, x, 2) = src.at<T>(y, x, 2);
            out.at<T>(y, x, 3) = alpha;
        }
    }
    return out;
}

template <typename T>
Mat rgba2rgb_reference(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 4);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC3 : CV_32FC3;
    Mat out({src.size[0], src.size[1]}, out_type);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            out.at<T>(y, x, 0) = src.at<T>(y, x, 0);
            out.at<T>(y, x, 1) = src.at<T>(y, x, 1);
            out.at<T>(y, x, 2) = src.at<T>(y, x, 2);
        }
    }
    return out;
}

template <typename T>
Mat bgr2rgba_reference(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 3);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC4 : CV_32FC4;
    const T alpha = std::is_same_v<T, uchar> ? static_cast<T>(255) : static_cast<T>(1.0f);
    Mat out({src.size[0], src.size[1]}, out_type);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            out.at<T>(y, x, 0) = src.at<T>(y, x, 2);
            out.at<T>(y, x, 1) = src.at<T>(y, x, 1);
            out.at<T>(y, x, 2) = src.at<T>(y, x, 0);
            out.at<T>(y, x, 3) = alpha;
        }
    }
    return out;
}

template <typename T>
Mat rgba2bgr_reference(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 4);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC3 : CV_32FC3;
    Mat out({src.size[0], src.size[1]}, out_type);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            out.at<T>(y, x, 0) = src.at<T>(y, x, 2);
            out.at<T>(y, x, 1) = src.at<T>(y, x, 1);
            out.at<T>(y, x, 2) = src.at<T>(y, x, 0);
        }
    }
    return out;
}

template <typename T>
Mat rgb2bgra_reference(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 3);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC4 : CV_32FC4;
    const T alpha = std::is_same_v<T, uchar> ? static_cast<T>(255) : static_cast<T>(1.0f);
    Mat out({src.size[0], src.size[1]}, out_type);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            out.at<T>(y, x, 0) = src.at<T>(y, x, 2);
            out.at<T>(y, x, 1) = src.at<T>(y, x, 1);
            out.at<T>(y, x, 2) = src.at<T>(y, x, 0);
            out.at<T>(y, x, 3) = alpha;
        }
    }
    return out;
}

template <typename T>
Mat bgra2rgb_reference(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 4);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC3 : CV_32FC3;
    Mat out({src.size[0], src.size[1]}, out_type);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            out.at<T>(y, x, 0) = src.at<T>(y, x, 2);
            out.at<T>(y, x, 1) = src.at<T>(y, x, 1);
            out.at<T>(y, x, 2) = src.at<T>(y, x, 0);
        }
    }
    return out;
}

template <typename T>
Mat swap_rb_4ch_reference(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 4);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC4 : CV_32FC4;
    Mat out({src.size[0], src.size[1]}, out_type);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            out.at<T>(y, x, 0) = src.at<T>(y, x, 2);
            out.at<T>(y, x, 1) = src.at<T>(y, x, 1);
            out.at<T>(y, x, 2) = src.at<T>(y, x, 0);
            out.at<T>(y, x, 3) = src.at<T>(y, x, 3);
        }
    }
    return out;
}

template <typename T>
Mat gray2bgra_reference(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 1);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC4 : CV_32FC4;
    const T alpha = std::is_same_v<T, uchar> ? static_cast<T>(255) : static_cast<T>(1.0f);
    Mat out({src.size[0], src.size[1]}, out_type);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            const T g = src.at<T>(y, x);
            out.at<T>(y, x, 0) = g;
            out.at<T>(y, x, 1) = g;
            out.at<T>(y, x, 2) = g;
            out.at<T>(y, x, 3) = alpha;
        }
    }
    return out;
}

template <typename T>
Mat color4_to_gray_reference(const Mat& src, bool rgba_order)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 4);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC1 : CV_32FC1;
    Mat out({src.size[0], src.size[1]}, out_type);

    if constexpr (std::is_same_v<T, uchar>)
    {
        constexpr int kB = 7471;
        constexpr int kG = 38470;
        constexpr int kR = 19595;
        constexpr int kRound = 1 << 15;

        for (int y = 0; y < src.size[0]; ++y)
        {
            for (int x = 0; x < src.size[1]; ++x)
            {
                const int b = src.at<uchar>(y, x, rgba_order ? 2 : 0);
                const int g = src.at<uchar>(y, x, 1);
                const int r = src.at<uchar>(y, x, rgba_order ? 0 : 2);
                out.at<uchar>(y, x) = static_cast<uchar>((kB * b + kG * g + kR * r + kRound) >> 16);
            }
        }
        return out;
    }

    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            const float b = src.at<float>(y, x, rgba_order ? 2 : 0);
            const float g = src.at<float>(y, x, 1);
            const float r = src.at<float>(y, x, rgba_order ? 0 : 2);
            out.at<float>(y, x) = 0.114f * b + 0.587f * g + 0.299f * r;
        }
    }
    return out;
}

template <typename T>
constexpr float yuv_delta_reference()
{
    return std::is_same_v<T, uchar> ? 128.0f : 0.5f;
}

template <typename T>
Mat color3_to_yuv_reference(const Mat& src, bool rgb_order)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 3);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC3 : CV_32FC3;
    const float delta = yuv_delta_reference<T>();
    Mat out({src.size[0], src.size[1]}, out_type);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            const float r = static_cast<float>(src.at<T>(y, x, rgb_order ? 0 : 2));
            const float g = static_cast<float>(src.at<T>(y, x, 1));
            const float b = static_cast<float>(src.at<T>(y, x, rgb_order ? 2 : 0));
            const float yy = 0.299f * r + 0.587f * g + 0.114f * b;
            const float uu = 0.492f * (b - yy) + delta;
            const float vv = 0.877f * (r - yy) + delta;

            if constexpr (std::is_same_v<T, uchar>)
            {
                out.at<uchar>(y, x, 0) = saturate_cast<uchar>(yy);
                out.at<uchar>(y, x, 1) = saturate_cast<uchar>(uu);
                out.at<uchar>(y, x, 2) = saturate_cast<uchar>(vv);
            }
            else
            {
                out.at<float>(y, x, 0) = yy;
                out.at<float>(y, x, 1) = uu;
                out.at<float>(y, x, 2) = vv;
            }
        }
    }
    return out;
}

template <typename T>
Mat yuv_to_color3_reference(const Mat& src, bool rgb_order)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 3);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC3 : CV_32FC3;
    const float delta = yuv_delta_reference<T>();
    Mat out({src.size[0], src.size[1]}, out_type);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            const float yy = static_cast<float>(src.at<T>(y, x, 0));
            const float uu = static_cast<float>(src.at<T>(y, x, 1)) - delta;
            const float vv = static_cast<float>(src.at<T>(y, x, 2)) - delta;

            const float b = yy + 2.032f * uu;
            const float g = yy - 0.395f * uu - 0.581f * vv;
            const float r = yy + 1.140f * vv;

            if constexpr (std::is_same_v<T, uchar>)
            {
                out.at<uchar>(y, x, rgb_order ? 0 : 2) = saturate_cast<uchar>(r);
                out.at<uchar>(y, x, 1) = saturate_cast<uchar>(g);
                out.at<uchar>(y, x, rgb_order ? 2 : 0) = saturate_cast<uchar>(b);
            }
            else
            {
                out.at<float>(y, x, rgb_order ? 0 : 2) = r;
                out.at<float>(y, x, 1) = g;
                out.at<float>(y, x, rgb_order ? 2 : 0) = b;
            }
        }
    }
    return out;
}

inline uchar color3_to_yuv_limited_u8(int bb, int gg, int rr, int channel);

inline uchar yuv420_limited_to_u8(int yy, int uu, int vv, int channel)
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

Mat yuv420sp_to_color3_reference_u8(const Mat& src, bool nv21_layout, bool rgb_order)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 1);
    CV_Assert((src.size[0] % 3) == 0);
    CV_Assert((src.size[1] % 2) == 0);

    const int rows = src.size[0] * 2 / 3;
    const int cols = src.size[1];
    Mat out({rows, cols}, CV_8UC3);

    for (int y = 0; y < rows; ++y)
    {
        const int uv_y = rows + y / 2;
        for (int x = 0; x < cols; ++x)
        {
            const int uv_x = x & ~1;
            const int yy = static_cast<int>(src.at<uchar>(y, x));
            const int first = static_cast<int>(src.at<uchar>(uv_y, uv_x + 0));
            const int second = static_cast<int>(src.at<uchar>(uv_y, uv_x + 1));
            const int uu = nv21_layout ? second : first;
            const int vv = nv21_layout ? first : second;

            const uchar b = yuv420_limited_to_u8(yy, uu, vv, 0);
            const uchar g = yuv420_limited_to_u8(yy, uu, vv, 1);
            const uchar r = yuv420_limited_to_u8(yy, uu, vv, 2);

            out.at<uchar>(y, x, rgb_order ? 0 : 2) = r;
            out.at<uchar>(y, x, 1) = g;
            out.at<uchar>(y, x, rgb_order ? 2 : 0) = b;
        }
    }

    return out;
}

Mat color3_to_yuv420sp_reference_u8(const Mat& src, bool rgb_order, bool nv21_layout)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 3);
    CV_Assert((src.size[0] % 2) == 0);
    CV_Assert((src.size[1] % 2) == 0);

    const int rows = src.size[0];
    const int cols = src.size[1];
    Mat out({rows * 3 / 2, cols}, CV_8UC1);

    for (int y = 0; y < rows; y += 2)
    {
        for (int x = 0; x < cols; x += 2)
        {
            int sum_b = 0;
            int sum_g = 0;
            int sum_r = 0;

            for (int dy = 0; dy < 2; ++dy)
            {
                for (int dx = 0; dx < 2; ++dx)
                {
                    const int yy_y = y + dy;
                    const int yy_x = x + dx;
                    const int bb = static_cast<int>(src.at<uchar>(yy_y, yy_x, rgb_order ? 2 : 0));
                    const int gg = static_cast<int>(src.at<uchar>(yy_y, yy_x, 1));
                    const int rr = static_cast<int>(src.at<uchar>(yy_y, yy_x, rgb_order ? 0 : 2));
                    const uchar yy = color3_to_yuv_limited_u8(bb, gg, rr, 0);

                    out.at<uchar>(yy_y, yy_x) = yy;
                    sum_b += bb;
                    sum_g += gg;
                    sum_r += rr;
                }
            }

            const int avg_b = (sum_b + 2) >> 2;
            const int avg_g = (sum_g + 2) >> 2;
            const int avg_r = (sum_r + 2) >> 2;
            const uchar uu = color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 1);
            const uchar vv = color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 2);
            const int uv_y = rows + y / 2;

            out.at<uchar>(uv_y, x + 0) = nv21_layout ? vv : uu;
            out.at<uchar>(uv_y, x + 1) = nv21_layout ? uu : vv;
        }
    }

    return out;
}

inline uchar yuv420p_plane_byte_at_u8(const Mat& src, int rows, int cols, int plane_offset, int plane_index)
{
    const int logical_offset = plane_offset + plane_index;
    return src.at<uchar>(rows + logical_offset / cols, logical_offset % cols);
}

inline void set_yuv420p_plane_byte_u8(Mat& dst, int rows, int cols, int plane_offset, int plane_index, uchar value)
{
    const int logical_offset = plane_offset + plane_index;
    dst.at<uchar>(rows + logical_offset / cols, logical_offset % cols) = value;
}

Mat color3_to_yuv420p_reference_u8(const Mat& src, bool rgb_order, bool yv12_layout)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 3);
    CV_Assert((src.size[0] % 2) == 0);
    CV_Assert((src.size[1] % 2) == 0);

    const int rows = src.size[0];
    const int cols = src.size[1];
    const int uv_size = rows * cols / 4;
    const int u_plane_offset = yv12_layout ? uv_size : 0;
    const int v_plane_offset = yv12_layout ? 0 : uv_size;
    Mat out({rows * 3 / 2, cols}, CV_8UC1);

    for (int y = 0; y < rows; y += 2)
    {
        for (int x = 0; x < cols; x += 2)
        {
            int sum_b = 0;
            int sum_g = 0;
            int sum_r = 0;

            for (int dy = 0; dy < 2; ++dy)
            {
                for (int dx = 0; dx < 2; ++dx)
                {
                    const int yy_y = y + dy;
                    const int yy_x = x + dx;
                    const int bb = static_cast<int>(src.at<uchar>(yy_y, yy_x, rgb_order ? 2 : 0));
                    const int gg = static_cast<int>(src.at<uchar>(yy_y, yy_x, 1));
                    const int rr = static_cast<int>(src.at<uchar>(yy_y, yy_x, rgb_order ? 0 : 2));
                    const uchar yy = color3_to_yuv_limited_u8(bb, gg, rr, 0);

                    out.at<uchar>(yy_y, yy_x) = yy;
                    sum_b += bb;
                    sum_g += gg;
                    sum_r += rr;
                }
            }

            const int avg_b = (sum_b + 2) >> 2;
            const int avg_g = (sum_g + 2) >> 2;
            const int avg_r = (sum_r + 2) >> 2;
            const uchar uu = color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 1);
            const uchar vv = color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 2);
            const int chroma_index = (y / 2) * (cols / 2) + (x / 2);

            set_yuv420p_plane_byte_u8(out, rows, cols, u_plane_offset, chroma_index, uu);
            set_yuv420p_plane_byte_u8(out, rows, cols, v_plane_offset, chroma_index, vv);
        }
    }

    return out;
}

inline uchar yuv444p_plane_byte_at_u8(const Mat& src, int rows, int cols, int plane_offset, int plane_index)
{
    const int logical_offset = plane_offset + plane_index;
    return src.at<uchar>(rows + logical_offset / cols, logical_offset % cols);
}

inline void set_yuv444p_plane_byte_u8(Mat& dst, int rows, int cols, int plane_offset, int plane_index, uchar value)
{
    const int logical_offset = plane_offset + plane_index;
    dst.at<uchar>(rows + logical_offset / cols, logical_offset % cols) = value;
}

Mat yuv420p_to_color3_reference_u8(const Mat& src, bool yv12_layout, bool rgb_order)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 1);
    CV_Assert((src.size[0] % 3) == 0);
    CV_Assert((src.size[1] % 2) == 0);

    const int rows = src.size[0] * 2 / 3;
    const int cols = src.size[1];
    const int uv_size = rows * cols / 4;
    const int u_plane_offset = yv12_layout ? uv_size : 0;
    const int v_plane_offset = yv12_layout ? 0 : uv_size;

    Mat out({rows, cols}, CV_8UC3);
    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            const int yy = static_cast<int>(src.at<uchar>(y, x));
            const int chroma_index = (y / 2) * (cols / 2) + (x / 2);
            const int uu = static_cast<int>(yuv420p_plane_byte_at_u8(src, rows, cols, u_plane_offset, chroma_index));
            const int vv = static_cast<int>(yuv420p_plane_byte_at_u8(src, rows, cols, v_plane_offset, chroma_index));

            const uchar b = yuv420_limited_to_u8(yy, uu, vv, 0);
            const uchar g = yuv420_limited_to_u8(yy, uu, vv, 1);
            const uchar r = yuv420_limited_to_u8(yy, uu, vv, 2);

            out.at<uchar>(y, x, rgb_order ? 0 : 2) = r;
            out.at<uchar>(y, x, 1) = g;
            out.at<uchar>(y, x, rgb_order ? 2 : 0) = b;
        }
    }

    return out;
}

Mat yuv444p_to_color3_reference_u8(const Mat& src, bool yv24_layout, bool rgb_order)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 1);
    CV_Assert((src.size[0] % 3) == 0);

    const int rows = src.size[0] / 3;
    const int cols = src.size[1];
    const int plane_size = rows * cols;
    const int u_plane_offset = yv24_layout ? plane_size : 0;
    const int v_plane_offset = yv24_layout ? 0 : plane_size;

    Mat out({rows, cols}, CV_8UC3);
    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            const int yy = static_cast<int>(src.at<uchar>(y, x));
            const int chroma_index = y * cols + x;
            const int uu = static_cast<int>(yuv444p_plane_byte_at_u8(src, rows, cols, u_plane_offset, chroma_index));
            const int vv = static_cast<int>(yuv444p_plane_byte_at_u8(src, rows, cols, v_plane_offset, chroma_index));

            const uchar b = yuv420_limited_to_u8(yy, uu, vv, 0);
            const uchar g = yuv420_limited_to_u8(yy, uu, vv, 1);
            const uchar r = yuv420_limited_to_u8(yy, uu, vv, 2);

            out.at<uchar>(y, x, rgb_order ? 0 : 2) = r;
            out.at<uchar>(y, x, 1) = g;
            out.at<uchar>(y, x, rgb_order ? 2 : 0) = b;
        }
    }

    return out;
}

Mat yuv422sp_to_color3_reference_u8(const Mat& src, bool nv61_layout, bool rgb_order)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 1);
    CV_Assert((src.size[0] % 2) == 0);
    CV_Assert((src.size[1] % 2) == 0);

    const int rows = src.size[0] / 2;
    const int cols = src.size[1];
    Mat out({rows, cols}, CV_8UC3);

    for (int y = 0; y < rows; ++y)
    {
        const int uv_y = rows + y;
        for (int x = 0; x < cols; ++x)
        {
            const int uv_x = x & ~1;
            const int yy = static_cast<int>(src.at<uchar>(y, x));
            const int first = static_cast<int>(src.at<uchar>(uv_y, uv_x + 0));
            const int second = static_cast<int>(src.at<uchar>(uv_y, uv_x + 1));
            const int uu = nv61_layout ? second : first;
            const int vv = nv61_layout ? first : second;

            const uchar b = yuv420_limited_to_u8(yy, uu, vv, 0);
            const uchar g = yuv420_limited_to_u8(yy, uu, vv, 1);
            const uchar r = yuv420_limited_to_u8(yy, uu, vv, 2);

            out.at<uchar>(y, x, rgb_order ? 0 : 2) = r;
            out.at<uchar>(y, x, 1) = g;
            out.at<uchar>(y, x, rgb_order ? 2 : 0) = b;
        }
    }

    return out;
}

inline uchar yuv444sp_plane_byte_at_u8(const Mat& src, int rows, int cols, int plane_index)
{
    return src.at<uchar>(rows + plane_index / cols, plane_index % cols);
}

inline void set_yuv444sp_plane_byte_u8(Mat& dst, int rows, int cols, int plane_index, uchar value)
{
    dst.at<uchar>(rows + plane_index / cols, plane_index % cols) = value;
}

inline uchar yuv422sp_plane_byte_at_u8(const Mat& src, int rows, int cols, int plane_index)
{
    return src.at<uchar>(rows + plane_index / cols, plane_index % cols);
}

inline void set_yuv422sp_plane_byte_u8(Mat& dst, int rows, int cols, int plane_index, uchar value)
{
    dst.at<uchar>(rows + plane_index / cols, plane_index % cols) = value;
}

inline uchar color3_to_yuv_limited_u8(int bb, int gg, int rr, int channel)
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

Mat yuv444sp_to_color3_reference_u8(const Mat& src, bool nv42_layout, bool rgb_order)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 1);
    CV_Assert((src.size[0] % 3) == 0);

    const int rows = src.size[0] / 3;
    const int cols = src.size[1];
    Mat out({rows, cols}, CV_8UC3);

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            const int yy = static_cast<int>(src.at<uchar>(y, x));
            const int base = y * (cols * 2) + x * 2;
            const int uu = static_cast<int>(yuv444sp_plane_byte_at_u8(src, rows, cols, base + (nv42_layout ? 1 : 0)));
            const int vv = static_cast<int>(yuv444sp_plane_byte_at_u8(src, rows, cols, base + (nv42_layout ? 0 : 1)));

            const uchar b = yuv420_limited_to_u8(yy, uu, vv, 0);
            const uchar g = yuv420_limited_to_u8(yy, uu, vv, 1);
            const uchar r = yuv420_limited_to_u8(yy, uu, vv, 2);

            out.at<uchar>(y, x, rgb_order ? 0 : 2) = r;
            out.at<uchar>(y, x, 1) = g;
            out.at<uchar>(y, x, rgb_order ? 2 : 0) = b;
        }
    }

    return out;
}

Mat color3_to_yuv444sp_reference_u8(const Mat& src, bool rgb_order, bool nv42_layout)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 3);

    const int rows = src.size[0];
    const int cols = src.size[1];
    Mat out({rows * 3, cols}, CV_8UC1);

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            const int bb = static_cast<int>(src.at<uchar>(y, x, rgb_order ? 2 : 0));
            const int gg = static_cast<int>(src.at<uchar>(y, x, 1));
            const int rr = static_cast<int>(src.at<uchar>(y, x, rgb_order ? 0 : 2));
            const uchar yy = color3_to_yuv_limited_u8(bb, gg, rr, 0);
            const uchar uu = color3_to_yuv_limited_u8(bb, gg, rr, 1);
            const uchar vv = color3_to_yuv_limited_u8(bb, gg, rr, 2);
            const int base = y * (cols * 2) + x * 2;

            out.at<uchar>(y, x) = yy;
            set_yuv444sp_plane_byte_u8(out, rows, cols, base + 0, nv42_layout ? vv : uu);
            set_yuv444sp_plane_byte_u8(out, rows, cols, base + 1, nv42_layout ? uu : vv);
        }
    }

    return out;
}

Mat color3_to_yuv444p_reference_u8(const Mat& src, bool rgb_order, bool yv24_layout)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 3);

    const int rows = src.size[0];
    const int cols = src.size[1];
    const int plane_size = rows * cols;
    Mat out({rows * 3, cols}, CV_8UC1);

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            const int bb = static_cast<int>(src.at<uchar>(y, x, rgb_order ? 2 : 0));
            const int gg = static_cast<int>(src.at<uchar>(y, x, 1));
            const int rr = static_cast<int>(src.at<uchar>(y, x, rgb_order ? 0 : 2));
            const uchar yy = color3_to_yuv_limited_u8(bb, gg, rr, 0);
            const uchar uu = color3_to_yuv_limited_u8(bb, gg, rr, 1);
            const uchar vv = color3_to_yuv_limited_u8(bb, gg, rr, 2);
            const int chroma_index = y * cols + x;

            out.at<uchar>(y, x) = yy;
            set_yuv444p_plane_byte_u8(out, rows, cols, yv24_layout ? plane_size : 0, chroma_index, uu);
            set_yuv444p_plane_byte_u8(out, rows, cols, yv24_layout ? 0 : plane_size, chroma_index, vv);
        }
    }

    return out;
}

Mat color3_to_yuv422sp_reference_u8(const Mat& src, bool rgb_order, bool nv61_layout)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 3);
    CV_Assert((src.size[1] % 2) == 0);

    const int rows = src.size[0];
    const int cols = src.size[1];
    Mat out({rows * 2, cols}, CV_8UC1);

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; x += 2)
        {
            int sum_b = 0;
            int sum_g = 0;
            int sum_r = 0;

            for (int dx = 0; dx < 2; ++dx)
            {
                const int xx = x + dx;
                const int bb = static_cast<int>(src.at<uchar>(y, xx, rgb_order ? 2 : 0));
                const int gg = static_cast<int>(src.at<uchar>(y, xx, 1));
                const int rr = static_cast<int>(src.at<uchar>(y, xx, rgb_order ? 0 : 2));
                const uchar yy = color3_to_yuv_limited_u8(bb, gg, rr, 0);

                out.at<uchar>(y, xx) = yy;
                sum_b += bb;
                sum_g += gg;
                sum_r += rr;
            }

            const int avg_b = (sum_b + 1) >> 1;
            const int avg_g = (sum_g + 1) >> 1;
            const int avg_r = (sum_r + 1) >> 1;
            const uchar uu = color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 1);
            const uchar vv = color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 2);
            const int base = y * cols + x;

            set_yuv422sp_plane_byte_u8(out, rows, cols, base + 0, nv61_layout ? vv : uu);
            set_yuv422sp_plane_byte_u8(out, rows, cols, base + 1, nv61_layout ? uu : vv);
        }
    }

    return out;
}

inline void set_yuv422_packed_pair_u8(Mat& dst, int y, int pair_x, bool uyvy_layout, uchar yy0, uchar yy1, uchar uu, uchar vv)
{
    CV_Assert(dst.dims == 2);
    CV_Assert(dst.depth() == CV_8U);
    CV_Assert(dst.channels() == 2);
    CV_Assert((pair_x % 2) == 0);
    CV_Assert(pair_x >= 0 && pair_x + 1 < dst.size[1]);

    if (uyvy_layout)
    {
        dst.at<uchar>(y, pair_x + 0, 0) = uu;
        dst.at<uchar>(y, pair_x + 0, 1) = yy0;
        dst.at<uchar>(y, pair_x + 1, 0) = vv;
        dst.at<uchar>(y, pair_x + 1, 1) = yy1;
        return;
    }

    dst.at<uchar>(y, pair_x + 0, 0) = yy0;
    dst.at<uchar>(y, pair_x + 0, 1) = uu;
    dst.at<uchar>(y, pair_x + 1, 0) = yy1;
    dst.at<uchar>(y, pair_x + 1, 1) = vv;
}

Mat color3_to_yuv422packed_reference_u8(const Mat& src, bool rgb_order, bool uyvy_layout)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 3);
    CV_Assert((src.size[1] % 2) == 0);

    const int rows = src.size[0];
    const int cols = src.size[1];
    Mat out({rows, cols}, CV_8UC2);

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; x += 2)
        {
            int sum_b = 0;
            int sum_g = 0;
            int sum_r = 0;
            uchar yy[2] = {0, 0};

            for (int dx = 0; dx < 2; ++dx)
            {
                const int xx = x + dx;
                const int bb = static_cast<int>(src.at<uchar>(y, xx, rgb_order ? 2 : 0));
                const int gg = static_cast<int>(src.at<uchar>(y, xx, 1));
                const int rr = static_cast<int>(src.at<uchar>(y, xx, rgb_order ? 0 : 2));

                yy[dx] = color3_to_yuv_limited_u8(bb, gg, rr, 0);
                sum_b += bb;
                sum_g += gg;
                sum_r += rr;
            }

            const int avg_b = (sum_b + 1) >> 1;
            const int avg_g = (sum_g + 1) >> 1;
            const int avg_r = (sum_r + 1) >> 1;
            const uchar uu = color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 1);
            const uchar vv = color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 2);

            set_yuv422_packed_pair_u8(out, y, x, uyvy_layout, yy[0], yy[1], uu, vv);
        }
    }

    return out;
}

Mat yuv422packed_to_color3_reference_u8(const Mat& src, bool uyvy_layout, bool rgb_order)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 2);
    CV_Assert((src.size[1] % 2) == 0);

    const int rows = src.size[0];
    const int cols = src.size[1];
    Mat out({rows, cols}, CV_8UC3);

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            const int pair_x = x & ~1;
            const int yy = static_cast<int>(src.at<uchar>(y, x, uyvy_layout ? 1 : 0));
            const int uu = static_cast<int>(src.at<uchar>(y, pair_x + 0, uyvy_layout ? 0 : 1));
            const int vv = static_cast<int>(src.at<uchar>(y, pair_x + 1, uyvy_layout ? 0 : 1));

            const uchar b = yuv420_limited_to_u8(yy, uu, vv, 0);
            const uchar g = yuv420_limited_to_u8(yy, uu, vv, 1);
            const uchar r = yuv420_limited_to_u8(yy, uu, vv, 2);

            out.at<uchar>(y, x, rgb_order ? 0 : 2) = r;
            out.at<uchar>(y, x, 1) = g;
            out.at<uchar>(y, x, rgb_order ? 2 : 0) = b;
        }
    }

    return out;
}

}  // namespace

TEST(ImgprocCvtColor_TEST, bgr2gray_matches_known_values)
{
    Mat src({2, 2}, CV_8UC3);
    src.at<uchar>(0, 0, 0) = 10;  src.at<uchar>(0, 0, 1) = 20;  src.at<uchar>(0, 0, 2) = 30;
    src.at<uchar>(0, 1, 0) = 100; src.at<uchar>(0, 1, 1) = 110; src.at<uchar>(0, 1, 2) = 120;
    src.at<uchar>(1, 0, 0) = 0;   src.at<uchar>(1, 0, 1) = 0;   src.at<uchar>(1, 0, 2) = 255;
    src.at<uchar>(1, 1, 0) = 255; src.at<uchar>(1, 1, 1) = 0;   src.at<uchar>(1, 1, 2) = 0;

    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    ASSERT_EQ(gray.type(), CV_8UC1);
    ASSERT_EQ(gray.size[0], 2);
    ASSERT_EQ(gray.size[1], 2);

    const uchar expected[4] = {22, 112, 76, 29};
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 2; ++x)
        {
            EXPECT_EQ(gray.at<uchar>(y, x), expected[y * 2 + x]);
        }
    }
}

TEST(ImgprocCvtColor_TEST, gray2bgr_replicates_channels)
{
    Mat gray({2, 3}, CV_8UC1);
    gray.at<uchar>(0, 0) = 10;
    gray.at<uchar>(0, 1) = 20;
    gray.at<uchar>(0, 2) = 30;
    gray.at<uchar>(1, 0) = 40;
    gray.at<uchar>(1, 1) = 50;
    gray.at<uchar>(1, 2) = 60;

    Mat bgr;
    cvtColor(gray, bgr, COLOR_GRAY2BGR);
    ASSERT_EQ(bgr.type(), CV_8UC3);
    ASSERT_EQ(bgr.size[0], gray.size[0]);
    ASSERT_EQ(bgr.size[1], gray.size[1]);

    for (int y = 0; y < bgr.size[0]; ++y)
    {
        for (int x = 0; x < bgr.size[1]; ++x)
        {
            const uchar v = gray.at<uchar>(y, x);
            EXPECT_EQ(bgr.at<uchar>(y, x, 0), v);
            EXPECT_EQ(bgr.at<uchar>(y, x, 1), v);
            EXPECT_EQ(bgr.at<uchar>(y, x, 2), v);
        }
    }
}

TEST(ImgprocCvtColor_TEST, gray2bgr_then_bgr2gray_roundtrip_is_identity)
{
    Mat gray({4, 5}, CV_8UC1);
    for (int y = 0; y < gray.size[0]; ++y)
    {
        for (int x = 0; x < gray.size[1]; ++x)
        {
            gray.at<uchar>(y, x) = static_cast<uchar>((y * 31 + x * 17) % 256);
        }
    }

    Mat bgr;
    cvtColor(gray, bgr, COLOR_GRAY2BGR);

    Mat back;
    cvtColor(bgr, back, COLOR_BGR2GRAY);
    ASSERT_EQ(back.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(gray, back), 0);
}

TEST(ImgprocCvtColor_TEST, bgr2rgb_swaps_blue_and_red_channels)
{
    Mat src({2, 3}, CV_8UC3);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            src.at<uchar>(y, x, 0) = static_cast<uchar>(10 + y * 20 + x * 3);
            src.at<uchar>(y, x, 1) = static_cast<uchar>(30 + y * 20 + x * 5);
            src.at<uchar>(y, x, 2) = static_cast<uchar>(50 + y * 20 + x * 7);
        }
    }

    Mat expected = bgr2rgb_reference<uchar>(src);
    Mat actual;
    cvtColor(src, actual, COLOR_BGR2RGB);
    ASSERT_EQ(actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(expected, actual), 0);

    Mat roundtrip;
    cvtColor(actual, roundtrip, COLOR_RGB2BGR);
    EXPECT_EQ(max_abs_diff_u8(src, roundtrip), 0);
}

TEST(ImgprocCvtColor_TEST, bgr2bgra_and_bgra2bgr_match_reference)
{
    Mat src({2, 4}, CV_8UC3);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            src.at<uchar>(y, x, 0) = static_cast<uchar>(1 + y * 17 + x * 2);
            src.at<uchar>(y, x, 1) = static_cast<uchar>(2 + y * 19 + x * 3);
            src.at<uchar>(y, x, 2) = static_cast<uchar>(3 + y * 23 + x * 5);
        }
    }

    Mat bgra_expected = bgr2bgra_reference<uchar>(src);
    Mat bgra_actual;
    cvtColor(src, bgra_actual, COLOR_BGR2BGRA);
    ASSERT_EQ(bgra_actual.type(), CV_8UC4);
    EXPECT_EQ(max_abs_diff_u8(bgra_expected, bgra_actual), 0);

    Mat bgr_expected = bgra2bgr_reference<uchar>(bgra_actual);
    Mat bgr_actual;
    cvtColor(bgra_actual, bgr_actual, COLOR_BGRA2BGR);
    ASSERT_EQ(bgr_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_expected, bgr_actual), 0);
}

TEST(ImgprocCvtColor_TEST, rgb_rgba_bgr_bgra_family_u8_matches_reference)
{
    Mat rgb({2, 4}, CV_8UC3);
    for (int y = 0; y < rgb.size[0]; ++y)
    {
        for (int x = 0; x < rgb.size[1]; ++x)
        {
            rgb.at<uchar>(y, x, 0) = static_cast<uchar>(11 + y * 13 + x * 2);
            rgb.at<uchar>(y, x, 1) = static_cast<uchar>(21 + y * 17 + x * 3);
            rgb.at<uchar>(y, x, 2) = static_cast<uchar>(31 + y * 19 + x * 5);
        }
    }

    Mat rgba_expected = rgb2rgba_reference<uchar>(rgb);
    Mat rgba_actual;
    cvtColor(rgb, rgba_actual, COLOR_RGB2RGBA);
    ASSERT_EQ(rgba_actual.type(), CV_8UC4);
    EXPECT_EQ(max_abs_diff_u8(rgba_expected, rgba_actual), 0);

    Mat rgb_roundtrip;
    cvtColor(rgba_actual, rgb_roundtrip, COLOR_RGBA2RGB);
    EXPECT_EQ(max_abs_diff_u8(rgb, rgb_roundtrip), 0);

    Mat bgra_expected = rgb2bgra_reference<uchar>(rgb);
    Mat bgra_actual;
    cvtColor(rgb, bgra_actual, COLOR_RGB2BGRA);
    ASSERT_EQ(bgra_actual.type(), CV_8UC4);
    EXPECT_EQ(max_abs_diff_u8(bgra_expected, bgra_actual), 0);

    Mat rgb_from_bgra;
    cvtColor(bgra_actual, rgb_from_bgra, COLOR_BGRA2RGB);
    EXPECT_EQ(max_abs_diff_u8(rgb, rgb_from_bgra), 0);

    Mat bgr({2, 4}, CV_8UC3);
    for (int y = 0; y < bgr.size[0]; ++y)
    {
        for (int x = 0; x < bgr.size[1]; ++x)
        {
            bgr.at<uchar>(y, x, 0) = static_cast<uchar>(7 + y * 23 + x * 2);
            bgr.at<uchar>(y, x, 1) = static_cast<uchar>(9 + y * 11 + x * 7);
            bgr.at<uchar>(y, x, 2) = static_cast<uchar>(13 + y * 5 + x * 9);
        }
    }

    Mat rgba_from_bgr_expected = bgr2rgba_reference<uchar>(bgr);
    Mat rgba_from_bgr_actual;
    cvtColor(bgr, rgba_from_bgr_actual, COLOR_BGR2RGBA);
    ASSERT_EQ(rgba_from_bgr_actual.type(), CV_8UC4);
    EXPECT_EQ(max_abs_diff_u8(rgba_from_bgr_expected, rgba_from_bgr_actual), 0);

    Mat bgr_roundtrip;
    cvtColor(rgba_from_bgr_actual, bgr_roundtrip, COLOR_RGBA2BGR);
    EXPECT_EQ(max_abs_diff_u8(bgr, bgr_roundtrip), 0);

    Mat rgba_swapped_expected = swap_rb_4ch_reference<uchar>(bgra_actual);
    Mat rgba_swapped_actual;
    cvtColor(bgra_actual, rgba_swapped_actual, COLOR_BGRA2RGBA);
    ASSERT_EQ(rgba_swapped_actual.type(), CV_8UC4);
    EXPECT_EQ(max_abs_diff_u8(rgba_swapped_expected, rgba_swapped_actual), 0);

    Mat bgra_roundtrip;
    cvtColor(rgba_swapped_actual, bgra_roundtrip, COLOR_RGBA2BGRA);
    EXPECT_EQ(max_abs_diff_u8(bgra_actual, bgra_roundtrip), 0);
}

TEST(ImgprocCvtColor_TEST, gray_rgba_bgra_family_u8_matches_reference)
{
    Mat gray({2, 5}, CV_8UC1);
    for (int y = 0; y < gray.size[0]; ++y)
    {
        for (int x = 0; x < gray.size[1]; ++x)
        {
            gray.at<uchar>(y, x) = static_cast<uchar>(10 + y * 31 + x * 7);
        }
    }

    Mat bgra_expected = gray2bgra_reference<uchar>(gray);
    Mat bgra_actual;
    cvtColor(gray, bgra_actual, COLOR_GRAY2BGRA);
    ASSERT_EQ(bgra_actual.type(), CV_8UC4);
    EXPECT_EQ(max_abs_diff_u8(bgra_expected, bgra_actual), 0);

    Mat rgba_actual;
    cvtColor(gray, rgba_actual, COLOR_GRAY2RGBA);
    ASSERT_EQ(rgba_actual.type(), CV_8UC4);
    EXPECT_EQ(max_abs_diff_u8(bgra_expected, rgba_actual), 0);

    Mat gray_from_bgra_expected = color4_to_gray_reference<uchar>(bgra_actual, false);
    Mat gray_from_bgra_actual;
    cvtColor(bgra_actual, gray_from_bgra_actual, COLOR_BGRA2GRAY);
    ASSERT_EQ(gray_from_bgra_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(gray_from_bgra_expected, gray_from_bgra_actual), 0);

    Mat gray_from_rgba_expected = color4_to_gray_reference<uchar>(rgba_actual, true);
    Mat gray_from_rgba_actual;
    cvtColor(rgba_actual, gray_from_rgba_actual, COLOR_RGBA2GRAY);
    ASSERT_EQ(gray_from_rgba_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(gray_from_rgba_expected, gray_from_rgba_actual), 0);
}

TEST(ImgprocCvtColor_TEST, bgr_rgb_yuv_family_u8_matches_reference)
{
    Mat bgr({2, 5}, CV_8UC3);
    for (int y = 0; y < bgr.size[0]; ++y)
    {
        for (int x = 0; x < bgr.size[1]; ++x)
        {
            bgr.at<uchar>(y, x, 0) = static_cast<uchar>(7 + y * 29 + x * 3);
            bgr.at<uchar>(y, x, 1) = static_cast<uchar>(13 + y * 17 + x * 5);
            bgr.at<uchar>(y, x, 2) = static_cast<uchar>(19 + y * 11 + x * 7);
        }
    }

    Mat yuv_expected = color3_to_yuv_reference<uchar>(bgr, false);
    Mat yuv_actual;
    cvtColor(bgr, yuv_actual, COLOR_BGR2YUV);
    ASSERT_EQ(yuv_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(yuv_expected, yuv_actual), 0);

    Mat bgr_expected = yuv_to_color3_reference<uchar>(yuv_actual, false);
    Mat bgr_actual;
    cvtColor(yuv_actual, bgr_actual, COLOR_YUV2BGR);
    ASSERT_EQ(bgr_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_expected, bgr_actual), 0);

    Mat rgb = bgr2rgb_reference<uchar>(bgr);
    Mat yuv_from_rgb_expected = color3_to_yuv_reference<uchar>(rgb, true);
    Mat yuv_from_rgb_actual;
    cvtColor(rgb, yuv_from_rgb_actual, COLOR_RGB2YUV);
    ASSERT_EQ(yuv_from_rgb_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(yuv_from_rgb_expected, yuv_from_rgb_actual), 0);

    Mat rgb_expected = yuv_to_color3_reference<uchar>(yuv_from_rgb_actual, true);
    Mat rgb_actual;
    cvtColor(yuv_from_rgb_actual, rgb_actual, COLOR_YUV2RGB);
    ASSERT_EQ(rgb_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(rgb_expected, rgb_actual), 0);
}

TEST(ImgprocCvtColor_TEST, bgr_rgb_to_nv24_nv42_yuv444sp_u8_matches_reference)
{
    constexpr int kRows = 4;
    constexpr int kCols = 5;

    Mat bgr({kRows, kCols}, CV_8UC3);
    Mat rgb({kRows, kCols}, CV_8UC3);
    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar b = static_cast<uchar>(17 + (y * 19 + x * 11) % 200);
            const uchar g = static_cast<uchar>(33 + (y * 13 + x * 7) % 180);
            const uchar r = static_cast<uchar>(49 + (y * 9 + x * 5) % 160);
            bgr.at<uchar>(y, x, 0) = b;
            bgr.at<uchar>(y, x, 1) = g;
            bgr.at<uchar>(y, x, 2) = r;
            rgb.at<uchar>(y, x, 0) = r;
            rgb.at<uchar>(y, x, 1) = g;
            rgb.at<uchar>(y, x, 2) = b;
        }
    }

    Mat nv24_from_bgr_expected = color3_to_yuv444sp_reference_u8(bgr, false, false);
    Mat nv24_from_bgr_actual;
    cvtColor(bgr, nv24_from_bgr_actual, COLOR_BGR2YUV_NV24);
    ASSERT_EQ(nv24_from_bgr_actual.type(), CV_8UC1);
    EXPECT_EQ(nv24_from_bgr_actual.size[0], kRows * 3);
    EXPECT_EQ(nv24_from_bgr_actual.size[1], kCols);
    EXPECT_EQ(max_abs_diff_u8(nv24_from_bgr_expected, nv24_from_bgr_actual), 0);

    Mat nv42_from_bgr_expected = color3_to_yuv444sp_reference_u8(bgr, false, true);
    Mat nv42_from_bgr_actual;
    cvtColor(bgr, nv42_from_bgr_actual, COLOR_BGR2YUV_NV42);
    ASSERT_EQ(nv42_from_bgr_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(nv42_from_bgr_expected, nv42_from_bgr_actual), 0);

    Mat nv24_from_rgb_expected = color3_to_yuv444sp_reference_u8(rgb, true, false);
    Mat nv24_from_rgb_actual;
    cvtColor(rgb, nv24_from_rgb_actual, COLOR_RGB2YUV_NV24);
    ASSERT_EQ(nv24_from_rgb_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(nv24_from_rgb_expected, nv24_from_rgb_actual), 0);

    Mat nv42_from_rgb_expected = color3_to_yuv444sp_reference_u8(rgb, true, true);
    Mat nv42_from_rgb_actual;
    cvtColor(rgb, nv42_from_rgb_actual, COLOR_RGB2YUV_NV42);
    ASSERT_EQ(nv42_from_rgb_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(nv42_from_rgb_expected, nv42_from_rgb_actual), 0);
}

TEST(ImgprocCvtColor_TEST, bgr_rgb_to_i444_yv24_yuv444p_u8_matches_reference)
{
    constexpr int kRows = 4;
    constexpr int kCols = 5;

    Mat bgr({kRows, kCols}, CV_8UC3);
    Mat rgb({kRows, kCols}, CV_8UC3);
    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar b = static_cast<uchar>(21 + (y * 17 + x * 9) % 190);
            const uchar g = static_cast<uchar>(37 + (y * 11 + x * 7) % 170);
            const uchar r = static_cast<uchar>(53 + (y * 13 + x * 5) % 150);
            bgr.at<uchar>(y, x, 0) = b;
            bgr.at<uchar>(y, x, 1) = g;
            bgr.at<uchar>(y, x, 2) = r;
            rgb.at<uchar>(y, x, 0) = r;
            rgb.at<uchar>(y, x, 1) = g;
            rgb.at<uchar>(y, x, 2) = b;
        }
    }

    Mat i444_from_bgr_expected = color3_to_yuv444p_reference_u8(bgr, false, false);
    Mat i444_from_bgr_actual;
    cvtColor(bgr, i444_from_bgr_actual, COLOR_BGR2YUV_I444);
    ASSERT_EQ(i444_from_bgr_actual.type(), CV_8UC1);
    EXPECT_EQ(i444_from_bgr_actual.size[0], kRows * 3);
    EXPECT_EQ(i444_from_bgr_actual.size[1], kCols);
    EXPECT_EQ(max_abs_diff_u8(i444_from_bgr_expected, i444_from_bgr_actual), 0);

    Mat yv24_from_bgr_expected = color3_to_yuv444p_reference_u8(bgr, false, true);
    Mat yv24_from_bgr_actual;
    cvtColor(bgr, yv24_from_bgr_actual, COLOR_BGR2YUV_YV24);
    ASSERT_EQ(yv24_from_bgr_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(yv24_from_bgr_expected, yv24_from_bgr_actual), 0);

    Mat i444_from_rgb_expected = color3_to_yuv444p_reference_u8(rgb, true, false);
    Mat i444_from_rgb_actual;
    cvtColor(rgb, i444_from_rgb_actual, COLOR_RGB2YUV_I444);
    ASSERT_EQ(i444_from_rgb_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(i444_from_rgb_expected, i444_from_rgb_actual), 0);

    Mat yv24_from_rgb_expected = color3_to_yuv444p_reference_u8(rgb, true, true);
    Mat yv24_from_rgb_actual;
    cvtColor(rgb, yv24_from_rgb_actual, COLOR_RGB2YUV_YV24);
    ASSERT_EQ(yv24_from_rgb_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(yv24_from_rgb_expected, yv24_from_rgb_actual), 0);
}

TEST(ImgprocCvtColor_TEST, bgr_rgb_to_nv16_nv61_yuv422sp_u8_matches_reference)
{
    constexpr int kRows = 4;
    constexpr int kCols = 6;

    Mat bgr({kRows, kCols}, CV_8UC3);
    Mat rgb({kRows, kCols}, CV_8UC3);
    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar b = static_cast<uchar>(19 + (y * 17 + x * 9) % 180);
            const uchar g = static_cast<uchar>(35 + (y * 13 + x * 7) % 170);
            const uchar r = static_cast<uchar>(51 + (y * 11 + x * 5) % 160);
            bgr.at<uchar>(y, x, 0) = b;
            bgr.at<uchar>(y, x, 1) = g;
            bgr.at<uchar>(y, x, 2) = r;
            rgb.at<uchar>(y, x, 0) = r;
            rgb.at<uchar>(y, x, 1) = g;
            rgb.at<uchar>(y, x, 2) = b;
        }
    }

    Mat nv16_from_bgr_expected = color3_to_yuv422sp_reference_u8(bgr, false, false);
    Mat nv16_from_bgr_actual;
    cvtColor(bgr, nv16_from_bgr_actual, COLOR_BGR2YUV_NV16);
    ASSERT_EQ(nv16_from_bgr_actual.type(), CV_8UC1);
    EXPECT_EQ(nv16_from_bgr_actual.size[0], kRows * 2);
    EXPECT_EQ(nv16_from_bgr_actual.size[1], kCols);
    EXPECT_EQ(max_abs_diff_u8(nv16_from_bgr_expected, nv16_from_bgr_actual), 0);

    Mat nv61_from_bgr_expected = color3_to_yuv422sp_reference_u8(bgr, false, true);
    Mat nv61_from_bgr_actual;
    cvtColor(bgr, nv61_from_bgr_actual, COLOR_BGR2YUV_NV61);
    ASSERT_EQ(nv61_from_bgr_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(nv61_from_bgr_expected, nv61_from_bgr_actual), 0);

    Mat nv16_from_rgb_expected = color3_to_yuv422sp_reference_u8(rgb, true, false);
    Mat nv16_from_rgb_actual;
    cvtColor(rgb, nv16_from_rgb_actual, COLOR_RGB2YUV_NV16);
    ASSERT_EQ(nv16_from_rgb_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(nv16_from_rgb_expected, nv16_from_rgb_actual), 0);

    Mat nv61_from_rgb_expected = color3_to_yuv422sp_reference_u8(rgb, true, true);
    Mat nv61_from_rgb_actual;
    cvtColor(rgb, nv61_from_rgb_actual, COLOR_RGB2YUV_NV61);
    ASSERT_EQ(nv61_from_rgb_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(nv61_from_rgb_expected, nv61_from_rgb_actual), 0);
}

TEST(ImgprocCvtColor_TEST, bgr_rgb_to_yuy2_uyvy_yuv422packed_u8_matches_reference)
{
    constexpr int kRows = 4;
    constexpr int kCols = 6;

    Mat bgr({kRows, kCols}, CV_8UC3);
    Mat rgb({kRows, kCols}, CV_8UC3);
    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar b = static_cast<uchar>(23 + (y * 17 + x * 9) % 180);
            const uchar g = static_cast<uchar>(41 + (y * 13 + x * 7) % 170);
            const uchar r = static_cast<uchar>(59 + (y * 11 + x * 5) % 160);
            bgr.at<uchar>(y, x, 0) = b;
            bgr.at<uchar>(y, x, 1) = g;
            bgr.at<uchar>(y, x, 2) = r;
            rgb.at<uchar>(y, x, 0) = r;
            rgb.at<uchar>(y, x, 1) = g;
            rgb.at<uchar>(y, x, 2) = b;
        }
    }

    Mat yuy2_from_bgr_expected = color3_to_yuv422packed_reference_u8(bgr, false, false);
    Mat yuy2_from_bgr_actual;
    cvtColor(bgr, yuy2_from_bgr_actual, COLOR_BGR2YUV_YUY2);
    ASSERT_EQ(yuy2_from_bgr_actual.type(), CV_8UC2);
    EXPECT_EQ(max_abs_diff_u8(yuy2_from_bgr_expected, yuy2_from_bgr_actual), 0);

    Mat uyvy_from_bgr_expected = color3_to_yuv422packed_reference_u8(bgr, false, true);
    Mat uyvy_from_bgr_actual;
    cvtColor(bgr, uyvy_from_bgr_actual, COLOR_BGR2YUV_UYVY);
    ASSERT_EQ(uyvy_from_bgr_actual.type(), CV_8UC2);
    EXPECT_EQ(max_abs_diff_u8(uyvy_from_bgr_expected, uyvy_from_bgr_actual), 0);

    Mat yuy2_from_rgb_expected = color3_to_yuv422packed_reference_u8(rgb, true, false);
    Mat yuy2_from_rgb_actual;
    cvtColor(rgb, yuy2_from_rgb_actual, COLOR_RGB2YUV_YUY2);
    ASSERT_EQ(yuy2_from_rgb_actual.type(), CV_8UC2);
    EXPECT_EQ(max_abs_diff_u8(yuy2_from_rgb_expected, yuy2_from_rgb_actual), 0);

    Mat uyvy_from_rgb_expected = color3_to_yuv422packed_reference_u8(rgb, true, true);
    Mat uyvy_from_rgb_actual;
    cvtColor(rgb, uyvy_from_rgb_actual, COLOR_RGB2YUV_UYVY);
    ASSERT_EQ(uyvy_from_rgb_actual.type(), CV_8UC2);
    EXPECT_EQ(max_abs_diff_u8(uyvy_from_rgb_expected, uyvy_from_rgb_actual), 0);
}

TEST(ImgprocCvtColor_TEST, bgr_rgb_to_nv12_nv21_yuv420sp_u8_matches_reference)
{
    constexpr int kRows = 4;
    constexpr int kCols = 6;

    Mat bgr({kRows, kCols}, CV_8UC3);
    Mat rgb({kRows, kCols}, CV_8UC3);
    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar b = static_cast<uchar>(17 + (y * 19 + x * 9) % 180);
            const uchar g = static_cast<uchar>(33 + (y * 13 + x * 7) % 170);
            const uchar r = static_cast<uchar>(49 + (y * 11 + x * 5) % 160);
            bgr.at<uchar>(y, x, 0) = b;
            bgr.at<uchar>(y, x, 1) = g;
            bgr.at<uchar>(y, x, 2) = r;
            rgb.at<uchar>(y, x, 0) = r;
            rgb.at<uchar>(y, x, 1) = g;
            rgb.at<uchar>(y, x, 2) = b;
        }
    }

    Mat nv12_from_bgr_expected = color3_to_yuv420sp_reference_u8(bgr, false, false);
    Mat nv12_from_bgr_actual;
    cvtColor(bgr, nv12_from_bgr_actual, COLOR_BGR2YUV_NV12);
    ASSERT_EQ(nv12_from_bgr_actual.type(), CV_8UC1);
    EXPECT_EQ(nv12_from_bgr_actual.size[0], kRows * 3 / 2);
    EXPECT_EQ(nv12_from_bgr_actual.size[1], kCols);
    EXPECT_EQ(max_abs_diff_u8(nv12_from_bgr_expected, nv12_from_bgr_actual), 0);

    Mat nv21_from_bgr_expected = color3_to_yuv420sp_reference_u8(bgr, false, true);
    Mat nv21_from_bgr_actual;
    cvtColor(bgr, nv21_from_bgr_actual, COLOR_BGR2YUV_NV21);
    ASSERT_EQ(nv21_from_bgr_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(nv21_from_bgr_expected, nv21_from_bgr_actual), 0);

    Mat nv12_from_rgb_expected = color3_to_yuv420sp_reference_u8(rgb, true, false);
    Mat nv12_from_rgb_actual;
    cvtColor(rgb, nv12_from_rgb_actual, COLOR_RGB2YUV_NV12);
    ASSERT_EQ(nv12_from_rgb_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(nv12_from_rgb_expected, nv12_from_rgb_actual), 0);

    Mat nv21_from_rgb_expected = color3_to_yuv420sp_reference_u8(rgb, true, true);
    Mat nv21_from_rgb_actual;
    cvtColor(rgb, nv21_from_rgb_actual, COLOR_RGB2YUV_NV21);
    ASSERT_EQ(nv21_from_rgb_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(nv21_from_rgb_expected, nv21_from_rgb_actual), 0);
}

TEST(ImgprocCvtColor_TEST, bgr_rgb_to_i420_yv12_yuv420p_u8_matches_reference)
{
    constexpr int kRows = 4;
    constexpr int kCols = 6;

    Mat bgr({kRows, kCols}, CV_8UC3);
    Mat rgb({kRows, kCols}, CV_8UC3);
    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar b = static_cast<uchar>(21 + (y * 17 + x * 9) % 180);
            const uchar g = static_cast<uchar>(37 + (y * 11 + x * 7) % 170);
            const uchar r = static_cast<uchar>(53 + (y * 13 + x * 5) % 160);
            bgr.at<uchar>(y, x, 0) = b;
            bgr.at<uchar>(y, x, 1) = g;
            bgr.at<uchar>(y, x, 2) = r;
            rgb.at<uchar>(y, x, 0) = r;
            rgb.at<uchar>(y, x, 1) = g;
            rgb.at<uchar>(y, x, 2) = b;
        }
    }

    Mat i420_from_bgr_expected = color3_to_yuv420p_reference_u8(bgr, false, false);
    Mat i420_from_bgr_actual;
    cvtColor(bgr, i420_from_bgr_actual, COLOR_BGR2YUV_I420);
    ASSERT_EQ(i420_from_bgr_actual.type(), CV_8UC1);
    EXPECT_EQ(i420_from_bgr_actual.size[0], kRows * 3 / 2);
    EXPECT_EQ(i420_from_bgr_actual.size[1], kCols);
    EXPECT_EQ(max_abs_diff_u8(i420_from_bgr_expected, i420_from_bgr_actual), 0);

    Mat yv12_from_bgr_expected = color3_to_yuv420p_reference_u8(bgr, false, true);
    Mat yv12_from_bgr_actual;
    cvtColor(bgr, yv12_from_bgr_actual, COLOR_BGR2YUV_YV12);
    ASSERT_EQ(yv12_from_bgr_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(yv12_from_bgr_expected, yv12_from_bgr_actual), 0);

    Mat i420_from_rgb_expected = color3_to_yuv420p_reference_u8(rgb, true, false);
    Mat i420_from_rgb_actual;
    cvtColor(rgb, i420_from_rgb_actual, COLOR_RGB2YUV_I420);
    ASSERT_EQ(i420_from_rgb_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(i420_from_rgb_expected, i420_from_rgb_actual), 0);

    Mat yv12_from_rgb_expected = color3_to_yuv420p_reference_u8(rgb, true, true);
    Mat yv12_from_rgb_actual;
    cvtColor(rgb, yv12_from_rgb_actual, COLOR_RGB2YUV_YV12);
    ASSERT_EQ(yv12_from_rgb_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(yv12_from_rgb_expected, yv12_from_rgb_actual), 0);
}

TEST(ImgprocCvtColor_TEST, nv12_nv21_yuv420sp_u8_matches_reference)
{
    constexpr int kRows = 4;
    constexpr int kCols = 6;

    Mat nv12({kRows * 3 / 2, kCols}, CV_8UC1);
    Mat nv21({kRows * 3 / 2, kCols}, CV_8UC1);
    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar yy = static_cast<uchar>(16 + (y * 23 + x * 11) % 220);
            nv12.at<uchar>(y, x) = yy;
            nv21.at<uchar>(y, x) = yy;
        }
    }

    for (int y = 0; y < kRows / 2; ++y)
    {
        for (int x = 0; x < kCols; x += 2)
        {
            const uchar uu = static_cast<uchar>(40 + (y * 17 + x * 9) % 160);
            const uchar vv = static_cast<uchar>(60 + (y * 13 + x * 7) % 150);

            nv12.at<uchar>(kRows + y, x + 0) = uu;
            nv12.at<uchar>(kRows + y, x + 1) = vv;
            nv21.at<uchar>(kRows + y, x + 0) = vv;
            nv21.at<uchar>(kRows + y, x + 1) = uu;
        }
    }

    Mat bgr_nv12_expected = yuv420sp_to_color3_reference_u8(nv12, false, false);
    Mat bgr_nv12_actual;
    cvtColor(nv12, bgr_nv12_actual, COLOR_YUV2BGR_NV12);
    ASSERT_EQ(bgr_nv12_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_nv12_expected, bgr_nv12_actual), 0);

    Mat rgb_nv12_expected = yuv420sp_to_color3_reference_u8(nv12, false, true);
    Mat rgb_nv12_actual;
    cvtColor(nv12, rgb_nv12_actual, COLOR_YUV2RGB_NV12);
    ASSERT_EQ(rgb_nv12_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(rgb_nv12_expected, rgb_nv12_actual), 0);

    Mat bgr_nv21_expected = yuv420sp_to_color3_reference_u8(nv21, true, false);
    Mat bgr_nv21_actual;
    cvtColor(nv21, bgr_nv21_actual, COLOR_YUV2BGR_NV21);
    ASSERT_EQ(bgr_nv21_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_nv21_expected, bgr_nv21_actual), 0);

    Mat rgb_nv21_expected = yuv420sp_to_color3_reference_u8(nv21, true, true);
    Mat rgb_nv21_actual;
    cvtColor(nv21, rgb_nv21_actual, COLOR_YUV2RGB_NV21);
    ASSERT_EQ(rgb_nv21_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(rgb_nv21_expected, rgb_nv21_actual), 0);
}

TEST(ImgprocCvtColor_TEST, i420_yv12_yuv420p_u8_matches_reference)
{
    constexpr int kRows = 4;
    constexpr int kCols = 6;
    constexpr int kUvSize = kRows * kCols / 4;

    Mat i420({kRows * 3 / 2, kCols}, CV_8UC1);
    Mat yv12({kRows * 3 / 2, kCols}, CV_8UC1);
    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar yy = static_cast<uchar>(16 + (y * 19 + x * 13) % 220);
            i420.at<uchar>(y, x) = yy;
            yv12.at<uchar>(y, x) = yy;
        }
    }

    for (int i = 0; i < kUvSize; ++i)
    {
        const uchar uu = static_cast<uchar>(44 + (i * 9) % 160);
        const uchar vv = static_cast<uchar>(58 + (i * 11) % 150);
        set_yuv420p_plane_byte_u8(i420, kRows, kCols, 0, i, uu);
        set_yuv420p_plane_byte_u8(i420, kRows, kCols, kUvSize, i, vv);
        set_yuv420p_plane_byte_u8(yv12, kRows, kCols, 0, i, vv);
        set_yuv420p_plane_byte_u8(yv12, kRows, kCols, kUvSize, i, uu);
    }

    Mat bgr_i420_expected = yuv420p_to_color3_reference_u8(i420, false, false);
    Mat bgr_i420_actual;
    cvtColor(i420, bgr_i420_actual, COLOR_YUV2BGR_I420);
    ASSERT_EQ(bgr_i420_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_i420_expected, bgr_i420_actual), 0);

    Mat rgb_i420_expected = yuv420p_to_color3_reference_u8(i420, false, true);
    Mat rgb_i420_actual;
    cvtColor(i420, rgb_i420_actual, COLOR_YUV2RGB_I420);
    ASSERT_EQ(rgb_i420_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(rgb_i420_expected, rgb_i420_actual), 0);

    Mat bgr_yv12_expected = yuv420p_to_color3_reference_u8(yv12, true, false);
    Mat bgr_yv12_actual;
    cvtColor(yv12, bgr_yv12_actual, COLOR_YUV2BGR_YV12);
    ASSERT_EQ(bgr_yv12_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_yv12_expected, bgr_yv12_actual), 0);

    Mat rgb_yv12_expected = yuv420p_to_color3_reference_u8(yv12, true, true);
    Mat rgb_yv12_actual;
    cvtColor(yv12, rgb_yv12_actual, COLOR_YUV2RGB_YV12);
    ASSERT_EQ(rgb_yv12_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(rgb_yv12_expected, rgb_yv12_actual), 0);
}

TEST(ImgprocCvtColor_TEST, nv16_nv61_yuv422sp_u8_matches_reference)
{
    constexpr int kRows = 5;
    constexpr int kCols = 6;

    Mat nv16({kRows * 2, kCols}, CV_8UC1);
    Mat nv61({kRows * 2, kCols}, CV_8UC1);
    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar yy = static_cast<uchar>(20 + (y * 19 + x * 11) % 200);
            nv16.at<uchar>(y, x) = yy;
            nv61.at<uchar>(y, x) = yy;
        }
    }

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; x += 2)
        {
            const uchar uu = static_cast<uchar>(46 + (y * 13 + x * 5) % 150);
            const uchar vv = static_cast<uchar>(60 + (y * 7 + x * 9) % 140);
            nv16.at<uchar>(kRows + y, x + 0) = uu;
            nv16.at<uchar>(kRows + y, x + 1) = vv;
            nv61.at<uchar>(kRows + y, x + 0) = vv;
            nv61.at<uchar>(kRows + y, x + 1) = uu;
        }
    }

    Mat bgr_nv16_expected = yuv422sp_to_color3_reference_u8(nv16, false, false);
    Mat bgr_nv16_actual;
    cvtColor(nv16, bgr_nv16_actual, COLOR_YUV2BGR_NV16);
    ASSERT_EQ(bgr_nv16_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_nv16_expected, bgr_nv16_actual), 0);

    Mat rgb_nv16_expected = yuv422sp_to_color3_reference_u8(nv16, false, true);
    Mat rgb_nv16_actual;
    cvtColor(nv16, rgb_nv16_actual, COLOR_YUV2RGB_NV16);
    ASSERT_EQ(rgb_nv16_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(rgb_nv16_expected, rgb_nv16_actual), 0);

    Mat bgr_nv61_expected = yuv422sp_to_color3_reference_u8(nv61, true, false);
    Mat bgr_nv61_actual;
    cvtColor(nv61, bgr_nv61_actual, COLOR_YUV2BGR_NV61);
    ASSERT_EQ(bgr_nv61_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_nv61_expected, bgr_nv61_actual), 0);

    Mat rgb_nv61_expected = yuv422sp_to_color3_reference_u8(nv61, true, true);
    Mat rgb_nv61_actual;
    cvtColor(nv61, rgb_nv61_actual, COLOR_YUV2RGB_NV61);
    ASSERT_EQ(rgb_nv61_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(rgb_nv61_expected, rgb_nv61_actual), 0);
}

TEST(ImgprocCvtColor_TEST, i444_yv24_yuv444p_u8_matches_reference)
{
    constexpr int kRows = 4;
    constexpr int kCols = 5;
    constexpr int kPlaneSize = kRows * kCols;

    Mat i444({kRows * 3, kCols}, CV_8UC1);
    Mat yv24({kRows * 3, kCols}, CV_8UC1);
    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar yy = static_cast<uchar>(20 + (y * 19 + x * 13) % 200);
            i444.at<uchar>(y, x) = yy;
            yv24.at<uchar>(y, x) = yy;
        }
    }

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const int chroma_index = y * kCols + x;
            const uchar uu = static_cast<uchar>(48 + (y * 11 + x * 7) % 150);
            const uchar vv = static_cast<uchar>(62 + (y * 17 + x * 5) % 140);
            set_yuv444p_plane_byte_u8(i444, kRows, kCols, 0, chroma_index, uu);
            set_yuv444p_plane_byte_u8(i444, kRows, kCols, kPlaneSize, chroma_index, vv);
            set_yuv444p_plane_byte_u8(yv24, kRows, kCols, 0, chroma_index, vv);
            set_yuv444p_plane_byte_u8(yv24, kRows, kCols, kPlaneSize, chroma_index, uu);
        }
    }

    Mat bgr_i444_expected = yuv444p_to_color3_reference_u8(i444, false, false);
    Mat bgr_i444_actual;
    cvtColor(i444, bgr_i444_actual, COLOR_YUV2BGR_I444);
    ASSERT_EQ(bgr_i444_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_i444_expected, bgr_i444_actual), 0);

    Mat rgb_i444_expected = yuv444p_to_color3_reference_u8(i444, false, true);
    Mat rgb_i444_actual;
    cvtColor(i444, rgb_i444_actual, COLOR_YUV2RGB_I444);
    ASSERT_EQ(rgb_i444_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(rgb_i444_expected, rgb_i444_actual), 0);

    Mat bgr_yv24_expected = yuv444p_to_color3_reference_u8(yv24, true, false);
    Mat bgr_yv24_actual;
    cvtColor(yv24, bgr_yv24_actual, COLOR_YUV2BGR_YV24);
    ASSERT_EQ(bgr_yv24_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_yv24_expected, bgr_yv24_actual), 0);

    Mat rgb_yv24_expected = yuv444p_to_color3_reference_u8(yv24, true, true);
    Mat rgb_yv24_actual;
    cvtColor(yv24, rgb_yv24_actual, COLOR_YUV2RGB_YV24);
    ASSERT_EQ(rgb_yv24_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(rgb_yv24_expected, rgb_yv24_actual), 0);
}

TEST(ImgprocCvtColor_TEST, nv24_nv42_yuv444sp_u8_matches_reference)
{
    constexpr int kRows = 4;
    constexpr int kCols = 5;

    Mat nv24({kRows * 3, kCols}, CV_8UC1);
    Mat nv42({kRows * 3, kCols}, CV_8UC1);
    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar yy = static_cast<uchar>(18 + (y * 23 + x * 13) % 210);
            nv24.at<uchar>(y, x) = yy;
            nv42.at<uchar>(y, x) = yy;
        }
    }

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar uu = static_cast<uchar>(44 + (y * 17 + x * 7) % 160);
            const uchar vv = static_cast<uchar>(58 + (y * 11 + x * 9) % 150);
            const int base = y * (kCols * 2) + x * 2;
            set_yuv444sp_plane_byte_u8(nv24, kRows, kCols, base + 0, uu);
            set_yuv444sp_plane_byte_u8(nv24, kRows, kCols, base + 1, vv);
            set_yuv444sp_plane_byte_u8(nv42, kRows, kCols, base + 0, vv);
            set_yuv444sp_plane_byte_u8(nv42, kRows, kCols, base + 1, uu);
        }
    }

    Mat bgr_nv24_expected = yuv444sp_to_color3_reference_u8(nv24, false, false);
    Mat bgr_nv24_actual;
    cvtColor(nv24, bgr_nv24_actual, COLOR_YUV2BGR_NV24);
    ASSERT_EQ(bgr_nv24_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_nv24_expected, bgr_nv24_actual), 0);

    Mat rgb_nv24_expected = yuv444sp_to_color3_reference_u8(nv24, false, true);
    Mat rgb_nv24_actual;
    cvtColor(nv24, rgb_nv24_actual, COLOR_YUV2RGB_NV24);
    ASSERT_EQ(rgb_nv24_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(rgb_nv24_expected, rgb_nv24_actual), 0);

    Mat bgr_nv42_expected = yuv444sp_to_color3_reference_u8(nv42, true, false);
    Mat bgr_nv42_actual;
    cvtColor(nv42, bgr_nv42_actual, COLOR_YUV2BGR_NV42);
    ASSERT_EQ(bgr_nv42_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_nv42_expected, bgr_nv42_actual), 0);

    Mat rgb_nv42_expected = yuv444sp_to_color3_reference_u8(nv42, true, true);
    Mat rgb_nv42_actual;
    cvtColor(nv42, rgb_nv42_actual, COLOR_YUV2RGB_NV42);
    ASSERT_EQ(rgb_nv42_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(rgb_nv42_expected, rgb_nv42_actual), 0);
}

TEST(ImgprocCvtColor_TEST, yuy2_uyvy_yuv422packed_u8_matches_reference)
{
    constexpr int kRows = 5;
    constexpr int kCols = 6;

    Mat yuy2({kRows, kCols}, CV_8UC2);
    Mat uyvy({kRows, kCols}, CV_8UC2);
    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; x += 2)
        {
            const uchar yy0 = static_cast<uchar>(22 + (y * 17 + x * 9) % 190);
            const uchar yy1 = static_cast<uchar>(35 + (y * 11 + x * 7) % 180);
            const uchar uu = static_cast<uchar>(48 + (y * 13 + x * 5) % 150);
            const uchar vv = static_cast<uchar>(62 + (y * 7 + x * 3) % 140);
            set_yuv422_packed_pair_u8(yuy2, y, x, false, yy0, yy1, uu, vv);
            set_yuv422_packed_pair_u8(uyvy, y, x, true, yy0, yy1, uu, vv);
        }
    }

    Mat bgr_yuy2_expected = yuv422packed_to_color3_reference_u8(yuy2, false, false);
    Mat bgr_yuy2_actual;
    cvtColor(yuy2, bgr_yuy2_actual, COLOR_YUV2BGR_YUY2);
    ASSERT_EQ(bgr_yuy2_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_yuy2_expected, bgr_yuy2_actual), 0);

    Mat rgb_yuy2_expected = yuv422packed_to_color3_reference_u8(yuy2, false, true);
    Mat rgb_yuy2_actual;
    cvtColor(yuy2, rgb_yuy2_actual, COLOR_YUV2RGB_YUY2);
    ASSERT_EQ(rgb_yuy2_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(rgb_yuy2_expected, rgb_yuy2_actual), 0);

    Mat bgr_uyvy_expected = yuv422packed_to_color3_reference_u8(uyvy, true, false);
    Mat bgr_uyvy_actual;
    cvtColor(uyvy, bgr_uyvy_actual, COLOR_YUV2BGR_UYVY);
    ASSERT_EQ(bgr_uyvy_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_uyvy_expected, bgr_uyvy_actual), 0);

    Mat rgb_uyvy_expected = yuv422packed_to_color3_reference_u8(uyvy, true, true);
    Mat rgb_uyvy_actual;
    cvtColor(uyvy, rgb_uyvy_actual, COLOR_YUV2RGB_UYVY);
    ASSERT_EQ(rgb_uyvy_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(rgb_uyvy_expected, rgb_uyvy_actual), 0);
}

TEST(ImgprocCvtColor_TEST, throws_on_invalid_input_channels_or_code)
{
    // Ported idea from OpenCV:
    // modules/imgproc/test/test_color.cpp
    // TEST(ImgProc_cvtColor_InvalidNumOfChannels, regression_25971)
    Mat src_gray({8, 8}, CV_8UC1);
    Mat src_bgr({8, 8}, CV_8UC3);
    Mat dst;

    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_BGR2GRAY), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, COLOR_GRAY2BGR), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_BGR2RGB), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_BGR2BGRA), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_RGB2RGBA), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_BGR2RGBA), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, COLOR_RGBA2RGB), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, COLOR_BGRA2RGBA), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, COLOR_GRAY2BGRA), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, COLOR_GRAY2RGBA), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_BGRA2GRAY), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_RGBA2GRAY), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_BGR2YUV), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_RGB2YUV), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_BGR2YUV_NV24), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_RGB2YUV_NV24), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_BGR2YUV_NV42), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_RGB2YUV_NV42), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2BGR), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2RGB), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2BGR_NV12), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2RGB_NV12), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2BGR_NV21), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2RGB_NV21), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2BGR_I420), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2RGB_I420), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2BGR_YV12), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2RGB_YV12), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, COLOR_YUV2BGR_NV24), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, COLOR_YUV2RGB_NV24), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, COLOR_YUV2BGR_NV42), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, COLOR_YUV2RGB_NV42), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, COLOR_YUV2BGR_NV16), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, COLOR_YUV2RGB_NV16), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, COLOR_YUV2BGR_NV61), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, COLOR_YUV2RGB_NV61), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2BGR_YUY2), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2RGB_YUY2), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2BGR_UYVY), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2RGB_UYVY), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, -999), Exception);

    Mat src_u16({8, 8}, CV_16UC3);
    EXPECT_THROW(cvtColor(src_u16, dst, COLOR_BGR2GRAY), Exception);
}

TEST(ImgprocCvtColor_TEST, non_contiguous_roi_matches_reference)
{
    Mat base_bgr({7, 11}, CV_8UC3);
    for (int y = 0; y < base_bgr.size[0]; ++y)
    {
        for (int x = 0; x < base_bgr.size[1]; ++x)
        {
            base_bgr.at<uchar>(y, x, 0) = static_cast<uchar>((y * 13 + x * 3 + 1) % 256);
            base_bgr.at<uchar>(y, x, 1) = static_cast<uchar>((y * 5 + x * 17 + 2) % 256);
            base_bgr.at<uchar>(y, x, 2) = static_cast<uchar>((y * 19 + x * 7 + 3) % 256);
        }
    }
    Mat bgr_roi = base_bgr.colRange(2, 10);
    ASSERT_FALSE(bgr_roi.isContinuous());

    Mat gray_expected = bgr2gray_reference_u8(bgr_roi);
    Mat gray_actual;
    cvtColor(bgr_roi, gray_actual, COLOR_BGR2GRAY);
    EXPECT_EQ(max_abs_diff_u8(gray_expected, gray_actual), 0);

    Mat base_gray({6, 12}, CV_8UC1);
    for (int y = 0; y < base_gray.size[0]; ++y)
    {
        for (int x = 0; x < base_gray.size[1]; ++x)
        {
            base_gray.at<uchar>(y, x) = static_cast<uchar>((y * 29 + x * 11 + 7) % 256);
        }
    }
    Mat gray_roi = base_gray.colRange(1, 10);
    ASSERT_FALSE(gray_roi.isContinuous());

    Mat bgr_expected = gray2bgr_reference_u8(gray_roi);
    Mat bgr_actual;
    cvtColor(gray_roi, bgr_actual, COLOR_GRAY2BGR);
    EXPECT_EQ(max_abs_diff_u8(bgr_expected, bgr_actual), 0);
}

TEST(ImgprocCvtColor_TEST, non_contiguous_roi_for_rgb_and_bgra_paths_matches_reference)
{
    Mat base_bgr({6, 10}, CV_8UC3);
    for (int y = 0; y < base_bgr.size[0]; ++y)
    {
        for (int x = 0; x < base_bgr.size[1]; ++x)
        {
            base_bgr.at<uchar>(y, x, 0) = static_cast<uchar>((y * 11 + x * 3 + 1) % 256);
            base_bgr.at<uchar>(y, x, 1) = static_cast<uchar>((y * 7 + x * 5 + 2) % 256);
            base_bgr.at<uchar>(y, x, 2) = static_cast<uchar>((y * 13 + x * 9 + 3) % 256);
        }
    }
    Mat bgr_roi = base_bgr.colRange(1, 9);
    ASSERT_FALSE(bgr_roi.isContinuous());

    Mat rgb_expected = bgr2rgb_reference<uchar>(bgr_roi);
    Mat rgb_actual;
    cvtColor(bgr_roi, rgb_actual, COLOR_BGR2RGB);
    EXPECT_EQ(max_abs_diff_u8(rgb_expected, rgb_actual), 0);

    Mat base_bgr_f32({7, 11}, CV_32FC3);
    for (int y = 0; y < base_bgr_f32.size[0]; ++y)
    {
        for (int x = 0; x < base_bgr_f32.size[1]; ++x)
        {
            base_bgr_f32.at<float>(y, x, 0) = static_cast<float>(y * 0.30 - x * 0.12 + 0.75);
            base_bgr_f32.at<float>(y, x, 1) = static_cast<float>(y * 0.55 + x * 0.18 - 1.25);
            base_bgr_f32.at<float>(y, x, 2) = static_cast<float>(y * 0.08 + x * 0.72 + 2.50);
        }
    }
    Mat bgr_f32_roi = base_bgr_f32.colRange(2, 10);
    ASSERT_FALSE(bgr_f32_roi.isContinuous());

    Mat bgra_expected = bgr2bgra_reference<float>(bgr_f32_roi);
    Mat bgra_actual;
    cvtColor(bgr_f32_roi, bgra_actual, COLOR_BGR2BGRA);
    EXPECT_LE(max_abs_diff_f32(bgra_expected, bgra_actual), 1e-6f);

    Mat bgr_expected = bgra2bgr_reference<float>(bgra_actual);
    Mat bgr_actual;
    cvtColor(bgra_actual, bgr_actual, COLOR_BGRA2BGR);
    EXPECT_LE(max_abs_diff_f32(bgr_expected, bgr_actual), 1e-6f);
}

TEST(ImgprocCvtColor_TEST, non_contiguous_roi_for_rgba_family_matches_reference)
{
    Mat base_rgb({6, 10}, CV_8UC3);
    for (int y = 0; y < base_rgb.size[0]; ++y)
    {
        for (int x = 0; x < base_rgb.size[1]; ++x)
        {
            base_rgb.at<uchar>(y, x, 0) = static_cast<uchar>((y * 13 + x * 3 + 5) % 256);
            base_rgb.at<uchar>(y, x, 1) = static_cast<uchar>((y * 7 + x * 11 + 9) % 256);
            base_rgb.at<uchar>(y, x, 2) = static_cast<uchar>((y * 5 + x * 17 + 1) % 256);
        }
    }
    Mat rgb_roi = base_rgb.colRange(1, 9);
    ASSERT_FALSE(rgb_roi.isContinuous());

    Mat rgba_expected = rgb2rgba_reference<uchar>(rgb_roi);
    Mat rgba_actual;
    cvtColor(rgb_roi, rgba_actual, COLOR_RGB2RGBA);
    EXPECT_EQ(max_abs_diff_u8(rgba_expected, rgba_actual), 0);

    Mat bgra_expected = rgb2bgra_reference<uchar>(rgb_roi);
    Mat bgra_actual;
    cvtColor(rgb_roi, bgra_actual, COLOR_RGB2BGRA);
    EXPECT_EQ(max_abs_diff_u8(bgra_expected, bgra_actual), 0);

    Mat base_rgba_f32({7, 11}, CV_32FC4);
    for (int y = 0; y < base_rgba_f32.size[0]; ++y)
    {
        for (int x = 0; x < base_rgba_f32.size[1]; ++x)
        {
            base_rgba_f32.at<float>(y, x, 0) = static_cast<float>(y * 0.35 + x * 0.07 - 1.00);
            base_rgba_f32.at<float>(y, x, 1) = static_cast<float>(y * 0.12 - x * 0.28 + 2.25);
            base_rgba_f32.at<float>(y, x, 2) = static_cast<float>(y * 0.50 + x * 0.21 - 0.75);
            base_rgba_f32.at<float>(y, x, 3) = static_cast<float>(0.25 + y * 0.03 + x * 0.04);
        }
    }
    Mat rgba_roi = base_rgba_f32.colRange(2, 10);
    ASSERT_FALSE(rgba_roi.isContinuous());

    Mat rgb_expected = rgba2rgb_reference<float>(rgba_roi);
    Mat rgb_actual;
    cvtColor(rgba_roi, rgb_actual, COLOR_RGBA2RGB);
    EXPECT_LE(max_abs_diff_f32(rgb_expected, rgb_actual), 1e-6f);

    Mat bgr_expected = rgba2bgr_reference<float>(rgba_roi);
    Mat bgr_actual;
    cvtColor(rgba_roi, bgr_actual, COLOR_RGBA2BGR);
    EXPECT_LE(max_abs_diff_f32(bgr_expected, bgr_actual), 1e-6f);

    Mat bgra_expected_f32 = swap_rb_4ch_reference<float>(rgba_roi);
    Mat bgra_actual_f32;
    cvtColor(rgba_roi, bgra_actual_f32, COLOR_RGBA2BGRA);
    EXPECT_LE(max_abs_diff_f32(bgra_expected_f32, bgra_actual_f32), 1e-6f);
}

TEST(ImgprocCvtColor_TEST, non_contiguous_roi_for_gray_rgba_family_matches_reference)
{
    Mat base_gray({6, 10}, CV_8UC1);
    for (int y = 0; y < base_gray.size[0]; ++y)
    {
        for (int x = 0; x < base_gray.size[1]; ++x)
        {
            base_gray.at<uchar>(y, x) = static_cast<uchar>((y * 19 + x * 13 + 4) % 256);
        }
    }
    Mat gray_roi = base_gray.colRange(1, 9);
    ASSERT_FALSE(gray_roi.isContinuous());

    Mat bgra_expected = gray2bgra_reference<uchar>(gray_roi);
    Mat bgra_actual;
    cvtColor(gray_roi, bgra_actual, COLOR_GRAY2BGRA);
    EXPECT_EQ(max_abs_diff_u8(bgra_expected, bgra_actual), 0);

    Mat rgba_actual;
    cvtColor(gray_roi, rgba_actual, COLOR_GRAY2RGBA);
    EXPECT_EQ(max_abs_diff_u8(bgra_expected, rgba_actual), 0);

    Mat base_bgra_f32({7, 11}, CV_32FC4);
    for (int y = 0; y < base_bgra_f32.size[0]; ++y)
    {
        for (int x = 0; x < base_bgra_f32.size[1]; ++x)
        {
            base_bgra_f32.at<float>(y, x, 0) = static_cast<float>(y * 0.25 + x * 0.05 - 1.5);
            base_bgra_f32.at<float>(y, x, 1) = static_cast<float>(y * 0.18 - x * 0.12 + 0.5);
            base_bgra_f32.at<float>(y, x, 2) = static_cast<float>(y * 0.07 + x * 0.31 + 2.0);
            base_bgra_f32.at<float>(y, x, 3) = static_cast<float>(0.2 + y * 0.01 + x * 0.03);
        }
    }
    Mat bgra_roi = base_bgra_f32.colRange(2, 10);
    ASSERT_FALSE(bgra_roi.isContinuous());

    Mat gray_from_bgra_expected = color4_to_gray_reference<float>(bgra_roi, false);
    Mat gray_from_bgra_actual;
    cvtColor(bgra_roi, gray_from_bgra_actual, COLOR_BGRA2GRAY);
    EXPECT_LE(max_abs_diff_f32(gray_from_bgra_expected, gray_from_bgra_actual), 1e-6f);

    Mat rgba_roi = swap_rb_4ch_reference<float>(bgra_roi);
    Mat gray_from_rgba_expected = color4_to_gray_reference<float>(rgba_roi, true);
    Mat gray_from_rgba_actual;
    cvtColor(rgba_roi, gray_from_rgba_actual, COLOR_RGBA2GRAY);
    EXPECT_LE(max_abs_diff_f32(gray_from_rgba_expected, gray_from_rgba_actual), 1e-6f);
}

TEST(ImgprocCvtColor_TEST, non_contiguous_roi_for_yuv_family_matches_reference)
{
    Mat base_bgr({6, 10}, CV_8UC3);
    for (int y = 0; y < base_bgr.size[0]; ++y)
    {
        for (int x = 0; x < base_bgr.size[1]; ++x)
        {
            base_bgr.at<uchar>(y, x, 0) = static_cast<uchar>((y * 23 + x * 3 + 1) % 256);
            base_bgr.at<uchar>(y, x, 1) = static_cast<uchar>((y * 17 + x * 5 + 2) % 256);
            base_bgr.at<uchar>(y, x, 2) = static_cast<uchar>((y * 11 + x * 7 + 3) % 256);
        }
    }
    Mat bgr_roi = base_bgr.colRange(1, 9);
    ASSERT_FALSE(bgr_roi.isContinuous());

    Mat yuv_expected = color3_to_yuv_reference<uchar>(bgr_roi, false);
    Mat yuv_actual;
    cvtColor(bgr_roi, yuv_actual, COLOR_BGR2YUV);
    EXPECT_EQ(max_abs_diff_u8(yuv_expected, yuv_actual), 0);

    Mat base_yuv_f32({7, 11}, CV_32FC3);
    for (int y = 0; y < base_yuv_f32.size[0]; ++y)
    {
        for (int x = 0; x < base_yuv_f32.size[1]; ++x)
        {
            base_yuv_f32.at<float>(y, x, 0) = static_cast<float>(0.15 + y * 0.04 + x * 0.03);
            base_yuv_f32.at<float>(y, x, 1) = static_cast<float>(0.50 - y * 0.02 + x * 0.01);
            base_yuv_f32.at<float>(y, x, 2) = static_cast<float>(0.45 + y * 0.03 - x * 0.02);
        }
    }
    Mat yuv_roi = base_yuv_f32.colRange(2, 10);
    ASSERT_FALSE(yuv_roi.isContinuous());

    Mat rgb_expected = yuv_to_color3_reference<float>(yuv_roi, true);
    Mat rgb_actual;
    cvtColor(yuv_roi, rgb_actual, COLOR_YUV2RGB);
    EXPECT_LE(max_abs_diff_f32(rgb_expected, rgb_actual), 1e-6f);
}

TEST(ImgprocCvtColor_TEST, non_contiguous_roi_for_nv24_nv42_encode_matches_reference)
{
    constexpr int kRows = 5;
    constexpr int kCols = 6;

    Mat base_bgr({kRows, kCols + 3}, CV_8UC3);
    Mat bgr_roi = base_bgr.colRange(1, 1 + kCols);
    ASSERT_FALSE(bgr_roi.isContinuous());

    Mat base_rgb({kRows, kCols + 3}, CV_8UC3);
    Mat rgb_roi = base_rgb.colRange(1, 1 + kCols);
    ASSERT_FALSE(rgb_roi.isContinuous());

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar b = static_cast<uchar>(23 + (y * 17 + x * 7) % 190);
            const uchar g = static_cast<uchar>(41 + (y * 11 + x * 5) % 170);
            const uchar r = static_cast<uchar>(59 + (y * 13 + x * 9) % 150);
            bgr_roi.at<uchar>(y, x, 0) = b;
            bgr_roi.at<uchar>(y, x, 1) = g;
            bgr_roi.at<uchar>(y, x, 2) = r;
            rgb_roi.at<uchar>(y, x, 0) = r;
            rgb_roi.at<uchar>(y, x, 1) = g;
            rgb_roi.at<uchar>(y, x, 2) = b;
        }
    }

    Mat nv24_expected = color3_to_yuv444sp_reference_u8(bgr_roi, false, false);
    Mat nv24_actual;
    cvtColor(bgr_roi, nv24_actual, COLOR_BGR2YUV_NV24);
    EXPECT_EQ(max_abs_diff_u8(nv24_expected, nv24_actual), 0);

    Mat nv42_expected = color3_to_yuv444sp_reference_u8(rgb_roi, true, true);
    Mat nv42_actual;
    cvtColor(rgb_roi, nv42_actual, COLOR_RGB2YUV_NV42);
    EXPECT_EQ(max_abs_diff_u8(nv42_expected, nv42_actual), 0);
}

TEST(ImgprocCvtColor_TEST, non_contiguous_roi_for_i444_yv24_encode_matches_reference)
{
    constexpr int kRows = 5;
    constexpr int kCols = 6;

    Mat base_bgr({kRows, kCols + 3}, CV_8UC3);
    Mat bgr_roi = base_bgr.colRange(1, 1 + kCols);
    ASSERT_FALSE(bgr_roi.isContinuous());

    Mat base_rgb({kRows, kCols + 3}, CV_8UC3);
    Mat rgb_roi = base_rgb.colRange(1, 1 + kCols);
    ASSERT_FALSE(rgb_roi.isContinuous());

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar b = static_cast<uchar>(25 + (y * 19 + x * 7) % 180);
            const uchar g = static_cast<uchar>(43 + (y * 13 + x * 5) % 170);
            const uchar r = static_cast<uchar>(61 + (y * 11 + x * 9) % 150);
            bgr_roi.at<uchar>(y, x, 0) = b;
            bgr_roi.at<uchar>(y, x, 1) = g;
            bgr_roi.at<uchar>(y, x, 2) = r;
            rgb_roi.at<uchar>(y, x, 0) = r;
            rgb_roi.at<uchar>(y, x, 1) = g;
            rgb_roi.at<uchar>(y, x, 2) = b;
        }
    }

    Mat i444_expected = color3_to_yuv444p_reference_u8(bgr_roi, false, false);
    Mat i444_actual;
    cvtColor(bgr_roi, i444_actual, COLOR_BGR2YUV_I444);
    EXPECT_EQ(max_abs_diff_u8(i444_expected, i444_actual), 0);

    Mat yv24_expected = color3_to_yuv444p_reference_u8(rgb_roi, true, true);
    Mat yv24_actual;
    cvtColor(rgb_roi, yv24_actual, COLOR_RGB2YUV_YV24);
    EXPECT_EQ(max_abs_diff_u8(yv24_expected, yv24_actual), 0);
}

TEST(ImgprocCvtColor_TEST, non_contiguous_roi_for_nv16_nv61_encode_matches_reference)
{
    constexpr int kRows = 5;
    constexpr int kCols = 8;

    Mat base_bgr({kRows, kCols + 4}, CV_8UC3);
    Mat bgr_roi = base_bgr.colRange(2, 2 + kCols);
    ASSERT_FALSE(bgr_roi.isContinuous());

    Mat base_rgb({kRows, kCols + 4}, CV_8UC3);
    Mat rgb_roi = base_rgb.colRange(2, 2 + kCols);
    ASSERT_FALSE(rgb_roi.isContinuous());

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar b = static_cast<uchar>(27 + (y * 19 + x * 9) % 170);
            const uchar g = static_cast<uchar>(45 + (y * 11 + x * 7) % 160);
            const uchar r = static_cast<uchar>(63 + (y * 13 + x * 5) % 150);
            bgr_roi.at<uchar>(y, x, 0) = b;
            bgr_roi.at<uchar>(y, x, 1) = g;
            bgr_roi.at<uchar>(y, x, 2) = r;
            rgb_roi.at<uchar>(y, x, 0) = r;
            rgb_roi.at<uchar>(y, x, 1) = g;
            rgb_roi.at<uchar>(y, x, 2) = b;
        }
    }

    Mat nv16_expected = color3_to_yuv422sp_reference_u8(bgr_roi, false, false);
    Mat nv16_actual;
    cvtColor(bgr_roi, nv16_actual, COLOR_BGR2YUV_NV16);
    EXPECT_EQ(max_abs_diff_u8(nv16_expected, nv16_actual), 0);

    Mat nv61_expected = color3_to_yuv422sp_reference_u8(rgb_roi, true, true);
    Mat nv61_actual;
    cvtColor(rgb_roi, nv61_actual, COLOR_RGB2YUV_NV61);
    EXPECT_EQ(max_abs_diff_u8(nv61_expected, nv61_actual), 0);
}

TEST(ImgprocCvtColor_TEST, non_contiguous_roi_for_yuy2_uyvy_encode_matches_reference)
{
    constexpr int kRows = 6;
    constexpr int kCols = 8;

    Mat base_bgr({kRows, kCols + 3}, CV_8UC3);
    Mat bgr_roi = base_bgr.colRange(1, 1 + kCols);
    ASSERT_FALSE(bgr_roi.isContinuous());

    Mat base_rgb({kRows, kCols + 3}, CV_8UC3);
    Mat rgb_roi = base_rgb.colRange(1, 1 + kCols);
    ASSERT_FALSE(rgb_roi.isContinuous());

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar b = static_cast<uchar>(31 + (y * 19 + x * 9) % 170);
            const uchar g = static_cast<uchar>(49 + (y * 11 + x * 7) % 160);
            const uchar r = static_cast<uchar>(67 + (y * 13 + x * 5) % 150);
            bgr_roi.at<uchar>(y, x, 0) = b;
            bgr_roi.at<uchar>(y, x, 1) = g;
            bgr_roi.at<uchar>(y, x, 2) = r;
            rgb_roi.at<uchar>(y, x, 0) = r;
            rgb_roi.at<uchar>(y, x, 1) = g;
            rgb_roi.at<uchar>(y, x, 2) = b;
        }
    }

    Mat yuy2_expected = color3_to_yuv422packed_reference_u8(bgr_roi, false, false);
    Mat yuy2_actual;
    cvtColor(bgr_roi, yuy2_actual, COLOR_BGR2YUV_YUY2);
    EXPECT_EQ(max_abs_diff_u8(yuy2_expected, yuy2_actual), 0);

    Mat uyvy_expected = color3_to_yuv422packed_reference_u8(rgb_roi, true, true);
    Mat uyvy_actual;
    cvtColor(rgb_roi, uyvy_actual, COLOR_RGB2YUV_UYVY);
    EXPECT_EQ(max_abs_diff_u8(uyvy_expected, uyvy_actual), 0);
}

TEST(ImgprocCvtColor_TEST, non_contiguous_roi_for_nv12_nv21_encode_matches_reference)
{
    constexpr int kRows = 6;
    constexpr int kCols = 8;

    Mat base_bgr({kRows + 2, kCols + 3}, CV_8UC3);
    Mat bgr_roi = base_bgr.rowRange(1, 1 + kRows).colRange(1, 1 + kCols);
    ASSERT_FALSE(bgr_roi.isContinuous());

    Mat base_rgb({kRows + 2, kCols + 3}, CV_8UC3);
    Mat rgb_roi = base_rgb.rowRange(1, 1 + kRows).colRange(1, 1 + kCols);
    ASSERT_FALSE(rgb_roi.isContinuous());

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar b = static_cast<uchar>(29 + (y * 17 + x * 9) % 170);
            const uchar g = static_cast<uchar>(47 + (y * 11 + x * 7) % 160);
            const uchar r = static_cast<uchar>(65 + (y * 13 + x * 5) % 150);
            bgr_roi.at<uchar>(y, x, 0) = b;
            bgr_roi.at<uchar>(y, x, 1) = g;
            bgr_roi.at<uchar>(y, x, 2) = r;
            rgb_roi.at<uchar>(y, x, 0) = r;
            rgb_roi.at<uchar>(y, x, 1) = g;
            rgb_roi.at<uchar>(y, x, 2) = b;
        }
    }

    Mat nv12_expected = color3_to_yuv420sp_reference_u8(bgr_roi, false, false);
    Mat nv12_actual;
    cvtColor(bgr_roi, nv12_actual, COLOR_BGR2YUV_NV12);
    EXPECT_EQ(max_abs_diff_u8(nv12_expected, nv12_actual), 0);

    Mat nv21_expected = color3_to_yuv420sp_reference_u8(rgb_roi, true, true);
    Mat nv21_actual;
    cvtColor(rgb_roi, nv21_actual, COLOR_RGB2YUV_NV21);
    EXPECT_EQ(max_abs_diff_u8(nv21_expected, nv21_actual), 0);
}

TEST(ImgprocCvtColor_TEST, non_contiguous_roi_for_i420_yv12_encode_matches_reference)
{
    constexpr int kRows = 6;
    constexpr int kCols = 8;

    Mat base_bgr({kRows + 2, kCols + 3}, CV_8UC3);
    Mat bgr_roi = base_bgr.rowRange(1, 1 + kRows).colRange(1, 1 + kCols);
    ASSERT_FALSE(bgr_roi.isContinuous());

    Mat base_rgb({kRows + 2, kCols + 3}, CV_8UC3);
    Mat rgb_roi = base_rgb.rowRange(1, 1 + kRows).colRange(1, 1 + kCols);
    ASSERT_FALSE(rgb_roi.isContinuous());

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar b = static_cast<uchar>(33 + (y * 17 + x * 9) % 170);
            const uchar g = static_cast<uchar>(51 + (y * 11 + x * 7) % 160);
            const uchar r = static_cast<uchar>(69 + (y * 13 + x * 5) % 150);
            bgr_roi.at<uchar>(y, x, 0) = b;
            bgr_roi.at<uchar>(y, x, 1) = g;
            bgr_roi.at<uchar>(y, x, 2) = r;
            rgb_roi.at<uchar>(y, x, 0) = r;
            rgb_roi.at<uchar>(y, x, 1) = g;
            rgb_roi.at<uchar>(y, x, 2) = b;
        }
    }

    Mat i420_expected = color3_to_yuv420p_reference_u8(bgr_roi, false, false);
    Mat i420_actual;
    cvtColor(bgr_roi, i420_actual, COLOR_BGR2YUV_I420);
    EXPECT_EQ(max_abs_diff_u8(i420_expected, i420_actual), 0);

    Mat yv12_expected = color3_to_yuv420p_reference_u8(rgb_roi, true, true);
    Mat yv12_actual;
    cvtColor(rgb_roi, yv12_actual, COLOR_RGB2YUV_YV12);
    EXPECT_EQ(max_abs_diff_u8(yv12_expected, yv12_actual), 0);
}

TEST(ImgprocCvtColor_TEST, non_contiguous_step_for_nv12_nv21_matches_reference)
{
    constexpr int kRows = 6;
    constexpr int kCols = 8;

    Mat base_nv12({kRows * 3 / 2, kCols + 4}, CV_8UC1);
    Mat nv12_roi = base_nv12.colRange(2, 2 + kCols);
    ASSERT_FALSE(nv12_roi.isContinuous());

    Mat base_nv21({kRows * 3 / 2, kCols + 4}, CV_8UC1);
    Mat nv21_roi = base_nv21.colRange(2, 2 + kCols);
    ASSERT_FALSE(nv21_roi.isContinuous());

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar yy = static_cast<uchar>(32 + (y * 19 + x * 7) % 180);
            nv12_roi.at<uchar>(y, x) = yy;
            nv21_roi.at<uchar>(y, x) = yy;
        }
    }

    for (int y = 0; y < kRows / 2; ++y)
    {
        for (int x = 0; x < kCols; x += 2)
        {
            const uchar uu = static_cast<uchar>(48 + (y * 9 + x * 5) % 150);
            const uchar vv = static_cast<uchar>(70 + (y * 11 + x * 3) % 140);

            nv12_roi.at<uchar>(kRows + y, x + 0) = uu;
            nv12_roi.at<uchar>(kRows + y, x + 1) = vv;
            nv21_roi.at<uchar>(kRows + y, x + 0) = vv;
            nv21_roi.at<uchar>(kRows + y, x + 1) = uu;
        }
    }

    Mat bgr_nv12_expected = yuv420sp_to_color3_reference_u8(nv12_roi, false, false);
    Mat bgr_nv12_actual;
    cvtColor(nv12_roi, bgr_nv12_actual, COLOR_YUV2BGR_NV12);
    EXPECT_EQ(max_abs_diff_u8(bgr_nv12_expected, bgr_nv12_actual), 0);

    Mat rgb_nv21_expected = yuv420sp_to_color3_reference_u8(nv21_roi, true, true);
    Mat rgb_nv21_actual;
    cvtColor(nv21_roi, rgb_nv21_actual, COLOR_YUV2RGB_NV21);
    EXPECT_EQ(max_abs_diff_u8(rgb_nv21_expected, rgb_nv21_actual), 0);
}

TEST(ImgprocCvtColor_TEST, non_contiguous_step_for_i420_yv12_matches_reference)
{
    constexpr int kRows = 6;
    constexpr int kCols = 8;
    constexpr int kUvSize = kRows * kCols / 4;

    Mat base_i420({kRows * 3 / 2, kCols + 4}, CV_8UC1);
    Mat i420_roi = base_i420.colRange(2, 2 + kCols);
    ASSERT_FALSE(i420_roi.isContinuous());

    Mat base_yv12({kRows * 3 / 2, kCols + 4}, CV_8UC1);
    Mat yv12_roi = base_yv12.colRange(2, 2 + kCols);
    ASSERT_FALSE(yv12_roi.isContinuous());

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar yy = static_cast<uchar>(28 + (y * 17 + x * 5) % 190);
            i420_roi.at<uchar>(y, x) = yy;
            yv12_roi.at<uchar>(y, x) = yy;
        }
    }

    for (int i = 0; i < kUvSize; ++i)
    {
        const uchar uu = static_cast<uchar>(52 + (i * 7) % 150);
        const uchar vv = static_cast<uchar>(66 + (i * 9) % 140);
        set_yuv420p_plane_byte_u8(i420_roi, kRows, kCols, 0, i, uu);
        set_yuv420p_plane_byte_u8(i420_roi, kRows, kCols, kUvSize, i, vv);
        set_yuv420p_plane_byte_u8(yv12_roi, kRows, kCols, 0, i, vv);
        set_yuv420p_plane_byte_u8(yv12_roi, kRows, kCols, kUvSize, i, uu);
    }

    Mat bgr_i420_expected = yuv420p_to_color3_reference_u8(i420_roi, false, false);
    Mat bgr_i420_actual;
    cvtColor(i420_roi, bgr_i420_actual, COLOR_YUV2BGR_I420);
    EXPECT_EQ(max_abs_diff_u8(bgr_i420_expected, bgr_i420_actual), 0);

    Mat rgb_yv12_expected = yuv420p_to_color3_reference_u8(yv12_roi, true, true);
    Mat rgb_yv12_actual;
    cvtColor(yv12_roi, rgb_yv12_actual, COLOR_YUV2RGB_YV12);
    EXPECT_EQ(max_abs_diff_u8(rgb_yv12_expected, rgb_yv12_actual), 0);
}

TEST(ImgprocCvtColor_TEST, non_contiguous_step_for_i444_yv24_matches_reference)
{
    constexpr int kRows = 5;
    constexpr int kCols = 6;
    constexpr int kPlaneSize = kRows * kCols;

    Mat base_i444({kRows * 3, kCols + 3}, CV_8UC1);
    Mat i444_roi = base_i444.colRange(1, 1 + kCols);
    ASSERT_FALSE(i444_roi.isContinuous());

    Mat base_yv24({kRows * 3, kCols + 3}, CV_8UC1);
    Mat yv24_roi = base_yv24.colRange(1, 1 + kCols);
    ASSERT_FALSE(yv24_roi.isContinuous());

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar yy = static_cast<uchar>(24 + (y * 17 + x * 9) % 190);
            i444_roi.at<uchar>(y, x) = yy;
            yv24_roi.at<uchar>(y, x) = yy;
        }
    }

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const int chroma_index = y * kCols + x;
            const uchar uu = static_cast<uchar>(50 + (y * 13 + x * 5) % 150);
            const uchar vv = static_cast<uchar>(68 + (y * 7 + x * 11) % 140);
            set_yuv444p_plane_byte_u8(i444_roi, kRows, kCols, 0, chroma_index, uu);
            set_yuv444p_plane_byte_u8(i444_roi, kRows, kCols, kPlaneSize, chroma_index, vv);
            set_yuv444p_plane_byte_u8(yv24_roi, kRows, kCols, 0, chroma_index, vv);
            set_yuv444p_plane_byte_u8(yv24_roi, kRows, kCols, kPlaneSize, chroma_index, uu);
        }
    }

    Mat bgr_i444_expected = yuv444p_to_color3_reference_u8(i444_roi, false, false);
    Mat bgr_i444_actual;
    cvtColor(i444_roi, bgr_i444_actual, COLOR_YUV2BGR_I444);
    EXPECT_EQ(max_abs_diff_u8(bgr_i444_expected, bgr_i444_actual), 0);

    Mat rgb_yv24_expected = yuv444p_to_color3_reference_u8(yv24_roi, true, true);
    Mat rgb_yv24_actual;
    cvtColor(yv24_roi, rgb_yv24_actual, COLOR_YUV2RGB_YV24);
    EXPECT_EQ(max_abs_diff_u8(rgb_yv24_expected, rgb_yv24_actual), 0);
}

TEST(ImgprocCvtColor_TEST, non_contiguous_step_for_nv16_nv61_matches_reference)
{
    constexpr int kRows = 6;
    constexpr int kCols = 8;

    Mat base_nv16({kRows * 2, kCols + 4}, CV_8UC1);
    Mat nv16_roi = base_nv16.colRange(2, 2 + kCols);
    ASSERT_FALSE(nv16_roi.isContinuous());

    Mat base_nv61({kRows * 2, kCols + 4}, CV_8UC1);
    Mat nv61_roi = base_nv61.colRange(2, 2 + kCols);
    ASSERT_FALSE(nv61_roi.isContinuous());

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar yy = static_cast<uchar>(24 + (y * 17 + x * 7) % 200);
            nv16_roi.at<uchar>(y, x) = yy;
            nv61_roi.at<uchar>(y, x) = yy;
        }
    }

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; x += 2)
        {
            const uchar uu = static_cast<uchar>(50 + (y * 9 + x * 5) % 150);
            const uchar vv = static_cast<uchar>(66 + (y * 11 + x * 3) % 140);
            nv16_roi.at<uchar>(kRows + y, x + 0) = uu;
            nv16_roi.at<uchar>(kRows + y, x + 1) = vv;
            nv61_roi.at<uchar>(kRows + y, x + 0) = vv;
            nv61_roi.at<uchar>(kRows + y, x + 1) = uu;
        }
    }

    Mat bgr_nv16_expected = yuv422sp_to_color3_reference_u8(nv16_roi, false, false);
    Mat bgr_nv16_actual;
    cvtColor(nv16_roi, bgr_nv16_actual, COLOR_YUV2BGR_NV16);
    EXPECT_EQ(max_abs_diff_u8(bgr_nv16_expected, bgr_nv16_actual), 0);

    Mat rgb_nv61_expected = yuv422sp_to_color3_reference_u8(nv61_roi, true, true);
    Mat rgb_nv61_actual;
    cvtColor(nv61_roi, rgb_nv61_actual, COLOR_YUV2RGB_NV61);
    EXPECT_EQ(max_abs_diff_u8(rgb_nv61_expected, rgb_nv61_actual), 0);
}

TEST(ImgprocCvtColor_TEST, non_contiguous_step_for_nv24_nv42_matches_reference)
{
    constexpr int kRows = 4;
    constexpr int kCols = 5;

    Mat base_nv24({kRows * 3, kCols + 3}, CV_8UC1);
    Mat nv24_roi = base_nv24.colRange(1, 1 + kCols);
    ASSERT_FALSE(nv24_roi.isContinuous());

    Mat base_nv42({kRows * 3, kCols + 3}, CV_8UC1);
    Mat nv42_roi = base_nv42.colRange(1, 1 + kCols);
    ASSERT_FALSE(nv42_roi.isContinuous());

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar yy = static_cast<uchar>(26 + (y * 19 + x * 11) % 200);
            nv24_roi.at<uchar>(y, x) = yy;
            nv42_roi.at<uchar>(y, x) = yy;
        }
    }

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar uu = static_cast<uchar>(52 + (y * 13 + x * 5) % 150);
            const uchar vv = static_cast<uchar>(70 + (y * 7 + x * 9) % 140);
            const int base = y * (kCols * 2) + x * 2;
            set_yuv444sp_plane_byte_u8(nv24_roi, kRows, kCols, base + 0, uu);
            set_yuv444sp_plane_byte_u8(nv24_roi, kRows, kCols, base + 1, vv);
            set_yuv444sp_plane_byte_u8(nv42_roi, kRows, kCols, base + 0, vv);
            set_yuv444sp_plane_byte_u8(nv42_roi, kRows, kCols, base + 1, uu);
        }
    }

    Mat bgr_nv24_expected = yuv444sp_to_color3_reference_u8(nv24_roi, false, false);
    Mat bgr_nv24_actual;
    cvtColor(nv24_roi, bgr_nv24_actual, COLOR_YUV2BGR_NV24);
    EXPECT_EQ(max_abs_diff_u8(bgr_nv24_expected, bgr_nv24_actual), 0);

    Mat rgb_nv42_expected = yuv444sp_to_color3_reference_u8(nv42_roi, true, true);
    Mat rgb_nv42_actual;
    cvtColor(nv42_roi, rgb_nv42_actual, COLOR_YUV2RGB_NV42);
    EXPECT_EQ(max_abs_diff_u8(rgb_nv42_expected, rgb_nv42_actual), 0);
}

TEST(ImgprocCvtColor_TEST, non_contiguous_step_for_yuy2_uyvy_matches_reference)
{
    constexpr int kRows = 6;
    constexpr int kCols = 8;

    Mat base_yuy2({kRows, kCols + 3}, CV_8UC2);
    Mat yuy2_roi = base_yuy2.colRange(1, 1 + kCols);
    ASSERT_FALSE(yuy2_roi.isContinuous());

    Mat base_uyvy({kRows, kCols + 3}, CV_8UC2);
    Mat uyvy_roi = base_uyvy.colRange(1, 1 + kCols);
    ASSERT_FALSE(uyvy_roi.isContinuous());

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; x += 2)
        {
            const uchar yy0 = static_cast<uchar>(26 + (y * 15 + x * 7) % 190);
            const uchar yy1 = static_cast<uchar>(41 + (y * 9 + x * 11) % 180);
            const uchar uu = static_cast<uchar>(54 + (y * 5 + x * 3) % 150);
            const uchar vv = static_cast<uchar>(68 + (y * 7 + x * 5) % 140);
            set_yuv422_packed_pair_u8(yuy2_roi, y, x, false, yy0, yy1, uu, vv);
            set_yuv422_packed_pair_u8(uyvy_roi, y, x, true, yy0, yy1, uu, vv);
        }
    }

    Mat bgr_yuy2_expected = yuv422packed_to_color3_reference_u8(yuy2_roi, false, false);
    Mat bgr_yuy2_actual;
    cvtColor(yuy2_roi, bgr_yuy2_actual, COLOR_YUV2BGR_YUY2);
    EXPECT_EQ(max_abs_diff_u8(bgr_yuy2_expected, bgr_yuy2_actual), 0);

    Mat rgb_uyvy_expected = yuv422packed_to_color3_reference_u8(uyvy_roi, true, true);
    Mat rgb_uyvy_actual;
    cvtColor(uyvy_roi, rgb_uyvy_actual, COLOR_YUV2RGB_UYVY);
    EXPECT_EQ(max_abs_diff_u8(rgb_uyvy_expected, rgb_uyvy_actual), 0);
}

TEST(ImgprocCvtColor_TEST, throws_on_invalid_nv12_nv21_layouts)
{
    Mat dst;

    Mat odd_width({6, 5}, CV_8UC1);
    EXPECT_THROW(cvtColor(odd_width, dst, COLOR_YUV2BGR_NV12), Exception);

    Mat bad_rows({5, 6}, CV_8UC1);
    EXPECT_THROW(cvtColor(bad_rows, dst, COLOR_YUV2RGB_NV12), Exception);

    Mat three_channel({6, 6}, CV_8UC3);
    EXPECT_THROW(cvtColor(three_channel, dst, COLOR_YUV2BGR_NV21), Exception);

    Mat f32_src({6, 6}, CV_32FC1);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_YUV2RGB_NV21), Exception);
}

TEST(ImgprocCvtColor_TEST, throws_on_invalid_i420_yv12_layouts)
{
    Mat dst;

    Mat odd_width({6, 5}, CV_8UC1);
    EXPECT_THROW(cvtColor(odd_width, dst, COLOR_YUV2BGR_I420), Exception);

    Mat bad_rows({5, 6}, CV_8UC1);
    EXPECT_THROW(cvtColor(bad_rows, dst, COLOR_YUV2RGB_YV12), Exception);

    Mat three_channel({6, 6}, CV_8UC3);
    EXPECT_THROW(cvtColor(three_channel, dst, COLOR_YUV2BGR_YV12), Exception);

    Mat f32_src({6, 6}, CV_32FC1);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_YUV2RGB_I420), Exception);
}

TEST(ImgprocCvtColor_TEST, throws_on_invalid_i444_yv24_layouts)
{
    Mat dst;

    Mat bad_rows({11, 5}, CV_8UC1);
    EXPECT_THROW(cvtColor(bad_rows, dst, COLOR_YUV2BGR_I444), Exception);

    Mat three_channel({12, 5}, CV_8UC3);
    EXPECT_THROW(cvtColor(three_channel, dst, COLOR_YUV2RGB_YV24), Exception);

    Mat f32_src({12, 5}, CV_32FC1);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_YUV2RGB_I444), Exception);
}

TEST(ImgprocCvtColor_TEST, throws_on_invalid_nv16_nv61_layouts)
{
    Mat dst;

    Mat odd_width({8, 5}, CV_8UC1);
    EXPECT_THROW(cvtColor(odd_width, dst, COLOR_YUV2BGR_NV16), Exception);

    Mat bad_rows({7, 6}, CV_8UC1);
    EXPECT_THROW(cvtColor(bad_rows, dst, COLOR_YUV2RGB_NV61), Exception);

    Mat three_channel({8, 6}, CV_8UC3);
    EXPECT_THROW(cvtColor(three_channel, dst, COLOR_YUV2BGR_NV61), Exception);

    Mat f32_src({8, 6}, CV_32FC1);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_YUV2RGB_NV16), Exception);
}

TEST(ImgprocCvtColor_TEST, throws_on_invalid_nv24_nv42_layouts)
{
    Mat dst;

    Mat bad_rows({11, 5}, CV_8UC1);
    EXPECT_THROW(cvtColor(bad_rows, dst, COLOR_YUV2BGR_NV24), Exception);

    Mat three_channel({12, 5}, CV_8UC3);
    EXPECT_THROW(cvtColor(three_channel, dst, COLOR_YUV2RGB_NV42), Exception);

    Mat f32_src({12, 5}, CV_32FC1);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_YUV2BGR_NV42), Exception);
}

TEST(ImgprocCvtColor_TEST, throws_on_invalid_bgr_rgb_to_nv24_nv42_inputs)
{
    Mat dst;

    Mat gray_src({5, 6}, CV_8UC1);
    EXPECT_THROW(cvtColor(gray_src, dst, COLOR_BGR2YUV_NV24), Exception);
    EXPECT_THROW(cvtColor(gray_src, dst, COLOR_RGB2YUV_NV42), Exception);

    Mat bgra_src({5, 6}, CV_8UC4);
    EXPECT_THROW(cvtColor(bgra_src, dst, COLOR_RGB2YUV_NV24), Exception);
    EXPECT_THROW(cvtColor(bgra_src, dst, COLOR_BGR2YUV_NV42), Exception);

    Mat f32_src({5, 6}, CV_32FC3);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_BGR2YUV_NV24), Exception);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_RGB2YUV_NV42), Exception);
}

TEST(ImgprocCvtColor_TEST, throws_on_invalid_bgr_rgb_to_i444_yv24_inputs)
{
    Mat dst;

    Mat gray_src({5, 6}, CV_8UC1);
    EXPECT_THROW(cvtColor(gray_src, dst, COLOR_BGR2YUV_I444), Exception);
    EXPECT_THROW(cvtColor(gray_src, dst, COLOR_RGB2YUV_YV24), Exception);

    Mat bgra_src({5, 6}, CV_8UC4);
    EXPECT_THROW(cvtColor(bgra_src, dst, COLOR_RGB2YUV_I444), Exception);
    EXPECT_THROW(cvtColor(bgra_src, dst, COLOR_BGR2YUV_YV24), Exception);

    Mat f32_src({5, 6}, CV_32FC3);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_BGR2YUV_I444), Exception);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_RGB2YUV_YV24), Exception);
}

TEST(ImgprocCvtColor_TEST, throws_on_invalid_bgr_rgb_to_nv16_nv61_inputs)
{
    Mat dst;

    Mat gray_src({5, 6}, CV_8UC1);
    EXPECT_THROW(cvtColor(gray_src, dst, COLOR_BGR2YUV_NV16), Exception);
    EXPECT_THROW(cvtColor(gray_src, dst, COLOR_RGB2YUV_NV61), Exception);

    Mat bgra_src({5, 6}, CV_8UC4);
    EXPECT_THROW(cvtColor(bgra_src, dst, COLOR_RGB2YUV_NV16), Exception);
    EXPECT_THROW(cvtColor(bgra_src, dst, COLOR_BGR2YUV_NV61), Exception);

    Mat odd_width({5, 5}, CV_8UC3);
    EXPECT_THROW(cvtColor(odd_width, dst, COLOR_BGR2YUV_NV16), Exception);
    EXPECT_THROW(cvtColor(odd_width, dst, COLOR_RGB2YUV_NV61), Exception);

    Mat f32_src({5, 6}, CV_32FC3);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_BGR2YUV_NV16), Exception);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_RGB2YUV_NV61), Exception);
}

TEST(ImgprocCvtColor_TEST, throws_on_invalid_bgr_rgb_to_yuy2_uyvy_inputs)
{
    Mat dst;

    Mat gray_src({5, 6}, CV_8UC1);
    EXPECT_THROW(cvtColor(gray_src, dst, COLOR_BGR2YUV_YUY2), Exception);
    EXPECT_THROW(cvtColor(gray_src, dst, COLOR_RGB2YUV_UYVY), Exception);

    Mat bgra_src({5, 6}, CV_8UC4);
    EXPECT_THROW(cvtColor(bgra_src, dst, COLOR_RGB2YUV_YUY2), Exception);
    EXPECT_THROW(cvtColor(bgra_src, dst, COLOR_BGR2YUV_UYVY), Exception);

    Mat odd_width({5, 5}, CV_8UC3);
    EXPECT_THROW(cvtColor(odd_width, dst, COLOR_BGR2YUV_YUY2), Exception);
    EXPECT_THROW(cvtColor(odd_width, dst, COLOR_RGB2YUV_UYVY), Exception);

    Mat f32_src({5, 6}, CV_32FC3);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_BGR2YUV_YUY2), Exception);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_RGB2YUV_UYVY), Exception);
}

TEST(ImgprocCvtColor_TEST, throws_on_invalid_bgr_rgb_to_nv12_nv21_inputs)
{
    Mat dst;

    Mat gray_src({6, 6}, CV_8UC1);
    EXPECT_THROW(cvtColor(gray_src, dst, COLOR_BGR2YUV_NV12), Exception);
    EXPECT_THROW(cvtColor(gray_src, dst, COLOR_RGB2YUV_NV21), Exception);

    Mat bgra_src({6, 6}, CV_8UC4);
    EXPECT_THROW(cvtColor(bgra_src, dst, COLOR_RGB2YUV_NV12), Exception);
    EXPECT_THROW(cvtColor(bgra_src, dst, COLOR_BGR2YUV_NV21), Exception);

    Mat odd_width({6, 5}, CV_8UC3);
    EXPECT_THROW(cvtColor(odd_width, dst, COLOR_BGR2YUV_NV12), Exception);
    EXPECT_THROW(cvtColor(odd_width, dst, COLOR_RGB2YUV_NV21), Exception);

    Mat odd_height({5, 6}, CV_8UC3);
    EXPECT_THROW(cvtColor(odd_height, dst, COLOR_BGR2YUV_NV12), Exception);
    EXPECT_THROW(cvtColor(odd_height, dst, COLOR_RGB2YUV_NV21), Exception);

    Mat f32_src({6, 6}, CV_32FC3);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_BGR2YUV_NV12), Exception);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_RGB2YUV_NV21), Exception);
}

TEST(ImgprocCvtColor_TEST, throws_on_invalid_bgr_rgb_to_i420_yv12_inputs)
{
    Mat dst;

    Mat gray_src({6, 6}, CV_8UC1);
    EXPECT_THROW(cvtColor(gray_src, dst, COLOR_BGR2YUV_I420), Exception);
    EXPECT_THROW(cvtColor(gray_src, dst, COLOR_RGB2YUV_YV12), Exception);

    Mat bgra_src({6, 6}, CV_8UC4);
    EXPECT_THROW(cvtColor(bgra_src, dst, COLOR_RGB2YUV_I420), Exception);
    EXPECT_THROW(cvtColor(bgra_src, dst, COLOR_BGR2YUV_YV12), Exception);

    Mat odd_width({6, 5}, CV_8UC3);
    EXPECT_THROW(cvtColor(odd_width, dst, COLOR_BGR2YUV_I420), Exception);
    EXPECT_THROW(cvtColor(odd_width, dst, COLOR_RGB2YUV_YV12), Exception);

    Mat odd_height({5, 6}, CV_8UC3);
    EXPECT_THROW(cvtColor(odd_height, dst, COLOR_BGR2YUV_I420), Exception);
    EXPECT_THROW(cvtColor(odd_height, dst, COLOR_RGB2YUV_YV12), Exception);

    Mat f32_src({6, 6}, CV_32FC3);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_BGR2YUV_I420), Exception);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_RGB2YUV_YV12), Exception);
}

TEST(ImgprocCvtColor_TEST, throws_on_invalid_yuy2_uyvy_layouts)
{
    Mat dst;

    Mat odd_width({6, 5}, CV_8UC2);
    EXPECT_THROW(cvtColor(odd_width, dst, COLOR_YUV2BGR_YUY2), Exception);

    Mat one_channel({6, 6}, CV_8UC1);
    EXPECT_THROW(cvtColor(one_channel, dst, COLOR_YUV2RGB_YUY2), Exception);

    Mat three_channel({6, 6}, CV_8UC3);
    EXPECT_THROW(cvtColor(three_channel, dst, COLOR_YUV2BGR_UYVY), Exception);

    Mat f32_src({6, 6}, CV_32FC2);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_YUV2RGB_UYVY), Exception);
}

TEST(ImgprocCvtColor_TEST, supports_single_row_and_single_col_images)
{
    Mat row_bgr({1, 9}, CV_8UC3);
    for (int x = 0; x < row_bgr.size[1]; ++x)
    {
        row_bgr.at<uchar>(0, x, 0) = static_cast<uchar>((x * 3 + 1) % 256);
        row_bgr.at<uchar>(0, x, 1) = static_cast<uchar>((x * 5 + 2) % 256);
        row_bgr.at<uchar>(0, x, 2) = static_cast<uchar>((x * 7 + 3) % 256);
    }

    Mat row_gray;
    cvtColor(row_bgr, row_gray, COLOR_BGR2GRAY);
    Mat row_expected = bgr2gray_reference_u8(row_bgr);
    EXPECT_EQ(max_abs_diff_u8(row_gray, row_expected), 0);

    Mat col_gray({9, 1}, CV_8UC1);
    for (int y = 0; y < col_gray.size[0]; ++y)
    {
        col_gray.at<uchar>(y, 0) = static_cast<uchar>((y * 17 + 9) % 256);
    }

    Mat col_bgr;
    cvtColor(col_gray, col_bgr, COLOR_GRAY2BGR);
    Mat col_expected = gray2bgr_reference_u8(col_gray);
    EXPECT_EQ(max_abs_diff_u8(col_bgr, col_expected), 0);
}

TEST(ImgprocCvtColor_TEST, supports_cv32f_bgr2gray_and_gray2bgr)
{
    Mat src({3, 4}, CV_32FC3);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            src.at<float>(y, x, 0) = static_cast<float>(-1.5 + y * 0.25 + x * 0.10);
            src.at<float>(y, x, 1) = static_cast<float>(2.0 + y * 0.75 - x * 0.20);
            src.at<float>(y, x, 2) = static_cast<float>(0.5 - y * 0.10 + x * 1.30);
        }
    }

    Mat gray_expected = bgr2gray_reference_f32(src);
    Mat gray_actual;
    cvtColor(src, gray_actual, COLOR_BGR2GRAY);
    ASSERT_EQ(gray_actual.type(), CV_32FC1);
    EXPECT_LE(max_abs_diff_f32(gray_expected, gray_actual), 1e-6f);

    Mat bgr_expected = gray2bgr_reference_f32(gray_actual);
    Mat bgr_actual;
    cvtColor(gray_actual, bgr_actual, COLOR_GRAY2BGR);
    ASSERT_EQ(bgr_actual.type(), CV_32FC3);
    EXPECT_LE(max_abs_diff_f32(bgr_expected, bgr_actual), 1e-6f);
}

TEST(ImgprocCvtColor_TEST, cv32f_non_contiguous_roi_matches_reference)
{
    Mat base_bgr({6, 10}, CV_32FC3);
    for (int y = 0; y < base_bgr.size[0]; ++y)
    {
        for (int x = 0; x < base_bgr.size[1]; ++x)
        {
            base_bgr.at<float>(y, x, 0) = static_cast<float>(y * 0.45 - x * 0.15 + 0.25);
            base_bgr.at<float>(y, x, 1) = static_cast<float>(y * 1.20 + x * 0.35 - 0.50);
            base_bgr.at<float>(y, x, 2) = static_cast<float>(-y * 0.30 + x * 0.90 + 1.75);
        }
    }
    Mat bgr_roi = base_bgr.colRange(1, 9);
    ASSERT_FALSE(bgr_roi.isContinuous());

    Mat gray_expected = bgr2gray_reference_f32(bgr_roi);
    Mat gray_actual;
    cvtColor(bgr_roi, gray_actual, COLOR_BGR2GRAY);
    EXPECT_LE(max_abs_diff_f32(gray_expected, gray_actual), 1e-6f);

    Mat base_gray({7, 11}, CV_32FC1);
    for (int y = 0; y < base_gray.size[0]; ++y)
    {
        for (int x = 0; x < base_gray.size[1]; ++x)
        {
            base_gray.at<float>(y, x) = static_cast<float>(y * 0.60 - x * 0.22 + 3.0);
        }
    }
    Mat gray_roi = base_gray.colRange(2, 10);
    ASSERT_FALSE(gray_roi.isContinuous());

    Mat bgr_expected = gray2bgr_reference_f32(gray_roi);
    Mat bgr_actual;
    cvtColor(gray_roi, bgr_actual, COLOR_GRAY2BGR);
    EXPECT_LE(max_abs_diff_f32(bgr_expected, bgr_actual), 1e-6f);
}

TEST(ImgprocCvtColor_TEST, supports_cv32f_bgr2rgb_and_bgr2bgra)
{
    Mat src({3, 5}, CV_32FC3);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            src.at<float>(y, x, 0) = static_cast<float>(-0.5 + y * 0.20 + x * 0.10);
            src.at<float>(y, x, 1) = static_cast<float>(1.5 + y * 0.35 - x * 0.40);
            src.at<float>(y, x, 2) = static_cast<float>(2.5 - y * 0.15 + x * 0.60);
        }
    }

    Mat rgb_expected = bgr2rgb_reference<float>(src);
    Mat rgb_actual;
    cvtColor(src, rgb_actual, COLOR_BGR2RGB);
    ASSERT_EQ(rgb_actual.type(), CV_32FC3);
    EXPECT_LE(max_abs_diff_f32(rgb_expected, rgb_actual), 1e-6f);

    Mat bgr_roundtrip;
    cvtColor(rgb_actual, bgr_roundtrip, COLOR_RGB2BGR);
    EXPECT_LE(max_abs_diff_f32(src, bgr_roundtrip), 1e-6f);

    Mat bgra_expected = bgr2bgra_reference<float>(src);
    Mat bgra_actual;
    cvtColor(src, bgra_actual, COLOR_BGR2BGRA);
    ASSERT_EQ(bgra_actual.type(), CV_32FC4);
    EXPECT_LE(max_abs_diff_f32(bgra_expected, bgra_actual), 1e-6f);

    Mat bgr_expected = bgra2bgr_reference<float>(bgra_actual);
    Mat bgr_actual;
    cvtColor(bgra_actual, bgr_actual, COLOR_BGRA2BGR);
    ASSERT_EQ(bgr_actual.type(), CV_32FC3);
    EXPECT_LE(max_abs_diff_f32(bgr_expected, bgr_actual), 1e-6f);
}

TEST(ImgprocCvtColor_TEST, supports_cv32f_rgba_family_conversions)
{
    Mat rgb({3, 5}, CV_32FC3);
    for (int y = 0; y < rgb.size[0]; ++y)
    {
        for (int x = 0; x < rgb.size[1]; ++x)
        {
            rgb.at<float>(y, x, 0) = static_cast<float>(-1.0 + y * 0.15 + x * 0.40);
            rgb.at<float>(y, x, 1) = static_cast<float>(0.5 + y * 0.22 - x * 0.17);
            rgb.at<float>(y, x, 2) = static_cast<float>(2.0 - y * 0.31 + x * 0.09);
        }
    }

    Mat rgba_expected = rgb2rgba_reference<float>(rgb);
    Mat rgba_actual;
    cvtColor(rgb, rgba_actual, COLOR_RGB2RGBA);
    ASSERT_EQ(rgba_actual.type(), CV_32FC4);
    EXPECT_LE(max_abs_diff_f32(rgba_expected, rgba_actual), 1e-6f);

    Mat bgra_expected = rgb2bgra_reference<float>(rgb);
    Mat bgra_actual;
    cvtColor(rgb, bgra_actual, COLOR_RGB2BGRA);
    ASSERT_EQ(bgra_actual.type(), CV_32FC4);
    EXPECT_LE(max_abs_diff_f32(bgra_expected, bgra_actual), 1e-6f);

    Mat rgb_roundtrip;
    cvtColor(rgba_actual, rgb_roundtrip, COLOR_RGBA2RGB);
    EXPECT_LE(max_abs_diff_f32(rgb, rgb_roundtrip), 1e-6f);

    Mat rgb_from_bgra;
    cvtColor(bgra_actual, rgb_from_bgra, COLOR_BGRA2RGB);
    EXPECT_LE(max_abs_diff_f32(rgb, rgb_from_bgra), 1e-6f);

    Mat bgr({3, 5}, CV_32FC3);
    for (int y = 0; y < bgr.size[0]; ++y)
    {
        for (int x = 0; x < bgr.size[1]; ++x)
        {
            bgr.at<float>(y, x, 0) = static_cast<float>(1.5 + y * 0.11 + x * 0.27);
            bgr.at<float>(y, x, 1) = static_cast<float>(-0.5 + y * 0.45 - x * 0.12);
            bgr.at<float>(y, x, 2) = static_cast<float>(0.25 - y * 0.08 + x * 0.51);
        }
    }

    Mat rgba_from_bgr_expected = bgr2rgba_reference<float>(bgr);
    Mat rgba_from_bgr_actual;
    cvtColor(bgr, rgba_from_bgr_actual, COLOR_BGR2RGBA);
    ASSERT_EQ(rgba_from_bgr_actual.type(), CV_32FC4);
    EXPECT_LE(max_abs_diff_f32(rgba_from_bgr_expected, rgba_from_bgr_actual), 1e-6f);

    Mat bgr_roundtrip;
    cvtColor(rgba_from_bgr_actual, bgr_roundtrip, COLOR_RGBA2BGR);
    EXPECT_LE(max_abs_diff_f32(bgr, bgr_roundtrip), 1e-6f);

    Mat rgba_swapped_expected = swap_rb_4ch_reference<float>(bgra_actual);
    Mat rgba_swapped_actual;
    cvtColor(bgra_actual, rgba_swapped_actual, COLOR_BGRA2RGBA);
    ASSERT_EQ(rgba_swapped_actual.type(), CV_32FC4);
    EXPECT_LE(max_abs_diff_f32(rgba_swapped_expected, rgba_swapped_actual), 1e-6f);

    Mat bgra_roundtrip;
    cvtColor(rgba_swapped_actual, bgra_roundtrip, COLOR_RGBA2BGRA);
    EXPECT_LE(max_abs_diff_f32(bgra_actual, bgra_roundtrip), 1e-6f);
}

TEST(ImgprocCvtColor_TEST, supports_cv32f_gray_rgba_family_conversions)
{
    Mat gray({3, 4}, CV_32FC1);
    for (int y = 0; y < gray.size[0]; ++y)
    {
        for (int x = 0; x < gray.size[1]; ++x)
        {
            gray.at<float>(y, x) = static_cast<float>(-0.75 + y * 0.40 + x * 0.15);
        }
    }

    Mat bgra_expected = gray2bgra_reference<float>(gray);
    Mat bgra_actual;
    cvtColor(gray, bgra_actual, COLOR_GRAY2BGRA);
    ASSERT_EQ(bgra_actual.type(), CV_32FC4);
    EXPECT_LE(max_abs_diff_f32(bgra_expected, bgra_actual), 1e-6f);

    Mat rgba_actual;
    cvtColor(gray, rgba_actual, COLOR_GRAY2RGBA);
    ASSERT_EQ(rgba_actual.type(), CV_32FC4);
    EXPECT_LE(max_abs_diff_f32(bgra_expected, rgba_actual), 1e-6f);

    Mat gray_from_bgra_expected = color4_to_gray_reference<float>(bgra_actual, false);
    Mat gray_from_bgra_actual;
    cvtColor(bgra_actual, gray_from_bgra_actual, COLOR_BGRA2GRAY);
    ASSERT_EQ(gray_from_bgra_actual.type(), CV_32FC1);
    EXPECT_LE(max_abs_diff_f32(gray_from_bgra_expected, gray_from_bgra_actual), 1e-6f);

    Mat gray_from_rgba_expected = color4_to_gray_reference<float>(rgba_actual, true);
    Mat gray_from_rgba_actual;
    cvtColor(rgba_actual, gray_from_rgba_actual, COLOR_RGBA2GRAY);
    ASSERT_EQ(gray_from_rgba_actual.type(), CV_32FC1);
    EXPECT_LE(max_abs_diff_f32(gray_from_rgba_expected, gray_from_rgba_actual), 1e-6f);
}

TEST(ImgprocCvtColor_TEST, supports_cv32f_yuv_family_conversions)
{
    Mat bgr({3, 4}, CV_32FC3);
    for (int y = 0; y < bgr.size[0]; ++y)
    {
        for (int x = 0; x < bgr.size[1]; ++x)
        {
            bgr.at<float>(y, x, 0) = static_cast<float>(0.10 + y * 0.07 + x * 0.03);
            bgr.at<float>(y, x, 1) = static_cast<float>(0.20 + y * 0.05 + x * 0.04);
            bgr.at<float>(y, x, 2) = static_cast<float>(0.30 + y * 0.06 + x * 0.02);
        }
    }

    Mat yuv_expected = color3_to_yuv_reference<float>(bgr, false);
    Mat yuv_actual;
    cvtColor(bgr, yuv_actual, COLOR_BGR2YUV);
    ASSERT_EQ(yuv_actual.type(), CV_32FC3);
    EXPECT_LE(max_abs_diff_f32(yuv_expected, yuv_actual), 1e-6f);

    Mat bgr_expected = yuv_to_color3_reference<float>(yuv_actual, false);
    Mat bgr_actual;
    cvtColor(yuv_actual, bgr_actual, COLOR_YUV2BGR);
    ASSERT_EQ(bgr_actual.type(), CV_32FC3);
    EXPECT_LE(max_abs_diff_f32(bgr_expected, bgr_actual), 1e-6f);

    Mat rgb = bgr2rgb_reference<float>(bgr);
    Mat yuv_from_rgb_expected = color3_to_yuv_reference<float>(rgb, true);
    Mat yuv_from_rgb_actual;
    cvtColor(rgb, yuv_from_rgb_actual, COLOR_RGB2YUV);
    ASSERT_EQ(yuv_from_rgb_actual.type(), CV_32FC3);
    EXPECT_LE(max_abs_diff_f32(yuv_from_rgb_expected, yuv_from_rgb_actual), 1e-6f);

    Mat rgb_expected = yuv_to_color3_reference<float>(yuv_from_rgb_actual, true);
    Mat rgb_actual;
    cvtColor(yuv_from_rgb_actual, rgb_actual, COLOR_YUV2RGB);
    ASSERT_EQ(rgb_actual.type(), CV_32FC3);
    EXPECT_LE(max_abs_diff_f32(rgb_expected, rgb_actual), 1e-6f);
}
