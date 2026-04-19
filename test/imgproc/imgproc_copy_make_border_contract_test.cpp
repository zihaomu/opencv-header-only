#include "cvh.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

using namespace cvh;

namespace
{

int normalize_border_type(int borderType)
{
    return borderType & (~BORDER_ISOLATED);
}

int border_interpolate_ref(int p, int len, int borderType)
{
    if (borderType == BORDER_WRAP)
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

    if (static_cast<unsigned>(p) < static_cast<unsigned>(len))
    {
        return p;
    }

    if (borderType == BORDER_CONSTANT)
    {
        return -1;
    }
    if (borderType == BORDER_REPLICATE)
    {
        return p < 0 ? 0 : (len - 1);
    }
    if (borderType == BORDER_REFLECT || borderType == BORDER_REFLECT_101)
    {
        if (len == 1)
        {
            return 0;
        }
        const int delta = borderType == BORDER_REFLECT_101 ? 1 : 0;
        while (p < 0 || p >= len)
        {
            if (p < 0)
            {
                p = -p - 1 + delta;
            }
            else
            {
                p = len - 1 - (p - len) - delta;
            }
        }
        return p;
    }
    return -1;
}

template <typename T>
T border_scalar_value(const Scalar& value, int channel)
{
    const int idx = channel < 4 ? channel : 3;
    if constexpr (std::is_same<T, uchar>::value)
    {
        return saturate_cast<uchar>(value.val[idx]);
    }
    return static_cast<float>(value.val[idx]);
}

template <typename T>
Mat copy_make_border_reference(const Mat& src,
                               int top,
                               int bottom,
                               int left,
                               int right,
                               int borderType,
                               const Scalar& value)
{
    CV_Assert(src.dims == 2);
    const int rows = src.size[0];
    const int cols = src.size[1];
    const int cn = src.channels();
    const size_t src_step = src.step(0);

    Mat dst({rows + top + bottom, cols + left + right}, src.type());
    const size_t dst_step = dst.step(0);
    const int border = normalize_border_type(borderType);

    for (int y = 0; y < dst.size[0]; ++y)
    {
        T* dst_row = reinterpret_cast<T*>(dst.data + static_cast<size_t>(y) * dst_step);
        const int sy_raw = y - top;
        for (int x = 0; x < dst.size[1]; ++x)
        {
            T* dst_px = dst_row + static_cast<size_t>(x) * cn;
            const int sx_raw = x - left;

            const bool inside = static_cast<unsigned>(sy_raw) < static_cast<unsigned>(rows) &&
                                static_cast<unsigned>(sx_raw) < static_cast<unsigned>(cols);
            if (inside)
            {
                const T* src_row = reinterpret_cast<const T*>(src.data + static_cast<size_t>(sy_raw) * src_step);
                const T* src_px = src_row + static_cast<size_t>(sx_raw) * cn;
                for (int c = 0; c < cn; ++c)
                {
                    dst_px[c] = src_px[c];
                }
                continue;
            }

            if (border == BORDER_CONSTANT)
            {
                for (int c = 0; c < cn; ++c)
                {
                    dst_px[c] = border_scalar_value<T>(value, c);
                }
                continue;
            }

            const int sy = border_interpolate_ref(sy_raw, rows, border);
            const int sx = border_interpolate_ref(sx_raw, cols, border);
            CV_Assert(sy >= 0 && sx >= 0);
            const T* src_row = reinterpret_cast<const T*>(src.data + static_cast<size_t>(sy) * src_step);
            const T* src_px = src_row + static_cast<size_t>(sx) * cn;
            for (int c = 0; c < cn; ++c)
            {
                dst_px[c] = src_px[c];
            }
        }
    }

    return dst;
}

void fill_u8_pattern(Mat& src, std::uint32_t seed)
{
    CV_Assert(src.depth() == CV_8U);
    const size_t count = src.total() * static_cast<size_t>(src.channels());
    for (size_t i = 0; i < count; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        src.data[i] = static_cast<uchar>((seed >> 24) & 0xFFu);
    }
}

void fill_f32_pattern(Mat& src)
{
    CV_Assert(src.depth() == CV_32F);
    const int rows = src.size[0];
    const int cols = src.size[1];
    const int cn = src.channels();
    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            for (int c = 0; c < cn; ++c)
            {
                src.at<float>(y, x, c) = static_cast<float>(y * 0.25f + x * 0.5f + c * 1.25f);
            }
        }
    }
}

int max_abs_diff_u8(const Mat& a, const Mat& b)
{
    CV_Assert(a.type() == b.type());
    CV_Assert(a.total() == b.total());
    CV_Assert(a.channels() == b.channels());
    const size_t count = a.total() * static_cast<size_t>(a.channels());
    int max_diff = 0;
    for (size_t i = 0; i < count; ++i)
    {
        max_diff = std::max(max_diff, std::abs(static_cast<int>(a.data[i]) - static_cast<int>(b.data[i])));
    }
    return max_diff;
}

float max_abs_diff_f32(const Mat& a, const Mat& b)
{
    CV_Assert(a.type() == b.type());
    CV_Assert(a.total() == b.total());
    CV_Assert(a.channels() == b.channels());
    CV_Assert(a.depth() == CV_32F);
    const float* pa = reinterpret_cast<const float*>(a.data);
    const float* pb = reinterpret_cast<const float*>(b.data);
    const size_t count = a.total() * static_cast<size_t>(a.channels());
    float max_diff = 0.0f;
    for (size_t i = 0; i < count; ++i)
    {
        max_diff = std::max(max_diff, std::fabs(pa[i] - pb[i]));
    }
    return max_diff;
}

}  // namespace

TEST(ImgprocCopyMakeBorder_TEST, u8_c3_matches_reference_across_border_modes)
{
    Mat src({6, 8}, CV_8UC3);
    fill_u8_pattern(src, 0x1234u);

    const std::vector<int> border_modes = {
        BORDER_CONSTANT,
        BORDER_REPLICATE,
        BORDER_REFLECT,
        BORDER_REFLECT_101,
        BORDER_WRAP,
        BORDER_REFLECT | BORDER_ISOLATED,
    };
    const Scalar border_value(17.0, 29.0, 43.0, 251.0);

    for (int border_mode : border_modes)
    {
        Mat actual;
        copyMakeBorder(src, actual, 2, 1, 3, 2, border_mode, border_value);
        const Mat expected = copy_make_border_reference<uchar>(src, 2, 1, 3, 2, border_mode, border_value);
        EXPECT_EQ(max_abs_diff_u8(actual, expected), 0) << "border_mode=" << border_mode;
    }
}

TEST(ImgprocCopyMakeBorder_TEST, f32_c4_matches_reference_for_replicate_and_wrap)
{
    Mat src({5, 7}, CV_32FC4);
    fill_f32_pattern(src);

    const std::vector<int> border_modes = {BORDER_REPLICATE, BORDER_WRAP, BORDER_REFLECT_101};
    for (int border_mode : border_modes)
    {
        Mat actual;
        copyMakeBorder(src, actual, 3, 2, 1, 4, border_mode, Scalar::all(0.0));
        const Mat expected = copy_make_border_reference<float>(src, 3, 2, 1, 4, border_mode, Scalar::all(0.0));
        EXPECT_LE(max_abs_diff_f32(actual, expected), 1e-6f) << "border_mode=" << border_mode;
    }
}

TEST(ImgprocCopyMakeBorder_TEST, roi_non_contiguous_matches_reference)
{
    Mat full({9, 12}, CV_8UC4);
    fill_u8_pattern(full, 0x5a5au);
    Mat roi = full(Range(2, 8), Range(1, 10));
    ASSERT_FALSE(roi.isContinuous());

    Mat actual;
    copyMakeBorder(roi, actual, 1, 2, 2, 1, BORDER_REFLECT, Scalar::all(0.0));
    const Mat expected = copy_make_border_reference<uchar>(roi, 1, 2, 2, 1, BORDER_REFLECT, Scalar::all(0.0));
    EXPECT_EQ(max_abs_diff_u8(actual, expected), 0);
}

TEST(ImgprocCopyMakeBorder_TEST, in_place_same_mat_is_supported)
{
    Mat src({4, 5}, CV_8UC1);
    fill_u8_pattern(src, 0x77u);

    const Mat expected = copy_make_border_reference<uchar>(src, 1, 2, 3, 1, BORDER_WRAP, Scalar::all(0.0));
    copyMakeBorder(src, src, 1, 2, 3, 1, BORDER_WRAP, Scalar::all(0.0));

    EXPECT_EQ(src.size[0], expected.size[0]);
    EXPECT_EQ(src.size[1], expected.size[1]);
    EXPECT_EQ(max_abs_diff_u8(src, expected), 0);
}

TEST(ImgprocCopyMakeBorder_TEST, throws_on_invalid_arguments)
{
    Mat empty;
    Mat dst;
    EXPECT_THROW(copyMakeBorder(empty, dst, 1, 1, 1, 1, BORDER_CONSTANT), Exception);

    Mat src_u16({3, 4}, CV_16UC1);
    src_u16.setTo(Scalar::all(7.0));
    EXPECT_THROW(copyMakeBorder(src_u16, dst, 1, 1, 1, 1, BORDER_CONSTANT), Exception);

    Mat src_u8({3, 4}, CV_8UC1);
    src_u8.setTo(Scalar::all(7.0));
    EXPECT_THROW(copyMakeBorder(src_u8, dst, -1, 0, 0, 0, BORDER_CONSTANT), Exception);
    EXPECT_THROW(copyMakeBorder(src_u8, dst, 1, 1, 1, 1, BORDER_TRANSPARENT), Exception);
}

TEST(ImgprocCopyMakeBorder_TEST, upstream_findcontours_border_preamble_semantics)
{
    Mat src({8, 10}, CV_8UC1);
    src.setTo(Scalar::all(0.0));

    Mat img;
    copyMakeBorder(src, img, 1, 1, 1, 1, BORDER_CONSTANT, Scalar::all(1.0));

    ASSERT_EQ(img.size[0], 10);
    ASSERT_EQ(img.size[1], 12);
    for (int y = 0; y < img.size[0]; ++y)
    {
        for (int x = 0; x < img.size[1]; ++x)
        {
            const bool is_border = (y == 0 || y == img.size[0] - 1 || x == 0 || x == img.size[1] - 1);
            const uchar expected = is_border ? static_cast<uchar>(1) : static_cast<uchar>(0);
            EXPECT_EQ(img.at<uchar>(y, x), expected) << "y=" << y << ", x=" << x;
        }
    }
}
