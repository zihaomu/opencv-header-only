#include "cvh.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
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

double kernel_value_ref(const Mat& kernel, int y, int x)
{
    return static_cast<double>(kernel.at<float>(y, x));
}

double sample_value_ref(const Mat& src, int y, int x, int c)
{
    if (src.depth() == CV_8U)
    {
        return static_cast<double>(src.at<uchar>(y, x, c));
    }
    return static_cast<double>(src.at<float>(y, x, c));
}

Mat filter2d_reference(const Mat& src,
                       int ddepth,
                       const Mat& kernel,
                       Point anchor,
                       double delta,
                       int borderType)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
    CV_Assert(kernel.dims == 2 && kernel.channels() == 1);
    CV_Assert(kernel.depth() == CV_32F);

    const int out_depth = ddepth < 0 ? src.depth() : ddepth;
    CV_Assert(out_depth == CV_8U || out_depth == CV_32F);

    const int rows = src.size[0];
    const int cols = src.size[1];
    const int cn = src.channels();
    const int krows = kernel.size[0];
    const int kcols = kernel.size[1];
    const int ax = anchor.x >= 0 ? anchor.x : (kcols / 2);
    const int ay = anchor.y >= 0 ? anchor.y : (krows / 2);
    const int border = normalize_border_type(borderType);

    Mat dst({rows, cols}, CV_MAKETYPE(out_depth, cn));
    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            for (int c = 0; c < cn; ++c)
            {
                double acc = delta;
                for (int ky = 0; ky < krows; ++ky)
                {
                    const int src_y = border_interpolate_ref(y + ky - ay, rows, border);
                    if (src_y < 0)
                    {
                        continue;
                    }
                    for (int kx = 0; kx < kcols; ++kx)
                    {
                        const int src_x = border_interpolate_ref(x + kx - ax, cols, border);
                        if (src_x < 0)
                        {
                            continue;
                        }
                        acc += sample_value_ref(src, src_y, src_x, c) * kernel_value_ref(kernel, ky, kx);
                    }
                }

                if (out_depth == CV_8U)
                {
                    dst.at<uchar>(y, x, c) = saturate_cast<uchar>(acc);
                }
                else
                {
                    dst.at<float>(y, x, c) = static_cast<float>(acc);
                }
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
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            for (int c = 0; c < src.channels(); ++c)
            {
                src.at<float>(y, x, c) = static_cast<float>(y * 0.375f + x * 0.625f + c * 1.5f);
            }
        }
    }
}

int max_abs_diff_u8(const Mat& a, const Mat& b)
{
    CV_Assert(a.type() == b.type());
    CV_Assert(a.dims == 2 && b.dims == 2);
    CV_Assert(a.size[0] == b.size[0] && a.size[1] == b.size[1]);

    int max_diff = 0;
    for (int y = 0; y < a.size[0]; ++y)
    {
        const uchar* pa = a.data + static_cast<size_t>(y) * a.step(0);
        const uchar* pb = b.data + static_cast<size_t>(y) * b.step(0);
        for (int x = 0; x < a.size[1] * a.channels(); ++x)
        {
            max_diff = std::max(max_diff, std::abs(static_cast<int>(pa[x]) - static_cast<int>(pb[x])));
        }
    }
    return max_diff;
}

float max_abs_diff_f32(const Mat& a, const Mat& b)
{
    CV_Assert(a.type() == b.type());
    CV_Assert(a.depth() == CV_32F);
    CV_Assert(a.dims == 2 && b.dims == 2);
    CV_Assert(a.size[0] == b.size[0] && a.size[1] == b.size[1]);

    float max_diff = 0.0f;
    for (int y = 0; y < a.size[0]; ++y)
    {
        const float* pa = reinterpret_cast<const float*>(a.data + static_cast<size_t>(y) * a.step(0));
        const float* pb = reinterpret_cast<const float*>(b.data + static_cast<size_t>(y) * b.step(0));
        for (int x = 0; x < a.size[1] * a.channels(); ++x)
        {
            max_diff = std::max(max_diff, std::fabs(pa[x] - pb[x]));
        }
    }
    return max_diff;
}

}  // namespace

TEST(ImgprocFilter2D_TEST, u8_c3_matches_reference_replicate_border)
{
    Mat src({7, 9}, CV_8UC3);
    fill_u8_pattern(src, 0x1234u);

    Mat kernel({3, 3}, CV_32FC1);
    kernel.at<float>(0, 0) = 0.0f;   kernel.at<float>(0, 1) = 0.25f;  kernel.at<float>(0, 2) = 0.0f;
    kernel.at<float>(1, 0) = 0.25f;  kernel.at<float>(1, 1) = 0.0f;   kernel.at<float>(1, 2) = 0.25f;
    kernel.at<float>(2, 0) = 0.0f;   kernel.at<float>(2, 1) = 0.25f;  kernel.at<float>(2, 2) = 0.0f;

    Mat actual;
    filter2D(src, actual, -1, kernel, Point(-1, -1), 3.0, BORDER_REPLICATE);
    const Mat expected = filter2d_reference(src, -1, kernel, Point(-1, -1), 3.0, BORDER_REPLICATE);

    EXPECT_EQ(max_abs_diff_u8(actual, expected), 0);
}

TEST(ImgprocFilter2D_TEST, f32_c1_custom_anchor_and_delta_matches_reference)
{
    Mat src({6, 8}, CV_32FC1);
    fill_f32_pattern(src);

    Mat kernel({2, 3}, CV_32FC1);
    kernel.at<float>(0, 0) = -0.25f;
    kernel.at<float>(0, 1) = 0.5f;
    kernel.at<float>(0, 2) = 0.125f;
    kernel.at<float>(1, 0) = 1.0f;
    kernel.at<float>(1, 1) = -0.375f;
    kernel.at<float>(1, 2) = 0.75f;

    Mat actual;
    filter2D(src, actual, CV_32F, kernel, Point(1, 0), -2.5, BORDER_REFLECT_101);
    const Mat expected = filter2d_reference(src, CV_32F, kernel, Point(1, 0), -2.5, BORDER_REFLECT_101);

    EXPECT_LE(max_abs_diff_f32(actual, expected), 1e-5f);
}

TEST(ImgprocFilter2D_TEST, u8_to_f32_output_depth_matches_reference)
{
    Mat src({5, 7}, CV_8UC1);
    fill_u8_pattern(src, 0x77u);

    Mat kernel({3, 1}, CV_32FC1);
    kernel.at<float>(0, 0) = 1.5f;
    kernel.at<float>(1, 0) = -1.0f;
    kernel.at<float>(2, 0) = 0.25f;

    Mat actual;
    filter2D(src, actual, CV_32F, kernel, Point(0, 1), 0.0, BORDER_CONSTANT);
    const Mat expected = filter2d_reference(src, CV_32F, kernel, Point(0, 1), 0.0, BORDER_CONSTANT);

    ASSERT_EQ(actual.type(), CV_32FC1);
    EXPECT_LE(max_abs_diff_f32(actual, expected), 1e-5f);
}

TEST(ImgprocFilter2D_TEST, roi_non_contiguous_and_inplace_are_supported)
{
    Mat full({10, 12}, CV_8UC4);
    fill_u8_pattern(full, 0x8877u);
    Mat roi = full(Range(2, 9), Range(1, 11));
    ASSERT_FALSE(roi.isContinuous());

    Mat kernel({3, 3}, CV_32FC1);
    kernel.setTo(0.0f);
    kernel.at<float>(1, 1) = 1.0f;

    Mat roi_actual;
    filter2D(roi, roi_actual, -1, kernel, Point(-1, -1), 0.0, BORDER_REFLECT);
    EXPECT_EQ(max_abs_diff_u8(roi_actual, roi), 0);

    Mat in_place = roi.clone();
    Mat expected = filter2d_reference(in_place, -1, kernel, Point(-1, -1), 0.0, BORDER_REFLECT);
    filter2D(in_place, in_place, -1, kernel, Point(-1, -1), 0.0, BORDER_REFLECT);
    EXPECT_EQ(max_abs_diff_u8(in_place, expected), 0);
}

TEST(ImgprocFilter2D_TEST, throws_on_invalid_arguments)
{
    Mat dst;
    Mat empty;
    Mat src({4, 5}, CV_8UC1);
    fill_u8_pattern(src, 0x123u);

    Mat kernel_good({3, 3}, CV_32FC1);
    kernel_good.setTo(1.0f / 9.0f);

    EXPECT_THROW(filter2D(empty, dst, -1, kernel_good), Exception);

    Mat kernel_bad_depth({3, 3}, CV_8UC1);
    kernel_bad_depth.setTo(1.0f);
    EXPECT_THROW(filter2D(src, dst, -1, kernel_bad_depth), Exception);

    Mat kernel_bad_channels({3, 3}, CV_32FC2);
    EXPECT_THROW(filter2D(src, dst, -1, kernel_bad_channels), Exception);

    Mat src_u16({4, 5}, CV_16UC1);
    EXPECT_THROW(filter2D(src_u16, dst, -1, kernel_good), Exception);

    EXPECT_THROW(filter2D(src, dst, CV_16S, kernel_good), Exception);
    EXPECT_THROW(filter2D(src, dst, -1, kernel_good, Point(3, 1)), Exception);
    EXPECT_THROW(filter2D(src, dst, -1, kernel_good, Point(-1, -1), 0.0, BORDER_WRAP), Exception);
}
