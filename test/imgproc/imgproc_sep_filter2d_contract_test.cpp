#include "cvh.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace cvh;

namespace
{

std::vector<float> collect_kernel_coeffs(const Mat& kernel)
{
    CV_Assert(kernel.dims == 2);
    CV_Assert(kernel.channels() == 1);
    CV_Assert(kernel.depth() == CV_32F);
    CV_Assert((kernel.size[0] == 1) || (kernel.size[1] == 1));

    const int len = kernel.size[0] == 1 ? kernel.size[1] : kernel.size[0];
    std::vector<float> coeffs(static_cast<size_t>(len), 0.0f);
    if (kernel.size[0] == 1)
    {
        for (int i = 0; i < len; ++i)
        {
            coeffs[static_cast<size_t>(i)] = kernel.at<float>(0, i);
        }
        return coeffs;
    }

    for (int i = 0; i < len; ++i)
    {
        coeffs[static_cast<size_t>(i)] = kernel.at<float>(i, 0);
    }
    return coeffs;
}

Mat outer_product_kernel(const Mat& kernelX, const Mat& kernelY)
{
    const std::vector<float> kx = collect_kernel_coeffs(kernelX);
    const std::vector<float> ky = collect_kernel_coeffs(kernelY);
    Mat kernel({static_cast<int>(ky.size()), static_cast<int>(kx.size())}, CV_32FC1);
    for (int y = 0; y < static_cast<int>(ky.size()); ++y)
    {
        for (int x = 0; x < static_cast<int>(kx.size()); ++x)
        {
            kernel.at<float>(y, x) = ky[static_cast<size_t>(y)] * kx[static_cast<size_t>(x)];
        }
    }
    return kernel;
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
                src.at<float>(y, x, c) = static_cast<float>(y * 0.75f + x * 0.5f + c * 1.25f);
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

TEST(ImgprocSepFilter2D_TEST, u8_c3_matches_filter2d_outer_product)
{
    Mat src({7, 9}, CV_8UC3);
    fill_u8_pattern(src, 0x1234u);

    Mat kernelX({1, 3}, CV_32FC1);
    kernelX.at<float>(0, 0) = 0.25f;
    kernelX.at<float>(0, 1) = 0.5f;
    kernelX.at<float>(0, 2) = 0.25f;

    Mat kernelY({3, 1}, CV_32FC1);
    kernelY.at<float>(0, 0) = 0.25f;
    kernelY.at<float>(1, 0) = 0.5f;
    kernelY.at<float>(2, 0) = 0.25f;

    Mat sep_actual;
    sepFilter2D(src, sep_actual, -1, kernelX, kernelY, Point(-1, -1), 2.0, BORDER_REPLICATE);

    Mat filter_actual;
    const Mat kernel2d = outer_product_kernel(kernelX, kernelY);
    filter2D(src, filter_actual, -1, kernel2d, Point(-1, -1), 2.0, BORDER_REPLICATE);

    EXPECT_EQ(max_abs_diff_u8(sep_actual, filter_actual), 0);
}

TEST(ImgprocSepFilter2D_TEST, f32_custom_anchor_delta_matches_filter2d_outer_product)
{
    Mat src({6, 8}, CV_32FC1);
    fill_f32_pattern(src);

    Mat kernelX({3, 1}, CV_32FC1);
    kernelX.at<float>(0, 0) = -0.25f;
    kernelX.at<float>(1, 0) = 0.75f;
    kernelX.at<float>(2, 0) = 0.5f;

    Mat kernelY({1, 2}, CV_32FC1);
    kernelY.at<float>(0, 0) = 1.25f;
    kernelY.at<float>(0, 1) = -0.5f;

    const Point anchor(1, 0);
    Mat sep_actual;
    sepFilter2D(src, sep_actual, CV_32F, kernelX, kernelY, anchor, -1.5, BORDER_REFLECT_101);

    Mat filter_actual;
    const Mat kernel2d = outer_product_kernel(kernelX, kernelY);
    filter2D(src, filter_actual, CV_32F, kernel2d, anchor, -1.5, BORDER_REFLECT_101);

    EXPECT_LE(max_abs_diff_f32(sep_actual, filter_actual), 1e-5f);
}

TEST(ImgprocSepFilter2D_TEST, u8_to_f32_depth_matches_filter2d_outer_product)
{
    Mat src({5, 7}, CV_8UC1);
    fill_u8_pattern(src, 0x55u);

    Mat kernelX({1, 5}, CV_32FC1);
    kernelX.at<float>(0, 0) = 0.0f;
    kernelX.at<float>(0, 1) = 0.25f;
    kernelX.at<float>(0, 2) = 0.5f;
    kernelX.at<float>(0, 3) = 0.25f;
    kernelX.at<float>(0, 4) = 0.0f;

    Mat kernelY({3, 1}, CV_32FC1);
    kernelY.at<float>(0, 0) = 0.5f;
    kernelY.at<float>(1, 0) = 0.0f;
    kernelY.at<float>(2, 0) = -0.5f;

    Mat sep_actual;
    sepFilter2D(src, sep_actual, CV_32F, kernelX, kernelY, Point(-1, -1), 0.0, BORDER_CONSTANT);

    Mat filter_actual;
    const Mat kernel2d = outer_product_kernel(kernelX, kernelY);
    filter2D(src, filter_actual, CV_32F, kernel2d, Point(-1, -1), 0.0, BORDER_CONSTANT);

    ASSERT_EQ(sep_actual.type(), CV_32FC1);
    EXPECT_LE(max_abs_diff_f32(sep_actual, filter_actual), 1e-5f);
}

TEST(ImgprocSepFilter2D_TEST, roi_non_contiguous_and_inplace_are_supported)
{
    Mat full({10, 12}, CV_8UC4);
    fill_u8_pattern(full, 0x8877u);
    Mat roi = full(Range(2, 9), Range(1, 11));
    ASSERT_FALSE(roi.isContinuous());

    Mat kernelX({1, 3}, CV_32FC1);
    kernelX.at<float>(0, 0) = 0.0f;
    kernelX.at<float>(0, 1) = 1.0f;
    kernelX.at<float>(0, 2) = 0.0f;

    Mat kernelY({3, 1}, CV_32FC1);
    kernelY.at<float>(0, 0) = 0.0f;
    kernelY.at<float>(1, 0) = 1.0f;
    kernelY.at<float>(2, 0) = 0.0f;

    Mat sep_actual;
    sepFilter2D(roi, sep_actual, -1, kernelX, kernelY, Point(-1, -1), 0.0, BORDER_REFLECT);
    EXPECT_EQ(max_abs_diff_u8(sep_actual, roi), 0);

    Mat in_place = roi.clone();
    Mat expected;
    sepFilter2D(in_place, expected, -1, kernelX, kernelY, Point(-1, -1), 0.0, BORDER_REFLECT);
    sepFilter2D(in_place, in_place, -1, kernelX, kernelY, Point(-1, -1), 0.0, BORDER_REFLECT);
    EXPECT_EQ(max_abs_diff_u8(in_place, expected), 0);
}

TEST(ImgprocSepFilter2D_TEST, throws_on_invalid_arguments)
{
    Mat dst;
    Mat empty;
    Mat src({4, 5}, CV_8UC1);
    fill_u8_pattern(src, 0x123u);

    Mat kx_good({1, 3}, CV_32FC1);
    kx_good.at<float>(0, 0) = 0.25f;
    kx_good.at<float>(0, 1) = 0.5f;
    kx_good.at<float>(0, 2) = 0.25f;

    Mat ky_good({3, 1}, CV_32FC1);
    ky_good.at<float>(0, 0) = 0.25f;
    ky_good.at<float>(1, 0) = 0.5f;
    ky_good.at<float>(2, 0) = 0.25f;

    EXPECT_THROW(sepFilter2D(empty, dst, -1, kx_good, ky_good), Exception);

    Mat kx_bad_depth({1, 3}, CV_8UC1);
    kx_bad_depth.setTo(1.0f);
    EXPECT_THROW(sepFilter2D(src, dst, -1, kx_bad_depth, ky_good), Exception);

    Mat ky_bad_shape({3, 3}, CV_32FC1);
    EXPECT_THROW(sepFilter2D(src, dst, -1, kx_good, ky_bad_shape), Exception);

    Mat src_u16({4, 5}, CV_16UC1);
    EXPECT_THROW(sepFilter2D(src_u16, dst, -1, kx_good, ky_good), Exception);

    EXPECT_THROW(sepFilter2D(src, dst, CV_16S, kx_good, ky_good), Exception);
    EXPECT_THROW(sepFilter2D(src, dst, -1, kx_good, ky_good, Point(3, 0)), Exception);
    EXPECT_THROW(sepFilter2D(src, dst, -1, kx_good, ky_good, Point(-1, -1), 0.0, BORDER_WRAP), Exception);
}
