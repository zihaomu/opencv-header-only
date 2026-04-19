#include "cvh.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

using namespace cvh;

namespace
{

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

uchar lut_u8_value(const Mat& lut, int index, int channel)
{
    CV_Assert(lut.depth() == CV_8U);
    CV_Assert(lut.dims == 2);
    CV_Assert(index >= 0 && index < 256);
    const int lut_ch = lut.channels();
    CV_Assert(lut_ch == 1 || channel < lut_ch);
    const int row = index / lut.size[1];
    const int col = index - row * lut.size[1];
    return lut.at<uchar>(row, col, lut_ch == 1 ? 0 : channel);
}

float lut_f32_value(const Mat& lut, int index, int channel)
{
    CV_Assert(lut.depth() == CV_32F);
    CV_Assert(lut.dims == 2);
    CV_Assert(index >= 0 && index < 256);
    const int lut_ch = lut.channels();
    CV_Assert(lut_ch == 1 || channel < lut_ch);
    const int row = index / lut.size[1];
    const int col = index - row * lut.size[1];
    return lut.at<float>(row, col, lut_ch == 1 ? 0 : channel);
}

Mat lut_reference_u8(const Mat& src, const Mat& lut)
{
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(lut.depth() == CV_8U);
    CV_Assert(src.dims == 2 && lut.dims == 2);
    CV_Assert(lut.total() == 256);
    CV_Assert(lut.channels() == 1 || lut.channels() == src.channels());

    Mat dst({src.size[0], src.size[1]}, CV_MAKETYPE(CV_8U, src.channels()));
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            for (int c = 0; c < src.channels(); ++c)
            {
                const int index = static_cast<int>(src.at<uchar>(y, x, c));
                dst.at<uchar>(y, x, c) = lut_u8_value(lut, index, c);
            }
        }
    }
    return dst;
}

Mat lut_reference_f32(const Mat& src, const Mat& lut)
{
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(lut.depth() == CV_32F);
    CV_Assert(src.dims == 2 && lut.dims == 2);
    CV_Assert(lut.total() == 256);
    CV_Assert(lut.channels() == 1 || lut.channels() == src.channels());

    Mat dst({src.size[0], src.size[1]}, CV_MAKETYPE(CV_32F, src.channels()));
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            for (int c = 0; c < src.channels(); ++c)
            {
                const int index = static_cast<int>(src.at<uchar>(y, x, c));
                dst.at<float>(y, x, c) = lut_f32_value(lut, index, c);
            }
        }
    }
    return dst;
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

TEST(ImgprocLUT_TEST, u8_c1_lut_c1_matches_reference)
{
    Mat src({5, 7}, CV_8UC1);
    fill_u8_pattern(src, 0x1234u);

    Mat lut({1, 256}, CV_8UC1);
    for (int i = 0; i < 256; ++i)
    {
        lut.at<uchar>(0, i) = static_cast<uchar>(255 - i);
    }

    Mat actual;
    LUT(src, lut, actual);
    const Mat expected = lut_reference_u8(src, lut);

    ASSERT_EQ(actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(actual, expected), 0);
}

TEST(ImgprocLUT_TEST, u8_c3_lut_c1_roi_matches_reference)
{
    Mat full({8, 10}, CV_8UC3);
    fill_u8_pattern(full, 0x88u);
    Mat src = full(Range(1, 7), Range(2, 9));
    ASSERT_FALSE(src.isContinuous());

    Mat lut({1, 256}, CV_8UC1);
    for (int i = 0; i < 256; ++i)
    {
        lut.at<uchar>(0, i) = static_cast<uchar>((i * 7 + 13) & 0xFF);
    }

    Mat actual;
    LUT(src, lut, actual);
    const Mat expected = lut_reference_u8(src, lut);

    ASSERT_EQ(actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(actual, expected), 0);
}

TEST(ImgprocLUT_TEST, u8_c4_lut_c4_matches_reference)
{
    Mat src({4, 6}, CV_8UC4);
    fill_u8_pattern(src, 0xaaaa55u);

    Mat lut({1, 256}, CV_8UC4);
    for (int i = 0; i < 256; ++i)
    {
        lut.at<uchar>(0, i, 0) = static_cast<uchar>(i);
        lut.at<uchar>(0, i, 1) = static_cast<uchar>(255 - i);
        lut.at<uchar>(0, i, 2) = static_cast<uchar>((i * 3) & 0xFF);
        lut.at<uchar>(0, i, 3) = static_cast<uchar>((i * 5 + 1) & 0xFF);
    }

    Mat actual;
    LUT(src, lut, actual);
    const Mat expected = lut_reference_u8(src, lut);

    ASSERT_EQ(actual.type(), CV_8UC4);
    EXPECT_EQ(max_abs_diff_u8(actual, expected), 0);
}

TEST(ImgprocLUT_TEST, output_depth_follows_lut_f32)
{
    Mat src({3, 5}, CV_8UC1);
    fill_u8_pattern(src, 0x42u);

    Mat lut({1, 256}, CV_32FC1);
    for (int i = 0; i < 256; ++i)
    {
        lut.at<float>(0, i) = static_cast<float>(i) * 0.25f + 1.5f;
    }

    Mat actual;
    LUT(src, lut, actual);
    const Mat expected = lut_reference_f32(src, lut);

    ASSERT_EQ(actual.type(), CV_32FC1);
    EXPECT_LE(max_abs_diff_f32(actual, expected), 1e-6f);
}

TEST(ImgprocLUT_TEST, supports_non_contiguous_lut_storage)
{
    Mat lut_storage({2, 300}, CV_8UC1);
    lut_storage.setTo(Scalar::all(0.0));
    Mat lut = lut_storage(Range(0, 2), Range(20, 148));
    ASSERT_FALSE(lut.isContinuous());
    ASSERT_EQ(lut.total(), 256u);

    for (int i = 0; i < 256; ++i)
    {
        const int row = i / lut.size[1];
        const int col = i - row * lut.size[1];
        lut.at<uchar>(row, col) = static_cast<uchar>((i * 11 + 3) & 0xFF);
    }

    Mat src({4, 7}, CV_8UC1);
    fill_u8_pattern(src, 0x9876u);

    Mat actual;
    LUT(src, lut, actual);
    const Mat expected = lut_reference_u8(src, lut);

    EXPECT_EQ(max_abs_diff_u8(actual, expected), 0);
}

TEST(ImgprocLUT_TEST, supports_in_place_src_equals_dst)
{
    Mat src({5, 5}, CV_8UC1);
    fill_u8_pattern(src, 0x2222u);

    Mat lut({1, 256}, CV_8UC1);
    for (int i = 0; i < 256; ++i)
    {
        lut.at<uchar>(0, i) = static_cast<uchar>((255 - i) ^ 0x5Au);
    }

    const Mat expected = lut_reference_u8(src, lut);
    LUT(src, lut, src);

    ASSERT_EQ(src.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(src, expected), 0);
}

TEST(ImgprocLUT_TEST, throws_on_invalid_arguments)
{
    Mat dst;
    Mat empty;
    Mat lut_ok({1, 256}, CV_8UC1);
    lut_ok.setTo(Scalar::all(0.0));

    EXPECT_THROW(LUT(empty, lut_ok, dst), Exception);

    Mat src_ok({3, 4}, CV_8UC1);
    src_ok.setTo(Scalar::all(7.0));
    EXPECT_THROW(LUT(src_ok, empty, dst), Exception);

    Mat src_u16({3, 4}, CV_16UC1);
    src_u16.setTo(Scalar::all(7.0));
    EXPECT_THROW(LUT(src_u16, lut_ok, dst), Exception);

    Mat lut_bad_total({1, 255}, CV_8UC1);
    lut_bad_total.setTo(Scalar::all(1.0));
    EXPECT_THROW(LUT(src_ok, lut_bad_total, dst), Exception);

    Mat src_c3({2, 3}, CV_8UC3);
    fill_u8_pattern(src_c3, 1u);
    Mat lut_c2({1, 256}, CV_8UC2);
    lut_c2.setTo(Scalar::all(0.0));
    EXPECT_THROW(LUT(src_c3, lut_c2, dst), Exception);

    Mat src_3d({2, 3, 4}, CV_8UC1);
    src_3d.setTo(Scalar::all(0.0));
    EXPECT_THROW(LUT(src_3d, lut_ok, dst), Exception);
}
