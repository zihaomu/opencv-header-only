#include "cvh.h"
#include "gtest/gtest.h"

#include <cstdlib>

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
    EXPECT_THROW(cvtColor(src_bgr, dst, -999), Exception);

    Mat src_fp32({8, 8}, CV_32FC3);
    EXPECT_THROW(cvtColor(src_fp32, dst, COLOR_BGR2GRAY), Exception);
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
