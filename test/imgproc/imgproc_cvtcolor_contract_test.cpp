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
