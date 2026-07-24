#include "cvh.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstring>
#include <limits>

using namespace cvh;

namespace
{

void fill_pattern(Mat& mat)
{
    for (int y = 0; y < mat.size.p[0]; ++y)
    {
        for (int x = 0; x < mat.size.p[1]; ++x)
        {
            for (int ch = 0; ch < mat.channels(); ++ch)
            {
                mat.at<uchar>(y, x, ch) =
                    static_cast<uchar>((17 * y + 23 * x + 31 * ch) & 255);
            }
        }
    }
}

}  // namespace

TEST(ImgprocPhase1Intensity_TEST, median_blur_handles_boundaries_roi_and_in_place)
{
    Mat impulse({3, 3}, CV_8UC1);
    impulse.setTo(Scalar::all(0));
    impulse.at<uchar>(1, 1) = 255;
    Mat filtered;
    medianBlur(impulse, filtered, 3);
    EXPECT_EQ(filtered.at<uchar>(1, 1), 0);
    EXPECT_EQ(filtered.at<uchar>(0, 0), 0);

    Mat parent({6, 8}, CV_8UC3);
    fill_pattern(parent);
    Mat roi = parent(Range(1, 6), Range(2, 8));
    Mat expected;
    medianBlur(roi, expected, 5);
    Mat in_place = roi.clone();
    medianBlur(in_place, in_place, 5);
    EXPECT_EQ(
        std::memcmp(
            expected.data,
            in_place.data,
            expected.total() * expected.elemSize()),
        0);

    Mat one_row({1, 5}, CV_32FC1);
    for (int x = 0; x < 5; ++x)
    {
        one_row.at<float>(0, x) = static_cast<float>(x);
    }
    medianBlur(one_row, filtered, 3);
    EXPECT_FLOAT_EQ(filtered.at<float>(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(filtered.at<float>(0, 4), 4.0f);
    EXPECT_THROW(medianBlur(one_row, filtered, 7), Exception);
    EXPECT_THROW(medianBlur(impulse, filtered, 4), Exception);
}

TEST(ImgprocPhase1Intensity_TEST, bilateral_filter_preserves_constants_and_rejects_alias)
{
    Mat constant({1, 7}, CV_8UC3);
    constant.setTo(Scalar(11, 37, 201));
    Mat filtered;
    bilateralFilter(
        constant, filtered, 5, 30.0, 2.0, BORDER_REFLECT_101);
    for (int x = 0; x < 7; ++x)
    {
        EXPECT_EQ(filtered.at<uchar>(0, x, 0), 11);
        EXPECT_EQ(filtered.at<uchar>(0, x, 1), 37);
        EXPECT_EQ(filtered.at<uchar>(0, x, 2), 201);
    }

    Mat edge({5, 5}, CV_32FC1);
    for (int y = 0; y < 5; ++y)
    {
        for (int x = 0; x < 5; ++x)
        {
            edge.at<float>(y, x) = x < 2 ? 0.0f : 100.0f;
        }
    }
    bilateralFilter(edge, filtered, 3, 1.0, 2.0);
    EXPECT_LT(filtered.at<float>(2, 1), 1.0f);
    EXPECT_GT(filtered.at<float>(2, 2), 99.0f);
    EXPECT_THROW(
        bilateralFilter(edge, edge, 3, 1.0, 2.0),
        Exception);
    EXPECT_THROW(
        bilateralFilter(edge, filtered, 3, 1.0, 2.0, BORDER_WRAP),
        Exception);
    EXPECT_THROW(
        bilateralFilter(
            edge,
            filtered,
            3,
            std::numeric_limits<double>::quiet_NaN(),
            2.0),
        Exception);
}

TEST(ImgprocPhase1Intensity_TEST, stack_blur_has_triangular_kernel_and_alias_contract)
{
    Mat impulse({1, 5}, CV_32FC1);
    impulse.setTo(Scalar::all(0));
    impulse.at<float>(0, 2) = 9.0f;
    Mat filtered;
    stackBlur(impulse, filtered, Size(3, 1));
    EXPECT_FLOAT_EQ(filtered.at<float>(0, 1), 2.25f);
    EXPECT_FLOAT_EQ(filtered.at<float>(0, 2), 4.5f);
    EXPECT_FLOAT_EQ(filtered.at<float>(0, 3), 2.25f);

    Mat color({5, 4}, CV_8UC4);
    color.setTo(Scalar(2, 20, 90, 255));
    stackBlur(color, color, Size(5, 3));
    EXPECT_EQ(color.at<uchar>(2, 2, 0), 2);
    EXPECT_EQ(color.at<uchar>(2, 2, 3), 255);
    EXPECT_THROW(stackBlur(color, filtered, Size(4, 3)), Exception);
}

TEST(ImgprocPhase1Intensity_TEST, adaptive_threshold_covers_mean_gaussian_and_in_place)
{
    Mat ramp({7, 9}, CV_8UC1);
    for (int y = 0; y < ramp.size.p[0]; ++y)
    {
        for (int x = 0; x < ramp.size.p[1]; ++x)
        {
            ramp.at<uchar>(y, x) =
                static_cast<uchar>(10 * x + y);
        }
    }
    Mat mean_binary;
    adaptiveThreshold(
        ramp,
        mean_binary,
        200,
        ADAPTIVE_THRESH_MEAN_C,
        THRESH_BINARY,
        3,
        2.0);
    EXPECT_EQ(mean_binary.type(), CV_8UC1);
    EXPECT_TRUE(
        mean_binary.at<uchar>(3, 4) == 0 ||
        mean_binary.at<uchar>(3, 4) == 200);

    Mat gaussian_inverse = ramp.clone();
    adaptiveThreshold(
        gaussian_inverse,
        gaussian_inverse,
        255,
        ADAPTIVE_THRESH_GAUSSIAN_C,
        THRESH_BINARY_INV,
        5,
        -1.25);
    EXPECT_EQ(gaussian_inverse.shape(), ramp.shape());
    EXPECT_THROW(
        adaptiveThreshold(
            ramp,
            mean_binary,
            255,
            ADAPTIVE_THRESH_MEAN_C,
            THRESH_BINARY,
            4,
            1.0),
        Exception);
    EXPECT_THROW(
        adaptiveThreshold(
            ramp,
            mean_binary,
            255,
            ADAPTIVE_THRESH_MEAN_C,
            THRESH_BINARY,
            3,
            std::numeric_limits<double>::infinity()),
        Exception);
    EXPECT_THROW(
        adaptiveThreshold(
            ramp,
            mean_binary,
            255,
            99,
            THRESH_BINARY,
            3,
            1.0),
        Exception);
}

TEST(ImgprocPhase1Intensity_TEST, threshold_with_mask_preserves_unselected_pixels)
{
    Mat src({3, 4}, CV_8UC3);
    fill_pattern(src);
    Mat mask({3, 4}, CV_8UC1);
    for (int y = 0; y < 3; ++y)
    {
        for (int x = 0; x < 4; ++x)
        {
            mask.at<uchar>(y, x) = ((x + y) & 1) ? 255 : 0;
        }
    }
    Mat dst({3, 4}, CV_8UC3);
    dst.setTo(Scalar(7, 8, 9));
    const double used = thresholdWithMask(
        src, dst, mask, 60.8, 200.0, THRESH_BINARY);
    EXPECT_DOUBLE_EQ(used, 60.0);
    for (int y = 0; y < 3; ++y)
    {
        for (int x = 0; x < 4; ++x)
        {
            if (mask.at<uchar>(y, x) == 0)
            {
                EXPECT_EQ(dst.at<uchar>(y, x, 0), 7);
                EXPECT_EQ(dst.at<uchar>(y, x, 2), 9);
            }
        }
    }

    Mat auto_src({1, 6}, CV_8UC1);
    for (int x = 0; x < 6; ++x)
    {
        auto_src.at<uchar>(0, x) =
            static_cast<uchar>(x < 3 ? 10 : 200);
    }
    Mat auto_mask({1, 6}, CV_8UC1);
    auto_mask.setTo(Scalar::all(255));
    Mat auto_dst({1, 6}, CV_8UC1);
    auto_dst.setTo(Scalar::all(17));
    EXPECT_GE(
        thresholdWithMask(
            auto_src,
            auto_dst,
            auto_mask,
            0,
            255,
            THRESH_BINARY | THRESH_OTSU),
        0.0);

    Mat wrong_dst;
    EXPECT_THROW(
        thresholdWithMask(
            src, wrong_dst, mask, 10, 255, THRESH_BINARY),
        Exception);
}

TEST(ImgprocPhase1Intensity_TEST, equalize_hist_handles_constant_bimodal_ramp_and_roi)
{
    Mat constant({4, 5}, CV_8UC1);
    constant.setTo(Scalar::all(73));
    Mat output;
    equalizeHist(constant, output);
    EXPECT_EQ(output.at<uchar>(2, 3), 73);

    Mat bimodal({1, 6}, CV_8UC1);
    for (int x = 0; x < 6; ++x)
    {
        bimodal.at<uchar>(0, x) =
            static_cast<uchar>(x < 3 ? 10 : 200);
    }
    equalizeHist(bimodal, output);
    EXPECT_EQ(output.at<uchar>(0, 0), 0);
    EXPECT_EQ(output.at<uchar>(0, 5), 255);

    Mat parent({3, 258}, CV_8UC1);
    parent.setTo(Scalar::all(0));
    Mat ramp = parent(Range(1, 2), Range(1, 257));
    for (int x = 0; x < 256; ++x)
    {
        ramp.at<uchar>(0, x) = static_cast<uchar>(x);
    }
    equalizeHist(ramp, ramp);
    for (int x = 0; x < 256; ++x)
    {
        EXPECT_EQ(ramp.at<uchar>(0, x), x);
    }
}

TEST(ImgprocPhase1Intensity_TEST, apply_color_map_uses_bgr_and_user_lut)
{
    Mat values({1, 3}, CV_8UC1);
    values.at<uchar>(0, 0) = 0;
    values.at<uchar>(0, 1) = 128;
    values.at<uchar>(0, 2) = 255;
    Mat colored;
    applyColorMap(values, colored, COLORMAP_AUTUMN);
    EXPECT_EQ(colored.type(), CV_8UC3);
    EXPECT_EQ(colored.at<uchar>(0, 0, 0), 0);
    EXPECT_EQ(colored.at<uchar>(0, 0, 2), 255);
    EXPECT_EQ(colored.at<uchar>(0, 2, 1), 255);

    Mat lookup_parent({258, 3}, CV_8UC3);
    Mat lookup = lookup_parent(Range(1, 257), Range(1, 2));
    for (int i = 0; i < 256; ++i)
    {
        lookup.at<uchar>(i, 0, 0) = static_cast<uchar>(i);
        lookup.at<uchar>(i, 0, 1) = static_cast<uchar>(255 - i);
        lookup.at<uchar>(i, 0, 2) = 17;
    }
    applyColorMap(values, colored, lookup);
    EXPECT_EQ(colored.at<uchar>(0, 1, 0), 128);
    EXPECT_EQ(colored.at<uchar>(0, 1, 1), 127);
    EXPECT_EQ(colored.at<uchar>(0, 1, 2), 17);
    EXPECT_THROW(
        applyColorMap(values, colored, COLORMAP_VIRIDIS),
        Exception);
}
