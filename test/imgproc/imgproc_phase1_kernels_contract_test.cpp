#include "cvh.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstring>

using namespace cvh;

TEST(ImgprocPhase1Kernels_TEST, structuring_elements_cover_shapes_anchor_and_errors)
{
    Mat rectangle = getStructuringElement(MORPH_RECT, Size(5, 3));
    EXPECT_EQ(countNonZero(rectangle), 15);

    Mat cross = getStructuringElement(MORPH_CROSS, Size(5, 3), Point(1, 0));
    EXPECT_EQ(countNonZero(cross), 7);
    EXPECT_EQ(cross.at<uchar>(0, 4), 1);
    EXPECT_EQ(cross.at<uchar>(2, 1), 1);
    EXPECT_EQ(cross.at<uchar>(2, 2), 0);

    Mat ellipse = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    EXPECT_EQ(ellipse.at<uchar>(0, 2), 1);
    EXPECT_EQ(ellipse.at<uchar>(2, 0), 1);
    EXPECT_EQ(ellipse.at<uchar>(0, 0), 0);

    EXPECT_THROW(
        getStructuringElement(MORPH_RECT, Size(0, 3)),
        Exception);
    EXPECT_THROW(
        getStructuringElement(MORPH_CROSS, Size(3, 3), Point(3, 1)),
        Exception);
}

TEST(ImgprocPhase1Kernels_TEST, gaussian_and_hanning_have_fixed_numeric_contracts)
{
    for (const int type : {CV_32F, CV_64F})
    {
        Mat gaussian = getGaussianKernel(7, 0.0, type);
        EXPECT_EQ(gaussian.shape(), MatShape({7, 1}));
        EXPECT_NEAR(sum(gaussian)[0], 1.0, type == CV_32F ? 1e-7 : 1e-15);
        for (int i = 0; i < 7; ++i)
        {
            const double left = type == CV_32F
                                    ? gaussian.at<float>(i, 0)
                                    : gaussian.at<double>(i, 0);
            const double right = type == CV_32F
                                     ? gaussian.at<float>(6 - i, 0)
                                     : gaussian.at<double>(6 - i, 0);
            EXPECT_DOUBLE_EQ(left, right);
        }

        Mat hanning;
        createHanningWindow(hanning, Size(5, 5), type);
        EXPECT_DOUBLE_EQ(
            type == CV_32F ? hanning.at<float>(0, 2) : hanning.at<double>(0, 2),
            0.0);
        EXPECT_NEAR(
            type == CV_32F ? hanning.at<float>(2, 2) : hanning.at<double>(2, 2),
            1.0,
            type == CV_32F ? 1e-7 : 1e-15);
    }
}

TEST(ImgprocPhase1Kernels_TEST, derivative_and_gabor_generators_are_stable)
{
    Mat kx;
    Mat ky;
    getDerivKernels(kx, ky, 1, 0, 3, false, CV_64F);
    EXPECT_DOUBLE_EQ(kx.at<double>(0, 0), -1.0);
    EXPECT_DOUBLE_EQ(kx.at<double>(2, 0), 1.0);
    EXPECT_DOUBLE_EQ(ky.at<double>(1, 0), 2.0);

    getDerivKernels(kx, ky, 1, 0, -1, true, CV_32F);
    EXPECT_FLOAT_EQ(kx.at<float>(0, 0), -1.0f);
    EXPECT_FLOAT_EQ(ky.at<float>(0, 0), 3.0f / 32.0f);
    EXPECT_NEAR(sum(ky)[0], 0.5, 1e-7);

    Mat gabor = getGaborKernel(
        Size(7, 5), 2.0, 0.3, 4.0, 0.8, 0.0, CV_64F);
    EXPECT_EQ(gabor.shape(), MatShape({5, 7}));
    EXPECT_TRUE(std::isfinite(gabor.at<double>(2, 3)));
    EXPECT_THROW(
        getGaborKernel(Size(3, 3), 0.0, 0.0, 2.0, 1.0),
        Exception);
}

TEST(ImgprocPhase1Kernels_TEST, integral_has_zero_border_and_multichannel_values)
{
    Mat src({2, 3}, CV_8UC3);
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 3; ++x)
        {
            for (int ch = 0; ch < 3; ++ch)
            {
                src.at<uchar>(y, x, ch) =
                    static_cast<uchar>(1 + y * 3 + x + ch);
            }
        }
    }
    Mat sum32;
    integral(src, sum32);
    ASSERT_EQ(sum32.shape(), MatShape({3, 4}));
    ASSERT_EQ(sum32.type(), CV_32SC3);
    EXPECT_EQ(sum32.at<int>(0, 3, 2), 0);
    EXPECT_EQ(sum32.at<int>(2, 3, 0), 21);
    EXPECT_EQ(sum32.at<int>(2, 3, 2), 33);

    Mat sum64;
    integral(src, sum64, CV_64F);
    EXPECT_DOUBLE_EQ(sum64.at<double>(2, 3, 0), 21.0);

    Mat roi_parent({4, 5}, CV_8UC1);
    roi_parent.setTo(Scalar::all(2));
    Mat roi = roi_parent(Range(1, 4), Range(1, 5));
    integral(roi, sum32);
    EXPECT_EQ(sum32.at<int>(3, 4), 24);
}

TEST(ImgprocPhase1Kernels_TEST, scharr_laplacian_and_spatial_gradient_share_semantics)
{
    Mat ramp({7, 9}, CV_8UC1);
    for (int y = 0; y < 7; ++y)
    {
        for (int x = 0; x < 9; ++x)
        {
            ramp.at<uchar>(y, x) = static_cast<uchar>(x + 2 * y);
        }
    }
    Mat scharr_x;
    Scharr(ramp, scharr_x, CV_16S, 1, 0);
    EXPECT_EQ(scharr_x.at<short>(3, 4), 32);

    Mat constant({7, 9}, CV_32FC3);
    constant.setTo(Scalar::all(5.0));
    Mat laplacian;
    Laplacian(constant, laplacian, CV_32F, 3);
    EXPECT_NEAR(norm(laplacian, NORM_INF), 0.0, 1e-6);

    Mat impulse({5, 5}, CV_32FC1);
    impulse.setTo(Scalar::all(0.0));
    impulse.at<float>(2, 2) = 1.0f;
    Laplacian(impulse, laplacian, CV_32F, 1, 1.0, 0.0, BORDER_CONSTANT);
    EXPECT_FLOAT_EQ(laplacian.at<float>(2, 2), -4.0f);

    Mat dx;
    Mat dy;
    Mat expected_dx;
    Mat expected_dy;
    spatialGradient(ramp, dx, dy);
    Sobel(ramp, expected_dx, CV_16S, 1, 0, 3);
    Sobel(ramp, expected_dy, CV_16S, 0, 1, 3);
    EXPECT_EQ(
        std::memcmp(
            dx.data,
            expected_dx.data,
            dx.total() * dx.elemSize()),
        0);
    EXPECT_EQ(
        std::memcmp(
            dy.data,
            expected_dy.data,
            dy.total() * dy.elemSize()),
        0);
}

TEST(ImgprocPhase1Kernels_TEST, squared_box_filter_uses_wide_accumulation)
{
    Mat large({35, 35}, CV_8UC1);
    large.setTo(Scalar::all(255));
    Mat normalized;
    sqrBoxFilter(
        large,
        normalized,
        CV_64F,
        Size(31, 31),
        Point(-1, -1),
        true,
        BORDER_REPLICATE);
    EXPECT_DOUBLE_EQ(normalized.at<double>(17, 17), 65025.0);

    Mat unnormalized;
    sqrBoxFilter(
        large,
        unnormalized,
        CV_64F,
        Size(31, 31),
        Point(-1, -1),
        false,
        BORDER_REPLICATE);
    EXPECT_DOUBLE_EQ(
        unnormalized.at<double>(17, 17), 65025.0 * 31.0 * 31.0);

    Mat color_parent({5, 7}, CV_32FC3);
    color_parent.setTo(Scalar(1.0, 2.0, 3.0));
    Mat color_roi = color_parent(Range(1, 5), Range(1, 6));
    Mat color_result;
    sqrBoxFilter(
        color_roi,
        color_result,
        CV_32F,
        Size(3, 3),
        Point(-1, -1),
        true,
        BORDER_REPLICATE | BORDER_ISOLATED);
    EXPECT_FLOAT_EQ(color_result.at<float>(2, 2, 0), 1.0f);
    EXPECT_FLOAT_EQ(color_result.at<float>(2, 2, 2), 9.0f);
}

TEST(ImgprocPhase1Kernels_TEST, invalid_integral_derivative_and_square_filter_inputs_throw)
{
    Mat f32({3, 3}, CV_32FC1);
    Mat out;
    EXPECT_THROW(integral(f32, out), Exception);
    EXPECT_THROW(Scharr(f32, out, CV_32F, 1, 1), Exception);
    EXPECT_THROW(Laplacian(f32, out, CV_32F, 7), Exception);
    EXPECT_THROW(spatialGradient(f32, out, out, 5), Exception);
    EXPECT_THROW(
        sqrBoxFilter(f32, out, CV_16S, Size(3, 3)),
        Exception);
}
