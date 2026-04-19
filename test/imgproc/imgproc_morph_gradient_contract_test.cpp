#include "cvh.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
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

void fill_u8_pattern(Mat& src)
{
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.dims == 2);
    const int rows = src.size[0];
    const int cols = src.size[1];
    const int cn = src.channels();

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            for (int c = 0; c < cn; ++c)
            {
                src.at<uchar>(y, x, c) = static_cast<uchar>((y * 31 + x * 17 + c * 13) & 0xFF);
            }
        }
    }
}

Mat morph_reference_u8(const Mat& src, bool is_erode, int borderType, const Scalar& borderValue)
{
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.dims == 2);
    const int rows = src.size[0];
    const int cols = src.size[1];
    const int cn = src.channels();
    const size_t src_step = src.step(0);

    Mat dst({rows, cols}, src.type());
    const size_t dst_step = dst.step(0);
    const int border = normalize_border_type(borderType);

    for (int y = 0; y < rows; ++y)
    {
        uchar* dst_row = dst.data + static_cast<size_t>(y) * dst_step;
        for (int x = 0; x < cols; ++x)
        {
            uchar* out_px = dst_row + static_cast<size_t>(x) * cn;
            for (int c = 0; c < cn; ++c)
            {
                int best = is_erode ? 255 : 0;
                for (int ky = -1; ky <= 1; ++ky)
                {
                    for (int kx = -1; kx <= 1; ++kx)
                    {
                        const int sy = border_interpolate_ref(y + ky, rows, border);
                        const int sx = border_interpolate_ref(x + kx, cols, border);
                        int value = 0;
                        if (sy < 0 || sx < 0)
                        {
                            value = saturate_cast<uchar>(borderValue.val[c]);
                        }
                        else
                        {
                            const uchar* src_row = src.data + static_cast<size_t>(sy) * src_step;
                            value = src_row[static_cast<size_t>(sx) * cn + c];
                        }

                        if (is_erode)
                        {
                            best = std::min(best, value);
                        }
                        else
                        {
                            best = std::max(best, value);
                        }
                    }
                }
                out_px[c] = static_cast<uchar>(best);
            }
        }
    }

    return dst;
}

Mat sobel_reference_u8_to_f32(const Mat& src, int dx, int dy, int borderType)
{
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.dims == 2);
    CV_Assert((dx == 1 && dy == 0) || (dx == 0 && dy == 1));

    const int rows = src.size[0];
    const int cols = src.size[1];
    const int cn = src.channels();
    const size_t src_step = src.step(0);
    const int border = normalize_border_type(borderType);

    Mat dst({rows, cols}, CV_MAKETYPE(CV_32F, cn));
    const size_t dst_step = dst.step(0);

    for (int y = 0; y < rows; ++y)
    {
        float* dst_row = reinterpret_cast<float*>(dst.data + static_cast<size_t>(y) * dst_step);
        for (int x = 0; x < cols; ++x)
        {
            const int y0 = border_interpolate_ref(y - 1, rows, border);
            const int y1 = border_interpolate_ref(y, rows, border);
            const int y2 = border_interpolate_ref(y + 1, rows, border);
            const int x0 = border_interpolate_ref(x - 1, cols, border);
            const int x1 = border_interpolate_ref(x, cols, border);
            const int x2 = border_interpolate_ref(x + 1, cols, border);

            for (int c = 0; c < cn; ++c)
            {
                const auto sample = [&](int sy, int sx) -> float {
                    const uchar* src_row = src.data + static_cast<size_t>(sy) * src_step;
                    return static_cast<float>(src_row[static_cast<size_t>(sx) * cn + c]);
                };

                float value = 0.0f;
                if (dx == 1)
                {
                    value = (sample(y0, x2) + 2.0f * sample(y1, x2) + sample(y2, x2)) -
                            (sample(y0, x0) + 2.0f * sample(y1, x0) + sample(y2, x0));
                }
                else
                {
                    value = (sample(y2, x0) + 2.0f * sample(y2, x1) + sample(y2, x2)) -
                            (sample(y0, x0) + 2.0f * sample(y0, x1) + sample(y0, x2));
                }
                dst_row[static_cast<size_t>(x) * cn + c] = value;
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

int max_abs_diff_s16(const Mat& a, const Mat& b)
{
    CV_Assert(a.type() == b.type());
    CV_Assert(a.total() == b.total());
    CV_Assert(a.channels() == b.channels());
    CV_Assert(a.depth() == CV_16S);
    const short* pa = reinterpret_cast<const short*>(a.data);
    const short* pb = reinterpret_cast<const short*>(b.data);
    const size_t count = a.total() * static_cast<size_t>(a.channels());
    int max_diff = 0;
    for (size_t i = 0; i < count; ++i)
    {
        max_diff = std::max(max_diff, std::abs(static_cast<int>(pa[i]) - static_cast<int>(pb[i])));
    }
    return max_diff;
}

std::uint32_t lcg_next(std::uint32_t state)
{
    return state * 1664525u + 1013904223u;
}

void fill_u8_lcg(Mat& src, std::uint32_t seed)
{
    CV_Assert(src.depth() == CV_8U);
    const size_t count = src.total() * static_cast<size_t>(src.channels());
    for (size_t i = 0; i < count; ++i)
    {
        seed = lcg_next(seed);
        src.data[i] = static_cast<uchar>((seed >> 24) & 0xFFu);
    }
}

}  // namespace

TEST(ImgprocMorphGradient_TEST, erode_dilate_u8_c1_matches_reference)
{
    Mat src({7, 9}, CV_8UC1);
    fill_u8_pattern(src);

    Mat erode_actual;
    Mat dilate_actual;
    erode(src, erode_actual, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
    dilate(src, dilate_actual, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);

    const Mat erode_expected = morph_reference_u8(src, true, BORDER_REPLICATE, Scalar::all(255.0));
    const Mat dilate_expected = morph_reference_u8(src, false, BORDER_REPLICATE, Scalar::all(0.0));

    EXPECT_EQ(max_abs_diff_u8(erode_actual, erode_expected), 0);
    EXPECT_EQ(max_abs_diff_u8(dilate_actual, dilate_expected), 0);
}

TEST(ImgprocMorphGradient_TEST, erode_dilate_u8_c3_roi_matches_reference)
{
    Mat src_full({8, 10}, CV_8UC3);
    fill_u8_pattern(src_full);
    Mat roi = src_full(Range(1, 7), Range(2, 9));

    Mat erode_actual;
    Mat dilate_actual;
    erode(roi, erode_actual, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
    dilate(roi, dilate_actual, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);

    const Mat erode_expected = morph_reference_u8(roi, true, BORDER_REPLICATE, Scalar::all(255.0));
    const Mat dilate_expected = morph_reference_u8(roi, false, BORDER_REPLICATE, Scalar::all(0.0));

    EXPECT_EQ(max_abs_diff_u8(erode_actual, erode_expected), 0);
    EXPECT_EQ(max_abs_diff_u8(dilate_actual, dilate_expected), 0);
}

TEST(ImgprocMorphGradient_TEST, morphologyEx_open_close_gradient_match_reference)
{
    Mat src({9, 11}, CV_8UC3);
    fill_u8_pattern(src);

    Mat expected_erode = morph_reference_u8(src, true, BORDER_REPLICATE, Scalar::all(255.0));
    Mat expected_dilate = morph_reference_u8(src, false, BORDER_REPLICATE, Scalar::all(0.0));

    Mat expected_open;
    dilate(expected_erode, expected_open, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
    Mat expected_close;
    erode(expected_dilate, expected_close, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);

    Mat expected_gradient({src.size[0], src.size[1]}, src.type());
    const size_t count = src.total() * static_cast<size_t>(src.channels());
    for (size_t i = 0; i < count; ++i)
    {
        expected_gradient.data[i] =
            static_cast<uchar>(static_cast<int>(expected_dilate.data[i]) - static_cast<int>(expected_erode.data[i]));
    }

    Mat actual_open;
    Mat actual_close;
    Mat actual_gradient;
    morphologyEx(src, actual_open, MORPH_OPEN, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
    morphologyEx(src, actual_close, MORPH_CLOSE, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
    morphologyEx(src, actual_gradient, MORPH_GRADIENT, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);

    EXPECT_EQ(max_abs_diff_u8(actual_open, expected_open), 0);
    EXPECT_EQ(max_abs_diff_u8(actual_close, expected_close), 0);
    EXPECT_EQ(max_abs_diff_u8(actual_gradient, expected_gradient), 0);
}

TEST(ImgprocMorphGradient_TEST, morphologyEx_tophat_blackhat_match_reference)
{
    Mat src({9, 12}, CV_8UC4);
    fill_u8_pattern(src);

    const Mat expected_erode = morph_reference_u8(src, true, BORDER_REPLICATE, Scalar::all(255.0));
    const Mat expected_dilate = morph_reference_u8(src, false, BORDER_REPLICATE, Scalar::all(0.0));
    const Mat expected_open = morph_reference_u8(expected_erode, false, BORDER_REPLICATE, Scalar::all(0.0));
    const Mat expected_close = morph_reference_u8(expected_dilate, true, BORDER_REPLICATE, Scalar::all(255.0));

    Mat expected_tophat({src.size[0], src.size[1]}, src.type());
    Mat expected_blackhat({src.size[0], src.size[1]}, src.type());
    const size_t count = src.total() * static_cast<size_t>(src.channels());
    for (size_t i = 0; i < count; ++i)
    {
        const int tophat = static_cast<int>(src.data[i]) - static_cast<int>(expected_open.data[i]);
        const int blackhat = static_cast<int>(expected_close.data[i]) - static_cast<int>(src.data[i]);
        expected_tophat.data[i] = static_cast<uchar>(tophat < 0 ? 0 : tophat);
        expected_blackhat.data[i] = static_cast<uchar>(blackhat < 0 ? 0 : blackhat);
    }

    Mat actual_tophat;
    Mat actual_blackhat;
    morphologyEx(src, actual_tophat, MORPH_TOPHAT, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
    morphologyEx(src, actual_blackhat, MORPH_BLACKHAT, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);

    EXPECT_EQ(max_abs_diff_u8(actual_tophat, expected_tophat), 0);
    EXPECT_EQ(max_abs_diff_u8(actual_blackhat, expected_blackhat), 0);
}

TEST(ImgprocMorphGradient_TEST, morphologyEx_hitmiss_signed_kernel_semantics)
{
    Mat src({3, 3}, CV_8UC1);
    src = 0;
    src.at<uchar>(1, 1) = 255;

    Mat kernel({3, 3}, CV_8SC1);
    kernel = 0;
    kernel.at<schar>(0, 1) = -1;
    kernel.at<schar>(1, 0) = -1;
    kernel.at<schar>(1, 1) = 1;
    kernel.at<schar>(1, 2) = -1;
    kernel.at<schar>(2, 1) = -1;

    Mat dst;
    morphologyEx(src, dst, MORPH_HITMISS, kernel);

    Mat expected({3, 3}, CV_8UC1);
    expected = 0;
    expected.at<uchar>(1, 1) = 255;
    EXPECT_EQ(max_abs_diff_u8(dst, expected), 0);

    src.at<uchar>(0, 1) = 255;
    morphologyEx(src, dst, MORPH_HITMISS, kernel);

    expected = 0;
    EXPECT_EQ(max_abs_diff_u8(dst, expected), 0);
}

TEST(ImgprocMorphGradient_TEST, sobel_u8_to_f32_c1_matches_reference)
{
    Mat src({6, 7}, CV_8UC1);
    fill_u8_pattern(src);

    Mat actual;
    Sobel(src, actual, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
    const Mat expected = sobel_reference_u8_to_f32(src, 1, 0, BORDER_REPLICATE);

    EXPECT_LE(max_abs_diff_f32(actual, expected), 1e-6f);
}

TEST(ImgprocMorphGradient_TEST, sobel_u8_to_f32_c4_roi_isolated_matches_reference)
{
    Mat src_full({9, 11}, CV_8UC4);
    fill_u8_pattern(src_full);
    Mat roi = src_full(Range(2, 8), Range(3, 10));

    Mat actual;
    // Keep this contract as ROI-local sampling; non-isolated ROI behavior is
    // covered by ImgprocMorphGradientUpstreamPort_TEST.Imgproc_Sobel_borderTypes.
    Sobel(roi, actual, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE | BORDER_ISOLATED);
    const Mat expected = sobel_reference_u8_to_f32(roi, 1, 0, BORDER_REPLICATE);

    EXPECT_LE(max_abs_diff_f32(actual, expected), 1e-6f);
}

TEST(ImgprocMorphGradient_TEST, invalid_arguments_throw)
{
    Mat src_u8({5, 6}, CV_8UC1);
    fill_u8_pattern(src_u8);
    Mat src_u16({5, 6}, CV_16UC1);
    src_u16 = 7;
    Mat dst;
    Mat empty;

    EXPECT_THROW(Sobel(empty, dst, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE), Exception);
    EXPECT_THROW(Sobel(src_u16, dst, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE), Exception);
    EXPECT_THROW(Sobel(src_u8, dst, CV_8U, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE), Exception);
    EXPECT_THROW(Sobel(src_u8, dst, CV_32F, 2, 0, 3, 1.0, 0.0, BORDER_REPLICATE), Exception);

    EXPECT_THROW(erode(src_u16, dst), Exception);
    EXPECT_THROW(dilate(src_u16, dst), Exception);
    EXPECT_THROW(erode(src_u8, dst, Mat(), Point(-1, -1), 0, BORDER_REPLICATE), Exception);
}

// Ported from OpenCV:
// modules/imgproc/test/test_filter.cpp
// TEST(Imgproc_Morphology, iterated)
TEST(ImgprocMorphGradientUpstreamPort_TEST, Imgproc_Morphology_iterated)
{
    std::uint32_t state = 0xC0FFEEu;
    for (int iter = 0; iter < 20; ++iter)
    {
        state = lcg_next(state);
        const int width = 5 + static_cast<int>(state % 28u);
        state = lcg_next(state);
        const int height = 5 + static_cast<int>(state % 28u);
        state = lcg_next(state);
        const int cn = 1 + static_cast<int>(state % 4u);
        state = lcg_next(state);
        const int iterations = 1 + static_cast<int>(state % 10u);
        state = lcg_next(state);
        const bool do_dilate = (state & 1u) == 0u;

        Mat src({height, width}, CV_MAKETYPE(CV_8U, cn));
        fill_u8_lcg(src, state ^ 0x91u);

        Mat dst0;
        Mat dst1;
        Mat dst2;

        if (do_dilate)
        {
            dilate(src, dst0, Mat(), Point(-1, -1), iterations);
        }
        else
        {
            erode(src, dst0, Mat(), Point(-1, -1), iterations);
        }

        for (int i = 0; i < iterations; ++i)
        {
            if (do_dilate)
            {
                dilate(i == 0 ? src : dst1, dst1, Mat(), Point(-1, -1), 1);
            }
            else
            {
                erode(i == 0 ? src : dst1, dst1, Mat(), Point(-1, -1), 1);
            }
        }

        Mat kern({3, 3}, CV_8UC1);
        kern = 1;
        if (do_dilate)
        {
            dilate(src, dst2, kern, Point(-1, -1), iterations);
        }
        else
        {
            erode(src, dst2, kern, Point(-1, -1), iterations);
        }

        EXPECT_EQ(0, max_abs_diff_u8(dst0, dst1));
        EXPECT_EQ(0, max_abs_diff_u8(dst0, dst2));
    }
}

// Ported from OpenCV:
// modules/imgproc/test/test_filter.cpp
// TEST(Imgproc, morphologyEx_small_input_22893)
TEST(ImgprocMorphGradientUpstreamPort_TEST, Imgproc_morphologyEx_small_input_22893)
{
    Mat img({1, 4}, CV_8UC1);
    img.at<uchar>(0, 0) = 1;
    img.at<uchar>(0, 1) = 2;
    img.at<uchar>(0, 2) = 3;
    img.at<uchar>(0, 3) = 4;

    Mat gold({1, 4}, CV_8UC1);
    gold.at<uchar>(0, 0) = 2;
    gold.at<uchar>(0, 1) = 3;
    gold.at<uchar>(0, 2) = 4;
    gold.at<uchar>(0, 3) = 4;

    Mat kernel({4, 4}, CV_8UC1);
    kernel = 1;

    Mat result;
    morphologyEx(img, result, MORPH_DILATE, kernel);

    ASSERT_EQ(result.type(), gold.type());
    ASSERT_EQ(result.size[0], gold.size[0]);
    ASSERT_EQ(result.size[1], gold.size[1]);
    EXPECT_EQ(max_abs_diff_u8(result, gold), 0);
}

// Ported from OpenCV:
// modules/imgproc/test/test_filter.cpp
// TEST(Imgproc_MorphEx, hitmiss_regression_8957)
TEST(ImgprocMorphGradientUpstreamPort_TEST, Imgproc_MorphEx_hitmiss_regression_8957)
{
    Mat src({3, 3}, CV_8UC1);
    src.at<uchar>(0, 0) = 0;   src.at<uchar>(0, 1) = 255; src.at<uchar>(0, 2) = 0;
    src.at<uchar>(1, 0) = 0;   src.at<uchar>(1, 1) = 0;   src.at<uchar>(1, 2) = 0;
    src.at<uchar>(2, 0) = 0;   src.at<uchar>(2, 1) = 255; src.at<uchar>(2, 2) = 0;

    Mat kernel({3, 3}, CV_8UC1);
    kernel.at<uchar>(0, 0) = 0; kernel.at<uchar>(0, 1) = 1; kernel.at<uchar>(0, 2) = 0;
    kernel.at<uchar>(1, 0) = 0; kernel.at<uchar>(1, 1) = 0; kernel.at<uchar>(1, 2) = 0;
    kernel.at<uchar>(2, 0) = 0; kernel.at<uchar>(2, 1) = 1; kernel.at<uchar>(2, 2) = 0;

    Mat dst;
    morphologyEx(src, dst, MORPH_HITMISS, kernel);

    Mat ref({3, 3}, CV_8UC1);
    ref = 0;
    ref.at<uchar>(1, 1) = 255;
    EXPECT_EQ(max_abs_diff_u8(dst, ref), 0);

    src.at<uchar>(1, 1) = 255;
    ref.at<uchar>(0, 1) = 255;
    ref.at<uchar>(2, 1) = 255;
    morphologyEx(src, dst, MORPH_HITMISS, kernel);
    EXPECT_EQ(max_abs_diff_u8(dst, ref), 0);
}

// Ported from OpenCV:
// modules/imgproc/test/test_filter.cpp
// TEST(Imgproc_MorphEx, hitmiss_zero_kernel)
TEST(ImgprocMorphGradientUpstreamPort_TEST, Imgproc_MorphEx_hitmiss_zero_kernel)
{
    Mat src({3, 3}, CV_8UC1);
    src.at<uchar>(0, 0) = 0;   src.at<uchar>(0, 1) = 255; src.at<uchar>(0, 2) = 0;
    src.at<uchar>(1, 0) = 0;   src.at<uchar>(1, 1) = 0;   src.at<uchar>(1, 2) = 0;
    src.at<uchar>(2, 0) = 0;   src.at<uchar>(2, 1) = 255; src.at<uchar>(2, 2) = 0;

    Mat kernel({3, 3}, CV_8UC1);
    kernel = 0;

    Mat dst;
    morphologyEx(src, dst, MORPH_HITMISS, kernel);
    EXPECT_EQ(max_abs_diff_u8(dst, src), 0);
}

// Ported from OpenCV (implemented-ops coverage):
// modules/imgproc/test/test_filter.cpp
// TEST(Imgproc, filter_empty_src_16857)
TEST(ImgprocMorphGradientUpstreamPort_TEST, Imgproc_filter_empty_src_16857)
{
    Mat src, dst, dst2;

    EXPECT_THROW(blur(src, dst, Size(3, 3)), Exception);
    EXPECT_THROW(boxFilter(src, dst, CV_8U, Size(3, 3)), Exception);
    EXPECT_THROW(GaussianBlur(src, dst, Size(3, 3), 0.0), Exception);
    EXPECT_THROW(Sobel(src, dst, CV_32F, 1, 0, 3), Exception);
    EXPECT_THROW(dilate(src, dst, Mat()), Exception);
    EXPECT_THROW(erode(src, dst, Mat()), Exception);
    EXPECT_THROW(morphologyEx(src, dst, MORPH_OPEN, Mat()), Exception);

    EXPECT_TRUE(src.empty());
    EXPECT_TRUE(dst.empty());
    EXPECT_TRUE(dst2.empty());
}

// Ported from OpenCV:
// modules/imgproc/test/test_filter.cpp
// TEST(Imgproc_Sobel, borderTypes)
TEST(ImgprocMorphGradientUpstreamPort_TEST, Imgproc_Sobel_borderTypes)
{
    const int kernelSize = 3;
    Mat dst;

    Mat src({3, 3}, CV_8UC1);
    src.at<uchar>(0, 0) = 1; src.at<uchar>(0, 1) = 2; src.at<uchar>(0, 2) = 3;
    src.at<uchar>(1, 0) = 4; src.at<uchar>(1, 1) = 5; src.at<uchar>(1, 2) = 6;
    src.at<uchar>(2, 0) = 7; src.at<uchar>(2, 1) = 8; src.at<uchar>(2, 2) = 9;

    Mat src_roi = src(Range(1, 2), Range(1, 2));
    src_roi.setTo(0);

    Sobel(src_roi, dst, CV_32F, 1, 0, kernelSize, 1.0, 0.0, BORDER_REPLICATE);
    EXPECT_FLOAT_EQ(8.0f, dst.at<float>(0, 0));
    Sobel(src_roi, dst, CV_32F, 1, 0, kernelSize, 1.0, 0.0, BORDER_REFLECT);
    EXPECT_FLOAT_EQ(8.0f, dst.at<float>(0, 0));

    Sobel(src_roi, dst, CV_32F, 1, 0, kernelSize, 1.0, 0.0, BORDER_REPLICATE | BORDER_ISOLATED);
    EXPECT_FLOAT_EQ(0.0f, dst.at<float>(0, 0));
    Sobel(src_roi, dst, CV_32F, 1, 0, kernelSize, 1.0, 0.0, BORDER_REFLECT | BORDER_ISOLATED);
    EXPECT_FLOAT_EQ(0.0f, dst.at<float>(0, 0));

    src = Mat({5, 5}, CV_8UC1);
    src = 5;
    src_roi = src(Range(1, 4), Range(1, 4));
    src_roi.setTo(0);

    Mat expected({3, 3}, CV_32FC1);
    expected.at<float>(0, 0) = -15.0f; expected.at<float>(0, 1) = 0.0f; expected.at<float>(0, 2) = 15.0f;
    expected.at<float>(1, 0) = -20.0f; expected.at<float>(1, 1) = 0.0f; expected.at<float>(1, 2) = 20.0f;
    expected.at<float>(2, 0) = -15.0f; expected.at<float>(2, 1) = 0.0f; expected.at<float>(2, 2) = 15.0f;

    Sobel(src_roi, dst, CV_32F, 1, 0, kernelSize, 1.0, 0.0, BORDER_REPLICATE);
    EXPECT_LE(max_abs_diff_f32(expected, dst), 1e-6f);
    Sobel(src_roi, dst, CV_32F, 1, 0, kernelSize, 1.0, 0.0, BORDER_REFLECT);
    EXPECT_LE(max_abs_diff_f32(expected, dst), 1e-6f);

    Mat expected_zero({3, 3}, CV_32FC1);
    expected_zero = 0.0f;
    Sobel(src_roi, dst, CV_32F, 1, 0, kernelSize, 1.0, 0.0, BORDER_REPLICATE | BORDER_ISOLATED);
    EXPECT_LE(max_abs_diff_f32(expected_zero, dst), 1e-6f);
    Sobel(src_roi, dst, CV_32F, 1, 0, kernelSize, 1.0, 0.0, BORDER_REFLECT | BORDER_ISOLATED);
    EXPECT_LE(max_abs_diff_f32(expected_zero, dst), 1e-6f);
}

// Ported from OpenCV:
// modules/imgproc/test/test_filter.cpp
// TEST(Imgproc_Sobel, s16_regression_13506)
TEST(ImgprocMorphGradientUpstreamPort_TEST, Imgproc_Sobel_s16_regression_13506)
{
    static const short src_values[8 * 16] = {
        127, 138, 130, 102, 118, 97, 76, 84, 124, 90, 146, 63, 130, 87, 212, 85,
        164, 3, 51, 124, 151, 89, 154, 117, 36, 88, 116, 117, 180, 112, 147, 124,
        63, 50, 115, 103, 83, 148, 106, 79, 213, 106, 135, 53, 79, 106, 122, 112,
        218, 107, 81, 126, 78, 138, 85, 142, 151, 108, 104, 158, 155, 81, 112, 178,
        184, 96, 187, 148, 150, 112, 138, 162, 222, 146, 128, 49, 124, 46, 165, 104,
        119, 164, 77, 144, 186, 98, 106, 148, 155, 157, 160, 151, 156, 149, 43, 122,
        106, 155, 120, 132, 159, 115, 126, 188, 44, 79, 164, 201, 153, 97, 139, 133,
        133, 98, 111, 165, 66, 106, 131, 85, 176, 156, 67, 108, 142, 91, 74, 137,
    };

    static const short ref_values[8 * 16] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        -1020, -796, -489, -469, -247, 317, 760, 1429, 1983, 1384, 254, -459, -899, -1197, -1172, -1058,
        2552, 2340, 1617, 591, 9, 96, 722, 1985, 2746, 1916, 676, 9, -635, -1115, -779, -380,
        3546, 3349, 2838, 2206, 1388, 669, 938, 1880, 2252, 1785, 1083, 606, 180, -298, -464, -418,
        816, 966, 1255, 1652, 1619, 924, 535, 288, 5, 601, 1581, 1870, 1520, 625, -627, -1260,
        -782, -610, -395, -267, -122, -42, -317, -1378, -2293, -1451, 596, 1870, 1679, 763, -69, -394,
        -882, -681, -463, -818, -1167, -732, -463, -1042, -1604, -1592, -1047, -334, -104, -117, 229, 512,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    Mat src({8, 16}, CV_16SC1);
    Mat ref({8, 16}, CV_16SC1);
    short* src_ptr = reinterpret_cast<short*>(src.data);
    short* ref_ptr = reinterpret_cast<short*>(ref.data);
    for (size_t i = 0; i < 8u * 16u; ++i)
    {
        src_ptr[i] = src_values[i];
        ref_ptr[i] = ref_values[i];
    }

    Mat dst;
    Sobel(src, dst, CV_16S, 0, 1, 5);

    ASSERT_EQ(dst.type(), ref.type());
    ASSERT_EQ(dst.size[0], ref.size[0]);
    ASSERT_EQ(dst.size[1], ref.size[1]);
    EXPECT_EQ(max_abs_diff_s16(dst, ref), 0);
}
