#include "cvh.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

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
        max_diff = std::max(max_diff, diff);
    }
    return max_diff;
}

float max_abs_diff_f32(const Mat& a, const Mat& b)
{
    if (a.type() != b.type() || a.size[0] != b.size[0] || a.size[1] != b.size[1])
    {
        return 1e9f;
    }
    CV_Assert(a.depth() == CV_32F);
    const size_t count = a.total() * static_cast<size_t>(a.channels());
    const float* pa = reinterpret_cast<const float*>(a.data);
    const float* pb = reinterpret_cast<const float*>(b.data);
    float max_diff = 0.0f;
    for (size_t i = 0; i < count; ++i)
    {
        const float diff = std::fabs(pa[i] - pb[i]);
        max_diff = std::max(max_diff, diff);
    }
    return max_diff;
}

double l2_norm_diff_f32(const Mat& a, const Mat& b)
{
    if (a.type() != b.type() || a.size[0] != b.size[0] || a.size[1] != b.size[1])
    {
        return 1e12;
    }
    CV_Assert(a.depth() == CV_32F);
    const size_t count = a.total() * static_cast<size_t>(a.channels());
    const float* pa = reinterpret_cast<const float*>(a.data);
    const float* pb = reinterpret_cast<const float*>(b.data);

    double sum_sq = 0.0;
    for (size_t i = 0; i < count; ++i)
    {
        const double diff = static_cast<double>(pa[i]) - static_cast<double>(pb[i]);
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq);
}

int normalize_border_type(int borderType)
{
    return borderType & (~BORDER_ISOLATED);
}

int border_interpolate(int p, int len, int borderType)
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

Mat box_filter_reference_u8(const Mat& src, Size ksize, Point anchor, bool normalize, int borderType)
{
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.dims == 2);
    CV_Assert(ksize.width > 0 && ksize.height > 0);

    const int rows = src.size[0];
    const int cols = src.size[1];
    const int channels = src.channels();
    const size_t src_step = src.step(0);

    const int ax = anchor.x >= 0 ? anchor.x : (ksize.width / 2);
    const int ay = anchor.y >= 0 ? anchor.y : (ksize.height / 2);
    CV_Assert(ax >= 0 && ax < ksize.width);
    CV_Assert(ay >= 0 && ay < ksize.height);

    const int border = normalize_border_type(borderType);
    CV_Assert(border == BORDER_CONSTANT ||
              border == BORDER_REPLICATE ||
              border == BORDER_REFLECT ||
              border == BORDER_REFLECT_101);

    Mat dst({rows, cols}, src.type());
    const size_t dst_step = dst.step(0);

    const int kernel_area = ksize.width * ksize.height;
    const float inv_kernel_area = 1.0f / static_cast<float>(kernel_area);

    for (int y = 0; y < rows; ++y)
    {
        uchar* out_row = dst.data + static_cast<size_t>(y) * dst_step;
        for (int x = 0; x < cols; ++x)
        {
            uchar* out_px = out_row + static_cast<size_t>(x) * channels;
            for (int c = 0; c < channels; ++c)
            {
                int64_t sum = 0;
                for (int ky = 0; ky < ksize.height; ++ky)
                {
                    const int sy = y + ky - ay;
                    const int src_y = border_interpolate(sy, rows, border);
                    if (src_y < 0)
                    {
                        continue;
                    }

                    const uchar* src_row = src.data + static_cast<size_t>(src_y) * src_step;
                    for (int kx = 0; kx < ksize.width; ++kx)
                    {
                        const int sx = x + kx - ax;
                        const int src_x = border_interpolate(sx, cols, border);
                        if (src_x < 0)
                        {
                            continue;
                        }
                        sum += static_cast<int64_t>(src_row[static_cast<size_t>(src_x) * channels + c]);
                    }
                }

                if (normalize)
                {
                    out_px[c] = saturate_cast<uchar>(static_cast<float>(sum) * inv_kernel_area);
                }
                else
                {
                    out_px[c] = saturate_cast<uchar>(sum);
                }
            }
        }
    }

    return dst;
}

Mat box_filter_reference_f32(const Mat& src, Size ksize, Point anchor, bool normalize, int borderType)
{
    CV_Assert(src.depth() == CV_32F);
    CV_Assert(src.dims == 2);
    CV_Assert(ksize.width > 0 && ksize.height > 0);

    const int rows = src.size[0];
    const int cols = src.size[1];
    const int channels = src.channels();
    const size_t src_step = src.step(0);

    const int ax = anchor.x >= 0 ? anchor.x : (ksize.width / 2);
    const int ay = anchor.y >= 0 ? anchor.y : (ksize.height / 2);
    CV_Assert(ax >= 0 && ax < ksize.width);
    CV_Assert(ay >= 0 && ay < ksize.height);

    const int border = normalize_border_type(borderType);
    CV_Assert(border == BORDER_CONSTANT ||
              border == BORDER_REPLICATE ||
              border == BORDER_REFLECT ||
              border == BORDER_REFLECT_101);

    Mat dst({rows, cols}, src.type());
    const size_t dst_step = dst.step(0);

    const int kernel_area = ksize.width * ksize.height;
    const double inv_kernel_area = 1.0 / static_cast<double>(kernel_area);

    for (int y = 0; y < rows; ++y)
    {
        float* out_row = reinterpret_cast<float*>(dst.data + static_cast<size_t>(y) * dst_step);
        for (int x = 0; x < cols; ++x)
        {
            float* out_px = out_row + static_cast<size_t>(x) * channels;
            for (int c = 0; c < channels; ++c)
            {
                double sum = 0.0;
                for (int ky = 0; ky < ksize.height; ++ky)
                {
                    const int sy = y + ky - ay;
                    const int src_y = border_interpolate(sy, rows, border);
                    if (src_y < 0)
                    {
                        continue;
                    }

                    const float* src_row = reinterpret_cast<const float*>(src.data + static_cast<size_t>(src_y) * src_step);
                    for (int kx = 0; kx < ksize.width; ++kx)
                    {
                        const int sx = x + kx - ax;
                        const int src_x = border_interpolate(sx, cols, border);
                        if (src_x < 0)
                        {
                            continue;
                        }
                        sum += static_cast<double>(src_row[static_cast<size_t>(src_x) * channels + c]);
                    }
                }

                out_px[c] = normalize ? static_cast<float>(sum * inv_kernel_area) : static_cast<float>(sum);
            }
        }
    }

    return dst;
}

double default_sigma_for_ksize(int ksize)
{
    return ((static_cast<double>(ksize) - 1.0) * 0.5 - 1.0) * 0.3 + 0.8;
}

std::vector<float> gaussian_kernel_1d(int ksize, double sigma)
{
    CV_Assert(ksize > 0 && (ksize & 1));
    CV_Assert(sigma > 0.0);

    const int radius = ksize / 2;
    const double scale = -0.5 / (sigma * sigma);

    std::vector<float> kernel(static_cast<size_t>(ksize), 0.0f);
    double sum = 0.0;
    for (int i = 0; i < ksize; ++i)
    {
        const double x = static_cast<double>(i - radius);
        const double w = std::exp(x * x * scale);
        kernel[static_cast<size_t>(i)] = static_cast<float>(w);
        sum += w;
    }

    const float inv_sum = static_cast<float>(1.0 / sum);
    for (float& w : kernel)
    {
        w *= inv_sum;
    }

    return kernel;
}

Mat gaussian_blur_reference_u8(const Mat& src, Size ksize, double sigmaX, double sigmaY, int borderType)
{
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.dims == 2);
    CV_Assert(ksize.width > 0 && ksize.height > 0);
    CV_Assert((ksize.width & 1) && (ksize.height & 1));

    if (sigmaX <= 0.0)
    {
        sigmaX = default_sigma_for_ksize(ksize.width);
    }
    if (sigmaY <= 0.0)
    {
        sigmaY = sigmaX;
    }

    const int border = normalize_border_type(borderType);
    CV_Assert(border == BORDER_CONSTANT ||
              border == BORDER_REPLICATE ||
              border == BORDER_REFLECT ||
              border == BORDER_REFLECT_101);

    const int rows = src.size[0];
    const int cols = src.size[1];
    const int channels = src.channels();
    const size_t src_step = src.step(0);

    const int rx = ksize.width / 2;
    const int ry = ksize.height / 2;
    const std::vector<float> kx = gaussian_kernel_1d(ksize.width, sigmaX);
    const std::vector<float> ky = gaussian_kernel_1d(ksize.height, sigmaY);

    std::vector<float> tmp(static_cast<size_t>(rows) * cols * channels, 0.0f);

    for (int y = 0; y < rows; ++y)
    {
        const uchar* src_row = src.data + static_cast<size_t>(y) * src_step;
        for (int x = 0; x < cols; ++x)
        {
            const size_t tmp_base = (static_cast<size_t>(y) * cols + x) * channels;
            for (int c = 0; c < channels; ++c)
            {
                double acc = 0.0;
                for (int i = 0; i < ksize.width; ++i)
                {
                    const int sx = x + i - rx;
                    const int src_x = border_interpolate(sx, cols, border);
                    if (src_x < 0)
                    {
                        continue;
                    }
                    acc += static_cast<double>(kx[static_cast<size_t>(i)]) *
                           static_cast<double>(src_row[static_cast<size_t>(src_x) * channels + c]);
                }
                tmp[tmp_base + c] = static_cast<float>(acc);
            }
        }
    }

    Mat dst({rows, cols}, src.type());
    const size_t dst_step = dst.step(0);
    for (int y = 0; y < rows; ++y)
    {
        uchar* dst_row = dst.data + static_cast<size_t>(y) * dst_step;
        for (int x = 0; x < cols; ++x)
        {
            uchar* out_px = dst_row + static_cast<size_t>(x) * channels;
            for (int c = 0; c < channels; ++c)
            {
                double acc = 0.0;
                for (int i = 0; i < ksize.height; ++i)
                {
                    const int sy = y + i - ry;
                    const int src_y = border_interpolate(sy, rows, border);
                    if (src_y < 0)
                    {
                        continue;
                    }
                    const size_t tmp_idx = (static_cast<size_t>(src_y) * cols + x) * channels + c;
                    acc += static_cast<double>(ky[static_cast<size_t>(i)]) * static_cast<double>(tmp[tmp_idx]);
                }
                out_px[c] = saturate_cast<uchar>(acc);
            }
        }
    }

    return dst;
}

Mat gaussian_blur_reference_f32(const Mat& src, Size ksize, double sigmaX, double sigmaY, int borderType)
{
    CV_Assert(src.depth() == CV_32F);
    CV_Assert(src.dims == 2);
    CV_Assert(ksize.width > 0 && ksize.height > 0);
    CV_Assert((ksize.width & 1) && (ksize.height & 1));

    if (sigmaX <= 0.0)
    {
        sigmaX = default_sigma_for_ksize(ksize.width);
    }
    if (sigmaY <= 0.0)
    {
        sigmaY = sigmaX;
    }

    const int border = normalize_border_type(borderType);
    CV_Assert(border == BORDER_CONSTANT ||
              border == BORDER_REPLICATE ||
              border == BORDER_REFLECT ||
              border == BORDER_REFLECT_101);

    const int rows = src.size[0];
    const int cols = src.size[1];
    const int channels = src.channels();
    const size_t src_step = src.step(0);

    const int rx = ksize.width / 2;
    const int ry = ksize.height / 2;
    const std::vector<float> kx = gaussian_kernel_1d(ksize.width, sigmaX);
    const std::vector<float> ky = gaussian_kernel_1d(ksize.height, sigmaY);

    std::vector<float> tmp(static_cast<size_t>(rows) * cols * channels, 0.0f);

    for (int y = 0; y < rows; ++y)
    {
        const float* src_row = reinterpret_cast<const float*>(src.data + static_cast<size_t>(y) * src_step);
        for (int x = 0; x < cols; ++x)
        {
            const size_t tmp_base = (static_cast<size_t>(y) * cols + x) * channels;
            for (int c = 0; c < channels; ++c)
            {
                double acc = 0.0;
                for (int i = 0; i < ksize.width; ++i)
                {
                    const int sx = x + i - rx;
                    const int src_x = border_interpolate(sx, cols, border);
                    if (src_x < 0)
                    {
                        continue;
                    }
                    acc += static_cast<double>(kx[static_cast<size_t>(i)]) *
                           static_cast<double>(src_row[static_cast<size_t>(src_x) * channels + c]);
                }
                tmp[tmp_base + c] = static_cast<float>(acc);
            }
        }
    }

    Mat dst({rows, cols}, src.type());
    const size_t dst_step = dst.step(0);
    for (int y = 0; y < rows; ++y)
    {
        float* dst_row = reinterpret_cast<float*>(dst.data + static_cast<size_t>(y) * dst_step);
        for (int x = 0; x < cols; ++x)
        {
            float* out_px = dst_row + static_cast<size_t>(x) * channels;
            for (int c = 0; c < channels; ++c)
            {
                double acc = 0.0;
                for (int i = 0; i < ksize.height; ++i)
                {
                    const int sy = y + i - ry;
                    const int src_y = border_interpolate(sy, rows, border);
                    if (src_y < 0)
                    {
                        continue;
                    }
                    const size_t tmp_idx = (static_cast<size_t>(src_y) * cols + x) * channels + c;
                    acc += static_cast<double>(ky[static_cast<size_t>(i)]) * static_cast<double>(tmp[tmp_idx]);
                }
                out_px[c] = static_cast<float>(acc);
            }
        }
    }

    return dst;
}

}  // namespace

TEST(OpenCVUpstreamFilterPort_TEST, Imgproc_Blur_borderTypes)
{
    // Upstream reference:
    // modules/imgproc/test/test_filter.cpp :: TEST(Imgproc_Blur, borderTypes)
    Mat parent({9, 11}, CV_8UC3);
    for (int y = 0; y < parent.size[0]; ++y)
    {
        for (int x = 0; x < parent.size[1]; ++x)
        {
            for (int c = 0; c < 3; ++c)
            {
                parent.at<uchar>(y, x, c) = static_cast<uchar>((y * 19 + x * 13 + c * 23) % 256);
            }
        }
    }

    Mat src_roi = parent(Range(2, 8), Range(3, 10));
    ASSERT_FALSE(src_roi.isContinuous());

    Mat dst;
    blur(src_roi, dst, Size(3, 3), Point(-1, -1), BORDER_REPLICATE);

    Mat dst_isolated;
    blur(src_roi, dst_isolated, Size(3, 3), Point(-1, -1), BORDER_REPLICATE | BORDER_ISOLATED);

    EXPECT_EQ(max_abs_diff_u8(dst, dst_isolated), 0);

    const Mat ref = box_filter_reference_u8(src_roi, Size(3, 3), Point(-1, -1), true, BORDER_REPLICATE);
    EXPECT_EQ(max_abs_diff_u8(dst, ref), 0);
}

TEST(OpenCVUpstreamFilterPort_TEST, Imgproc_GaussianBlur_borderTypes)
{
    // Upstream reference:
    // modules/imgproc/test/test_filter.cpp :: TEST(Imgproc_GaussianBlur, borderTypes)
    Mat parent({10, 12}, CV_8UC1);
    for (int y = 0; y < parent.size[0]; ++y)
    {
        for (int x = 0; x < parent.size[1]; ++x)
        {
            parent.at<uchar>(y, x) = static_cast<uchar>((y * 37 + x * 11) % 256);
        }
    }

    Mat src_roi = parent(Range(2, 9), Range(1, 10));
    ASSERT_FALSE(src_roi.isContinuous());

    Mat dst;
    GaussianBlur(src_roi, dst, Size(5, 5), 0.0, 0.0, BORDER_REPLICATE);

    Mat dst_isolated;
    GaussianBlur(src_roi, dst_isolated, Size(5, 5), 0.0, 0.0, BORDER_REPLICATE | BORDER_ISOLATED);
    EXPECT_EQ(max_abs_diff_u8(dst, dst_isolated), 0);

    Mat dst_default;
    GaussianBlur(src_roi, dst_default, Size(5, 5), 0.0, 0.0, BORDER_DEFAULT);
    const Mat ref_default = gaussian_blur_reference_u8(src_roi, Size(5, 5), 0.0, 0.0, BORDER_DEFAULT);
    EXPECT_LE(max_abs_diff_u8(dst_default, ref_default), 1);
}

TEST(OpenCVUpstreamFilterPort_TEST, GaussianBlur_Bitexact_regression_15015)
{
    // Upstream reference:
    // modules/imgproc/test/test_smooth_bitexact.cpp :: TEST(GaussianBlur_Bitexact, regression_15015)
    Mat src({100, 100}, CV_8UC3);
    src = Scalar(255.0, 255.0, 255.0, 0.0);

    Mat dst;
    GaussianBlur(src, dst, Size(5, 5), 0.0);

    ASSERT_EQ(dst.type(), src.type());
    ASSERT_EQ(dst.size[0], src.size[0]);
    ASSERT_EQ(dst.size[1], src.size[1]);
    EXPECT_EQ(max_abs_diff_u8(dst, src), 0);
}

TEST(OpenCVUpstreamFilterPort_TEST, Imgproc_GaussianBlur_regression_11303)
{
    // Upstream reference:
    // modules/imgproc/test/test_filter.cpp :: TEST(Imgproc_GaussianBlur, regression_11303)
    const int width = 2115;
    const int height = 211;
    const double sigma = 8.64421;

    Mat src({height, width}, CV_32FC1);
    src = 1.0f;

    Mat dst;
    GaussianBlur(src, dst, Size(), sigma, sigma);

    ASSERT_EQ(dst.type(), src.type());
    ASSERT_EQ(dst.size[0], src.size[0]);
    ASSERT_EQ(dst.size[1], src.size[1]);
    EXPECT_LE(l2_norm_diff_f32(src, dst), 1e-3);
}

TEST(ImgprocFilterFastpath_TEST, boxfilter_non_contiguous_roi_custom_anchor_and_normalize_off_matches_reference)
{
    Mat parent({9, 13}, CV_8UC4);
    for (int y = 0; y < parent.size[0]; ++y)
    {
        for (int x = 0; x < parent.size[1]; ++x)
        {
            parent.at<uchar>(y, x, 0) = static_cast<uchar>((y * 17 + x * 3 + 11) % 256);
            parent.at<uchar>(y, x, 1) = static_cast<uchar>((y * 5 + x * 29 + 7) % 256);
            parent.at<uchar>(y, x, 2) = static_cast<uchar>((y * 13 + x * 19 + 3) % 256);
            parent.at<uchar>(y, x, 3) = static_cast<uchar>((y * 23 + x * 11 + 1) % 256);
        }
    }

    Mat roi = parent(Range(1, 8), Range(2, 12));
    ASSERT_FALSE(roi.isContinuous());

    const Size ksize(5, 3);
    const Point anchor(1, 0);

    Mat actual;
    boxFilter(roi, actual, -1, ksize, anchor, false, BORDER_REFLECT_101);

    const Mat expected = box_filter_reference_u8(roi, ksize, anchor, false, BORDER_REFLECT_101);
    EXPECT_EQ(max_abs_diff_u8(actual, expected), 0);
}

TEST(ImgprocFilterFastpath_TEST, boxfilter_inplace_matches_reference_with_constant_border)
{
    Mat src({17, 19}, CV_8UC3);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            src.at<uchar>(y, x, 0) = static_cast<uchar>((y * 31 + x * 7 + 3) % 256);
            src.at<uchar>(y, x, 1) = static_cast<uchar>((y * 11 + x * 5 + 9) % 256);
            src.at<uchar>(y, x, 2) = static_cast<uchar>((y * 13 + x * 17 + 15) % 256);
        }
    }

    const Size ksize(7, 5);
    const Point anchor(-1, -1);
    const Mat expected = box_filter_reference_u8(src, ksize, anchor, true, BORDER_CONSTANT);

    Mat in_place = src.clone();
    boxFilter(in_place, in_place, -1, ksize, anchor, true, BORDER_CONSTANT);

    EXPECT_EQ(max_abs_diff_u8(in_place, expected), 0);
}

TEST(ImgprocFilterFastpath_TEST, gaussian_blur_roi_and_inplace_match_reference)
{
    Mat parent({11, 14}, CV_8UC4);
    for (int y = 0; y < parent.size[0]; ++y)
    {
        for (int x = 0; x < parent.size[1]; ++x)
        {
            parent.at<uchar>(y, x, 0) = static_cast<uchar>((y * 7 + x * 37 + 5) % 256);
            parent.at<uchar>(y, x, 1) = static_cast<uchar>((y * 19 + x * 13 + 17) % 256);
            parent.at<uchar>(y, x, 2) = static_cast<uchar>((y * 29 + x * 3 + 23) % 256);
            parent.at<uchar>(y, x, 3) = static_cast<uchar>((y * 11 + x * 41 + 31) % 256);
        }
    }

    Mat roi = parent(Range(2, 10), Range(1, 13));
    ASSERT_FALSE(roi.isContinuous());

    const Size roi_ksize(7, 5);
    const Mat roi_expected = gaussian_blur_reference_u8(roi, roi_ksize, 1.4, 1.2, BORDER_REFLECT);
    Mat roi_actual;
    GaussianBlur(roi, roi_actual, roi_ksize, 1.4, 1.2, BORDER_REFLECT);
    EXPECT_LE(max_abs_diff_u8(roi_actual, roi_expected), 1);

    Mat in_place_src({13, 15}, CV_8UC1);
    for (int y = 0; y < in_place_src.size[0]; ++y)
    {
        for (int x = 0; x < in_place_src.size[1]; ++x)
        {
            in_place_src.at<uchar>(y, x) = static_cast<uchar>((y * 43 + x * 17 + 29) % 256);
        }
    }

    const Size in_place_ksize(5, 5);
    const Mat in_place_expected = gaussian_blur_reference_u8(in_place_src, in_place_ksize, 0.0, 0.0, BORDER_CONSTANT);
    Mat in_place_actual = in_place_src.clone();
    GaussianBlur(in_place_actual, in_place_actual, in_place_ksize, 0.0, 0.0, BORDER_CONSTANT);
    EXPECT_LE(max_abs_diff_u8(in_place_actual, in_place_expected), 1);
}

TEST(ImgprocFilterFastpath_TEST, supports_cv32f_boxfilter_roi_and_inplace)
{
    Mat base({9, 12}, CV_32FC4);
    for (int y = 0; y < base.size[0]; ++y)
    {
        for (int x = 0; x < base.size[1]; ++x)
        {
            for (int c = 0; c < 4; ++c)
            {
                base.at<float>(y, x, c) = static_cast<float>(y * 0.8 - x * 0.35 + c * 1.2);
            }
        }
    }
    Mat roi = base(Range(1, 8), Range(2, 11));
    ASSERT_FALSE(roi.isContinuous());

    Mat actual;
    boxFilter(roi, actual, -1, Size(5, 3), Point(1, 0), true, BORDER_REFLECT_101);
    const Mat expected = box_filter_reference_f32(roi, Size(5, 3), Point(1, 0), true, BORDER_REFLECT_101);
    EXPECT_LE(max_abs_diff_f32(actual, expected), 1e-5f);

    Mat in_place = roi.clone();
    const Mat in_place_ref = box_filter_reference_f32(in_place, Size(3, 3), Point(-1, -1), false, BORDER_CONSTANT);
    boxFilter(in_place, in_place, -1, Size(3, 3), Point(-1, -1), false, BORDER_CONSTANT);
    EXPECT_LE(max_abs_diff_f32(in_place, in_place_ref), 1e-5f);
}

TEST(ImgprocFilterFastpath_TEST, supports_cv32f_gaussian_roi_and_inplace)
{
    Mat base({10, 13}, CV_32FC3);
    for (int y = 0; y < base.size[0]; ++y)
    {
        for (int x = 0; x < base.size[1]; ++x)
        {
            for (int c = 0; c < 3; ++c)
            {
                base.at<float>(y, x, c) = static_cast<float>(y * 1.1 + x * 0.4 - c * 0.9);
            }
        }
    }
    Mat roi = base(Range(2, 9), Range(1, 12));
    ASSERT_FALSE(roi.isContinuous());

    Mat actual;
    GaussianBlur(roi, actual, Size(5, 7), 1.2, 1.6, BORDER_REPLICATE);
    const Mat expected = gaussian_blur_reference_f32(roi, Size(5, 7), 1.2, 1.6, BORDER_REPLICATE);
    EXPECT_LE(max_abs_diff_f32(actual, expected), 2e-4f);

    Mat in_place = roi.clone();
    const Mat in_place_ref = gaussian_blur_reference_f32(in_place, Size(3, 3), 0.0, 0.0, BORDER_CONSTANT);
    GaussianBlur(in_place, in_place, Size(3, 3), 0.0, 0.0, BORDER_CONSTANT);
    EXPECT_LE(max_abs_diff_f32(in_place, in_place_ref), 2e-4f);
}
