#include "cvh.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace cvh;

namespace
{

double l1_u8(const Mat& src)
{
    const size_t count = src.total() * static_cast<size_t>(src.channels());
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i)
    {
        sum += static_cast<double>(src.data[i]);
    }
    return sum;
}

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
        const float diff = std::abs(pa[i] - pb[i]);
        if (diff > max_diff)
        {
            max_diff = diff;
        }
    }
    return max_diff;
}

Mat mat_u8(int rows, int cols, const std::vector<int>& values)
{
    CV_Assert(static_cast<int>(values.size()) == rows * cols);
    Mat out({rows, cols}, CV_8UC1);
    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            out.at<uchar>(y, x) = static_cast<uchar>(values[static_cast<size_t>(y * cols + x)]);
        }
    }
    return out;
}

Mat transpose_u8(const Mat& src)
{
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 1);
    CV_Assert(src.dims == 2);

    Mat out({src.size[1], src.size[0]}, src.type());
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            out.at<uchar>(x, y) = src.at<uchar>(y, x);
        }
    }
    return out;
}

Mat resize_reference_linear_u8(const Mat& src, Size dsize, double fx, double fy)
{
    CV_Assert(!src.empty());
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);

    const int src_rows = src.size[0];
    const int src_cols = src.size[1];
    const int dst_cols = dsize.width > 0 ? dsize.width : std::max(1, static_cast<int>(std::lround(src_cols * fx)));
    const int dst_rows = dsize.height > 0 ? dsize.height : std::max(1, static_cast<int>(std::lround(src_rows * fy)));
    CV_Assert(dst_cols > 0 && dst_rows > 0);

    Mat out({dst_rows, dst_cols}, src.type());
    const int cn = src.channels();

    const float scale_x = static_cast<float>(src_cols) / static_cast<float>(dst_cols);
    const float scale_y = static_cast<float>(src_rows) / static_cast<float>(dst_rows);

    for (int y = 0; y < dst_rows; ++y)
    {
        const float fy_src = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const int y0 = std::clamp(static_cast<int>(std::floor(fy_src)), 0, src_rows - 1);
        const int y1 = std::min(y0 + 1, src_rows - 1);
        const float wy = fy_src - static_cast<float>(y0);

        for (int x = 0; x < dst_cols; ++x)
        {
            const float fx_src = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const int x0 = std::clamp(static_cast<int>(std::floor(fx_src)), 0, src_cols - 1);
            const int x1 = std::min(x0 + 1, src_cols - 1);
            const float wx = fx_src - static_cast<float>(x0);

            for (int c = 0; c < cn; ++c)
            {
                const float p00 = static_cast<float>(src.at<uchar>(y0, x0, c));
                const float p01 = static_cast<float>(src.at<uchar>(y0, x1, c));
                const float p10 = static_cast<float>(src.at<uchar>(y1, x0, c));
                const float p11 = static_cast<float>(src.at<uchar>(y1, x1, c));

                const float top = p00 + (p01 - p00) * wx;
                const float bot = p10 + (p11 - p10) * wx;
                out.at<uchar>(y, x, c) = saturate_cast<uchar>(top + (bot - top) * wy);
            }
        }
    }

    return out;
}

Mat resize_reference_nearest_u8(const Mat& src, Size dsize, double fx, double fy)
{
    CV_Assert(!src.empty());
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);

    const int src_rows = src.size[0];
    const int src_cols = src.size[1];
    const int dst_cols = dsize.width > 0 ? dsize.width : std::max(1, static_cast<int>(std::lround(src_cols * fx)));
    const int dst_rows = dsize.height > 0 ? dsize.height : std::max(1, static_cast<int>(std::lround(src_rows * fy)));
    CV_Assert(dst_cols > 0 && dst_rows > 0);

    Mat out({dst_rows, dst_cols}, src.type());
    const int cn = src.channels();
    for (int y = 0; y < dst_rows; ++y)
    {
        const int sy = std::min(src_rows - 1, (y * src_rows) / dst_rows);
        for (int x = 0; x < dst_cols; ++x)
        {
            const int sx = std::min(src_cols - 1, (x * src_cols) / dst_cols);
            for (int c = 0; c < cn; ++c)
            {
                out.at<uchar>(y, x, c) = src.at<uchar>(sy, sx, c);
            }
        }
    }
    return out;
}

Mat resize_reference_nearest_exact_u8(const Mat& src, Size dsize, double fx, double fy)
{
    CV_Assert(!src.empty());
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);

    const int src_rows = src.size[0];
    const int src_cols = src.size[1];
    const int dst_cols = dsize.width > 0 ? dsize.width : std::max(1, static_cast<int>(std::lround(src_cols * fx)));
    const int dst_rows = dsize.height > 0 ? dsize.height : std::max(1, static_cast<int>(std::lround(src_rows * fy)));
    CV_Assert(dst_cols > 0 && dst_rows > 0);

    Mat out({dst_rows, dst_cols}, src.type());
    const int cn = src.channels();
    const int64_t ifx = ((static_cast<int64_t>(src_cols) << 16) + dst_cols / 2) / dst_cols;
    const int64_t ifx0 = ifx / 2 - (src_cols % 2);
    const int64_t ify = ((static_cast<int64_t>(src_rows) << 16) + dst_rows / 2) / dst_rows;
    const int64_t ify0 = ify / 2 - (src_rows % 2);

    for (int y = 0; y < dst_rows; ++y)
    {
        const int sy = std::clamp(static_cast<int>((ify * y + ify0) >> 16), 0, src_rows - 1);
        for (int x = 0; x < dst_cols; ++x)
        {
            const int sx = std::clamp(static_cast<int>((ifx * x + ifx0) >> 16), 0, src_cols - 1);
            for (int c = 0; c < cn; ++c)
            {
                out.at<uchar>(y, x, c) = src.at<uchar>(sy, sx, c);
            }
        }
    }
    return out;
}

Mat resize_reference_linear_f32(const Mat& src, Size dsize, double fx, double fy)
{
    CV_Assert(!src.empty());
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_32F);

    const int src_rows = src.size[0];
    const int src_cols = src.size[1];
    const int dst_cols = dsize.width > 0 ? dsize.width : std::max(1, static_cast<int>(std::lround(src_cols * fx)));
    const int dst_rows = dsize.height > 0 ? dsize.height : std::max(1, static_cast<int>(std::lround(src_rows * fy)));
    CV_Assert(dst_cols > 0 && dst_rows > 0);

    Mat out({dst_rows, dst_cols}, src.type());
    const int cn = src.channels();

    const float scale_x = static_cast<float>(src_cols) / static_cast<float>(dst_cols);
    const float scale_y = static_cast<float>(src_rows) / static_cast<float>(dst_rows);

    for (int y = 0; y < dst_rows; ++y)
    {
        const float fy_src = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const int y0 = std::clamp(static_cast<int>(std::floor(fy_src)), 0, src_rows - 1);
        const int y1 = std::min(y0 + 1, src_rows - 1);
        const float wy = fy_src - static_cast<float>(y0);

        for (int x = 0; x < dst_cols; ++x)
        {
            const float fx_src = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const int x0 = std::clamp(static_cast<int>(std::floor(fx_src)), 0, src_cols - 1);
            const int x1 = std::min(x0 + 1, src_cols - 1);
            const float wx = fx_src - static_cast<float>(x0);

            for (int c = 0; c < cn; ++c)
            {
                const float p00 = src.at<float>(y0, x0, c);
                const float p01 = src.at<float>(y0, x1, c);
                const float p10 = src.at<float>(y1, x0, c);
                const float p11 = src.at<float>(y1, x1, c);

                const float top = p00 + (p01 - p00) * wx;
                const float bot = p10 + (p11 - p10) * wx;
                out.at<float>(y, x, c) = top + (bot - top) * wy;
            }
        }
    }

    return out;
}

Mat resize_reference_nearest_f32(const Mat& src, Size dsize, double fx, double fy)
{
    CV_Assert(!src.empty());
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_32F);

    const int src_rows = src.size[0];
    const int src_cols = src.size[1];
    const int dst_cols = dsize.width > 0 ? dsize.width : std::max(1, static_cast<int>(std::lround(src_cols * fx)));
    const int dst_rows = dsize.height > 0 ? dsize.height : std::max(1, static_cast<int>(std::lround(src_rows * fy)));
    CV_Assert(dst_cols > 0 && dst_rows > 0);

    Mat out({dst_rows, dst_cols}, src.type());
    const int cn = src.channels();
    for (int y = 0; y < dst_rows; ++y)
    {
        const int sy = std::min(src_rows - 1, (y * src_rows) / dst_rows);
        for (int x = 0; x < dst_cols; ++x)
        {
            const int sx = std::min(src_cols - 1, (x * src_cols) / dst_cols);
            for (int c = 0; c < cn; ++c)
            {
                out.at<float>(y, x, c) = src.at<float>(sy, sx, c);
            }
        }
    }
    return out;
}

Mat resize_reference_nearest_exact_f32(const Mat& src, Size dsize, double fx, double fy)
{
    CV_Assert(!src.empty());
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_32F);

    const int src_rows = src.size[0];
    const int src_cols = src.size[1];
    const int dst_cols = dsize.width > 0 ? dsize.width : std::max(1, static_cast<int>(std::lround(src_cols * fx)));
    const int dst_rows = dsize.height > 0 ? dsize.height : std::max(1, static_cast<int>(std::lround(src_rows * fy)));
    CV_Assert(dst_cols > 0 && dst_rows > 0);

    Mat out({dst_rows, dst_cols}, src.type());
    const int cn = src.channels();
    const int64_t ifx = ((static_cast<int64_t>(src_cols) << 16) + dst_cols / 2) / dst_cols;
    const int64_t ifx0 = ifx / 2 - (src_cols % 2);
    const int64_t ify = ((static_cast<int64_t>(src_rows) << 16) + dst_rows / 2) / dst_rows;
    const int64_t ify0 = ify / 2 - (src_rows % 2);

    for (int y = 0; y < dst_rows; ++y)
    {
        const int sy = std::clamp(static_cast<int>((ify * y + ify0) >> 16), 0, src_rows - 1);
        for (int x = 0; x < dst_cols; ++x)
        {
            const int sx = std::clamp(static_cast<int>((ifx * x + ifx0) >> 16), 0, src_cols - 1);
            for (int c = 0; c < cn; ++c)
            {
                out.at<float>(y, x, c) = src.at<float>(sy, sx, c);
            }
        }
    }
    return out;
}

}  // namespace

TEST(ImgprocResize_TEST, nearest_regression_15075_from_upstream_imgwarp)
{
    // Ported from OpenCV: modules/imgproc/test/test_imgwarp.cpp
    // TEST(Resize, nearest_regression_15075)
    const int channels = 5;
    const int col = 5;
    const int row = 5;

    Mat src({12, 12}, CV_8UC(channels));
    src = 0;
    for (int ch = 0; ch < channels; ++ch)
    {
        src.at<uchar>(row, col, ch) = 1;
    }

    Mat dst;
    resize(src, dst, Size(11, 11), 0.0, 0.0, INTER_NEAREST);

    ASSERT_EQ(dst.type(), CV_8UC(channels));
    EXPECT_EQ(dst.size[0], 11);
    EXPECT_EQ(dst.size[1], 11);
    EXPECT_EQ(l1_u8(dst), static_cast<double>(channels));
}

TEST(ImgprocResize_TEST, nearest_matches_smoke_reference_grid)
{
    Mat src({2, 2}, CV_8UC1);
    src.at<uchar>(0, 0) = 1;
    src.at<uchar>(0, 1) = 2;
    src.at<uchar>(1, 0) = 3;
    src.at<uchar>(1, 1) = 4;

    Mat dst;
    resize(src, dst, Size(3, 3), 0.0, 0.0, INTER_NEAREST);
    ASSERT_EQ(dst.type(), CV_8UC1);
    ASSERT_EQ(dst.size[0], 3);
    ASSERT_EQ(dst.size[1], 3);

    const uchar expected[9] = {
        1, 1, 2,
        1, 1, 2,
        3, 3, 4,
    };
    for (int y = 0; y < 3; ++y)
    {
        for (int x = 0; x < 3; ++x)
        {
            EXPECT_EQ(dst.at<uchar>(y, x), expected[y * 3 + x]);
        }
    }
}

TEST(ImgprocResize_TEST, linear_matches_independent_reference_on_u8)
{
    Mat src({4, 5}, CV_8UC3);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            src.at<uchar>(y, x, 0) = static_cast<uchar>((y * 17 + x * 11) % 256);
            src.at<uchar>(y, x, 1) = static_cast<uchar>((y * 7 + x * 23) % 256);
            src.at<uchar>(y, x, 2) = static_cast<uchar>((y * 19 + x * 5) % 256);
        }
    }

    Mat expected = resize_reference_linear_u8(src, Size(7, 6), 0.0, 0.0);
    Mat actual;
    resize(src, actual, Size(7, 6), 0.0, 0.0, INTER_LINEAR);

    ASSERT_EQ(actual.type(), expected.type());
    ASSERT_EQ(actual.size[0], expected.size[0]);
    ASSERT_EQ(actual.size[1], expected.size[1]);
    EXPECT_EQ(max_abs_diff_u8(actual, expected), 0);
}

TEST(ImgprocResize_TEST, dsize_takes_precedence_over_fx_fy)
{
    Mat src({8, 6}, CV_8UC1);
    src = 9;

    Mat dst;
    resize(src, dst, Size(3, 4), 10.0, 10.0, INTER_NEAREST);
    EXPECT_EQ(dst.size[0], 4);
    EXPECT_EQ(dst.size[1], 3);
}

TEST(ImgprocResize_TEST, fx_fy_are_used_when_dsize_is_empty)
{
    Mat src({6, 10}, CV_8UC1);
    src = 1;

    Mat dst;
    resize(src, dst, Size(), 0.5, 0.5, INTER_NEAREST);
    EXPECT_EQ(dst.size[0], 3);
    EXPECT_EQ(dst.size[1], 5);
}

TEST(ImgprocResize_TEST, nearest_exact_nearest8u_port_from_upstream)
{
    // Ported from OpenCV: modules/imgproc/test/test_resize_bitexact.cpp
    // TEST(Resize_Bitexact, Nearest8U)
    struct Case
    {
        Mat src;
        Mat expected;
    };

    const std::vector<Case> cases = {
        {mat_u8(1, 6, {0, 1, 2, 3, 4, 5}), mat_u8(1, 3, {1, 3, 5})},
        {mat_u8(1, 5, {0, 1, 2, 3, 4}), mat_u8(1, 1, {2})},
        {mat_u8(1, 5, {0, 1, 2, 3, 4}), mat_u8(1, 3, {0, 2, 4})},
        {mat_u8(1, 5, {0, 1, 2, 3, 4}), mat_u8(1, 2, {1, 3})},
        {
            mat_u8(3, 5, {
                0, 1, 2, 3, 4,
                5, 6, 7, 8, 9,
                10, 11, 12, 13, 14,
            }),
            mat_u8(5, 7, {
                0, 1, 1, 2, 3, 3, 4,
                0, 1, 1, 2, 3, 3, 4,
                5, 6, 6, 7, 8, 8, 9,
                10, 11, 11, 12, 13, 13, 14,
                10, 11, 11, 12, 13, 13, 14,
            }),
        },
        {
            mat_u8(2, 3, {
                0, 1, 2,
                3, 4, 5,
            }),
            mat_u8(4, 6, {
                0, 0, 1, 1, 2, 2,
                0, 0, 1, 1, 2, 2,
                3, 3, 4, 4, 5, 5,
                3, 3, 4, 4, 5, 5,
            }),
        },
    };

    for (size_t i = 0; i < cases.size(); ++i)
    {
        SCOPED_TRACE(i);
        Mat calc;
        resize(cases[i].src, calc, Size(cases[i].expected.size[1], cases[i].expected.size[0]), 0.0, 0.0, INTER_NEAREST_EXACT);
        EXPECT_EQ(max_abs_diff_u8(calc, cases[i].expected), 0);

        const Mat src_t = transpose_u8(cases[i].src);
        const Mat expected_t = transpose_u8(cases[i].expected);
        Mat calc_t;
        resize(src_t, calc_t, Size(expected_t.size[1], expected_t.size[0]), 0.0, 0.0, INTER_NEAREST_EXACT);
        EXPECT_EQ(max_abs_diff_u8(calc_t, expected_t), 0);
    }
}

TEST(ImgprocResize_TEST, throws_on_invalid_inputs)
{
    Mat dst;
    const Mat empty;
    EXPECT_THROW(resize(empty, dst, Size(2, 2), 0.0, 0.0, INTER_NEAREST), Exception);

    const Mat u16({4, 4}, CV_16UC1);
    EXPECT_THROW(resize(u16, dst, Size(2, 2), 0.0, 0.0, INTER_NEAREST), Exception);

    const Mat src({4, 4}, CV_8UC1);
    EXPECT_THROW(resize(src, dst, Size(), 0.0, 0.0, INTER_NEAREST), Exception);
    EXPECT_THROW(resize(src, dst, Size(2, 2), 0.0, 0.0, 1234), Exception);
}

TEST(ImgprocResize_TEST, non_contiguous_roi_matches_reference_for_all_interpolations)
{
    Mat base({9, 13}, CV_8UC4);
    for (int y = 0; y < base.size[0]; ++y)
    {
        for (int x = 0; x < base.size[1]; ++x)
        {
            base.at<uchar>(y, x, 0) = static_cast<uchar>((y * 13 + x * 7 + 1) % 256);
            base.at<uchar>(y, x, 1) = static_cast<uchar>((y * 3 + x * 19 + 2) % 256);
            base.at<uchar>(y, x, 2) = static_cast<uchar>((y * 11 + x * 5 + 3) % 256);
            base.at<uchar>(y, x, 3) = static_cast<uchar>((y * 17 + x * 9 + 4) % 256);
        }
    }

    Mat roi = base.colRange(2, 11);
    ASSERT_FALSE(roi.isContinuous());
    ASSERT_EQ(roi.channels(), 4);

    struct Case
    {
        int interpolation;
        Size dsize;
    };

    const std::vector<Case> cases = {
        {INTER_NEAREST, Size(7, 5)},
        {INTER_NEAREST_EXACT, Size(8, 6)},
        {INTER_LINEAR, Size(6, 7)},
    };

    for (const auto& c : cases)
    {
        SCOPED_TRACE(c.interpolation);
        Mat expected;
        if (c.interpolation == INTER_NEAREST)
        {
            expected = resize_reference_nearest_u8(roi, c.dsize, 0.0, 0.0);
        }
        else if (c.interpolation == INTER_NEAREST_EXACT)
        {
            expected = resize_reference_nearest_exact_u8(roi, c.dsize, 0.0, 0.0);
        }
        else
        {
            expected = resize_reference_linear_u8(roi, c.dsize, 0.0, 0.0);
        }

        Mat actual;
        resize(roi, actual, c.dsize, 0.0, 0.0, c.interpolation);

        ASSERT_EQ(actual.type(), expected.type());
        ASSERT_EQ(actual.size[0], expected.size[0]);
        ASSERT_EQ(actual.size[1], expected.size[1]);
        EXPECT_EQ(max_abs_diff_u8(actual, expected), 0);
    }
}

TEST(ImgprocResize_TEST, boundary_sizes_single_row_and_single_col_match_reference)
{
    Mat row_src({1, 9}, CV_8UC3);
    for (int x = 0; x < row_src.size[1]; ++x)
    {
        row_src.at<uchar>(0, x, 0) = static_cast<uchar>((x * 13 + 5) % 256);
        row_src.at<uchar>(0, x, 1) = static_cast<uchar>((x * 7 + 9) % 256);
        row_src.at<uchar>(0, x, 2) = static_cast<uchar>((x * 3 + 17) % 256);
    }

    Mat col_src({9, 1}, CV_8UC3);
    for (int y = 0; y < col_src.size[0]; ++y)
    {
        col_src.at<uchar>(y, 0, 0) = static_cast<uchar>((y * 5 + 3) % 256);
        col_src.at<uchar>(y, 0, 1) = static_cast<uchar>((y * 11 + 1) % 256);
        col_src.at<uchar>(y, 0, 2) = static_cast<uchar>((y * 17 + 7) % 256);
    }

    {
        Mat expected = resize_reference_linear_u8(row_src, Size(13, 1), 0.0, 0.0);
        Mat actual;
        resize(row_src, actual, Size(13, 1), 0.0, 0.0, INTER_LINEAR);
        EXPECT_EQ(max_abs_diff_u8(actual, expected), 0);
    }

    {
        Mat expected = resize_reference_nearest_exact_u8(row_src, Size(4, 1), 0.0, 0.0);
        Mat actual;
        resize(row_src, actual, Size(4, 1), 0.0, 0.0, INTER_NEAREST_EXACT);
        EXPECT_EQ(max_abs_diff_u8(actual, expected), 0);
    }

    {
        Mat expected = resize_reference_linear_u8(col_src, Size(1, 13), 0.0, 0.0);
        Mat actual;
        resize(col_src, actual, Size(1, 13), 0.0, 0.0, INTER_LINEAR);
        EXPECT_EQ(max_abs_diff_u8(actual, expected), 0);
    }

    {
        Mat expected = resize_reference_nearest_u8(col_src, Size(1, 4), 0.0, 0.0);
        Mat actual;
        resize(col_src, actual, Size(1, 4), 0.0, 0.0, INTER_NEAREST);
        EXPECT_EQ(max_abs_diff_u8(actual, expected), 0);
    }
}

TEST(ImgprocResize_TEST, supports_cv32f_all_interpolations_for_c1_c3_c4)
{
    for (int cn : {1, 3, 4})
    {
        SCOPED_TRACE(cn);
        Mat src({6, 7}, CV_MAKETYPE(CV_32F, cn));
        for (int y = 0; y < src.size[0]; ++y)
        {
            for (int x = 0; x < src.size[1]; ++x)
            {
                for (int c = 0; c < cn; ++c)
                {
                    src.at<float>(y, x, c) = static_cast<float>((y - 2) * 0.75 + (x - 3) * 0.42 + c * 1.13);
                }
            }
        }

        struct Case
        {
            int interpolation;
            Size dsize;
        };
        const std::vector<Case> cases = {
            {INTER_NEAREST, Size(5, 4)},
            {INTER_NEAREST_EXACT, Size(8, 5)},
            {INTER_LINEAR, Size(9, 6)},
        };

        for (const auto& c : cases)
        {
            SCOPED_TRACE(c.interpolation);
            Mat expected;
            if (c.interpolation == INTER_NEAREST)
            {
                expected = resize_reference_nearest_f32(src, c.dsize, 0.0, 0.0);
            }
            else if (c.interpolation == INTER_NEAREST_EXACT)
            {
                expected = resize_reference_nearest_exact_f32(src, c.dsize, 0.0, 0.0);
            }
            else
            {
                expected = resize_reference_linear_f32(src, c.dsize, 0.0, 0.0);
            }

            Mat actual;
            resize(src, actual, c.dsize, 0.0, 0.0, c.interpolation);
            ASSERT_EQ(actual.type(), expected.type());
            ASSERT_EQ(actual.size[0], expected.size[0]);
            ASSERT_EQ(actual.size[1], expected.size[1]);
            EXPECT_LE(max_abs_diff_f32(actual, expected), 1e-5f);
        }
    }
}

TEST(ImgprocResize_TEST, cv32f_non_contiguous_roi_matches_reference)
{
    Mat base({8, 11}, CV_32FC3);
    for (int y = 0; y < base.size[0]; ++y)
    {
        for (int x = 0; x < base.size[1]; ++x)
        {
            for (int c = 0; c < 3; ++c)
            {
                base.at<float>(y, x, c) = static_cast<float>((y * 0.8) - (x * 0.15) + c * 1.7);
            }
        }
    }

    Mat roi = base.colRange(2, 10);
    ASSERT_FALSE(roi.isContinuous());

    Mat expected = resize_reference_linear_f32(roi, Size(7, 6), 0.0, 0.0);
    Mat actual;
    resize(roi, actual, Size(7, 6), 0.0, 0.0, INTER_LINEAR);

    ASSERT_EQ(actual.type(), expected.type());
    ASSERT_EQ(actual.size[0], expected.size[0]);
    ASSERT_EQ(actual.size[1], expected.size[1]);
    EXPECT_LE(max_abs_diff_f32(actual, expected), 1e-5f);
}
