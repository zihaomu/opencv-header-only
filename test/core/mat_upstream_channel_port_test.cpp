#include "cvh.h"
#include "gtest/gtest.h"

#include <cstdlib>
#include <cstdint>

using namespace cvh;

namespace
{
void skip_pending(const char* upstream_case, const char* reason)
{
    GTEST_SKIP() << "Pending upstream port: " << upstream_case << " | " << reason;
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

void fill_lut_u8_c1(Mat& lut)
{
    CV_Assert(lut.total() == 256 && lut.type() == CV_8UC1);
    for (int i = 0; i < 256; ++i)
    {
        lut.at<uchar>(0, i) = static_cast<uchar>((i * 13 + 7) & 0xFF);
    }
}

void fill_lut_u8_c3(Mat& lut)
{
    CV_Assert(lut.total() == 256 && lut.type() == CV_8UC3);
    for (int i = 0; i < 256; ++i)
    {
        lut.at<uchar>(0, i, 0) = static_cast<uchar>((i * 5 + 1) & 0xFF);
        lut.at<uchar>(0, i, 1) = static_cast<uchar>((i * 7 + 3) & 0xFF);
        lut.at<uchar>(0, i, 2) = static_cast<uchar>((i * 11 + 9) & 0xFF);
    }
}

Mat lut_reference_u8(const Mat& input, const Mat& table)
{
    CV_Assert(input.depth() == CV_8U);
    CV_Assert(table.depth() == CV_8U);
    CV_Assert(table.total() == 256);
    CV_Assert(table.channels() == 1 || table.channels() == input.channels());

    Mat ref({input.size[0], input.size[1]}, CV_MAKETYPE(CV_8U, input.channels()));
    const int lut_cn = table.channels();
    for (int y = 0; y < input.size[0]; ++y)
    {
        for (int x = 0; x < input.size[1]; ++x)
        {
            for (int c = 0; c < input.channels(); ++c)
            {
                const int idx = static_cast<int>(input.at<uchar>(y, x, c));
                ref.at<uchar>(y, x, c) = table.at<uchar>(0, idx, lut_cn == 1 ? 0 : c);
            }
        }
    }
    return ref;
}

int max_abs_diff_u8(const Mat& a, const Mat& b)
{
    CV_Assert(a.type() == b.type());
    CV_Assert(a.total() == b.total());
    CV_Assert(a.channels() == b.channels());
    CV_Assert(a.depth() == CV_8U);
    const size_t count = a.total() * static_cast<size_t>(a.channels());
    int max_diff = 0;
    for (size_t i = 0; i < count; ++i)
    {
        const int d = std::abs(static_cast<int>(a.data[i]) - static_cast<int>(b.data[i]));
        if (d > max_diff)
        {
            max_diff = d;
        }
    }
    return max_diff;
}
} // namespace

// From modules/core/test/test_mat.cpp
TEST(OpenCVUpstreamChannelPort_TEST, Core_Merge_shape_operations)
{
    Mat c0({2, 3}, CV_8UC1);
    Mat c1({2, 3}, CV_8UC2);
    c0.setTo(Scalar::all(7));
    c1.setTo(Scalar(11, 13, 0, 0));

    std::vector<Mat> src = {c0, c1};
    Mat merged;
    merge(src, merged);

    ASSERT_EQ(merged.type(), CV_8UC3);
    ASSERT_EQ(merged.shape(), (MatShape{2, 3}));
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 3; ++x)
        {
            EXPECT_EQ(merged.at<uchar>(y, x, 0), static_cast<uchar>(7));
            EXPECT_EQ(merged.at<uchar>(y, x, 1), static_cast<uchar>(11));
            EXPECT_EQ(merged.at<uchar>(y, x, 2), static_cast<uchar>(13));
        }
    }
}

TEST(OpenCVUpstreamChannelPort_TEST, Core_Split_shape_operations)
{
    Mat src({2, 2}, CV_16SC3);
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 2; ++x)
        {
            for (int ch = 0; ch < 3; ++ch)
            {
                src.at<short>(y, x, ch) = static_cast<short>(y * 100 + x * 10 + ch);
            }
        }
    }

    Mat planes[3];
    split(src, planes);

    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 2; ++x)
        {
            EXPECT_EQ(planes[0].at<short>(y, x), static_cast<short>(y * 100 + x * 10 + 0));
            EXPECT_EQ(planes[1].at<short>(y, x), static_cast<short>(y * 100 + x * 10 + 1));
            EXPECT_EQ(planes[2].at<short>(y, x), static_cast<short>(y * 100 + x * 10 + 2));
        }
    }
}

TEST(OpenCVUpstreamChannelPort_TEST, Core_Merge_hang_12171)
{
    Mat src1({4, 24}, CV_8UC1);
    Mat src2({4, 24}, CV_8UC1);
    src1.setTo(Scalar::all(1));
    src2.setTo(Scalar::all(2));

    Mat src_channels[2] = {
        src1.colRange(0, 23),
        src2.colRange(0, 23),
    };

    Mat dst({4, 24}, CV_8UC2);
    dst.setTo(Scalar::all(5));
    Mat dst_roi = dst.colRange(1, 24);
    merge(src_channels, 2, dst_roi);

    EXPECT_EQ(dst.at<uchar>(0, 0, 0), static_cast<uchar>(5));
    EXPECT_EQ(dst.at<uchar>(0, 0, 1), static_cast<uchar>(5));
    EXPECT_EQ(dst.at<uchar>(0, 1, 0), static_cast<uchar>(1));
    EXPECT_EQ(dst.at<uchar>(0, 1, 1), static_cast<uchar>(2));
    EXPECT_EQ(dst.at<uchar>(1, 0, 0), static_cast<uchar>(5));
    EXPECT_EQ(dst.at<uchar>(1, 0, 1), static_cast<uchar>(5));
    EXPECT_EQ(dst.at<uchar>(1, 1, 0), static_cast<uchar>(1));
    EXPECT_EQ(dst.at<uchar>(1, 1, 1), static_cast<uchar>(2));
}

TEST(OpenCVUpstreamChannelPort_TEST, Core_Split_hang_12171)
{
    Mat src({4, 24}, CV_8UC2);
    src.setTo(Scalar(1, 2, 3, 4));
    Mat src_roi = src.colRange(0, 23);

    Mat dst1({4, 24}, CV_8UC1);
    Mat dst2({4, 24}, CV_8UC1);
    dst1.setTo(Scalar::all(5));
    dst2.setTo(Scalar::all(10));
    Mat dst_channels[2] = {
        dst1.colRange(0, 23),
        dst2.colRange(0, 23),
    };
    split(src_roi, dst_channels);

    EXPECT_EQ(dst1.at<uchar>(0, 0), static_cast<uchar>(1));
    EXPECT_EQ(dst1.at<uchar>(0, 1), static_cast<uchar>(1));
    EXPECT_EQ(dst2.at<uchar>(0, 0), static_cast<uchar>(2));
    EXPECT_EQ(dst2.at<uchar>(0, 1), static_cast<uchar>(2));
    EXPECT_EQ(dst1.at<uchar>(1, 0), static_cast<uchar>(1));
    EXPECT_EQ(dst1.at<uchar>(1, 1), static_cast<uchar>(1));
    EXPECT_EQ(dst2.at<uchar>(1, 0), static_cast<uchar>(2));
    EXPECT_EQ(dst2.at<uchar>(1, 1), static_cast<uchar>(2));
}

TEST(OpenCVUpstreamChannelPort_TEST, Core_Split_crash_12171)
{
    Mat src({4, 40}, CV_8UC2);
    src.setTo(Scalar(1, 2, 3, 4));
    Mat src_roi = src.colRange(0, 39);

    Mat dst1({4, 40}, CV_8UC1);
    Mat dst2({4, 40}, CV_8UC1);
    dst1.setTo(Scalar::all(5));
    dst2.setTo(Scalar::all(10));
    Mat dst_channels[2] = {
        dst1.colRange(0, 39),
        dst2.colRange(0, 39),
    };
    split(src_roi, dst_channels);

    EXPECT_EQ(dst1.at<uchar>(0, 0), static_cast<uchar>(1));
    EXPECT_EQ(dst1.at<uchar>(0, 1), static_cast<uchar>(1));
    EXPECT_EQ(dst2.at<uchar>(0, 0), static_cast<uchar>(2));
    EXPECT_EQ(dst2.at<uchar>(0, 1), static_cast<uchar>(2));
    EXPECT_EQ(dst1.at<uchar>(1, 0), static_cast<uchar>(1));
    EXPECT_EQ(dst1.at<uchar>(1, 1), static_cast<uchar>(1));
    EXPECT_EQ(dst2.at<uchar>(1, 0), static_cast<uchar>(2));
    EXPECT_EQ(dst2.at<uchar>(1, 1), static_cast<uchar>(2));
}

TEST(OpenCVUpstreamChannelPort_TEST, Core_Merge_bug_13544)
{
    Mat src1({2, 2}, CV_8UC3);
    Mat src2({2, 2}, CV_8UC3);
    Mat src3({2, 2}, CV_8UC3);
    src1.setTo(Scalar::all(1));
    src2.setTo(Scalar::all(2));
    src3.setTo(Scalar::all(3));

    Mat src_arr[] = {src1, src2, src3};
    Mat dst;
    merge(src_arr, 3, dst);

    ASSERT_EQ(dst.channels(), 9);
    EXPECT_EQ(dst.at<uchar>(0, 0, 6), static_cast<uchar>(3));
    EXPECT_EQ(dst.at<uchar>(0, 0, 7), static_cast<uchar>(3));
    EXPECT_EQ(dst.at<uchar>(0, 0, 8), static_cast<uchar>(3));
    EXPECT_EQ(dst.at<uchar>(1, 0, 0), static_cast<uchar>(1));
    EXPECT_EQ(dst.at<uchar>(1, 0, 1), static_cast<uchar>(1));
    EXPECT_EQ(dst.at<uchar>(1, 0, 2), static_cast<uchar>(1));
    EXPECT_EQ(dst.at<uchar>(1, 0, 3), static_cast<uchar>(2));
    EXPECT_EQ(dst.at<uchar>(1, 0, 4), static_cast<uchar>(2));
    EXPECT_EQ(dst.at<uchar>(1, 0, 5), static_cast<uchar>(2));
    EXPECT_EQ(dst.at<uchar>(1, 0, 6), static_cast<uchar>(3));
    EXPECT_EQ(dst.at<uchar>(1, 0, 7), static_cast<uchar>(3));
    EXPECT_EQ(dst.at<uchar>(1, 0, 8), static_cast<uchar>(3));
}

TEST(OpenCVUpstreamChannelPort_TEST, Core_Mat_reinterpret_Mat_8UC3_8SC3)
{
    Mat A({8, 16}, CV_8UC3);
    A.setTo(Scalar(1, 2, 3));
    Mat B = A.reinterpret(CV_8SC3);

    EXPECT_EQ(A.data, B.data);
    EXPECT_EQ(B.type(), CV_8SC3);
    EXPECT_EQ(B.shape(), A.shape());
}

TEST(OpenCVUpstreamChannelPort_TEST, Core_Mat_reinterpret_Mat_8UC4_32FC1)
{
    Mat A({8, 16}, CV_8UC4);
    A.setTo(Scalar(1, 2, 3, 4));
    Mat B = A.reinterpret(CV_32FC1);

    EXPECT_EQ(A.data, B.data);
    EXPECT_EQ(B.type(), CV_32FC1);
    EXPECT_EQ(B.shape(), A.shape());
}

TEST(OpenCVUpstreamChannelPort_TEST, Core_Mat_reinterpret_OutputArray_8UC3_8SC3)
{
    skip_pending("Core_Mat.reinterpret_OutputArray_8UC3_8SC3",
                 "by design: OutputArray compatibility is out of scope in Mat-only v1");
}

TEST(OpenCVUpstreamChannelPort_TEST, Core_Mat_reinterpret_OutputArray_8UC4_32FC1)
{
    skip_pending("Core_Mat.reinterpret_OutputArray_8UC4_32FC1",
                 "by design: OutputArray compatibility is out of scope in Mat-only v1");
}

TEST(OpenCVUpstreamChannelPort_TEST, Core_MatExpr_issue_16655)
{
    Mat a({5, 5}, CV_32FC3);
    Mat b({5, 5}, CV_32FC3);
    a.setTo(Scalar::all(1.0));
    b.setTo(Scalar::all(2.0));

    MatExpr ab_expr = a != b;
    Mat ab_mat = ab_expr;
    EXPECT_EQ(CV_8UC3, ab_expr.type());
    EXPECT_EQ(CV_8UC3, ab_mat.type());
}

// From modules/core/test/test_arithm.cpp
TEST(OpenCVUpstreamChannelPort_TEST, Subtract_scalarc1_matc3)
{
    int scalar = 255;
    Mat srcImage({5, 5}, CV_8UC3);
    srcImage = Scalar::all(5.0);

    Mat destImage;
    subtract(scalar, srcImage, destImage);

    ASSERT_EQ(destImage.type(), CV_8UC3);
    ASSERT_EQ(destImage.shape(), srcImage.shape());
    for (int y = 0; y < 5; ++y)
    {
        for (int x = 0; x < 5; ++x)
        {
            for (int ch = 0; ch < 3; ++ch)
            {
                EXPECT_EQ(destImage.at<uchar>(y, x, ch), static_cast<uchar>(250));
            }
        }
    }
}

TEST(OpenCVUpstreamChannelPort_TEST, Subtract_scalarc4_matc4)
{
    const Scalar sc(255.0, 255.0, 255.0, 255.0);
    Mat srcImage({5, 5}, CV_8UC4);
    srcImage = Scalar::all(5.0);

    Mat destImage;
    subtract(sc, srcImage, destImage);

    ASSERT_EQ(destImage.type(), CV_8UC4);
    ASSERT_EQ(destImage.shape(), srcImage.shape());
    for (int y = 0; y < 5; ++y)
    {
        for (int x = 0; x < 5; ++x)
        {
            for (int ch = 0; ch < 4; ++ch)
            {
                EXPECT_EQ(destImage.at<uchar>(y, x, ch), static_cast<uchar>(250));
            }
        }
    }
}

TEST(OpenCVUpstreamChannelPort_TEST, Compare_empty)
{
    Mat temp;
    Mat dst1;
    Mat dst2;

    EXPECT_NO_THROW(compare(temp, temp, dst1, CV_CMP_EQ));
    EXPECT_TRUE(dst1.empty());
    EXPECT_THROW(dst2 = temp > 5, Exception);
}

TEST(OpenCVUpstreamChannelPort_TEST, Compare_regression_8999)
{
    Mat A({4, 1}, CV_32F);
    Mat B({1, 1}, CV_32F);
    A.at<float>(0, 0) = 1.0f;
    A.at<float>(1, 0) = 3.0f;
    A.at<float>(2, 0) = 2.0f;
    A.at<float>(3, 0) = 4.0f;
    B.at<float>(0, 0) = 2.0f;

    Mat C;
    EXPECT_THROW(compare(A, B, C, CV_CMP_LT), Exception);
}

TEST(OpenCVUpstreamChannelPort_TEST, Compare_regression_16F_do_not_crash)
{
    Mat mat1({2, 2}, CV_16FC1);
    Mat mat2({2, 2}, CV_16FC1);
    mat1.setTo(Scalar::all(1));
    mat2.setTo(Scalar::all(2));

    Mat dst;
    EXPECT_THROW(compare(mat1, mat2, dst, CV_CMP_EQ), Exception);
}

// From modules/core/test/test_operations.cpp
TEST(OpenCVUpstreamChannelPort_TEST, Core_Array_expressions)
{
    // Keep this runnable as a minimal channel-semantics checkpoint extracted from CV_OperationsTest.
    Mat m1({1, 1}, CV_8UC1);
    Mat m2({1, 1}, CV_32FC2);
    Mat m3({1, 1}, CV_16SC3);

    EXPECT_EQ(m1.channels(), 1);
    EXPECT_EQ(m2.channels(), 2);
    EXPECT_EQ(m3.channels(), 3);
}

TEST(OpenCVUpstreamChannelPort_TEST, Core_LUT_accuracy)
{
    Mat input({117, 113}, CV_8UC1);
    fill_u8_pattern(input, 0x1234u);

    Mat table({1, 256}, CV_8UC1);
    fill_lut_u8_c1(table);

    Mat output;
    ASSERT_NO_THROW(LUT(input, table, output));
    ASSERT_FALSE(output.empty());

    const Mat gt = lut_reference_u8(input, table);
    ASSERT_FALSE(gt.empty());
    ASSERT_EQ(0, max_abs_diff_u8(output, gt));
}

TEST(OpenCVUpstreamChannelPort_TEST, Core_LUT_accuracy_multi)
{
    Mat input({117, 113}, CV_8UC3);
    fill_u8_pattern(input, 0x5678u);

    Mat table({1, 256}, CV_8UC1);
    fill_lut_u8_c1(table);

    Mat output;
    ASSERT_NO_THROW(LUT(input, table, output));
    ASSERT_FALSE(output.empty());

    const Mat gt = lut_reference_u8(input, table);
    ASSERT_FALSE(gt.empty());
    ASSERT_EQ(0, max_abs_diff_u8(output, gt));
}

TEST(OpenCVUpstreamChannelPort_TEST, Core_LUT_accuracy_multi2)
{
    Mat input({117, 113}, CV_8UC3);
    fill_u8_pattern(input, 0x9abcu);

    Mat table({1, 256}, CV_8UC3);
    fill_lut_u8_c3(table);

    Mat output;
    ASSERT_NO_THROW(LUT(input, table, output));
    ASSERT_FALSE(output.empty());

    const Mat gt = lut_reference_u8(input, table);
    ASSERT_FALSE(gt.empty());
    ASSERT_EQ(0, max_abs_diff_u8(output, gt));
}
