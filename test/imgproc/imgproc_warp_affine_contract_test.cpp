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

int max_abs_diff_u8(const Mat& a, const Mat& b)
{
    CV_Assert(a.type() == b.type());
    CV_Assert(a.dims == 2 && b.dims == 2);
    CV_Assert(a.size[0] == b.size[0] && a.size[1] == b.size[1]);

    const int rows = a.size[0];
    const int cols = a.size[1];
    const int cn = a.channels();
    int max_diff = 0;
    for (int y = 0; y < rows; ++y)
    {
        const uchar* pa = a.data + static_cast<size_t>(y) * a.step(0);
        const uchar* pb = b.data + static_cast<size_t>(y) * b.step(0);
        for (int x = 0; x < cols; ++x)
        {
            const int base = x * cn;
            for (int c = 0; c < cn; ++c)
            {
                max_diff = std::max(max_diff, std::abs(static_cast<int>(pa[base + c]) - static_cast<int>(pb[base + c])));
            }
        }
    }
    return max_diff;
}

Mat make_affine_f32(double m00, double m01, double m02, double m10, double m11, double m12)
{
    Mat M({2, 3}, CV_32FC1);
    M.setTo(0.0f);
    M.at<float>(0, 0) = static_cast<float>(m00);
    M.at<float>(0, 1) = static_cast<float>(m01);
    M.at<float>(0, 2) = static_cast<float>(m02);
    M.at<float>(1, 0) = static_cast<float>(m10);
    M.at<float>(1, 1) = static_cast<float>(m11);
    M.at<float>(1, 2) = static_cast<float>(m12);
    return M;
}

}  // namespace

TEST(ImgprocWarpAffine_TEST, nearest_identity_u8_c3_keeps_values)
{
    Mat src({5, 7}, CV_8UC3);
    fill_u8_pattern(src, 0x1234u);

    Mat dst;
    const Mat M = make_affine_f32(1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    warpAffine(src, dst, M, Size(7, 5), INTER_NEAREST, BORDER_CONSTANT, Scalar::all(0.0));

    ASSERT_EQ(dst.type(), CV_8UC3);
    ASSERT_EQ(dst.size[0], 5);
    ASSERT_EQ(dst.size[1], 7);
    EXPECT_EQ(max_abs_diff_u8(src, dst), 0);
}

TEST(ImgprocWarpAffine_TEST, nearest_inverse_translation_constant_border_matches_expected)
{
    Mat src({2, 3}, CV_8UC1);
    src.at<uchar>(0, 0) = 1;
    src.at<uchar>(0, 1) = 2;
    src.at<uchar>(0, 2) = 3;
    src.at<uchar>(1, 0) = 4;
    src.at<uchar>(1, 1) = 5;
    src.at<uchar>(1, 2) = 6;

    Mat dst;
    const Mat M = make_affine_f32(1.0, 0.0, -1.0, 0.0, 1.0, 0.0);  // dst->src
    warpAffine(src, dst, M, Size(3, 2), INTER_NEAREST | WARP_INVERSE_MAP, BORDER_CONSTANT, Scalar::all(0.0));

    ASSERT_EQ(dst.type(), CV_8UC1);
    ASSERT_EQ(dst.size[0], 2);
    ASSERT_EQ(dst.size[1], 3);

    EXPECT_EQ(dst.at<uchar>(0, 0), 0);
    EXPECT_EQ(dst.at<uchar>(0, 1), 1);
    EXPECT_EQ(dst.at<uchar>(0, 2), 2);
    EXPECT_EQ(dst.at<uchar>(1, 0), 0);
    EXPECT_EQ(dst.at<uchar>(1, 1), 4);
    EXPECT_EQ(dst.at<uchar>(1, 2), 5);
}

TEST(ImgprocWarpAffine_TEST, linear_half_pixel_center_for_f32_is_expected_average)
{
    Mat src({2, 2}, CV_32FC1);
    src.at<float>(0, 0) = 0.0f;
    src.at<float>(0, 1) = 10.0f;
    src.at<float>(1, 0) = 20.0f;
    src.at<float>(1, 1) = 30.0f;

    Mat dst;
    const Mat M = make_affine_f32(0.0, 0.0, 0.5, 0.0, 0.0, 0.5);  // always sample (0.5, 0.5)
    warpAffine(src, dst, M, Size(1, 1), INTER_LINEAR | WARP_INVERSE_MAP, BORDER_REPLICATE, Scalar::all(0.0));

    ASSERT_EQ(dst.type(), CV_32FC1);
    ASSERT_EQ(dst.size[0], 1);
    ASSERT_EQ(dst.size[1], 1);
    EXPECT_NEAR(dst.at<float>(0, 0), 15.0f, 1e-5f);
}

TEST(ImgprocWarpAffine_TEST, inverse_map_flag_matches_internal_inverse_path)
{
    Mat src({6, 8}, CV_8UC1);
    fill_u8_pattern(src, 0x55u);

    const Mat M_forward = make_affine_f32(1.0, 0.0, 1.25, 0.0, 1.0, -0.5);  // src->dst
    const Mat M_inverse = make_affine_f32(1.0, 0.0, -1.25, 0.0, 1.0, 0.5);  // dst->src

    Mat dst_from_forward;
    Mat dst_from_inverse;
    warpAffine(src, dst_from_forward, M_forward, Size(8, 6), INTER_LINEAR, BORDER_REPLICATE, Scalar::all(0.0));
    warpAffine(src, dst_from_inverse, M_inverse, Size(8, 6), INTER_LINEAR | WARP_INVERSE_MAP, BORDER_REPLICATE, Scalar::all(0.0));

    EXPECT_EQ(max_abs_diff_u8(dst_from_forward, dst_from_inverse), 0);
}

TEST(ImgprocWarpAffine_TEST, roi_non_contiguous_input_is_supported)
{
    Mat full({10, 12}, CV_8UC1);
    fill_u8_pattern(full, 0x9911u);
    Mat roi = full(Range(2, 9), Range(1, 11));
    ASSERT_FALSE(roi.isContinuous());

    const Mat M = make_affine_f32(1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    Mat dst;
    warpAffine(roi, dst, M, Size(roi.size[1], roi.size[0]), INTER_NEAREST, BORDER_REFLECT_101, Scalar::all(0.0));

    ASSERT_EQ(dst.type(), roi.type());
    ASSERT_EQ(dst.size[0], roi.size[0]);
    ASSERT_EQ(dst.size[1], roi.size[1]);
    EXPECT_EQ(max_abs_diff_u8(dst, roi), 0);
}

TEST(ImgprocWarpAffine_TEST, in_place_same_mat_is_supported)
{
    Mat src({4, 5}, CV_8UC1);
    fill_u8_pattern(src, 0x42u);

    Mat expected;
    const Mat M = make_affine_f32(1.0, 0.0, -1.0, 0.0, 1.0, 0.0);
    warpAffine(src, expected, M, Size(5, 4), INTER_NEAREST | WARP_INVERSE_MAP, BORDER_REPLICATE, Scalar::all(0.0));

    warpAffine(src, src, M, Size(5, 4), INTER_NEAREST | WARP_INVERSE_MAP, BORDER_REPLICATE, Scalar::all(0.0));
    EXPECT_EQ(max_abs_diff_u8(src, expected), 0);
}

TEST(ImgprocWarpAffine_TEST, constant_border_uses_per_channel_scalar_for_outside_pixels)
{
    Mat src({3, 4}, CV_8UC4);
    src.setTo(Scalar(0.0, 0.0, 0.0, 0.0));

    Mat dst;
    const Mat M = make_affine_f32(1.0, 0.0, 100.0, 0.0, 1.0, 100.0);  // sample fully outside source
    const Scalar border(7.0, 11.0, 19.0, 255.0);
    warpAffine(src, dst, M, Size(4, 3), INTER_LINEAR | WARP_INVERSE_MAP, BORDER_CONSTANT, border);

    ASSERT_EQ(dst.type(), CV_8UC4);
    for (int y = 0; y < dst.size[0]; ++y)
    {
        const uchar* row = dst.data + static_cast<size_t>(y) * dst.step(0);
        for (int x = 0; x < dst.size[1]; ++x)
        {
            const uchar* px = row + static_cast<size_t>(x) * 4;
            EXPECT_EQ(px[0], 7);
            EXPECT_EQ(px[1], 11);
            EXPECT_EQ(px[2], 19);
            EXPECT_EQ(px[3], 255);
        }
    }
}

TEST(ImgprocWarpAffine_TEST, throws_on_invalid_arguments)
{
    Mat dst;
    Mat empty;
    Mat src({4, 4}, CV_8UC1);
    fill_u8_pattern(src, 0x33u);

    const Mat M_identity = make_affine_f32(1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    EXPECT_THROW(warpAffine(empty, dst, M_identity, Size(4, 4), INTER_NEAREST), Exception);

    Mat M_bad_shape({3, 3}, CV_32FC1);
    M_bad_shape.setTo(0.0f);
    EXPECT_THROW(warpAffine(src, dst, M_bad_shape, Size(4, 4), INTER_NEAREST), Exception);

    Mat M_bad_depth({2, 3}, CV_8UC1);
    M_bad_depth.setTo(0.0f);
    EXPECT_THROW(warpAffine(src, dst, M_bad_depth, Size(4, 4), INTER_NEAREST), Exception);

    Mat src_u16({4, 4}, CV_16UC1);
    EXPECT_THROW(warpAffine(src_u16, dst, M_identity, Size(4, 4), INTER_NEAREST), Exception);

    EXPECT_THROW(warpAffine(src, dst, M_identity, Size(4, 4), INTER_NEAREST_EXACT), Exception);
    EXPECT_THROW(warpAffine(src, dst, M_identity, Size(4, 4), INTER_NEAREST, BORDER_WRAP), Exception);

    Mat M_singular = make_affine_f32(1.0, 2.0, 0.0, 2.0, 4.0, 0.0);
    EXPECT_THROW(warpAffine(src, dst, M_singular, Size(4, 4), INTER_LINEAR), Exception);
}
