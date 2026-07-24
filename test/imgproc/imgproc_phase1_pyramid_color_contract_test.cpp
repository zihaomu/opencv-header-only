#include "cvh.h"
#include "gtest/gtest.h"

#include <cstring>
#include <vector>

using namespace cvh;

TEST(ImgprocPhase1PyramidColor_TEST, accumulate_family_covers_mask_and_repeated_updates)
{
    Mat src1({3, 4}, CV_8UC3);
    Mat src2({3, 4}, CV_8UC3);
    for (int y = 0; y < 3; ++y)
    {
        for (int x = 0; x < 4; ++x)
        {
            for (int ch = 0; ch < 3; ++ch)
            {
                src1.at<uchar>(y, x, ch) =
                    static_cast<uchar>(1 + x + y + ch);
                src2.at<uchar>(y, x, ch) =
                    static_cast<uchar>(2 + 2 * x + ch);
            }
        }
    }
    Mat mask({3, 4}, CV_8UC1);
    for (int y = 0; y < 3; ++y)
    {
        for (int x = 0; x < 4; ++x)
        {
            mask.at<uchar>(y, x) = ((x + y) & 1) ? 255 : 0;
        }
    }

    Mat dst({3, 4}, CV_32FC3);
    dst.setTo(Scalar::all(1.0));
    accumulate(src1, dst, mask);
    accumulate(src1, dst, mask);
    EXPECT_FLOAT_EQ(dst.at<float>(0, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(
        dst.at<float>(0, 1, 2),
        1.0f + 2.0f * src1.at<uchar>(0, 1, 2));

    dst.setTo(Scalar::all(0.0));
    accumulateSquare(src1, dst);
    EXPECT_FLOAT_EQ(
        dst.at<float>(2, 3, 1),
        static_cast<float>(
            src1.at<uchar>(2, 3, 1) *
            src1.at<uchar>(2, 3, 1)));

    dst.setTo(Scalar::all(0.0));
    accumulateProduct(src1, src2, dst, mask);
    EXPECT_FLOAT_EQ(dst.at<float>(0, 0, 0), 0.0f);
    EXPECT_FLOAT_EQ(
        dst.at<float>(1, 0, 2),
        static_cast<float>(
            src1.at<uchar>(1, 0, 2) *
            src2.at<uchar>(1, 0, 2)));

    Mat wrong;
    EXPECT_THROW(accumulate(src1, wrong), Exception);
}

TEST(ImgprocPhase1PyramidColor_TEST, accumulate_weighted_handles_alpha_extremes)
{
    Mat src({2, 3}, CV_32FC1);
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 3; ++x)
        {
            src.at<float>(y, x) = static_cast<float>(10 + y * 3 + x);
        }
    }
    Mat dst({2, 3}, CV_32FC1);
    dst.setTo(Scalar::all(3.0));
    accumulateWeighted(src, dst, 0.0);
    EXPECT_FLOAT_EQ(dst.at<float>(1, 2), 3.0f);
    accumulateWeighted(src, dst, 1.0);
    EXPECT_FLOAT_EQ(dst.at<float>(1, 2), src.at<float>(1, 2));
    dst.setTo(Scalar::all(2.0));
    accumulateWeighted(src, dst, 0.25);
    EXPECT_FLOAT_EQ(
        dst.at<float>(0, 0),
        0.75f * 2.0f + 0.25f * src.at<float>(0, 0));
}

TEST(ImgprocPhase1PyramidColor_TEST, blend_linear_handles_zero_and_non_normalized_weights)
{
    Mat first({2, 3}, CV_8UC3);
    Mat second({2, 3}, CV_8UC3);
    first.setTo(Scalar(20, 40, 60));
    second.setTo(Scalar(100, 120, 140));
    Mat weight1({2, 3}, CV_32FC1);
    Mat weight2({2, 3}, CV_32FC1);
    weight1.setTo(Scalar::all(0.0));
    weight2.setTo(Scalar::all(0.0));
    Mat result;
    blendLinear(first, second, weight1, weight2, result);
    EXPECT_EQ(result.at<uchar>(0, 0, 0), 0);

    weight1.setTo(Scalar::all(2.0));
    weight2.setTo(Scalar::all(1.0));
    blendLinear(first, second, weight1, weight2, result);
    EXPECT_NEAR(result.at<uchar>(1, 2, 0), 47, 1);

    weight1.setTo(Scalar::all(0.25));
    weight2.setTo(Scalar::all(0.75));
    blendLinear(first, second, weight1, weight2, first);
    EXPECT_NEAR(first.at<uchar>(0, 0, 2), 120, 1);
}

TEST(ImgprocPhase1PyramidColor_TEST, pyramid_sizes_constants_and_build_contract)
{
    Mat constant({7, 9}, CV_8UC3);
    constant.setTo(Scalar(12, 34, 210));
    Mat down;
    pyrDown(constant, down);
    EXPECT_EQ(down.shape(), MatShape({4, 5}));
    EXPECT_EQ(down.at<uchar>(2, 3, 0), 12);
    EXPECT_EQ(down.at<uchar>(2, 3, 2), 210);

    Mat up;
    pyrUp(down, up, Size(9, 7));
    EXPECT_EQ(up.shape(), constant.shape());
    EXPECT_EQ(up.at<uchar>(3, 4, 1), 34);

    std::vector<Mat> pyramid;
    buildPyramid(constant, pyramid, 3);
    ASSERT_EQ(pyramid.size(), 4u);
    for (size_t level = 1; level < pyramid.size(); ++level)
    {
        Mat expected;
        pyrDown(pyramid[level - 1], expected);
        EXPECT_EQ(expected.shape(), pyramid[level].shape());
        EXPECT_EQ(
            std::memcmp(
                expected.data,
                pyramid[level].data,
                expected.total() * expected.elemSize()),
            0);
    }
    EXPECT_THROW(pyrDown(constant, down, Size(20, 20)), Exception);
    EXPECT_THROW(pyrUp(down, up, Size(), BORDER_REPLICATE), Exception);
}

TEST(ImgprocPhase1PyramidColor_TEST, two_plane_matches_existing_packed_nv12_and_nv21)
{
    constexpr int rows = 6;
    constexpr int cols = 8;
    Mat y_parent({rows + 2, cols + 3}, CV_8UC1);
    Mat uv_parent({rows / 2 + 2, cols / 2 + 3}, CV_8UC2);
    Mat y = y_parent(Range(1, rows + 1), Range(1, cols + 1));
    Mat uv = uv_parent(
        Range(1, rows / 2 + 1), Range(1, cols / 2 + 1));
    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            y.at<uchar>(row, col) =
                static_cast<uchar>(16 + (row * 23 + col * 17) % 220);
        }
    }
    for (int row = 0; row < rows / 2; ++row)
    {
        for (int col = 0; col < cols / 2; ++col)
        {
            uv.at<uchar>(row, col, 0) =
                static_cast<uchar>(70 + row * 7 + col);
            uv.at<uchar>(row, col, 1) =
                static_cast<uchar>(150 + row + col * 3);
        }
    }

    for (const int code :
         {COLOR_YUV2BGR_NV12,
          COLOR_YUV2RGB_NV12,
          COLOR_YUV2BGR_NV21,
          COLOR_YUV2RGB_NV21})
    {
        Mat packed({rows + rows / 2, cols}, CV_8UC1);
        for (int row = 0; row < rows; ++row)
        {
            std::memcpy(
                packed.data + static_cast<size_t>(row) * packed.step(0),
                y.data + static_cast<size_t>(row) * y.step(0),
                cols);
        }
        for (int row = 0; row < rows / 2; ++row)
        {
            std::memcpy(
                packed.data +
                    static_cast<size_t>(rows + row) * packed.step(0),
                uv.data + static_cast<size_t>(row) * uv.step(0),
                cols);
        }
        Mat expected;
        Mat actual;
        cvtColor(packed, expected, code);
        cvtColorTwoPlane(y, uv, actual, code);
        EXPECT_EQ(
            std::memcmp(
                expected.data,
                actual.data,
                actual.total() * actual.elemSize()),
            0);
    }
}

TEST(ImgprocPhase1PyramidColor_TEST, demosaicing_covers_four_bayer_patterns)
{
    const int codes[] = {
        COLOR_BayerBG2BGR,
        COLOR_BayerGB2BGR,
        COLOR_BayerRG2BGR,
        COLOR_BayerGR2BGR,
    };
    for (const int code : codes)
    {
        const int pattern =
            demosaicing_detail::pattern_from_code(code);
        Mat bayer({7, 9}, CV_8UC1);
        for (int y = 0; y < bayer.size.p[0]; ++y)
        {
            for (int x = 0; x < bayer.size.p[1]; ++x)
            {
                const auto channel =
                    demosaicing_detail::color_at(y, x, pattern);
                bayer.at<uchar>(y, x) =
                    channel == demosaicing_detail::Blue
                        ? 20
                        : (channel == demosaicing_detail::Green ? 90 : 180);
            }
        }
        Mat color;
        demosaicing(bayer, color, code);
        EXPECT_EQ(color.at<uchar>(3, 4, 0), 20);
        EXPECT_EQ(color.at<uchar>(3, 4, 1), 90);
        EXPECT_EQ(color.at<uchar>(3, 4, 2), 180);
        EXPECT_EQ(color.at<uchar>(0, 0, 1), 90);
    }
    Mat source({3, 3}, CV_8UC1);
    Mat output;
    EXPECT_THROW(demosaicing(source, output, 999), Exception);
    EXPECT_THROW(demosaicing(source, output, COLOR_BayerBG2BGR, 4), Exception);
}
