#include "cvh.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstring>

using namespace cvh;

namespace {

void fill_u8(Mat& matrix)
{
    for (int row = 0; row < matrix.size[0]; ++row)
    {
        for (int col = 0; col < matrix.size[1]; ++col)
        {
            for (int channel = 0; channel < matrix.channels(); ++channel)
            {
                matrix.at<uchar>(row, col, channel) =
                    static_cast<uchar>(
                        (row * 37 + col * 19 + channel * 53) & 255);
            }
        }
    }
}

void make_maps(Mat& map_x, Mat& map_y, int rows, int cols)
{
    map_x.create({rows, cols}, CV_32FC1);
    map_y.create({rows, cols}, CV_32FC1);
    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            map_x.at<float>(row, col) =
                static_cast<float>(col) + 0.28125f;
            map_y.at<float>(row, col) =
                static_cast<float>(row) - 0.34375f;
        }
    }
}

int max_u8_difference(const Mat& first, const Mat& second)
{
    EXPECT_EQ(first.type(), second.type());
    EXPECT_EQ(first.shape(), second.shape());
    int maximum = 0;
    for (int row = 0; row < first.size[0]; ++row)
    {
        const uchar* first_row =
            first.data + static_cast<size_t>(row) * first.step(0);
        const uchar* second_row =
            second.data + static_cast<size_t>(row) * second.step(0);
        const size_t count =
            static_cast<size_t>(first.size[1]) * first.channels();
        for (size_t index = 0; index < count; ++index)
        {
            maximum = std::max(
                maximum,
                std::abs(
                    static_cast<int>(first_row[index]) -
                    static_cast<int>(second_row[index])));
        }
    }
    return maximum;
}

Mat identity_perspective(int depth)
{
    Mat matrix({3, 3}, CV_MAKETYPE(depth, 1));
    matrix.setTo(Scalar::all(0.0));
    if (depth == CV_32F)
    {
        matrix.at<float>(0, 0) = 1.0f;
        matrix.at<float>(1, 1) = 1.0f;
        matrix.at<float>(2, 2) = 1.0f;
    }
    else
    {
        matrix.at<double>(0, 0) = 1.0;
        matrix.at<double>(1, 1) = 1.0;
        matrix.at<double>(2, 2) = 1.0;
    }
    return matrix;
}

}  // namespace

TEST(ImgprocPhase1GeometricSampling_TEST, remap_identity_and_alias_are_exact)
{
    Mat source({5, 7}, CV_8UC3);
    fill_u8(source);
    Mat map({5, 7}, CV_32FC2);
    for (int row = 0; row < 5; ++row)
    {
        for (int col = 0; col < 7; ++col)
        {
            map.at<float>(row, col, 0) = static_cast<float>(col);
            map.at<float>(row, col, 1) = static_cast<float>(row);
        }
    }
    Mat output;
    remap(
        source,
        output,
        map,
        Mat(),
        INTER_NEAREST,
        BORDER_CONSTANT);
    EXPECT_EQ(max_u8_difference(source, output), 0);

    remap(
        source,
        source,
        map,
        Mat(),
        INTER_NEAREST,
        BORDER_CONSTANT);
    EXPECT_EQ(max_u8_difference(source, output), 0);
}

TEST(ImgprocPhase1GeometricSampling_TEST, map_representations_produce_same_linear_result)
{
    Mat parent({9, 12}, CV_8UC4);
    fill_u8(parent);
    Mat source = parent(Range(1, 8), Range(2, 11));
    ASSERT_FALSE(source.isContinuous());
    Mat map_x;
    Mat map_y;
    make_maps(map_x, map_y, 7, 9);

    Mat interleaved;
    Mat unused;
    convertMaps(
        map_x,
        map_y,
        interleaved,
        unused,
        CV_32FC2);
    ASSERT_TRUE(unused.empty());

    Mat fixed_coordinates;
    Mat fixed_fractions;
    convertMaps(
        interleaved,
        Mat(),
        fixed_coordinates,
        fixed_fractions,
        CV_16SC2);
    EXPECT_EQ(fixed_coordinates.type(), CV_16SC2);
    EXPECT_EQ(fixed_fractions.type(), CV_16UC1);

    Mat from_pair;
    Mat from_interleaved;
    Mat from_fixed;
    remap(
        source,
        from_pair,
        map_x,
        map_y,
        INTER_LINEAR,
        BORDER_REFLECT_101);
    remap(
        source,
        from_interleaved,
        interleaved,
        Mat(),
        INTER_LINEAR,
        BORDER_REFLECT_101);
    remap(
        source,
        from_fixed,
        fixed_coordinates,
        fixed_fractions,
        INTER_LINEAR,
        BORDER_REFLECT_101);
    EXPECT_EQ(max_u8_difference(from_pair, from_interleaved), 0);
    EXPECT_EQ(max_u8_difference(from_pair, from_fixed), 0);

    Mat restored_x;
    Mat restored_y;
    convertMaps(
        fixed_coordinates,
        fixed_fractions,
        restored_x,
        restored_y,
        CV_32FC1);
    for (int row = 0; row < map_x.size[0]; ++row)
    {
        for (int col = 0; col < map_x.size[1]; ++col)
        {
            EXPECT_NEAR(
                restored_x.at<float>(row, col),
                map_x.at<float>(row, col),
                1.0f / INTER_TAB_SIZE);
            EXPECT_NEAR(
                restored_y.at<float>(row, col),
                map_y.at<float>(row, col),
                1.0f / INTER_TAB_SIZE);
        }
    }
}

TEST(ImgprocPhase1GeometricSampling_TEST, nearest_fixed_map_and_border_modes_are_defined)
{
    Mat source({2, 3}, CV_32FC1);
    for (int row = 0; row < 2; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            source.at<float>(row, col) =
                static_cast<float>(row * 10 + col);
        }
    }
    Mat map_x({1, 4}, CV_32FC1);
    Mat map_y({1, 4}, CV_32FC1);
    const float coordinates[] = {-1.0f, 0.49f, 1.51f, 4.0f};
    for (int col = 0; col < 4; ++col)
    {
        map_x.at<float>(0, col) = coordinates[col];
        map_y.at<float>(0, col) = 0.0f;
    }
    Mat fixed;
    Mat fractions;
    convertMaps(
        map_x,
        map_y,
        fixed,
        fractions,
        CV_16SC2,
        true);
    EXPECT_TRUE(fractions.empty());

    Mat output;
    remap(
        source,
        output,
        fixed,
        fractions,
        INTER_NEAREST,
        BORDER_CONSTANT,
        Scalar::all(99.0));
    EXPECT_FLOAT_EQ(output.at<float>(0, 0), 99.0f);
    EXPECT_FLOAT_EQ(output.at<float>(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(output.at<float>(0, 2), 2.0f);
    EXPECT_FLOAT_EQ(output.at<float>(0, 3), 99.0f);
}

TEST(ImgprocPhase1GeometricSampling_TEST, warp_perspective_identity_inverse_and_alias)
{
    Mat source({7, 9}, CV_8UC1);
    fill_u8(source);
    const Mat identity = identity_perspective(CV_64F);
    Mat output;
    warpPerspective(
        source,
        output,
        identity,
        Size(9, 7),
        INTER_NEAREST);
    EXPECT_EQ(max_u8_difference(source, output), 0);

    Mat inverse_map = identity.clone();
    inverse_map.at<double>(0, 2) = -1.0;
    Mat from_inverse;
    warpPerspective(
        source,
        from_inverse,
        inverse_map,
        Size(9, 7),
        INTER_NEAREST | WARP_INVERSE_MAP,
        BORDER_REPLICATE);
    Mat forward = identity.clone();
    forward.at<double>(0, 2) = 1.0;
    Mat from_forward;
    warpPerspective(
        source,
        from_forward,
        forward,
        Size(9, 7),
        INTER_NEAREST,
        BORDER_REPLICATE);
    EXPECT_EQ(max_u8_difference(from_inverse, from_forward), 0);

    warpPerspective(
        source,
        source,
        identity,
        Size(9, 7),
        INTER_LINEAR,
        BORDER_REFLECT);
    EXPECT_EQ(max_u8_difference(source, output), 0);
}

TEST(ImgprocPhase1GeometricSampling_TEST, warp_perspective_handles_true_projective_map)
{
    Mat source({8, 10}, CV_32FC3);
    for (int row = 0; row < source.size[0]; ++row)
    {
        for (int col = 0; col < source.size[1]; ++col)
        {
            for (int channel = 0; channel < 3; ++channel)
            {
                source.at<float>(row, col, channel) =
                    static_cast<float>(row * 10 + col + channel);
            }
        }
    }
    Mat inverse = identity_perspective(CV_32F);
    inverse.at<float>(0, 1) = 0.1f;
    inverse.at<float>(0, 2) = 0.25f;
    inverse.at<float>(1, 0) = -0.05f;
    inverse.at<float>(1, 2) = 0.5f;
    inverse.at<float>(2, 0) = 0.002f;
    inverse.at<float>(2, 1) = -0.003f;
    Mat output;
    warpPerspective(
        source,
        output,
        inverse,
        Size(7, 5),
        INTER_LINEAR | WARP_INVERSE_MAP,
        BORDER_REFLECT_101);
    EXPECT_EQ(output.type(), CV_32FC3);
    EXPECT_EQ(output.shape(), MatShape({5, 7}));
    EXPECT_TRUE(std::isfinite(output.at<float>(4, 6, 2)));
}

TEST(ImgprocPhase1GeometricSampling_TEST, get_rect_sub_pix_covers_depth_and_edge)
{
    Mat source({4, 5}, CV_8UC1);
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 5; ++col)
        {
            source.at<uchar>(row, col) =
                static_cast<uchar>(row * 20 + col * 4);
        }
    }
    Mat center_patch;
    getRectSubPix(
        source,
        Size(1, 1),
        Point2f(1.5f, 2.5f),
        center_patch,
        CV_32F);
    EXPECT_EQ(center_patch.type(), CV_32FC1);
    EXPECT_FLOAT_EQ(center_patch.at<float>(0, 0), 56.0f);

    Mat edge_patch;
    getRectSubPix(
        source,
        Size(3, 3),
        Point2f(0.0f, 0.0f),
        edge_patch);
    EXPECT_EQ(edge_patch.type(), CV_8UC1);
    EXPECT_EQ(edge_patch.at<uchar>(0, 0), source.at<uchar>(0, 0));
    EXPECT_EQ(edge_patch.at<uchar>(2, 2), source.at<uchar>(1, 1));
}

TEST(ImgprocPhase1GeometricSampling_TEST, invalid_inputs_are_rejected)
{
    Mat source({4, 5}, CV_8UC1);
    Mat output;
    Mat bad_map({4, 5}, CV_8UC2);
    EXPECT_THROW(
        remap(
            source,
            output,
            bad_map,
            Mat(),
            INTER_LINEAR),
        Exception);

    Mat map_x({4, 5}, CV_32FC1);
    Mat map_y({3, 5}, CV_32FC1);
    EXPECT_THROW(
        convertMaps(
            map_x,
            map_y,
            output,
            bad_map,
            CV_16SC2),
        Exception);

    Mat singular({3, 3}, CV_64FC1);
    singular.setTo(Scalar::all(0.0));
    EXPECT_THROW(
        warpPerspective(
            source,
            output,
            singular,
            Size(5, 4)),
        Exception);
    EXPECT_THROW(
        getRectSubPix(
            source,
            Size(3, 3),
            Point2f(-0.1f, 0.0f),
            output),
        Exception);
}
