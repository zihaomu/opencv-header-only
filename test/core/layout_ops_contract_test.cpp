#include "cvh.h"
#include "gtest/gtest.h"

#include <cstring>
#include <vector>

using namespace cvh;

namespace
{

void fill_incrementing_bytes(Mat& mat, unsigned char start = 0)
{
    ASSERT_TRUE(mat.isContinuous());
    const size_t bytes = mat.total() * mat.elemSize();
    for (size_t i = 0; i < bytes; ++i)
    {
        mat.data[i] = static_cast<unsigned char>(start + i);
    }
}

void expect_same_bytes(const Mat& expected, const Mat& actual)
{
    ASSERT_EQ(actual.shape(), expected.shape());
    ASSERT_EQ(actual.type(), expected.type());
    for (size_t i = 0; i < expected.total(); ++i)
    {
        EXPECT_EQ(
            std::memcmp(
                array_detail::pixel_at(expected, i),
                array_detail::pixel_at(actual, i),
                expected.elemSize()),
            0);
    }
}

}  // namespace

TEST(LayoutOpsContract_TEST, border_interpolate_matches_documented_modes)
{
    EXPECT_EQ(borderInterpolate(-3, 5, BORDER_REPLICATE), 0);
    EXPECT_EQ(borderInterpolate(8, 5, BORDER_REPLICATE), 4);
    EXPECT_EQ(borderInterpolate(-1, 5, BORDER_REFLECT), 0);
    EXPECT_EQ(borderInterpolate(5, 5, BORDER_REFLECT), 4);
    EXPECT_EQ(borderInterpolate(-1, 5, BORDER_REFLECT_101), 1);
    EXPECT_EQ(borderInterpolate(5, 5, BORDER_REFLECT_101), 3);
    EXPECT_EQ(borderInterpolate(-7, 5, BORDER_WRAP), 3);
    EXPECT_EQ(borderInterpolate(7, 5, BORDER_WRAP), 2);
    EXPECT_EQ(borderInterpolate(2, 5, BORDER_CONSTANT), 2);
    EXPECT_EQ(borderInterpolate(-1, 5, BORDER_CONSTANT), -1);
    EXPECT_EQ(borderInterpolate(9, 1, BORDER_REFLECT_101), 0);
    EXPECT_EQ(
        borderInterpolate(-1, 5, BORDER_REFLECT_101 | BORDER_ISOLATED),
        1);
    EXPECT_THROW(borderInterpolate(0, 0, BORDER_REPLICATE), Exception);
    EXPECT_THROW(borderInterpolate(-1, 5, BORDER_TRANSPARENT), Exception);
}

TEST(LayoutOpsContract_TEST, copy_to_mask_roi_and_overlap_are_defined)
{
    Mat parent({4, 6}, CV_16SC1);
    for (int y = 0; y < 4; ++y)
    {
        for (int x = 0; x < 6; ++x)
        {
            parent.at<short>(y, x) = static_cast<short>(10 * y + x);
        }
    }
    Mat src = parent(Range(1, 4), Range(1, 5));
    ASSERT_FALSE(src.isContinuous());

    Mat copied;
    copyTo(src, copied);
    ASSERT_EQ(copied.shape(), MatShape({3, 4}));
    EXPECT_EQ(copied.at<short>(2, 3), 34);

    Mat mask({3, 4}, CV_8UC1);
    mask.setTo(Scalar::all(0));
    mask.at<uchar>(0, 1) = 255;
    mask.at<uchar>(2, 3) = 255;
    Mat masked;
    copyTo(src, masked, mask);
    EXPECT_EQ(masked.at<short>(0, 0), 0);
    EXPECT_EQ(masked.at<short>(0, 1), 12);
    EXPECT_EQ(masked.at<short>(2, 3), 34);

    masked.setTo(Scalar::all(-5));
    copyTo(src, masked, mask);
    EXPECT_EQ(masked.at<short>(0, 0), -5);
    EXPECT_EQ(masked.at<short>(0, 1), 12);

    mask.setTo(Scalar::all(0));
    Mat all_zero;
    copyTo(src, all_zero, mask);
    EXPECT_EQ(countNonZero(all_zero), 0);
    mask.setTo(Scalar::all(255));
    Mat all_one;
    copyTo(src, all_one, mask);
    expect_same_bytes(src, all_one);

    Mat overlapping({1, 6}, CV_8UC1);
    for (int x = 0; x < 6; ++x)
    {
        overlapping.at<uchar>(0, x) = static_cast<uchar>(x);
    }
    Mat left = overlapping.colRange(0, 5);
    Mat right = overlapping.colRange(1, 6);
    copyTo(left, right);
    const uchar expected[] = {0, 0, 1, 2, 3, 4};
    EXPECT_EQ(std::memcmp(overlapping.data, expected, sizeof(expected)), 0);
}

TEST(LayoutOpsContract_TEST, copy_class_preserves_bytes_across_depths_and_channels)
{
    const int types[] = {
        CV_8UC4,
        CV_8SC3,
        CV_16UC2,
        CV_16SC4,
        CV_32SC3,
        CV_32UC1,
        CV_16FC4,
        CV_32FC3,
        CV_64FC4,
    };
    for (int type : types)
    {
        SCOPED_TRACE(type);
        Mat src({2, 3}, type);
        fill_incrementing_bytes(src, 17);
        Mat copied;
        copyTo(src, copied);
        expect_same_bytes(src, copied);

        Mat flipped;
        flip(src, flipped, 1);
        Mat restored;
        flip(flipped, restored, 1);
        expect_same_bytes(src, restored);
    }
}

TEST(LayoutOpsContract_TEST, channel_extract_insert_and_mix_route_raw_scalars)
{
    Mat bgra({1, 2}, CV_8UC4);
    const uchar values[] = {1, 2, 3, 4, 5, 6, 7, 8};
    std::memcpy(bgra.data, values, sizeof(values));

    Mat red;
    extractChannel(bgra, red, 2);
    EXPECT_EQ(red.type(), CV_8UC1);
    EXPECT_EQ(red.at<uchar>(0, 0), 3);
    EXPECT_EQ(red.at<uchar>(0, 1), 7);

    Mat replaced = bgra.clone();
    Mat alpha({1, 2}, CV_8UC1);
    alpha.at<uchar>(0, 0) = 40;
    alpha.at<uchar>(0, 1) = 80;
    insertChannel(alpha, replaced, 3);
    EXPECT_EQ(replaced.at<uchar>(0, 0, 3), 40);
    EXPECT_EQ(replaced.at<uchar>(0, 1, 3), 80);

    Mat c2({1, 2}, CV_16SC2);
    c2.at<short>(0, 0, 0) = 11;
    c2.at<short>(0, 0, 1) = 12;
    c2.at<short>(0, 1, 0) = 21;
    c2.at<short>(0, 1, 1) = 22;
    Mat second_channel;
    extractChannel(c2, second_channel, 1);
    EXPECT_EQ(second_channel.at<short>(0, 0), 12);
    EXPECT_EQ(second_channel.at<short>(0, 1), 22);

    Mat bgr({1, 2}, CV_8UC3);
    Mat output_alpha({1, 2}, CV_8UC1);
    Mat outputs[] = {bgr, output_alpha};
    const int routes[] = {0, 2, 1, 1, 2, 0, 3, 3};
    mixChannels(&bgra, 1, outputs, 2, routes, 4);
    EXPECT_EQ(outputs[0].at<uchar>(0, 0, 0), 3);
    EXPECT_EQ(outputs[0].at<uchar>(0, 0, 1), 2);
    EXPECT_EQ(outputs[0].at<uchar>(0, 0, 2), 1);
    EXPECT_EQ(outputs[1].at<uchar>(0, 1), 8);

    Mat zeroed({1, 2}, CV_8UC1);
    zeroed.setTo(Scalar::all(9));
    const int zero_route[] = {-1, 0};
    mixChannels(&bgra, 1, &zeroed, 1, zero_route, 1);
    EXPECT_EQ(zeroed.at<uchar>(0, 0), 0);
    EXPECT_EQ(zeroed.at<uchar>(0, 1), 0);

    Mat in_place({1, 1}, CV_8UC3);
    in_place.at<uchar>(0, 0, 0) = 10;
    in_place.at<uchar>(0, 0, 1) = 20;
    in_place.at<uchar>(0, 0, 2) = 30;
    const int swap_red_blue[] = {0, 2, 2, 0};
    mixChannels(
        &in_place, 1, &in_place, 1, swap_red_blue, 2);
    EXPECT_EQ(in_place.at<uchar>(0, 0, 0), 30);
    EXPECT_EQ(in_place.at<uchar>(0, 0, 1), 20);
    EXPECT_EQ(in_place.at<uchar>(0, 0, 2), 10);
}

TEST(LayoutOpsContract_TEST, flip_rotate_repeat_and_alias_match_geometry)
{
    Mat src({2, 3}, CV_32SC1);
    for (int i = 0; i < 6; ++i)
    {
        reinterpret_cast<int*>(src.data)[i] = i + 1;
    }

    Mat horizontal;
    flip(src, horizontal, 1);
    EXPECT_EQ(horizontal.at<int>(0, 0), 3);
    EXPECT_EQ(horizontal.at<int>(1, 2), 4);
    Mat vertical;
    flip(src, vertical, 0);
    EXPECT_EQ(vertical.at<int>(0, 0), 4);
    Mat both;
    flip(src, both, -1);
    EXPECT_EQ(both.at<int>(0, 0), 6);

    Mat clockwise;
    rotate(src, clockwise, ROTATE_90_CLOCKWISE);
    ASSERT_EQ(clockwise.shape(), MatShape({3, 2}));
    EXPECT_EQ(clockwise.at<int>(0, 0), 4);
    EXPECT_EQ(clockwise.at<int>(0, 1), 1);

    Mat counterclockwise;
    rotate(src, counterclockwise, ROTATE_90_COUNTERCLOCKWISE);
    EXPECT_EQ(counterclockwise.at<int>(0, 0), 3);
    EXPECT_EQ(counterclockwise.at<int>(2, 1), 4);

    Mat alias = src.clone();
    rotate(alias, alias, ROTATE_180);
    EXPECT_EQ(alias.at<int>(0, 0), 6);
    EXPECT_EQ(alias.at<int>(1, 2), 1);

    Mat tiled;
    repeat(src, 2, 3, tiled);
    ASSERT_EQ(tiled.shape(), MatShape({4, 9}));
    EXPECT_EQ(tiled.at<int>(0, 0), 1);
    EXPECT_EQ(tiled.at<int>(2, 4), 2);
    EXPECT_EQ(tiled.at<int>(3, 8), 6);
}

TEST(LayoutOpsContract_TEST, flip_nd_covers_1d_2d_3d_and_negative_axis)
{
    Mat one_dimensional({4}, CV_8UC1);
    for (int i = 0; i < 4; ++i)
    {
        one_dimensional.at<uchar>(i) = static_cast<uchar>(i + 1);
    }
    Mat one_flipped;
    flipND(one_dimensional, one_flipped, 0);
    EXPECT_EQ(one_flipped.at<uchar>(0), 4);
    EXPECT_EQ(one_flipped.at<uchar>(3), 1);

    Mat three_dimensional({2, 2, 3}, CV_16SC1);
    for (int i = 0; i < 12; ++i)
    {
        reinterpret_cast<short*>(three_dimensional.data)[i] =
            static_cast<short>(i);
    }
    Mat last_axis;
    flipND(three_dimensional, last_axis, -1);
    const short expected_last[] = {2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9};
    EXPECT_EQ(
        std::memcmp(last_axis.data, expected_last, sizeof(expected_last)),
        0);

    Mat first_axis;
    flipND(three_dimensional, first_axis, 0);
    EXPECT_EQ(reinterpret_cast<short*>(first_axis.data)[0], 6);
    EXPECT_EQ(reinterpret_cast<short*>(first_axis.data)[11], 5);

    Mat two_dimensional = three_dimensional.reshape({4, 3});
    Mat two_dimensional_flipped;
    flipND(two_dimensional, two_dimensional_flipped, 1);
    EXPECT_EQ(two_dimensional_flipped.at<short>(0, 0), 2);
    EXPECT_EQ(two_dimensional_flipped.at<short>(3, 2), 9);
}

TEST(LayoutOpsContract_TEST, concat_validates_shapes_and_supports_output_alias)
{
    Mat left({2, 2}, CV_8UC1);
    Mat right({2, 1}, CV_8UC1);
    left.setTo(Scalar::all(1));
    right.setTo(Scalar::all(2));
    hconcat(left, right, left);
    ASSERT_EQ(left.shape(), MatShape({2, 3}));
    EXPECT_EQ(left.at<uchar>(0, 0), 1);
    EXPECT_EQ(left.at<uchar>(0, 2), 2);

    Mat top({1, 3}, CV_8UC1);
    top.setTo(Scalar::all(3));
    Mat vertical;
    vconcat(left, top, vertical);
    ASSERT_EQ(vertical.shape(), MatShape({3, 3}));
    EXPECT_EQ(vertical.at<uchar>(2, 1), 3);

    Mat wrong_rows({1, 2}, CV_8UC1);
    EXPECT_THROW(hconcat(left, wrong_rows, vertical), Exception);
    Mat wrong_cols({2, 2}, CV_8UC1);
    EXPECT_THROW(vconcat(left, wrong_cols, vertical), Exception);
}

TEST(LayoutOpsContract_TEST, broadcast_uses_trailing_dimension_rules)
{
    Mat src({2, 1, 3}, CV_32SC1);
    for (int i = 0; i < 6; ++i)
    {
        reinterpret_cast<int*>(src.data)[i] = i + 1;
    }
    Mat dst;
    broadcast(src, std::vector<int>({4, 2, 5, 3}), dst);
    ASSERT_EQ(dst.shape(), MatShape({4, 2, 5, 3}));
    const int* values = reinterpret_cast<const int*>(dst.data);
    EXPECT_EQ(values[0], 1);
    EXPECT_EQ(values[3], 1);
    EXPECT_EQ(values[14], 3);
    EXPECT_EQ(values[15], 4);
    EXPECT_EQ(values[119], 6);

    Mat shape({1, 3}, CV_32SC1);
    shape.at<int>(0, 0) = 2;
    shape.at<int>(0, 1) = 2;
    shape.at<int>(0, 2) = 3;
    Mat matrix_shape_output;
    broadcast(src, shape, matrix_shape_output);
    ASSERT_EQ(matrix_shape_output.shape(), MatShape({2, 2, 3}));
    EXPECT_EQ(reinterpret_cast<int*>(matrix_shape_output.data)[3], 1);

    Mat c3({1, 1}, CV_8UC3);
    c3.at<uchar>(0, 0, 0) = 7;
    c3.at<uchar>(0, 0, 1) = 8;
    c3.at<uchar>(0, 0, 2) = 9;
    broadcast(c3, std::vector<int>({2, 3}), dst);
    EXPECT_EQ(dst.type(), CV_8UC3);
    EXPECT_EQ(dst.at<uchar>(1, 2, 2), 9);

    EXPECT_THROW(
        broadcast(src, std::vector<int>({3, 4, 3}), dst),
        Exception);
}

TEST(LayoutOpsContract_TEST, swap_exchanges_headers_without_copying_pixels)
{
    Mat a({2, 3}, CV_8UC1);
    Mat b({4, 1}, CV_32FC2);
    a.setTo(Scalar::all(7));
    b.setTo(Scalar(2.0, 3.0));
    uchar* a_data = a.data;
    uchar* b_data = b.data;
    Mat a_alias = a;
    Mat b_alias = b;
    MatData* a_storage = a.u;
    MatData* b_storage = b.u;
    const int a_refcount = a_storage->refcount;
    const int b_refcount = b_storage->refcount;

    swap(a, b);
    EXPECT_EQ(a.data, b_data);
    EXPECT_EQ(b.data, a_data);
    EXPECT_EQ(a.shape(), MatShape({4, 1}));
    EXPECT_EQ(b.shape(), MatShape({2, 3}));
    EXPECT_EQ(a.type(), CV_32FC2);
    EXPECT_EQ(b.type(), CV_8UC1);
    EXPECT_EQ(a.u, b_storage);
    EXPECT_EQ(b.u, a_storage);
    EXPECT_EQ(a_storage->refcount, a_refcount);
    EXPECT_EQ(b_storage->refcount, b_refcount);
    EXPECT_EQ(a_alias.data, a_data);
    EXPECT_EQ(b_alias.data, b_data);
    EXPECT_FLOAT_EQ(a.at<float>(0, 0, 0), 2.0f);
    EXPECT_EQ(b.at<uchar>(0, 0), 7);
}

TEST(LayoutOpsContract_TEST, invalid_channel_layout_and_geometry_inputs_throw)
{
    Mat c3({2, 3}, CV_8UC3);
    Mat c1({2, 3}, CV_8UC1);
    Mat other_shape({2, 4}, CV_8UC1);
    Mat dst;

    EXPECT_THROW(extractChannel(c3, dst, 3), Exception);
    EXPECT_THROW(insertChannel(c3, c3, 0), Exception);
    EXPECT_THROW(insertChannel(other_shape, c3, 0), Exception);
    EXPECT_THROW(flipND(c1, dst, 2), Exception);
    EXPECT_THROW(rotate(c1, dst, 99), Exception);
    EXPECT_THROW(repeat(c1, 0, 1, dst), Exception);

    Mat unallocated;
    const int route[] = {0, 0};
    EXPECT_THROW(
        mixChannels(&c3, 1, &unallocated, 1, route, 1),
        Exception);
}
