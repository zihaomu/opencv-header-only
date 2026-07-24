#include "cvh.h"
#include "gtest/gtest.h"

#include <cmath>
#include <limits>
#include <vector>

using namespace cvh;

TEST(ReductionOpsContract_TEST, norm_values_mask_and_two_input_forms)
{
    EXPECT_EQ(NORM_INF, 1);
    EXPECT_EQ(NORM_L1, 2);
    EXPECT_EQ(NORM_L2, 4);
    EXPECT_EQ(NORM_MINMAX, 32);

    Mat src({1, 3}, CV_32FC2);
    const float values[] = {3.0f, 4.0f, -5.0f, 0.0f, 0.0f, 12.0f};
    for (int x = 0; x < 3; ++x)
    {
        src.at<float>(0, x, 0) = values[2 * x];
        src.at<float>(0, x, 1) = values[2 * x + 1];
    }

    EXPECT_DOUBLE_EQ(norm(src, NORM_INF), 12.0);
    EXPECT_DOUBLE_EQ(norm(src, NORM_L1), 24.0);
    EXPECT_DOUBLE_EQ(norm(src, NORM_L2), std::sqrt(194.0));

    Mat mask({1, 3}, CV_8UC1);
    mask.at<uchar>(0, 0) = 255;
    mask.at<uchar>(0, 1) = 0;
    mask.at<uchar>(0, 2) = 255;
    EXPECT_DOUBLE_EQ(norm(src, NORM_L1, mask), 19.0);

    Mat zeros(src.shape(), src.type());
    zeros.setTo(Scalar::all(0.0));
    EXPECT_DOUBLE_EQ(norm(src, zeros, NORM_L2), std::sqrt(194.0));
}

TEST(ReductionOpsContract_TEST, sum_mean_stddev_cover_c3_mask_and_roi)
{
    Mat parent({2, 6}, CV_32FC3);
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 6; ++x)
        {
            parent.at<float>(y, x, 0) = static_cast<float>(y * 10 + x);
            parent.at<float>(y, x, 1) = static_cast<float>(2 * (y * 10 + x));
            parent.at<float>(y, x, 2) = 5.0f;
        }
    }
    Mat src = parent.colRange(1, 5);
    ASSERT_FALSE(src.isContinuous());

    const Scalar sums = sum(src);
    EXPECT_DOUBLE_EQ(sums[0], 60.0);
    EXPECT_DOUBLE_EQ(sums[1], 120.0);
    EXPECT_DOUBLE_EQ(sums[2], 40.0);

    Mat mask({2, 4}, CV_8UC1);
    mask.setTo(Scalar::all(0.0));
    mask.at<uchar>(0, 0) = 255;
    mask.at<uchar>(0, 2) = 255;
    mask.at<uchar>(1, 1) = 255;
    mask.at<uchar>(1, 3) = 255;

    const Scalar means = mean(src, mask);
    EXPECT_DOUBLE_EQ(means[0], 7.5);
    EXPECT_DOUBLE_EQ(means[1], 15.0);
    EXPECT_DOUBLE_EQ(means[2], 5.0);

    Scalar mean_value;
    Scalar stddev_value;
    meanStdDev(src, mean_value, stddev_value, mask);
    EXPECT_EQ(mean_value, means);
    EXPECT_NEAR(stddev_value[0], std::sqrt(31.25), 1e-12);
    EXPECT_NEAR(stddev_value[1], std::sqrt(125.0), 1e-12);
    EXPECT_DOUBLE_EQ(stddev_value[2], 0.0);

    mask.setTo(Scalar::all(0.0));
    EXPECT_EQ(mean(src, mask), Scalar());
    meanStdDev(src, mean_value, stddev_value, mask);
    EXPECT_EQ(mean_value, Scalar());
    EXPECT_EQ(stddev_value, Scalar());
}

TEST(ReductionOpsContract_TEST, statistics_cover_c4_single_and_identical_values)
{
    Mat c4({1, 2}, CV_64FC4);
    for (int ch = 0; ch < 4; ++ch)
    {
        c4.at<double>(0, 0, ch) = static_cast<double>(ch + 1);
        c4.at<double>(0, 1, ch) = static_cast<double>(2 * (ch + 1));
    }
    const Scalar sums = sum(c4);
    EXPECT_DOUBLE_EQ(sums[0], 3.0);
    EXPECT_DOUBLE_EQ(sums[1], 6.0);
    EXPECT_DOUBLE_EQ(sums[2], 9.0);
    EXPECT_DOUBLE_EQ(sums[3], 12.0);

    Mat identical({3, 5}, CV_64FC1);
    identical.setTo(Scalar::all(1.0e12));
    Scalar mean_value;
    Scalar stddev_value;
    meanStdDev(identical, mean_value, stddev_value);
    EXPECT_DOUBLE_EQ(mean_value[0], 1.0e12);
    EXPECT_DOUBLE_EQ(stddev_value[0], 0.0);

    Mat single({1, 1}, CV_32FC1);
    single.at<float>(0, 0) = -7.0f;
    meanStdDev(single, mean_value, stddev_value);
    EXPECT_DOUBLE_EQ(mean_value[0], -7.0);
    EXPECT_DOUBLE_EQ(stddev_value[0], 0.0);
}

TEST(ReductionOpsContract_TEST, nonzero_predicates_and_locations_are_row_major)
{
    Mat src({3, 4}, CV_16SC1);
    src.setTo(Scalar::all(0.0));
    src.at<short>(0, 3) = 2;
    src.at<short>(1, 1) = -4;
    src.at<short>(2, 0) = 7;

    EXPECT_TRUE(hasNonZero(src));
    EXPECT_EQ(countNonZero(src), 3);

    std::vector<Point> points;
    findNonZero(src, points);
    ASSERT_EQ(points.size(), 3u);
    EXPECT_EQ(points[0], Point(3, 0));
    EXPECT_EQ(points[1], Point(1, 1));
    EXPECT_EQ(points[2], Point(0, 2));

    Mat point_mat;
    findNonZero(src, point_mat);
    ASSERT_EQ(point_mat.type(), CV_32SC2);
    ASSERT_EQ(point_mat.shape(), MatShape({3, 1}));
    EXPECT_EQ(point_mat.at<int>(1, 0, 0), 1);
    EXPECT_EQ(point_mat.at<int>(1, 0, 1), 1);

    src.setTo(Scalar::all(0.0));
    EXPECT_FALSE(hasNonZero(src));
    EXPECT_EQ(countNonZero(src), 0);
    findNonZero(src, point_mat);
    EXPECT_TRUE(point_mat.empty());
}

TEST(ReductionOpsContract_TEST, nonzero_apis_reject_multichannel_input)
{
    Mat src({1, 3}, CV_8UC3);
    Mat out;
    std::vector<Point> points;
    EXPECT_THROW(countNonZero(src), Exception);
    EXPECT_THROW(hasNonZero(src), Exception);
    EXPECT_THROW(findNonZero(src, points), Exception);
    EXPECT_THROW(findNonZero(src, out), Exception);
}

TEST(ReductionOpsContract_TEST, minmax_ties_use_first_row_major_location)
{
    Mat src({2, 4}, CV_32SC1);
    const int values[] = {5, -2, 9, -2, 9, 3, 9, 4};
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 4; ++x)
        {
            src.at<int>(y, x) = values[y * 4 + x];
        }
    }

    double min_value = 0.0;
    double max_value = 0.0;
    Point min_location;
    Point max_location;
    minMaxLoc(src, &min_value, &max_value, &min_location, &max_location);
    EXPECT_DOUBLE_EQ(min_value, -2.0);
    EXPECT_DOUBLE_EQ(max_value, 9.0);
    EXPECT_EQ(min_location, Point(1, 0));
    EXPECT_EQ(max_location, Point(2, 0));

    Mat mask({2, 4}, CV_8UC1);
    mask.setTo(Scalar::all(0.0));
    mask.at<uchar>(0, 3) = 255;
    mask.at<uchar>(1, 1) = 255;
    minMaxLoc(src, &min_value, &max_value, &min_location, &max_location, mask);
    EXPECT_DOUBLE_EQ(min_value, -2.0);
    EXPECT_DOUBLE_EQ(max_value, 3.0);
    EXPECT_EQ(min_location, Point(3, 0));
    EXPECT_EQ(max_location, Point(1, 1));
}

TEST(ReductionOpsContract_TEST, minmaxidx_reports_nd_coordinates)
{
    Mat src({2, 3, 4}, CV_64FC1);
    double* values = reinterpret_cast<double*>(src.data);
    for (int i = 0; i < 24; ++i)
    {
        values[i] = static_cast<double>(i);
    }
    values[17] = -10.0;
    values[22] = 100.0;

    double min_value = 0.0;
    double max_value = 0.0;
    int min_index[3] = {-1, -1, -1};
    int max_index[3] = {-1, -1, -1};
    minMaxIdx(src, &min_value, &max_value, min_index, max_index);
    EXPECT_DOUBLE_EQ(min_value, -10.0);
    EXPECT_DOUBLE_EQ(max_value, 100.0);
    EXPECT_EQ(min_index[0], 1);
    EXPECT_EQ(min_index[1], 1);
    EXPECT_EQ(min_index[2], 1);
    EXPECT_EQ(max_index[0], 1);
    EXPECT_EQ(max_index[1], 2);
    EXPECT_EQ(max_index[2], 2);
}

TEST(ReductionOpsContract_TEST, reduce_axis_shape_type_and_values_match_contract)
{
    Mat src({2, 3}, CV_16SC2);
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 3; ++x)
        {
            src.at<short>(y, x, 0) = static_cast<short>(10 * y + x + 1);
            src.at<short>(y, x, 1) = static_cast<short>(100 + 10 * y + x);
        }
    }

    Mat dst;
    reduce(src, dst, 0, REDUCE_SUM, CV_64F);
    ASSERT_EQ(dst.shape(), MatShape({1, 3}));
    ASSERT_EQ(dst.type(), CV_64FC2);
    EXPECT_DOUBLE_EQ(dst.at<double>(0, 0, 0), 12.0);
    EXPECT_DOUBLE_EQ(dst.at<double>(0, 2, 1), 214.0);

    reduce(src, dst, 1, REDUCE_AVG, CV_32F);
    ASSERT_EQ(dst.shape(), MatShape({2, 1}));
    ASSERT_EQ(dst.type(), CV_32FC2);
    EXPECT_FLOAT_EQ(dst.at<float>(0, 0, 0), 2.0f);
    EXPECT_FLOAT_EQ(dst.at<float>(1, 0, 1), 111.0f);

    reduce(src, dst, 0, REDUCE_MAX);
    ASSERT_EQ(dst.type(), CV_16SC2);
    EXPECT_EQ(dst.at<short>(0, 1, 0), 12);

    reduce(src, dst, 1, REDUCE_SUM2, CV_64F);
    EXPECT_DOUBLE_EQ(dst.at<double>(0, 0, 0), 1.0 + 4.0 + 9.0);

    Mat alias = src.clone();
    reduce(alias, alias, 0, REDUCE_SUM, CV_64F);
    ASSERT_EQ(alias.shape(), MatShape({1, 3}));
    EXPECT_DOUBLE_EQ(alias.at<double>(0, 0, 0), 12.0);
}

TEST(ReductionOpsContract_TEST, reduce_arg_ties_support_first_and_last_index)
{
    Mat src({3, 4}, CV_32FC1);
    const float values[] = {
        1.0f, 9.0f, 3.0f, 9.0f,
        1.0f, 5.0f, 3.0f, 9.0f,
        2.0f, 5.0f, 0.0f, 8.0f,
    };
    for (int y = 0; y < 3; ++y)
    {
        for (int x = 0; x < 4; ++x)
        {
            src.at<float>(y, x) = values[y * 4 + x];
        }
    }

    Mat indices;
    reduceArgMin(src, indices, 0, false);
    EXPECT_EQ(indices.at<int>(0, 0), 0);
    EXPECT_EQ(indices.at<int>(0, 1), 1);
    EXPECT_EQ(indices.at<int>(0, 2), 2);

    reduceArgMin(src, indices, 0, true);
    EXPECT_EQ(indices.at<int>(0, 0), 1);
    EXPECT_EQ(indices.at<int>(0, 1), 2);

    reduceArgMax(src, indices, 1, false);
    EXPECT_EQ(indices.at<int>(0, 0), 1);
    EXPECT_EQ(indices.at<int>(1, 0), 3);

    reduceArgMax(src, indices, 1, true);
    EXPECT_EQ(indices.at<int>(0, 0), 3);
    EXPECT_EQ(indices.at<int>(1, 0), 3);
}

TEST(ReductionOpsContract_TEST, normalize_supports_norms_minmax_dtype_mask_and_alias)
{
    Mat src({1, 4}, CV_32FC1);
    src.at<float>(0, 0) = 1.0f;
    src.at<float>(0, 1) = 2.0f;
    src.at<float>(0, 2) = 3.0f;
    src.at<float>(0, 3) = 4.0f;

    Mat dst;
    normalize(src, dst, 1.0, 0.0, NORM_L1);
    EXPECT_NEAR(norm(dst, NORM_L1), 1.0, 1e-6);
    normalize(src, dst, 2.0, 0.0, NORM_L2);
    EXPECT_NEAR(norm(dst, NORM_L2), 2.0, 1e-6);
    normalize(src, dst, 5.0, 0.0, NORM_INF);
    EXPECT_NEAR(norm(dst, NORM_INF), 5.0, 1e-6);

    normalize(src, dst, 10.0, 20.0, NORM_MINMAX, CV_64F);
    ASSERT_EQ(dst.type(), CV_64FC1);
    EXPECT_DOUBLE_EQ(dst.at<double>(0, 0), 10.0);
    EXPECT_DOUBLE_EQ(dst.at<double>(0, 3), 20.0);

    Mat mask({1, 4}, CV_8UC1);
    mask.at<uchar>(0, 0) = 0;
    mask.at<uchar>(0, 1) = 255;
    mask.at<uchar>(0, 2) = 255;
    mask.at<uchar>(0, 3) = 0;
    dst = Mat();
    normalize(src, dst, 1.0, 0.0, NORM_L1, -1, mask);
    EXPECT_FLOAT_EQ(dst.at<float>(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(dst.at<float>(0, 1), 0.4f);
    EXPECT_FLOAT_EQ(dst.at<float>(0, 2), 0.6f);
    EXPECT_FLOAT_EQ(dst.at<float>(0, 3), 0.0f);

    normalize(src, src, 0.0, 1.0, NORM_MINMAX);
    EXPECT_FLOAT_EQ(src.at<float>(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(src.at<float>(0, 3), 1.0f);
}

TEST(ReductionOpsContract_TEST, empty_zero_nan_inf_and_thread_settings_are_stable)
{
    Mat empty;
    EXPECT_EQ(sum(empty), Scalar());
    EXPECT_EQ(mean(empty), Scalar());
    EXPECT_DOUBLE_EQ(norm(empty), 0.0);
    EXPECT_EQ(countNonZero(empty), 0);
    EXPECT_FALSE(hasNonZero(empty));
    std::vector<Point> empty_points = {Point(1, 2)};
    findNonZero(empty, empty_points);
    EXPECT_TRUE(empty_points.empty());
    Mat empty_indices({1, 1}, CV_32SC2);
    findNonZero(empty, empty_indices);
    EXPECT_TRUE(empty_indices.empty());

    double empty_min = -1.0;
    double empty_max = -1.0;
    Point empty_min_location;
    Point empty_max_location;
    minMaxLoc(
        empty,
        &empty_min,
        &empty_max,
        &empty_min_location,
        &empty_max_location);
    EXPECT_DOUBLE_EQ(empty_min, 0.0);
    EXPECT_DOUBLE_EQ(empty_max, 0.0);
    EXPECT_EQ(empty_min_location, Point(-1, -1));
    EXPECT_EQ(empty_max_location, Point(-1, -1));

    Mat zero({20, 30}, CV_32FC1);
    zero.setTo(Scalar::all(0.0));
    EXPECT_DOUBLE_EQ(norm(zero, NORM_L2), 0.0);
    Mat normalized;
    normalize(zero, normalized, 1.0, 0.0, NORM_L2);
    EXPECT_DOUBLE_EQ(norm(normalized, NORM_L2), 0.0);

    Mat special({1, 3}, CV_64FC1);
    special.at<double>(0, 0) = 1.0;
    special.at<double>(0, 1) = std::numeric_limits<double>::infinity();
    special.at<double>(0, 2) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(sum(special)[0]));
    EXPECT_TRUE(std::isnan(norm(special, NORM_L2)));

    Mat large({200, 300}, CV_32FC1);
    for (size_t i = 0; i < large.total(); ++i)
    {
        reinterpret_cast<float*>(large.data)[i] =
            static_cast<float>(static_cast<int>(i % 37) - 18) * 0.25f;
    }
    const int previous_threads = getNumThreads();
    setNumThreads(1);
    const double single_thread = norm(large, NORM_L2);
    setNumThreads(4);
    const double configured_multi = norm(large, NORM_L2);
    setNumThreads(previous_threads);
    EXPECT_DOUBLE_EQ(single_thread, configured_multi);
}

TEST(ReductionOpsContract_TEST, invalid_axes_types_and_masks_throw)
{
    Mat c3({2, 3}, CV_8UC3);
    Mat c1({2, 3}, CV_8UC1);
    Mat bad_mask({2, 3}, CV_8UC3);
    Mat out;

    EXPECT_THROW(mean(c3, bad_mask), Exception);
    EXPECT_THROW(reduce(c1, out, 2, REDUCE_SUM), Exception);
    EXPECT_THROW(reduce(c1, out, 0, 99), Exception);
    EXPECT_THROW(reduceArgMin(c3, out, 0), Exception);
    EXPECT_THROW(normalize(c1, out, 1.0, 0.0, 999), Exception);
}
