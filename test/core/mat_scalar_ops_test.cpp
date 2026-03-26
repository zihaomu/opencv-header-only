#include "cvh.h"
#include "gtest/gtest.h"

using namespace cvh;

TEST(MatScalarOps_TEST, setto_scalar_fills_all_channels_per_pixel)
{
    Mat m({2, 2}, CV_8UC3);
    m.setTo(Scalar(1.0, 2.0, 3.0));

    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 2; ++x)
        {
            EXPECT_EQ(m.at<uchar>(y, x, 0), static_cast<uchar>(1));
            EXPECT_EQ(m.at<uchar>(y, x, 1), static_cast<uchar>(2));
            EXPECT_EQ(m.at<uchar>(y, x, 2), static_cast<uchar>(3));
        }
    }
}

TEST(MatScalarOps_TEST, operator_assign_scalar_matches_setto_behavior)
{
    Mat m({1, 3}, CV_32SC2);
    m = Scalar(10.0, -7.0);

    EXPECT_EQ(m.at<int>(0, 0, 0), 10);
    EXPECT_EQ(m.at<int>(0, 0, 1), -7);
    EXPECT_EQ(m.at<int>(0, 1, 0), 10);
    EXPECT_EQ(m.at<int>(0, 1, 1), -7);
    EXPECT_EQ(m.at<int>(0, 2, 0), 10);
    EXPECT_EQ(m.at<int>(0, 2, 1), -7);
}

TEST(MatScalarOps_TEST, setto_scalar_supports_non_continuous_roi)
{
    Mat base({2, 5}, CV_8UC3);
    base.setTo(Scalar::all(0.0));

    Mat roi = base.colRange(1, 4);
    ASSERT_FALSE(roi.isContinuous());
    roi.setTo(Scalar(9.0, 8.0, 7.0));

    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 5; ++x)
        {
            if (x >= 1 && x < 4)
            {
                EXPECT_EQ(base.at<uchar>(y, x, 0), static_cast<uchar>(9));
                EXPECT_EQ(base.at<uchar>(y, x, 1), static_cast<uchar>(8));
                EXPECT_EQ(base.at<uchar>(y, x, 2), static_cast<uchar>(7));
            }
            else
            {
                EXPECT_EQ(base.at<uchar>(y, x, 0), static_cast<uchar>(0));
                EXPECT_EQ(base.at<uchar>(y, x, 1), static_cast<uchar>(0));
                EXPECT_EQ(base.at<uchar>(y, x, 2), static_cast<uchar>(0));
            }
        }
    }
}

TEST(MatScalarOps_TEST, setto_scalar_rejects_more_than_four_channels)
{
    Mat m({1, 1}, CV_8UC(5));
    EXPECT_THROW(m.setTo(Scalar::all(1.0)), Exception);
}

TEST(MatScalarOps_TEST, setto_scalar_saturates_values_like_float_path)
{
    Mat m({1, 2}, CV_8UC4);
    m.setTo(Scalar(-5.0, 12.4, 260.0, 7.6));

    EXPECT_EQ(m.at<uchar>(0, 0, 0), static_cast<uchar>(0));
    EXPECT_EQ(m.at<uchar>(0, 0, 1), static_cast<uchar>(12));
    EXPECT_EQ(m.at<uchar>(0, 0, 2), static_cast<uchar>(255));
    EXPECT_EQ(m.at<uchar>(0, 0, 3), static_cast<uchar>(8));
    EXPECT_EQ(m.at<uchar>(0, 1, 0), static_cast<uchar>(0));
    EXPECT_EQ(m.at<uchar>(0, 1, 1), static_cast<uchar>(12));
    EXPECT_EQ(m.at<uchar>(0, 1, 2), static_cast<uchar>(255));
    EXPECT_EQ(m.at<uchar>(0, 1, 3), static_cast<uchar>(8));
}

TEST(MatScalarOps_TEST, add_mat_scalar_and_scalar_mat_support_multichannel)
{
    Mat src({2, 2}, CV_32SC3);
    int seed = 1;
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 2; ++x)
        {
            src.at<int>(y, x, 0) = seed;
            src.at<int>(y, x, 1) = seed + 9;
            src.at<int>(y, x, 2) = -seed;
            ++seed;
        }
    }

    const Scalar s(5.0, -2.0, 9.0);
    Mat lhs_plus_rhs;
    Mat rhs_plus_lhs;
    add(src, s, lhs_plus_rhs);
    add(s, src, rhs_plus_lhs);

    ASSERT_EQ(lhs_plus_rhs.type(), CV_32SC3);
    ASSERT_EQ(lhs_plus_rhs.shape(), src.shape());
    ASSERT_EQ(rhs_plus_lhs.type(), CV_32SC3);
    ASSERT_EQ(rhs_plus_lhs.shape(), src.shape());

    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 2; ++x)
        {
            EXPECT_EQ(lhs_plus_rhs.at<int>(y, x, 0), src.at<int>(y, x, 0) + 5);
            EXPECT_EQ(lhs_plus_rhs.at<int>(y, x, 1), src.at<int>(y, x, 1) - 2);
            EXPECT_EQ(lhs_plus_rhs.at<int>(y, x, 2), src.at<int>(y, x, 2) + 9);

            EXPECT_EQ(rhs_plus_lhs.at<int>(y, x, 0), src.at<int>(y, x, 0) + 5);
            EXPECT_EQ(rhs_plus_lhs.at<int>(y, x, 1), src.at<int>(y, x, 1) - 2);
            EXPECT_EQ(rhs_plus_lhs.at<int>(y, x, 2), src.at<int>(y, x, 2) + 9);
        }
    }
}

TEST(MatScalarOps_TEST, subtract_mat_scalar_and_scalar_mat_support_multichannel)
{
    Mat src({2, 2}, CV_32SC3);
    int seed = 1;
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 2; ++x)
        {
            src.at<int>(y, x, 0) = seed;
            src.at<int>(y, x, 1) = seed + 9;
            src.at<int>(y, x, 2) = -seed;
            ++seed;
        }
    }

    const Scalar s(5.0, -2.0, 9.0);
    Mat mat_minus_scalar;
    Mat scalar_minus_mat;
    subtract(src, s, mat_minus_scalar);
    subtract(s, src, scalar_minus_mat);

    ASSERT_EQ(mat_minus_scalar.type(), CV_32SC3);
    ASSERT_EQ(mat_minus_scalar.shape(), src.shape());
    ASSERT_EQ(scalar_minus_mat.type(), CV_32SC3);
    ASSERT_EQ(scalar_minus_mat.shape(), src.shape());

    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 2; ++x)
        {
            EXPECT_EQ(mat_minus_scalar.at<int>(y, x, 0), src.at<int>(y, x, 0) - 5);
            EXPECT_EQ(mat_minus_scalar.at<int>(y, x, 1), src.at<int>(y, x, 1) + 2);
            EXPECT_EQ(mat_minus_scalar.at<int>(y, x, 2), src.at<int>(y, x, 2) - 9);

            EXPECT_EQ(scalar_minus_mat.at<int>(y, x, 0), 5 - src.at<int>(y, x, 0));
            EXPECT_EQ(scalar_minus_mat.at<int>(y, x, 1), -2 - src.at<int>(y, x, 1));
            EXPECT_EQ(scalar_minus_mat.at<int>(y, x, 2), 9 - src.at<int>(y, x, 2));
        }
    }
}

TEST(MatScalarOps_TEST, compare_mat_scalar_and_scalar_mat_return_u8_mask_with_channels)
{
    Mat src({2, 2}, CV_32SC3);
    src.at<int>(0, 0, 0) = 5;   src.at<int>(0, 0, 1) = 9;   src.at<int>(0, 0, 2) = 12;
    src.at<int>(0, 1, 0) = 10;  src.at<int>(0, 1, 1) = 3;   src.at<int>(0, 1, 2) = 11;
    src.at<int>(1, 0, 0) = -1;  src.at<int>(1, 0, 1) = 8;   src.at<int>(1, 0, 2) = 20;
    src.at<int>(1, 1, 0) = 7;   src.at<int>(1, 1, 1) = 7;   src.at<int>(1, 1, 2) = 7;

    const Scalar s(7.0, 8.0, 11.0);
    Mat mat_scalar_mask;
    Mat scalar_mat_mask;
    compare(src, s, mat_scalar_mask, CV_CMP_GE);
    compare(s, src, scalar_mat_mask, CV_CMP_GT);

    ASSERT_EQ(mat_scalar_mask.type(), CV_8UC3);
    ASSERT_EQ(mat_scalar_mask.shape(), src.shape());
    ASSERT_EQ(scalar_mat_mask.type(), CV_8UC3);
    ASSERT_EQ(scalar_mat_mask.shape(), src.shape());

    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 2; ++x)
        {
            EXPECT_EQ(mat_scalar_mask.at<uchar>(y, x, 0),
                      src.at<int>(y, x, 0) >= 7 ? static_cast<uchar>(255) : static_cast<uchar>(0));
            EXPECT_EQ(mat_scalar_mask.at<uchar>(y, x, 1),
                      src.at<int>(y, x, 1) >= 8 ? static_cast<uchar>(255) : static_cast<uchar>(0));
            EXPECT_EQ(mat_scalar_mask.at<uchar>(y, x, 2),
                      src.at<int>(y, x, 2) >= 11 ? static_cast<uchar>(255) : static_cast<uchar>(0));

            EXPECT_EQ(scalar_mat_mask.at<uchar>(y, x, 0),
                      7 > src.at<int>(y, x, 0) ? static_cast<uchar>(255) : static_cast<uchar>(0));
            EXPECT_EQ(scalar_mat_mask.at<uchar>(y, x, 1),
                      8 > src.at<int>(y, x, 1) ? static_cast<uchar>(255) : static_cast<uchar>(0));
            EXPECT_EQ(scalar_mat_mask.at<uchar>(y, x, 2),
                      11 > src.at<int>(y, x, 2) ? static_cast<uchar>(255) : static_cast<uchar>(0));
        }
    }
}

TEST(MatScalarOps_TEST, scalar_binary_and_compare_support_non_continuous_roi)
{
    Mat base({2, 5}, CV_32SC3);
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 5; ++x)
        {
            base.at<int>(y, x, 0) = 100 * y + 10 * x + 0;
            base.at<int>(y, x, 1) = 100 * y + 10 * x + 1;
            base.at<int>(y, x, 2) = 100 * y + 10 * x + 2;
        }
    }

    Mat roi = base.colRange(1, 4);
    ASSERT_FALSE(roi.isContinuous());

    Mat add_out;
    add(roi, Scalar(1.0, 2.0, 3.0), add_out);
    ASSERT_EQ(add_out.shape(), roi.shape());
    ASSERT_EQ(add_out.type(), CV_32SC3);

    Mat cmp_out;
    compare(Scalar(120.0, 120.0, 120.0), roi, cmp_out, CV_CMP_GT);
    ASSERT_EQ(cmp_out.shape(), roi.shape());
    ASSERT_EQ(cmp_out.type(), CV_8UC3);

    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 3; ++x)
        {
            const int src_x = x + 1;
            const int v0 = 100 * y + 10 * src_x + 0;
            const int v1 = 100 * y + 10 * src_x + 1;
            const int v2 = 100 * y + 10 * src_x + 2;

            EXPECT_EQ(add_out.at<int>(y, x, 0), v0 + 1);
            EXPECT_EQ(add_out.at<int>(y, x, 1), v1 + 2);
            EXPECT_EQ(add_out.at<int>(y, x, 2), v2 + 3);

            EXPECT_EQ(cmp_out.at<uchar>(y, x, 0), 120 > v0 ? static_cast<uchar>(255) : static_cast<uchar>(0));
            EXPECT_EQ(cmp_out.at<uchar>(y, x, 1), 120 > v1 ? static_cast<uchar>(255) : static_cast<uchar>(0));
            EXPECT_EQ(cmp_out.at<uchar>(y, x, 2), 120 > v2 ? static_cast<uchar>(255) : static_cast<uchar>(0));
        }
    }
}

TEST(MatScalarOps_TEST, scalar_binary_and_compare_reject_more_than_four_channels)
{
    Mat src({1, 2}, CV_32SC(5));
    Mat dst;

    EXPECT_THROW(add(src, Scalar::all(1.0), dst), Exception);
    EXPECT_THROW(add(Scalar::all(1.0), src, dst), Exception);
    EXPECT_THROW(subtract(src, Scalar::all(1.0), dst), Exception);
    EXPECT_THROW(subtract(Scalar::all(1.0), src, dst), Exception);
    EXPECT_THROW(compare(src, Scalar::all(1.0), dst, CV_CMP_EQ), Exception);
    EXPECT_THROW(compare(Scalar::all(1.0), src, dst, CV_CMP_EQ), Exception);
}
