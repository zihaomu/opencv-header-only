#include "cvh.h"
#include "gtest/gtest.h"

#include <cmath>

using namespace cvh;

namespace {

double mat_value(const Mat& matrix, int row, int col)
{
    return matrix.depth() == CV_32F
        ? static_cast<double>(matrix.at<float>(row, col))
        : matrix.at<double>(row, col);
}

Point2d apply_affine(const Mat& matrix, Point2d point)
{
    return Point2d(
        mat_value(matrix, 0, 0) * point.x +
            mat_value(matrix, 0, 1) * point.y +
            mat_value(matrix, 0, 2),
        mat_value(matrix, 1, 0) * point.x +
            mat_value(matrix, 1, 1) * point.y +
            mat_value(matrix, 1, 2));
}

Point2d apply_perspective(const Mat& matrix, Point2d point)
{
    const double denominator =
        mat_value(matrix, 2, 0) * point.x +
        mat_value(matrix, 2, 1) * point.y +
        mat_value(matrix, 2, 2);
    return Point2d(
        (mat_value(matrix, 0, 0) * point.x +
         mat_value(matrix, 0, 1) * point.y +
         mat_value(matrix, 0, 2)) /
            denominator,
        (mat_value(matrix, 1, 0) * point.x +
         mat_value(matrix, 1, 1) * point.y +
         mat_value(matrix, 1, 2)) /
            denominator);
}

void expect_point_near(Point2d actual, Point2d expected, double tolerance)
{
    EXPECT_NEAR(actual.x, expected.x, tolerance);
    EXPECT_NEAR(actual.y, expected.y, tolerance);
}

}  // namespace

TEST(ImgprocPhase1GeometryMatrix_TEST, rotation_matrix_covers_identity_scale_and_center)
{
    const Point2d center(12.5, -7.25);
    const AffineMatrix2x3d identity =
        getRotationMatrix2D_(center, 0.0, 1.0);
    EXPECT_DOUBLE_EQ(identity(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(identity(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(identity(0, 2), 0.0);
    EXPECT_DOUBLE_EQ(identity(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(identity(1, 1), 1.0);
    EXPECT_DOUBLE_EQ(identity(1, 2), 0.0);

    const Mat rotation = getRotationMatrix2D(center, 90.0, 2.0);
    EXPECT_EQ(rotation.type(), CV_64FC1);
    EXPECT_EQ(rotation.shape(), MatShape({2, 3}));
    expect_point_near(
        apply_affine(rotation, center),
        center,
        1e-12);
    expect_point_near(
        apply_affine(rotation, Point2d(center.x + 1.0, center.y)),
        Point2d(center.x, center.y - 2.0),
        1e-12);
}

TEST(ImgprocPhase1GeometryMatrix_TEST, affine_maps_float_and_double_control_points)
{
    const Point2f source_f[] = {
        Point2f(0.0f, 0.0f),
        Point2f(4.0f, 0.0f),
        Point2f(0.0f, 3.0f)};
    const Point2f target_f[] = {
        Point2f(5.0f, -2.0f),
        Point2f(13.0f, -6.0f),
        Point2f(6.5f, 7.0f)};
    const Mat affine_f = getAffineTransform(source_f, target_f);
    for (int index = 0; index < 3; ++index)
    {
        expect_point_near(
            apply_affine(
                affine_f,
                Point2d(source_f[index].x, source_f[index].y)),
            Point2d(target_f[index].x, target_f[index].y),
            1e-12);
    }

    const Point2d source_d[] = {
        Point2d(1.0e8, -2.0e8),
        Point2d(1.0e8 + 128.0, -2.0e8),
        Point2d(1.0e8, -2.0e8 + 64.0)};
    const Point2d target_d[] = {
        Point2d(-3.0, 8.0),
        Point2d(253.0, -120.0),
        Point2d(29.0, 200.0)};
    const Mat affine_d = getAffineTransform(source_d, target_d);
    for (int index = 0; index < 3; ++index)
    {
        expect_point_near(
            apply_affine(affine_d, source_d[index]),
            target_d[index],
            1e-7);
    }
}

TEST(ImgprocPhase1GeometryMatrix_TEST, perspective_maps_four_points)
{
    const Point2f source[] = {
        Point2f(0.0f, 0.0f),
        Point2f(8.0f, 0.0f),
        Point2f(8.0f, 6.0f),
        Point2f(0.0f, 6.0f)};
    const Point2f target[] = {
        Point2f(1.0f, 2.0f),
        Point2f(9.0f, 1.0f),
        Point2f(7.5f, 8.0f),
        Point2f(-0.5f, 6.5f)};
    const Mat perspective = getPerspectiveTransform(source, target);
    ASSERT_EQ(perspective.type(), CV_64FC1);
    ASSERT_EQ(perspective.shape(), MatShape({3, 3}));
    EXPECT_DOUBLE_EQ(perspective.at<double>(2, 2), 1.0);
    for (int index = 0; index < 4; ++index)
    {
        expect_point_near(
            apply_perspective(
                perspective,
                Point2d(source[index].x, source[index].y)),
            Point2d(target[index].x, target[index].y),
            1e-10);
    }
}

TEST(ImgprocPhase1GeometryMatrix_TEST, inverse_composes_and_supports_aliasing)
{
    Mat matrix({2, 3}, CV_32FC1);
    matrix.at<float>(0, 0) = 1.5f;
    matrix.at<float>(0, 1) = 0.25f;
    matrix.at<float>(0, 2) = -7.0f;
    matrix.at<float>(1, 0) = -0.5f;
    matrix.at<float>(1, 1) = 2.0f;
    matrix.at<float>(1, 2) = 3.0f;
    const Mat original = matrix.clone();
    invertAffineTransform(matrix, matrix);
    EXPECT_EQ(matrix.type(), CV_32FC1);
    for (const Point2d point :
         {Point2d(0.0, 0.0), Point2d(1.25, -3.5), Point2d(100.0, 50.0)})
    {
        expect_point_near(
            apply_affine(matrix, apply_affine(original, point)),
            point,
            2e-5);
    }
}

TEST(ImgprocPhase1GeometryMatrix_TEST, degenerate_inputs_have_fixed_behavior)
{
    const Point2f repeated_source[] = {
        Point2f(1.0f, 1.0f),
        Point2f(1.0f, 1.0f),
        Point2f(2.0f, 2.0f)};
    const Point2f affine_target[] = {
        Point2f(0.0f, 0.0f),
        Point2f(1.0f, 0.0f),
        Point2f(0.0f, 1.0f)};
    EXPECT_THROW(
        getAffineTransform(repeated_source, affine_target),
        Exception);

    const Point2f collinear_source[] = {
        Point2f(0.0f, 0.0f),
        Point2f(1.0f, 1.0f),
        Point2f(2.0f, 2.0f),
        Point2f(3.0f, 3.0f)};
    const Point2f perspective_target[] = {
        Point2f(0.0f, 0.0f),
        Point2f(1.0f, 0.0f),
        Point2f(1.0f, 1.0f),
        Point2f(0.0f, 1.0f)};
    EXPECT_THROW(
        getPerspectiveTransform(collinear_source, perspective_target),
        Exception);

    Mat singular({2, 3}, CV_64FC1);
    singular.setTo(Scalar::all(0.0));
    Mat inverse;
    invertAffineTransform(singular, inverse);
    for (int row = 0; row < 2; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            EXPECT_DOUBLE_EQ(inverse.at<double>(row, col), 0.0);
        }
    }
}
