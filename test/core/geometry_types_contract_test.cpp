#include "cvh.h"
#include "gtest/gtest.h"

#include <type_traits>

using namespace cvh;

TEST(GeometryTypesContract_TEST, point_alias_preserves_existing_integer_api)
{
    static_assert(std::is_same<Point, Point2i>::value, "Point must remain the integer alias");

    const Point point(3, -7);
    EXPECT_EQ(point.x, 3);
    EXPECT_EQ(point.y, -7);
    EXPECT_EQ(point, Point2i(3, -7));
}

TEST(GeometryTypesContract_TEST, point_float_double_and_conversion_are_available)
{
    const Point2f point_f(1.5f, -2.25f);
    const Point2d point_d(point_f);
    EXPECT_DOUBLE_EQ(point_d.x, 1.5);
    EXPECT_DOUBLE_EQ(point_d.y, -2.25);

    const Point point_i(point_f);
    EXPECT_EQ(point_i.x, 1);
    EXPECT_EQ(point_i.y, -2);
}

TEST(GeometryTypesContract_TEST, size_alias_preserves_existing_integer_api)
{
    static_assert(std::is_same<Size, Size2i>::value, "Size must remain the integer alias");

    const Size size(13, 9);
    EXPECT_EQ(size.width, 13);
    EXPECT_EQ(size.height, 9);
    EXPECT_FALSE(size.empty());
    EXPECT_TRUE(Size().empty());
}

TEST(GeometryTypesContract_TEST, floating_size_conversion_is_available)
{
    const Size2f size_f(7.5f, 3.25f);
    const Size2d size_d(size_f);
    EXPECT_DOUBLE_EQ(size_d.width, 7.5);
    EXPECT_DOUBLE_EQ(size_d.height, 3.25);
}
