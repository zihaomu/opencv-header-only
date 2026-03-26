#include "cvh.h"
#include "gtest/gtest.h"

using namespace cvh;

TEST(ScalarContract_TEST, default_constructor_is_zero)
{
    const Scalar s;
    EXPECT_DOUBLE_EQ(s[0], 0.0);
    EXPECT_DOUBLE_EQ(s[1], 0.0);
    EXPECT_DOUBLE_EQ(s[2], 0.0);
    EXPECT_DOUBLE_EQ(s[3], 0.0);
}

TEST(ScalarContract_TEST, constructors_and_all_fill_lanes)
{
    const Scalar s(1.0, 2.5, -3.0, 4.25);
    EXPECT_DOUBLE_EQ(s[0], 1.0);
    EXPECT_DOUBLE_EQ(s[1], 2.5);
    EXPECT_DOUBLE_EQ(s[2], -3.0);
    EXPECT_DOUBLE_EQ(s[3], 4.25);

    const Scalar a = Scalar::all(7.0);
    EXPECT_DOUBLE_EQ(a[0], 7.0);
    EXPECT_DOUBLE_EQ(a[1], 7.0);
    EXPECT_DOUBLE_EQ(a[2], 7.0);
    EXPECT_DOUBLE_EQ(a[3], 7.0);
}

TEST(ScalarContract_TEST, equality_operators_compare_all_lanes)
{
    const Scalar a(1.0, 2.0, 3.0, 4.0);
    const Scalar b(1.0, 2.0, 3.0, 4.0);
    const Scalar c(1.0, 2.0, 3.0, 5.0);

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
    EXPECT_FALSE(a == c);
    EXPECT_TRUE(a != c);
}
