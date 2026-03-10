#include "cvh.h"
#include "gtest/gtest.h"

#include <cmath>

using namespace cvh;

TEST(CoreBasic_TEST, mat_clone_preserves_data)
{
    Mat src({2, 3}, CV_32F);
    float* src_data = reinterpret_cast<float*>(src.data);
    for (int i = 0; i < 6; ++i)
    {
        src_data[i] = static_cast<float>(i) * 1.5f - 2.0f;
    }

    Mat cloned = src.clone();
    const float* cloned_data = reinterpret_cast<const float*>(cloned.data);

    ASSERT_EQ(cloned.total(), src.total());
    ASSERT_EQ(cloned.type(), src.type());
    for (int i = 0; i < 6; ++i)
    {
        EXPECT_FLOAT_EQ(cloned_data[i], src_data[i]);
    }
}

TEST(CoreBasic_TEST, convert_to_int32_roundtrip)
{
    Mat src({1, 5}, CV_32F);
    float* src_data = reinterpret_cast<float*>(src.data);
    src_data[0] = -2.0f;
    src_data[1] = -1.2f;
    src_data[2] = 0.0f;
    src_data[3] = 2.4f;
    src_data[4] = 9.9f;

    Mat as_int32;
    src.convertTo(as_int32, CV_32S);
    ASSERT_EQ(as_int32.type(), CV_32S);

    Mat back_to_fp32;
    as_int32.convertTo(back_to_fp32, CV_32F);
    ASSERT_EQ(back_to_fp32.type(), CV_32F);

    const float* out = reinterpret_cast<const float*>(back_to_fp32.data);
    EXPECT_NEAR(out[0], -2.0f, 1e-6f);
    EXPECT_NEAR(out[1], -1.0f, 1e-6f);
    EXPECT_NEAR(out[2], 0.0f, 1e-6f);
    EXPECT_NEAR(out[3], 2.0f, 1e-6f);
    EXPECT_NEAR(out[4], 10.0f, 1e-6f);
}
