#include "cvh.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

using namespace cvh;

namespace
{

template<typename T>
void set_raw_bits(T& value, const void* bits)
{
    std::memcpy(&value, bits, sizeof(T));
}

template<typename UInt, typename T>
UInt raw_bits(const T& value)
{
    static_assert(sizeof(UInt) == sizeof(T), "raw bit type size mismatch");
    UInt bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

}  // namespace

TEST(ArrayOpsContract_TEST, public_numeric_apis_cover_mat_and_scalar_inputs)
{
    Mat a({1, 7}, CV_8UC1);
    Mat b({1, 7}, CV_8UC1);
    const uchar a_values[] = {0, 10, 20, 30, 200, 250, 255};
    const uchar b_values[] = {1, 5, 30, 20, 210, 240, 0};
    for (int x = 0; x < 7; ++x)
    {
        a.at<uchar>(0, x) = a_values[x];
        b.at<uchar>(0, x) = b_values[x];
    }

    Mat out;
    absdiff(a, b, out);
    const uchar abs_expected[] = {1, 5, 10, 10, 10, 10, 255};
    for (int x = 0; x < 7; ++x)
    {
        EXPECT_EQ(out.at<uchar>(0, x), abs_expected[x]);
    }

    absdiff(a, Scalar::all(20.0), out);
    EXPECT_EQ(out.at<uchar>(0, 0), 20);
    EXPECT_EQ(out.at<uchar>(0, 2), 0);
    EXPECT_EQ(out.at<uchar>(0, 6), 235);

    min(a, b, out);
    EXPECT_EQ(out.at<uchar>(0, 1), 5);
    EXPECT_EQ(out.at<uchar>(0, 4), 200);
    min(a, Scalar::all(25.0), out);
    EXPECT_EQ(out.at<uchar>(0, 0), 0);
    EXPECT_EQ(out.at<uchar>(0, 3), 25);
    max(a, Scalar::all(25.0), out);
    EXPECT_EQ(out.at<uchar>(0, 0), 25);
    EXPECT_EQ(out.at<uchar>(0, 3), 30);
}

TEST(ArrayOpsContract_TEST, bitwise_apis_use_raw_float_bits)
{
    Mat a({1, 3}, CV_32FC1);
    Mat b({1, 3}, CV_32FC1);
    const std::uint32_t a_bits[] = {0x3f800000u, 0x80000000u, 0x7fc12345u};
    const std::uint32_t b_bits[] = {0x00ff00ffu, 0xffffffffu, 0x0f0f0f0fu};
    for (int x = 0; x < 3; ++x)
    {
        set_raw_bits(a.at<float>(0, x), &a_bits[x]);
        set_raw_bits(b.at<float>(0, x), &b_bits[x]);
    }

    Mat out;
    bitwise_and(a, b, out);
    for (int x = 0; x < 3; ++x)
    {
        EXPECT_EQ(raw_bits<std::uint32_t>(out.at<float>(0, x)), a_bits[x] & b_bits[x]);
    }

    bitwise_or(a, b, out);
    for (int x = 0; x < 3; ++x)
    {
        EXPECT_EQ(raw_bits<std::uint32_t>(out.at<float>(0, x)), a_bits[x] | b_bits[x]);
    }

    bitwise_xor(a, b, out);
    for (int x = 0; x < 3; ++x)
    {
        EXPECT_EQ(raw_bits<std::uint32_t>(out.at<float>(0, x)), a_bits[x] ^ b_bits[x]);
    }

    bitwise_not(a, out);
    for (int x = 0; x < 3; ++x)
    {
        EXPECT_EQ(raw_bits<std::uint32_t>(out.at<float>(0, x)), ~a_bits[x]);
    }
}

TEST(ArrayOpsContract_TEST, bitwise_scalar_and_mask_preserve_unselected_pixels)
{
    Mat src({1, 5}, CV_8UC4);
    for (int x = 0; x < 5; ++x)
    {
        for (int ch = 0; ch < 4; ++ch)
        {
            src.at<uchar>(0, x, ch) = static_cast<uchar>(0x10 * (ch + 1) + x);
        }
    }
    Mat mask({1, 5}, CV_8UC1);
    mask.at<uchar>(0, 0) = 0;
    mask.at<uchar>(0, 1) = 255;
    mask.at<uchar>(0, 2) = 0;
    mask.at<uchar>(0, 3) = 1;
    mask.at<uchar>(0, 4) = 0;

    Mat dst({1, 5}, CV_8UC4);
    dst.setTo(Scalar::all(0xA5));
    bitwise_xor(src, Scalar(0x0F, 0xF0, 0x55, 0xAA), dst, mask);
    for (int x = 0; x < 5; ++x)
    {
        for (int ch = 0; ch < 4; ++ch)
        {
            const uchar expected =
                mask.at<uchar>(0, x) != 0
                    ? static_cast<uchar>(src.at<uchar>(0, x, ch) ^
                                         static_cast<uchar>(Scalar(0x0F, 0xF0, 0x55, 0xAA)[ch]))
                    : static_cast<uchar>(0xA5);
            EXPECT_EQ(dst.at<uchar>(0, x, ch), expected);
        }
    }

    Mat scalar_out;
    bitwise_and(src, Scalar::all(0x0F), scalar_out);
    EXPECT_EQ(scalar_out.at<uchar>(0, 2, 0),
              static_cast<uchar>(src.at<uchar>(0, 2, 0) & 0x0F));
    bitwise_or(Scalar::all(0x80), src, scalar_out);
    EXPECT_EQ(scalar_out.at<uchar>(0, 2, 1),
              static_cast<uchar>(0x80 | src.at<uchar>(0, 2, 1)));
    bitwise_xor(Scalar::all(0xFF), src, scalar_out);
    EXPECT_EQ(scalar_out.at<uchar>(0, 2, 2),
              static_cast<uchar>(0xFF ^ src.at<uchar>(0, 2, 2)));

    Mat allocated;
    bitwise_not(src, allocated, mask);
    for (int x = 0; x < 5; ++x)
    {
        for (int ch = 0; ch < 4; ++ch)
        {
            const uchar expected = mask.at<uchar>(0, x) != 0
                                       ? static_cast<uchar>(~src.at<uchar>(0, x, ch))
                                       : static_cast<uchar>(0);
            EXPECT_EQ(allocated.at<uchar>(0, x, ch), expected);
        }
    }
}

TEST(ArrayOpsContract_TEST, inrange_combines_channels_into_single_channel_mask)
{
    Mat src({1, 4}, CV_16SC3);
    const short values[4][3] = {
        {1, 10, 100},
        {2, 20, 200},
        {3, 30, 300},
        {4, 40, 400},
    };
    for (int x = 0; x < 4; ++x)
    {
        for (int ch = 0; ch < 3; ++ch)
        {
            src.at<short>(0, x, ch) = values[x][ch];
        }
    }

    Mat scalar_mask;
    inRange(src, Scalar(2.0, 15.0, 150.0), Scalar(4.0, 35.0, 350.0), scalar_mask);
    ASSERT_EQ(scalar_mask.type(), CV_8UC1);
    EXPECT_EQ(scalar_mask.at<uchar>(0, 0), 0);
    EXPECT_EQ(scalar_mask.at<uchar>(0, 1), 255);
    EXPECT_EQ(scalar_mask.at<uchar>(0, 2), 255);
    EXPECT_EQ(scalar_mask.at<uchar>(0, 3), 0);

    Mat lower(src.shape(), src.type());
    Mat upper(src.shape(), src.type());
    lower.setTo(Scalar(1.0, 10.0, 100.0));
    upper.setTo(Scalar(3.0, 30.0, 300.0));
    Mat mat_mask;
    inRange(src, lower, upper, mat_mask);
    EXPECT_EQ(mat_mask.at<uchar>(0, 0), 255);
    EXPECT_EQ(mat_mask.at<uchar>(0, 1), 255);
    EXPECT_EQ(mat_mask.at<uchar>(0, 2), 255);
    EXPECT_EQ(mat_mask.at<uchar>(0, 3), 0);
}

TEST(ArrayOpsContract_TEST, inrange_integer_scalar_uses_inclusive_fractional_bounds)
{
    Mat src({1, 5}, CV_32SC1);
    for (int x = 0; x < 5; ++x)
    {
        src.at<int>(0, x) = x + 1;
    }

    Mat mask;
    inRange(src, Scalar::all(2.5), Scalar::all(4.5), mask);
    const uchar expected[] = {0, 0, 255, 255, 0};
    for (int x = 0; x < 5; ++x)
    {
        EXPECT_EQ(mask.at<uchar>(0, x), expected[x]);
    }
}

TEST(ArrayOpsContract_TEST, public_ops_support_non_contiguous_roi_and_in_place_output)
{
    Mat a_base({3, 8}, CV_8UC3);
    Mat b_base({3, 8}, CV_8UC3);
    for (int y = 0; y < 3; ++y)
    {
        for (int x = 0; x < 8; ++x)
        {
            for (int ch = 0; ch < 3; ++ch)
            {
                a_base.at<uchar>(y, x, ch) = static_cast<uchar>(10 * y + x + ch);
                b_base.at<uchar>(y, x, ch) = static_cast<uchar>(2 * x + ch);
            }
        }
    }

    Mat a = a_base.colRange(1, 7);
    Mat b = b_base.colRange(1, 7);
    ASSERT_FALSE(a.isContinuous());
    ASSERT_FALSE(b.isContinuous());

    Mat expected;
    absdiff(a, b, expected);
    absdiff(a, b, a);
    for (int y = 0; y < 3; ++y)
    {
        for (int x = 0; x < 6; ++x)
        {
            for (int ch = 0; ch < 3; ++ch)
            {
                EXPECT_EQ(a.at<uchar>(y, x, ch), expected.at<uchar>(y, x, ch));
            }
        }
    }

    Mat range_mask;
    inRange(a, Scalar::all(0.0), Scalar::all(20.0), range_mask);
    ASSERT_EQ(range_mask.shape(), a.shape());
    ASSERT_EQ(range_mask.type(), CV_8UC1);
}

TEST(ArrayOpsContract_TEST, floating_numeric_edges_have_stable_operand_order_semantics)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    Mat a({1, 5}, CV_32FC1);
    Mat b({1, 5}, CV_32FC1);
    const float av[] = {nan, 1.0f, inf, -0.0f, 0.0f};
    const float bv[] = {2.0f, nan, inf, 0.0f, -0.0f};
    for (int x = 0; x < 5; ++x)
    {
        a.at<float>(0, x) = av[x];
        b.at<float>(0, x) = bv[x];
    }

    Mat out;
    min(a, b, out);
    EXPECT_TRUE(std::isnan(out.at<float>(0, 0)));
    EXPECT_TRUE(std::isnan(out.at<float>(0, 1)));
    EXPECT_TRUE(std::signbit(out.at<float>(0, 3)));
    EXPECT_FALSE(std::signbit(out.at<float>(0, 4)));

    max(a, b, out);
    EXPECT_TRUE(std::isnan(out.at<float>(0, 0)));
    EXPECT_TRUE(std::isnan(out.at<float>(0, 1)));
    EXPECT_FALSE(std::signbit(out.at<float>(0, 3)));
    EXPECT_FALSE(std::signbit(out.at<float>(0, 4)));

    absdiff(a, b, out);
    EXPECT_TRUE(std::isnan(out.at<float>(0, 0)));
    EXPECT_TRUE(std::isnan(out.at<float>(0, 1)));
    EXPECT_TRUE(std::isnan(out.at<float>(0, 2)));
    EXPECT_FALSE(std::signbit(out.at<float>(0, 3)));
    EXPECT_FALSE(std::signbit(out.at<float>(0, 4)));

    Mat a64({1, 2}, CV_64FC1);
    Mat b64({1, 2}, CV_64FC1);
    a64.at<double>(0, 0) = -std::numeric_limits<double>::infinity();
    a64.at<double>(0, 1) = -0.0;
    b64.at<double>(0, 0) = std::numeric_limits<double>::infinity();
    b64.at<double>(0, 1) = 0.0;
    out.release();
    absdiff(a64, b64, out);
    EXPECT_TRUE(std::isinf(out.at<double>(0, 0)));
    EXPECT_FALSE(std::signbit(out.at<double>(0, 1)));

    out.release();
    min(a64, b64, out);
    EXPECT_TRUE(std::isinf(out.at<double>(0, 0)));
    EXPECT_TRUE(std::signbit(out.at<double>(0, 0)));
    EXPECT_TRUE(std::signbit(out.at<double>(0, 1)));

    max(a64, b64, out);
    EXPECT_TRUE(std::isinf(out.at<double>(0, 0)));
    EXPECT_FALSE(std::signbit(out.at<double>(0, 0)));
    EXPECT_FALSE(std::signbit(out.at<double>(0, 1)));
}

TEST(ArrayOpsContract_TEST, invalid_shapes_types_and_masks_throw)
{
    Mat a({2, 3}, CV_8UC1);
    Mat wrong_shape({2, 4}, CV_8UC1);
    Mat wrong_type({2, 3}, CV_16UC1);
    Mat bad_mask({2, 3}, CV_8UC3);
    Mat out;

    EXPECT_THROW(absdiff(a, wrong_shape, out), Exception);
    EXPECT_THROW(min(a, wrong_type, out), Exception);
    EXPECT_THROW(bitwise_and(a, a, out, bad_mask), Exception);
    EXPECT_THROW(inRange(a, wrong_shape, wrong_shape, out), Exception);
}
