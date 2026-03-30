#include "cvh.h"
#include "gtest/gtest.h"

#include <cmath>

using namespace cvh;

namespace
{

template<typename T>
Mat make_vec_mat(const std::initializer_list<T>& values, int type)
{
    Mat out({1, static_cast<int>(values.size())}, type);
    int idx = 0;
    for (const T v : values)
    {
        out.at<T>(0, idx++) = v;
    }
    return out;
}

template<typename T>
void expect_vec_eq(const Mat& m, const std::initializer_list<T>& values)
{
    ASSERT_EQ(m.size[0], 1);
    ASSERT_EQ(m.size[1], static_cast<int>(values.size()));
    int idx = 0;
    for (const T v : values)
    {
        EXPECT_EQ(m.at<T>(0, idx++), v);
    }
}

void expect_vec_near_f32(const Mat& m, const std::initializer_list<float>& values, float eps = 1e-5f)
{
    ASSERT_EQ(m.type(), CV_32FC1);
    ASSERT_EQ(m.size[0], 1);
    ASSERT_EQ(m.size[1], static_cast<int>(values.size()));
    int idx = 0;
    for (const float v : values)
    {
        EXPECT_NEAR(m.at<float>(0, idx++), v, eps);
    }
}

}  // namespace

TEST(BinaryOpContract_TEST, add_sub_mul_div_work_on_same_shape_int32)
{
    const Mat a = make_vec_mat<int>({10, 20, -30, 40}, CV_32SC1);
    const Mat b = make_vec_mat<int>({3, -5, 7, 0}, CV_32SC1);

    Mat out;
    binaryFunc(BinaryOp::ADD, a, b, out);
    expect_vec_eq<int>(out, {13, 15, -23, 40});

    binaryFunc(BinaryOp::SUB, a, b, out);
    expect_vec_eq<int>(out, {7, 25, -37, 40});

    binaryFunc(BinaryOp::MUL, a, b, out);
    expect_vec_eq<int>(out, {30, -100, -210, 0});

    binaryFunc(BinaryOp::DIV, a, b, out);
    expect_vec_eq<int>(out, {3, -4, -4, 0});
}

TEST(BinaryOpContract_TEST, sum_is_alias_of_add)
{
    const Mat a = make_vec_mat<int>({10, 20, -30, 40}, CV_32SC1);
    const Mat b = make_vec_mat<int>({3, -5, 7, 0}, CV_32SC1);

    Mat out;
    binaryFunc(BinaryOp::SUM, a, b, out);
    expect_vec_eq<int>(out, {13, 15, -23, 40});
}

TEST(BinaryOpContract_TEST, bitwise_and_or_xor_work_on_u8)
{
    const Mat a = make_vec_mat<uchar>({0xF0, 0x0F, 0xAA, 0x55}, CV_8UC1);
    const Mat b = make_vec_mat<uchar>({0xCC, 0x33, 0x0F, 0xF0}, CV_8UC1);

    Mat out;
    binaryFunc(BinaryOp::AND, a, b, out);
    expect_vec_eq<uchar>(out, {0xC0, 0x03, 0x0A, 0x50});

    binaryFunc(BinaryOp::OR, a, b, out);
    expect_vec_eq<uchar>(out, {0xFC, 0x3F, 0xAF, 0xF5});

    binaryFunc(BinaryOp::XOR, a, b, out);
    expect_vec_eq<uchar>(out, {0x3C, 0x3C, 0xA5, 0xA5});
}

TEST(BinaryOpContract_TEST, mod_follows_divisor_sign_on_int32)
{
    const Mat a = make_vec_mat<int>({5, -5, 5, -5}, CV_32SC1);
    const Mat b = make_vec_mat<int>({2, 2, -2, -2}, CV_32SC1);

    Mat out;
    binaryFunc(BinaryOp::MOD, a, b, out);
    // sign(divisor) semantics: [1, 1, -1, -1]
    expect_vec_eq<int>(out, {1, 1, -1, -1});
}

TEST(BinaryOpContract_TEST, fmod_follows_dividend_sign)
{
    const Mat a = make_vec_mat<float>({5.f, -5.f, 5.f, -5.f}, CV_32FC1);
    const Mat b = make_vec_mat<float>({2.f, 2.f, -2.f, -2.f}, CV_32FC1);

    Mat out;
    binaryFunc(BinaryOp::FMOD, a, b, out);
    // sign(dividend) semantics: [1, -1, 1, -1]
    expect_vec_near_f32(out, {1.f, -1.f, 1.f, -1.f});
}

TEST(BinaryOpContract_TEST, pow_and_bitshift_work_on_int32)
{
    const Mat base = make_vec_mat<int>({2, 3, 4, 5}, CV_32SC1);
    const Mat exp = make_vec_mat<int>({3, 2, 1, 0}, CV_32SC1);
    Mat out;

    binaryFunc(BinaryOp::POW, base, exp, out);
    expect_vec_eq<int>(out, {8, 9, 4, 1});

    const Mat lhs = make_vec_mat<int>({8, 8, 1, 16}, CV_32SC1);
    const Mat shift = make_vec_mat<int>({-1, 1, 3, -2}, CV_32SC1);
    binaryFunc(BinaryOp::BITSHIFT, lhs, shift, out);
    // rhs<0 => right shift, rhs>0 => left shift
    expect_vec_eq<int>(out, {4, 16, 8, 4});
}

TEST(BinaryOpContract_TEST, compare_binary_ops_return_u8_mask)
{
    const Mat a = make_vec_mat<int>({1, 2, 3, 4}, CV_32SC1);
    const Mat b = make_vec_mat<int>({1, 1, 4, 3}, CV_32SC1);
    Mat out;

    binaryFunc(BinaryOp::EQUAL, a, b, out);
    ASSERT_EQ(out.type(), CV_8UC1);
    expect_vec_eq<uchar>(out, {255, 0, 0, 0});

    binaryFunc(BinaryOp::GREATER, a, b, out);
    expect_vec_eq<uchar>(out, {0, 255, 0, 255});

    binaryFunc(BinaryOp::GREATER_EQUAL, a, b, out);
    expect_vec_eq<uchar>(out, {255, 255, 0, 255});

    binaryFunc(BinaryOp::LESS, a, b, out);
    expect_vec_eq<uchar>(out, {0, 0, 255, 0});

    binaryFunc(BinaryOp::LESS_EQUAL, a, b, out);
    expect_vec_eq<uchar>(out, {255, 0, 255, 0});
}

TEST(BinaryOpContract_TEST, max_and_min_work_on_int32)
{
    const Mat a = make_vec_mat<int>({1, -5, 10, 8}, CV_32SC1);
    const Mat b = make_vec_mat<int>({2, -9, 7, 8}, CV_32SC1);
    Mat out;

    binaryFunc(BinaryOp::MAX, a, b, out);
    expect_vec_eq<int>(out, {2, -5, 10, 8});

    binaryFunc(BinaryOp::MIN, a, b, out);
    expect_vec_eq<int>(out, {1, -9, 7, 8});
}

TEST(BinaryOpContract_TEST, mean_is_elementwise_average)
{
    const Mat ai = make_vec_mat<int>({2, 4, -6, 8}, CV_32SC1);
    const Mat bi = make_vec_mat<int>({4, 6, -2, 10}, CV_32SC1);
    Mat out_i;

    binaryFunc(BinaryOp::MEAN, ai, bi, out_i);
    expect_vec_eq<int>(out_i, {3, 5, -4, 9});

    const Mat af = make_vec_mat<float>({1.f, 2.5f, -1.f}, CV_32FC1);
    const Mat bf = make_vec_mat<float>({3.f, -0.5f, 1.f}, CV_32FC1);
    Mat out_f;
    binaryFunc(BinaryOp::MEAN, af, bf, out_f);
    expect_vec_near_f32(out_f, {2.f, 1.f, 0.f});
}

TEST(BinaryOpContract_TEST, atan2_and_hypot_work_on_float32)
{
    const Mat y = make_vec_mat<float>({0.f, 1.f, -1.f, 1.f}, CV_32FC1);
    const Mat x = make_vec_mat<float>({1.f, 0.f, 0.f, 1.f}, CV_32FC1);
    Mat out;

    binaryFunc(BinaryOp::ATAN2, y, x, out);
    expect_vec_near_f32(out, {0.f, static_cast<float>(CV_PI / 2.0), static_cast<float>(-CV_PI / 2.0), static_cast<float>(CV_PI / 4.0)});

    const Mat a = make_vec_mat<float>({3.f, 5.f, 0.f, -8.f}, CV_32FC1);
    const Mat b = make_vec_mat<float>({4.f, 12.f, 0.f, 15.f}, CV_32FC1);
    binaryFunc(BinaryOp::HYPOT, a, b, out);
    expect_vec_near_f32(out, {5.f, 13.f, 0.f, 17.f});
}

TEST(BinaryOpContract_TEST, not_is_bitwise_andnot_on_u8)
{
    const Mat a = make_vec_mat<uchar>({0xF0, 0x0F, 0xAA, 0x55}, CV_8UC1);
    const Mat b = make_vec_mat<uchar>({0x0F, 0xF0, 0xFF, 0x00}, CV_8UC1);
    Mat out;

    binaryFunc(BinaryOp::NOT, a, b, out);
    expect_vec_eq<uchar>(out, {0xF0, 0x0F, 0x00, 0x55});
}

TEST(BinaryOpContract_TEST, unsupported_combinations_throw)
{
    const Mat f = make_vec_mat<float>({1.f, 2.f, 3.f}, CV_32FC1);
    const Mat g = make_vec_mat<float>({1.f, 2.f, 3.f}, CV_32FC1);
    Mat out;

    EXPECT_THROW(binaryFunc(BinaryOp::AND, f, g, out), Exception);
    EXPECT_THROW(binaryFunc(BinaryOp::OR, f, g, out), Exception);
    EXPECT_THROW(binaryFunc(BinaryOp::XOR, f, g, out), Exception);
    EXPECT_THROW(binaryFunc(BinaryOp::MOD, f, g, out), Exception);
    EXPECT_THROW(binaryFunc(BinaryOp::BITSHIFT, f, g, out), Exception);
    EXPECT_THROW(binaryFunc(BinaryOp::NOT, f, g, out), Exception);
}
