#include "cvh.h"
#include "cvh/core/detail/dispatch_control.h"
#include "gtest/gtest.h"

#include <cmath>
#include <vector>

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

Mat make_vec_mat_from_doubles(const std::initializer_list<double>& values, int type)
{
    Mat out({1, static_cast<int>(values.size())}, type);
    int idx = 0;
    const int depth = CV_MAT_DEPTH(type);
    for (const double v : values)
    {
        switch (depth)
        {
            case CV_8U:
                out.at<uchar>(0, idx) = saturate_cast<uchar>(v);
                break;
            case CV_8S:
                out.at<schar>(0, idx) = saturate_cast<schar>(v);
                break;
            case CV_16U:
                out.at<ushort>(0, idx) = saturate_cast<ushort>(v);
                break;
            case CV_16S:
                out.at<short>(0, idx) = saturate_cast<short>(v);
                break;
            case CV_32S:
                out.at<int>(0, idx) = saturate_cast<int>(v);
                break;
            case CV_32U:
                out.at<uint>(0, idx) = saturate_cast<uint>(v);
                break;
            case CV_32F:
                out.at<float>(0, idx) = saturate_cast<float>(v);
                break;
            case CV_16F:
                out.at<hfloat>(0, idx) = saturate_cast<hfloat>(v);
                break;
            default:
                CV_Error_(Error::StsNotImplemented, ("Unsupported depth=%d in test helper", depth));
        }
        ++idx;
    }
    return out;
}

double read_vec_value_as_double(const Mat& m, int idx)
{
    switch (m.depth())
    {
        case CV_8U: return static_cast<double>(m.at<uchar>(0, idx));
        case CV_8S: return static_cast<double>(m.at<schar>(0, idx));
        case CV_16U: return static_cast<double>(m.at<ushort>(0, idx));
        case CV_16S: return static_cast<double>(m.at<short>(0, idx));
        case CV_32S: return static_cast<double>(m.at<int>(0, idx));
        case CV_32U: return static_cast<double>(m.at<uint>(0, idx));
        case CV_32F: return static_cast<double>(m.at<float>(0, idx));
        case CV_16F: return static_cast<double>(static_cast<float>(m.at<hfloat>(0, idx)));
        default:
            CV_Error_(Error::StsNotImplemented, ("Unsupported depth=%d in test helper", m.depth()));
            return 0.0;
    }
}

void expect_vec_match_by_depth(const Mat& m,
                               const std::initializer_list<double>& values,
                               double float_eps = 1e-6,
                               double half_eps = 2e-2)
{
    ASSERT_EQ(m.size[0], 1);
    ASSERT_EQ(m.size[1], static_cast<int>(values.size()));
    int idx = 0;
    for (const double expected : values)
    {
        const double actual = read_vec_value_as_double(m, idx++);
        if (m.depth() == CV_16F)
        {
            EXPECT_NEAR(actual, expected, half_eps);
        }
        else if (m.depth() == CV_32F)
        {
            EXPECT_NEAR(actual, expected, float_eps);
        }
        else
        {
            EXPECT_EQ(actual, expected);
        }
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

TEST(BinaryOpContract_TEST, add_supports_all_declared_depths)
{
    const int types[] = {
        CV_8UC1, CV_8SC1, CV_16UC1, CV_16SC1, CV_32SC1, CV_32UC1, CV_32FC1, CV_16FC1
    };

    for (const int type : types)
    {
        SCOPED_TRACE(type);
        const Mat a = make_vec_mat_from_doubles({1.0, 2.0, 3.0}, type);
        const Mat b = make_vec_mat_from_doubles({4.0, 5.0, 6.0}, type);
        Mat out;
        binaryFunc(BinaryOp::ADD, a, b, out);

        ASSERT_EQ(out.type(), type);
        expect_vec_match_by_depth(out, {5.0, 7.0, 9.0});
    }
}

TEST(BinaryOpContract_TEST, mat_mat_add_sub_mul_int32_and_uint32_support_scalar_only_mode)
{
    const auto previous_mode = cpu::dispatch_mode();
    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);

    Mat out;

    const Mat a32s = make_vec_mat<int>({10, -20, 30, -40}, CV_32SC1);
    const Mat b32s = make_vec_mat<int>({3, 5, -7, 2}, CV_32SC1);
    binaryFunc(BinaryOp::ADD, a32s, b32s, out);
    expect_vec_eq<int>(out, {13, -15, 23, -38});
    binaryFunc(BinaryOp::SUB, a32s, b32s, out);
    expect_vec_eq<int>(out, {7, -25, 37, -42});
    binaryFunc(BinaryOp::MUL, a32s, b32s, out);
    expect_vec_eq<int>(out, {30, -100, -210, -80});

    out = Mat();
    const Mat a32u = make_vec_mat<uint>({10u, 20u, 30u, 40u}, CV_32UC1);
    const Mat b32u = make_vec_mat<uint>({3u, 5u, 7u, 2u}, CV_32UC1);
    binaryFunc(BinaryOp::ADD, a32u, b32u, out);
    expect_vec_eq<uint>(out, {13u, 25u, 37u, 42u});
    binaryFunc(BinaryOp::SUB, a32u, b32u, out);
    expect_vec_eq<uint>(out, {7u, 15u, 23u, 38u});
    binaryFunc(BinaryOp::MUL, a32u, b32u, out);
    expect_vec_eq<uint>(out, {30u, 100u, 210u, 80u});

    cpu::set_dispatch_mode(previous_mode);
}

TEST(BinaryOpContract_TEST, mat_mat_add_sub_u8_s8_u16_s16_support_scalar_only_with_saturation)
{
    const auto previous_mode = cpu::dispatch_mode();
    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);

    Mat out;

    const Mat a8u = make_vec_mat<uchar>({250, 10, 5, 0}, CV_8UC1);
    const Mat b8u = make_vec_mat<uchar>({10, 250, 7, 1}, CV_8UC1);
    binaryFunc(BinaryOp::ADD, a8u, b8u, out);
    expect_vec_eq<uchar>(out, {255, 255, 12, 1});
    binaryFunc(BinaryOp::SUB, a8u, b8u, out);
    expect_vec_eq<uchar>(out, {240, 0, 0, 0});

    out = Mat();
    const Mat a8s = make_vec_mat<schar>({120, -120, 50, -50}, CV_8SC1);
    const Mat b8s = make_vec_mat<schar>({20, -20, -100, 100}, CV_8SC1);
    binaryFunc(BinaryOp::ADD, a8s, b8s, out);
    expect_vec_eq<schar>(out, {127, -128, -50, 50});
    binaryFunc(BinaryOp::SUB, a8s, b8s, out);
    expect_vec_eq<schar>(out, {100, -100, 127, -128});

    out = Mat();
    const Mat a16u = make_vec_mat<ushort>({65530, 10, 3, 0}, CV_16UC1);
    const Mat b16u = make_vec_mat<ushort>({100, 65530, 9, 1}, CV_16UC1);
    binaryFunc(BinaryOp::ADD, a16u, b16u, out);
    expect_vec_eq<ushort>(out, {65535, 65535, 12, 1});
    binaryFunc(BinaryOp::SUB, a16u, b16u, out);
    expect_vec_eq<ushort>(out, {65430, 0, 0, 0});

    out = Mat();
    const Mat a16s = make_vec_mat<short>({32760, -32760, 1000, -1000}, CV_16SC1);
    const Mat b16s = make_vec_mat<short>({100, -100, -2000, 2000}, CV_16SC1);
    binaryFunc(BinaryOp::ADD, a16s, b16s, out);
    expect_vec_eq<short>(out, {32767, -32768, -1000, 1000});
    binaryFunc(BinaryOp::SUB, a16s, b16s, out);
    expect_vec_eq<short>(out, {32660, -32660, 3000, -3000});

    cpu::set_dispatch_mode(previous_mode);
}

TEST(BinaryOpContract_TEST, mat_scalar_add_sub_mul_int32_and_uint32_support_scalar_only_mode)
{
    const auto previous_mode = cpu::dispatch_mode();
    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);

    Mat out;

    const Mat src32s = make_vec_mat<int>({10, -20, 30, -40}, CV_32SC1);
    add(src32s, Scalar(5.0), out);
    expect_vec_eq<int>(out, {15, -15, 35, -35});
    subtract(Scalar(7.0), src32s, out);
    expect_vec_eq<int>(out, {-3, 27, -23, 47});
    multiply(src32s, Scalar(-2.0), out);
    expect_vec_eq<int>(out, {-20, 40, -60, 80});

    out = Mat();
    Mat src32u({1, 2}, CV_32UC3);
    src32u.at<uint>(0, 0, 0) = 10u;
    src32u.at<uint>(0, 0, 1) = 20u;
    src32u.at<uint>(0, 0, 2) = 30u;
    src32u.at<uint>(0, 1, 0) = 40u;
    src32u.at<uint>(0, 1, 1) = 50u;
    src32u.at<uint>(0, 1, 2) = 60u;

    add(src32u, Scalar(3.0, 5.0, 7.0), out);
    EXPECT_EQ(out.at<uint>(0, 0, 0), 13u);
    EXPECT_EQ(out.at<uint>(0, 0, 1), 25u);
    EXPECT_EQ(out.at<uint>(0, 0, 2), 37u);
    EXPECT_EQ(out.at<uint>(0, 1, 0), 43u);
    EXPECT_EQ(out.at<uint>(0, 1, 1), 55u);
    EXPECT_EQ(out.at<uint>(0, 1, 2), 67u);

    subtract(Scalar(100.0, 80.0, 60.0), src32u, out);
    EXPECT_EQ(out.at<uint>(0, 0, 0), 90u);
    EXPECT_EQ(out.at<uint>(0, 0, 1), 60u);
    EXPECT_EQ(out.at<uint>(0, 0, 2), 30u);
    EXPECT_EQ(out.at<uint>(0, 1, 0), 60u);
    EXPECT_EQ(out.at<uint>(0, 1, 1), 30u);
    EXPECT_EQ(out.at<uint>(0, 1, 2), 0u);

    multiply(src32u, Scalar(2.0, 3.0, 4.0), out);
    EXPECT_EQ(out.at<uint>(0, 0, 0), 20u);
    EXPECT_EQ(out.at<uint>(0, 0, 1), 60u);
    EXPECT_EQ(out.at<uint>(0, 0, 2), 120u);
    EXPECT_EQ(out.at<uint>(0, 1, 0), 80u);
    EXPECT_EQ(out.at<uint>(0, 1, 1), 150u);
    EXPECT_EQ(out.at<uint>(0, 1, 2), 240u);

    cpu::set_dispatch_mode(previous_mode);
}

TEST(BinaryOpContract_TEST, mat_scalar_add_sub_u8_s8_u16_s16_support_scalar_only_with_saturation)
{
    const auto previous_mode = cpu::dispatch_mode();
    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);

    Mat out;

    const Mat src8u = make_vec_mat<uchar>({250, 10, 5, 0}, CV_8UC1);
    add(src8u, Scalar(10.0), out);
    expect_vec_eq<uchar>(out, {255, 20, 15, 10});
    subtract(Scalar(3.0), src8u, out);
    expect_vec_eq<uchar>(out, {0, 0, 0, 3});

    out = Mat();
    const Mat src8s = make_vec_mat<schar>({120, -120, 50, -50}, CV_8SC1);
    add(src8s, Scalar(20.0), out);
    expect_vec_eq<schar>(out, {127, -100, 70, -30});
    subtract(Scalar(-100.0), src8s, out);
    expect_vec_eq<schar>(out, {-128, 20, -128, -50});

    out = Mat();
    Mat src16u({1, 2}, CV_16UC3);
    src16u.at<ushort>(0, 0, 0) = 65530;
    src16u.at<ushort>(0, 0, 1) = 10;
    src16u.at<ushort>(0, 0, 2) = 3;
    src16u.at<ushort>(0, 1, 0) = 100;
    src16u.at<ushort>(0, 1, 1) = 200;
    src16u.at<ushort>(0, 1, 2) = 300;
    add(src16u, Scalar(10.0, 65530.0, 40.0), out);
    EXPECT_EQ(out.at<ushort>(0, 0, 0), 65535);
    EXPECT_EQ(out.at<ushort>(0, 0, 1), 65535);
    EXPECT_EQ(out.at<ushort>(0, 0, 2), 43);
    EXPECT_EQ(out.at<ushort>(0, 1, 0), 110);
    EXPECT_EQ(out.at<ushort>(0, 1, 1), 65535);
    EXPECT_EQ(out.at<ushort>(0, 1, 2), 340);

    subtract(Scalar(50.0, 20.0, 100.0), src16u, out);
    EXPECT_EQ(out.at<ushort>(0, 0, 0), 0);
    EXPECT_EQ(out.at<ushort>(0, 0, 1), 10);
    EXPECT_EQ(out.at<ushort>(0, 0, 2), 97);
    EXPECT_EQ(out.at<ushort>(0, 1, 0), 0);
    EXPECT_EQ(out.at<ushort>(0, 1, 1), 0);
    EXPECT_EQ(out.at<ushort>(0, 1, 2), 0);

    out = Mat();
    Mat src16s({1, 2}, CV_16SC3);
    src16s.at<short>(0, 0, 0) = 32760;
    src16s.at<short>(0, 0, 1) = -32760;
    src16s.at<short>(0, 0, 2) = 1000;
    src16s.at<short>(0, 1, 0) = -1000;
    src16s.at<short>(0, 1, 1) = 2000;
    src16s.at<short>(0, 1, 2) = -2000;
    add(src16s, Scalar(100.0, -100.0, 30000.0), out);
    EXPECT_EQ(out.at<short>(0, 0, 0), 32767);
    EXPECT_EQ(out.at<short>(0, 0, 1), -32768);
    EXPECT_EQ(out.at<short>(0, 0, 2), 31000);
    EXPECT_EQ(out.at<short>(0, 1, 0), -900);
    EXPECT_EQ(out.at<short>(0, 1, 1), 1900);
    EXPECT_EQ(out.at<short>(0, 1, 2), 28000);

    subtract(Scalar(-32000.0, 32000.0, -32000.0), src16s, out);
    EXPECT_EQ(out.at<short>(0, 0, 0), -32768);
    EXPECT_EQ(out.at<short>(0, 0, 1), 32767);
    EXPECT_EQ(out.at<short>(0, 0, 2), -32768);
    EXPECT_EQ(out.at<short>(0, 1, 0), -31000);
    EXPECT_EQ(out.at<short>(0, 1, 1), 30000);
    EXPECT_EQ(out.at<short>(0, 1, 2), -30000);

    cpu::set_dispatch_mode(previous_mode);
}

TEST(BinaryOpContract_TEST, mat_mat_and_mat_scalar_mul_u8_s8_u16_s16_support_scalar_only_with_saturation)
{
    const auto previous_mode = cpu::dispatch_mode();
    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);

    Mat out;

    const Mat a8u = make_vec_mat<uchar>({20, 30, 200, 255}, CV_8UC1);
    const Mat b8u = make_vec_mat<uchar>({10, 9, 2, 2}, CV_8UC1);
    multiply(a8u, b8u, out);
    expect_vec_eq<uchar>(out, {200, 255, 255, 255});
    multiply(a8u, Scalar(3.0), out);
    expect_vec_eq<uchar>(out, {60, 90, 255, 255});

    out = Mat();
    const Mat a8s = make_vec_mat<schar>({50, -50, 100, -100}, CV_8SC1);
    const Mat b8s = make_vec_mat<schar>({3, 3, 2, 2}, CV_8SC1);
    multiply(a8s, b8s, out);
    expect_vec_eq<schar>(out, {127, -128, 127, -128});
    multiply(a8s, Scalar(-3.0), out);
    expect_vec_eq<schar>(out, {-128, 127, -128, 127});

    out = Mat();
    const Mat a16u = make_vec_mat<ushort>({1000, 40000, 60000, 65535}, CV_16UC1);
    const Mat b16u = make_vec_mat<ushort>({100, 2, 2, 2}, CV_16UC1);
    multiply(a16u, b16u, out);
    expect_vec_eq<ushort>(out, {65535, 65535, 65535, 65535});
    multiply(a16u, Scalar(2.0), out);
    expect_vec_eq<ushort>(out, {2000, 65535, 65535, 65535});

    out = Mat();
    const Mat a16s = make_vec_mat<short>({1000, -1000, 20000, -20000}, CV_16SC1);
    const Mat b16s = make_vec_mat<short>({40, 40, 2, 2}, CV_16SC1);
    multiply(a16s, b16s, out);
    expect_vec_eq<short>(out, {32767, -32768, 32767, -32768});
    multiply(a16s, Scalar(-3.0), out);
    expect_vec_eq<short>(out, {-3000, 3000, -32768, 32767});

    cpu::set_dispatch_mode(previous_mode);
}

TEST(BinaryOpContract_TEST, mat_mat_integer_compare_supports_scalar_only_mode)
{
    const auto previous_mode = cpu::dispatch_mode();
    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);

    Mat out;

    const Mat a8u = make_vec_mat<uchar>({1, 5, 9, 9}, CV_8UC1);
    const Mat b8u = make_vec_mat<uchar>({1, 3, 9, 10}, CV_8UC1);
    compare(a8u, b8u, out, CV_CMP_EQ);
    expect_vec_eq<uchar>(out, {255, 0, 255, 0});

    const Mat a16s = make_vec_mat<short>({-3, 4, 7, 7}, CV_16SC1);
    const Mat b16s = make_vec_mat<short>({-4, 4, 8, 6}, CV_16SC1);
    compare(a16s, b16s, out, CV_CMP_GT);
    expect_vec_eq<uchar>(out, {255, 0, 0, 255});

    const Mat a32u = make_vec_mat<uint>({3u, 4u, 5u, 6u}, CV_32UC1);
    const Mat b32u = make_vec_mat<uint>({3u, 7u, 4u, 6u}, CV_32UC1);
    compare(a32u, b32u, out, CV_CMP_LE);
    expect_vec_eq<uchar>(out, {255, 255, 0, 255});

    cpu::set_dispatch_mode(previous_mode);
}

TEST(BinaryOpContract_TEST, mat_scalar_integer_compare_supports_scalar_only_mode)
{
    const auto previous_mode = cpu::dispatch_mode();
    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);

    Mat out;

    const Mat src8s = make_vec_mat<schar>({-2, 0, 3, 7}, CV_8SC1);
    compare(src8s, Scalar(0.0), out, CV_CMP_GT);
    expect_vec_eq<uchar>(out, {0, 0, 255, 255});

    out = Mat();
    Mat src16u({1, 2}, CV_16UC3);
    src16u.at<ushort>(0, 0, 0) = 10;
    src16u.at<ushort>(0, 0, 1) = 20;
    src16u.at<ushort>(0, 0, 2) = 30;
    src16u.at<ushort>(0, 1, 0) = 40;
    src16u.at<ushort>(0, 1, 1) = 50;
    src16u.at<ushort>(0, 1, 2) = 60;
    compare(Scalar(10.0, 25.0, 30.0), src16u, out, CV_CMP_NE);
    EXPECT_EQ(out.at<uchar>(0, 0, 0), 0);
    EXPECT_EQ(out.at<uchar>(0, 0, 1), 255);
    EXPECT_EQ(out.at<uchar>(0, 0, 2), 0);
    EXPECT_EQ(out.at<uchar>(0, 1, 0), 255);
    EXPECT_EQ(out.at<uchar>(0, 1, 1), 255);
    EXPECT_EQ(out.at<uchar>(0, 1, 2), 255);

    out = Mat();
    const Mat src32s = make_vec_mat<int>({5, -1, 0, 9}, CV_32SC1);
    compare(src32s, Scalar(0.0), out, CV_CMP_GE);
    expect_vec_eq<uchar>(out, {255, 0, 255, 255});

    cpu::set_dispatch_mode(previous_mode);
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

TEST(BinaryOpContract_TEST, integral_ops_support_all_integral_depths)
{
    const int types[] = {CV_8UC1, CV_8SC1, CV_16UC1, CV_16SC1, CV_32SC1, CV_32UC1};
    for (const int type : types)
    {
        SCOPED_TRACE(type);
        Mat out;

        const Mat a = make_vec_mat_from_doubles({15.0, 3.0, 10.0}, type);
        const Mat b = make_vec_mat_from_doubles({1.0, 7.0, 12.0}, type);
        binaryFunc(BinaryOp::AND, a, b, out);
        expect_vec_match_by_depth(out, {1.0, 3.0, 8.0});
        binaryFunc(BinaryOp::OR, a, b, out);
        expect_vec_match_by_depth(out, {15.0, 7.0, 14.0});
        binaryFunc(BinaryOp::XOR, a, b, out);
        expect_vec_match_by_depth(out, {14.0, 4.0, 6.0});
        binaryFunc(BinaryOp::NOT, a, b, out);
        expect_vec_match_by_depth(out, {14.0, 0.0, 2.0});

        const Mat m0 = make_vec_mat_from_doubles({9.0, 10.0, 11.0}, type);
        const Mat m1 = make_vec_mat_from_doubles({4.0, 5.0, 2.0}, type);
        binaryFunc(BinaryOp::MOD, m0, m1, out);
        expect_vec_match_by_depth(out, {1.0, 0.0, 1.0});

        const Mat s0 = make_vec_mat_from_doubles({1.0, 2.0, 3.0}, type);
        const Mat s1 = make_vec_mat_from_doubles({1.0, 0.0, 2.0}, type);
        binaryFunc(BinaryOp::BITSHIFT, s0, s1, out);
        expect_vec_match_by_depth(out, {2.0, 2.0, 12.0});
    }
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

TEST(BinaryOpContract_TEST, compare_binary_ops_return_u8_mask_on_float32)
{
    const Mat a = make_vec_mat<float>({1.f, 2.f, -3.f, 4.f}, CV_32FC1);
    const Mat b = make_vec_mat<float>({1.f, 3.f, -5.f, 4.f}, CV_32FC1);
    Mat out;

    binaryFunc(BinaryOp::EQUAL, a, b, out);
    ASSERT_EQ(out.type(), CV_8UC1);
    expect_vec_eq<uchar>(out, {255, 0, 0, 255});

    binaryFunc(BinaryOp::GREATER, a, b, out);
    expect_vec_eq<uchar>(out, {0, 0, 255, 0});

    binaryFunc(BinaryOp::LESS_EQUAL, a, b, out);
    expect_vec_eq<uchar>(out, {255, 255, 0, 255});
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

TEST(BinaryOpContract_TEST, fmod_and_mean_support_float_and_half)
{
    const int types[] = {CV_32FC1, CV_16FC1};
    for (const int type : types)
    {
        SCOPED_TRACE(type);
        Mat out_fmod;
        Mat out_mean;

        const Mat a = make_vec_mat_from_doubles({5.0, -5.0, 5.0, -5.0}, type);
        const Mat b = make_vec_mat_from_doubles({2.0, 2.0, -2.0, -2.0}, type);
        binaryFunc(BinaryOp::FMOD, a, b, out_fmod);
        expect_vec_match_by_depth(out_fmod, {1.0, -1.0, 1.0, -1.0}, 1e-6, 2e-2);

        const Mat c = make_vec_mat_from_doubles({1.0, 2.5, -1.0}, type);
        const Mat d = make_vec_mat_from_doubles({3.0, -0.5, 1.0}, type);
        binaryFunc(BinaryOp::MEAN, c, d, out_mean);
        expect_vec_match_by_depth(out_mean, {2.0, 1.0, 0.0}, 1e-6, 2e-2);
    }
}

TEST(BinaryOpContract_TEST, mat_mat_add_fp16_supports_scalar_only_mode)
{
    const auto previous_mode = cpu::dispatch_mode();
    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);

    const Mat a = make_vec_mat_from_doubles({1.0, -2.0, 3.5, 4.0}, CV_16FC1);
    const Mat b = make_vec_mat_from_doubles({0.5, 1.0, -1.5, 2.0}, CV_16FC1);

    Mat out;
    binaryFunc(BinaryOp::ADD, a, b, out);
    expect_vec_match_by_depth(out, {1.5, -1.0, 2.0, 6.0}, 1e-6, 2e-2);

    cpu::set_dispatch_mode(previous_mode);
}

TEST(BinaryOpContract_TEST, mat_scalar_add_fp16_supports_scalar_only_mode)
{
    const auto previous_mode = cpu::dispatch_mode();
    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);

    const Mat src = make_vec_mat_from_doubles({1.0, -2.0, 3.5, 4.0}, CV_16FC1);

    Mat out;
    add(src, Scalar(0.5), out);
    expect_vec_match_by_depth(out, {1.5, -1.5, 4.0, 4.5}, 1e-6, 2e-2);

    cpu::set_dispatch_mode(previous_mode);
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

TEST(BinaryOpContract_TEST, binaryfunc_supports_multichannel_non_continuous_roi)
{
    Mat a_base({2, 5}, CV_32SC3);
    Mat b_base({2, 5}, CV_32SC3);
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 5; ++x)
        {
            for (int ch = 0; ch < 3; ++ch)
            {
                a_base.at<int>(y, x, ch) = 100 * y + 10 * x + ch;
                b_base.at<int>(y, x, ch) = 1 + y + x + ch;
            }
        }
    }

    Mat a = a_base.colRange(1, 4);
    Mat b = b_base.colRange(1, 4);
    ASSERT_FALSE(a.isContinuous());
    ASSERT_FALSE(b.isContinuous());

    Mat add_out;
    binaryFunc(BinaryOp::ADD, a, b, add_out);
    ASSERT_EQ(add_out.type(), CV_32SC3);
    ASSERT_EQ(add_out.shape(), a.shape());

    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 3; ++x)
        {
            const int src_x = x + 1;
            for (int ch = 0; ch < 3; ++ch)
            {
                const int lhs = 100 * y + 10 * src_x + ch;
                const int rhs = 1 + y + src_x + ch;
                EXPECT_EQ(add_out.at<int>(y, x, ch), lhs + rhs);
            }
        }
    }

    Mat eq_out;
    binaryFunc(BinaryOp::EQUAL, a, a, eq_out);
    ASSERT_EQ(eq_out.type(), CV_8UC3);
    ASSERT_EQ(eq_out.shape(), a.shape());
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 3; ++x)
        {
            for (int ch = 0; ch < 3; ++ch)
            {
                EXPECT_EQ(eq_out.at<uchar>(y, x, ch), static_cast<uchar>(255));
            }
        }
    }
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
