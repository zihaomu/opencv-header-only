#include "opencv_contract_backend.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>

namespace cvh_test_opencv_contract {
namespace {

std::uint32_t lcg_next(std::uint32_t state)
{
    return state * 1664525u + 1013904223u;
}

void fill_u8(cv::Mat& mat, std::uint32_t seed)
{
    std::uint32_t state = seed;
    const int scalars_per_row = mat.cols * mat.channels();
    for (int y = 0; y < mat.rows; ++y)
    {
        unsigned char* row = mat.ptr<unsigned char>(y);
        for (int x = 0; x < scalars_per_row; ++x)
        {
            state = lcg_next(state);
            row[x] = static_cast<unsigned char>((state >> 24) ^ static_cast<std::uint32_t>(x + y * 17));
        }
    }
}

void fill_f32(cv::Mat& mat, std::uint32_t seed)
{
    std::uint32_t state = seed;
    const int scalars_per_row = mat.cols * mat.channels();
    for (int y = 0; y < mat.rows; ++y)
    {
        float* row = mat.ptr<float>(y);
        for (int x = 0; x < scalars_per_row; ++x)
        {
            state = lcg_next(state);
            row[x] = static_cast<float>(static_cast<int>(state & 0xffffu) - 32768) / 4096.0f;
        }
    }
}

void fill_f64(cv::Mat& mat, std::uint32_t seed)
{
    std::uint32_t state = seed;
    const int scalars_per_row = mat.cols * mat.channels();
    for (int y = 0; y < mat.rows; ++y)
    {
        double* row = mat.ptr<double>(y);
        for (int x = 0; x < scalars_per_row; ++x)
        {
            state = lcg_next(state);
            row[x] = static_cast<double>(static_cast<int>(state & 0xffffu) - 32768) / 4096.0;
        }
    }
}

template<typename T>
void fill_integer(cv::Mat& mat, std::uint32_t seed)
{
    std::uint32_t state = seed;
    const int scalars_per_row = mat.cols * mat.channels();
    for (int y = 0; y < mat.rows; ++y)
    {
        T* row = mat.ptr<T>(y);
        for (int x = 0; x < scalars_per_row; ++x)
        {
            state = lcg_next(state);
            if constexpr (std::is_same<T, int>::value)
            {
                row[x] = static_cast<int>(state % 2000001u) - 1000000;
            }
            else
            {
                row[x] = static_cast<T>(state ^ (state >> 16));
            }
        }
    }
}

void fill_by_depth(cv::Mat& mat, CoreDepthId depth, std::uint32_t seed)
{
    switch (depth)
    {
        case CoreDepthId::U8: fill_u8(mat, seed); return;
        case CoreDepthId::S8: fill_integer<signed char>(mat, seed); return;
        case CoreDepthId::U16: fill_integer<unsigned short>(mat, seed); return;
        case CoreDepthId::S16: fill_integer<short>(mat, seed); return;
        case CoreDepthId::S32: fill_integer<int>(mat, seed); return;
        case CoreDepthId::F32: fill_f32(mat, seed); return;
        case CoreDepthId::F64: fill_f64(mat, seed); return;
    }
}

void fill_by_depth_contiguous(cv::Mat& mat,
                              CoreDepthId depth,
                              std::uint32_t seed)
{
    cv::Mat flat(
        1,
        static_cast<int>(mat.total()),
        CV_MAKETYPE(mat.depth(), mat.channels()),
        mat.data);
    fill_by_depth(flat, depth, seed);
}

int cv_depth(CoreDepthId depth)
{
    switch (depth)
    {
        case CoreDepthId::U8: return CV_8U;
        case CoreDepthId::S8: return CV_8S;
        case CoreDepthId::U16: return CV_16U;
        case CoreDepthId::S16: return CV_16S;
        case CoreDepthId::S32: return CV_32S;
        case CoreDepthId::F32: return CV_32F;
        case CoreDepthId::F64: return CV_64F;
    }
    return -1;
}

void run_core_array_op(CoreArrayOpId op,
                       const cv::Mat& a,
                       const cv::Mat& b,
                       cv::Mat& dst)
{
    switch (op)
    {
        case CoreArrayOpId::AbsDiff: cv::absdiff(a, b, dst); return;
        case CoreArrayOpId::BitwiseAnd: cv::bitwise_and(a, b, dst); return;
        case CoreArrayOpId::BitwiseNot: cv::bitwise_not(a, dst); return;
        case CoreArrayOpId::BitwiseOr: cv::bitwise_or(a, b, dst); return;
        case CoreArrayOpId::BitwiseXor: cv::bitwise_xor(a, b, dst); return;
        case CoreArrayOpId::InRange:
            cv::inRange(a, cv::Scalar::all(-2.5), cv::Scalar::all(3.5), dst);
            return;
        case CoreArrayOpId::Min: cv::min(a, b, dst); return;
        case CoreArrayOpId::Max: cv::max(a, b, dst); return;
    }
}

void fill_math_input(cv::Mat& mat, CoreMathOpId op, std::uint32_t seed)
{
    std::uint32_t state = seed;
    const int scalar_count = mat.cols * mat.channels();
    for (int y = 0; y < mat.rows; ++y)
    {
        for (int x = 0; x < scalar_count; ++x)
        {
            state = lcg_next(state);
            double value =
                static_cast<double>(static_cast<int>(state % 20001u) - 10000) / 2000.0;
            if (op == CoreMathOpId::Sqrt || op == CoreMathOpId::Log)
            {
                value = std::fabs(value) + 0.01;
            }
            if (mat.depth() == CV_32F)
            {
                mat.ptr<float>(y)[x] = static_cast<float>(value);
            }
            else
            {
                mat.ptr<double>(y)[x] = value;
            }
        }
    }
}

void run_core_math_op(CoreMathOpId op, const cv::Mat& src, cv::Mat& dst)
{
    switch (op)
    {
        case CoreMathOpId::Sqrt: cv::sqrt(src, dst); return;
        case CoreMathOpId::Pow: cv::pow(src, 1.75, dst); return;
        case CoreMathOpId::Exp: cv::exp(src, dst); return;
        case CoreMathOpId::Log: cv::log(src, dst); return;
    }
}

bool matches_bytes(const cv::Mat& expected,
                   const void* actual_data,
                   std::size_t actual_bytes,
                   int u8_tolerance)
{
    const std::size_t expected_bytes = expected.total() * expected.elemSize();
    if (!expected.isContinuous() || !actual_data || actual_bytes != expected_bytes)
    {
        return false;
    }

    if (expected.depth() != CV_8U)
    {
        if (std::memcmp(expected.data, actual_data, expected_bytes) == 0)
        {
            return true;
        }
        const unsigned char* actual = static_cast<const unsigned char*>(actual_data);
        for (std::size_t i = 0; i < expected_bytes; ++i)
        {
            if (expected.data[i] != actual[i])
            {
                std::cerr << "OpenCV byte mismatch at offset " << i
                          << ": expected=" << static_cast<int>(expected.data[i])
                          << " actual=" << static_cast<int>(actual[i]) << "\n";
                break;
            }
        }
        return false;
    }

    const unsigned char* actual = static_cast<const unsigned char*>(actual_data);
    for (std::size_t i = 0; i < expected_bytes; ++i)
    {
        if (std::abs(static_cast<int>(expected.data[i]) - static_cast<int>(actual[i])) > u8_tolerance)
        {
            std::cerr << "OpenCV U8 mismatch at offset " << i
                      << ": expected=" << static_cast<int>(expected.data[i])
                      << " actual=" << static_cast<int>(actual[i])
                      << " tolerance=" << u8_tolerance << "\n";
            return false;
        }
    }
    return true;
}

template<typename T>
void write_bits(T& value, const void* bits)
{
    std::memcpy(&value, bits, sizeof(T));
}

void fill_mask(cv::Mat& mask)
{
    for (int y = 0; y < mask.rows; ++y)
    {
        unsigned char* row = mask.ptr<unsigned char>(y);
        for (int x = 0; x < mask.cols; ++x)
        {
            row[x] = ((x + 2 * y) % 3) != 0 ? 255 : 0;
        }
    }
}

bool nearly_equal(double expected, double actual, double relative_tolerance)
{
    if (std::isnan(expected) || std::isnan(actual))
    {
        return std::isnan(expected) && std::isnan(actual);
    }
    if (std::isinf(expected) || std::isinf(actual))
    {
        return expected == actual;
    }
    const double tolerance =
        relative_tolerance * std::max(1.0, std::fabs(expected));
    return std::fabs(expected - actual) <= tolerance;
}

bool matches_f64_values(const cv::Mat& expected,
                        const void* actual_data,
                        std::size_t actual_bytes,
                        double relative_tolerance)
{
    const std::size_t expected_bytes = expected.total() * expected.elemSize();
    if (!expected.isContinuous() || expected.depth() != CV_64F ||
        actual_data == nullptr || actual_bytes != expected_bytes)
    {
        return false;
    }
    const std::size_t count =
        expected.total() * static_cast<std::size_t>(expected.channels());
    const double* expected_values = expected.ptr<double>();
    const double* actual_values = static_cast<const double*>(actual_data);
    for (std::size_t i = 0; i < count; ++i)
    {
        if (!nearly_equal(
                expected_values[i], actual_values[i], relative_tolerance))
        {
            std::cerr << "OpenCV f64 mismatch at scalar " << i
                      << ": expected=" << expected_values[i]
                      << " actual=" << actual_values[i] << "\n";
            return false;
        }
    }
    return true;
}

bool matches_float_values(const cv::Mat& expected,
                          const void* actual_data,
                          std::size_t actual_bytes,
                          double relative_tolerance)
{
    const std::size_t expected_bytes = expected.total() * expected.elemSize();
    if (!expected.isContinuous() || actual_data == nullptr ||
        actual_bytes != expected_bytes ||
        (expected.depth() != CV_32F && expected.depth() != CV_64F))
    {
        return false;
    }
    const std::size_t count =
        expected.total() * static_cast<std::size_t>(expected.channels());
    for (std::size_t i = 0; i < count; ++i)
    {
        const double expected_value =
            expected.depth() == CV_32F
                ? static_cast<double>(expected.ptr<float>()[i])
                : expected.ptr<double>()[i];
        const double actual_value =
            expected.depth() == CV_32F
                ? static_cast<double>(
                      static_cast<const float*>(actual_data)[i])
                : static_cast<const double*>(actual_data)[i];
        if (!nearly_equal(
                expected_value, actual_value, relative_tolerance))
        {
            std::cerr << "OpenCV floating output mismatch at scalar " << i
                      << ": expected=" << expected_value
                      << " actual=" << actual_value << "\n";
            return false;
        }
    }
    return true;
}

}  // namespace

bool validate_core_convert_u8_to_f64(int rows,
                                     int cols,
                                     int channels,
                                     std::uint32_t seed,
                                     const void* actual_data,
                                     std::size_t actual_bytes)
{
    cv::Mat src(rows, cols, CV_MAKETYPE(CV_8U, channels));
    fill_u8(src, seed);

    cv::Mat expected;
    src.convertTo(expected, CV_MAKETYPE(CV_64F, channels));
    return matches_bytes(expected, actual_data, actual_bytes, 0);
}

bool validate_core_array_op(CoreArrayOpId op,
                            CoreDepthId depth,
                            int rows,
                            int cols,
                            int channels,
                            std::uint32_t seed_a,
                            std::uint32_t seed_b,
                            const void* actual_data,
                            std::size_t actual_bytes)
{
    const int type = CV_MAKETYPE(cv_depth(depth), channels);
    cv::Mat a(rows, cols, type);
    cv::Mat b(rows, cols, type);
    fill_by_depth(a, depth, seed_a);
    fill_by_depth(b, depth, seed_b);

    cv::Mat expected;
    run_core_array_op(op, a, b, expected);
    return matches_bytes(expected, actual_data, actual_bytes, 0);
}

bool validate_core_float_edge_op(CoreArrayOpId op,
                                 const void* actual_data,
                                 std::size_t actual_bytes)
{
    cv::Mat a(1, 5, CV_32FC1);
    cv::Mat b(1, 5, CV_32FC1);
    const std::uint32_t a_bits[] = {
        0x7fc12345u, 0x3f800000u, 0x7f800000u, 0x80000000u, 0x00000000u,
    };
    const std::uint32_t b_bits[] = {
        0x40000000u, 0x7fc54321u, 0x7f800000u, 0x00000000u, 0x80000000u,
    };
    for (int x = 0; x < 5; ++x)
    {
        write_bits(a.at<float>(0, x), &a_bits[x]);
        write_bits(b.at<float>(0, x), &b_bits[x]);
    }

    cv::Mat expected;
    run_core_array_op(op, a, b, expected);
    return matches_bytes(expected, actual_data, actual_bytes, 0);
}

bool validate_core_double_edge_op(CoreArrayOpId op,
                                  const void* actual_data,
                                  std::size_t actual_bytes)
{
    cv::Mat a(1, 5, CV_64FC1);
    cv::Mat b(1, 5, CV_64FC1);
    const std::uint64_t a_bits[] = {
        0x7ff8123456789abcULL,
        0x3ff0000000000000ULL,
        0x7ff0000000000000ULL,
        0x8000000000000000ULL,
        0x0000000000000000ULL,
    };
    const std::uint64_t b_bits[] = {
        0x4000000000000000ULL,
        0x7ff854321abcdef0ULL,
        0x7ff0000000000000ULL,
        0x0000000000000000ULL,
        0x8000000000000000ULL,
    };
    for (int x = 0; x < 5; ++x)
    {
        write_bits(a.at<double>(0, x), &a_bits[x]);
        write_bits(b.at<double>(0, x), &b_bits[x]);
    }

    cv::Mat expected;
    run_core_array_op(op, a, b, expected);
    return matches_bytes(expected, actual_data, actual_bytes, 0);
}

bool validate_convert_scale_abs_edges(const void* actual_data,
                                      std::size_t actual_bytes)
{
    cv::Mat src(1, 9, CV_32FC1);
    const float values[] = {
        -300.0f, -2.5f, -1.5f, -0.5f, 0.5f, 1.5f, 2.5f, 254.5f, 300.0f,
    };
    std::memcpy(src.data, values, sizeof(values));

    cv::Mat expected;
    cv::convertScaleAbs(src, expected);
    return matches_bytes(expected, actual_data, actual_bytes, 0);
}

bool validate_convert_fp16_edges(const void* actual_data,
                                 std::size_t actual_bytes)
{
    cv::Mat src(1, 11, CV_32FC1);
    const float denorm = std::ldexp(1.0f, -24);
    const float values[] = {
        0.0f,
        -0.0f,
        1.0f,
        -2.0f,
        65504.0f,
        std::ldexp(1.0f, -14),
        denorm,
        denorm * 0.25f,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
    };
    std::memcpy(src.data, values, sizeof(values));

    cv::Mat expected;
    cv::convertFp16(src, expected);
    return matches_bytes(expected, actual_data, actual_bytes, 0);
}

bool validate_core_math_op(CoreMathOpId op,
                           CoreDepthId depth,
                           int rows,
                           int cols,
                           int channels,
                           std::uint32_t seed,
                           const void* actual_data,
                           std::size_t actual_bytes)
{
    if (depth != CoreDepthId::F32 && depth != CoreDepthId::F64)
    {
        return false;
    }
    const int type = CV_MAKETYPE(cv_depth(depth), channels);
    cv::Mat src(rows, cols, type);
    fill_math_input(src, op, seed);

    cv::Mat expected;
    run_core_math_op(op, src, expected);
    const std::size_t expected_bytes = expected.total() * expected.elemSize();
    if (!actual_data || actual_bytes != expected_bytes)
    {
        return false;
    }

    const std::size_t count = expected.total() * static_cast<std::size_t>(channels);
    const double relative_tolerance =
        depth == CoreDepthId::F32
            ? (op == CoreMathOpId::Exp ? 7e-6 : 2e-6)
            : (op == CoreMathOpId::Exp ? 1e-10 : 2e-12);
    for (std::size_t i = 0; i < count; ++i)
    {
        const double expected_value =
            depth == CoreDepthId::F32
                ? static_cast<double>(expected.ptr<float>()[i])
                : expected.ptr<double>()[i];
        const double actual_value =
            depth == CoreDepthId::F32
                ? static_cast<double>(static_cast<const float*>(actual_data)[i])
                : static_cast<const double*>(actual_data)[i];
        if (std::isnan(expected_value) || std::isnan(actual_value))
        {
            if (!(std::isnan(expected_value) && std::isnan(actual_value)))
            {
                std::cerr << "OpenCV math NaN category mismatch at scalar " << i
                          << ": expected=" << expected_value
                          << " actual=" << actual_value << "\n";
                return false;
            }
            continue;
        }
        if (std::isinf(expected_value) || std::isinf(actual_value))
        {
            if (expected_value != actual_value)
            {
                std::cerr << "OpenCV math Inf category mismatch at scalar " << i
                          << ": expected=" << expected_value
                          << " actual=" << actual_value << "\n";
                return false;
            }
            continue;
        }
        const double tolerance =
            relative_tolerance * std::max(1.0, std::fabs(expected_value));
        if (std::fabs(expected_value - actual_value) > tolerance)
        {
            std::cerr << "OpenCV math mismatch at scalar " << i
                      << ": expected=" << expected_value
                      << " actual=" << actual_value
                      << " tolerance=" << tolerance << "\n";
            return false;
        }
    }
    return true;
}

bool validate_core_reduction_summary(CoreDepthId depth,
                                     int rows,
                                     int cols,
                                     int channels,
                                     std::uint32_t seed,
                                     bool use_mask,
                                     const CoreReductionSummary& actual)
{
    const int type = CV_MAKETYPE(cv_depth(depth), channels);
    cv::Mat src(rows, cols, type);
    fill_by_depth(src, depth, seed);
    cv::Mat mask;
    if (use_mask)
    {
        mask.create(rows, cols, CV_8UC1);
        fill_mask(mask);
    }

    const cv::Scalar expected_sums = cv::sum(src);
    const cv::Scalar expected_means = cv::mean(src, mask);
    cv::Scalar expected_stddev;
    cv::Scalar mean_from_stddev;
    cv::meanStdDev(src, mean_from_stddev, expected_stddev, mask);
    const double relative_tolerance =
        depth == CoreDepthId::F32 ? 2e-6 : 2e-12;
    for (int ch = 0; ch < channels; ++ch)
    {
        if (!nearly_equal(
                expected_sums[ch], actual.sums[ch], relative_tolerance) ||
            !nearly_equal(
                expected_means[ch], actual.means[ch], relative_tolerance) ||
            !nearly_equal(
                expected_stddev[ch], actual.stddevs[ch], relative_tolerance))
        {
            std::cerr << "OpenCV reduction summary mismatch at channel " << ch
                      << "\n";
            return false;
        }
    }

    if (!nearly_equal(
            cv::norm(src, cv::NORM_INF, mask),
            actual.norm_inf,
            relative_tolerance) ||
        !nearly_equal(
            cv::norm(src, cv::NORM_L1, mask),
            actual.norm_l1,
            relative_tolerance) ||
        !nearly_equal(
            cv::norm(src, cv::NORM_L2, mask),
            actual.norm_l2,
            relative_tolerance))
    {
        std::cerr << "OpenCV norm summary mismatch\n";
        return false;
    }

    if (channels == 1)
    {
        double expected_min = 0.0;
        double expected_max = 0.0;
        cv::Point expected_min_location;
        cv::Point expected_max_location;
        cv::minMaxLoc(
            src,
            &expected_min,
            &expected_max,
            &expected_min_location,
            &expected_max_location,
            mask);
        if (!nearly_equal(
                expected_min, actual.min_value, relative_tolerance) ||
            !nearly_equal(
                expected_max, actual.max_value, relative_tolerance) ||
            expected_min_location.x != actual.min_x ||
            expected_min_location.y != actual.min_y ||
            expected_max_location.x != actual.max_x ||
            expected_max_location.y != actual.max_y ||
            cv::countNonZero(src) != actual.count_non_zero)
        {
            std::cerr << "OpenCV single-channel reduction metadata mismatch\n";
            return false;
        }
    }
    return true;
}

bool validate_core_nonzero_locations(CoreDepthId depth,
                                     int rows,
                                     int cols,
                                     std::uint32_t seed,
                                     const int* actual_xy,
                                     std::size_t actual_count)
{
    cv::Mat src(rows, cols, CV_MAKETYPE(cv_depth(depth), 1));
    fill_by_depth(src, depth, seed);
    cv::Mat expected;
    cv::findNonZero(src, expected);
    if (expected.total() != actual_count ||
        (actual_count != 0 && actual_xy == nullptr))
    {
        return false;
    }
    for (std::size_t i = 0; i < actual_count; ++i)
    {
        const cv::Point point = expected.at<cv::Point>(static_cast<int>(i), 0);
        if (point.x != actual_xy[2 * i] || point.y != actual_xy[2 * i + 1])
        {
            std::cerr << "OpenCV findNonZero mismatch at index " << i << "\n";
            return false;
        }
    }
    return true;
}

bool validate_core_reduce_f64(CoreDepthId depth,
                              int rows,
                              int cols,
                              int channels,
                              std::uint32_t seed,
                              int dim,
                              int reduce_type,
                              const void* actual_data,
                              std::size_t actual_bytes)
{
    cv::Mat src(rows, cols, CV_MAKETYPE(cv_depth(depth), channels));
    fill_by_depth(src, depth, seed);
    cv::Mat expected;
    cv::reduce(src, expected, dim, reduce_type, CV_64F);
    const double tolerance =
        depth == CoreDepthId::F32 ? 2e-6 : 2e-12;
    return matches_f64_values(
        expected, actual_data, actual_bytes, tolerance);
}

bool validate_core_reduce_arg(CoreDepthId depth,
                              int rows,
                              int cols,
                              std::uint32_t seed,
                              int axis,
                              bool find_max,
                              bool last_index,
                              const void* actual_data,
                              std::size_t actual_bytes)
{
    cv::Mat src(rows, cols, CV_MAKETYPE(cv_depth(depth), 1));
    fill_by_depth(src, depth, seed);
    cv::Mat expected;
    if (find_max)
    {
        cv::reduceArgMax(src, expected, axis, last_index);
    }
    else
    {
        cv::reduceArgMin(src, expected, axis, last_index);
    }
    return matches_bytes(expected, actual_data, actual_bytes, 0);
}

bool validate_core_normalize_l2_f64(CoreDepthId depth,
                                    int rows,
                                    int cols,
                                    int channels,
                                    std::uint32_t seed,
                                    bool use_mask,
                                    const void* actual_data,
                                    std::size_t actual_bytes)
{
    cv::Mat src(rows, cols, CV_MAKETYPE(cv_depth(depth), channels));
    fill_by_depth(src, depth, seed);
    cv::Mat mask;
    if (use_mask)
    {
        mask.create(rows, cols, CV_8UC1);
        fill_mask(mask);
    }
    cv::Mat expected;
    cv::normalize(src, expected, 2.0, 0.0, cv::NORM_L2, CV_64F, mask);
    const double tolerance =
        depth == CoreDepthId::F32 ? 2e-6 : 2e-12;
    return matches_f64_values(
        expected, actual_data, actual_bytes, tolerance);
}

bool validate_core_border_interpolate(int p,
                                      int len,
                                      int border_type,
                                      int actual)
{
    return cv::borderInterpolate(p, len, border_type) == actual;
}

bool validate_core_layout_op(CoreLayoutOpId op,
                             CoreDepthId depth,
                             int rows,
                             int cols,
                             int channels,
                             std::uint32_t seed,
                             const void* actual_data,
                             std::size_t actual_bytes)
{
    const int type = CV_MAKETYPE(cv_depth(depth), channels);
    cv::Mat src(rows, cols, type);
    fill_by_depth(src, depth, seed);
    cv::Mat expected;
    switch (op)
    {
        case CoreLayoutOpId::CopyMask:
        {
            cv::Mat mask(rows, cols, CV_8UC1);
            fill_mask(mask);
            src.copyTo(expected, mask);
            break;
        }
        case CoreLayoutOpId::ExtractLastChannel:
            cv::extractChannel(src, expected, channels - 1);
            break;
        case CoreLayoutOpId::FlipHorizontal:
            cv::flip(src, expected, 1);
            break;
        case CoreLayoutOpId::FlipVertical:
            cv::flip(src, expected, 0);
            break;
        case CoreLayoutOpId::FlipBoth:
            cv::flip(src, expected, -1);
            break;
        case CoreLayoutOpId::RotateClockwise:
            cv::rotate(src, expected, cv::ROTATE_90_CLOCKWISE);
            break;
        case CoreLayoutOpId::Rotate180:
            cv::rotate(src, expected, cv::ROTATE_180);
            break;
        case CoreLayoutOpId::RotateCounterclockwise:
            cv::rotate(src, expected, cv::ROTATE_90_COUNTERCLOCKWISE);
            break;
        case CoreLayoutOpId::Repeat2x3:
            cv::repeat(src, 2, 3, expected);
            break;
        case CoreLayoutOpId::HConcat:
        {
            cv::Mat other(rows, cols, type);
            fill_by_depth(other, depth, seed ^ 0x9e3779b9u);
            cv::hconcat(src, other, expected);
            break;
        }
        case CoreLayoutOpId::VConcat:
        {
            cv::Mat other(rows, cols, type);
            fill_by_depth(other, depth, seed ^ 0x9e3779b9u);
            cv::vconcat(src, other, expected);
            break;
        }
    }
    return matches_bytes(expected, actual_data, actual_bytes, 0);
}

bool validate_core_mix_channels(CoreDepthId depth,
                                int rows,
                                int cols,
                                std::uint32_t seed,
                                const void* actual_bgr,
                                std::size_t actual_bgr_bytes,
                                const void* actual_alpha,
                                std::size_t actual_alpha_bytes)
{
    const int cv_depth_value = cv_depth(depth);
    cv::Mat src(rows, cols, CV_MAKETYPE(cv_depth_value, 4));
    fill_by_depth(src, depth, seed);
    cv::Mat bgr(rows, cols, CV_MAKETYPE(cv_depth_value, 3));
    cv::Mat alpha(rows, cols, CV_MAKETYPE(cv_depth_value, 1));
    cv::Mat outputs[] = {bgr, alpha};
    const int routes[] = {0, 2, 1, 1, 2, 0, 3, 3};
    cv::mixChannels(&src, 1, outputs, 2, routes, 4);
    return matches_bytes(
               outputs[0], actual_bgr, actual_bgr_bytes, 0) &&
           matches_bytes(
               outputs[1], actual_alpha, actual_alpha_bytes, 0);
}

bool validate_core_flip_nd(CoreDepthId depth,
                           std::uint32_t seed,
                           int axis,
                           const void* actual_data,
                           std::size_t actual_bytes)
{
    const int sizes[] = {2, 3, 4};
    cv::Mat src(3, sizes, CV_MAKETYPE(cv_depth(depth), 1));
    fill_by_depth_contiguous(src, depth, seed);
    cv::Mat expected;
    cv::flipND(src, expected, axis);
    return matches_bytes(expected, actual_data, actual_bytes, 0);
}

bool validate_core_broadcast(CoreDepthId depth,
                             std::uint32_t seed,
                             const void* actual_data,
                             std::size_t actual_bytes)
{
    const int source_sizes[] = {2, 1, 3};
    cv::Mat src(3, source_sizes, CV_MAKETYPE(cv_depth(depth), 1));
    fill_by_depth_contiguous(src, depth, seed);
    const int target_values[] = {4, 2, 5, 3};
    cv::Mat target_shape(
        1, 4, CV_32SC1, const_cast<int*>(target_values));
    cv::Mat expected;
    cv::broadcast(src, target_shape, expected);
    return matches_bytes(expected, actual_data, actual_bytes, 0);
}

bool validate_imgproc_structuring_element(int shape,
                                          int width,
                                          int height,
                                          int anchor_x,
                                          int anchor_y,
                                          const void* actual_data,
                                          std::size_t actual_bytes)
{
    const cv::Mat expected = cv::getStructuringElement(
        shape, cv::Size(width, height), cv::Point(anchor_x, anchor_y));
    return matches_bytes(expected, actual_data, actual_bytes, 0);
}

bool validate_imgproc_gaussian_kernel(int ksize,
                                      double sigma,
                                      CoreDepthId depth,
                                      const void* actual_data,
                                      std::size_t actual_bytes)
{
    const cv::Mat expected =
        cv::getGaussianKernel(ksize, sigma, cv_depth(depth));
    return matches_float_values(
        expected,
        actual_data,
        actual_bytes,
        depth == CoreDepthId::F32 ? 2e-7 : 2e-14);
}

bool validate_imgproc_deriv_kernels(int dx,
                                    int dy,
                                    int ksize,
                                    bool normalize,
                                    CoreDepthId depth,
                                    const void* actual_kx,
                                    std::size_t actual_kx_bytes,
                                    const void* actual_ky,
                                    std::size_t actual_ky_bytes)
{
    cv::Mat expected_x;
    cv::Mat expected_y;
    cv::getDerivKernels(
        expected_x,
        expected_y,
        dx,
        dy,
        ksize,
        normalize,
        cv_depth(depth));
    const double tolerance = depth == CoreDepthId::F32 ? 1e-7 : 1e-15;
    return matches_float_values(
               expected_x, actual_kx, actual_kx_bytes, tolerance) &&
           matches_float_values(
               expected_y, actual_ky, actual_ky_bytes, tolerance);
}

bool validate_imgproc_gabor_kernel(int width,
                                   int height,
                                   CoreDepthId depth,
                                   const void* actual_data,
                                   std::size_t actual_bytes)
{
    const cv::Mat expected = cv::getGaborKernel(
        cv::Size(width, height),
        2.0,
        0.3,
        4.0,
        0.8,
        0.0,
        cv_depth(depth));
    return matches_float_values(
        expected,
        actual_data,
        actual_bytes,
        depth == CoreDepthId::F32 ? 2e-6 : 2e-13);
}

bool validate_imgproc_hanning_window(int width,
                                     int height,
                                     CoreDepthId depth,
                                     const void* actual_data,
                                     std::size_t actual_bytes)
{
    cv::Mat expected;
    cv::createHanningWindow(
        expected, cv::Size(width, height), cv_depth(depth));
    return matches_float_values(
        expected,
        actual_data,
        actual_bytes,
        depth == CoreDepthId::F32 ? 2e-6 : 2e-13);
}

bool validate_imgproc_integral_u8(int rows,
                                  int cols,
                                  int channels,
                                  std::uint32_t seed,
                                  CoreDepthId output_depth,
                                  const void* actual_data,
                                  std::size_t actual_bytes)
{
    cv::Mat src(rows, cols, CV_MAKETYPE(CV_8U, channels));
    fill_u8(src, seed);
    cv::Mat expected;
    cv::integral(src, expected, cv_depth(output_depth));
    return matches_bytes(expected, actual_data, actual_bytes, 0);
}

bool validate_imgproc_derivative_filters_u8(int rows,
                                            int cols,
                                            int channels,
                                            std::uint32_t seed,
                                            const void* actual_scharr,
                                            std::size_t actual_scharr_bytes,
                                            const void* actual_laplacian,
                                            std::size_t actual_laplacian_bytes)
{
    cv::Mat src(rows, cols, CV_MAKETYPE(CV_8U, channels));
    fill_u8(src, seed);
    cv::Mat expected_scharr;
    cv::Mat expected_laplacian;
    cv::Scharr(
        src, expected_scharr, CV_16S, 1, 0, 1.0, 0.0, cv::BORDER_REFLECT_101);
    cv::Laplacian(
        src, expected_laplacian, CV_16S, 3, 1.0, 0.0, cv::BORDER_REFLECT_101);
    return matches_bytes(
               expected_scharr,
               actual_scharr,
               actual_scharr_bytes,
               0) &&
           matches_bytes(
               expected_laplacian,
               actual_laplacian,
               actual_laplacian_bytes,
               0);
}

bool validate_imgproc_spatial_gradient_u8(int rows,
                                          int cols,
                                          std::uint32_t seed,
                                          const void* actual_dx,
                                          std::size_t actual_dx_bytes,
                                          const void* actual_dy,
                                          std::size_t actual_dy_bytes)
{
    cv::Mat src(rows, cols, CV_8UC1);
    fill_u8(src, seed);
    cv::Mat expected_x;
    cv::Mat expected_y;
    cv::spatialGradient(
        src, expected_x, expected_y, 3, cv::BORDER_REFLECT_101);
    return matches_bytes(expected_x, actual_dx, actual_dx_bytes, 0) &&
           matches_bytes(expected_y, actual_dy, actual_dy_bytes, 0);
}

bool validate_imgproc_sqr_box_filter_u8(int rows,
                                        int cols,
                                        int channels,
                                        std::uint32_t seed,
                                        const void* actual_data,
                                        std::size_t actual_bytes)
{
    cv::Mat src(rows, cols, CV_MAKETYPE(CV_8U, channels));
    fill_u8(src, seed);
    cv::Mat expected;
    cv::sqrBoxFilter(
        src,
        expected,
        CV_64F,
        cv::Size(7, 5),
        cv::Point(-1, -1),
        true,
        cv::BORDER_REFLECT_101);
    return matches_float_values(
        expected, actual_data, actual_bytes, 2e-13);
}

bool validate_imgproc_intensity_u8(ImgprocIntensityOpId op,
                                   int rows,
                                   int cols,
                                   std::uint32_t seed,
                                   const void* actual_data,
                                   std::size_t actual_bytes)
{
    const bool color_source =
        op == ImgprocIntensityOpId::BilateralU8 ||
        op == ImgprocIntensityOpId::StackBlurU8;
    cv::Mat src(
        rows,
        cols,
        color_source ? CV_8UC3 : CV_8UC1);
    fill_u8(src, seed);
    cv::Mat expected;
    int tolerance = 0;
    switch (op)
    {
        case ImgprocIntensityOpId::MedianU8:
            cv::medianBlur(src, expected, 5);
            break;
        case ImgprocIntensityOpId::BilateralU8:
            cv::bilateralFilter(
                src,
                expected,
                5,
                35.0,
                2.0,
                cv::BORDER_REFLECT_101);
            tolerance = 1;
            break;
        case ImgprocIntensityOpId::StackBlurU8:
            cv::stackBlur(src, expected, cv::Size(5, 3));
            tolerance = 2;
            break;
        case ImgprocIntensityOpId::AdaptiveMeanU8:
            cv::adaptiveThreshold(
                src,
                expected,
                200.0,
                cv::ADAPTIVE_THRESH_MEAN_C,
                cv::THRESH_BINARY,
                5,
                2.25);
            tolerance = 1;
            break;
        case ImgprocIntensityOpId::AdaptiveGaussianU8:
            cv::adaptiveThreshold(
                src,
                expected,
                200.0,
                cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                cv::THRESH_BINARY_INV,
                5,
                -1.25);
            tolerance = 1;
            break;
        case ImgprocIntensityOpId::ThresholdMaskU8:
        {
            cv::Mat mask(rows, cols, CV_8UC1);
            fill_mask(mask);
            expected.create(src.size(), src.type());
            expected.setTo(cv::Scalar::all(17));
            cv::thresholdWithMask(
                src,
                expected,
                mask,
                110.0,
                200.0,
                cv::THRESH_BINARY);
            break;
        }
        case ImgprocIntensityOpId::EqualizeHistU8:
            cv::equalizeHist(src, expected);
            break;
        case ImgprocIntensityOpId::ColorMapJetU8:
            cv::applyColorMap(src, expected, cv::COLORMAP_JET);
            tolerance = 1;
            break;
        case ImgprocIntensityOpId::ColorMapUserU8:
        {
            cv::Mat lookup(256, 1, CV_8UC3);
            for (int i = 0; i < 256; ++i)
            {
                lookup.at<cv::Vec3b>(i, 0) =
                    cv::Vec3b(
                        static_cast<uchar>(i),
                        static_cast<uchar>(255 - i),
                        17);
            }
            cv::applyColorMap(src, expected, lookup);
            break;
        }
    }
    return matches_bytes(
        expected, actual_data, actual_bytes, tolerance);
}

bool validate_imgproc_pyramid_color_u8(ImgprocPyramidColorOpId op,
                                       int rows,
                                       int cols,
                                       std::uint32_t seed,
                                       const void* actual_data,
                                       std::size_t actual_bytes)
{
    const bool demosaic =
        op == ImgprocPyramidColorOpId::DemosaicBgU8 ||
        op == ImgprocPyramidColorOpId::DemosaicGbU8 ||
        op == ImgprocPyramidColorOpId::DemosaicRgU8 ||
        op == ImgprocPyramidColorOpId::DemosaicGrU8;
    cv::Mat src(
        rows, cols, demosaic ? CV_8UC1 : CV_8UC3);
    fill_u8(src, seed);
    cv::Mat expected;
    int tolerance = 0;
    switch (op)
    {
        case ImgprocPyramidColorOpId::AccumulateU8:
        case ImgprocPyramidColorOpId::AccumulateSquareU8:
        case ImgprocPyramidColorOpId::AccumulateProductU8:
        case ImgprocPyramidColorOpId::AccumulateWeightedU8:
        {
            cv::Mat mask(rows, cols, CV_8UC1);
            fill_mask(mask);
            expected = cv::Mat(rows, cols, CV_32FC3, cv::Scalar::all(1.0));
            if (op == ImgprocPyramidColorOpId::AccumulateU8)
            {
                cv::accumulate(src, expected);
            }
            else if (op == ImgprocPyramidColorOpId::AccumulateSquareU8)
            {
                cv::accumulateSquare(src, expected, mask);
            }
            else if (op == ImgprocPyramidColorOpId::AccumulateProductU8)
            {
                cv::Mat second(rows, cols, CV_8UC3);
                fill_u8(second, seed + 17u);
                cv::accumulateProduct(src, second, expected, mask);
            }
            else
            {
                cv::accumulateWeighted(src, expected, 0.375, mask);
            }
            return matches_float_values(
                expected, actual_data, actual_bytes, 1e-6);
        }
        case ImgprocPyramidColorOpId::BlendLinearU8:
        {
            cv::Mat second(rows, cols, CV_8UC3);
            fill_u8(second, seed + 17u);
            cv::Mat weight1(rows, cols, CV_32FC1);
            cv::Mat weight2(rows, cols, CV_32FC1);
            for (int y = 0; y < rows; ++y)
            {
                for (int x = 0; x < cols; ++x)
                {
                    weight1.at<float>(y, x) =
                        static_cast<float>((x + y) % 5) * 0.25f;
                    weight2.at<float>(y, x) =
                        static_cast<float>((2 * x + y + 1) % 7) * 0.2f;
                }
            }
            cv::blendLinear(src, second, weight1, weight2, expected);
            tolerance = 1;
            break;
        }
        case ImgprocPyramidColorOpId::PyrDownU8:
            cv::pyrDown(src, expected);
            tolerance = 1;
            break;
        case ImgprocPyramidColorOpId::PyrUpU8:
            cv::pyrUp(src, expected);
            tolerance = 1;
            break;
        case ImgprocPyramidColorOpId::TwoPlaneNv12U8:
        case ImgprocPyramidColorOpId::TwoPlaneNv21U8:
        {
            cv::Mat y_plane(rows, cols, CV_8UC1);
            cv::Mat uv_plane(rows / 2, cols / 2, CV_8UC2);
            fill_u8(y_plane, seed);
            fill_u8(uv_plane, seed + 17u);
            cv::cvtColorTwoPlane(
                y_plane,
                uv_plane,
                expected,
                op == ImgprocPyramidColorOpId::TwoPlaneNv12U8
                    ? cv::COLOR_YUV2BGR_NV12
                    : cv::COLOR_YUV2RGB_NV21);
            tolerance = 1;
            break;
        }
        case ImgprocPyramidColorOpId::DemosaicBgU8:
            cv::demosaicing(src, expected, cv::COLOR_BayerBG2BGR);
            tolerance = 1;
            break;
        case ImgprocPyramidColorOpId::DemosaicGbU8:
            cv::demosaicing(src, expected, cv::COLOR_BayerGB2BGR);
            tolerance = 1;
            break;
        case ImgprocPyramidColorOpId::DemosaicRgU8:
            cv::demosaicing(src, expected, cv::COLOR_BayerRG2BGR);
            tolerance = 1;
            break;
        case ImgprocPyramidColorOpId::DemosaicGrU8:
            cv::demosaicing(src, expected, cv::COLOR_BayerGR2BGR);
            tolerance = 1;
            break;
    }
    return matches_bytes(
        expected, actual_data, actual_bytes, tolerance);
}

bool validate_imgproc_resize_linear_u8(int src_rows,
                                       int src_cols,
                                       int dst_rows,
                                       int dst_cols,
                                       int channels,
                                       std::uint32_t seed,
                                       const void* actual_data,
                                       std::size_t actual_bytes)
{
    cv::Mat src(src_rows, src_cols, CV_MAKETYPE(CV_8U, channels));
    fill_u8(src, seed);

    cv::Mat expected;
    cv::resize(src, expected, cv::Size(dst_cols, dst_rows), 0.0, 0.0, cv::INTER_LINEAR);
    return matches_bytes(expected, actual_data, actual_bytes, 1);
}

bool validate_imgproc_geometry_matrices(const double* actual_rotation,
                                        const double* actual_affine,
                                        const double* actual_perspective,
                                        const double* actual_inverse)
{
    const auto matches = [](const double* expected,
                            const double* actual,
                            std::size_t count,
                            double tolerance) {
        for (std::size_t index = 0; index < count; ++index)
        {
            if (std::abs(expected[index] - actual[index]) > tolerance)
            {
                std::cerr
                    << "geometry matrix mismatch at " << index
                    << ": expected=" << expected[index]
                    << ", actual=" << actual[index] << '\n';
                return false;
            }
        }
        return true;
    };

    const cv::Matx23d rotation =
        cv::getRotationMatrix2D_(
            cv::Point2f(12.5f, -7.25f),
            37.0,
            0.75);
    const cv::Point2f affine_source[] = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(4.0f, 0.0f),
        cv::Point2f(0.0f, 3.0f)};
    const cv::Point2f affine_target[] = {
        cv::Point2f(5.0f, -2.0f),
        cv::Point2f(13.0f, -6.0f),
        cv::Point2f(6.5f, 7.0f)};
    const cv::Mat affine =
        cv::getAffineTransform(affine_source, affine_target);

    const cv::Point2f perspective_source[] = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(8.0f, 0.0f),
        cv::Point2f(8.0f, 6.0f),
        cv::Point2f(0.0f, 6.0f)};
    const cv::Point2f perspective_target[] = {
        cv::Point2f(1.0f, 2.0f),
        cv::Point2f(9.0f, 1.0f),
        cv::Point2f(7.5f, 8.0f),
        cv::Point2f(-0.5f, 6.5f)};
    const cv::Mat perspective =
        cv::getPerspectiveTransform(
            perspective_source,
            perspective_target);

    cv::Mat inverse;
    cv::invertAffineTransform(affine, inverse);
    return matches(rotation.val, actual_rotation, 6, 1e-12) &&
        matches(affine.ptr<double>(), actual_affine, 6, 1e-12) &&
        matches(
            perspective.ptr<double>(),
            actual_perspective,
            9,
            1e-10) &&
        matches(inverse.ptr<double>(), actual_inverse, 6, 1e-12);
}

bool validate_imgproc_geometry_sampling(
    ImgprocGeometrySamplingOpId op,
    std::uint32_t seed,
    const void* actual_data,
    std::size_t actual_bytes)
{
    cv::Mat source(9, 11, CV_8UC3);
    fill_u8(source, seed);
    cv::Mat expected;
    switch (op)
    {
        case ImgprocGeometrySamplingOpId::RemapFloatU8:
        case ImgprocGeometrySamplingOpId::RemapFixedU8:
        {
            cv::Mat map_x(7, 9, CV_32FC1);
            cv::Mat map_y(7, 9, CV_32FC1);
            for (int row = 0; row < map_x.rows; ++row)
            {
                for (int col = 0; col < map_x.cols; ++col)
                {
                    map_x.at<float>(row, col) =
                        static_cast<float>(col) + 0.28125f;
                    map_y.at<float>(row, col) =
                        static_cast<float>(row) - 0.34375f;
                }
            }
            if (op == ImgprocGeometrySamplingOpId::RemapFloatU8)
            {
                cv::remap(
                    source,
                    expected,
                    map_x,
                    map_y,
                    cv::INTER_LINEAR,
                    cv::BORDER_REFLECT_101);
            }
            else
            {
                cv::Mat fixed_coordinates;
                cv::Mat fixed_fractions;
                cv::convertMaps(
                    map_x,
                    map_y,
                    fixed_coordinates,
                    fixed_fractions,
                    CV_16SC2);
                cv::remap(
                    source,
                    expected,
                    fixed_coordinates,
                    fixed_fractions,
                    cv::INTER_LINEAR,
                    cv::BORDER_REFLECT_101);
            }
            return matches_bytes(
                expected,
                actual_data,
                actual_bytes,
                1);
        }
        case ImgprocGeometrySamplingOpId::WarpPerspectiveU8:
        {
            cv::Mat matrix = cv::Mat::eye(3, 3, CV_64FC1);
            matrix.at<double>(0, 1) = 0.1;
            matrix.at<double>(0, 2) = 0.25;
            matrix.at<double>(1, 0) = -0.05;
            matrix.at<double>(1, 2) = 0.5;
            matrix.at<double>(2, 0) = 0.002;
            matrix.at<double>(2, 1) = -0.003;
            cv::warpPerspective(
                source,
                expected,
                matrix,
                cv::Size(9, 7),
                cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                cv::BORDER_REFLECT_101);
            return matches_bytes(
                expected,
                actual_data,
                actual_bytes,
                1);
        }
        case ImgprocGeometrySamplingOpId::RectSubPixU8:
            cv::getRectSubPix(
                source,
                cv::Size(7, 5),
                cv::Point2f(0.25f, 0.75f),
                expected);
            return matches_bytes(
                expected,
                actual_data,
                actual_bytes,
                1);
        case ImgprocGeometrySamplingOpId::RectSubPixU8F32:
            cv::getRectSubPix(
                source,
                cv::Size(7, 5),
                cv::Point2f(4.25f, 3.75f),
                expected,
                CV_32F);
            return matches_float_values(
                expected,
                actual_data,
                actual_bytes,
                1e-4);
    }
    return false;
}

}  // namespace cvh_test_opencv_contract
