#include "cvh.h"
#include "gtest/gtest.h"
#include "opencv_contract_backend.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

using namespace cvh;

namespace {

std::uint32_t lcg_next(std::uint32_t state)
{
    return state * 1664525u + 1013904223u;
}

void fill_u8(Mat& mat, std::uint32_t seed)
{
    std::uint32_t state = seed;
    const int rows = mat.size[0];
    const int scalars_per_row = mat.size[1] * mat.channels();
    for (int y = 0; y < rows; ++y)
    {
        unsigned char* row = mat.data + static_cast<std::size_t>(y) * mat.step(0);
        for (int x = 0; x < scalars_per_row; ++x)
        {
            state = lcg_next(state);
            row[x] = static_cast<unsigned char>((state >> 24) ^ static_cast<std::uint32_t>(x + y * 17));
        }
    }
}

void fill_f32(Mat& mat, std::uint32_t seed)
{
    std::uint32_t state = seed;
    const int rows = mat.size[0];
    const int scalars_per_row = mat.size[1] * mat.channels();
    for (int y = 0; y < rows; ++y)
    {
        float* row = reinterpret_cast<float*>(mat.data + static_cast<std::size_t>(y) * mat.step(0));
        for (int x = 0; x < scalars_per_row; ++x)
        {
            state = lcg_next(state);
            row[x] = static_cast<float>(static_cast<int>(state & 0xffffu) - 32768) / 4096.0f;
        }
    }
}

void fill_f64(Mat& mat, std::uint32_t seed)
{
    std::uint32_t state = seed;
    const int rows = mat.size[0];
    const int scalars_per_row = mat.size[1] * mat.channels();
    for (int y = 0; y < rows; ++y)
    {
        double* row =
            reinterpret_cast<double*>(mat.data + static_cast<std::size_t>(y) * mat.step(0));
        for (int x = 0; x < scalars_per_row; ++x)
        {
            state = lcg_next(state);
            row[x] = static_cast<double>(static_cast<int>(state & 0xffffu) - 32768) / 4096.0;
        }
    }
}

template<typename T>
void fill_integer(Mat& mat, std::uint32_t seed)
{
    std::uint32_t state = seed;
    const int rows = mat.size[0];
    const int scalars_per_row = mat.size[1] * mat.channels();
    for (int y = 0; y < rows; ++y)
    {
        T* row = reinterpret_cast<T*>(mat.data + static_cast<std::size_t>(y) * mat.step(0));
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

int cvh_depth(cvh_test_opencv_contract::CoreDepthId depth)
{
    using cvh_test_opencv_contract::CoreDepthId;
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

void fill_core_depth(Mat& mat,
                     cvh_test_opencv_contract::CoreDepthId depth,
                     std::uint32_t seed)
{
    using cvh_test_opencv_contract::CoreDepthId;
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

void fill_core_depth_contiguous(
    Mat& mat,
    cvh_test_opencv_contract::CoreDepthId depth,
    std::uint32_t seed)
{
    Mat flat(
        {1, static_cast<int>(mat.total())},
        mat.type(),
        mat.data);
    fill_core_depth(flat, depth, seed);
}

void fill_mask(Mat& mask)
{
    for (int y = 0; y < mask.size[0]; ++y)
    {
        unsigned char* row =
            mask.data + static_cast<std::size_t>(y) * mask.step(0);
        for (int x = 0; x < mask.size[1]; ++x)
        {
            row[x] = ((x + 2 * y) % 3) != 0 ? 255 : 0;
        }
    }
}

cvh_test_opencv_contract::CoreReductionSummary collect_reduction_summary(
    const Mat& src,
    const Mat& mask)
{
    cvh_test_opencv_contract::CoreReductionSummary result{};
    const Scalar sums = sum(src);
    const Scalar means = mean(src, mask);
    Scalar mean_from_stddev;
    Scalar stddevs;
    meanStdDev(src, mean_from_stddev, stddevs, mask);
    for (int ch = 0; ch < src.channels(); ++ch)
    {
        result.sums[ch] = sums[ch];
        result.means[ch] = means[ch];
        result.stddevs[ch] = stddevs[ch];
    }
    result.norm_inf = norm(src, NORM_INF, mask);
    result.norm_l1 = norm(src, NORM_L1, mask);
    result.norm_l2 = norm(src, NORM_L2, mask);
    if (src.channels() == 1)
    {
        Point min_location;
        Point max_location;
        minMaxLoc(
            src,
            &result.min_value,
            &result.max_value,
            &min_location,
            &max_location,
            mask);
        result.count_non_zero = countNonZero(src);
        result.min_x = min_location.x;
        result.min_y = min_location.y;
        result.max_x = max_location.x;
        result.max_y = max_location.y;
    }
    return result;
}

void run_core_array_op(cvh_test_opencv_contract::CoreArrayOpId op,
                       const Mat& a,
                       const Mat& b,
                       Mat& dst)
{
    using cvh_test_opencv_contract::CoreArrayOpId;
    switch (op)
    {
        case CoreArrayOpId::AbsDiff: absdiff(a, b, dst); return;
        case CoreArrayOpId::BitwiseAnd: bitwise_and(a, b, dst); return;
        case CoreArrayOpId::BitwiseNot: bitwise_not(a, dst); return;
        case CoreArrayOpId::BitwiseOr: bitwise_or(a, b, dst); return;
        case CoreArrayOpId::BitwiseXor: bitwise_xor(a, b, dst); return;
        case CoreArrayOpId::InRange:
            inRange(a, Scalar::all(-2.5), Scalar::all(3.5), dst);
            return;
        case CoreArrayOpId::Min: min(a, b, dst); return;
        case CoreArrayOpId::Max: max(a, b, dst); return;
    }
}

void write_f32_bits(float& value, std::uint32_t bits)
{
    std::memcpy(&value, &bits, sizeof(bits));
}

void write_f64_bits(double& value, std::uint64_t bits)
{
    std::memcpy(&value, &bits, sizeof(bits));
}

void fill_math_input(Mat& mat,
                     cvh_test_opencv_contract::CoreMathOpId op,
                     std::uint32_t seed)
{
    using cvh_test_opencv_contract::CoreMathOpId;
    std::uint32_t state = seed;
    const int scalar_count = mat.size[1] * mat.channels();
    for (int y = 0; y < mat.size[0]; ++y)
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
                reinterpret_cast<float*>(
                    mat.data + static_cast<std::size_t>(y) * mat.step(0))[x] =
                    static_cast<float>(value);
            }
            else
            {
                reinterpret_cast<double*>(
                    mat.data + static_cast<std::size_t>(y) * mat.step(0))[x] = value;
            }
        }
    }
}

void run_core_math_op(cvh_test_opencv_contract::CoreMathOpId op,
                      const Mat& src,
                      Mat& dst)
{
    using cvh_test_opencv_contract::CoreMathOpId;
    switch (op)
    {
        case CoreMathOpId::Sqrt: cvh::sqrt(src, dst); return;
        case CoreMathOpId::Pow: cvh::pow(src, 1.75, dst); return;
        case CoreMathOpId::Exp: cvh::exp(src, dst); return;
        case CoreMathOpId::Log: cvh::log(src, dst); return;
    }
}

void run_core_layout_op(cvh_test_opencv_contract::CoreLayoutOpId op,
                        const Mat& src,
                        cvh_test_opencv_contract::CoreDepthId depth,
                        std::uint32_t seed,
                        Mat& dst)
{
    using cvh_test_opencv_contract::CoreLayoutOpId;
    switch (op)
    {
        case CoreLayoutOpId::CopyMask:
        {
            Mat mask(src.shape(), CV_8UC1);
            fill_mask(mask);
            copyTo(src, dst, mask);
            return;
        }
        case CoreLayoutOpId::ExtractLastChannel:
            extractChannel(src, dst, src.channels() - 1);
            return;
        case CoreLayoutOpId::FlipHorizontal: flip(src, dst, 1); return;
        case CoreLayoutOpId::FlipVertical: flip(src, dst, 0); return;
        case CoreLayoutOpId::FlipBoth: flip(src, dst, -1); return;
        case CoreLayoutOpId::RotateClockwise:
            rotate(src, dst, ROTATE_90_CLOCKWISE);
            return;
        case CoreLayoutOpId::Rotate180:
            rotate(src, dst, ROTATE_180);
            return;
        case CoreLayoutOpId::RotateCounterclockwise:
            rotate(src, dst, ROTATE_90_COUNTERCLOCKWISE);
            return;
        case CoreLayoutOpId::Repeat2x3: repeat(src, 2, 3, dst); return;
        case CoreLayoutOpId::HConcat:
        {
            Mat other(src.shape(), src.type());
            fill_core_depth(other, depth, seed ^ 0x9e3779b9u);
            hconcat(src, other, dst);
            return;
        }
        case CoreLayoutOpId::VConcat:
        {
            Mat other(src.shape(), src.type());
            fill_core_depth(other, depth, seed ^ 0x9e3779b9u);
            vconcat(src, other, dst);
            return;
        }
    }
}

}  // namespace

TEST(OpenCVContractSmoke_TEST, core_u8_to_f64_matches_upstream)
{
    constexpr int rows = 5;
    constexpr int cols = 7;
    constexpr int channels = 3;
    constexpr std::uint32_t seed = 0x12345678u;

    Mat src({rows, cols}, CV_8UC3);
    fill_u8(src, seed);

    Mat dst;
    src.convertTo(dst, CV_64F);
    ASSERT_EQ(dst.type(), CV_64FC3);
    ASSERT_TRUE(dst.isContinuous());

    EXPECT_TRUE(cvh_test_opencv_contract::validate_core_convert_u8_to_f64(
        rows,
        cols,
        channels,
        seed,
        dst.data,
        dst.total() * dst.elemSize()));
}

TEST(OpenCVContractSmoke_TEST, imgproc_resize_linear_u8_matches_upstream)
{
    constexpr int src_rows = 7;
    constexpr int src_cols = 9;
    constexpr int dst_rows = 5;
    constexpr int dst_cols = 6;
    constexpr int channels = 1;
    constexpr std::uint32_t seed = 0x9e3779b9u;

    Mat src({src_rows, src_cols}, CV_8UC1);
    fill_u8(src, seed);

    Mat dst;
    resize(src, dst, Size(dst_cols, dst_rows), 0.0, 0.0, INTER_LINEAR);
    ASSERT_EQ(dst.type(), CV_8UC1);
    ASSERT_TRUE(dst.isContinuous());

    EXPECT_TRUE(cvh_test_opencv_contract::validate_imgproc_resize_linear_u8(
        src_rows,
        src_cols,
        dst_rows,
        dst_cols,
        channels,
        seed,
        dst.data,
        dst.total() * dst.elemSize()));
}

TEST(OpenCVContractSmoke_TEST, core_array_ops_match_upstream_for_standard_depths)
{
    using cvh_test_opencv_contract::CoreArrayOpId;
    using cvh_test_opencv_contract::CoreDepthId;
    constexpr int rows = 5;
    constexpr int cols = 7;
    constexpr int channels = 3;
    constexpr std::uint32_t seed_a = 0x10203040u;
    constexpr std::uint32_t seed_b = 0x55667788u;
    const CoreArrayOpId ops[] = {
        CoreArrayOpId::AbsDiff,
        CoreArrayOpId::BitwiseAnd,
        CoreArrayOpId::BitwiseNot,
        CoreArrayOpId::BitwiseOr,
        CoreArrayOpId::BitwiseXor,
        CoreArrayOpId::InRange,
        CoreArrayOpId::Min,
        CoreArrayOpId::Max,
    };

    for (const CoreDepthId depth :
         {CoreDepthId::U8,
          CoreDepthId::S8,
          CoreDepthId::U16,
          CoreDepthId::S16,
          CoreDepthId::S32,
          CoreDepthId::F32})
    {
        const int type = CV_MAKETYPE(cvh_depth(depth), channels);
        Mat a({rows, cols}, type);
        Mat b({rows, cols}, type);
        fill_core_depth(a, depth, seed_a);
        fill_core_depth(b, depth, seed_b);

        for (const CoreArrayOpId op : ops)
        {
            SCOPED_TRACE(static_cast<int>(depth));
            SCOPED_TRACE(static_cast<int>(op));
            Mat dst;
            run_core_array_op(op, a, b, dst);
            EXPECT_TRUE(cvh_test_opencv_contract::validate_core_array_op(
                op,
                depth,
                rows,
                cols,
                channels,
                seed_a,
                seed_b,
                dst.data,
                dst.total() * dst.elemSize()));
        }
    }
}

TEST(OpenCVContractSmoke_TEST, core_float_numeric_edges_match_upstream_bits)
{
    using cvh_test_opencv_contract::CoreArrayOpId;
    Mat a({1, 5}, CV_32FC1);
    Mat b({1, 5}, CV_32FC1);
    const std::uint32_t a_bits[] = {
        0x7fc12345u, 0x3f800000u, 0x7f800000u, 0x80000000u, 0x00000000u,
    };
    const std::uint32_t b_bits[] = {
        0x40000000u, 0x7fc54321u, 0x7f800000u, 0x00000000u, 0x80000000u,
    };
    for (int x = 0; x < 5; ++x)
    {
        write_f32_bits(a.at<float>(0, x), a_bits[x]);
        write_f32_bits(b.at<float>(0, x), b_bits[x]);
    }

    for (const CoreArrayOpId op :
         {CoreArrayOpId::AbsDiff, CoreArrayOpId::Min, CoreArrayOpId::Max})
    {
        SCOPED_TRACE(static_cast<int>(op));
        Mat dst;
        run_core_array_op(op, a, b, dst);
        EXPECT_TRUE(cvh_test_opencv_contract::validate_core_float_edge_op(
            op, dst.data, dst.total() * dst.elemSize()));
    }
}

TEST(OpenCVContractSmoke_TEST, core_double_numeric_edges_match_upstream_bits)
{
    using cvh_test_opencv_contract::CoreArrayOpId;
    Mat a({1, 5}, CV_64FC1);
    Mat b({1, 5}, CV_64FC1);
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
        write_f64_bits(a.at<double>(0, x), a_bits[x]);
        write_f64_bits(b.at<double>(0, x), b_bits[x]);
    }

    for (const CoreArrayOpId op :
         {CoreArrayOpId::AbsDiff, CoreArrayOpId::Min, CoreArrayOpId::Max})
    {
        SCOPED_TRACE(static_cast<int>(op));
        Mat dst;
        run_core_array_op(op, a, b, dst);
        EXPECT_TRUE(cvh_test_opencv_contract::validate_core_double_edge_op(
            op, dst.data, dst.total() * dst.elemSize()));
    }
}

TEST(OpenCVContractSmoke_TEST, core_convert_scale_abs_and_fp16_match_upstream_bits)
{
    Mat scale_src({1, 9}, CV_32FC1);
    const float scale_values[] = {
        -300.0f, -2.5f, -1.5f, -0.5f, 0.5f, 1.5f, 2.5f, 254.5f, 300.0f,
    };
    std::memcpy(scale_src.data, scale_values, sizeof(scale_values));
    Mat scale_dst;
    convertScaleAbs(scale_src, scale_dst);
    EXPECT_TRUE(cvh_test_opencv_contract::validate_convert_scale_abs_edges(
        scale_dst.data, scale_dst.total() * scale_dst.elemSize()));

    Mat fp32({1, 11}, CV_32FC1);
    const float denorm = std::ldexp(1.0f, -24);
    const float fp32_values[] = {
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
    std::memcpy(fp32.data, fp32_values, sizeof(fp32_values));
    Mat fp16;
    convertFp16(fp32, fp16);
    EXPECT_TRUE(cvh_test_opencv_contract::validate_convert_fp16_edges(
        fp16.data, fp16.total() * fp16.elemSize()));
}

TEST(OpenCVContractSmoke_TEST, core_math_functions_match_upstream_tolerance)
{
    using cvh_test_opencv_contract::CoreDepthId;
    using cvh_test_opencv_contract::CoreMathOpId;
    constexpr int rows = 3;
    constexpr int cols = 7;
    constexpr int channels = 3;
    constexpr std::uint32_t seed = 0xa17c93e5u;

    for (const CoreDepthId depth : {CoreDepthId::F32, CoreDepthId::F64})
    {
        const int type = CV_MAKETYPE(cvh_depth(depth), channels);
        for (const CoreMathOpId op :
             {CoreMathOpId::Sqrt, CoreMathOpId::Pow, CoreMathOpId::Exp, CoreMathOpId::Log})
        {
            SCOPED_TRACE(static_cast<int>(depth));
            SCOPED_TRACE(static_cast<int>(op));
            Mat src({rows, cols}, type);
            fill_math_input(src, op, seed);
            Mat dst;
            run_core_math_op(op, src, dst);
            EXPECT_TRUE(cvh_test_opencv_contract::validate_core_math_op(
                op,
                depth,
                rows,
                cols,
                channels,
                seed,
                dst.data,
                dst.total() * dst.elemSize()));
        }
    }
}

TEST(OpenCVContractSmoke_TEST, core_reduction_summaries_match_upstream)
{
    using cvh_test_opencv_contract::CoreDepthId;
    constexpr int rows = 7;
    constexpr int cols = 11;
    constexpr std::uint32_t seed = 0x4f13c2a9u;

    for (const CoreDepthId depth :
         {CoreDepthId::U8,
          CoreDepthId::S16,
          CoreDepthId::S32,
          CoreDepthId::F32,
          CoreDepthId::F64})
    {
        SCOPED_TRACE(static_cast<int>(depth));
        Mat src({rows, cols}, CV_MAKETYPE(cvh_depth(depth), 1));
        fill_core_depth(src, depth, seed);
        Mat mask({rows, cols}, CV_8UC1);
        fill_mask(mask);
        const auto actual = collect_reduction_summary(src, mask);
        EXPECT_TRUE(
            cvh_test_opencv_contract::validate_core_reduction_summary(
                depth, rows, cols, 1, seed, true, actual));

        std::vector<Point> points;
        findNonZero(src, points);
        std::vector<int> xy(points.size() * 2);
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            xy[2 * i] = points[i].x;
            xy[2 * i + 1] = points[i].y;
        }
        EXPECT_TRUE(cvh_test_opencv_contract::validate_core_nonzero_locations(
            depth,
            rows,
            cols,
            seed,
            xy.empty() ? nullptr : xy.data(),
            points.size()));
    }

    for (const auto depth_and_channels :
         {std::pair<CoreDepthId, int>(CoreDepthId::U8, 3),
          std::pair<CoreDepthId, int>(CoreDepthId::F64, 4)})
    {
        const CoreDepthId depth = depth_and_channels.first;
        const int channels = depth_and_channels.second;
        SCOPED_TRACE(channels);
        Mat src(
            {rows, cols}, CV_MAKETYPE(cvh_depth(depth), channels));
        fill_core_depth(src, depth, seed);
        Mat mask({rows, cols}, CV_8UC1);
        fill_mask(mask);
        const auto actual = collect_reduction_summary(src, mask);
        EXPECT_TRUE(
            cvh_test_opencv_contract::validate_core_reduction_summary(
                depth, rows, cols, channels, seed, true, actual));
    }
}

TEST(OpenCVContractSmoke_TEST, core_reduce_and_normalize_match_upstream)
{
    using cvh_test_opencv_contract::CoreDepthId;
    constexpr int rows = 7;
    constexpr int cols = 11;
    constexpr int channels = 3;
    constexpr std::uint32_t seed = 0x6ac19e35u;

    for (const CoreDepthId depth :
         {CoreDepthId::U8,
          CoreDepthId::S16,
          CoreDepthId::S32,
          CoreDepthId::F32,
          CoreDepthId::F64})
    {
        SCOPED_TRACE(static_cast<int>(depth));
        Mat src(
            {rows, cols}, CV_MAKETYPE(cvh_depth(depth), channels));
        fill_core_depth(src, depth, seed);
        if (depth != CoreDepthId::S32)
        {
            for (const int dim : {0, 1})
            {
                for (const int reduce_type :
                     {REDUCE_SUM, REDUCE_AVG, REDUCE_SUM2})
                {
                    SCOPED_TRACE(dim);
                    SCOPED_TRACE(reduce_type);
                    Mat reduced;
                    reduce(src, reduced, dim, reduce_type, CV_64F);
                    EXPECT_TRUE(
                        cvh_test_opencv_contract::validate_core_reduce_f64(
                            depth,
                            rows,
                            cols,
                            channels,
                            seed,
                            dim,
                            reduce_type,
                            reduced.data,
                            reduced.total() * reduced.elemSize()));
                }
            }
        }

        Mat mask({rows, cols}, CV_8UC1);
        fill_mask(mask);
        Mat normalized;
        normalize(src, normalized, 2.0, 0.0, NORM_L2, CV_64F, mask);
        EXPECT_TRUE(
            cvh_test_opencv_contract::validate_core_normalize_l2_f64(
                depth,
                rows,
                cols,
                channels,
                seed,
                true,
                normalized.data,
                normalized.total() * normalized.elemSize()));
    }
}

TEST(OpenCVContractSmoke_TEST, core_reduce_arg_matches_upstream)
{
    using cvh_test_opencv_contract::CoreDepthId;
    constexpr int rows = 9;
    constexpr int cols = 13;
    constexpr std::uint32_t seed = 0xe31b2475u;

    for (const CoreDepthId depth :
         {CoreDepthId::U8,
          CoreDepthId::S16,
          CoreDepthId::S32,
          CoreDepthId::F32,
          CoreDepthId::F64})
    {
        Mat src({rows, cols}, CV_MAKETYPE(cvh_depth(depth), 1));
        fill_core_depth(src, depth, seed);
        for (const int axis : {0, 1})
        {
            for (const bool find_max : {false, true})
            {
                for (const bool last_index : {false, true})
                {
                    SCOPED_TRACE(static_cast<int>(depth));
                    SCOPED_TRACE(axis);
                    SCOPED_TRACE(find_max);
                    SCOPED_TRACE(last_index);
                    Mat actual;
                    if (find_max)
                    {
                        reduceArgMax(src, actual, axis, last_index);
                    }
                    else
                    {
                        reduceArgMin(src, actual, axis, last_index);
                    }
                    EXPECT_TRUE(
                        cvh_test_opencv_contract::validate_core_reduce_arg(
                            depth,
                            rows,
                            cols,
                            seed,
                            axis,
                            find_max,
                            last_index,
                            actual.data,
                            actual.total() * actual.elemSize()));
                }
            }
        }
    }
}

TEST(OpenCVContractSmoke_TEST, core_border_interpolate_matches_upstream)
{
    for (const int border_type :
         {BORDER_CONSTANT,
          BORDER_REPLICATE,
          BORDER_REFLECT,
          BORDER_WRAP,
          BORDER_REFLECT_101})
    {
        for (const int coordinate : {-13, -1, 0, 4, 5, 17})
        {
            SCOPED_TRACE(border_type);
            SCOPED_TRACE(coordinate);
            const int actual =
                borderInterpolate(coordinate, 5, border_type);
            EXPECT_TRUE(
                cvh_test_opencv_contract::validate_core_border_interpolate(
                    coordinate, 5, border_type, actual));
        }
    }
}

TEST(OpenCVContractSmoke_TEST, core_layout_ops_match_upstream_bytes)
{
    using cvh_test_opencv_contract::CoreDepthId;
    using cvh_test_opencv_contract::CoreLayoutOpId;
    constexpr int rows = 5;
    constexpr int cols = 7;
    constexpr std::uint32_t seed = 0x38a7d12fu;
    const CoreLayoutOpId operations[] = {
        CoreLayoutOpId::CopyMask,
        CoreLayoutOpId::ExtractLastChannel,
        CoreLayoutOpId::FlipHorizontal,
        CoreLayoutOpId::FlipVertical,
        CoreLayoutOpId::FlipBoth,
        CoreLayoutOpId::RotateClockwise,
        CoreLayoutOpId::Rotate180,
        CoreLayoutOpId::RotateCounterclockwise,
        CoreLayoutOpId::Repeat2x3,
        CoreLayoutOpId::HConcat,
        CoreLayoutOpId::VConcat,
    };
    for (const CoreDepthId depth :
         {CoreDepthId::U8,
          CoreDepthId::S16,
          CoreDepthId::S32,
          CoreDepthId::F32,
          CoreDepthId::F64})
    {
        for (const int channels : {1, 3, 4})
        {
            Mat src(
                {rows, cols}, CV_MAKETYPE(cvh_depth(depth), channels));
            fill_core_depth(src, depth, seed);
            for (const CoreLayoutOpId op : operations)
            {
                SCOPED_TRACE(static_cast<int>(depth));
                SCOPED_TRACE(channels);
                SCOPED_TRACE(static_cast<int>(op));
                Mat actual;
                run_core_layout_op(op, src, depth, seed, actual);
                EXPECT_TRUE(
                    cvh_test_opencv_contract::validate_core_layout_op(
                        op,
                        depth,
                        rows,
                        cols,
                        channels,
                        seed,
                        actual.data,
                        actual.total() * actual.elemSize()));
            }
        }
    }
}

TEST(OpenCVContractSmoke_TEST, core_mix_flip_nd_and_broadcast_match_upstream)
{
    using cvh_test_opencv_contract::CoreDepthId;
    constexpr int rows = 5;
    constexpr int cols = 7;
    constexpr std::uint32_t seed = 0x91c6e24bu;

    for (const CoreDepthId depth :
         {CoreDepthId::U8, CoreDepthId::F64})
    {
        Mat src({rows, cols}, CV_MAKETYPE(cvh_depth(depth), 4));
        fill_core_depth(src, depth, seed);
        Mat bgr({rows, cols}, CV_MAKETYPE(cvh_depth(depth), 3));
        Mat alpha({rows, cols}, CV_MAKETYPE(cvh_depth(depth), 1));
        Mat outputs[] = {bgr, alpha};
        const int routes[] = {0, 2, 1, 1, 2, 0, 3, 3};
        mixChannels(&src, 1, outputs, 2, routes, 4);
        EXPECT_TRUE(cvh_test_opencv_contract::validate_core_mix_channels(
            depth,
            rows,
            cols,
            seed,
            outputs[0].data,
            outputs[0].total() * outputs[0].elemSize(),
            outputs[1].data,
            outputs[1].total() * outputs[1].elemSize()));
    }

    for (const CoreDepthId depth :
         {CoreDepthId::U8,
          CoreDepthId::S16,
          CoreDepthId::S32,
          CoreDepthId::F32,
          CoreDepthId::F64})
    {
        Mat nd_src({2, 3, 4}, CV_MAKETYPE(cvh_depth(depth), 1));
        fill_core_depth_contiguous(nd_src, depth, seed);
        for (const int axis : {0, -1})
        {
            Mat flipped;
            flipND(nd_src, flipped, axis);
            EXPECT_TRUE(
                cvh_test_opencv_contract::validate_core_flip_nd(
                    depth,
                    seed,
                    axis,
                    flipped.data,
                    flipped.total() * flipped.elemSize()));
        }

        Mat broadcast_src(
            {2, 1, 3}, CV_MAKETYPE(cvh_depth(depth), 1));
        fill_core_depth_contiguous(broadcast_src, depth, seed);
        Mat broadcast_dst;
        broadcast(
            broadcast_src,
            std::vector<int>({4, 2, 5, 3}),
            broadcast_dst);
        EXPECT_TRUE(cvh_test_opencv_contract::validate_core_broadcast(
            depth,
            seed,
            broadcast_dst.data,
            broadcast_dst.total() * broadcast_dst.elemSize()));
    }
}

TEST(OpenCVContractSmoke_TEST, imgproc_phase1_kernels_match_upstream)
{
    using cvh_test_opencv_contract::CoreDepthId;
    for (const int shape : {MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE})
    {
        Mat kernel =
            getStructuringElement(shape, Size(7, 5), Point(2, 1));
        EXPECT_TRUE(
            cvh_test_opencv_contract::validate_imgproc_structuring_element(
                shape,
                7,
                5,
                2,
                1,
                kernel.data,
                kernel.total() * kernel.elemSize()));
    }

    for (const CoreDepthId depth :
         {CoreDepthId::F32, CoreDepthId::F64})
    {
        const int type =
            depth == CoreDepthId::F32 ? CV_32F : CV_64F;
        for (const double sigma : {0.0, 1.7})
        {
            Mat gaussian = getGaussianKernel(7, sigma, type);
            EXPECT_TRUE(
                cvh_test_opencv_contract::validate_imgproc_gaussian_kernel(
                    7,
                    sigma,
                    depth,
                    gaussian.data,
                    gaussian.total() * gaussian.elemSize()));
        }

        Mat kx;
        Mat ky;
        getDerivKernels(kx, ky, 1, 0, 5, true, type);
        EXPECT_TRUE(
            cvh_test_opencv_contract::validate_imgproc_deriv_kernels(
                1,
                0,
                5,
                true,
                depth,
                kx.data,
                kx.total() * kx.elemSize(),
                ky.data,
                ky.total() * ky.elemSize()));

        Mat gabor = getGaborKernel(
            Size(7, 5), 2.0, 0.3, 4.0, 0.8, 0.0, type);
        EXPECT_TRUE(
            cvh_test_opencv_contract::validate_imgproc_gabor_kernel(
                7,
                5,
                depth,
                gabor.data,
                gabor.total() * gabor.elemSize()));

        Mat hanning;
        createHanningWindow(hanning, Size(7, 5), type);
        EXPECT_TRUE(
            cvh_test_opencv_contract::validate_imgproc_hanning_window(
                7,
                5,
                depth,
                hanning.data,
                hanning.total() * hanning.elemSize()));
    }
}

TEST(OpenCVContractSmoke_TEST, imgproc_phase1_integral_derivatives_and_square_box_match_upstream)
{
    using cvh_test_opencv_contract::CoreDepthId;
    constexpr int rows = 11;
    constexpr int cols = 13;
    constexpr std::uint32_t seed = 0x73ad91e5u;
    for (const int channels : {1, 3, 4})
    {
        Mat src({rows, cols}, CV_MAKETYPE(CV_8U, channels));
        fill_u8(src, seed);
        for (const auto depth_and_type :
             {std::pair<CoreDepthId, int>(CoreDepthId::S32, CV_32S),
              std::pair<CoreDepthId, int>(CoreDepthId::F64, CV_64F)})
        {
            Mat actual;
            integral(src, actual, depth_and_type.second);
            EXPECT_TRUE(
                cvh_test_opencv_contract::validate_imgproc_integral_u8(
                    rows,
                    cols,
                    channels,
                    seed,
                    depth_and_type.first,
                    actual.data,
                    actual.total() * actual.elemSize()));
        }

        Mat scharr;
        Mat laplacian;
        Scharr(src, scharr, CV_16S, 1, 0);
        Laplacian(src, laplacian, CV_16S, 3);
        EXPECT_TRUE(
            cvh_test_opencv_contract::validate_imgproc_derivative_filters_u8(
                rows,
                cols,
                channels,
                seed,
                scharr.data,
                scharr.total() * scharr.elemSize(),
                laplacian.data,
                laplacian.total() * laplacian.elemSize()));

        Mat squared;
        sqrBoxFilter(
            src, squared, CV_64F, Size(7, 5));
        EXPECT_TRUE(
            cvh_test_opencv_contract::validate_imgproc_sqr_box_filter_u8(
                rows,
                cols,
                channels,
                seed,
                squared.data,
                squared.total() * squared.elemSize()));
    }

    Mat src({rows, cols}, CV_8UC1);
    fill_u8(src, seed);
    Mat dx;
    Mat dy;
    spatialGradient(src, dx, dy);
    EXPECT_TRUE(
        cvh_test_opencv_contract::validate_imgproc_spatial_gradient_u8(
            rows,
            cols,
            seed,
            dx.data,
            dx.total() * dx.elemSize(),
            dy.data,
            dy.total() * dy.elemSize()));
}

TEST(OpenCVContractSmoke_TEST, imgproc_phase1_intensity_ops_match_upstream)
{
    using cvh_test_opencv_contract::ImgprocIntensityOpId;
    constexpr int rows = 13;
    constexpr int cols = 17;
    constexpr std::uint32_t seed = 0x4e912bd3u;
    const auto run = [&](ImgprocIntensityOpId op) {
        const bool color_source =
            op == ImgprocIntensityOpId::BilateralU8 ||
            op == ImgprocIntensityOpId::StackBlurU8;
        Mat src(
            {rows, cols},
            color_source ? CV_8UC3 : CV_8UC1);
        fill_u8(src, seed);
        Mat actual;
        switch (op)
        {
            case ImgprocIntensityOpId::MedianU8:
                medianBlur(src, actual, 5);
                break;
            case ImgprocIntensityOpId::BilateralU8:
                bilateralFilter(
                    src,
                    actual,
                    5,
                    35.0,
                    2.0,
                    BORDER_REFLECT_101);
                break;
            case ImgprocIntensityOpId::StackBlurU8:
                stackBlur(src, actual, Size(5, 3));
                break;
            case ImgprocIntensityOpId::AdaptiveMeanU8:
                adaptiveThreshold(
                    src,
                    actual,
                    200.0,
                    ADAPTIVE_THRESH_MEAN_C,
                    THRESH_BINARY,
                    5,
                    2.25);
                break;
            case ImgprocIntensityOpId::AdaptiveGaussianU8:
                adaptiveThreshold(
                    src,
                    actual,
                    200.0,
                    ADAPTIVE_THRESH_GAUSSIAN_C,
                    THRESH_BINARY_INV,
                    5,
                    -1.25);
                break;
            case ImgprocIntensityOpId::ThresholdMaskU8:
            {
                Mat mask({rows, cols}, CV_8UC1);
                fill_mask(mask);
                actual.create(src.shape(), src.type());
                actual.setTo(Scalar::all(17));
                thresholdWithMask(
                    src,
                    actual,
                    mask,
                    110.0,
                    200.0,
                    THRESH_BINARY);
                break;
            }
            case ImgprocIntensityOpId::EqualizeHistU8:
                equalizeHist(src, actual);
                break;
            case ImgprocIntensityOpId::ColorMapJetU8:
                applyColorMap(src, actual, COLORMAP_JET);
                break;
            case ImgprocIntensityOpId::ColorMapUserU8:
            {
                Mat lookup({256, 1}, CV_8UC3);
                for (int i = 0; i < 256; ++i)
                {
                    lookup.at<uchar>(i, 0, 0) =
                        static_cast<uchar>(i);
                    lookup.at<uchar>(i, 0, 1) =
                        static_cast<uchar>(255 - i);
                    lookup.at<uchar>(i, 0, 2) = 17;
                }
                applyColorMap(src, actual, lookup);
                break;
            }
        }
        EXPECT_TRUE(
            cvh_test_opencv_contract::validate_imgproc_intensity_u8(
                op,
                rows,
                cols,
                seed,
                actual.data,
                actual.total() * actual.elemSize()))
            << "intensity op id=" << static_cast<int>(op);
    };

    for (const ImgprocIntensityOpId op :
         {ImgprocIntensityOpId::MedianU8,
          ImgprocIntensityOpId::BilateralU8,
          ImgprocIntensityOpId::StackBlurU8,
          ImgprocIntensityOpId::AdaptiveMeanU8,
          ImgprocIntensityOpId::AdaptiveGaussianU8,
          ImgprocIntensityOpId::ThresholdMaskU8,
          ImgprocIntensityOpId::EqualizeHistU8,
          ImgprocIntensityOpId::ColorMapJetU8,
          ImgprocIntensityOpId::ColorMapUserU8})
    {
        run(op);
    }
}

TEST(OpenCVContractSmoke_TEST, imgproc_phase1_pyramid_color_ops_match_upstream)
{
    using cvh_test_opencv_contract::ImgprocPyramidColorOpId;
    constexpr int rows = 12;
    constexpr int cols = 16;
    constexpr std::uint32_t seed = 0x8d2346b1u;
    const auto run = [&](ImgprocPyramidColorOpId op) {
        const bool demosaic =
            op == ImgprocPyramidColorOpId::DemosaicBgU8 ||
            op == ImgprocPyramidColorOpId::DemosaicGbU8 ||
            op == ImgprocPyramidColorOpId::DemosaicRgU8 ||
            op == ImgprocPyramidColorOpId::DemosaicGrU8;
        Mat src(
            {rows, cols}, demosaic ? CV_8UC1 : CV_8UC3);
        fill_u8(src, seed);
        Mat actual;
        switch (op)
        {
            case ImgprocPyramidColorOpId::AccumulateU8:
            case ImgprocPyramidColorOpId::AccumulateSquareU8:
            case ImgprocPyramidColorOpId::AccumulateProductU8:
            case ImgprocPyramidColorOpId::AccumulateWeightedU8:
            {
                Mat mask({rows, cols}, CV_8UC1);
                fill_mask(mask);
                actual.create({rows, cols}, CV_32FC3);
                actual.setTo(Scalar::all(1.0));
                if (op == ImgprocPyramidColorOpId::AccumulateU8)
                {
                    accumulate(src, actual);
                }
                else if (
                    op == ImgprocPyramidColorOpId::AccumulateSquareU8)
                {
                    accumulateSquare(src, actual, mask);
                }
                else if (
                    op == ImgprocPyramidColorOpId::AccumulateProductU8)
                {
                    Mat second({rows, cols}, CV_8UC3);
                    fill_u8(second, seed + 17u);
                    accumulateProduct(src, second, actual, mask);
                }
                else
                {
                    accumulateWeighted(src, actual, 0.375, mask);
                }
                break;
            }
            case ImgprocPyramidColorOpId::BlendLinearU8:
            {
                Mat second({rows, cols}, CV_8UC3);
                fill_u8(second, seed + 17u);
                Mat weight1({rows, cols}, CV_32FC1);
                Mat weight2({rows, cols}, CV_32FC1);
                for (int y = 0; y < rows; ++y)
                {
                    for (int x = 0; x < cols; ++x)
                    {
                        weight1.at<float>(y, x) =
                            static_cast<float>((x + y) % 5) * 0.25f;
                        weight2.at<float>(y, x) =
                            static_cast<float>((2 * x + y + 1) % 7) *
                            0.2f;
                    }
                }
                blendLinear(src, second, weight1, weight2, actual);
                break;
            }
            case ImgprocPyramidColorOpId::PyrDownU8:
                pyrDown(src, actual);
                break;
            case ImgprocPyramidColorOpId::PyrUpU8:
                pyrUp(src, actual);
                break;
            case ImgprocPyramidColorOpId::TwoPlaneNv12U8:
            case ImgprocPyramidColorOpId::TwoPlaneNv21U8:
            {
                Mat y_plane({rows, cols}, CV_8UC1);
                Mat uv_plane({rows / 2, cols / 2}, CV_8UC2);
                fill_u8(y_plane, seed);
                fill_u8(uv_plane, seed + 17u);
                cvtColorTwoPlane(
                    y_plane,
                    uv_plane,
                    actual,
                    op == ImgprocPyramidColorOpId::TwoPlaneNv12U8
                        ? COLOR_YUV2BGR_NV12
                        : COLOR_YUV2RGB_NV21);
                break;
            }
            case ImgprocPyramidColorOpId::DemosaicBgU8:
                demosaicing(src, actual, COLOR_BayerBG2BGR);
                break;
            case ImgprocPyramidColorOpId::DemosaicGbU8:
                demosaicing(src, actual, COLOR_BayerGB2BGR);
                break;
            case ImgprocPyramidColorOpId::DemosaicRgU8:
                demosaicing(src, actual, COLOR_BayerRG2BGR);
                break;
            case ImgprocPyramidColorOpId::DemosaicGrU8:
                demosaicing(src, actual, COLOR_BayerGR2BGR);
                break;
        }
        EXPECT_TRUE(
            cvh_test_opencv_contract::validate_imgproc_pyramid_color_u8(
                op,
                rows,
                cols,
                seed,
                actual.data,
                actual.total() * actual.elemSize()))
            << "pyramid/color op id=" << static_cast<int>(op);
    };

    for (const ImgprocPyramidColorOpId op :
         {ImgprocPyramidColorOpId::AccumulateU8,
          ImgprocPyramidColorOpId::AccumulateSquareU8,
          ImgprocPyramidColorOpId::AccumulateProductU8,
          ImgprocPyramidColorOpId::AccumulateWeightedU8,
          ImgprocPyramidColorOpId::BlendLinearU8,
          ImgprocPyramidColorOpId::PyrDownU8,
          ImgprocPyramidColorOpId::PyrUpU8,
          ImgprocPyramidColorOpId::TwoPlaneNv12U8,
          ImgprocPyramidColorOpId::TwoPlaneNv21U8,
          ImgprocPyramidColorOpId::DemosaicBgU8,
          ImgprocPyramidColorOpId::DemosaicGbU8,
          ImgprocPyramidColorOpId::DemosaicRgU8,
          ImgprocPyramidColorOpId::DemosaicGrU8})
    {
        run(op);
    }
}

TEST(OpenCVContractSmoke_TEST, imgproc_geometry_matrices_match_upstream)
{
    const Point2f center(12.5f, -7.25f);
    const AffineMatrix2x3d rotation =
        getRotationMatrix2D_(center, 37.0, 0.75);

    const Point2f affine_source[] = {
        Point2f(0.0f, 0.0f),
        Point2f(4.0f, 0.0f),
        Point2f(0.0f, 3.0f)};
    const Point2f affine_target[] = {
        Point2f(5.0f, -2.0f),
        Point2f(13.0f, -6.0f),
        Point2f(6.5f, 7.0f)};
    const Mat affine =
        getAffineTransform(affine_source, affine_target);

    const Point2f perspective_source[] = {
        Point2f(0.0f, 0.0f),
        Point2f(8.0f, 0.0f),
        Point2f(8.0f, 6.0f),
        Point2f(0.0f, 6.0f)};
    const Point2f perspective_target[] = {
        Point2f(1.0f, 2.0f),
        Point2f(9.0f, 1.0f),
        Point2f(7.5f, 8.0f),
        Point2f(-0.5f, 6.5f)};
    const Mat perspective =
        getPerspectiveTransform(
            perspective_source,
            perspective_target);

    Mat inverse;
    invertAffineTransform(affine, inverse);
    EXPECT_TRUE(
        cvh_test_opencv_contract::validate_imgproc_geometry_matrices(
            rotation.val,
            reinterpret_cast<const double*>(affine.data),
            reinterpret_cast<const double*>(perspective.data),
            reinterpret_cast<const double*>(inverse.data)));
}

TEST(OpenCVContractSmoke_TEST, imgproc_geometry_sampling_matches_upstream)
{
    using cvh_test_opencv_contract::ImgprocGeometrySamplingOpId;
    constexpr std::uint32_t seed = 0x7193u;
    Mat source({9, 11}, CV_8UC3);
    fill_u8(source, seed);

    const auto validate =
        [&](ImgprocGeometrySamplingOpId op, const Mat& actual) {
            EXPECT_TRUE(
                cvh_test_opencv_contract::
                    validate_imgproc_geometry_sampling(
                        op,
                        seed,
                        actual.data,
                        actual.total() * actual.elemSize()))
                << "geometry sampling op=" << static_cast<int>(op);
        };

    Mat map_x({7, 9}, CV_32FC1);
    Mat map_y({7, 9}, CV_32FC1);
    for (int row = 0; row < 7; ++row)
    {
        for (int col = 0; col < 9; ++col)
        {
            map_x.at<float>(row, col) =
                static_cast<float>(col) + 0.28125f;
            map_y.at<float>(row, col) =
                static_cast<float>(row) - 0.34375f;
        }
    }
    Mat actual;
    remap(
        source,
        actual,
        map_x,
        map_y,
        INTER_LINEAR,
        BORDER_REFLECT_101);
    validate(ImgprocGeometrySamplingOpId::RemapFloatU8, actual);

    Mat fixed_coordinates;
    Mat fixed_fractions;
    convertMaps(
        map_x,
        map_y,
        fixed_coordinates,
        fixed_fractions,
        CV_16SC2);
    remap(
        source,
        actual,
        fixed_coordinates,
        fixed_fractions,
        INTER_LINEAR,
        BORDER_REFLECT_101);
    validate(ImgprocGeometrySamplingOpId::RemapFixedU8, actual);

    Mat perspective({3, 3}, CV_64FC1);
    perspective.setTo(Scalar::all(0.0));
    perspective.at<double>(0, 0) = 1.0;
    perspective.at<double>(0, 1) = 0.1;
    perspective.at<double>(0, 2) = 0.25;
    perspective.at<double>(1, 0) = -0.05;
    perspective.at<double>(1, 1) = 1.0;
    perspective.at<double>(1, 2) = 0.5;
    perspective.at<double>(2, 0) = 0.002;
    perspective.at<double>(2, 1) = -0.003;
    perspective.at<double>(2, 2) = 1.0;
    warpPerspective(
        source,
        actual,
        perspective,
        Size(9, 7),
        INTER_LINEAR | WARP_INVERSE_MAP,
        BORDER_REFLECT_101);
    validate(ImgprocGeometrySamplingOpId::WarpPerspectiveU8, actual);

    getRectSubPix(
        source,
        Size(7, 5),
        Point2f(0.25f, 0.75f),
        actual);
    validate(ImgprocGeometrySamplingOpId::RectSubPixU8, actual);
    getRectSubPix(
        source,
        Size(7, 5),
        Point2f(4.25f, 3.75f),
        actual,
        CV_32F);
    validate(ImgprocGeometrySamplingOpId::RectSubPixU8F32, actual);
}
