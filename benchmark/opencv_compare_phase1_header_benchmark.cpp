#include "common/benchmark_common.h"
#include "cvh.h"
#include "opencv_compare_phase1_benchmark.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cvh_bench_compare {
namespace {

volatile double g_phase1_cvh_sink = 0.0;

std::uint32_t p1_lcg_next(std::uint32_t state)
{
    return state * 1664525u + 1013904223u;
}

cvh::Mat p1_make_mat(int rows, int cols, int type)
{
    return cvh::Mat({rows, cols}, type);
}

void p1_fill_u8(cvh::Mat& mat, std::uint32_t seed)
{
    std::uint32_t state = seed;
    const int scalars_per_row = mat.size[1] * mat.channels();
    for (int row = 0; row < mat.size[0]; ++row)
    {
        uchar* output = mat.data + static_cast<std::size_t>(row) * mat.step(0);
        for (int index = 0; index < scalars_per_row; ++index)
        {
            state = p1_lcg_next(state);
            output[index] = static_cast<uchar>(
                (state >> 24) ^ static_cast<std::uint32_t>(index + row * 17));
        }
    }
}

void p1_fill_f32(cvh::Mat& mat, std::uint32_t seed)
{
    std::uint32_t state = seed;
    const int scalars_per_row = mat.size[1] * mat.channels();
    for (int row = 0; row < mat.size[0]; ++row)
    {
        float* output = reinterpret_cast<float*>(
            mat.data + static_cast<std::size_t>(row) * mat.step(0));
        for (int index = 0; index < scalars_per_row; ++index)
        {
            state = p1_lcg_next(state);
            output[index] =
                static_cast<float>(static_cast<int>(state & 0xFFFFu) - 32768) /
                4096.0f;
        }
    }
}

void p1_make_positive(cvh::Mat& mat)
{
    const int scalars_per_row = mat.size[1] * mat.channels();
    for (int row = 0; row < mat.size[0]; ++row)
    {
        float* values = reinterpret_cast<float*>(
            mat.data + static_cast<std::size_t>(row) * mat.step(0));
        for (int index = 0; index < scalars_per_row; ++index)
        {
            values[index] = std::fabs(values[index]) * 0.1f + 0.25f;
        }
    }
}

void p1_fill_identity_maps(cvh::Mat& map_x, cvh::Mat& map_y)
{
    for (int row = 0; row < map_x.size[0]; ++row)
    {
        for (int col = 0; col < map_x.size[1]; ++col)
        {
            map_x.at<float>(row, col) = static_cast<float>(col) + 0.25f;
            map_y.at<float>(row, col) = static_cast<float>(row) + 0.5f;
        }
    }
}

double p1_checksum(const cvh::Mat& mat)
{
    return static_cast<double>(cvh_bench::common::checksum_mat_bytes(mat));
}

double p1_rotation_value(const cvh::AffineMatrix2x3d& matrix)
{
    return matrix.val[0];
}

template <typename RunFn, typename ProbeFn>
double measure_ms(RunFn&& run,
                  ProbeFn&& probe,
                  int warmup,
                  int iters,
                  int repeats)
{
    const auto timing = cvh_bench::common::measure_repeated_ms(
        std::forward<RunFn>(run), warmup, iters, repeats);
    g_phase1_cvh_sink += static_cast<double>(probe());
    return timing.min_ms;
}

#define P1_BENCH_FUNCTION bench_cvh_phase1
#define P1_NAMESPACE cvh
#define P1_MAT cvh::Mat
#define P1_POINT_TYPE cvh::Point
#define P1_POINT2F_TYPE cvh::Point2f
#define P1_SIZE_TYPE cvh::Size
#include "opencv_compare_phase1_cases.inl"
#undef P1_SIZE_TYPE
#undef P1_POINT2F_TYPE
#undef P1_POINT_TYPE
#undef P1_MAT
#undef P1_NAMESPACE
#undef P1_BENCH_FUNCTION

struct Phase1CaseSpec
{
    Phase1OpId id;
    const char* suite;
    const char* op;
    const char* variant;
    const char* depth;
    int channels;
    bool micro;
};

const std::vector<Phase1CaseSpec>& phase1_case_specs()
{
    static const std::vector<Phase1CaseSpec> specs = {
        {Phase1OpId::Absdiff, "core", "ABSDIFF", "mat_mat_u8c3", "CV_8U", 3, false},
        {Phase1OpId::BitwiseAnd, "core", "BITWISE_AND", "mat_mat_u8c3", "CV_8U", 3, false},
        {Phase1OpId::BitwiseNot, "core", "BITWISE_NOT", "u8c3", "CV_8U", 3, false},
        {Phase1OpId::BitwiseOr, "core", "BITWISE_OR", "mat_mat_u8c3", "CV_8U", 3, false},
        {Phase1OpId::BitwiseXor, "core", "BITWISE_XOR", "mat_mat_u8c3", "CV_8U", 3, false},
        {Phase1OpId::InRange, "core", "IN_RANGE", "scalar_bounds_u8c3", "CV_8U", 3, false},
        {Phase1OpId::Min, "core", "MIN", "mat_mat_u8c3", "CV_8U", 3, false},
        {Phase1OpId::Max, "core", "MAX", "mat_mat_u8c3", "CV_8U", 3, false},
        {Phase1OpId::ScaleAdd, "core", "SCALE_ADD", "f32c3", "CV_32F", 3, false},
        {Phase1OpId::ConvertScaleAbs, "core", "CONVERT_SCALE_ABS", "f32c3_to_u8c3", "CV_32F", 3, false},
        {Phase1OpId::ConvertFp16, "core", "CONVERT_FP16", "f32c1_to_fp16", "CV_32F", 1, false},
        {Phase1OpId::Sqrt, "core", "SQRT", "positive_f32c1", "CV_32F", 1, false},
        {Phase1OpId::Pow, "core", "POW", "power_1_75_f32c1", "CV_32F", 1, false},
        {Phase1OpId::Exp, "core", "EXP", "bounded_f32c1", "CV_32F", 1, false},
        {Phase1OpId::Log, "core", "LOG", "positive_f32c1", "CV_32F", 1, false},
        {Phase1OpId::CheckRange, "core", "CHECK_RANGE", "quiet_f32c1", "CV_32F", 1, false},
        {Phase1OpId::PatchNaNs, "core", "PATCH_NANS", "one_nan_f32c1", "CV_32F", 1, false},
        {Phase1OpId::Norm, "core", "NORM", "l2_f32c1", "CV_32F", 1, false},
        {Phase1OpId::Sum, "core", "SUM", "f32c3", "CV_32F", 3, false},
        {Phase1OpId::Mean, "core", "MEAN", "f32c3", "CV_32F", 3, false},
        {Phase1OpId::MeanStdDev, "core", "MEAN_STD_DEV", "f32c3", "CV_32F", 3, false},
        {Phase1OpId::CountNonZero, "core", "COUNT_NON_ZERO", "u8c1", "CV_8U", 1, false},
        {Phase1OpId::HasNonZero, "core", "HAS_NON_ZERO", "u8c1", "CV_8U", 1, false},
        {Phase1OpId::FindNonZero, "core", "FIND_NON_ZERO", "u8c1", "CV_8U", 1, false},
        {Phase1OpId::MinMaxIdx, "core", "MIN_MAX_IDX", "f32c1", "CV_32F", 1, false},
        {Phase1OpId::MinMaxLoc, "core", "MIN_MAX_LOC", "f32c1", "CV_32F", 1, false},
        {Phase1OpId::Reduce, "core", "REDUCE", "axis0_sum_f32c1", "CV_32F", 1, false},
        {Phase1OpId::ReduceArgMax, "core", "REDUCE_ARG_MAX", "axis0_f32c1", "CV_32F", 1, false},
        {Phase1OpId::ReduceArgMin, "core", "REDUCE_ARG_MIN", "axis0_f32c1", "CV_32F", 1, false},
        {Phase1OpId::Normalize, "core", "NORMALIZE", "l2_f32c1", "CV_32F", 1, false},
        {Phase1OpId::BorderInterpolate, "core", "BORDER_INTERPOLATE", "reflect101_batch4096", "S32", 1, true},
        {Phase1OpId::CopyTo, "core", "COPY_TO", "masked_u8c3", "CV_8U", 3, false},
        {Phase1OpId::ExtractChannel, "core", "EXTRACT_CHANNEL", "channel1_u8c3", "CV_8U", 3, false},
        {Phase1OpId::InsertChannel, "core", "INSERT_CHANNEL", "channel1_u8c3", "CV_8U", 3, false},
        {Phase1OpId::MixChannels, "core", "MIX_CHANNELS", "reverse_u8c3", "CV_8U", 3, false},
        {Phase1OpId::Flip, "core", "FLIP", "horizontal_u8c3", "CV_8U", 3, false},
        {Phase1OpId::FlipND, "core", "FLIP_ND", "axis1_u8c3", "CV_8U", 3, false},
        {Phase1OpId::Rotate, "core", "ROTATE", "clockwise90_u8c3", "CV_8U", 3, false},
        {Phase1OpId::Repeat, "core", "REPEAT", "two_by_two_u8c1", "CV_8U", 1, false},
        {Phase1OpId::Hconcat, "core", "HCONCAT", "two_halves_u8c1", "CV_8U", 1, false},
        {Phase1OpId::Vconcat, "core", "VCONCAT", "two_halves_u8c1", "CV_8U", 1, false},
        {Phase1OpId::Broadcast, "core", "BROADCAST", "row_to_image_u8c1", "CV_8U", 1, false},
        {Phase1OpId::Swap, "core", "SWAP", "mat_headers", "CV_8U", 1, true},
        {Phase1OpId::GetStructuringElement, "imgproc", "GET_STRUCTURING_ELEMENT", "ellipse7x7", "CV_8U", 1, true},
        {Phase1OpId::GetGaussianKernel, "imgproc", "GET_GAUSSIAN_KERNEL", "ksize15_f32", "CV_32F", 1, true},
        {Phase1OpId::GetDerivKernels, "imgproc", "GET_DERIV_KERNELS", "dx1_ksize5_f32", "CV_32F", 1, true},
        {Phase1OpId::GetGaborKernel, "imgproc", "GET_GABOR_KERNEL", "15x15_f32", "CV_32F", 1, true},
        {Phase1OpId::CreateHanningWindow, "imgproc", "CREATE_HANNING_WINDOW", "64x64_f32", "CV_32F", 1, false},
        {Phase1OpId::Integral, "imgproc", "INTEGRAL", "u8c1_to_s32", "CV_8U", 1, false},
        {Phase1OpId::Scharr, "imgproc", "SCHARR", "dx1_u8_to_f32", "CV_8U", 1, false},
        {Phase1OpId::Laplacian, "imgproc", "LAPLACIAN", "ksize3_u8_to_f32", "CV_8U", 1, false},
        {Phase1OpId::SpatialGradient, "imgproc", "SPATIAL_GRADIENT", "ksize3_u8_to_s16", "CV_8U", 1, false},
        {Phase1OpId::SqrBoxFilter, "imgproc", "SQR_BOX_FILTER", "3x3_u8_to_f32", "CV_8U", 1, false},
        {Phase1OpId::MedianBlur, "imgproc", "MEDIAN_BLUR", "ksize5_u8c1", "CV_8U", 1, false},
        {Phase1OpId::BilateralFilter, "imgproc", "BILATERAL_FILTER", "d5_u8c1", "CV_8U", 1, false},
        {Phase1OpId::StackBlur, "imgproc", "STACK_BLUR", "5x5_u8c1", "CV_8U", 1, false},
        {Phase1OpId::AdaptiveThreshold, "imgproc", "ADAPTIVE_THRESHOLD", "mean11_u8c1", "CV_8U", 1, false},
        {Phase1OpId::ThresholdWithMask, "imgproc", "THRESHOLD_WITH_MASK", "binary_masked_u8c1", "CV_8U", 1, false},
        {Phase1OpId::EqualizeHist, "imgproc", "EQUALIZE_HIST", "u8c1", "CV_8U", 1, false},
        {Phase1OpId::ApplyColorMap, "imgproc", "APPLY_COLOR_MAP", "jet_u8c1", "CV_8U", 1, false},
        {Phase1OpId::Accumulate, "imgproc", "ACCUMULATE", "u8c1_to_f32", "CV_8U", 1, false},
        {Phase1OpId::AccumulateProduct, "imgproc", "ACCUMULATE_PRODUCT", "u8c1_to_f32", "CV_8U", 1, false},
        {Phase1OpId::AccumulateSquare, "imgproc", "ACCUMULATE_SQUARE", "u8c1_to_f32", "CV_8U", 1, false},
        {Phase1OpId::AccumulateWeighted, "imgproc", "ACCUMULATE_WEIGHTED", "alpha0_1_u8c1", "CV_8U", 1, false},
        {Phase1OpId::BlendLinear, "imgproc", "BLEND_LINEAR", "u8c3_f32_weights", "CV_8U", 3, false},
        {Phase1OpId::PyrDown, "imgproc", "PYR_DOWN", "u8c3", "CV_8U", 3, false},
        {Phase1OpId::PyrUp, "imgproc", "PYR_UP", "u8c3", "CV_8U", 3, false},
        {Phase1OpId::BuildPyramid, "imgproc", "BUILD_PYRAMID", "levels3_u8c1", "CV_8U", 1, false},
        {Phase1OpId::CvtColorTwoPlane, "imgproc", "CVT_COLOR_TWO_PLANE", "nv12_to_bgr", "CV_8U", 1, false},
        {Phase1OpId::Demosaicing, "imgproc", "DEMOSAICING", "bayer_bg_to_bgr", "CV_8U", 1, false},
        {Phase1OpId::ConvertMaps, "imgproc", "CONVERT_MAPS", "f32_pair_to_fixed", "CV_32F", 2, false},
        {Phase1OpId::GetAffineTransform, "imgproc", "GET_AFFINE_TRANSFORM", "three_points", "CV_32F", 1, true},
        {Phase1OpId::GetPerspectiveTransform, "imgproc", "GET_PERSPECTIVE_TRANSFORM", "four_points_lu", "CV_32F", 1, true},
        {Phase1OpId::GetRotationMatrix2D, "imgproc", "GET_ROTATION_MATRIX_2D", "point_angle_scale", "CV_32F", 1, true},
        {Phase1OpId::GetRotationMatrix2DUnderscore, "imgproc", "GET_ROTATION_MATRIX_2D_", "matx23d", "CV_64F", 1, true},
        {Phase1OpId::InvertAffineTransform, "imgproc", "INVERT_AFFINE_TRANSFORM", "f64_2x3", "CV_64F", 1, true},
    };
    return specs;
}

}  // namespace

std::vector<Phase1BenchmarkResult> run_phase1_benchmarks(
    const Phase1BenchmarkConfig& config)
{
    const int rows =
        config.profile == "quick" ? 120 : (config.profile == "stable" ? 240 : 480);
    const int cols =
        config.profile == "quick" ? 160 : (config.profile == "stable" ? 320 : 640);
    const std::uint32_t seed = 0x51A7u;
    std::vector<Phase1BenchmarkResult> results;
    results.reserve(phase1_case_specs().size());

    if (phase1_case_specs().size() != 76u)
    {
        throw std::logic_error("P1 Mode B additional case list must contain 76 operations");
    }

    for (const Phase1CaseSpec& spec : phase1_case_specs())
    {
        int warmup = config.warmup;
        int iters = config.iters;
        if (spec.micro)
        {
            warmup = std::max(20, config.warmup * 20);
            iters = std::max(1000, config.iters * 1000);
        }
        const double cvh_ms = bench_cvh_phase1(
            spec.id, rows, cols, warmup, iters, config.repeats, seed);
        const double opencv_ms = bench_opencv_phase1(
            spec.id, rows, cols, warmup, iters, config.repeats, seed);
        if (cvh_ms <= 0.0 || opencv_ms <= 0.0)
        {
            throw std::runtime_error(
                std::string("P1 benchmark failed for ") + spec.op);
        }

        Phase1BenchmarkResult result;
        result.suite =
            std::string(spec.suite) == "core" ? "core_mat" : spec.suite;
        result.op = spec.op;
        result.variant = spec.variant;
        result.dispatch_path = "public_header_baseline";
        result.depth = spec.depth;
        result.channels = spec.channels;
        result.shape =
            spec.micro
                ? "micro_batch"
                : std::to_string(rows) + "x" + std::to_string(cols);
        result.cvh_ms = cvh_ms;
        result.opencv_ms = opencv_ms;
        result.note =
            spec.micro
                ? "phase1_representative_case;micro_iterations=" +
                      std::to_string(iters)
                : "phase1_representative_case";
        results.push_back(std::move(result));
    }

    return results;
}

}  // namespace cvh_bench_compare
