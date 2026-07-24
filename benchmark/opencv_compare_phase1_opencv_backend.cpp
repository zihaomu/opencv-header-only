#include "opencv_compare_phase1_benchmark.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace cvh_bench_compare {
namespace {

volatile double g_phase1_opencv_sink = 0.0;

std::uint32_t p1_lcg_next(std::uint32_t state)
{
    return state * 1664525u + 1013904223u;
}

cv::Mat p1_make_mat(int rows, int cols, int type)
{
    return cv::Mat(rows, cols, type);
}

void p1_fill_u8(cv::Mat& mat, std::uint32_t seed)
{
    std::uint32_t state = seed;
    const int scalars_per_row = mat.cols * mat.channels();
    for (int row = 0; row < mat.rows; ++row)
    {
        unsigned char* output = mat.ptr<unsigned char>(row);
        for (int index = 0; index < scalars_per_row; ++index)
        {
            state = p1_lcg_next(state);
            output[index] = static_cast<unsigned char>(
                (state >> 24) ^ static_cast<std::uint32_t>(index + row * 17));
        }
    }
}

void p1_fill_f32(cv::Mat& mat, std::uint32_t seed)
{
    std::uint32_t state = seed;
    const int scalars_per_row = mat.cols * mat.channels();
    for (int row = 0; row < mat.rows; ++row)
    {
        float* output = mat.ptr<float>(row);
        for (int index = 0; index < scalars_per_row; ++index)
        {
            state = p1_lcg_next(state);
            output[index] =
                static_cast<float>(static_cast<int>(state & 0xFFFFu) - 32768) /
                4096.0f;
        }
    }
}

void p1_make_positive(cv::Mat& mat)
{
    const int scalars_per_row = mat.cols * mat.channels();
    for (int row = 0; row < mat.rows; ++row)
    {
        float* values = mat.ptr<float>(row);
        for (int index = 0; index < scalars_per_row; ++index)
        {
            values[index] = std::fabs(values[index]) * 0.1f + 0.25f;
        }
    }
}

void p1_fill_identity_maps(cv::Mat& map_x, cv::Mat& map_y)
{
    for (int row = 0; row < map_x.rows; ++row)
    {
        for (int col = 0; col < map_x.cols; ++col)
        {
            map_x.at<float>(row, col) = static_cast<float>(col) + 0.25f;
            map_y.at<float>(row, col) = static_cast<float>(row) + 0.5f;
        }
    }
}

double p1_checksum(const cv::Mat& mat)
{
    if (mat.empty())
    {
        return 0.0;
    }

    std::uint64_t hash = 1469598103934665603ull;
    for (int row = 0; row < mat.rows; ++row)
    {
        const unsigned char* values = mat.ptr<unsigned char>(row);
        const std::size_t bytes =
            static_cast<std::size_t>(mat.cols) * mat.elemSize();
        const std::size_t stride = std::max<std::size_t>(1, bytes / 128);
        for (std::size_t index = 0; index < bytes; index += stride)
        {
            hash ^= values[index];
            hash *= 1099511628211ull;
        }
    }
    return static_cast<double>(hash);
}

double p1_rotation_value(const cv::Matx23d& matrix)
{
    return matrix(0, 0);
}

template <typename RunFn, typename ProbeFn>
double measure_ms(RunFn&& run,
                  ProbeFn&& probe,
                  int warmup,
                  int iters,
                  int repeats)
{
    for (int index = 0; index < warmup; ++index)
    {
        run();
    }

    double best_ms = std::numeric_limits<double>::max();
    for (int repeat = 0; repeat < repeats; ++repeat)
    {
        const auto begin = std::chrono::steady_clock::now();
        for (int index = 0; index < iters; ++index)
        {
            run();
        }
        const auto end = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                end - begin)
                .count();
        best_ms =
            std::min(best_ms, elapsed_ms / static_cast<double>(iters));
        g_phase1_opencv_sink += static_cast<double>(probe());
    }
    return best_ms;
}

}  // namespace

#define P1_BENCH_FUNCTION bench_opencv_phase1
#define P1_NAMESPACE cv
#define P1_MAT cv::Mat
#define P1_POINT_TYPE cv::Point
#define P1_POINT2F_TYPE cv::Point2f
#define P1_SIZE_TYPE cv::Size
#include "opencv_compare_phase1_cases.inl"
#undef P1_SIZE_TYPE
#undef P1_POINT2F_TYPE
#undef P1_POINT_TYPE
#undef P1_MAT
#undef P1_NAMESPACE
#undef P1_BENCH_FUNCTION

}  // namespace cvh_bench_compare
