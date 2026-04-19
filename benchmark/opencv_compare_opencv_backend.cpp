#include "opencv_compare_backend.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>

namespace cvh_bench_compare {
namespace {

volatile double g_backend_sink = 0.0;

std::uint32_t lcg_next(std::uint32_t state)
{
    return state * 1664525u + 1013904223u;
}

int to_cv_depth(DepthId depth)
{
    return depth == DepthId::U8 ? CV_8U : CV_32F;
}

void fill_u8(cv::Mat& mat, std::uint32_t seed)
{
    const int rows = mat.rows;
    const int cols = mat.cols;
    const int scalars_per_row = cols * mat.channels();
    std::uint32_t state = seed;

    for (int y = 0; y < rows; ++y)
    {
        uchar* row = mat.ptr<uchar>(y);
        for (int x = 0; x < scalars_per_row; ++x)
        {
            state = lcg_next(state);
            row[x] = static_cast<uchar>((state >> 24) ^ static_cast<std::uint32_t>(x + y * 17));
        }
    }
}

void fill_f32(cv::Mat& mat, std::uint32_t seed)
{
    const int rows = mat.rows;
    const int cols = mat.cols;
    const int scalars_per_row = cols * mat.channels();
    std::uint32_t state = seed;

    for (int y = 0; y < rows; ++y)
    {
        float* row = mat.ptr<float>(y);
        for (int x = 0; x < scalars_per_row; ++x)
        {
            state = lcg_next(state);
            const float v = static_cast<float>(static_cast<int>(state & 0xFFFFu) - 32768) / 4096.0f;
            row[x] = v;
        }
    }
}

void fill_by_depth(cv::Mat& mat, DepthId depth, std::uint32_t seed)
{
    if (depth == DepthId::U8)
    {
        fill_u8(mat, seed);
    }
    else
    {
        fill_f32(mat, seed);
    }
}

double checksum(const cv::Mat& mat)
{
    if (mat.empty())
    {
        return 0.0;
    }

    const int rows = mat.rows;
    const int cols = mat.cols;
    const int channels = mat.channels();
    const int scalars_per_row = cols * channels;
    const int stride = std::max(1, scalars_per_row / 64);

    double sum = 0.0;
    if (mat.depth() == CV_8U)
    {
        for (int y = 0; y < rows; ++y)
        {
            const uchar* row = mat.ptr<uchar>(y);
            for (int x = 0; x < scalars_per_row; x += stride)
            {
                sum += static_cast<double>(row[x]) * static_cast<double>(x + 1 + y);
            }
        }
    }
    else if (mat.depth() == CV_32F)
    {
        for (int y = 0; y < rows; ++y)
        {
            const float* row = mat.ptr<float>(y);
            for (int x = 0; x < scalars_per_row; x += stride)
            {
                sum += static_cast<double>(row[x]) * static_cast<double>((x % 13) + 1);
            }
        }
    }

    return sum;
}

template <typename RunFn, typename ProbeFn>
double measure_ms(RunFn&& run, ProbeFn&& probe, int warmup, int iters, int repeats)
{
    for (int i = 0; i < warmup; ++i)
    {
        run();
    }

    double best_ms = std::numeric_limits<double>::max();
    for (int rep = 0; rep < repeats; ++rep)
    {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i)
        {
            run();
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double total_ms =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
        best_ms = std::min(best_ms, total_ms / static_cast<double>(iters));
        g_backend_sink += probe();
    }

    return best_ms;
}

}  // namespace

double bench_opencv_add(int rows,
                        int cols,
                        DepthId depth,
                        int channels,
                        int warmup,
                        int iters,
                        int repeats,
                        std::uint32_t seed_a,
                        std::uint32_t seed_b)
{
    const int type = CV_MAKETYPE(to_cv_depth(depth), channels);
    cv::Mat a(rows, cols, type);
    cv::Mat b(rows, cols, type);
    cv::Mat dst;

    fill_by_depth(a, depth, seed_a);
    fill_by_depth(b, depth, seed_b);

    return measure_ms(
        [&]() { cv::add(a, b, dst); },
        [&]() { return checksum(dst); },
        warmup,
        iters,
        repeats);
}

double bench_opencv_sub(int rows,
                        int cols,
                        DepthId depth,
                        int channels,
                        int warmup,
                        int iters,
                        int repeats,
                        std::uint32_t seed_a,
                        std::uint32_t seed_b)
{
    const int type = CV_MAKETYPE(to_cv_depth(depth), channels);
    cv::Mat a(rows, cols, type);
    cv::Mat b(rows, cols, type);
    cv::Mat dst;

    fill_by_depth(a, depth, seed_a);
    fill_by_depth(b, depth, seed_b);

    return measure_ms(
        [&]() { cv::subtract(a, b, dst); },
        [&]() { return checksum(dst); },
        warmup,
        iters,
        repeats);
}

double bench_opencv_gemm(int m,
                         int k,
                         int n,
                         int warmup,
                         int iters,
                         int repeats,
                         std::uint32_t seed_a,
                         std::uint32_t seed_b)
{
    cv::Mat a(m, k, CV_32F);
    cv::Mat b(k, n, CV_32F);
    cv::Mat dst;

    fill_f32(a, seed_a);
    fill_f32(b, seed_b);

    return measure_ms(
        [&]() { cv::gemm(a, b, 1.0, cv::Mat(), 0.0, dst, 0); },
        [&]() { return checksum(dst); },
        warmup,
        iters,
        repeats);
}

double bench_opencv_gemm_prepack(int m,
                                 int k,
                                 int n,
                                 int warmup,
                                 int iters,
                                 int repeats,
                                 std::uint32_t seed_a,
                                 std::uint32_t seed_b)
{
    // OpenCV public API has no explicit B-pack handle; this measures repeated gemm with the same B matrix.
    return bench_opencv_gemm(m, k, n, warmup, iters, repeats, seed_a, seed_b);
}

double bench_opencv_gaussian(int rows,
                             int cols,
                             DepthId depth,
                             int channels,
                             int ksize,
                             int warmup,
                             int iters,
                             int repeats,
                             std::uint32_t seed)
{
    const int type = CV_MAKETYPE(to_cv_depth(depth), channels);
    cv::Mat src(rows, cols, type);
    cv::Mat dst;

    fill_by_depth(src, depth, seed);

    return measure_ms(
        [&]() { cv::GaussianBlur(src, dst, cv::Size(ksize, ksize), 0.0, 0.0, cv::BORDER_DEFAULT); },
        [&]() { return checksum(dst); },
        warmup,
        iters,
        repeats);
}

double bench_opencv_box(int rows,
                        int cols,
                        DepthId depth,
                        int channels,
                        int ksize,
                        int warmup,
                        int iters,
                        int repeats,
                        std::uint32_t seed)
{
    const int type = CV_MAKETYPE(to_cv_depth(depth), channels);
    cv::Mat src(rows, cols, type);
    cv::Mat dst;

    fill_by_depth(src, depth, seed);

    return measure_ms(
        [&]() {
            cv::boxFilter(
                src, dst, -1, cv::Size(ksize, ksize), cv::Point(-1, -1), true, cv::BORDER_DEFAULT);
        },
        [&]() { return checksum(dst); },
        warmup,
        iters,
        repeats);
}

double bench_opencv_lut(int rows,
                        int cols,
                        int channels,
                        int warmup,
                        int iters,
                        int repeats,
                        std::uint32_t seed)
{
    cv::Mat src(rows, cols, CV_MAKETYPE(CV_8U, channels));
    cv::Mat dst;
    cv::Mat lut(1, 256, CV_8UC1);
    fill_u8(src, seed);

    for (int i = 0; i < 256; ++i)
    {
        lut.at<uchar>(0, i) = static_cast<uchar>(255 - i);
    }

    return measure_ms(
        [&]() { cv::LUT(src, lut, dst); },
        [&]() { return checksum(dst); },
        warmup,
        iters,
        repeats);
}

double bench_opencv_copy_make_border(int rows,
                                     int cols,
                                     DepthId depth,
                                     int channels,
                                     int top,
                                     int bottom,
                                     int left,
                                     int right,
                                     int warmup,
                                     int iters,
                                     int repeats,
                                     std::uint32_t seed)
{
    const int type = CV_MAKETYPE(to_cv_depth(depth), channels);
    cv::Mat src(rows, cols, type);
    cv::Mat dst;
    fill_by_depth(src, depth, seed);

    return measure_ms(
        [&]() {
            cv::copyMakeBorder(
                src, dst, top, bottom, left, right, cv::BORDER_REPLICATE, cv::Scalar::all(0.0));
        },
        [&]() { return checksum(dst); },
        warmup,
        iters,
        repeats);
}

double bench_opencv_filter2d(int rows,
                             int cols,
                             DepthId depth,
                             int channels,
                             int warmup,
                             int iters,
                             int repeats,
                             std::uint32_t seed)
{
    const int type = CV_MAKETYPE(to_cv_depth(depth), channels);
    cv::Mat src(rows, cols, type);
    cv::Mat dst;
    fill_by_depth(src, depth, seed);

    cv::Mat kernel(3, 3, CV_32FC1);
    kernel.at<float>(0, 0) = 0.0f;
    kernel.at<float>(0, 1) = 0.25f;
    kernel.at<float>(0, 2) = 0.0f;
    kernel.at<float>(1, 0) = 0.25f;
    kernel.at<float>(1, 1) = 0.0f;
    kernel.at<float>(1, 2) = 0.25f;
    kernel.at<float>(2, 0) = 0.0f;
    kernel.at<float>(2, 1) = 0.25f;
    kernel.at<float>(2, 2) = 0.0f;

    return measure_ms(
        [&]() { cv::filter2D(src, dst, -1, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_DEFAULT); },
        [&]() { return checksum(dst); },
        warmup,
        iters,
        repeats);
}

double bench_opencv_sep_filter2d(int rows,
                                 int cols,
                                 DepthId depth,
                                 int channels,
                                 int warmup,
                                 int iters,
                                 int repeats,
                                 std::uint32_t seed)
{
    const int type = CV_MAKETYPE(to_cv_depth(depth), channels);
    cv::Mat src(rows, cols, type);
    cv::Mat dst;
    fill_by_depth(src, depth, seed);

    cv::Mat kernelX(1, 3, CV_32FC1);
    kernelX.at<float>(0, 0) = 0.25f;
    kernelX.at<float>(0, 1) = 0.5f;
    kernelX.at<float>(0, 2) = 0.25f;
    cv::Mat kernelY(3, 1, CV_32FC1);
    kernelY.at<float>(0, 0) = 0.25f;
    kernelY.at<float>(1, 0) = 0.5f;
    kernelY.at<float>(2, 0) = 0.25f;

    return measure_ms(
        [&]() {
            cv::sepFilter2D(src, dst, -1, kernelX, kernelY, cv::Point(-1, -1), 0.0, cv::BORDER_DEFAULT);
        },
        [&]() { return checksum(dst); },
        warmup,
        iters,
        repeats);
}

double bench_opencv_warp_affine(int rows,
                                int cols,
                                DepthId depth,
                                int channels,
                                int warmup,
                                int iters,
                                int repeats,
                                std::uint32_t seed)
{
    const int type = CV_MAKETYPE(to_cv_depth(depth), channels);
    cv::Mat src(rows, cols, type);
    cv::Mat dst;
    fill_by_depth(src, depth, seed);

    cv::Mat M(2, 3, CV_32FC1);
    M.at<float>(0, 0) = 1.0f;
    M.at<float>(0, 1) = 0.0f;
    M.at<float>(0, 2) = -1.25f;
    M.at<float>(1, 0) = 0.0f;
    M.at<float>(1, 1) = 1.0f;
    M.at<float>(1, 2) = 0.75f;

    return measure_ms(
        [&]() {
            cv::warpAffine(src,
                           dst,
                           M,
                           cv::Size(cols, rows),
                           cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                           cv::BORDER_REPLICATE,
                           cv::Scalar::all(0.0));
        },
        [&]() { return checksum(dst); },
        warmup,
        iters,
        repeats);
}

double bench_opencv_sobel(int rows,
                          int cols,
                          int channels,
                          int warmup,
                          int iters,
                          int repeats,
                          std::uint32_t seed)
{
    cv::Mat src(rows, cols, CV_MAKETYPE(CV_8U, channels));
    cv::Mat dst;
    fill_u8(src, seed);

    return measure_ms(
        [&]() { cv::Sobel(src, dst, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_DEFAULT); },
        [&]() { return checksum(dst); },
        warmup,
        iters,
        repeats);
}

double bench_opencv_canny(int rows,
                          int cols,
                          int warmup,
                          int iters,
                          int repeats,
                          std::uint32_t seed,
                          double threshold1,
                          double threshold2,
                          int aperture_size,
                          bool l2gradient)
{
    cv::Mat src(rows, cols, CV_8UC1);
    cv::Mat dst;
    fill_u8(src, seed);

    return measure_ms(
        [&]() { cv::Canny(src, dst, threshold1, threshold2, aperture_size, l2gradient); },
        [&]() { return checksum(dst); },
        warmup,
        iters,
        repeats);
}

double bench_opencv_erode(int rows,
                          int cols,
                          int channels,
                          int warmup,
                          int iters,
                          int repeats,
                          std::uint32_t seed)
{
    cv::Mat src(rows, cols, CV_MAKETYPE(CV_8U, channels));
    cv::Mat dst;
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    fill_u8(src, seed);

    return measure_ms(
        [&]() { cv::erode(src, dst, kernel, cv::Point(-1, -1), 1, cv::BORDER_DEFAULT); },
        [&]() { return checksum(dst); },
        warmup,
        iters,
        repeats);
}

double bench_opencv_dilate(int rows,
                           int cols,
                           int channels,
                           int warmup,
                           int iters,
                           int repeats,
                           std::uint32_t seed)
{
    cv::Mat src(rows, cols, CV_MAKETYPE(CV_8U, channels));
    cv::Mat dst;
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    fill_u8(src, seed);

    return measure_ms(
        [&]() { cv::dilate(src, dst, kernel, cv::Point(-1, -1), 1, cv::BORDER_DEFAULT); },
        [&]() { return checksum(dst); },
        warmup,
        iters,
        repeats);
}

}  // namespace cvh_bench_compare
