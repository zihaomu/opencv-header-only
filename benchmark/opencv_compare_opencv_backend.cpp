#include "opencv_compare_backend.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iostream>
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

void run_binary(CoreBinaryOpId op, const cv::Mat& a, const cv::Mat& b, cv::Mat& dst)
{
    switch (op)
    {
        case CoreBinaryOpId::Add:
            cv::add(a, b, dst);
            return;
        case CoreBinaryOpId::Subtract:
            cv::subtract(a, b, dst);
            return;
        case CoreBinaryOpId::Multiply:
            cv::multiply(a, b, dst);
            return;
        case CoreBinaryOpId::Divide:
            cv::divide(a, b, dst);
            return;
    }
}

bool output_matches(const cv::Mat& expected,
                    DepthId depth,
                    const void* actual_data,
                    std::uint64_t actual_bytes,
                    int u8_tolerance = 0)
{
    const std::uint64_t expected_bytes =
        static_cast<std::uint64_t>(expected.total()) * static_cast<std::uint64_t>(expected.elemSize());
    if (!expected.isContinuous() || actual_data == nullptr || actual_bytes != expected_bytes)
    {
        return false;
    }
    if (depth == DepthId::U8)
    {
        const unsigned char* actual = static_cast<const unsigned char*>(actual_data);
        for (std::size_t i = 0; i < static_cast<std::size_t>(expected_bytes); ++i)
        {
            if (std::abs(static_cast<int>(expected.data[i]) - static_cast<int>(actual[i])) >
                u8_tolerance)
            {
                std::cerr << "OpenCV U8 mismatch at scalar " << i
                          << ": expected=" << static_cast<int>(expected.data[i])
                          << " actual=" << static_cast<int>(actual[i]) << "\n";
                return false;
            }
        }
        return true;
    }

    const float* expected_values = expected.ptr<float>();
    const float* actual_values = static_cast<const float*>(actual_data);
    const std::size_t count = expected.total() * static_cast<std::size_t>(expected.channels());
    for (std::size_t i = 0; i < count; ++i)
    {
        const float tolerance = 1e-5f * std::max(1.0f, std::fabs(expected_values[i]));
        if (std::fabs(expected_values[i] - actual_values[i]) > tolerance)
        {
            return false;
        }
    }
    return true;
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

void configure_opencv_threads(int threads)
{
    cv::setNumThreads(std::max(1, threads));
    cv::setUseOptimized(true);
}

double bench_opencv_mat_op(MatOpId op,
                           int rows,
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

    switch (op)
    {
        case MatOpId::CreateReuse:
            dst.create(rows, cols, type);
            return measure_ms(
                [&]() { dst.create(rows, cols, type); },
                [&]() { return static_cast<double>(dst.total()); },
                warmup,
                iters,
                repeats);
        case MatOpId::Clone:
            return measure_ms(
                [&]() { dst = src.clone(); },
                [&]() { return checksum(dst); },
                warmup,
                iters,
                repeats);
        case MatOpId::CopyTo:
            return measure_ms(
                [&]() { src.copyTo(dst); },
                [&]() { return checksum(dst); },
                warmup,
                iters,
                repeats);
        case MatOpId::SetTo:
            dst.create(rows, cols, type);
            return measure_ms(
                [&]() { dst.setTo(cv::Scalar::all(7.0)); },
                [&]() { return checksum(dst); },
                warmup,
                iters,
                repeats);
        case MatOpId::ConvertTo:
        {
            const int dst_depth = depth == DepthId::U8 ? CV_32F : CV_8U;
            return measure_ms(
                [&]() { src.convertTo(dst, CV_MAKETYPE(dst_depth, channels)); },
                [&]() { return checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case MatOpId::Reshape:
            return measure_ms(
                [&]() { dst = src.reshape(channels, rows * cols); },
                [&]() { return static_cast<double>(dst.total()); },
                warmup,
                iters,
                repeats);
    }

    return -1.0;
}

double bench_opencv_binary(CoreBinaryOpId op,
                           int rows,
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
        [&]() { run_binary(op, a, b, dst); },
        [&]() { return checksum(dst); },
        warmup,
        iters,
        repeats);
}

double bench_opencv_transpose(int rows,
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

    return measure_ms(
        [&]() { cv::transpose(src, dst); },
        [&]() { return checksum(dst); },
        warmup,
        iters,
        repeats);
}

bool validate_opencv_binary(CoreBinaryOpId op,
                            int rows,
                            int cols,
                            DepthId depth,
                            int channels,
                            std::uint32_t seed_a,
                            std::uint32_t seed_b,
                            const void* cvh_data,
                            std::uint64_t cvh_bytes)
{
    const int type = CV_MAKETYPE(to_cv_depth(depth), channels);
    cv::Mat a(rows, cols, type);
    cv::Mat b(rows, cols, type);
    cv::Mat expected;
    fill_by_depth(a, depth, seed_a);
    fill_by_depth(b, depth, seed_b);
    run_binary(op, a, b, expected);
    const int u8_tolerance = op == CoreBinaryOpId::Divide ? 1 : 0;
    return output_matches(expected, depth, cvh_data, cvh_bytes, u8_tolerance);
}

bool validate_opencv_transpose(int rows,
                               int cols,
                               DepthId depth,
                               int channels,
                               std::uint32_t seed,
                               const void* cvh_data,
                               std::uint64_t cvh_bytes)
{
    const int type = CV_MAKETYPE(to_cv_depth(depth), channels);
    cv::Mat src(rows, cols, type);
    cv::Mat expected;
    fill_by_depth(src, depth, seed);
    cv::transpose(src, expected);
    return output_matches(expected, depth, cvh_data, cvh_bytes);
}

bool validate_opencv_gemm(int m,
                          int k,
                          int n,
                          std::uint32_t seed_a,
                          std::uint32_t seed_b,
                          const void* cvh_data,
                          std::uint64_t cvh_bytes)
{
    cv::Mat a(m, k, CV_32F);
    cv::Mat b(k, n, CV_32F);
    cv::Mat expected;
    fill_f32(a, seed_a);
    fill_f32(b, seed_b);
    cv::gemm(a, b, 1.0, cv::Mat(), 0.0, expected, 0);
    return output_matches(expected, DepthId::F32, cvh_data, cvh_bytes);
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

double bench_opencv_resize_linear_half(int dst_rows,
                                        int dst_cols,
                                        int warmup,
                                        int iters,
                                        int repeats,
                                        std::uint32_t seed)
{
    cv::Mat src(dst_rows * 2, dst_cols * 2, CV_8UC1);
    cv::Mat dst;
    fill_u8(src, seed);

    return measure_ms(
        [&]() { cv::resize(src, dst, cv::Size(dst_cols, dst_rows), 0.0, 0.0, cv::INTER_LINEAR); },
        [&]() { return checksum(dst); },
        warmup,
        iters,
        repeats);
}

double bench_opencv_resize(int src_rows,
                           int src_cols,
                           int dst_rows,
                           int dst_cols,
                           DepthId depth,
                           int channels,
                           int interpolation,
                           int warmup,
                           int iters,
                           int repeats,
                           std::uint32_t seed)
{
    cv::Mat src(src_rows, src_cols, CV_MAKETYPE(to_cv_depth(depth), channels));
    cv::Mat dst;
    fill_by_depth(src, depth, seed);
    return measure_ms(
        [&]() { cv::resize(src, dst, cv::Size(dst_cols, dst_rows), 0.0, 0.0, interpolation); },
        [&]() { return checksum(dst); },
        warmup,
        iters,
        repeats);
}

double bench_opencv_cvtcolor(ImgprocColorOpId op,
                             int rows,
                             int cols,
                             DepthId depth,
                             int warmup,
                             int iters,
                             int repeats,
                             std::uint32_t seed)
{
    int code = cv::COLOR_BGR2RGB;
    int src_channels = 3;
    int src_rows = rows;
    switch (op)
    {
        case ImgprocColorOpId::Bgr2Rgb:
            code = cv::COLOR_BGR2RGB;
            break;
        case ImgprocColorOpId::Bgr2Bgra:
            code = cv::COLOR_BGR2BGRA;
            break;
        case ImgprocColorOpId::Bgra2Gray:
            code = cv::COLOR_BGRA2GRAY;
            src_channels = 4;
            break;
        case ImgprocColorOpId::Bgr2Gray:
            code = cv::COLOR_BGR2GRAY;
            break;
        case ImgprocColorOpId::Bgr2Yuv:
            code = cv::COLOR_BGR2YUV;
            break;
        case ImgprocColorOpId::Yuv2Bgr:
            code = cv::COLOR_YUV2BGR;
            break;
        case ImgprocColorOpId::Bgr2YuvI420:
            code = cv::COLOR_BGR2YUV_I420;
            break;
        case ImgprocColorOpId::YuvI420ToBgr:
            code = cv::COLOR_YUV2BGR_I420;
            src_channels = 1;
            src_rows = rows * 3 / 2;
            break;
        case ImgprocColorOpId::Bgr2YuvYuy2:
            code = cv::COLOR_BGR2YUV_YUY2;
            break;
        case ImgprocColorOpId::YuvYuy2ToBgr:
            code = cv::COLOR_YUV2BGR_YUY2;
            src_channels = 2;
            break;
        case ImgprocColorOpId::YuvNv12ToBgr:
            code = cv::COLOR_YUV2BGR_NV12;
            src_channels = 1;
            src_rows = rows * 3 / 2;
            break;
    }

    cv::Mat src(src_rows, cols, CV_MAKETYPE(to_cv_depth(depth), src_channels));
    cv::Mat dst;
    fill_by_depth(src, depth, seed);
    return measure_ms(
        [&]() { cv::cvtColor(src, dst, code); },
        [&]() { return checksum(dst); },
        warmup,
        iters,
        repeats);
}

double bench_opencv_cvtcolor_bgr2gray(int rows,
                                      int cols,
                                      int warmup,
                                      int iters,
                                      int repeats,
                                      std::uint32_t seed)
{
    cv::Mat src(rows, cols, CV_8UC3);
    cv::Mat dst;
    fill_u8(src, seed);

    return measure_ms(
        [&]() { cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY); },
        [&]() { return checksum(dst); },
        warmup,
        iters,
        repeats);
}

double bench_opencv_threshold_binary(int rows,
                                      int cols,
                                      int warmup,
                                      int iters,
                                      int repeats,
                                      std::uint32_t seed)
{
    cv::Mat src(rows, cols, CV_8UC1);
    cv::Mat dst;
    fill_u8(src, seed);

    return measure_ms(
        [&]() { cv::threshold(src, dst, 127.0, 255.0, cv::THRESH_BINARY); },
        [&]() { return checksum(dst); },
        warmup,
        iters,
        repeats);
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
        [&]() { cv::GaussianBlur(src, dst, cv::Size(ksize, ksize), 0.0, 0.0, cv::BORDER_REPLICATE); },
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
                src, dst, -1, cv::Size(ksize, ksize), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
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
        [&]() { cv::filter2D(src, dst, -1, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE); },
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
            cv::sepFilter2D(src, dst, -1, kernelX, kernelY, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
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
        [&]() { cv::Sobel(src, dst, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE); },
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
        [&]() { cv::erode(src, dst, kernel, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE); },
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
        [&]() { cv::dilate(src, dst, kernel, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE); },
        [&]() { return checksum(dst); },
        warmup,
        iters,
        repeats);
}

double bench_opencv_imgproc_roi(ImgprocRoiOpId op,
                                int rows,
                                int cols,
                                int warmup,
                                int iters,
                                int repeats,
                                std::uint32_t seed)
{
    const bool is_f32 = op == ImgprocRoiOpId::ThresholdF32;
    const int type = is_f32 ? CV_32FC3 : CV_8UC3;
    cv::Mat owner(rows + 2, cols + 3, type);
    cv::Mat src = owner(cv::Rect(1, 1, cols, rows));
    cv::Mat dst;
    fill_by_depth(owner, is_f32 ? DepthId::F32 : DepthId::U8, seed);

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
    cv::Mat kernel_x(1, 3, CV_32FC1);
    cv::Mat kernel_y(3, 1, CV_32FC1);
    kernel_x.at<float>(0, 0) = 0.25f;
    kernel_x.at<float>(0, 1) = 0.5f;
    kernel_x.at<float>(0, 2) = 0.25f;
    kernel_y.at<float>(0, 0) = 0.25f;
    kernel_y.at<float>(1, 0) = 0.5f;
    kernel_y.at<float>(2, 0) = 0.25f;

    return measure_ms(
        [&]() {
            switch (op)
            {
                case ImgprocRoiOpId::ResizeLinear:
                    cv::resize(src, dst, cv::Size(cols * 3 / 4, rows * 3 / 4), 0.0, 0.0, cv::INTER_LINEAR);
                    break;
                case ImgprocRoiOpId::CvtColorBgr2Gray:
                    cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
                    break;
                case ImgprocRoiOpId::ThresholdF32:
                    cv::threshold(src, dst, 0.5, 1.0, cv::THRESH_BINARY);
                    break;
                case ImgprocRoiOpId::Box:
                    cv::boxFilter(
                        src, dst, -1, cv::Size(3, 3), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
                    break;
                case ImgprocRoiOpId::Gaussian:
                    cv::GaussianBlur(src, dst, cv::Size(5, 5), 0.0, 0.0, cv::BORDER_REPLICATE);
                    break;
                case ImgprocRoiOpId::Filter2D:
                    cv::filter2D(src, dst, -1, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
                    break;
                case ImgprocRoiOpId::SepFilter2D:
                    cv::sepFilter2D(
                        src, dst, -1, kernel_x, kernel_y, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
                    break;
            }
        },
        [&]() { return checksum(dst); },
        warmup,
        iters,
        repeats);
}

}  // namespace cvh_bench_compare
