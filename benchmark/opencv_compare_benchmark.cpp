#include "cvh.h"

#include "opencv_compare_backend.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace cvh_bench_compare {

struct Args
{
    std::string profile = "quick";
    int warmup = 2;
    int iters = 20;
    int repeats = 5;
    std::string impl;
    std::string output_csv;
};

struct CompareRow
{
    std::string profile;
    std::string op;
    std::string depth;
    int channels = 0;
    std::string shape;
    double cvh_ms = -1.0;
    double opencv_ms = -1.0;
    double speedup = 0.0;
    std::string status;
    std::string note;
};

struct SizeCase
{
    int rows = 0;
    int cols = 0;
};

struct GemmCase
{
    int m = 0;
    int k = 0;
    int n = 0;
};

volatile double g_sink = 0.0;

std::string default_impl_name()
{
#if defined(CVH_FULL)
    return "full";
#else
    return "lite";
#endif
}

std::string depth_to_name(DepthId depth)
{
    return depth == DepthId::U8 ? "CV_8U" : "CV_32F";
}

std::string shape_to_string(int rows, int cols)
{
    std::ostringstream oss;
    oss << rows << "x" << cols;
    return oss.str();
}

std::string gemm_shape_to_string(const GemmCase& c)
{
    std::ostringstream oss;
    oss << c.m << "x" << c.k << "x" << c.n;
    return oss.str();
}

int cvh_depth(DepthId depth)
{
    return depth == DepthId::U8 ? CV_8U : CV_32F;
}

std::uint32_t lcg_next(std::uint32_t state)
{
    return state * 1664525u + 1013904223u;
}

void fill_u8(cvh::Mat& mat, std::uint32_t seed)
{
    const int rows = mat.size[0];
    const int cols = mat.size[1];
    const int scalars_per_row = cols * mat.channels();
    const size_t step = mat.step(0);
    std::uint32_t state = seed;

    for (int y = 0; y < rows; ++y)
    {
        uchar* row = mat.data + static_cast<size_t>(y) * step;
        for (int x = 0; x < scalars_per_row; ++x)
        {
            state = lcg_next(state);
            row[x] = static_cast<uchar>((state >> 24) ^ static_cast<std::uint32_t>(x + y * 17));
        }
    }
}

void fill_f32(cvh::Mat& mat, std::uint32_t seed)
{
    const int rows = mat.size[0];
    const int cols = mat.size[1];
    const int scalars_per_row = cols * mat.channels();
    const size_t step = mat.step(0);
    std::uint32_t state = seed;

    for (int y = 0; y < rows; ++y)
    {
        float* row = reinterpret_cast<float*>(mat.data + static_cast<size_t>(y) * step);
        for (int x = 0; x < scalars_per_row; ++x)
        {
            state = lcg_next(state);
            const float v = static_cast<float>(static_cast<int>(state & 0xFFFFu) - 32768) / 4096.0f;
            row[x] = v;
        }
    }
}

void fill_by_depth(cvh::Mat& mat, DepthId depth, std::uint32_t seed)
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

double checksum(const cvh::Mat& mat)
{
    if (mat.empty())
    {
        return 0.0;
    }

    const int rows = mat.size[0];
    const int cols = mat.size[1];
    const int channels = mat.channels();
    const int scalars_per_row = cols * channels;
    const int stride = std::max(1, scalars_per_row / 64);
    const size_t step = mat.step(0);

    double sum = 0.0;
    if (mat.depth() == CV_8U)
    {
        for (int y = 0; y < rows; ++y)
        {
            const uchar* row = mat.data + static_cast<size_t>(y) * step;
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
            const float* row = reinterpret_cast<const float*>(mat.data + static_cast<size_t>(y) * step);
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
        g_sink += probe();
    }

    return best_ms;
}

double safe_speedup(double cvh_ms, double opencv_ms)
{
    if (cvh_ms <= 0.0 || opencv_ms <= 0.0)
    {
        return 0.0;
    }
    return opencv_ms / cvh_ms;
}

Args parse_args(int argc, char** argv)
{
    Args args;
    args.impl = default_impl_name();
    for (int i = 1; i < argc; ++i)
    {
        const std::string token(argv[i]);
        auto next_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for " << name << "\n";
                std::exit(2);
            }
            return std::string(argv[++i]);
        };

        if (token == "--profile")
        {
            args.profile = next_value("--profile");
        }
        else if (token == "--warmup")
        {
            args.warmup = std::max(0, std::stoi(next_value("--warmup")));
        }
        else if (token == "--iters")
        {
            args.iters = std::max(1, std::stoi(next_value("--iters")));
        }
        else if (token == "--repeats")
        {
            args.repeats = std::max(1, std::stoi(next_value("--repeats")));
        }
        else if (token == "--output")
        {
            args.output_csv = next_value("--output");
        }
        else if (token == "--impl")
        {
            args.impl = next_value("--impl");
        }
        else if (token == "--help")
        {
            std::cout << "Usage: cvh_benchmark_compare [--profile quick|stable|full] [--warmup N] [--iters N] "
                         "[--repeats N] [--impl full|lite] [--output path]\n";
            std::exit(0);
        }
        else
        {
            std::cerr << "Unknown arg: " << token << "\n";
            std::exit(2);
        }
    }

    if (args.profile != "quick" && args.profile != "stable" && args.profile != "full")
    {
        std::cerr << "Unsupported profile: " << args.profile << " (expected quick/stable/full)\n";
        std::exit(2);
    }
    if (args.impl.empty())
    {
        std::cerr << "impl must not be empty\n";
        std::exit(2);
    }

    return args;
}

std::vector<SizeCase> elementwise_shapes(const std::string& profile)
{
    if (profile == "full")
    {
        return {{480, 640}, {720, 1280}};
    }
    return {{480, 640}};
}

std::vector<SizeCase> filter_shapes(const std::string& profile)
{
    if (profile == "full")
    {
        return {{480, 640}, {720, 1280}};
    }
    return {{480, 640}};
}

std::vector<GemmCase> gemm_shapes(const std::string& profile)
{
    if (profile == "full")
    {
        return {{256, 256, 256}, {512, 512, 512}};
    }
    return {{256, 256, 256}};
}

void append_add_sub_rows(const Args& args, std::vector<CompareRow>& rows)
{
    const std::vector<DepthId> depths = {DepthId::U8, DepthId::F32};
    const std::vector<int> channels = {1, 3, 4};
    const auto shapes = elementwise_shapes(args.profile);

    for (const auto& shape : shapes)
    {
        for (DepthId depth : depths)
        {
            for (int cn : channels)
            {
                const int type = CV_MAKETYPE(cvh_depth(depth), cn);

                cvh::Mat a_cvh(std::vector<int>{shape.rows, shape.cols}, type);
                cvh::Mat b_cvh(std::vector<int>{shape.rows, shape.cols}, type);
                cvh::Mat dst_cvh;

                constexpr std::uint32_t seed_a = 0xA1u;
                constexpr std::uint32_t seed_b = 0xB2u;
                fill_by_depth(a_cvh, depth, seed_a);
                fill_by_depth(b_cvh, depth, seed_b);

                for (int op_idx = 0; op_idx < 2; ++op_idx)
                {
                    const bool is_add = (op_idx == 0);
                    const std::string op_name = is_add ? "ADD" : "SUB";

                    const double cvh_ms = measure_ms(
                        [&]() {
                            if (is_add)
                            {
                                cvh::add(a_cvh, b_cvh, dst_cvh);
                            }
                            else
                            {
                                cvh::subtract(a_cvh, b_cvh, dst_cvh);
                            }
                        },
                        [&]() { return checksum(dst_cvh); },
                        args.warmup,
                        args.iters,
                        args.repeats);

                    const double opencv_ms = is_add
                        ? bench_opencv_add(shape.rows,
                                           shape.cols,
                                           depth,
                                           cn,
                                           args.warmup,
                                           args.iters,
                                           args.repeats,
                                           seed_a,
                                           seed_b)
                        : bench_opencv_sub(shape.rows,
                                           shape.cols,
                                           depth,
                                           cn,
                                           args.warmup,
                                           args.iters,
                                           args.repeats,
                                           seed_a,
                                           seed_b);

                    CompareRow row;
                    row.profile = args.profile;
                    row.op = op_name;
                    row.depth = depth_to_name(depth);
                    row.channels = cn;
                    row.shape = shape_to_string(shape.rows, shape.cols);
                    row.cvh_ms = cvh_ms;
                    row.opencv_ms = opencv_ms;
                    row.speedup = safe_speedup(cvh_ms, opencv_ms);
                    row.status = "OK";
                    rows.push_back(row);
                }
            }
        }
    }
}

void append_gemm_rows(const Args& args, std::vector<CompareRow>& rows)
{
    const auto shapes = gemm_shapes(args.profile);

    for (const auto& shape : shapes)
    {
#if defined(CVH_FULL)
        cvh::Mat a_cvh(std::vector<int>{shape.m, shape.k}, CV_32F);
        cvh::Mat b_cvh(std::vector<int>{shape.k, shape.n}, CV_32F);
        constexpr std::uint32_t seed_a = 0xC3u;
        constexpr std::uint32_t seed_b = 0xD4u;
        fill_f32(a_cvh, seed_a);
        fill_f32(b_cvh, seed_b);
        cvh::Mat dst_cvh;

        const double cvh_ms = measure_ms(
            [&]() { dst_cvh = cvh::gemm(a_cvh, b_cvh, false, false); },
            [&]() { return checksum(dst_cvh); },
            args.warmup,
            args.iters,
            args.repeats);

        const double opencv_ms = bench_opencv_gemm(shape.m,
                                                   shape.k,
                                                   shape.n,
                                                   args.warmup,
                                                   args.iters,
                                                   args.repeats,
                                                   seed_a,
                                                   seed_b);

        CompareRow row;
        row.profile = args.profile;
        row.op = "GEMM";
        row.depth = "CV_32F";
        row.channels = 1;
        row.shape = gemm_shape_to_string(shape);
        row.cvh_ms = cvh_ms;
        row.opencv_ms = opencv_ms;
        row.speedup = safe_speedup(cvh_ms, opencv_ms);
        row.status = "OK";
        rows.push_back(row);

        const cvh::GemmPackedB packed_b = cvh::gemm_pack_b(b_cvh, false);
        const double cvh_prepack_ms = measure_ms(
            [&]() { dst_cvh = cvh::gemm(a_cvh, packed_b, false); },
            [&]() { return checksum(dst_cvh); },
            args.warmup,
            args.iters,
            args.repeats);

        const double opencv_prepack_ms = bench_opencv_gemm_prepack(shape.m,
                                                                    shape.k,
                                                                    shape.n,
                                                                    args.warmup,
                                                                    args.iters,
                                                                    args.repeats,
                                                                    seed_a,
                                                                    seed_b);

        CompareRow prepack_row;
        prepack_row.profile = args.profile;
        prepack_row.op = "GEMM_PREPACK";
        prepack_row.depth = "CV_32F";
        prepack_row.channels = 1;
        prepack_row.shape = gemm_shape_to_string(shape);
        prepack_row.cvh_ms = cvh_prepack_ms;
        prepack_row.opencv_ms = opencv_prepack_ms;
        prepack_row.speedup = safe_speedup(cvh_prepack_ms, opencv_prepack_ms);
        prepack_row.status = "OK";
        prepack_row.note = "pack_B_once";
        rows.push_back(prepack_row);
#else
        constexpr std::uint32_t seed_a = 0xC3u;
        constexpr std::uint32_t seed_b = 0xD4u;
        const double opencv_ms = bench_opencv_gemm(shape.m,
                                                   shape.k,
                                                   shape.n,
                                                   args.warmup,
                                                   args.iters,
                                                   args.repeats,
                                                   seed_a,
                                                   seed_b);

        CompareRow row;
        row.profile = args.profile;
        row.op = "GEMM";
        row.depth = "CV_32F";
        row.channels = 1;
        row.shape = gemm_shape_to_string(shape);
        row.cvh_ms = -1.0;
        row.opencv_ms = opencv_ms;
        row.speedup = 0.0;
        row.status = "UNSUPPORTED";
        row.note = "requires_CVH_FULL_backend";
        rows.push_back(row);

        const double opencv_prepack_ms = bench_opencv_gemm_prepack(shape.m,
                                                                    shape.k,
                                                                    shape.n,
                                                                    args.warmup,
                                                                    args.iters,
                                                                    args.repeats,
                                                                    seed_a,
                                                                    seed_b);

        CompareRow prepack_row;
        prepack_row.profile = args.profile;
        prepack_row.op = "GEMM_PREPACK";
        prepack_row.depth = "CV_32F";
        prepack_row.channels = 1;
        prepack_row.shape = gemm_shape_to_string(shape);
        prepack_row.cvh_ms = -1.0;
        prepack_row.opencv_ms = opencv_prepack_ms;
        prepack_row.speedup = 0.0;
        prepack_row.status = "UNSUPPORTED";
        prepack_row.note = "requires_CVH_FULL_backend";
        rows.push_back(prepack_row);
#endif
    }
}

void append_filter_rows(const Args& args, std::vector<CompareRow>& rows)
{
    const auto shapes = filter_shapes(args.profile);
    const std::vector<DepthId> depths = {DepthId::U8, DepthId::F32};
    const std::vector<int> channels = {1, 3, 4};
    const std::vector<int> kernels = {3, 5, 11};

    for (const auto& shape : shapes)
    {
        for (DepthId depth : depths)
        {
            for (int cn : channels)
            {
                for (int ksize : kernels)
                {
                    const int type = CV_MAKETYPE(cvh_depth(depth), cn);
                    cvh::Mat src_cvh(std::vector<int>{shape.rows, shape.cols}, type);
                    cvh::Mat dst_cvh;

                    const std::uint32_t seed = static_cast<std::uint32_t>(0xE5u + ksize * 17 + cn);
                    fill_by_depth(src_cvh, depth, seed);

                    const double cvh_gaussian_ms = measure_ms(
                        [&]() {
                            cvh::GaussianBlur(src_cvh,
                                              dst_cvh,
                                              cvh::Size(ksize, ksize),
                                              0.0,
                                              0.0,
                                              cvh::BORDER_DEFAULT);
                        },
                        [&]() { return checksum(dst_cvh); },
                        args.warmup,
                        args.iters,
                        args.repeats);

                    const double opencv_gaussian_ms = bench_opencv_gaussian(shape.rows,
                                                                            shape.cols,
                                                                            depth,
                                                                            cn,
                                                                            ksize,
                                                                            args.warmup,
                                                                            args.iters,
                                                                            args.repeats,
                                                                            seed);

                    CompareRow gaussian_row;
                    gaussian_row.profile = args.profile;
                    gaussian_row.op = "GAUSSIAN_" + std::to_string(ksize) + "X" + std::to_string(ksize);
                    gaussian_row.depth = depth_to_name(depth);
                    gaussian_row.channels = cn;
                    gaussian_row.shape = shape_to_string(shape.rows, shape.cols);
                    gaussian_row.cvh_ms = cvh_gaussian_ms;
                    gaussian_row.opencv_ms = opencv_gaussian_ms;
                    gaussian_row.speedup = safe_speedup(cvh_gaussian_ms, opencv_gaussian_ms);
                    gaussian_row.status = "OK";
                    rows.push_back(gaussian_row);

                    const double cvh_box_ms = measure_ms(
                        [&]() {
                            cvh::boxFilter(src_cvh,
                                           dst_cvh,
                                           -1,
                                           cvh::Size(ksize, ksize),
                                           cvh::Point(-1, -1),
                                           true,
                                           cvh::BORDER_DEFAULT);
                        },
                        [&]() { return checksum(dst_cvh); },
                        args.warmup,
                        args.iters,
                        args.repeats);

                    const double opencv_box_ms = bench_opencv_box(shape.rows,
                                                                  shape.cols,
                                                                  depth,
                                                                  cn,
                                                                  ksize,
                                                                  args.warmup,
                                                                  args.iters,
                                                                  args.repeats,
                                                                  seed);

                    CompareRow box_row;
                    box_row.profile = args.profile;
                    box_row.op = "BOX_" + std::to_string(ksize) + "X" + std::to_string(ksize);
                    box_row.depth = depth_to_name(depth);
                    box_row.channels = cn;
                    box_row.shape = shape_to_string(shape.rows, shape.cols);
                    box_row.cvh_ms = cvh_box_ms;
                    box_row.opencv_ms = opencv_box_ms;
                    box_row.speedup = safe_speedup(cvh_box_ms, opencv_box_ms);
                    box_row.status = "OK";
                    rows.push_back(box_row);
                }
            }
        }
    }
}

void append_morph_gradient_rows(const Args& args, std::vector<CompareRow>& rows)
{
    const auto shapes = filter_shapes(args.profile);
    const std::vector<int> channels = {1, 3, 4};

    for (const auto& shape : shapes)
    {
        for (int cn : channels)
        {
            const std::uint32_t sobel_seed = 0x91u + static_cast<std::uint32_t>(cn);
            const std::uint32_t canny_seed = 0x9Du + static_cast<std::uint32_t>(cn);
            const std::uint32_t erode_seed = 0xA3u + static_cast<std::uint32_t>(cn);
            const std::uint32_t dilate_seed = 0xB5u + static_cast<std::uint32_t>(cn);

            cvh::Mat src_sobel(std::vector<int>{shape.rows, shape.cols}, CV_MAKETYPE(CV_8U, cn));
            cvh::Mat src_erode(std::vector<int>{shape.rows, shape.cols}, CV_MAKETYPE(CV_8U, cn));
            cvh::Mat src_dilate(std::vector<int>{shape.rows, shape.cols}, CV_MAKETYPE(CV_8U, cn));
            fill_u8(src_sobel, sobel_seed);
            fill_u8(src_erode, erode_seed);
            fill_u8(src_dilate, dilate_seed);

            cvh::Mat dst_sobel;
            cvh::Mat dst_erode;
            cvh::Mat dst_dilate;

            const double cvh_sobel_ms = measure_ms(
                [&]() {
                    cvh::Sobel(
                        src_sobel, dst_sobel, CV_32F, 1, 0, 3, 1.0, 0.0, cvh::BORDER_DEFAULT);
                },
                [&]() { return checksum(dst_sobel); },
                args.warmup,
                args.iters,
                args.repeats);

            const double cvh_erode_ms = measure_ms(
                [&]() {
                    cvh::erode(
                        src_erode, dst_erode, cvh::Mat(), cvh::Point(-1, -1), 1, cvh::BORDER_DEFAULT);
                },
                [&]() { return checksum(dst_erode); },
                args.warmup,
                args.iters,
                args.repeats);

            const double cvh_dilate_ms = measure_ms(
                [&]() {
                    cvh::dilate(
                        src_dilate, dst_dilate, cvh::Mat(), cvh::Point(-1, -1), 1, cvh::BORDER_DEFAULT);
                },
                [&]() { return checksum(dst_dilate); },
                args.warmup,
                args.iters,
                args.repeats);

            const double opencv_sobel_ms = bench_opencv_sobel(
                shape.rows, shape.cols, cn, args.warmup, args.iters, args.repeats, sobel_seed);
            const double opencv_erode_ms = bench_opencv_erode(
                shape.rows, shape.cols, cn, args.warmup, args.iters, args.repeats, erode_seed);
            const double opencv_dilate_ms = bench_opencv_dilate(
                shape.rows, shape.cols, cn, args.warmup, args.iters, args.repeats, dilate_seed);

            CompareRow sobel_row;
            sobel_row.profile = args.profile;
            sobel_row.op = "SOBEL";
            sobel_row.depth = "CV_8U";
            sobel_row.channels = cn;
            sobel_row.shape = shape_to_string(shape.rows, shape.cols);
            sobel_row.cvh_ms = cvh_sobel_ms;
            sobel_row.opencv_ms = opencv_sobel_ms;
            sobel_row.speedup = safe_speedup(cvh_sobel_ms, opencv_sobel_ms);
            sobel_row.status = "OK";
            rows.push_back(sobel_row);

            if (cn == 1)
            {
                cvh::Mat src_canny(std::vector<int>{shape.rows, shape.cols}, CV_8UC1);
                cvh::Mat dst_canny;
                fill_u8(src_canny, canny_seed);
                struct CannyMode
                {
                    const char* name;
                    double threshold1;
                    double threshold2;
                    int aperture;
                    bool l2;
                };

                const CannyMode canny_modes[] = {
                    {"CANNY_A3_L1", 100.0, 200.0, 3, false},
                    {"CANNY_A3_L2", 100.0, 200.0, 3, true},
                    {"CANNY_A5_L1", 300.0, 600.0, 5, false},
                    {"CANNY_A5_L2", 300.0, 600.0, 5, true},
                };

                for (const CannyMode& mode : canny_modes)
                {
                    const double cvh_canny_ms = measure_ms(
                        [&]() {
                            cvh::Canny(src_canny,
                                       dst_canny,
                                       mode.threshold1,
                                       mode.threshold2,
                                       mode.aperture,
                                       mode.l2);
                        },
                        [&]() { return checksum(dst_canny); },
                        args.warmup,
                        args.iters,
                        args.repeats);

                    const double opencv_canny_ms = bench_opencv_canny(shape.rows,
                                                                      shape.cols,
                                                                      args.warmup,
                                                                      args.iters,
                                                                      args.repeats,
                                                                      canny_seed,
                                                                      mode.threshold1,
                                                                      mode.threshold2,
                                                                      mode.aperture,
                                                                      mode.l2);

                    CompareRow canny_row;
                    canny_row.profile = args.profile;
                    canny_row.op = mode.name;
                    canny_row.depth = "CV_8U";
                    canny_row.channels = 1;
                    canny_row.shape = shape_to_string(shape.rows, shape.cols);
                    canny_row.cvh_ms = cvh_canny_ms;
                    canny_row.opencv_ms = opencv_canny_ms;
                    canny_row.speedup = safe_speedup(cvh_canny_ms, opencv_canny_ms);
                    canny_row.status = "OK";
                    rows.push_back(canny_row);
                }
            }

            CompareRow erode_row;
            erode_row.profile = args.profile;
            erode_row.op = "ERODE";
            erode_row.depth = "CV_8U";
            erode_row.channels = cn;
            erode_row.shape = shape_to_string(shape.rows, shape.cols);
            erode_row.cvh_ms = cvh_erode_ms;
            erode_row.opencv_ms = opencv_erode_ms;
            erode_row.speedup = safe_speedup(cvh_erode_ms, opencv_erode_ms);
            erode_row.status = "OK";
            rows.push_back(erode_row);

            CompareRow dilate_row;
            dilate_row.profile = args.profile;
            dilate_row.op = "DILATE";
            dilate_row.depth = "CV_8U";
            dilate_row.channels = cn;
            dilate_row.shape = shape_to_string(shape.rows, shape.cols);
            dilate_row.cvh_ms = cvh_dilate_ms;
            dilate_row.opencv_ms = opencv_dilate_ms;
            dilate_row.speedup = safe_speedup(cvh_dilate_ms, opencv_dilate_ms);
            dilate_row.status = "OK";
            rows.push_back(dilate_row);
        }
    }
}

void append_m2_rows(const Args& args, std::vector<CompareRow>& rows)
{
    const auto shapes = filter_shapes(args.profile);
    const std::vector<DepthId> depths = {DepthId::U8, DepthId::F32};
    const std::vector<int> channels = {1, 3, 4};

    for (const auto& shape : shapes)
    {
        for (DepthId depth : depths)
        {
            for (int cn : channels)
            {
                const int type = CV_MAKETYPE(cvh_depth(depth), cn);
                const std::uint32_t seed = static_cast<std::uint32_t>(0xC0u + cn * 19 + (depth == DepthId::U8 ? 1 : 7));

                cvh::Mat src_cvh(std::vector<int>{shape.rows, shape.cols}, type);
                fill_by_depth(src_cvh, depth, seed);

                cvh::Mat copy_dst;
                const double cvh_copy_ms = measure_ms(
                    [&]() {
                        cvh::copyMakeBorder(src_cvh,
                                            copy_dst,
                                            1,
                                            1,
                                            2,
                                            2,
                                            cvh::BORDER_REPLICATE,
                                            cvh::Scalar::all(0.0));
                    },
                    [&]() { return checksum(copy_dst); },
                    args.warmup,
                    args.iters,
                    args.repeats);

                const double opencv_copy_ms = bench_opencv_copy_make_border(shape.rows,
                                                                             shape.cols,
                                                                             depth,
                                                                             cn,
                                                                             1,
                                                                             1,
                                                                             2,
                                                                             2,
                                                                             args.warmup,
                                                                             args.iters,
                                                                             args.repeats,
                                                                             seed);

                CompareRow copy_row;
                copy_row.profile = args.profile;
                copy_row.op = "COPYMAKEBORDER";
                copy_row.depth = depth_to_name(depth);
                copy_row.channels = cn;
                copy_row.shape = shape_to_string(shape.rows, shape.cols);
                copy_row.cvh_ms = cvh_copy_ms;
                copy_row.opencv_ms = opencv_copy_ms;
                copy_row.speedup = safe_speedup(cvh_copy_ms, opencv_copy_ms);
                copy_row.status = "OK";
                rows.push_back(copy_row);

                cvh::Mat kernel2d({3, 3}, CV_32FC1);
                kernel2d.at<float>(0, 0) = 0.0f;
                kernel2d.at<float>(0, 1) = 0.25f;
                kernel2d.at<float>(0, 2) = 0.0f;
                kernel2d.at<float>(1, 0) = 0.25f;
                kernel2d.at<float>(1, 1) = 0.0f;
                kernel2d.at<float>(1, 2) = 0.25f;
                kernel2d.at<float>(2, 0) = 0.0f;
                kernel2d.at<float>(2, 1) = 0.25f;
                kernel2d.at<float>(2, 2) = 0.0f;

                cvh::Mat filter_dst;
                const double cvh_filter_ms = measure_ms(
                    [&]() {
                        cvh::filter2D(src_cvh, filter_dst, -1, kernel2d, cvh::Point(-1, -1), 0.0, cvh::BORDER_DEFAULT);
                    },
                    [&]() { return checksum(filter_dst); },
                    args.warmup,
                    args.iters,
                    args.repeats);

                const double opencv_filter_ms = bench_opencv_filter2d(shape.rows,
                                                                      shape.cols,
                                                                      depth,
                                                                      cn,
                                                                      args.warmup,
                                                                      args.iters,
                                                                      args.repeats,
                                                                      seed);

                CompareRow filter_row;
                filter_row.profile = args.profile;
                filter_row.op = "FILTER2D_3X3";
                filter_row.depth = depth_to_name(depth);
                filter_row.channels = cn;
                filter_row.shape = shape_to_string(shape.rows, shape.cols);
                filter_row.cvh_ms = cvh_filter_ms;
                filter_row.opencv_ms = opencv_filter_ms;
                filter_row.speedup = safe_speedup(cvh_filter_ms, opencv_filter_ms);
                filter_row.status = "OK";
                rows.push_back(filter_row);

                cvh::Mat kernel_x({1, 3}, CV_32FC1);
                kernel_x.at<float>(0, 0) = 0.25f;
                kernel_x.at<float>(0, 1) = 0.5f;
                kernel_x.at<float>(0, 2) = 0.25f;
                cvh::Mat kernel_y({3, 1}, CV_32FC1);
                kernel_y.at<float>(0, 0) = 0.25f;
                kernel_y.at<float>(1, 0) = 0.5f;
                kernel_y.at<float>(2, 0) = 0.25f;

                cvh::Mat sep_dst;
                const double cvh_sep_ms = measure_ms(
                    [&]() {
                        cvh::sepFilter2D(src_cvh,
                                         sep_dst,
                                         -1,
                                         kernel_x,
                                         kernel_y,
                                         cvh::Point(-1, -1),
                                         0.0,
                                         cvh::BORDER_DEFAULT);
                    },
                    [&]() { return checksum(sep_dst); },
                    args.warmup,
                    args.iters,
                    args.repeats);

                const double opencv_sep_ms = bench_opencv_sep_filter2d(shape.rows,
                                                                        shape.cols,
                                                                        depth,
                                                                        cn,
                                                                        args.warmup,
                                                                        args.iters,
                                                                        args.repeats,
                                                                        seed);

                CompareRow sep_row;
                sep_row.profile = args.profile;
                sep_row.op = "SEPFILTER2D_3X3";
                sep_row.depth = depth_to_name(depth);
                sep_row.channels = cn;
                sep_row.shape = shape_to_string(shape.rows, shape.cols);
                sep_row.cvh_ms = cvh_sep_ms;
                sep_row.opencv_ms = opencv_sep_ms;
                sep_row.speedup = safe_speedup(cvh_sep_ms, opencv_sep_ms);
                sep_row.status = "OK";
                rows.push_back(sep_row);

                cvh::Mat warp_mat({2, 3}, CV_32FC1);
                warp_mat.at<float>(0, 0) = 1.0f;
                warp_mat.at<float>(0, 1) = 0.0f;
                warp_mat.at<float>(0, 2) = -1.25f;
                warp_mat.at<float>(1, 0) = 0.0f;
                warp_mat.at<float>(1, 1) = 1.0f;
                warp_mat.at<float>(1, 2) = 0.75f;

                cvh::Mat warp_dst;
                const double cvh_warp_ms = measure_ms(
                    [&]() {
                        cvh::warpAffine(src_cvh,
                                        warp_dst,
                                        warp_mat,
                                        cvh::Size(shape.cols, shape.rows),
                                        cvh::INTER_LINEAR | cvh::WARP_INVERSE_MAP,
                                        cvh::BORDER_REPLICATE,
                                        cvh::Scalar::all(0.0));
                    },
                    [&]() { return checksum(warp_dst); },
                    args.warmup,
                    args.iters,
                    args.repeats);

                const double opencv_warp_ms = bench_opencv_warp_affine(shape.rows,
                                                                        shape.cols,
                                                                        depth,
                                                                        cn,
                                                                        args.warmup,
                                                                        args.iters,
                                                                        args.repeats,
                                                                        seed);

                CompareRow warp_row;
                warp_row.profile = args.profile;
                warp_row.op = "WARP_AFFINE";
                warp_row.depth = depth_to_name(depth);
                warp_row.channels = cn;
                warp_row.shape = shape_to_string(shape.rows, shape.cols);
                warp_row.cvh_ms = cvh_warp_ms;
                warp_row.opencv_ms = opencv_warp_ms;
                warp_row.speedup = safe_speedup(cvh_warp_ms, opencv_warp_ms);
                warp_row.status = "OK";
                rows.push_back(warp_row);
            }
        }
    }

    for (const auto& shape : shapes)
    {
        for (int cn : channels)
        {
            const std::uint32_t seed = static_cast<std::uint32_t>(0xE1u + cn * 23);
            cvh::Mat src_cvh(std::vector<int>{shape.rows, shape.cols}, CV_MAKETYPE(CV_8U, cn));
            fill_u8(src_cvh, seed);

            cvh::Mat lut_table({1, 256}, CV_8UC1);
            for (int i = 0; i < 256; ++i)
            {
                lut_table.at<uchar>(0, i) = static_cast<uchar>(255 - i);
            }

            cvh::Mat lut_dst;
            const double cvh_lut_ms = measure_ms(
                [&]() { cvh::LUT(src_cvh, lut_table, lut_dst); },
                [&]() { return checksum(lut_dst); },
                args.warmup,
                args.iters,
                args.repeats);

            const double opencv_lut_ms = bench_opencv_lut(shape.rows,
                                                          shape.cols,
                                                          cn,
                                                          args.warmup,
                                                          args.iters,
                                                          args.repeats,
                                                          seed);

            CompareRow lut_row;
            lut_row.profile = args.profile;
            lut_row.op = "LUT";
            lut_row.depth = "CV_8U";
            lut_row.channels = cn;
            lut_row.shape = shape_to_string(shape.rows, shape.cols);
            lut_row.cvh_ms = cvh_lut_ms;
            lut_row.opencv_ms = opencv_lut_ms;
            lut_row.speedup = safe_speedup(cvh_lut_ms, opencv_lut_ms);
            lut_row.status = "OK";
            rows.push_back(lut_row);
        }
    }
}

void write_csv(const std::vector<CompareRow>& rows, const std::string& path, const std::string& impl)
{
    std::ofstream out(path);
    if (!out)
    {
        std::cerr << "Failed to open output CSV: " << path << "\n";
        std::exit(2);
    }

    out << "impl,profile,op,depth,channels,shape,cvh_ms,opencv_ms,speedup,status,note\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& row : rows)
    {
        out << impl << ","
            << row.profile << ","
            << row.op << ","
            << row.depth << ","
            << row.channels << ","
            << row.shape << ","
            << row.cvh_ms << ","
            << row.opencv_ms << ","
            << row.speedup << ","
            << row.status << ","
            << row.note << "\n";
    }
}

}  // namespace cvh_bench_compare

int main(int argc, char** argv)
{
    using namespace cvh_bench_compare;

    Args args = parse_args(argc, argv);

    std::vector<CompareRow> rows;
    rows.reserve(256);

    append_add_sub_rows(args, rows);
    append_gemm_rows(args, rows);
    append_filter_rows(args, rows);
    append_morph_gradient_rows(args, rows);
    append_m2_rows(args, rows);

    if (!args.output_csv.empty())
    {
        write_csv(rows, args.output_csv, args.impl);
    }

    int supported = 0;
    int unsupported = 0;
    for (const auto& row : rows)
    {
        if (row.status == "OK")
        {
            ++supported;
        }
        else
        {
            ++unsupported;
        }
    }

    std::cout << "cvh_benchmark_compare: impl=" << args.impl
              << ", profile=" << args.profile
              << ", rows=" << rows.size()
              << ", supported=" << supported
              << ", unsupported=" << unsupported
              << ", sink=" << g_sink << "\n";
    if (!args.output_csv.empty())
    {
        std::cout << "cvh_benchmark_compare_output: " << args.output_csv << "\n";
    }

    return 0;
}
