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
        else if (token == "--help")
        {
            std::cout << "Usage: cvh_benchmark_compare [--profile quick|full] [--warmup N] [--iters N] "
                         "[--repeats N] [--output path]\n";
            std::exit(0);
        }
        else
        {
            std::cerr << "Unknown arg: " << token << "\n";
            std::exit(2);
        }
    }

    if (args.profile != "quick" && args.profile != "full")
    {
        std::cerr << "Unsupported profile: " << args.profile << " (expected quick/full)\n";
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
        cvh::Mat a_cvh(std::vector<int>{shape.m, shape.k}, CV_32F);
        cvh::Mat b_cvh(std::vector<int>{shape.k, shape.n}, CV_32F);
        cvh::Mat dst_cvh;

        constexpr std::uint32_t seed_a = 0xC3u;
        constexpr std::uint32_t seed_b = 0xD4u;
        fill_f32(a_cvh, seed_a);
        fill_f32(b_cvh, seed_b);

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

void append_opencv_only_rows(const Args& args, std::vector<CompareRow>& rows)
{
    const auto shapes = filter_shapes(args.profile);
    const std::vector<int> channels = {1, 3, 4};

    for (const auto& shape : shapes)
    {
        for (int cn : channels)
        {
            const double sobel_ms =
                bench_opencv_sobel(shape.rows, shape.cols, cn, args.warmup, args.iters, args.repeats, 0x91u + cn);
            const double erode_ms =
                bench_opencv_erode(shape.rows, shape.cols, cn, args.warmup, args.iters, args.repeats, 0xA3u + cn);
            const double dilate_ms =
                bench_opencv_dilate(shape.rows, shape.cols, cn, args.warmup, args.iters, args.repeats, 0xB5u + cn);

            for (int i = 0; i < 3; ++i)
            {
                CompareRow row;
                row.profile = args.profile;
                row.op = (i == 0) ? "SOBEL" : ((i == 1) ? "ERODE" : "DILATE");
                row.depth = "CV_8U";
                row.channels = cn;
                row.shape = shape_to_string(shape.rows, shape.cols);
                row.cvh_ms = -1.0;
                row.opencv_ms = (i == 0) ? sobel_ms : ((i == 1) ? erode_ms : dilate_ms);
                row.speedup = 0.0;
                row.status = "UNSUPPORTED_CVH";
                row.note = "cvh_api_not_available_yet";
                rows.push_back(row);
            }
        }
    }
}

void write_csv(const std::vector<CompareRow>& rows, const std::string& path)
{
    std::ofstream out(path);
    if (!out)
    {
        std::cerr << "Failed to open output CSV: " << path << "\n";
        std::exit(2);
    }

    out << "profile,op,depth,channels,shape,cvh_ms,opencv_ms,speedup,status,note\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& row : rows)
    {
        out << row.profile << ","
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
    append_opencv_only_rows(args, rows);

    if (!args.output_csv.empty())
    {
        write_csv(rows, args.output_csv);
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

    std::cout << "cvh_benchmark_compare: profile=" << args.profile
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
