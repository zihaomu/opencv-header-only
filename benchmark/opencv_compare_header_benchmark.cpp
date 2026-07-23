#include "cvh.h"
#include "common/benchmark_common.h"
#include "opencv_compare_backend.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifndef CVH_COMPARE_IMPL_NAME
#define CVH_COMPARE_IMPL_NAME "cvh_headers_fast"
#endif

namespace cvh_bench_compare {

struct Args
{
    std::string profile = "quick";
    std::string impl = CVH_COMPARE_IMPL_NAME;
    int warmup = 1;
    int iters = 5;
    int repeats = 1;
    int threads = 1;
    std::string output_csv;
};

struct CompareRow
{
    std::string impl;
    std::string profile;
    std::string suite;
    std::string op;
    std::string variant;
    std::string dispatch_path;
    std::string depth;
    int channels = 0;
    std::string shape;
    double cvh_ms = -1.0;
    double opencv_ms = -1.0;
    double speedup = 0.0;
    std::string status = "OK";
    std::string note;
};

struct ShapeCase
{
    int rows = 0;
    int cols = 0;
};

volatile std::uint64_t g_sink = 0;

void usage()
{
    std::cout
        << "Usage: cvh_benchmark_opencv_compare_headers_fast "
        << "[--profile quick|stable|full] [--warmup N] [--iters N] [--repeats N] "
        << "[--threads N] [--impl name] [--output path]\n";
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
        else if (token == "--threads")
        {
            args.threads = std::max(1, std::stoi(next_value("--threads")));
        }
        else if (token == "--impl")
        {
            args.impl = next_value("--impl");
        }
        else if (token == "--output")
        {
            args.output_csv = next_value("--output");
        }
        else if (token == "--help")
        {
            usage();
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
        std::cerr << "Unsupported profile: " << args.profile << "\n";
        std::exit(2);
    }
    if (args.impl == "headers_fast")
    {
        args.impl = "cvh_headers_fast";
    }
    if (args.impl != "cvh_headers_fast")
    {
        std::cerr << "Unsupported impl: " << args.impl
                  << " (Mode B only compares cvh_headers_fast against upstream OpenCV)\n";
        std::exit(2);
    }
    return args;
}

std::vector<ShapeCase> build_shapes(const std::string& profile)
{
    std::vector<ShapeCase> shapes = {{480, 640}};
    if (profile == "stable" || profile == "full")
    {
        shapes.push_back({720, 1280});
        shapes.push_back({1080, 1920});
    }
    if (profile == "full")
    {
        shapes.push_back({479, 641});
    }
    return shapes;
}

std::string shape_string(const ShapeCase& shape)
{
    std::ostringstream oss;
    oss << shape.rows << "x" << shape.cols;
    return oss.str();
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
    std::uint32_t state = seed;
    for (int y = 0; y < rows; ++y)
    {
        uchar* row = mat.data + static_cast<std::size_t>(y) * mat.step(0);
        for (int x = 0; x < scalars_per_row; ++x)
        {
            state = lcg_next(state);
            row[x] = static_cast<uchar>((state >> 24) ^ static_cast<std::uint32_t>(x + y * 17));
        }
    }
}

double safe_speedup(double cvh_ms, double opencv_ms)
{
    return cvh_ms > 0.0 && opencv_ms > 0.0 ? opencv_ms / cvh_ms : 0.0;
}

template <typename RunFn, typename ProbeFn>
double measure_cvh_ms(RunFn&& run, ProbeFn&& probe, const Args& args)
{
    const auto timing = cvh_bench::common::measure_repeated_ms(run, args.warmup, args.iters, args.repeats);
    g_sink ^= static_cast<std::uint64_t>(probe());
    return timing.min_ms;
}

template <typename RunFn>
double measure_cvh_mat_ms(RunFn&& run, cvh::Mat& dst, const Args& args)
{
    return measure_cvh_ms(
        std::forward<RunFn>(run),
        [&]() { return cvh_bench::common::checksum_mat_bytes(dst); },
        args);
}

void append_row(std::vector<CompareRow>& rows,
                const Args& args,
                const std::string& suite,
                const std::string& op,
                const std::string& variant,
                const std::string& dispatch_path,
                const std::string& depth,
                int channels,
                const std::string& shape,
                double cvh_ms,
                double opencv_ms,
                const std::string& note = "")
{
    CompareRow row;
    row.impl = args.impl;
    row.profile = args.profile;
    row.suite = suite;
    row.op = op;
    row.variant = variant;
    row.dispatch_path = dispatch_path;
    row.depth = depth;
    row.channels = channels;
    row.shape = shape;
    row.cvh_ms = cvh_ms;
    row.opencv_ms = opencv_ms;
    row.speedup = safe_speedup(cvh_ms, opencv_ms);
    row.note = note;
    rows.push_back(row);
}

void append_core_mat_cases(const Args& args, std::vector<CompareRow>& rows)
{
    constexpr std::uint32_t seed = 0xA501u;
    Args micro_args = args;
    micro_args.warmup = std::max(100, args.warmup * 100);
    micro_args.iters = args.iters * 1000;
    for (const auto& shape : build_shapes(args.profile))
    {
        const std::vector<int> dims {shape.rows, shape.cols};
        const std::string shape_name = shape_string(shape);
        cvh::Mat src(dims, CV_8UC1);
        fill_u8(src, seed);

        {
            cvh::Mat dst(dims, CV_8UC1);
            const double cvh_ms = measure_cvh_ms(
                [&]() { dst.create(dims, CV_8UC1); },
                [&]() { return dst.total(); },
                micro_args);
            const double opencv_ms = bench_opencv_mat_op(
                MatOpId::CreateReuse,
                shape.rows,
                shape.cols,
                DepthId::U8,
                1,
                micro_args.warmup,
                micro_args.iters,
                args.repeats,
                seed);
            append_row(
                rows,
                args,
                "core_mat",
                "MAT_CREATE",
                "reuse_same_shape",
                "headers_baseline",
                "CV_8U",
                1,
                shape_name,
                cvh_ms,
                opencv_ms,
                "cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000");
        }

        {
            cvh::Mat dst;
            const double cvh_ms = measure_cvh_mat_ms([&]() { dst = src.clone(); }, dst, args);
            const double opencv_ms = bench_opencv_mat_op(
                MatOpId::Clone,
                shape.rows,
                shape.cols,
                DepthId::U8,
                1,
                args.warmup,
                args.iters,
                args.repeats,
                seed);
            append_row(rows, args, "core_mat", "MAT_CLONE", "full_copy", "headers_baseline", "CV_8U", 1, shape_name, cvh_ms, opencv_ms);
        }

        {
            cvh::Mat dst;
            const double cvh_ms = measure_cvh_mat_ms([&]() { src.copyTo(dst); }, dst, args);
            const double opencv_ms = bench_opencv_mat_op(
                MatOpId::CopyTo,
                shape.rows,
                shape.cols,
                DepthId::U8,
                1,
                args.warmup,
                args.iters,
                args.repeats,
                seed);
            append_row(rows, args, "core_mat", "MAT_COPYTO", "continuous_reuse", "headers_baseline", "CV_8U", 1, shape_name, cvh_ms, opencv_ms);
        }

        {
            cvh::Mat dst(dims, CV_8UC1);
            const double cvh_ms = measure_cvh_mat_ms(
                [&]() { dst.setTo(cvh::Scalar::all(7.0)); },
                dst,
                args);
            const double opencv_ms = bench_opencv_mat_op(
                MatOpId::SetTo,
                shape.rows,
                shape.cols,
                DepthId::U8,
                1,
                args.warmup,
                args.iters,
                args.repeats,
                seed);
            append_row(rows, args, "core_mat", "MAT_SETTO", "scalar_all", "headers_baseline", "CV_8U", 1, shape_name, cvh_ms, opencv_ms);
        }

        {
            cvh::Mat dst;
            const double cvh_ms = measure_cvh_mat_ms([&]() { src.convertTo(dst, CV_32FC1); }, dst, args);
            const double opencv_ms = bench_opencv_mat_op(
                MatOpId::ConvertTo,
                shape.rows,
                shape.cols,
                DepthId::U8,
                1,
                args.warmup,
                args.iters,
                args.repeats,
                seed);
            append_row(rows, args, "core_mat", "MAT_CONVERTTO", "CV_8U_to_CV_32F", "headers_baseline", "CV_8U", 1, shape_name, cvh_ms, opencv_ms);
        }

        {
            cvh::Mat view;
            const std::vector<int> view_shape {shape.rows * shape.cols, 1};
            const double cvh_ms = measure_cvh_ms(
                [&]() { view = src.reshape(view_shape); },
                [&]() { return view.total(); },
                micro_args);
            const double opencv_ms = bench_opencv_mat_op(
                MatOpId::Reshape,
                shape.rows,
                shape.cols,
                DepthId::U8,
                1,
                micro_args.warmup,
                micro_args.iters,
                args.repeats,
                seed);
            append_row(
                rows,
                args,
                "core_mat",
                "MAT_RESHAPE",
                "to_column_view",
                "headers_baseline",
                "CV_8U",
                1,
                shape_name,
                cvh_ms,
                opencv_ms,
                "micro_iters_x1000");
        }
    }
}

void append_imgproc_cases(const Args& args, std::vector<CompareRow>& rows)
{
    constexpr std::uint32_t seed = 0xC001u;
    for (const auto& shape : build_shapes(args.profile))
    {
        const std::string shape_name = shape_string(shape);
        {
            cvh::Mat src({shape.rows * 2, shape.cols * 2}, CV_8UC1);
            cvh::Mat dst;
            fill_u8(src, seed);
            const double cvh_ms = measure_cvh_mat_ms(
                [&]() { cvh::resize(src, dst, cvh::Size(shape.cols, shape.rows), 0.0, 0.0, cvh::INTER_LINEAR); },
                dst,
                args);
            const double opencv_ms = bench_opencv_resize_linear_half(
                shape.rows, shape.cols, args.warmup, args.iters, args.repeats, seed);
            append_row(rows, args, "imgproc", "RESIZE", "linear_half_u8c1", "opencv_ui", "CV_8U", 1, shape_name, cvh_ms, opencv_ms);
        }

        {
            cvh::Mat src({shape.rows, shape.cols}, CV_8UC3);
            cvh::Mat dst;
            fill_u8(src, seed);
            const double cvh_ms = measure_cvh_mat_ms(
                [&]() { cvh::cvtColor(src, dst, cvh::COLOR_BGR2GRAY); },
                dst,
                args);
            const double opencv_ms = bench_opencv_cvtcolor_bgr2gray(
                shape.rows, shape.cols, args.warmup, args.iters, args.repeats, seed);
            append_row(rows, args, "imgproc", "CVTCOLOR", "BGR2GRAY_u8", "opencv_ui", "CV_8U", 3, shape_name, cvh_ms, opencv_ms);
        }

        {
            cvh::Mat src({shape.rows, shape.cols}, CV_8UC1);
            cvh::Mat dst;
            fill_u8(src, seed);
            const double cvh_ms = measure_cvh_mat_ms(
                [&]() { cvh::threshold(src, dst, 127.0, 255.0, cvh::THRESH_BINARY); },
                dst,
                args);
            const double opencv_ms = bench_opencv_threshold_binary(
                shape.rows, shape.cols, args.warmup, args.iters, args.repeats, seed);
            append_row(rows, args, "imgproc", "THRESHOLD", "binary_u8", "headers_baseline", "CV_8U", 1, shape_name, cvh_ms, opencv_ms);
        }

        {
            cvh::Mat src({shape.rows, shape.cols}, CV_8UC1);
            cvh::Mat dst;
            fill_u8(src, seed);
            const double cvh_ms = measure_cvh_mat_ms(
                [&]() { cvh::GaussianBlur(src, dst, cvh::Size(5, 5), 0.0, 0.0, cvh::BORDER_REPLICATE); },
                dst,
                args);
            const double opencv_ms = bench_opencv_gaussian(shape.rows, shape.cols, DepthId::U8, 1, 5, args.warmup, args.iters, args.repeats, seed);
            append_row(rows, args, "imgproc", "GAUSSIAN", "5x5_replicate", "headers_baseline", "CV_8U", 1, shape_name, cvh_ms, opencv_ms);
        }

        {
            cvh::Mat src({shape.rows, shape.cols}, CV_8UC1);
            cvh::Mat dst;
            fill_u8(src, seed);
            const double cvh_ms = measure_cvh_mat_ms(
                [&]() { cvh::boxFilter(src, dst, -1, cvh::Size(3, 3), cvh::Point(-1, -1), true, cvh::BORDER_REPLICATE); },
                dst,
                args);
            const double opencv_ms = bench_opencv_box(shape.rows, shape.cols, DepthId::U8, 1, 3, args.warmup, args.iters, args.repeats, seed);
            append_row(rows, args, "imgproc", "BOX_FILTER", "3x3_replicate", "headers_baseline", "CV_8U", 1, shape_name, cvh_ms, opencv_ms);
        }

        {
            cvh::Mat src({shape.rows, shape.cols}, CV_8UC1);
            cvh::Mat lut({256, 1}, CV_8UC1);
            cvh::Mat dst;
            fill_u8(src, seed);
            for (int i = 0; i < 256; ++i)
            {
                lut.data[i] = static_cast<uchar>(255 - i);
            }
            const double cvh_ms = measure_cvh_mat_ms([&]() { cvh::LUT(src, lut, dst); }, dst, args);
            const double opencv_ms = bench_opencv_lut(shape.rows, shape.cols, 1, args.warmup, args.iters, args.repeats, seed);
            append_row(rows, args, "imgproc", "LUT", "invert_u8", "headers_baseline", "CV_8U", 1, shape_name, cvh_ms, opencv_ms);
        }

        {
            cvh::Mat src({shape.rows, shape.cols}, CV_8UC1);
            cvh::Mat dst;
            fill_u8(src, seed);
            const double cvh_ms = measure_cvh_mat_ms(
                [&]() { cvh::copyMakeBorder(src, dst, 2, 2, 2, 2, cvh::BORDER_REPLICATE, cvh::Scalar::all(0.0)); },
                dst,
                args);
            const double opencv_ms = bench_opencv_copy_make_border(shape.rows, shape.cols, DepthId::U8, 1, 2, 2, 2, 2, args.warmup, args.iters, args.repeats, seed);
            append_row(rows, args, "imgproc", "COPY_MAKE_BORDER", "2px_replicate", "headers_baseline", "CV_8U", 1, shape_name, cvh_ms, opencv_ms);
        }

        {
            cvh::Mat src({shape.rows, shape.cols}, CV_8UC1);
            cvh::Mat dst;
            cvh::Mat kernel({3, 3}, CV_32FC1);
            float* values = reinterpret_cast<float*>(kernel.data);
            const float kernel_values[9] = {0.0f, 0.25f, 0.0f, 0.25f, 0.0f, 0.25f, 0.0f, 0.25f, 0.0f};
            std::copy(kernel_values, kernel_values + 9, values);
            fill_u8(src, seed);
            const double cvh_ms = measure_cvh_mat_ms(
                [&]() { cvh::filter2D(src, dst, -1, kernel, cvh::Point(-1, -1), 0.0, cvh::BORDER_REPLICATE); },
                dst,
                args);
            const double opencv_ms = bench_opencv_filter2d(
                shape.rows, shape.cols, DepthId::U8, 1, args.warmup, args.iters, args.repeats, seed);
            append_row(rows, args, "imgproc", "FILTER2D", "3x3_replicate", "headers_baseline", "CV_8U", 1, shape_name, cvh_ms, opencv_ms);
        }

        {
            cvh::Mat src({shape.rows, shape.cols}, CV_8UC1);
            cvh::Mat dst;
            cvh::Mat kernel_x({1, 3}, CV_32FC1);
            cvh::Mat kernel_y({3, 1}, CV_32FC1);
            float* kx = reinterpret_cast<float*>(kernel_x.data);
            float* ky = reinterpret_cast<float*>(kernel_y.data);
            kx[0] = 0.25f;
            kx[1] = 0.5f;
            kx[2] = 0.25f;
            ky[0] = 0.25f;
            ky[1] = 0.5f;
            ky[2] = 0.25f;
            fill_u8(src, seed);
            const double cvh_ms = measure_cvh_mat_ms(
                [&]() { cvh::sepFilter2D(src, dst, -1, kernel_x, kernel_y, cvh::Point(-1, -1), 0.0, cvh::BORDER_REPLICATE); },
                dst,
                args);
            const double opencv_ms = bench_opencv_sep_filter2d(
                shape.rows, shape.cols, DepthId::U8, 1, args.warmup, args.iters, args.repeats, seed);
            append_row(rows, args, "imgproc", "SEP_FILTER2D", "3x3_replicate", "headers_baseline", "CV_8U", 1, shape_name, cvh_ms, opencv_ms);
        }

        {
            cvh::Mat src({shape.rows, shape.cols}, CV_8UC1);
            cvh::Mat dst;
            fill_u8(src, seed);
            const double cvh_ms = measure_cvh_mat_ms(
                [&]() { cvh::Sobel(src, dst, CV_32F, 1, 0, 3, 1.0, 0.0, cvh::BORDER_REPLICATE); },
                dst,
                args);
            const double opencv_ms = bench_opencv_sobel(shape.rows, shape.cols, 1, args.warmup, args.iters, args.repeats, seed);
            append_row(rows, args, "imgproc", "SOBEL", "dx1_ksize3_replicate", "headers_baseline", "CV_8U", 1, shape_name, cvh_ms, opencv_ms);
        }

        {
            cvh::Mat src({shape.rows, shape.cols}, CV_8UC1);
            cvh::Mat dst;
            fill_u8(src, seed);
            const double cvh_ms = measure_cvh_mat_ms([&]() { cvh::Canny(src, dst, 50.0, 130.0, 3, false); }, dst, args);
            const double opencv_ms = bench_opencv_canny(shape.rows, shape.cols, args.warmup, args.iters, args.repeats, seed, 50.0, 130.0, 3, false);
            append_row(rows, args, "imgproc", "CANNY", "aperture3_l1", "headers_baseline", "CV_8U", 1, shape_name, cvh_ms, opencv_ms);
        }

        {
            cvh::Mat src({shape.rows, shape.cols}, CV_8UC1);
            cvh::Mat dst;
            fill_u8(src, seed);
            const double cvh_ms = measure_cvh_mat_ms(
                [&]() { cvh::erode(src, dst, cvh::Mat(), cvh::Point(-1, -1), 1, cvh::BORDER_REPLICATE); },
                dst,
                args);
            const double opencv_ms = bench_opencv_erode(shape.rows, shape.cols, 1, args.warmup, args.iters, args.repeats, seed);
            append_row(rows, args, "imgproc", "ERODE", "3x3_replicate", "headers_baseline", "CV_8U", 1, shape_name, cvh_ms, opencv_ms);
        }

        {
            cvh::Mat src({shape.rows, shape.cols}, CV_8UC1);
            cvh::Mat dst;
            fill_u8(src, seed);
            const double cvh_ms = measure_cvh_mat_ms(
                [&]() { cvh::dilate(src, dst, cvh::Mat(), cvh::Point(-1, -1), 1, cvh::BORDER_REPLICATE); },
                dst,
                args);
            const double opencv_ms = bench_opencv_dilate(shape.rows, shape.cols, 1, args.warmup, args.iters, args.repeats, seed);
            append_row(rows, args, "imgproc", "DILATE", "3x3_replicate", "headers_baseline", "CV_8U", 1, shape_name, cvh_ms, opencv_ms);
        }

        {
            cvh::Mat src({shape.rows, shape.cols}, CV_8UC1);
            cvh::Mat dst;
            cvh::Mat transform({2, 3}, CV_32FC1);
            float* m = reinterpret_cast<float*>(transform.data);
            m[0] = 1.0f;
            m[1] = 0.0f;
            m[2] = -1.25f;
            m[3] = 0.0f;
            m[4] = 1.0f;
            m[5] = 0.75f;
            fill_u8(src, seed);
            const double cvh_ms = measure_cvh_mat_ms(
                [&]() {
                    cvh::warpAffine(
                        src,
                        dst,
                        transform,
                        cvh::Size(shape.cols, shape.rows),
                        cvh::INTER_LINEAR | cvh::WARP_INVERSE_MAP,
                        cvh::BORDER_REPLICATE,
                        cvh::Scalar::all(0.0));
                },
                dst,
                args);
            const double opencv_ms = bench_opencv_warp_affine(
                shape.rows, shape.cols, DepthId::U8, 1, args.warmup, args.iters, args.repeats, seed);
            append_row(rows, args, "imgproc", "WARP_AFFINE", "linear_inverse_replicate", "headers_baseline", "CV_8U", 1, shape_name, cvh_ms, opencv_ms);
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

    out << "impl,profile,suite,op,variant,dispatch_path,depth,channels,shape,cvh_ms,opencv_ms,speedup,status,note\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& row : rows)
    {
        out << row.impl << ","
            << row.profile << ","
            << row.suite << ","
            << row.op << ","
            << row.variant << ","
            << row.dispatch_path << ","
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
    const auto args = cvh_bench_compare::parse_args(argc, argv);
    cvh_bench_compare::configure_opencv_threads(args.threads);
    std::vector<cvh_bench_compare::CompareRow> rows;
    cvh_bench_compare::append_core_mat_cases(args, rows);
    cvh_bench_compare::append_imgproc_cases(args, rows);

    if (!args.output_csv.empty())
    {
        cvh_bench_compare::write_csv(rows, args.output_csv);
    }

    std::cout << "cvh_benchmark_opencv_compare_headers_fast: impl=" << args.impl
              << ", profile=" << args.profile
              << ", threads=" << args.threads
              << ", rows=" << rows.size()
              << ", sink=" << cvh_bench_compare::g_sink << "\n";
    return 0;
}
