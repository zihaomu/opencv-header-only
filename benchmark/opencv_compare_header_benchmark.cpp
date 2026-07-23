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
    std::string layout = "continuous";
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

void fill_f32(cvh::Mat& mat, std::uint32_t seed)
{
    const int rows = mat.size[0];
    const int scalars_per_row = mat.size[1] * mat.channels();
    std::uint32_t state = seed;
    for (int y = 0; y < rows; ++y)
    {
        float* row = reinterpret_cast<float*>(mat.data + static_cast<std::size_t>(y) * mat.step(0));
        for (int x = 0; x < scalars_per_row; ++x)
        {
            state = lcg_next(state);
            row[x] = static_cast<float>(static_cast<int>(state & 0xFFFFu) - 32768) / 4096.0f;
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

void append_unsupported_row(std::vector<CompareRow>& rows,
                            const Args& args,
                            const std::string& suite,
                            const std::string& op,
                            const std::string& variant,
                            const std::string& depth,
                            int channels,
                            const std::string& layout,
                            const std::string& shape,
                            const std::string& note)
{
    CompareRow row;
    row.impl = args.impl;
    row.profile = args.profile;
    row.suite = suite;
    row.op = op;
    row.variant = variant;
    row.dispatch_path = "unsupported";
    row.depth = depth;
    row.channels = channels;
    row.layout = layout;
    row.shape = shape;
    row.status = "UNSUPPORTED";
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

void run_cvh_binary(CoreBinaryOpId op, const cvh::Mat& a, const cvh::Mat& b, cvh::Mat& dst)
{
    switch (op)
    {
        case CoreBinaryOpId::Add:
            cvh::add(a, b, dst);
            return;
        case CoreBinaryOpId::Subtract:
            cvh::subtract(a, b, dst);
            return;
        case CoreBinaryOpId::Multiply:
            cvh::multiply(a, b, dst);
            return;
        case CoreBinaryOpId::Divide:
            cvh::divide(a, b, dst);
            return;
    }
}

void require_correct(bool correct, const std::string& case_name)
{
    if (!correct)
    {
        std::cerr << "OpenCV correctness gate failed: " << case_name << "\n";
        std::exit(3);
    }
}

void append_core_compute_cases(const Args& args, std::vector<CompareRow>& rows)
{
    constexpr std::uint32_t seed_a = 0xB101u;
    constexpr std::uint32_t seed_b = 0xB202u;
    const std::vector<std::pair<CoreBinaryOpId, std::string>> binary_ops = {
        {CoreBinaryOpId::Add, "ADD"},
        {CoreBinaryOpId::Subtract, "SUBTRACT"},
        {CoreBinaryOpId::Multiply, "MULTIPLY"},
        {CoreBinaryOpId::Divide, "DIVIDE"},
    };
    const std::vector<std::pair<DepthId, std::string>> depths = {
        {DepthId::U8, "CV_8U"},
        {DepthId::F32, "CV_32F"},
    };

    for (const auto& shape : build_shapes(args.profile))
    {
        const std::string shape_name = shape_string(shape);
        for (const auto& depth_case : depths)
        {
            for (const int channels : {1, 3})
            {
                const int type = CV_MAKETYPE(
                    depth_case.first == DepthId::U8 ? CV_8U : CV_32F,
                    channels);
                cvh::Mat a({shape.rows, shape.cols}, type);
                cvh::Mat b({shape.rows, shape.cols}, type);
                fill_by_depth(a, depth_case.first, seed_a);
                fill_by_depth(b, depth_case.first, seed_b);

                for (const auto& op_case : binary_ops)
                {
                    cvh::Mat dst;
                    run_cvh_binary(op_case.first, a, b, dst);
                    const std::uint64_t output_bytes =
                        static_cast<std::uint64_t>(dst.total()) * dst.elemSize();
                    require_correct(
                        validate_opencv_binary(
                            op_case.first,
                            shape.rows,
                            shape.cols,
                            depth_case.first,
                            channels,
                            seed_a,
                            seed_b,
                            dst.data,
                            output_bytes),
                        op_case.second + "/" + depth_case.second + "C" + std::to_string(channels) + "/" + shape_name);

                    const double cvh_ms = measure_cvh_mat_ms(
                        [&]() { run_cvh_binary(op_case.first, a, b, dst); },
                        dst,
                        args);
                    const double opencv_ms = bench_opencv_binary(
                        op_case.first,
                        shape.rows,
                        shape.cols,
                        depth_case.first,
                        channels,
                        args.warmup,
                        args.iters,
                        args.repeats,
                        seed_a,
                        seed_b);
                    append_row(
                        rows,
                        args,
                        "core_mat",
                        op_case.second,
                        "mat_mat_continuous",
                        "headers_baseline",
                        depth_case.second,
                        channels,
                        shape_name,
                        cvh_ms,
                        opencv_ms,
                        op_case.first == CoreBinaryOpId::Divide && depth_case.first == DepthId::U8
                            ? "correctness=upstream_pass;u8_divide_abs_tolerance=1"
                            : "correctness=upstream_pass");
                }

                cvh::Mat transposed = cvh::transpose(a);
                const std::uint64_t transpose_bytes =
                    static_cast<std::uint64_t>(transposed.total()) * transposed.elemSize();
                require_correct(
                    validate_opencv_transpose(
                        shape.rows,
                        shape.cols,
                        depth_case.first,
                        channels,
                        seed_a,
                        transposed.data,
                        transpose_bytes),
                    "TRANSPOSE/" + depth_case.second + "C" + std::to_string(channels) + "/" + shape_name);
                const double cvh_ms = measure_cvh_mat_ms(
                    [&]() { transposed = cvh::transpose(a); },
                    transposed,
                    args);
                const double opencv_ms = bench_opencv_transpose(
                    shape.rows,
                    shape.cols,
                    depth_case.first,
                    channels,
                    args.warmup,
                    args.iters,
                    args.repeats,
                    seed_a);
                append_row(
                    rows,
                    args,
                    "core_mat",
                    "TRANSPOSE",
                    "continuous",
                    "headers_baseline",
                    depth_case.second,
                    channels,
                    shape_name,
                    cvh_ms,
                    opencv_ms,
                    "correctness=upstream_pass");
            }
        }
    }

    std::vector<int> gemm_sizes {128};
    if (args.profile == "stable" || args.profile == "full")
    {
        gemm_sizes = {128, 256, 512};
    }
    for (const int size : gemm_sizes)
    {
        cvh::Mat a({size, size}, CV_32F);
        cvh::Mat b({size, size}, CV_32F);
        fill_f32(a, seed_a);
        fill_f32(b, seed_b);

        Args gemm_args = args;
        const std::uint64_t work =
            static_cast<std::uint64_t>(size) * static_cast<std::uint64_t>(size) * static_cast<std::uint64_t>(size);
        gemm_args.iters = std::min(args.iters, std::max(1, static_cast<int>((16u << 20) / work)));
        gemm_args.warmup = std::min(args.warmup, 1);

        cvh::Mat dst = cvh::gemm(a, b);
        const std::uint64_t output_bytes =
            static_cast<std::uint64_t>(dst.total()) * dst.elemSize();
        require_correct(
            validate_opencv_gemm(size, size, size, seed_a, seed_b, dst.data, output_bytes),
            "GEMM/NN/" + std::to_string(size));
        const double cvh_ms = measure_cvh_mat_ms([&]() { dst = cvh::gemm(a, b); }, dst, gemm_args);
        const double opencv_ms = bench_opencv_gemm(
            size,
            size,
            size,
            gemm_args.warmup,
            gemm_args.iters,
            gemm_args.repeats,
            seed_a,
            seed_b);
        append_row(
            rows,
            args,
            "core_mat",
            "GEMM",
            "fp32_nn_end_to_end",
            "headers_baseline",
            "CV_32F",
            1,
            std::to_string(size) + "x" + std::to_string(size) + "x" + std::to_string(size),
            cvh_ms,
            opencv_ms,
            "correctness=upstream_pass;iters=" + std::to_string(gemm_args.iters));

        const cvh::GemmPackedB packed_b = cvh::gemm_pack_b(b);
        dst = cvh::gemm(a, packed_b);
        require_correct(
            validate_opencv_gemm(size, size, size, seed_a, seed_b, dst.data, output_bytes),
            "GEMM/NN/pack_once/" + std::to_string(size));
        const double packed_cvh_ms =
            measure_cvh_mat_ms([&]() { dst = cvh::gemm(a, packed_b); }, dst, gemm_args);
        const double packed_opencv_ms = bench_opencv_gemm_prepack(
            size,
            size,
            size,
            gemm_args.warmup,
            gemm_args.iters,
            gemm_args.repeats,
            seed_a,
            seed_b);
        append_row(
            rows,
            args,
            "core_mat",
            "GEMM",
            "fp32_nn_pack_once",
            "headers_baseline",
            "CV_32F",
            1,
            std::to_string(size) + "x" + std::to_string(size) + "x" + std::to_string(size),
            packed_cvh_ms,
            packed_opencv_ms,
            "correctness=upstream_pass;opencv_reuses_B_without_public_pack_handle;iters=" +
                std::to_string(gemm_args.iters));
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
            append_row(rows, args, "imgproc", "GAUSSIAN", "5x5_replicate", "header_fastpath", "CV_8U", 1, shape_name, cvh_ms, opencv_ms);
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
            append_row(rows, args, "imgproc", "BOX_FILTER", "3x3_replicate", "header_fastpath", "CV_8U", 1, shape_name, cvh_ms, opencv_ms);
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
            append_row(rows, args, "imgproc", "LUT", "invert_u8", "header_fastpath", "CV_8U", 1, shape_name, cvh_ms, opencv_ms);
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
            append_row(rows, args, "imgproc", "COPY_MAKE_BORDER", "2px_replicate", "header_fastpath", "CV_8U", 1, shape_name, cvh_ms, opencv_ms);
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
            append_row(rows, args, "imgproc", "FILTER2D", "3x3_replicate", "header_fastpath", "CV_8U", 1, shape_name, cvh_ms, opencv_ms);
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
            append_row(rows, args, "imgproc", "SEP_FILTER2D", "3x3_replicate", "header_fastpath", "CV_8U", 1, shape_name, cvh_ms, opencv_ms);
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
            append_row(rows, args, "imgproc", "SOBEL", "dx1_ksize3_replicate", "header_fastpath", "CV_8U", 1, shape_name, cvh_ms, opencv_ms);
        }

        {
            cvh::Mat src({shape.rows, shape.cols}, CV_8UC1);
            cvh::Mat dst;
            fill_u8(src, seed);
            const double cvh_ms = measure_cvh_mat_ms([&]() { cvh::Canny(src, dst, 50.0, 130.0, 3, false); }, dst, args);
            const double opencv_ms = bench_opencv_canny(shape.rows, shape.cols, args.warmup, args.iters, args.repeats, seed, 50.0, 130.0, 3, false);
            append_row(rows, args, "imgproc", "CANNY", "aperture3_l1", "header_fastpath", "CV_8U", 1, shape_name, cvh_ms, opencv_ms);
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
            append_row(rows, args, "imgproc", "ERODE", "3x3_replicate", "header_fastpath", "CV_8U", 1, shape_name, cvh_ms, opencv_ms);
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
            append_row(rows, args, "imgproc", "DILATE", "3x3_replicate", "header_fastpath", "CV_8U", 1, shape_name, cvh_ms, opencv_ms);
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

void append_imgproc_type_channel_cases(const Args& args, std::vector<CompareRow>& rows)
{
    if (args.profile == "quick")
    {
        return;
    }

    constexpr std::uint32_t seed = 0xD101u;
    const ShapeCase shape {480, 640};
    const std::string shape_name = shape_string(shape);
    struct TypeCase
    {
        DepthId depth;
        const char* depth_name;
        int channels;
    };
    const TypeCase type_cases[] = {
        {DepthId::U8, "CV_8U", 3},
        {DepthId::U8, "CV_8U", 4},
        {DepthId::F32, "CV_32F", 1},
        {DepthId::F32, "CV_32F", 3},
        {DepthId::F32, "CV_32F", 4},
    };

    cvh::Mat kernel({3, 3}, CV_32FC1);
    float* kernel_values = reinterpret_cast<float*>(kernel.data);
    const float filter_values[9] = {0.0f, 0.25f, 0.0f, 0.25f, 0.0f, 0.25f, 0.0f, 0.25f, 0.0f};
    std::copy(filter_values, filter_values + 9, kernel_values);
    cvh::Mat kernel_x({1, 3}, CV_32FC1);
    cvh::Mat kernel_y({3, 1}, CV_32FC1);
    float* kx = reinterpret_cast<float*>(kernel_x.data);
    float* ky = reinterpret_cast<float*>(kernel_y.data);
    kx[0] = ky[0] = 0.25f;
    kx[1] = ky[1] = 0.5f;
    kx[2] = ky[2] = 0.25f;

    for (const auto& type_case : type_cases)
    {
        const int type = CV_MAKETYPE(
            type_case.depth == DepthId::U8 ? CV_8U : CV_32F,
            type_case.channels);
        const std::string suffix =
            std::string(type_case.depth == DepthId::U8 ? "u8c" : "f32c") +
            std::to_string(type_case.channels);
        cvh::Mat src({shape.rows, shape.cols}, type);
        fill_by_depth(src, type_case.depth, seed + static_cast<std::uint32_t>(type_case.channels));

        {
            cvh::Mat dst;
            const double cvh_ms = measure_cvh_mat_ms(
                [&]() { cvh::GaussianBlur(src, dst, cvh::Size(5, 5), 0.0, 0.0, cvh::BORDER_REPLICATE); },
                dst,
                args);
            const double opencv_ms = bench_opencv_gaussian(
                shape.rows, shape.cols, type_case.depth, type_case.channels, 5,
                args.warmup, args.iters, args.repeats, seed);
            append_row(rows, args, "imgproc", "GAUSSIAN", "5x5_replicate_" + suffix,
                       cvh::detail::last_gaussianblur_dispatch_path(), type_case.depth_name,
                       type_case.channels, shape_name, cvh_ms, opencv_ms);
        }

        {
            cvh::Mat dst;
            const double cvh_ms = measure_cvh_mat_ms(
                [&]() { cvh::boxFilter(src, dst, -1, cvh::Size(3, 3), cvh::Point(-1, -1), true, cvh::BORDER_REPLICATE); },
                dst,
                args);
            const double opencv_ms = bench_opencv_box(
                shape.rows, shape.cols, type_case.depth, type_case.channels, 3,
                args.warmup, args.iters, args.repeats, seed);
            append_row(rows, args, "imgproc", "BOX_FILTER", "3x3_replicate_" + suffix,
                       cvh::detail::last_boxfilter_dispatch_path(), type_case.depth_name,
                       type_case.channels, shape_name, cvh_ms, opencv_ms);
        }

        {
            cvh::Mat dst;
            const double cvh_ms = measure_cvh_mat_ms(
                [&]() { cvh::copyMakeBorder(src, dst, 2, 2, 2, 2, cvh::BORDER_REPLICATE, cvh::Scalar::all(0.0)); },
                dst,
                args);
            const double opencv_ms = bench_opencv_copy_make_border(
                shape.rows, shape.cols, type_case.depth, type_case.channels, 2, 2, 2, 2,
                args.warmup, args.iters, args.repeats, seed);
            append_row(rows, args, "imgproc", "COPY_MAKE_BORDER", "2px_replicate_" + suffix,
                       "header_fastpath", type_case.depth_name, type_case.channels,
                       shape_name, cvh_ms, opencv_ms);
        }

        {
            cvh::Mat dst;
            const double cvh_ms = measure_cvh_mat_ms(
                [&]() { cvh::filter2D(src, dst, -1, kernel, cvh::Point(-1, -1), 0.0, cvh::BORDER_REPLICATE); },
                dst,
                args);
            const double opencv_ms = bench_opencv_filter2d(
                shape.rows, shape.cols, type_case.depth, type_case.channels,
                args.warmup, args.iters, args.repeats, seed);
            append_row(rows, args, "imgproc", "FILTER2D", "3x3_replicate_" + suffix,
                       "header_fastpath", type_case.depth_name, type_case.channels,
                       shape_name, cvh_ms, opencv_ms);
        }

        {
            cvh::Mat dst;
            const double cvh_ms = measure_cvh_mat_ms(
                [&]() { cvh::sepFilter2D(src, dst, -1, kernel_x, kernel_y, cvh::Point(-1, -1), 0.0, cvh::BORDER_REPLICATE); },
                dst,
                args);
            const double opencv_ms = bench_opencv_sep_filter2d(
                shape.rows, shape.cols, type_case.depth, type_case.channels,
                args.warmup, args.iters, args.repeats, seed);
            append_row(rows, args, "imgproc", "SEP_FILTER2D", "3x3_replicate_" + suffix,
                       "header_fastpath", type_case.depth_name, type_case.channels,
                       shape_name, cvh_ms, opencv_ms);
        }

        {
            cvh::Mat dst;
            cvh::Mat transform({2, 3}, CV_32FC1);
            float* m = reinterpret_cast<float*>(transform.data);
            m[0] = 1.0f;
            m[1] = 0.0f;
            m[2] = -1.25f;
            m[3] = 0.0f;
            m[4] = 1.0f;
            m[5] = 0.75f;
            const double cvh_ms = measure_cvh_mat_ms(
                [&]() {
                    cvh::warpAffine(
                        src, dst, transform, cvh::Size(shape.cols, shape.rows),
                        cvh::INTER_LINEAR | cvh::WARP_INVERSE_MAP,
                        cvh::BORDER_REPLICATE, cvh::Scalar::all(0.0));
                },
                dst,
                args);
            const double opencv_ms = bench_opencv_warp_affine(
                shape.rows, shape.cols, type_case.depth, type_case.channels,
                args.warmup, args.iters, args.repeats, seed);
            append_row(rows, args, "imgproc", "WARP_AFFINE", "linear_inverse_replicate_" + suffix,
                       "headers_baseline", type_case.depth_name, type_case.channels,
                       shape_name, cvh_ms, opencv_ms);
        }
    }

    for (const int channels : {3, 4})
    {
        const std::string suffix = "u8c" + std::to_string(channels);
        cvh::Mat src({shape.rows, shape.cols}, CV_MAKETYPE(CV_8U, channels));
        fill_u8(src, seed + static_cast<std::uint32_t>(channels));

        {
            cvh::Mat lut({256, 1}, CV_8UC1);
            for (int i = 0; i < 256; ++i)
            {
                lut.data[i] = static_cast<uchar>(255 - i);
            }
            cvh::Mat dst;
            const double cvh_ms = measure_cvh_mat_ms([&]() { cvh::LUT(src, lut, dst); }, dst, args);
            const double opencv_ms = bench_opencv_lut(
                shape.rows, shape.cols, channels, args.warmup, args.iters, args.repeats, seed);
            append_row(rows, args, "imgproc", "LUT", "invert_" + suffix,
                       "header_fastpath", "CV_8U", channels, shape_name, cvh_ms, opencv_ms);
        }

        {
            cvh::Mat dst;
            const double cvh_ms = measure_cvh_mat_ms(
                [&]() { cvh::Sobel(src, dst, CV_32F, 1, 0, 3, 1.0, 0.0, cvh::BORDER_REPLICATE); },
                dst,
                args);
            const double opencv_ms = bench_opencv_sobel(
                shape.rows, shape.cols, channels, args.warmup, args.iters, args.repeats, seed);
            append_row(rows, args, "imgproc", "SOBEL", "dx1_ksize3_replicate_" + suffix,
                       "header_fastpath", "CV_8U", channels, shape_name, cvh_ms, opencv_ms);
        }

        {
            cvh::Mat dst;
            const double cvh_ms = measure_cvh_mat_ms(
                [&]() { cvh::erode(src, dst, cvh::Mat(), cvh::Point(-1, -1), 1, cvh::BORDER_REPLICATE); },
                dst,
                args);
            const double opencv_ms = bench_opencv_erode(
                shape.rows, shape.cols, channels, args.warmup, args.iters, args.repeats, seed);
            append_row(rows, args, "imgproc", "ERODE", "3x3_replicate_" + suffix,
                       "header_fastpath", "CV_8U", channels, shape_name, cvh_ms, opencv_ms);
        }

        {
            cvh::Mat dst;
            const double cvh_ms = measure_cvh_mat_ms(
                [&]() { cvh::dilate(src, dst, cvh::Mat(), cvh::Point(-1, -1), 1, cvh::BORDER_REPLICATE); },
                dst,
                args);
            const double opencv_ms = bench_opencv_dilate(
                shape.rows, shape.cols, channels, args.warmup, args.iters, args.repeats, seed);
            append_row(rows, args, "imgproc", "DILATE", "3x3_replicate_" + suffix,
                       "header_fastpath", "CV_8U", channels, shape_name, cvh_ms, opencv_ms);
        }
    }
}

void append_imgproc_resize_color_cases(const Args& args, std::vector<CompareRow>& rows)
{
    if (args.profile == "quick")
    {
        return;
    }

    constexpr std::uint32_t seed = 0xD707u;
    const ShapeCase shape {480, 640};
    const std::string shape_name = shape_string(shape);
    struct ResizeCase
    {
        const char* variant;
        DepthId depth;
        const char* depth_name;
        int channels;
        int interpolation;
        const char* dispatch_path;
    };
    const ResizeCase resize_cases[] = {
        {"nearest_0.75_u8c3", DepthId::U8, "CV_8U", 3, cvh::INTER_NEAREST, "header_fastpath"},
        {"nearest_exact_0.75_u8c3", DepthId::U8, "CV_8U", 3, cvh::INTER_NEAREST_EXACT, "header_fastpath"},
        {"linear_0.75_u8c3", DepthId::U8, "CV_8U", 3, cvh::INTER_LINEAR, "header_fastpath"},
        {"linear_0.75_f32c1", DepthId::F32, "CV_32F", 1, cvh::INTER_LINEAR, "headers_baseline"},
        {"linear_0.75_f32c3", DepthId::F32, "CV_32F", 3, cvh::INTER_LINEAR, "headers_baseline"},
    };
    for (const auto& resize_case : resize_cases)
    {
        const int type = CV_MAKETYPE(
            resize_case.depth == DepthId::U8 ? CV_8U : CV_32F,
            resize_case.channels);
        cvh::Mat src({shape.rows, shape.cols}, type);
        fill_by_depth(src, resize_case.depth, seed);
        cvh::Mat dst;
        const int dst_rows = shape.rows * 3 / 4;
        const int dst_cols = shape.cols * 3 / 4;
        const double cvh_ms = measure_cvh_mat_ms(
            [&]() {
                cvh::resize(
                    src, dst, cvh::Size(dst_cols, dst_rows), 0.0, 0.0,
                    resize_case.interpolation);
            },
            dst,
            args);
        const double opencv_ms = bench_opencv_resize(
            shape.rows, shape.cols, dst_rows, dst_cols,
            resize_case.depth, resize_case.channels, resize_case.interpolation,
            args.warmup, args.iters, args.repeats, seed);
        append_row(
            rows, args, "imgproc", "RESIZE", resize_case.variant,
            resize_case.dispatch_path, resize_case.depth_name, resize_case.channels,
            shape_name, cvh_ms, opencv_ms);
    }

    struct ColorCase
    {
        const char* variant;
        const char* layout;
        ImgprocColorOpId op;
        int cvh_code;
        DepthId depth;
        const char* depth_name;
        int src_channels;
        int src_rows_numerator;
        int src_rows_denominator;
    };
    const ColorCase stable_cases[] = {
        {"BGR2RGB_u8", "continuous", ImgprocColorOpId::Bgr2Rgb, cvh::COLOR_BGR2RGB, DepthId::U8, "CV_8U", 3, 1, 1},
        {"BGR2BGRA_u8", "continuous", ImgprocColorOpId::Bgr2Bgra, cvh::COLOR_BGR2BGRA, DepthId::U8, "CV_8U", 3, 1, 1},
        {"BGRA2GRAY_u8", "continuous", ImgprocColorOpId::Bgra2Gray, cvh::COLOR_BGRA2GRAY, DepthId::U8, "CV_8U", 4, 1, 1},
        {"BGR2GRAY_f32", "continuous", ImgprocColorOpId::Bgr2Gray, cvh::COLOR_BGR2GRAY, DepthId::F32, "CV_32F", 3, 1, 1},
        {"BGR2RGB_f32", "continuous", ImgprocColorOpId::Bgr2Rgb, cvh::COLOR_BGR2RGB, DepthId::F32, "CV_32F", 3, 1, 1},
        {"BGR2YUV_u8", "yuv444_interleaved", ImgprocColorOpId::Bgr2Yuv, cvh::COLOR_BGR2YUV, DepthId::U8, "CV_8U", 3, 1, 1},
        {"YUV2BGR_u8", "yuv444_interleaved", ImgprocColorOpId::Yuv2Bgr, cvh::COLOR_YUV2BGR, DepthId::U8, "CV_8U", 3, 1, 1},
    };
    for (const auto& color_case : stable_cases)
    {
        const int src_rows =
            shape.rows * color_case.src_rows_numerator / color_case.src_rows_denominator;
        const int type = CV_MAKETYPE(
            color_case.depth == DepthId::U8 ? CV_8U : CV_32F,
            color_case.src_channels);
        cvh::Mat src({src_rows, shape.cols}, type);
        fill_by_depth(src, color_case.depth, seed);
        cvh::Mat dst;
        const double cvh_ms = measure_cvh_mat_ms(
            [&]() { cvh::cvtColor(src, dst, color_case.cvh_code); },
            dst,
            args);
        const double opencv_ms = bench_opencv_cvtcolor(
            color_case.op, shape.rows, shape.cols, color_case.depth,
            args.warmup, args.iters, args.repeats, seed);
        append_row(
            rows, args, "imgproc", "CVTCOLOR", color_case.variant,
            "header_fastpath", color_case.depth_name, color_case.src_channels,
            shape_name, cvh_ms, opencv_ms);
        rows.back().layout = color_case.layout;
    }

    if (args.profile != "full")
    {
        return;
    }

    const ColorCase full_cases[] = {
        {"BGR2I420_u8", "yuv420_i420", ImgprocColorOpId::Bgr2YuvI420, cvh::COLOR_BGR2YUV_I420, DepthId::U8, "CV_8U", 3, 1, 1},
        {"I420_TO_BGR_u8", "yuv420_i420", ImgprocColorOpId::YuvI420ToBgr, cvh::COLOR_YUV2BGR_I420, DepthId::U8, "CV_8U", 1, 3, 2},
        {"BGR2YUY2_u8", "yuv422_yuy2", ImgprocColorOpId::Bgr2YuvYuy2, cvh::COLOR_BGR2YUV_YUY2, DepthId::U8, "CV_8U", 3, 1, 1},
        {"YUY2_TO_BGR_u8", "yuv422_yuy2", ImgprocColorOpId::YuvYuy2ToBgr, cvh::COLOR_YUV2BGR_YUY2, DepthId::U8, "CV_8U", 2, 1, 1},
        {"NV12_TO_BGR_u8", "yuv420_nv12", ImgprocColorOpId::YuvNv12ToBgr, cvh::COLOR_YUV2BGR_NV12, DepthId::U8, "CV_8U", 1, 3, 2},
    };
    for (const auto& color_case : full_cases)
    {
        const int src_rows =
            shape.rows * color_case.src_rows_numerator / color_case.src_rows_denominator;
        cvh::Mat src({src_rows, shape.cols}, CV_MAKETYPE(CV_8U, color_case.src_channels));
        fill_u8(src, seed);
        cvh::Mat dst;
        const double cvh_ms = measure_cvh_mat_ms(
            [&]() { cvh::cvtColor(src, dst, color_case.cvh_code); },
            dst,
            args);
        const double opencv_ms = bench_opencv_cvtcolor(
            color_case.op, shape.rows, shape.cols, DepthId::U8,
            args.warmup, args.iters, args.repeats, seed);
        append_row(
            rows, args, "imgproc", "CVTCOLOR", color_case.variant,
            "header_fastpath", "CV_8U", color_case.src_channels,
            shape_name, cvh_ms, opencv_ms);
        rows.back().layout = color_case.layout;
    }

    append_unsupported_row(
        rows, args, "imgproc", "CVTCOLOR", "BGR2NV12_u8",
        "CV_8U", 3, "yuv420_nv12", shape_name,
        "upstream OpenCV has NV12 decode but no single-call BGR-to-NV12 encoder");
}

void append_imgproc_roi_cases(const Args& args, std::vector<CompareRow>& rows)
{
    if (args.profile != "full")
    {
        return;
    }

    constexpr std::uint32_t seed = 0xE201u;
    const ShapeCase shape {479, 641};
    const std::string shape_name = shape_string(shape);
    cvh::Mat u8_owner({shape.rows + 2, shape.cols + 3}, CV_8UC3);
    cvh::Mat u8_src = u8_owner(cvh::Range(1, shape.rows + 1), cvh::Range(1, shape.cols + 1));
    fill_u8(u8_owner, seed);
    cvh::Mat f32_owner({shape.rows + 2, shape.cols + 3}, CV_32FC3);
    cvh::Mat f32_src = f32_owner(cvh::Range(1, shape.rows + 1), cvh::Range(1, shape.cols + 1));
    fill_f32(f32_owner, seed);

    cvh::Mat kernel({3, 3}, CV_32FC1);
    float* kernel_values = reinterpret_cast<float*>(kernel.data);
    const float filter_values[9] = {0.0f, 0.25f, 0.0f, 0.25f, 0.0f, 0.25f, 0.0f, 0.25f, 0.0f};
    std::copy(filter_values, filter_values + 9, kernel_values);
    cvh::Mat kernel_x({1, 3}, CV_32FC1);
    cvh::Mat kernel_y({3, 1}, CV_32FC1);
    float* kx = reinterpret_cast<float*>(kernel_x.data);
    float* ky = reinterpret_cast<float*>(kernel_y.data);
    kx[0] = ky[0] = 0.25f;
    kx[1] = ky[1] = 0.5f;
    kx[2] = ky[2] = 0.25f;

    const auto add_roi_row = [&](const char* op,
                                 const char* variant,
                                 const char* dispatch_path,
                                 const char* depth,
                                 int channels,
                                 double cvh_ms,
                                 double opencv_ms) {
        append_row(
            rows, args, "imgproc", op, variant, dispatch_path, depth, channels,
            shape_name, cvh_ms, opencv_ms);
        rows.back().layout = "roi";
    };

    {
        cvh::Mat dst;
        const double cvh_ms = measure_cvh_mat_ms(
            [&]() {
                cvh::resize(
                    u8_src, dst, cvh::Size(shape.cols * 3 / 4, shape.rows * 3 / 4),
                    0.0, 0.0, cvh::INTER_LINEAR);
            },
            dst,
            args);
        const double opencv_ms = bench_opencv_imgproc_roi(
            ImgprocRoiOpId::ResizeLinear, shape.rows, shape.cols,
            args.warmup, args.iters, args.repeats, seed);
        add_roi_row(
            "RESIZE", "linear_0.75_u8c3_roi", "header_fastpath", "CV_8U", 3,
            cvh_ms, opencv_ms);
    }

    {
        cvh::Mat dst;
        const double cvh_ms = measure_cvh_mat_ms(
            [&]() { cvh::cvtColor(u8_src, dst, cvh::COLOR_BGR2GRAY); },
            dst,
            args);
        const double opencv_ms = bench_opencv_imgproc_roi(
            ImgprocRoiOpId::CvtColorBgr2Gray, shape.rows, shape.cols,
            args.warmup, args.iters, args.repeats, seed);
        add_roi_row(
            "CVTCOLOR", "BGR2GRAY_u8_roi", "opencv_ui", "CV_8U", 3,
            cvh_ms, opencv_ms);
    }

    {
        cvh::Mat dst;
        const double cvh_ms = measure_cvh_mat_ms(
            [&]() { cvh::threshold(f32_src, dst, 0.5, 1.0, cvh::THRESH_BINARY); },
            dst,
            args);
        const double opencv_ms = bench_opencv_imgproc_roi(
            ImgprocRoiOpId::ThresholdF32, shape.rows, shape.cols,
            args.warmup, args.iters, args.repeats, seed);
        add_roi_row(
            "THRESHOLD", "binary_f32c3_roi", "header_fastpath", "CV_32F", 3,
            cvh_ms, opencv_ms);
    }

    {
        cvh::Mat dst;
        const double cvh_ms = measure_cvh_mat_ms(
            [&]() { cvh::boxFilter(u8_src, dst, -1, cvh::Size(3, 3), cvh::Point(-1, -1), true, cvh::BORDER_REPLICATE); },
            dst,
            args);
        const double opencv_ms = bench_opencv_imgproc_roi(
            ImgprocRoiOpId::Box, shape.rows, shape.cols,
            args.warmup, args.iters, args.repeats, seed);
        add_roi_row(
            "BOX_FILTER", "3x3_replicate_u8c3_roi",
            cvh::detail::last_boxfilter_dispatch_path(), "CV_8U", 3,
            cvh_ms, opencv_ms);
    }

    {
        cvh::Mat dst;
        const double cvh_ms = measure_cvh_mat_ms(
            [&]() { cvh::GaussianBlur(u8_src, dst, cvh::Size(5, 5), 0.0, 0.0, cvh::BORDER_REPLICATE); },
            dst,
            args);
        const double opencv_ms = bench_opencv_imgproc_roi(
            ImgprocRoiOpId::Gaussian, shape.rows, shape.cols,
            args.warmup, args.iters, args.repeats, seed);
        add_roi_row(
            "GAUSSIAN", "5x5_replicate_u8c3_roi",
            cvh::detail::last_gaussianblur_dispatch_path(), "CV_8U", 3,
            cvh_ms, opencv_ms);
    }

    {
        cvh::Mat dst;
        const double cvh_ms = measure_cvh_mat_ms(
            [&]() { cvh::filter2D(u8_src, dst, -1, kernel, cvh::Point(-1, -1), 0.0, cvh::BORDER_REPLICATE); },
            dst,
            args);
        const double opencv_ms = bench_opencv_imgproc_roi(
            ImgprocRoiOpId::Filter2D, shape.rows, shape.cols,
            args.warmup, args.iters, args.repeats, seed);
        add_roi_row(
            "FILTER2D", "3x3_replicate_u8c3_roi", "header_fastpath", "CV_8U", 3,
            cvh_ms, opencv_ms);
    }

    {
        cvh::Mat dst;
        const double cvh_ms = measure_cvh_mat_ms(
            [&]() { cvh::sepFilter2D(u8_src, dst, -1, kernel_x, kernel_y, cvh::Point(-1, -1), 0.0, cvh::BORDER_REPLICATE); },
            dst,
            args);
        const double opencv_ms = bench_opencv_imgproc_roi(
            ImgprocRoiOpId::SepFilter2D, shape.rows, shape.cols,
            args.warmup, args.iters, args.repeats, seed);
        add_roi_row(
            "SEP_FILTER2D", "3x3_replicate_u8c3_roi", "header_fastpath", "CV_8U", 3,
            cvh_ms, opencv_ms);
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

    out << "impl,profile,suite,op,variant,dispatch_path,depth,channels,layout,shape,cvh_ms,opencv_ms,speedup,status,note\n";
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
            << row.layout << ","
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
    cvh_bench_compare::append_core_compute_cases(args, rows);
    cvh_bench_compare::append_imgproc_cases(args, rows);
    cvh_bench_compare::append_imgproc_type_channel_cases(args, rows);
    cvh_bench_compare::append_imgproc_resize_color_cases(args, rows);
    cvh_bench_compare::append_imgproc_roi_cases(args, rows);

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
