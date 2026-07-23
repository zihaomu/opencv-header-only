#include "cvh.h"
#include "common/benchmark_common.h"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace cvh_bench {

using Result = common::BenchmarkResult;

struct Args
{
    std::string profile = "quick";
    int warmup = 3;
    int iters = 10;
    int repeats = 7;
    std::string output_csv;
};

struct ShapeCase
{
    const char* name;
    int rows;
    int cols;
    int type;
};

struct ResultRow
{
    std::string mode = "internal";
    std::string suite = "core_mat";
    std::string module = "core";
    std::string op;
    std::string variant;
    std::string depth;
    int channels = 0;
    std::string layout;
    std::string shape;
    std::size_t elements = 0;
    std::size_t pixels = 0;
    std::string implementation = "cvh_headers";
    std::string dispatch_path = "header_only";
    std::string allocation_mode;
    int warmup = 0;
    int iters = 0;
    int repeats = 0;
    int threads = 1;
    double min_ms = 0.0;
    double median_ms = 0.0;
    double mpix_per_sec = 0.0;
    double melems_per_sec = 0.0;
    double gb_per_sec = 0.0;
    std::uint64_t checksum = 0;
    std::string status = "OK";
    std::string note;
};

volatile std::uint64_t g_sink = 0;

void usage()
{
    std::cout
        << "Usage: cvh_benchmark_core_mat_header "
        << "[--profile quick|stable|full] [--warmup N] [--iters N] [--repeats N] [--output path]\n";
}

Args parse_args(int argc, char** argv)
{
    const auto parsed = common::parse_basic_args(
        argc,
        argv,
        common::BasicArgs {"quick", 3, 10, 7, ""},
        {"quick", "stable", "full"},
        usage);
    return Args {parsed.profile, parsed.warmup, parsed.iters, parsed.repeats, parsed.output_csv};
}

std::string depth_name(int depth)
{
    switch (depth)
    {
        case CV_8U: return "CV_8U";
        case CV_16S: return "CV_16S";
        case CV_32F: return "CV_32F";
        default: return "UNKNOWN";
    }
}

std::string shape_name(const ShapeCase& shape)
{
    std::ostringstream oss;
    oss << shape.cols << "x" << shape.rows << "C" << CV_MAT_CN(shape.type);
    return oss.str();
}

std::string fmt(double value)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << value;
    return oss.str();
}

std::vector<ShapeCase> build_shapes(const std::string& profile)
{
    std::vector<ShapeCase> shapes = {
        {"tiny_u8c1", 32, 32, CV_8UC1},
        {"vga_u8c1", 480, 640, CV_8UC1},
        {"vga_u8c3", 480, 640, CV_8UC3},
        {"vga_f32c1", 480, 640, CV_32FC1},
    };

    if (profile == "stable" || profile == "full")
    {
        shapes.push_back({"hd_u8c1", 1080, 1920, CV_8UC1});
        shapes.push_back({"hd_u8c3", 1080, 1920, CV_8UC3});
        shapes.push_back({"hd_f32c1", 1080, 1920, CV_32FC1});
    }
    if (profile == "full")
    {
        shapes.push_back({"nonaligned_u8c1", 479, 641, CV_8UC1});
        shapes.push_back({"uhd_u8c1", 2160, 3840, CV_8UC1});
        shapes.push_back({"uhd_u8c3", 2160, 3840, CV_8UC3});
    }

    return shapes;
}

void fill_mat(cvh::Mat& mat, std::uint32_t seed)
{
    if (mat.depth() == CV_8U)
    {
        common::fill_mat_u8_lcg(mat, seed);
        return;
    }

    if (mat.depth() == CV_32F)
    {
        const std::size_t count = mat.total() * static_cast<std::size_t>(mat.channels());
        float* ptr = reinterpret_cast<float*>(mat.data);
        for (std::size_t i = 0; i < count; ++i)
        {
            seed = seed * 1664525u + 1013904223u;
            ptr[i] = static_cast<float>(static_cast<int>((seed >> 8) & 0xffffu) - 32768) / 1024.0f;
        }
    }
}

template <typename RunFn, typename ChecksumFn>
Result measure(RunFn&& run_once, ChecksumFn&& checksum_fn, const Args& args)
{
    const auto timing = common::measure_repeated_ms(run_once, args.warmup, args.iters, args.repeats);
    const std::uint64_t hash = checksum_fn();
    g_sink ^= hash;
    return Result {timing.min_ms, timing.median_ms, hash};
}

ResultRow make_row(const Args& args,
                   const ShapeCase& shape,
                   const std::string& op,
                   const std::string& variant,
                   const std::string& layout,
                   const std::string& allocation_mode,
                   std::size_t bytes_touched,
                   const Result& result)
{
    const std::size_t pixels = static_cast<std::size_t>(shape.rows) * static_cast<std::size_t>(shape.cols);
    const std::size_t elements = pixels * static_cast<std::size_t>(CV_MAT_CN(shape.type));
    ResultRow row;
    row.op = op;
    row.variant = variant;
    row.depth = depth_name(CV_MAT_DEPTH(shape.type));
    row.channels = CV_MAT_CN(shape.type);
    row.layout = layout;
    row.shape = shape_name(shape);
    row.elements = elements;
    row.pixels = pixels;
    row.allocation_mode = allocation_mode;
    row.warmup = args.warmup;
    row.iters = args.iters;
    row.repeats = args.repeats;
    row.min_ms = result.min_ms;
    row.median_ms = result.median_ms;
    row.mpix_per_sec = common::mpix_per_sec(pixels, result.median_ms);
    row.melems_per_sec = result.median_ms > 0.0 ? static_cast<double>(elements) / result.median_ms / 1000.0 : 0.0;
    row.gb_per_sec = result.median_ms > 0.0 ? static_cast<double>(bytes_touched) / result.median_ms / 1.0e6 : 0.0;
    row.checksum = result.checksum;
    return row;
}

void print_csv(const std::vector<ResultRow>& rows, std::ostream& os)
{
    common::write_csv_row(
        os,
        {
            "mode", "suite", "module", "op", "variant", "depth", "channels", "layout", "shape", "elements",
            "pixels", "implementation", "dispatch_path", "allocation_mode", "warmup", "iters", "repeats",
            "threads", "min_ms", "median_ms", "mpix_per_sec", "melems_per_sec", "gb_per_sec", "checksum",
            "status", "note",
        });

    for (const auto& row : rows)
    {
        common::write_csv_row(
            os,
            {
                row.mode,
                row.suite,
                row.module,
                row.op,
                row.variant,
                row.depth,
                std::to_string(row.channels),
                row.layout,
                row.shape,
                std::to_string(row.elements),
                std::to_string(row.pixels),
                row.implementation,
                row.dispatch_path,
                row.allocation_mode,
                std::to_string(row.warmup),
                std::to_string(row.iters),
                std::to_string(row.repeats),
                std::to_string(row.threads),
                fmt(row.min_ms),
                fmt(row.median_ms),
                fmt(row.mpix_per_sec),
                fmt(row.melems_per_sec),
                fmt(row.gb_per_sec),
                std::to_string(row.checksum),
                row.status,
                row.note,
            });
    }
}

void append_shape_rows(const Args& args, const ShapeCase& shape, std::vector<ResultRow>& rows)
{
    const std::vector<int> dims {shape.rows, shape.cols};
    const std::vector<int> alt_dims {shape.rows + 1, shape.cols};
    const std::size_t bytes = static_cast<std::size_t>(shape.rows) * static_cast<std::size_t>(shape.cols) *
                              static_cast<std::size_t>(CV_MAT_CN(shape.type)) *
                              static_cast<std::size_t>(CV_ELEM_SIZE1(shape.type));

    cvh::Mat src(dims, shape.type);
    fill_mat(src, static_cast<std::uint32_t>(shape.rows * 131 + shape.cols * 17 + CV_MAT_CN(shape.type)));

    {
        cvh::Mat dst(dims, shape.type);
        const auto result = measure(
            [&]() { dst.create(dims, shape.type); },
            [&]() { return common::fnv1a64_mix_u64(common::fnv1a64_basis(), dst.total()); },
            args);
        rows.push_back(make_row(args, shape, "MAT_CREATE", "reuse_same_shape", "none", "reuse", 0, result));
    }

    {
        cvh::Mat dst;
        bool toggle = false;
        const auto result = measure(
            [&]() {
                dst.create(toggle ? dims : alt_dims, shape.type);
                toggle = !toggle;
            },
            [&]() { return common::fnv1a64_mix_u64(common::fnv1a64_basis(), dst.total()); },
            args);
        rows.push_back(make_row(args, shape, "MAT_CREATE", "alternate_shape", "none", "recreate", 0, result));
    }

    {
        cvh::Mat dst(dims, shape.type);
        const auto result = measure(
            [&]() {
                dst.release();
                dst.create(dims, shape.type);
            },
            [&]() { return common::fnv1a64_mix_u64(common::fnv1a64_basis(), dst.total()); },
            args);
        rows.push_back(make_row(args, shape, "MAT_RELEASE_CREATE", "release_then_create", "none", "recreate", 0, result));
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() { dst = src.clone(); },
            [&]() { return common::checksum_mat_bytes(dst); },
            args);
        rows.push_back(make_row(args, shape, "MAT_CLONE", "full_copy", "continuous", "recreate", bytes, result));
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() { src.copyTo(dst); },
            [&]() { return common::checksum_mat_bytes(dst); },
            args);
        rows.push_back(make_row(args, shape, "MAT_COPYTO", "continuous", "continuous", "reuse", bytes, result));
    }

    {
        cvh::Mat parent({shape.rows + 2, shape.cols + 2}, shape.type);
        fill_mat(parent, static_cast<std::uint32_t>(shape.rows * 19 + shape.cols * 23));
        cvh::Mat roi = parent(cvh::Range(1, shape.rows + 1), cvh::Range(1, shape.cols + 1));
        cvh::Mat dst;
        const auto result = measure(
            [&]() { roi.copyTo(dst); },
            [&]() { return common::checksum_mat_bytes(dst); },
            args);
        rows.push_back(make_row(args, shape, "MAT_COPYTO", "roi_to_continuous", "roi", "reuse", bytes, result));
    }

    {
        cvh::Mat dst(dims, shape.type);
        const auto result = measure(
            [&]() { dst.setTo(cvh::Scalar::all(7.0)); },
            [&]() { return common::checksum_mat_bytes(dst); },
            args);
        rows.push_back(make_row(args, shape, "MAT_SETTO", "scalar_all", "continuous", "reuse", bytes, result));
    }

    {
        const int dst_type = CV_MAKETYPE(shape.type == CV_32FC1 ? CV_8U : CV_32F, CV_MAT_CN(shape.type));
        cvh::Mat dst;
        const auto result = measure(
            [&]() { src.convertTo(dst, dst_type); },
            [&]() { return common::checksum_mat_bytes(dst); },
            args);
        rows.push_back(make_row(args, shape, "MAT_CONVERTTO", depth_name(CV_MAT_DEPTH(dst_type)), "continuous", "reuse", bytes, result));
    }

    {
        const std::vector<int> reshaped_dims {shape.rows * shape.cols, 1};
        cvh::Mat view;
        const auto result = measure(
            [&]() { view = src.reshape(reshaped_dims); },
            [&]() { return common::fnv1a64_mix_u64(common::fnv1a64_basis(), view.total()); },
            args);
        rows.push_back(make_row(args, shape, "MAT_RESHAPE", "to_vector_view", "continuous", "none", 0, result));
    }
}

}  // namespace cvh_bench

int main(int argc, char** argv)
{
    const auto args = cvh_bench::parse_args(argc, argv);
    const auto shapes = cvh_bench::build_shapes(args.profile);
    std::vector<cvh_bench::ResultRow> rows;
    rows.reserve(shapes.size() * 9);

    for (const auto& shape : shapes)
    {
        cvh_bench::append_shape_rows(args, shape, rows);
    }

    cvh_bench::print_csv(rows, std::cout);
    if (!args.output_csv.empty())
    {
        std::ofstream ofs(args.output_csv);
        if (!ofs)
        {
            std::cerr << "Failed to open output: " << args.output_csv << "\n";
            return 4;
        }
        cvh_bench::print_csv(rows, ofs);
    }

    return static_cast<int>(cvh_bench::g_sink & 0u);
}
