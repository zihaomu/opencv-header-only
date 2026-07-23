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
    int warmup = 2;
    int iters = 5;
    int repeats = 3;
    std::string output_csv;
};

struct ShapeCase
{
    const char* name;
    int rows;
    int cols;
};

struct ResultRow
{
    std::string mode = "internal";
    std::string suite = "imgproc";
    std::string module = "imgproc";
    std::string op;
    std::string variant;
    std::string depth;
    int channels = 0;
    std::string layout = "continuous";
    std::string shape;
    std::size_t elements = 0;
    std::size_t pixels = 0;
    std::string implementation = "cvh_headers_fast";
    std::string dispatch_path = "public_header";
    std::string allocation_mode = "reuse";
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
        << "Usage: cvh_benchmark_imgproc_header "
        << "[--profile quick|stable|full] [--warmup N] [--iters N] [--repeats N] [--output path]\n";
}

Args parse_args(int argc, char** argv)
{
    const auto parsed = common::parse_basic_args(
        argc,
        argv,
        common::BasicArgs {"quick", 2, 5, 3, ""},
        {"quick", "stable", "full"},
        usage);
    return Args {parsed.profile, parsed.warmup, parsed.iters, parsed.repeats, parsed.output_csv};
}

std::string fmt(double value)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << value;
    return oss.str();
}

std::vector<ShapeCase> build_shapes(const std::string& profile)
{
    std::vector<ShapeCase> shapes = {{"640x480", 480, 640}};
    if (profile == "stable" || profile == "full")
    {
        shapes.push_back({"1280x720", 720, 1280});
        shapes.push_back({"1920x1080", 1080, 1920});
    }
    if (profile == "full")
    {
        shapes.push_back({"641x479", 479, 641});
        shapes.push_back({"3840x2160", 2160, 3840});
    }
    return shapes;
}

void fill_lut(cvh::Mat& lut)
{
    lut.create(std::vector<int>{256, 1}, CV_8UC1);
    for (int i = 0; i < 256; ++i)
    {
        lut.data[i] = static_cast<uchar>(255 - i);
    }
}

template <typename RunFn>
Result measure(RunFn&& run_once, cvh::Mat& dst, const Args& args)
{
    const auto timing = common::measure_repeated_ms(run_once, args.warmup, args.iters, args.repeats);
    const std::uint64_t hash = common::checksum_mat_bytes(dst);
    g_sink ^= hash;
    return Result {timing.min_ms, timing.median_ms, hash};
}

ResultRow make_row(const Args& args,
                   const ShapeCase& shape,
                   const std::string& op,
                   const std::string& variant,
                   int type,
                   int output_rows,
                   int output_cols,
                   std::size_t bytes_touched,
                   const Result& result)
{
    const std::size_t pixels = static_cast<std::size_t>(output_rows) * static_cast<std::size_t>(output_cols);
    const std::size_t elements = pixels * static_cast<std::size_t>(CV_MAT_CN(type));
    ResultRow row;
    row.op = op;
    row.variant = variant;
    row.depth = CV_MAT_DEPTH(type) == CV_32F ? "CV_32F" : CV_MAT_DEPTH(type) == CV_16S ? "CV_16S" : "CV_8U";
    row.channels = CV_MAT_CN(type);
    row.shape = shape.name;
    row.elements = elements;
    row.pixels = pixels;
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
    cvh::Mat gray({shape.rows, shape.cols}, CV_8UC1);
    cvh::Mat bgr({shape.rows, shape.cols}, CV_8UC3);
    common::fill_mat_u8_lcg(gray, static_cast<std::uint32_t>(shape.rows * 131 + shape.cols * 17));
    common::fill_mat_u8_lcg(bgr, static_cast<std::uint32_t>(shape.rows * 19 + shape.cols * 23));

    const std::size_t gray_bytes = static_cast<std::size_t>(shape.rows) * static_cast<std::size_t>(shape.cols);
    const std::size_t bgr_bytes = gray_bytes * 3u;

    {
        cvh::Mat dst;
        const int out_rows = shape.rows / 2;
        const int out_cols = shape.cols / 2;
        const auto result = measure(
            [&]() { cvh::resize(gray, dst, cvh::Size(out_cols, out_rows), 0.0, 0.0, cvh::INTER_LINEAR); },
            dst,
            args);
        rows.push_back(make_row(args, shape, "RESIZE", "INTER_LINEAR_0.5_U8C1", CV_8UC1, out_rows, out_cols, gray_bytes, result));
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() { cvh::cvtColor(bgr, dst, cvh::COLOR_BGR2GRAY); },
            dst,
            args);
        rows.push_back(make_row(args, shape, "CVTCOLOR", "BGR2GRAY_U8", CV_8UC1, shape.rows, shape.cols, bgr_bytes, result));
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() { cvh::cvtColor(gray, dst, cvh::COLOR_GRAY2BGR); },
            dst,
            args);
        rows.push_back(make_row(args, shape, "CVTCOLOR", "GRAY2BGR_U8", CV_8UC3, shape.rows, shape.cols, gray_bytes, result));
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() { (void)cvh::threshold(gray, dst, 96.0, 255.0, cvh::THRESH_BINARY); },
            dst,
            args);
        rows.push_back(make_row(args, shape, "THRESHOLD", "BINARY_U8", CV_8UC1, shape.rows, shape.cols, gray_bytes, result));
    }

    {
        cvh::Mat lut;
        fill_lut(lut);
        cvh::Mat dst;
        const auto result = measure(
            [&]() { cvh::LUT(gray, lut, dst); },
            dst,
            args);
        rows.push_back(make_row(args, shape, "LUT", "U8C1_TABLE_C1", CV_8UC1, shape.rows, shape.cols, gray_bytes, result));
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() { cvh::copyMakeBorder(gray, dst, 2, 2, 2, 2, cvh::BORDER_REPLICATE, cvh::Scalar::all(0.0)); },
            dst,
            args);
        rows.push_back(make_row(args, shape, "COPYMAKEBORDER", "REPLICATE_2PX_U8C1", CV_8UC1, shape.rows + 4, shape.cols + 4, gray_bytes, result));
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() { cvh::boxFilter(gray, dst, -1, cvh::Size(3, 3), cvh::Point(-1, -1), true, cvh::BORDER_REPLICATE); },
            dst,
            args);
        rows.push_back(make_row(args, shape, "BOXFILTER", "3x3_NORMALIZED_U8C1", CV_8UC1, shape.rows, shape.cols, gray_bytes, result));
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() { cvh::GaussianBlur(gray, dst, cvh::Size(5, 5), 0.0, 0.0, cvh::BORDER_REPLICATE); },
            dst,
            args);
        rows.push_back(make_row(args, shape, "GAUSSIANBLUR", "5x5_U8C1", CV_8UC1, shape.rows, shape.cols, gray_bytes, result));
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() { cvh::Sobel(gray, dst, CV_32F, 1, 0, 3, 1.0, 0.0, cvh::BORDER_REPLICATE); },
            dst,
            args);
        rows.push_back(make_row(args, shape, "SOBEL", "DX1_K3_U8C1_TO_F32", CV_32FC1, shape.rows, shape.cols, gray_bytes, result));
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() { cvh::Canny(gray, dst, 50.0, 130.0, 3, false); },
            dst,
            args);
        rows.push_back(make_row(args, shape, "CANNY", "A3_L1_U8C1", CV_8UC1, shape.rows, shape.cols, gray_bytes, result));
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() { cvh::erode(gray, dst, cvh::Mat(), cvh::Point(-1, -1), 1, cvh::BORDER_REPLICATE); },
            dst,
            args);
        rows.push_back(make_row(args, shape, "ERODE", "3x3_U8C1", CV_8UC1, shape.rows, shape.cols, gray_bytes, result));
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() { cvh::dilate(gray, dst, cvh::Mat(), cvh::Point(-1, -1), 1, cvh::BORDER_REPLICATE); },
            dst,
            args);
        rows.push_back(make_row(args, shape, "DILATE", "3x3_U8C1", CV_8UC1, shape.rows, shape.cols, gray_bytes, result));
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() { cvh::morphologyEx(gray, dst, cvh::MORPH_GRADIENT, cvh::Mat(), cvh::Point(-1, -1), 1, cvh::BORDER_REPLICATE); },
            dst,
            args);
        rows.push_back(make_row(args, shape, "MORPHOLOGYEX", "GRADIENT_3x3_U8C1", CV_8UC1, shape.rows, shape.cols, gray_bytes, result));
    }
}

}  // namespace cvh_bench

int main(int argc, char** argv)
{
    const auto args = cvh_bench::parse_args(argc, argv);
    const auto shapes = cvh_bench::build_shapes(args.profile);
    std::vector<cvh_bench::ResultRow> rows;
    rows.reserve(shapes.size() * 13);

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
