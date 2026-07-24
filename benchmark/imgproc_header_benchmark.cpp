#include "cvh.h"
#include "common/benchmark_common.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace cvh_bench {

using Result = common::BenchmarkResult;
constexpr int kBenchmarkSchemaVersion = 2;

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
    double tail_ratio = 0.0;
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

std::string type_suffix(int type)
{
    const char* depth = CV_MAT_DEPTH(type) == CV_32F ? "F32" : "U8";
    return std::string(depth) + "C" + std::to_string(CV_MAT_CN(type));
}

void fill_input(cvh::Mat& mat, std::uint32_t seed)
{
    if (mat.depth() == CV_32F)
    {
        common::fill_mat_f32_lcg(mat, seed);
    }
    else
    {
        common::fill_mat_u8_lcg(mat, seed);
    }
}

std::size_t logical_bytes(const cvh::Mat& mat)
{
    return mat.total() * mat.elemSize();
}

bool is_matrix_anchor(const ShapeCase& shape)
{
    return shape.rows == 480 && shape.cols == 640;
}

template <typename RunFn>
Result measure(RunFn&& run_once, cvh::Mat& dst, const Args& args)
{
    run_once();
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
    switch (CV_MAT_DEPTH(type))
    {
        case CV_16S: row.depth = "CV_16S"; break;
        case CV_32S: row.depth = "CV_32S"; break;
        case CV_32F: row.depth = "CV_32F"; break;
        case CV_64F: row.depth = "CV_64F"; break;
        default: row.depth = "CV_8U"; break;
    }
    row.channels = CV_MAT_CN(type);
    row.shape = shape.name;
    row.elements = elements;
    row.pixels = pixels;
    row.warmup = args.warmup;
    row.iters = args.iters;
    row.repeats = args.repeats;
    if (CV_MAT_DEPTH(type) == CV_8U && output_cols > 0)
    {
        const int lanes = common::simd_u8_lanes();
        row.tail_ratio = static_cast<double>(output_cols % lanes) / static_cast<double>(output_cols);
    }
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
            "schema_version", "mode", "suite", "module", "op", "variant", "depth", "channels", "layout",
            "shape", "elements", "pixels", "implementation", "dispatch_path", "allocation_mode",
            "tail_ratio", "warmup", "iters", "repeats", "threads", "min_ms", "median_ms",
            "mpix_per_sec", "melems_per_sec", "gb_per_sec", "checksum", "status", "note",
        });
    for (const auto& row : rows)
    {
        common::write_csv_row(
            os,
            {
                std::to_string(kBenchmarkSchemaVersion),
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
                fmt(row.tail_ratio),
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

void append_resize_matrix_rows(const Args& args, const ShapeCase& shape, std::vector<ResultRow>& rows)
{
    const int types[] = {CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4};
    const int interpolations[] = {cvh::INTER_NEAREST, cvh::INTER_NEAREST_EXACT, cvh::INTER_LINEAR};
    const int out_rows = shape.rows * 3 / 4;
    const int out_cols = shape.cols * 3 / 4;

    for (const int type : types)
    {
        cvh::Mat src({shape.rows, shape.cols}, type);
        fill_input(src, static_cast<std::uint32_t>(shape.rows * 31 + shape.cols * 43 + type * 7));
        for (const int interpolation : interpolations)
        {
            const char* interpolation_name =
                interpolation == cvh::INTER_NEAREST
                    ? "INTER_NEAREST"
                    : interpolation == cvh::INTER_NEAREST_EXACT ? "INTER_NEAREST_EXACT" : "INTER_LINEAR";
            cvh::Mat dst;
            const auto result = measure(
                [&]() { cvh::resize(src, dst, cvh::Size(out_cols, out_rows), 0.0, 0.0, interpolation); },
                dst,
                args);
            auto row = make_row(
                args,
                shape,
                "RESIZE",
                std::string(interpolation_name) + "_0.75_" + type_suffix(type),
                type,
                out_rows,
                out_cols,
                logical_bytes(src),
                result);
            row.dispatch_path = CV_MAT_DEPTH(type) == CV_8U ? "header_fastpath" : "headers_baseline";
            rows.push_back(row);
        }
    }

    auto unsupported = make_row(
        args,
        shape,
        "RESIZE",
        "INTER_CUBIC_OUTSIDE_ACCEPTED_CONTRACT",
        CV_8UC1,
        shape.rows * 3 / 4,
        shape.cols * 3 / 4,
        0,
        Result {});
    unsupported.dispatch_path = "unsupported";
    unsupported.allocation_mode = "none";
    unsupported.status = "UNSUPPORTED";
    unsupported.note = "accepted resize contract is nearest/nearest-exact/linear";
    rows.push_back(unsupported);
}

struct ColorMatrixCase
{
    const char* variant;
    int code;
    int src_type;
    int dst_type;
};

void append_color_case(const Args& args,
                       const ShapeCase& shape,
                       const ColorMatrixCase& color_case,
                       std::vector<ResultRow>& rows)
{
    cvh::Mat src({shape.rows, shape.cols}, color_case.src_type);
    fill_input(src, static_cast<std::uint32_t>(shape.rows * 47 + shape.cols * 53 + color_case.code * 11));
    cvh::Mat dst;
    const auto result = measure(
        [&]() { cvh::cvtColor(src, dst, color_case.code); },
        dst,
        args);
    auto row = make_row(
        args,
        shape,
        "CVTCOLOR",
        color_case.variant,
        color_case.dst_type,
        shape.rows,
        shape.cols,
        logical_bytes(src),
        result);
    row.dispatch_path = "header_fastpath";
    rows.push_back(row);
}

void append_color_matrix_rows(const Args& args, const ShapeCase& shape, std::vector<ResultRow>& rows)
{
    const ColorMatrixCase stable_cases[] = {
        {"BGR2RGB_U8", cvh::COLOR_BGR2RGB, CV_8UC3, CV_8UC3},
        {"BGR2BGRA_U8", cvh::COLOR_BGR2BGRA, CV_8UC3, CV_8UC4},
        {"BGRA2BGR_U8", cvh::COLOR_BGRA2BGR, CV_8UC4, CV_8UC3},
        {"RGB2RGBA_U8", cvh::COLOR_RGB2RGBA, CV_8UC3, CV_8UC4},
        {"RGBA2RGB_U8", cvh::COLOR_RGBA2RGB, CV_8UC4, CV_8UC3},
        {"BGRA2RGBA_U8", cvh::COLOR_BGRA2RGBA, CV_8UC4, CV_8UC4},
        {"GRAY2BGRA_U8", cvh::COLOR_GRAY2BGRA, CV_8UC1, CV_8UC4},
        {"BGRA2GRAY_U8", cvh::COLOR_BGRA2GRAY, CV_8UC4, CV_8UC1},
        {"BGR2YUV_U8", cvh::COLOR_BGR2YUV, CV_8UC3, CV_8UC3},
        {"YUV2BGR_U8", cvh::COLOR_YUV2BGR, CV_8UC3, CV_8UC3},
        {"BGR2GRAY_F32", cvh::COLOR_BGR2GRAY, CV_32FC3, CV_32FC1},
        {"GRAY2BGR_F32", cvh::COLOR_GRAY2BGR, CV_32FC1, CV_32FC3},
        {"BGR2RGB_F32", cvh::COLOR_BGR2RGB, CV_32FC3, CV_32FC3},
        {"BGR2YUV_F32", cvh::COLOR_BGR2YUV, CV_32FC3, CV_32FC3},
        {"YUV2BGR_F32", cvh::COLOR_YUV2BGR, CV_32FC3, CV_32FC3},
    };
    for (const auto& color_case : stable_cases)
    {
        append_color_case(args, shape, color_case, rows);
    }

    if (args.profile != "full")
    {
        return;
    }

    const ColorMatrixCase full_cases[] = {
        {"BGR2RGBA_U8", cvh::COLOR_BGR2RGBA, CV_8UC3, CV_8UC4},
        {"RGBA2BGR_U8", cvh::COLOR_RGBA2BGR, CV_8UC4, CV_8UC3},
        {"RGB2BGRA_U8", cvh::COLOR_RGB2BGRA, CV_8UC3, CV_8UC4},
        {"BGRA2RGB_U8", cvh::COLOR_BGRA2RGB, CV_8UC4, CV_8UC3},
        {"RGBA2BGRA_U8", cvh::COLOR_RGBA2BGRA, CV_8UC4, CV_8UC4},
        {"GRAY2RGBA_U8", cvh::COLOR_GRAY2RGBA, CV_8UC1, CV_8UC4},
        {"RGBA2GRAY_U8", cvh::COLOR_RGBA2GRAY, CV_8UC4, CV_8UC1},
        {"BGR2BGRA_F32", cvh::COLOR_BGR2BGRA, CV_32FC3, CV_32FC4},
        {"BGRA2BGR_F32", cvh::COLOR_BGRA2BGR, CV_32FC4, CV_32FC3},
        {"BGR2RGBA_F32", cvh::COLOR_BGR2RGBA, CV_32FC3, CV_32FC4},
        {"RGBA2BGR_F32", cvh::COLOR_RGBA2BGR, CV_32FC4, CV_32FC3},
        {"GRAY2BGRA_F32", cvh::COLOR_GRAY2BGRA, CV_32FC1, CV_32FC4},
        {"BGRA2GRAY_F32", cvh::COLOR_BGRA2GRAY, CV_32FC4, CV_32FC1},
        {"GRAY2RGBA_F32", cvh::COLOR_GRAY2RGBA, CV_32FC1, CV_32FC4},
        {"RGBA2GRAY_F32", cvh::COLOR_RGBA2GRAY, CV_32FC4, CV_32FC1},
    };
    for (const auto& color_case : full_cases)
    {
        append_color_case(args, shape, color_case, rows);
    }
}

enum class MatrixFilterOp
{
    Box,
    Blur,
    Gaussian,
    Filter2D,
    SepFilter2D,
};

void append_filter_matrix_case(const Args& args,
                               const ShapeCase& shape,
                               int type,
                               MatrixFilterOp op,
                               std::vector<ResultRow>& rows)
{
    cvh::Mat src({shape.rows, shape.cols}, type);
    fill_input(src, static_cast<std::uint32_t>(shape.rows * 59 + shape.cols * 61 + type * 13));
    cvh::Mat kernel({3, 3}, CV_32FC1);
    float* values = reinterpret_cast<float*>(kernel.data);
    const float kernel_values[9] = {0.0f, 0.25f, 0.0f, 0.25f, 0.0f, 0.25f, 0.0f, 0.25f, 0.0f};
    std::copy(kernel_values, kernel_values + 9, values);
    cvh::Mat kernel_x({1, 3}, CV_32FC1);
    cvh::Mat kernel_y({3, 1}, CV_32FC1);
    float* kx = reinterpret_cast<float*>(kernel_x.data);
    float* ky = reinterpret_cast<float*>(kernel_y.data);
    kx[0] = ky[0] = 0.25f;
    kx[1] = ky[1] = 0.5f;
    kx[2] = ky[2] = 0.25f;

    cvh::Mat dst;
    const char* op_name = "";
    std::string dispatch_path = "header_fastpath";
    Result result;
    switch (op)
    {
        case MatrixFilterOp::Box:
            op_name = "BOXFILTER";
            result = measure(
                [&]() { cvh::boxFilter(src, dst, -1, cvh::Size(3, 3), cvh::Point(-1, -1), true, cvh::BORDER_REPLICATE); },
                dst,
                args);
            dispatch_path = cvh::detail::last_boxfilter_dispatch_path();
            break;
        case MatrixFilterOp::Blur:
            op_name = "BLUR";
            result = measure(
                [&]() { cvh::blur(src, dst, cvh::Size(3, 3), cvh::Point(-1, -1), cvh::BORDER_REPLICATE); },
                dst,
                args);
            dispatch_path = cvh::detail::last_boxfilter_dispatch_path();
            break;
        case MatrixFilterOp::Gaussian:
            op_name = "GAUSSIANBLUR";
            result = measure(
                [&]() { cvh::GaussianBlur(src, dst, cvh::Size(5, 5), 0.0, 0.0, cvh::BORDER_REPLICATE); },
                dst,
                args);
            dispatch_path = cvh::detail::last_gaussianblur_dispatch_path();
            break;
        case MatrixFilterOp::Filter2D:
            op_name = "FILTER2D";
            result = measure(
                [&]() { cvh::filter2D(src, dst, -1, kernel, cvh::Point(-1, -1), 0.0, cvh::BORDER_REPLICATE); },
                dst,
                args);
            break;
        case MatrixFilterOp::SepFilter2D:
            op_name = "SEPFILTER2D";
            result = measure(
                [&]() { cvh::sepFilter2D(src, dst, -1, kernel_x, kernel_y, cvh::Point(-1, -1), 0.0, cvh::BORDER_REPLICATE); },
                dst,
                args);
            break;
    }

    auto row = make_row(
        args,
        shape,
        op_name,
        std::string("3x3_") + type_suffix(type),
        type,
        shape.rows,
        shape.cols,
        logical_bytes(src),
        result);
    row.dispatch_path = dispatch_path;
    rows.push_back(row);
}

void append_threshold_filter_matrix_rows(const Args& args,
                                         const ShapeCase& shape,
                                         std::vector<ResultRow>& rows)
{
    const int threshold_types[] = {CV_8UC3, CV_32FC1, CV_32FC3, CV_32FC4};
    for (const int type : threshold_types)
    {
        cvh::Mat src({shape.rows, shape.cols}, type);
        fill_input(src, static_cast<std::uint32_t>(shape.rows * 67 + shape.cols * 71 + type * 17));
        cvh::Mat dst;
        const double threshold_value = CV_MAT_DEPTH(type) == CV_32F ? 0.5 : 96.0;
        const double max_value = CV_MAT_DEPTH(type) == CV_32F ? 1.0 : 255.0;
        const auto result = measure(
            [&]() { (void)cvh::threshold(src, dst, threshold_value, max_value, cvh::THRESH_BINARY); },
            dst,
            args);
        auto row = make_row(
            args,
            shape,
            "THRESHOLD",
            std::string("BINARY_") + type_suffix(type),
            type,
            shape.rows,
            shape.cols,
            logical_bytes(src),
            result);
        row.dispatch_path = CV_MAT_DEPTH(type) == CV_32F ? "header_fastpath" : "headers_baseline";
        rows.push_back(row);
    }

    const int broad_filter_types[] = {CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4};
    for (const int type : broad_filter_types)
    {
        append_filter_matrix_case(args, shape, type, MatrixFilterOp::Box, rows);
        append_filter_matrix_case(args, shape, type, MatrixFilterOp::Gaussian, rows);
    }

    const int focused_filter_types[] = {CV_8UC3, CV_32FC1};
    for (const int type : focused_filter_types)
    {
        append_filter_matrix_case(args, shape, type, MatrixFilterOp::Blur, rows);
        append_filter_matrix_case(args, shape, type, MatrixFilterOp::Filter2D, rows);
        append_filter_matrix_case(args, shape, type, MatrixFilterOp::SepFilter2D, rows);
    }
}

struct YuvMatrixCase
{
    const char* variant;
    const char* layout;
    int code;
    int src_type;
    int src_rows_numerator;
    int src_rows_denominator;
    int dst_type;
};

void append_yuv_matrix_rows(const Args& args, const ShapeCase& shape, std::vector<ResultRow>& rows)
{
    const YuvMatrixCase cases[] = {
        {"BGR2NV12_U8", "yuv420_nv12", cvh::COLOR_BGR2YUV_NV12, CV_8UC3, 1, 1, CV_8UC1},
        {"NV12_TO_BGR_U8", "yuv420_nv12", cvh::COLOR_YUV2BGR_NV12, CV_8UC1, 3, 2, CV_8UC3},
        {"BGR2I420_U8", "yuv420_i420", cvh::COLOR_BGR2YUV_I420, CV_8UC3, 1, 1, CV_8UC1},
        {"I420_TO_BGR_U8", "yuv420_i420", cvh::COLOR_YUV2BGR_I420, CV_8UC1, 3, 2, CV_8UC3},
        {"BGR2NV16_U8", "yuv422_nv16", cvh::COLOR_BGR2YUV_NV16, CV_8UC3, 1, 1, CV_8UC1},
        {"NV16_TO_BGR_U8", "yuv422_nv16", cvh::COLOR_YUV2BGR_NV16, CV_8UC1, 2, 1, CV_8UC3},
        {"BGR2YUY2_U8", "yuv422_yuy2", cvh::COLOR_BGR2YUV_YUY2, CV_8UC3, 1, 1, CV_8UC2},
        {"YUY2_TO_BGR_U8", "yuv422_yuy2", cvh::COLOR_YUV2BGR_YUY2, CV_8UC2, 1, 1, CV_8UC3},
        {"BGR2NV24_U8", "yuv444_nv24", cvh::COLOR_BGR2YUV_NV24, CV_8UC3, 1, 1, CV_8UC1},
        {"NV24_TO_BGR_U8", "yuv444_nv24", cvh::COLOR_YUV2BGR_NV24, CV_8UC1, 3, 1, CV_8UC3},
        {"BGR2I444_U8", "yuv444_i444", cvh::COLOR_BGR2YUV_I444, CV_8UC3, 1, 1, CV_8UC1},
        {"I444_TO_BGR_U8", "yuv444_i444", cvh::COLOR_YUV2BGR_I444, CV_8UC1, 3, 1, CV_8UC3},
    };

    for (const auto& color_case : cases)
    {
        const int src_rows = shape.rows * color_case.src_rows_numerator / color_case.src_rows_denominator;
        cvh::Mat src({src_rows, shape.cols}, color_case.src_type);
        fill_input(src, static_cast<std::uint32_t>(shape.rows * 73 + shape.cols * 79 + color_case.code * 19));
        cvh::Mat dst;
        const auto result = measure(
            [&]() { cvh::cvtColor(src, dst, color_case.code); },
            dst,
            args);
        auto row = make_row(
            args,
            shape,
            "CVTCOLOR",
            color_case.variant,
            color_case.dst_type,
            shape.rows,
            shape.cols,
            logical_bytes(src),
            result);
        row.layout = color_case.layout;
        row.dispatch_path = "header_fastpath";
        rows.push_back(row);
    }
}

void init_filter_kernels(cvh::Mat& kernel, cvh::Mat& kernel_x, cvh::Mat& kernel_y)
{
    kernel.create(std::vector<int>{3, 3}, CV_32FC1);
    float* values = reinterpret_cast<float*>(kernel.data);
    const float kernel_values[9] = {0.0f, 0.25f, 0.0f, 0.25f, 0.0f, 0.25f, 0.0f, 0.25f, 0.0f};
    std::copy(kernel_values, kernel_values + 9, values);

    kernel_x.create(std::vector<int>{1, 3}, CV_32FC1);
    kernel_y.create(std::vector<int>{3, 1}, CV_32FC1);
    float* kx = reinterpret_cast<float*>(kernel_x.data);
    float* ky = reinterpret_cast<float*>(kernel_y.data);
    kx[0] = ky[0] = 0.25f;
    kx[1] = ky[1] = 0.5f;
    kx[2] = ky[2] = 0.25f;
}

cvh::Mat make_roi_input(cvh::Mat& owner, const ShapeCase& shape, int type, std::uint32_t seed)
{
    owner.create(std::vector<int>{shape.rows + 2, shape.cols + 3}, type);
    fill_input(owner, seed);
    return owner(cvh::Range(1, shape.rows + 1), cvh::Range(1, shape.cols + 1));
}

void append_roi_rows(const Args& args, const ShapeCase& shape, std::vector<ResultRow>& rows)
{
    cvh::Mat u8_owner;
    cvh::Mat u8c3 = make_roi_input(u8_owner, shape, CV_8UC3, 0x13579u);
    cvh::Mat f32_owner;
    cvh::Mat f32c3 = make_roi_input(f32_owner, shape, CV_32FC3, 0x24680u);
    cvh::Mat kernel;
    cvh::Mat kernel_x;
    cvh::Mat kernel_y;
    init_filter_kernels(kernel, kernel_x, kernel_y);

    {
        cvh::Mat dst;
        const int out_rows = shape.rows * 3 / 4;
        const int out_cols = shape.cols * 3 / 4;
        const auto result = measure(
            [&]() { cvh::resize(u8c3, dst, cvh::Size(out_cols, out_rows), 0.0, 0.0, cvh::INTER_LINEAR); },
            dst,
            args);
        auto row = make_row(args, shape, "RESIZE", "INTER_LINEAR_0.75_U8C3_ROI", CV_8UC3, out_rows, out_cols, logical_bytes(u8c3), result);
        row.layout = "roi";
        row.dispatch_path = "header_fastpath";
        rows.push_back(row);
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() { cvh::cvtColor(u8c3, dst, cvh::COLOR_BGR2GRAY); },
            dst,
            args);
        auto row = make_row(args, shape, "CVTCOLOR", "BGR2GRAY_U8_ROI", CV_8UC1, shape.rows, shape.cols, logical_bytes(u8c3), result);
        row.layout = "roi";
        row.dispatch_path = "opencv_ui";
        rows.push_back(row);
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() { (void)cvh::threshold(f32c3, dst, 0.5, 1.0, cvh::THRESH_BINARY); },
            dst,
            args);
        auto row = make_row(args, shape, "THRESHOLD", "BINARY_F32C3_ROI", CV_32FC3, shape.rows, shape.cols, logical_bytes(f32c3), result);
        row.layout = "roi";
        row.dispatch_path = "header_fastpath";
        rows.push_back(row);
    }

    cvh::Mat box_dst;
    const auto box_result = measure(
        [&]() { cvh::boxFilter(u8c3, box_dst, -1, cvh::Size(3, 3), cvh::Point(-1, -1), true, cvh::BORDER_REPLICATE); },
        box_dst,
        args);
    auto box_row = make_row(args, shape, "BOXFILTER", "3x3_U8C3_ROI", CV_8UC3, shape.rows, shape.cols, logical_bytes(u8c3), box_result);
    box_row.layout = "roi";
    box_row.dispatch_path = cvh::detail::last_boxfilter_dispatch_path();
    rows.push_back(box_row);

    cvh::Mat gaussian_dst;
    const auto gaussian_result = measure(
        [&]() { cvh::GaussianBlur(u8c3, gaussian_dst, cvh::Size(5, 5), 0.0, 0.0, cvh::BORDER_REPLICATE); },
        gaussian_dst,
        args);
    auto gaussian_row = make_row(args, shape, "GAUSSIANBLUR", "5x5_U8C3_ROI", CV_8UC3, shape.rows, shape.cols, logical_bytes(u8c3), gaussian_result);
    gaussian_row.layout = "roi";
    gaussian_row.dispatch_path = cvh::detail::last_gaussianblur_dispatch_path();
    rows.push_back(gaussian_row);

    cvh::Mat filter_dst;
    const auto filter_result = measure(
        [&]() { cvh::filter2D(u8c3, filter_dst, -1, kernel, cvh::Point(-1, -1), 0.0, cvh::BORDER_REPLICATE); },
        filter_dst,
        args);
    auto filter_row = make_row(args, shape, "FILTER2D", "3x3_U8C3_ROI", CV_8UC3, shape.rows, shape.cols, logical_bytes(u8c3), filter_result);
    filter_row.layout = "roi";
    filter_row.dispatch_path = "header_fastpath";
    rows.push_back(filter_row);

    cvh::Mat sep_dst;
    const auto sep_result = measure(
        [&]() { cvh::sepFilter2D(u8c3, sep_dst, -1, kernel_x, kernel_y, cvh::Point(-1, -1), 0.0, cvh::BORDER_REPLICATE); },
        sep_dst,
        args);
    auto sep_row = make_row(args, shape, "SEPFILTER2D", "3x3_U8C3_ROI", CV_8UC3, shape.rows, shape.cols, logical_bytes(u8c3), sep_result);
    sep_row.layout = "roi";
    sep_row.dispatch_path = "header_fastpath";
    rows.push_back(sep_row);
}

void append_recreate_rows(const Args& args, const ShapeCase& shape, std::vector<ResultRow>& rows)
{
    cvh::Mat gray({shape.rows, shape.cols}, CV_8UC1);
    cvh::Mat bgr({shape.rows, shape.cols}, CV_8UC3);
    fill_input(gray, 0x10203u);
    fill_input(bgr, 0x40506u);
    cvh::Mat kernel;
    cvh::Mat kernel_x;
    cvh::Mat kernel_y;
    init_filter_kernels(kernel, kernel_x, kernel_y);

    {
        cvh::Mat dst;
        const int out_rows = shape.rows / 2;
        const int out_cols = shape.cols / 2;
        const auto result = measure(
            [&]() {
                dst.release();
                cvh::resize(gray, dst, cvh::Size(out_cols, out_rows), 0.0, 0.0, cvh::INTER_LINEAR);
            },
            dst,
            args);
        auto row = make_row(args, shape, "RESIZE", "INTER_LINEAR_0.5_U8C1_RECREATE", CV_8UC1, out_rows, out_cols, logical_bytes(gray), result);
        row.allocation_mode = "recreate";
        row.dispatch_path = "opencv_ui";
        rows.push_back(row);
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() {
                dst.release();
                cvh::cvtColor(bgr, dst, cvh::COLOR_BGR2GRAY);
            },
            dst,
            args);
        auto row = make_row(args, shape, "CVTCOLOR", "BGR2GRAY_U8_RECREATE", CV_8UC1, shape.rows, shape.cols, logical_bytes(bgr), result);
        row.allocation_mode = "recreate";
        row.dispatch_path = "opencv_ui";
        rows.push_back(row);
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() {
                dst.release();
                cvh::GaussianBlur(gray, dst, cvh::Size(5, 5), 0.0, 0.0, cvh::BORDER_REPLICATE);
            },
            dst,
            args);
        auto row = make_row(args, shape, "GAUSSIANBLUR", "5x5_U8C1_RECREATE", CV_8UC1, shape.rows, shape.cols, logical_bytes(gray), result);
        row.allocation_mode = "recreate";
        row.dispatch_path = cvh::detail::last_gaussianblur_dispatch_path();
        rows.push_back(row);
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() {
                dst.release();
                cvh::filter2D(gray, dst, -1, kernel, cvh::Point(-1, -1), 0.0, cvh::BORDER_REPLICATE);
            },
            dst,
            args);
        auto row = make_row(args, shape, "FILTER2D", "3x3_U8C1_RECREATE", CV_8UC1, shape.rows, shape.cols, logical_bytes(gray), result);
        row.allocation_mode = "recreate";
        row.dispatch_path = "header_fastpath";
        rows.push_back(row);
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
        auto row = make_row(args, shape, "BOXFILTER", "3x3_NORMALIZED_U8C1", CV_8UC1, shape.rows, shape.cols, gray_bytes, result);
        row.dispatch_path = cvh::detail::last_boxfilter_dispatch_path();
        rows.push_back(row);
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() { cvh::blur(gray, dst, cvh::Size(3, 3), cvh::Point(-1, -1), cvh::BORDER_REPLICATE); },
            dst,
            args);
        auto row = make_row(args, shape, "BLUR", "3x3_U8C1", CV_8UC1, shape.rows, shape.cols, gray_bytes, result);
        row.dispatch_path = cvh::detail::last_boxfilter_dispatch_path();
        rows.push_back(row);
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() { cvh::GaussianBlur(gray, dst, cvh::Size(5, 5), 0.0, 0.0, cvh::BORDER_REPLICATE); },
            dst,
            args);
        auto row = make_row(args, shape, "GAUSSIANBLUR", "5x5_U8C1", CV_8UC1, shape.rows, shape.cols, gray_bytes, result);
        row.dispatch_path = cvh::detail::last_gaussianblur_dispatch_path();
        rows.push_back(row);
    }

    for (const int kernel_size : {3, 5})
    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() { cvh::medianBlur(gray, dst, kernel_size); },
            dst,
            args);
        auto row = make_row(
            args,
            shape,
            "MEDIAN_BLUR",
            std::to_string(kernel_size) + "x" +
                std::to_string(kernel_size) + "_U8C1",
            CV_8UC1,
            shape.rows,
            shape.cols,
            gray_bytes + logical_bytes(dst),
            result);
        row.dispatch_path = "scalar_baseline";
        row.note = "Not qualified as a fast path";
        rows.push_back(row);
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() {
                cvh::bilateralFilter(
                    gray,
                    dst,
                    5,
                    35.0,
                    2.0,
                    cvh::BORDER_REFLECT_101);
            },
            dst,
            args);
        auto row = make_row(
            args,
            shape,
            "BILATERAL_FILTER",
            "D5_SIGMA35_2_U8C1",
            CV_8UC1,
            shape.rows,
            shape.cols,
            gray_bytes + logical_bytes(dst),
            result);
        row.dispatch_path = "scalar_baseline";
        row.note = "Not qualified as a fast path";
        rows.push_back(row);
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() {
                cvh::adaptiveThreshold(
                    gray,
                    dst,
                    255.0,
                    cvh::ADAPTIVE_THRESH_MEAN_C,
                    cvh::THRESH_BINARY,
                    5,
                    2.0);
            },
            dst,
            args);
        auto row = make_row(
            args,
            shape,
            "ADAPTIVE_THRESHOLD",
            "MEAN_K5_BINARY_U8C1",
            CV_8UC1,
            shape.rows,
            shape.cols,
            gray_bytes + logical_bytes(dst),
            result);
        row.dispatch_path = "scalar_baseline";
        row.note = "Not qualified as a fast path";
        rows.push_back(row);
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() { cvh::equalizeHist(gray, dst); },
            dst,
            args);
        auto row = make_row(
            args,
            shape,
            "EQUALIZE_HIST",
            "U8C1",
            CV_8UC1,
            shape.rows,
            shape.cols,
            gray_bytes + logical_bytes(dst),
            result);
        row.dispatch_path = "scalar_baseline";
        row.note = "Not qualified as a fast path";
        rows.push_back(row);
    }

    {
        cvh::Mat accumulator({shape.rows, shape.cols}, CV_32FC1);
        accumulator.setTo(cvh::Scalar::all(0.0));
        const auto result = measure(
            [&]() {
                cvh::accumulateWeighted(
                    gray, accumulator, 0.125);
            },
            accumulator,
            args);
        auto row = make_row(
            args,
            shape,
            "ACCUMULATE_WEIGHTED",
            "ALPHA_0.125_U8C1_TO_F32",
            CV_32FC1,
            shape.rows,
            shape.cols,
            gray_bytes + logical_bytes(accumulator),
            result);
        row.dispatch_path = "scalar_baseline";
        row.note = "In-place accumulator update";
        rows.push_back(row);
    }

    {
        cvh::Mat weights1({shape.rows, shape.cols}, CV_32FC1);
        cvh::Mat weights2({shape.rows, shape.cols}, CV_32FC1);
        weights1.setTo(cvh::Scalar::all(0.35));
        weights2.setTo(cvh::Scalar::all(0.65));
        cvh::Mat dst;
        const auto result = measure(
            [&]() {
                cvh::blendLinear(
                    bgr, bgr, weights1, weights2, dst);
            },
            dst,
            args);
        auto row = make_row(
            args,
            shape,
            "BLEND_LINEAR",
            "U8C3_F32_WEIGHTS",
            CV_8UC3,
            shape.rows,
            shape.cols,
            bgr_bytes * 2 + logical_bytes(weights1) * 2,
            result);
        row.dispatch_path = "scalar_baseline";
        row.note = "Not qualified as a fast path";
        rows.push_back(row);
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() { cvh::pyrDown(gray, dst); },
            dst,
            args);
        auto row = make_row(
            args,
            shape,
            "PYR_DOWN",
            "GAUSSIAN5_U8C1",
            CV_8UC1,
            (shape.rows + 1) / 2,
            (shape.cols + 1) / 2,
            gray_bytes + logical_bytes(dst),
            result);
        row.dispatch_path = "scalar_baseline";
        row.note = "Not qualified as a fast path";
        rows.push_back(row);
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() { cvh::pyrUp(gray, dst); },
            dst,
            args);
        auto row = make_row(
            args,
            shape,
            "PYR_UP",
            "GAUSSIAN5_U8C1",
            CV_8UC1,
            shape.rows * 2,
            shape.cols * 2,
            gray_bytes + logical_bytes(dst),
            result);
        row.dispatch_path = "scalar_baseline";
        row.note = "Not qualified as a fast path";
        rows.push_back(row);
    }

    {
        cvh::Mat uv(
            {shape.rows / 2, shape.cols / 2}, CV_8UC2);
        common::fill_mat_u8_lcg(
            uv,
            static_cast<std::uint32_t>(
                shape.rows * 29 + shape.cols * 31));
        for (const auto code_and_name :
             {std::pair<int, const char*>(
                  cvh::COLOR_YUV2BGR_NV12, "NV12_TO_BGR"),
              std::pair<int, const char*>(
                  cvh::COLOR_YUV2RGB_NV21, "NV21_TO_RGB")})
        {
            cvh::Mat dst;
            const auto result = measure(
                [&]() {
                    cvh::cvtColorTwoPlane(
                        gray, uv, dst, code_and_name.first);
                },
                dst,
                args);
            auto row = make_row(
                args,
                shape,
                "CVTCOLOR_TWO_PLANE",
                code_and_name.second,
                CV_8UC3,
                shape.rows,
                shape.cols,
                gray_bytes + logical_bytes(uv) + logical_bytes(dst),
                result);
            row.dispatch_path = "scalar_baseline";
            row.note = "Separate Y and UV planes";
            rows.push_back(row);
        }
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() {
                cvh::demosaicing(
                    gray, dst, cvh::COLOR_BayerBG2BGR);
            },
            dst,
            args);
        auto row = make_row(
            args,
            shape,
            "DEMOSAICING",
            "BAYER_BG_TO_BGR_U8",
            CV_8UC3,
            shape.rows,
            shape.cols,
            gray_bytes + logical_bytes(dst),
            result);
        row.dispatch_path = "scalar_baseline";
        row.note = "Bilinear Bayer decode";
        rows.push_back(row);
    }

    {
        cvh::Mat map_x({shape.rows, shape.cols}, CV_32FC1);
        cvh::Mat map_y({shape.rows, shape.cols}, CV_32FC1);
        for (int row_index = 0; row_index < shape.rows; ++row_index)
        {
            for (int col_index = 0; col_index < shape.cols; ++col_index)
            {
                map_x.at<float>(row_index, col_index) =
                    static_cast<float>(col_index) + 0.28125f;
                map_y.at<float>(row_index, col_index) =
                    static_cast<float>(row_index) - 0.34375f;
            }
        }
        cvh::Mat fixed_coordinates;
        cvh::Mat fixed_fractions;
        cvh::convertMaps(
            map_x,
            map_y,
            fixed_coordinates,
            fixed_fractions,
            CV_16SC2);
        for (const bool fixed : {false, true})
        {
            cvh::Mat dst;
            const auto result = measure(
                [&]() {
                    cvh::remap(
                        bgr,
                        dst,
                        fixed ? fixed_coordinates : map_x,
                        fixed ? fixed_fractions : map_y,
                        cvh::INTER_LINEAR,
                        cvh::BORDER_REFLECT_101);
                },
                dst,
                args);
            auto row = make_row(
                args,
                shape,
                "REMAP",
                fixed ? "FIXED_LINEAR_U8C3" : "FLOAT_LINEAR_U8C3",
                CV_8UC3,
                shape.rows,
                shape.cols,
                bgr_bytes +
                    (fixed
                         ? logical_bytes(fixed_coordinates) +
                               logical_bytes(fixed_fractions)
                         : logical_bytes(map_x) + logical_bytes(map_y)),
                result);
            row.dispatch_path = "public_header_scalar";
            row.note = "No qualified SIMD fast path";
            rows.push_back(row);
        }
    }

    {
        cvh::Mat matrix({3, 3}, CV_64FC1);
        matrix.setTo(cvh::Scalar::all(0.0));
        matrix.at<double>(0, 0) = 1.0;
        matrix.at<double>(0, 1) = 0.01;
        matrix.at<double>(0, 2) = 0.25;
        matrix.at<double>(1, 0) = -0.005;
        matrix.at<double>(1, 1) = 1.0;
        matrix.at<double>(1, 2) = 0.5;
        matrix.at<double>(2, 0) = 0.00002;
        matrix.at<double>(2, 1) = -0.00003;
        matrix.at<double>(2, 2) = 1.0;
        cvh::Mat dst;
        const auto result = measure(
            [&]() {
                cvh::warpPerspective(
                    bgr,
                    dst,
                    matrix,
                    cvh::Size(shape.cols, shape.rows),
                    cvh::INTER_LINEAR | cvh::WARP_INVERSE_MAP,
                    cvh::BORDER_REFLECT_101);
            },
            dst,
            args);
        auto row = make_row(
            args,
            shape,
            "WARP_PERSPECTIVE",
            "PROJECTIVE_LINEAR_U8C3",
            CV_8UC3,
            shape.rows,
            shape.cols,
            bgr_bytes + logical_bytes(dst),
            result);
        row.dispatch_path = "public_header_scalar";
        row.note = "No qualified SIMD fast path";
        rows.push_back(row);
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() {
                cvh::getRectSubPix(
                    bgr,
                    cvh::Size(shape.cols, shape.rows),
                    cvh::Point2f(
                        (static_cast<float>(shape.cols) - 1.0f) * 0.5f,
                        (static_cast<float>(shape.rows) - 1.0f) * 0.5f),
                    dst);
            },
            dst,
            args);
        auto row = make_row(
            args,
            shape,
            "GET_RECT_SUB_PIX",
            "FULL_FRAME_U8C3",
            CV_8UC3,
            shape.rows,
            shape.cols,
            bgr_bytes + logical_bytes(dst),
            result);
        row.dispatch_path = "public_header_scalar";
        row.note = "No qualified SIMD fast path";
        rows.push_back(row);
    }

    {
        cvh::Mat kernel({3, 3}, CV_32FC1);
        float* values = reinterpret_cast<float*>(kernel.data);
        const float kernel_values[9] = {0.0f, 0.25f, 0.0f, 0.25f, 0.0f, 0.25f, 0.0f, 0.25f, 0.0f};
        std::copy(kernel_values, kernel_values + 9, values);
        cvh::Mat dst;
        const auto result = measure(
            [&]() { cvh::filter2D(gray, dst, -1, kernel, cvh::Point(-1, -1), 0.0, cvh::BORDER_REPLICATE); },
            dst,
            args);
        auto row = make_row(args, shape, "FILTER2D", "3x3_U8C1", CV_8UC1, shape.rows, shape.cols, gray_bytes, result);
        row.dispatch_path = "header_fastpath";
        rows.push_back(row);
    }

    {
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
        cvh::Mat dst;
        const auto result = measure(
            [&]() { cvh::sepFilter2D(gray, dst, -1, kernel_x, kernel_y, cvh::Point(-1, -1), 0.0, cvh::BORDER_REPLICATE); },
            dst,
            args);
        auto row = make_row(args, shape, "SEPFILTER2D", "3x3_U8C1", CV_8UC1, shape.rows, shape.cols, gray_bytes, result);
        row.dispatch_path = "header_fastpath";
        rows.push_back(row);
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
            [&]() { cvh::integral(gray, dst, CV_32S); },
            dst,
            args);
        rows.push_back(make_row(
            args,
            shape,
            "INTEGRAL",
            "U8C1_TO_S32",
            CV_32SC1,
            shape.rows + 1,
            shape.cols + 1,
            gray_bytes + logical_bytes(dst),
            result));
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() {
                cvh::Scharr(
                    gray,
                    dst,
                    CV_16S,
                    1,
                    0,
                    1.0,
                    0.0,
                    cvh::BORDER_REPLICATE);
            },
            dst,
            args);
        rows.push_back(make_row(
            args,
            shape,
            "SCHARR",
            "DX1_U8C1_TO_S16",
            CV_16SC1,
            shape.rows,
            shape.cols,
            gray_bytes + logical_bytes(dst),
            result));
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() {
                cvh::Laplacian(
                    gray,
                    dst,
                    CV_16S,
                    3,
                    1.0,
                    0.0,
                    cvh::BORDER_REPLICATE);
            },
            dst,
            args);
        rows.push_back(make_row(
            args,
            shape,
            "LAPLACIAN",
            "K3_U8C1_TO_S16",
            CV_16SC1,
            shape.rows,
            shape.cols,
            gray_bytes + logical_bytes(dst),
            result));
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() {
                cvh::sqrBoxFilter(
                    gray,
                    dst,
                    CV_64F,
                    cvh::Size(3, 3),
                    cvh::Point(-1, -1),
                    true,
                    cvh::BORDER_REPLICATE);
            },
            dst,
            args);
        rows.push_back(make_row(
            args,
            shape,
            "SQR_BOX_FILTER",
            "3x3_U8C1_TO_F64",
            CV_64FC1,
            shape.rows,
            shape.cols,
            gray_bytes + logical_bytes(dst),
            result));
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

    if (args.profile != "quick" && is_matrix_anchor(shape))
    {
        append_resize_matrix_rows(args, shape, rows);
        append_color_matrix_rows(args, shape, rows);
        append_threshold_filter_matrix_rows(args, shape, rows);
        if (args.profile == "full")
        {
            append_yuv_matrix_rows(args, shape, rows);
            append_recreate_rows(args, shape, rows);
        }
    }
    if (args.profile == "full" && shape.rows == 479 && shape.cols == 641)
    {
        append_roi_rows(args, shape, rows);
    }
}

}  // namespace cvh_bench

int main(int argc, char** argv)
{
    const auto args = cvh_bench::parse_args(argc, argv);
    const auto shapes = cvh_bench::build_shapes(args.profile);
    std::vector<cvh_bench::ResultRow> rows;
    rows.reserve(shapes.size() * 32 + 96);

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
