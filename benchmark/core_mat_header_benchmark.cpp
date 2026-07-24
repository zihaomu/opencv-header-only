#include "cvh.h"
#include "common/benchmark_common.h"

#include <cmath>
#include <cstdint>
#include <cstring>
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
    std::string implementation = "cvh_headers_fast";
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

template<typename RunFn>
void append_array_op_row(const Args& args,
                         const ShapeCase& shape,
                         const std::string& op,
                         const std::string& variant,
                         std::size_t bytes_touched,
                         cvh::Mat& dst,
                         RunFn&& run,
                         std::vector<ResultRow>& rows)
{
    const auto result = measure(
        std::forward<RunFn>(run),
        [&]() { return common::checksum_mat_bytes(dst); },
        args);
    rows.push_back(make_row(
        args,
        shape,
        op,
        variant,
        "continuous",
        "reuse",
        bytes_touched,
        result));
}

std::uint64_t checksum_doubles(const double* values, std::size_t count)
{
    std::uint64_t hash = common::fnv1a64_basis();
    for (std::size_t i = 0; i < count; ++i)
    {
        std::uint64_t bits = 0;
        std::memcpy(&bits, values + i, sizeof(bits));
        hash = common::fnv1a64_mix_u64(hash, bits);
    }
    return hash;
}

void append_measured_row(const Args& args,
                         const ShapeCase& shape,
                         const std::string& op,
                         const std::string& variant,
                         const std::string& allocation_mode,
                         std::size_t bytes_touched,
                         int threads,
                         const std::string& note,
                         const Result& result,
                         std::vector<ResultRow>& rows)
{
    ResultRow row = make_row(
        args,
        shape,
        op,
        variant,
        "continuous",
        allocation_mode,
        bytes_touched,
        result);
    row.threads = threads;
    row.note = note;
    rows.push_back(std::move(row));
}

void append_reduction_rows(const Args& args,
                           const ShapeCase& shape,
                           const cvh::Mat& src,
                           std::size_t bytes,
                           std::vector<ResultRow>& rows)
{
    const int saved_threads = cvh::getNumThreads();
    const int thread_counts[] = {1, saved_threads};
    const char* thread_labels[] = {"threads_1", "project_default"};
    for (int mode = 0; mode < 2; ++mode)
    {
        const int threads = thread_counts[mode];
        cvh::setNumThreads(threads);
        const std::string suffix = thread_labels[mode];
        const std::string note =
            "deterministic_header_loop;configured_threads=" +
            std::to_string(threads);

        {
            cvh::Scalar value;
            const auto result = measure(
                [&]() { value = cvh::sum(src); },
                [&]() { return checksum_doubles(value.val, 4); },
                args);
            append_measured_row(
                args,
                shape,
                "SUM",
                "all_channels_" + suffix,
                "none",
                bytes,
                threads,
                note,
                result,
                rows);
        }

        {
            cvh::Scalar mean_value;
            cvh::Scalar stddev_value;
            const auto result = measure(
                [&]() {
                    cvh::meanStdDev(src, mean_value, stddev_value);
                },
                [&]() {
                    std::uint64_t hash = checksum_doubles(mean_value.val, 4);
                    return common::fnv1a64_mix_u64(
                        hash, checksum_doubles(stddev_value.val, 4));
                },
                args);
            append_measured_row(
                args,
                shape,
                "MEAN_STDDEV",
                "all_channels_" + suffix,
                "none",
                bytes,
                threads,
                note,
                result,
                rows);
        }

        {
            double value = 0.0;
            const auto result = measure(
                [&]() { value = cvh::norm(src, cvh::NORM_L2); },
                [&]() { return checksum_doubles(&value, 1); },
                args);
            append_measured_row(
                args,
                shape,
                "NORM",
                "l2_" + suffix,
                "none",
                bytes,
                threads,
                note,
                result,
                rows);
        }

        if (src.channels() == 1)
        {
            double min_value = 0.0;
            double max_value = 0.0;
            cvh::Point min_location;
            cvh::Point max_location;
            const auto result = measure(
                [&]() {
                    cvh::minMaxLoc(
                        src,
                        &min_value,
                        &max_value,
                        &min_location,
                        &max_location);
                },
                [&]() {
                    const double values[] = {
                        min_value,
                        max_value,
                        static_cast<double>(min_location.x),
                        static_cast<double>(min_location.y),
                        static_cast<double>(max_location.x),
                        static_cast<double>(max_location.y),
                    };
                    return checksum_doubles(values, 6);
                },
                args);
            append_measured_row(
                args,
                shape,
                "MIN_MAX_LOC",
                "first_tie_" + suffix,
                "none",
                bytes,
                threads,
                note,
                result,
                rows);
        }

        {
            cvh::Mat reduced;
            const auto result = measure(
                [&]() {
                    cvh::reduce(src, reduced, 0, cvh::REDUCE_SUM, CV_64F);
                },
                [&]() { return common::checksum_mat_bytes(reduced); },
                args);
            const std::size_t output_bytes =
                static_cast<std::size_t>(shape.cols) *
                static_cast<std::size_t>(src.channels()) * sizeof(double);
            append_measured_row(
                args,
                shape,
                "REDUCE",
                "axis_0_sum_f64_" + suffix,
                "reuse",
                bytes + output_bytes,
                threads,
                note,
                result,
                rows);
        }

        {
            cvh::Mat normalized;
            const auto result = measure(
                [&]() {
                    cvh::normalize(
                        src,
                        normalized,
                        1.0,
                        0.0,
                        cvh::NORM_L2);
                },
                [&]() { return common::checksum_mat_bytes(normalized); },
                args);
            append_measured_row(
                args,
                shape,
                "NORMALIZE",
                "l2_" + suffix,
                "reuse",
                bytes * 2,
                threads,
                note,
                result,
                rows);
        }
    }
    cvh::setNumThreads(saved_threads);
}

void append_layout_rows(const Args& args,
                        const ShapeCase& shape,
                        const cvh::Mat& src,
                        std::size_t bytes,
                        std::vector<ResultRow>& rows)
{
    const std::vector<int> dims {shape.rows, shape.cols};
    {
        cvh::Mat mask(dims, CV_8UC1);
        for (int y = 0; y < shape.rows; ++y)
        {
            uchar* mask_row =
                mask.data + static_cast<std::size_t>(y) * mask.step(0);
            for (int x = 0; x < shape.cols; ++x)
            {
                mask_row[x] = ((x + y) & 1) != 0 ? 255 : 0;
            }
        }
        cvh::Mat dst(dims, shape.type);
        append_array_op_row(
            args,
            shape,
            "COPY_TO",
            "partial_mask",
            bytes * 2 + mask.total(),
            dst,
            [&]() { cvh::copyTo(src, dst, mask); },
            rows);
    }

    {
        cvh::Mat dst(dims, shape.type);
        std::vector<int> routes;
        routes.reserve(static_cast<std::size_t>(src.channels()) * 2);
        for (int ch = 0; ch < src.channels(); ++ch)
        {
            routes.push_back(ch);
            routes.push_back(src.channels() - 1 - ch);
        }
        append_array_op_row(
            args,
            shape,
            "MIX_CHANNELS",
            "reverse_channels",
            bytes * 2,
            dst,
            [&]() {
                cvh::mixChannels(
                    &src,
                    1,
                    &dst,
                    1,
                    routes.data(),
                    routes.size() / 2);
            },
            rows);
    }

    {
        cvh::Mat dst;
        append_array_op_row(
            args,
            shape,
            "FLIP",
            "horizontal",
            bytes * 2,
            dst,
            [&]() { cvh::flip(src, dst, 1); },
            rows);
    }

    {
        cvh::Mat dst;
        append_array_op_row(
            args,
            shape,
            "HCONCAT",
            "two_equal_inputs",
            bytes * 3,
            dst,
            [&]() { cvh::hconcat(src, src, dst); },
            rows);
    }

    {
        cvh::Mat dst;
        append_array_op_row(
            args,
            shape,
            "VCONCAT",
            "two_equal_inputs",
            bytes * 3,
            dst,
            [&]() { cvh::vconcat(src, src, dst); },
            rows);
    }

    {
        cvh::Mat dst;
        append_array_op_row(
            args,
            shape,
            "BROADCAST",
            "prepend_extent_2",
            bytes * 3,
            dst,
            [&]() {
                cvh::broadcast(
                    src,
                    std::vector<int>({2, shape.rows, shape.cols}),
                    dst);
            },
            rows);
    }
}

void print_csv(const std::vector<ResultRow>& rows, std::ostream& os)
{
    common::write_csv_row(
        os,
        {
            "schema_version", "mode", "suite", "module", "op", "variant", "depth", "channels", "layout",
            "shape", "elements", "pixels", "implementation", "dispatch_path", "allocation_mode", "warmup",
            "iters", "repeats", "threads", "min_ms", "median_ms", "mpix_per_sec", "melems_per_sec",
            "gb_per_sec", "checksum", "status", "note",
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

    cvh::Mat rhs(dims, shape.type);
    fill_mat(rhs, static_cast<std::uint32_t>(shape.rows * 211 + shape.cols * 29 + shape.type));

    {
        cvh::Mat dst(dims, shape.type);
        append_array_op_row(
            args,
            shape,
            "ABSDIFF",
            "mat_mat",
            bytes * 3,
            dst,
            [&]() { cvh::absdiff(src, rhs, dst); },
            rows);
    }

    {
        cvh::Mat dst(dims, shape.type);
        append_array_op_row(
            args,
            shape,
            "BITWISE_AND",
            "mat_mat_raw_bits",
            bytes * 3,
            dst,
            [&]() { cvh::bitwise_and(src, rhs, dst); },
            rows);
    }

    {
        cvh::Mat dst(dims, CV_8UC1);
        const std::size_t mask_bytes =
            static_cast<std::size_t>(shape.rows) * static_cast<std::size_t>(shape.cols);
        append_array_op_row(
            args,
            shape,
            "IN_RANGE",
            "scalar_bounds",
            bytes + mask_bytes,
            dst,
            [&]() {
                cvh::inRange(src, cvh::Scalar::all(-2.5), cvh::Scalar::all(127.5), dst);
            },
            rows);
    }

    {
        cvh::Mat dst(dims, shape.type);
        append_array_op_row(
            args,
            shape,
            "MIN",
            "mat_mat",
            bytes * 3,
            dst,
            [&]() { cvh::min(src, rhs, dst); },
            rows);
    }

    {
        cvh::Mat dst(dims, shape.type);
        append_array_op_row(
            args,
            shape,
            "MAX",
            "mat_mat",
            bytes * 3,
            dst,
            [&]() { cvh::max(src, rhs, dst); },
            rows);
    }

    {
        cvh::Mat dst(dims, CV_MAKETYPE(CV_8U, CV_MAT_CN(shape.type)));
        const std::size_t output_bytes =
            static_cast<std::size_t>(shape.rows) * static_cast<std::size_t>(shape.cols) *
            static_cast<std::size_t>(CV_MAT_CN(shape.type));
        append_array_op_row(
            args,
            shape,
            "CONVERT_SCALE_ABS",
            "alpha_1_25_beta_3",
            bytes + output_bytes,
            dst,
            [&]() { cvh::convertScaleAbs(src, dst, 1.25, 3.0); },
            rows);
    }

    if (CV_MAT_DEPTH(shape.type) == CV_32F)
    {
        cvh::Mat positive_src = src.clone();
        float* values = reinterpret_cast<float*>(positive_src.data);
        const size_t scalar_count =
            positive_src.total() * static_cast<size_t>(positive_src.channels());
        for (size_t i = 0; i < scalar_count; ++i)
        {
            values[i] = std::fabs(values[i]) + 0.01f;
        }

        {
            cvh::Mat dst(dims, shape.type);
            append_array_op_row(
                args,
                shape,
                "SQRT",
                "positive_f32",
                bytes * 2,
                dst,
                [&]() { cvh::sqrt(positive_src, dst); },
                rows);
        }

        {
            cvh::Mat dst(dims, shape.type);
            append_array_op_row(
                args,
                shape,
                "EXP",
                "f32",
                bytes * 2,
                dst,
                [&]() { cvh::exp(src, dst); },
                rows);
        }

        {
            cvh::Mat dst(dims, shape.type);
            append_array_op_row(
                args,
                shape,
                "LOG",
                "positive_f32",
                bytes * 2,
                dst,
                [&]() { cvh::log(positive_src, dst); },
            rows);
        }
    }

    append_reduction_rows(args, shape, src, bytes, rows);
    append_layout_rows(args, shape, src, bytes, rows);
}

}  // namespace cvh_bench

int main(int argc, char** argv)
{
    const auto args = cvh_bench::parse_args(argc, argv);
    const auto shapes = cvh_bench::build_shapes(args.profile);
    std::vector<cvh_bench::ResultRow> rows;
    rows.reserve(shapes.size() * 36);

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
