#include "cvh.h"

#if !CVH_ENABLE_OPENCV_INTRIN
#error "cvh_benchmark_cvtcolor_bgr2gray_header must be compiled with CVH_ENABLE_OPENCV_INTRIN=1"
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace cvh_bench {

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
};

struct Result
{
    double min_ms = 0.0;
    double median_ms = 0.0;
    std::uint64_t checksum = 0;
};

struct ResultRow
{
    std::string profile;
    std::string backend;
    ShapeCase shape;
    std::size_t pixels = 0;
    int warmup = 0;
    int iters = 0;
    int repeats = 0;
    double min_ms = 0.0;
    double median_ms = 0.0;
    double mpix_per_sec = 0.0;
    double speedup_vs_scalar = 1.0;
    std::uint64_t checksum = 0;
};

using BenchFn = void (*)(const cvh::Mat&, cvh::Mat&);

volatile std::uint64_t g_sink = 0;

void usage()
{
    std::cout
        << "Usage: cvh_benchmark_cvtcolor_bgr2gray_header "
        << "[--profile quick|full] [--warmup N] [--iters N] [--repeats N] [--output path]\n";
}

Args parse_args(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc; ++i)
    {
        const std::string token = argv[i];
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
            usage();
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

std::vector<ShapeCase> build_shapes(const std::string& profile)
{
    std::vector<ShapeCase> shapes = {
        {"640x480", 480, 640},
        {"1280x720", 720, 1280},
        {"1920x1080", 1080, 1920},
        {"3840x2160", 2160, 3840},
    };

    if (profile == "full")
    {
        shapes.insert(
            shapes.begin(),
            {
                {"320x240", 240, 320},
                {"641x479", 479, 641},
            });
        shapes.push_back({"4096x2160", 2160, 4096});
    }

    return shapes;
}

std::string opencv_intrin_backend_name()
{
#if defined(CV_FORCE_SIMD128_CPP)
    return "opencv_intrin_cpp";
#elif defined(CV_NEON) && CV_NEON
    return "opencv_intrin_neon";
#else
    return "opencv_intrin_cpp";
#endif
}

void fill_bgr(cvh::Mat& mat, std::uint32_t seed)
{
    const std::size_t count = mat.total() * static_cast<std::size_t>(mat.channels());
    for (std::size_t i = 0; i < count; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        mat.data[i] = static_cast<uchar>((seed >> 16) & 0xffu);
    }
}

std::uint64_t checksum(const cvh::Mat& mat)
{
    const std::size_t count = mat.total() * static_cast<std::size_t>(mat.channels()) *
                              static_cast<std::size_t>(mat.elemSize1());
    std::uint64_t value = 1469598103934665603ull;
    for (std::size_t i = 0; i < count; ++i)
    {
        value ^= static_cast<std::uint64_t>(mat.data[i]);
        value *= 1099511628211ull;
    }
    return value;
}

bool same_mat(const cvh::Mat& lhs, const cvh::Mat& rhs)
{
    if (lhs.type() != rhs.type() || lhs.dims != rhs.dims || lhs.size != rhs.size)
    {
        return false;
    }

    const std::size_t count = lhs.total() * static_cast<std::size_t>(lhs.channels()) *
                              static_cast<std::size_t>(lhs.elemSize1());
    for (std::size_t i = 0; i < count; ++i)
    {
        if (lhs.data[i] != rhs.data[i])
        {
            return false;
        }
    }
    return true;
}

void run_scalar(const cvh::Mat& src, cvh::Mat& dst)
{
    cvh::detail::cvtcolor_bgr2gray_u8_scalar_impl(src, dst);
}

void run_opencv_intrin(const cvh::Mat& src, cvh::Mat& dst)
{
    cvh::cvtColor(src, dst, cvh::COLOR_BGR2GRAY);
}

Result measure(BenchFn fn, const cvh::Mat& src, cvh::Mat& dst, int warmup, int iters, int repeats)
{
    for (int i = 0; i < warmup; ++i)
    {
        fn(src, dst);
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));

    using Clock = std::chrono::steady_clock;
    for (int r = 0; r < repeats; ++r)
    {
        const auto t0 = Clock::now();
        for (int i = 0; i < iters; ++i)
        {
            fn(src, dst);
        }
        const auto t1 = Clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        samples.push_back(elapsed_ms / static_cast<double>(iters));
    }

    std::sort(samples.begin(), samples.end());
    const std::uint64_t hash = checksum(dst);
    g_sink ^= hash;

    return Result {samples.front(), samples[samples.size() / 2], hash};
}

ResultRow make_row(const Args& args,
                   const std::string& backend,
                   const ShapeCase& shape,
                   const Result& result,
                   double scalar_median_ms)
{
    const std::size_t pixels = static_cast<std::size_t>(shape.rows) * static_cast<std::size_t>(shape.cols);
    return ResultRow {
        args.profile,
        backend,
        shape,
        pixels,
        args.warmup,
        args.iters,
        args.repeats,
        result.min_ms,
        result.median_ms,
        static_cast<double>(pixels) / result.median_ms / 1000.0,
        scalar_median_ms / result.median_ms,
        result.checksum,
    };
}

void print_csv(const std::vector<ResultRow>& rows, std::ostream& os)
{
    os << "profile,op,backend,shape,width,height,pixels,warmup,iters,repeats,"
       << "min_ms_per_call,median_ms_per_call,mpix_per_sec,speedup_vs_scalar,checksum\n";
    os << std::fixed << std::setprecision(6);
    for (const auto& row : rows)
    {
        os << row.profile << ","
           << "CVTCOLOR_BGR2GRAY_U8" << ","
           << row.backend << ","
           << row.shape.name << ","
           << row.shape.cols << ","
           << row.shape.rows << ","
           << row.pixels << ","
           << row.warmup << ","
           << row.iters << ","
           << row.repeats << ","
           << row.min_ms << ","
           << row.median_ms << ","
           << row.mpix_per_sec << ","
           << row.speedup_vs_scalar << ","
           << row.checksum << "\n";
    }
}

}  // namespace cvh_bench

int main(int argc, char** argv)
{
    const auto args = cvh_bench::parse_args(argc, argv);
    const auto shapes = cvh_bench::build_shapes(args.profile);
    std::vector<cvh_bench::ResultRow> rows;
    rows.reserve(shapes.size() * 2);

    for (const auto& shape : shapes)
    {
        cvh::Mat src({shape.rows, shape.cols}, CV_8UC3);
        cvh_bench::fill_bgr(src, static_cast<std::uint32_t>(shape.rows * 131 + shape.cols * 17));

        cvh::Mat scalar_check;
        cvh::Mat opencv_check;
        cvh_bench::run_scalar(src, scalar_check);
        cvh_bench::run_opencv_intrin(src, opencv_check);
        if (!cvh_bench::same_mat(scalar_check, opencv_check))
        {
            std::cerr << "Correctness mismatch for shape " << shape.name << "\n";
            return 3;
        }

        cvh::Mat scalar_dst;
        cvh::Mat opencv_dst;
        const auto scalar = cvh_bench::measure(
            cvh_bench::run_scalar, src, scalar_dst, args.warmup, args.iters, args.repeats);
        const auto opencv_intrin = cvh_bench::measure(
            cvh_bench::run_opencv_intrin, src, opencv_dst, args.warmup, args.iters, args.repeats);

        rows.push_back(cvh_bench::make_row(args, "scalar", shape, scalar, scalar.median_ms));
        rows.push_back(cvh_bench::make_row(
            args,
            cvh_bench::opencv_intrin_backend_name(),
            shape,
            opencv_intrin,
            scalar.median_ms));
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
