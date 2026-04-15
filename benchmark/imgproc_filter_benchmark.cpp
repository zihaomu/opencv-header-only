#include "cvh.h"

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

enum class FilterOp
{
    BoxFilter = 0,
    GaussianBlur,
};

enum class DispatchMode
{
    Auto = 0,
    OptimizedOnly,
    FallbackOnly,
};

struct Args
{
    std::string profile = "quick";
    std::string dispatch_mode = "auto";
    std::string parallel_backend = "auto";
    int threads = 0;
    int warmup = 2;
    int iters = 8;
    int repeats = 3;
    std::string output_csv;
};

struct InputCase
{
    std::string shape;
    int rows = 0;
    int cols = 0;
    bool roi_layout = false;
};

struct KernelCase
{
    FilterOp op = FilterOp::BoxFilter;
    std::string kernel;
    cvh::Size ksize;
    double sigma_x = 0.0;
    double sigma_y = 0.0;
};

struct BorderCase
{
    std::string name;
    int border_type = cvh::BORDER_DEFAULT;
};

struct ResultRow
{
    std::string op;
    std::string kernel;
    std::string depth;
    int channels = 0;
    std::string shape;
    std::string layout;
    std::string border;
    std::string dispatch_path;
    std::string parallel_backend;
    int parallel_chunks = 1;
    double ms_per_iter = 0.0;
};

volatile double g_sink = 0.0;

const char* op_name(FilterOp op)
{
    switch (op)
    {
        case FilterOp::BoxFilter: return "boxFilter";
        case FilterOp::GaussianBlur: return "GaussianBlur";
    }
    return "unknown";
}

std::string dispatch_path_for(FilterOp op)
{
    if (op == FilterOp::BoxFilter)
    {
        return cvh::detail::last_boxfilter_dispatch_path();
    }
    return cvh::detail::last_gaussianblur_dispatch_path();
}

DispatchMode parse_dispatch_mode(const std::string& mode)
{
    if (mode == "auto")
    {
        return DispatchMode::Auto;
    }
    if (mode == "optimized-only")
    {
        return DispatchMode::OptimizedOnly;
    }
    if (mode == "fallback-only")
    {
        return DispatchMode::FallbackOnly;
    }

    std::cerr << "Unsupported dispatch mode: " << mode
              << " (expected auto/optimized-only/fallback-only)\n";
    std::exit(2);
}

cvh::ParallelBackend parse_parallel_backend(const std::string& mode)
{
    if (mode == "auto")
    {
        return cvh::ParallelBackend::Auto;
    }
    if (mode == "stdthread")
    {
        return cvh::ParallelBackend::StdThread;
    }
    if (mode == "openmp")
    {
        return cvh::ParallelBackend::OpenMP;
    }
    if (mode == "serial")
    {
        return cvh::ParallelBackend::Serial;
    }

    std::cerr << "Unsupported parallel backend: " << mode
              << " (expected auto/stdthread/openmp/serial)\n";
    std::exit(2);
}

std::vector<InputCase> build_inputs(const std::string& profile)
{
    std::vector<InputCase> cases;
    cases.push_back({"720x1280", 720, 1280, false});
    cases.push_back({"513x769", 513, 769, true});
    if (profile == "full")
    {
        cases.push_back({"1080x1920", 1080, 1920, false});
    }
    return cases;
}

std::vector<int> build_channels()
{
    return {1, 3, 4};
}

std::vector<KernelCase> build_kernels(const std::string& profile)
{
    std::vector<KernelCase> cases = {
        {FilterOp::BoxFilter, "3x3", cvh::Size(3, 3), 0.0, 0.0},
        {FilterOp::GaussianBlur, "3x3", cvh::Size(3, 3), 0.0, 0.0},
    };

    if (profile == "full")
    {
        cases.push_back({FilterOp::GaussianBlur, "5x5", cvh::Size(5, 5), 0.0, 0.0});
        cases.push_back({FilterOp::BoxFilter, "5x5", cvh::Size(5, 5), 0.0, 0.0});
        cases.push_back({FilterOp::GaussianBlur, "7x7", cvh::Size(7, 7), 1.2, 1.2});
    }

    return cases;
}

std::vector<BorderCase> build_borders(const std::string& profile)
{
    std::vector<BorderCase> cases = {
        {"BORDER_REPLICATE", cvh::BORDER_REPLICATE},
        {"BORDER_REFLECT_101", cvh::BORDER_REFLECT_101},
    };

    if (profile == "full")
    {
        cases.push_back({"BORDER_CONSTANT", cvh::BORDER_CONSTANT});
    }

    return cases;
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
        else if (token == "--dispatch")
        {
            args.dispatch_mode = next_value("--dispatch");
        }
        else if (token == "--parallel-backend")
        {
            args.parallel_backend = next_value("--parallel-backend");
        }
        else if (token == "--threads")
        {
            args.threads = std::max(1, std::stoi(next_value("--threads")));
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
            std::cout << "Usage: cvh_benchmark_imgproc_filter [--profile quick|full] "
                         "[--dispatch auto|optimized-only|fallback-only] [--warmup N] [--iters N] "
                         "[--repeats N] [--parallel-backend auto|stdthread|openmp|serial] "
                         "[--threads N] [--output path]\n";
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

    (void)parse_dispatch_mode(args.dispatch_mode);
    (void)parse_parallel_backend(args.parallel_backend);

    return args;
}

void fill_u8(cvh::Mat& mat, std::uint32_t seed)
{
    if (mat.empty())
    {
        return;
    }

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
            state = state * 1664525u + 1013904223u;
            row[x] = static_cast<uchar>((state >> 24) ^ static_cast<std::uint32_t>(x * 13 + y * 7));
        }
    }
}

double probe_checksum(const cvh::Mat& mat)
{
    if (mat.empty())
    {
        return 0.0;
    }

    const int rows = mat.size[0];
    const int cols = mat.size[1];
    const int scalars_per_row = cols * mat.channels();
    const size_t step = mat.step(0);

    std::uint64_t sum = 0;
    std::uint64_t count = 0;
    for (int y = 0; y < rows; ++y)
    {
        const uchar* row = mat.data + static_cast<size_t>(y) * step;
        const int stride = std::max(1, scalars_per_row / 64);
        for (int x = 0; x < scalars_per_row; x += stride)
        {
            sum += static_cast<std::uint64_t>(row[x]) * static_cast<std::uint64_t>((x + 1) + (y + 1));
            ++count;
        }
    }
    return count > 0 ? static_cast<double>(sum) / static_cast<double>(count) : 0.0;
}

template <typename Fn>
double measure_case(Fn&& fn, const cvh::Mat& dst_probe, int warmup, int iters, int repeats)
{
    for (int i = 0; i < warmup; ++i)
    {
        fn();
    }

    std::vector<double> samples_ms_per_iter;
    samples_ms_per_iter.reserve(static_cast<size_t>(repeats));

    using Clock = std::chrono::steady_clock;
    for (int r = 0; r < repeats; ++r)
    {
        const auto t0 = Clock::now();
        for (int i = 0; i < iters; ++i)
        {
            fn();
        }
        const auto t1 = Clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        samples_ms_per_iter.push_back(elapsed_ms / static_cast<double>(iters));
    }

    std::sort(samples_ms_per_iter.begin(), samples_ms_per_iter.end());
    g_sink += probe_checksum(dst_probe);
    return samples_ms_per_iter[samples_ms_per_iter.size() / 2];
}

void print_csv(const std::vector<ResultRow>& rows, std::ostream& os)
{
    os << "op,kernel,depth,channels,shape,layout,border,dispatch_path,parallel_backend,parallel_chunks,ms_per_iter\n";
    os << std::fixed << std::setprecision(6);
    for (const auto& row : rows)
    {
        os << row.op << ","
           << row.kernel << ","
           << row.depth << ","
           << row.channels << ","
           << row.shape << ","
           << row.layout << ","
           << row.border << ","
           << row.dispatch_path << ","
           << row.parallel_backend << ","
           << row.parallel_chunks << ","
           << row.ms_per_iter << "\n";
    }
}

}  // namespace cvh_bench

int main(int argc, char** argv)
{
    const auto args = cvh_bench::parse_args(argc, argv);
    const auto dispatch_mode = cvh_bench::parse_dispatch_mode(args.dispatch_mode);
    const auto parallel_backend = cvh_bench::parse_parallel_backend(args.parallel_backend);
    cvh::setParallelBackend(parallel_backend);
    if (args.threads > 0)
    {
        cvh::setNumThreads(args.threads);
    }
    if (parallel_backend == cvh::ParallelBackend::OpenMP && !cvh::isOpenMPBackendAvailable())
    {
        std::cerr << "warning: OpenMP backend requested but unavailable, runtime will fallback\n";
    }
    const auto inputs = cvh_bench::build_inputs(args.profile);
    const auto channels = cvh_bench::build_channels();
    const auto kernels = cvh_bench::build_kernels(args.profile);
    const auto borders = cvh_bench::build_borders(args.profile);

    std::vector<cvh_bench::ResultRow> rows;
    rows.reserve(512);

    for (const auto& input : inputs)
    {
        for (const int cn : channels)
        {
            const int type = CV_MAKETYPE(CV_8U, cn);
            for (const auto& kernel : kernels)
            {
                for (const auto& border : borders)
                {
                    cvh::Mat src_owner;
                    cvh::Mat src_base;
                    cvh::Mat src;
                    if (input.roi_layout)
                    {
                        src_base = cvh::Mat(std::vector<int>{input.rows + 2, input.cols + 2}, type);
                        cvh_bench::fill_u8(src_base, 0xC0FFEEu + static_cast<std::uint32_t>(cn * 7));
                        src = src_base(cvh::Range(1, input.rows + 1), cvh::Range(1, input.cols + 1));
                    }
                    else
                    {
                        src_owner = cvh::Mat(std::vector<int>{input.rows, input.cols}, type);
                        cvh_bench::fill_u8(src_owner, 0xC0FFEEu + static_cast<std::uint32_t>(cn * 7));
                        src = src_owner;
                    }

                    cvh::Mat dst;
                    double ms_per_iter = 0.0;
                    std::string dispatch_path = "unknown";
                    std::string parallel_backend_used = "unknown";
                    int parallel_chunks = 1;
                    try
                    {
                        if (kernel.op == cvh_bench::FilterOp::BoxFilter)
                        {
                            auto run = [&]() {
                                if (dispatch_mode == cvh_bench::DispatchMode::FallbackOnly)
                                {
                                    cvh::detail::boxFilter_fallback(src,
                                                                    dst,
                                                                    -1,
                                                                    kernel.ksize,
                                                                    cvh::Point(-1, -1),
                                                                    true,
                                                                    border.border_type);
                                }
                                else
                                {
                                    cvh::boxFilter(src,
                                                   dst,
                                                   -1,
                                                   kernel.ksize,
                                                   cvh::Point(-1, -1),
                                                   true,
                                                   border.border_type);
                                }
                            };
                            ms_per_iter = cvh_bench::measure_case(run, dst, args.warmup, args.iters, args.repeats);
                            dispatch_path = (dispatch_mode == cvh_bench::DispatchMode::FallbackOnly)
                                                ? "forced_fallback"
                                                : cvh_bench::dispatch_path_for(kernel.op);
                            parallel_backend_used = cvh::last_parallel_backend();
                            parallel_chunks = cvh::last_parallel_chunks();
                        }
                        else
                        {
                            auto run = [&]() {
                                if (dispatch_mode == cvh_bench::DispatchMode::FallbackOnly)
                                {
                                    cvh::detail::gaussian_blur_fallback(src,
                                                                        dst,
                                                                        kernel.ksize,
                                                                        kernel.sigma_x,
                                                                        kernel.sigma_y,
                                                                        border.border_type);
                                }
                                else
                                {
                                    cvh::GaussianBlur(src,
                                                      dst,
                                                      kernel.ksize,
                                                      kernel.sigma_x,
                                                      kernel.sigma_y,
                                                      border.border_type);
                                }
                            };
                            ms_per_iter = cvh_bench::measure_case(run, dst, args.warmup, args.iters, args.repeats);
                            dispatch_path = (dispatch_mode == cvh_bench::DispatchMode::FallbackOnly)
                                                ? "forced_fallback"
                                                : cvh_bench::dispatch_path_for(kernel.op);
                            parallel_backend_used = cvh::last_parallel_backend();
                            parallel_chunks = cvh::last_parallel_chunks();
                        }
                    }
                    catch (const cvh::Exception&)
                    {
                        continue;
                    }

                    rows.push_back({
                        cvh_bench::op_name(kernel.op),
                        kernel.kernel,
                        "CV_8U",
                        cn,
                        input.shape,
                        input.roi_layout ? "roi_noncontiguous" : "continuous",
                        border.name,
                        dispatch_path,
                        parallel_backend_used,
                        parallel_chunks,
                        ms_per_iter,
                    });
                }
            }
        }
    }

    cvh_bench::print_csv(rows, std::cout);

    if (!args.output_csv.empty())
    {
        std::ofstream ofs(args.output_csv);
        if (!ofs)
        {
            std::cerr << "Failed to open output file: " << args.output_csv << "\n";
            return 2;
        }
        cvh_bench::print_csv(rows, ofs);
    }

    if (cvh_bench::g_sink == 0.123456789)
    {
        std::cerr << "sink=" << cvh_bench::g_sink << "\n";
    }

    return 0;
}
