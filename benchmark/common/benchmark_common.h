#ifndef CVH_BENCHMARK_COMMON_BENCHMARK_COMMON_H
#define CVH_BENCHMARK_COMMON_BENCHMARK_COMMON_H

#include "cvh.h"
#include "benchmark_csv.h"
#include "benchmark_metadata.h"

#if defined(CVH_ENABLE_OPENCV_INTRIN) && CVH_ENABLE_OPENCV_INTRIN
#include "cvh/core/simd/opencv_ui.h"
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace cvh_bench {
namespace common {

struct BasicArgs
{
    std::string profile = "quick";
    int warmup = 3;
    int iters = 10;
    int repeats = 7;
    std::string output_csv;
};

struct TimingResult
{
    double min_ms = 0.0;
    double median_ms = 0.0;
};

struct BenchmarkResult
{
    double min_ms = 0.0;
    double median_ms = 0.0;
    std::uint64_t checksum = 0;
};

inline bool profile_is_allowed(const std::string& profile, std::initializer_list<const char*> allowed_profiles)
{
    for (const char* allowed : allowed_profiles)
    {
        if (profile == allowed)
        {
            return true;
        }
    }
    return false;
}

inline std::string profile_list_label(std::initializer_list<const char*> allowed_profiles)
{
    std::ostringstream oss;
    bool first = true;
    for (const char* allowed : allowed_profiles)
    {
        if (!first)
        {
            oss << "/";
        }
        first = false;
        oss << allowed;
    }
    return oss.str();
}

template <typename UsageFn>
BasicArgs parse_basic_args(int argc,
                           char** argv,
                           const BasicArgs& defaults,
                           std::initializer_list<const char*> allowed_profiles,
                           UsageFn&& usage)
{
    BasicArgs args = defaults;
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

    if (!profile_is_allowed(args.profile, allowed_profiles))
    {
        std::cerr << "Unsupported profile: " << args.profile
                  << " (expected " << profile_list_label(allowed_profiles) << ")\n";
        std::exit(2);
    }

    return args;
}

inline TimingResult summarize_samples(std::vector<double>& samples)
{
    if (samples.empty())
    {
        return {};
    }
    std::sort(samples.begin(), samples.end());
    return TimingResult {samples.front(), samples[samples.size() / 2]};
}

template <typename RunOnceFn>
TimingResult measure_repeated_ms(RunOnceFn&& run_once, int warmup, int iters, int repeats)
{
    for (int i = 0; i < warmup; ++i)
    {
        run_once();
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));

    using Clock = std::chrono::steady_clock;
    for (int r = 0; r < repeats; ++r)
    {
        const auto t0 = Clock::now();
        for (int i = 0; i < iters; ++i)
        {
            run_once();
        }
        const auto t1 = Clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        samples.push_back(elapsed_ms / static_cast<double>(iters));
    }

    return summarize_samples(samples);
}

inline constexpr std::uint64_t fnv1a64_basis()
{
    return 1469598103934665603ull;
}

inline constexpr std::uint64_t fnv1a64_prime()
{
    return 1099511628211ull;
}

inline std::uint64_t fnv1a64_mix_byte(std::uint64_t hash, std::uint8_t value)
{
    hash ^= static_cast<std::uint64_t>(value);
    hash *= fnv1a64_prime();
    return hash;
}

inline std::uint64_t fnv1a64_mix_u64(std::uint64_t hash, std::uint64_t value)
{
    hash ^= value;
    hash *= fnv1a64_prime();
    return hash;
}

inline std::uint64_t checksum_bytes(const uchar* data, std::size_t count)
{
    std::uint64_t value = fnv1a64_basis();
    for (std::size_t i = 0; i < count; ++i)
    {
        value = fnv1a64_mix_byte(value, data[i]);
    }
    return value;
}

inline std::uint64_t checksum_mat_bytes(const cvh::Mat& mat)
{
    if (mat.dims == 2)
    {
        std::uint64_t value = fnv1a64_basis();
        const std::size_t row_bytes = static_cast<std::size_t>(mat.size[1]) * mat.elemSize();
        for (int row = 0; row < mat.size[0]; ++row)
        {
            const uchar* row_data = mat.data + static_cast<std::size_t>(row) * mat.step(0);
            for (std::size_t i = 0; i < row_bytes; ++i)
            {
                value = fnv1a64_mix_byte(value, row_data[i]);
            }
        }
        return value;
    }

    return checksum_bytes(mat.data, mat.total() * mat.elemSize());
}

inline bool same_mat_bytes(const cvh::Mat& lhs, const cvh::Mat& rhs)
{
    if (lhs.type() != rhs.type() || lhs.dims != rhs.dims || lhs.size != rhs.size)
    {
        return false;
    }

    if (lhs.dims == 2)
    {
        const std::size_t row_bytes = static_cast<std::size_t>(lhs.size[1]) * lhs.elemSize();
        for (int row = 0; row < lhs.size[0]; ++row)
        {
            const uchar* lhs_row = lhs.data + static_cast<std::size_t>(row) * lhs.step(0);
            const uchar* rhs_row = rhs.data + static_cast<std::size_t>(row) * rhs.step(0);
            for (std::size_t i = 0; i < row_bytes; ++i)
            {
                if (lhs_row[i] != rhs_row[i])
                {
                    return false;
                }
            }
        }
        return true;
    }

    const std::size_t count = lhs.total() * lhs.elemSize();
    for (std::size_t i = 0; i < count; ++i)
    {
        if (lhs.data[i] != rhs.data[i])
        {
            return false;
        }
    }
    return true;
}

inline void fill_bytes_lcg(uchar* data, std::size_t count, std::uint32_t seed)
{
    for (std::size_t i = 0; i < count; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        data[i] = static_cast<uchar>((seed >> 16) & 0xffu);
    }
}

inline void fill_mat_u8_lcg(cvh::Mat& mat, std::uint32_t seed)
{
    if (mat.dims != 2)
    {
        fill_bytes_lcg(mat.data, mat.total() * mat.elemSize(), seed);
        return;
    }

    const std::size_t row_bytes = static_cast<std::size_t>(mat.size[1]) * mat.elemSize();
    for (int row = 0; row < mat.size[0]; ++row)
    {
        uchar* row_data = mat.data + static_cast<std::size_t>(row) * mat.step(0);
        for (std::size_t i = 0; i < row_bytes; ++i)
        {
            seed = seed * 1664525u + 1013904223u;
            row_data[i] = static_cast<uchar>((seed >> 16) & 0xffu);
        }
    }
}

inline void fill_mat_f32_lcg(cvh::Mat& mat, std::uint32_t seed)
{
    const std::size_t row_scalars =
        mat.dims == 2
            ? static_cast<std::size_t>(mat.size[1]) * static_cast<std::size_t>(mat.channels())
            : mat.total() * static_cast<std::size_t>(mat.channels());
    const int rows = mat.dims == 2 ? mat.size[0] : 1;
    for (int row = 0; row < rows; ++row)
    {
        float* row_data = reinterpret_cast<float*>(
            mat.data + (mat.dims == 2 ? static_cast<std::size_t>(row) * mat.step(0) : 0u));
        for (std::size_t i = 0; i < row_scalars; ++i)
        {
            seed = seed * 1664525u + 1013904223u;
            row_data[i] = static_cast<float>((seed >> 8) & 0xffffu) / 65535.0f;
        }
    }
}

inline std::string opencv_intrin_backend_name()
{
#if defined(CV_FORCE_SIMD128_CPP)
    return "opencv_intrin_cpp";
#elif defined(CV_NEON) && CV_NEON
    return "opencv_intrin_neon";
#else
    return "opencv_intrin_cpp";
#endif
}

inline int simd_u8_lanes()
{
#if defined(CVH_ENABLE_OPENCV_INTRIN) && CVH_ENABLE_OPENCV_INTRIN
    return cv::VTraits<cv::v_uint8x16>::vlanes();
#else
    return 1;
#endif
}

inline double mpix_per_sec(std::size_t pixels, double median_ms)
{
    return median_ms > 0.0 ? static_cast<double>(pixels) / median_ms / 1000.0 : 0.0;
}

}  // namespace common
}  // namespace cvh_bench

#endif  // CVH_BENCHMARK_COMMON_BENCHMARK_COMMON_H
