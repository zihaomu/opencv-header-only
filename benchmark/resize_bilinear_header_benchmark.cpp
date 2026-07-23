#include "cvh.h"
#include "cvh/core/simd/opencv_ui.h"

#if !CVH_ENABLE_OPENCV_INTRIN
#error "cvh_benchmark_resize_bilinear_header must be compiled with CVH_ENABLE_OPENCV_INTRIN=1"
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

struct ResizeCase
{
    const char* name;
    int src_rows;
    int src_cols;
    int dst_rows;
    int dst_cols;
    int channels;
};

struct LinearTables
{
    std::vector<int> x0b;
    std::vector<int> x1b;
    std::vector<int> y0;
    std::vector<int> y1;
    std::vector<float> wx;
    std::vector<float> wy;
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
    std::string op;
    std::string backend;
    std::string entry;
    std::string allocation_mode;
    ResizeCase shape;
    std::size_t pixels = 0;
    int simd_lanes = 0;
    std::size_t tail_pixels = 0;
    double tail_ratio = 0.0;
    int warmup = 0;
    int iters = 0;
    int repeats = 0;
    double min_ms = 0.0;
    double median_ms = 0.0;
    double mpix_per_sec = 0.0;
    double speedup_vs_scalar = 1.0;
    std::uint64_t checksum = 0;
};

using ResizeFn = void (*)(const cvh::Mat&, cvh::Mat&, const ResizeCase&);
using TableFn = std::uint64_t (*)(const ResizeCase&);
using PrecomputedResizeFn = void (*)(const cvh::Mat&, cvh::Mat&, const ResizeCase&, const LinearTables&);

enum class AllocationMode
{
    Reuse,
    Recreate,
};

volatile std::uint64_t g_sink = 0;

void usage()
{
    std::cout
        << "Usage: cvh_benchmark_resize_bilinear_header "
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

std::vector<ResizeCase> build_cases(const std::string& profile)
{
    std::vector<ResizeCase> cases = {
        {"640x480_to_320x240", 480, 640, 240, 320, 1},
        {"1280x720_to_640x360", 720, 1280, 360, 640, 1},
        {"1920x1080_to_960x540", 1080, 1920, 540, 960, 1},
        {"3840x2160_to_1920x1080", 2160, 3840, 1080, 1920, 1},
    };

    if (profile == "full")
    {
        cases.push_back({"641x479_to_321x239", 479, 641, 239, 321, 1});
        cases.push_back({"1919x1080_to_961x541", 1080, 1919, 541, 961, 1});
        cases.push_back({"1920x1080_to_853x480", 1080, 1920, 480, 853, 1});
        cases.push_back({"3839x2160_to_1917x1079", 2160, 3839, 1079, 1917, 1});
        cases.push_back({"3840x2160_to_1280x720", 2160, 3840, 720, 1280, 1});
    }

    return cases;
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

bool has_resize_opencv_intrin_fastpath(const ResizeCase& shape)
{
    return shape.channels == 1 &&
           shape.src_rows == shape.dst_rows * 2 &&
           shape.src_cols == shape.dst_cols * 2;
}

std::string public_resize_backend_name(const ResizeCase& shape)
{
    if (has_resize_opencv_intrin_fastpath(shape))
    {
        return opencv_intrin_backend_name();
    }
    return opencv_intrin_backend_name() + "_no_resize_fastpath";
}

std::string allocation_mode_name(AllocationMode mode)
{
    return mode == AllocationMode::Reuse ? "reuse" : "recreate";
}

int simd_lanes()
{
    return cv::VTraits<cv::v_uint8x16>::vlanes();
}

std::size_t output_pixels(const ResizeCase& shape)
{
    return static_cast<std::size_t>(shape.dst_rows) * static_cast<std::size_t>(shape.dst_cols);
}

std::size_t tail_pixels(const ResizeCase& shape)
{
    const int lanes = simd_lanes();
    if (lanes <= 0)
    {
        return 0;
    }
    return static_cast<std::size_t>(shape.dst_rows) * static_cast<std::size_t>(shape.dst_cols % lanes);
}

void fill_u8(cvh::Mat& mat, std::uint32_t seed)
{
    const std::size_t count = mat.total() * static_cast<std::size_t>(mat.channels());
    for (std::size_t i = 0; i < count; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        mat.data[i] = static_cast<uchar>((seed >> 16) & 0xffu);
    }
}

std::uint64_t checksum_bytes(const uchar* data, std::size_t count)
{
    std::uint64_t value = 1469598103934665603ull;
    for (std::size_t i = 0; i < count; ++i)
    {
        value ^= static_cast<std::uint64_t>(data[i]);
        value *= 1099511628211ull;
    }
    return value;
}

std::uint64_t checksum(const cvh::Mat& mat)
{
    const std::size_t count = mat.total() * static_cast<std::size_t>(mat.channels()) *
                              static_cast<std::size_t>(mat.elemSize1());
    return checksum_bytes(mat.data, count);
}

std::uint64_t checksum_tables(const LinearTables& tables)
{
    std::uint64_t value = 1469598103934665603ull;
    auto mix_u64 = [&](std::uint64_t v) {
        value ^= v;
        value *= 1099511628211ull;
    };
    for (const int v : tables.x0b)
    {
        mix_u64(static_cast<std::uint64_t>(static_cast<std::uint32_t>(v)));
    }
    for (const int v : tables.x1b)
    {
        mix_u64(static_cast<std::uint64_t>(static_cast<std::uint32_t>(v)));
    }
    for (const int v : tables.y0)
    {
        mix_u64(static_cast<std::uint64_t>(static_cast<std::uint32_t>(v)));
    }
    for (const int v : tables.y1)
    {
        mix_u64(static_cast<std::uint64_t>(static_cast<std::uint32_t>(v)));
    }
    for (const float v : tables.wx)
    {
        std::uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(v), "float checksum expects 32-bit float");
        std::memcpy(&bits, &v, sizeof(bits));
        mix_u64(static_cast<std::uint64_t>(bits));
    }
    for (const float v : tables.wy)
    {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        mix_u64(static_cast<std::uint64_t>(bits));
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

LinearTables build_linear_tables(const ResizeCase& shape)
{
    LinearTables tables;
    tables.x0b.resize(static_cast<std::size_t>(shape.dst_cols));
    tables.x1b.resize(static_cast<std::size_t>(shape.dst_cols));
    tables.y0.resize(static_cast<std::size_t>(shape.dst_rows));
    tables.y1.resize(static_cast<std::size_t>(shape.dst_rows));
    tables.wx.resize(static_cast<std::size_t>(shape.dst_cols));
    tables.wy.resize(static_cast<std::size_t>(shape.dst_rows));

    const float scale_x = static_cast<float>(shape.src_cols) / static_cast<float>(shape.dst_cols);
    const float scale_y = static_cast<float>(shape.src_rows) / static_cast<float>(shape.dst_rows);

    for (int x = 0; x < shape.dst_cols; ++x)
    {
        const float fx_src = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
        const int x0 = std::clamp(static_cast<int>(std::floor(fx_src)), 0, shape.src_cols - 1);
        const int x1 = std::min(x0 + 1, shape.src_cols - 1);
        tables.x0b[static_cast<std::size_t>(x)] = x0 * shape.channels;
        tables.x1b[static_cast<std::size_t>(x)] = x1 * shape.channels;
        tables.wx[static_cast<std::size_t>(x)] = fx_src - static_cast<float>(x0);
    }

    for (int y = 0; y < shape.dst_rows; ++y)
    {
        const float fy_src = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const int y0 = std::clamp(static_cast<int>(std::floor(fy_src)), 0, shape.src_rows - 1);
        const int y1 = std::min(y0 + 1, shape.src_rows - 1);
        tables.y0[static_cast<std::size_t>(y)] = y0;
        tables.y1[static_cast<std::size_t>(y)] = y1;
        tables.wy[static_cast<std::size_t>(y)] = fy_src - static_cast<float>(y0);
    }

    return tables;
}

std::uint64_t build_linear_tables_for_measure(const ResizeCase& shape)
{
    const LinearTables tables = build_linear_tables(shape);
    return checksum_tables(tables);
}

void resize_public(const cvh::Mat& src, cvh::Mat& dst, const ResizeCase& shape)
{
    cvh::resize(src, dst, cvh::Size(shape.dst_cols, shape.dst_rows), 0.0, 0.0, cvh::INTER_LINEAR);
}

void resize_direct_scalar(const cvh::Mat& src, cvh::Mat& dst, const ResizeCase& shape)
{
    dst.create(std::vector<int>{shape.dst_rows, shape.dst_cols}, src.type());
    cvh::detail::resize_fallback_impl_typed<uchar>(src, dst, cvh::INTER_LINEAR);
}

void resize_direct_opencv_intrin(const cvh::Mat& src, cvh::Mat& dst, const ResizeCase& shape)
{
    CV_Assert(has_resize_opencv_intrin_fastpath(shape));
    cvh::detail::resize_linear_u8c1_downsample2_opencv_intrin_impl(src, dst);
}

void resize_precomputed_linear_u8c1(const cvh::Mat& src,
                                    cvh::Mat& dst,
                                    const ResizeCase& shape,
                                    const LinearTables& tables)
{
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 1);
    CV_Assert(shape.channels == 1);

    dst.create(std::vector<int>{shape.dst_rows, shape.dst_cols}, CV_8UC1);

    for (int y = 0; y < shape.dst_rows; ++y)
    {
        const int y0 = tables.y0[static_cast<std::size_t>(y)];
        const int y1 = tables.y1[static_cast<std::size_t>(y)];
        const float wy = tables.wy[static_cast<std::size_t>(y)];
        const uchar* src_row0 = src.data + static_cast<std::size_t>(y0) * src.step(0);
        const uchar* src_row1 = src.data + static_cast<std::size_t>(y1) * src.step(0);
        uchar* dst_row = dst.data + static_cast<std::size_t>(y) * dst.step(0);

        for (int x = 0; x < shape.dst_cols; ++x)
        {
            const std::size_t ix = static_cast<std::size_t>(x);
            const int x0b = tables.x0b[ix];
            const int x1b = tables.x1b[ix];
            const float wx = tables.wx[ix];
            const float p00 = static_cast<float>(src_row0[x0b]);
            const float p01 = static_cast<float>(src_row0[x1b]);
            const float p10 = static_cast<float>(src_row1[x0b]);
            const float p11 = static_cast<float>(src_row1[x1b]);
            const float top = cvh::detail::lerp(p00, p01, wx);
            const float bot = cvh::detail::lerp(p10, p11, wx);
            dst_row[x] = cvh::saturate_cast<uchar>(cvh::detail::lerp(top, bot, wy));
        }
    }
}

void prepare_dst(cvh::Mat& dst, const ResizeCase& shape, AllocationMode allocation_mode)
{
    if (allocation_mode == AllocationMode::Recreate)
    {
        dst.release();
        return;
    }
    dst.create(std::vector<int>{shape.dst_rows, shape.dst_cols}, CV_8UC1);
}

Result measure_resize(ResizeFn fn,
                      const cvh::Mat& src,
                      cvh::Mat& dst,
                      const ResizeCase& shape,
                      AllocationMode allocation_mode,
                      int warmup,
                      int iters,
                      int repeats)
{
    for (int i = 0; i < warmup; ++i)
    {
        prepare_dst(dst, shape, allocation_mode);
        fn(src, dst, shape);
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));

    using Clock = std::chrono::steady_clock;
    for (int r = 0; r < repeats; ++r)
    {
        const auto t0 = Clock::now();
        for (int i = 0; i < iters; ++i)
        {
            prepare_dst(dst, shape, allocation_mode);
            fn(src, dst, shape);
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

Result measure_precomputed_resize(PrecomputedResizeFn fn,
                                  const cvh::Mat& src,
                                  cvh::Mat& dst,
                                  const ResizeCase& shape,
                                  const LinearTables& tables,
                                  AllocationMode allocation_mode,
                                  int warmup,
                                  int iters,
                                  int repeats)
{
    for (int i = 0; i < warmup; ++i)
    {
        prepare_dst(dst, shape, allocation_mode);
        fn(src, dst, shape, tables);
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));

    using Clock = std::chrono::steady_clock;
    for (int r = 0; r < repeats; ++r)
    {
        const auto t0 = Clock::now();
        for (int i = 0; i < iters; ++i)
        {
            prepare_dst(dst, shape, allocation_mode);
            fn(src, dst, shape, tables);
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

Result measure_tables(TableFn fn, const ResizeCase& shape, int warmup, int iters, int repeats)
{
    for (int i = 0; i < warmup; ++i)
    {
        g_sink ^= fn(shape);
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    std::uint64_t hash = 0;

    using Clock = std::chrono::steady_clock;
    for (int r = 0; r < repeats; ++r)
    {
        std::uint64_t sample_hash = 1469598103934665603ull;
        const auto t0 = Clock::now();
        for (int i = 0; i < iters; ++i)
        {
            sample_hash ^= fn(shape);
            sample_hash *= 1099511628211ull;
        }
        const auto t1 = Clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        samples.push_back(elapsed_ms / static_cast<double>(iters));
        hash ^= sample_hash;
        hash *= 1099511628211ull;
    }

    std::sort(samples.begin(), samples.end());
    g_sink ^= hash;
    return Result {samples.front(), samples[samples.size() / 2], hash};
}

ResultRow make_row(const Args& args,
                   const std::string& op,
                   const std::string& backend,
                   const std::string& entry,
                   const std::string& allocation_mode,
                   const ResizeCase& shape,
                   const Result& result,
                   double scalar_median_ms)
{
    const std::size_t pixels = output_pixels(shape);
    const std::size_t tails = tail_pixels(shape);
    return ResultRow {
        args.profile,
        op,
        backend,
        entry,
        allocation_mode,
        shape,
        pixels,
        simd_lanes(),
        tails,
        static_cast<double>(tails) / static_cast<double>(pixels),
        args.warmup,
        args.iters,
        args.repeats,
        result.min_ms,
        result.median_ms,
        static_cast<double>(pixels) / result.median_ms / 1000.0,
        scalar_median_ms > 0.0 ? scalar_median_ms / result.median_ms : 0.0,
        result.checksum,
    };
}

void print_csv(const std::vector<ResultRow>& rows, std::ostream& os)
{
    os << "profile,op,backend,entry,allocation_mode,shape,"
       << "src_width,src_height,dst_width,dst_height,channels,pixels,"
       << "simd_lanes,tail_pixels,tail_ratio,warmup,iters,repeats,"
       << "min_ms_per_call,median_ms_per_call,mpix_per_sec,speedup_vs_scalar,checksum\n";
    os << std::fixed << std::setprecision(6);
    for (const auto& row : rows)
    {
        os << row.profile << ","
           << row.op << ","
           << row.backend << ","
           << row.entry << ","
           << row.allocation_mode << ","
           << row.shape.name << ","
           << row.shape.src_cols << ","
           << row.shape.src_rows << ","
           << row.shape.dst_cols << ","
           << row.shape.dst_rows << ","
           << row.shape.channels << ","
           << row.pixels << ","
           << row.simd_lanes << ","
           << row.tail_pixels << ","
           << row.tail_ratio << ","
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
    const auto cases = cvh_bench::build_cases(args.profile);
    const std::vector<cvh_bench::AllocationMode> allocation_modes = {
        cvh_bench::AllocationMode::Reuse,
        cvh_bench::AllocationMode::Recreate,
    };
    std::vector<cvh_bench::ResultRow> rows;
    rows.reserve(cases.size() * 7);

    for (const auto& shape : cases)
    {
        cvh::Mat src({shape.src_rows, shape.src_cols}, CV_8UC1);
        cvh_bench::fill_u8(
            src,
            static_cast<std::uint32_t>(
                shape.src_rows * 131 + shape.src_cols * 17 + shape.dst_rows * 19 + shape.dst_cols * 23));

        const cvh_bench::LinearTables tables = cvh_bench::build_linear_tables(shape);

        cvh::Mat scalar_check;
        cvh::Mat public_check;
        cvh::Mat precomputed_check;
        cvh_bench::resize_direct_scalar(src, scalar_check, shape);
        cvh_bench::resize_public(src, public_check, shape);
        cvh_bench::resize_precomputed_linear_u8c1(src, precomputed_check, shape, tables);
        if (!cvh_bench::same_mat(scalar_check, public_check) ||
            !cvh_bench::same_mat(scalar_check, precomputed_check))
        {
            std::cerr << "Correctness mismatch for shape " << shape.name << "\n";
            return 3;
        }
        if (cvh_bench::has_resize_opencv_intrin_fastpath(shape))
        {
            cvh::Mat direct_opencv_intrin_check;
            cvh_bench::resize_direct_opencv_intrin(src, direct_opencv_intrin_check, shape);
            if (!cvh_bench::same_mat(scalar_check, direct_opencv_intrin_check))
            {
                std::cerr << "OpenCV intrinsics correctness mismatch for shape " << shape.name << "\n";
                return 4;
            }
        }

        for (const auto allocation_mode : allocation_modes)
        {
            cvh::Mat scalar_dst;
            cvh::Mat public_dst;
            cvh::Mat precomputed_dst;
            cvh::Mat direct_opencv_intrin_dst;
            const auto scalar = cvh_bench::measure_resize(
                cvh_bench::resize_direct_scalar,
                src,
                scalar_dst,
                shape,
                allocation_mode,
                args.warmup,
                args.iters,
                args.repeats);
            const auto public_entry = cvh_bench::measure_resize(
                cvh_bench::resize_public,
                src,
                public_dst,
                shape,
                allocation_mode,
                args.warmup,
                args.iters,
                args.repeats);
            cvh_bench::Result direct_opencv_intrin;
            if (cvh_bench::has_resize_opencv_intrin_fastpath(shape))
            {
                direct_opencv_intrin = cvh_bench::measure_resize(
                    cvh_bench::resize_direct_opencv_intrin,
                    src,
                    direct_opencv_intrin_dst,
                    shape,
                    allocation_mode,
                    args.warmup,
                    args.iters,
                    args.repeats);
            }
            const auto precomputed = cvh_bench::measure_precomputed_resize(
                cvh_bench::resize_precomputed_linear_u8c1,
                src,
                precomputed_dst,
                shape,
                tables,
                allocation_mode,
                args.warmup,
                args.iters,
                args.repeats);

            rows.push_back(cvh_bench::make_row(
                args,
                "RESIZE_LINEAR_U8_C1",
                "scalar",
                "direct_detail",
                cvh_bench::allocation_mode_name(allocation_mode),
                shape,
                scalar,
                scalar.median_ms));
            rows.push_back(cvh_bench::make_row(
                args,
                "RESIZE_LINEAR_U8_C1",
                cvh_bench::public_resize_backend_name(shape),
                "public_headers_fast_resize",
                cvh_bench::allocation_mode_name(allocation_mode),
                shape,
                public_entry,
                scalar.median_ms));
            if (cvh_bench::has_resize_opencv_intrin_fastpath(shape))
            {
                rows.push_back(cvh_bench::make_row(
                    args,
                    "RESIZE_LINEAR_U8_C1",
                    cvh_bench::opencv_intrin_backend_name(),
                    "direct_detail",
                    cvh_bench::allocation_mode_name(allocation_mode),
                    shape,
                    direct_opencv_intrin,
                    scalar.median_ms));
            }
            rows.push_back(cvh_bench::make_row(
                args,
                "MICRO_RESIZE_LINEAR_U8_C1_PRECOMPUTED_TABLES",
                "scalar",
                "pixel_kernel_precomputed_tables",
                cvh_bench::allocation_mode_name(allocation_mode),
                shape,
                precomputed,
                scalar.median_ms));
        }

        const auto table_build = cvh_bench::measure_tables(
            cvh_bench::build_linear_tables_for_measure,
            shape,
            args.warmup,
            args.iters,
            args.repeats);
        rows.push_back(cvh_bench::make_row(
            args,
            "MICRO_RESIZE_LINEAR_TABLES",
            "scalar",
            "build_xy_weight_tables",
            "none",
            shape,
            table_build,
            0.0));
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
