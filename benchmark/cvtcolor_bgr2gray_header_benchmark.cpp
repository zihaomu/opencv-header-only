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
    std::string op;
    std::string backend;
    std::string entry;
    std::string allocation_mode;
    ShapeCase shape;
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

using BenchFn = void (*)(const cvh::Mat&, cvh::Mat&);
using MicroFn = void (*)(const cvh::Mat&, const cvh::Mat&, const ShapeCase&, cvh::Mat&);

struct OperationCase
{
    const char* name;
    BenchFn scalar_direct;
    BenchFn public_entry;
    BenchFn simd_direct;
};

struct MicroCase
{
    const char* name;
    std::string backend;
    const char* entry;
    MicroFn fn;
};

enum class AllocationMode
{
    Reuse,
    Recreate,
};

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
                {"63x480", 480, 63},
                {"641x479", 479, 641},
            });
        shapes.push_back({"1919x1080", 1080, 1919});
        shapes.push_back({"3839x2160", 2160, 3839});
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

std::string allocation_mode_name(AllocationMode mode)
{
    return mode == AllocationMode::Reuse ? "reuse" : "recreate";
}

int simd_lanes()
{
    return static_cast<int>(cvh::detail::simd::u8_lanes());
}

std::size_t tail_pixels(const ShapeCase& shape)
{
    const int lanes = simd_lanes();
    if (lanes <= 0)
    {
        return 0;
    }
    return static_cast<std::size_t>(shape.rows) * static_cast<std::size_t>(shape.cols % lanes);
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

cvh::Mat make_planar_bgr(const cvh::Mat& bgr)
{
    const int rows = bgr.size[0];
    const int cols = bgr.size[1];
    const std::size_t plane_size = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    cvh::Mat planar({rows, cols * 3}, CV_8UC1);

    std::uint8_t* b_plane = reinterpret_cast<std::uint8_t*>(planar.data);
    std::uint8_t* g_plane = b_plane + plane_size;
    std::uint8_t* r_plane = g_plane + plane_size;

    for (int y = 0; y < rows; ++y)
    {
        const std::uint8_t* src_row = reinterpret_cast<const std::uint8_t*>(bgr.data + static_cast<std::size_t>(y) * bgr.step(0));
        std::uint8_t* b_row = b_plane + static_cast<std::size_t>(y) * static_cast<std::size_t>(cols);
        std::uint8_t* g_row = g_plane + static_cast<std::size_t>(y) * static_cast<std::size_t>(cols);
        std::uint8_t* r_row = r_plane + static_cast<std::size_t>(y) * static_cast<std::size_t>(cols);
        for (int x = 0; x < cols; ++x)
        {
            const std::size_t sx = static_cast<std::size_t>(x) * 3;
            b_row[x] = src_row[sx + 0];
            g_row[x] = src_row[sx + 1];
            r_row[x] = src_row[sx + 2];
        }
    }

    return planar;
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

void run_rgb_scalar(const cvh::Mat& src, cvh::Mat& dst)
{
    cvh::detail::cvtcolor_rgb2gray_u8_scalar_impl(src, dst);
}

void run_opencv_intrin(const cvh::Mat& src, cvh::Mat& dst)
{
    cvh::cvtColor(src, dst, cvh::COLOR_BGR2GRAY);
}

void run_rgb_opencv_intrin(const cvh::Mat& src, cvh::Mat& dst)
{
    cvh::cvtColor(src, dst, cvh::COLOR_RGB2GRAY);
}

void run_opencv_intrin_direct(const cvh::Mat& src, cvh::Mat& dst)
{
    cvh::detail::cvtcolor_bgr2gray_u8_simd_impl(src, dst);
}

void run_rgb_opencv_intrin_direct(const cvh::Mat& src, cvh::Mat& dst)
{
    cvh::detail::cvtcolor_rgb2gray_u8_simd_impl(src, dst);
}

void prepare_dst(cvh::Mat& dst, const cvh::Mat& src, AllocationMode allocation_mode)
{
    if (allocation_mode == AllocationMode::Recreate)
    {
        dst.release();
        return;
    }
    dst.create(std::vector<int>{src.size[0], src.size[1]}, CV_8UC1);
}

void prepare_dst(cvh::Mat& dst, const ShapeCase& shape, AllocationMode allocation_mode)
{
    if (allocation_mode == AllocationMode::Recreate)
    {
        dst.release();
        return;
    }
    dst.create(std::vector<int>{shape.rows, shape.cols}, CV_8UC1);
}

Result measure(BenchFn fn,
               const cvh::Mat& src,
               cvh::Mat& dst,
               AllocationMode allocation_mode,
               int warmup,
               int iters,
               int repeats)
{
    for (int i = 0; i < warmup; ++i)
    {
        prepare_dst(dst, src, allocation_mode);
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
            prepare_dst(dst, src, allocation_mode);
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

void run_micro_store_u8(const cvh::Mat&, const cvh::Mat&, const ShapeCase& shape, cvh::Mat& dst)
{
    namespace simd = cvh::detail::simd;

    dst.create(std::vector<int>{shape.rows, shape.cols}, CV_8UC1);
    const int lanes = simd_lanes();
    const simd::u16 wide = simd::setall_u16(123);
    const simd::u8 value = simd::pack_u16_to_u8(wide, wide);

    for (int y = 0; y < shape.rows; ++y)
    {
        std::uint8_t* dst_row = reinterpret_cast<std::uint8_t*>(dst.data + static_cast<std::size_t>(y) * dst.step(0));
        int x = 0;
        for (; x + lanes <= shape.cols; x += lanes)
        {
            simd::store_u8(dst_row + x, value);
        }
        for (; x < shape.cols; ++x)
        {
            dst_row[x] = 123;
        }
    }
}

void run_micro_scalar_read3_write1_u8(const cvh::Mat& bgr,
                                      const cvh::Mat&,
                                      const ShapeCase& shape,
                                      cvh::Mat& dst)
{
    dst.create(std::vector<int>{shape.rows, shape.cols}, CV_8UC1);
    for (int y = 0; y < shape.rows; ++y)
    {
        const std::uint8_t* src_row = reinterpret_cast<const std::uint8_t*>(
            bgr.data + static_cast<std::size_t>(y) * bgr.step(0));
        std::uint8_t* dst_row = reinterpret_cast<std::uint8_t*>(dst.data + static_cast<std::size_t>(y) * dst.step(0));
        for (int x = 0; x < shape.cols; ++x)
        {
            const std::size_t sx = static_cast<std::size_t>(x) * 3;
            dst_row[x] = static_cast<std::uint8_t>(src_row[sx + 0] ^ src_row[sx + 1] ^ src_row[sx + 2]);
        }
    }
}

void run_micro_load_deinterleave_store_u8(const cvh::Mat& bgr,
                                          const cvh::Mat&,
                                          const ShapeCase& shape,
                                          cvh::Mat& dst)
{
    namespace simd = cvh::detail::simd;

    dst.create(std::vector<int>{shape.rows, shape.cols}, CV_8UC1);
    const int lanes = simd_lanes();
    for (int y = 0; y < shape.rows; ++y)
    {
        const std::uint8_t* src_row = reinterpret_cast<const std::uint8_t*>(
            bgr.data + static_cast<std::size_t>(y) * bgr.step(0));
        std::uint8_t* dst_row = reinterpret_cast<std::uint8_t*>(dst.data + static_cast<std::size_t>(y) * dst.step(0));
        int x = 0;
        for (; x + lanes <= shape.cols; x += lanes)
        {
            simd::u8 c0;
            simd::u8 c1;
            simd::u8 c2;
            simd::load_deinterleave3_u8(src_row + static_cast<std::size_t>(x) * 3, c0, c1, c2);
            simd::store_u8(dst_row + x, c0);
        }
        for (; x < shape.cols; ++x)
        {
            dst_row[x] = src_row[static_cast<std::size_t>(x) * 3];
        }
    }
}

void run_micro_planar_widen_mul_pack_store_u8(const cvh::Mat&,
                                              const cvh::Mat& planar,
                                              const ShapeCase& shape,
                                              cvh::Mat& dst)
{
    namespace simd = cvh::detail::simd;

    dst.create(std::vector<int>{shape.rows, shape.cols}, CV_8UC1);
    const int lanes = simd_lanes();
    const std::size_t plane_size = static_cast<std::size_t>(shape.rows) * static_cast<std::size_t>(shape.cols);
    const std::uint8_t* b_plane = reinterpret_cast<const std::uint8_t*>(planar.data);
    const std::uint8_t* g_plane = b_plane + plane_size;
    const std::uint8_t* r_plane = g_plane + plane_size;
    const simd::u16 weight_b = simd::setall_u16(7471);
    const simd::u16 weight_g = simd::setall_u16(38470);
    const simd::u16 weight_r = simd::setall_u16(19595);

    for (int y = 0; y < shape.rows; ++y)
    {
        const std::size_t row_offset = static_cast<std::size_t>(y) * static_cast<std::size_t>(shape.cols);
        const std::uint8_t* b_row = b_plane + row_offset;
        const std::uint8_t* g_row = g_plane + row_offset;
        const std::uint8_t* r_row = r_plane + row_offset;
        std::uint8_t* dst_row = reinterpret_cast<std::uint8_t*>(dst.data + static_cast<std::size_t>(y) * dst.step(0));

        int x = 0;
        for (; x + lanes <= shape.cols; x += lanes)
        {
            const simd::u8 b8 = simd::load_u8(b_row + x);
            const simd::u8 g8 = simd::load_u8(g_row + x);
            const simd::u8 r8 = simd::load_u8(r_row + x);

            simd::u16 b16_lo;
            simd::u16 b16_hi;
            simd::u16 g16_lo;
            simd::u16 g16_hi;
            simd::u16 r16_lo;
            simd::u16 r16_hi;
            simd::expand_u8(b8, b16_lo, b16_hi);
            simd::expand_u8(g8, g16_lo, g16_hi);
            simd::expand_u8(r8, r16_lo, r16_hi);

            const simd::u16 y16_lo = cvh::detail::cvtcolor_bgr2gray_u8_wide_simd(
                b16_lo, g16_lo, r16_lo, weight_b, weight_g, weight_r);
            const simd::u16 y16_hi = cvh::detail::cvtcolor_bgr2gray_u8_wide_simd(
                b16_hi, g16_hi, r16_hi, weight_b, weight_g, weight_r);
            simd::store_u8(dst_row + x, simd::pack_u16_to_u8(y16_lo, y16_hi));
        }

        for (; x < shape.cols; ++x)
        {
            dst_row[x] = cvh::detail::cvtcolor_bgr2gray_pixel_u8(b_row[x], g_row[x], r_row[x]);
        }
    }
}

Result measure_micro(MicroFn fn,
                     const cvh::Mat& bgr,
                     const cvh::Mat& planar,
                     const ShapeCase& shape,
                     AllocationMode allocation_mode,
                     int warmup,
                     int iters,
                     int repeats)
{
    cvh::Mat dst;
    for (int i = 0; i < warmup; ++i)
    {
        prepare_dst(dst, shape, allocation_mode);
        fn(bgr, planar, shape, dst);
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
            fn(bgr, planar, shape, dst);
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
                   const std::string& op,
                   const std::string& backend,
                   const std::string& entry,
                   AllocationMode allocation_mode,
                   const ShapeCase& shape,
                   const Result& result,
                   double scalar_median_ms)
{
    const std::size_t pixels = static_cast<std::size_t>(shape.rows) * static_cast<std::size_t>(shape.cols);
    const std::size_t tails = tail_pixels(shape);
    return ResultRow {
        args.profile,
        op,
        backend,
        entry,
        allocation_mode_name(allocation_mode),
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
        scalar_median_ms / result.median_ms,
        result.checksum,
    };
}

ResultRow make_micro_row(const Args& args,
                         const MicroCase& micro,
                         AllocationMode allocation_mode,
                         const ShapeCase& shape,
                         const Result& result)
{
    const std::size_t pixels = static_cast<std::size_t>(shape.rows) * static_cast<std::size_t>(shape.cols);
    const std::size_t tails = tail_pixels(shape);
    return ResultRow {
        args.profile,
        micro.name,
        micro.backend,
        micro.entry,
        allocation_mode_name(allocation_mode),
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
        0.0,
        result.checksum,
    };
}

void print_csv(const std::vector<ResultRow>& rows, std::ostream& os)
{
    os << "profile,op,backend,entry,allocation_mode,shape,width,height,pixels,simd_lanes,tail_pixels,tail_ratio,"
       << "warmup,iters,repeats,"
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
           << row.shape.cols << ","
           << row.shape.rows << ","
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
    const auto shapes = cvh_bench::build_shapes(args.profile);
    const std::vector<cvh_bench::OperationCase> operations = {
        {
            "CVTCOLOR_BGR2GRAY_U8",
            cvh_bench::run_scalar,
            cvh_bench::run_opencv_intrin,
            cvh_bench::run_opencv_intrin_direct,
        },
        {
            "CVTCOLOR_RGB2GRAY_U8",
            cvh_bench::run_rgb_scalar,
            cvh_bench::run_rgb_opencv_intrin,
            cvh_bench::run_rgb_opencv_intrin_direct,
        },
    };
    const std::vector<cvh_bench::AllocationMode> allocation_modes = {
        cvh_bench::AllocationMode::Reuse,
        cvh_bench::AllocationMode::Recreate,
    };
    const std::vector<cvh_bench::MicroCase> micro_cases = {
        {
            "MICRO_STORE_U8",
            cvh_bench::opencv_intrin_backend_name(),
            "store_only",
            cvh_bench::run_micro_store_u8,
        },
        {
            "MICRO_SCALAR_READ3_WRITE1_U8",
            "scalar",
            "scalar_read3_write1",
            cvh_bench::run_micro_scalar_read3_write1_u8,
        },
        {
            "MICRO_LOAD_DEINTERLEAVE_STORE_U8",
            cvh_bench::opencv_intrin_backend_name(),
            "load_deinterleave_store",
            cvh_bench::run_micro_load_deinterleave_store_u8,
        },
        {
            "MICRO_PLANAR_LOAD3_WIDEN_MUL_PACK_STORE_U8",
            cvh_bench::opencv_intrin_backend_name(),
            "planar_load3_widen_mul_pack_store",
            cvh_bench::run_micro_planar_widen_mul_pack_store_u8,
        },
    };
    std::vector<cvh_bench::ResultRow> rows;
    rows.reserve(shapes.size() * (operations.size() * allocation_modes.size() * 3 + micro_cases.size()));

    for (const auto& shape : shapes)
    {
        cvh::Mat src({shape.rows, shape.cols}, CV_8UC3);
        cvh_bench::fill_bgr(src, static_cast<std::uint32_t>(shape.rows * 131 + shape.cols * 17));
        const cvh::Mat planar = cvh_bench::make_planar_bgr(src);

        for (const auto& operation : operations)
        {
            cvh::Mat scalar_check;
            cvh::Mat public_check;
            cvh::Mat direct_check;
            operation.scalar_direct(src, scalar_check);
            operation.public_entry(src, public_check);
            operation.simd_direct(src, direct_check);
            if (!cvh_bench::same_mat(scalar_check, public_check) ||
                !cvh_bench::same_mat(scalar_check, direct_check))
            {
                std::cerr << "Correctness mismatch for op " << operation.name
                          << ", shape " << shape.name << "\n";
                return 3;
            }

            for (const auto allocation_mode : allocation_modes)
            {
                cvh::Mat scalar_dst;
                cvh::Mat public_dst;
                cvh::Mat direct_dst;
                const auto scalar = cvh_bench::measure(
                    operation.scalar_direct, src, scalar_dst, allocation_mode, args.warmup, args.iters, args.repeats);
                const auto public_entry = cvh_bench::measure(
                    operation.public_entry, src, public_dst, allocation_mode, args.warmup, args.iters, args.repeats);
                const auto direct_entry = cvh_bench::measure(
                    operation.simd_direct, src, direct_dst, allocation_mode, args.warmup, args.iters, args.repeats);

                rows.push_back(cvh_bench::make_row(
                    args, operation.name, "scalar", "direct_detail", allocation_mode, shape, scalar, scalar.median_ms));
                rows.push_back(cvh_bench::make_row(
                    args,
                    operation.name,
                    cvh_bench::opencv_intrin_backend_name(),
                    "public_headers_fast_cvtColor",
                    allocation_mode,
                    shape,
                    public_entry,
                    scalar.median_ms));
                rows.push_back(cvh_bench::make_row(
                    args,
                    operation.name,
                    cvh_bench::opencv_intrin_backend_name(),
                    "direct_detail",
                    allocation_mode,
                    shape,
                    direct_entry,
                    scalar.median_ms));
            }
        }

        for (const auto& micro : micro_cases)
        {
            const auto result = cvh_bench::measure_micro(
                micro.fn,
                src,
                planar,
                shape,
                cvh_bench::AllocationMode::Reuse,
                args.warmup,
                args.iters,
                args.repeats);
            rows.push_back(cvh_bench::make_micro_row(
                args,
                micro,
                cvh_bench::AllocationMode::Reuse,
                shape,
                result));
        }
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
