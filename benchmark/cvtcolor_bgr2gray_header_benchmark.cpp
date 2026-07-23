#include "cvh.h"
#include "cvh/core/simd/opencv_ui.h"
#include "common/benchmark_common.h"

#if !CVH_ENABLE_OPENCV_INTRIN
#error "cvh_benchmark_cvtcolor_bgr2gray_header must be compiled with CVH_ENABLE_OPENCV_INTRIN=1"
#endif

#include <algorithm>
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

using Result = common::BenchmarkResult;

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
    const auto parsed = common::parse_basic_args(
        argc,
        argv,
        common::BasicArgs {"quick", 3, 10, 7, ""},
        {"quick", "full"},
        usage);
    return Args {parsed.profile, parsed.warmup, parsed.iters, parsed.repeats, parsed.output_csv};
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
    return common::opencv_intrin_backend_name();
}

std::string allocation_mode_name(AllocationMode mode)
{
    return mode == AllocationMode::Reuse ? "reuse" : "recreate";
}

int simd_lanes()
{
    return common::simd_u8_lanes();
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
    common::fill_mat_u8_lcg(mat, seed);
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
    return common::checksum_mat_bytes(mat);
}

bool same_mat(const cvh::Mat& lhs, const cvh::Mat& rhs)
{
    return common::same_mat_bytes(lhs, rhs);
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
    cvh::detail::cvtcolor_bgr2gray_u8_opencv_intrin_impl(src, dst);
}

void run_rgb_opencv_intrin_direct(const cvh::Mat& src, cvh::Mat& dst)
{
    cvh::detail::cvtcolor_rgb2gray_u8_opencv_intrin_impl(src, dst);
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
    const auto timing = common::measure_repeated_ms(
        [&]() {
            prepare_dst(dst, src, allocation_mode);
            fn(src, dst);
        },
        warmup,
        iters,
        repeats);
    const std::uint64_t hash = checksum(dst);
    g_sink ^= hash;

    return Result {timing.min_ms, timing.median_ms, hash};
}

void run_micro_store_u8(const cvh::Mat&, const cvh::Mat&, const ShapeCase& shape, cvh::Mat& dst)
{
    dst.create(std::vector<int>{shape.rows, shape.cols}, CV_8UC1);
    const int lanes = simd_lanes();
    const cv::v_uint16x8 wide = cv::v_setall_u16(static_cast<ushort>(123));
    const cv::v_uint8x16 value = cv::v_pack(wide, wide);

    for (int y = 0; y < shape.rows; ++y)
    {
        std::uint8_t* dst_row = reinterpret_cast<std::uint8_t*>(dst.data + static_cast<std::size_t>(y) * dst.step(0));
        int x = 0;
        for (; x + lanes <= shape.cols; x += lanes)
        {
            cv::v_store(reinterpret_cast<uchar*>(dst_row + x), value);
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
            cv::v_uint8x16 c0;
            cv::v_uint8x16 c1;
            cv::v_uint8x16 c2;
            cv::v_load_deinterleave(
                reinterpret_cast<const uchar*>(src_row + static_cast<std::size_t>(x) * 3),
                c0,
                c1,
                c2);
            cv::v_store(reinterpret_cast<uchar*>(dst_row + x), c0);
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
    dst.create(std::vector<int>{shape.rows, shape.cols}, CV_8UC1);
    const int lanes = simd_lanes();
    const std::size_t plane_size = static_cast<std::size_t>(shape.rows) * static_cast<std::size_t>(shape.cols);
    const std::uint8_t* b_plane = reinterpret_cast<const std::uint8_t*>(planar.data);
    const std::uint8_t* g_plane = b_plane + plane_size;
    const std::uint8_t* r_plane = g_plane + plane_size;
    const cv::v_uint16x8 weight_b = cv::v_setall_u16(static_cast<ushort>(7471));
    const cv::v_uint16x8 weight_g = cv::v_setall_u16(static_cast<ushort>(38470));
    const cv::v_uint16x8 weight_r = cv::v_setall_u16(static_cast<ushort>(19595));

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
            const cv::v_uint8x16 b8 = cv::v_load(reinterpret_cast<const uchar*>(b_row + x));
            const cv::v_uint8x16 g8 = cv::v_load(reinterpret_cast<const uchar*>(g_row + x));
            const cv::v_uint8x16 r8 = cv::v_load(reinterpret_cast<const uchar*>(r_row + x));

            cv::v_uint16x8 b16_lo;
            cv::v_uint16x8 b16_hi;
            cv::v_uint16x8 g16_lo;
            cv::v_uint16x8 g16_hi;
            cv::v_uint16x8 r16_lo;
            cv::v_uint16x8 r16_hi;
            cv::v_expand(b8, b16_lo, b16_hi);
            cv::v_expand(g8, g16_lo, g16_hi);
            cv::v_expand(r8, r16_lo, r16_hi);

            const cv::v_uint16x8 y16_lo = cvh::detail::cvtcolor_bgr2gray_u8_wide_opencv_intrin(
                b16_lo, g16_lo, r16_lo, weight_b, weight_g, weight_r);
            const cv::v_uint16x8 y16_hi = cvh::detail::cvtcolor_bgr2gray_u8_wide_opencv_intrin(
                b16_hi, g16_hi, r16_hi, weight_b, weight_g, weight_r);
            cv::v_store(reinterpret_cast<uchar*>(dst_row + x), cv::v_pack(y16_lo, y16_hi));
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
    const auto timing = common::measure_repeated_ms(
        [&]() {
            prepare_dst(dst, shape, allocation_mode);
            fn(bgr, planar, shape, dst);
        },
        warmup,
        iters,
        repeats);
    const std::uint64_t hash = checksum(dst);
    g_sink ^= hash;

    return Result {timing.min_ms, timing.median_ms, hash};
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
