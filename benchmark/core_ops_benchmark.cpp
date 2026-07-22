#include "cvh.h"
#include "cvh/core/detail/dispatch_control.h"

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

using cvh::BinaryOp;

enum class BenchOp
{
    Add = 0,
    Sub,
    Mul,
    Div,
    Mean,
    Max,
    Min,
    And,
    Or,
    Xor,
    Not,
    Mod,
    Bitshift,
    Fmod,
    Atan2,
    Hypot,
    CmpEq,
    CmpGt,
    CmpGe,
    CmpLt,
    CmpLe,
    CmpNe
};

struct Args
{
    std::string profile = "quick";
    std::string benchmark_mode = "matmat";
    std::string dispatch_mode = "auto";
    std::string scalar_pattern = "both";
    std::string scalar_order = "both";
    int warmup = 2;
    int iters = 20;
    int repeats = 5;
    std::string output_csv;
};

enum class ScalarBenchOp
{
    Add = 0,
    Sub,
    Mul,
    Div,
    CmpGt
};

struct ScalarPatternCase
{
    std::string name;
    bool nonuniform = false;
};

struct ScalarOrderCase
{
    std::string name;
    bool scalar_first = false;
};

struct ShapeCase
{
    std::string name;
    std::vector<int> dims;
};

struct ResultRow
{
    std::string profile;
    std::string op;
    std::string dispatch;
    std::string depth;
    int channels = 0;
    std::string shape;
    std::size_t elements = 0;
    double ms_per_iter = 0.0;
    double melems_per_sec = 0.0;
    double gb_per_sec = 0.0;
};

struct MeasureResult
{
    double ms_per_iter = 0.0;
    std::string dispatch = "unknown";
};

volatile double g_sink = 0.0;

std::string depth_to_name(int depth)
{
    switch (depth)
    {
        case CV_8U: return "CV_8U";
        case CV_8S: return "CV_8S";
        case CV_16U: return "CV_16U";
        case CV_16S: return "CV_16S";
        case CV_32S: return "CV_32S";
        case CV_32U: return "CV_32U";
        case CV_32F: return "CV_32F";
        case CV_16F: return "CV_16F";
        case CV_64F: return "CV_64F";
        default: return "UNKNOWN";
    }
}

std::string shape_to_string(const std::vector<int>& dims)
{
    std::ostringstream oss;
    for (std::size_t i = 0; i < dims.size(); ++i)
    {
        if (i > 0)
        {
            oss << "x";
        }
        oss << dims[i];
    }
    return oss.str();
}

const char* op_name(BenchOp op)
{
    switch (op)
    {
        case BenchOp::Add: return "ADD";
        case BenchOp::Sub: return "SUB";
        case BenchOp::Mul: return "MUL";
        case BenchOp::Div: return "DIV";
        case BenchOp::Mean: return "MEAN";
        case BenchOp::Max: return "MAX";
        case BenchOp::Min: return "MIN";
        case BenchOp::And: return "AND";
        case BenchOp::Or: return "OR";
        case BenchOp::Xor: return "XOR";
        case BenchOp::Not: return "NOT";
        case BenchOp::Mod: return "MOD";
        case BenchOp::Bitshift: return "BITSHIFT";
        case BenchOp::Fmod: return "FMOD";
        case BenchOp::Atan2: return "ATAN2";
        case BenchOp::Hypot: return "HYPOT";
        case BenchOp::CmpEq: return "CMP_EQ";
        case BenchOp::CmpGt: return "CMP_GT";
        case BenchOp::CmpGe: return "CMP_GE";
        case BenchOp::CmpLt: return "CMP_LT";
        case BenchOp::CmpLe: return "CMP_LE";
        case BenchOp::CmpNe: return "CMP_NE";
    }

    return "UNKNOWN";
}

const char* scalar_op_name(ScalarBenchOp op)
{
    switch (op)
    {
        case ScalarBenchOp::Add: return "ADD";
        case ScalarBenchOp::Sub: return "SUB";
        case ScalarBenchOp::Mul: return "MUL";
        case ScalarBenchOp::Div: return "DIV";
        case ScalarBenchOp::CmpGt: return "CMP_GT";
    }
    return "UNKNOWN";
}

bool scalar_op_is_compare(ScalarBenchOp op)
{
    return op == ScalarBenchOp::CmpGt;
}

bool scalar_op_supported_for_depth(ScalarBenchOp op, int depth)
{
    (void)op;
    return depth != CV_64F;
}

BinaryOp to_binary_op(BenchOp op)
{
    switch (op)
    {
        case BenchOp::Add: return BinaryOp::ADD;
        case BenchOp::Sub: return BinaryOp::SUB;
        case BenchOp::Mul: return BinaryOp::MUL;
        case BenchOp::Div: return BinaryOp::DIV;
        case BenchOp::Mean: return BinaryOp::MEAN;
        case BenchOp::Max: return BinaryOp::MAX;
        case BenchOp::Min: return BinaryOp::MIN;
        case BenchOp::And: return BinaryOp::AND;
        case BenchOp::Or: return BinaryOp::OR;
        case BenchOp::Xor: return BinaryOp::XOR;
        case BenchOp::Not: return BinaryOp::NOT;
        case BenchOp::Mod: return BinaryOp::MOD;
        case BenchOp::Bitshift: return BinaryOp::BITSHIFT;
        case BenchOp::Fmod: return BinaryOp::FMOD;
        case BenchOp::Atan2: return BinaryOp::ATAN2;
        case BenchOp::Hypot: return BinaryOp::HYPOT;
        case BenchOp::CmpEq: return BinaryOp::EQUAL;
        case BenchOp::CmpGt: return BinaryOp::GREATER;
        case BenchOp::CmpGe: return BinaryOp::GREATER_EQUAL;
        case BenchOp::CmpLt: return BinaryOp::LESS;
        case BenchOp::CmpLe: return BinaryOp::LESS_EQUAL;
        case BenchOp::CmpNe: return BinaryOp::EQUAL;
    }

    return BinaryOp::ADD;
}

bool op_is_compare(BenchOp op)
{
    switch (op)
    {
        case BenchOp::CmpEq:
        case BenchOp::CmpGt:
        case BenchOp::CmpGe:
        case BenchOp::CmpLt:
        case BenchOp::CmpLe:
        case BenchOp::CmpNe:
            return true;
        default:
            return false;
    }
}

int compare_code(BenchOp op)
{
    switch (op)
    {
        case BenchOp::CmpEq: return CV_CMP_EQ;
        case BenchOp::CmpGt: return CV_CMP_GT;
        case BenchOp::CmpGe: return CV_CMP_GE;
        case BenchOp::CmpLt: return CV_CMP_LT;
        case BenchOp::CmpLe: return CV_CMP_LE;
        case BenchOp::CmpNe: return CV_CMP_NE;
        default:
            return CV_CMP_EQ;
    }
}

bool is_integral_depth(int depth)
{
    return depth == CV_8U || depth == CV_8S || depth == CV_16U || depth == CV_16S ||
           depth == CV_32S || depth == CV_32U;
}

bool is_unsigned_integral_depth(int depth)
{
    return depth == CV_8U || depth == CV_16U || depth == CV_32U;
}

bool is_float_depth(int depth)
{
    return depth == CV_32F || depth == CV_16F || depth == CV_64F;
}

bool op_supported_for_depth(BenchOp op, int depth)
{
    if (op_is_compare(op))
    {
        // Current compare(Mat,Mat) kernel supports [CV_8U..CV_16F], excludes CV_64F.
        return depth != CV_64F;
    }

    switch (op)
    {
        case BenchOp::And:
        case BenchOp::Or:
        case BenchOp::Xor:
        case BenchOp::Not:
        case BenchOp::Mod:
        case BenchOp::Bitshift:
            return is_integral_depth(depth);
        case BenchOp::Fmod:
        case BenchOp::Atan2:
        case BenchOp::Hypot:
            return is_float_depth(depth);
        default:
            return true;
    }
}

std::string scalar_case_op_label(ScalarBenchOp op, const ScalarPatternCase& pattern, const ScalarOrderCase& order)
{
    return std::string(scalar_op_name(op)) + "_MS_" + pattern.name + "_" + order.name;
}

double seed_value(std::size_t idx, int depth, bool rhs, BenchOp op)
{
    if (is_float_depth(depth))
    {
        double base = rhs ? static_cast<double>((idx * 17u) % 401u) * 0.05 - 10.0
                          : static_cast<double>((idx * 131u) % 401u) * 0.05 - 10.0;
        if (op == BenchOp::Bitshift)
        {
            base = static_cast<double>((idx % 4u) + 1u);
        }
        if (rhs && (op == BenchOp::Div || op == BenchOp::Mod || op == BenchOp::Fmod) &&
            std::abs(base) < 1e-6)
        {
            base = 1.25;
        }
        return base;
    }

    if (depth == CV_8U || depth == CV_16U || depth == CV_32U)
    {
        double base = rhs ? static_cast<double>((idx * 17u) % 251u + 1u)
                          : static_cast<double>((idx * 131u) % 251u + 1u);
        if (op == BenchOp::Bitshift && rhs)
        {
            base = static_cast<double>((idx % 4u) + 1u);
        }
        return base;
    }

    double base = rhs ? static_cast<double>((idx * 17u) % 255u) - 127.0
                      : static_cast<double>((idx * 131u) % 255u) - 127.0;
    if (op == BenchOp::Bitshift && rhs)
    {
        base = static_cast<double>((idx % 4u) + 1u);
    }
    if (rhs && (op == BenchOp::Div || op == BenchOp::Mod || op == BenchOp::Fmod) &&
        std::abs(base) < 1e-9)
    {
        base = 3.0;
    }
    return base;
}

void fill_mat(cvh::Mat& mat, bool rhs, BenchOp op)
{
    const std::size_t count = mat.total() * static_cast<std::size_t>(mat.channels());
    switch (mat.depth())
    {
        case CV_8U:
        {
            auto* ptr = reinterpret_cast<uchar*>(mat.data);
            for (std::size_t i = 0; i < count; ++i) ptr[i] = cvh::saturate_cast<uchar>(seed_value(i, CV_8U, rhs, op));
            break;
        }
        case CV_8S:
        {
            auto* ptr = reinterpret_cast<schar*>(mat.data);
            for (std::size_t i = 0; i < count; ++i) ptr[i] = cvh::saturate_cast<schar>(seed_value(i, CV_8S, rhs, op));
            break;
        }
        case CV_16U:
        {
            auto* ptr = reinterpret_cast<ushort*>(mat.data);
            for (std::size_t i = 0; i < count; ++i) ptr[i] = cvh::saturate_cast<ushort>(seed_value(i, CV_16U, rhs, op));
            break;
        }
        case CV_16S:
        {
            auto* ptr = reinterpret_cast<short*>(mat.data);
            for (std::size_t i = 0; i < count; ++i) ptr[i] = cvh::saturate_cast<short>(seed_value(i, CV_16S, rhs, op));
            break;
        }
        case CV_32S:
        {
            auto* ptr = reinterpret_cast<int*>(mat.data);
            for (std::size_t i = 0; i < count; ++i) ptr[i] = cvh::saturate_cast<int>(seed_value(i, CV_32S, rhs, op));
            break;
        }
        case CV_32U:
        {
            auto* ptr = reinterpret_cast<uint*>(mat.data);
            for (std::size_t i = 0; i < count; ++i) ptr[i] = cvh::saturate_cast<uint>(seed_value(i, CV_32U, rhs, op));
            break;
        }
        case CV_32F:
        {
            auto* ptr = reinterpret_cast<float*>(mat.data);
            for (std::size_t i = 0; i < count; ++i) ptr[i] = cvh::saturate_cast<float>(seed_value(i, CV_32F, rhs, op));
            break;
        }
        case CV_16F:
        {
            auto* ptr = reinterpret_cast<hfloat*>(mat.data);
            for (std::size_t i = 0; i < count; ++i) ptr[i] = cvh::saturate_cast<hfloat>(seed_value(i, CV_16F, rhs, op));
            break;
        }
        case CV_64F:
        {
            auto* ptr = reinterpret_cast<double*>(mat.data);
            for (std::size_t i = 0; i < count; ++i) ptr[i] = cvh::saturate_cast<double>(seed_value(i, CV_64F, rhs, op));
            break;
        }
        default:
            break;
    }
}

double probe_checksum(const cvh::Mat& mat)
{
    const std::size_t count = mat.total() * static_cast<std::size_t>(mat.channels());
    if (count == 0)
    {
        return 0.0;
    }

    const std::size_t idx0 = 0;
    const std::size_t idx1 = count / 2;
    const std::size_t idx2 = count - 1;

    auto sample = [&](std::size_t idx) -> double {
        switch (mat.depth())
        {
            case CV_8U: return static_cast<double>(reinterpret_cast<const uchar*>(mat.data)[idx]);
            case CV_8S: return static_cast<double>(reinterpret_cast<const schar*>(mat.data)[idx]);
            case CV_16U: return static_cast<double>(reinterpret_cast<const ushort*>(mat.data)[idx]);
            case CV_16S: return static_cast<double>(reinterpret_cast<const short*>(mat.data)[idx]);
            case CV_32S: return static_cast<double>(reinterpret_cast<const int*>(mat.data)[idx]);
            case CV_32U: return static_cast<double>(reinterpret_cast<const uint*>(mat.data)[idx]);
            case CV_32F: return static_cast<double>(reinterpret_cast<const float*>(mat.data)[idx]);
            case CV_16F: return static_cast<double>(static_cast<float>(reinterpret_cast<const hfloat*>(mat.data)[idx]));
            case CV_64F: return reinterpret_cast<const double*>(mat.data)[idx];
            default: return 0.0;
        }
    };

    return sample(idx0) + sample(idx1) + sample(idx2);
}

void run_one_op(BenchOp op, const cvh::Mat& a, const cvh::Mat& b, cvh::Mat& out)
{
    if (op_is_compare(op))
    {
        cvh::compare(a, b, out, compare_code(op));
        return;
    }
    cvh::binaryFunc(to_binary_op(op), a, b, out);
}

void run_one_scalar_op(ScalarBenchOp op,
                       const cvh::Mat& src,
                       const cvh::Scalar& scalar,
                       cvh::Mat& out,
                       bool scalar_first)
{
    switch (op)
    {
        case ScalarBenchOp::Add:
            if (scalar_first)
            {
                cvh::add(scalar, src, out);
            }
            else
            {
                cvh::add(src, scalar, out);
            }
            return;
        case ScalarBenchOp::Sub:
            if (scalar_first)
            {
                cvh::subtract(scalar, src, out);
            }
            else
            {
                cvh::subtract(src, scalar, out);
            }
            return;
        case ScalarBenchOp::Mul:
            if (scalar_first)
            {
                cvh::multiply(scalar, src, out);
            }
            else
            {
                cvh::multiply(src, scalar, out);
            }
            return;
        case ScalarBenchOp::Div:
            if (scalar_first)
            {
                cvh::divide(scalar, src, out);
            }
            else
            {
                cvh::divide(src, scalar, out);
            }
            return;
        case ScalarBenchOp::CmpGt:
            if (scalar_first)
            {
                cvh::compare(scalar, src, out, CV_CMP_GT);
            }
            else
            {
                cvh::compare(src, scalar, out, CV_CMP_GT);
            }
            return;
    }
}

cvh::Mat run_one_transpose(const cvh::Mat& src)
{
    return cvh::transpose(src);
}

std::vector<ShapeCase> build_shapes(const std::string& profile)
{
    if (profile == "full")
    {
        return {
            {"1d_1m", {1024 * 1024}},
            {"2d_vga", {480, 640}},
            {"2d_fhd", {1080, 1920}},
            {"3d_8x256x256", {8, 256, 256}},
        };
    }

    return {
        {"1d_1m", {1024 * 1024}},
        {"2d_hd", {720, 1280}},
        {"3d_8x256x256", {8, 256, 256}},
    };
}

std::vector<ShapeCase> build_transpose_shapes(const std::string& profile)
{
    if (profile == "full")
    {
        return {
            {"2d_vga", {480, 640}},
            {"2d_fhd", {1080, 1920}},
            {"3d_8x256x384", {8, 256, 384}},
            {"4d_4x8x128x192", {4, 8, 128, 192}},
        };
    }

    return {
        {"2d_hd", {720, 1280}},
        {"2d_tall", {1536, 512}},
        {"3d_8x256x384", {8, 256, 384}},
    };
}

std::vector<int> build_channels(const std::string& profile)
{
    if (profile == "full")
    {
        return {1, 3, 4};
    }
    return {1, 3};
}

std::vector<int> build_depths(const std::string& profile)
{
    if (profile == "full")
    {
        return {CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32U, CV_32F, CV_16F};
    }
    return {CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32U, CV_32F, CV_16F};
}

std::vector<BenchOp> build_ops(const std::string& profile)
{
    if (profile == "full")
    {
        return {
            BenchOp::Add, BenchOp::Sub, BenchOp::Mul, BenchOp::Div,
            BenchOp::Mean, BenchOp::Max, BenchOp::Min,
            BenchOp::And, BenchOp::Or, BenchOp::Xor, BenchOp::Not,
            BenchOp::Mod, BenchOp::Bitshift, BenchOp::Fmod,
            BenchOp::Atan2, BenchOp::Hypot,
            BenchOp::CmpEq, BenchOp::CmpGt, BenchOp::CmpGe,
            BenchOp::CmpLt, BenchOp::CmpLe, BenchOp::CmpNe};
    }
    return {
        BenchOp::Add, BenchOp::Sub, BenchOp::Mul, BenchOp::Div,
        BenchOp::Mean, BenchOp::Max, BenchOp::Min,
        BenchOp::And, BenchOp::Xor, BenchOp::Mod, BenchOp::Fmod,
        BenchOp::CmpEq, BenchOp::CmpGt, BenchOp::CmpGe,
        BenchOp::CmpLt, BenchOp::CmpLe, BenchOp::CmpNe};
}

std::vector<ScalarBenchOp> build_scalar_ops(const std::string& profile)
{
    if (profile == "full")
    {
        return {ScalarBenchOp::Add, ScalarBenchOp::Sub, ScalarBenchOp::Mul,
                ScalarBenchOp::Div, ScalarBenchOp::CmpGt};
    }
    return {ScalarBenchOp::Add, ScalarBenchOp::Sub, ScalarBenchOp::Mul,
            ScalarBenchOp::Div, ScalarBenchOp::CmpGt};
}

std::vector<ScalarPatternCase> build_scalar_patterns(const std::string& scalar_pattern)
{
    if (scalar_pattern == "uniform")
    {
        return {{"U", false}};
    }
    if (scalar_pattern == "nonuniform")
    {
        return {{"NU", true}};
    }
    return {{"U", false}, {"NU", true}};
}

std::vector<ScalarOrderCase> build_scalar_orders(const std::string& scalar_order)
{
    if (scalar_order == "mat_first")
    {
        return {{"MF", false}};
    }
    if (scalar_order == "scalar_first")
    {
        return {{"SF", true}};
    }
    return {{"MF", false}, {"SF", true}};
}

cvh::Scalar make_scalar_for_case(int depth, int channels, bool nonuniform)
{
    double lanes[4] = {0.0, 0.0, 0.0, 0.0};
    if (is_float_depth(depth) || depth == CV_64F)
    {
        lanes[0] = 1.25;
        lanes[1] = -0.75;
        lanes[2] = 0.5;
        lanes[3] = 2.0;
    }
    else if (is_unsigned_integral_depth(depth))
    {
        lanes[0] = 17.0;
        lanes[1] = 3.0;
        lanes[2] = 29.0;
        lanes[3] = 7.0;
    }
    else
    {
        lanes[0] = 5.0;
        lanes[1] = -3.0;
        lanes[2] = 11.0;
        lanes[3] = 2.0;
    }

    if (!nonuniform || channels <= 1)
    {
        lanes[1] = lanes[0];
        lanes[2] = lanes[0];
        lanes[3] = lanes[0];
    }

    return cvh::Scalar(lanes[0], lanes[1], lanes[2], lanes[3]);
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
        else if (token == "--bench")
        {
            args.benchmark_mode = next_value("--bench");
        }
        else if (token == "--dispatch")
        {
            args.dispatch_mode = next_value("--dispatch");
        }
        else if (token == "--scalar-pattern")
        {
            args.scalar_pattern = next_value("--scalar-pattern");
        }
        else if (token == "--scalar-order")
        {
            args.scalar_order = next_value("--scalar-order");
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
            std::cout
                << "Usage: cvh_benchmark_core_ops [--profile quick|full] [--bench matmat|matscalar|transpose|all] "
                   "[--dispatch auto|scalar-only] "
                   "[--scalar-pattern uniform|nonuniform|both] [--scalar-order mat_first|scalar_first|both] "
                   "[--warmup N] [--iters N] [--repeats N] [--output path]\n";
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

    if (args.benchmark_mode != "matmat" &&
        args.benchmark_mode != "matscalar" &&
        args.benchmark_mode != "transpose" &&
        args.benchmark_mode != "all")
    {
        std::cerr << "Unsupported bench mode: " << args.benchmark_mode
                  << " (expected matmat/matscalar/transpose/all)\n";
        std::exit(2);
    }

    if (args.scalar_pattern != "uniform" && args.scalar_pattern != "nonuniform" &&
        args.scalar_pattern != "both")
    {
        std::cerr << "Unsupported scalar-pattern: " << args.scalar_pattern
                  << " (expected uniform/nonuniform/both)\n";
        std::exit(2);
    }

    if (args.scalar_order != "mat_first" && args.scalar_order != "scalar_first" &&
        args.scalar_order != "both")
    {
        std::cerr << "Unsupported scalar-order: " << args.scalar_order
                  << " (expected mat_first/scalar_first/both)\n";
        std::exit(2);
    }

    if (args.dispatch_mode != "auto" &&
        args.dispatch_mode != "scalar-only")
    {
        std::cerr << "Unsupported dispatch mode: " << args.dispatch_mode
                  << " (expected auto/scalar-only)\n";
        std::exit(2);
    }

    return args;
}

MeasureResult measure_case(BenchOp op, const cvh::Mat& a, const cvh::Mat& b, cvh::Mat& out, int warmup, int iters, int repeats)
{
    cvh::cpu::reset_last_dispatch_tag();
    for (int i = 0; i < warmup; ++i)
    {
        run_one_op(op, a, b, out);
    }

    std::vector<double> samples_ms_per_iter;
    samples_ms_per_iter.reserve(static_cast<std::size_t>(repeats));

    using Clock = std::chrono::steady_clock;
    for (int r = 0; r < repeats; ++r)
    {
        const auto t0 = Clock::now();
        for (int i = 0; i < iters; ++i)
        {
            run_one_op(op, a, b, out);
        }
        const auto t1 = Clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        samples_ms_per_iter.push_back(elapsed_ms / static_cast<double>(iters));
    }

    std::sort(samples_ms_per_iter.begin(), samples_ms_per_iter.end());
    const double median_ms = samples_ms_per_iter[samples_ms_per_iter.size() / 2];
    g_sink += probe_checksum(out);
    return {median_ms, cvh::cpu::dispatch_tag_name(cvh::cpu::last_dispatch_tag())};
}

MeasureResult measure_scalar_case(ScalarBenchOp op,
                                  const cvh::Mat& src,
                                  const cvh::Scalar& scalar,
                                  cvh::Mat& out,
                                  bool scalar_first,
                                  int warmup,
                                  int iters,
                                  int repeats)
{
    cvh::cpu::reset_last_dispatch_tag();
    for (int i = 0; i < warmup; ++i)
    {
        run_one_scalar_op(op, src, scalar, out, scalar_first);
    }

    std::vector<double> samples_ms_per_iter;
    samples_ms_per_iter.reserve(static_cast<std::size_t>(repeats));

    using Clock = std::chrono::steady_clock;
    for (int r = 0; r < repeats; ++r)
    {
        const auto t0 = Clock::now();
        for (int i = 0; i < iters; ++i)
        {
            run_one_scalar_op(op, src, scalar, out, scalar_first);
        }
        const auto t1 = Clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        samples_ms_per_iter.push_back(elapsed_ms / static_cast<double>(iters));
    }

    std::sort(samples_ms_per_iter.begin(), samples_ms_per_iter.end());
    const double median_ms = samples_ms_per_iter[samples_ms_per_iter.size() / 2];
    g_sink += probe_checksum(out);
    return {median_ms, cvh::cpu::dispatch_tag_name(cvh::cpu::last_dispatch_tag())};
}

MeasureResult measure_transpose_case(const cvh::Mat& src, cvh::Mat& out, int warmup, int iters, int repeats)
{
    cvh::cpu::reset_last_dispatch_tag();
    for (int i = 0; i < warmup; ++i)
    {
        out = run_one_transpose(src);
    }

    std::vector<double> samples_ms_per_iter;
    samples_ms_per_iter.reserve(static_cast<std::size_t>(repeats));

    using Clock = std::chrono::steady_clock;
    for (int r = 0; r < repeats; ++r)
    {
        const auto t0 = Clock::now();
        for (int i = 0; i < iters; ++i)
        {
            out = run_one_transpose(src);
        }
        const auto t1 = Clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        samples_ms_per_iter.push_back(elapsed_ms / static_cast<double>(iters));
    }

    std::sort(samples_ms_per_iter.begin(), samples_ms_per_iter.end());
    const double median_ms = samples_ms_per_iter[samples_ms_per_iter.size() / 2];
    g_sink += probe_checksum(out);
    return {median_ms, cvh::cpu::dispatch_tag_name(cvh::cpu::last_dispatch_tag())};
}

void print_csv(const std::vector<ResultRow>& rows, std::ostream& os)
{
    os << "profile,op,dispatch,depth,channels,shape,elements,ms_per_iter,melems_per_sec,gb_per_sec\n";
    os << std::fixed << std::setprecision(6);
    for (const auto& row : rows)
    {
        os << row.profile << ","
           << row.op << ","
           << row.dispatch << ","
           << row.depth << ","
           << row.channels << ","
           << row.shape << ","
           << row.elements << ","
           << row.ms_per_iter << ","
           << row.melems_per_sec << ","
           << row.gb_per_sec << "\n";
    }
}

}  // namespace cvh_bench

int main(int argc, char** argv)
{
    const auto args = cvh_bench::parse_args(argc, argv);
    if (args.dispatch_mode == "scalar-only")
    {
        cvh::cpu::set_dispatch_mode(cvh::cpu::DispatchMode::ScalarOnly);
    }
    else
    {
        cvh::cpu::set_dispatch_mode(cvh::cpu::DispatchMode::Auto);
    }

    const auto shapes = cvh_bench::build_shapes(args.profile);
    const auto transpose_shapes = cvh_bench::build_transpose_shapes(args.profile);
    const auto channels = cvh_bench::build_channels(args.profile);
    const auto depths = cvh_bench::build_depths(args.profile);
    const auto ops = cvh_bench::build_ops(args.profile);
    const auto scalar_ops = cvh_bench::build_scalar_ops(args.profile);
    const auto scalar_patterns = cvh_bench::build_scalar_patterns(args.scalar_pattern);
    const auto scalar_orders = cvh_bench::build_scalar_orders(args.scalar_order);

    std::vector<cvh_bench::ResultRow> rows;
    rows.reserve(1024);

    if (args.benchmark_mode == "matmat" || args.benchmark_mode == "all")
    {
        for (const auto& shape_case : shapes)
        {
            for (const int cn : channels)
            {
                for (const int depth : depths)
                {
                    const int type = CV_MAKETYPE(depth, cn);
                    cvh::Mat a(shape_case.dims, type);
                    cvh::Mat b(shape_case.dims, type);

                    for (const auto op : ops)
                    {
                        if (!cvh_bench::op_supported_for_depth(op, depth))
                        {
                            continue;
                        }

                        const int out_depth = cvh_bench::op_is_compare(op) ? CV_8U : depth;
                        const int out_type = CV_MAKETYPE(out_depth, cn);
                        cvh::Mat out(shape_case.dims, out_type);

                        cvh_bench::fill_mat(a, false, op);
                        cvh_bench::fill_mat(b, true, op);

                        cvh_bench::MeasureResult measure;
                        try
                        {
                            measure = cvh_bench::measure_case(op, a, b, out, args.warmup, args.iters, args.repeats);
                        }
                        catch (const cvh::Exception&)
                        {
                            continue;
                        }
                        const std::size_t elements = out.total() * static_cast<std::size_t>(out.channels());
                        const std::size_t in_bytes =
                            elements * static_cast<std::size_t>(CV_ELEM_SIZE1(type)) * 2u;
                        const std::size_t out_bytes =
                            elements * static_cast<std::size_t>(CV_ELEM_SIZE1(out_type));
                        const std::size_t bytes_per_iter = in_bytes + out_bytes;
                        const double sec = measure.ms_per_iter / 1000.0;
                        const double melems_per_sec = elements / sec / 1e6;
                        const double gb_per_sec = bytes_per_iter / sec / 1e9;

                        rows.push_back({
                            args.profile,
                            cvh_bench::op_name(op),
                            measure.dispatch,
                            cvh_bench::depth_to_name(depth),
                            cn,
                            cvh_bench::shape_to_string(shape_case.dims),
                            elements,
                            measure.ms_per_iter,
                            melems_per_sec,
                            gb_per_sec,
                        });
                    }
                }
            }
        }
    }

    if (args.benchmark_mode == "matscalar" || args.benchmark_mode == "all")
    {
        for (const auto& shape_case : shapes)
        {
            for (const int cn : channels)
            {
                for (const int depth : depths)
                {
                    const int src_type = CV_MAKETYPE(depth, cn);
                    cvh::Mat src(shape_case.dims, src_type);
                    cvh_bench::fill_mat(src, false, cvh_bench::BenchOp::Add);

                    for (const auto& pattern_case : scalar_patterns)
                    {
                        if (pattern_case.nonuniform && cn <= 1)
                        {
                            continue;
                        }
                        const cvh::Scalar scalar = cvh_bench::make_scalar_for_case(depth, cn, pattern_case.nonuniform);
                        for (const auto& order_case : scalar_orders)
                        {
                            for (const auto op : scalar_ops)
                            {
                                if (!cvh_bench::scalar_op_supported_for_depth(op, depth))
                                {
                                    continue;
                                }
                                const bool is_compare = cvh_bench::scalar_op_is_compare(op);
                                const int out_depth = is_compare ? CV_8U : depth;
                                const int out_type = CV_MAKETYPE(out_depth, cn);
                                cvh::Mat out(shape_case.dims, out_type);

                                cvh_bench::MeasureResult measure;
                                try
                                {
                                    measure = cvh_bench::measure_scalar_case(op,
                                                                            src,
                                                                            scalar,
                                                                            out,
                                                                            order_case.scalar_first,
                                                                            args.warmup,
                                                                            args.iters,
                                                                            args.repeats);
                                }
                                catch (const cvh::Exception&)
                                {
                                    continue;
                                }
                                const std::size_t elements = out.total() * static_cast<std::size_t>(out.channels());
                                const std::size_t src_bytes =
                                    src.total() * static_cast<std::size_t>(src.channels()) *
                                    static_cast<std::size_t>(CV_ELEM_SIZE1(src_type));
                                const std::size_t out_bytes =
                                    out.total() * static_cast<std::size_t>(out.channels()) *
                                    static_cast<std::size_t>(CV_ELEM_SIZE1(out_type));
                                const std::size_t bytes_per_iter = src_bytes + out_bytes;
                                const double sec = measure.ms_per_iter / 1000.0;
                                const double melems_per_sec = elements / sec / 1e6;
                                const double gb_per_sec = bytes_per_iter / sec / 1e9;

                                rows.push_back({
                                    args.profile,
                                    cvh_bench::scalar_case_op_label(op, pattern_case, order_case),
                                    measure.dispatch,
                                    cvh_bench::depth_to_name(depth),
                                    cn,
                                    cvh_bench::shape_to_string(shape_case.dims),
                                    elements,
                                    measure.ms_per_iter,
                                    melems_per_sec,
                                    gb_per_sec,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    if (args.benchmark_mode == "transpose" || args.benchmark_mode == "all")
    {
        for (const auto& shape_case : transpose_shapes)
        {
            for (const int cn : channels)
            {
                for (const int depth : depths)
                {
                    const int type = CV_MAKETYPE(depth, cn);
                    cvh::Mat src(shape_case.dims, type);
                    cvh_bench::fill_mat(src, false, cvh_bench::BenchOp::Add);
                    cvh::Mat out;

                    cvh_bench::MeasureResult measure;
                    try
                    {
                        measure = cvh_bench::measure_transpose_case(src, out, args.warmup, args.iters, args.repeats);
                    }
                    catch (const cvh::Exception&)
                    {
                        continue;
                    }
                    const std::size_t elements = out.total() * static_cast<std::size_t>(out.channels());
                    const std::size_t bytes_per_iter =
                        elements * static_cast<std::size_t>(CV_ELEM_SIZE1(type)) * 2u;
                    const double sec = measure.ms_per_iter / 1000.0;
                    const double melems_per_sec = elements / sec / 1e6;
                    const double gb_per_sec = bytes_per_iter / sec / 1e9;

                    rows.push_back({
                        args.profile,
                        "TRANSPOSE",
                        measure.dispatch,
                        cvh_bench::depth_to_name(depth),
                        cn,
                        cvh_bench::shape_to_string(shape_case.dims),
                        elements,
                        measure.ms_per_iter,
                        melems_per_sec,
                        gb_per_sec,
                    });
                }
            }
        }
    }

    cvh_bench::print_csv(rows, std::cout);

    if (!args.output_csv.empty())
    {
        std::ofstream ofs(args.output_csv);
        if (!ofs.is_open())
        {
            std::cerr << "Failed to open output file: " << args.output_csv << "\n";
            return 3;
        }
        cvh_bench::print_csv(rows, ofs);
    }

    if (cvh_bench::g_sink == -1.0)
    {
        std::cerr << "unreachable\n";
    }

    return 0;
}
struct MeasureResult
{
    double ms_per_iter = 0.0;
    std::string dispatch = "unknown";
};
