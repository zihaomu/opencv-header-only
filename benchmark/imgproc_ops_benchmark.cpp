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

enum class ImgprocOp
{
    ResizeNearest = 0,
    ResizeLinear,
    ResizeNearestF32,
    ResizeLinearF32,
    CvtColorBgr2Gray,
    CvtColorGray2Bgr,
    ThresholdBinary,
    ThresholdBinaryF32,
    BoxFilter3x3,
    BoxFilter3x3F32,
    GaussianBlur5x5,
    GaussianBlur5x5F32,
};

struct Args
{
    std::string profile = "quick";
    int warmup = 2;
    int iters = 20;
    int repeats = 5;
    std::string output_csv;
};

struct ShapeCase
{
    std::string name;
    int rows = 0;
    int cols = 0;
};

struct ResultRow
{
    std::string profile;
    std::string op;
    std::string depth;
    int channels = 0;
    std::string shape;
    std::size_t elements = 0;
    double ms_per_iter = 0.0;
    double melems_per_sec = 0.0;
    double gb_per_sec = 0.0;
};

struct BenchCase
{
    ImgprocOp op;
    ShapeCase shape;
    int channels = 0;
};

struct CaseData
{
    cvh::Mat src;
    cvh::Mat dst;
    cvh::Size dsize;
    double thresh = 120.0;
    double maxval = 255.0;
    int interpolation = cvh::INTER_LINEAR;
    int color_code = cvh::COLOR_BGR2GRAY;
    int threshold_type = cvh::THRESH_BINARY;
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

const char* op_name(ImgprocOp op)
{
    switch (op)
    {
        case ImgprocOp::ResizeNearest: return "RESIZE_NEAREST";
        case ImgprocOp::ResizeLinear: return "RESIZE_LINEAR";
        case ImgprocOp::ResizeNearestF32: return "RESIZE_NEAREST_F32";
        case ImgprocOp::ResizeLinearF32: return "RESIZE_LINEAR_F32";
        case ImgprocOp::CvtColorBgr2Gray: return "CVTCOLOR_BGR2GRAY";
        case ImgprocOp::CvtColorGray2Bgr: return "CVTCOLOR_GRAY2BGR";
        case ImgprocOp::ThresholdBinary: return "THRESH_BINARY";
        case ImgprocOp::ThresholdBinaryF32: return "THRESH_BINARY_F32";
        case ImgprocOp::BoxFilter3x3: return "BOXFILTER_3X3";
        case ImgprocOp::BoxFilter3x3F32: return "BOXFILTER_3X3_F32";
        case ImgprocOp::GaussianBlur5x5: return "GAUSSIAN_5X5";
        case ImgprocOp::GaussianBlur5x5F32: return "GAUSSIAN_5X5_F32";
    }
    return "UNKNOWN";
}

std::vector<ShapeCase> build_shapes(const std::string& profile)
{
    if (profile == "full")
    {
        return {
            {"vga", 480, 640},
            {"hd", 720, 1280},
            {"fhd", 1080, 1920},
        };
    }

    return {
        {"vga", 480, 640},
        {"hd", 720, 1280},
    };
}

std::vector<BenchCase> build_cases(const std::string& profile)
{
    const auto shapes = build_shapes(profile);
    std::vector<BenchCase> cases;
    cases.reserve(shapes.size() * 32);

    for (const auto& shape : shapes)
    {
        for (const int cn : {1, 3, 4})
        {
            cases.push_back({ImgprocOp::ResizeNearest, shape, cn});
            cases.push_back({ImgprocOp::ResizeLinear, shape, cn});
            cases.push_back({ImgprocOp::BoxFilter3x3, shape, cn});
            cases.push_back({ImgprocOp::GaussianBlur5x5, shape, cn});
            cases.push_back({ImgprocOp::ResizeNearestF32, shape, cn});
            cases.push_back({ImgprocOp::ResizeLinearF32, shape, cn});
            cases.push_back({ImgprocOp::BoxFilter3x3F32, shape, cn});
            cases.push_back({ImgprocOp::GaussianBlur5x5F32, shape, cn});
        }

        cases.push_back({ImgprocOp::CvtColorBgr2Gray, shape, 3});
        cases.push_back({ImgprocOp::CvtColorGray2Bgr, shape, 1});
        cases.push_back({ImgprocOp::ThresholdBinary, shape, 1});
        cases.push_back({ImgprocOp::ThresholdBinary, shape, 3});
        cases.push_back({ImgprocOp::ThresholdBinaryF32, shape, 1});
        cases.push_back({ImgprocOp::ThresholdBinaryF32, shape, 3});
        cases.push_back({ImgprocOp::ThresholdBinaryF32, shape, 4});
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
                << "Usage: cvh_benchmark_imgproc_ops [--profile quick|full] [--warmup N] [--iters N] [--repeats N] [--output path]\n";
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

std::string case_shape_string(const BenchCase& c, const CaseData& d)
{
    std::ostringstream oss;
    if (c.op == ImgprocOp::ResizeNearest ||
        c.op == ImgprocOp::ResizeLinear ||
        c.op == ImgprocOp::ResizeNearestF32 ||
        c.op == ImgprocOp::ResizeLinearF32)
    {
        oss << c.shape.rows << "x" << c.shape.cols << "->" << d.dsize.height << "x" << d.dsize.width;
    }
    else
    {
        oss << c.shape.rows << "x" << c.shape.cols;
    }
    return oss.str();
}

void fill_u8(cvh::Mat& mat, std::uint32_t seed)
{
    const std::size_t count = mat.total() * static_cast<std::size_t>(mat.channels());
    for (std::size_t i = 0; i < count; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        mat.data[i] = static_cast<uchar>((seed >> 16) & 0xFFu);
    }
}

void fill_f32(cvh::Mat& mat, std::uint32_t seed)
{
    const std::size_t count = mat.total() * static_cast<std::size_t>(mat.channels());
    float* ptr = reinterpret_cast<float*>(mat.data);
    for (std::size_t i = 0; i < count; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        const float unit = static_cast<float>((seed >> 8) & 0xFFFFFFu) / static_cast<float>(0xFFFFFFu);
        ptr[i] = unit * 8.0f - 4.0f;
    }
}

double probe_checksum(const cvh::Mat& mat)
{
    const std::size_t count = mat.total() * static_cast<std::size_t>(mat.channels());
    if (count == 0)
    {
        return 0.0;
    }

    const std::size_t i0 = 0;
    const std::size_t i1 = count / 2;
    const std::size_t i2 = count - 1;
    return static_cast<double>(mat.data[i0]) + static_cast<double>(mat.data[i1]) + static_cast<double>(mat.data[i2]);
}

void prepare_case_data(const BenchCase& c, CaseData& d)
{
    int src_type = CV_MAKETYPE(CV_8U, c.channels);
    if (c.op == ImgprocOp::ThresholdBinaryF32 ||
        c.op == ImgprocOp::ResizeNearestF32 ||
        c.op == ImgprocOp::ResizeLinearF32 ||
        c.op == ImgprocOp::BoxFilter3x3F32 ||
        c.op == ImgprocOp::GaussianBlur5x5F32)
    {
        src_type = CV_MAKETYPE(CV_32F, c.channels);
    }
    d.src.create(std::vector<int>{c.shape.rows, c.shape.cols}, src_type);
    const std::uint32_t seed = static_cast<std::uint32_t>(c.shape.rows * 131 + c.shape.cols * 17 + c.channels * 31 + static_cast<int>(c.op) * 7);
    if (d.src.depth() == CV_8U)
    {
        fill_u8(d.src, seed);
    }
    else
    {
        fill_f32(d.src, seed);
    }

    switch (c.op)
    {
        case ImgprocOp::ResizeNearest:
        case ImgprocOp::ResizeNearestF32:
            d.interpolation = cvh::INTER_NEAREST;
            d.dsize = cvh::Size(std::max(1, c.shape.cols / 2), std::max(1, c.shape.rows / 2));
            break;
        case ImgprocOp::ResizeLinear:
        case ImgprocOp::ResizeLinearF32:
            d.interpolation = cvh::INTER_LINEAR;
            d.dsize = cvh::Size(std::max(1, c.shape.cols / 2), std::max(1, c.shape.rows / 2));
            break;
        case ImgprocOp::CvtColorBgr2Gray:
            d.color_code = cvh::COLOR_BGR2GRAY;
            break;
        case ImgprocOp::CvtColorGray2Bgr:
            d.color_code = cvh::COLOR_GRAY2BGR;
            break;
        case ImgprocOp::ThresholdBinary:
            d.thresh = 120.0;
            d.maxval = 255.0;
            d.threshold_type = cvh::THRESH_BINARY;
            break;
        case ImgprocOp::ThresholdBinaryF32:
            d.thresh = 0.2;
            d.maxval = 3.0;
            d.threshold_type = cvh::THRESH_BINARY;
            break;
        case ImgprocOp::BoxFilter3x3:
        case ImgprocOp::BoxFilter3x3F32:
            break;
        case ImgprocOp::GaussianBlur5x5:
        case ImgprocOp::GaussianBlur5x5F32:
            break;
    }
}

void run_one_op(const BenchCase& c, CaseData& d)
{
    switch (c.op)
    {
        case ImgprocOp::ResizeNearest:
        case ImgprocOp::ResizeLinear:
        case ImgprocOp::ResizeNearestF32:
        case ImgprocOp::ResizeLinearF32:
            cvh::resize(d.src, d.dst, d.dsize, 0.0, 0.0, d.interpolation);
            break;
        case ImgprocOp::CvtColorBgr2Gray:
        case ImgprocOp::CvtColorGray2Bgr:
            cvh::cvtColor(d.src, d.dst, d.color_code);
            break;
        case ImgprocOp::ThresholdBinary:
        case ImgprocOp::ThresholdBinaryF32:
            cvh::threshold(d.src, d.dst, d.thresh, d.maxval, d.threshold_type);
            break;
        case ImgprocOp::BoxFilter3x3:
        case ImgprocOp::BoxFilter3x3F32:
            cvh::boxFilter(d.src, d.dst, -1, cvh::Size(3, 3), cvh::Point(-1, -1), true, cvh::BORDER_REPLICATE);
            break;
        case ImgprocOp::GaussianBlur5x5:
        case ImgprocOp::GaussianBlur5x5F32:
            cvh::GaussianBlur(d.src, d.dst, cvh::Size(5, 5), 1.1, 0.0, cvh::BORDER_REPLICATE);
            break;
    }
}

double measure_case(const BenchCase& c, CaseData& d, int warmup, int iters, int repeats)
{
    for (int i = 0; i < warmup; ++i)
    {
        run_one_op(c, d);
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));

    using Clock = std::chrono::steady_clock;
    for (int r = 0; r < repeats; ++r)
    {
        const auto t0 = Clock::now();
        for (int i = 0; i < iters; ++i)
        {
            run_one_op(c, d);
        }
        const auto t1 = Clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        samples.push_back(elapsed_ms / static_cast<double>(iters));
    }

    std::sort(samples.begin(), samples.end());
    const double median_ms = samples[samples.size() / 2];
    g_sink += probe_checksum(d.dst);
    return median_ms;
}

std::size_t bytes_of_mat(const cvh::Mat& m)
{
    return m.total() * static_cast<std::size_t>(m.channels()) * static_cast<std::size_t>(m.elemSize1());
}

void print_csv(const std::vector<ResultRow>& rows, std::ostream& os)
{
    os << "profile,op,depth,channels,shape,elements,ms_per_iter,melems_per_sec,gb_per_sec\n";
    os << std::fixed << std::setprecision(6);
    for (const auto& row : rows)
    {
        os << row.profile << ","
           << row.op << ","
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
    const auto cases = cvh_bench::build_cases(args.profile);

    std::vector<cvh_bench::ResultRow> rows;
    rows.reserve(cases.size());

    for (const auto& c : cases)
    {
        cvh_bench::CaseData d;
        cvh_bench::prepare_case_data(c, d);

        const double ms_per_iter = cvh_bench::measure_case(c, d, args.warmup, args.iters, args.repeats);
        const std::size_t elements = d.dst.total() * static_cast<std::size_t>(d.dst.channels());
        const double sec = ms_per_iter / 1000.0;
        const std::size_t bytes_per_iter = cvh_bench::bytes_of_mat(d.src) + cvh_bench::bytes_of_mat(d.dst);
        const double melems_per_sec = sec > 0.0 ? (elements / sec / 1e6) : 0.0;
        const double gb_per_sec = sec > 0.0 ? (bytes_per_iter / sec / 1e9) : 0.0;

        rows.push_back({
            args.profile,
            cvh_bench::op_name(c.op),
            cvh_bench::depth_to_name(d.src.depth()),
            c.channels,
            cvh_bench::case_shape_string(c, d),
            elements,
            ms_per_iter,
            melems_per_sec,
            gb_per_sec,
        });
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
