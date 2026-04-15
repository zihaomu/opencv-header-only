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
    CvtColorBgr2Rgb,
    CvtColorBgr2Bgra,
    CvtColorBgra2Bgr,
    CvtColorRgb2Rgba,
    CvtColorRgba2Rgb,
    CvtColorBgr2Rgba,
    CvtColorRgba2Bgr,
    CvtColorRgb2Bgra,
    CvtColorBgra2Rgb,
    CvtColorBgra2Rgba,
    CvtColorRgba2Bgra,
    CvtColorGray2Bgra,
    CvtColorBgra2Gray,
    CvtColorGray2Rgba,
    CvtColorRgba2Gray,
    CvtColorBgr2Yuv,
    CvtColorYuv2Bgr,
    CvtColorRgb2Yuv,
    CvtColorYuv2Rgb,
    CvtColorBgr2YuvNv12,
    CvtColorRgb2YuvNv12,
    CvtColorBgr2YuvNv21,
    CvtColorRgb2YuvNv21,
    CvtColorBgr2YuvI420,
    CvtColorRgb2YuvI420,
    CvtColorBgr2YuvYv12,
    CvtColorRgb2YuvYv12,
    CvtColorBgr2YuvNv24,
    CvtColorRgb2YuvNv24,
    CvtColorBgr2YuvNv42,
    CvtColorRgb2YuvNv42,
    CvtColorBgr2YuvNv16,
    CvtColorRgb2YuvNv16,
    CvtColorBgr2YuvNv61,
    CvtColorRgb2YuvNv61,
    CvtColorBgr2YuvYuy2,
    CvtColorRgb2YuvYuy2,
    CvtColorBgr2YuvUyvy,
    CvtColorRgb2YuvUyvy,
    CvtColorBgr2YuvI444,
    CvtColorRgb2YuvI444,
    CvtColorBgr2YuvYv24,
    CvtColorRgb2YuvYv24,
    CvtColorYuv2BgrNv12,
    CvtColorYuv2RgbNv12,
    CvtColorYuv2BgrNv21,
    CvtColorYuv2RgbNv21,
    CvtColorYuv2BgrI420,
    CvtColorYuv2RgbI420,
    CvtColorYuv2BgrYv12,
    CvtColorYuv2RgbYv12,
    CvtColorYuv2BgrI444,
    CvtColorYuv2RgbI444,
    CvtColorYuv2BgrYv24,
    CvtColorYuv2RgbYv24,
    CvtColorYuv2BgrNv16,
    CvtColorYuv2RgbNv16,
    CvtColorYuv2BgrNv61,
    CvtColorYuv2RgbNv61,
    CvtColorYuv2BgrNv24,
    CvtColorYuv2RgbNv24,
    CvtColorYuv2BgrNv42,
    CvtColorYuv2RgbNv42,
    CvtColorYuv2BgrYuy2,
    CvtColorYuv2RgbYuy2,
    CvtColorYuv2BgrUyvy,
    CvtColorYuv2RgbUyvy,
    CvtColorBgr2GrayF32,
    CvtColorGray2BgrF32,
    CvtColorBgr2RgbF32,
    CvtColorBgr2BgraF32,
    CvtColorBgra2BgrF32,
    CvtColorRgb2RgbaF32,
    CvtColorRgba2RgbF32,
    CvtColorBgr2RgbaF32,
    CvtColorRgba2BgrF32,
    CvtColorRgb2BgraF32,
    CvtColorBgra2RgbF32,
    CvtColorBgra2RgbaF32,
    CvtColorRgba2BgraF32,
    CvtColorGray2BgraF32,
    CvtColorBgra2GrayF32,
    CvtColorGray2RgbaF32,
    CvtColorRgba2GrayF32,
    CvtColorBgr2YuvF32,
    CvtColorYuv2BgrF32,
    CvtColorRgb2YuvF32,
    CvtColorYuv2RgbF32,
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
        case ImgprocOp::CvtColorBgr2Rgb: return "CVTCOLOR_BGR2RGB";
        case ImgprocOp::CvtColorBgr2Bgra: return "CVTCOLOR_BGR2BGRA";
        case ImgprocOp::CvtColorBgra2Bgr: return "CVTCOLOR_BGRA2BGR";
        case ImgprocOp::CvtColorRgb2Rgba: return "CVTCOLOR_RGB2RGBA";
        case ImgprocOp::CvtColorRgba2Rgb: return "CVTCOLOR_RGBA2RGB";
        case ImgprocOp::CvtColorBgr2Rgba: return "CVTCOLOR_BGR2RGBA";
        case ImgprocOp::CvtColorRgba2Bgr: return "CVTCOLOR_RGBA2BGR";
        case ImgprocOp::CvtColorRgb2Bgra: return "CVTCOLOR_RGB2BGRA";
        case ImgprocOp::CvtColorBgra2Rgb: return "CVTCOLOR_BGRA2RGB";
        case ImgprocOp::CvtColorBgra2Rgba: return "CVTCOLOR_BGRA2RGBA";
        case ImgprocOp::CvtColorRgba2Bgra: return "CVTCOLOR_RGBA2BGRA";
        case ImgprocOp::CvtColorGray2Bgra: return "CVTCOLOR_GRAY2BGRA";
        case ImgprocOp::CvtColorBgra2Gray: return "CVTCOLOR_BGRA2GRAY";
        case ImgprocOp::CvtColorGray2Rgba: return "CVTCOLOR_GRAY2RGBA";
        case ImgprocOp::CvtColorRgba2Gray: return "CVTCOLOR_RGBA2GRAY";
        case ImgprocOp::CvtColorBgr2Yuv: return "CVTCOLOR_BGR2YUV";
        case ImgprocOp::CvtColorYuv2Bgr: return "CVTCOLOR_YUV2BGR";
        case ImgprocOp::CvtColorRgb2Yuv: return "CVTCOLOR_RGB2YUV";
        case ImgprocOp::CvtColorYuv2Rgb: return "CVTCOLOR_YUV2RGB";
        case ImgprocOp::CvtColorBgr2YuvNv12: return "CVTCOLOR_BGR2YUV_NV12";
        case ImgprocOp::CvtColorRgb2YuvNv12: return "CVTCOLOR_RGB2YUV_NV12";
        case ImgprocOp::CvtColorBgr2YuvNv21: return "CVTCOLOR_BGR2YUV_NV21";
        case ImgprocOp::CvtColorRgb2YuvNv21: return "CVTCOLOR_RGB2YUV_NV21";
        case ImgprocOp::CvtColorBgr2YuvI420: return "CVTCOLOR_BGR2YUV_I420";
        case ImgprocOp::CvtColorRgb2YuvI420: return "CVTCOLOR_RGB2YUV_I420";
        case ImgprocOp::CvtColorBgr2YuvYv12: return "CVTCOLOR_BGR2YUV_YV12";
        case ImgprocOp::CvtColorRgb2YuvYv12: return "CVTCOLOR_RGB2YUV_YV12";
        case ImgprocOp::CvtColorBgr2YuvNv24: return "CVTCOLOR_BGR2YUV_NV24";
        case ImgprocOp::CvtColorRgb2YuvNv24: return "CVTCOLOR_RGB2YUV_NV24";
        case ImgprocOp::CvtColorBgr2YuvNv42: return "CVTCOLOR_BGR2YUV_NV42";
        case ImgprocOp::CvtColorRgb2YuvNv42: return "CVTCOLOR_RGB2YUV_NV42";
        case ImgprocOp::CvtColorBgr2YuvNv16: return "CVTCOLOR_BGR2YUV_NV16";
        case ImgprocOp::CvtColorRgb2YuvNv16: return "CVTCOLOR_RGB2YUV_NV16";
        case ImgprocOp::CvtColorBgr2YuvNv61: return "CVTCOLOR_BGR2YUV_NV61";
        case ImgprocOp::CvtColorRgb2YuvNv61: return "CVTCOLOR_RGB2YUV_NV61";
        case ImgprocOp::CvtColorBgr2YuvYuy2: return "CVTCOLOR_BGR2YUV_YUY2";
        case ImgprocOp::CvtColorRgb2YuvYuy2: return "CVTCOLOR_RGB2YUV_YUY2";
        case ImgprocOp::CvtColorBgr2YuvUyvy: return "CVTCOLOR_BGR2YUV_UYVY";
        case ImgprocOp::CvtColorRgb2YuvUyvy: return "CVTCOLOR_RGB2YUV_UYVY";
        case ImgprocOp::CvtColorBgr2YuvI444: return "CVTCOLOR_BGR2YUV_I444";
        case ImgprocOp::CvtColorRgb2YuvI444: return "CVTCOLOR_RGB2YUV_I444";
        case ImgprocOp::CvtColorBgr2YuvYv24: return "CVTCOLOR_BGR2YUV_YV24";
        case ImgprocOp::CvtColorRgb2YuvYv24: return "CVTCOLOR_RGB2YUV_YV24";
        case ImgprocOp::CvtColorYuv2BgrNv12: return "CVTCOLOR_YUV2BGR_NV12";
        case ImgprocOp::CvtColorYuv2RgbNv12: return "CVTCOLOR_YUV2RGB_NV12";
        case ImgprocOp::CvtColorYuv2BgrNv21: return "CVTCOLOR_YUV2BGR_NV21";
        case ImgprocOp::CvtColorYuv2RgbNv21: return "CVTCOLOR_YUV2RGB_NV21";
        case ImgprocOp::CvtColorYuv2BgrI420: return "CVTCOLOR_YUV2BGR_I420";
        case ImgprocOp::CvtColorYuv2RgbI420: return "CVTCOLOR_YUV2RGB_I420";
        case ImgprocOp::CvtColorYuv2BgrYv12: return "CVTCOLOR_YUV2BGR_YV12";
        case ImgprocOp::CvtColorYuv2RgbYv12: return "CVTCOLOR_YUV2RGB_YV12";
        case ImgprocOp::CvtColorYuv2BgrI444: return "CVTCOLOR_YUV2BGR_I444";
        case ImgprocOp::CvtColorYuv2RgbI444: return "CVTCOLOR_YUV2RGB_I444";
        case ImgprocOp::CvtColorYuv2BgrYv24: return "CVTCOLOR_YUV2BGR_YV24";
        case ImgprocOp::CvtColorYuv2RgbYv24: return "CVTCOLOR_YUV2RGB_YV24";
        case ImgprocOp::CvtColorYuv2BgrNv16: return "CVTCOLOR_YUV2BGR_NV16";
        case ImgprocOp::CvtColorYuv2RgbNv16: return "CVTCOLOR_YUV2RGB_NV16";
        case ImgprocOp::CvtColorYuv2BgrNv61: return "CVTCOLOR_YUV2BGR_NV61";
        case ImgprocOp::CvtColorYuv2RgbNv61: return "CVTCOLOR_YUV2RGB_NV61";
        case ImgprocOp::CvtColorYuv2BgrNv24: return "CVTCOLOR_YUV2BGR_NV24";
        case ImgprocOp::CvtColorYuv2RgbNv24: return "CVTCOLOR_YUV2RGB_NV24";
        case ImgprocOp::CvtColorYuv2BgrNv42: return "CVTCOLOR_YUV2BGR_NV42";
        case ImgprocOp::CvtColorYuv2RgbNv42: return "CVTCOLOR_YUV2RGB_NV42";
        case ImgprocOp::CvtColorYuv2BgrYuy2: return "CVTCOLOR_YUV2BGR_YUY2";
        case ImgprocOp::CvtColorYuv2RgbYuy2: return "CVTCOLOR_YUV2RGB_YUY2";
        case ImgprocOp::CvtColorYuv2BgrUyvy: return "CVTCOLOR_YUV2BGR_UYVY";
        case ImgprocOp::CvtColorYuv2RgbUyvy: return "CVTCOLOR_YUV2RGB_UYVY";
        case ImgprocOp::CvtColorBgr2GrayF32: return "CVTCOLOR_BGR2GRAY_F32";
        case ImgprocOp::CvtColorGray2BgrF32: return "CVTCOLOR_GRAY2BGR_F32";
        case ImgprocOp::CvtColorBgr2RgbF32: return "CVTCOLOR_BGR2RGB_F32";
        case ImgprocOp::CvtColorBgr2BgraF32: return "CVTCOLOR_BGR2BGRA_F32";
        case ImgprocOp::CvtColorBgra2BgrF32: return "CVTCOLOR_BGRA2BGR_F32";
        case ImgprocOp::CvtColorRgb2RgbaF32: return "CVTCOLOR_RGB2RGBA_F32";
        case ImgprocOp::CvtColorRgba2RgbF32: return "CVTCOLOR_RGBA2RGB_F32";
        case ImgprocOp::CvtColorBgr2RgbaF32: return "CVTCOLOR_BGR2RGBA_F32";
        case ImgprocOp::CvtColorRgba2BgrF32: return "CVTCOLOR_RGBA2BGR_F32";
        case ImgprocOp::CvtColorRgb2BgraF32: return "CVTCOLOR_RGB2BGRA_F32";
        case ImgprocOp::CvtColorBgra2RgbF32: return "CVTCOLOR_BGRA2RGB_F32";
        case ImgprocOp::CvtColorBgra2RgbaF32: return "CVTCOLOR_BGRA2RGBA_F32";
        case ImgprocOp::CvtColorRgba2BgraF32: return "CVTCOLOR_RGBA2BGRA_F32";
        case ImgprocOp::CvtColorGray2BgraF32: return "CVTCOLOR_GRAY2BGRA_F32";
        case ImgprocOp::CvtColorBgra2GrayF32: return "CVTCOLOR_BGRA2GRAY_F32";
        case ImgprocOp::CvtColorGray2RgbaF32: return "CVTCOLOR_GRAY2RGBA_F32";
        case ImgprocOp::CvtColorRgba2GrayF32: return "CVTCOLOR_RGBA2GRAY_F32";
        case ImgprocOp::CvtColorBgr2YuvF32: return "CVTCOLOR_BGR2YUV_F32";
        case ImgprocOp::CvtColorYuv2BgrF32: return "CVTCOLOR_YUV2BGR_F32";
        case ImgprocOp::CvtColorRgb2YuvF32: return "CVTCOLOR_RGB2YUV_F32";
        case ImgprocOp::CvtColorYuv2RgbF32: return "CVTCOLOR_YUV2RGB_F32";
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
    cases.reserve(shapes.size() * 120);

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
        cases.push_back({ImgprocOp::CvtColorBgr2Rgb, shape, 3});
        cases.push_back({ImgprocOp::CvtColorBgr2Bgra, shape, 3});
        cases.push_back({ImgprocOp::CvtColorBgra2Bgr, shape, 4});
        cases.push_back({ImgprocOp::CvtColorRgb2Rgba, shape, 3});
        cases.push_back({ImgprocOp::CvtColorRgba2Rgb, shape, 4});
        cases.push_back({ImgprocOp::CvtColorBgr2Rgba, shape, 3});
        cases.push_back({ImgprocOp::CvtColorRgba2Bgr, shape, 4});
        cases.push_back({ImgprocOp::CvtColorRgb2Bgra, shape, 3});
        cases.push_back({ImgprocOp::CvtColorBgra2Rgb, shape, 4});
        cases.push_back({ImgprocOp::CvtColorBgra2Rgba, shape, 4});
        cases.push_back({ImgprocOp::CvtColorRgba2Bgra, shape, 4});
        cases.push_back({ImgprocOp::CvtColorGray2Bgra, shape, 1});
        cases.push_back({ImgprocOp::CvtColorBgra2Gray, shape, 4});
        cases.push_back({ImgprocOp::CvtColorGray2Rgba, shape, 1});
        cases.push_back({ImgprocOp::CvtColorRgba2Gray, shape, 4});
        cases.push_back({ImgprocOp::CvtColorBgr2Yuv, shape, 3});
        cases.push_back({ImgprocOp::CvtColorYuv2Bgr, shape, 3});
        cases.push_back({ImgprocOp::CvtColorRgb2Yuv, shape, 3});
        cases.push_back({ImgprocOp::CvtColorYuv2Rgb, shape, 3});
        cases.push_back({ImgprocOp::CvtColorBgr2YuvNv12, shape, 3});
        cases.push_back({ImgprocOp::CvtColorRgb2YuvNv12, shape, 3});
        cases.push_back({ImgprocOp::CvtColorBgr2YuvNv21, shape, 3});
        cases.push_back({ImgprocOp::CvtColorRgb2YuvNv21, shape, 3});
        cases.push_back({ImgprocOp::CvtColorBgr2YuvI420, shape, 3});
        cases.push_back({ImgprocOp::CvtColorRgb2YuvI420, shape, 3});
        cases.push_back({ImgprocOp::CvtColorBgr2YuvYv12, shape, 3});
        cases.push_back({ImgprocOp::CvtColorRgb2YuvYv12, shape, 3});
        cases.push_back({ImgprocOp::CvtColorBgr2YuvNv24, shape, 3});
        cases.push_back({ImgprocOp::CvtColorRgb2YuvNv24, shape, 3});
        cases.push_back({ImgprocOp::CvtColorBgr2YuvNv42, shape, 3});
        cases.push_back({ImgprocOp::CvtColorRgb2YuvNv42, shape, 3});
        cases.push_back({ImgprocOp::CvtColorBgr2YuvNv16, shape, 3});
        cases.push_back({ImgprocOp::CvtColorRgb2YuvNv16, shape, 3});
        cases.push_back({ImgprocOp::CvtColorBgr2YuvNv61, shape, 3});
        cases.push_back({ImgprocOp::CvtColorRgb2YuvNv61, shape, 3});
        cases.push_back({ImgprocOp::CvtColorBgr2YuvYuy2, shape, 3});
        cases.push_back({ImgprocOp::CvtColorRgb2YuvYuy2, shape, 3});
        cases.push_back({ImgprocOp::CvtColorBgr2YuvUyvy, shape, 3});
        cases.push_back({ImgprocOp::CvtColorRgb2YuvUyvy, shape, 3});
        cases.push_back({ImgprocOp::CvtColorBgr2YuvI444, shape, 3});
        cases.push_back({ImgprocOp::CvtColorRgb2YuvI444, shape, 3});
        cases.push_back({ImgprocOp::CvtColorBgr2YuvYv24, shape, 3});
        cases.push_back({ImgprocOp::CvtColorRgb2YuvYv24, shape, 3});
        cases.push_back({ImgprocOp::CvtColorYuv2BgrNv12, shape, 1});
        cases.push_back({ImgprocOp::CvtColorYuv2RgbNv12, shape, 1});
        cases.push_back({ImgprocOp::CvtColorYuv2BgrNv21, shape, 1});
        cases.push_back({ImgprocOp::CvtColorYuv2RgbNv21, shape, 1});
        cases.push_back({ImgprocOp::CvtColorYuv2BgrI420, shape, 1});
        cases.push_back({ImgprocOp::CvtColorYuv2RgbI420, shape, 1});
        cases.push_back({ImgprocOp::CvtColorYuv2BgrYv12, shape, 1});
        cases.push_back({ImgprocOp::CvtColorYuv2RgbYv12, shape, 1});
        cases.push_back({ImgprocOp::CvtColorYuv2BgrI444, shape, 1});
        cases.push_back({ImgprocOp::CvtColorYuv2RgbI444, shape, 1});
        cases.push_back({ImgprocOp::CvtColorYuv2BgrYv24, shape, 1});
        cases.push_back({ImgprocOp::CvtColorYuv2RgbYv24, shape, 1});
        cases.push_back({ImgprocOp::CvtColorYuv2BgrNv16, shape, 1});
        cases.push_back({ImgprocOp::CvtColorYuv2RgbNv16, shape, 1});
        cases.push_back({ImgprocOp::CvtColorYuv2BgrNv61, shape, 1});
        cases.push_back({ImgprocOp::CvtColorYuv2RgbNv61, shape, 1});
        cases.push_back({ImgprocOp::CvtColorYuv2BgrNv24, shape, 1});
        cases.push_back({ImgprocOp::CvtColorYuv2RgbNv24, shape, 1});
        cases.push_back({ImgprocOp::CvtColorYuv2BgrNv42, shape, 1});
        cases.push_back({ImgprocOp::CvtColorYuv2RgbNv42, shape, 1});
        cases.push_back({ImgprocOp::CvtColorYuv2BgrYuy2, shape, 2});
        cases.push_back({ImgprocOp::CvtColorYuv2RgbYuy2, shape, 2});
        cases.push_back({ImgprocOp::CvtColorYuv2BgrUyvy, shape, 2});
        cases.push_back({ImgprocOp::CvtColorYuv2RgbUyvy, shape, 2});
        cases.push_back({ImgprocOp::CvtColorBgr2GrayF32, shape, 3});
        cases.push_back({ImgprocOp::CvtColorGray2BgrF32, shape, 1});
        cases.push_back({ImgprocOp::CvtColorBgr2RgbF32, shape, 3});
        cases.push_back({ImgprocOp::CvtColorBgr2BgraF32, shape, 3});
        cases.push_back({ImgprocOp::CvtColorBgra2BgrF32, shape, 4});
        cases.push_back({ImgprocOp::CvtColorRgb2RgbaF32, shape, 3});
        cases.push_back({ImgprocOp::CvtColorRgba2RgbF32, shape, 4});
        cases.push_back({ImgprocOp::CvtColorBgr2RgbaF32, shape, 3});
        cases.push_back({ImgprocOp::CvtColorRgba2BgrF32, shape, 4});
        cases.push_back({ImgprocOp::CvtColorRgb2BgraF32, shape, 3});
        cases.push_back({ImgprocOp::CvtColorBgra2RgbF32, shape, 4});
        cases.push_back({ImgprocOp::CvtColorBgra2RgbaF32, shape, 4});
        cases.push_back({ImgprocOp::CvtColorRgba2BgraF32, shape, 4});
        cases.push_back({ImgprocOp::CvtColorGray2BgraF32, shape, 1});
        cases.push_back({ImgprocOp::CvtColorBgra2GrayF32, shape, 4});
        cases.push_back({ImgprocOp::CvtColorGray2RgbaF32, shape, 1});
        cases.push_back({ImgprocOp::CvtColorRgba2GrayF32, shape, 4});
        cases.push_back({ImgprocOp::CvtColorBgr2YuvF32, shape, 3});
        cases.push_back({ImgprocOp::CvtColorYuv2BgrF32, shape, 3});
        cases.push_back({ImgprocOp::CvtColorRgb2YuvF32, shape, 3});
        cases.push_back({ImgprocOp::CvtColorYuv2RgbF32, shape, 3});
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
    const bool is_yuv420sp_decode =
        c.op == ImgprocOp::CvtColorYuv2BgrNv12 ||
        c.op == ImgprocOp::CvtColorYuv2RgbNv12 ||
        c.op == ImgprocOp::CvtColorYuv2BgrNv21 ||
        c.op == ImgprocOp::CvtColorYuv2RgbNv21;
    const bool is_yuv422sp_decode =
        c.op == ImgprocOp::CvtColorYuv2BgrNv16 ||
        c.op == ImgprocOp::CvtColorYuv2RgbNv16 ||
        c.op == ImgprocOp::CvtColorYuv2BgrNv61 ||
        c.op == ImgprocOp::CvtColorYuv2RgbNv61;
    const bool is_yuv444sp_decode =
        c.op == ImgprocOp::CvtColorYuv2BgrNv24 ||
        c.op == ImgprocOp::CvtColorYuv2RgbNv24 ||
        c.op == ImgprocOp::CvtColorYuv2BgrNv42 ||
        c.op == ImgprocOp::CvtColorYuv2RgbNv42;
    const bool is_yuv444p_decode =
        c.op == ImgprocOp::CvtColorYuv2BgrI444 ||
        c.op == ImgprocOp::CvtColorYuv2RgbI444 ||
        c.op == ImgprocOp::CvtColorYuv2BgrYv24 ||
        c.op == ImgprocOp::CvtColorYuv2RgbYv24;
    const bool is_yuv420_decode =
        is_yuv420sp_decode ||
        c.op == ImgprocOp::CvtColorYuv2BgrI420 ||
        c.op == ImgprocOp::CvtColorYuv2RgbI420 ||
        c.op == ImgprocOp::CvtColorYuv2BgrYv12 ||
        c.op == ImgprocOp::CvtColorYuv2RgbYv12;

    int src_type = CV_MAKETYPE(CV_8U, c.channels);
    if (c.op == ImgprocOp::ThresholdBinaryF32 ||
        c.op == ImgprocOp::ResizeNearestF32 ||
        c.op == ImgprocOp::ResizeLinearF32 ||
        c.op == ImgprocOp::CvtColorBgr2RgbF32 ||
        c.op == ImgprocOp::CvtColorBgr2BgraF32 ||
        c.op == ImgprocOp::CvtColorBgra2BgrF32 ||
        c.op == ImgprocOp::CvtColorRgb2RgbaF32 ||
        c.op == ImgprocOp::CvtColorRgba2RgbF32 ||
        c.op == ImgprocOp::CvtColorBgr2RgbaF32 ||
        c.op == ImgprocOp::CvtColorRgba2BgrF32 ||
        c.op == ImgprocOp::CvtColorRgb2BgraF32 ||
        c.op == ImgprocOp::CvtColorBgra2RgbF32 ||
        c.op == ImgprocOp::CvtColorBgra2RgbaF32 ||
        c.op == ImgprocOp::CvtColorRgba2BgraF32 ||
        c.op == ImgprocOp::CvtColorGray2BgraF32 ||
        c.op == ImgprocOp::CvtColorBgra2GrayF32 ||
        c.op == ImgprocOp::CvtColorGray2RgbaF32 ||
        c.op == ImgprocOp::CvtColorRgba2GrayF32 ||
        c.op == ImgprocOp::CvtColorBgr2YuvF32 ||
        c.op == ImgprocOp::CvtColorYuv2BgrF32 ||
        c.op == ImgprocOp::CvtColorRgb2YuvF32 ||
        c.op == ImgprocOp::CvtColorYuv2RgbF32 ||
        c.op == ImgprocOp::CvtColorBgr2GrayF32 ||
        c.op == ImgprocOp::CvtColorGray2BgrF32 ||
        c.op == ImgprocOp::BoxFilter3x3F32 ||
        c.op == ImgprocOp::GaussianBlur5x5F32)
    {
        src_type = CV_MAKETYPE(CV_32F, c.channels);
    }
    const int src_rows = is_yuv420_decode ? (c.shape.rows * 3 / 2) :
                         (is_yuv422sp_decode ? (c.shape.rows * 2) :
                         ((is_yuv444sp_decode || is_yuv444p_decode) ? (c.shape.rows * 3) : c.shape.rows));
    d.src.create(std::vector<int>{src_rows, c.shape.cols}, src_type);
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
        case ImgprocOp::CvtColorBgr2Rgb:
            d.color_code = cvh::COLOR_BGR2RGB;
            break;
        case ImgprocOp::CvtColorBgr2Bgra:
            d.color_code = cvh::COLOR_BGR2BGRA;
            break;
        case ImgprocOp::CvtColorBgra2Bgr:
            d.color_code = cvh::COLOR_BGRA2BGR;
            break;
        case ImgprocOp::CvtColorRgb2Rgba:
            d.color_code = cvh::COLOR_RGB2RGBA;
            break;
        case ImgprocOp::CvtColorRgba2Rgb:
            d.color_code = cvh::COLOR_RGBA2RGB;
            break;
        case ImgprocOp::CvtColorBgr2Rgba:
            d.color_code = cvh::COLOR_BGR2RGBA;
            break;
        case ImgprocOp::CvtColorRgba2Bgr:
            d.color_code = cvh::COLOR_RGBA2BGR;
            break;
        case ImgprocOp::CvtColorRgb2Bgra:
            d.color_code = cvh::COLOR_RGB2BGRA;
            break;
        case ImgprocOp::CvtColorBgra2Rgb:
            d.color_code = cvh::COLOR_BGRA2RGB;
            break;
        case ImgprocOp::CvtColorBgra2Rgba:
            d.color_code = cvh::COLOR_BGRA2RGBA;
            break;
        case ImgprocOp::CvtColorRgba2Bgra:
            d.color_code = cvh::COLOR_RGBA2BGRA;
            break;
        case ImgprocOp::CvtColorGray2Bgra:
            d.color_code = cvh::COLOR_GRAY2BGRA;
            break;
        case ImgprocOp::CvtColorBgra2Gray:
            d.color_code = cvh::COLOR_BGRA2GRAY;
            break;
        case ImgprocOp::CvtColorGray2Rgba:
            d.color_code = cvh::COLOR_GRAY2RGBA;
            break;
        case ImgprocOp::CvtColorRgba2Gray:
            d.color_code = cvh::COLOR_RGBA2GRAY;
            break;
        case ImgprocOp::CvtColorBgr2Yuv:
            d.color_code = cvh::COLOR_BGR2YUV;
            break;
        case ImgprocOp::CvtColorYuv2Bgr:
            d.color_code = cvh::COLOR_YUV2BGR;
            break;
        case ImgprocOp::CvtColorRgb2Yuv:
            d.color_code = cvh::COLOR_RGB2YUV;
            break;
        case ImgprocOp::CvtColorBgr2YuvNv12:
            d.color_code = cvh::COLOR_BGR2YUV_NV12;
            break;
        case ImgprocOp::CvtColorRgb2YuvNv12:
            d.color_code = cvh::COLOR_RGB2YUV_NV12;
            break;
        case ImgprocOp::CvtColorBgr2YuvNv21:
            d.color_code = cvh::COLOR_BGR2YUV_NV21;
            break;
        case ImgprocOp::CvtColorRgb2YuvNv21:
            d.color_code = cvh::COLOR_RGB2YUV_NV21;
            break;
        case ImgprocOp::CvtColorBgr2YuvI420:
            d.color_code = cvh::COLOR_BGR2YUV_I420;
            break;
        case ImgprocOp::CvtColorRgb2YuvI420:
            d.color_code = cvh::COLOR_RGB2YUV_I420;
            break;
        case ImgprocOp::CvtColorBgr2YuvYv12:
            d.color_code = cvh::COLOR_BGR2YUV_YV12;
            break;
        case ImgprocOp::CvtColorRgb2YuvYv12:
            d.color_code = cvh::COLOR_RGB2YUV_YV12;
            break;
        case ImgprocOp::CvtColorYuv2Rgb:
            d.color_code = cvh::COLOR_YUV2RGB;
            break;
        case ImgprocOp::CvtColorBgr2YuvNv24:
            d.color_code = cvh::COLOR_BGR2YUV_NV24;
            break;
        case ImgprocOp::CvtColorRgb2YuvNv24:
            d.color_code = cvh::COLOR_RGB2YUV_NV24;
            break;
        case ImgprocOp::CvtColorBgr2YuvNv42:
            d.color_code = cvh::COLOR_BGR2YUV_NV42;
            break;
        case ImgprocOp::CvtColorRgb2YuvNv42:
            d.color_code = cvh::COLOR_RGB2YUV_NV42;
            break;
        case ImgprocOp::CvtColorBgr2YuvNv16:
            d.color_code = cvh::COLOR_BGR2YUV_NV16;
            break;
        case ImgprocOp::CvtColorRgb2YuvNv16:
            d.color_code = cvh::COLOR_RGB2YUV_NV16;
            break;
        case ImgprocOp::CvtColorBgr2YuvNv61:
            d.color_code = cvh::COLOR_BGR2YUV_NV61;
            break;
        case ImgprocOp::CvtColorRgb2YuvNv61:
            d.color_code = cvh::COLOR_RGB2YUV_NV61;
            break;
        case ImgprocOp::CvtColorBgr2YuvYuy2:
            d.color_code = cvh::COLOR_BGR2YUV_YUY2;
            break;
        case ImgprocOp::CvtColorRgb2YuvYuy2:
            d.color_code = cvh::COLOR_RGB2YUV_YUY2;
            break;
        case ImgprocOp::CvtColorBgr2YuvUyvy:
            d.color_code = cvh::COLOR_BGR2YUV_UYVY;
            break;
        case ImgprocOp::CvtColorRgb2YuvUyvy:
            d.color_code = cvh::COLOR_RGB2YUV_UYVY;
            break;
        case ImgprocOp::CvtColorBgr2YuvI444:
            d.color_code = cvh::COLOR_BGR2YUV_I444;
            break;
        case ImgprocOp::CvtColorRgb2YuvI444:
            d.color_code = cvh::COLOR_RGB2YUV_I444;
            break;
        case ImgprocOp::CvtColorBgr2YuvYv24:
            d.color_code = cvh::COLOR_BGR2YUV_YV24;
            break;
        case ImgprocOp::CvtColorRgb2YuvYv24:
            d.color_code = cvh::COLOR_RGB2YUV_YV24;
            break;
        case ImgprocOp::CvtColorYuv2BgrNv12:
            d.color_code = cvh::COLOR_YUV2BGR_NV12;
            break;
        case ImgprocOp::CvtColorYuv2RgbNv12:
            d.color_code = cvh::COLOR_YUV2RGB_NV12;
            break;
        case ImgprocOp::CvtColorYuv2BgrNv21:
            d.color_code = cvh::COLOR_YUV2BGR_NV21;
            break;
        case ImgprocOp::CvtColorYuv2RgbNv21:
            d.color_code = cvh::COLOR_YUV2RGB_NV21;
            break;
        case ImgprocOp::CvtColorYuv2BgrI420:
            d.color_code = cvh::COLOR_YUV2BGR_I420;
            break;
        case ImgprocOp::CvtColorYuv2RgbI420:
            d.color_code = cvh::COLOR_YUV2RGB_I420;
            break;
        case ImgprocOp::CvtColorYuv2BgrYv12:
            d.color_code = cvh::COLOR_YUV2BGR_YV12;
            break;
        case ImgprocOp::CvtColorYuv2RgbYv12:
            d.color_code = cvh::COLOR_YUV2RGB_YV12;
            break;
        case ImgprocOp::CvtColorYuv2BgrI444:
            d.color_code = cvh::COLOR_YUV2BGR_I444;
            break;
        case ImgprocOp::CvtColorYuv2RgbI444:
            d.color_code = cvh::COLOR_YUV2RGB_I444;
            break;
        case ImgprocOp::CvtColorYuv2BgrYv24:
            d.color_code = cvh::COLOR_YUV2BGR_YV24;
            break;
        case ImgprocOp::CvtColorYuv2RgbYv24:
            d.color_code = cvh::COLOR_YUV2RGB_YV24;
            break;
        case ImgprocOp::CvtColorYuv2BgrNv16:
            d.color_code = cvh::COLOR_YUV2BGR_NV16;
            break;
        case ImgprocOp::CvtColorYuv2RgbNv16:
            d.color_code = cvh::COLOR_YUV2RGB_NV16;
            break;
        case ImgprocOp::CvtColorYuv2BgrNv61:
            d.color_code = cvh::COLOR_YUV2BGR_NV61;
            break;
        case ImgprocOp::CvtColorYuv2RgbNv61:
            d.color_code = cvh::COLOR_YUV2RGB_NV61;
            break;
        case ImgprocOp::CvtColorYuv2BgrNv24:
            d.color_code = cvh::COLOR_YUV2BGR_NV24;
            break;
        case ImgprocOp::CvtColorYuv2RgbNv24:
            d.color_code = cvh::COLOR_YUV2RGB_NV24;
            break;
        case ImgprocOp::CvtColorYuv2BgrNv42:
            d.color_code = cvh::COLOR_YUV2BGR_NV42;
            break;
        case ImgprocOp::CvtColorYuv2RgbNv42:
            d.color_code = cvh::COLOR_YUV2RGB_NV42;
            break;
        case ImgprocOp::CvtColorYuv2BgrYuy2:
            d.color_code = cvh::COLOR_YUV2BGR_YUY2;
            break;
        case ImgprocOp::CvtColorYuv2RgbYuy2:
            d.color_code = cvh::COLOR_YUV2RGB_YUY2;
            break;
        case ImgprocOp::CvtColorYuv2BgrUyvy:
            d.color_code = cvh::COLOR_YUV2BGR_UYVY;
            break;
        case ImgprocOp::CvtColorYuv2RgbUyvy:
            d.color_code = cvh::COLOR_YUV2RGB_UYVY;
            break;
        case ImgprocOp::CvtColorBgr2GrayF32:
            d.color_code = cvh::COLOR_BGR2GRAY;
            break;
        case ImgprocOp::CvtColorGray2BgrF32:
            d.color_code = cvh::COLOR_GRAY2BGR;
            break;
        case ImgprocOp::CvtColorBgr2RgbF32:
            d.color_code = cvh::COLOR_BGR2RGB;
            break;
        case ImgprocOp::CvtColorBgr2BgraF32:
            d.color_code = cvh::COLOR_BGR2BGRA;
            break;
        case ImgprocOp::CvtColorBgra2BgrF32:
            d.color_code = cvh::COLOR_BGRA2BGR;
            break;
        case ImgprocOp::CvtColorRgb2RgbaF32:
            d.color_code = cvh::COLOR_RGB2RGBA;
            break;
        case ImgprocOp::CvtColorRgba2RgbF32:
            d.color_code = cvh::COLOR_RGBA2RGB;
            break;
        case ImgprocOp::CvtColorBgr2RgbaF32:
            d.color_code = cvh::COLOR_BGR2RGBA;
            break;
        case ImgprocOp::CvtColorRgba2BgrF32:
            d.color_code = cvh::COLOR_RGBA2BGR;
            break;
        case ImgprocOp::CvtColorRgb2BgraF32:
            d.color_code = cvh::COLOR_RGB2BGRA;
            break;
        case ImgprocOp::CvtColorBgra2RgbF32:
            d.color_code = cvh::COLOR_BGRA2RGB;
            break;
        case ImgprocOp::CvtColorBgra2RgbaF32:
            d.color_code = cvh::COLOR_BGRA2RGBA;
            break;
        case ImgprocOp::CvtColorRgba2BgraF32:
            d.color_code = cvh::COLOR_RGBA2BGRA;
            break;
        case ImgprocOp::CvtColorGray2BgraF32:
            d.color_code = cvh::COLOR_GRAY2BGRA;
            break;
        case ImgprocOp::CvtColorBgra2GrayF32:
            d.color_code = cvh::COLOR_BGRA2GRAY;
            break;
        case ImgprocOp::CvtColorGray2RgbaF32:
            d.color_code = cvh::COLOR_GRAY2RGBA;
            break;
        case ImgprocOp::CvtColorRgba2GrayF32:
            d.color_code = cvh::COLOR_RGBA2GRAY;
            break;
        case ImgprocOp::CvtColorBgr2YuvF32:
            d.color_code = cvh::COLOR_BGR2YUV;
            break;
        case ImgprocOp::CvtColorYuv2BgrF32:
            d.color_code = cvh::COLOR_YUV2BGR;
            break;
        case ImgprocOp::CvtColorRgb2YuvF32:
            d.color_code = cvh::COLOR_RGB2YUV;
            break;
        case ImgprocOp::CvtColorYuv2RgbF32:
            d.color_code = cvh::COLOR_YUV2RGB;
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
        case ImgprocOp::CvtColorBgr2Rgb:
        case ImgprocOp::CvtColorBgr2Bgra:
        case ImgprocOp::CvtColorBgra2Bgr:
        case ImgprocOp::CvtColorRgb2Rgba:
        case ImgprocOp::CvtColorRgba2Rgb:
        case ImgprocOp::CvtColorBgr2Rgba:
        case ImgprocOp::CvtColorRgba2Bgr:
        case ImgprocOp::CvtColorRgb2Bgra:
        case ImgprocOp::CvtColorBgra2Rgb:
        case ImgprocOp::CvtColorBgra2Rgba:
        case ImgprocOp::CvtColorRgba2Bgra:
        case ImgprocOp::CvtColorGray2Bgra:
        case ImgprocOp::CvtColorBgra2Gray:
        case ImgprocOp::CvtColorGray2Rgba:
        case ImgprocOp::CvtColorRgba2Gray:
        case ImgprocOp::CvtColorBgr2Yuv:
        case ImgprocOp::CvtColorYuv2Bgr:
        case ImgprocOp::CvtColorRgb2Yuv:
        case ImgprocOp::CvtColorYuv2Rgb:
        case ImgprocOp::CvtColorBgr2YuvNv12:
        case ImgprocOp::CvtColorRgb2YuvNv12:
        case ImgprocOp::CvtColorBgr2YuvNv21:
        case ImgprocOp::CvtColorRgb2YuvNv21:
        case ImgprocOp::CvtColorBgr2YuvI420:
        case ImgprocOp::CvtColorRgb2YuvI420:
        case ImgprocOp::CvtColorBgr2YuvYv12:
        case ImgprocOp::CvtColorRgb2YuvYv12:
        case ImgprocOp::CvtColorBgr2YuvNv24:
        case ImgprocOp::CvtColorRgb2YuvNv24:
        case ImgprocOp::CvtColorBgr2YuvNv42:
        case ImgprocOp::CvtColorRgb2YuvNv42:
        case ImgprocOp::CvtColorBgr2YuvNv16:
        case ImgprocOp::CvtColorRgb2YuvNv16:
        case ImgprocOp::CvtColorBgr2YuvNv61:
        case ImgprocOp::CvtColorRgb2YuvNv61:
        case ImgprocOp::CvtColorBgr2YuvYuy2:
        case ImgprocOp::CvtColorRgb2YuvYuy2:
        case ImgprocOp::CvtColorBgr2YuvUyvy:
        case ImgprocOp::CvtColorRgb2YuvUyvy:
        case ImgprocOp::CvtColorBgr2YuvI444:
        case ImgprocOp::CvtColorRgb2YuvI444:
        case ImgprocOp::CvtColorBgr2YuvYv24:
        case ImgprocOp::CvtColorRgb2YuvYv24:
        case ImgprocOp::CvtColorYuv2BgrNv12:
        case ImgprocOp::CvtColorYuv2RgbNv12:
        case ImgprocOp::CvtColorYuv2BgrNv21:
        case ImgprocOp::CvtColorYuv2RgbNv21:
        case ImgprocOp::CvtColorYuv2BgrI420:
        case ImgprocOp::CvtColorYuv2RgbI420:
        case ImgprocOp::CvtColorYuv2BgrYv12:
        case ImgprocOp::CvtColorYuv2RgbYv12:
        case ImgprocOp::CvtColorYuv2BgrI444:
        case ImgprocOp::CvtColorYuv2RgbI444:
        case ImgprocOp::CvtColorYuv2BgrYv24:
        case ImgprocOp::CvtColorYuv2RgbYv24:
        case ImgprocOp::CvtColorYuv2BgrNv16:
        case ImgprocOp::CvtColorYuv2RgbNv16:
        case ImgprocOp::CvtColorYuv2BgrNv61:
        case ImgprocOp::CvtColorYuv2RgbNv61:
        case ImgprocOp::CvtColorYuv2BgrNv24:
        case ImgprocOp::CvtColorYuv2RgbNv24:
        case ImgprocOp::CvtColorYuv2BgrNv42:
        case ImgprocOp::CvtColorYuv2RgbNv42:
        case ImgprocOp::CvtColorYuv2BgrYuy2:
        case ImgprocOp::CvtColorYuv2RgbYuy2:
        case ImgprocOp::CvtColorYuv2BgrUyvy:
        case ImgprocOp::CvtColorYuv2RgbUyvy:
        case ImgprocOp::CvtColorBgr2GrayF32:
        case ImgprocOp::CvtColorGray2BgrF32:
        case ImgprocOp::CvtColorBgr2RgbF32:
        case ImgprocOp::CvtColorBgr2BgraF32:
        case ImgprocOp::CvtColorBgra2BgrF32:
        case ImgprocOp::CvtColorRgb2RgbaF32:
        case ImgprocOp::CvtColorRgba2RgbF32:
        case ImgprocOp::CvtColorBgr2RgbaF32:
        case ImgprocOp::CvtColorRgba2BgrF32:
        case ImgprocOp::CvtColorRgb2BgraF32:
        case ImgprocOp::CvtColorBgra2RgbF32:
        case ImgprocOp::CvtColorBgra2RgbaF32:
        case ImgprocOp::CvtColorRgba2BgraF32:
        case ImgprocOp::CvtColorGray2BgraF32:
        case ImgprocOp::CvtColorBgra2GrayF32:
        case ImgprocOp::CvtColorGray2RgbaF32:
        case ImgprocOp::CvtColorRgba2GrayF32:
        case ImgprocOp::CvtColorBgr2YuvF32:
        case ImgprocOp::CvtColorYuv2BgrF32:
        case ImgprocOp::CvtColorRgb2YuvF32:
        case ImgprocOp::CvtColorYuv2RgbF32:
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
