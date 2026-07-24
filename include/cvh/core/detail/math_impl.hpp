#ifndef CVH_CORE_DETAIL_MATH_IMPL_HPP
#define CVH_CORE_DETAIL_MATH_IMPL_HPP

#include "../saturate.h"

#include <cmath>
#include <cstring>
#include <type_traits>

namespace cvh
{
namespace math_detail
{

inline void ensure_same_input(const Mat& a, const Mat& b, const char* fn_name)
{
    if (a.empty() || b.empty())
    {
        CV_Error_(Error::StsBadArg, ("%s expects non-empty inputs", fn_name));
    }
    if (a.type() != b.type())
    {
        CV_Error_(Error::StsBadType,
                  ("%s type mismatch, src1=%d src2=%d", fn_name, a.type(), b.type()));
    }
    if (a.shape() != b.shape())
    {
        CV_Error_(Error::StsUnmatchedSizes, ("%s shape mismatch", fn_name));
    }
}

inline void prepare_dst(const Mat& src, Mat& dst, int type, const char* fn_name)
{
    if (src.empty())
    {
        CV_Error_(Error::StsBadArg, ("%s expects non-empty src", fn_name));
    }
    const int dst_type = CV_MAKETYPE(CV_MAT_DEPTH(type), src.channels());
    if (dst.empty() || dst.type() != dst_type || dst.shape() != src.shape())
    {
        dst.create(src.dims, src.size.p, dst_type);
    }
}

template<typename T, typename Op>
void apply_unary_same_type(const Mat& src, Mat& dst, Op op)
{
    const size_t outer = src.dims > 1 ? static_cast<size_t>(src.size.p[0]) : 1;
    const size_t pixels_per_outer = src.dims > 1 ? src.total(1, src.dims) : src.total();
    const size_t scalar_count = pixels_per_outer * static_cast<size_t>(src.channels());
    const size_t src_step0 = src.dims > 1 ? src.step(0) : scalar_count * sizeof(T);
    const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : scalar_count * sizeof(T);

    for (size_t row = 0; row < outer; ++row)
    {
        const T* src_row = reinterpret_cast<const T*>(src.data + row * src_step0);
        T* dst_row = reinterpret_cast<T*>(dst.data + row * dst_step0);
        for (size_t i = 0; i < scalar_count; ++i)
        {
            dst_row[i] = static_cast<T>(op(src_row[i]));
        }
    }
}

template<typename Op>
void dispatch_float_unary(const Mat& src, Mat& dst, Op op, const char* fn_name)
{
    prepare_dst(src, dst, src.type(), fn_name);
    switch (src.depth())
    {
        case CV_32F: apply_unary_same_type<float>(src, dst, op); break;
        case CV_64F: apply_unary_same_type<double>(src, dst, op); break;
        default:
            CV_Error_(Error::StsUnsupportedFormat,
                      ("%s supports CV_32F and CV_64F, depth=%d", fn_name, src.depth()));
    }
}

template<typename T>
void apply_scale_add(const Mat& src1, double alpha, const Mat& src2, Mat& dst)
{
    const size_t outer = src1.dims > 1 ? static_cast<size_t>(src1.size.p[0]) : 1;
    const size_t pixels_per_outer =
        src1.dims > 1 ? src1.total(1, src1.dims) : src1.total();
    const size_t scalar_count = pixels_per_outer * static_cast<size_t>(src1.channels());
    const size_t src1_step0 = src1.dims > 1 ? src1.step(0) : scalar_count * sizeof(T);
    const size_t src2_step0 = src2.dims > 1 ? src2.step(0) : scalar_count * sizeof(T);
    const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : scalar_count * sizeof(T);

    for (size_t row = 0; row < outer; ++row)
    {
        const T* src1_row = reinterpret_cast<const T*>(src1.data + row * src1_step0);
        const T* src2_row = reinterpret_cast<const T*>(src2.data + row * src2_step0);
        T* dst_row = reinterpret_cast<T*>(dst.data + row * dst_step0);
        for (size_t i = 0; i < scalar_count; ++i)
        {
            const double value =
                static_cast<double>(src1_row[i]) * alpha + static_cast<double>(src2_row[i]);
            dst_row[i] = saturate_cast<T>(value);
        }
    }
}

inline void dispatch_scale_add(const Mat& src1,
                               double alpha,
                               const Mat& src2,
                               Mat& dst,
                               const char* fn_name)
{
    ensure_same_input(src1, src2, fn_name);
    prepare_dst(src1, dst, src1.type(), fn_name);
    switch (src1.depth())
    {
        case CV_8U: apply_scale_add<uchar>(src1, alpha, src2, dst); break;
        case CV_8S: apply_scale_add<schar>(src1, alpha, src2, dst); break;
        case CV_16U: apply_scale_add<ushort>(src1, alpha, src2, dst); break;
        case CV_16S: apply_scale_add<short>(src1, alpha, src2, dst); break;
        case CV_32S: apply_scale_add<int>(src1, alpha, src2, dst); break;
        case CV_32U: apply_scale_add<uint>(src1, alpha, src2, dst); break;
        case CV_16F: apply_scale_add<hfloat>(src1, alpha, src2, dst); break;
        case CV_32F: apply_scale_add<float>(src1, alpha, src2, dst); break;
        case CV_64F: apply_scale_add<double>(src1, alpha, src2, dst); break;
        default:
            CV_Error_(Error::StsUnsupportedFormat,
                      ("%s does not support depth=%d", fn_name, src1.depth()));
    }
}

template<typename T>
inline uchar convert_scale_abs_cast(T value)
{
    const double numeric_value = static_cast<double>(value);
    if (!(numeric_value > 0.0))
    {
        return 0;
    }
    if (numeric_value >= 255.0)
    {
        return 255;
    }
    return static_cast<uchar>(std::nearbyint(numeric_value));
}

template<typename T>
void apply_convert_scale_abs(const Mat& src,
                             Mat& dst,
                             double alpha,
                             double beta)
{
    const size_t outer = src.dims > 1 ? static_cast<size_t>(src.size.p[0]) : 1;
    const size_t pixels_per_outer = src.dims > 1 ? src.total(1, src.dims) : src.total();
    const size_t scalar_count = pixels_per_outer * static_cast<size_t>(src.channels());
    const size_t src_step0 = src.dims > 1 ? src.step(0) : scalar_count * sizeof(T);
    const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : scalar_count;

    for (size_t row = 0; row < outer; ++row)
    {
        const T* src_row = reinterpret_cast<const T*>(src.data + row * src_step0);
        uchar* dst_row = dst.data + row * dst_step0;
        for (size_t i = 0; i < scalar_count; ++i)
        {
            const double value =
                std::fabs(static_cast<double>(src_row[i]) * alpha + beta);
            dst_row[i] = convert_scale_abs_cast(value);
        }
    }
}

inline void dispatch_convert_scale_abs(const Mat& src,
                                       Mat& dst,
                                       double alpha,
                                       double beta,
                                       const char* fn_name)
{
    prepare_dst(src, dst, CV_8U, fn_name);
    switch (src.depth())
    {
        case CV_8U: apply_convert_scale_abs<uchar>(src, dst, alpha, beta); break;
        case CV_8S: apply_convert_scale_abs<schar>(src, dst, alpha, beta); break;
        case CV_16U: apply_convert_scale_abs<ushort>(src, dst, alpha, beta); break;
        case CV_16S: apply_convert_scale_abs<short>(src, dst, alpha, beta); break;
        case CV_32S: apply_convert_scale_abs<int>(src, dst, alpha, beta); break;
        case CV_32U: apply_convert_scale_abs<uint>(src, dst, alpha, beta); break;
        case CV_16F: apply_convert_scale_abs<hfloat>(src, dst, alpha, beta); break;
        case CV_32F: apply_convert_scale_abs<float>(src, dst, alpha, beta); break;
        case CV_64F: apply_convert_scale_abs<double>(src, dst, alpha, beta); break;
        default:
            CV_Error_(Error::StsUnsupportedFormat,
                      ("%s does not support depth=%d", fn_name, src.depth()));
    }
}

inline ushort float_to_half_bits(float value)
{
    hfloat half(value);
    ushort bits = 0;
    std::memcpy(&bits, half.get_ptr(), sizeof(bits));
    return bits;
}

inline float half_bits_to_float(ushort bits)
{
    hfloat half;
    std::memcpy(half.get_ptr(), &bits, sizeof(bits));
    return static_cast<float>(half);
}

inline void convert_f32_to_fp16_bits(const Mat& src, Mat& dst)
{
    const size_t outer = src.dims > 1 ? static_cast<size_t>(src.size.p[0]) : 1;
    const size_t pixels_per_outer = src.dims > 1 ? src.total(1, src.dims) : src.total();
    const size_t scalar_count = pixels_per_outer * static_cast<size_t>(src.channels());
    const size_t src_step0 = src.dims > 1 ? src.step(0) : scalar_count * sizeof(float);
    const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : scalar_count * sizeof(short);

    for (size_t row = 0; row < outer; ++row)
    {
        const float* src_row =
            reinterpret_cast<const float*>(src.data + row * src_step0);
        short* dst_row = reinterpret_cast<short*>(dst.data + row * dst_step0);
        for (size_t i = 0; i < scalar_count; ++i)
        {
            const ushort bits = float_to_half_bits(src_row[i]);
            std::memcpy(dst_row + i, &bits, sizeof(bits));
        }
    }
}

inline void convert_fp16_bits_to_f32(const Mat& src, Mat& dst)
{
    const size_t outer = src.dims > 1 ? static_cast<size_t>(src.size.p[0]) : 1;
    const size_t pixels_per_outer = src.dims > 1 ? src.total(1, src.dims) : src.total();
    const size_t scalar_count = pixels_per_outer * static_cast<size_t>(src.channels());
    const size_t src_step0 = src.dims > 1 ? src.step(0) : scalar_count * sizeof(short);
    const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : scalar_count * sizeof(float);

    for (size_t row = 0; row < outer; ++row)
    {
        const short* src_row =
            reinterpret_cast<const short*>(src.data + row * src_step0);
        float* dst_row = reinterpret_cast<float*>(dst.data + row * dst_step0);
        for (size_t i = 0; i < scalar_count; ++i)
        {
            ushort bits = 0;
            std::memcpy(&bits, src_row + i, sizeof(bits));
            dst_row[i] = half_bits_to_float(bits);
        }
    }
}

inline void convert_native_f16_to_f32(const Mat& src, Mat& dst)
{
    const size_t outer = src.dims > 1 ? static_cast<size_t>(src.size.p[0]) : 1;
    const size_t pixels_per_outer = src.dims > 1 ? src.total(1, src.dims) : src.total();
    const size_t scalar_count = pixels_per_outer * static_cast<size_t>(src.channels());
    const size_t src_step0 = src.dims > 1 ? src.step(0) : scalar_count * sizeof(hfloat);
    const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : scalar_count * sizeof(float);

    for (size_t row = 0; row < outer; ++row)
    {
        const hfloat* src_row =
            reinterpret_cast<const hfloat*>(src.data + row * src_step0);
        float* dst_row = reinterpret_cast<float*>(dst.data + row * dst_step0);
        for (size_t i = 0; i < scalar_count; ++i)
        {
            dst_row[i] = static_cast<float>(src_row[i]);
        }
    }
}

template<typename T>
bool find_out_of_range(const Mat& src,
                       double min_value,
                       double max_value,
                       Point& position)
{
    const int channels = src.channels();
    const size_t outer = src.dims > 1 ? static_cast<size_t>(src.size.p[0]) : 1;
    const size_t pixels_per_outer = src.dims > 1 ? src.total(1, src.dims) : src.total();
    const size_t scalar_count = pixels_per_outer * static_cast<size_t>(channels);
    const size_t src_step0 = src.dims > 1 ? src.step(0) : scalar_count * sizeof(T);

    for (size_t row = 0; row < outer; ++row)
    {
        const T* src_row = reinterpret_cast<const T*>(src.data + row * src_step0);
        for (size_t pixel = 0; pixel < pixels_per_outer; ++pixel)
        {
            const size_t offset = pixel * static_cast<size_t>(channels);
            for (int ch = 0; ch < channels; ++ch)
            {
                const double value =
                    static_cast<double>(src_row[offset + static_cast<size_t>(ch)]);
                if (!std::isfinite(value) || value < min_value || value >= max_value)
                {
                    position = Point(
                        static_cast<int>(pixel),
                        src.dims > 1 ? static_cast<int>(row) : 0);
                    return true;
                }
            }
        }
    }
    return false;
}

inline bool dispatch_find_out_of_range(const Mat& src,
                                       double min_value,
                                       double max_value,
                                       Point& position)
{
    switch (src.depth())
    {
        case CV_8U: return find_out_of_range<uchar>(src, min_value, max_value, position);
        case CV_8S: return find_out_of_range<schar>(src, min_value, max_value, position);
        case CV_16U: return find_out_of_range<ushort>(src, min_value, max_value, position);
        case CV_16S: return find_out_of_range<short>(src, min_value, max_value, position);
        case CV_32S: return find_out_of_range<int>(src, min_value, max_value, position);
        case CV_32U: return find_out_of_range<uint>(src, min_value, max_value, position);
        case CV_16F: return find_out_of_range<hfloat>(src, min_value, max_value, position);
        case CV_32F: return find_out_of_range<float>(src, min_value, max_value, position);
        case CV_64F: return find_out_of_range<double>(src, min_value, max_value, position);
        default:
            CV_Error_(Error::StsUnsupportedFormat,
                      ("checkRange does not support depth=%d", src.depth()));
            return true;
    }
}

template<typename T>
void patch_nans_impl(Mat& src, T replacement)
{
    const size_t outer = src.dims > 1 ? static_cast<size_t>(src.size.p[0]) : 1;
    const size_t pixels_per_outer = src.dims > 1 ? src.total(1, src.dims) : src.total();
    const size_t scalar_count = pixels_per_outer * static_cast<size_t>(src.channels());
    const size_t src_step0 = src.dims > 1 ? src.step(0) : scalar_count * sizeof(T);

    for (size_t row = 0; row < outer; ++row)
    {
        T* src_row = reinterpret_cast<T*>(src.data + row * src_step0);
        for (size_t i = 0; i < scalar_count; ++i)
        {
            if (std::isnan(src_row[i]))
            {
                src_row[i] = replacement;
            }
        }
    }
}

}  // namespace math_detail

inline void scaleAdd(const Mat& src1, double alpha, const Mat& src2, Mat& dst)
{
    math_detail::dispatch_scale_add(src1, alpha, src2, dst, "scaleAdd");
}

inline void convertScaleAbs(const Mat& src,
                            Mat& dst,
                            double alpha,
                            double beta)
{
    if (&src == &dst)
    {
        Mat source_copy = src.clone();
        math_detail::dispatch_convert_scale_abs(
            source_copy, dst, alpha, beta, "convertScaleAbs");
        return;
    }
    math_detail::dispatch_convert_scale_abs(src, dst, alpha, beta, "convertScaleAbs");
}

inline void convertFp16(const Mat& src, Mat& dst)
{
    if (src.empty())
    {
        CV_Error(Error::StsBadArg, "convertFp16 expects non-empty src");
    }
    if (&src == &dst)
    {
        Mat source_copy = src.clone();
        convertFp16(source_copy, dst);
        return;
    }

    if (src.depth() == CV_32F)
    {
        math_detail::prepare_dst(src, dst, CV_16S, "convertFp16");
        math_detail::convert_f32_to_fp16_bits(src, dst);
        return;
    }
    if (src.depth() == CV_16S)
    {
        math_detail::prepare_dst(src, dst, CV_32F, "convertFp16");
        math_detail::convert_fp16_bits_to_f32(src, dst);
        return;
    }
    if (src.depth() == CV_16F)
    {
        math_detail::prepare_dst(src, dst, CV_32F, "convertFp16");
        math_detail::convert_native_f16_to_f32(src, dst);
        return;
    }
    CV_Error_(Error::StsUnsupportedFormat,
              ("convertFp16 expects CV_32F, CV_16S, or CV_16F, depth=%d", src.depth()));
}

inline void sqrt(const Mat& src, Mat& dst)
{
    math_detail::dispatch_float_unary(
        src,
        dst,
        [](const auto value) { return std::sqrt(value); },
        "sqrt");
}

inline void pow(const Mat& src, double power, Mat& dst)
{
    math_detail::dispatch_float_unary(
        src,
        dst,
        [power](const auto value) { return std::pow(value, power); },
        "pow");
}

inline void exp(const Mat& src, Mat& dst)
{
    math_detail::dispatch_float_unary(
        src,
        dst,
        [](const auto value) { return std::exp(value); },
        "exp");
}

inline void log(const Mat& src, Mat& dst)
{
    math_detail::dispatch_float_unary(
        src,
        dst,
        [](const auto value) { return std::log(value); },
        "log");
}

inline bool checkRange(const Mat& src,
                       bool quiet,
                       Point* pos,
                       double minVal,
                       double maxVal)
{
    if (src.empty())
    {
        return true;
    }
    if (src.dims > 2 && pos != nullptr)
    {
        CV_Error(Error::StsBadArg, "checkRange position is only available for dims <= 2");
    }

    Point bad_position(-1, -1);
    const bool invalid =
        minVal >= maxVal ||
        math_detail::dispatch_find_out_of_range(src, minVal, maxVal, bad_position);
    if (!invalid)
    {
        return true;
    }

    if (bad_position.x < 0)
    {
        bad_position = Point(0, 0);
    }
    if (pos != nullptr)
    {
        *pos = bad_position;
    }
    if (!quiet)
    {
        CV_Error_(Error::StsOutOfRange,
                  ("value at (%d, %d) is outside [%g, %g)",
                   bad_position.x,
                   bad_position.y,
                   minVal,
                   maxVal));
    }
    return false;
}

inline void patchNaNs(Mat& src, double value)
{
    if (src.empty())
    {
        return;
    }
    if (src.depth() != CV_32F)
    {
        CV_Error_(Error::StsUnsupportedFormat,
                  ("patchNaNs follows the local OpenCV CPU path and supports CV_32F only, depth=%d",
                   src.depth()));
    }
    math_detail::patch_nans_impl<float>(src, static_cast<float>(value));
}

}  // namespace cvh

#endif  // CVH_CORE_DETAIL_MATH_IMPL_HPP
