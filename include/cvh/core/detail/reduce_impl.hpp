#ifndef CVH_CORE_DETAIL_REDUCE_IMPL_HPP
#define CVH_CORE_DETAIL_REDUCE_IMPL_HPP

#include "../saturate.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace cvh
{
namespace reduce_detail
{

inline void validate_mask(const Mat& src, const Mat& mask, const char* fn_name)
{
    if (!mask.empty() && (mask.type() != CV_8UC1 || mask.shape() != src.shape()))
    {
        CV_Error_(Error::StsBadArg,
                  ("%s mask must be CV_8UC1 with the same shape as src", fn_name));
    }
}

inline void validate_channels_1_to_4(const Mat& src, const char* fn_name)
{
    if (src.channels() < 1 || src.channels() > 4)
    {
        CV_Error_(Error::StsBadArg,
                  ("%s supports channels in [1,4], channels=%d",
                   fn_name,
                   src.channels()));
    }
}

inline double read_scalar(const uchar* row, size_t scalar_index, int depth)
{
    switch (depth)
    {
        case CV_8U:
            return reinterpret_cast<const uchar*>(row)[scalar_index];
        case CV_8S:
            return reinterpret_cast<const schar*>(row)[scalar_index];
        case CV_16U:
            return reinterpret_cast<const ushort*>(row)[scalar_index];
        case CV_16S:
            return reinterpret_cast<const short*>(row)[scalar_index];
        case CV_32S:
            return reinterpret_cast<const int*>(row)[scalar_index];
        case CV_32U:
            return reinterpret_cast<const uint*>(row)[scalar_index];
        case CV_16F:
            return static_cast<float>(reinterpret_cast<const hfloat*>(row)[scalar_index]);
        case CV_32F:
            return reinterpret_cast<const float*>(row)[scalar_index];
        case CV_64F:
            return reinterpret_cast<const double*>(row)[scalar_index];
        default:
            CV_Error_(Error::StsUnsupportedFormat,
                      ("reduction does not support depth=%d", depth));
            return 0.0;
    }
}

inline void write_scalar(uchar* row, size_t scalar_index, int depth, double value)
{
    switch (depth)
    {
        case CV_8U:
            reinterpret_cast<uchar*>(row)[scalar_index] = saturate_cast<uchar>(value);
            return;
        case CV_8S:
            reinterpret_cast<schar*>(row)[scalar_index] = saturate_cast<schar>(value);
            return;
        case CV_16U:
            reinterpret_cast<ushort*>(row)[scalar_index] = saturate_cast<ushort>(value);
            return;
        case CV_16S:
            reinterpret_cast<short*>(row)[scalar_index] = saturate_cast<short>(value);
            return;
        case CV_32S:
            reinterpret_cast<int*>(row)[scalar_index] = saturate_cast<int>(value);
            return;
        case CV_32U:
            reinterpret_cast<uint*>(row)[scalar_index] = saturate_cast<uint>(value);
            return;
        case CV_16F:
            reinterpret_cast<hfloat*>(row)[scalar_index] = saturate_cast<hfloat>(value);
            return;
        case CV_32F:
            reinterpret_cast<float*>(row)[scalar_index] = static_cast<float>(value);
            return;
        case CV_64F:
            reinterpret_cast<double*>(row)[scalar_index] = value;
            return;
        default:
            CV_Error_(Error::StsUnsupportedFormat,
                      ("reduction output does not support depth=%d", depth));
    }
}

template<typename Fn>
void for_each_selected_scalar(const Mat& src, const Mat& mask, Fn&& fn)
{
    const int channels = src.channels();
    const size_t outer = src.dims > 1 ? static_cast<size_t>(src.size.p[0]) : 1;
    const size_t pixels_per_outer = src.dims > 1 ? src.total(1, src.dims) : src.total();
    const size_t src_row_bytes = pixels_per_outer * src.elemSize();
    const size_t src_step0 = src.dims > 1 ? src.step(0) : src_row_bytes;
    const size_t mask_step0 =
        mask.empty() ? 0 : (mask.dims > 1 ? mask.step(0) : pixels_per_outer);

    for (size_t row = 0; row < outer; ++row)
    {
        const uchar* src_row = src.data + row * src_step0;
        const uchar* mask_row = mask.empty() ? nullptr : mask.data + row * mask_step0;
        for (size_t pixel = 0; pixel < pixels_per_outer; ++pixel)
        {
            if (mask_row != nullptr && mask_row[pixel] == 0)
            {
                continue;
            }
            const size_t scalar_offset = pixel * static_cast<size_t>(channels);
            for (int ch = 0; ch < channels; ++ch)
            {
                fn(row,
                   pixel,
                   ch,
                   read_scalar(
                       src_row, scalar_offset + static_cast<size_t>(ch), src.depth()));
            }
        }
    }
}

inline size_t selected_pixel_count(const Mat& src, const Mat& mask)
{
    if (mask.empty())
    {
        return src.total();
    }
    size_t count = 0;
    const size_t outer = mask.dims > 1 ? static_cast<size_t>(mask.size.p[0]) : 1;
    const size_t pixels_per_outer =
        mask.dims > 1 ? mask.total(1, mask.dims) : mask.total();
    const size_t step0 = mask.dims > 1 ? mask.step(0) : pixels_per_outer;
    for (size_t row = 0; row < outer; ++row)
    {
        const uchar* mask_row = mask.data + row * step0;
        for (size_t pixel = 0; pixel < pixels_per_outer; ++pixel)
        {
            count += mask_row[pixel] != 0 ? 1u : 0u;
        }
    }
    return count;
}

inline void ensure_single_channel(const Mat& src, const char* fn_name)
{
    if (src.empty())
    {
        CV_Error_(Error::StsBadArg, ("%s expects non-empty src", fn_name));
    }
    if (src.channels() != 1)
    {
        CV_Error_(Error::StsBadArg,
                  ("%s expects single-channel src, channels=%d",
                   fn_name,
                   src.channels()));
    }
}

inline bool nonzero(double value)
{
    return value != 0.0;
}

inline std::vector<int> coordinates_from_linear(const Mat& src, size_t linear)
{
    std::vector<int> coordinates(static_cast<size_t>(src.dims), 0);
    for (int dim = src.dims - 1; dim >= 0; --dim)
    {
        const size_t dim_size = static_cast<size_t>(src.size.p[dim]);
        coordinates[static_cast<size_t>(dim)] =
            static_cast<int>(linear % dim_size);
        linear /= dim_size;
    }
    return coordinates;
}

struct Extrema
{
    bool found = false;
    double min_value = 0.0;
    double max_value = 0.0;
    size_t min_linear = 0;
    size_t max_linear = 0;
};

inline Extrema find_extrema(const Mat& src, const Mat& mask)
{
    Extrema extrema;
    const size_t pixels_per_outer = src.dims > 1 ? src.total(1, src.dims) : src.total();
    for_each_selected_scalar(
        src,
        mask,
        [&](size_t row, size_t pixel, int, double value) {
            if (std::isnan(value))
            {
                return;
            }
            const size_t linear = row * pixels_per_outer + pixel;
            if (!extrema.found)
            {
                extrema.found = true;
                extrema.min_value = value;
                extrema.max_value = value;
                extrema.min_linear = linear;
                extrema.max_linear = linear;
                return;
            }
            if (value < extrema.min_value)
            {
                extrema.min_value = value;
                extrema.min_linear = linear;
            }
            if (value > extrema.max_value)
            {
                extrema.max_value = value;
                extrema.max_linear = linear;
            }
        });
    return extrema;
}

inline int output_depth(int dtype, int source_depth)
{
    return dtype < 0 ? source_depth : CV_MAT_DEPTH(dtype);
}

inline double reduce_value(const Mat& src,
                           int fixed_index,
                           int channel,
                           int dim,
                           int rtype)
{
    const int length = dim == 0 ? src.size.p[0] : src.size.p[1];
    bool initialized = false;
    long double accumulator = 0.0L;
    double extrema = 0.0;
    for (int i = 0; i < length; ++i)
    {
        const int y = dim == 0 ? i : fixed_index;
        const int x = dim == 0 ? fixed_index : i;
        const uchar* row = src.data + static_cast<size_t>(y) * src.step(0);
        const size_t scalar_index =
            static_cast<size_t>(x) * static_cast<size_t>(src.channels()) +
            static_cast<size_t>(channel);
        const double value = read_scalar(row, scalar_index, src.depth());
        if (rtype == REDUCE_MAX || rtype == REDUCE_MIN)
        {
            if (!initialized)
            {
                extrema = value;
                initialized = true;
            }
            else if ((rtype == REDUCE_MAX && value > extrema) ||
                     (rtype == REDUCE_MIN && value < extrema))
            {
                extrema = value;
            }
        }
        else if (rtype == REDUCE_SUM2)
        {
            accumulator += static_cast<long double>(value) * value;
        }
        else
        {
            accumulator += value;
        }
    }

    if (rtype == REDUCE_MAX || rtype == REDUCE_MIN)
    {
        return extrema;
    }
    if (rtype == REDUCE_AVG)
    {
        accumulator /= static_cast<long double>(length);
    }
    return static_cast<double>(accumulator);
}

inline void prepare_normalize_dst(const Mat& src,
                                  Mat& dst,
                                  int dtype,
                                  const Mat& mask)
{
    const int depth = output_depth(dtype, src.depth());
    const int type = CV_MAKETYPE(depth, src.channels());
    const bool allocate =
        dst.empty() || dst.type() != type || dst.shape() != src.shape();
    if (allocate)
    {
        dst.create(src.dims, src.size.p, type);
        if (!mask.empty())
        {
            dst.setTo(Scalar::all(0.0));
        }
    }
}

inline void apply_normalize(const Mat& src,
                            Mat& dst,
                            const Mat& mask,
                            double scale,
                            double shift)
{
    const int channels = src.channels();
    const size_t outer = src.dims > 1 ? static_cast<size_t>(src.size.p[0]) : 1;
    const size_t pixels_per_outer = src.dims > 1 ? src.total(1, src.dims) : src.total();
    const size_t src_step0 =
        src.dims > 1 ? src.step(0) : pixels_per_outer * src.elemSize();
    const size_t dst_step0 =
        dst.dims > 1 ? dst.step(0) : pixels_per_outer * dst.elemSize();
    const size_t mask_step0 =
        mask.empty() ? 0 : (mask.dims > 1 ? mask.step(0) : pixels_per_outer);

    for (size_t row = 0; row < outer; ++row)
    {
        const uchar* src_row = src.data + row * src_step0;
        uchar* dst_row = dst.data + row * dst_step0;
        const uchar* mask_row = mask.empty() ? nullptr : mask.data + row * mask_step0;
        for (size_t pixel = 0; pixel < pixels_per_outer; ++pixel)
        {
            if (mask_row != nullptr && mask_row[pixel] == 0)
            {
                continue;
            }
            const size_t offset = pixel * static_cast<size_t>(channels);
            for (int ch = 0; ch < channels; ++ch)
            {
                const double value =
                    read_scalar(src_row, offset + static_cast<size_t>(ch), src.depth());
                write_scalar(
                    dst_row,
                    offset + static_cast<size_t>(ch),
                    dst.depth(),
                    value * scale + shift);
            }
        }
    }
}

}  // namespace reduce_detail

inline Scalar sum(const Mat& src)
{
    Scalar result;
    if (src.empty())
    {
        return result;
    }
    reduce_detail::validate_channels_1_to_4(src, "sum");
    long double accumulators[4] = {0.0L, 0.0L, 0.0L, 0.0L};
    reduce_detail::for_each_selected_scalar(
        src,
        Mat(),
        [&](size_t, size_t, int channel, double value) {
            accumulators[channel] += value;
        });
    for (int ch = 0; ch < src.channels(); ++ch)
    {
        result[ch] = static_cast<double>(accumulators[ch]);
    }
    return result;
}

inline Scalar mean(const Mat& src, const Mat& mask)
{
    Scalar result;
    if (src.empty())
    {
        return result;
    }
    reduce_detail::validate_channels_1_to_4(src, "mean");
    reduce_detail::validate_mask(src, mask, "mean");
    const size_t count = reduce_detail::selected_pixel_count(src, mask);
    if (count == 0)
    {
        return result;
    }
    long double accumulators[4] = {0.0L, 0.0L, 0.0L, 0.0L};
    reduce_detail::for_each_selected_scalar(
        src,
        mask,
        [&](size_t, size_t, int channel, double value) {
            accumulators[channel] += value;
        });
    for (int ch = 0; ch < src.channels(); ++ch)
    {
        result[ch] =
            static_cast<double>(accumulators[ch] / static_cast<long double>(count));
    }
    return result;
}

inline void meanStdDev(const Mat& src,
                       Scalar& mean_value,
                       Scalar& stddev_value,
                       const Mat& mask)
{
    mean_value = Scalar();
    stddev_value = Scalar();
    if (src.empty())
    {
        return;
    }
    reduce_detail::validate_channels_1_to_4(src, "meanStdDev");
    reduce_detail::validate_mask(src, mask, "meanStdDev");
    const size_t count = reduce_detail::selected_pixel_count(src, mask);
    if (count == 0)
    {
        return;
    }

    size_t counts[4] = {0, 0, 0, 0};
    long double means[4] = {0.0L, 0.0L, 0.0L, 0.0L};
    long double squared_differences[4] = {0.0L, 0.0L, 0.0L, 0.0L};
    reduce_detail::for_each_selected_scalar(
        src,
        mask,
        [&](size_t, size_t, int channel, double value) {
            ++counts[channel];
            const long double delta = static_cast<long double>(value) - means[channel];
            means[channel] += delta / static_cast<long double>(counts[channel]);
            const long double delta_after = static_cast<long double>(value) - means[channel];
            squared_differences[channel] += delta * delta_after;
        });
    for (int ch = 0; ch < src.channels(); ++ch)
    {
        long double variance =
            squared_differences[ch] / static_cast<long double>(count);
        if (variance < 0.0L && variance > -1e-18L)
        {
            variance = 0.0L;
        }
        mean_value[ch] = static_cast<double>(means[ch]);
        stddev_value[ch] = std::sqrt(static_cast<double>(variance));
    }
}

inline double norm(const Mat& src, int normType, const Mat& mask)
{
    if (src.empty())
    {
        return 0.0;
    }
    reduce_detail::validate_mask(src, mask, "norm");
    if (normType != NORM_INF && normType != NORM_L1 && normType != NORM_L2)
    {
        CV_Error_(Error::StsBadArg, ("norm unsupported normType=%d", normType));
    }

    long double accumulator = 0.0L;
    double maximum = 0.0;
    bool has_nan = false;
    reduce_detail::for_each_selected_scalar(
        src,
        mask,
        [&](size_t, size_t, int, double value) {
            const double magnitude = std::fabs(value);
            has_nan = has_nan || std::isnan(magnitude);
            if (normType == NORM_INF)
            {
                maximum = std::max(maximum, magnitude);
            }
            else if (normType == NORM_L1)
            {
                accumulator += magnitude;
            }
            else
            {
                accumulator += static_cast<long double>(value) * value;
            }
        });
    if (has_nan)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (normType == NORM_INF)
    {
        return maximum;
    }
    if (normType == NORM_L1)
    {
        return static_cast<double>(accumulator);
    }
    return std::sqrt(static_cast<double>(accumulator));
}

inline double norm(const Mat& src1,
                   const Mat& src2,
                   int normType,
                   const Mat& mask)
{
    if (src1.empty() && src2.empty())
    {
        return 0.0;
    }
    if (src1.type() != src2.type() || src1.shape() != src2.shape())
    {
        CV_Error(Error::StsUnmatchedSizes, "norm inputs must have the same shape and type");
    }
    reduce_detail::validate_mask(src1, mask, "norm");
    if (normType != NORM_INF && normType != NORM_L1 && normType != NORM_L2)
    {
        CV_Error_(Error::StsBadArg, ("norm unsupported normType=%d", normType));
    }

    const int channels = src1.channels();
    const size_t outer = src1.dims > 1 ? static_cast<size_t>(src1.size.p[0]) : 1;
    const size_t pixels_per_outer =
        src1.dims > 1 ? src1.total(1, src1.dims) : src1.total();
    const size_t src1_step0 =
        src1.dims > 1 ? src1.step(0) : pixels_per_outer * src1.elemSize();
    const size_t src2_step0 =
        src2.dims > 1 ? src2.step(0) : pixels_per_outer * src2.elemSize();
    const size_t mask_step0 =
        mask.empty() ? 0 : (mask.dims > 1 ? mask.step(0) : pixels_per_outer);
    long double accumulator = 0.0L;
    double maximum = 0.0;
    bool has_nan = false;
    for (size_t row = 0; row < outer; ++row)
    {
        const uchar* row1 = src1.data + row * src1_step0;
        const uchar* row2 = src2.data + row * src2_step0;
        const uchar* mask_row = mask.empty() ? nullptr : mask.data + row * mask_step0;
        for (size_t pixel = 0; pixel < pixels_per_outer; ++pixel)
        {
            if (mask_row != nullptr && mask_row[pixel] == 0)
            {
                continue;
            }
            const size_t offset = pixel * static_cast<size_t>(channels);
            for (int ch = 0; ch < channels; ++ch)
            {
                const size_t idx = offset + static_cast<size_t>(ch);
                const double difference =
                    reduce_detail::read_scalar(row1, idx, src1.depth()) -
                    reduce_detail::read_scalar(row2, idx, src2.depth());
                const double magnitude = std::fabs(difference);
                has_nan = has_nan || std::isnan(magnitude);
                if (normType == NORM_INF)
                {
                    maximum = std::max(maximum, magnitude);
                }
                else if (normType == NORM_L1)
                {
                    accumulator += magnitude;
                }
                else
                {
                    accumulator += static_cast<long double>(difference) * difference;
                }
            }
        }
    }
    if (has_nan)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (normType == NORM_INF)
    {
        return maximum;
    }
    if (normType == NORM_L1)
    {
        return static_cast<double>(accumulator);
    }
    return std::sqrt(static_cast<double>(accumulator));
}

inline int countNonZero(const Mat& src)
{
    if (src.empty())
    {
        return 0;
    }
    reduce_detail::ensure_single_channel(src, "countNonZero");
    size_t count = 0;
    reduce_detail::for_each_selected_scalar(
        src,
        Mat(),
        [&](size_t, size_t, int, double value) {
            count += reduce_detail::nonzero(value) ? 1u : 0u;
        });
    if (count > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        CV_Error(Error::StsOutOfRange, "countNonZero result exceeds int range");
    }
    return static_cast<int>(count);
}

inline bool hasNonZero(const Mat& src)
{
    if (src.empty())
    {
        return false;
    }
    reduce_detail::ensure_single_channel(src, "hasNonZero");
    bool found = false;
    reduce_detail::for_each_selected_scalar(
        src,
        Mat(),
        [&](size_t, size_t, int, double value) {
            found = found || reduce_detail::nonzero(value);
        });
    return found;
}

inline void findNonZero(const Mat& src, std::vector<Point>& indices)
{
    indices.clear();
    if (src.empty())
    {
        return;
    }
    reduce_detail::ensure_single_channel(src, "findNonZero");
    if (src.dims != 2)
    {
        CV_Error(Error::StsBadArg, "findNonZero currently supports 2D input");
    }
    reduce_detail::for_each_selected_scalar(
        src,
        Mat(),
        [&](size_t row, size_t pixel, int, double value) {
            if (reduce_detail::nonzero(value))
            {
                indices.emplace_back(static_cast<int>(pixel), static_cast<int>(row));
            }
        });
}

inline void findNonZero(const Mat& src, Mat& indices)
{
    if (src.empty())
    {
        indices.release();
        return;
    }
    std::vector<Point> points;
    findNonZero(src, points);
    if (points.empty())
    {
        indices.release();
        return;
    }
    indices.create({static_cast<int>(points.size()), 1}, CV_32SC2);
    for (size_t i = 0; i < points.size(); ++i)
    {
        indices.at<int>(static_cast<int>(i), 0, 0) = points[i].x;
        indices.at<int>(static_cast<int>(i), 0, 1) = points[i].y;
    }
}

inline void minMaxIdx(const Mat& src,
                      double* minVal,
                      double* maxVal,
                      int* minIdx,
                      int* maxIdx,
                      const Mat& mask)
{
    if (src.empty())
    {
        CV_Error(Error::StsBadArg, "minMaxIdx expects non-empty src");
    }
    reduce_detail::validate_mask(src, mask, "minMaxIdx");
    if ((minIdx != nullptr || maxIdx != nullptr || !mask.empty()) && src.channels() != 1)
    {
        CV_Error(Error::StsBadArg, "minMaxIdx indices and mask require single-channel src");
    }
    const reduce_detail::Extrema extrema = reduce_detail::find_extrema(src, mask);
    if (minVal != nullptr)
    {
        *minVal = extrema.found ? extrema.min_value : 0.0;
    }
    if (maxVal != nullptr)
    {
        *maxVal = extrema.found ? extrema.max_value : 0.0;
    }
    if (minIdx != nullptr)
    {
        const std::vector<int> coordinates =
            extrema.found
                ? reduce_detail::coordinates_from_linear(src, extrema.min_linear)
                : std::vector<int>(static_cast<size_t>(src.dims), -1);
        std::copy(coordinates.begin(), coordinates.end(), minIdx);
    }
    if (maxIdx != nullptr)
    {
        const std::vector<int> coordinates =
            extrema.found
                ? reduce_detail::coordinates_from_linear(src, extrema.max_linear)
                : std::vector<int>(static_cast<size_t>(src.dims), -1);
        std::copy(coordinates.begin(), coordinates.end(), maxIdx);
    }
}

inline void minMaxLoc(const Mat& src,
                      double* minVal,
                      double* maxVal,
                      Point* minLoc,
                      Point* maxLoc,
                      const Mat& mask)
{
    if (src.empty())
    {
        if (minVal != nullptr)
        {
            *minVal = 0.0;
        }
        if (maxVal != nullptr)
        {
            *maxVal = 0.0;
        }
        if (minLoc != nullptr)
        {
            *minLoc = Point(-1, -1);
        }
        if (maxLoc != nullptr)
        {
            *maxLoc = Point(-1, -1);
        }
        return;
    }
    if (src.dims != 2)
    {
        CV_Error(Error::StsBadArg, "minMaxLoc expects 2D src");
    }
    int min_index[2] = {-1, -1};
    int max_index[2] = {-1, -1};
    minMaxIdx(
        src,
        minVal,
        maxVal,
        minLoc != nullptr ? min_index : nullptr,
        maxLoc != nullptr ? max_index : nullptr,
        mask);
    if (minLoc != nullptr)
    {
        *minLoc = Point(min_index[1], min_index[0]);
    }
    if (maxLoc != nullptr)
    {
        *maxLoc = Point(max_index[1], max_index[0]);
    }
}

inline void reduce(const Mat& src, Mat& dst, int dim, int rtype, int dtype)
{
    if (src.empty() || src.dims != 2)
    {
        CV_Error(Error::StsBadArg, "reduce expects non-empty 2D src");
    }
    if (dim != 0 && dim != 1)
    {
        CV_Error_(Error::StsBadArg, ("reduce axis must be 0 or 1, axis=%d", dim));
    }
    if (rtype < REDUCE_SUM || rtype > REDUCE_SUM2)
    {
        CV_Error_(Error::StsBadArg, ("reduce unsupported rtype=%d", rtype));
    }
    if (&src == &dst)
    {
        Mat source_copy = src.clone();
        reduce(source_copy, dst, dim, rtype, dtype);
        return;
    }
    const int depth = reduce_detail::output_depth(dtype, src.depth());
    const int type = CV_MAKETYPE(depth, src.channels());
    const int rows = dim == 0 ? 1 : src.size.p[0];
    const int cols = dim == 0 ? src.size.p[1] : 1;
    if (dst.empty() || dst.type() != type || dst.shape() != MatShape({rows, cols}))
    {
        dst.create({rows, cols}, type);
    }

    const int output_length = dim == 0 ? cols : rows;
    for (int fixed = 0; fixed < output_length; ++fixed)
    {
        uchar* dst_row = dst.data +
                         static_cast<size_t>(dim == 0 ? 0 : fixed) * dst.step(0);
        for (int ch = 0; ch < src.channels(); ++ch)
        {
            const double value =
                reduce_detail::reduce_value(src, fixed, ch, dim, rtype);
            const size_t scalar_index =
                dim == 0
                    ? static_cast<size_t>(fixed) * static_cast<size_t>(src.channels()) +
                          static_cast<size_t>(ch)
                    : static_cast<size_t>(ch);
            reduce_detail::write_scalar(dst_row, scalar_index, depth, value);
        }
    }
}

template<bool FindMax>
inline void reduce_arg_impl(const Mat& src,
                            Mat& dst,
                            int axis,
                            bool lastIndex,
                            const char* fn_name)
{
    reduce_detail::ensure_single_channel(src, fn_name);
    if (src.dims != 2 || (axis != 0 && axis != 1))
    {
        CV_Error_(Error::StsBadArg, ("%s expects 2D src and axis 0/1", fn_name));
    }
    if (&src == &dst)
    {
        Mat source_copy = src.clone();
        reduce_arg_impl<FindMax>(
            source_copy, dst, axis, lastIndex, fn_name);
        return;
    }
    const int rows = axis == 0 ? 1 : src.size.p[0];
    const int cols = axis == 0 ? src.size.p[1] : 1;
    if (dst.empty() || dst.type() != CV_32SC1 ||
        dst.shape() != MatShape({rows, cols}))
    {
        dst.create({rows, cols}, CV_32SC1);
    }
    const int output_length = axis == 0 ? cols : rows;
    const int reduce_length = axis == 0 ? src.size.p[0] : src.size.p[1];
    for (int fixed = 0; fixed < output_length; ++fixed)
    {
        int best_index = 0;
        const int first_y = axis == 0 ? 0 : fixed;
        const int first_x = axis == 0 ? fixed : 0;
        const uchar* first_row =
            src.data + static_cast<size_t>(first_y) * src.step(0);
        double best_value =
            reduce_detail::read_scalar(first_row, static_cast<size_t>(first_x), src.depth());
        for (int i = 1; i < reduce_length; ++i)
        {
            const int y = axis == 0 ? i : fixed;
            const int x = axis == 0 ? fixed : i;
            const uchar* row = src.data + static_cast<size_t>(y) * src.step(0);
            const double value =
                reduce_detail::read_scalar(row, static_cast<size_t>(x), src.depth());
            const bool better = FindMax ? value > best_value : value < best_value;
            const bool equal_and_last = lastIndex && value == best_value;
            if (better || equal_and_last)
            {
                best_value = value;
                best_index = i;
            }
        }
        if (axis == 0)
        {
            dst.at<int>(0, fixed) = best_index;
        }
        else
        {
            dst.at<int>(fixed, 0) = best_index;
        }
    }
}

inline void reduceArgMin(const Mat& src, Mat& dst, int axis, bool lastIndex)
{
    reduce_arg_impl<false>(src, dst, axis, lastIndex, "reduceArgMin");
}

inline void reduceArgMax(const Mat& src, Mat& dst, int axis, bool lastIndex)
{
    reduce_arg_impl<true>(src, dst, axis, lastIndex, "reduceArgMax");
}

inline void normalize(const Mat& src,
                      Mat& dst,
                      double alpha,
                      double beta,
                      int normType,
                      int dtype,
                      const Mat& mask)
{
    if (src.empty())
    {
        dst.release();
        return;
    }
    reduce_detail::validate_mask(src, mask, "normalize");
    if (&src == &dst && dtype >= 0 && CV_MAT_DEPTH(dtype) != src.depth())
    {
        Mat source_copy = src.clone();
        normalize(source_copy, dst, alpha, beta, normType, dtype, mask);
        return;
    }
    reduce_detail::prepare_normalize_dst(src, dst, dtype, mask);

    double scale = 0.0;
    double shift = 0.0;
    if (normType == NORM_MINMAX)
    {
        const reduce_detail::Extrema extrema = reduce_detail::find_extrema(src, mask);
        if (!extrema.found)
        {
            return;
        }
        if (extrema.max_value != extrema.min_value)
        {
            scale = (beta - alpha) / (extrema.max_value - extrema.min_value);
        }
        shift = alpha - extrema.min_value * scale;
    }
    else if (normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2)
    {
        const double source_norm = norm(src, normType, mask);
        scale = source_norm > 0.0 ? alpha / source_norm : 0.0;
    }
    else
    {
        CV_Error_(Error::StsBadArg, ("normalize unsupported normType=%d", normType));
    }
    reduce_detail::apply_normalize(src, dst, mask, scale, shift);
}

}  // namespace cvh

#endif  // CVH_CORE_DETAIL_REDUCE_IMPL_HPP
