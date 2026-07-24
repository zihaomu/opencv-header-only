#ifndef CVH_IMGPROC_DETAIL_GEOMETRIC_SAMPLING_HPP
#define CVH_IMGPROC_DETAIL_GEOMETRIC_SAMPLING_HPP

#include "common.h"

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace cvh {
namespace detail {

template<typename T>
inline T geometric_cast(double value)
{
    if constexpr (std::is_same<T, uchar>::value)
    {
        return saturate_cast<uchar>(value);
    }
    return static_cast<T>(value);
}

template<typename T>
inline T geometric_border_value(const Scalar& value, int channel)
{
    const double scalar = value.val[channel < 4 ? channel : 3];
    return geometric_cast<T>(scalar);
}

template<typename T>
inline T geometric_read(const Mat& src,
                        int y,
                        int x,
                        int channel,
                        int border_type,
                        const Scalar& border_value)
{
    const int rows = src.size[0];
    const int cols = src.size[1];
    if (static_cast<unsigned>(y) < static_cast<unsigned>(rows) &&
        static_cast<unsigned>(x) < static_cast<unsigned>(cols))
    {
        const T* row = reinterpret_cast<const T*>(
            src.data + static_cast<size_t>(y) * src.step(0));
        return row[static_cast<size_t>(x) * src.channels() + channel];
    }
    if (border_type == BORDER_CONSTANT)
    {
        return geometric_border_value<T>(border_value, channel);
    }
    const int source_y = border_interpolate(y, rows, border_type);
    const int source_x = border_interpolate(x, cols, border_type);
    if (source_y < 0 || source_x < 0)
    {
        return geometric_border_value<T>(border_value, channel);
    }
    const T* row = reinterpret_cast<const T*>(
        src.data + static_cast<size_t>(source_y) * src.step(0));
    return row[
        static_cast<size_t>(source_x) * src.channels() + channel];
}

template<typename T>
inline void geometric_write_nearest(const Mat& src,
                                    T* destination,
                                    int source_x,
                                    int source_y,
                                    int border_type,
                                    const Scalar& border_value)
{
    for (int channel = 0; channel < src.channels(); ++channel)
    {
        destination[channel] = geometric_read<T>(
            src,
            source_y,
            source_x,
            channel,
            border_type,
            border_value);
    }
}

template<typename SourceT, typename DestinationT>
inline void geometric_write_linear_as(const Mat& src,
                                      DestinationT* destination,
                                      int source_x,
                                      int source_y,
                                      double fraction_x,
                                      double fraction_y,
                                      int border_type,
                                      const Scalar& border_value)
{
    for (int channel = 0; channel < src.channels(); ++channel)
    {
        const double top_left = geometric_read<SourceT>(
            src,
            source_y,
            source_x,
            channel,
            border_type,
            border_value);
        const double top_right = geometric_read<SourceT>(
            src,
            source_y,
            source_x + 1,
            channel,
            border_type,
            border_value);
        const double bottom_left = geometric_read<SourceT>(
            src,
            source_y + 1,
            source_x,
            channel,
            border_type,
            border_value);
        const double bottom_right = geometric_read<SourceT>(
            src,
            source_y + 1,
            source_x + 1,
            channel,
            border_type,
            border_value);
        const double top =
            top_left + (top_right - top_left) * fraction_x;
        const double bottom =
            bottom_left + (bottom_right - bottom_left) * fraction_x;
        destination[channel] = geometric_cast<DestinationT>(
            top + (bottom - top) * fraction_y);
    }
}

template<typename T>
inline void geometric_write_linear(const Mat& src,
                                   T* destination,
                                   int source_x,
                                   int source_y,
                                   double fraction_x,
                                   double fraction_y,
                                   int border_type,
                                   const Scalar& border_value)
{
    geometric_write_linear_as<T, T>(
        src,
        destination,
        source_x,
        source_y,
        fraction_x,
        fraction_y,
        border_type,
        border_value);
}

inline void geometric_linear_coordinate(double coordinate,
                                        bool quantize,
                                        int& integer,
                                        double& fraction)
{
    if (!quantize)
    {
        integer = static_cast<int>(std::floor(coordinate));
        fraction = coordinate - static_cast<double>(integer);
        return;
    }
    const long scaled = std::lrint(
        coordinate * static_cast<double>(INTER_TAB_SIZE));
    integer = static_cast<int>(std::floor(
        static_cast<double>(scaled) /
        static_cast<double>(INTER_TAB_SIZE)));
    fraction =
        static_cast<double>(
            scaled - static_cast<long>(integer) * INTER_TAB_SIZE) /
        static_cast<double>(INTER_TAB_SIZE);
}

template<typename T>
inline void geometric_write_coordinate(const Mat& src,
                                       T* destination,
                                       double source_x,
                                       double source_y,
                                       int interpolation,
                                       int border_type,
                                       const Scalar& border_value,
                                       bool quantize_linear,
                                       bool nearest_even)
{
    if (interpolation == INTER_NEAREST)
    {
        const int nearest_x = nearest_even
            ? static_cast<int>(std::lrint(source_x))
            : static_cast<int>(std::floor(source_x + 0.5));
        const int nearest_y = nearest_even
            ? static_cast<int>(std::lrint(source_y))
            : static_cast<int>(std::floor(source_y + 0.5));
        geometric_write_nearest(
            src,
            destination,
            nearest_x,
            nearest_y,
            border_type,
            border_value);
        return;
    }
    int integer_x = 0;
    int integer_y = 0;
    double fraction_x = 0.0;
    double fraction_y = 0.0;
    geometric_linear_coordinate(
        source_x,
        quantize_linear,
        integer_x,
        fraction_x);
    geometric_linear_coordinate(
        source_y,
        quantize_linear,
        integer_y,
        fraction_y);
    geometric_write_linear(
        src,
        destination,
        integer_x,
        integer_y,
        fraction_x,
        fraction_y,
        border_type,
        border_value);
}

}  // namespace detail
}  // namespace cvh

#endif  // CVH_IMGPROC_DETAIL_GEOMETRIC_SAMPLING_HPP
