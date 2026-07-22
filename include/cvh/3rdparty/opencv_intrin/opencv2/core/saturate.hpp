#ifndef CVH_3RDPARTY_OPENCV_INTRIN_SATURATE_HPP
#define CVH_3RDPARTY_OPENCV_INTRIN_SATURATE_HPP

#include "opencv2/core/cvdef.h"

#include <cmath>
#include <climits>
#include <limits>
#include <type_traits>

namespace cv {

CV_INLINE int cvRound(double value)
{
    return static_cast<int>(std::lrint(value));
}

CV_INLINE int cvRound(float value)
{
    return static_cast<int>(std::lrint(value));
}

CV_INLINE int cvRound(int value)
{
    return value;
}

CV_INLINE int cvFloor(double value)
{
    return static_cast<int>(std::floor(value));
}

CV_INLINE int cvFloor(float value)
{
    return static_cast<int>(std::floor(value));
}

CV_INLINE int cvFloor(int value)
{
    return value;
}

CV_INLINE int cvCeil(double value)
{
    return static_cast<int>(std::ceil(value));
}

CV_INLINE int cvCeil(float value)
{
    return static_cast<int>(std::ceil(value));
}

CV_INLINE int cvCeil(int value)
{
    return value;
}

template <typename T, typename U>
static inline T saturate_cast(U value)
{
    if constexpr (std::is_same<T, uchar>::value)
    {
        const long long rounded = std::is_floating_point<U>::value
                                      ? static_cast<long long>(cvRound(static_cast<double>(value)))
                                      : static_cast<long long>(value);
        return static_cast<uchar>(rounded < 0 ? 0 : rounded > UCHAR_MAX ? UCHAR_MAX : rounded);
    }
    else if constexpr (std::is_same<T, schar>::value)
    {
        const long long rounded = std::is_floating_point<U>::value
                                      ? static_cast<long long>(cvRound(static_cast<double>(value)))
                                      : static_cast<long long>(value);
        return static_cast<schar>(rounded < SCHAR_MIN ? SCHAR_MIN : rounded > SCHAR_MAX ? SCHAR_MAX : rounded);
    }
    else if constexpr (std::is_same<T, ushort>::value)
    {
        const long long rounded = std::is_floating_point<U>::value
                                      ? static_cast<long long>(cvRound(static_cast<double>(value)))
                                      : static_cast<long long>(value);
        return static_cast<ushort>(rounded < 0 ? 0 : rounded > USHRT_MAX ? USHRT_MAX : rounded);
    }
    else if constexpr (std::is_same<T, short>::value)
    {
        const long long rounded = std::is_floating_point<U>::value
                                      ? static_cast<long long>(cvRound(static_cast<double>(value)))
                                      : static_cast<long long>(value);
        return static_cast<short>(rounded < SHRT_MIN ? SHRT_MIN : rounded > SHRT_MAX ? SHRT_MAX : rounded);
    }
    else if constexpr (std::is_same<T, int>::value && std::is_unsigned<U>::value && sizeof(U) >= sizeof(int))
    {
        return static_cast<int>(value > static_cast<U>(INT_MAX) ? INT_MAX : value);
    }
    else
    {
        return static_cast<T>(value);
    }
}

}  // namespace cv

#endif  // CVH_3RDPARTY_OPENCV_INTRIN_SATURATE_HPP
