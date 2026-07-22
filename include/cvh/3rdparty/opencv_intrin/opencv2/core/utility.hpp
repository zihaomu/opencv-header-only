#ifndef CVH_3RDPARTY_OPENCV_INTRIN_UTILITY_HPP
#define CVH_3RDPARTY_OPENCV_INTRIN_UTILITY_HPP

#include "opencv2/core/cvdef.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cv {

template <int N, typename T>
inline bool isAligned(const T& data)
{
    if constexpr (std::is_pointer<T>::value)
    {
        return (reinterpret_cast<std::uintptr_t>(data) & (N - 1U)) == 0U;
    }
    else
    {
        return (static_cast<std::uintptr_t>(data) & (N - 1U)) == 0U;
    }
}

template <int N>
inline bool isAligned(const void* ptr)
{
    return ((reinterpret_cast<std::uintptr_t>(ptr)) & (N - 1U)) == 0U;
}

template <int N>
inline bool isAligned(const void* p1, const void* p2)
{
    return isAligned<N>(reinterpret_cast<std::uintptr_t>(p1) |
                        reinterpret_cast<std::uintptr_t>(p2));
}

template <int N>
inline bool isAligned(const void* p1, const void* p2, const void* p3)
{
    return isAligned<N>(reinterpret_cast<std::uintptr_t>(p1) |
                        reinterpret_cast<std::uintptr_t>(p2) |
                        reinterpret_cast<std::uintptr_t>(p3));
}

template <int N>
inline bool isAligned(const void* p1, const void* p2, const void* p3, const void* p4)
{
    return isAligned<N>(reinterpret_cast<std::uintptr_t>(p1) |
                        reinterpret_cast<std::uintptr_t>(p2) |
                        reinterpret_cast<std::uintptr_t>(p3) |
                        reinterpret_cast<std::uintptr_t>(p4));
}

}  // namespace cv

#endif  // CVH_3RDPARTY_OPENCV_INTRIN_UTILITY_HPP
