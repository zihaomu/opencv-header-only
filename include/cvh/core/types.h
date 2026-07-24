#ifndef CVH_CORE_TYPES_H
#define CVH_CORE_TYPES_H

#include <cassert>

namespace cvh {

template<typename T>
struct Point_
{
    T x;
    T y;

    constexpr Point_() : x(0), y(0) {}
    constexpr Point_(T x_, T y_) : x(x_), y(y_) {}

    template<typename U>
    constexpr Point_(const Point_<U>& point)
        : x(static_cast<T>(point.x)), y(static_cast<T>(point.y))
    {
    }
};

template<typename T>
inline bool operator==(const Point_<T>& lhs, const Point_<T>& rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y;
}

template<typename T>
inline bool operator!=(const Point_<T>& lhs, const Point_<T>& rhs)
{
    return !(lhs == rhs);
}

using Point2i = Point_<int>;
using Point2f = Point_<float>;
using Point2d = Point_<double>;
using Point = Point2i;

template<typename T>
struct Size_
{
    T width;
    T height;

    constexpr Size_() : width(0), height(0) {}
    constexpr Size_(T width_, T height_) : width(width_), height(height_) {}

    template<typename U>
    constexpr Size_(const Size_<U>& size)
        : width(static_cast<T>(size.width)), height(static_cast<T>(size.height))
    {
    }

    constexpr bool empty() const { return width <= 0 || height <= 0; }
};

template<typename T>
inline bool operator==(const Size_<T>& lhs, const Size_<T>& rhs)
{
    return lhs.width == rhs.width && lhs.height == rhs.height;
}

template<typename T>
inline bool operator!=(const Size_<T>& lhs, const Size_<T>& rhs)
{
    return !(lhs == rhs);
}

using Size2i = Size_<int>;
using Size2f = Size_<float>;
using Size2d = Size_<double>;
using Size = Size2i;

enum DecompTypes
{
    DECOMP_LU = 0,
};

struct Scalar
{
    double val[4];

    Scalar() : val{0.0, 0.0, 0.0, 0.0} {}
    explicit Scalar(double v0) : val{v0, 0.0, 0.0, 0.0} {}
    Scalar(double v0, double v1, double v2 = 0.0, double v3 = 0.0) : val{v0, v1, v2, v3} {}

    static Scalar all(double v)
    {
        return Scalar(v, v, v, v);
    }

    double& operator[](int i)
    {
        assert(i >= 0 && i < 4);
        return val[i];
    }

    const double& operator[](int i) const
    {
        assert(i >= 0 && i < 4);
        return val[i];
    }
};

inline bool operator==(const Scalar& lhs, const Scalar& rhs)
{
    return lhs.val[0] == rhs.val[0] &&
           lhs.val[1] == rhs.val[1] &&
           lhs.val[2] == rhs.val[2] &&
           lhs.val[3] == rhs.val[3];
}

inline bool operator!=(const Scalar& lhs, const Scalar& rhs)
{
    return !(lhs == rhs);
}

}  // namespace cvh

#endif  // CVH_CORE_TYPES_H
