#ifndef CVH_CORE_TYPES_H
#define CVH_CORE_TYPES_H

#include <cassert>

namespace cvh {

struct Point
{
    int x;
    int y;

    Point() : x(0), y(0) {}
    Point(int x_, int y_) : x(x_), y(y_) {}
};

inline bool operator==(const Point& lhs, const Point& rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y;
}

inline bool operator!=(const Point& lhs, const Point& rhs)
{
    return !(lhs == rhs);
}

struct Size
{
    int width;
    int height;

    Size() : width(0), height(0) {}
    Size(int w, int h) : width(w), height(h) {}

    bool empty() const { return width <= 0 || height <= 0; }
};

inline bool operator==(const Size& lhs, const Size& rhs)
{
    return lhs.width == rhs.width && lhs.height == rhs.height;
}

inline bool operator!=(const Size& lhs, const Size& rhs)
{
    return !(lhs == rhs);
}

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
