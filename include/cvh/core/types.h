#ifndef CVH_CORE_TYPES_H
#define CVH_CORE_TYPES_H

namespace cvh {

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

}  // namespace cvh

#endif  // CVH_CORE_TYPES_H
