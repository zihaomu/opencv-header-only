#ifndef CVH_IMGPROC_DETAIL_COMMON_H
#define CVH_IMGPROC_DETAIL_COMMON_H

#include "../../core/mat.h"
#include "../../core/saturate.h"
#include "../../core/types.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace cvh {

enum InterpolationFlags
{
    INTER_NEAREST = 0,
    INTER_LINEAR = 1,
    INTER_NEAREST_EXACT = 6,
};

enum ColorConversionCodes
{
    COLOR_BGR2GRAY = 0,
    COLOR_GRAY2BGR = 1,
};

enum ThresholdTypes
{
    THRESH_BINARY = 0,
    THRESH_BINARY_INV = 1,
    THRESH_TRUNC = 2,
    THRESH_TOZERO = 3,
    THRESH_TOZERO_INV = 4,
    THRESH_MASK = 7,
    THRESH_OTSU = 8,
    THRESH_TRIANGLE = 16,
    THRESH_DRYRUN = 128,
};

enum BorderTypes
{
    BORDER_CONSTANT = 0,
    BORDER_REPLICATE = 1,
    BORDER_REFLECT = 2,
    BORDER_WRAP = 3,
    BORDER_REFLECT_101 = 4,
    BORDER_TRANSPARENT = 5,

    BORDER_REFLECT101 = BORDER_REFLECT_101,
    BORDER_DEFAULT = BORDER_REFLECT_101,
    BORDER_ISOLATED = 16,
};

#if defined(CVH_FULL)
CV_EXPORTS void register_all_backends();
#endif

namespace detail {

inline int resolve_resize_dim(int src_dim, int dsize_dim, double scale)
{
    if (dsize_dim > 0)
    {
        return dsize_dim;
    }
    if (scale > 0.0)
    {
        const int scaled = static_cast<int>(std::lround(static_cast<double>(src_dim) * scale));
        return std::max(1, scaled);
    }
    return 0;
}

inline float lerp(float a, float b, float t)
{
    return a + (b - a) * t;
}

inline int normalize_border_type(int border_type)
{
    return border_type & (~BORDER_ISOLATED);
}

inline bool is_supported_filter_border(int border_type)
{
    return border_type == BORDER_CONSTANT ||
           border_type == BORDER_REPLICATE ||
           border_type == BORDER_REFLECT ||
           border_type == BORDER_REFLECT_101;
}

inline int border_interpolate(int p, int len, int border_type)
{
    CV_Assert(len > 0);

    if (static_cast<unsigned>(p) < static_cast<unsigned>(len))
    {
        return p;
    }

    if (border_type == BORDER_CONSTANT)
    {
        return -1;
    }

    if (border_type == BORDER_REPLICATE)
    {
        return p < 0 ? 0 : (len - 1);
    }

    if (border_type == BORDER_REFLECT || border_type == BORDER_REFLECT_101)
    {
        if (len == 1)
        {
            return 0;
        }

        const int delta = border_type == BORDER_REFLECT_101 ? 1 : 0;
        while (p < 0 || p >= len)
        {
            if (p < 0)
            {
                p = -p - 1 + delta;
            }
            else
            {
                p = len - 1 - (p - len) - delta;
            }
        }
        return p;
    }

    CV_Error_(Error::StsBadArg, ("filter: unsupported borderType=%d", border_type));
    return -1;
}

inline double default_gaussian_sigma_for_ksize(int ksize)
{
    CV_Assert(ksize > 0);
    return ((static_cast<double>(ksize) - 1.0) * 0.5 - 1.0) * 0.3 + 0.8;
}

inline int auto_gaussian_ksize(double sigma)
{
    CV_Assert(sigma > 0.0);
    int ksize = static_cast<int>(std::lround(sigma * 6.0 + 1.0));
    ksize = std::max(ksize, 3);
    if ((ksize & 1) == 0)
    {
        ++ksize;
    }
    return ksize;
}

inline std::vector<float> build_gaussian_kernel_1d(int ksize, double sigma)
{
    CV_Assert(ksize > 0 && (ksize & 1));
    CV_Assert(sigma > 0.0);

    std::vector<float> kernel(static_cast<size_t>(ksize), 0.0f);
    const int radius = ksize / 2;
    const double scale = -0.5 / (sigma * sigma);

    double sum = 0.0;
    for (int i = 0; i < ksize; ++i)
    {
        const double x = static_cast<double>(i - radius);
        const double w = std::exp(x * x * scale);
        kernel[static_cast<size_t>(i)] = static_cast<float>(w);
        sum += w;
    }

    if (sum <= 0.0)
    {
        CV_Error(Error::StsBadArg, "GaussianBlur: invalid gaussian kernel sum");
    }

    const float inv_sum = static_cast<float>(1.0 / sum);
    for (float& w : kernel)
    {
        w *= inv_sum;
    }

    return kernel;
}

inline void ensure_backends_registered_once()
{
#if defined(CVH_FULL)
    static bool initialized = []() {
        cvh::register_all_backends();
        return true;
    }();
    (void)initialized;
#endif
}

}  // namespace detail

}  // namespace cvh

#endif  // CVH_IMGPROC_DETAIL_COMMON_H
