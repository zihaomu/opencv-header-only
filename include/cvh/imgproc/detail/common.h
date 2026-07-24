#ifndef CVH_IMGPROC_DETAIL_COMMON_H
#define CVH_IMGPROC_DETAIL_COMMON_H

#include "../../core/array.h"
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
    INTER_MAX = 7,
    INTER_NEAREST_EXACT = 6,
    WARP_INVERSE_MAP = 16,
    WARP_RELATIVE_MAP = 32,
};

enum InterpolationMasks
{
    INTER_BITS = 5,
    INTER_BITS2 = INTER_BITS * 2,
    INTER_TAB_SIZE = 1 << INTER_BITS,
    INTER_TAB_SIZE2 = INTER_TAB_SIZE * INTER_TAB_SIZE,
};

enum ColorConversionCodes
{
    COLOR_BGR2GRAY = 0,
    COLOR_GRAY2BGR = 1,
    COLOR_BGR2RGB = 2,
    COLOR_RGB2BGR = 3,
    COLOR_BGR2BGRA = 4,
    COLOR_BGRA2BGR = 5,
    COLOR_RGB2RGBA = 6,
    COLOR_RGBA2RGB = 7,
    COLOR_BGR2RGBA = 8,
    COLOR_RGBA2BGR = 9,
    COLOR_RGB2BGRA = 10,
    COLOR_BGRA2RGB = 11,
    COLOR_BGRA2RGBA = 12,
    COLOR_RGBA2BGRA = 13,
    COLOR_GRAY2BGRA = 14,
    COLOR_BGRA2GRAY = 15,
    COLOR_GRAY2RGBA = 16,
    COLOR_RGBA2GRAY = 17,
    COLOR_BGR2YUV = 18,
    COLOR_YUV2BGR = 19,
    COLOR_RGB2YUV = 20,
    COLOR_YUV2RGB = 21,
    COLOR_YUV2BGR_NV12 = 22,
    COLOR_YUV2RGB_NV12 = 23,
    COLOR_YUV2BGR_NV21 = 24,
    COLOR_YUV2RGB_NV21 = 25,
    COLOR_YUV2BGR_I420 = 26,
    COLOR_YUV2RGB_I420 = 27,
    COLOR_YUV2BGR_YV12 = 28,
    COLOR_YUV2RGB_YV12 = 29,
    COLOR_YUV2BGR_YUY2 = 30,
    COLOR_YUV2RGB_YUY2 = 31,
    COLOR_YUV2BGR_UYVY = 32,
    COLOR_YUV2RGB_UYVY = 33,
    COLOR_YUV2BGR_NV16 = 34,
    COLOR_YUV2RGB_NV16 = 35,
    COLOR_YUV2BGR_NV61 = 36,
    COLOR_YUV2RGB_NV61 = 37,
    COLOR_YUV2BGR_NV24 = 38,
    COLOR_YUV2RGB_NV24 = 39,
    COLOR_YUV2BGR_NV42 = 40,
    COLOR_YUV2RGB_NV42 = 41,
    COLOR_BGR2YUV_NV24 = 42,
    COLOR_RGB2YUV_NV24 = 43,
    COLOR_BGR2YUV_NV42 = 44,
    COLOR_RGB2YUV_NV42 = 45,
    COLOR_YUV2BGR_I444 = 46,
    COLOR_YUV2RGB_I444 = 47,
    COLOR_YUV2BGR_YV24 = 48,
    COLOR_YUV2RGB_YV24 = 49,
    COLOR_BGR2YUV_I444 = 50,
    COLOR_RGB2YUV_I444 = 51,
    COLOR_BGR2YUV_YV24 = 52,
    COLOR_RGB2YUV_YV24 = 53,
    COLOR_BGR2YUV_NV16 = 54,
    COLOR_RGB2YUV_NV16 = 55,
    COLOR_BGR2YUV_NV61 = 56,
    COLOR_RGB2YUV_NV61 = 57,
    COLOR_BGR2YUV_YUY2 = 58,
    COLOR_RGB2YUV_YUY2 = 59,
    COLOR_BGR2YUV_UYVY = 60,
    COLOR_RGB2YUV_UYVY = 61,
    COLOR_BGR2YUV_NV12 = 62,
    COLOR_RGB2YUV_NV12 = 63,
    COLOR_BGR2YUV_NV21 = 64,
    COLOR_RGB2YUV_NV21 = 65,
    COLOR_BGR2YUV_I420 = 66,
    COLOR_RGB2YUV_I420 = 67,
    COLOR_BGR2YUV_YV12 = 68,
    COLOR_RGB2YUV_YV12 = 69,
    COLOR_RGB2GRAY = 70,
    COLOR_BayerBG2BGR = 71,
    COLOR_BayerGB2BGR = 72,
    COLOR_BayerRG2BGR = 73,
    COLOR_BayerGR2BGR = 74,
    COLOR_BayerBG2RGB = COLOR_BayerRG2BGR,
    COLOR_BayerGB2RGB = COLOR_BayerGR2BGR,
    COLOR_BayerRG2RGB = COLOR_BayerBG2BGR,
    COLOR_BayerGR2RGB = COLOR_BayerGB2BGR,
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

enum AdaptiveThresholdTypes
{
    ADAPTIVE_THRESH_MEAN_C = 0,
    ADAPTIVE_THRESH_GAUSSIAN_C = 1,
};

enum ColormapTypes
{
    COLORMAP_AUTUMN = 0,
    COLORMAP_BONE = 1,
    COLORMAP_JET = 2,
    COLORMAP_WINTER = 3,
    COLORMAP_RAINBOW = 4,
    COLORMAP_OCEAN = 5,
    COLORMAP_SUMMER = 6,
    COLORMAP_SPRING = 7,
    COLORMAP_COOL = 8,
    COLORMAP_HSV = 9,
    COLORMAP_PINK = 10,
    COLORMAP_HOT = 11,
    COLORMAP_PARULA = 12,
    COLORMAP_MAGMA = 13,
    COLORMAP_INFERNO = 14,
    COLORMAP_PLASMA = 15,
    COLORMAP_VIRIDIS = 16,
    COLORMAP_CIVIDIS = 17,
    COLORMAP_TWILIGHT = 18,
    COLORMAP_TWILIGHT_SHIFTED = 19,
    COLORMAP_TURBO = 20,
    COLORMAP_DEEPGREEN = 21,
};

enum MorphTypes
{
    MORPH_ERODE = 0,
    MORPH_DILATE = 1,
    MORPH_OPEN = 2,
    MORPH_CLOSE = 3,
    MORPH_GRADIENT = 4,
    MORPH_TOPHAT = 5,
    MORPH_BLACKHAT = 6,
    MORPH_HITMISS = 7,
};

enum MorphShapes
{
    MORPH_RECT = 0,
    MORPH_CROSS = 1,
    MORPH_ELLIPSE = 2,
    MORPH_DIAMOND = 3,
};

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
    return cvh::borderInterpolate(p, len, border_type);
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

}  // namespace detail

}  // namespace cvh

#endif  // CVH_IMGPROC_DETAIL_COMMON_H
