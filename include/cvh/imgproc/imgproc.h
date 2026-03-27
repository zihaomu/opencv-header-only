#ifndef CVH_IMGPROC_H
#define CVH_IMGPROC_H

#include "../core/mat.h"
#include "../core/saturate.h"
#include "../core/types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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

using ResizeFn = void (*)(const Mat&, Mat&, Size, double, double, int);
using CvtColorFn = void (*)(const Mat&, Mat&, int);
using ThresholdFn = double (*)(const Mat&, Mat&, double, double, int);
using BoxFilterFn = void (*)(const Mat&, Mat&, int, Size, Point, bool, int);
using GaussianBlurFn = void (*)(const Mat&, Mat&, Size, double, double, int);

inline void resize_fallback(const Mat& src, Mat& dst, Size dsize, double fx, double fy, int interpolation)
{
    CV_Assert(!src.empty() && "resize: source image can not be empty");
    CV_Assert(src.dims == 2 && "resize: only 2D Mat is supported");
    CV_Assert(src.depth() == CV_8U && "resize: v1 supports CV_8U only");

    const int src_rows = src.size[0];
    const int src_cols = src.size[1];
    const int dst_cols = detail::resolve_resize_dim(src_cols, dsize.width, fx);
    const int dst_rows = detail::resolve_resize_dim(src_rows, dsize.height, fy);
    CV_Assert(dst_cols > 0 && dst_rows > 0 && "resize: invalid output size");

    dst.create(std::vector<int>{dst_rows, dst_cols}, src.type());

    const int channels = src.channels();
    const size_t src_step = src.step(0);
    const size_t dst_step = dst.step(0);
    const uchar* src_data = src.data;
    uchar* dst_data = dst.data;

    if (interpolation == INTER_NEAREST_EXACT)
    {
        const int64_t ifx = ((static_cast<int64_t>(src_cols) << 16) + dst_cols / 2) / dst_cols;
        const int64_t ifx0 = ifx / 2 - (src_cols % 2);
        const int64_t ify = ((static_cast<int64_t>(src_rows) << 16) + dst_rows / 2) / dst_rows;
        const int64_t ify0 = ify / 2 - (src_rows % 2);

        std::vector<int> x_ofs(static_cast<size_t>(dst_cols), 0);
        for (int x = 0; x < dst_cols; ++x)
        {
            const int sx = static_cast<int>((ifx * x + ifx0) >> 16);
            x_ofs[static_cast<size_t>(x)] = std::clamp(sx, 0, src_cols - 1);
        }

        for (int y = 0; y < dst_rows; ++y)
        {
            const int sy = static_cast<int>((ify * y + ify0) >> 16);
            const int clamped_sy = std::clamp(sy, 0, src_rows - 1);

            const uchar* src_row = src_data + static_cast<size_t>(clamped_sy) * src_step;
            uchar* dst_row = dst_data + static_cast<size_t>(y) * dst_step;
            for (int x = 0; x < dst_cols; ++x)
            {
                const int sx = x_ofs[static_cast<size_t>(x)];
                const uchar* src_px = src_row + static_cast<size_t>(sx) * channels;
                uchar* dst_px = dst_row + static_cast<size_t>(x) * channels;
                for (int c = 0; c < channels; ++c)
                {
                    dst_px[c] = src_px[c];
                }
            }
        }
        return;
    }

    if (interpolation == INTER_NEAREST)
    {
        for (int y = 0; y < dst_rows; ++y)
        {
            const int sy = std::min(src_rows - 1, (y * src_rows) / dst_rows);
            const uchar* src_row = src_data + static_cast<size_t>(sy) * src_step;
            uchar* dst_row = dst_data + static_cast<size_t>(y) * dst_step;
            for (int x = 0; x < dst_cols; ++x)
            {
                const int sx = std::min(src_cols - 1, (x * src_cols) / dst_cols);
                const uchar* src_px = src_row + static_cast<size_t>(sx) * channels;
                uchar* dst_px = dst_row + static_cast<size_t>(x) * channels;
                for (int c = 0; c < channels; ++c)
                {
                    dst_px[c] = src_px[c];
                }
            }
        }
        return;
    }

    if (interpolation != INTER_LINEAR)
    {
        CV_Error_(Error::StsBadArg, ("resize: unsupported interpolation=%d", interpolation));
    }

    const float scale_x = static_cast<float>(src_cols) / static_cast<float>(dst_cols);
    const float scale_y = static_cast<float>(src_rows) / static_cast<float>(dst_rows);

    for (int y = 0; y < dst_rows; ++y)
    {
        const float fy_src = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const int y0 = std::clamp(static_cast<int>(std::floor(fy_src)), 0, src_rows - 1);
        const int y1 = std::min(y0 + 1, src_rows - 1);
        const float wy = fy_src - static_cast<float>(y0);

        const uchar* src_row0 = src_data + static_cast<size_t>(y0) * src_step;
        const uchar* src_row1 = src_data + static_cast<size_t>(y1) * src_step;
        uchar* dst_row = dst_data + static_cast<size_t>(y) * dst_step;

        for (int x = 0; x < dst_cols; ++x)
        {
            const float fx_src = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const int x0 = std::clamp(static_cast<int>(std::floor(fx_src)), 0, src_cols - 1);
            const int x1 = std::min(x0 + 1, src_cols - 1);
            const float wx = fx_src - static_cast<float>(x0);

            const uchar* p00 = src_row0 + static_cast<size_t>(x0) * channels;
            const uchar* p01 = src_row0 + static_cast<size_t>(x1) * channels;
            const uchar* p10 = src_row1 + static_cast<size_t>(x0) * channels;
            const uchar* p11 = src_row1 + static_cast<size_t>(x1) * channels;
            uchar* dst_px = dst_row + static_cast<size_t>(x) * channels;

            for (int c = 0; c < channels; ++c)
            {
                const float top = detail::lerp(static_cast<float>(p00[c]), static_cast<float>(p01[c]), wx);
                const float bot = detail::lerp(static_cast<float>(p10[c]), static_cast<float>(p11[c]), wx);
                dst_px[c] = saturate_cast<uchar>(detail::lerp(top, bot, wy));
            }
        }
    }
}

inline ResizeFn& resize_dispatch()
{
    static ResizeFn fn = &resize_fallback;
    return fn;
}

inline void register_resize_backend(ResizeFn fn)
{
    if (fn)
    {
        resize_dispatch() = fn;
    }
}

inline bool is_resize_backend_registered()
{
    return resize_dispatch() != &resize_fallback;
}

inline void cvtColor_fallback(const Mat& src, Mat& dst, int code)
{
    CV_Assert(!src.empty() && "cvtColor: source image can not be empty");
    CV_Assert(src.dims == 2 && "cvtColor: only 2D Mat is supported");
    CV_Assert(src.depth() == CV_8U && "cvtColor: v1 supports CV_8U only");

    const int rows = src.size[0];
    const int cols = src.size[1];
    const size_t src_step = src.step(0);

    if (code == COLOR_BGR2GRAY)
    {
        CV_Assert(src.channels() == 3 && "cvtColor(BGR2GRAY): source must be CV_8UC3");
        dst.create(std::vector<int>{rows, cols}, CV_8UC1);
        const size_t dst_step = dst.step(0);
        for (int y = 0; y < rows; ++y)
        {
            const uchar* src_row = src.data + static_cast<size_t>(y) * src_step;
            uchar* dst_row = dst.data + static_cast<size_t>(y) * dst_step;
            for (int x = 0; x < cols; ++x)
            {
                const uchar* px = src_row + static_cast<size_t>(x) * 3;
                const float gray = 0.114f * static_cast<float>(px[0]) +
                                   0.587f * static_cast<float>(px[1]) +
                                   0.299f * static_cast<float>(px[2]);
                dst_row[x] = saturate_cast<uchar>(gray);
            }
        }
        return;
    }

    if (code == COLOR_GRAY2BGR)
    {
        CV_Assert(src.channels() == 1 && "cvtColor(GRAY2BGR): source must be CV_8UC1");
        dst.create(std::vector<int>{rows, cols}, CV_8UC3);
        const size_t dst_step = dst.step(0);
        for (int y = 0; y < rows; ++y)
        {
            const uchar* src_row = src.data + static_cast<size_t>(y) * src_step;
            uchar* dst_row = dst.data + static_cast<size_t>(y) * dst_step;
            for (int x = 0; x < cols; ++x)
            {
                const uchar g = src_row[x];
                uchar* out = dst_row + static_cast<size_t>(x) * 3;
                out[0] = g;
                out[1] = g;
                out[2] = g;
            }
        }
        return;
    }

    CV_Error_(Error::StsBadArg, ("cvtColor: unsupported conversion code=%d", code));
}

inline CvtColorFn& cvtcolor_dispatch()
{
    static CvtColorFn fn = &cvtColor_fallback;
    return fn;
}

inline void register_cvtcolor_backend(CvtColorFn fn)
{
    if (fn)
    {
        cvtcolor_dispatch() = fn;
    }
}

inline bool is_cvtcolor_backend_registered()
{
    return cvtcolor_dispatch() != &cvtColor_fallback;
}

inline std::array<int, 256> histogram_u8_single_channel(const Mat& src)
{
    CV_Assert(src.depth() == CV_8U && "threshold auto mode requires CV_8U source");
    CV_Assert(src.channels() == 1 && "threshold auto mode requires single-channel source");

    std::array<int, 256> hist = {};
    if (src.isContinuous())
    {
        const size_t count = src.total();
        for (size_t i = 0; i < count; ++i)
        {
            ++hist[src.data[i]];
        }
        return hist;
    }

    CV_Assert(src.dims == 2 && "threshold auto mode non-contiguous path supports 2D Mat only");
    const int rows = src.size[0];
    const int cols = src.size[1];
    const size_t src_step = src.step(0);
    for (int y = 0; y < rows; ++y)
    {
        const uchar* row = src.data + static_cast<size_t>(y) * src_step;
        for (int x = 0; x < cols; ++x)
        {
            ++hist[row[x]];
        }
    }
    return hist;
}

inline double threshold_otsu_u8(const Mat& src)
{
    const std::array<int, 256> hist = histogram_u8_single_channel(src);

    int64_t total = 0;
    double sum = 0.0;
    for (int i = 0; i < 256; ++i)
    {
        total += hist[static_cast<size_t>(i)];
        sum += static_cast<double>(i) * static_cast<double>(hist[static_cast<size_t>(i)]);
    }

    if (total == 0)
    {
        return 0.0;
    }

    int64_t w_b = 0;
    double sum_b = 0.0;
    double max_between = -1.0;
    int best_thresh = 0;

    for (int t = 0; t < 256; ++t)
    {
        const int h = hist[static_cast<size_t>(t)];
        w_b += h;
        if (w_b == 0)
        {
            continue;
        }

        const int64_t w_f = total - w_b;
        if (w_f == 0)
        {
            break;
        }

        sum_b += static_cast<double>(t) * static_cast<double>(h);
        const double m_b = sum_b / static_cast<double>(w_b);
        const double m_f = (sum - sum_b) / static_cast<double>(w_f);
        const double between = static_cast<double>(w_b) * static_cast<double>(w_f) * (m_b - m_f) * (m_b - m_f);
        if (between > max_between)
        {
            max_between = between;
            best_thresh = t;
        }
    }

    return static_cast<double>(best_thresh);
}

inline double threshold_triangle_u8(const Mat& src)
{
    std::array<int, 256> hist = histogram_u8_single_channel(src);

    int left_bound = 0;
    int right_bound = 0;
    int max_ind = 0;
    int max_val = 0;

    for (int i = 0; i < 256; ++i)
    {
        if (hist[static_cast<size_t>(i)] > 0)
        {
            left_bound = i;
            break;
        }
    }
    if (left_bound > 0)
    {
        --left_bound;
    }

    for (int i = 255; i > 0; --i)
    {
        if (hist[static_cast<size_t>(i)] > 0)
        {
            right_bound = i;
            break;
        }
    }
    if (right_bound < 255)
    {
        ++right_bound;
    }

    for (int i = 0; i < 256; ++i)
    {
        if (hist[static_cast<size_t>(i)] > max_val)
        {
            max_val = hist[static_cast<size_t>(i)];
            max_ind = i;
        }
    }

    bool flipped = false;
    if (max_ind - left_bound < right_bound - max_ind)
    {
        flipped = true;
        for (int i = 0, j = 255; i < j; ++i, --j)
        {
            std::swap(hist[static_cast<size_t>(i)], hist[static_cast<size_t>(j)]);
        }
        left_bound = 255 - right_bound;
        max_ind = 255 - max_ind;
    }

    double thresh = static_cast<double>(left_bound);
    const double a = static_cast<double>(max_val);
    const double b = static_cast<double>(left_bound - max_ind);
    double dist = 0.0;
    for (int i = left_bound + 1; i <= max_ind; ++i)
    {
        const double tempdist = a * static_cast<double>(i) + b * static_cast<double>(hist[static_cast<size_t>(i)]);
        if (tempdist > dist)
        {
            dist = tempdist;
            thresh = static_cast<double>(i);
        }
    }
    thresh -= 1.0;

    if (flipped)
    {
        thresh = 255.0 - thresh;
    }
    return thresh;
}

inline double threshold_fallback(const Mat& src, Mat& dst, double thresh, double maxval, int type)
{
    CV_Assert(!src.empty() && "threshold: source image can not be empty");
    CV_Assert(src.depth() == CV_8U && "threshold: v1 supports CV_8U only");

    const bool is_dryrun = (type & THRESH_DRYRUN) != 0;
    type &= ~THRESH_DRYRUN;

    const int automatic_thresh = type & (~THRESH_MASK);
    const int thresh_type = type & THRESH_MASK;

    if (automatic_thresh != 0 &&
        automatic_thresh != THRESH_OTSU &&
        automatic_thresh != THRESH_TRIANGLE)
    {
        CV_Error_(Error::StsBadArg, ("threshold: unsupported automatic threshold flag=%d", automatic_thresh));
    }

    if (thresh_type != THRESH_BINARY &&
        thresh_type != THRESH_BINARY_INV &&
        thresh_type != THRESH_TRUNC &&
        thresh_type != THRESH_TOZERO &&
        thresh_type != THRESH_TOZERO_INV)
    {
        CV_Error_(Error::StsBadArg, ("threshold: unsupported threshold type=%d", thresh_type));
    }

    if (automatic_thresh == THRESH_OTSU)
    {
        thresh = threshold_otsu_u8(src);
    }
    else if (automatic_thresh == THRESH_TRIANGLE)
    {
        thresh = threshold_triangle_u8(src);
    }

    const double effective_thresh = std::floor(thresh);
    if (is_dryrun)
    {
        return effective_thresh;
    }

    dst.create(src.dims, src.size.p, src.type());

    const uchar max_u8 = saturate_cast<uchar>(maxval);
    const uchar trunc_u8 = saturate_cast<uchar>(effective_thresh);
    const size_t scalar_count = src.total() * static_cast<size_t>(src.channels());
    const uchar* src_ptr = src.data;
    uchar* dst_ptr = dst.data;

    if (src.isContinuous() && dst.isContinuous())
    {
        for (size_t i = 0; i < scalar_count; ++i)
        {
            const uchar s = src_ptr[i];
            const bool cond = static_cast<double>(s) > effective_thresh;
            switch (thresh_type)
            {
            case THRESH_BINARY:
                dst_ptr[i] = cond ? max_u8 : 0;
                break;
            case THRESH_BINARY_INV:
                dst_ptr[i] = cond ? 0 : max_u8;
                break;
            case THRESH_TRUNC:
                dst_ptr[i] = cond ? trunc_u8 : s;
                break;
            case THRESH_TOZERO:
                dst_ptr[i] = cond ? s : 0;
                break;
            case THRESH_TOZERO_INV:
                dst_ptr[i] = cond ? 0 : s;
                break;
            default:
                CV_Error_(Error::StsBadArg, ("threshold: unsupported threshold type=%d", thresh_type));
            }
        }
        return effective_thresh;
    }

    CV_Assert(src.dims == 2 && "threshold: non-contiguous path supports 2D Mat only");
    const int rows = src.size[0];
    const int cols_scalar = src.size[1] * src.channels();
    const size_t src_step = src.step(0);
    const size_t dst_step = dst.step(0);
    for (int y = 0; y < rows; ++y)
    {
        const uchar* src_row = src_ptr + static_cast<size_t>(y) * src_step;
        uchar* dst_row = dst_ptr + static_cast<size_t>(y) * dst_step;
        for (int x = 0; x < cols_scalar; ++x)
        {
            const uchar s = src_row[x];
            const bool cond = static_cast<double>(s) > effective_thresh;
            switch (thresh_type)
            {
            case THRESH_BINARY:
                dst_row[x] = cond ? max_u8 : 0;
                break;
            case THRESH_BINARY_INV:
                dst_row[x] = cond ? 0 : max_u8;
                break;
            case THRESH_TRUNC:
                dst_row[x] = cond ? trunc_u8 : s;
                break;
            case THRESH_TOZERO:
                dst_row[x] = cond ? s : 0;
                break;
            case THRESH_TOZERO_INV:
                dst_row[x] = cond ? 0 : s;
                break;
            default:
                CV_Error_(Error::StsBadArg, ("threshold: unsupported threshold type=%d", thresh_type));
            }
        }
    }

    return effective_thresh;
}

inline ThresholdFn& threshold_dispatch()
{
    static ThresholdFn fn = &threshold_fallback;
    return fn;
}

inline void register_threshold_backend(ThresholdFn fn)
{
    if (fn)
    {
        threshold_dispatch() = fn;
    }
}

inline bool is_threshold_backend_registered()
{
    return threshold_dispatch() != &threshold_fallback;
}

inline void boxFilter_fallback(const Mat& src, Mat& dst, int ddepth, Size ksize, Point anchor, bool normalize, int borderType)
{
    CV_Assert(!src.empty() && "boxFilter: source image can not be empty");
    CV_Assert(src.dims == 2 && "boxFilter: only 2D Mat is supported");
    CV_Assert(src.depth() == CV_8U && "boxFilter: v1 supports CV_8U only");

    if (ddepth != -1 && ddepth != CV_8U)
    {
        CV_Error_(Error::StsBadArg, ("boxFilter: unsupported ddepth=%d (only -1/CV_8U)", ddepth));
    }

    if (ksize.width <= 0 || ksize.height <= 0)
    {
        CV_Error_(Error::StsBadArg, ("boxFilter: invalid ksize=(%d,%d)", ksize.width, ksize.height));
    }

    const int anchor_x = anchor.x >= 0 ? anchor.x : (ksize.width / 2);
    const int anchor_y = anchor.y >= 0 ? anchor.y : (ksize.height / 2);
    if (anchor_x < 0 || anchor_x >= ksize.width || anchor_y < 0 || anchor_y >= ksize.height)
    {
        CV_Error_(Error::StsBadArg,
                  ("boxFilter: invalid anchor=(%d,%d) for ksize=(%d,%d)",
                   anchor_x, anchor_y, ksize.width, ksize.height));
    }

    const int border_type = normalize_border_type(borderType);
    if (!is_supported_filter_border(border_type))
    {
        CV_Error_(Error::StsBadArg, ("boxFilter: unsupported borderType=%d", borderType));
    }

    Mat src_local;
    const Mat* src_ref = &src;
    if (src.data == dst.data)
    {
        src_local = src.clone();
        src_ref = &src_local;
    }

    const int rows = src_ref->size[0];
    const int cols = src_ref->size[1];
    const int channels = src_ref->channels();
    const size_t src_step = src_ref->step(0);

    dst.create(std::vector<int>{rows, cols}, src_ref->type());
    const size_t dst_step = dst.step(0);

    const int kernel_area = ksize.width * ksize.height;
    const float inv_kernel_area = kernel_area > 0 ? (1.0f / static_cast<float>(kernel_area)) : 0.0f;

    for (int y = 0; y < rows; ++y)
    {
        uchar* dst_row = dst.data + static_cast<size_t>(y) * dst_step;
        for (int x = 0; x < cols; ++x)
        {
            uchar* dst_px = dst_row + static_cast<size_t>(x) * channels;
            for (int c = 0; c < channels; ++c)
            {
                int64_t sum = 0;
                for (int ky = 0; ky < ksize.height; ++ky)
                {
                    const int sy = y + ky - anchor_y;
                    const int src_y = border_interpolate(sy, rows, border_type);
                    if (src_y < 0)
                    {
                        continue;
                    }

                    const uchar* src_row = src_ref->data + static_cast<size_t>(src_y) * src_step;
                    for (int kx = 0; kx < ksize.width; ++kx)
                    {
                        const int sx = x + kx - anchor_x;
                        const int src_x = border_interpolate(sx, cols, border_type);
                        if (src_x < 0)
                        {
                            continue;
                        }
                        sum += static_cast<int64_t>(src_row[static_cast<size_t>(src_x) * channels + c]);
                    }
                }

                if (normalize)
                {
                    dst_px[c] = saturate_cast<uchar>(static_cast<float>(sum) * inv_kernel_area);
                }
                else
                {
                    dst_px[c] = saturate_cast<uchar>(sum);
                }
            }
        }
    }
}

inline BoxFilterFn& boxfilter_dispatch()
{
    static BoxFilterFn fn = &boxFilter_fallback;
    return fn;
}

inline void register_boxfilter_backend(BoxFilterFn fn)
{
    if (fn)
    {
        boxfilter_dispatch() = fn;
    }
}

inline bool is_boxfilter_backend_registered()
{
    return boxfilter_dispatch() != &boxFilter_fallback;
}

inline void gaussian_blur_fallback(const Mat& src,
                                   Mat& dst,
                                   Size ksize,
                                   double sigmaX,
                                   double sigmaY,
                                   int borderType)
{
    CV_Assert(!src.empty() && "GaussianBlur: source image can not be empty");
    CV_Assert(src.dims == 2 && "GaussianBlur: only 2D Mat is supported");
    CV_Assert(src.depth() == CV_8U && "GaussianBlur: v1 supports CV_8U only");

    int kx = ksize.width;
    int ky = ksize.height;

    if (kx <= 0 && sigmaX > 0.0)
    {
        kx = auto_gaussian_ksize(sigmaX);
    }
    if (ky <= 0 && sigmaY > 0.0)
    {
        ky = auto_gaussian_ksize(sigmaY);
    }

    if (kx <= 0 && ky > 0)
    {
        kx = ky;
    }
    if (ky <= 0 && kx > 0)
    {
        ky = kx;
    }

    if (kx <= 0 || ky <= 0)
    {
        CV_Error_(Error::StsBadArg,
                  ("GaussianBlur: invalid ksize=(%d,%d), sigmaX=%.6f sigmaY=%.6f",
                   ksize.width,
                   ksize.height,
                   sigmaX,
                   sigmaY));
    }

    if ((kx & 1) == 0 || (ky & 1) == 0)
    {
        CV_Error_(Error::StsBadArg, ("GaussianBlur: ksize must be odd, got (%d,%d)", kx, ky));
    }

    if (sigmaX <= 0.0)
    {
        sigmaX = default_gaussian_sigma_for_ksize(kx);
    }
    if (sigmaY <= 0.0)
    {
        sigmaY = sigmaX;
    }

    if (sigmaX <= 0.0 || sigmaY <= 0.0)
    {
        CV_Error_(Error::StsBadArg,
                  ("GaussianBlur: invalid sigmaX/sigmaY after resolve (%.6f, %.6f)", sigmaX, sigmaY));
    }

    const int border_type = normalize_border_type(borderType);
    if (!is_supported_filter_border(border_type))
    {
        CV_Error_(Error::StsBadArg, ("GaussianBlur: unsupported borderType=%d", borderType));
    }

    Mat src_local;
    const Mat* src_ref = &src;
    if (src.data == dst.data)
    {
        src_local = src.clone();
        src_ref = &src_local;
    }

    const int rows = src_ref->size[0];
    const int cols = src_ref->size[1];
    const int channels = src_ref->channels();
    const size_t src_step = src_ref->step(0);

    dst.create(std::vector<int>{rows, cols}, src_ref->type());
    const size_t dst_step = dst.step(0);

    const std::vector<float> kernel_x = build_gaussian_kernel_1d(kx, sigmaX);
    const std::vector<float> kernel_y = build_gaussian_kernel_1d(ky, sigmaY);
    const int rx = kx / 2;
    const int ry = ky / 2;

    std::vector<float> tmp(static_cast<size_t>(rows) * cols * channels, 0.0f);

    for (int y = 0; y < rows; ++y)
    {
        const uchar* src_row = src_ref->data + static_cast<size_t>(y) * src_step;
        for (int x = 0; x < cols; ++x)
        {
            const size_t tmp_base = (static_cast<size_t>(y) * cols + x) * channels;
            for (int c = 0; c < channels; ++c)
            {
                double acc = 0.0;
                for (int k = 0; k < kx; ++k)
                {
                    const int sx = x + k - rx;
                    const int src_x = border_interpolate(sx, cols, border_type);
                    if (src_x < 0)
                    {
                        continue;
                    }
                    acc += static_cast<double>(kernel_x[static_cast<size_t>(k)]) *
                           static_cast<double>(src_row[static_cast<size_t>(src_x) * channels + c]);
                }
                tmp[tmp_base + c] = static_cast<float>(acc);
            }
        }
    }

    for (int y = 0; y < rows; ++y)
    {
        uchar* dst_row = dst.data + static_cast<size_t>(y) * dst_step;
        for (int x = 0; x < cols; ++x)
        {
            uchar* out_px = dst_row + static_cast<size_t>(x) * channels;
            for (int c = 0; c < channels; ++c)
            {
                double acc = 0.0;
                for (int k = 0; k < ky; ++k)
                {
                    const int sy = y + k - ry;
                    const int src_y = border_interpolate(sy, rows, border_type);
                    if (src_y < 0)
                    {
                        continue;
                    }
                    const size_t tmp_idx = (static_cast<size_t>(src_y) * cols + x) * channels + c;
                    acc += static_cast<double>(kernel_y[static_cast<size_t>(k)]) * static_cast<double>(tmp[tmp_idx]);
                }
                out_px[c] = saturate_cast<uchar>(acc);
            }
        }
    }
}

inline GaussianBlurFn& gaussianblur_dispatch()
{
    static GaussianBlurFn fn = &gaussian_blur_fallback;
    return fn;
}

inline void register_gaussianblur_backend(GaussianBlurFn fn)
{
    if (fn)
    {
        gaussianblur_dispatch() = fn;
    }
}

inline bool is_gaussianblur_backend_registered()
{
    return gaussianblur_dispatch() != &gaussian_blur_fallback;
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

inline void resize(const Mat& src, Mat& dst, Size dsize, double fx = 0.0, double fy = 0.0, int interpolation = INTER_LINEAR)
{
    detail::ensure_backends_registered_once();
    detail::resize_dispatch()(src, dst, dsize, fx, fy, interpolation);
}

inline void cvtColor(const Mat& src, Mat& dst, int code)
{
    detail::ensure_backends_registered_once();
    detail::cvtcolor_dispatch()(src, dst, code);
}

inline double threshold(const Mat& src, Mat& dst, double thresh, double maxval, int type)
{
    detail::ensure_backends_registered_once();
    return detail::threshold_dispatch()(src, dst, thresh, maxval, type);
}

inline void boxFilter(const Mat& src,
                      Mat& dst,
                      int ddepth,
                      Size ksize,
                      Point anchor = Point(-1, -1),
                      bool normalize = true,
                      int borderType = BORDER_DEFAULT)
{
    detail::ensure_backends_registered_once();
    detail::boxfilter_dispatch()(src, dst, ddepth, ksize, anchor, normalize, borderType);
}

inline void blur(const Mat& src,
                 Mat& dst,
                 Size ksize,
                 Point anchor = Point(-1, -1),
                 int borderType = BORDER_DEFAULT)
{
    boxFilter(src, dst, -1, ksize, anchor, true, borderType);
}

inline void GaussianBlur(const Mat& src,
                         Mat& dst,
                         Size ksize,
                         double sigmaX,
                         double sigmaY = 0.0,
                         int borderType = BORDER_DEFAULT)
{
    detail::ensure_backends_registered_once();
    detail::gaussianblur_dispatch()(src, dst, ksize, sigmaX, sigmaY, borderType);
}

}  // namespace cvh

#endif  // CVH_IMGPROC_H
