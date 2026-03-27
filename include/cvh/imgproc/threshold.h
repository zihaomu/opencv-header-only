#ifndef CVH_IMGPROC_THRESHOLD_H
#define CVH_IMGPROC_THRESHOLD_H

#include "detail/common.h"

#include <array>
#include <cstdint>

namespace cvh {
namespace detail {

using ThresholdFn = double (*)(const Mat&, Mat&, double, double, int);

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

}  // namespace detail

inline double threshold(const Mat& src, Mat& dst, double thresh, double maxval, int type)
{
    detail::ensure_backends_registered_once();
    return detail::threshold_dispatch()(src, dst, thresh, maxval, type);
}

}  // namespace cvh

#endif  // CVH_IMGPROC_THRESHOLD_H
