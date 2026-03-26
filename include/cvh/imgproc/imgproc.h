#ifndef CVH_IMGPROC_H
#define CVH_IMGPROC_H

#include "../core/mat.h"
#include "../core/saturate.h"
#include "../core/types.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace cvh {

enum InterpolationFlags
{
    INTER_NEAREST = 0,
    INTER_LINEAR = 1,
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

}  // namespace detail

inline void resize(const Mat& src, Mat& dst, Size dsize, double fx = 0.0, double fy = 0.0, int interpolation = INTER_LINEAR)
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

inline void cvtColor(const Mat& src, Mat& dst, int code)
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

inline double threshold(const Mat& src, Mat& dst, double thresh, double maxval, int type)
{
    CV_Assert(!src.empty() && "threshold: source image can not be empty");
    CV_Assert(src.depth() == CV_8U && "threshold: v1 supports CV_8U only");

    if (type != THRESH_BINARY && type != THRESH_BINARY_INV)
    {
        CV_Error_(Error::StsBadArg, ("threshold: unsupported threshold type=%d", type));
    }

    dst.create(src.dims, src.size.p, src.type());

    const uchar max_u8 = saturate_cast<uchar>(maxval);
    const size_t scalar_count = src.total() * static_cast<size_t>(src.channels());
    const uchar* src_ptr = src.data;
    uchar* dst_ptr = dst.data;

    if (src.isContinuous() && dst.isContinuous())
    {
        for (size_t i = 0; i < scalar_count; ++i)
        {
            const bool cond = static_cast<double>(src_ptr[i]) > thresh;
            dst_ptr[i] = (type == THRESH_BINARY) ? (cond ? max_u8 : 0) : (cond ? 0 : max_u8);
        }
        return thresh;
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
            const bool cond = static_cast<double>(src_row[x]) > thresh;
            dst_row[x] = (type == THRESH_BINARY) ? (cond ? max_u8 : 0) : (cond ? 0 : max_u8);
        }
    }

    return thresh;
}

}  // namespace cvh

#endif  // CVH_IMGPROC_H
