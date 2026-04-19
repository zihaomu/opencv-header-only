#ifndef CVH_IMGPROC_SOBEL_H
#define CVH_IMGPROC_SOBEL_H

#include "detail/common.h"

#include <cstddef>
#include <vector>

namespace cvh {
namespace detail {

using SobelFn = void (*)(const Mat&, Mat&, int, int, int, int, double, double, int);

inline float sobel_sample_as_f32(const Mat& src, int y, int x, int c)
{
    const size_t src_step = src.step(0);
    if (src.depth() == CV_8U)
    {
        const uchar* row = src.data + static_cast<size_t>(y) * src_step;
        return static_cast<float>(row[static_cast<size_t>(x) * src.channels() + c]);
    }

    const float* row = reinterpret_cast<const float*>(src.data + static_cast<size_t>(y) * src_step);
    return row[static_cast<size_t>(x) * src.channels() + c];
}

inline double sobel_sample_window_as_f64(const uchar* base_data,
                                         size_t row_step,
                                         int depth,
                                         int channels,
                                         int y,
                                         int x,
                                         int c)
{
    if (y < 0 || x < 0)
    {
        return 0.0;
    }

    const uchar* row_ptr = base_data + static_cast<size_t>(y) * row_step;
    if (depth == CV_8U)
    {
        return static_cast<double>(row_ptr[static_cast<size_t>(x) * channels + c]);
    }

    if (depth == CV_16S)
    {
        const short* row = reinterpret_cast<const short*>(row_ptr);
        return static_cast<double>(row[static_cast<size_t>(x) * channels + c]);
    }

    const float* row = reinterpret_cast<const float*>(row_ptr);
    return static_cast<double>(row[static_cast<size_t>(x) * channels + c]);
}

inline std::vector<int> sobel_kernel_1d(int ksize, bool derivative)
{
    if (ksize == 3)
    {
        return derivative ? std::vector<int>{-1, 0, 1} : std::vector<int>{1, 2, 1};
    }
    if (ksize == 5)
    {
        return derivative ? std::vector<int>{-1, -2, 0, 2, 1} : std::vector<int>{1, 4, 6, 4, 1};
    }

    CV_Error_(Error::StsBadArg, ("Sobel: only ksize=3 or 5 is supported for now, got %d", ksize));
    return std::vector<int>();
}

struct SobelSamplingWindow
{
    const uchar* base_data = nullptr;
    int rows = 0;
    int cols = 0;
    int row_offset = 0;
    int col_offset = 0;
    bool use_parent_window = false;
};

inline SobelSamplingWindow resolve_sobel_sampling_window(const Mat& src, bool use_isolated_border)
{
    SobelSamplingWindow window;
    window.base_data = src.data;
    window.rows = src.size[0];
    window.cols = src.size[1];
    window.row_offset = 0;
    window.col_offset = 0;
    window.use_parent_window = false;

    if (use_isolated_border || !src.u || !src.u->data || src.data == src.u->data)
    {
        return window;
    }

    const size_t row_step = src.step(0);
    const size_t pixel_step = src.elemSize();
    if (row_step == 0 || pixel_step == 0 || row_step < pixel_step)
    {
        return window;
    }

    const std::ptrdiff_t byte_offset = src.data - src.u->data;
    if (byte_offset < 0)
    {
        return window;
    }

    const size_t offset = static_cast<size_t>(byte_offset);
    if (offset >= src.u->size)
    {
        return window;
    }

    const size_t row_offset = offset / row_step;
    const size_t row_inner_bytes = offset % row_step;
    if (row_inner_bytes % pixel_step != 0)
    {
        return window;
    }

    const size_t col_offset = row_inner_bytes / pixel_step;
    const size_t parent_rows = src.u->size / row_step;
    const size_t parent_cols = row_step / pixel_step;
    if (parent_rows == 0 || parent_cols == 0)
    {
        return window;
    }

    if (row_offset + static_cast<size_t>(src.size[0]) > parent_rows ||
        col_offset + static_cast<size_t>(src.size[1]) > parent_cols)
    {
        return window;
    }

    window.base_data = src.u->data;
    window.rows = static_cast<int>(parent_rows);
    window.cols = static_cast<int>(parent_cols);
    window.row_offset = static_cast<int>(row_offset);
    window.col_offset = static_cast<int>(col_offset);
    window.use_parent_window = true;
    return window;
}

inline void sobel_fallback(const Mat& src,
                           Mat& dst,
                           int ddepth,
                           int dx,
                           int dy,
                           int ksize,
                           double scale,
                           double delta,
                           int borderType)
{
    CV_Assert(!src.empty() && "Sobel: source image can not be empty");
    CV_Assert(src.dims == 2 && "Sobel: only 2D Mat is supported");

    const int src_depth = src.depth();
    if (src_depth != CV_8U && src_depth != CV_16S && src_depth != CV_32F)
    {
        CV_Error_(Error::StsBadArg, ("Sobel: unsupported src depth=%d", src_depth));
    }

    const std::vector<int> kernel_x = sobel_kernel_1d(ksize, dx == 1);
    const std::vector<int> kernel_y = sobel_kernel_1d(ksize, dy == 1);

    if (!((dx == 1 && dy == 0) || (dx == 0 && dy == 1)))
    {
        CV_Error_(Error::StsBadArg, ("Sobel: only (dx,dy)=(1,0) or (0,1) is supported, got (%d,%d)", dx, dy));
    }

    int out_depth = ddepth;
    if (out_depth < 0)
    {
        out_depth = CV_32F;
    }
    if (out_depth != CV_32F && out_depth != CV_16S)
    {
        CV_Error_(Error::StsBadArg, ("Sobel: only ddepth=CV_16S/CV_32F is supported, got %d", ddepth));
    }

    const bool isolated_border = (borderType & BORDER_ISOLATED) != 0;
    const int border_type = normalize_border_type(borderType);
    if (!is_supported_filter_border(border_type))
    {
        CV_Error_(Error::StsBadArg, ("Sobel: unsupported borderType=%d", borderType));
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
    const SobelSamplingWindow sample_window = resolve_sobel_sampling_window(*src_ref, isolated_border);

    dst.create(std::vector<int>{rows, cols}, CV_MAKETYPE(out_depth, channels));
    const size_t dst_step = dst.step(0);
    const int radius = ksize / 2;

    for (int y = 0; y < rows; ++y)
    {
        float* dst_row_f32 = out_depth == CV_32F
                                 ? reinterpret_cast<float*>(dst.data + static_cast<size_t>(y) * dst_step)
                                 : nullptr;
        short* dst_row_s16 = out_depth == CV_16S
                                 ? reinterpret_cast<short*>(dst.data + static_cast<size_t>(y) * dst_step)
                                 : nullptr;
        for (int x = 0; x < cols; ++x)
        {
            for (int c = 0; c < channels; ++c)
            {
                const int base_y = y + sample_window.row_offset;
                const int base_x = x + sample_window.col_offset;

                double value = 0.0;
                for (int ky = 0; ky < ksize; ++ky)
                {
                    const int sample_y = border_interpolate(base_y + ky - radius, sample_window.rows, border_type);
                    const int wy = kernel_y[static_cast<size_t>(ky)];
                    for (int kx = 0; kx < ksize; ++kx)
                    {
                        const int sample_x = border_interpolate(base_x + kx - radius, sample_window.cols, border_type);
                        const int wx = kernel_x[static_cast<size_t>(kx)];
                        const double sample = sobel_sample_window_as_f64(
                            sample_window.base_data, src_step, src_depth, channels, sample_y, sample_x, c);
                        value += sample * static_cast<double>(wy) * static_cast<double>(wx);
                    }
                }

                const double out_value = value * scale + delta;
                if (dst_row_f32)
                {
                    dst_row_f32[static_cast<size_t>(x) * channels + c] = static_cast<float>(out_value);
                }
                else
                {
                    dst_row_s16[static_cast<size_t>(x) * channels + c] = saturate_cast<short>(out_value);
                }
            }
        }
    }
}

inline SobelFn& sobel_dispatch()
{
    static SobelFn fn = &sobel_fallback;
    return fn;
}

inline void register_sobel_backend(SobelFn fn)
{
    if (fn)
    {
        sobel_dispatch() = fn;
    }
}

inline bool is_sobel_backend_registered()
{
    return sobel_dispatch() != &sobel_fallback;
}

}  // namespace detail

inline void Sobel(const Mat& src,
                  Mat& dst,
                  int ddepth,
                  int dx,
                  int dy,
                  int ksize = 3,
                  double scale = 1.0,
                  double delta = 0.0,
                  int borderType = BORDER_DEFAULT)
{
    detail::ensure_backends_registered_once();
    detail::sobel_dispatch()(src, dst, ddepth, dx, dy, ksize, scale, delta, borderType);
}

}  // namespace cvh

#endif  // CVH_IMGPROC_SOBEL_H
