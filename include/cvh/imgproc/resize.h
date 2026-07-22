#ifndef CVH_IMGPROC_RESIZE_H
#define CVH_IMGPROC_RESIZE_H

#include "../detail/config.h"
#include "detail/common.h"
#if CVH_ENABLE_OPENCV_INTRIN
#include "../core/simd/simd.h"
#endif

#include <cstdint>
#include <type_traits>
#include <vector>

namespace cvh {
namespace detail {

using ResizeFn = void (*)(const Mat&, Mat&, Size, double, double, int);

template <typename T>
inline T resize_interpolate_cast(float v)
{
    if constexpr (std::is_same<T, uchar>::value)
    {
        return saturate_cast<uchar>(v);
    }
    else
    {
        return static_cast<T>(v);
    }
}

#if CVH_ENABLE_OPENCV_INTRIN
inline bool resize_linear_u8c1_downsample2_opencv_intrin_supported(const Mat& src,
                                                                  int dst_rows,
                                                                  int dst_cols,
                                                                  int interpolation)
{
    return interpolation == INTER_LINEAR &&
           src.dims == 2 &&
           src.depth() == CV_8U &&
           src.channels() == 1 &&
           src.size[0] == dst_rows * 2 &&
           src.size[1] == dst_cols * 2;
}

inline uchar resize_linear_u8c1_downsample2_pixel(uchar p00, uchar p01, uchar p10, uchar p11)
{
    return static_cast<uchar>(
        (static_cast<int>(p00) +
         static_cast<int>(p01) +
         static_cast<int>(p10) +
         static_cast<int>(p11) +
         2) >> 2);
}

inline void resize_linear_u8c1_downsample2_opencv_intrin_impl(const Mat& src, Mat& dst)
{
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 1);
    CV_Assert(src.dims == 2);

    const int dst_rows = src.size[0] / 2;
    const int dst_cols = src.size[1] / 2;
    dst.create(std::vector<int>{dst_rows, dst_cols}, CV_8UC1);

    const size_t src_step = src.step(0);
    const size_t dst_step = dst.step(0);
    const int lanes = static_cast<int>(simd::u8_lanes());

    for (int y = 0; y < dst_rows; ++y)
    {
        const uchar* src_row0 = src.data + static_cast<size_t>(y * 2) * src_step;
        const uchar* src_row1 = src.data + static_cast<size_t>(y * 2 + 1) * src_step;
        uchar* dst_row = dst.data + static_cast<size_t>(y) * dst_step;

        int x = 0;
        for (; x + lanes <= dst_cols; x += lanes)
        {
            simd::u8 r0_even;
            simd::u8 r0_odd;
            simd::u8 r1_even;
            simd::u8 r1_odd;
            simd::load_deinterleave2_u8(
                reinterpret_cast<const std::uint8_t*>(src_row0 + static_cast<size_t>(x) * 2),
                r0_even,
                r0_odd);
            simd::load_deinterleave2_u8(
                reinterpret_cast<const std::uint8_t*>(src_row1 + static_cast<size_t>(x) * 2),
                r1_even,
                r1_odd);

            simd::u16 r0_even_lo;
            simd::u16 r0_even_hi;
            simd::u16 r0_odd_lo;
            simd::u16 r0_odd_hi;
            simd::u16 r1_even_lo;
            simd::u16 r1_even_hi;
            simd::u16 r1_odd_lo;
            simd::u16 r1_odd_hi;
            simd::expand_u8(r0_even, r0_even_lo, r0_even_hi);
            simd::expand_u8(r0_odd, r0_odd_lo, r0_odd_hi);
            simd::expand_u8(r1_even, r1_even_lo, r1_even_hi);
            simd::expand_u8(r1_odd, r1_odd_lo, r1_odd_hi);

            const simd::u16 sum_lo = simd::add(
                simd::add(r0_even_lo, r0_odd_lo),
                simd::add(r1_even_lo, r1_odd_lo));
            const simd::u16 sum_hi = simd::add(
                simd::add(r0_even_hi, r0_odd_hi),
                simd::add(r1_even_hi, r1_odd_hi));
            simd::store_u8(
                reinterpret_cast<std::uint8_t*>(dst_row + x),
                simd::rshr_pack_u16_to_u8<2>(sum_lo, sum_hi));
        }

        for (; x < dst_cols; ++x)
        {
            const int sx = x * 2;
            dst_row[x] = resize_linear_u8c1_downsample2_pixel(
                src_row0[sx],
                src_row0[sx + 1],
                src_row1[sx],
                src_row1[sx + 1]);
        }
    }
}
#endif

template <typename T>
inline void resize_fallback_impl_typed(const Mat& src, Mat& dst, int interpolation)
{
    const int src_rows = src.size[0];
    const int src_cols = src.size[1];
    const int dst_rows = dst.size[0];
    const int dst_cols = dst.size[1];

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

            const T* src_row = reinterpret_cast<const T*>(src_data + static_cast<size_t>(clamped_sy) * src_step);
            T* dst_row = reinterpret_cast<T*>(dst_data + static_cast<size_t>(y) * dst_step);
            for (int x = 0; x < dst_cols; ++x)
            {
                const int sx = x_ofs[static_cast<size_t>(x)];
                const T* src_px = src_row + static_cast<size_t>(sx) * channels;
                T* dst_px = dst_row + static_cast<size_t>(x) * channels;
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
            const T* src_row = reinterpret_cast<const T*>(src_data + static_cast<size_t>(sy) * src_step);
            T* dst_row = reinterpret_cast<T*>(dst_data + static_cast<size_t>(y) * dst_step);
            for (int x = 0; x < dst_cols; ++x)
            {
                const int sx = std::min(src_cols - 1, (x * src_cols) / dst_cols);
                const T* src_px = src_row + static_cast<size_t>(sx) * channels;
                T* dst_px = dst_row + static_cast<size_t>(x) * channels;
                for (int c = 0; c < channels; ++c)
                {
                    dst_px[c] = src_px[c];
                }
            }
        }
        return;
    }

    const float scale_x = static_cast<float>(src_cols) / static_cast<float>(dst_cols);
    const float scale_y = static_cast<float>(src_rows) / static_cast<float>(dst_rows);

    for (int y = 0; y < dst_rows; ++y)
    {
        const float fy_src = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const int y0 = std::clamp(static_cast<int>(std::floor(fy_src)), 0, src_rows - 1);
        const int y1 = std::min(y0 + 1, src_rows - 1);
        const float wy = fy_src - static_cast<float>(y0);

        const T* src_row0 = reinterpret_cast<const T*>(src_data + static_cast<size_t>(y0) * src_step);
        const T* src_row1 = reinterpret_cast<const T*>(src_data + static_cast<size_t>(y1) * src_step);
        T* dst_row = reinterpret_cast<T*>(dst_data + static_cast<size_t>(y) * dst_step);

        for (int x = 0; x < dst_cols; ++x)
        {
            const float fx_src = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const int x0 = std::clamp(static_cast<int>(std::floor(fx_src)), 0, src_cols - 1);
            const int x1 = std::min(x0 + 1, src_cols - 1);
            const float wx = fx_src - static_cast<float>(x0);

            const T* p00 = src_row0 + static_cast<size_t>(x0) * channels;
            const T* p01 = src_row0 + static_cast<size_t>(x1) * channels;
            const T* p10 = src_row1 + static_cast<size_t>(x0) * channels;
            const T* p11 = src_row1 + static_cast<size_t>(x1) * channels;
            T* dst_px = dst_row + static_cast<size_t>(x) * channels;

            for (int c = 0; c < channels; ++c)
            {
                const float top = detail::lerp(static_cast<float>(p00[c]), static_cast<float>(p01[c]), wx);
                const float bot = detail::lerp(static_cast<float>(p10[c]), static_cast<float>(p11[c]), wx);
                dst_px[c] = resize_interpolate_cast<T>(detail::lerp(top, bot, wy));
            }
        }
    }
}

inline void resize_fallback(const Mat& src, Mat& dst, Size dsize, double fx, double fy, int interpolation)
{
    CV_Assert(!src.empty() && "resize: source image can not be empty");
    CV_Assert(src.dims == 2 && "resize: only 2D Mat is supported");

    const int src_rows = src.size[0];
    const int src_cols = src.size[1];
    const int dst_cols = detail::resolve_resize_dim(src_cols, dsize.width, fx);
    const int dst_rows = detail::resolve_resize_dim(src_rows, dsize.height, fy);
    CV_Assert(dst_cols > 0 && dst_rows > 0 && "resize: invalid output size");

    if (interpolation != INTER_NEAREST &&
        interpolation != INTER_NEAREST_EXACT &&
        interpolation != INTER_LINEAR)
    {
        CV_Error_(Error::StsBadArg, ("resize: unsupported interpolation=%d", interpolation));
    }

    const int src_depth = src.depth();
    if (src_depth != CV_8U && src_depth != CV_32F)
    {
        CV_Error_(Error::StsBadArg, ("resize: unsupported src depth=%d", src_depth));
    }

#if CVH_ENABLE_OPENCV_INTRIN
    if (resize_linear_u8c1_downsample2_opencv_intrin_supported(src, dst_rows, dst_cols, interpolation))
    {
        resize_linear_u8c1_downsample2_opencv_intrin_impl(src, dst);
        return;
    }
#endif

    dst.create(std::vector<int>{dst_rows, dst_cols}, src.type());

    if (src_depth == CV_8U)
    {
        resize_fallback_impl_typed<uchar>(src, dst, interpolation);
        return;
    }

    resize_fallback_impl_typed<float>(src, dst, interpolation);
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

}  // namespace detail

inline void resize(const Mat& src, Mat& dst, Size dsize, double fx = 0.0, double fy = 0.0, int interpolation = INTER_LINEAR)
{
    detail::ensure_backends_registered_once();
    detail::resize_dispatch()(src, dst, dsize, fx, fy, interpolation);
}

}  // namespace cvh

#endif  // CVH_IMGPROC_RESIZE_H
