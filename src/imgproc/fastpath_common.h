#ifndef CVH_IMGPROC_BACKEND_FASTPATH_COMMON_H
#define CVH_IMGPROC_BACKEND_FASTPATH_COMMON_H

#include "cvh/imgproc/imgproc.h"
#include "cvh/core/parallel.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace cvh
{
namespace detail
{

inline bool is_u8_fastpath_channels(int cn)
{
    return cn == 1 || cn == 3 || cn == 4;
}

inline bool should_parallelize_resize(int rows, int cols, int channels)
{
    return cvh::cpu::should_parallelize_1d_loop(
        static_cast<std::size_t>(rows),
        static_cast<std::size_t>(cols) * static_cast<std::size_t>(channels),
        1LL << 16,
        2);
}

template <class Fn>
inline void parallel_for_index_if(bool do_parallel, int end, Fn&& fn)
{
    if (end <= 0)
    {
        return;
    }

    if (!do_parallel)
    {
        for (int i = 0; i < end; ++i)
        {
            fn(i);
        }
        return;
    }

    cvh::parallel_for_(
        cvh::Range(0, end),
        [&](const cvh::Range& range) {
            for (int i = range.start; i < range.end; ++i)
            {
                fn(i);
            }
        },
        static_cast<double>(end));
}

template <class Fn>
inline void parallel_for_index_if_step(bool do_parallel, int begin, int end, int step, Fn&& fn)
{
    CV_Assert(step > 0);
    if (begin >= end)
    {
        return;
    }

    const int count = (end - begin + step - 1) / step;
    parallel_for_index_if(
        do_parallel,
        count,
        [&](int idx) {
            fn(begin + idx * step);
        });
}

template <class Fn>
inline void parallel_for_count_if(bool do_parallel, std::size_t count, Fn&& fn)
{
    if (count == 0)
    {
        return;
    }

    const std::size_t max_parallel_range = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (!do_parallel || count > max_parallel_range)
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            fn(i);
        }
        return;
    }

    cvh::parallel_for_(
        cvh::Range(0, static_cast<int>(count)),
        [&](const cvh::Range& range) {
            for (int i = range.start; i < range.end; ++i)
            {
                fn(static_cast<std::size_t>(i));
            }
        },
        static_cast<double>(count));
}

inline bool is_boxfilter_3x3_candidate(Size ksize, Point anchor, bool normalize)
{
    if (!normalize || ksize.width != 3 || ksize.height != 3)
    {
        return false;
    }

    const int anchor_x = anchor.x >= 0 ? anchor.x : (ksize.width / 2);
    const int anchor_y = anchor.y >= 0 ? anchor.y : (ksize.height / 2);
    return anchor_x == 1 && anchor_y == 1;
}

inline bool resolve_gaussian_kernel_size(Size ksize, double sigmaX, double sigmaY, int& kx, int& ky)
{
    kx = ksize.width;
    ky = ksize.height;
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
    return kx > 0 && ky > 0 && (kx & 1) != 0 && (ky & 1) != 0;
}

inline bool should_parallelize_cvtcolor(int rows, int cols, int src_channels)
{
    return cvh::cpu::should_parallelize_1d_loop(
        static_cast<std::size_t>(rows),
        static_cast<std::size_t>(cols) * static_cast<std::size_t>(src_channels),
        1LL << 15,
        2);
}

inline bool should_parallelize_filter_rows(int rows, int cols, int channels, int taps)
{
    return cvh::cpu::should_parallelize_1d_loop(
        static_cast<std::size_t>(rows),
        static_cast<std::size_t>(cols) * static_cast<std::size_t>(channels) * static_cast<std::size_t>(std::max(1, taps)),
        1LL << 16,
        2);
}

inline bool should_parallelize_threshold_contiguous(std::size_t scalar_count)
{
    return cvh::cpu::should_parallelize_1d_loop(
        scalar_count,
        1,
        1LL << 17,
        2);
}

inline bool should_parallelize_threshold_rows(int rows, int cols_scalar)
{
    return cvh::cpu::should_parallelize_1d_loop(
        static_cast<std::size_t>(rows),
        static_cast<std::size_t>(cols_scalar),
        1LL << 16,
        2);
}

inline std::vector<int> build_extended_index_map(int len, int left, int right, int border_type)
{
    CV_Assert(len > 0);
    CV_Assert(left >= 0 && right >= 0);

    const int ext_len = len + left + right;
    std::vector<int> map(static_cast<std::size_t>(ext_len), -1);
    for (int i = 0; i < ext_len; ++i)
    {
        map[static_cast<std::size_t>(i)] = border_interpolate(i - left, len, border_type);
    }
    return map;
}

inline bool is_morph_rect3x3_kernel(const Mat& kernel, Point anchor)
{
    if (kernel.empty())
    {
        return true;
    }

    if (kernel.dims != 2 || kernel.depth() != CV_8U || kernel.channels() != 1)
    {
        return false;
    }

    if (kernel.size[1] != 3 || kernel.size[0] != 3)
    {
        return false;
    }

    const int anchor_x = anchor.x >= 0 ? anchor.x : 1;
    const int anchor_y = anchor.y >= 0 ? anchor.y : 1;
    if (anchor_x != 1 || anchor_y != 1)
    {
        return false;
    }

    const std::size_t kstep = kernel.step(0);
    for (int ky = 0; ky < 3; ++ky)
    {
        const uchar* row = kernel.data + static_cast<std::size_t>(ky) * kstep;
        for (int kx = 0; kx < 3; ++kx)
        {
            if (row[kx] == 0)
            {
                return false;
            }
        }
    }
    return true;
}

} // namespace detail
} // namespace cvh

#endif // CVH_IMGPROC_BACKEND_FASTPATH_COMMON_H
