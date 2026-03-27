#ifndef CVH_IMGPROC_CVTCOLOR_H
#define CVH_IMGPROC_CVTCOLOR_H

#include "detail/common.h"

#include <vector>

namespace cvh {
namespace detail {

using CvtColorFn = void (*)(const Mat&, Mat&, int);

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

}  // namespace detail

inline void cvtColor(const Mat& src, Mat& dst, int code)
{
    detail::ensure_backends_registered_once();
    detail::cvtcolor_dispatch()(src, dst, code);
}

}  // namespace cvh

#endif  // CVH_IMGPROC_CVTCOLOR_H
