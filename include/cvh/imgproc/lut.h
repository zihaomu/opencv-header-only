#ifndef CVH_IMGPROC_LUT_H
#define CVH_IMGPROC_LUT_H

#include "detail/common.h"

#include <cstddef>
#include <cstring>
#include <vector>

namespace cvh {
namespace detail {

using LUTFn = void (*)(const Mat&, const Mat&, Mat&);

inline const uchar* lut_entry_base_ptr(const Mat& lut, int index)
{
    CV_Assert(index >= 0 && index < 256);
    if (lut.isContinuous())
    {
        return lut.data + static_cast<size_t>(index) * lut.elemSize();
    }

    CV_Assert(lut.dims == 2 && "LUT: non-contiguous lut supports 2D Mat only");
    const int cols = lut.size[1];
    CV_Assert(cols > 0);
    const int row = index / cols;
    const int col = index - row * cols;
    return lut.data + static_cast<size_t>(row) * lut.step(0) + static_cast<size_t>(col) * lut.elemSize();
}

inline void LUT_fallback(const Mat& src, const Mat& lut, Mat& dst)
{
    CV_Assert(!src.empty() && "LUT: source image can not be empty");
    CV_Assert(!lut.empty() && "LUT: lookup table can not be empty");
    CV_Assert(src.dims == 2 && "LUT: only 2D source Mat is supported");
    CV_Assert(src.depth() == CV_8U && "LUT: source depth must be CV_8U");

    if (lut.total() != 256)
    {
        CV_Error_(Error::StsBadArg, ("LUT: lookup table must contain 256 elements, got total=%zu", lut.total()));
    }

    const int src_cn = src.channels();
    const int lut_cn = lut.channels();
    if (lut_cn != 1 && lut_cn != src_cn)
    {
        CV_Error_(Error::StsBadArg,
                  ("LUT: lut channels must be 1 or equal to src channels (src_cn=%d, lut_cn=%d)", src_cn, lut_cn));
    }

    Mat src_local;
    Mat lut_local;
    const Mat* src_ref = &src;
    const Mat* lut_ref = &lut;

    if (dst.data && dst.data == src.data)
    {
        src_local = src.clone();
        src_ref = &src_local;
    }
    if (dst.data && dst.data == lut.data)
    {
        lut_local = lut.clone();
        lut_ref = &lut_local;
    }

    const int rows = src_ref->size[0];
    const int cols = src_ref->size[1];
    const int dst_type = CV_MAKETYPE(lut_ref->depth(), src_cn);
    dst.create(std::vector<int>{rows, cols}, dst_type);

    const size_t src_step = src_ref->step(0);
    const size_t dst_step = dst.step(0);
    const size_t lut_elem_size1 = lut_ref->elemSize1();
    const size_t dst_pixel_size = static_cast<size_t>(src_cn) * lut_elem_size1;

    for (int y = 0; y < rows; ++y)
    {
        const uchar* src_row = src_ref->data + static_cast<size_t>(y) * src_step;
        uchar* dst_row = dst.data + static_cast<size_t>(y) * dst_step;

        for (int x = 0; x < cols; ++x)
        {
            const uchar* src_px = src_row + static_cast<size_t>(x) * src_cn;
            uchar* dst_px = dst_row + static_cast<size_t>(x) * dst_pixel_size;

            for (int c = 0; c < src_cn; ++c)
            {
                const int lut_index = static_cast<int>(src_px[c]);
                const int lut_channel = (lut_cn == 1) ? 0 : c;
                const uchar* lut_base = lut_entry_base_ptr(*lut_ref, lut_index);
                const uchar* lut_value = lut_base + static_cast<size_t>(lut_channel) * lut_elem_size1;
                std::memcpy(dst_px + static_cast<size_t>(c) * lut_elem_size1, lut_value, lut_elem_size1);
            }
        }
    }
}

inline LUTFn& lut_dispatch()
{
    static LUTFn fn = &LUT_fallback;
    return fn;
}

inline void register_lut_backend(LUTFn fn)
{
    if (fn)
    {
        lut_dispatch() = fn;
    }
}

inline bool is_lut_backend_registered()
{
    return lut_dispatch() != &LUT_fallback;
}

}  // namespace detail

inline void LUT(const Mat& src, const Mat& lut, Mat& dst)
{
    detail::ensure_backends_registered_once();
    detail::lut_dispatch()(src, lut, dst);
}

}  // namespace cvh

#endif  // CVH_IMGPROC_LUT_H
