#ifndef CVH_IMGPROC_MORPHOLOGY_H
#define CVH_IMGPROC_MORPHOLOGY_H

#include "detail/common.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace cvh {
namespace detail {

using MorphologyFn = void (*)(const Mat&, Mat&, const Mat&, Point, int, int, const Scalar&);

struct MorphKernelSpec
{
    int width = 0;
    int height = 0;
    int anchor_x = 0;
    int anchor_y = 0;
    std::vector<Point> offsets;
};

struct HitMissKernelSpec
{
    int width = 0;
    int height = 0;
    int anchor_x = 0;
    int anchor_y = 0;
    std::vector<Point> foreground_offsets;
    std::vector<Point> background_offsets;
};

inline MorphKernelSpec resolve_morph_kernel(const Mat& kernel, Point anchor)
{
    MorphKernelSpec spec;
    if (kernel.empty())
    {
        spec.width = 3;
        spec.height = 3;
        spec.anchor_x = 1;
        spec.anchor_y = 1;
        for (int ky = 0; ky < 3; ++ky)
        {
            for (int kx = 0; kx < 3; ++kx)
            {
                spec.offsets.push_back(Point(kx - spec.anchor_x, ky - spec.anchor_y));
            }
        }
        return spec;
    }

    CV_Assert(kernel.dims == 2 && "morphology: only 2D kernel is supported");
    CV_Assert(kernel.depth() == CV_8U && kernel.channels() == 1 && "morphology: kernel must be CV_8UC1");

    spec.height = kernel.size[0];
    spec.width = kernel.size[1];
    CV_Assert(spec.width > 0 && spec.height > 0 && "morphology: kernel can not be empty");

    spec.anchor_x = anchor.x >= 0 ? anchor.x : (spec.width / 2);
    spec.anchor_y = anchor.y >= 0 ? anchor.y : (spec.height / 2);
    CV_Assert(spec.anchor_x >= 0 && spec.anchor_x < spec.width);
    CV_Assert(spec.anchor_y >= 0 && spec.anchor_y < spec.height);

    const size_t kstep = kernel.step(0);
    for (int ky = 0; ky < spec.height; ++ky)
    {
        const uchar* row = kernel.data + static_cast<size_t>(ky) * kstep;
        for (int kx = 0; kx < spec.width; ++kx)
        {
            if (row[kx] != 0)
            {
                spec.offsets.push_back(Point(kx - spec.anchor_x, ky - spec.anchor_y));
            }
        }
    }

    CV_Assert(!spec.offsets.empty() && "morphology: kernel has no active elements");
    return spec;
}

inline HitMissKernelSpec resolve_hitmiss_kernel(const Mat& kernel, Point anchor)
{
    CV_Assert(!kernel.empty() && "morphologyEx(HITMISS): kernel can not be empty");
    CV_Assert(kernel.dims == 2 && "morphologyEx(HITMISS): only 2D kernel is supported");
    CV_Assert((kernel.depth() == CV_8U || kernel.depth() == CV_8S) && kernel.channels() == 1 &&
              "morphologyEx(HITMISS): kernel must be CV_8UC1 or CV_8SC1");

    HitMissKernelSpec spec;
    spec.height = kernel.size[0];
    spec.width = kernel.size[1];
    CV_Assert(spec.width > 0 && spec.height > 0);

    spec.anchor_x = anchor.x >= 0 ? anchor.x : (spec.width / 2);
    spec.anchor_y = anchor.y >= 0 ? anchor.y : (spec.height / 2);
    CV_Assert(spec.anchor_x >= 0 && spec.anchor_x < spec.width);
    CV_Assert(spec.anchor_y >= 0 && spec.anchor_y < spec.height);

    const size_t kstep = kernel.step(0);
    if (kernel.depth() == CV_8U)
    {
        for (int ky = 0; ky < spec.height; ++ky)
        {
            const uchar* row = kernel.data + static_cast<size_t>(ky) * kstep;
            for (int kx = 0; kx < spec.width; ++kx)
            {
                if (row[kx] > 0)
                {
                    spec.foreground_offsets.push_back(Point(kx - spec.anchor_x, ky - spec.anchor_y));
                }
            }
        }
        return spec;
    }

    for (int ky = 0; ky < spec.height; ++ky)
    {
        const schar* row = reinterpret_cast<const schar*>(kernel.data + static_cast<size_t>(ky) * kstep);
        for (int kx = 0; kx < spec.width; ++kx)
        {
            const schar v = row[kx];
            if (v > 0)
            {
                spec.foreground_offsets.push_back(Point(kx - spec.anchor_x, ky - spec.anchor_y));
            }
            else if (v < 0)
            {
                spec.background_offsets.push_back(Point(kx - spec.anchor_x, ky - spec.anchor_y));
            }
        }
    }

    return spec;
}

inline void morphology_single_pass_u8(const Mat& src,
                                      Mat& dst,
                                      const MorphKernelSpec& kernel,
                                      int border_type,
                                      const Scalar& border_value,
                                      bool is_erode)
{
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.dims == 2);

    const int rows = src.size[0];
    const int cols = src.size[1];
    const int channels = src.channels();
    const size_t src_step = src.step(0);

    dst.create(std::vector<int>{rows, cols}, src.type());
    const size_t dst_step = dst.step(0);

    for (int y = 0; y < rows; ++y)
    {
        uchar* dst_row = dst.data + static_cast<size_t>(y) * dst_step;
        for (int x = 0; x < cols; ++x)
        {
            uchar* out_px = dst_row + static_cast<size_t>(x) * channels;
            for (int c = 0; c < channels; ++c)
            {
                int best = is_erode ? 255 : 0;
                for (const Point& ofs : kernel.offsets)
                {
                    const int sy = y + ofs.y;
                    const int sx = x + ofs.x;
                    const int by = border_interpolate(sy, rows, border_type);
                    const int bx = border_interpolate(sx, cols, border_type);

                    int value = 0;
                    if (by < 0 || bx < 0)
                    {
                        value = saturate_cast<uchar>(border_value.val[c]);
                    }
                    else
                    {
                        const uchar* src_row = src.data + static_cast<size_t>(by) * src_step;
                        value = src_row[static_cast<size_t>(bx) * channels + c];
                    }

                    if (is_erode)
                    {
                        best = std::min(best, value);
                    }
                    else
                    {
                        best = std::max(best, value);
                    }
                }
                out_px[c] = static_cast<uchar>(best);
            }
        }
    }
}

inline void morphology_fallback(const Mat& src,
                                Mat& dst,
                                const Mat& kernel,
                                Point anchor,
                                int iterations,
                                int borderType,
                                const Scalar& borderValue,
                                bool is_erode)
{
    CV_Assert(!src.empty() && "morphology: source image can not be empty");
    CV_Assert(src.dims == 2 && "morphology: only 2D Mat is supported");
    CV_Assert(iterations > 0 && "morphology: iterations must be > 0");

    if (src.depth() != CV_8U)
    {
        CV_Error_(Error::StsBadArg, ("morphology: only CV_8U is supported for now, got depth=%d", src.depth()));
    }

    const int border_type = normalize_border_type(borderType);
    if (!is_supported_filter_border(border_type))
    {
        CV_Error_(Error::StsBadArg, ("morphology: unsupported borderType=%d", borderType));
    }

    const MorphKernelSpec kernel_spec = resolve_morph_kernel(kernel, anchor);

    Mat src_local;
    const Mat* input0 = &src;
    if (src.data == dst.data)
    {
        src_local = src.clone();
        input0 = &src_local;
    }

    Mat buf_a;
    Mat buf_b;
    const Mat* cur = input0;
    Mat* out = &buf_a;

    for (int i = 0; i < iterations; ++i)
    {
        if (i == iterations - 1)
        {
            out = &dst;
        }

        morphology_single_pass_u8(*cur, *out, kernel_spec, border_type, borderValue, is_erode);

        if (i < iterations - 1)
        {
            cur = out;
            out = (out == &buf_a) ? &buf_b : &buf_a;
        }
    }
}

inline void erode_fallback(const Mat& src,
                           Mat& dst,
                           const Mat& kernel,
                           Point anchor,
                           int iterations,
                           int borderType,
                           const Scalar& borderValue)
{
    morphology_fallback(src, dst, kernel, anchor, iterations, borderType, borderValue, true);
}

inline void dilate_fallback(const Mat& src,
                            Mat& dst,
                            const Mat& kernel,
                            Point anchor,
                            int iterations,
                            int borderType,
                            const Scalar& borderValue)
{
    morphology_fallback(src, dst, kernel, anchor, iterations, borderType, borderValue, false);
}

inline MorphologyFn& erode_dispatch()
{
    static MorphologyFn fn = &erode_fallback;
    return fn;
}

inline MorphologyFn& dilate_dispatch()
{
    static MorphologyFn fn = &dilate_fallback;
    return fn;
}

inline void register_erode_backend(MorphologyFn fn)
{
    if (fn)
    {
        erode_dispatch() = fn;
    }
}

inline void register_dilate_backend(MorphologyFn fn)
{
    if (fn)
    {
        dilate_dispatch() = fn;
    }
}

inline bool is_erode_backend_registered()
{
    return erode_dispatch() != &erode_fallback;
}

inline bool is_dilate_backend_registered()
{
    return dilate_dispatch() != &dilate_fallback;
}

}  // namespace detail

inline void erode(const Mat& src,
                  Mat& dst,
                  const Mat& kernel = Mat(),
                  Point anchor = Point(-1, -1),
                  int iterations = 1,
                  int borderType = BORDER_DEFAULT,
                  const Scalar& borderValue = Scalar::all(255.0))
{
    detail::ensure_backends_registered_once();
    detail::erode_dispatch()(src, dst, kernel, anchor, iterations, borderType, borderValue);
}

inline void dilate(const Mat& src,
                   Mat& dst,
                   const Mat& kernel = Mat(),
                   Point anchor = Point(-1, -1),
                   int iterations = 1,
                   int borderType = BORDER_DEFAULT,
                   const Scalar& borderValue = Scalar::all(0.0))
{
    detail::ensure_backends_registered_once();
    detail::dilate_dispatch()(src, dst, kernel, anchor, iterations, borderType, borderValue);
}

namespace detail {

inline void morphology_gradient_u8(const Mat& dilated, const Mat& eroded, Mat& dst)
{
    CV_Assert(dilated.type() == eroded.type());
    CV_Assert(dilated.size[0] == eroded.size[0] && dilated.size[1] == eroded.size[1]);
    CV_Assert(dilated.depth() == CV_8U);

    dst.create(std::vector<int>{dilated.size[0], dilated.size[1]}, dilated.type());
    const size_t count = dilated.total() * static_cast<size_t>(dilated.channels());
    for (size_t i = 0; i < count; ++i)
    {
        const int value = static_cast<int>(dilated.data[i]) - static_cast<int>(eroded.data[i]);
        dst.data[i] = static_cast<uchar>(value < 0 ? 0 : value);
    }
}

inline void morphology_sub_u8(const Mat& lhs, const Mat& rhs, Mat& dst)
{
    CV_Assert(lhs.type() == rhs.type());
    CV_Assert(lhs.size[0] == rhs.size[0] && lhs.size[1] == rhs.size[1]);
    CV_Assert(lhs.depth() == CV_8U);

    dst.create(std::vector<int>{lhs.size[0], lhs.size[1]}, lhs.type());
    const size_t count = lhs.total() * static_cast<size_t>(lhs.channels());
    for (size_t i = 0; i < count; ++i)
    {
        const int value = static_cast<int>(lhs.data[i]) - static_cast<int>(rhs.data[i]);
        dst.data[i] = static_cast<uchar>(value < 0 ? 0 : value);
    }
}

inline void morphology_hitmiss_u8(const Mat& src, Mat& dst, const HitMissKernelSpec& kernel)
{
    CV_Assert(src.depth() == CV_8U && src.channels() == 1 && src.dims == 2);
    const int rows = src.size[0];
    const int cols = src.size[1];
    const size_t src_step = src.step(0);

    dst.create(std::vector<int>{rows, cols}, src.type());
    const size_t dst_step = dst.step(0);

    // OpenCV regression behavior: all-zero kernel is identity.
    if (kernel.foreground_offsets.empty() && kernel.background_offsets.empty())
    {
        src.copyTo(dst);
        return;
    }

    for (int y = 0; y < rows; ++y)
    {
        uchar* out_row = dst.data + static_cast<size_t>(y) * dst_step;
        for (int x = 0; x < cols; ++x)
        {
            bool matched = true;
            for (const Point& ofs : kernel.foreground_offsets)
            {
                const int sy = y + ofs.y;
                const int sx = x + ofs.x;
                if (sy < 0 || sy >= rows || sx < 0 || sx >= cols)
                {
                    continue;
                }

                const uchar* src_row = src.data + static_cast<size_t>(sy) * src_step;
                if (src_row[sx] == 0)
                {
                    matched = false;
                    break;
                }
            }

            if (matched)
            {
                for (const Point& ofs : kernel.background_offsets)
                {
                    const int sy = y + ofs.y;
                    const int sx = x + ofs.x;
                    if (sy < 0 || sy >= rows || sx < 0 || sx >= cols)
                    {
                        continue;
                    }

                    const uchar* src_row = src.data + static_cast<size_t>(sy) * src_step;
                    if (src_row[sx] != 0)
                    {
                        matched = false;
                        break;
                    }
                }
            }

            out_row[x] = matched ? static_cast<uchar>(255) : static_cast<uchar>(0);
        }
    }
}

}  // namespace detail

inline void morphologyEx(const Mat& src,
                         Mat& dst,
                         int op,
                         const Mat& kernel,
                         Point anchor = Point(-1, -1),
                         int iterations = 1,
                         int borderType = BORDER_CONSTANT,
                         const Scalar& borderValue = Scalar::all(0.0))
{
    detail::ensure_backends_registered_once();

    if (op == MORPH_ERODE)
    {
        detail::erode_dispatch()(src, dst, kernel, anchor, iterations, borderType, borderValue);
        return;
    }

    if (op == MORPH_DILATE)
    {
        detail::dilate_dispatch()(src, dst, kernel, anchor, iterations, borderType, borderValue);
        return;
    }

    if (op == MORPH_OPEN)
    {
        Mat tmp;
        detail::erode_dispatch()(src, tmp, kernel, anchor, iterations, borderType, borderValue);
        detail::dilate_dispatch()(tmp, dst, kernel, anchor, iterations, borderType, borderValue);
        return;
    }

    if (op == MORPH_CLOSE)
    {
        Mat tmp;
        detail::dilate_dispatch()(src, tmp, kernel, anchor, iterations, borderType, borderValue);
        detail::erode_dispatch()(tmp, dst, kernel, anchor, iterations, borderType, borderValue);
        return;
    }

    if (op == MORPH_GRADIENT)
    {
        Mat dilated;
        Mat eroded;
        detail::dilate_dispatch()(src, dilated, kernel, anchor, iterations, borderType, borderValue);
        detail::erode_dispatch()(src, eroded, kernel, anchor, iterations, borderType, borderValue);
        detail::morphology_gradient_u8(dilated, eroded, dst);
        return;
    }

    if (op == MORPH_TOPHAT)
    {
        Mat opened;
        morphologyEx(src, opened, MORPH_OPEN, kernel, anchor, iterations, borderType, borderValue);
        detail::morphology_sub_u8(src, opened, dst);
        return;
    }

    if (op == MORPH_BLACKHAT)
    {
        Mat closed;
        morphologyEx(src, closed, MORPH_CLOSE, kernel, anchor, iterations, borderType, borderValue);
        detail::morphology_sub_u8(closed, src, dst);
        return;
    }

    if (op == MORPH_HITMISS)
    {
        CV_Assert(!src.empty() && "morphologyEx(HITMISS): source image can not be empty");
        CV_Assert(src.dims == 2 && "morphologyEx(HITMISS): only 2D Mat is supported");
        if (src.depth() != CV_8U || src.channels() != 1)
        {
            CV_Error_(Error::StsBadArg,
                      ("morphologyEx(HITMISS): only CV_8UC1 is supported, got type=%d", src.type()));
        }

        Mat src_local;
        const Mat* src_ref = &src;
        if (src.data == dst.data)
        {
            src_local = src.clone();
            src_ref = &src_local;
        }

        const detail::HitMissKernelSpec spec = detail::resolve_hitmiss_kernel(kernel, anchor);
        detail::morphology_hitmiss_u8(*src_ref, dst, spec);
        return;
    }

    CV_Error_(Error::StsBadArg,
              ("morphologyEx: unsupported op=%d (supported ops: MORPH_ERODE, MORPH_DILATE, MORPH_OPEN, MORPH_CLOSE, MORPH_GRADIENT, MORPH_TOPHAT, MORPH_BLACKHAT, MORPH_HITMISS)", op));
}

}  // namespace cvh

#endif  // CVH_IMGPROC_MORPHOLOGY_H
