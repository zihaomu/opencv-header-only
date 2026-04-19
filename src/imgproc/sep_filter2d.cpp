#include "fastpath_common.h"

namespace cvh
{
namespace detail
{

namespace
{
bool try_sep_filter2d_fastpath(const Mat& src,
                               Mat& dst,
                               int ddepth,
                               const Mat& kernelX,
                               const Mat& kernelY,
                               Point anchor,
                               double delta,
                               int borderType)
{
    if (src.empty() || src.dims != 2)
    {
        return false;
    }

    const int src_depth = src.depth();
    if (src_depth != CV_8U && src_depth != CV_32F)
    {
        return false;
    }

    std::vector<float> kx;
    std::vector<float> ky;
    if (kernelX.empty() || kernelY.empty())
    {
        return false;
    }
    if (kernelX.dims != 2 || kernelY.dims != 2 ||
        kernelX.channels() != 1 || kernelY.channels() != 1 ||
        kernelX.depth() != CV_32F || kernelY.depth() != CV_32F)
    {
        return false;
    }
    sepfilter2d_collect_kernel(kernelX, kx, "kernelX");
    sepfilter2d_collect_kernel(kernelY, ky, "kernelY");

    const int kx_len = static_cast<int>(kx.size());
    const int ky_len = static_cast<int>(ky.size());
    const int ax = anchor.x >= 0 ? anchor.x : (kx_len / 2);
    const int ay = anchor.y >= 0 ? anchor.y : (ky_len / 2);
    if (ax < 0 || ax >= kx_len || ay < 0 || ay >= ky_len)
    {
        return false;
    }

    int out_depth = ddepth;
    if (out_depth == -1)
    {
        out_depth = src_depth;
    }
    if (out_depth != CV_8U && out_depth != CV_32F)
    {
        return false;
    }

    const int border_type = normalize_border_type(borderType);
    if (!is_supported_filter_border(border_type))
    {
        return false;
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
    if (rows <= 0 || cols <= 0 || channels <= 0)
    {
        return false;
    }

    const int row_stride = cols * channels;
    std::vector<int> x_offsets(static_cast<std::size_t>(cols) * static_cast<std::size_t>(kx_len), -1);
    for (int x = 0; x < cols; ++x)
    {
        int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kx_len);
        for (int i = 0; i < kx_len; ++i)
        {
            const int sx = border_interpolate(x + i - ax, cols, border_type);
            x_ofs[i] = sx >= 0 ? sx * channels : -1;
        }
    }

    std::vector<int> y_offsets(static_cast<std::size_t>(rows) * static_cast<std::size_t>(ky_len), -1);
    for (int y = 0; y < rows; ++y)
    {
        int* y_ofs = y_offsets.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(ky_len);
        for (int i = 0; i < ky_len; ++i)
        {
            const int sy = border_interpolate(y + i - ay, rows, border_type);
            y_ofs[i] = sy >= 0 ? sy * row_stride : -1;
        }
    }

    std::vector<float> tmp(static_cast<std::size_t>(rows) * static_cast<std::size_t>(row_stride), 0.0f);
    const std::size_t src_step = src_ref->step(0);
    const bool do_parallel_h = should_parallelize_filter_rows(rows, cols, channels, kx_len);

    if (src_depth == CV_8U)
    {
        parallel_for_index_if(do_parallel_h, rows, [&](int y) {
            const uchar* src_row = src_ref->data + static_cast<std::size_t>(y) * src_step;
            float* tmp_row = tmp.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(row_stride);

            for (int x = 0; x < cols; ++x)
            {
                const int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kx_len);
                const int dx = x * channels;

                if (channels == 1)
                {
                    float acc0 = 0.0f;
                    for (int i = 0; i < kx_len; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        acc0 += kx[static_cast<std::size_t>(i)] * static_cast<float>(src_row[sx]);
                    }
                    tmp_row[dx] = acc0;
                    continue;
                }

                if (channels == 3)
                {
                    float acc0 = 0.0f;
                    float acc1 = 0.0f;
                    float acc2 = 0.0f;
                    for (int i = 0; i < kx_len; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        const float w = kx[static_cast<std::size_t>(i)];
                        const uchar* px = src_row + sx;
                        acc0 += w * static_cast<float>(px[0]);
                        acc1 += w * static_cast<float>(px[1]);
                        acc2 += w * static_cast<float>(px[2]);
                    }
                    tmp_row[dx + 0] = acc0;
                    tmp_row[dx + 1] = acc1;
                    tmp_row[dx + 2] = acc2;
                    continue;
                }

                if (channels == 4)
                {
                    float acc0 = 0.0f;
                    float acc1 = 0.0f;
                    float acc2 = 0.0f;
                    float acc3 = 0.0f;
                    for (int i = 0; i < kx_len; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        const float w = kx[static_cast<std::size_t>(i)];
                        const uchar* px = src_row + sx;
                        acc0 += w * static_cast<float>(px[0]);
                        acc1 += w * static_cast<float>(px[1]);
                        acc2 += w * static_cast<float>(px[2]);
                        acc3 += w * static_cast<float>(px[3]);
                    }
                    tmp_row[dx + 0] = acc0;
                    tmp_row[dx + 1] = acc1;
                    tmp_row[dx + 2] = acc2;
                    tmp_row[dx + 3] = acc3;
                    continue;
                }

                for (int c = 0; c < channels; ++c)
                {
                    float acc = 0.0f;
                    for (int i = 0; i < kx_len; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        acc += kx[static_cast<std::size_t>(i)] * static_cast<float>(src_row[sx + c]);
                    }
                    tmp_row[dx + c] = acc;
                }
            }
        });
    }
    else
    {
        parallel_for_index_if(do_parallel_h, rows, [&](int y) {
            const float* src_row = reinterpret_cast<const float*>(src_ref->data + static_cast<std::size_t>(y) * src_step);
            float* tmp_row = tmp.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(row_stride);

            for (int x = 0; x < cols; ++x)
            {
                const int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kx_len);
                const int dx = x * channels;

                if (channels == 1)
                {
                    float acc0 = 0.0f;
                    for (int i = 0; i < kx_len; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        acc0 += kx[static_cast<std::size_t>(i)] * src_row[sx];
                    }
                    tmp_row[dx] = acc0;
                    continue;
                }

                if (channels == 3)
                {
                    float acc0 = 0.0f;
                    float acc1 = 0.0f;
                    float acc2 = 0.0f;
                    for (int i = 0; i < kx_len; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        const float w = kx[static_cast<std::size_t>(i)];
                        const float* px = src_row + sx;
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                    }
                    tmp_row[dx + 0] = acc0;
                    tmp_row[dx + 1] = acc1;
                    tmp_row[dx + 2] = acc2;
                    continue;
                }

                if (channels == 4)
                {
                    float acc0 = 0.0f;
                    float acc1 = 0.0f;
                    float acc2 = 0.0f;
                    float acc3 = 0.0f;
                    for (int i = 0; i < kx_len; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        const float w = kx[static_cast<std::size_t>(i)];
                        const float* px = src_row + sx;
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                        acc3 += w * px[3];
                    }
                    tmp_row[dx + 0] = acc0;
                    tmp_row[dx + 1] = acc1;
                    tmp_row[dx + 2] = acc2;
                    tmp_row[dx + 3] = acc3;
                    continue;
                }

                for (int c = 0; c < channels; ++c)
                {
                    float acc = 0.0f;
                    for (int i = 0; i < kx_len; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        acc += kx[static_cast<std::size_t>(i)] * src_row[sx + c];
                    }
                    tmp_row[dx + c] = acc;
                }
            }
        });
    }

    dst.create(std::vector<int>{rows, cols}, CV_MAKETYPE(out_depth, channels));
    const std::size_t dst_step = dst.step(0);
    const bool do_parallel_v = should_parallelize_filter_rows(rows, cols, channels, ky_len);

    parallel_for_index_if(do_parallel_v, rows, [&](int y) {
        const int* y_ofs = y_offsets.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(ky_len);
        uchar* dst_row_u8 = out_depth == CV_8U ? (dst.data + static_cast<std::size_t>(y) * dst_step) : nullptr;
        float* dst_row_f32 =
            out_depth == CV_32F ? reinterpret_cast<float*>(dst.data + static_cast<std::size_t>(y) * dst_step) : nullptr;

        for (int x = 0; x < cols; ++x)
        {
            const int dx = x * channels;
            if (channels == 1)
            {
                float acc0 = static_cast<float>(delta);
                for (int i = 0; i < ky_len; ++i)
                {
                    const int sy = y_ofs[i];
                    if (sy < 0)
                    {
                        continue;
                    }
                    acc0 += ky[static_cast<std::size_t>(i)] * tmp[static_cast<std::size_t>(sy + dx)];
                }
                if (dst_row_f32)
                {
                    dst_row_f32[dx] = acc0;
                }
                else
                {
                    dst_row_u8[dx] = saturate_cast<uchar>(acc0);
                }
                continue;
            }

            if (channels == 3)
            {
                float acc0 = static_cast<float>(delta);
                float acc1 = static_cast<float>(delta);
                float acc2 = static_cast<float>(delta);
                for (int i = 0; i < ky_len; ++i)
                {
                    const int sy = y_ofs[i];
                    if (sy < 0)
                    {
                        continue;
                    }
                    const float w = ky[static_cast<std::size_t>(i)];
                    const float* px = tmp.data() + static_cast<std::size_t>(sy + dx);
                    acc0 += w * px[0];
                    acc1 += w * px[1];
                    acc2 += w * px[2];
                }
                if (dst_row_f32)
                {
                    dst_row_f32[dx + 0] = acc0;
                    dst_row_f32[dx + 1] = acc1;
                    dst_row_f32[dx + 2] = acc2;
                }
                else
                {
                    dst_row_u8[dx + 0] = saturate_cast<uchar>(acc0);
                    dst_row_u8[dx + 1] = saturate_cast<uchar>(acc1);
                    dst_row_u8[dx + 2] = saturate_cast<uchar>(acc2);
                }
                continue;
            }

            if (channels == 4)
            {
                float acc0 = static_cast<float>(delta);
                float acc1 = static_cast<float>(delta);
                float acc2 = static_cast<float>(delta);
                float acc3 = static_cast<float>(delta);
                for (int i = 0; i < ky_len; ++i)
                {
                    const int sy = y_ofs[i];
                    if (sy < 0)
                    {
                        continue;
                    }
                    const float w = ky[static_cast<std::size_t>(i)];
                    const float* px = tmp.data() + static_cast<std::size_t>(sy + dx);
                    acc0 += w * px[0];
                    acc1 += w * px[1];
                    acc2 += w * px[2];
                    acc3 += w * px[3];
                }
                if (dst_row_f32)
                {
                    dst_row_f32[dx + 0] = acc0;
                    dst_row_f32[dx + 1] = acc1;
                    dst_row_f32[dx + 2] = acc2;
                    dst_row_f32[dx + 3] = acc3;
                }
                else
                {
                    dst_row_u8[dx + 0] = saturate_cast<uchar>(acc0);
                    dst_row_u8[dx + 1] = saturate_cast<uchar>(acc1);
                    dst_row_u8[dx + 2] = saturate_cast<uchar>(acc2);
                    dst_row_u8[dx + 3] = saturate_cast<uchar>(acc3);
                }
                continue;
            }

            for (int c = 0; c < channels; ++c)
            {
                float acc = static_cast<float>(delta);
                for (int i = 0; i < ky_len; ++i)
                {
                    const int sy = y_ofs[i];
                    if (sy < 0)
                    {
                        continue;
                    }
                    acc += ky[static_cast<std::size_t>(i)] * tmp[static_cast<std::size_t>(sy + dx + c)];
                }
                if (dst_row_f32)
                {
                    dst_row_f32[dx + c] = acc;
                }
                else
                {
                    dst_row_u8[dx + c] = saturate_cast<uchar>(acc);
                }
            }
        }
    });

    return true;
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


} // namespace

void sep_filter2d_backend_impl(const Mat& src,
                               Mat& dst,
                               int ddepth,
                               const Mat& kernelX,
                               const Mat& kernelY,
                               Point anchor,
                               double delta,
                               int borderType)
{
    if (try_sep_filter2d_fastpath(src, dst, ddepth, kernelX, kernelY, anchor, delta, borderType))
    {
        return;
    }

    sepFilter2D_fallback(src, dst, ddepth, kernelX, kernelY, anchor, delta, borderType);
}

} // namespace detail
} // namespace cvh
