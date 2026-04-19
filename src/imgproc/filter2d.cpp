#include "fastpath_common.h"

namespace cvh
{
namespace detail
{

namespace
{
bool try_filter2d_fastpath(const Mat& src,
                           Mat& dst,
                           int ddepth,
                           const Mat& kernel,
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

    if (kernel.empty() || kernel.dims != 2 || kernel.channels() != 1 || kernel.depth() != CV_32F)
    {
        return false;
    }

    const int krows = kernel.size[0];
    const int kcols = kernel.size[1];
    if (krows <= 0 || kcols <= 0)
    {
        return false;
    }

    const int ax = anchor.x >= 0 ? anchor.x : (kcols / 2);
    const int ay = anchor.y >= 0 ? anchor.y : (krows / 2);
    if (ax < 0 || ax >= kcols || ay < 0 || ay >= krows)
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

    std::vector<float> kernel_coeffs(static_cast<std::size_t>(krows) * static_cast<std::size_t>(kcols), 0.0f);
    for (int ky = 0; ky < krows; ++ky)
    {
        for (int kx = 0; kx < kcols; ++kx)
        {
            kernel_coeffs[static_cast<std::size_t>(ky) * static_cast<std::size_t>(kcols) + static_cast<std::size_t>(kx)] =
                kernel.at<float>(ky, kx);
        }
    }

    std::vector<int> x_offsets(static_cast<std::size_t>(cols) * static_cast<std::size_t>(kcols), -1);
    for (int x = 0; x < cols; ++x)
    {
        int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kcols);
        for (int kx = 0; kx < kcols; ++kx)
        {
            const int sx = border_interpolate(x + kx - ax, cols, border_type);
            x_ofs[kx] = sx >= 0 ? sx * channels : -1;
        }
    }

    std::vector<int> y_indices(static_cast<std::size_t>(rows) * static_cast<std::size_t>(krows), -1);
    for (int y = 0; y < rows; ++y)
    {
        int* y_idx = y_indices.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(krows);
        for (int ky = 0; ky < krows; ++ky)
        {
            y_idx[ky] = border_interpolate(y + ky - ay, rows, border_type);
        }
    }

    dst.create(std::vector<int>{rows, cols}, CV_MAKETYPE(out_depth, channels));
    const std::size_t src_step = src_ref->step(0);
    const std::size_t dst_step = dst.step(0);
    const bool do_parallel = should_parallelize_filter_rows(rows, cols, channels, krows * kcols);

    if (src_depth == CV_8U)
    {
        const uchar* src_data = src_ref->data;
        parallel_for_index_if(do_parallel, rows, [&](int y) {
            const int* y_idx = y_indices.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(krows);
            uchar* dst_row_u8 = out_depth == CV_8U ? (dst.data + static_cast<std::size_t>(y) * dst_step) : nullptr;
            float* dst_row_f32 =
                out_depth == CV_32F ? reinterpret_cast<float*>(dst.data + static_cast<std::size_t>(y) * dst_step) : nullptr;

            for (int x = 0; x < cols; ++x)
            {
                const int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kcols);
                const std::size_t out_base = static_cast<std::size_t>(x) * static_cast<std::size_t>(channels);

                if (channels == 1)
                {
                    double acc0 = delta;
                    for (int ky = 0; ky < krows; ++ky)
                    {
                        const int sy = y_idx[ky];
                        if (sy < 0)
                        {
                            continue;
                        }
                        const uchar* src_row = src_data + static_cast<std::size_t>(sy) * src_step;
                        const float* krow = kernel_coeffs.data() + static_cast<std::size_t>(ky) * static_cast<std::size_t>(kcols);
                        for (int kx = 0; kx < kcols; ++kx)
                        {
                            const int sx = x_ofs[kx];
                            if (sx < 0)
                            {
                                continue;
                            }
                            acc0 += static_cast<double>(krow[kx]) * static_cast<double>(src_row[sx]);
                        }
                    }

                    if (dst_row_f32)
                    {
                        dst_row_f32[out_base] = static_cast<float>(acc0);
                    }
                    else
                    {
                        dst_row_u8[out_base] = saturate_cast<uchar>(acc0);
                    }
                    continue;
                }

                if (channels == 3)
                {
                    double acc0 = delta;
                    double acc1 = delta;
                    double acc2 = delta;
                    for (int ky = 0; ky < krows; ++ky)
                    {
                        const int sy = y_idx[ky];
                        if (sy < 0)
                        {
                            continue;
                        }
                        const uchar* src_row = src_data + static_cast<std::size_t>(sy) * src_step;
                        const float* krow = kernel_coeffs.data() + static_cast<std::size_t>(ky) * static_cast<std::size_t>(kcols);
                        for (int kx = 0; kx < kcols; ++kx)
                        {
                            const int sx = x_ofs[kx];
                            if (sx < 0)
                            {
                                continue;
                            }
                            const float w = krow[kx];
                            acc0 += static_cast<double>(w) * static_cast<double>(src_row[sx + 0]);
                            acc1 += static_cast<double>(w) * static_cast<double>(src_row[sx + 1]);
                            acc2 += static_cast<double>(w) * static_cast<double>(src_row[sx + 2]);
                        }
                    }

                    if (dst_row_f32)
                    {
                        dst_row_f32[out_base + 0] = static_cast<float>(acc0);
                        dst_row_f32[out_base + 1] = static_cast<float>(acc1);
                        dst_row_f32[out_base + 2] = static_cast<float>(acc2);
                    }
                    else
                    {
                        dst_row_u8[out_base + 0] = saturate_cast<uchar>(acc0);
                        dst_row_u8[out_base + 1] = saturate_cast<uchar>(acc1);
                        dst_row_u8[out_base + 2] = saturate_cast<uchar>(acc2);
                    }
                    continue;
                }

                if (channels == 4)
                {
                    double acc0 = delta;
                    double acc1 = delta;
                    double acc2 = delta;
                    double acc3 = delta;
                    for (int ky = 0; ky < krows; ++ky)
                    {
                        const int sy = y_idx[ky];
                        if (sy < 0)
                        {
                            continue;
                        }
                        const uchar* src_row = src_data + static_cast<std::size_t>(sy) * src_step;
                        const float* krow = kernel_coeffs.data() + static_cast<std::size_t>(ky) * static_cast<std::size_t>(kcols);
                        for (int kx = 0; kx < kcols; ++kx)
                        {
                            const int sx = x_ofs[kx];
                            if (sx < 0)
                            {
                                continue;
                            }
                            const float w = krow[kx];
                            acc0 += static_cast<double>(w) * static_cast<double>(src_row[sx + 0]);
                            acc1 += static_cast<double>(w) * static_cast<double>(src_row[sx + 1]);
                            acc2 += static_cast<double>(w) * static_cast<double>(src_row[sx + 2]);
                            acc3 += static_cast<double>(w) * static_cast<double>(src_row[sx + 3]);
                        }
                    }

                    if (dst_row_f32)
                    {
                        dst_row_f32[out_base + 0] = static_cast<float>(acc0);
                        dst_row_f32[out_base + 1] = static_cast<float>(acc1);
                        dst_row_f32[out_base + 2] = static_cast<float>(acc2);
                        dst_row_f32[out_base + 3] = static_cast<float>(acc3);
                    }
                    else
                    {
                        dst_row_u8[out_base + 0] = saturate_cast<uchar>(acc0);
                        dst_row_u8[out_base + 1] = saturate_cast<uchar>(acc1);
                        dst_row_u8[out_base + 2] = saturate_cast<uchar>(acc2);
                        dst_row_u8[out_base + 3] = saturate_cast<uchar>(acc3);
                    }
                    continue;
                }

                for (int c = 0; c < channels; ++c)
                {
                    double acc = delta;
                    for (int ky = 0; ky < krows; ++ky)
                    {
                        const int sy = y_idx[ky];
                        if (sy < 0)
                        {
                            continue;
                        }
                        const uchar* src_row = src_data + static_cast<std::size_t>(sy) * src_step;
                        const float* krow = kernel_coeffs.data() + static_cast<std::size_t>(ky) * static_cast<std::size_t>(kcols);
                        for (int kx = 0; kx < kcols; ++kx)
                        {
                            const int sx = x_ofs[kx];
                            if (sx < 0)
                            {
                                continue;
                            }
                            acc += static_cast<double>(krow[kx]) * static_cast<double>(src_row[sx + c]);
                        }
                    }

                    if (dst_row_f32)
                    {
                        dst_row_f32[out_base + static_cast<std::size_t>(c)] = static_cast<float>(acc);
                    }
                    else
                    {
                        dst_row_u8[out_base + static_cast<std::size_t>(c)] = saturate_cast<uchar>(acc);
                    }
                }
            }
        });
        return true;
    }

    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const int* y_idx = y_indices.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(krows);
        uchar* dst_row_u8 = out_depth == CV_8U ? (dst.data + static_cast<std::size_t>(y) * dst_step) : nullptr;
        float* dst_row_f32 =
            out_depth == CV_32F ? reinterpret_cast<float*>(dst.data + static_cast<std::size_t>(y) * dst_step) : nullptr;

        for (int x = 0; x < cols; ++x)
        {
            const int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kcols);
            const std::size_t out_base = static_cast<std::size_t>(x) * static_cast<std::size_t>(channels);

            if (channels == 1)
            {
                double acc0 = delta;
                for (int ky = 0; ky < krows; ++ky)
                {
                    const int sy = y_idx[ky];
                    if (sy < 0)
                    {
                        continue;
                    }
                    const float* src_row =
                        reinterpret_cast<const float*>(src_ref->data + static_cast<std::size_t>(sy) * src_step);
                    const float* krow = kernel_coeffs.data() + static_cast<std::size_t>(ky) * static_cast<std::size_t>(kcols);
                    for (int kx = 0; kx < kcols; ++kx)
                    {
                        const int sx = x_ofs[kx];
                        if (sx < 0)
                        {
                            continue;
                        }
                        acc0 += static_cast<double>(krow[kx]) * static_cast<double>(src_row[sx]);
                    }
                }

                if (dst_row_f32)
                {
                    dst_row_f32[out_base] = static_cast<float>(acc0);
                }
                else
                {
                    dst_row_u8[out_base] = saturate_cast<uchar>(acc0);
                }
                continue;
            }

            if (channels == 3)
            {
                double acc0 = delta;
                double acc1 = delta;
                double acc2 = delta;
                for (int ky = 0; ky < krows; ++ky)
                {
                    const int sy = y_idx[ky];
                    if (sy < 0)
                    {
                        continue;
                    }
                    const float* src_row =
                        reinterpret_cast<const float*>(src_ref->data + static_cast<std::size_t>(sy) * src_step);
                    const float* krow = kernel_coeffs.data() + static_cast<std::size_t>(ky) * static_cast<std::size_t>(kcols);
                    for (int kx = 0; kx < kcols; ++kx)
                    {
                        const int sx = x_ofs[kx];
                        if (sx < 0)
                        {
                            continue;
                        }
                        const float w = krow[kx];
                        acc0 += static_cast<double>(w) * static_cast<double>(src_row[sx + 0]);
                        acc1 += static_cast<double>(w) * static_cast<double>(src_row[sx + 1]);
                        acc2 += static_cast<double>(w) * static_cast<double>(src_row[sx + 2]);
                    }
                }

                if (dst_row_f32)
                {
                    dst_row_f32[out_base + 0] = static_cast<float>(acc0);
                    dst_row_f32[out_base + 1] = static_cast<float>(acc1);
                    dst_row_f32[out_base + 2] = static_cast<float>(acc2);
                }
                else
                {
                    dst_row_u8[out_base + 0] = saturate_cast<uchar>(acc0);
                    dst_row_u8[out_base + 1] = saturate_cast<uchar>(acc1);
                    dst_row_u8[out_base + 2] = saturate_cast<uchar>(acc2);
                }
                continue;
            }

            if (channels == 4)
            {
                double acc0 = delta;
                double acc1 = delta;
                double acc2 = delta;
                double acc3 = delta;
                for (int ky = 0; ky < krows; ++ky)
                {
                    const int sy = y_idx[ky];
                    if (sy < 0)
                    {
                        continue;
                    }
                    const float* src_row =
                        reinterpret_cast<const float*>(src_ref->data + static_cast<std::size_t>(sy) * src_step);
                    const float* krow = kernel_coeffs.data() + static_cast<std::size_t>(ky) * static_cast<std::size_t>(kcols);
                    for (int kx = 0; kx < kcols; ++kx)
                    {
                        const int sx = x_ofs[kx];
                        if (sx < 0)
                        {
                            continue;
                        }
                        const float w = krow[kx];
                        acc0 += static_cast<double>(w) * static_cast<double>(src_row[sx + 0]);
                        acc1 += static_cast<double>(w) * static_cast<double>(src_row[sx + 1]);
                        acc2 += static_cast<double>(w) * static_cast<double>(src_row[sx + 2]);
                        acc3 += static_cast<double>(w) * static_cast<double>(src_row[sx + 3]);
                    }
                }

                if (dst_row_f32)
                {
                    dst_row_f32[out_base + 0] = static_cast<float>(acc0);
                    dst_row_f32[out_base + 1] = static_cast<float>(acc1);
                    dst_row_f32[out_base + 2] = static_cast<float>(acc2);
                    dst_row_f32[out_base + 3] = static_cast<float>(acc3);
                }
                else
                {
                    dst_row_u8[out_base + 0] = saturate_cast<uchar>(acc0);
                    dst_row_u8[out_base + 1] = saturate_cast<uchar>(acc1);
                    dst_row_u8[out_base + 2] = saturate_cast<uchar>(acc2);
                    dst_row_u8[out_base + 3] = saturate_cast<uchar>(acc3);
                }
                continue;
            }

            for (int c = 0; c < channels; ++c)
            {
                double acc = delta;
                for (int ky = 0; ky < krows; ++ky)
                {
                    const int sy = y_idx[ky];
                    if (sy < 0)
                    {
                        continue;
                    }
                    const float* src_row =
                        reinterpret_cast<const float*>(src_ref->data + static_cast<std::size_t>(sy) * src_step);
                    const float* krow = kernel_coeffs.data() + static_cast<std::size_t>(ky) * static_cast<std::size_t>(kcols);
                    for (int kx = 0; kx < kcols; ++kx)
                    {
                        const int sx = x_ofs[kx];
                        if (sx < 0)
                        {
                            continue;
                        }
                        acc += static_cast<double>(krow[kx]) * static_cast<double>(src_row[sx + c]);
                    }
                }

                if (dst_row_f32)
                {
                    dst_row_f32[out_base + static_cast<std::size_t>(c)] = static_cast<float>(acc);
                }
                else
                {
                    dst_row_u8[out_base + static_cast<std::size_t>(c)] = saturate_cast<uchar>(acc);
                }
            }
        }
    });

    return true;
}


} // namespace

void filter2d_backend_impl(const Mat& src,
                           Mat& dst,
                           int ddepth,
                           const Mat& kernel,
                           Point anchor,
                           double delta,
                           int borderType)
{
    if (try_filter2d_fastpath(src, dst, ddepth, kernel, anchor, delta, borderType))
    {
        return;
    }

    filter2D_fallback(src, dst, ddepth, kernel, anchor, delta, borderType);
}

} // namespace detail
} // namespace cvh
