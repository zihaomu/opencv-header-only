#include "fastpath_common.h"

namespace cvh
{
namespace detail
{

namespace
{
bool try_sobel_fastpath_u8(const Mat& src,
                           Mat& dst,
                           int ddepth,
                           int dx,
                           int dy,
                           int ksize,
                           double scale,
                           double delta,
                           int borderType)
{
    if (src.empty() || src.dims != 2 || src.depth() != CV_8U)
    {
        return false;
    }

    if (!((dx == 1 && dy == 0) || (dx == 0 && dy == 1)))
    {
        return false;
    }

    if (ksize != 3 && ksize != 5)
    {
        return false;
    }

    int out_depth = ddepth;
    if (out_depth < 0)
    {
        out_depth = CV_32F;
    }
    if (out_depth != CV_32F && out_depth != CV_16S)
    {
        return false;
    }

    const bool isolated_border = (borderType & BORDER_ISOLATED) != 0;
    const int border_type = normalize_border_type(borderType);
    if (!is_supported_filter_border(border_type))
    {
        return false;
    }
    if (border_type == BORDER_CONSTANT)
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

    const SobelSamplingWindow sample_window = resolve_sobel_sampling_window(*src_ref, isolated_border);
    const int rows = src_ref->size[0];
    const int cols = src_ref->size[1];
    const int channels = src_ref->channels();
    if (rows <= 0 || cols <= 0 || channels <= 0)
    {
        return false;
    }

    if (ksize == 5)
    {
        const int taps = 5;
        const int radius = 2;

        int hcoeff[5] = {0, 0, 0, 0, 0};
        int vcoeff[5] = {0, 0, 0, 0, 0};
        if (dx == 1)
        {
            hcoeff[0] = -1; hcoeff[1] = -2; hcoeff[2] = 0; hcoeff[3] = 2; hcoeff[4] = 1;
            vcoeff[0] = 1; vcoeff[1] = 4; vcoeff[2] = 6; vcoeff[3] = 4; vcoeff[4] = 1;
        }
        else
        {
            hcoeff[0] = 1; hcoeff[1] = 4; hcoeff[2] = 6; hcoeff[3] = 4; hcoeff[4] = 1;
            vcoeff[0] = -1; vcoeff[1] = -2; vcoeff[2] = 0; vcoeff[3] = 2; vcoeff[4] = 1;
        }

        std::vector<int> x_offsets(static_cast<std::size_t>(cols) * static_cast<std::size_t>(taps), -1);
        for (int x = 0; x < cols; ++x)
        {
            const int base_x = x + sample_window.col_offset;
            int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(taps);
            for (int i = 0; i < taps; ++i)
            {
                const int sx = border_interpolate(base_x + i - radius, sample_window.cols, border_type);
                x_ofs[i] = sx * channels;
            }
        }

        std::vector<int> y_indices(static_cast<std::size_t>(rows) * static_cast<std::size_t>(taps), -1);
        for (int y = 0; y < rows; ++y)
        {
            const int base_y = y + sample_window.row_offset;
            int* y_idx = y_indices.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(taps);
            for (int i = 0; i < taps; ++i)
            {
                y_idx[i] = border_interpolate(base_y + i - radius, sample_window.rows, border_type);
            }
        }

        const int tmp_rows = sample_window.rows;
        const int tmp_stride = cols * channels;
        std::vector<int> tmp(static_cast<std::size_t>(tmp_rows) * static_cast<std::size_t>(tmp_stride), 0);

        const std::size_t src_step = src_ref->step(0);
        const uchar* base_data = sample_window.base_data;
        const bool do_parallel_h = should_parallelize_filter_rows(tmp_rows, cols, channels, taps);
        parallel_for_index_if(do_parallel_h, tmp_rows, [&](int py) {
            const uchar* src_row = base_data + static_cast<std::size_t>(py) * src_step;
            int* tmp_row = tmp.data() + static_cast<std::size_t>(py) * static_cast<std::size_t>(tmp_stride);

            for (int x = 0; x < cols; ++x)
            {
                const int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(taps);
                const int dx_base = x * channels;
                for (int c = 0; c < channels; ++c)
                {
                    int acc = 0;
                    acc += hcoeff[0] * static_cast<int>(src_row[x_ofs[0] + c]);
                    acc += hcoeff[1] * static_cast<int>(src_row[x_ofs[1] + c]);
                    acc += hcoeff[2] * static_cast<int>(src_row[x_ofs[2] + c]);
                    acc += hcoeff[3] * static_cast<int>(src_row[x_ofs[3] + c]);
                    acc += hcoeff[4] * static_cast<int>(src_row[x_ofs[4] + c]);
                    tmp_row[dx_base + c] = acc;
                }
            }
        });

        dst.create(std::vector<int>{rows, cols}, CV_MAKETYPE(out_depth, channels));
        const std::size_t dst_step = dst.step(0);
        const bool do_parallel_v = should_parallelize_filter_rows(rows, cols, channels, taps);

        if (out_depth == CV_32F)
        {
            parallel_for_index_if(do_parallel_v, rows, [&](int y) {
                float* dst_row = reinterpret_cast<float*>(dst.data + static_cast<std::size_t>(y) * dst_step);
                const int* y_idx = y_indices.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(taps);
                const int* row0 = tmp.data() + static_cast<std::size_t>(y_idx[0]) * static_cast<std::size_t>(tmp_stride);
                const int* row1 = tmp.data() + static_cast<std::size_t>(y_idx[1]) * static_cast<std::size_t>(tmp_stride);
                const int* row2 = tmp.data() + static_cast<std::size_t>(y_idx[2]) * static_cast<std::size_t>(tmp_stride);
                const int* row3 = tmp.data() + static_cast<std::size_t>(y_idx[3]) * static_cast<std::size_t>(tmp_stride);
                const int* row4 = tmp.data() + static_cast<std::size_t>(y_idx[4]) * static_cast<std::size_t>(tmp_stride);

                for (int x = 0; x < cols; ++x)
                {
                    const int dx_base = x * channels;
                    for (int c = 0; c < channels; ++c)
                    {
                        int acc = 0;
                        acc += vcoeff[0] * row0[dx_base + c];
                        acc += vcoeff[1] * row1[dx_base + c];
                        acc += vcoeff[2] * row2[dx_base + c];
                        acc += vcoeff[3] * row3[dx_base + c];
                        acc += vcoeff[4] * row4[dx_base + c];
                        dst_row[dx_base + c] = static_cast<float>(static_cast<double>(acc) * scale + delta);
                    }
                }
            });
            return true;
        }

        parallel_for_index_if(do_parallel_v, rows, [&](int y) {
            short* dst_row = reinterpret_cast<short*>(dst.data + static_cast<std::size_t>(y) * dst_step);
            const int* y_idx = y_indices.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(taps);
            const int* row0 = tmp.data() + static_cast<std::size_t>(y_idx[0]) * static_cast<std::size_t>(tmp_stride);
            const int* row1 = tmp.data() + static_cast<std::size_t>(y_idx[1]) * static_cast<std::size_t>(tmp_stride);
            const int* row2 = tmp.data() + static_cast<std::size_t>(y_idx[2]) * static_cast<std::size_t>(tmp_stride);
            const int* row3 = tmp.data() + static_cast<std::size_t>(y_idx[3]) * static_cast<std::size_t>(tmp_stride);
            const int* row4 = tmp.data() + static_cast<std::size_t>(y_idx[4]) * static_cast<std::size_t>(tmp_stride);

            for (int x = 0; x < cols; ++x)
            {
                const int dx_base = x * channels;
                for (int c = 0; c < channels; ++c)
                {
                    int acc = 0;
                    acc += vcoeff[0] * row0[dx_base + c];
                    acc += vcoeff[1] * row1[dx_base + c];
                    acc += vcoeff[2] * row2[dx_base + c];
                    acc += vcoeff[3] * row3[dx_base + c];
                    acc += vcoeff[4] * row4[dx_base + c];
                    dst_row[dx_base + c] = saturate_cast<short>(static_cast<double>(acc) * scale + delta);
                }
            }
        });
        return true;
    }

    std::vector<int> x_offsets(static_cast<std::size_t>(cols) * 3u, -1);
    for (int x = 0; x < cols; ++x)
    {
        const int base_x = x + sample_window.col_offset;
        int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * 3u;
        for (int i = 0; i < 3; ++i)
        {
            const int sx = border_interpolate(base_x + i - 1, sample_window.cols, border_type);
            x_ofs[i] = sx >= 0 ? sx * channels : -1;
        }
    }

    std::vector<int> y_indices(static_cast<std::size_t>(rows) * 3u, -1);
    for (int y = 0; y < rows; ++y)
    {
        const int base_y = y + sample_window.row_offset;
        int* y_idx = y_indices.data() + static_cast<std::size_t>(y) * 3u;
        for (int i = 0; i < 3; ++i)
        {
            y_idx[i] = border_interpolate(base_y + i - 1, sample_window.rows, border_type);
        }
    }

    dst.create(std::vector<int>{rows, cols}, CV_MAKETYPE(out_depth, channels));
    const std::size_t dst_step = dst.step(0);
    const std::size_t src_step = src_ref->step(0);
    const uchar* base_data = sample_window.base_data;
    const bool do_parallel = should_parallelize_filter_rows(rows, cols, channels, 9);

    if (out_depth == CV_32F)
    {
        parallel_for_index_if(do_parallel, rows, [&](int y) {
            float* dst_row = reinterpret_cast<float*>(dst.data + static_cast<std::size_t>(y) * dst_step);
            const int* y_idx = y_indices.data() + static_cast<std::size_t>(y) * 3u;

            const uchar* row0 = base_data + static_cast<std::size_t>(y_idx[0]) * src_step;
            const uchar* row1 = base_data + static_cast<std::size_t>(y_idx[1]) * src_step;
            const uchar* row2 = base_data + static_cast<std::size_t>(y_idx[2]) * src_step;

            for (int x = 0; x < cols; ++x)
            {
                const int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * 3u;
                const int sx0 = x_ofs[0];
                const int sx1 = x_ofs[1];
                const int sx2 = x_ofs[2];
                const int dx_base = x * channels;

                for (int c = 0; c < channels; ++c)
                {
                    const int p00 = static_cast<int>(row0[sx0 + c]);
                    const int p01 = static_cast<int>(row0[sx1 + c]);
                    const int p02 = static_cast<int>(row0[sx2 + c]);
                    const int p10 = static_cast<int>(row1[sx0 + c]);
                    const int p12 = static_cast<int>(row1[sx2 + c]);
                    const int p20 = static_cast<int>(row2[sx0 + c]);
                    const int p21 = static_cast<int>(row2[sx1 + c]);
                    const int p22 = static_cast<int>(row2[sx2 + c]);

                    int gv = 0;
                    if (dx == 1)
                    {
                        gv = (p02 - p00) + ((p12 - p10) << 1) + (p22 - p20);
                    }
                    else
                    {
                        gv = (p20 + (p21 << 1) + p22) - (p00 + (p01 << 1) + p02);
                    }

                    dst_row[dx_base + c] = static_cast<float>(static_cast<double>(gv) * scale + delta);
                }
            }
        });
        return true;
    }

    parallel_for_index_if(do_parallel, rows, [&](int y) {
        short* dst_row = reinterpret_cast<short*>(dst.data + static_cast<std::size_t>(y) * dst_step);
        const int* y_idx = y_indices.data() + static_cast<std::size_t>(y) * 3u;

        const uchar* row0 = base_data + static_cast<std::size_t>(y_idx[0]) * src_step;
        const uchar* row1 = base_data + static_cast<std::size_t>(y_idx[1]) * src_step;
        const uchar* row2 = base_data + static_cast<std::size_t>(y_idx[2]) * src_step;

        for (int x = 0; x < cols; ++x)
        {
            const int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * 3u;
            const int sx0 = x_ofs[0];
            const int sx1 = x_ofs[1];
            const int sx2 = x_ofs[2];
            const int dx_base = x * channels;

            for (int c = 0; c < channels; ++c)
            {
                const int p00 = static_cast<int>(row0[sx0 + c]);
                const int p01 = static_cast<int>(row0[sx1 + c]);
                const int p02 = static_cast<int>(row0[sx2 + c]);
                const int p10 = static_cast<int>(row1[sx0 + c]);
                const int p12 = static_cast<int>(row1[sx2 + c]);
                const int p20 = static_cast<int>(row2[sx0 + c]);
                const int p21 = static_cast<int>(row2[sx1 + c]);
                const int p22 = static_cast<int>(row2[sx2 + c]);

                int gv = 0;
                if (dx == 1)
                {
                    gv = (p02 - p00) + ((p12 - p10) << 1) + (p22 - p20);
                }
                else
                {
                    gv = (p20 + (p21 << 1) + p22) - (p00 + (p01 << 1) + p02);
                }

                const double out_v = static_cast<double>(gv) * scale + delta;
                dst_row[dx_base + c] = saturate_cast<short>(out_v);
            }
        }
    });

    return true;
}


} // namespace

void sobel_backend_impl(const Mat& src,
                        Mat& dst,
                        int ddepth,
                        int dx,
                        int dy,
                        int ksize,
                        double scale,
                        double delta,
                        int borderType)
{
    if (try_sobel_fastpath_u8(src, dst, ddepth, dx, dy, ksize, scale, delta, borderType))
    {
        return;
    }

    sobel_fallback(src, dst, ddepth, dx, dy, ksize, scale, delta, borderType);
}

} // namespace detail
} // namespace cvh
