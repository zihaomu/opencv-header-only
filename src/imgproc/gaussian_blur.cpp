#include "fastpath_common.h"

namespace cvh
{
namespace detail
{

namespace
{
thread_local const char* g_last_gaussianblur_dispatch_path = "fallback";

inline void set_last_gaussianblur_dispatch_path(const char* path)
{
    g_last_gaussianblur_dispatch_path = path ? path : "fallback";
}

bool try_gaussian_blur_fastpath_u8(const Mat& src, Mat& dst, Size ksize, double sigmaX, double sigmaY, int borderType)
{
    if (src.empty() || src.dims != 2 || src.depth() != CV_8U)
    {
        return false;
    }

    if (!is_u8_fastpath_channels(src.channels()))
    {
        return false;
    }

    int kx = ksize.width;
    int ky = ksize.height;

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

    if (kx <= 0 || ky <= 0 || (kx & 1) == 0 || (ky & 1) == 0)
    {
        return false;
    }

    if (sigmaX <= 0.0)
    {
        sigmaX = default_gaussian_sigma_for_ksize(kx);
    }
    if (sigmaY <= 0.0)
    {
        sigmaY = sigmaX;
    }
    if (sigmaX <= 0.0 || sigmaY <= 0.0)
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
    const std::size_t src_step = src_ref->step(0);
    const int row_stride = cols * channels;

    dst.create(std::vector<int>{rows, cols}, src_ref->type());
    const std::size_t dst_step = dst.step(0);

    const std::vector<float> kernel_x = build_gaussian_kernel_1d(kx, sigmaX);
    const std::vector<float> kernel_y = build_gaussian_kernel_1d(ky, sigmaY);
    const int rx = kx / 2;
    const int ry = ky / 2;
    const bool has_constant_border = border_type == BORDER_CONSTANT;

    std::vector<int> x_offsets(static_cast<std::size_t>(cols) * static_cast<std::size_t>(kx), -1);
    for (int x = 0; x < cols; ++x)
    {
        int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kx);
        for (int i = 0; i < kx; ++i)
        {
            const int sx = border_interpolate(x + i - rx, cols, border_type);
            x_ofs[i] = sx >= 0 ? sx * channels : -1;
        }
    }

    std::vector<int> y_offsets(static_cast<std::size_t>(rows) * static_cast<std::size_t>(ky), -1);
    for (int y = 0; y < rows; ++y)
    {
        int* y_ofs = y_offsets.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(ky);
        for (int i = 0; i < ky; ++i)
        {
            const int sy = border_interpolate(y + i - ry, rows, border_type);
            y_ofs[i] = sy >= 0 ? sy * row_stride : -1;
        }
    }

    std::vector<float> tmp(static_cast<std::size_t>(rows) * static_cast<std::size_t>(row_stride), 0.0f);

    const bool do_parallel_h = should_parallelize_filter_rows(rows, cols, channels, kx);
    parallel_for_index_if(do_parallel_h, rows, [&](int y) {
        const uchar* src_row = src_ref->data + static_cast<std::size_t>(y) * src_step;
        float* tmp_row = tmp.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(row_stride);

        if (channels == 1)
        {
            for (int x = 0; x < cols; ++x)
            {
                const int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kx);
                float acc0 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        acc0 += kernel_x[static_cast<std::size_t>(i)] * static_cast<float>(src_row[sx]);
                    }
                }
                else
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        acc0 += kernel_x[static_cast<std::size_t>(i)] * static_cast<float>(src_row[sx]);
                    }
                }
                tmp_row[x] = acc0;
            }
            return;
        }

        if (channels == 3)
        {
            for (int x = 0; x < cols; ++x)
            {
                const int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kx);
                const int dx = x * 3;
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        const float w = kernel_x[static_cast<std::size_t>(i)];
                        const uchar* px = src_row + sx;
                        acc0 += w * static_cast<float>(px[0]);
                        acc1 += w * static_cast<float>(px[1]);
                        acc2 += w * static_cast<float>(px[2]);
                    }
                }
                else
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        const float w = kernel_x[static_cast<std::size_t>(i)];
                        const uchar* px = src_row + sx;
                        acc0 += w * static_cast<float>(px[0]);
                        acc1 += w * static_cast<float>(px[1]);
                        acc2 += w * static_cast<float>(px[2]);
                    }
                }
                tmp_row[dx + 0] = acc0;
                tmp_row[dx + 1] = acc1;
                tmp_row[dx + 2] = acc2;
            }
            return;
        }

        if (channels == 4)
        {
            for (int x = 0; x < cols; ++x)
            {
                const int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kx);
                const int dx = x * 4;
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        const float w = kernel_x[static_cast<std::size_t>(i)];
                        const uchar* px = src_row + sx;
                        acc0 += w * static_cast<float>(px[0]);
                        acc1 += w * static_cast<float>(px[1]);
                        acc2 += w * static_cast<float>(px[2]);
                        acc3 += w * static_cast<float>(px[3]);
                    }
                }
                else
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        const float w = kernel_x[static_cast<std::size_t>(i)];
                        const uchar* px = src_row + sx;
                        acc0 += w * static_cast<float>(px[0]);
                        acc1 += w * static_cast<float>(px[1]);
                        acc2 += w * static_cast<float>(px[2]);
                        acc3 += w * static_cast<float>(px[3]);
                    }
                }
                tmp_row[dx + 0] = acc0;
                tmp_row[dx + 1] = acc1;
                tmp_row[dx + 2] = acc2;
                tmp_row[dx + 3] = acc3;
            }
        }
    });

    const bool do_parallel_v = should_parallelize_filter_rows(rows, cols, channels, ky);
    parallel_for_index_if(do_parallel_v, rows, [&](int y) {
        uchar* dst_row = dst.data + static_cast<std::size_t>(y) * dst_step;
        const int* y_ofs = y_offsets.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(ky);

        if (channels == 1)
        {
            for (int x = 0; x < cols; ++x)
            {
                float acc0 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        if (sy < 0)
                        {
                            continue;
                        }
                        acc0 += kernel_y[static_cast<std::size_t>(i)] *
                                tmp[static_cast<std::size_t>(sy + x)];
                    }
                }
                else
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        acc0 += kernel_y[static_cast<std::size_t>(i)] *
                                tmp[static_cast<std::size_t>(sy + x)];
                    }
                }
                dst_row[x] = saturate_cast<uchar>(acc0);
            }
            return;
        }

        if (channels == 3)
        {
            for (int x = 0; x < cols; ++x)
            {
                const int dx = x * 3;
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        if (sy < 0)
                        {
                            continue;
                        }
                        const float w = kernel_y[static_cast<std::size_t>(i)];
                        const float* px = tmp.data() + static_cast<std::size_t>(sy + dx);
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                    }
                }
                else
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        const float w = kernel_y[static_cast<std::size_t>(i)];
                        const float* px = tmp.data() + static_cast<std::size_t>(sy + dx);
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                    }
                }
                dst_row[dx + 0] = saturate_cast<uchar>(acc0);
                dst_row[dx + 1] = saturate_cast<uchar>(acc1);
                dst_row[dx + 2] = saturate_cast<uchar>(acc2);
            }
            return;
        }

        if (channels == 4)
        {
            for (int x = 0; x < cols; ++x)
            {
                const int dx = x * 4;
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        if (sy < 0)
                        {
                            continue;
                        }
                        const float w = kernel_y[static_cast<std::size_t>(i)];
                        const float* px = tmp.data() + static_cast<std::size_t>(sy + dx);
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                        acc3 += w * px[3];
                    }
                }
                else
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        const float w = kernel_y[static_cast<std::size_t>(i)];
                        const float* px = tmp.data() + static_cast<std::size_t>(sy + dx);
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                        acc3 += w * px[3];
                    }
                }
                dst_row[dx + 0] = saturate_cast<uchar>(acc0);
                dst_row[dx + 1] = saturate_cast<uchar>(acc1);
                dst_row[dx + 2] = saturate_cast<uchar>(acc2);
                dst_row[dx + 3] = saturate_cast<uchar>(acc3);
            }
        }
    });

    return true;
}

bool try_gaussian_blur_fastpath_f32(const Mat& src, Mat& dst, Size ksize, double sigmaX, double sigmaY, int borderType)
{
    if (src.empty() || src.dims != 2 || src.depth() != CV_32F)
    {
        return false;
    }

    if (!is_u8_fastpath_channels(src.channels()))
    {
        return false;
    }

    int kx = ksize.width;
    int ky = ksize.height;

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

    if (kx <= 0 || ky <= 0 || (kx & 1) == 0 || (ky & 1) == 0)
    {
        return false;
    }

    if (sigmaX <= 0.0)
    {
        sigmaX = default_gaussian_sigma_for_ksize(kx);
    }
    if (sigmaY <= 0.0)
    {
        sigmaY = sigmaX;
    }
    if (sigmaX <= 0.0 || sigmaY <= 0.0)
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
    const std::size_t src_step = src_ref->step(0);
    const int row_stride = cols * channels;

    dst.create(std::vector<int>{rows, cols}, src_ref->type());
    const std::size_t dst_step = dst.step(0);

    const std::vector<float> kernel_x = build_gaussian_kernel_1d(kx, sigmaX);
    const std::vector<float> kernel_y = build_gaussian_kernel_1d(ky, sigmaY);
    const int rx = kx / 2;
    const int ry = ky / 2;
    const bool has_constant_border = border_type == BORDER_CONSTANT;

    std::vector<int> x_offsets(static_cast<std::size_t>(cols) * static_cast<std::size_t>(kx), -1);
    for (int x = 0; x < cols; ++x)
    {
        int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kx);
        for (int i = 0; i < kx; ++i)
        {
            const int sx = border_interpolate(x + i - rx, cols, border_type);
            x_ofs[i] = sx >= 0 ? sx * channels : -1;
        }
    }

    std::vector<int> y_offsets(static_cast<std::size_t>(rows) * static_cast<std::size_t>(ky), -1);
    for (int y = 0; y < rows; ++y)
    {
        int* y_ofs = y_offsets.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(ky);
        for (int i = 0; i < ky; ++i)
        {
            const int sy = border_interpolate(y + i - ry, rows, border_type);
            y_ofs[i] = sy >= 0 ? sy * row_stride : -1;
        }
    }

    std::vector<float> tmp(static_cast<std::size_t>(rows) * static_cast<std::size_t>(row_stride), 0.0f);

    const bool do_parallel_h = should_parallelize_filter_rows(rows, cols, channels, kx);
    parallel_for_index_if(do_parallel_h, rows, [&](int y) {
        const float* src_row = reinterpret_cast<const float*>(src_ref->data + static_cast<std::size_t>(y) * src_step);
        float* tmp_row = tmp.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(row_stride);

        if (channels == 1)
        {
            for (int x = 0; x < cols; ++x)
            {
                const int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kx);
                float acc0 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        acc0 += kernel_x[static_cast<std::size_t>(i)] * src_row[sx];
                    }
                }
                else
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        acc0 += kernel_x[static_cast<std::size_t>(i)] * src_row[sx];
                    }
                }
                tmp_row[x] = acc0;
            }
            return;
        }

        if (channels == 3)
        {
            for (int x = 0; x < cols; ++x)
            {
                const int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kx);
                const int dx = x * 3;
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        const float w = kernel_x[static_cast<std::size_t>(i)];
                        const float* px = src_row + sx;
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                    }
                }
                else
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        const float w = kernel_x[static_cast<std::size_t>(i)];
                        const float* px = src_row + sx;
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                    }
                }
                tmp_row[dx + 0] = acc0;
                tmp_row[dx + 1] = acc1;
                tmp_row[dx + 2] = acc2;
            }
            return;
        }

        if (channels == 4)
        {
            for (int x = 0; x < cols; ++x)
            {
                const int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kx);
                const int dx = x * 4;
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        const float w = kernel_x[static_cast<std::size_t>(i)];
                        const float* px = src_row + sx;
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                        acc3 += w * px[3];
                    }
                }
                else
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        const float w = kernel_x[static_cast<std::size_t>(i)];
                        const float* px = src_row + sx;
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                        acc3 += w * px[3];
                    }
                }
                tmp_row[dx + 0] = acc0;
                tmp_row[dx + 1] = acc1;
                tmp_row[dx + 2] = acc2;
                tmp_row[dx + 3] = acc3;
            }
        }
    });

    const bool do_parallel_v = should_parallelize_filter_rows(rows, cols, channels, ky);
    parallel_for_index_if(do_parallel_v, rows, [&](int y) {
        float* dst_row = reinterpret_cast<float*>(dst.data + static_cast<std::size_t>(y) * dst_step);
        const int* y_ofs = y_offsets.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(ky);

        if (channels == 1)
        {
            for (int x = 0; x < cols; ++x)
            {
                float acc0 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        if (sy < 0)
                        {
                            continue;
                        }
                        acc0 += kernel_y[static_cast<std::size_t>(i)] *
                                tmp[static_cast<std::size_t>(sy + x)];
                    }
                }
                else
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        acc0 += kernel_y[static_cast<std::size_t>(i)] *
                                tmp[static_cast<std::size_t>(sy + x)];
                    }
                }
                dst_row[x] = acc0;
            }
            return;
        }

        if (channels == 3)
        {
            for (int x = 0; x < cols; ++x)
            {
                const int dx = x * 3;
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        if (sy < 0)
                        {
                            continue;
                        }
                        const float w = kernel_y[static_cast<std::size_t>(i)];
                        const float* px = tmp.data() + static_cast<std::size_t>(sy + dx);
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                    }
                }
                else
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        const float w = kernel_y[static_cast<std::size_t>(i)];
                        const float* px = tmp.data() + static_cast<std::size_t>(sy + dx);
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                    }
                }
                dst_row[dx + 0] = acc0;
                dst_row[dx + 1] = acc1;
                dst_row[dx + 2] = acc2;
            }
            return;
        }

        if (channels == 4)
        {
            for (int x = 0; x < cols; ++x)
            {
                const int dx = x * 4;
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        if (sy < 0)
                        {
                            continue;
                        }
                        const float w = kernel_y[static_cast<std::size_t>(i)];
                        const float* px = tmp.data() + static_cast<std::size_t>(sy + dx);
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                        acc3 += w * px[3];
                    }
                }
                else
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        const float w = kernel_y[static_cast<std::size_t>(i)];
                        const float* px = tmp.data() + static_cast<std::size_t>(sy + dx);
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                        acc3 += w * px[3];
                    }
                }
                dst_row[dx + 0] = acc0;
                dst_row[dx + 1] = acc1;
                dst_row[dx + 2] = acc2;
                dst_row[dx + 3] = acc3;
            }
        }
    });

    return true;
}


} // namespace

const char* last_gaussianblur_dispatch_path()
{
    return g_last_gaussianblur_dispatch_path;
}

void gaussianBlur_backend_impl(const Mat& src, Mat& dst, Size ksize, double sigmaX, double sigmaY, int borderType)
{
    set_last_gaussianblur_dispatch_path("fallback");

    if (try_gaussian_blur_fastpath_u8(src, dst, ksize, sigmaX, sigmaY, borderType))
    {
        int kx = 0;
        int ky = 0;
        if (resolve_gaussian_kernel_size(ksize, sigmaX, sigmaY, kx, ky) && kx == 3 && ky == 3)
        {
            set_last_gaussianblur_dispatch_path("gauss3x3");
        }
        else
        {
            set_last_gaussianblur_dispatch_path("gauss_separable");
        }
        return;
    }

    if (try_gaussian_blur_fastpath_f32(src, dst, ksize, sigmaX, sigmaY, borderType))
    {
        int kx = 0;
        int ky = 0;
        if (resolve_gaussian_kernel_size(ksize, sigmaX, sigmaY, kx, ky) && kx == 3 && ky == 3)
        {
            set_last_gaussianblur_dispatch_path("gauss3x3");
        }
        else
        {
            set_last_gaussianblur_dispatch_path("gauss_separable");
        }
        return;
    }

    gaussian_blur_fallback(src, dst, ksize, sigmaX, sigmaY, borderType);
}

} // namespace detail
} // namespace cvh
