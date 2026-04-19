#include "fastpath_common.h"
#include "cvtcolor_internal.h"

namespace cvh
{
namespace detail
{

namespace
{
void cvtcolor_yuv422sp_to_3ch_u8(const uchar* src_data,
                                 std::size_t src_step,
                                 uchar* dst_data,
                                 std::size_t dst_step,
                                 int rows,
                                 int cols,
                                 bool nv61_layout,
                                 bool rgb_order)
{
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 1);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* y_row = src_data + static_cast<std::size_t>(y) * src_step;
        const uchar* uv_row = src_data + static_cast<std::size_t>(rows + y) * src_step;
        uchar* dst_row = dst_data + static_cast<std::size_t>(y) * dst_step;

        for (int x = 0; x < cols; x += 2)
        {
            const int first = static_cast<int>(uv_row[x + 0]);
            const int second = static_cast<int>(uv_row[x + 1]);
            const int uu = nv61_layout ? second : first;
            const int vv = nv61_layout ? first : second;

            for (int i = 0; i < 2; ++i)
            {
                const int dx = (x + i) * 3;
                const int yy = static_cast<int>(y_row[x + i]);
                const uchar b = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 0);
                const uchar g = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 1);
                const uchar r = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 2);

                dst_row[dx + (rgb_order ? 0 : 2)] = r;
                dst_row[dx + 1] = g;
                dst_row[dx + (rgb_order ? 2 : 0)] = b;
            }
        }
    });
}

void cvtcolor_3ch_to_yuv422sp_u8(const uchar* src_data,
                                 std::size_t src_step,
                                 uchar* dst_data,
                                 std::size_t dst_step,
                                 int rows,
                                 int cols,
                                 bool rgb_order,
                                 bool nv61_layout)
{
    CV_Assert((cols % 2) == 0 && "cvtColor(BGR/RGB2YUV422sp): source width must be even");

    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 3);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* src_row = src_data + static_cast<std::size_t>(y) * src_step;
        uchar* dst_y_row = dst_data + static_cast<std::size_t>(y) * dst_step;
        uchar* dst_uv_row = dst_data + static_cast<std::size_t>(rows + y) * dst_step;

        for (int x = 0; x < cols; x += 2)
        {
            int sum_b = 0;
            int sum_g = 0;
            int sum_r = 0;

            for (int i = 0; i < 2; ++i)
            {
                const int sx = (x + i) * 3;
                const int bb = static_cast<int>(src_row[sx + (rgb_order ? 2 : 0)]);
                const int gg = static_cast<int>(src_row[sx + 1]);
                const int rr = static_cast<int>(src_row[sx + (rgb_order ? 0 : 2)]);
                const uchar yy = cvtcolor_color3_to_yuv_limited_u8(bb, gg, rr, 0);

                dst_y_row[x + i] = yy;
                sum_b += bb;
                sum_g += gg;
                sum_r += rr;
            }

            const int avg_b = (sum_b + 1) >> 1;
            const int avg_g = (sum_g + 1) >> 1;
            const int avg_r = (sum_r + 1) >> 1;
            const uchar uu = cvtcolor_color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 1);
            const uchar vv = cvtcolor_color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 2);

            dst_uv_row[x + 0] = nv61_layout ? vv : uu;
            dst_uv_row[x + 1] = nv61_layout ? uu : vv;
        }
    });
}

void cvtcolor_3ch_to_yuv422packed_u8(const uchar* src_data,
                                     std::size_t src_step,
                                     uchar* dst_data,
                                     std::size_t dst_step,
                                     int rows,
                                     int cols,
                                     bool rgb_order,
                                     bool uyvy_layout)
{
    CV_Assert((cols % 2) == 0 && "cvtColor(BGR/RGB2YUV422packed): source width must be even");

    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 3);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* src_row = src_data + static_cast<std::size_t>(y) * src_step;
        uchar* dst_row = dst_data + static_cast<std::size_t>(y) * dst_step;

        for (int x = 0; x < cols; x += 2)
        {
            int sum_b = 0;
            int sum_g = 0;
            int sum_r = 0;
            uchar yy[2] = {0, 0};

            for (int i = 0; i < 2; ++i)
            {
                const int sx = (x + i) * 3;
                const int bb = static_cast<int>(src_row[sx + (rgb_order ? 2 : 0)]);
                const int gg = static_cast<int>(src_row[sx + 1]);
                const int rr = static_cast<int>(src_row[sx + (rgb_order ? 0 : 2)]);

                yy[i] = cvtcolor_color3_to_yuv_limited_u8(bb, gg, rr, 0);
                sum_b += bb;
                sum_g += gg;
                sum_r += rr;
            }

            const int avg_b = (sum_b + 1) >> 1;
            const int avg_g = (sum_g + 1) >> 1;
            const int avg_r = (sum_r + 1) >> 1;
            const uchar uu = cvtcolor_color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 1);
            const uchar vv = cvtcolor_color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 2);
            const int base = x * 2;

            if (uyvy_layout)
            {
                dst_row[base + 0] = uu;
                dst_row[base + 1] = yy[0];
                dst_row[base + 2] = vv;
                dst_row[base + 3] = yy[1];
            }
            else
            {
                dst_row[base + 0] = yy[0];
                dst_row[base + 1] = uu;
                dst_row[base + 2] = yy[1];
                dst_row[base + 3] = vv;
            }
        }
    });
}

inline uchar cvtcolor_yuv444sp_plane_byte_u8(const uchar* src_data,
                                             std::size_t src_step,
                                             int rows,
                                             int cols,
                                             int plane_index)
{
    return *(src_data +
             static_cast<std::size_t>(rows + plane_index / cols) * src_step +
             static_cast<std::size_t>(plane_index % cols));
}

void cvtcolor_yuv422packed_to_3ch_u8(const uchar* src_data,
                                     std::size_t src_step,
                                     uchar* dst_data,
                                     std::size_t dst_step,
                                     int rows,
                                     int cols,
                                     bool uyvy_layout,
                                     bool rgb_order)
{
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 2);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* src_row = src_data + static_cast<std::size_t>(y) * src_step;
        uchar* dst_row = dst_data + static_cast<std::size_t>(y) * dst_step;

        for (int x = 0; x < cols; x += 2)
        {
            const int base = x * 2;
            const int first0 = static_cast<int>(src_row[base + 0]);
            const int first1 = static_cast<int>(src_row[base + 1]);
            const int second0 = static_cast<int>(src_row[base + 2]);
            const int second1 = static_cast<int>(src_row[base + 3]);

            const int yy0 = uyvy_layout ? first1 : first0;
            const int uu = uyvy_layout ? first0 : first1;
            const int yy1 = uyvy_layout ? second1 : second0;
            const int vv = uyvy_layout ? second0 : second1;

            for (int i = 0; i < 2; ++i)
            {
                const int dx = (x + i) * 3;
                const int yy = (i == 0) ? yy0 : yy1;
                const uchar b = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 0);
                const uchar g = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 1);
                const uchar r = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 2);

                dst_row[dx + (rgb_order ? 0 : 2)] = r;
                dst_row[dx + 1] = g;
                dst_row[dx + (rgb_order ? 2 : 0)] = b;
            }
        }
    });
}


} // namespace

bool try_cvtcolor_fastpath_u8_yuv422(const Mat& src, Mat& dst, int code)
{
    if (src.empty() || src.dims != 2 || src.depth() != CV_8U)
    {
        return false;
    }

    const int rows = src.size[0];
    const int cols = src.size[1];
    const size_t src_step = src.step(0);

    if (code == COLOR_BGR2YUV_NV16 ||
        code == COLOR_RGB2YUV_NV16 ||
        code == COLOR_BGR2YUV_NV61 ||
        code == COLOR_RGB2YUV_NV61)
    {
        if (src.channels() != 3)
        {
            return false;
        }

        dst.create(std::vector<int>{rows * 2, cols}, CV_8UC1);
        cvtcolor_3ch_to_yuv422sp_u8(
            src.data,
            src_step,
            dst.data,
            dst.step(0),
            rows,
            cols,
            code == COLOR_RGB2YUV_NV16 || code == COLOR_RGB2YUV_NV61,
            code == COLOR_BGR2YUV_NV61 || code == COLOR_RGB2YUV_NV61);
        return true;
    }

    if (code == COLOR_BGR2YUV_YUY2 ||
        code == COLOR_RGB2YUV_YUY2 ||
        code == COLOR_BGR2YUV_UYVY ||
        code == COLOR_RGB2YUV_UYVY)
    {
        if (src.channels() != 3)
        {
            return false;
        }

        dst.create(std::vector<int>{rows, cols}, CV_8UC2);
        cvtcolor_3ch_to_yuv422packed_u8(
            src.data,
            src_step,
            dst.data,
            dst.step(0),
            rows,
            cols,
            code == COLOR_RGB2YUV_YUY2 || code == COLOR_RGB2YUV_UYVY,
            code == COLOR_BGR2YUV_UYVY || code == COLOR_RGB2YUV_UYVY);
        return true;
    }

    if (code == COLOR_YUV2BGR_NV16 ||
        code == COLOR_YUV2RGB_NV16 ||
        code == COLOR_YUV2BGR_NV61 ||
        code == COLOR_YUV2RGB_NV61)
    {
        if (src.channels() != 1)
        {
            return false;
        }

        const int y_rows = cvtcolor_validate_yuv422sp_layout_u8(src);
        dst.create(std::vector<int>{y_rows, cols}, CV_8UC3);
        cvtcolor_yuv422sp_to_3ch_u8(
            src.data,
            src_step,
            dst.data,
            dst.step(0),
            y_rows,
            cols,
            code == COLOR_YUV2BGR_NV61 || code == COLOR_YUV2RGB_NV61,
            code == COLOR_YUV2RGB_NV16 || code == COLOR_YUV2RGB_NV61);
        return true;
    }

    if (code == COLOR_YUV2BGR_YUY2 ||
        code == COLOR_YUV2RGB_YUY2 ||
        code == COLOR_YUV2BGR_UYVY ||
        code == COLOR_YUV2RGB_UYVY)
    {
        if (src.channels() != 2)
        {
            return false;
        }

        cvtcolor_validate_yuv422packed_layout_u8(src);
        dst.create(std::vector<int>{rows, cols}, CV_8UC3);
        cvtcolor_yuv422packed_to_3ch_u8(
            src.data,
            src_step,
            dst.data,
            dst.step(0),
            rows,
            cols,
            code == COLOR_YUV2BGR_UYVY || code == COLOR_YUV2RGB_UYVY,
            code == COLOR_YUV2RGB_YUY2 || code == COLOR_YUV2RGB_UYVY);
        return true;
    }

    return false;
}

} // namespace detail
} // namespace cvh
