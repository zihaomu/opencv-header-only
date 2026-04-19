#include "fastpath_common.h"
#include "cvtcolor_internal.h"

namespace cvh
{
namespace detail
{

namespace
{
inline uchar cvtcolor_yuv420p_plane_byte_u8(const uchar* src_data,
                                            std::size_t src_step,
                                            int rows,
                                            int cols,
                                            int plane_offset,
                                            int plane_index)
{
    const int logical_offset = plane_offset + plane_index;
    return *(src_data +
             static_cast<std::size_t>(rows + logical_offset / cols) * src_step +
             static_cast<std::size_t>(logical_offset % cols));
}

void cvtcolor_yuv420sp_to_3ch_u8(const uchar* src_data,
                                 std::size_t src_step,
                                 uchar* dst_data,
                                 std::size_t dst_step,
                                 int rows,
                                 int cols,
                                 bool nv21_layout,
                                 bool rgb_order)
{
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 1);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* y_row = src_data + static_cast<std::size_t>(y) * src_step;
        const uchar* uv_row = src_data + static_cast<std::size_t>(rows + y / 2) * src_step;
        uchar* dst_row = dst_data + static_cast<std::size_t>(y) * dst_step;

        for (int x = 0; x < cols; x += 2)
        {
            const int first = static_cast<int>(uv_row[x + 0]);
            const int second = static_cast<int>(uv_row[x + 1]);
            const int uu = nv21_layout ? second : first;
            const int vv = nv21_layout ? first : second;

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

void cvtcolor_3ch_to_yuv420sp_u8(const uchar* src_data,
                                 std::size_t src_step,
                                 uchar* dst_data,
                                 std::size_t dst_step,
                                 int rows,
                                 int cols,
                                 bool rgb_order,
                                 bool nv21_layout)
{
    CV_Assert((rows % 2) == 0 && "cvtColor(BGR/RGB2YUV420sp): source height must be even");
    CV_Assert((cols % 2) == 0 && "cvtColor(BGR/RGB2YUV420sp): source width must be even");

    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 3);
    parallel_for_index_if_step(do_parallel, 0, rows, 2, [&](int y) {
        const uchar* src_row0 = src_data + static_cast<std::size_t>(y + 0) * src_step;
        const uchar* src_row1 = src_data + static_cast<std::size_t>(y + 1) * src_step;
        uchar* dst_y_row0 = dst_data + static_cast<std::size_t>(y + 0) * dst_step;
        uchar* dst_y_row1 = dst_data + static_cast<std::size_t>(y + 1) * dst_step;
        uchar* dst_uv_row = dst_data + static_cast<std::size_t>(rows + y / 2) * dst_step;

        for (int x = 0; x < cols; x += 2)
        {
            int sum_b = 0;
            int sum_g = 0;
            int sum_r = 0;

            for (int dy = 0; dy < 2; ++dy)
            {
                const uchar* src_row = (dy == 0) ? src_row0 : src_row1;
                uchar* dst_y_row = (dy == 0) ? dst_y_row0 : dst_y_row1;

                for (int dx = 0; dx < 2; ++dx)
                {
                    const int sx = (x + dx) * 3;
                    const int bb = static_cast<int>(src_row[sx + (rgb_order ? 2 : 0)]);
                    const int gg = static_cast<int>(src_row[sx + 1]);
                    const int rr = static_cast<int>(src_row[sx + (rgb_order ? 0 : 2)]);
                    const uchar yy = cvtcolor_color3_to_yuv_limited_u8(bb, gg, rr, 0);

                    dst_y_row[x + dx] = yy;
                    sum_b += bb;
                    sum_g += gg;
                    sum_r += rr;
                }
            }

            const int avg_b = (sum_b + 2) >> 2;
            const int avg_g = (sum_g + 2) >> 2;
            const int avg_r = (sum_r + 2) >> 2;
            const uchar uu = cvtcolor_color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 1);
            const uchar vv = cvtcolor_color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 2);

            dst_uv_row[x + 0] = nv21_layout ? vv : uu;
            dst_uv_row[x + 1] = nv21_layout ? uu : vv;
        }
    });
}

void cvtcolor_3ch_to_yuv420p_u8(const uchar* src_data,
                                std::size_t src_step,
                                uchar* dst_data,
                                std::size_t dst_step,
                                int rows,
                                int cols,
                                bool rgb_order,
                                bool yv12_layout)
{
    CV_Assert((rows % 2) == 0 && "cvtColor(BGR/RGB2YUV420p): source height must be even");
    CV_Assert((cols % 2) == 0 && "cvtColor(BGR/RGB2YUV420p): source width must be even");

    const int uv_size = rows * cols / 4;
    const int u_plane_offset = yv12_layout ? uv_size : 0;
    const int v_plane_offset = yv12_layout ? 0 : uv_size;
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 3);
    parallel_for_index_if_step(do_parallel, 0, rows, 2, [&](int y) {
        const uchar* src_row0 = src_data + static_cast<std::size_t>(y + 0) * src_step;
        const uchar* src_row1 = src_data + static_cast<std::size_t>(y + 1) * src_step;
        uchar* dst_y_row0 = dst_data + static_cast<std::size_t>(y + 0) * dst_step;
        uchar* dst_y_row1 = dst_data + static_cast<std::size_t>(y + 1) * dst_step;

        for (int x = 0; x < cols; x += 2)
        {
            int sum_b = 0;
            int sum_g = 0;
            int sum_r = 0;

            for (int dy = 0; dy < 2; ++dy)
            {
                const uchar* src_row = (dy == 0) ? src_row0 : src_row1;
                uchar* dst_y_row = (dy == 0) ? dst_y_row0 : dst_y_row1;

                for (int dx = 0; dx < 2; ++dx)
                {
                    const int sx = (x + dx) * 3;
                    const int bb = static_cast<int>(src_row[sx + (rgb_order ? 2 : 0)]);
                    const int gg = static_cast<int>(src_row[sx + 1]);
                    const int rr = static_cast<int>(src_row[sx + (rgb_order ? 0 : 2)]);
                    const uchar yy = cvtcolor_color3_to_yuv_limited_u8(bb, gg, rr, 0);

                    dst_y_row[x + dx] = yy;
                    sum_b += bb;
                    sum_g += gg;
                    sum_r += rr;
                }
            }

            const int avg_b = (sum_b + 2) >> 2;
            const int avg_g = (sum_g + 2) >> 2;
            const int avg_r = (sum_r + 2) >> 2;
            const uchar uu = cvtcolor_color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 1);
            const uchar vv = cvtcolor_color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 2);
            const int chroma_index = (y / 2) * (cols / 2) + (x / 2);

            *(dst_data +
              static_cast<std::size_t>(rows + (u_plane_offset + chroma_index) / cols) * dst_step +
              static_cast<std::size_t>((u_plane_offset + chroma_index) % cols)) = uu;
            *(dst_data +
              static_cast<std::size_t>(rows + (v_plane_offset + chroma_index) / cols) * dst_step +
              static_cast<std::size_t>((v_plane_offset + chroma_index) % cols)) = vv;
        }
    });
}

void cvtcolor_yuv420p_to_3ch_u8(const uchar* src_data,
                                std::size_t src_step,
                                uchar* dst_data,
                                std::size_t dst_step,
                                int rows,
                                int cols,
                                bool yv12_layout,
                                bool rgb_order)
{
    const int uv_size = rows * cols / 4;
    const int u_plane_offset = yv12_layout ? uv_size : 0;
    const int v_plane_offset = yv12_layout ? 0 : uv_size;
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 1);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* y_row = src_data + static_cast<std::size_t>(y) * src_step;
        uchar* dst_row = dst_data + static_cast<std::size_t>(y) * dst_step;

        for (int x = 0; x < cols; ++x)
        {
            const int yy = static_cast<int>(y_row[x]);
            const int chroma_index = (y / 2) * (cols / 2) + (x / 2);
            const int uu = static_cast<int>(cvtcolor_yuv420p_plane_byte_u8(src_data, src_step, rows, cols, u_plane_offset, chroma_index));
            const int vv = static_cast<int>(cvtcolor_yuv420p_plane_byte_u8(src_data, src_step, rows, cols, v_plane_offset, chroma_index));
            const uchar b = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 0);
            const uchar g = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 1);
            const uchar r = cvtcolor_yuv420sp_channel_u8(yy, uu, vv, 2);

            const int dx = x * 3;
            dst_row[dx + (rgb_order ? 0 : 2)] = r;
            dst_row[dx + 1] = g;
            dst_row[dx + (rgb_order ? 2 : 0)] = b;
        }
    });
}


} // namespace

bool try_cvtcolor_fastpath_u8_yuv420(const Mat& src, Mat& dst, int code)
{
    if (src.empty() || src.dims != 2 || src.depth() != CV_8U)
    {
        return false;
    }

    const int rows = src.size[0];
    const int cols = src.size[1];
    const size_t src_step = src.step(0);

    if (code == COLOR_BGR2YUV_NV12 ||
        code == COLOR_RGB2YUV_NV12 ||
        code == COLOR_BGR2YUV_NV21 ||
        code == COLOR_RGB2YUV_NV21)
    {
        if (src.channels() != 3)
        {
            return false;
        }

        dst.create(std::vector<int>{rows * 3 / 2, cols}, CV_8UC1);
        cvtcolor_3ch_to_yuv420sp_u8(
            src.data,
            src_step,
            dst.data,
            dst.step(0),
            rows,
            cols,
            code == COLOR_RGB2YUV_NV12 || code == COLOR_RGB2YUV_NV21,
            code == COLOR_BGR2YUV_NV21 || code == COLOR_RGB2YUV_NV21);
        return true;
    }

    if (code == COLOR_BGR2YUV_I420 ||
        code == COLOR_RGB2YUV_I420 ||
        code == COLOR_BGR2YUV_YV12 ||
        code == COLOR_RGB2YUV_YV12)
    {
        if (src.channels() != 3)
        {
            return false;
        }

        dst.create(std::vector<int>{rows * 3 / 2, cols}, CV_8UC1);
        cvtcolor_3ch_to_yuv420p_u8(
            src.data,
            src_step,
            dst.data,
            dst.step(0),
            rows,
            cols,
            code == COLOR_RGB2YUV_I420 || code == COLOR_RGB2YUV_YV12,
            code == COLOR_BGR2YUV_YV12 || code == COLOR_RGB2YUV_YV12);
        return true;
    }

    if (code == COLOR_YUV2BGR_NV12 ||
        code == COLOR_YUV2RGB_NV12 ||
        code == COLOR_YUV2BGR_NV21 ||
        code == COLOR_YUV2RGB_NV21)
    {
        if (src.channels() != 1)
        {
            return false;
        }

        const int y_rows = cvtcolor_validate_yuv420sp_layout_u8(src);
        dst.create(std::vector<int>{y_rows, cols}, CV_8UC3);
        cvtcolor_yuv420sp_to_3ch_u8(
            src.data,
            src_step,
            dst.data,
            dst.step(0),
            y_rows,
            cols,
            code == COLOR_YUV2BGR_NV21 || code == COLOR_YUV2RGB_NV21,
            code == COLOR_YUV2RGB_NV12 || code == COLOR_YUV2RGB_NV21);
        return true;
    }

    if (code == COLOR_YUV2BGR_I420 ||
        code == COLOR_YUV2RGB_I420 ||
        code == COLOR_YUV2BGR_YV12 ||
        code == COLOR_YUV2RGB_YV12)
    {
        if (src.channels() != 1)
        {
            return false;
        }

        const int y_rows = cvtcolor_validate_yuv420sp_layout_u8(src);
        dst.create(std::vector<int>{y_rows, cols}, CV_8UC3);
        cvtcolor_yuv420p_to_3ch_u8(
            src.data,
            src_step,
            dst.data,
            dst.step(0),
            y_rows,
            cols,
            code == COLOR_YUV2BGR_YV12 || code == COLOR_YUV2RGB_YV12,
            code == COLOR_YUV2RGB_I420 || code == COLOR_YUV2RGB_YV12);
        return true;
    }

    return false;
}

} // namespace detail
} // namespace cvh
