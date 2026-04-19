#include "fastpath_common.h"
#include "cvtcolor_internal.h"

namespace cvh
{
namespace detail
{

namespace
{
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

void cvtcolor_yuv444sp_to_3ch_u8(const uchar* src_data,
                                 std::size_t src_step,
                                 uchar* dst_data,
                                 std::size_t dst_step,
                                 int rows,
                                 int cols,
                                 bool nv42_layout,
                                 bool rgb_order)
{
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 1);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* y_row = src_data + static_cast<std::size_t>(y) * src_step;
        uchar* dst_row = dst_data + static_cast<std::size_t>(y) * dst_step;

        for (int x = 0; x < cols; ++x)
        {
            const int yy = static_cast<int>(y_row[x]);
            const int base = y * (cols * 2) + x * 2;
            const int uu = static_cast<int>(cvtcolor_yuv444sp_plane_byte_u8(src_data, src_step, rows, cols, base + (nv42_layout ? 1 : 0)));
            const int vv = static_cast<int>(cvtcolor_yuv444sp_plane_byte_u8(src_data, src_step, rows, cols, base + (nv42_layout ? 0 : 1)));
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

void cvtcolor_3ch_to_yuv444sp_u8(const uchar* src_data,
                                 std::size_t src_step,
                                 uchar* dst_data,
                                 std::size_t dst_step,
                                 int rows,
                                 int cols,
                                 bool rgb_order,
                                 bool nv42_layout)
{
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 3);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* src_row = src_data + static_cast<std::size_t>(y) * src_step;
        uchar* dst_y_row = dst_data + static_cast<std::size_t>(y) * dst_step;

        for (int x = 0; x < cols; ++x)
        {
            const int sx = x * 3;
            const int bb = static_cast<int>(src_row[sx + (rgb_order ? 2 : 0)]);
            const int gg = static_cast<int>(src_row[sx + 1]);
            const int rr = static_cast<int>(src_row[sx + (rgb_order ? 0 : 2)]);
            const uchar yy = cvtcolor_color3_to_yuv_limited_u8(bb, gg, rr, 0);
            const uchar uu = cvtcolor_color3_to_yuv_limited_u8(bb, gg, rr, 1);
            const uchar vv = cvtcolor_color3_to_yuv_limited_u8(bb, gg, rr, 2);
            const int base = y * (cols * 2) + x * 2;

            dst_y_row[x] = yy;
            *(dst_data +
              static_cast<std::size_t>(rows + (base + 0) / cols) * dst_step +
              static_cast<std::size_t>((base + 0) % cols)) = nv42_layout ? vv : uu;
            *(dst_data +
              static_cast<std::size_t>(rows + (base + 1) / cols) * dst_step +
              static_cast<std::size_t>((base + 1) % cols)) = nv42_layout ? uu : vv;
        }
    });
}

void cvtcolor_3ch_to_yuv444p_u8(const uchar* src_data,
                                std::size_t src_step,
                                uchar* dst_data,
                                std::size_t dst_step,
                                int rows,
                                int cols,
                                bool rgb_order,
                                bool yv24_layout)
{
    const int plane_size = rows * cols;
    const int u_plane_offset = yv24_layout ? plane_size : 0;
    const int v_plane_offset = yv24_layout ? 0 : plane_size;
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 3);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* src_row = src_data + static_cast<std::size_t>(y) * src_step;
        uchar* dst_y_row = dst_data + static_cast<std::size_t>(y) * dst_step;

        for (int x = 0; x < cols; ++x)
        {
            const int sx = x * 3;
            const int bb = static_cast<int>(src_row[sx + (rgb_order ? 2 : 0)]);
            const int gg = static_cast<int>(src_row[sx + 1]);
            const int rr = static_cast<int>(src_row[sx + (rgb_order ? 0 : 2)]);
            const uchar yy = cvtcolor_color3_to_yuv_limited_u8(bb, gg, rr, 0);
            const uchar uu = cvtcolor_color3_to_yuv_limited_u8(bb, gg, rr, 1);
            const uchar vv = cvtcolor_color3_to_yuv_limited_u8(bb, gg, rr, 2);
            const int chroma_index = y * cols + x;

            dst_y_row[x] = yy;
            *(dst_data +
              static_cast<std::size_t>(rows + (u_plane_offset + chroma_index) / cols) * dst_step +
              static_cast<std::size_t>((u_plane_offset + chroma_index) % cols)) = uu;
            *(dst_data +
              static_cast<std::size_t>(rows + (v_plane_offset + chroma_index) / cols) * dst_step +
              static_cast<std::size_t>((v_plane_offset + chroma_index) % cols)) = vv;
        }
    });
}

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

inline uchar cvtcolor_yuv444p_plane_byte_u8(const uchar* src_data,
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

void cvtcolor_yuv444p_to_3ch_u8(const uchar* src_data,
                                std::size_t src_step,
                                uchar* dst_data,
                                std::size_t dst_step,
                                int rows,
                                int cols,
                                bool yv24_layout,
                                bool rgb_order)
{
    const int plane_size = rows * cols;
    const int u_plane_offset = yv24_layout ? plane_size : 0;
    const int v_plane_offset = yv24_layout ? 0 : plane_size;
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 1);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* y_row = src_data + static_cast<std::size_t>(y) * src_step;
        uchar* dst_row = dst_data + static_cast<std::size_t>(y) * dst_step;

        for (int x = 0; x < cols; ++x)
        {
            const int yy = static_cast<int>(y_row[x]);
            const int chroma_index = y * cols + x;
            const int uu = static_cast<int>(cvtcolor_yuv444p_plane_byte_u8(src_data, src_step, rows, cols, u_plane_offset, chroma_index));
            const int vv = static_cast<int>(cvtcolor_yuv444p_plane_byte_u8(src_data, src_step, rows, cols, v_plane_offset, chroma_index));
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

bool try_cvtcolor_fastpath_u8_yuv444(const Mat& src, Mat& dst, int code)
{
    if (src.empty() || src.dims != 2 || src.depth() != CV_8U)
    {
        return false;
    }

    const int rows = src.size[0];
    const int cols = src.size[1];
    const size_t src_step = src.step(0);

    if (code == COLOR_BGR2YUV_NV24 ||
        code == COLOR_RGB2YUV_NV24 ||
        code == COLOR_BGR2YUV_NV42 ||
        code == COLOR_RGB2YUV_NV42)
    {
        if (src.channels() != 3)
        {
            return false;
        }

        dst.create(std::vector<int>{rows * 3, cols}, CV_8UC1);
        cvtcolor_3ch_to_yuv444sp_u8(
            src.data,
            src_step,
            dst.data,
            dst.step(0),
            rows,
            cols,
            code == COLOR_RGB2YUV_NV24 || code == COLOR_RGB2YUV_NV42,
            code == COLOR_BGR2YUV_NV42 || code == COLOR_RGB2YUV_NV42);
        return true;
    }

    if (code == COLOR_BGR2YUV_I444 ||
        code == COLOR_RGB2YUV_I444 ||
        code == COLOR_BGR2YUV_YV24 ||
        code == COLOR_RGB2YUV_YV24)
    {
        if (src.channels() != 3)
        {
            return false;
        }

        dst.create(std::vector<int>{rows * 3, cols}, CV_8UC1);
        cvtcolor_3ch_to_yuv444p_u8(
            src.data,
            src_step,
            dst.data,
            dst.step(0),
            rows,
            cols,
            code == COLOR_RGB2YUV_I444 || code == COLOR_RGB2YUV_YV24,
            code == COLOR_BGR2YUV_YV24 || code == COLOR_RGB2YUV_YV24);
        return true;
    }

    if (code == COLOR_YUV2BGR_NV24 ||
        code == COLOR_YUV2RGB_NV24 ||
        code == COLOR_YUV2BGR_NV42 ||
        code == COLOR_YUV2RGB_NV42)
    {
        if (src.channels() != 1)
        {
            return false;
        }

        const int y_rows = cvtcolor_validate_yuv444sp_layout_u8(src);
        dst.create(std::vector<int>{y_rows, cols}, CV_8UC3);
        cvtcolor_yuv444sp_to_3ch_u8(
            src.data,
            src_step,
            dst.data,
            dst.step(0),
            y_rows,
            cols,
            code == COLOR_YUV2BGR_NV42 || code == COLOR_YUV2RGB_NV42,
            code == COLOR_YUV2RGB_NV24 || code == COLOR_YUV2RGB_NV42);
        return true;
    }

    if (code == COLOR_YUV2BGR_I444 ||
        code == COLOR_YUV2RGB_I444 ||
        code == COLOR_YUV2BGR_YV24 ||
        code == COLOR_YUV2RGB_YV24)
    {
        if (src.channels() != 1)
        {
            return false;
        }

        const int y_rows = cvtcolor_validate_yuv444p_layout_u8(src);
        dst.create(std::vector<int>{y_rows, cols}, CV_8UC3);
        cvtcolor_yuv444p_to_3ch_u8(
            src.data,
            src_step,
            dst.data,
            dst.step(0),
            y_rows,
            cols,
            code == COLOR_YUV2BGR_YV24 || code == COLOR_YUV2RGB_YV24,
            code == COLOR_YUV2RGB_I444 || code == COLOR_YUV2RGB_YV24);
        return true;
    }

    return false;
}

} // namespace detail
} // namespace cvh
