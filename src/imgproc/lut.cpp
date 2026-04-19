#include "fastpath_common.h"

namespace cvh
{
namespace detail
{

namespace
{
bool try_lut_fastpath_u8(const Mat& src, const Mat& lut, Mat& dst)
{
    if (src.empty() || src.dims != 2 || src.depth() != CV_8U)
    {
        return false;
    }

    if (lut.empty() || lut.total() != 256 || lut.depth() != CV_8U)
    {
        return false;
    }

    const int src_cn = src.channels();
    const int lut_cn = lut.channels();
    if (lut_cn != 1 && lut_cn != src_cn)
    {
        return false;
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
    if (rows <= 0 || cols <= 0 || src_cn <= 0)
    {
        return false;
    }

    std::vector<uchar> table;
    if (lut_cn == 1)
    {
        table.resize(256u);
        for (int i = 0; i < 256; ++i)
        {
            const uchar* base = lut_entry_base_ptr(*lut_ref, i);
            table[static_cast<std::size_t>(i)] = base[0];
        }
    }
    else
    {
        table.resize(256u * static_cast<std::size_t>(src_cn));
        for (int i = 0; i < 256; ++i)
        {
            const uchar* base = lut_entry_base_ptr(*lut_ref, i);
            std::memcpy(
                table.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(src_cn),
                base,
                static_cast<std::size_t>(src_cn));
        }
    }

    dst.create(std::vector<int>{rows, cols}, src_ref->type());
    const std::size_t src_step = src_ref->step(0);
    const std::size_t dst_step = dst.step(0);
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, src_cn);

    if (lut_cn == 1)
    {
        const uchar* map = table.data();
        parallel_for_index_if(do_parallel, rows, [&](int y) {
            const uchar* src_row = src_ref->data + static_cast<std::size_t>(y) * src_step;
            uchar* dst_row = dst.data + static_cast<std::size_t>(y) * dst_step;

            const int row_elems = cols * src_cn;
            int i = 0;
            for (; i + 7 < row_elems; i += 8)
            {
                dst_row[i + 0] = map[src_row[i + 0]];
                dst_row[i + 1] = map[src_row[i + 1]];
                dst_row[i + 2] = map[src_row[i + 2]];
                dst_row[i + 3] = map[src_row[i + 3]];
                dst_row[i + 4] = map[src_row[i + 4]];
                dst_row[i + 5] = map[src_row[i + 5]];
                dst_row[i + 6] = map[src_row[i + 6]];
                dst_row[i + 7] = map[src_row[i + 7]];
            }
            for (; i < row_elems; ++i)
            {
                dst_row[i] = map[src_row[i]];
            }
        });
        return true;
    }

    const uchar* map = table.data();
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* src_row = src_ref->data + static_cast<std::size_t>(y) * src_step;
        uchar* dst_row = dst.data + static_cast<std::size_t>(y) * dst_step;

        for (int x = 0; x < cols; ++x)
        {
            const uchar* src_px = src_row + static_cast<std::size_t>(x) * static_cast<std::size_t>(src_cn);
            uchar* dst_px = dst_row + static_cast<std::size_t>(x) * static_cast<std::size_t>(src_cn);

            for (int c = 0; c < src_cn; ++c)
            {
                dst_px[c] = map[static_cast<std::size_t>(src_px[c]) * static_cast<std::size_t>(src_cn) + static_cast<std::size_t>(c)];
            }
        }
    });

    return true;
}


} // namespace

void lut_backend_impl(const Mat& src, const Mat& lut, Mat& dst)
{
    if (try_lut_fastpath_u8(src, lut, dst))
    {
        return;
    }

    LUT_fallback(src, lut, dst);
}

} // namespace detail
} // namespace cvh
