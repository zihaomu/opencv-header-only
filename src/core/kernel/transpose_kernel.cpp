#include "transpose_kernel.h"
#include "cvh/core/parallel.h"
#include "cvh/core/detail/dispatch_control.h"
#include "cvh/core/detail/openmp_utils.h"
#include "cvh/core/system.h"

#include <algorithm>
#include <cstddef>
#include <cstring>

namespace cvh {
namespace cpu {

namespace {

template <class RowBlockFn>
void for_each_row_block(int rows, int cols, int tile, RowBlockFn&& fn)
{
    const int row_blocks = (rows + tile - 1) / tile;
    const bool do_parallel = should_parallelize_1d_loop(
        static_cast<size_t>(row_blocks),
        static_cast<size_t>(tile) * static_cast<size_t>(cols),
        1LL << 14,
        2);

    if (!do_parallel)
    {
        for (int block_idx = 0; block_idx < row_blocks; ++block_idx)
        {
            const int row0 = block_idx * tile;
            fn(row0);
        }
        return;
    }

    cvh::parallel_for_(
        cvh::Range(0, row_blocks),
        [&](const cvh::Range& range) {
            for (int block_idx = range.start; block_idx < range.end; ++block_idx)
            {
                const int row0 = block_idx * tile;
                fn(row0);
            }
        },
        static_cast<double>(row_blocks));
}

template<typename T>
void transpose2d_tiled(const unsigned char* src_raw, unsigned char* dst_raw, int rows, int cols)
{
    const T* src = reinterpret_cast<const T*>(src_raw);
    T* dst = reinterpret_cast<T*>(dst_raw);

    constexpr int TILE = 32;
    for_each_row_block(rows, cols, TILE, [&](int row0) {
        const int row1 = std::min(row0 + TILE, rows);
        for (int col0 = 0; col0 < cols; col0 += TILE)
        {
            const int col1 = std::min(col0 + TILE, cols);
            for (int row = row0; row < row1; ++row)
            {
                for (int col = col0; col < col1; ++col)
                {
                    dst[static_cast<size_t>(col) * rows + row] = src[static_cast<size_t>(row) * cols + col];
                }
            }
        }
    });
}


template<size_t Bytes>
struct FixedPixel
{
    unsigned char data[Bytes];
};

inline void transpose2d_memcpy_fallback(const unsigned char* src,
                                        unsigned char* dst,
                                        int rows,
                                        int cols,
                                        size_t elem_size)
{
    constexpr int TILE = 32;
    for_each_row_block(rows, cols, TILE, [&](int row0) {
        const int row1 = std::min(row0 + TILE, rows);
        for (int col0 = 0; col0 < cols; col0 += TILE)
        {
            const int col1 = std::min(col0 + TILE, cols);
            for (int row = row0; row < row1; ++row)
            {
                for (int col = col0; col < col1; ++col)
                {
                    std::memcpy(dst + (static_cast<size_t>(col) * rows + row) * elem_size,
                                src + (static_cast<size_t>(row) * cols + col) * elem_size,
                                elem_size);
                }
            }
        }
    });
}

}  // namespace

void transpose2d_kernel_blocked(const unsigned char* src,
                                unsigned char* dst,
                                int rows,
                                int cols,
                                size_t elem_size1,
                                int channels)
{
    if (rows <= 0 || cols <= 0 || elem_size1 == 0 || channels <= 0)
    {
        return;
    }

    const size_t elem_size = elem_size1 * static_cast<size_t>(channels);
    set_last_dispatch_tag(DispatchTag::Scalar);

    // Fixed-size pixel fallback avoids per-element memcpy call overhead for
    // common multi-channel layouts not representable as 1/2/4/8-byte lanes.
    switch (elem_size)
    {
        case 3:
            transpose2d_tiled<FixedPixel<3>>(src, dst, rows, cols);
            return;
        case 6:
            transpose2d_tiled<FixedPixel<6>>(src, dst, rows, cols);
            return;
        case 12:
            transpose2d_tiled<FixedPixel<12>>(src, dst, rows, cols);
            return;
        case 16:
            transpose2d_tiled<FixedPixel<16>>(src, dst, rows, cols);
            return;
        default:
            transpose2d_memcpy_fallback(src, dst, rows, cols, elem_size);
            return;
    }
}

}  // namespace cpu
}  // namespace cvh
