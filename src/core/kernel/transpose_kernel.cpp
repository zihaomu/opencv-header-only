#include "transpose_kernel.h"
#include "cvh/core/parallel.h"
#include "cvh/core/detail/dispatch_control.h"
#include "cvh/core/detail/openmp_utils.h"
#include "cvh/core/system.h"
#include "xsimd/xsimd.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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


template<typename T>
void transpose2d_xsimd(const unsigned char* src_raw, unsigned char* dst_raw, int rows, int cols)
{
    using batch_type = xsimd::batch<T>;
    constexpr int N = batch_type::size;
    constexpr size_t lane_bytes = sizeof(T) * static_cast<size_t>(N);
    constexpr int TILE = 64; // Tile should be a multiple of N (which is usually 4, 8, 16)

    for_each_row_block(rows, cols, TILE, [&](int row0) {
        const int row1 = std::min(row0 + TILE, rows);
        for (int col0 = 0; col0 < cols; col0 += TILE)
        {
            const int col1 = std::min(col0 + TILE, cols);

            int row = row0;
            for (; row + N <= row1; row += N)
            {
                int col = col0;
                for (; col + N <= col1; col += N)
                {
                    batch_type matrix[N];
                    for (int i = 0; i < N; ++i)
                    {
                        std::array<T, N> lane {};
                        const size_t src_byte_offset =
                            (static_cast<size_t>(row + i) * static_cast<size_t>(cols) +
                             static_cast<size_t>(col)) * sizeof(T);
                        std::memcpy(lane.data(), src_raw + src_byte_offset, lane_bytes);
                        matrix[i] = batch_type::load_unaligned(lane.data());
                    }

                    xsimd::transpose(matrix, matrix + N);

                    for (int i = 0; i < N; ++i)
                    {
                        std::array<T, N> lane {};
                        matrix[i].store_unaligned(lane.data());

                        const size_t dst_byte_offset =
                            (static_cast<size_t>(col + i) * static_cast<size_t>(rows) +
                             static_cast<size_t>(row)) * sizeof(T);
                        std::memcpy(dst_raw + dst_byte_offset, lane.data(), lane_bytes);
                    }
                }
                // Handle remaining columns in this block of N rows
                for (; col < col1; ++col)
                {
                    for (int i = 0; i < N; ++i)
                    {
                        std::memcpy(
                            dst_raw + (static_cast<size_t>(col) * static_cast<size_t>(rows) +
                                       static_cast<size_t>(row + i)) * sizeof(T),
                            src_raw + (static_cast<size_t>(row + i) * static_cast<size_t>(cols) +
                                       static_cast<size_t>(col)) * sizeof(T),
                            sizeof(T));
                    }
                }
            }
            // Handle remaining rows in this TILE block
            for (; row < row1; ++row)
            {
                for (int col = col0; col < col1; ++col)
                {
                    std::memcpy(
                        dst_raw + (static_cast<size_t>(col) * static_cast<size_t>(rows) +
                                   static_cast<size_t>(row)) * sizeof(T),
                        src_raw + (static_cast<size_t>(row) * static_cast<size_t>(cols) +
                                   static_cast<size_t>(col)) * sizeof(T),
                        sizeof(T));
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

inline bool try_transpose2d_xsimd_for_element_size(const unsigned char* src,
                                                   unsigned char* dst,
                                                   int rows,
                                                   int cols,
                                                   size_t elem_size)
{
    switch (elem_size)
    {
        case 1:
            transpose2d_xsimd<int8_t>(src, dst, rows, cols);
            return true;
        case 2:
            transpose2d_xsimd<int16_t>(src, dst, rows, cols);
            return true;
        case 4:
            transpose2d_xsimd<float>(src, dst, rows, cols);
            return true;
        case 8:
            transpose2d_xsimd<double>(src, dst, rows, cols);
            return true;
        default:
            return false;
    }
}

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

inline int xsimd_probe_index_from_elem_size(size_t elem_size)
{
    switch (elem_size)
    {
        case 1: return 0;
        case 2: return 1;
        case 4: return 2;
        case 8: return 3;
        default: return -1;
    }
}

inline bool transpose_xsimd_probe_log_enabled()
{
    static const bool enabled = [] {
        const char* env = std::getenv("CVH_TRANSPOSE_XSIMD_PROBE_LOG");
        if (env == nullptr || env[0] == '\0')
        {
            return false;
        }
        if (std::strcmp(env, "0") == 0 ||
            std::strcmp(env, "false") == 0 ||
            std::strcmp(env, "FALSE") == 0 ||
            std::strcmp(env, "off") == 0 ||
            std::strcmp(env, "OFF") == 0)
        {
            return false;
        }
        return true;
    }();
    return enabled;
}

inline void log_transpose_xsimd_probe(size_t elem_size, const char* stage, const char* result)
{
    if (!transpose_xsimd_probe_log_enabled())
    {
        return;
    }
    std::fprintf(stderr,
                 "[cvh][transpose][xsimd-probe] elem_size=%zu stage=%s result=%s\n",
                 elem_size,
                 stage,
                 result);
    std::fflush(stderr);
}

inline void log_transpose_xsimd_probe_detail(size_t elem_size,
                                             int rows,
                                             int cols,
                                             size_t mismatch_byte,
                                             unsigned char got,
                                             unsigned char expected)
{
    if (!transpose_xsimd_probe_log_enabled())
    {
        return;
    }
    std::fprintf(stderr,
                 "[cvh][transpose][xsimd-probe] elem_size=%zu stage=probe-detail shape=%dx%d mismatch_byte=%zu got=%u expected=%u\n",
                 elem_size,
                 rows,
                 cols,
                 mismatch_byte,
                 static_cast<unsigned int>(got),
                 static_cast<unsigned int>(expected));
    std::fflush(stderr);
}

inline bool probe_transpose2d_xsimd_elem_size(size_t elem_size)
{
    // Probe multiple shapes to catch tail handling and non-square layout issues.
    constexpr std::array<std::array<int, 2>, 4> kProbeShapes = {{
        {{11, 29}},
        {{5, 7}},
        {{13, 29}},
        {{64, 65}},
    }};

    for (const auto& shape : kProbeShapes)
    {
        const int rows = shape[0];
        const int cols = shape[1];
        const size_t count =
            static_cast<size_t>(rows) * static_cast<size_t>(cols) * elem_size;

        std::vector<unsigned char> src(count);
        std::vector<unsigned char> dst(count);
        std::vector<unsigned char> ref(count);

        for (size_t i = 0; i < count; ++i)
        {
            src[i] = static_cast<unsigned char>((i * 131u + 17u) & 0xFFu);
        }

        if (!try_transpose2d_xsimd_for_element_size(src.data(), dst.data(), rows, cols, elem_size))
        {
            log_transpose_xsimd_probe_detail(elem_size, rows, cols, 0, 0, 0);
            return false;
        }

        transpose2d_memcpy_fallback(src.data(), ref.data(), rows, cols, elem_size);
        if (std::memcmp(dst.data(), ref.data(), count) != 0)
        {
            size_t mismatch_byte = 0;
            while (mismatch_byte < count && dst[mismatch_byte] == ref[mismatch_byte])
            {
                ++mismatch_byte;
            }
            const unsigned char got = mismatch_byte < count ? dst[mismatch_byte] : 0;
            const unsigned char expected = mismatch_byte < count ? ref[mismatch_byte] : 0;
            log_transpose_xsimd_probe_detail(elem_size, rows, cols, mismatch_byte, got, expected);
            return false;
        }
    }

    return true;
}

inline bool xsimd_transpose_allowed_for_elem_size(size_t elem_size)
{
    // 0 unknown, 1 pass, 2 fail
    static std::array<std::atomic<int>, 4> states = {
        std::atomic<int>{0},
        std::atomic<int>{0},
        std::atomic<int>{0},
        std::atomic<int>{0},
    };

    const int idx = xsimd_probe_index_from_elem_size(elem_size);
    if (idx < 0)
    {
        log_transpose_xsimd_probe(elem_size, "cache", "unsupported-elem-size");
        return false;
    }

    const int cached = states[static_cast<size_t>(idx)].load(std::memory_order_acquire);
    if (cached == 1)
    {
        log_transpose_xsimd_probe(elem_size, "cache", "pass");
        return true;
    }
    if (cached == 2)
    {
        log_transpose_xsimd_probe(elem_size, "cache", "fail");
        return false;
    }

    const bool ok = probe_transpose2d_xsimd_elem_size(elem_size);
    const int desired = ok ? 1 : 2;
    int expected = 0;
    const bool first_writer = states[static_cast<size_t>(idx)].compare_exchange_strong(
        expected, desired, std::memory_order_acq_rel);

    const int final_state = first_writer ?
                            desired :
                            states[static_cast<size_t>(idx)].load(std::memory_order_acquire);
    log_transpose_xsimd_probe(elem_size,
                              first_writer ? "probe" : "cache",
                              final_state == 1 ? "pass" : "fail");
    return final_state == 1;
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
    const DispatchMode mode = dispatch_mode();
    const bool allow_xsimd_transpose =
        mode != DispatchMode::ScalarOnly &&
        xsimd_transpose_allowed_for_elem_size(elem_size);

    if (allow_xsimd_transpose &&
        try_transpose2d_xsimd_for_element_size(src, dst, rows, cols, elem_size))
    {
        set_last_dispatch_tag(DispatchTag::XSimd);
        return;
    }

    if (mode == DispatchMode::XSimdOnly)
    {
        CV_Error(Error::StsNotImplemented, "transpose2d_kernel_blocked xsimd-only mode requested but no xsimd path is available");
    }

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
