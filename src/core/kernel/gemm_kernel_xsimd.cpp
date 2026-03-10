#include "gemm_kernel_xsimd.h"
#include "cvh/core/detail/openmp_utils.h"
#include "cvh/core/detail/xsimd_kernel_utils.h"

#include "xsimd/xsimd.hpp"

#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cvh {
namespace cpu {

namespace {

constexpr int kKernelMR = 4;
constexpr int kKernelNRVecs = 2;
constexpr int kKernelNR = kKernelNRVecs * static_cast<int>(kXSimdBatchSize);

constexpr int kBlockKC = 256;
constexpr int kBlockMC = 128;
constexpr int kBlockNC = 256;

constexpr size_t kBlockedMinWork = 1ULL << 15;

inline int ceil_div(int value, int divisor)
{
    return (value + divisor - 1) / divisor;
}

inline bool should_use_blocked_kernel(int m, int n, int k)
{
    if (m <= 1 || n < kKernelNR || k < static_cast<int>(kXSimdBatchSize))
    {
        return false;
    }

    const size_t work = static_cast<size_t>(std::max(m, 0)) *
                        static_cast<size_t>(std::max(n, 0)) *
                        static_cast<size_t>(std::max(k, 0));
    return work >= kBlockedMinWork;
}

inline float dot_fp32_xsimd(const float* a_row, const float* b_row, int k)
{
    XSimdBatch sum_vec(0.0f);
    int ki = 0;
    for (; ki + static_cast<int>(kXSimdBatchSize) <= k; ki += static_cast<int>(kXSimdBatchSize))
    {
        const XSimdBatch va = XSimdBatch::load_unaligned(a_row + ki);
        const XSimdBatch vb = XSimdBatch::load_unaligned(b_row + ki);
        sum_vec = xsimd::fma(va, vb, sum_vec);
    }

    float sum = xsimd::reduce_add(sum_vec);
    for (; ki < k; ++ki)
    {
        sum += a_row[ki] * b_row[ki];
    }
    return sum;
}

inline float dot_fp16_xsimd(const float* a_row, const hfloat* b_row, int k)
{
    XSimdBatch sum_vec(0.0f);
    int ki = 0;
    for (; ki + static_cast<int>(kXSimdBatchSize) <= k; ki += static_cast<int>(kXSimdBatchSize))
    {
        const XSimdBatch va = XSimdBatch::load_unaligned(a_row + ki);
        const XSimdBatch vb = load_hfloat_batch(b_row + ki);
        sum_vec = xsimd::fma(va, vb, sum_vec);
    }

    float sum = xsimd::reduce_add(sum_vec);
    for (; ki < k; ++ki)
    {
        sum += a_row[ki] * static_cast<float>(b_row[ki]);
    }
    return sum;
}

inline float dot_i8_rowwise_xsimd(const float* a_row, const int8_t* b_row, float scale, int k)
{
    XSimdBatch sum_vec(0.0f);
    const XSimdBatch scale_vec(scale);
    int ki = 0;
    for (; ki + static_cast<int>(kXSimdBatchSize) <= k; ki += static_cast<int>(kXSimdBatchSize))
    {
        const XSimdBatch va = XSimdBatch::load_unaligned(a_row + ki);
        const XSimdBatch vb = load_int8_batch(b_row + ki) * scale_vec;
        sum_vec = xsimd::fma(va, vb, sum_vec);
    }

    float sum = xsimd::reduce_add(sum_vec);
    for (; ki < k; ++ki)
    {
        sum += a_row[ki] * (static_cast<float>(b_row[ki]) * scale);
    }
    return sum;
}

inline void microkernel_4x2v(const float* packed_a,
                             const float* packed_b,
                             float* c,
                             int ldc,
                             int kc,
                             int mr,
                             int nr)
{
    XSimdBatch acc[kKernelMR][kKernelNRVecs];
    for (int r = 0; r < kKernelMR; ++r)
    {
        for (int v = 0; v < kKernelNRVecs; ++v)
        {
            acc[r][v] = XSimdBatch(0.0f);
        }
    }

    for (int p = 0; p < kc; ++p)
    {
        const float* a_ptr = packed_a + static_cast<size_t>(p) * kKernelMR;
        const float* b_ptr = packed_b + static_cast<size_t>(p) * kKernelNR;

        const XSimdBatch b0 = XSimdBatch::load_unaligned(b_ptr);
        const XSimdBatch b1 = XSimdBatch::load_unaligned(b_ptr + kXSimdBatchSize);

        acc[0][0] = xsimd::fma(XSimdBatch(a_ptr[0]), b0, acc[0][0]);
        acc[0][1] = xsimd::fma(XSimdBatch(a_ptr[0]), b1, acc[0][1]);
        acc[1][0] = xsimd::fma(XSimdBatch(a_ptr[1]), b0, acc[1][0]);
        acc[1][1] = xsimd::fma(XSimdBatch(a_ptr[1]), b1, acc[1][1]);
        acc[2][0] = xsimd::fma(XSimdBatch(a_ptr[2]), b0, acc[2][0]);
        acc[2][1] = xsimd::fma(XSimdBatch(a_ptr[2]), b1, acc[2][1]);
        acc[3][0] = xsimd::fma(XSimdBatch(a_ptr[3]), b0, acc[3][0]);
        acc[3][1] = xsimd::fma(XSimdBatch(a_ptr[3]), b1, acc[3][1]);
    }

    if (mr == kKernelMR && nr == kKernelNR)
    {
        for (int r = 0; r < kKernelMR; ++r)
        {
            float* c_row = c + static_cast<size_t>(r) * ldc;
            (XSimdBatch::load_unaligned(c_row) + acc[r][0]).store_unaligned(c_row);
            (XSimdBatch::load_unaligned(c_row + kXSimdBatchSize) + acc[r][1]).store_unaligned(c_row + kXSimdBatchSize);
        }
        return;
    }

    alignas(64) float tile[kKernelMR * kKernelNR];
    for (int r = 0; r < kKernelMR; ++r)
    {
        acc[r][0].store_unaligned(tile + r * kKernelNR);
        acc[r][1].store_unaligned(tile + r * kKernelNR + kXSimdBatchSize);
    }

    for (int r = 0; r < mr; ++r)
    {
        float* c_row = c + static_cast<size_t>(r) * ldc;
        const float* tile_row = tile + r * kKernelNR;
        for (int col = 0; col < nr; ++col)
        {
            c_row[col] += tile_row[col];
        }
    }
}

inline void pack_a_panel(const float* a,
                         int lda,
                         int row_start,
                         int col_start,
                         int mc,
                         int kc,
                         std::vector<float>& packed_a)
{
    const int row_blocks = ceil_div(mc, kKernelMR);
    packed_a.assign(static_cast<size_t>(row_blocks) * kc * kKernelMR, 0.0f);

    for (int block = 0; block < row_blocks; ++block)
    {
        float* dst_block = packed_a.data() + static_cast<size_t>(block) * kc * kKernelMR;
        const int local_row_base = block * kKernelMR;

        for (int p = 0; p < kc; ++p)
        {
            float* dst = dst_block + static_cast<size_t>(p) * kKernelMR;
            const int src_col = col_start + p;
            for (int r = 0; r < kKernelMR; ++r)
            {
                const int local_row = local_row_base + r;
                if (local_row < mc)
                {
                    dst[r] = a[static_cast<size_t>(row_start + local_row) * lda + src_col];
                }
            }
        }
    }
}

template <class LoadBScalar>
inline void pack_b_panel(int kc,
                         int nc,
                         LoadBScalar&& load_b_scalar,
                         std::vector<float>& packed_b)
{
    const int col_blocks = ceil_div(nc, kKernelNR);
    packed_b.assign(static_cast<size_t>(col_blocks) * kc * kKernelNR, 0.0f);

    for (int block = 0; block < col_blocks; ++block)
    {
        float* dst_block = packed_b.data() + static_cast<size_t>(block) * kc * kKernelNR;
        const int local_col_base = block * kKernelNR;

        for (int p = 0; p < kc; ++p)
        {
            float* dst = dst_block + static_cast<size_t>(p) * kKernelNR;
            for (int j = 0; j < kKernelNR; ++j)
            {
                const int local_col = local_col_base + j;
                if (local_col < nc)
                {
                    dst[j] = load_b_scalar(p, local_col);
                }
            }
        }
    }
}

template <class LoadBScalar>
void gemm_kernel_blocked_impl(const float* a,
                              float* c,
                              int m,
                              int n,
                              int k,
                              LoadBScalar&& load_b_scalar)
{
    std::fill_n(c, static_cast<size_t>(m) * static_cast<size_t>(n), 0.0f);

    const int jc_blocks = ceil_div(n, kBlockNC);
    const bool parallel_jc = should_parallelize_1d_loop(
        static_cast<size_t>(jc_blocks),
        static_cast<size_t>(std::max(m, 1)) * static_cast<size_t>(std::max(k, 1)) * static_cast<size_t>(kBlockNC),
        1LL << 16,
        1);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(parallel_jc)
#endif
    for (int jb = 0; jb < jc_blocks; ++jb)
    {
        std::vector<float> packed_b;
        std::vector<float> packed_a;
        packed_b.reserve(static_cast<size_t>(kBlockKC) * kBlockNC);
        packed_a.reserve(static_cast<size_t>(kBlockKC) * kBlockMC);

        const int jc = jb * kBlockNC;
        const int nc = std::min(kBlockNC, n - jc);

        for (int pc = 0; pc < k; pc += kBlockKC)
        {
            const int kc = std::min(kBlockKC, k - pc);
            pack_b_panel(kc, nc,
                         [&](int p, int local_col) {
                             return load_b_scalar(pc + p, jc + local_col);
                         },
                         packed_b);

            const int col_blocks = ceil_div(nc, kKernelNR);

            for (int ic = 0; ic < m; ic += kBlockMC)
            {
                const int mc = std::min(kBlockMC, m - ic);
                pack_a_panel(a, k, ic, pc, mc, kc, packed_a);

                const int row_blocks = ceil_div(mc, kKernelMR);
                for (int block_col = 0; block_col < col_blocks; ++block_col)
                {
                    const int nr = std::min(kKernelNR, nc - block_col * kKernelNR);
                    const float* packed_b_block = packed_b.data() + static_cast<size_t>(block_col) * kc * kKernelNR;

                    for (int block_row = 0; block_row < row_blocks; ++block_row)
                    {
                        const int mr = std::min(kKernelMR, mc - block_row * kKernelMR);
                        const float* packed_a_block = packed_a.data() + static_cast<size_t>(block_row) * kc * kKernelMR;
                        float* c_block = c + static_cast<size_t>(ic + block_row * kKernelMR) * n +
                                         (jc + block_col * kKernelNR);

                        microkernel_4x2v(packed_a_block, packed_b_block, c_block, n, kc, mr, nr);
                    }
                }
            }
        }
    }
}

template <class PackedT>
inline void pack_b_raw_from_kn(const PackedT* b, PackedT* packed_b, int n, int k)
{
    const int col_blocks = ceil_div(n, kKernelNR);
    std::fill_n(packed_b,
                static_cast<size_t>(col_blocks) * static_cast<size_t>(k) * static_cast<size_t>(kKernelNR),
                static_cast<PackedT>(0));

    for (int block = 0; block < col_blocks; ++block)
    {
        PackedT* dst_block = packed_b + static_cast<size_t>(block) * k * kKernelNR;
        const int col_base = block * kKernelNR;
        for (int p = 0; p < k; ++p)
        {
            PackedT* dst = dst_block + static_cast<size_t>(p) * kKernelNR;
            for (int lane = 0; lane < kKernelNR; ++lane)
            {
                const int col = col_base + lane;
                if (col < n)
                {
                    dst[lane] = b[static_cast<size_t>(p) * n + col];
                }
            }
        }
    }
}

inline void pack_scales_rowwise(const float* scales, float* packed_scales, int n)
{
    const int col_blocks = ceil_div(n, kKernelNR);
    std::fill_n(packed_scales, static_cast<size_t>(col_blocks) * kKernelNR, 0.0f);

    for (int block = 0; block < col_blocks; ++block)
    {
        float* dst = packed_scales + static_cast<size_t>(block) * kKernelNR;
        const int col_base = block * kKernelNR;
        for (int lane = 0; lane < kKernelNR; ++lane)
        {
            const int col = col_base + lane;
            if (col < n)
            {
                dst[lane] = scales[col];
            }
        }
    }
}

inline void store_row_block(float* c, int nr, const XSimdBatch& sum0, const XSimdBatch& sum1)
{
    if (nr == kKernelNR)
    {
        sum0.store_unaligned(c);
        sum1.store_unaligned(c + kXSimdBatchSize);
        return;
    }

    alignas(64) float tmp[kKernelNR];
    sum0.store_unaligned(tmp);
    sum1.store_unaligned(tmp + kXSimdBatchSize);
    for (int lane = 0; lane < nr; ++lane)
    {
        c[lane] = tmp[lane];
    }
}

template <class PackedT, class LoadPackedVec>
inline void gemm_kernel_xsimd_row_packed_impl(const float* a,
                                              const PackedT* packed_b,
                                              float* c,
                                              int n,
                                              int k,
                                              LoadPackedVec&& load_packed_vec)
{
    const int col_blocks = ceil_div(n, kKernelNR);
    for (int block = 0; block < col_blocks; ++block)
    {
        const PackedT* block_ptr = packed_b + static_cast<size_t>(block) * k * kKernelNR;
        XSimdBatch sum0(0.0f);
        XSimdBatch sum1(0.0f);

        for (int p = 0; p < k; ++p)
        {
            XSimdBatch b0(0.0f);
            XSimdBatch b1(0.0f);
            load_packed_vec(block_ptr + static_cast<size_t>(p) * kKernelNR, b0, b1);

            const XSimdBatch a_vec(a[p]);
            sum0 = xsimd::fma(a_vec, b0, sum0);
            sum1 = xsimd::fma(a_vec, b1, sum1);
        }

        const int nr = std::min(kKernelNR, n - block * kKernelNR);
        store_row_block(c + static_cast<size_t>(block) * kKernelNR, nr, sum0, sum1);
    }
}

inline void gemm_kernel_xsimd_row_packed_i8_impl(const float* a,
                                                 const int8_t* packed_b,
                                                 const float* packed_scales,
                                                 float* c,
                                                 int n,
                                                 int k)
{
    const int col_blocks = ceil_div(n, kKernelNR);
    for (int block = 0; block < col_blocks; ++block)
    {
        const int8_t* block_ptr = packed_b + static_cast<size_t>(block) * k * kKernelNR;
        const float* scale_ptr = packed_scales + static_cast<size_t>(block) * kKernelNR;
        XSimdBatch sum0(0.0f);
        XSimdBatch sum1(0.0f);

        for (int p = 0; p < k; ++p)
        {
            const int8_t* b_ptr = block_ptr + static_cast<size_t>(p) * kKernelNR;
            const XSimdBatch a_vec(a[p]);
            const XSimdBatch b0 = load_int8_batch(b_ptr);
            const XSimdBatch b1 = load_int8_batch(b_ptr + kXSimdBatchSize);
            sum0 = xsimd::fma(a_vec, b0, sum0);
            sum1 = xsimd::fma(a_vec, b1, sum1);
        }

        const XSimdBatch scale0 = XSimdBatch::load_unaligned(scale_ptr);
        const XSimdBatch scale1 = XSimdBatch::load_unaligned(scale_ptr + kXSimdBatchSize);
        const int nr = std::min(kKernelNR, n - block * kKernelNR);
        store_row_block(c + static_cast<size_t>(block) * kKernelNR, nr, sum0 * scale0, sum1 * scale1);
    }
}

void gemm_kernel_xsimd_nn_simple_fp32(const float* a, const float* b, float* c,
                                      int m, int n, int k)
{
    const int lanes = static_cast<int>(kXSimdBatchSize);

#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(m, static_cast<size_t>(n) * static_cast<size_t>(k), 1LL << 16, 1))
#endif
    for (int mi = 0; mi < m; ++mi)
    {
        const float* a_row = a + static_cast<size_t>(mi) * k;
        float* c_row = c + static_cast<size_t>(mi) * n;

        int ni = 0;
        for (; ni + lanes <= n; ni += lanes)
        {
            XSimdBatch sum_vec(0.0f);
            for (int ki = 0; ki < k; ++ki)
            {
                const XSimdBatch a_vec(a_row[ki]);
                const XSimdBatch b_vec = XSimdBatch::load_unaligned(b + static_cast<size_t>(ki) * n + ni);
                sum_vec = xsimd::fma(a_vec, b_vec, sum_vec);
            }
            sum_vec.store_unaligned(c_row + ni);
        }

        for (; ni < n; ++ni)
        {
            float sum = 0.0f;
            for (int ki = 0; ki < k; ++ki)
            {
                sum += a_row[ki] * b[static_cast<size_t>(ki) * n + ni];
            }
            c_row[ni] = sum;
        }
    }
}

void gemm_kernel_xsimd_nt_simple_fp32(const float* a, const float* b, float* c,
                                      int m, int n, int k)
{
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(m, static_cast<size_t>(n) * static_cast<size_t>(k), 1LL << 16, 1))
#endif
    for (int mi = 0; mi < m; ++mi)
    {
        const float* a_row = a + static_cast<size_t>(mi) * k;
        float* c_row = c + static_cast<size_t>(mi) * n;

        for (int ni = 0; ni < n; ++ni)
        {
            const float* b_row = b + static_cast<size_t>(ni) * k;
            c_row[ni] = dot_fp32_xsimd(a_row, b_row, k);
        }
    }
}

void gemm_kernel_xsimd_nn_simple_fp16(const float* a, const hfloat* b, float* c,
                                      int m, int n, int k)
{
    const int lanes = static_cast<int>(kXSimdBatchSize);

#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(m, static_cast<size_t>(n) * static_cast<size_t>(k), 1LL << 16, 1))
#endif
    for (int mi = 0; mi < m; ++mi)
    {
        const float* a_row = a + static_cast<size_t>(mi) * k;
        float* c_row = c + static_cast<size_t>(mi) * n;

        int ni = 0;
        for (; ni + lanes <= n; ni += lanes)
        {
            XSimdBatch sum_vec(0.0f);
            for (int ki = 0; ki < k; ++ki)
            {
                const XSimdBatch a_vec(a_row[ki]);
                const XSimdBatch b_vec = load_hfloat_batch(b + static_cast<size_t>(ki) * n + ni);
                sum_vec = xsimd::fma(a_vec, b_vec, sum_vec);
            }
            sum_vec.store_unaligned(c_row + ni);
        }

        for (; ni < n; ++ni)
        {
            float sum = 0.0f;
            for (int ki = 0; ki < k; ++ki)
            {
                sum += a_row[ki] * static_cast<float>(b[static_cast<size_t>(ki) * n + ni]);
            }
            c_row[ni] = sum;
        }
    }
}

void gemm_kernel_xsimd_nt_simple_fp16(const float* a, const hfloat* b, float* c,
                                      int m, int n, int k)
{
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(m, static_cast<size_t>(n) * static_cast<size_t>(k), 1LL << 16, 1))
#endif
    for (int mi = 0; mi < m; ++mi)
    {
        const float* a_row = a + static_cast<size_t>(mi) * k;
        float* c_row = c + static_cast<size_t>(mi) * n;

        for (int ni = 0; ni < n; ++ni)
        {
            const hfloat* b_row = b + static_cast<size_t>(ni) * k;
            c_row[ni] = dot_fp16_xsimd(a_row, b_row, k);
        }
    }
}

void gemm_kernel_xsimd_nn_simple_i8_rowwise(const float* a, const int8_t* b, const float* scales, float* c,
                                            int m, int n, int k)
{
    const int lanes = static_cast<int>(kXSimdBatchSize);

#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(m, static_cast<size_t>(n) * static_cast<size_t>(k), 1LL << 16, 1))
#endif
    for (int mi = 0; mi < m; ++mi)
    {
        const float* a_row = a + static_cast<size_t>(mi) * k;
        float* c_row = c + static_cast<size_t>(mi) * n;

        int ni = 0;
        for (; ni + lanes <= n; ni += lanes)
        {
            XSimdBatch sum_vec(0.0f);
            const XSimdBatch scale_vec = XSimdBatch::load_unaligned(scales + ni);
            for (int ki = 0; ki < k; ++ki)
            {
                const XSimdBatch a_vec(a_row[ki]);
                const XSimdBatch b_vec = load_int8_batch(b + static_cast<size_t>(ki) * n + ni);
                sum_vec = xsimd::fma(a_vec, b_vec, sum_vec);
            }
            (sum_vec * scale_vec).store_unaligned(c_row + ni);
        }

        for (; ni < n; ++ni)
        {
            float sum = 0.0f;
            for (int ki = 0; ki < k; ++ki)
            {
                sum += a_row[ki] * static_cast<float>(b[static_cast<size_t>(ki) * n + ni]);
            }
            c_row[ni] = sum * scales[ni];
        }
    }
}

void gemm_kernel_xsimd_nt_simple_i8_rowwise(const float* a, const int8_t* b, const float* scales, float* c,
                                            int m, int n, int k)
{
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(m, static_cast<size_t>(n) * static_cast<size_t>(k), 1LL << 16, 1))
#endif
    for (int mi = 0; mi < m; ++mi)
    {
        const float* a_row = a + static_cast<size_t>(mi) * k;
        float* c_row = c + static_cast<size_t>(mi) * n;

        for (int ni = 0; ni < n; ++ni)
        {
            const int8_t* b_row = b + static_cast<size_t>(ni) * k;
            c_row[ni] = dot_i8_rowwise_xsimd(a_row, b_row, scales[ni], k);
        }
    }
}

}  // namespace

std::size_t gemm_xsimd_packed_b_elements(int n, int k)
{
    return static_cast<std::size_t>(ceil_div(n, kKernelNR)) *
           static_cast<std::size_t>(std::max(k, 0)) *
           static_cast<std::size_t>(kKernelNR);
}

std::size_t gemm_xsimd_packed_scale_elements(int n)
{
    return static_cast<std::size_t>(ceil_div(n, kKernelNR)) *
           static_cast<std::size_t>(kKernelNR);
}

void gemm_pack_xsimd_nn_fp32(const float* b, float* packed_b, int n, int k)
{
    pack_b_raw_from_kn(b, packed_b, n, k);
}

void gemm_pack_xsimd_nn_fp16(const hfloat* b, hfloat* packed_b, int n, int k)
{
    pack_b_raw_from_kn(b, packed_b, n, k);
}

void gemm_pack_xsimd_nn_i8_rowwise(const int8_t* b, const float* scales, int8_t* packed_b, float* packed_scales,
                                   int n, int k)
{
    pack_b_raw_from_kn(b, packed_b, n, k);
    pack_scales_rowwise(scales, packed_scales, n);
}

void gemm_kernel_xsimd_row_packed_fp32(const float* a, const float* packed_b, float* c, int n, int k)
{
    gemm_kernel_xsimd_row_packed_impl(
        a, packed_b, c, n, k,
        [](const float* src, XSimdBatch& b0, XSimdBatch& b1) {
            b0 = XSimdBatch::load_unaligned(src);
            b1 = XSimdBatch::load_unaligned(src + kXSimdBatchSize);
        });
}

void gemm_kernel_xsimd_row_packed_fp16(const float* a, const hfloat* packed_b, float* c, int n, int k)
{
    gemm_kernel_xsimd_row_packed_impl(
        a, packed_b, c, n, k,
        [](const hfloat* src, XSimdBatch& b0, XSimdBatch& b1) {
            b0 = load_hfloat_batch(src);
            b1 = load_hfloat_batch(src + kXSimdBatchSize);
        });
}

void gemm_kernel_xsimd_row_packed_i8_rowwise(const float* a, const int8_t* packed_b, const float* packed_scales,
                                             float* c, int n, int k)
{
    gemm_kernel_xsimd_row_packed_i8_impl(a, packed_b, packed_scales, c, n, k);
}

void gemm_kernel_xsimd_nn(const float* a, const float* b, float* c,
                          int m, int n, int k)
{
    if (!should_use_blocked_kernel(m, n, k))
    {
        gemm_kernel_xsimd_nn_simple_fp32(a, b, c, m, n, k);
        return;
    }

    gemm_kernel_blocked_impl(
        a, c, m, n, k,
        [&](int src_k, int src_n) {
            return b[static_cast<size_t>(src_k) * n + src_n];
        });
}

void gemm_kernel_xsimd_nt(const float* a, const float* b, float* c,
                          int m, int n, int k)
{
    if (!should_use_blocked_kernel(m, n, k))
    {
        gemm_kernel_xsimd_nt_simple_fp32(a, b, c, m, n, k);
        return;
    }

    gemm_kernel_blocked_impl(
        a, c, m, n, k,
        [&](int src_k, int src_n) {
            return b[static_cast<size_t>(src_n) * k + src_k];
        });
}

void gemm_kernel_xsimd_nn_fp16(const float* a, const hfloat* b, float* c,
                               int m, int n, int k)
{
    if (!should_use_blocked_kernel(m, n, k))
    {
        gemm_kernel_xsimd_nn_simple_fp16(a, b, c, m, n, k);
        return;
    }

    gemm_kernel_blocked_impl(
        a, c, m, n, k,
        [&](int src_k, int src_n) {
            return static_cast<float>(b[static_cast<size_t>(src_k) * n + src_n]);
        });
}

void gemm_kernel_xsimd_nn_i8_rowwise(const float* a, const int8_t* b, const float* scales, float* c,
                                     int m, int n, int k)
{
    if (!should_use_blocked_kernel(m, n, k))
    {
        gemm_kernel_xsimd_nn_simple_i8_rowwise(a, b, scales, c, m, n, k);
        return;
    }

    gemm_kernel_blocked_impl(
        a, c, m, n, k,
        [&](int src_k, int src_n) {
            return static_cast<float>(b[static_cast<size_t>(src_k) * n + src_n]) * scales[src_n];
        });
}

void gemm_kernel_xsimd_nt_fp16(const float* a, const hfloat* b, float* c,
                               int m, int n, int k)
{
    if (!should_use_blocked_kernel(m, n, k))
    {
        gemm_kernel_xsimd_nt_simple_fp16(a, b, c, m, n, k);
        return;
    }

    gemm_kernel_blocked_impl(
        a, c, m, n, k,
        [&](int src_k, int src_n) {
            return static_cast<float>(b[static_cast<size_t>(src_n) * k + src_k]);
        });
}

void gemm_kernel_xsimd_nt_i8_rowwise(const float* a, const int8_t* b, const float* scales, float* c,
                                     int m, int n, int k)
{
    if (!should_use_blocked_kernel(m, n, k))
    {
        gemm_kernel_xsimd_nt_simple_i8_rowwise(a, b, scales, c, m, n, k);
        return;
    }

    gemm_kernel_blocked_impl(
        a, c, m, n, k,
        [&](int src_k, int src_n) {
            return static_cast<float>(b[static_cast<size_t>(src_n) * k + src_k]) * scales[src_n];
        });
}

}  // namespace cpu
}  // namespace cvh
