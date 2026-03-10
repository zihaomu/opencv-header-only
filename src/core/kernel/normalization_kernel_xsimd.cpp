#include "normalization_kernel_xsimd.h"
#include "cvh/core/detail/openmp_utils.h"
#include "cvh/core/detail/xsimd_kernel_utils.h"

#include "xsimd/xsimd.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cvh {
namespace cpu {

namespace {

inline float rms_scale_xsimd(const float* input_row, size_t channels, float eps)
{
    XSimdBatch sum_vec(0.0f);
    size_t idx = 0;
    for (; idx + kXSimdBatchSize <= channels; idx += kXSimdBatchSize)
    {
        const XSimdBatch x = XSimdBatch::load_unaligned(input_row + idx);
        sum_vec = xsimd::fma(x, x, sum_vec);
    }

    float sum_sq = xsimd::reduce_add(sum_vec);
    for (; idx < channels; ++idx)
    {
        sum_sq += input_row[idx] * input_row[idx];
    }

    return 1.0f / std::sqrt(sum_sq / static_cast<float>(channels) + eps);
}

}  // namespace

void softmax_lastdim_xsimd(const float* input, float* output, size_t outer, size_t inner)
{
    const long long outer_ll = static_cast<long long>(outer);
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(outer, inner, 1LL << 14, 2))
#endif
    for (long long outer_idx = 0; outer_idx < outer_ll; ++outer_idx)
    {
        const size_t outer_i = static_cast<size_t>(outer_idx);
        const float* input_row = input + outer_i * inner;
        float* output_row = output + outer_i * inner;

        XSimdBatch max_vec(std::numeric_limits<float>::lowest());
        size_t idx = 0;
        for (; idx + kXSimdBatchSize <= inner; idx += kXSimdBatchSize)
        {
            max_vec = xsimd::max(max_vec, XSimdBatch::load_unaligned(input_row + idx));
        }

        float max_val = xsimd::reduce_max(max_vec);
        for (; idx < inner; ++idx)
        {
            max_val = std::max(max_val, input_row[idx]);
        }

        XSimdBatch sum_vec(0.0f);
        const XSimdBatch max_batch(max_val);
        idx = 0;
        for (; idx + kXSimdBatchSize <= inner; idx += kXSimdBatchSize)
        {
            const XSimdBatch exp_vec = xsimd::exp(XSimdBatch::load_unaligned(input_row + idx) - max_batch);
            exp_vec.store_unaligned(output_row + idx);
            sum_vec += exp_vec;
        }

        float sum_val = xsimd::reduce_add(sum_vec);
        for (; idx < inner; ++idx)
        {
            const float exp_v = std::exp(input_row[idx] - max_val);
            output_row[idx] = exp_v;
            sum_val += exp_v;
        }

        const float inv_sum = 1.0f / sum_val;
        const XSimdBatch inv_sum_vec(inv_sum);
        idx = 0;
        for (; idx + kXSimdBatchSize <= inner; idx += kXSimdBatchSize)
        {
            const XSimdBatch out_vec = XSimdBatch::load_unaligned(output_row + idx) * inv_sum_vec;
            out_vec.store_unaligned(output_row + idx);
        }
        for (; idx < inner; ++idx)
        {
            output_row[idx] *= inv_sum;
        }
    }
}

void causal_masked_softmax_square_xsimd(const float* input,
                                        float* output,
                                        size_t outer,
                                        size_t seq_len,
                                        float scale)
{
    const long long outer_ll = static_cast<long long>(outer);
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(outer, seq_len * seq_len, 1LL << 14, 2))
#endif
    for (long long outer_idx = 0; outer_idx < outer_ll; ++outer_idx)
    {
        const size_t outer_i = static_cast<size_t>(outer_idx);
        const float* input_mat = input + outer_i * seq_len * seq_len;
        float* output_mat = output + outer_i * seq_len * seq_len;
        const XSimdBatch scale_vec(scale);

        for (size_t row = 0; row < seq_len; ++row)
        {
            const size_t valid_cols = row + 1;
            const float* input_row = input_mat + row * seq_len;
            float* output_row = output_mat + row * seq_len;

            XSimdBatch max_vec(std::numeric_limits<float>::lowest());
            size_t idx = 0;
            for (; idx + kXSimdBatchSize <= valid_cols; idx += kXSimdBatchSize)
            {
                const XSimdBatch scaled = XSimdBatch::load_unaligned(input_row + idx) * scale_vec;
                max_vec = xsimd::max(max_vec, scaled);
            }

            float max_val = xsimd::reduce_max(max_vec);
            for (; idx < valid_cols; ++idx)
            {
                max_val = std::max(max_val, input_row[idx] * scale);
            }

            XSimdBatch sum_vec(0.0f);
            const XSimdBatch max_batch(max_val);
            idx = 0;
            for (; idx + kXSimdBatchSize <= valid_cols; idx += kXSimdBatchSize)
            {
                const XSimdBatch scaled = XSimdBatch::load_unaligned(input_row + idx) * scale_vec;
                const XSimdBatch exp_vec = xsimd::exp(scaled - max_batch);
                exp_vec.store_unaligned(output_row + idx);
                sum_vec += exp_vec;
            }

            float sum_val = xsimd::reduce_add(sum_vec);
            for (; idx < valid_cols; ++idx)
            {
                const float exp_v = std::exp(input_row[idx] * scale - max_val);
                output_row[idx] = exp_v;
                sum_val += exp_v;
            }

            const float inv_sum = 1.0f / sum_val;
            const XSimdBatch inv_sum_vec(inv_sum);
            idx = 0;
            for (; idx + kXSimdBatchSize <= valid_cols; idx += kXSimdBatchSize)
            {
                const XSimdBatch out_vec = XSimdBatch::load_unaligned(output_row + idx) * inv_sum_vec;
                out_vec.store_unaligned(output_row + idx);
            }
            for (; idx < valid_cols; ++idx)
            {
                output_row[idx] *= inv_sum;
            }

            std::fill(output_row + valid_cols, output_row + seq_len, 0.0f);
        }
    }
}

void rmsnorm_lastdim_xsimd_fp16_weight(const float* input,
                                       const hfloat* weight,
                                       float* output,
                                       size_t outer,
                                       size_t channels,
                                       float eps)
{
    const long long outer_ll = static_cast<long long>(outer);
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(outer, channels, 1LL << 14, 2))
#endif
    for (long long outer_idx = 0; outer_idx < outer_ll; ++outer_idx)
    {
        const size_t outer_i = static_cast<size_t>(outer_idx);
        const float* input_row = input + outer_i * channels;
        float* output_row = output + outer_i * channels;

        const float scale = rms_scale_xsimd(input_row, channels, eps);
        const XSimdBatch scale_vec(scale);

        size_t idx = 0;
        for (; idx + kXSimdBatchSize <= channels; idx += kXSimdBatchSize)
        {
            const XSimdBatch x = XSimdBatch::load_unaligned(input_row + idx);
            const XSimdBatch w = load_hfloat_batch(weight + idx);
            const XSimdBatch out_vec = x * scale_vec * w;
            out_vec.store_unaligned(output_row + idx);
        }
        for (; idx < channels; ++idx)
        {
            output_row[idx] = input_row[idx] * scale * static_cast<float>(weight[idx]);
        }
    }
}

void rmsnorm_lastdim_xsimd_i8_weight(const float* input,
                                     const int8_t* weight,
                                     const float* scales,
                                     float* output,
                                     size_t outer,
                                     size_t channels,
                                     float eps)
{
    const float weight_scale = scales[0];
    const long long outer_ll = static_cast<long long>(outer);
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(outer, channels, 1LL << 14, 2))
#endif
    for (long long outer_idx = 0; outer_idx < outer_ll; ++outer_idx)
    {
        const size_t outer_i = static_cast<size_t>(outer_idx);
        const float* input_row = input + outer_i * channels;
        float* output_row = output + outer_i * channels;

        const float scale = rms_scale_xsimd(input_row, channels, eps) * weight_scale;
        const XSimdBatch scale_vec(scale);

        size_t idx = 0;
        for (; idx + kXSimdBatchSize <= channels; idx += kXSimdBatchSize)
        {
            const XSimdBatch x = XSimdBatch::load_unaligned(input_row + idx);
            const XSimdBatch w = load_int8_batch(weight + idx);
            const XSimdBatch out_vec = x * scale_vec * w;
            out_vec.store_unaligned(output_row + idx);
        }
        for (; idx < channels; ++idx)
        {
            output_row[idx] = input_row[idx] * scale * static_cast<float>(weight[idx]);
        }
    }
}

void rmsnorm_lastdim_xsimd(const float* input,
                           const float* weight,
                           float* output,
                           size_t outer,
                           size_t channels,
                           float eps)
{
    const long long outer_ll = static_cast<long long>(outer);
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(outer, channels, 1LL << 14, 2))
#endif
    for (long long outer_idx = 0; outer_idx < outer_ll; ++outer_idx)
    {
        const size_t outer_i = static_cast<size_t>(outer_idx);
        const float* input_row = input + outer_i * channels;
        float* output_row = output + outer_i * channels;

        const float scale = rms_scale_xsimd(input_row, channels, eps);
        const XSimdBatch scale_vec(scale);

        size_t idx = 0;
        for (; idx + kXSimdBatchSize <= channels; idx += kXSimdBatchSize)
        {
            const XSimdBatch x = XSimdBatch::load_unaligned(input_row + idx);
            const XSimdBatch w = XSimdBatch::load_unaligned(weight + idx);
            const XSimdBatch out_vec = x * scale_vec * w;
            out_vec.store_unaligned(output_row + idx);
        }
        for (; idx < channels; ++idx)
        {
            output_row[idx] = input_row[idx] * scale * weight[idx];
        }
    }
}

}  // namespace cpu
}  // namespace cvh
