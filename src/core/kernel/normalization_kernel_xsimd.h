#ifndef CVH_NORMALIZATION_KERNEL_XSIMD_H
#define CVH_NORMALIZATION_KERNEL_XSIMD_H

#include "cvh/core/define.h"

#include <cstddef>
#include <cstdint>

namespace cvh {
namespace cpu {

void softmax_lastdim_xsimd(const float* input, float* output, size_t outer, size_t inner);

void causal_masked_softmax_square_xsimd(const float* input,
                                        float* output,
                                        size_t outer,
                                        size_t seq_len,
                                        float scale = 1.0f);

void rmsnorm_lastdim_xsimd_fp16_weight(const float* input,
                                       const hfloat* weight,
                                       float* output,
                                       size_t outer,
                                       size_t channels,
                                       float eps);

void rmsnorm_lastdim_xsimd_i8_weight(const float* input,
                                     const int8_t* weight,
                                     const float* scales,
                                     float* output,
                                     size_t outer,
                                     size_t channels,
                                     float eps);

void rmsnorm_lastdim_xsimd(const float* input,
                           const float* weight,
                           float* output,
                           size_t outer,
                           size_t channels,
                           float eps);

}  // namespace cpu
}  // namespace cvh

#endif  // CVH_NORMALIZATION_KERNEL_XSIMD_H
