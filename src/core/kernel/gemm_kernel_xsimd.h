#ifndef CVH_GEMM_KERNEL_XSIMD_H
#define CVH_GEMM_KERNEL_XSIMD_H

#include "cvh/core/define.h"

#include <cstddef>

namespace cvh {
namespace cpu {

// A[M, K] x B[K, N] -> C[M, N]
void gemm_kernel_xsimd_nn(const float* a, const float* b, float* c,
                          int m, int n, int k);

// A[M, K] x B[N, K] -> C[M, N], where B is row-major [N, K].
void gemm_kernel_xsimd_nt(const float* a, const float* b, float* c,
                          int m, int n, int k);

void gemm_kernel_xsimd_nn_fp16(const float* a, const hfloat* b, float* c,
                               int m, int n, int k);

void gemm_kernel_xsimd_nn_i8_rowwise(const float* a, const int8_t* b, const float* scales, float* c,
                                     int m, int n, int k);

void gemm_kernel_xsimd_nt_fp16(const float* a, const hfloat* b, float* c,
                               int m, int n, int k);

void gemm_kernel_xsimd_nt_i8_rowwise(const float* a, const int8_t* b, const float* scales, float* c,
                                     int m, int n, int k);

std::size_t gemm_xsimd_packed_b_elements(int n, int k);
std::size_t gemm_xsimd_packed_scale_elements(int n);

void gemm_pack_xsimd_nn_fp32(const float* b, float* packed_b, int n, int k);
void gemm_pack_xsimd_nn_fp16(const hfloat* b, hfloat* packed_b, int n, int k);
void gemm_pack_xsimd_nn_i8_rowwise(const int8_t* b, const float* scales, int8_t* packed_b, float* packed_scales,
                                   int n, int k);

void gemm_kernel_xsimd_row_packed_fp32(const float* a, const float* packed_b, float* c, int n, int k);
void gemm_kernel_xsimd_row_packed_fp16(const float* a, const hfloat* packed_b, float* c, int n, int k);
void gemm_kernel_xsimd_row_packed_i8_rowwise(const float* a, const int8_t* packed_b, const float* packed_scales,
                                             float* c, int n, int k);

}  // namespace cpu
}  // namespace cvh

#endif  // CVH_GEMM_KERNEL_XSIMD_H
