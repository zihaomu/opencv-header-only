#include "cvh.h"

#include <cmath>

float cvh_core_header_odr_peer();

int main()
{
    cvh::Mat a({2, 2}, CV_32F);
    cvh::Mat b({2, 2}, CV_32F);
    a = 2.0f;
    b = 3.0f;

    cvh::Mat sum;
    cvh::add(a, b, sum);
    cvh::Mat transposed = cvh::transpose(sum);
    cvh::Mat product = cvh::gemm(a, b);
    cvh::Mat expression = a + b;

    const float sum_value = reinterpret_cast<const float*>(transposed.data)[0];
    const float gemm_value = reinterpret_cast<const float*>(product.data)[0];
    const float expr_value = reinterpret_cast<const float*>(expression.data)[0];
    const float peer_value = cvh_core_header_odr_peer();

    return std::fabs(sum_value - 5.0f) < 1e-6f &&
                   std::fabs(gemm_value - 12.0f) < 1e-6f &&
                   std::fabs(expr_value - 5.0f) < 1e-6f &&
                   std::fabs(peer_value - 4.0f) < 1e-6f
               ? 0
               : 1;
}
