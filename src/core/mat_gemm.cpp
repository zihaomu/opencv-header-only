//
// Created by mzh on 2024/1/31.
//

#include "cvh/core/mat.h"
#include "cvh/core/basic_op.h"
#include "cvh/core/parallel.h"
#include "cvh/core/system.h"
#include "cvh/core/utils.h"
#include "kernel/gemm_kernel_xsimd.h"

#include <limits>

namespace cvh
{

// GEMM 的输入输出对Mat Shape的要求是：目前只支持单通道的Mat计算，支持多维度，支持广播机制，输出维度为输入维度的广播结果，输入维度必须大于等于2，最后两个维度分别是M和K，K和N，输出维度的最后两个维度分别是M和N。
static inline
MatShape make_strides(const MatShape& shape)
{
    MatShape strides(shape.size() - 2);
    size_t stride = 1;
    // 排除最后2个维度，他们作为内部循环，而其他的作为外部循环
    for (int i = (int)shape.size() - 2 - 1; i >= 0; i--)
    {
        strides[i] = stride;
        stride *= shape[i];
    }

    return strides;
}

static inline
size_t broadcast_linear_index(const MatShape& src_shape, const MatShape& src_strides,
                              const std::vector<int>& out_batch_index, int out_batch_dims)
{
    const int src_batch_dims = static_cast<int>(src_shape.size()) - 2;
    if (src_batch_dims <= 0)
        return 0;

    const int offset = out_batch_dims - src_batch_dims;
    CV_Assert(offset >= 0 && "Invalid broadcast mapping for GEMM batch dims!");

    size_t linear = 0;
    for (int d = 0; d < src_batch_dims; ++d)
    {
        int coord = src_shape[d] == 1 ? 0 : out_batch_index[d + offset];
        linear += static_cast<size_t>(coord) * src_strides[d];
    }
    return linear;
}

template <class Fn>
inline void for_each_gemm_batch(size_t out_loop, int m, int n, int k, Fn&& fn)
{
    const bool do_parallel = static_cast<long long>(m) * n * k <= (1LL << 15) &&
                             cpu::should_parallelize_1d_loop(
                                 out_loop,
                                 static_cast<size_t>(m) * static_cast<size_t>(n) * static_cast<size_t>(k),
                                 1LL << 15,
                                 2);
    const size_t max_parallel_range = static_cast<size_t>(std::numeric_limits<int>::max());
    if (!do_parallel || out_loop > max_parallel_range)
    {
        for (size_t i = 0; i < out_loop; ++i)
        {
            fn(i);
        }
        return;
    }

    cvh::parallel_for_(
        cvh::Range(0, static_cast<int>(out_loop)),
        [&](const cvh::Range& range) {
            for (int idx = range.start; idx < range.end; ++idx)
            {
                fn(static_cast<size_t>(idx));
            }
        },
        static_cast<double>(out_loop));
}

// naive impl, [M x K] x [K x N] = M x N
static inline
void gemm_impl_naive(const Mat& a, const Mat& b, Mat& c)
{
    MatShape shape_a = a.shape();
    MatShape shape_b = b.shape();

    // 目前不处理 K x KxN 这种情况。
    CV_Assert(shape_a.size() >= 2 && shape_b.size() >= 2 && "Mat shapes on gemm function are miss matching!");

    // generate output shape with brodcast rule
    // need to handle When a and b shape is 1xK x KxN, or MxK x 1xN, or MxK x Kx1
    MatShape shape_c = get_gemm_shape(shape_a, shape_b);

    // CV_Assert(shape_a.size() == shape_b.size() && "Two Mat dimension on gemm function are different!");
    int len_s = shape_a.size();
    CV_Assert(len_s >= 2 && "Only multi-dimension Mat is supported for gemm function!");

    int M = shape_a.size() == 1 ? 1 : shape_a[shape_a.size() - 2];
    int K = shape_a[shape_a.size() - 1];
    int N = shape_b.size() == 1 ? 1 : shape_b[shape_b.size() - 1];

    if (K != shape_b[shape_b.size() - 2])
    {
        std::string errorInfo = "Mat shapes on gemm([M,K] x [K,N]) function are miss matching!\n";
        errorInfo += "shape_a: ";
        errorInfo += shape_to_str(shape_a);
        errorInfo += "\n";
        errorInfo += "shape_b: ";
        errorInfo += shape_to_str(shape_b);
        errorInfo += "\n";
        errorInfo += "Expect gemm K = ";
        errorInfo += std::to_string(shape_b[shape_b.size() - 2]);
        errorInfo += ", but got ";
        errorInfo += std::to_string(K);
        errorInfo += "\n";
        CV_Error(NULL, errorInfo.c_str());
    }

    CV_Assert(a.type() == CV_32F && "Currently only FP32 activation mat is supported!");
    CV_Assert((b.type() == CV_32F || b.type() == CV_16F) && "NN gemm currently supports FP32/FP16 weights only!");

    // For dimension > 2, use numpy broadcasting rule for previous dimension.
    c = Mat(shape_c, CV_32F);

    const int out_batch_dims = static_cast<int>(shape_c.size()) - 2;
    size_t out_loop = out_batch_dims > 0 ? total(shape_c, 0, shape_c.size() - 2): 1;
    size_t step_a = M * K;
    size_t step_b = K * N;
    size_t step_c = M * N;

    MatShape stride_a = make_strides(shape_a);
    MatShape stride_b = make_strides(shape_b);
    MatShape stride_c = make_strides(shape_c);

    const float* pa = (const float*)a.data;
    float* pc = (float*)c.data;

    for_each_gemm_batch(out_loop, M, N, K, [&](size_t i) {
        size_t tmp = i;
        std::vector<int> idx_c(out_batch_dims);
        for (int d = 0; d < out_batch_dims; d++)
        {
            idx_c[d] = tmp / stride_c[d];
            tmp %= stride_c[d];
        }

        size_t lin_a = broadcast_linear_index(shape_a, stride_a, idx_c, out_batch_dims);
        size_t lin_b = broadcast_linear_index(shape_b, stride_b, idx_c, out_batch_dims);

        const float* pai = lin_a * step_a + pa;
        float* pci = i * step_c + pc;

        if (b.type() == CV_32F)
        {
            const float* pb = reinterpret_cast<const float*>(b.data);
            const float* pbi = lin_b * step_b + pb;
            cpu::gemm_kernel_xsimd_nn(pai, pbi, pci, M, N, K);
        }
        else
        {
            const hfloat* pb = reinterpret_cast<const hfloat*>(b.data);
            const hfloat* pbi = lin_b * step_b + pb;
            cpu::gemm_kernel_xsimd_nn_fp16(pai, pbi, pci, M, N, K);
        }
    });
}

// Mat b is not transposed! [M x K] x [N x K] = M x N
static inline
void gemm_impl_row(const Mat& a, const Mat& b, const Mat* b_scales, Mat& c)
{
    MatShape shape_a = a.shape();
    MatShape shape_b = b.shape();

#if 0
    // print shape_a and shape_b
    std::cout << "shape_a: " << shape_to_str(shape_a) << std::endl;
    std::cout << "shape_b: " << shape_to_str(shape_b) << std::endl;
#endif

    // 目前不处理 K x KxN 这种情况。
    CV_Assert(shape_a.size() >= 2 && shape_b.size() >= 2 && "Mat shapes on gemm function are miss matching!");

    // generate output shape with brodcast rule
    // need to handle When a and b shape is 1xK x KxN, or MxK x 1xN, or MxK x Kx1
    MatShape shape_c;

    int index_a = shape_a.size() - 2;
    int index_b = shape_b.size() - 2;

    while (index_a > 0 || index_b > 0)
    {
        if (index_a > 0 && index_b > 0)
        {
            if (shape_a[index_a - 1] == shape_b[index_b - 1])
            {
                shape_c.insert(shape_c.begin(), shape_a[index_a - 1]);
            }
            else if (shape_a[index_a - 1] == 1)
            {
                shape_c.insert(shape_c.begin(), shape_b[index_b - 1]);
            }
            else if (shape_b[index_b - 1] == 1)
            {
                shape_c.insert(shape_c.begin(), shape_a[index_a - 1]);
            }
            else
            {
                CV_Error(NULL, "Mat shapes on gemm function are miss matching!");
            }
            index_a--;
            index_b--;
        }
        else if (index_a > 0)
        {
            shape_c.insert(shape_c.begin(), shape_a[index_a - 1]);
            index_a--;
        }
        else if (index_b > 0)
        {
            shape_c.insert(shape_c.begin(), shape_b[index_b - 1]);
            index_b--;
        }
        else
            CV_Error(NULL, "Mat shapes on gemm function are miss matching!");
    }

    // 如果有一个维度为1维度，说明 出现 MxK x K = M的情况
    if (shape_a.size() == 1 || shape_b.size() == 1)
    {
        if (shape_a.size() == 1 && shape_b.size() == 1)
        {
            shape_c = {1}; // 处理 1xK x Kx1 = 1的情况
        }
    }
    else
    {
        shape_c.push_back(shape_a[shape_a.size() - 2]);
        shape_c.push_back(shape_b[shape_b.size() - 2]);
    }

    // CV_Assert(shape_a.size() == shape_b.size() && "Two Mat dimension on gemm function are different!");
    int len_s = shape_a.size();
    CV_Assert(len_s >= 2 && "Only multi-dimension Mat is supported for gemm function!");

    int M = shape_a.size() == 1 ? 1 : shape_a[shape_a.size() - 2];
    int K = shape_a[shape_a.size() - 1];
    int N = shape_b.size() == 1 ? 1 : shape_b[shape_b.size() - 2];

    // Check if K == shape_b[shape_b.size() - 1]
    if (K != shape_b[shape_b.size() - 1])
    {
        std::string errorInfo = "Mat shapes on gemm([M,K] x [N,K]) function are miss matching!\n";
        errorInfo += "shape_a: ";
        errorInfo += shape_to_str(shape_a);
        errorInfo += "\n";
        errorInfo += "shape_b: ";
        errorInfo += shape_to_str(shape_b);
        errorInfo += "\n";
        errorInfo += "Expact gemm K = " + std::to_string(K) + ", but got " + std::to_string(shape_b[shape_b.size() - 1]) + "\n";
        CV_Error(NULL, errorInfo.c_str());
    }

    CV_Assert(a.type() == CV_32F && "Currently only FP32 activation mat is supported!");
    CV_Assert((b.type() == CV_32F || b.type() == CV_16F || b.type() == CV_8S) &&
             "NT gemm currently supports FP32/FP16/INT8 weights!");
    if (b.type() == CV_8S)
    {
        CV_Assert(b_scales && !b_scales->empty() && "INT8 gemm requires non-empty weight scales!");
        CV_Assert(b_scales->type() == CV_32F);
        CV_Assert(b_scales->total() == static_cast<size_t>(N));
    }

    // For dimension > 2, use numpy broadcasting rule for previous dimension.
    c = Mat(shape_c, CV_32F);

    const int out_batch_dims = static_cast<int>(shape_c.size()) - 2;
    size_t out_loop = out_batch_dims > 0 ? total(shape_c, 0, shape_c.size() - 2): 1;
    size_t step_a = M * K;
    size_t step_b = K * N;
    size_t step_c = M * N;

    MatShape stride_a = make_strides(shape_a);
    MatShape stride_b = make_strides(shape_b);
    MatShape stride_c = make_strides(shape_c);
    const float* pa = (const float*)a.data;
    float* pc = (float*)c.data;

    for_each_gemm_batch(out_loop, M, N, K, [&](size_t i) {
        size_t tmp = i;
        std::vector<int> idx_c(out_batch_dims);
        for (int d = 0; d < out_batch_dims; d++)
        {
            idx_c[d] = tmp / stride_c[d];
            tmp %= stride_c[d];
        }

        size_t lin_a = broadcast_linear_index(shape_a, stride_a, idx_c, out_batch_dims);
        size_t lin_b = broadcast_linear_index(shape_b, stride_b, idx_c, out_batch_dims);

        const float* pai = lin_a * step_a + pa;
        float* pci = i * step_c + pc;

        if (b.type() == CV_32F)
        {
            const float* pb = reinterpret_cast<const float*>(b.data);
            const float* pbi = lin_b * step_b + pb;
            cpu::gemm_kernel_xsimd_nt(pai, pbi, pci, M, N, K);
        }
        else if (b.type() == CV_16F)
        {
            const hfloat* pb = reinterpret_cast<const hfloat*>(b.data);
            const hfloat* pbi = lin_b * step_b + pb;
            cpu::gemm_kernel_xsimd_nt_fp16(pai, pbi, pci, M, N, K);
        }
        else
        {
            const int8_t* pb = reinterpret_cast<const int8_t*>(b.data);
            const int8_t* pbi = lin_b * step_b + pb;
            const float* scale_ptr = reinterpret_cast<const float*>(b_scales->data);
            cpu::gemm_kernel_xsimd_nt_i8_rowwise(pai, pbi, scale_ptr, pci, M, N, K);
        }
    });
}

// implementation of rox x column
// TODO gemm support the different shape.
Mat gemm(const Mat& a, const Mat& b, bool transA, bool transB)
{
    Mat out;
    if (transA == false && transB == true)
    {
        gemm_impl_row(a, b, nullptr, out);
    }
    else
    {
        Mat aT;
        Mat bT;

        if (transA)
            aT = transpose(a);
        else
            aT = a;

        if (transB)
            bT = transpose(b);
        else
            bT = b;

        gemm_impl_naive(aT, bT, out);
    }

    return out;
}

Mat gemm(const Mat& a, const Mat& b, const Mat& b_scales, bool transA, bool transB)
{
    CV_Assert(!b_scales.empty() && "Quantized gemm requires non-empty scales!");
    if (b.type() != CV_8S)
    {
        return gemm(a, b, transA, transB);
    }

    Mat out;
    if (transA == false && transB == true)
    {
        gemm_impl_row(a, b, &b_scales, out);
        return out;
    }

    CV_Error_(Error::StsNotImplemented, ("INT8 gemm only supports transA=false, transB=true right now"));
    return {};
}

}
