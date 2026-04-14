#include "cvh/core/basic_op.h"
#include "cvh/core/detail/dispatch_control.h"
#include "cvh/core/saturate.h"
#include "binary_kernel_xsimd.h"
#include "transpose_kernel.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <vector>

namespace cvh
{

namespace
{

inline void ensure_non_empty(const Mat& src, const char* fn_name)
{
    if (src.empty())
    {
        CV_Error_(Error::StsBadArg, ("%s expects non-empty src Mat", fn_name));
    }
}

inline void ensure_same_type_and_shape(const Mat& a, const Mat& b, const char* fn_name)
{
    ensure_non_empty(a, fn_name);
    ensure_non_empty(b, fn_name);

    if (a.type() != b.type())
    {
        CV_Error_(Error::StsBadType,
                  ("%s type mismatch, a_type=%d b_type=%d", fn_name, a.type(), b.type()));
    }
    if (a.shape() != b.shape())
    {
        CV_Error_(Error::StsUnmatchedSizes, ("%s shape mismatch", fn_name));
    }
}

inline void ensure_binary_dst_like_src(const Mat& src, Mat& dst, const char* fn_name)
{
    ensure_non_empty(src, fn_name);

    if (dst.empty())
    {
        dst.create(src.dims, src.size.p, src.type());
        return;
    }

    if (dst.type() != src.type() || dst.shape() != src.shape())
    {
        CV_Error_(Error::StsBadType,
                  ("%s dst mismatch, src_type=%d dst_type=%d", fn_name, src.type(), dst.type()));
    }
}

inline void ensure_compare_dst_like_src(const Mat& src, Mat& dst, const char* fn_name)
{
    ensure_non_empty(src, fn_name);

    const int dst_type = CV_MAKETYPE(CV_8U, src.channels());
    if (dst.empty())
    {
        dst.create(src.dims, src.size.p, dst_type);
        return;
    }

    if (dst.type() != dst_type || dst.shape() != src.shape())
    {
        CV_Error_(Error::StsBadType,
                  ("%s dst mismatch, expected_type=%d got_type=%d", fn_name, dst_type, dst.type()));
    }
}

inline void check_scalar_channel_bound(const Mat& src, const char* fn_name)
{
    if (src.channels() > 4)
    {
        CV_Error_(Error::StsBadArg,
                  ("%s supports channels <= 4 in v1, channels=%d", fn_name, src.channels()));
    }
}

template<typename T>
inline auto safe_div_value(T lhs, T rhs)
{
    if constexpr (std::is_integral<T>::value)
    {
        if (rhs == 0)
        {
            return T(0);
        }
        return static_cast<T>(lhs / rhs);
    }
    else if constexpr (std::is_same<T, hfloat>::value)
    {
        const float r = static_cast<float>(rhs);
        if (r == 0.0f)
        {
            return 0.0f;
        }
        return static_cast<float>(lhs) / r;
    }
    else
    {
        return lhs / rhs;
    }
}

template<typename T, typename Op>
void apply_mat_mat_binary_impl(const Mat& a, const Mat& b, Mat& dst, Op op)
{
    const int cn = a.channels();
    const size_t outer = a.dims > 1 ? static_cast<size_t>(a.size.p[0]) : 1;
    const size_t pixel_per_outer = a.dims > 1 ? a.total(1, a.dims) : a.total();
    const size_t a_step0 = a.dims > 1 ? a.step(0) : pixel_per_outer * a.elemSize();
    const size_t b_step0 = b.dims > 1 ? b.step(0) : pixel_per_outer * b.elemSize();
    const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : pixel_per_outer * dst.elemSize();

    for (size_t i = 0; i < outer; ++i)
    {
        const T* a_row = reinterpret_cast<const T*>(a.data + i * a_step0);
        const T* b_row = reinterpret_cast<const T*>(b.data + i * b_step0);
        T* dst_row = reinterpret_cast<T*>(dst.data + i * dst_step0);

        for (size_t p = 0; p < pixel_per_outer; ++p)
        {
            const size_t base = p * static_cast<size_t>(cn);
            for (int ch = 0; ch < cn; ++ch)
            {
                const size_t idx = base + static_cast<size_t>(ch);
                dst_row[idx] = saturate_cast<T>(op(a_row[idx], b_row[idx]));
            }
        }
    }
}

template<typename Op>
void dispatch_mat_mat_binary_impl_by_depth(const Mat& a, const Mat& b, Mat& dst, Op op, const char* fn_name)
{
    switch (a.depth())
    {
        case CV_8U:
            apply_mat_mat_binary_impl<uchar>(a, b, dst, op);
            break;
        case CV_8S:
            apply_mat_mat_binary_impl<schar>(a, b, dst, op);
            break;
        case CV_16U:
            apply_mat_mat_binary_impl<ushort>(a, b, dst, op);
            break;
        case CV_16S:
            apply_mat_mat_binary_impl<short>(a, b, dst, op);
            break;
        case CV_32S:
            apply_mat_mat_binary_impl<int>(a, b, dst, op);
            break;
        case CV_32U:
            apply_mat_mat_binary_impl<uint>(a, b, dst, op);
            break;
        case CV_32F:
            apply_mat_mat_binary_impl<float>(a, b, dst, op);
            break;
        case CV_16F:
            apply_mat_mat_binary_impl<hfloat>(a, b, dst, op);
            break;
        case CV_64F:
            apply_mat_mat_binary_impl<double>(a, b, dst, op);
            break;
        default:
            CV_Error_(Error::StsNotImplemented,
                      ("%s supports depth in [CV_8U..CV_64F], depth=%d", fn_name, a.depth()));
    }
}

inline bool try_dispatch_mat_mat_binary_xsimd_fp32(const Mat& a,
                                                   const Mat& b,
                                                   Mat& dst,
                                                   cpu::BinaryKernelOp op)
{
    if (a.depth() != CV_32F)
    {
        return false;
    }

    const int cn = a.channels();
    const size_t outer = a.dims > 1 ? static_cast<size_t>(a.size.p[0]) : 1;
    const size_t pixel_per_outer = a.dims > 1 ? a.total(1, a.dims) : a.total();
    const size_t row_elements = pixel_per_outer * static_cast<size_t>(cn);

    const size_t a_step0 = a.dims > 1 ? a.step(0) : row_elements * sizeof(float);
    const size_t b_step0 = b.dims > 1 ? b.step(0) : row_elements * sizeof(float);
    const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : row_elements * sizeof(float);

    for (size_t i = 0; i < outer; ++i)
    {
        const float* a_row = reinterpret_cast<const float*>(a.data + i * a_step0);
        const float* b_row = reinterpret_cast<const float*>(b.data + i * b_step0);
        float* dst_row = reinterpret_cast<float*>(dst.data + i * dst_step0);

        cpu::binary_broadcast_xsimd(
            op,
            a_row,
            row_elements,
            1,
            b_row,
            row_elements,
            1,
            dst_row,
            1,
            row_elements);
    }

    return true;
}

inline bool is_uniform_scalar_for_channels(const Scalar& scalar, int channels)
{
    for (int ch = 1; ch < channels; ++ch)
    {
        if (scalar[ch] != scalar[0])
        {
            return false;
        }
    }
    return true;
}

inline bool is_int_xsimd_op_supported(int depth, cpu::BinaryKernelOp op)
{
    if (op == cpu::BinaryKernelOp::Max || op == cpu::BinaryKernelOp::Min)
    {
        return true;
    }

    if (depth == CV_8U || depth == CV_8S || depth == CV_16U || depth == CV_16S)
    {
        return op == cpu::BinaryKernelOp::Add ||
               op == cpu::BinaryKernelOp::Sub ||
               op == cpu::BinaryKernelOp::Mul;
    }

    if (depth == CV_32S || depth == CV_32U)
    {
        return op == cpu::BinaryKernelOp::Add ||
               op == cpu::BinaryKernelOp::Sub ||
               op == cpu::BinaryKernelOp::Mul;
    }

    return false;
}

inline bool try_dispatch_mat_mat_binary_xsimd(const Mat& a,
                                              const Mat& b,
                                              Mat& dst,
                                              cpu::BinaryKernelOp op)
{
    if (try_dispatch_mat_mat_binary_xsimd_fp32(a, b, dst, op))
    {
        return true;
    }

    const size_t outer = a.dims > 1 ? static_cast<size_t>(a.size.p[0]) : 1;
    const size_t pixel_per_outer = a.dims > 1 ? a.total(1, a.dims) : a.total();
    const size_t row_elements = pixel_per_outer * static_cast<size_t>(a.channels());

    switch (a.depth())
    {
        case CV_16F:
        {
            const size_t a_step0 = a.dims > 1 ? a.step(0) : row_elements * sizeof(hfloat);
            const size_t b_step0 = b.dims > 1 ? b.step(0) : row_elements * sizeof(hfloat);
            const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : row_elements * sizeof(hfloat);
            for (size_t i = 0; i < outer; ++i)
            {
                const void* a_row = a.data + i * a_step0;
                const void* b_row = b.data + i * b_step0;
                void* dst_row = dst.data + i * dst_step0;
                cpu::binary_broadcast_xsimd_hfloat(
                    op,
                    a_row,
                    row_elements,
                    1,
                    b_row,
                    row_elements,
                    1,
                    dst_row,
                    1,
                    row_elements);
            }
            return true;
        }
        case CV_64F:
        {
            const size_t a_step0 = a.dims > 1 ? a.step(0) : row_elements * sizeof(double);
            const size_t b_step0 = b.dims > 1 ? b.step(0) : row_elements * sizeof(double);
            const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : row_elements * sizeof(double);
            for (size_t i = 0; i < outer; ++i)
            {
                const void* a_row = a.data + i * a_step0;
                const void* b_row = b.data + i * b_step0;
                void* dst_row = dst.data + i * dst_step0;
                cpu::binary_broadcast_xsimd_double(
                    op,
                    a_row,
                    row_elements,
                    1,
                    b_row,
                    row_elements,
                    1,
                    dst_row,
                    1,
                    row_elements);
            }
            return true;
        }
        case CV_32S:
        {
            if (!is_int_xsimd_op_supported(a.depth(), op))
            {
                return false;
            }
            const size_t a_step0 = a.dims > 1 ? a.step(0) : row_elements * sizeof(int);
            const size_t b_step0 = b.dims > 1 ? b.step(0) : row_elements * sizeof(int);
            const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : row_elements * sizeof(int);
            for (size_t i = 0; i < outer; ++i)
            {
                const void* a_row = a.data + i * a_step0;
                const void* b_row = b.data + i * b_step0;
                void* dst_row = dst.data + i * dst_step0;
                cpu::binary_broadcast_xsimd_int32(
                    op,
                    a_row,
                    row_elements,
                    1,
                    b_row,
                    row_elements,
                    1,
                    dst_row,
                    1,
                    row_elements);
            }
            return true;
        }
        case CV_32U:
        {
            if (!is_int_xsimd_op_supported(a.depth(), op))
            {
                return false;
            }
            const size_t a_step0 = a.dims > 1 ? a.step(0) : row_elements * sizeof(uint);
            const size_t b_step0 = b.dims > 1 ? b.step(0) : row_elements * sizeof(uint);
            const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : row_elements * sizeof(uint);
            for (size_t i = 0; i < outer; ++i)
            {
                const void* a_row = a.data + i * a_step0;
                const void* b_row = b.data + i * b_step0;
                void* dst_row = dst.data + i * dst_step0;
                cpu::binary_broadcast_xsimd_uint32(
                    op,
                    a_row,
                    row_elements,
                    1,
                    b_row,
                    row_elements,
                    1,
                    dst_row,
                    1,
                    row_elements);
            }
            return true;
        }
        case CV_16S:
        {
            if (!is_int_xsimd_op_supported(a.depth(), op))
            {
                return false;
            }
            const size_t a_step0 = a.dims > 1 ? a.step(0) : row_elements * sizeof(short);
            const size_t b_step0 = b.dims > 1 ? b.step(0) : row_elements * sizeof(short);
            const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : row_elements * sizeof(short);
            for (size_t i = 0; i < outer; ++i)
            {
                const void* a_row = a.data + i * a_step0;
                const void* b_row = b.data + i * b_step0;
                void* dst_row = dst.data + i * dst_step0;
                cpu::binary_broadcast_xsimd_int16(
                    op,
                    a_row,
                    row_elements,
                    1,
                    b_row,
                    row_elements,
                    1,
                    dst_row,
                    1,
                    row_elements);
            }
            return true;
        }
        case CV_16U:
        {
            if (!is_int_xsimd_op_supported(a.depth(), op))
            {
                return false;
            }
            const size_t a_step0 = a.dims > 1 ? a.step(0) : row_elements * sizeof(ushort);
            const size_t b_step0 = b.dims > 1 ? b.step(0) : row_elements * sizeof(ushort);
            const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : row_elements * sizeof(ushort);
            for (size_t i = 0; i < outer; ++i)
            {
                const void* a_row = a.data + i * a_step0;
                const void* b_row = b.data + i * b_step0;
                void* dst_row = dst.data + i * dst_step0;
                cpu::binary_broadcast_xsimd_uint16(
                    op,
                    a_row,
                    row_elements,
                    1,
                    b_row,
                    row_elements,
                    1,
                    dst_row,
                    1,
                    row_elements);
            }
            return true;
        }
        case CV_8S:
        {
            if (!is_int_xsimd_op_supported(a.depth(), op))
            {
                return false;
            }
            const size_t a_step0 = a.dims > 1 ? a.step(0) : row_elements * sizeof(schar);
            const size_t b_step0 = b.dims > 1 ? b.step(0) : row_elements * sizeof(schar);
            const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : row_elements * sizeof(schar);
            for (size_t i = 0; i < outer; ++i)
            {
                const void* a_row = a.data + i * a_step0;
                const void* b_row = b.data + i * b_step0;
                void* dst_row = dst.data + i * dst_step0;
                cpu::binary_broadcast_xsimd_int8(
                    op,
                    a_row,
                    row_elements,
                    1,
                    b_row,
                    row_elements,
                    1,
                    dst_row,
                    1,
                    row_elements);
            }
            return true;
        }
        case CV_8U:
        {
            if (!is_int_xsimd_op_supported(a.depth(), op))
            {
                return false;
            }
            const size_t a_step0 = a.dims > 1 ? a.step(0) : row_elements * sizeof(uchar);
            const size_t b_step0 = b.dims > 1 ? b.step(0) : row_elements * sizeof(uchar);
            const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : row_elements * sizeof(uchar);
            for (size_t i = 0; i < outer; ++i)
            {
                const void* a_row = a.data + i * a_step0;
                const void* b_row = b.data + i * b_step0;
                void* dst_row = dst.data + i * dst_step0;
                cpu::binary_broadcast_xsimd_uint8(
                    op,
                    a_row,
                    row_elements,
                    1,
                    b_row,
                    row_elements,
                    1,
                    dst_row,
                    1,
                    row_elements);
            }
            return true;
        }
        default:
            return false;
    }
}

inline bool to_compare_kernel_op(int op, cpu::CompareKernelOp& kernel_op)
{
    switch (op)
    {
        case CV_CMP_EQ:
            kernel_op = cpu::CompareKernelOp::Eq;
            return true;
        case CV_CMP_GT:
            kernel_op = cpu::CompareKernelOp::Gt;
            return true;
        case CV_CMP_GE:
            kernel_op = cpu::CompareKernelOp::Ge;
            return true;
        case CV_CMP_LT:
            kernel_op = cpu::CompareKernelOp::Lt;
            return true;
        case CV_CMP_LE:
            kernel_op = cpu::CompareKernelOp::Le;
            return true;
        case CV_CMP_NE:
            kernel_op = cpu::CompareKernelOp::Ne;
            return true;
        default:
            return false;
    }
}

inline bool try_dispatch_mat_mat_compare_xsimd_fp32(const Mat& a,
                                                    const Mat& b,
                                                    Mat& dst,
                                                    int op)
{
    if (a.depth() != CV_32F)
    {
        return false;
    }

    cpu::CompareKernelOp kernel_op;
    if (!to_compare_kernel_op(op, kernel_op))
    {
        return false;
    }

    const int cn = a.channels();
    const size_t outer = a.dims > 1 ? static_cast<size_t>(a.size.p[0]) : 1;
    const size_t pixel_per_outer = a.dims > 1 ? a.total(1, a.dims) : a.total();
    const size_t row_elements = pixel_per_outer * static_cast<size_t>(cn);

    const size_t a_step0 = a.dims > 1 ? a.step(0) : row_elements * sizeof(float);
    const size_t b_step0 = b.dims > 1 ? b.step(0) : row_elements * sizeof(float);
    const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : row_elements * sizeof(uchar);
    cpu::compare_broadcast_xsimd(
        kernel_op,
        reinterpret_cast<const float*>(a.data),
        a_step0 / sizeof(float),
        1,
        reinterpret_cast<const float*>(b.data),
        b_step0 / sizeof(float),
        1,
        reinterpret_cast<uchar*>(dst.data),
        dst_step0 / sizeof(uchar),
        outer,
        row_elements);

    return true;
}

inline bool try_dispatch_mat_mat_compare_xsimd_fp16(const Mat& a,
                                                    const Mat& b,
                                                    Mat& dst,
                                                    int op)
{
#ifndef _OPENMP
    (void)a;
    (void)b;
    (void)dst;
    (void)op;
    return false;
#else
    if (a.depth() != CV_16F)
    {
        return false;
    }

    cpu::CompareKernelOp kernel_op;
    if (!to_compare_kernel_op(op, kernel_op))
    {
        return false;
    }

    const int cn = a.channels();
    const size_t outer = a.dims > 1 ? static_cast<size_t>(a.size.p[0]) : 1;
    const size_t pixel_per_outer = a.dims > 1 ? a.total(1, a.dims) : a.total();
    const size_t row_elements = pixel_per_outer * static_cast<size_t>(cn);

    const size_t a_step0 = a.dims > 1 ? a.step(0) : row_elements * sizeof(hfloat);
    const size_t b_step0 = b.dims > 1 ? b.step(0) : row_elements * sizeof(hfloat);
    const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : row_elements * sizeof(uchar);
    cpu::compare_broadcast_xsimd_hfloat(
        kernel_op,
        a.data,
        a_step0 / sizeof(hfloat),
        1,
        b.data,
        b_step0 / sizeof(hfloat),
        1,
        reinterpret_cast<uchar*>(dst.data),
        dst_step0 / sizeof(uchar),
        outer,
        row_elements);

    return true;
#endif
}

inline bool try_dispatch_mat_mat_compare_xsimd_int(const Mat& a,
                                                   const Mat& b,
                                                   Mat& dst,
                                                   int op)
{
    cpu::CompareKernelOp kernel_op;
    if (!to_compare_kernel_op(op, kernel_op))
    {
        return false;
    }

    const int cn = a.channels();
    const size_t outer = a.dims > 1 ? static_cast<size_t>(a.size.p[0]) : 1;
    const size_t pixel_per_outer = a.dims > 1 ? a.total(1, a.dims) : a.total();
    const size_t row_elements = pixel_per_outer * static_cast<size_t>(cn);
    const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : row_elements * sizeof(uchar);

    switch (a.depth())
    {
        case CV_8U:
        {
            const size_t a_step0 = a.dims > 1 ? a.step(0) : row_elements * sizeof(uchar);
            const size_t b_step0 = b.dims > 1 ? b.step(0) : row_elements * sizeof(uchar);
            cpu::compare_broadcast_xsimd_uint8(kernel_op, a.data, a_step0 / sizeof(uchar), 1,
                                               b.data, b_step0 / sizeof(uchar), 1,
                                               reinterpret_cast<uchar*>(dst.data),
                                               dst_step0 / sizeof(uchar), outer, row_elements);
            return true;
        }
        case CV_8S:
        {
            const size_t a_step0 = a.dims > 1 ? a.step(0) : row_elements * sizeof(schar);
            const size_t b_step0 = b.dims > 1 ? b.step(0) : row_elements * sizeof(schar);
            cpu::compare_broadcast_xsimd_int8(kernel_op, a.data, a_step0 / sizeof(schar), 1,
                                              b.data, b_step0 / sizeof(schar), 1,
                                              reinterpret_cast<uchar*>(dst.data),
                                              dst_step0 / sizeof(uchar), outer, row_elements);
            return true;
        }
        case CV_16U:
        {
            const size_t a_step0 = a.dims > 1 ? a.step(0) : row_elements * sizeof(ushort);
            const size_t b_step0 = b.dims > 1 ? b.step(0) : row_elements * sizeof(ushort);
            cpu::compare_broadcast_xsimd_uint16(kernel_op, a.data, a_step0 / sizeof(ushort), 1,
                                                b.data, b_step0 / sizeof(ushort), 1,
                                                reinterpret_cast<uchar*>(dst.data),
                                                dst_step0 / sizeof(uchar), outer, row_elements);
            return true;
        }
        case CV_16S:
        {
            const size_t a_step0 = a.dims > 1 ? a.step(0) : row_elements * sizeof(short);
            const size_t b_step0 = b.dims > 1 ? b.step(0) : row_elements * sizeof(short);
            cpu::compare_broadcast_xsimd_int16(kernel_op, a.data, a_step0 / sizeof(short), 1,
                                               b.data, b_step0 / sizeof(short), 1,
                                               reinterpret_cast<uchar*>(dst.data),
                                               dst_step0 / sizeof(uchar), outer, row_elements);
            return true;
        }
        case CV_32S:
        {
            const size_t a_step0 = a.dims > 1 ? a.step(0) : row_elements * sizeof(int);
            const size_t b_step0 = b.dims > 1 ? b.step(0) : row_elements * sizeof(int);
            cpu::compare_broadcast_xsimd_int32(kernel_op, a.data, a_step0 / sizeof(int), 1,
                                               b.data, b_step0 / sizeof(int), 1,
                                               reinterpret_cast<uchar*>(dst.data),
                                               dst_step0 / sizeof(uchar), outer, row_elements);
            return true;
        }
        case CV_32U:
        {
            const size_t a_step0 = a.dims > 1 ? a.step(0) : row_elements * sizeof(uint);
            const size_t b_step0 = b.dims > 1 ? b.step(0) : row_elements * sizeof(uint);
            cpu::compare_broadcast_xsimd_uint32(kernel_op, a.data, a_step0 / sizeof(uint), 1,
                                                b.data, b_step0 / sizeof(uint), 1,
                                                reinterpret_cast<uchar*>(dst.data),
                                                dst_step0 / sizeof(uchar), outer, row_elements);
            return true;
        }
        default:
            return false;
    }
}

inline bool try_dispatch_mat_scalar_binary_xsimd_fp32(const Mat& src,
                                                      const Scalar& scalar,
                                                      Mat& dst,
                                                      cpu::BinaryKernelOp op,
                                                      bool scalar_first)
{
    if (src.depth() != CV_32F)
    {
        return false;
    }

    const int cn = src.channels();
    if (cn <= 0)
    {
        return false;
    }

    float scalar_lanes[4] = {0.f, 0.f, 0.f, 0.f};
    for (int ch = 0; ch < cn; ++ch)
    {
        scalar_lanes[ch] = saturate_cast<float>(scalar[ch]);
    }

    const bool uniform_scalar = is_uniform_scalar_for_channels(scalar, cn);
    const float scalar_val = scalar_lanes[0];
    const size_t outer = src.dims > 1 ? static_cast<size_t>(src.size.p[0]) : 1;
    const size_t pixel_per_outer = src.dims > 1 ? src.total(1, src.dims) : src.total();
    const size_t row_elements = pixel_per_outer * static_cast<size_t>(cn);
    const size_t src_step0 = src.dims > 1 ? src.step(0) : row_elements * sizeof(float);
    const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : row_elements * sizeof(float);
    const size_t src_outer_stride = src_step0 / sizeof(float);
    const size_t dst_outer_stride = dst_step0 / sizeof(float);

    if (uniform_scalar)
    {
        for (size_t i = 0; i < outer; ++i)
        {
            const float* src_row = reinterpret_cast<const float*>(src.data + i * src_step0);
            float* dst_row = reinterpret_cast<float*>(dst.data + i * dst_step0);
            if (scalar_first)
            {
                cpu::binary_broadcast_xsimd(
                    op,
                    &scalar_val,
                    0,
                    0,
                    src_row,
                    row_elements,
                    1,
                    dst_row,
                    1,
                    row_elements);
            }
            else
            {
                cpu::binary_broadcast_xsimd(
                    op,
                    src_row,
                    row_elements,
                    1,
                    &scalar_val,
                    0,
                    0,
                    dst_row,
                    1,
                    row_elements);
            }
        }
    }
    else
    {
        cpu::binary_scalar_channels_xsimd(op,
                                          reinterpret_cast<const float*>(src.data),
                                          src_outer_stride,
                                          scalar_lanes,
                                          cn,
                                          reinterpret_cast<float*>(dst.data),
                                          dst_outer_stride,
                                          outer,
                                          row_elements,
                                          scalar_first);
    }

    return true;
}

inline bool try_dispatch_mat_scalar_binary_xsimd_fp16(const Mat& src,
                                                      const Scalar& scalar,
                                                      Mat& dst,
                                                      cpu::BinaryKernelOp op,
                                                      bool scalar_first)
{
    if (src.depth() != CV_16F)
    {
        return false;
    }

    const int cn = src.channels();
    if (cn <= 0 || cn > 4)
    {
        return false;
    }

    float scalar_lanes[4] = {0.f, 0.f, 0.f, 0.f};
    for (int ch = 0; ch < cn; ++ch)
    {
        scalar_lanes[ch] = saturate_cast<float>(scalar[ch]);
    }
    const bool uniform_scalar = is_uniform_scalar_for_channels(scalar, cn);
    const hfloat scalar_val = saturate_cast<hfloat>(scalar_lanes[0]);
    const size_t outer = src.dims > 1 ? static_cast<size_t>(src.size.p[0]) : 1;
    const size_t pixel_per_outer = src.dims > 1 ? src.total(1, src.dims) : src.total();
    const size_t row_elements = pixel_per_outer * static_cast<size_t>(cn);
    const size_t src_step0 = src.dims > 1 ? src.step(0) : row_elements * sizeof(hfloat);
    const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : row_elements * sizeof(hfloat);
    const size_t src_outer_stride = src_step0 / sizeof(hfloat);
    const size_t dst_outer_stride = dst_step0 / sizeof(hfloat);

    if (!uniform_scalar)
    {
        cpu::binary_scalar_channels_xsimd_hfloat(op,
                                                 src.data,
                                                 src_outer_stride,
                                                 scalar_lanes,
                                                 cn,
                                                 dst.data,
                                                 dst_outer_stride,
                                                 outer,
                                                 row_elements,
                                                 scalar_first);
        return true;
    }

    for (size_t i = 0; i < outer; ++i)
    {
        const void* src_row = src.data + i * src_step0;
        void* dst_row = dst.data + i * dst_step0;
        if (!scalar_first)
        {
            cpu::binary_broadcast_xsimd_hfloat(
                op,
                src_row,
                row_elements,
                1,
                &scalar_val,
                0,
                0,
                dst_row,
                1,
                row_elements);
        }
        else
        {
            cpu::binary_broadcast_xsimd_hfloat(
                op,
                &scalar_val,
                0,
                0,
                src_row,
                row_elements,
                1,
                dst_row,
                1,
                row_elements);
        }
    }

    return true;
}

inline bool try_dispatch_mat_scalar_binary_xsimd_int32(const Mat& src,
                                                       const Scalar& scalar,
                                                       Mat& dst,
                                                       cpu::BinaryKernelOp op,
                                                       bool scalar_first)
{
    if (src.depth() != CV_32S)
    {
        return false;
    }

    const int cn = src.channels();
    if (cn <= 0 || cn > 4)
    {
        return false;
    }

    std::int32_t scalar_lanes[4] = {0, 0, 0, 0};
    for (int ch = 0; ch < cn; ++ch)
    {
        scalar_lanes[ch] = saturate_cast<std::int32_t>(scalar[ch]);
    }

    const bool uniform_scalar = is_uniform_scalar_for_channels(scalar, cn);
    const std::int32_t scalar_val = scalar_lanes[0];
    const size_t outer = src.dims > 1 ? static_cast<size_t>(src.size.p[0]) : 1;
    const size_t pixel_per_outer = src.dims > 1 ? src.total(1, src.dims) : src.total();
    const size_t row_elements = pixel_per_outer * static_cast<size_t>(cn);
    const size_t src_step0 = src.dims > 1 ? src.step(0) : row_elements * sizeof(std::int32_t);
    const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : row_elements * sizeof(std::int32_t);
    const size_t src_outer_stride = src_step0 / sizeof(std::int32_t);
    const size_t dst_outer_stride = dst_step0 / sizeof(std::int32_t);

    if (uniform_scalar)
    {
        for (size_t i = 0; i < outer; ++i)
        {
            const void* src_row = src.data + i * src_step0;
            void* dst_row = dst.data + i * dst_step0;
            if (scalar_first)
            {
                cpu::binary_broadcast_xsimd_int32(
                    op,
                    &scalar_val,
                    0,
                    0,
                    src_row,
                    row_elements,
                    1,
                    dst_row,
                    1,
                    row_elements);
            }
            else
            {
                cpu::binary_broadcast_xsimd_int32(
                    op,
                    src_row,
                    row_elements,
                    1,
                    &scalar_val,
                    0,
                    0,
                    dst_row,
                    1,
                    row_elements);
            }
        }
    }
    else
    {
        cpu::binary_scalar_channels_xsimd_int32(op,
                                                src.data,
                                                src_outer_stride,
                                                scalar_lanes,
                                                cn,
                                                dst.data,
                                                dst_outer_stride,
                                                outer,
                                                row_elements,
                                                scalar_first);
    }

    return true;
}

template<typename T>
inline bool try_dispatch_mat_scalar_binary_xsimd_smallint(const Mat& src,
                                                          const Scalar& scalar,
                                                          Mat& dst,
                                                          cpu::BinaryKernelOp op,
                                                          bool scalar_first,
                                                          int expected_depth)
{
    if (op != cpu::BinaryKernelOp::Add &&
        op != cpu::BinaryKernelOp::Sub &&
        op != cpu::BinaryKernelOp::Mul)
    {
        return false;
    }

    if (src.depth() != expected_depth)
    {
        return false;
    }

    const int cn = src.channels();
    if (cn <= 0 || cn > 4)
    {
        return false;
    }

    T scalar_lanes[4] = {T(0), T(0), T(0), T(0)};
    for (int ch = 0; ch < cn; ++ch)
    {
        scalar_lanes[ch] = saturate_cast<T>(scalar[ch]);
    }

    const bool uniform_scalar = is_uniform_scalar_for_channels(scalar, cn);
    const T scalar_val = scalar_lanes[0];
    const size_t outer = src.dims > 1 ? static_cast<size_t>(src.size.p[0]) : 1;
    const size_t pixel_per_outer = src.dims > 1 ? src.total(1, src.dims) : src.total();
    const size_t row_elements = pixel_per_outer * static_cast<size_t>(cn);
    const size_t src_step0 = src.dims > 1 ? src.step(0) : row_elements * sizeof(T);
    const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : row_elements * sizeof(T);
    const size_t src_outer_stride = src_step0 / sizeof(T);
    const size_t dst_outer_stride = dst_step0 / sizeof(T);

    if (uniform_scalar)
    {
        for (size_t i = 0; i < outer; ++i)
        {
            const void* src_row = src.data + i * src_step0;
            void* dst_row = dst.data + i * dst_step0;
            if (expected_depth == CV_8U)
            {
                cpu::binary_broadcast_xsimd_uint8(op,
                                                  scalar_first ? static_cast<const void*>(&scalar_val) : src_row,
                                                  scalar_first ? 0 : row_elements,
                                                  scalar_first ? 0 : 1,
                                                  scalar_first ? src_row : static_cast<const void*>(&scalar_val),
                                                  scalar_first ? row_elements : 0,
                                                  scalar_first ? 1 : 0,
                                                  dst_row,
                                                  1,
                                                  row_elements);
            }
            else if (expected_depth == CV_8S)
            {
                cpu::binary_broadcast_xsimd_int8(op,
                                                 scalar_first ? static_cast<const void*>(&scalar_val) : src_row,
                                                 scalar_first ? 0 : row_elements,
                                                 scalar_first ? 0 : 1,
                                                 scalar_first ? src_row : static_cast<const void*>(&scalar_val),
                                                 scalar_first ? row_elements : 0,
                                                 scalar_first ? 1 : 0,
                                                 dst_row,
                                                 1,
                                                 row_elements);
            }
            else if (expected_depth == CV_16U)
            {
                cpu::binary_broadcast_xsimd_uint16(op,
                                                   scalar_first ? static_cast<const void*>(&scalar_val) : src_row,
                                                   scalar_first ? 0 : row_elements,
                                                   scalar_first ? 0 : 1,
                                                   scalar_first ? src_row : static_cast<const void*>(&scalar_val),
                                                   scalar_first ? row_elements : 0,
                                                   scalar_first ? 1 : 0,
                                                   dst_row,
                                                   1,
                                                   row_elements);
            }
            else
            {
                cpu::binary_broadcast_xsimd_int16(op,
                                                  scalar_first ? static_cast<const void*>(&scalar_val) : src_row,
                                                  scalar_first ? 0 : row_elements,
                                                  scalar_first ? 0 : 1,
                                                  scalar_first ? src_row : static_cast<const void*>(&scalar_val),
                                                  scalar_first ? row_elements : 0,
                                                  scalar_first ? 1 : 0,
                                                  dst_row,
                                                  1,
                                                  row_elements);
            }
        }
    }
    else
    {
        if (expected_depth == CV_8U)
        {
            cpu::binary_scalar_channels_xsimd_uint8(op,
                                                    src.data,
                                                    src_outer_stride,
                                                    reinterpret_cast<const std::uint8_t*>(scalar_lanes),
                                                    cn,
                                                    dst.data,
                                                    dst_outer_stride,
                                                    outer,
                                                    row_elements,
                                                    scalar_first);
        }
        else if (expected_depth == CV_8S)
        {
            cpu::binary_scalar_channels_xsimd_int8(op,
                                                   src.data,
                                                   src_outer_stride,
                                                   reinterpret_cast<const std::int8_t*>(scalar_lanes),
                                                   cn,
                                                   dst.data,
                                                   dst_outer_stride,
                                                   outer,
                                                   row_elements,
                                                   scalar_first);
        }
        else if (expected_depth == CV_16U)
        {
            cpu::binary_scalar_channels_xsimd_uint16(op,
                                                     src.data,
                                                     src_outer_stride,
                                                     reinterpret_cast<const std::uint16_t*>(scalar_lanes),
                                                     cn,
                                                     dst.data,
                                                     dst_outer_stride,
                                                     outer,
                                                     row_elements,
                                                     scalar_first);
        }
        else
        {
            cpu::binary_scalar_channels_xsimd_int16(op,
                                                    src.data,
                                                    src_outer_stride,
                                                    reinterpret_cast<const std::int16_t*>(scalar_lanes),
                                                    cn,
                                                    dst.data,
                                                    dst_outer_stride,
                                                    outer,
                                                    row_elements,
                                                    scalar_first);
        }
    }

    return true;
}

inline bool try_dispatch_mat_scalar_binary_xsimd_uint8(const Mat& src,
                                                       const Scalar& scalar,
                                                       Mat& dst,
                                                       cpu::BinaryKernelOp op,
                                                       bool scalar_first)
{
    return try_dispatch_mat_scalar_binary_xsimd_smallint<uchar>(src, scalar, dst, op, scalar_first, CV_8U);
}

inline bool try_dispatch_mat_scalar_binary_xsimd_int8(const Mat& src,
                                                      const Scalar& scalar,
                                                      Mat& dst,
                                                      cpu::BinaryKernelOp op,
                                                      bool scalar_first)
{
    return try_dispatch_mat_scalar_binary_xsimd_smallint<schar>(src, scalar, dst, op, scalar_first, CV_8S);
}

inline bool try_dispatch_mat_scalar_binary_xsimd_uint16(const Mat& src,
                                                        const Scalar& scalar,
                                                        Mat& dst,
                                                        cpu::BinaryKernelOp op,
                                                        bool scalar_first)
{
    return try_dispatch_mat_scalar_binary_xsimd_smallint<ushort>(src, scalar, dst, op, scalar_first, CV_16U);
}

inline bool try_dispatch_mat_scalar_binary_xsimd_int16(const Mat& src,
                                                       const Scalar& scalar,
                                                       Mat& dst,
                                                       cpu::BinaryKernelOp op,
                                                       bool scalar_first)
{
    return try_dispatch_mat_scalar_binary_xsimd_smallint<short>(src, scalar, dst, op, scalar_first, CV_16S);
}

inline bool try_dispatch_mat_scalar_binary_xsimd_uint32(const Mat& src,
                                                        const Scalar& scalar,
                                                        Mat& dst,
                                                        cpu::BinaryKernelOp op,
                                                        bool scalar_first)
{
    if (src.depth() != CV_32U)
    {
        return false;
    }

    const int cn = src.channels();
    if (cn <= 0 || cn > 4)
    {
        return false;
    }

    std::uint32_t scalar_lanes[4] = {0u, 0u, 0u, 0u};
    for (int ch = 0; ch < cn; ++ch)
    {
        scalar_lanes[ch] = saturate_cast<std::uint32_t>(scalar[ch]);
    }

    const bool uniform_scalar = is_uniform_scalar_for_channels(scalar, cn);
    const std::uint32_t scalar_val = scalar_lanes[0];
    const size_t outer = src.dims > 1 ? static_cast<size_t>(src.size.p[0]) : 1;
    const size_t pixel_per_outer = src.dims > 1 ? src.total(1, src.dims) : src.total();
    const size_t row_elements = pixel_per_outer * static_cast<size_t>(cn);
    const size_t src_step0 = src.dims > 1 ? src.step(0) : row_elements * sizeof(std::uint32_t);
    const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : row_elements * sizeof(std::uint32_t);
    const size_t src_outer_stride = src_step0 / sizeof(std::uint32_t);
    const size_t dst_outer_stride = dst_step0 / sizeof(std::uint32_t);

    if (uniform_scalar)
    {
        for (size_t i = 0; i < outer; ++i)
        {
            const void* src_row = src.data + i * src_step0;
            void* dst_row = dst.data + i * dst_step0;
            if (scalar_first)
            {
                cpu::binary_broadcast_xsimd_uint32(
                    op,
                    &scalar_val,
                    0,
                    0,
                    src_row,
                    row_elements,
                    1,
                    dst_row,
                    1,
                    row_elements);
            }
            else
            {
                cpu::binary_broadcast_xsimd_uint32(
                    op,
                    src_row,
                    row_elements,
                    1,
                    &scalar_val,
                    0,
                    0,
                    dst_row,
                    1,
                    row_elements);
            }
        }
    }
    else
    {
        cpu::binary_scalar_channels_xsimd_uint32(op,
                                                 src.data,
                                                 src_outer_stride,
                                                 scalar_lanes,
                                                 cn,
                                                 dst.data,
                                                 dst_outer_stride,
                                                 outer,
                                                 row_elements,
                                                 scalar_first);
    }

    return true;
}

inline bool try_dispatch_mat_scalar_compare_xsimd_fp32(const Mat& src,
                                                       const Scalar& scalar,
                                                       Mat& dst,
                                                       int op,
                                                       bool scalar_first)
{
    if (src.depth() != CV_32F)
    {
        return false;
    }

    const int cn = src.channels();
    if (cn <= 0)
    {
        return false;
    }

    cpu::CompareKernelOp kernel_op;
    if (!to_compare_kernel_op(op, kernel_op))
    {
        return false;
    }

    float scalar_lanes[4] = {0.f, 0.f, 0.f, 0.f};
    for (int ch = 0; ch < cn; ++ch)
    {
        scalar_lanes[ch] = saturate_cast<float>(scalar[ch]);
    }
    const bool uniform_scalar = is_uniform_scalar_for_channels(scalar, cn);
    const float scalar_val = scalar_lanes[0];
    const size_t outer = src.dims > 1 ? static_cast<size_t>(src.size.p[0]) : 1;
    const size_t pixel_per_outer = src.dims > 1 ? src.total(1, src.dims) : src.total();
    const size_t row_elements = pixel_per_outer * static_cast<size_t>(cn);
    const size_t src_step0 = src.dims > 1 ? src.step(0) : row_elements * sizeof(float);
    const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : row_elements * sizeof(uchar);
    const size_t src_outer_stride = src_step0 / sizeof(float);
    const size_t dst_outer_stride = dst_step0 / sizeof(uchar);

    if (uniform_scalar)
    {
        if (scalar_first)
        {
            cpu::compare_broadcast_xsimd(
                kernel_op,
                &scalar_val,
                0,
                0,
                reinterpret_cast<const float*>(src.data),
                src_outer_stride,
                1,
                reinterpret_cast<uchar*>(dst.data),
                dst_outer_stride,
                outer,
                row_elements);
        }
        else
        {
            cpu::compare_broadcast_xsimd(
                kernel_op,
                reinterpret_cast<const float*>(src.data),
                src_outer_stride,
                1,
                &scalar_val,
                0,
                0,
                reinterpret_cast<uchar*>(dst.data),
                dst_outer_stride,
                outer,
                row_elements);
        }
    }
    else
    {
        cpu::compare_scalar_channels_xsimd(kernel_op,
                                           reinterpret_cast<const float*>(src.data),
                                           src_outer_stride,
                                           scalar_lanes,
                                           cn,
                                           reinterpret_cast<uchar*>(dst.data),
                                           dst_outer_stride,
                                           outer,
                                           row_elements,
                                           scalar_first);
    }

    return true;
}

inline bool try_dispatch_mat_scalar_compare_xsimd_fp16(const Mat& src,
                                                       const Scalar& scalar,
                                                       Mat& dst,
                                                       int op,
                                                       bool scalar_first)
{
    if (src.depth() != CV_16F)
    {
        return false;
    }

    const int cn = src.channels();
    if (cn <= 0 || cn > 4)
    {
        return false;
    }

    cpu::CompareKernelOp kernel_op;
    if (!to_compare_kernel_op(op, kernel_op))
    {
        return false;
    }

    float scalar_lanes[4] = {0.f, 0.f, 0.f, 0.f};
    for (int ch = 0; ch < cn; ++ch)
    {
        scalar_lanes[ch] = saturate_cast<float>(scalar[ch]);
    }
    const bool uniform_scalar = is_uniform_scalar_for_channels(scalar, cn);
    const hfloat scalar_val = saturate_cast<hfloat>(scalar_lanes[0]);
    const size_t outer = src.dims > 1 ? static_cast<size_t>(src.size.p[0]) : 1;
    const size_t pixel_per_outer = src.dims > 1 ? src.total(1, src.dims) : src.total();
    const size_t row_elements = pixel_per_outer * static_cast<size_t>(cn);
    const size_t src_step0 = src.dims > 1 ? src.step(0) : row_elements * sizeof(hfloat);
    const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : row_elements * sizeof(uchar);
    const size_t src_outer_stride = src_step0 / sizeof(hfloat);
    const size_t dst_outer_stride = dst_step0 / sizeof(uchar);

    if (uniform_scalar)
    {
        if (scalar_first)
        {
            cpu::compare_broadcast_xsimd_hfloat(
                kernel_op,
                &scalar_val,
                0,
                0,
                src.data,
                src_outer_stride,
                1,
                reinterpret_cast<uchar*>(dst.data),
                dst_outer_stride,
                outer,
                row_elements);
        }
        else
        {
            cpu::compare_broadcast_xsimd_hfloat(
                kernel_op,
                src.data,
                src_outer_stride,
                1,
                &scalar_val,
                0,
                0,
                reinterpret_cast<uchar*>(dst.data),
                dst_outer_stride,
                outer,
                row_elements);
        }
    }
    else
    {
        cpu::compare_scalar_channels_xsimd_hfloat(kernel_op,
                                                  src.data,
                                                  src_outer_stride,
                                                  scalar_lanes,
                                                  cn,
                                                  reinterpret_cast<uchar*>(dst.data),
                                                  dst_outer_stride,
                                                  outer,
                                                  row_elements,
                                                  scalar_first);
    }

    return true;
}

template<typename T>
inline bool try_dispatch_mat_scalar_compare_xsimd_int(const Mat& src,
                                                      const Scalar& scalar,
                                                      Mat& dst,
                                                      int op,
                                                      bool scalar_first,
                                                      int expected_depth)
{
    if (src.depth() != expected_depth)
    {
        return false;
    }

    const int cn = src.channels();
    if (cn <= 0 || cn > 4)
    {
        return false;
    }

    cpu::CompareKernelOp kernel_op;
    if (!to_compare_kernel_op(op, kernel_op))
    {
        return false;
    }

    T scalar_lanes[4] = {T(0), T(0), T(0), T(0)};
    for (int ch = 0; ch < cn; ++ch)
    {
        scalar_lanes[ch] = saturate_cast<T>(scalar[ch]);
    }
    const bool uniform_scalar = is_uniform_scalar_for_channels(scalar, cn);
    const T scalar_val = scalar_lanes[0];
    const size_t outer = src.dims > 1 ? static_cast<size_t>(src.size.p[0]) : 1;
    const size_t pixel_per_outer = src.dims > 1 ? src.total(1, src.dims) : src.total();
    const size_t row_elements = pixel_per_outer * static_cast<size_t>(cn);
    const size_t src_step0 = src.dims > 1 ? src.step(0) : row_elements * sizeof(T);
    const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : row_elements * sizeof(uchar);
    const size_t src_outer_stride = src_step0 / sizeof(T);
    const size_t dst_outer_stride = dst_step0 / sizeof(uchar);

    if (uniform_scalar)
    {
        for (size_t i = 0; i < outer; ++i)
        {
            const void* src_row = src.data + i * src_step0;
            uchar* dst_row = reinterpret_cast<uchar*>(dst.data + i * dst_step0);
            if (expected_depth == CV_8U)
            {
                cpu::compare_broadcast_xsimd_uint8(kernel_op,
                                                   scalar_first ? static_cast<const void*>(&scalar_val) : src_row,
                                                   scalar_first ? 0 : row_elements,
                                                   scalar_first ? 0 : 1,
                                                   scalar_first ? src_row : static_cast<const void*>(&scalar_val),
                                                   scalar_first ? row_elements : 0,
                                                   scalar_first ? 1 : 0,
                                                   dst_row, row_elements, 1, row_elements);
            }
            else if (expected_depth == CV_8S)
            {
                cpu::compare_broadcast_xsimd_int8(kernel_op,
                                                  scalar_first ? static_cast<const void*>(&scalar_val) : src_row,
                                                  scalar_first ? 0 : row_elements,
                                                  scalar_first ? 0 : 1,
                                                  scalar_first ? src_row : static_cast<const void*>(&scalar_val),
                                                  scalar_first ? row_elements : 0,
                                                  scalar_first ? 1 : 0,
                                                  dst_row, row_elements, 1, row_elements);
            }
            else if (expected_depth == CV_16U)
            {
                cpu::compare_broadcast_xsimd_uint16(kernel_op,
                                                    scalar_first ? static_cast<const void*>(&scalar_val) : src_row,
                                                    scalar_first ? 0 : row_elements,
                                                    scalar_first ? 0 : 1,
                                                    scalar_first ? src_row : static_cast<const void*>(&scalar_val),
                                                    scalar_first ? row_elements : 0,
                                                    scalar_first ? 1 : 0,
                                                    dst_row, row_elements, 1, row_elements);
            }
            else if (expected_depth == CV_16S)
            {
                cpu::compare_broadcast_xsimd_int16(kernel_op,
                                                   scalar_first ? static_cast<const void*>(&scalar_val) : src_row,
                                                   scalar_first ? 0 : row_elements,
                                                   scalar_first ? 0 : 1,
                                                   scalar_first ? src_row : static_cast<const void*>(&scalar_val),
                                                   scalar_first ? row_elements : 0,
                                                   scalar_first ? 1 : 0,
                                                   dst_row, row_elements, 1, row_elements);
            }
            else if (expected_depth == CV_32S)
            {
                cpu::compare_broadcast_xsimd_int32(kernel_op,
                                                   scalar_first ? static_cast<const void*>(&scalar_val) : src_row,
                                                   scalar_first ? 0 : row_elements,
                                                   scalar_first ? 0 : 1,
                                                   scalar_first ? src_row : static_cast<const void*>(&scalar_val),
                                                   scalar_first ? row_elements : 0,
                                                   scalar_first ? 1 : 0,
                                                   dst_row, row_elements, 1, row_elements);
            }
            else
            {
                cpu::compare_broadcast_xsimd_uint32(kernel_op,
                                                    scalar_first ? static_cast<const void*>(&scalar_val) : src_row,
                                                    scalar_first ? 0 : row_elements,
                                                    scalar_first ? 0 : 1,
                                                    scalar_first ? src_row : static_cast<const void*>(&scalar_val),
                                                    scalar_first ? row_elements : 0,
                                                    scalar_first ? 1 : 0,
                                                    dst_row, row_elements, 1, row_elements);
            }
        }
    }
    else
    {
        if (expected_depth == CV_8U)
        {
            cpu::compare_scalar_channels_xsimd_uint8(kernel_op, src.data, src_outer_stride,
                                                     reinterpret_cast<const std::uint8_t*>(scalar_lanes), cn,
                                                     reinterpret_cast<uchar*>(dst.data), dst_outer_stride,
                                                     outer, row_elements, scalar_first);
        }
        else if (expected_depth == CV_8S)
        {
            cpu::compare_scalar_channels_xsimd_int8(kernel_op, src.data, src_outer_stride,
                                                    reinterpret_cast<const std::int8_t*>(scalar_lanes), cn,
                                                    reinterpret_cast<uchar*>(dst.data), dst_outer_stride,
                                                    outer, row_elements, scalar_first);
        }
        else if (expected_depth == CV_16U)
        {
            cpu::compare_scalar_channels_xsimd_uint16(kernel_op, src.data, src_outer_stride,
                                                      reinterpret_cast<const std::uint16_t*>(scalar_lanes), cn,
                                                      reinterpret_cast<uchar*>(dst.data), dst_outer_stride,
                                                      outer, row_elements, scalar_first);
        }
        else if (expected_depth == CV_16S)
        {
            cpu::compare_scalar_channels_xsimd_int16(kernel_op, src.data, src_outer_stride,
                                                     reinterpret_cast<const std::int16_t*>(scalar_lanes), cn,
                                                     reinterpret_cast<uchar*>(dst.data), dst_outer_stride,
                                                     outer, row_elements, scalar_first);
        }
        else if (expected_depth == CV_32S)
        {
            cpu::compare_scalar_channels_xsimd_int32(kernel_op, src.data, src_outer_stride,
                                                     reinterpret_cast<const std::int32_t*>(scalar_lanes), cn,
                                                     reinterpret_cast<uchar*>(dst.data), dst_outer_stride,
                                                     outer, row_elements, scalar_first);
        }
        else
        {
            cpu::compare_scalar_channels_xsimd_uint32(kernel_op, src.data, src_outer_stride,
                                                      reinterpret_cast<const std::uint32_t*>(scalar_lanes), cn,
                                                      reinterpret_cast<uchar*>(dst.data), dst_outer_stride,
                                                      outer, row_elements, scalar_first);
        }
    }

    return true;
}

template<typename Op>
void dispatch_mat_mat_binary_arith(const Mat& a,
                                   const Mat& b,
                                   Mat& dst,
                                   Op op,
                                   cpu::BinaryKernelOp xsimd_op,
                                   const char* fn_name)
{
    ensure_same_type_and_shape(a, b, fn_name);
    ensure_binary_dst_like_src(a, dst, fn_name);

    const cpu::DispatchMode mode = cpu::dispatch_mode();
    if (mode != cpu::DispatchMode::ScalarOnly &&
        try_dispatch_mat_mat_binary_xsimd(a, b, dst, xsimd_op))
    {
        cpu::set_last_dispatch_tag(cpu::DispatchTag::XSimd);
        return;
    }

    if (mode == cpu::DispatchMode::XSimdOnly)
    {
        CV_Error_(Error::StsNotImplemented,
                  ("%s xsimd-only mode requested but no xsimd path is available", fn_name));
    }

    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
    dispatch_mat_mat_binary_impl_by_depth(a, b, dst, op, fn_name);
}

template<typename Op>
void dispatch_mat_mat_binary(const Mat& a, const Mat& b, Mat& dst, Op op, const char* fn_name)
{
    ensure_same_type_and_shape(a, b, fn_name);
    ensure_binary_dst_like_src(a, dst, fn_name);
    dispatch_mat_mat_binary_impl_by_depth(a, b, dst, op, fn_name);
}

template<typename T>
void apply_unary_negate_impl(const Mat& src, Mat& dst)
{
    const int cn = src.channels();
    const size_t outer = src.dims > 1 ? static_cast<size_t>(src.size.p[0]) : 1;
    const size_t pixel_per_outer = src.dims > 1 ? src.total(1, src.dims) : src.total();
    const size_t src_step0 = src.dims > 1 ? src.step(0) : pixel_per_outer * src.elemSize();
    const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : pixel_per_outer * dst.elemSize();

    for (size_t i = 0; i < outer; ++i)
    {
        const T* src_row = reinterpret_cast<const T*>(src.data + i * src_step0);
        T* dst_row = reinterpret_cast<T*>(dst.data + i * dst_step0);

        for (size_t p = 0; p < pixel_per_outer; ++p)
        {
            const size_t base = p * static_cast<size_t>(cn);
            for (int ch = 0; ch < cn; ++ch)
            {
                const size_t idx = base + static_cast<size_t>(ch);
                if constexpr (std::is_same<T, hfloat>::value)
                {
                    dst_row[idx] = saturate_cast<T>(-static_cast<float>(src_row[idx]));
                }
                else
                {
                    dst_row[idx] = saturate_cast<T>(-src_row[idx]);
                }
            }
        }
    }
}

void dispatch_unary_negate(const Mat& src, Mat& dst, const char* fn_name)
{
    ensure_binary_dst_like_src(src, dst, fn_name);

    switch (src.depth())
    {
        case CV_8U:
            apply_unary_negate_impl<uchar>(src, dst);
            break;
        case CV_8S:
            apply_unary_negate_impl<schar>(src, dst);
            break;
        case CV_16U:
            apply_unary_negate_impl<ushort>(src, dst);
            break;
        case CV_16S:
            apply_unary_negate_impl<short>(src, dst);
            break;
        case CV_32S:
            apply_unary_negate_impl<int>(src, dst);
            break;
        case CV_32U:
            apply_unary_negate_impl<uint>(src, dst);
            break;
        case CV_32F:
            apply_unary_negate_impl<float>(src, dst);
            break;
        case CV_16F:
            apply_unary_negate_impl<hfloat>(src, dst);
            break;
        default:
            CV_Error_(Error::StsNotImplemented,
                      ("%s supports depth in [CV_8U..CV_16F], depth=%d", fn_name, src.depth()));
    }
}

template<typename T>
void apply_add_weighted_impl(const Mat& a, double alpha, const Mat& b, double beta, Mat& dst)
{
    const int cn = a.channels();
    const size_t outer = a.dims > 1 ? static_cast<size_t>(a.size.p[0]) : 1;
    const size_t pixel_per_outer = a.dims > 1 ? a.total(1, a.dims) : a.total();
    const size_t a_step0 = a.dims > 1 ? a.step(0) : pixel_per_outer * a.elemSize();
    const size_t b_step0 = b.dims > 1 ? b.step(0) : pixel_per_outer * b.elemSize();
    const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : pixel_per_outer * dst.elemSize();

    for (size_t i = 0; i < outer; ++i)
    {
        const T* a_row = reinterpret_cast<const T*>(a.data + i * a_step0);
        const T* b_row = reinterpret_cast<const T*>(b.data + i * b_step0);
        T* dst_row = reinterpret_cast<T*>(dst.data + i * dst_step0);

        for (size_t p = 0; p < pixel_per_outer; ++p)
        {
            const size_t base = p * static_cast<size_t>(cn);
            for (int ch = 0; ch < cn; ++ch)
            {
                const size_t idx = base + static_cast<size_t>(ch);
                const double value = static_cast<double>(a_row[idx]) * alpha +
                                     static_cast<double>(b_row[idx]) * beta;
                dst_row[idx] = saturate_cast<T>(value);
            }
        }
    }
}

void dispatch_add_weighted(const Mat& a, double alpha, const Mat& b, double beta, Mat& dst, const char* fn_name)
{
    ensure_same_type_and_shape(a, b, fn_name);
    ensure_binary_dst_like_src(a, dst, fn_name);

    switch (a.depth())
    {
        case CV_8U:
            apply_add_weighted_impl<uchar>(a, alpha, b, beta, dst);
            break;
        case CV_8S:
            apply_add_weighted_impl<schar>(a, alpha, b, beta, dst);
            break;
        case CV_16U:
            apply_add_weighted_impl<ushort>(a, alpha, b, beta, dst);
            break;
        case CV_16S:
            apply_add_weighted_impl<short>(a, alpha, b, beta, dst);
            break;
        case CV_32S:
            apply_add_weighted_impl<int>(a, alpha, b, beta, dst);
            break;
        case CV_32U:
            apply_add_weighted_impl<uint>(a, alpha, b, beta, dst);
            break;
        case CV_32F:
            apply_add_weighted_impl<float>(a, alpha, b, beta, dst);
            break;
        case CV_16F:
            apply_add_weighted_impl<hfloat>(a, alpha, b, beta, dst);
            break;
        default:
            CV_Error_(Error::StsNotImplemented,
                      ("%s supports depth in [CV_8U..CV_64F], depth=%d", fn_name, a.depth()));
    }
}

template<typename T, typename Op>
void apply_mat_scalar_binary_impl(const Mat& src,
                                  const Scalar& scalar,
                                  Mat& dst,
                                  Op op,
                                  bool scalar_first)
{
    const int cn = src.channels();
    T lane[4];
    for (int ch = 0; ch < cn; ++ch)
    {
        lane[ch] = saturate_cast<T>(scalar[ch]);
    }

    const size_t outer = src.dims > 1 ? static_cast<size_t>(src.size.p[0]) : 1;
    const size_t pixel_per_outer = src.dims > 1 ? src.total(1, src.dims) : src.total();
    const size_t src_step0 = src.dims > 1 ? src.step(0) : pixel_per_outer * src.elemSize();
    const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : pixel_per_outer * dst.elemSize();

    for (size_t i = 0; i < outer; ++i)
    {
        const T* src_row = reinterpret_cast<const T*>(src.data + i * src_step0);
        T* dst_row = reinterpret_cast<T*>(dst.data + i * dst_step0);

        for (size_t p = 0; p < pixel_per_outer; ++p)
        {
            const size_t base = p * static_cast<size_t>(cn);
            for (int ch = 0; ch < cn; ++ch)
            {
                const size_t idx = base + static_cast<size_t>(ch);
                const auto value = scalar_first ? op(lane[ch], src_row[idx]) : op(src_row[idx], lane[ch]);
                dst_row[idx] = saturate_cast<T>(value);
            }
        }
    }
}

template<typename Op>
void dispatch_mat_scalar_binary_by_depth(const Mat& src,
                                         const Scalar& scalar,
                                         Mat& dst,
                                         Op op,
                                         bool scalar_first,
                                         const char* fn_name)
{
    switch (src.depth())
    {
        case CV_8U:
            apply_mat_scalar_binary_impl<uchar>(src, scalar, dst, op, scalar_first);
            break;
        case CV_8S:
            apply_mat_scalar_binary_impl<schar>(src, scalar, dst, op, scalar_first);
            break;
        case CV_16U:
            apply_mat_scalar_binary_impl<ushort>(src, scalar, dst, op, scalar_first);
            break;
        case CV_16S:
            apply_mat_scalar_binary_impl<short>(src, scalar, dst, op, scalar_first);
            break;
        case CV_32S:
            apply_mat_scalar_binary_impl<int>(src, scalar, dst, op, scalar_first);
            break;
        case CV_32U:
            apply_mat_scalar_binary_impl<uint>(src, scalar, dst, op, scalar_first);
            break;
        case CV_32F:
            apply_mat_scalar_binary_impl<float>(src, scalar, dst, op, scalar_first);
            break;
        case CV_16F:
            apply_mat_scalar_binary_impl<hfloat>(src, scalar, dst, op, scalar_first);
            break;
        default:
            CV_Error_(Error::StsNotImplemented,
                      ("%s supports depth in [CV_8U..CV_16F], depth=%d", fn_name, src.depth()));
    }
}

template<typename Op>
void dispatch_mat_scalar_binary(const Mat& src,
                                const Scalar& scalar,
                                Mat& dst,
                                Op op,
                                bool scalar_first,
                                const char* fn_name)
{
    check_scalar_channel_bound(src, fn_name);
    ensure_binary_dst_like_src(src, dst, fn_name);
    dispatch_mat_scalar_binary_by_depth(src, scalar, dst, op, scalar_first, fn_name);
}

template<typename Op>
void dispatch_mat_scalar_binary_arith(const Mat& src,
                                      const Scalar& scalar,
                                      Mat& dst,
                                      Op op,
                                      cpu::BinaryKernelOp xsimd_op,
                                      bool scalar_first,
                                      const char* fn_name)
{
    check_scalar_channel_bound(src, fn_name);
    ensure_binary_dst_like_src(src, dst, fn_name);

    const cpu::DispatchMode mode = cpu::dispatch_mode();
    if (mode != cpu::DispatchMode::ScalarOnly &&
        (try_dispatch_mat_scalar_binary_xsimd_fp32(src, scalar, dst, xsimd_op, scalar_first) ||
         try_dispatch_mat_scalar_binary_xsimd_fp16(src, scalar, dst, xsimd_op, scalar_first) ||
         try_dispatch_mat_scalar_binary_xsimd_uint8(src, scalar, dst, xsimd_op, scalar_first) ||
         try_dispatch_mat_scalar_binary_xsimd_int8(src, scalar, dst, xsimd_op, scalar_first) ||
         try_dispatch_mat_scalar_binary_xsimd_uint16(src, scalar, dst, xsimd_op, scalar_first) ||
         try_dispatch_mat_scalar_binary_xsimd_int16(src, scalar, dst, xsimd_op, scalar_first) ||
         try_dispatch_mat_scalar_binary_xsimd_int32(src, scalar, dst, xsimd_op, scalar_first) ||
         try_dispatch_mat_scalar_binary_xsimd_uint32(src, scalar, dst, xsimd_op, scalar_first)))
    {
        cpu::set_last_dispatch_tag(cpu::DispatchTag::XSimd);
        return;
    }

    if (mode == cpu::DispatchMode::XSimdOnly)
    {
        CV_Error_(Error::StsNotImplemented,
                  ("%s xsimd-only mode requested but no xsimd path is available", fn_name));
    }

    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
    dispatch_mat_scalar_binary_by_depth(src, scalar, dst, op, scalar_first, fn_name);
}

template<typename T>
bool eval_compare_op(T lhs, T rhs, int op)
{
    switch (op)
    {
        case CV_CMP_EQ: return lhs == rhs;
        case CV_CMP_GT: return lhs > rhs;
        case CV_CMP_GE: return lhs >= rhs;
        case CV_CMP_LT: return lhs < rhs;
        case CV_CMP_LE: return lhs <= rhs;
        case CV_CMP_NE: return lhs != rhs;
        default:
            CV_Error_(Error::StsBadArg, ("Unsupported compare op=%d", op));
            return false;
    }
}

template<typename T>
void apply_mat_scalar_compare_impl(const Mat& src,
                                   const Scalar& scalar,
                                   Mat& dst,
                                   int op,
                                   bool scalar_first)
{
    const int cn = src.channels();
    T lane[4];
    for (int ch = 0; ch < cn; ++ch)
    {
        lane[ch] = saturate_cast<T>(scalar[ch]);
    }

    const size_t outer = src.dims > 1 ? static_cast<size_t>(src.size.p[0]) : 1;
    const size_t pixel_per_outer = src.dims > 1 ? src.total(1, src.dims) : src.total();
    const size_t src_step0 = src.dims > 1 ? src.step(0) : pixel_per_outer * src.elemSize();
    const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : pixel_per_outer * dst.elemSize();

    for (size_t i = 0; i < outer; ++i)
    {
        const T* src_row = reinterpret_cast<const T*>(src.data + i * src_step0);
        uchar* dst_row = reinterpret_cast<uchar*>(dst.data + i * dst_step0);

        for (size_t p = 0; p < pixel_per_outer; ++p)
        {
            const size_t base = p * static_cast<size_t>(cn);
            for (int ch = 0; ch < cn; ++ch)
            {
                const size_t idx = base + static_cast<size_t>(ch);
                const bool flag = scalar_first ?
                                  eval_compare_op<T>(lane[ch], src_row[idx], op) :
                                  eval_compare_op<T>(src_row[idx], lane[ch], op);
                dst_row[idx] = flag ? 255 : 0;
            }
        }
    }
}

void dispatch_mat_scalar_compare(const Mat& src,
                                 const Scalar& scalar,
                                 Mat& dst,
                                 int op,
                                 bool scalar_first,
                                 const char* fn_name)
{
    check_scalar_channel_bound(src, fn_name);
    ensure_compare_dst_like_src(src, dst, fn_name);

    const cpu::DispatchMode mode = cpu::dispatch_mode();
    if (mode != cpu::DispatchMode::ScalarOnly &&
        (try_dispatch_mat_scalar_compare_xsimd_fp32(src, scalar, dst, op, scalar_first) ||
         try_dispatch_mat_scalar_compare_xsimd_fp16(src, scalar, dst, op, scalar_first) ||
         try_dispatch_mat_scalar_compare_xsimd_int<uchar>(src, scalar, dst, op, scalar_first, CV_8U) ||
         try_dispatch_mat_scalar_compare_xsimd_int<schar>(src, scalar, dst, op, scalar_first, CV_8S) ||
         try_dispatch_mat_scalar_compare_xsimd_int<ushort>(src, scalar, dst, op, scalar_first, CV_16U) ||
         try_dispatch_mat_scalar_compare_xsimd_int<short>(src, scalar, dst, op, scalar_first, CV_16S) ||
         try_dispatch_mat_scalar_compare_xsimd_int<int>(src, scalar, dst, op, scalar_first, CV_32S) ||
         try_dispatch_mat_scalar_compare_xsimd_int<uint>(src, scalar, dst, op, scalar_first, CV_32U)))
    {
        cpu::set_last_dispatch_tag(cpu::DispatchTag::XSimd);
        return;
    }

    if (mode == cpu::DispatchMode::XSimdOnly)
    {
        CV_Error_(Error::StsNotImplemented,
                  ("%s xsimd-only mode requested but no xsimd path is available", fn_name));
    }

    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);

    switch (src.depth())
    {
        case CV_8U:
            apply_mat_scalar_compare_impl<uchar>(src, scalar, dst, op, scalar_first);
            break;
        case CV_8S:
            apply_mat_scalar_compare_impl<schar>(src, scalar, dst, op, scalar_first);
            break;
        case CV_16U:
            apply_mat_scalar_compare_impl<ushort>(src, scalar, dst, op, scalar_first);
            break;
        case CV_16S:
            apply_mat_scalar_compare_impl<short>(src, scalar, dst, op, scalar_first);
            break;
        case CV_32S:
            apply_mat_scalar_compare_impl<int>(src, scalar, dst, op, scalar_first);
            break;
        case CV_32U:
            apply_mat_scalar_compare_impl<uint>(src, scalar, dst, op, scalar_first);
            break;
        case CV_32F:
            apply_mat_scalar_compare_impl<float>(src, scalar, dst, op, scalar_first);
            break;
        case CV_16F:
            apply_mat_scalar_compare_impl<hfloat>(src, scalar, dst, op, scalar_first);
            break;
        default:
            CV_Error_(Error::StsNotImplemented,
                      ("%s supports depth in [CV_8U..CV_16F], depth=%d", fn_name, src.depth()));
    }
}

template<typename T>
void apply_mat_mat_compare_impl(const Mat& a, const Mat& b, Mat& dst, int op)
{
    const int cn = a.channels();
    const size_t outer = a.dims > 1 ? static_cast<size_t>(a.size.p[0]) : 1;
    const size_t pixel_per_outer = a.dims > 1 ? a.total(1, a.dims) : a.total();
    const size_t a_step0 = a.dims > 1 ? a.step(0) : pixel_per_outer * a.elemSize();
    const size_t b_step0 = b.dims > 1 ? b.step(0) : pixel_per_outer * b.elemSize();
    const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : pixel_per_outer * dst.elemSize();

    for (size_t i = 0; i < outer; ++i)
    {
        const T* a_row = reinterpret_cast<const T*>(a.data + i * a_step0);
        const T* b_row = reinterpret_cast<const T*>(b.data + i * b_step0);
        uchar* dst_row = reinterpret_cast<uchar*>(dst.data + i * dst_step0);

        for (size_t p = 0; p < pixel_per_outer; ++p)
        {
            const size_t base = p * static_cast<size_t>(cn);
            for (int ch = 0; ch < cn; ++ch)
            {
                const size_t idx = base + static_cast<size_t>(ch);
                const bool flag = eval_compare_op<T>(a_row[idx], b_row[idx], op);
                dst_row[idx] = flag ? 255 : 0;
            }
        }
    }
}

void dispatch_mat_mat_compare(const Mat& a, const Mat& b, Mat& dst, int op, const char* fn_name)
{
    ensure_same_type_and_shape(a, b, fn_name);
    ensure_compare_dst_like_src(a, dst, fn_name);

    const cpu::DispatchMode mode = cpu::dispatch_mode();
    if (mode != cpu::DispatchMode::ScalarOnly &&
        (try_dispatch_mat_mat_compare_xsimd_fp32(a, b, dst, op) ||
         try_dispatch_mat_mat_compare_xsimd_fp16(a, b, dst, op) ||
         try_dispatch_mat_mat_compare_xsimd_int(a, b, dst, op)))
    {
        cpu::set_last_dispatch_tag(cpu::DispatchTag::XSimd);
        return;
    }

    if (mode == cpu::DispatchMode::XSimdOnly)
    {
        CV_Error_(Error::StsNotImplemented,
                  ("%s xsimd-only mode requested but no xsimd path is available", fn_name));
    }

    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);

    switch (a.depth())
    {
        case CV_8U:
            apply_mat_mat_compare_impl<uchar>(a, b, dst, op);
            break;
        case CV_8S:
            apply_mat_mat_compare_impl<schar>(a, b, dst, op);
            break;
        case CV_16U:
            apply_mat_mat_compare_impl<ushort>(a, b, dst, op);
            break;
        case CV_16S:
            apply_mat_mat_compare_impl<short>(a, b, dst, op);
            break;
        case CV_32S:
            apply_mat_mat_compare_impl<int>(a, b, dst, op);
            break;
        case CV_32U:
            apply_mat_mat_compare_impl<uint>(a, b, dst, op);
            break;
        case CV_32F:
            apply_mat_mat_compare_impl<float>(a, b, dst, op);
            break;
        case CV_16F:
            apply_mat_mat_compare_impl<hfloat>(a, b, dst, op);
            break;
        default:
            CV_Error_(Error::StsNotImplemented,
                      ("%s supports depth in [CV_8U..CV_64F], depth=%d", fn_name, a.depth()));
    }
}

template<typename Op>
void dispatch_mat_mat_integral_binary(const Mat& a, const Mat& b, Mat& dst, Op op, const char* fn_name)
{
    ensure_same_type_and_shape(a, b, fn_name);
    ensure_binary_dst_like_src(a, dst, fn_name);

    switch (a.depth())
    {
        case CV_8U:
            apply_mat_mat_binary_impl<uchar>(a, b, dst, op);
            break;
        case CV_8S:
            apply_mat_mat_binary_impl<schar>(a, b, dst, op);
            break;
        case CV_16U:
            apply_mat_mat_binary_impl<ushort>(a, b, dst, op);
            break;
        case CV_16S:
            apply_mat_mat_binary_impl<short>(a, b, dst, op);
            break;
        case CV_32S:
            apply_mat_mat_binary_impl<int>(a, b, dst, op);
            break;
        case CV_32U:
            apply_mat_mat_binary_impl<uint>(a, b, dst, op);
            break;
        default:
            CV_Error_(Error::StsNotImplemented,
                      ("%s supports integral depth only, depth=%d", fn_name, a.depth()));
    }
}

template<typename T>
inline T integer_mod_divisor_sign(T lhs, T rhs)
{
    if (rhs == 0)
    {
        return T(0);
    }

    T rem = static_cast<T>(lhs % rhs);
    if (rem != 0 && ((rem > 0) != (rhs > 0)))
    {
        rem = static_cast<T>(rem + rhs);
    }
    return rem;
}

template<typename T>
inline T integer_bitshift_value(T lhs, T rhs)
{
    using U = typename std::make_unsigned<T>::type;

    const int bit_count = static_cast<int>(sizeof(T) * 8);
    const long long s = static_cast<long long>(rhs);
    if (s >= 0)
    {
        if (s >= bit_count)
        {
            return T(0);
        }
        const U shifted = static_cast<U>(static_cast<U>(lhs) << static_cast<int>(s));
        return static_cast<T>(shifted);
    }

    const long long rs = -s;
    if (rs >= bit_count)
    {
        return T(0);
    }
    const U shifted = static_cast<U>(static_cast<U>(lhs) >> static_cast<int>(rs));
    return static_cast<T>(shifted);
}

template<typename T>
inline auto floating_mod_dividend_sign(T lhs, T rhs)
{
    if constexpr (std::is_integral<T>::value)
    {
        if (rhs == 0)
        {
            return T(0);
        }
        return static_cast<T>(lhs % rhs);
    }
    else if constexpr (std::is_same<T, hfloat>::value)
    {
        const float r = static_cast<float>(rhs);
        if (r == 0.0f)
        {
            return 0.0f;
        }
        return std::fmod(static_cast<float>(lhs), r);
    }
    else
    {
        if (rhs == static_cast<T>(0))
        {
            return static_cast<T>(0);
        }
        return static_cast<T>(std::fmod(static_cast<double>(lhs), static_cast<double>(rhs)));
    }
}

void validate_merge_sources(const Mat* src,
                            size_t nsrc,
                            int& depth,
                            int& dims,
                            MatShape& shape,
                            int& total_channels,
                            const char* fn_name)
{
    if (src == nullptr)
    {
        CV_Error_(Error::StsBadArg, ("%s expects non-null src array", fn_name));
    }
    if (nsrc == 0)
    {
        CV_Error_(Error::StsBadArg, ("%s expects at least one source Mat", fn_name));
    }

    const Mat& ref = src[0];
    if (ref.empty())
    {
        CV_Error_(Error::StsBadArg, ("%s src[0] must be non-empty", fn_name));
    }

    depth = ref.depth();
    dims = ref.dims;
    shape = ref.shape();
    total_channels = 0;

    for (size_t i = 0; i < nsrc; ++i)
    {
        const Mat& plane = src[i];
        if (plane.empty())
        {
            CV_Error_(Error::StsBadArg, ("%s src[%zu] must be non-empty", fn_name, i));
        }
        if (plane.depth() != depth)
        {
            CV_Error_(Error::StsBadType,
                      ("%s depth mismatch at src[%zu], expected=%d actual=%d",
                       fn_name,
                       i,
                       depth,
                       plane.depth()));
        }
        if (plane.dims != dims || plane.shape() != shape)
        {
            CV_Error_(Error::StsUnmatchedSizes, ("%s shape mismatch at src[%zu]", fn_name, i));
        }

        const int cn = plane.channels();
        if (cn <= 0)
        {
            CV_Error_(Error::StsBadArg, ("%s src[%zu] has invalid channels=%d", fn_name, i, cn));
        }
        if (total_channels > CV_CN_MAX - cn)
        {
            CV_Error_(Error::StsOutOfRange, ("%s total channels exceed CV_CN_MAX", fn_name));
        }
        total_channels += cn;
    }
}

void ensure_merge_dst(const Mat& ref,
                      int depth,
                      int dims,
                      const MatShape& shape,
                      int total_channels,
                      Mat& dst,
                      const char* fn_name)
{
    const int dst_type = CV_MAKETYPE(depth, total_channels);
    if (dst.empty())
    {
        dst.create(dims, ref.size.p, dst_type);
        return;
    }

    if (dst.type() != dst_type || dst.shape() != shape)
    {
        CV_Error_(Error::StsBadType,
                  ("%s dst mismatch, expected_type=%d actual_type=%d",
                   fn_name,
                   dst_type,
                   dst.type()));
    }
}

void copy_merge_channels(const Mat* src, size_t nsrc, Mat& dst)
{
    const Mat& ref = src[0];
    const size_t outer = ref.dims > 1 ? static_cast<size_t>(ref.size.p[0]) : 1;
    const size_t pixel_per_outer = ref.dims > 1 ? ref.total(1, ref.dims) : ref.total();
    const size_t dst_pixel_bytes = dst.elemSize();
    const size_t dst_step0 = dst.dims > 1 ? dst.step(0) : pixel_per_outer * dst_pixel_bytes;

    std::vector<size_t> src_step0(nsrc, 0);
    std::vector<size_t> src_pixel_bytes(nsrc, 0);
    std::vector<const uchar*> src_rows(nsrc, nullptr);
    for (size_t i = 0; i < nsrc; ++i)
    {
        src_pixel_bytes[i] = src[i].elemSize();
        src_step0[i] = src[i].dims > 1 ? src[i].step(0) : pixel_per_outer * src_pixel_bytes[i];
    }

    for (size_t row = 0; row < outer; ++row)
    {
        uchar* dst_row = dst.data + row * dst_step0;
        for (size_t i = 0; i < nsrc; ++i)
        {
            src_rows[i] = src[i].data + row * src_step0[i];
        }

        for (size_t p = 0; p < pixel_per_outer; ++p)
        {
            uchar* dst_px = dst_row + p * dst_pixel_bytes;
            size_t dst_off = 0;
            for (size_t i = 0; i < nsrc; ++i)
            {
                const size_t bytes = src_pixel_bytes[i];
                const uchar* src_px = src_rows[i] + p * bytes;
                std::memcpy(dst_px + dst_off, src_px, bytes);
                dst_off += bytes;
            }
        }
    }
}

void ensure_split_dst(const Mat& src, Mat* dst, const char* fn_name)
{
    if (dst == nullptr)
    {
        CV_Error_(Error::StsBadArg, ("%s expects non-null dst array", fn_name));
    }

    const int cn = src.channels();
    const int dst_type = CV_MAKETYPE(src.depth(), 1);
    const MatShape src_shape = src.shape();
    for (int ch = 0; ch < cn; ++ch)
    {
        Mat& out = dst[ch];
        if (out.empty())
        {
            out.create(src.dims, src.size.p, dst_type);
            continue;
        }

        if (out.type() != dst_type || out.shape() != src_shape)
        {
            CV_Error_(Error::StsBadType,
                      ("%s dst[%d] mismatch, expected_type=%d actual_type=%d",
                       fn_name,
                       ch,
                       dst_type,
                       out.type()));
        }
    }
}

void copy_split_channels(const Mat& src, Mat* dst)
{
    const int cn = src.channels();
    const size_t src_elem_size1 = src.elemSize1();
    const size_t src_pixel_bytes = src.elemSize();
    const size_t outer = src.dims > 1 ? static_cast<size_t>(src.size.p[0]) : 1;
    const size_t pixel_per_outer = src.dims > 1 ? src.total(1, src.dims) : src.total();
    const size_t src_step0 = src.dims > 1 ? src.step(0) : pixel_per_outer * src_pixel_bytes;

    std::vector<size_t> dst_step0(static_cast<size_t>(cn), 0);
    std::vector<uchar*> dst_rows(static_cast<size_t>(cn), nullptr);
    for (int ch = 0; ch < cn; ++ch)
    {
        dst_step0[static_cast<size_t>(ch)] =
            dst[ch].dims > 1 ? dst[ch].step(0) : pixel_per_outer * src_elem_size1;
    }

    for (size_t row = 0; row < outer; ++row)
    {
        const uchar* src_row = src.data + row * src_step0;
        for (int ch = 0; ch < cn; ++ch)
        {
            dst_rows[static_cast<size_t>(ch)] = dst[ch].data + row * dst_step0[static_cast<size_t>(ch)];
        }

        for (size_t p = 0; p < pixel_per_outer; ++p)
        {
            const uchar* src_px = src_row + p * src_pixel_bytes;
            for (int ch = 0; ch < cn; ++ch)
            {
                uchar* dst_px = dst_rows[static_cast<size_t>(ch)] + p * src_elem_size1;
                std::memcpy(dst_px, src_px + static_cast<size_t>(ch) * src_elem_size1, src_elem_size1);
            }
        }
    }
}

} // namespace

void binaryFunc(BinaryOp op, const Mat& a, const Mat& b, Mat& c)
{
    switch (op)
    {
        case BinaryOp::ADD:
            add(a, b, c);
            return;
        case BinaryOp::SUB:
            subtract(a, b, c);
            return;
        case BinaryOp::MUL:
            multiply(a, b, c);
            return;
        case BinaryOp::DIV:
            divide(a, b, c);
            return;
        case BinaryOp::SUM:
            add(a, b, c);
            return;
        case BinaryOp::EQUAL:
            compare(a, b, c, CV_CMP_EQ);
            return;
        case BinaryOp::GREATER:
            compare(a, b, c, CV_CMP_GT);
            return;
        case BinaryOp::GREATER_EQUAL:
            compare(a, b, c, CV_CMP_GE);
            return;
        case BinaryOp::LESS:
            compare(a, b, c, CV_CMP_LT);
            return;
        case BinaryOp::LESS_EQUAL:
            compare(a, b, c, CV_CMP_LE);
            return;
        case BinaryOp::AND:
            dispatch_mat_mat_integral_binary(
                a,
                b,
                c,
                [](const auto& lhs, const auto& rhs) { return lhs & rhs; },
                "binaryFunc(AND)");
            return;
        case BinaryOp::OR:
            dispatch_mat_mat_integral_binary(
                a,
                b,
                c,
                [](const auto& lhs, const auto& rhs) { return lhs | rhs; },
                "binaryFunc(OR)");
            return;
        case BinaryOp::XOR:
            dispatch_mat_mat_integral_binary(
                a,
                b,
                c,
                [](const auto& lhs, const auto& rhs) { return lhs ^ rhs; },
                "binaryFunc(XOR)");
            return;
        case BinaryOp::MOD:
            dispatch_mat_mat_integral_binary(
                a,
                b,
                c,
                [](const auto& lhs, const auto& rhs) { return integer_mod_divisor_sign(lhs, rhs); },
                "binaryFunc(MOD)");
            return;
        case BinaryOp::BITSHIFT:
            dispatch_mat_mat_integral_binary(
                a,
                b,
                c,
                [](const auto& lhs, const auto& rhs) { return integer_bitshift_value(lhs, rhs); },
                "binaryFunc(BITSHIFT)");
            return;
        case BinaryOp::POW:
            dispatch_mat_mat_binary(
                a,
                b,
                c,
                [](const auto& lhs, const auto& rhs) {
                    return std::pow(static_cast<double>(lhs), static_cast<double>(rhs));
                },
                "binaryFunc(POW)");
            return;
        case BinaryOp::MAX:
            dispatch_mat_mat_binary_arith(
                a,
                b,
                c,
                [](const auto& lhs, const auto& rhs) { return lhs > rhs ? lhs : rhs; },
                cpu::BinaryKernelOp::Max,
                "binaryFunc(MAX)");
            return;
        case BinaryOp::MIN:
            dispatch_mat_mat_binary_arith(
                a,
                b,
                c,
                [](const auto& lhs, const auto& rhs) { return lhs < rhs ? lhs : rhs; },
                cpu::BinaryKernelOp::Min,
                "binaryFunc(MIN)");
            return;
        case BinaryOp::ATAN2:
            dispatch_mat_mat_binary(
                a,
                b,
                c,
                [](const auto& lhs, const auto& rhs) {
                    return std::atan2(static_cast<double>(lhs), static_cast<double>(rhs));
                },
                "binaryFunc(ATAN2)");
            return;
        case BinaryOp::HYPOT:
            dispatch_mat_mat_binary(
                a,
                b,
                c,
                [](const auto& lhs, const auto& rhs) {
                    return std::hypot(static_cast<double>(lhs), static_cast<double>(rhs));
                },
                "binaryFunc(HYPOT)");
            return;
        case BinaryOp::NOT:
            dispatch_mat_mat_integral_binary(
                a,
                b,
                c,
                [](const auto& lhs, const auto& rhs) {
                    using T = std::decay_t<decltype(lhs)>;
                    return static_cast<T>(static_cast<T>(lhs) & static_cast<T>(~rhs));
                },
                "binaryFunc(NOT)");
            return;
        case BinaryOp::FMOD:
            dispatch_mat_mat_binary(
                a,
                b,
                c,
                [](const auto& lhs, const auto& rhs) { return floating_mod_dividend_sign(lhs, rhs); },
                "binaryFunc(FMOD)");
            return;
        case BinaryOp::MEAN:
            dispatch_mat_mat_binary_arith(
                a,
                b,
                c,
                [](const auto& lhs, const auto& rhs) {
                    return (static_cast<double>(lhs) + static_cast<double>(rhs)) * 0.5;
                },
                cpu::BinaryKernelOp::Mean,
                "binaryFunc(MEAN)");
            return;
        default:
            CV_Error_(Error::StsNotImplemented, ("binaryFunc: unsupported BinaryOp=%d", static_cast<int>(op)));
    }
}

void add(const Mat& a, const Mat& b, Mat& c)
{
    dispatch_mat_mat_binary_arith(a,
                                  b,
                                  c,
                                  [](const auto& lhs, const auto& rhs) { return lhs + rhs; },
                                  cpu::BinaryKernelOp::Add,
                                  "add(Mat,Mat)");
}

void add(const Mat& a, const Scalar& b, Mat& c)
{
    dispatch_mat_scalar_binary_arith(a,
                                     b,
                                     c,
                                     [](const auto& lhs, const auto& rhs) { return lhs + rhs; },
                                     cpu::BinaryKernelOp::Add,
                                     false,
                                     "add(Mat,Scalar)");
}

void add(const Scalar& a, const Mat& b, Mat& c)
{
    dispatch_mat_scalar_binary_arith(b,
                                     a,
                                     c,
                                     [](const auto& lhs, const auto& rhs) { return lhs + rhs; },
                                     cpu::BinaryKernelOp::Add,
                                     true,
                                     "add(Scalar,Mat)");
}

void addWeighted(const Mat& a, double alpha, const Mat& b, double beta, Mat& c)
{
    dispatch_add_weighted(a, alpha, b, beta, c, "addWeighted");
}

void subtract(const Mat& a, Mat& c)
{
    dispatch_unary_negate(a, c, "subtract(Mat)");
}

void subtract(const Mat& a, const Mat& b, Mat& c)
{
    dispatch_mat_mat_binary_arith(a,
                                  b,
                                  c,
                                  [](const auto& lhs, const auto& rhs) { return lhs - rhs; },
                                  cpu::BinaryKernelOp::Sub,
                                  "subtract(Mat,Mat)");
}

void subtract(const Mat& a, const Scalar& b, Mat& c)
{
    dispatch_mat_scalar_binary_arith(a,
                                     b,
                                     c,
                                     [](const auto& lhs, const auto& rhs) { return lhs - rhs; },
                                     cpu::BinaryKernelOp::Sub,
                                     false,
                                     "subtract(Mat,Scalar)");
}

void subtract(const Scalar& a, const Mat& b, Mat& c)
{
    dispatch_mat_scalar_binary_arith(b,
                                     a,
                                     c,
                                     [](const auto& lhs, const auto& rhs) { return lhs - rhs; },
                                     cpu::BinaryKernelOp::Sub,
                                     true,
                                     "subtract(Scalar,Mat)");
}

void subtract(const Mat& a, double b, Mat& c)
{
    subtract(a, Scalar::all(b), c);
}

void subtract(double a, const Mat& b, Mat& c)
{
    subtract(Scalar::all(a), b, c);
}

void multiply(const Mat& a, const Mat& b, Mat& c)
{
    dispatch_mat_mat_binary_arith(a,
                                  b,
                                  c,
                                  [](const auto& lhs, const auto& rhs) { return lhs * rhs; },
                                  cpu::BinaryKernelOp::Mul,
                                  "multiply(Mat,Mat)");
}

void multiply(const Mat& a, const Scalar& b, Mat& c)
{
    dispatch_mat_scalar_binary_arith(a,
                                     b,
                                     c,
                                     [](const auto& lhs, const auto& rhs) { return lhs * rhs; },
                                     cpu::BinaryKernelOp::Mul,
                                     false,
                                     "multiply(Mat,Scalar)");
}

void multiply(const Scalar& a, const Mat& b, Mat& c)
{
    dispatch_mat_scalar_binary_arith(b,
                                     a,
                                     c,
                                     [](const auto& lhs, const auto& rhs) { return lhs * rhs; },
                                     cpu::BinaryKernelOp::Mul,
                                     true,
                                     "multiply(Scalar,Mat)");
}

void divide(const Mat& a, const Mat& b, Mat& c)
{
    dispatch_mat_mat_binary_arith(a,
                                  b,
                                  c,
                                  [](const auto& lhs, const auto& rhs) { return safe_div_value(lhs, rhs); },
                                  cpu::BinaryKernelOp::Div,
                                  "divide(Mat,Mat)");
}

void divide(const Mat& a, const Scalar& b, Mat& c)
{
    dispatch_mat_scalar_binary_arith(a,
                                     b,
                                     c,
                                     [](const auto& lhs, const auto& rhs) { return safe_div_value(lhs, rhs); },
                                     cpu::BinaryKernelOp::Div,
                                     false,
                                     "divide(Mat,Scalar)");
}

void divide(const Scalar& a, const Mat& b, Mat& c)
{
    dispatch_mat_scalar_binary_arith(b,
                                     a,
                                     c,
                                     [](const auto& lhs, const auto& rhs) { return safe_div_value(lhs, rhs); },
                                     cpu::BinaryKernelOp::Div,
                                     true,
                                     "divide(Scalar,Mat)");
}

void compare(const Mat& a, const Mat& b, Mat& c, int op)
{
    if (a.empty() && b.empty())
    {
        c.release();
        return;
    }
    if (a.empty() || b.empty())
    {
        CV_Error_(Error::StsBadArg, ("compare(Mat,Mat) expects both inputs non-empty or both empty"));
    }
    if (a.depth() == CV_16F)
    {
        CV_Error_(Error::StsNotImplemented,
                  ("compare(Mat,Mat) does not support CV_16F in v1 compatibility mode"));
    }

    dispatch_mat_mat_compare(a, b, c, op, "compare(Mat,Mat)");
}

void compare(const Mat& a, const Scalar& b, Mat& c, int op)
{
    dispatch_mat_scalar_compare(a, b, c, op, false, "compare(Mat,Scalar)");
}

void compare(const Scalar& a, const Mat& b, Mat& c, int op)
{
    dispatch_mat_scalar_compare(b, a, c, op, true, "compare(Scalar,Mat)");
}

void merge(const Mat* src, size_t nsrc, Mat& dst)
{
    int depth = 0;
    int dims = 0;
    int total_channels = 0;
    MatShape shape;
    validate_merge_sources(src, nsrc, depth, dims, shape, total_channels, "merge");
    ensure_merge_dst(src[0], depth, dims, shape, total_channels, dst, "merge");
    copy_merge_channels(src, nsrc, dst);
}

void merge(const std::vector<Mat>& src, Mat& dst)
{
    if (src.empty())
    {
        CV_Error_(Error::StsBadArg, ("merge expects at least one source Mat"));
    }
    merge(src.data(), src.size(), dst);
}

void split(const Mat& src, Mat* dst)
{
    if (src.empty())
    {
        CV_Error_(Error::StsBadArg, ("split expects non-empty src Mat"));
    }
    ensure_split_dst(src, dst, "split");
    copy_split_channels(src, dst);
}

void split(const Mat& src, std::vector<Mat>& dst)
{
    if (src.empty())
    {
        CV_Error_(Error::StsBadArg, ("split expects non-empty src Mat"));
    }
    dst.resize(static_cast<size_t>(src.channels()));
    split(src, dst.data());
}

Mat transpose(const Mat& input)
{
    CV_Assert(!input.empty() && "The transpose function get empty input!");
    const MatShape input_shape = input.shape();
    if (input_shape.size() == 1)
    {
        Mat out = input.clone();
        out.setSize({1, input_shape[0]});
        return out;
    }

    const int dim = static_cast<int>(input_shape.size());
    std::vector<int> out_order(static_cast<size_t>(dim));
    for (int i = 0; i < dim; ++i)
    {
        out_order[static_cast<size_t>(i)] = i;
    }

    std::swap(out_order[static_cast<size_t>(dim - 2)], out_order[static_cast<size_t>(dim - 1)]);
    return transposeND(input, out_order);
}

Mat transposeND(const Mat& input, const std::vector<int> order)
{
    if (input.dims != static_cast<int>(order.size()))
    {
        CV_Error_(Error::StsBadSize,
                  ("In transposeND, input dim is not equal to order size! input dim=%d, order size=%d",
                   input.dims,
                   static_cast<int>(order.size())));
    }

    std::vector<int> order_sorted = order;
    std::sort(order_sorted.begin(), order_sorted.end());
    for (int i = 0; i < static_cast<int>(order_sorted.size()); ++i)
    {
        if (order_sorted[static_cast<size_t>(i)] != i)
        {
            CV_Error(Error::StsBadSize, "New order should be a valid permutation of old order.");
        }
    }

    const MatShape old_shape = input.shape();
    std::vector<int> new_shape(order.size());
    for (int i = 0; i < static_cast<int>(order.size()); ++i)
    {
        new_shape[static_cast<size_t>(i)] = old_shape[static_cast<size_t>(order[static_cast<size_t>(i)])];
    }

    Mat out(new_shape, input.type());

    bool is_last_two_swap = order.size() >= 2;
    for (int i = 0; i < static_cast<int>(order.size()) && is_last_two_swap; ++i)
    {
        int expected = i;
        if (i == static_cast<int>(order.size()) - 2)
        {
            expected = static_cast<int>(order.size()) - 1;
        }
        else if (i == static_cast<int>(order.size()) - 1)
        {
            expected = static_cast<int>(order.size()) - 2;
        }

        if (order[static_cast<size_t>(i)] != expected)
        {
            is_last_two_swap = false;
        }
    }

    if (is_last_two_swap)
    {
        const int rows = old_shape[old_shape.size() - 2];
        const int cols = old_shape[old_shape.size() - 1];
        const size_t elem_size1 = CV_ELEM_SIZE1(input.type());
        const int channels = input.channels();
        const size_t elem_size = elem_size1 * static_cast<size_t>(channels);
        const size_t plane_bytes = static_cast<size_t>(rows) * static_cast<size_t>(cols) * elem_size;
        const size_t batch = input.total() / static_cast<size_t>(rows * cols);

        const unsigned char* src = input.data;
        unsigned char* dst = out.data;
        for (size_t batch_idx = 0; batch_idx < batch; ++batch_idx)
        {
            cpu::transpose2d_kernel_blocked(src + batch_idx * plane_bytes,
                                            dst + batch_idx * plane_bytes,
                                            rows,
                                            cols,
                                            elem_size1,
                                            channels);
        }
        return out;
    }

    const int dims = static_cast<int>(order.size());
    int continuous_idx = 0;
    for (int i = dims - 1; i >= 0; --i)
    {
        if (order[static_cast<size_t>(i)] != i)
        {
            continuous_idx = i + 1;
            break;
        }
    }

    size_t continuous_size = 1;
    if (continuous_idx != dims)
    {
        continuous_size = total(new_shape, continuous_idx);
    }
    if (continuous_idx == 0)
    {
        continuous_size = total(new_shape);
    }

    size_t step_in = 1;
    std::vector<size_t> input_steps(order.size());
    std::vector<size_t> input_steps_old(order.size());

    for (int i = dims - 1; i >= 0; --i)
    {
        input_steps_old[static_cast<size_t>(i)] = step_in;
        step_in *= static_cast<size_t>(old_shape[static_cast<size_t>(i)]);
    }

    for (int i = dims - 1; i >= 0; --i)
    {
        input_steps[static_cast<size_t>(i)] =
            input_steps_old[static_cast<size_t>(order[static_cast<size_t>(i)])];
    }

    const size_t out_size = input.total() / continuous_size;
    const unsigned char* src = input.data;
    unsigned char* dst = out.data;
    size_t src_offset = 0;
    const size_t elem_size = CV_ELEM_SIZE(out.type());
    const size_t continuous_bytes = elem_size * continuous_size;

    for (size_t i = 0; i < out_size; ++i)
    {
        std::memcpy(dst, src + elem_size * src_offset, continuous_bytes);
        dst += continuous_bytes;

        for (int j = continuous_idx - 1; j >= 0; --j)
        {
            src_offset += input_steps[static_cast<size_t>(j)];
            const size_t dim_size = static_cast<size_t>(out.size[j]);
            if ((src_offset / input_steps[static_cast<size_t>(j)]) % dim_size != 0)
            {
                break;
            }
            src_offset -= input_steps[static_cast<size_t>(j)] * dim_size;
        }
    }

    return out;
}

} // namespace cvh
