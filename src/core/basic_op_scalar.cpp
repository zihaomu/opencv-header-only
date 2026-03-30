#include "cvh/core/basic_op.h"
#include "cvh/core/saturate.h"

#include <cmath>
#include <type_traits>

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
void dispatch_mat_mat_binary(const Mat& a, const Mat& b, Mat& dst, Op op, const char* fn_name)
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
        case CV_32F:
            apply_mat_mat_binary_impl<float>(a, b, dst, op);
            break;
        case CV_16F:
            apply_mat_mat_binary_impl<hfloat>(a, b, dst, op);
            break;
        default:
            CV_Error_(Error::StsNotImplemented,
                      ("%s supports depth in [CV_8U..CV_16F], depth=%d", fn_name, a.depth()));
    }
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
                      ("%s supports depth in [CV_8U..CV_16F], depth=%d", fn_name, a.depth()));
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
void dispatch_mat_scalar_binary(const Mat& src,
                                const Scalar& scalar,
                                Mat& dst,
                                Op op,
                                bool scalar_first,
                                const char* fn_name)
{
    check_scalar_channel_bound(src, fn_name);
    ensure_binary_dst_like_src(src, dst, fn_name);

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
                      ("%s supports depth in [CV_8U..CV_16F], depth=%d", fn_name, a.depth()));
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
            dispatch_mat_mat_binary(
                a,
                b,
                c,
                [](const auto& lhs, const auto& rhs) { return lhs > rhs ? lhs : rhs; },
                "binaryFunc(MAX)");
            return;
        case BinaryOp::MIN:
            dispatch_mat_mat_binary(
                a,
                b,
                c,
                [](const auto& lhs, const auto& rhs) { return lhs < rhs ? lhs : rhs; },
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
            dispatch_mat_mat_binary(
                a,
                b,
                c,
                [](const auto& lhs, const auto& rhs) {
                    return (static_cast<double>(lhs) + static_cast<double>(rhs)) * 0.5;
                },
                "binaryFunc(MEAN)");
            return;
        default:
            CV_Error_(Error::StsNotImplemented, ("binaryFunc: unsupported BinaryOp=%d", static_cast<int>(op)));
    }
}

void add(const Mat& a, const Mat& b, Mat& c)
{
    dispatch_mat_mat_binary(a, b, c,
                            [](const auto& lhs, const auto& rhs) { return lhs + rhs; },
                            "add(Mat,Mat)");
}

void add(const Mat& a, const Scalar& b, Mat& c)
{
    dispatch_mat_scalar_binary(a, b, c,
                               [](const auto& lhs, const auto& rhs) { return lhs + rhs; },
                               false,
                               "add(Mat,Scalar)");
}

void add(const Scalar& a, const Mat& b, Mat& c)
{
    dispatch_mat_scalar_binary(b, a, c,
                               [](const auto& lhs, const auto& rhs) { return lhs + rhs; },
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
    dispatch_mat_mat_binary(a, b, c,
                            [](const auto& lhs, const auto& rhs) { return lhs - rhs; },
                            "subtract(Mat,Mat)");
}

void subtract(const Mat& a, const Scalar& b, Mat& c)
{
    dispatch_mat_scalar_binary(a, b, c,
                               [](const auto& lhs, const auto& rhs) { return lhs - rhs; },
                               false,
                               "subtract(Mat,Scalar)");
}

void subtract(const Scalar& a, const Mat& b, Mat& c)
{
    dispatch_mat_scalar_binary(b, a, c,
                               [](const auto& lhs, const auto& rhs) { return lhs - rhs; },
                               true,
                               "subtract(Scalar,Mat)");
}

void multiply(const Mat& a, const Mat& b, Mat& c)
{
    dispatch_mat_mat_binary(a, b, c,
                            [](const auto& lhs, const auto& rhs) { return lhs * rhs; },
                            "multiply(Mat,Mat)");
}

void divide(const Mat& a, const Mat& b, Mat& c)
{
    dispatch_mat_mat_binary(a, b, c,
                            [](const auto& lhs, const auto& rhs) { return safe_div_value(lhs, rhs); },
                            "divide(Mat,Mat)");
}

void compare(const Mat& a, const Mat& b, Mat& c, int op)
{
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

} // namespace cvh
