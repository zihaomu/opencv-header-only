#include "cvh/core/basic_op.h"
#include "cvh/core/saturate.h"

namespace cvh
{

namespace
{

inline void check_scalar_channel_bound(const Mat& src, const char* fn_name)
{
    if (src.channels() > 4)
    {
        CV_Error_(Error::StsBadArg,
                  ("%s supports channels <= 4 in v1, channels=%d", fn_name, src.channels()));
    }
}

inline void ensure_binary_dst_like_src(const Mat& src, Mat& dst, const char* fn_name)
{
    if (src.empty())
    {
        CV_Error_(Error::StsBadArg, ("%s expects non-empty src Mat", fn_name));
    }

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
    if (src.empty())
    {
        CV_Error_(Error::StsBadArg, ("%s expects non-empty src Mat", fn_name));
    }

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

} // namespace

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

void compare(const Mat& a, const Scalar& b, Mat& c, int op)
{
    dispatch_mat_scalar_compare(a, b, c, op, false, "compare(Mat,Scalar)");
}

void compare(const Scalar& a, const Mat& b, Mat& c, int op)
{
    dispatch_mat_scalar_compare(b, a, c, op, true, "compare(Scalar,Mat)");
}

} // namespace cvh
