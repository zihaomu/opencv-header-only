#include "binary_kernel_xsimd.h"
#include "cvh/core/parallel.h"
#include "cvh/core/saturate.h"
#include "cvh/core/detail/xsimd_kernel_utils.h"
#include "cvh/core/define.h"

#include "xsimd/xsimd.hpp"

#include <array>
#include <cstdint>
#include <limits>

namespace cvh {
namespace cpu {

namespace {

using Batch = xsimd::batch<float>;
constexpr size_t kLanes = Batch::size;

using Batch64 = xsimd::batch<double>;
constexpr size_t kLanes64 = Batch64::size;

using Batch32 = xsimd::batch<std::int32_t>;
constexpr size_t kLanes32 = Batch32::size;

using BatchU32 = xsimd::batch<std::uint32_t>;
constexpr size_t kLanesU32 = BatchU32::size;

using Batch16 = xsimd::batch<std::int16_t>;
constexpr size_t kLanes16 = Batch16::size;

using Batch8 = xsimd::batch<std::int8_t>;
constexpr size_t kLanes8 = Batch8::size;

inline float apply_scalar(BinaryKernelOp op, float lhs, float rhs)
{
    switch (op)
    {
        case BinaryKernelOp::Add:
            return lhs + rhs;
        case BinaryKernelOp::Sub:
            return lhs - rhs;
        case BinaryKernelOp::Mul:
            return lhs * rhs;
        case BinaryKernelOp::Div:
            return lhs / rhs;
        case BinaryKernelOp::Max:
            return lhs > rhs ? lhs : rhs;
        case BinaryKernelOp::Min:
            return lhs < rhs ? lhs : rhs;
        case BinaryKernelOp::Mean:
            return 0.5f * (lhs + rhs);
    }

    return 0.0f;
}

inline Batch apply_batch(BinaryKernelOp op, const Batch& lhs, const Batch& rhs)
{
    switch (op)
    {
        case BinaryKernelOp::Add:
            return lhs + rhs;
        case BinaryKernelOp::Sub:
            return lhs - rhs;
        case BinaryKernelOp::Mul:
            return lhs * rhs;
        case BinaryKernelOp::Div:
            return lhs / rhs;
        case BinaryKernelOp::Max:
            return xsimd::max(lhs, rhs);
        case BinaryKernelOp::Min:
            return xsimd::min(lhs, rhs);
        case BinaryKernelOp::Mean:
            return (lhs + rhs) * Batch(0.5f);
    }

    return Batch(0.0f);
}

inline bool apply_compare_scalar(CompareKernelOp op, float lhs, float rhs)
{
    switch (op)
    {
        case CompareKernelOp::Eq:
            return lhs == rhs;
        case CompareKernelOp::Gt:
            return lhs > rhs;
        case CompareKernelOp::Ge:
            return lhs >= rhs;
        case CompareKernelOp::Lt:
            return lhs < rhs;
        case CompareKernelOp::Le:
            return lhs <= rhs;
        case CompareKernelOp::Ne:
            return lhs != rhs;
    }

    return false;
}

inline Batch::batch_bool_type apply_compare_batch(CompareKernelOp op, const Batch& lhs, const Batch& rhs)
{
    switch (op)
    {
        case CompareKernelOp::Eq:
            return lhs == rhs;
        case CompareKernelOp::Gt:
            return lhs > rhs;
        case CompareKernelOp::Ge:
            return lhs >= rhs;
        case CompareKernelOp::Lt:
            return lhs < rhs;
        case CompareKernelOp::Le:
            return lhs <= rhs;
        case CompareKernelOp::Ne:
            return lhs != rhs;
    }

    return lhs != rhs;
}

template<typename T>
inline bool apply_compare_scalar_typed(CompareKernelOp op, T lhs, T rhs)
{
    switch (op)
    {
        case CompareKernelOp::Eq:
            return lhs == rhs;
        case CompareKernelOp::Gt:
            return lhs > rhs;
        case CompareKernelOp::Ge:
            return lhs >= rhs;
        case CompareKernelOp::Lt:
            return lhs < rhs;
        case CompareKernelOp::Le:
            return lhs <= rhs;
        case CompareKernelOp::Ne:
            return lhs != rhs;
    }

    return false;
}

template<typename BatchType>
inline typename BatchType::batch_bool_type apply_compare_batch_typed(CompareKernelOp op,
                                                                     const BatchType& lhs,
                                                                     const BatchType& rhs)
{
    switch (op)
    {
        case CompareKernelOp::Eq:
            return lhs == rhs;
        case CompareKernelOp::Gt:
            return lhs > rhs;
        case CompareKernelOp::Ge:
            return lhs >= rhs;
        case CompareKernelOp::Lt:
            return lhs < rhs;
        case CompareKernelOp::Le:
            return lhs <= rhs;
        case CompareKernelOp::Ne:
            return lhs != rhs;
    }

    return lhs != rhs;
}

template <class Fn>
inline void for_each_outer(size_t outer, size_t inner, Fn&& fn)
{
    const bool do_parallel = should_parallelize_1d_loop(outer, inner, 1LL << 15, 2);
    const size_t max_parallel_range = static_cast<size_t>(std::numeric_limits<int>::max());
    if (!do_parallel || outer > max_parallel_range)
    {
        for (size_t outer_i = 0; outer_i < outer; ++outer_i)
        {
            fn(outer_i);
        }
        return;
    }

    cvh::parallel_for_(
        cvh::Range(0, static_cast<int>(outer)),
        [&](const cvh::Range& range) {
            for (int outer_idx = range.start; outer_idx < range.end; ++outer_idx)
            {
                fn(static_cast<size_t>(outer_idx));
            }
        },
        static_cast<double>(outer));
}

// CV_16F (hfloat) variant - converts to float internally
inline void binary_broadcast_xsimd_hfloat_impl(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner)
{
    // For hfloat (CV_16F), process in float batch and convert back in batch.
    for_each_outer(outer, inner, [&](size_t outer_i) {
        const hfloat* lhs_row = reinterpret_cast<const hfloat*>(lhs) + outer_i * lhs_outer_stride;
        const hfloat* rhs_row = reinterpret_cast<const hfloat*>(rhs) + outer_i * rhs_outer_stride;
        hfloat* out_row = reinterpret_cast<hfloat*>(out) + outer_i * inner;

        size_t inner_idx = 0;
        if (lhs_inner_stride <= 1 && rhs_inner_stride <= 1)
        {
            const Batch lhs_scalar = lhs_inner_stride == 0 ? Batch(static_cast<float>(lhs_row[0])) : Batch(0.0f);
            const Batch rhs_scalar = rhs_inner_stride == 0 ? Batch(static_cast<float>(rhs_row[0])) : Batch(0.0f);
            for (; inner_idx + kLanes <= inner; inner_idx += kLanes)
            {
                const Batch lhs_vec = lhs_inner_stride == 0 ? lhs_scalar : load_hfloat_batch(lhs_row + inner_idx);
                const Batch rhs_vec = rhs_inner_stride == 0 ? rhs_scalar : load_hfloat_batch(rhs_row + inner_idx);
                const Batch out_vec = apply_batch(op, lhs_vec, rhs_vec);
                store_hfloat_batch(out_vec, out_row + inner_idx);
            }
        }

        for (; inner_idx < inner; ++inner_idx)
        {
            const float lhs_val = lhs_inner_stride == 0 ? static_cast<float>(lhs_row[0])
                                                        : static_cast<float>(lhs_row[inner_idx * lhs_inner_stride]);
            const float rhs_val = rhs_inner_stride == 0 ? static_cast<float>(rhs_row[0])
                                                        : static_cast<float>(rhs_row[inner_idx * rhs_inner_stride]);
            out_row[inner_idx] = hfloat(apply_scalar(op, lhs_val, rhs_val));
        }
    });
}

}  // namespace

void binary_broadcast_xsimd(BinaryKernelOp op,
                            const float* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const float* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            float* out,
                            size_t outer,
                            size_t inner)
{
    for_each_outer(outer, inner, [&](size_t outer_i) {
        const float* lhs_row = lhs + outer_i * lhs_outer_stride;
        const float* rhs_row = rhs + outer_i * rhs_outer_stride;
        float* out_row = out + outer_i * inner;

        size_t inner_idx = 0;
        if (lhs_inner_stride <= 1 && rhs_inner_stride <= 1)
        {
            for (; inner_idx + kLanes <= inner; inner_idx += kLanes)
            {
                const Batch lhs_vec = lhs_inner_stride == 0
                    ? Batch(lhs_row[0])
                    : Batch::load_unaligned(lhs_row + inner_idx);
                const Batch rhs_vec = rhs_inner_stride == 0
                    ? Batch(rhs_row[0])
                    : Batch::load_unaligned(rhs_row + inner_idx);
                const Batch out_vec = apply_batch(op, lhs_vec, rhs_vec);
                out_vec.store_unaligned(out_row + inner_idx);
            }
        }

        for (; inner_idx < inner; ++inner_idx)
        {
            const float lhs_val = lhs_row[inner_idx * lhs_inner_stride];
            const float rhs_val = rhs_row[inner_idx * rhs_inner_stride];
            out_row[inner_idx] = apply_scalar(op, lhs_val, rhs_val);
        }
    });
}

void binary_broadcast_xsimd_hfloat(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner)
{
    binary_broadcast_xsimd_hfloat_impl(op, lhs, lhs_outer_stride, lhs_inner_stride,
                                        rhs, rhs_outer_stride, rhs_inner_stride,
                                        out, outer, inner);
}

inline double apply_scalar64(BinaryKernelOp op, double lhs, double rhs)
{
    switch (op)
    {
        case BinaryKernelOp::Add:
            return lhs + rhs;
        case BinaryKernelOp::Sub:
            return lhs - rhs;
        case BinaryKernelOp::Mul:
            return lhs * rhs;
        case BinaryKernelOp::Div:
            return lhs / rhs;
        case BinaryKernelOp::Max:
            return lhs > rhs ? lhs : rhs;
        case BinaryKernelOp::Min:
            return lhs < rhs ? lhs : rhs;
        case BinaryKernelOp::Mean:
            return 0.5 * (lhs + rhs);
    }

    return 0.0;
}

inline Batch64 apply_batch64(BinaryKernelOp op, const Batch64& lhs, const Batch64& rhs)
{
    switch (op)
    {
        case BinaryKernelOp::Add:
            return lhs + rhs;
        case BinaryKernelOp::Sub:
            return lhs - rhs;
        case BinaryKernelOp::Mul:
            return lhs * rhs;
        case BinaryKernelOp::Div:
            return lhs / rhs;
        case BinaryKernelOp::Max:
            return xsimd::max(lhs, rhs);
        case BinaryKernelOp::Min:
            return xsimd::min(lhs, rhs);
        case BinaryKernelOp::Mean:
            return (lhs + rhs) * Batch64(0.5);
    }

    return Batch64(0.0);
}

inline void binary_broadcast_xsimd_double_impl(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner)
{
    for_each_outer(outer, inner, [&](size_t outer_i) {
        const double* lhs_row = reinterpret_cast<const double*>(lhs) + outer_i * lhs_outer_stride;
        const double* rhs_row = reinterpret_cast<const double*>(rhs) + outer_i * rhs_outer_stride;
        double* out_row = reinterpret_cast<double*>(out) + outer_i * inner;

        size_t inner_idx = 0;
        if (lhs_inner_stride <= 1 && rhs_inner_stride <= 1)
        {
            for (; inner_idx + kLanes64 <= inner; inner_idx += kLanes64)
            {
                const Batch64 lhs_vec = lhs_inner_stride == 0
                    ? Batch64(lhs_row[0])
                    : Batch64::load_unaligned(lhs_row + inner_idx);
                const Batch64 rhs_vec = rhs_inner_stride == 0
                    ? Batch64(rhs_row[0])
                    : Batch64::load_unaligned(rhs_row + inner_idx);
                const Batch64 out_vec = apply_batch64(op, lhs_vec, rhs_vec);
                out_vec.store_unaligned(out_row + inner_idx);
            }
        }

        for (; inner_idx < inner; ++inner_idx)
        {
            const double lhs_val = lhs_row[inner_idx * lhs_inner_stride];
            const double rhs_val = rhs_row[inner_idx * rhs_inner_stride];
            out_row[inner_idx] = apply_scalar64(op, lhs_val, rhs_val);
        }
    });
}

void binary_broadcast_xsimd_double(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner)
{
    binary_broadcast_xsimd_double_impl(op, lhs, lhs_outer_stride, lhs_inner_stride,
                                        rhs, rhs_outer_stride, rhs_inner_stride,
                                        out, outer, inner);
}

// Integer scalar functions
inline std::int32_t apply_scalar_i32(BinaryKernelOp op, std::int32_t lhs, std::int32_t rhs)
{
    switch (op)
    {
        case BinaryKernelOp::Add:
            return lhs + rhs;
        case BinaryKernelOp::Sub:
            return lhs - rhs;
        case BinaryKernelOp::Mul:
            return lhs * rhs;
        case BinaryKernelOp::Div:
            return rhs != 0 ? lhs / rhs : 0;
        case BinaryKernelOp::Max:
            return lhs > rhs ? lhs : rhs;
        case BinaryKernelOp::Min:
            return lhs < rhs ? lhs : rhs;
        case BinaryKernelOp::Mean:
            return (lhs + rhs) / 2;
        default:
            return 0;
    }
}

inline Batch32 apply_batch_i32(BinaryKernelOp op, const Batch32& lhs, const Batch32& rhs)
{
    switch (op)
    {
        case BinaryKernelOp::Add:
            return lhs + rhs;
        case BinaryKernelOp::Sub:
            return lhs - rhs;
        case BinaryKernelOp::Mul:
            return lhs * rhs;
        case BinaryKernelOp::Div:
            return lhs / rhs;
        case BinaryKernelOp::Max:
            return xsimd::max(lhs, rhs);
        case BinaryKernelOp::Min:
            return xsimd::min(lhs, rhs);
        case BinaryKernelOp::Mean:
            return (lhs + rhs) / Batch32(2);
        default:
            return Batch32(0);
    }
}

inline void binary_broadcast_xsimd_int32_impl(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner)
{
    for_each_outer(outer, inner, [&](size_t outer_i) {
        const std::int32_t* lhs_row = reinterpret_cast<const std::int32_t*>(lhs) + outer_i * lhs_outer_stride;
        const std::int32_t* rhs_row = reinterpret_cast<const std::int32_t*>(rhs) + outer_i * rhs_outer_stride;
        std::int32_t* out_row = reinterpret_cast<std::int32_t*>(out) + outer_i * inner;

        size_t inner_idx = 0;
        if (lhs_inner_stride <= 1 && rhs_inner_stride <= 1)
        {
            for (; inner_idx + kLanes32 <= inner; inner_idx += kLanes32)
            {
                const Batch32 lhs_vec = lhs_inner_stride == 0
                    ? Batch32(lhs_row[0])
                    : Batch32::load_unaligned(lhs_row + inner_idx);
                const Batch32 rhs_vec = rhs_inner_stride == 0
                    ? Batch32(rhs_row[0])
                    : Batch32::load_unaligned(rhs_row + inner_idx);
                const Batch32 out_vec = apply_batch_i32(op, lhs_vec, rhs_vec);
                out_vec.store_unaligned(out_row + inner_idx);
            }
        }

        for (; inner_idx < inner; ++inner_idx)
        {
            const std::int32_t lhs_val = lhs_row[inner_idx * lhs_inner_stride];
            const std::int32_t rhs_val = rhs_row[inner_idx * rhs_inner_stride];
            out_row[inner_idx] = apply_scalar_i32(op, lhs_val, rhs_val);
        }
    });
}

void binary_broadcast_xsimd_int32(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner)
{
    binary_broadcast_xsimd_int32_impl(op, lhs, lhs_outer_stride, lhs_inner_stride,
                                       rhs, rhs_outer_stride, rhs_inner_stride,
                                       out, outer, inner);
}

inline std::uint32_t apply_scalar_u32(BinaryKernelOp op, std::uint32_t lhs, std::uint32_t rhs)
{
    switch (op)
    {
        case BinaryKernelOp::Add:
            return lhs + rhs;
        case BinaryKernelOp::Sub:
            return lhs - rhs;
        case BinaryKernelOp::Mul:
            return lhs * rhs;
        case BinaryKernelOp::Div:
            return rhs != 0 ? lhs / rhs : 0u;
        case BinaryKernelOp::Max:
            return lhs > rhs ? lhs : rhs;
        case BinaryKernelOp::Min:
            return lhs < rhs ? lhs : rhs;
        case BinaryKernelOp::Mean:
            return (lhs + rhs) / 2u;
        default:
            return 0u;
    }
}

inline BatchU32 apply_batch_u32(BinaryKernelOp op, const BatchU32& lhs, const BatchU32& rhs)
{
    switch (op)
    {
        case BinaryKernelOp::Add:
            return lhs + rhs;
        case BinaryKernelOp::Sub:
            return lhs - rhs;
        case BinaryKernelOp::Mul:
            return lhs * rhs;
        case BinaryKernelOp::Div:
            return lhs / rhs;
        case BinaryKernelOp::Max:
            return xsimd::max(lhs, rhs);
        case BinaryKernelOp::Min:
            return xsimd::min(lhs, rhs);
        case BinaryKernelOp::Mean:
            return (lhs + rhs) / BatchU32(2u);
        default:
            return BatchU32(0u);
    }
}

inline void binary_broadcast_xsimd_uint32_impl(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner)
{
    for_each_outer(outer, inner, [&](size_t outer_i) {
        const std::uint32_t* lhs_row = reinterpret_cast<const std::uint32_t*>(lhs) + outer_i * lhs_outer_stride;
        const std::uint32_t* rhs_row = reinterpret_cast<const std::uint32_t*>(rhs) + outer_i * rhs_outer_stride;
        std::uint32_t* out_row = reinterpret_cast<std::uint32_t*>(out) + outer_i * inner;

        size_t inner_idx = 0;
        if (lhs_inner_stride <= 1 && rhs_inner_stride <= 1)
        {
            for (; inner_idx + kLanesU32 <= inner; inner_idx += kLanesU32)
            {
                const BatchU32 lhs_vec = lhs_inner_stride == 0
                    ? BatchU32(lhs_row[0])
                    : BatchU32::load_unaligned(lhs_row + inner_idx);
                const BatchU32 rhs_vec = rhs_inner_stride == 0
                    ? BatchU32(rhs_row[0])
                    : BatchU32::load_unaligned(rhs_row + inner_idx);
                const BatchU32 out_vec = apply_batch_u32(op, lhs_vec, rhs_vec);
                out_vec.store_unaligned(out_row + inner_idx);
            }
        }

        for (; inner_idx < inner; ++inner_idx)
        {
            const std::uint32_t lhs_val = lhs_row[inner_idx * lhs_inner_stride];
            const std::uint32_t rhs_val = rhs_row[inner_idx * rhs_inner_stride];
            out_row[inner_idx] = apply_scalar_u32(op, lhs_val, rhs_val);
        }
    });
}

void binary_broadcast_xsimd_uint32(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner)
{
    binary_broadcast_xsimd_uint32_impl(op, lhs, lhs_outer_stride, lhs_inner_stride,
                                       rhs, rhs_outer_stride, rhs_inner_stride,
                                       out, outer, inner);
}

// Generic integer kernel for 8-bit and 16-bit types with saturation
template<typename T, typename BatchType, size_t kLanesInt>
inline void binary_broadcast_xsimd_int_impl(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner)
{
    auto store_saturated_mul_batch = [](const BatchType& lhs_vec, const BatchType& rhs_vec, T* out_ptr)
    {
        const auto lhs_wide = xsimd::widen(lhs_vec);
        const auto rhs_wide = xsimd::widen(rhs_vec);
        using WideBatch = typename decltype(lhs_wide)::value_type;
        using WideT = xsimd::widen_t<T>;
        constexpr size_t kWideLanes = WideBatch::size;
        std::array<WideT, kLanesInt> tmp {};
        (lhs_wide[0] * rhs_wide[0]).store_unaligned(tmp.data());
        (lhs_wide[1] * rhs_wide[1]).store_unaligned(tmp.data() + kWideLanes);
        for (size_t lane = 0; lane < kLanesInt; ++lane)
        {
            out_ptr[lane] = saturate_cast<T>(tmp[lane]);
        }
    };

    for_each_outer(outer, inner, [&](size_t outer_i) {
        const T* lhs_row = reinterpret_cast<const T*>(lhs) + outer_i * lhs_outer_stride;
        const T* rhs_row = reinterpret_cast<const T*>(rhs) + outer_i * rhs_outer_stride;
        T* out_row = reinterpret_cast<T*>(out) + outer_i * inner;

        size_t inner_idx = 0;
        if (lhs_inner_stride <= 1 && rhs_inner_stride <= 1)
        {
            for (; inner_idx + kLanesInt <= inner; inner_idx += kLanesInt)
            {
                const BatchType lhs_vec = lhs_inner_stride == 0
                    ? BatchType(lhs_row[0])
                    : BatchType::load_unaligned(lhs_row + inner_idx);
                const BatchType rhs_vec = rhs_inner_stride == 0
                    ? BatchType(rhs_row[0])
                    : BatchType::load_unaligned(rhs_row + inner_idx);

                BatchType out_vec;
                switch (op)
                {
                    case BinaryKernelOp::Add:
                        out_vec = xsimd::sadd(lhs_vec, rhs_vec);
                        out_vec.store_unaligned(out_row + inner_idx);
                        break;
                    case BinaryKernelOp::Sub:
                        out_vec = xsimd::ssub(lhs_vec, rhs_vec);
                        out_vec.store_unaligned(out_row + inner_idx);
                        break;
                    case BinaryKernelOp::Mul:
                        store_saturated_mul_batch(lhs_vec, rhs_vec, out_row + inner_idx);
                        break;
                    case BinaryKernelOp::Div:
                        out_vec = lhs_vec / rhs_vec;
                        out_vec.store_unaligned(out_row + inner_idx);
                        break;
                    case BinaryKernelOp::Max:
                        out_vec = xsimd::max(lhs_vec, rhs_vec);
                        out_vec.store_unaligned(out_row + inner_idx);
                        break;
                    case BinaryKernelOp::Min:
                        out_vec = xsimd::min(lhs_vec, rhs_vec);
                        out_vec.store_unaligned(out_row + inner_idx);
                        break;
                    case BinaryKernelOp::Mean:
                        out_vec = (lhs_vec + rhs_vec) / BatchType(2);
                        out_vec.store_unaligned(out_row + inner_idx);
                        break;
                    default:
                        out_vec = BatchType(0);
                        out_vec.store_unaligned(out_row + inner_idx);
                }
            }
        }

        for (; inner_idx < inner; ++inner_idx)
        {
            const T lhs_val = lhs_row[inner_idx * lhs_inner_stride];
            const T rhs_val = rhs_row[inner_idx * rhs_inner_stride];
            T result;
            switch (op)
            {
                case BinaryKernelOp::Add:
                    result = saturate_cast<T>(static_cast<int64>(lhs_val) + static_cast<int64>(rhs_val));
                    break;
                case BinaryKernelOp::Sub:
                    result = saturate_cast<T>(static_cast<int64>(lhs_val) - static_cast<int64>(rhs_val));
                    break;
                case BinaryKernelOp::Mul:
                    result = saturate_cast<T>(static_cast<int64>(lhs_val) * static_cast<int64>(rhs_val));
                    break;
                case BinaryKernelOp::Div:
                    result = rhs_val != 0 ? lhs_val / rhs_val : 0;
                    break;
                case BinaryKernelOp::Max:
                    result = lhs_val > rhs_val ? lhs_val : rhs_val;
                    break;
                case BinaryKernelOp::Min:
                    result = lhs_val < rhs_val ? lhs_val : rhs_val;
                    break;
                case BinaryKernelOp::Mean:
                    result = static_cast<T>((static_cast<int>(lhs_val) + static_cast<int>(rhs_val)) / 2);
                    break;
                default:
                    result = 0;
            }
            out_row[inner_idx] = result;
        }
    });
}

void binary_broadcast_xsimd_int16(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner)
{
    binary_broadcast_xsimd_int_impl<std::int16_t, Batch16, kLanes16>(
        op, lhs, lhs_outer_stride, lhs_inner_stride,
        rhs, rhs_outer_stride, rhs_inner_stride,
        out, outer, inner);
}

void binary_broadcast_xsimd_uint16(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner)
{
    using BatchU16 = xsimd::batch<std::uint16_t>;
    constexpr size_t kLanesU16 = BatchU16::size;
    binary_broadcast_xsimd_int_impl<std::uint16_t, BatchU16, kLanesU16>(
        op, lhs, lhs_outer_stride, lhs_inner_stride,
        rhs, rhs_outer_stride, rhs_inner_stride,
        out, outer, inner);
}

void binary_broadcast_xsimd_int8(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner)
{
    binary_broadcast_xsimd_int_impl<std::int8_t, Batch8, kLanes8>(
        op, lhs, lhs_outer_stride, lhs_inner_stride,
        rhs, rhs_outer_stride, rhs_inner_stride,
        out, outer, inner);
}

void binary_broadcast_xsimd_uint8(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner)
{
    using BatchU8 = xsimd::batch<std::uint8_t>;
    constexpr size_t kLanesU8 = BatchU8::size;
    binary_broadcast_xsimd_int_impl<std::uint8_t, BatchU8, kLanesU8>(
        op, lhs, lhs_outer_stride, lhs_inner_stride,
        rhs, rhs_outer_stride, rhs_inner_stride,
        out, outer, inner);
}

void compare_broadcast_xsimd(CompareKernelOp op,
                             const float* lhs,
                             size_t lhs_outer_stride,
                             size_t lhs_inner_stride,
                             const float* rhs,
                             size_t rhs_outer_stride,
                             size_t rhs_inner_stride,
                             std::uint8_t* out,
                             size_t out_outer_stride,
                             size_t outer,
                             size_t inner)
{
    for_each_outer(outer, inner, [&](size_t outer_i) {
        const float* lhs_row = lhs + outer_i * lhs_outer_stride;
        const float* rhs_row = rhs + outer_i * rhs_outer_stride;
        std::uint8_t* out_row = out + outer_i * out_outer_stride;

        size_t inner_idx = 0;
        if (lhs_inner_stride <= 1 && rhs_inner_stride <= 1)
        {
            std::array<float, kLanes> tmp {};
            for (; inner_idx + kLanes <= inner; inner_idx += kLanes)
            {
                const Batch lhs_vec = lhs_inner_stride == 0
                    ? Batch(lhs_row[0])
                    : Batch::load_unaligned(lhs_row + inner_idx);
                const Batch rhs_vec = rhs_inner_stride == 0
                    ? Batch(rhs_row[0])
                    : Batch::load_unaligned(rhs_row + inner_idx);
                const Batch::batch_bool_type cmp_mask = apply_compare_batch(op, lhs_vec, rhs_vec);
                const Batch out_vec = xsimd::select(cmp_mask, Batch(255.0f), Batch(0.0f));
                out_vec.store_unaligned(tmp.data());

                for (size_t lane = 0; lane < kLanes; ++lane)
                {
                    out_row[inner_idx + lane] = static_cast<std::uint8_t>(tmp[lane]);
                }
            }
        }

        for (; inner_idx < inner; ++inner_idx)
        {
            const float lhs_val = lhs_row[inner_idx * lhs_inner_stride];
            const float rhs_val = rhs_row[inner_idx * rhs_inner_stride];
            out_row[inner_idx] = apply_compare_scalar(op, lhs_val, rhs_val) ? 255 : 0;
        }
    });
}

void compare_broadcast_xsimd_hfloat(CompareKernelOp op,
                                    const void* lhs,
                                    size_t lhs_outer_stride,
                                    size_t lhs_inner_stride,
                                    const void* rhs,
                                    size_t rhs_outer_stride,
                                    size_t rhs_inner_stride,
                                    std::uint8_t* out,
                                    size_t out_outer_stride,
                                    size_t outer,
                                    size_t inner)
{
    const hfloat* lhs_ptr = reinterpret_cast<const hfloat*>(lhs);
    const hfloat* rhs_ptr = reinterpret_cast<const hfloat*>(rhs);
    for_each_outer(outer, inner, [&](size_t outer_i) {
        const hfloat* lhs_row = lhs_ptr + outer_i * lhs_outer_stride;
        const hfloat* rhs_row = rhs_ptr + outer_i * rhs_outer_stride;
        std::uint8_t* out_row = out + outer_i * out_outer_stride;

        size_t inner_idx = 0;
        if (lhs_inner_stride <= 1 && rhs_inner_stride <= 1)
        {
            const Batch lhs_scalar = lhs_inner_stride == 0 ? Batch(static_cast<float>(lhs_row[0])) : Batch(0.0f);
            const Batch rhs_scalar = rhs_inner_stride == 0 ? Batch(static_cast<float>(rhs_row[0])) : Batch(0.0f);
            std::array<float, kLanes> tmp {};
            for (; inner_idx + kLanes <= inner; inner_idx += kLanes)
            {
                const Batch lhs_vec = lhs_inner_stride == 0 ? lhs_scalar : load_hfloat_batch(lhs_row + inner_idx);
                const Batch rhs_vec = rhs_inner_stride == 0 ? rhs_scalar : load_hfloat_batch(rhs_row + inner_idx);
                const Batch::batch_bool_type cmp_mask = apply_compare_batch(op, lhs_vec, rhs_vec);
                const Batch out_vec = xsimd::select(cmp_mask, Batch(255.0f), Batch(0.0f));
                out_vec.store_unaligned(tmp.data());

                for (size_t lane = 0; lane < kLanes; ++lane)
                {
                    out_row[inner_idx + lane] = static_cast<std::uint8_t>(tmp[lane]);
                }
            }
        }

        for (; inner_idx < inner; ++inner_idx)
        {
            const float lhs_val = lhs_inner_stride == 0 ? static_cast<float>(lhs_row[0])
                                                        : static_cast<float>(lhs_row[inner_idx * lhs_inner_stride]);
            const float rhs_val = rhs_inner_stride == 0 ? static_cast<float>(rhs_row[0])
                                                        : static_cast<float>(rhs_row[inner_idx * rhs_inner_stride]);
            out_row[inner_idx] = apply_compare_scalar(op, lhs_val, rhs_val) ? 255 : 0;
        }
    });
}

template<typename T, typename BatchType, size_t kLanesInt>
inline void compare_broadcast_xsimd_int_impl(CompareKernelOp op,
                                             const void* lhs,
                                             size_t lhs_outer_stride,
                                             size_t lhs_inner_stride,
                                             const void* rhs,
                                             size_t rhs_outer_stride,
                                             size_t rhs_inner_stride,
                                             std::uint8_t* out,
                                             size_t out_outer_stride,
                                             size_t outer,
                                             size_t inner)
{
    for_each_outer(outer, inner, [&](size_t outer_i) {
        const T* lhs_row = reinterpret_cast<const T*>(lhs) + outer_i * lhs_outer_stride;
        const T* rhs_row = reinterpret_cast<const T*>(rhs) + outer_i * rhs_outer_stride;
        std::uint8_t* out_row = out + outer_i * out_outer_stride;

        size_t inner_idx = 0;
        if (lhs_inner_stride <= 1 && rhs_inner_stride <= 1)
        {
            std::array<T, kLanesInt> tmp {};
            for (; inner_idx + kLanesInt <= inner; inner_idx += kLanesInt)
            {
                const BatchType lhs_vec = lhs_inner_stride == 0
                    ? BatchType(lhs_row[0])
                    : BatchType::load_unaligned(lhs_row + inner_idx);
                const BatchType rhs_vec = rhs_inner_stride == 0
                    ? BatchType(rhs_row[0])
                    : BatchType::load_unaligned(rhs_row + inner_idx);
                const auto cmp_mask = apply_compare_batch_typed(op, lhs_vec, rhs_vec);
                const BatchType out_vec = xsimd::select(cmp_mask, BatchType(T(255)), BatchType(T(0)));
                out_vec.store_unaligned(tmp.data());
                for (size_t lane = 0; lane < kLanesInt; ++lane)
                {
                    out_row[inner_idx + lane] = static_cast<std::uint8_t>(tmp[lane]);
                }
            }
        }

        for (; inner_idx < inner; ++inner_idx)
        {
            const T lhs_val = lhs_row[inner_idx * lhs_inner_stride];
            const T rhs_val = rhs_row[inner_idx * rhs_inner_stride];
            out_row[inner_idx] = apply_compare_scalar_typed(op, lhs_val, rhs_val) ? 255 : 0;
        }
    });
}

void compare_broadcast_xsimd_int32(CompareKernelOp op,
                                   const void* lhs,
                                   size_t lhs_outer_stride,
                                   size_t lhs_inner_stride,
                                   const void* rhs,
                                   size_t rhs_outer_stride,
                                   size_t rhs_inner_stride,
                                   std::uint8_t* out,
                                   size_t out_outer_stride,
                                   size_t outer,
                                   size_t inner)
{
    compare_broadcast_xsimd_int_impl<std::int32_t, Batch32, kLanes32>(
        op, lhs, lhs_outer_stride, lhs_inner_stride, rhs, rhs_outer_stride, rhs_inner_stride, out, out_outer_stride, outer, inner);
}

void compare_broadcast_xsimd_uint32(CompareKernelOp op,
                                    const void* lhs,
                                    size_t lhs_outer_stride,
                                    size_t lhs_inner_stride,
                                    const void* rhs,
                                    size_t rhs_outer_stride,
                                    size_t rhs_inner_stride,
                                    std::uint8_t* out,
                                    size_t out_outer_stride,
                                    size_t outer,
                                    size_t inner)
{
    compare_broadcast_xsimd_int_impl<std::uint32_t, BatchU32, kLanesU32>(
        op, lhs, lhs_outer_stride, lhs_inner_stride, rhs, rhs_outer_stride, rhs_inner_stride, out, out_outer_stride, outer, inner);
}

void compare_broadcast_xsimd_int16(CompareKernelOp op,
                                   const void* lhs,
                                   size_t lhs_outer_stride,
                                   size_t lhs_inner_stride,
                                   const void* rhs,
                                   size_t rhs_outer_stride,
                                   size_t rhs_inner_stride,
                                   std::uint8_t* out,
                                   size_t out_outer_stride,
                                   size_t outer,
                                   size_t inner)
{
    compare_broadcast_xsimd_int_impl<std::int16_t, Batch16, kLanes16>(
        op, lhs, lhs_outer_stride, lhs_inner_stride, rhs, rhs_outer_stride, rhs_inner_stride, out, out_outer_stride, outer, inner);
}

void compare_broadcast_xsimd_uint16(CompareKernelOp op,
                                    const void* lhs,
                                    size_t lhs_outer_stride,
                                    size_t lhs_inner_stride,
                                    const void* rhs,
                                    size_t rhs_outer_stride,
                                    size_t rhs_inner_stride,
                                    std::uint8_t* out,
                                    size_t out_outer_stride,
                                    size_t outer,
                                    size_t inner)
{
    using BatchU16 = xsimd::batch<std::uint16_t>;
    constexpr size_t kLanesU16 = BatchU16::size;
    compare_broadcast_xsimd_int_impl<std::uint16_t, BatchU16, kLanesU16>(
        op, lhs, lhs_outer_stride, lhs_inner_stride, rhs, rhs_outer_stride, rhs_inner_stride, out, out_outer_stride, outer, inner);
}

void compare_broadcast_xsimd_int8(CompareKernelOp op,
                                  const void* lhs,
                                  size_t lhs_outer_stride,
                                  size_t lhs_inner_stride,
                                  const void* rhs,
                                  size_t rhs_outer_stride,
                                  size_t rhs_inner_stride,
                                  std::uint8_t* out,
                                  size_t out_outer_stride,
                                  size_t outer,
                                  size_t inner)
{
    compare_broadcast_xsimd_int_impl<std::int8_t, Batch8, kLanes8>(
        op, lhs, lhs_outer_stride, lhs_inner_stride, rhs, rhs_outer_stride, rhs_inner_stride, out, out_outer_stride, outer, inner);
}

void compare_broadcast_xsimd_uint8(CompareKernelOp op,
                                   const void* lhs,
                                   size_t lhs_outer_stride,
                                   size_t lhs_inner_stride,
                                   const void* rhs,
                                   size_t rhs_outer_stride,
                                   size_t rhs_inner_stride,
                                   std::uint8_t* out,
                                   size_t out_outer_stride,
                                   size_t outer,
                                   size_t inner)
{
    using BatchU8 = xsimd::batch<std::uint8_t>;
    constexpr size_t kLanesU8 = BatchU8::size;
    compare_broadcast_xsimd_int_impl<std::uint8_t, BatchU8, kLanesU8>(
        op, lhs, lhs_outer_stride, lhs_inner_stride, rhs, rhs_outer_stride, rhs_inner_stride, out, out_outer_stride, outer, inner);
}

inline void build_phase_batches(const float* scalar_lanes, int channels, std::array<Batch, 4>& phase_batches)
{
    const size_t phase_count = static_cast<size_t>(channels);
    std::array<float, kLanes> lane_pattern {};
    for (size_t phase = 0; phase < phase_count; ++phase)
    {
        for (size_t lane = 0; lane < kLanes; ++lane)
        {
            lane_pattern[lane] = scalar_lanes[(phase + lane) % phase_count];
        }
        phase_batches[phase] = Batch::load_unaligned(lane_pattern.data());
    }
}

template<typename T, typename BatchType>
inline void build_phase_batches_typed(const T* scalar_lanes, int channels, std::array<BatchType, 4>& phase_batches)
{
    const size_t phase_count = static_cast<size_t>(channels);
    std::array<T, BatchType::size> lane_pattern {};
    for (size_t phase = 0; phase < phase_count; ++phase)
    {
        for (size_t lane = 0; lane < BatchType::size; ++lane)
        {
            lane_pattern[lane] = scalar_lanes[(phase + lane) % phase_count];
        }
        phase_batches[phase] = BatchType::load_unaligned(lane_pattern.data());
    }
}

void binary_scalar_channels_xsimd(BinaryKernelOp op,
                                  const float* src,
                                  size_t src_outer_stride,
                                  const float* scalar_lanes,
                                  int channels,
                                  float* out,
                                  size_t out_outer_stride,
                                  size_t outer,
                                  size_t inner,
                                  bool scalar_first)
{
    if (channels <= 0 || channels > 4 || outer == 0 || inner == 0)
    {
        return;
    }

    std::array<Batch, 4> phase_batches;
    build_phase_batches(scalar_lanes, channels, phase_batches);
    const size_t phase_count = static_cast<size_t>(channels);
    const size_t vec_phase_step = kLanes % phase_count;
    for_each_outer(outer, inner, [&](size_t outer_i) {
        const float* src_row = src + outer_i * src_outer_stride;
        float* out_row = out + outer_i * out_outer_stride;

        size_t inner_idx = 0;
        size_t phase = 0;
        for (; inner_idx + kLanes <= inner; inner_idx += kLanes)
        {
            const Batch src_vec = Batch::load_unaligned(src_row + inner_idx);
            const Batch& scalar_vec = phase_batches[phase];
            const Batch out_vec = scalar_first ? apply_batch(op, scalar_vec, src_vec)
                                               : apply_batch(op, src_vec, scalar_vec);
            out_vec.store_unaligned(out_row + inner_idx);
            if (vec_phase_step != 0)
            {
                phase += vec_phase_step;
                if (phase >= phase_count)
                {
                    phase -= phase_count;
                }
            }
        }

        for (; inner_idx < inner; ++inner_idx)
        {
            const float scalar_val = scalar_lanes[phase];
            out_row[inner_idx] = scalar_first ? apply_scalar(op, scalar_val, src_row[inner_idx])
                                              : apply_scalar(op, src_row[inner_idx], scalar_val);
            ++phase;
            if (phase == phase_count)
            {
                phase = 0;
            }
        }
    });
}

void binary_scalar_channels_xsimd_hfloat(BinaryKernelOp op,
                                         const void* src,
                                         size_t src_outer_stride,
                                         const float* scalar_lanes,
                                         int channels,
                                         void* out,
                                         size_t out_outer_stride,
                                         size_t outer,
                                         size_t inner,
                                         bool scalar_first)
{
    if (channels <= 0 || channels > 4 || outer == 0 || inner == 0)
    {
        return;
    }

    const hfloat* src_ptr = reinterpret_cast<const hfloat*>(src);
    hfloat* out_ptr = reinterpret_cast<hfloat*>(out);
    std::array<Batch, 4> phase_batches;
    build_phase_batches(scalar_lanes, channels, phase_batches);
    const size_t phase_count = static_cast<size_t>(channels);
    const size_t vec_phase_step = kLanes % phase_count;
    for_each_outer(outer, inner, [&](size_t outer_i) {
        const hfloat* src_row = src_ptr + outer_i * src_outer_stride;
        hfloat* out_row = out_ptr + outer_i * out_outer_stride;

        size_t inner_idx = 0;
        size_t phase = 0;
        for (; inner_idx + kLanes <= inner; inner_idx += kLanes)
        {
            const Batch src_vec = load_hfloat_batch(src_row + inner_idx);
            const Batch& scalar_vec = phase_batches[phase];
            const Batch out_vec = scalar_first ? apply_batch(op, scalar_vec, src_vec)
                                               : apply_batch(op, src_vec, scalar_vec);
            store_hfloat_batch(out_vec, out_row + inner_idx);
            if (vec_phase_step != 0)
            {
                phase += vec_phase_step;
                if (phase >= phase_count)
                {
                    phase -= phase_count;
                }
            }
        }

        for (; inner_idx < inner; ++inner_idx)
        {
            const float scalar_val = scalar_lanes[phase];
            const float src_val = static_cast<float>(src_row[inner_idx]);
            const float out_val = scalar_first ? apply_scalar(op, scalar_val, src_val)
                                               : apply_scalar(op, src_val, scalar_val);
            out_row[inner_idx] = hfloat(out_val);
            ++phase;
            if (phase == phase_count)
            {
                phase = 0;
            }
        }
    });
}

void binary_scalar_channels_xsimd_int32(BinaryKernelOp op,
                                        const void* src,
                                        size_t src_outer_stride,
                                        const std::int32_t* scalar_lanes,
                                        int channels,
                                        void* out,
                                        size_t out_outer_stride,
                                        size_t outer,
                                        size_t inner,
                                        bool scalar_first)
{
    if (channels <= 0 || channels > 4 || outer == 0 || inner == 0)
    {
        return;
    }

    const std::int32_t* src_ptr = reinterpret_cast<const std::int32_t*>(src);
    std::int32_t* out_ptr = reinterpret_cast<std::int32_t*>(out);
    std::array<Batch32, 4> phase_batches;
    build_phase_batches_typed<std::int32_t, Batch32>(scalar_lanes, channels, phase_batches);
    const size_t phase_count = static_cast<size_t>(channels);
    const size_t vec_phase_step = kLanes32 % phase_count;
    for_each_outer(outer, inner, [&](size_t outer_i) {
        const std::int32_t* src_row = src_ptr + outer_i * src_outer_stride;
        std::int32_t* out_row = out_ptr + outer_i * out_outer_stride;

        size_t inner_idx = 0;
        size_t phase = 0;
        for (; inner_idx + kLanes32 <= inner; inner_idx += kLanes32)
        {
            const Batch32 src_vec = Batch32::load_unaligned(src_row + inner_idx);
            const Batch32& scalar_vec = phase_batches[phase];
            const Batch32 out_vec = scalar_first ? apply_batch_i32(op, scalar_vec, src_vec)
                                                 : apply_batch_i32(op, src_vec, scalar_vec);
            out_vec.store_unaligned(out_row + inner_idx);
            if (vec_phase_step != 0)
            {
                phase += vec_phase_step;
                if (phase >= phase_count)
                {
                    phase -= phase_count;
                }
            }
        }

        for (; inner_idx < inner; ++inner_idx)
        {
            const std::int32_t scalar_val = scalar_lanes[phase];
            out_row[inner_idx] = scalar_first ? apply_scalar_i32(op, scalar_val, src_row[inner_idx])
                                              : apply_scalar_i32(op, src_row[inner_idx], scalar_val);
            ++phase;
            if (phase == phase_count)
            {
                phase = 0;
            }
        }
    });
}

void binary_scalar_channels_xsimd_uint32(BinaryKernelOp op,
                                         const void* src,
                                         size_t src_outer_stride,
                                         const std::uint32_t* scalar_lanes,
                                         int channels,
                                         void* out,
                                         size_t out_outer_stride,
                                         size_t outer,
                                         size_t inner,
                                         bool scalar_first)
{
    if (channels <= 0 || channels > 4 || outer == 0 || inner == 0)
    {
        return;
    }

    const std::uint32_t* src_ptr = reinterpret_cast<const std::uint32_t*>(src);
    std::uint32_t* out_ptr = reinterpret_cast<std::uint32_t*>(out);
    std::array<BatchU32, 4> phase_batches;
    build_phase_batches_typed<std::uint32_t, BatchU32>(scalar_lanes, channels, phase_batches);
    const size_t phase_count = static_cast<size_t>(channels);
    const size_t vec_phase_step = kLanesU32 % phase_count;
    for_each_outer(outer, inner, [&](size_t outer_i) {
        const std::uint32_t* src_row = src_ptr + outer_i * src_outer_stride;
        std::uint32_t* out_row = out_ptr + outer_i * out_outer_stride;

        size_t inner_idx = 0;
        size_t phase = 0;
        for (; inner_idx + kLanesU32 <= inner; inner_idx += kLanesU32)
        {
            const BatchU32 src_vec = BatchU32::load_unaligned(src_row + inner_idx);
            const BatchU32& scalar_vec = phase_batches[phase];
            const BatchU32 out_vec = scalar_first ? apply_batch_u32(op, scalar_vec, src_vec)
                                                  : apply_batch_u32(op, src_vec, scalar_vec);
            out_vec.store_unaligned(out_row + inner_idx);
            if (vec_phase_step != 0)
            {
                phase += vec_phase_step;
                if (phase >= phase_count)
                {
                    phase -= phase_count;
                }
            }
        }

        for (; inner_idx < inner; ++inner_idx)
        {
            const std::uint32_t scalar_val = scalar_lanes[phase];
            out_row[inner_idx] = scalar_first ? apply_scalar_u32(op, scalar_val, src_row[inner_idx])
                                              : apply_scalar_u32(op, src_row[inner_idx], scalar_val);
            ++phase;
            if (phase == phase_count)
            {
                phase = 0;
            }
        }
    });
}

template<typename T, typename BatchType>
inline T apply_scalar_smallint(BinaryKernelOp op, T lhs, T rhs)
{
    switch (op)
    {
        case BinaryKernelOp::Add:
            return saturate_cast<T>(static_cast<int64>(lhs) + static_cast<int64>(rhs));
        case BinaryKernelOp::Sub:
            return saturate_cast<T>(static_cast<int64>(lhs) - static_cast<int64>(rhs));
        case BinaryKernelOp::Mul:
            return lhs * rhs;
        case BinaryKernelOp::Div:
            return rhs != 0 ? static_cast<T>(lhs / rhs) : static_cast<T>(0);
        case BinaryKernelOp::Max:
            return lhs > rhs ? lhs : rhs;
        case BinaryKernelOp::Min:
            return lhs < rhs ? lhs : rhs;
        case BinaryKernelOp::Mean:
            return static_cast<T>((static_cast<long long>(lhs) + static_cast<long long>(rhs)) / 2);
        default:
            return static_cast<T>(0);
    }
}

template<typename T, typename BatchType, size_t kLanesInt>
inline BatchType apply_batch_smallint(BinaryKernelOp op, const BatchType& lhs, const BatchType& rhs)
{
    switch (op)
    {
        case BinaryKernelOp::Add:
            return xsimd::sadd(lhs, rhs);
        case BinaryKernelOp::Sub:
            return xsimd::ssub(lhs, rhs);
        case BinaryKernelOp::Mul:
            return lhs * rhs;
        case BinaryKernelOp::Div:
            return lhs / rhs;
        case BinaryKernelOp::Max:
            return xsimd::max(lhs, rhs);
        case BinaryKernelOp::Min:
            return xsimd::min(lhs, rhs);
        case BinaryKernelOp::Mean:
            return (lhs + rhs) / BatchType(2);
        default:
            return BatchType(0);
    }
}

template<typename T, typename BatchType, size_t kLanesInt>
inline void store_saturated_mul_smallint_batch(const BatchType& lhs_vec, const BatchType& rhs_vec, T* out_ptr)
{
    const auto lhs_wide = xsimd::widen(lhs_vec);
    const auto rhs_wide = xsimd::widen(rhs_vec);
    using WideBatch = typename decltype(lhs_wide)::value_type;
    using WideT = xsimd::widen_t<T>;
    constexpr size_t kWideLanes = WideBatch::size;
    std::array<WideT, kLanesInt> tmp {};
    (lhs_wide[0] * rhs_wide[0]).store_unaligned(tmp.data());
    (lhs_wide[1] * rhs_wide[1]).store_unaligned(tmp.data() + kWideLanes);
    for (size_t lane = 0; lane < kLanesInt; ++lane)
    {
        out_ptr[lane] = saturate_cast<T>(tmp[lane]);
    }
}

template<typename T, typename BatchType, size_t kLanesInt>
inline void binary_scalar_channels_xsimd_smallint_impl(BinaryKernelOp op,
                                                       const void* src,
                                                       size_t src_outer_stride,
                                                       const T* scalar_lanes,
                                                       int channels,
                                                       void* out,
                                                       size_t out_outer_stride,
                                                       size_t outer,
                                                       size_t inner,
                                                       bool scalar_first)
{
    if (channels <= 0 || channels > 4 || outer == 0 || inner == 0)
    {
        return;
    }

    const T* src_ptr = reinterpret_cast<const T*>(src);
    T* out_ptr = reinterpret_cast<T*>(out);
    std::array<BatchType, 4> phase_batches;
    build_phase_batches_typed<T, BatchType>(scalar_lanes, channels, phase_batches);
    const size_t phase_count = static_cast<size_t>(channels);
    const size_t vec_phase_step = kLanesInt % phase_count;
    for_each_outer(outer, inner, [&](size_t outer_i) {
        const T* src_row = src_ptr + outer_i * src_outer_stride;
        T* out_row = out_ptr + outer_i * out_outer_stride;

        size_t inner_idx = 0;
        size_t phase = 0;
        for (; inner_idx + kLanesInt <= inner; inner_idx += kLanesInt)
        {
            const BatchType src_vec = BatchType::load_unaligned(src_row + inner_idx);
            const BatchType& scalar_vec = phase_batches[phase];
            if (op == BinaryKernelOp::Mul)
            {
                if (scalar_first)
                {
                    store_saturated_mul_smallint_batch<T, BatchType, kLanesInt>(scalar_vec, src_vec, out_row + inner_idx);
                }
                else
                {
                    store_saturated_mul_smallint_batch<T, BatchType, kLanesInt>(src_vec, scalar_vec, out_row + inner_idx);
                }
            }
            else
            {
                const BatchType out_vec = scalar_first ? apply_batch_smallint<T, BatchType, kLanesInt>(op, scalar_vec, src_vec)
                                                       : apply_batch_smallint<T, BatchType, kLanesInt>(op, src_vec, scalar_vec);
                out_vec.store_unaligned(out_row + inner_idx);
            }
            if (vec_phase_step != 0)
            {
                phase += vec_phase_step;
                if (phase >= phase_count)
                {
                    phase -= phase_count;
                }
            }
        }

        for (; inner_idx < inner; ++inner_idx)
        {
            const T scalar_val = scalar_lanes[phase];
            out_row[inner_idx] = scalar_first ? apply_scalar_smallint<T, BatchType>(op, scalar_val, src_row[inner_idx])
                                              : apply_scalar_smallint<T, BatchType>(op, src_row[inner_idx], scalar_val);
            ++phase;
            if (phase == phase_count)
            {
                phase = 0;
            }
        }
    });
}

void binary_scalar_channels_xsimd_int16(BinaryKernelOp op,
                                        const void* src,
                                        size_t src_outer_stride,
                                        const std::int16_t* scalar_lanes,
                                        int channels,
                                        void* out,
                                        size_t out_outer_stride,
                                        size_t outer,
                                        size_t inner,
                                        bool scalar_first)
{
    binary_scalar_channels_xsimd_smallint_impl<std::int16_t, Batch16, kLanes16>(
        op, src, src_outer_stride, scalar_lanes, channels, out, out_outer_stride, outer, inner, scalar_first);
}

void binary_scalar_channels_xsimd_uint16(BinaryKernelOp op,
                                         const void* src,
                                         size_t src_outer_stride,
                                         const std::uint16_t* scalar_lanes,
                                         int channels,
                                         void* out,
                                         size_t out_outer_stride,
                                         size_t outer,
                                         size_t inner,
                                         bool scalar_first)
{
    using BatchU16 = xsimd::batch<std::uint16_t>;
    constexpr size_t kLanesU16 = BatchU16::size;
    binary_scalar_channels_xsimd_smallint_impl<std::uint16_t, BatchU16, kLanesU16>(
        op, src, src_outer_stride, scalar_lanes, channels, out, out_outer_stride, outer, inner, scalar_first);
}

void binary_scalar_channels_xsimd_int8(BinaryKernelOp op,
                                       const void* src,
                                       size_t src_outer_stride,
                                       const std::int8_t* scalar_lanes,
                                       int channels,
                                       void* out,
                                       size_t out_outer_stride,
                                       size_t outer,
                                       size_t inner,
                                       bool scalar_first)
{
    binary_scalar_channels_xsimd_smallint_impl<std::int8_t, Batch8, kLanes8>(
        op, src, src_outer_stride, scalar_lanes, channels, out, out_outer_stride, outer, inner, scalar_first);
}

void binary_scalar_channels_xsimd_uint8(BinaryKernelOp op,
                                        const void* src,
                                        size_t src_outer_stride,
                                        const std::uint8_t* scalar_lanes,
                                        int channels,
                                        void* out,
                                        size_t out_outer_stride,
                                        size_t outer,
                                        size_t inner,
                                        bool scalar_first)
{
    using BatchU8 = xsimd::batch<std::uint8_t>;
    constexpr size_t kLanesU8 = BatchU8::size;
    binary_scalar_channels_xsimd_smallint_impl<std::uint8_t, BatchU8, kLanesU8>(
        op, src, src_outer_stride, scalar_lanes, channels, out, out_outer_stride, outer, inner, scalar_first);
}

void compare_scalar_channels_xsimd(CompareKernelOp op,
                                   const float* src,
                                   size_t src_outer_stride,
                                   const float* scalar_lanes,
                                   int channels,
                                   std::uint8_t* out,
                                   size_t out_outer_stride,
                                   size_t outer,
                                   size_t inner,
                                   bool scalar_first)
{
    if (channels <= 0 || channels > 4 || outer == 0 || inner == 0)
    {
        return;
    }

    std::array<Batch, 4> phase_batches;
    build_phase_batches(scalar_lanes, channels, phase_batches);
    const size_t phase_count = static_cast<size_t>(channels);
    const size_t vec_phase_step = kLanes % phase_count;
    for_each_outer(outer, inner, [&](size_t outer_i) {
        const float* src_row = src + outer_i * src_outer_stride;
        std::uint8_t* out_row = out + outer_i * out_outer_stride;

        size_t inner_idx = 0;
        size_t phase = 0;
        std::array<float, kLanes> tmp {};
        for (; inner_idx + kLanes <= inner; inner_idx += kLanes)
        {
            const Batch src_vec = Batch::load_unaligned(src_row + inner_idx);
            const Batch& scalar_vec = phase_batches[phase];
            const Batch::batch_bool_type mask = scalar_first ? apply_compare_batch(op, scalar_vec, src_vec)
                                                             : apply_compare_batch(op, src_vec, scalar_vec);
            const Batch out_vec = xsimd::select(mask, Batch(255.0f), Batch(0.0f));
            out_vec.store_unaligned(tmp.data());
            for (size_t lane = 0; lane < kLanes; ++lane)
            {
                out_row[inner_idx + lane] = static_cast<std::uint8_t>(tmp[lane]);
            }
            if (vec_phase_step != 0)
            {
                phase += vec_phase_step;
                if (phase >= phase_count)
                {
                    phase -= phase_count;
                }
            }
        }

        for (; inner_idx < inner; ++inner_idx)
        {
            const float scalar_val = scalar_lanes[phase];
            out_row[inner_idx] = scalar_first ? (apply_compare_scalar(op, scalar_val, src_row[inner_idx]) ? 255 : 0)
                                              : (apply_compare_scalar(op, src_row[inner_idx], scalar_val) ? 255 : 0);
            ++phase;
            if (phase == phase_count)
            {
                phase = 0;
            }
        }
    });
}

void compare_scalar_channels_xsimd_hfloat(CompareKernelOp op,
                                          const void* src,
                                          size_t src_outer_stride,
                                          const float* scalar_lanes,
                                          int channels,
                                          std::uint8_t* out,
                                          size_t out_outer_stride,
                                          size_t outer,
                                          size_t inner,
                                          bool scalar_first)
{
    if (channels <= 0 || channels > 4 || outer == 0 || inner == 0)
    {
        return;
    }

    const hfloat* src_ptr = reinterpret_cast<const hfloat*>(src);
    std::array<Batch, 4> phase_batches;
    build_phase_batches(scalar_lanes, channels, phase_batches);
    const size_t phase_count = static_cast<size_t>(channels);
    const size_t vec_phase_step = kLanes % phase_count;
    for_each_outer(outer, inner, [&](size_t outer_i) {
        const hfloat* src_row = src_ptr + outer_i * src_outer_stride;
        std::uint8_t* out_row = out + outer_i * out_outer_stride;

        size_t inner_idx = 0;
        size_t phase = 0;
        std::array<float, kLanes> tmp {};
        for (; inner_idx + kLanes <= inner; inner_idx += kLanes)
        {
            const Batch src_vec = load_hfloat_batch(src_row + inner_idx);
            const Batch& scalar_vec = phase_batches[phase];
            const Batch::batch_bool_type mask = scalar_first ? apply_compare_batch(op, scalar_vec, src_vec)
                                                             : apply_compare_batch(op, src_vec, scalar_vec);
            const Batch out_vec = xsimd::select(mask, Batch(255.0f), Batch(0.0f));
            out_vec.store_unaligned(tmp.data());
            for (size_t lane = 0; lane < kLanes; ++lane)
            {
                out_row[inner_idx + lane] = static_cast<std::uint8_t>(tmp[lane]);
            }
            if (vec_phase_step != 0)
            {
                phase += vec_phase_step;
                if (phase >= phase_count)
                {
                    phase -= phase_count;
                }
            }
        }

        for (; inner_idx < inner; ++inner_idx)
        {
            const float scalar_val = scalar_lanes[phase];
            const float src_val = static_cast<float>(src_row[inner_idx]);
            out_row[inner_idx] = scalar_first ? (apply_compare_scalar(op, scalar_val, src_val) ? 255 : 0)
                                              : (apply_compare_scalar(op, src_val, scalar_val) ? 255 : 0);
            ++phase;
            if (phase == phase_count)
            {
                phase = 0;
            }
        }
    });
}

template<typename T, typename BatchType, size_t kLanesInt>
inline void compare_scalar_channels_xsimd_int_impl(CompareKernelOp op,
                                                   const void* src,
                                                   size_t src_outer_stride,
                                                   const T* scalar_lanes,
                                                   int channels,
                                                   std::uint8_t* out,
                                                   size_t out_outer_stride,
                                                   size_t outer,
                                                   size_t inner,
                                                   bool scalar_first)
{
    if (channels <= 0 || channels > 4 || outer == 0 || inner == 0)
    {
        return;
    }

    const T* src_ptr = reinterpret_cast<const T*>(src);
    std::array<BatchType, 4> phase_batches;
    build_phase_batches_typed<T, BatchType>(scalar_lanes, channels, phase_batches);
    const size_t phase_count = static_cast<size_t>(channels);
    const size_t vec_phase_step = kLanesInt % phase_count;
    for_each_outer(outer, inner, [&](size_t outer_i) {
        const T* src_row = src_ptr + outer_i * src_outer_stride;
        std::uint8_t* out_row = out + outer_i * out_outer_stride;

        size_t inner_idx = 0;
        size_t phase = 0;
        std::array<T, kLanesInt> tmp {};
        for (; inner_idx + kLanesInt <= inner; inner_idx += kLanesInt)
        {
            const BatchType src_vec = BatchType::load_unaligned(src_row + inner_idx);
            const BatchType& scalar_vec = phase_batches[phase];
            const auto cmp_mask = scalar_first ? apply_compare_batch_typed(op, scalar_vec, src_vec)
                                               : apply_compare_batch_typed(op, src_vec, scalar_vec);
            const BatchType out_vec = xsimd::select(cmp_mask, BatchType(T(255)), BatchType(T(0)));
            out_vec.store_unaligned(tmp.data());
            for (size_t lane = 0; lane < kLanesInt; ++lane)
            {
                out_row[inner_idx + lane] = static_cast<std::uint8_t>(tmp[lane]);
            }
            if (vec_phase_step != 0)
            {
                phase += vec_phase_step;
                if (phase >= phase_count)
                {
                    phase -= phase_count;
                }
            }
        }

        for (; inner_idx < inner; ++inner_idx)
        {
            const T scalar_val = scalar_lanes[phase];
            const T src_val = src_row[inner_idx];
            out_row[inner_idx] = scalar_first
                ? (apply_compare_scalar_typed(op, scalar_val, src_val) ? 255 : 0)
                : (apply_compare_scalar_typed(op, src_val, scalar_val) ? 255 : 0);
            ++phase;
            if (phase == phase_count)
            {
                phase = 0;
            }
        }
    });
}

void compare_scalar_channels_xsimd_int32(CompareKernelOp op,
                                         const void* src,
                                         size_t src_outer_stride,
                                         const std::int32_t* scalar_lanes,
                                         int channels,
                                         std::uint8_t* out,
                                         size_t out_outer_stride,
                                         size_t outer,
                                         size_t inner,
                                         bool scalar_first)
{
    compare_scalar_channels_xsimd_int_impl<std::int32_t, Batch32, kLanes32>(
        op, src, src_outer_stride, scalar_lanes, channels, out, out_outer_stride, outer, inner, scalar_first);
}

void compare_scalar_channels_xsimd_uint32(CompareKernelOp op,
                                          const void* src,
                                          size_t src_outer_stride,
                                          const std::uint32_t* scalar_lanes,
                                          int channels,
                                          std::uint8_t* out,
                                          size_t out_outer_stride,
                                          size_t outer,
                                          size_t inner,
                                          bool scalar_first)
{
    compare_scalar_channels_xsimd_int_impl<std::uint32_t, BatchU32, kLanesU32>(
        op, src, src_outer_stride, scalar_lanes, channels, out, out_outer_stride, outer, inner, scalar_first);
}

void compare_scalar_channels_xsimd_int16(CompareKernelOp op,
                                         const void* src,
                                         size_t src_outer_stride,
                                         const std::int16_t* scalar_lanes,
                                         int channels,
                                         std::uint8_t* out,
                                         size_t out_outer_stride,
                                         size_t outer,
                                         size_t inner,
                                         bool scalar_first)
{
    compare_scalar_channels_xsimd_int_impl<std::int16_t, Batch16, kLanes16>(
        op, src, src_outer_stride, scalar_lanes, channels, out, out_outer_stride, outer, inner, scalar_first);
}

void compare_scalar_channels_xsimd_uint16(CompareKernelOp op,
                                          const void* src,
                                          size_t src_outer_stride,
                                          const std::uint16_t* scalar_lanes,
                                          int channels,
                                          std::uint8_t* out,
                                          size_t out_outer_stride,
                                          size_t outer,
                                          size_t inner,
                                          bool scalar_first)
{
    using BatchU16 = xsimd::batch<std::uint16_t>;
    constexpr size_t kLanesU16 = BatchU16::size;
    compare_scalar_channels_xsimd_int_impl<std::uint16_t, BatchU16, kLanesU16>(
        op, src, src_outer_stride, scalar_lanes, channels, out, out_outer_stride, outer, inner, scalar_first);
}

void compare_scalar_channels_xsimd_int8(CompareKernelOp op,
                                        const void* src,
                                        size_t src_outer_stride,
                                        const std::int8_t* scalar_lanes,
                                        int channels,
                                        std::uint8_t* out,
                                        size_t out_outer_stride,
                                        size_t outer,
                                        size_t inner,
                                        bool scalar_first)
{
    compare_scalar_channels_xsimd_int_impl<std::int8_t, Batch8, kLanes8>(
        op, src, src_outer_stride, scalar_lanes, channels, out, out_outer_stride, outer, inner, scalar_first);
}

void compare_scalar_channels_xsimd_uint8(CompareKernelOp op,
                                         const void* src,
                                         size_t src_outer_stride,
                                         const std::uint8_t* scalar_lanes,
                                         int channels,
                                         std::uint8_t* out,
                                         size_t out_outer_stride,
                                         size_t outer,
                                         size_t inner,
                                         bool scalar_first)
{
    using BatchU8 = xsimd::batch<std::uint8_t>;
    constexpr size_t kLanesU8 = BatchU8::size;
    compare_scalar_channels_xsimd_int_impl<std::uint8_t, BatchU8, kLanesU8>(
        op, src, src_outer_stride, scalar_lanes, channels, out, out_outer_stride, outer, inner, scalar_first);
}

}  // namespace cpu
}  // namespace cvh
