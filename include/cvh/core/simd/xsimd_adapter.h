#ifndef CVH_CORE_SIMD_XSIMD_ADAPTER_H
#define CVH_CORE_SIMD_XSIMD_ADAPTER_H

#include "scalar_adapter.h"

#include "cvh/3rdparty/xsimd/include/xsimd/xsimd.hpp"

#include <cstddef>
#include <cstdint>

namespace cvh {
namespace detail {
namespace simd {
namespace xsimd_adapter {

using f32 = xsimd::batch<float>;
using u8 = scalar::u8;
using u16 = scalar::u16;
using u32 = scalar::u32;

inline const char* backend_name()
{
    return "xsimd";
}

inline f32 setzero_f32()
{
    return f32(0.0f);
}

inline f32 setall_f32(float value)
{
    return f32(value);
}

inline f32 load_f32(const float* src)
{
    return f32::load_unaligned(src);
}

inline void store_f32(float* dst, const f32& value)
{
    value.store_unaligned(dst);
}

inline u32 setall_u32(std::uint32_t value)
{
    return scalar::setall_u32(value);
}

inline u16 setall_u16(std::uint16_t value)
{
    return scalar::setall_u16(value);
}

inline u8 load_u8(const std::uint8_t* src)
{
    return scalar::load_u8(src);
}

inline void load_deinterleave2_u8(const std::uint8_t* src, u8& c0, u8& c1)
{
    scalar::load_deinterleave2_u8(src, c0, c1);
}

inline void load_deinterleave3_u8(const std::uint8_t* src, u8& c0, u8& c1, u8& c2)
{
    scalar::load_deinterleave3_u8(src, c0, c1, c2);
}

inline void store_u8(std::uint8_t* dst, const u8& value)
{
    scalar::store_u8(dst, value);
}

inline void expand_u8(const u8& value, u16& low, u16& high)
{
    scalar::expand_u8(value, low, high);
}

inline u32 expand_low_u16(const u16& value)
{
    return scalar::expand_low_u16(value);
}

inline u32 expand_high_u16(const u16& value)
{
    return scalar::expand_high_u16(value);
}

inline f32 add(const f32& lhs, const f32& rhs)
{
    return lhs + rhs;
}

inline u16 add(const u16& lhs, const u16& rhs)
{
    return scalar::add(lhs, rhs);
}

inline u32 add(const u32& lhs, const u32& rhs)
{
    return scalar::add(lhs, rhs);
}

inline f32 sub(const f32& lhs, const f32& rhs)
{
    return lhs - rhs;
}

inline f32 mul(const f32& lhs, const f32& rhs)
{
    return lhs * rhs;
}

inline u32 mul(const u32& lhs, const u32& rhs)
{
    return scalar::mul(lhs, rhs);
}

inline void mul_expand_u16(const u16& lhs, const u16& rhs, u32& low, u32& high)
{
    scalar::mul_expand_u16(lhs, rhs, low, high);
}

inline f32 min(const f32& lhs, const f32& rhs)
{
    return xsimd::min(lhs, rhs);
}

inline f32 max(const f32& lhs, const f32& rhs)
{
    return xsimd::max(lhs, rhs);
}

inline float reduce_sum(const f32& value)
{
    return xsimd::reduce_add(value);
}

inline constexpr std::size_t f32_lanes()
{
    return f32::size;
}

template <int shift>
inline u16 rshr_pack_u32_to_u16(const u32& low, const u32& high)
{
    return scalar::rshr_pack_u32_to_u16<shift>(low, high);
}

template <int shift>
inline u8 rshr_pack_u16_to_u8(const u16& low, const u16& high)
{
    return scalar::rshr_pack_u16_to_u8<shift>(low, high);
}

inline u8 pack_u16_to_u8(const u16& low, const u16& high)
{
    return scalar::pack_u16_to_u8(low, high);
}

inline constexpr std::size_t u8_lanes()
{
    return u8::lanes;
}

}  // namespace xsimd_adapter
}  // namespace simd
}  // namespace detail
}  // namespace cvh

#endif  // CVH_CORE_SIMD_XSIMD_ADAPTER_H
