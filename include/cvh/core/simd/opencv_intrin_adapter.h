#ifndef CVH_CORE_SIMD_OPENCV_INTRIN_ADAPTER_H
#define CVH_CORE_SIMD_OPENCV_INTRIN_ADAPTER_H

#include "cvh/detail/config.h"

#if CVH_ENABLE_OPENCV_INTRIN
#include "cvh/3rdparty/opencv_intrin/opencv2/core/hal/intrin.hpp"

#include <cstddef>
#include <cstdint>

namespace cvh {
namespace detail {
namespace simd {
namespace opencv_intrin {

using f32 = cv::v_float32x4;
using u8 = cv::v_uint8x16;
using u16 = cv::v_uint16x8;
using u32 = cv::v_uint32x4;

inline const char* backend_name()
{
    return "opencv_intrin";
}

inline f32 setzero_f32()
{
    return cv::v_setzero_f32();
}

inline f32 setall_f32(float value)
{
    return cv::v_setall_f32(value);
}

inline f32 load_f32(const float* src)
{
    return cv::v_load(src);
}

inline void store_f32(float* dst, const f32& value)
{
    cv::v_store(dst, value);
}

inline u32 setall_u32(std::uint32_t value)
{
    return cv::v_setall_u32(static_cast<unsigned>(value));
}

inline u16 setall_u16(std::uint16_t value)
{
    return cv::v_setall_u16(static_cast<ushort>(value));
}

inline void load_deinterleave3_u8(const std::uint8_t* src, u8& c0, u8& c1, u8& c2)
{
    cv::v_load_deinterleave(reinterpret_cast<const uchar*>(src), c0, c1, c2);
}

inline void store_u8(std::uint8_t* dst, const u8& value)
{
    cv::v_store(reinterpret_cast<uchar*>(dst), value);
}

inline void expand_u8(const u8& value, u16& low, u16& high)
{
    cv::v_expand(value, low, high);
}

inline u32 expand_low_u16(const u16& value)
{
    return cv::v_expand_low(value);
}

inline u32 expand_high_u16(const u16& value)
{
    return cv::v_expand_high(value);
}

inline f32 add(const f32& lhs, const f32& rhs)
{
    return cv::v_add(lhs, rhs);
}

inline u32 add(const u32& lhs, const u32& rhs)
{
    return cv::v_add(lhs, rhs);
}

inline f32 sub(const f32& lhs, const f32& rhs)
{
    return cv::v_sub(lhs, rhs);
}

inline f32 mul(const f32& lhs, const f32& rhs)
{
    return cv::v_mul(lhs, rhs);
}

inline u32 mul(const u32& lhs, const u32& rhs)
{
    return cv::v_mul(lhs, rhs);
}

inline void mul_expand_u16(const u16& lhs, const u16& rhs, u32& low, u32& high)
{
    cv::v_mul_expand(lhs, rhs, low, high);
}

inline f32 min(const f32& lhs, const f32& rhs)
{
    return cv::v_min(lhs, rhs);
}

inline f32 max(const f32& lhs, const f32& rhs)
{
    return cv::v_max(lhs, rhs);
}

inline float reduce_sum(const f32& value)
{
    return cv::v_reduce_sum(value);
}

inline constexpr std::size_t f32_lanes()
{
    return static_cast<std::size_t>(cv::VTraits<f32>::nlanes);
}

template <int shift>
inline u16 rshr_pack_u32_to_u16(const u32& low, const u32& high)
{
    return cv::v_rshr_pack<shift>(low, high);
}

inline u8 pack_u16_to_u8(const u16& low, const u16& high)
{
    return cv::v_pack(low, high);
}

inline constexpr std::size_t u8_lanes()
{
    return static_cast<std::size_t>(cv::VTraits<u8>::nlanes);
}

}  // namespace opencv_intrin
}  // namespace simd
}  // namespace detail
}  // namespace cvh
#endif

#endif  // CVH_CORE_SIMD_OPENCV_INTRIN_ADAPTER_H
