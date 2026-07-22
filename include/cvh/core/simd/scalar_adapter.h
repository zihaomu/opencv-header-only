#ifndef CVH_CORE_SIMD_SCALAR_ADAPTER_H
#define CVH_CORE_SIMD_SCALAR_ADAPTER_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace cvh {
namespace detail {
namespace simd {
namespace scalar {

struct f32
{
    static constexpr std::size_t lanes = 4;
    std::array<float, lanes> values {};
};

struct u8
{
    static constexpr std::size_t lanes = 16;
    std::array<std::uint8_t, lanes> values {};
};

struct u16
{
    static constexpr std::size_t lanes = 8;
    std::array<std::uint16_t, lanes> values {};
};

struct u32
{
    static constexpr std::size_t lanes = 4;
    std::array<std::uint32_t, lanes> values {};
};

inline const char* backend_name()
{
    return "scalar";
}

inline f32 setzero_f32()
{
    return f32 {};
}

inline f32 setall_f32(float value)
{
    f32 result;
    result.values.fill(value);
    return result;
}

inline f32 load_f32(const float* src)
{
    f32 result;
    for (std::size_t lane = 0; lane < f32::lanes; ++lane)
    {
        result.values[lane] = src[lane];
    }
    return result;
}

inline void store_f32(float* dst, const f32& value)
{
    for (std::size_t lane = 0; lane < f32::lanes; ++lane)
    {
        dst[lane] = value.values[lane];
    }
}

inline u32 setall_u32(std::uint32_t value)
{
    u32 result;
    result.values.fill(value);
    return result;
}

inline u16 setall_u16(std::uint16_t value)
{
    u16 result;
    result.values.fill(value);
    return result;
}

inline u8 load_u8(const std::uint8_t* src)
{
    u8 result;
    for (std::size_t lane = 0; lane < u8::lanes; ++lane)
    {
        result.values[lane] = src[lane];
    }
    return result;
}

inline void load_deinterleave2_u8(const std::uint8_t* src, u8& c0, u8& c1)
{
    for (std::size_t lane = 0; lane < u8::lanes; ++lane)
    {
        const std::size_t offset = lane * 2;
        c0.values[lane] = src[offset + 0];
        c1.values[lane] = src[offset + 1];
    }
}

inline void load_deinterleave3_u8(const std::uint8_t* src, u8& c0, u8& c1, u8& c2)
{
    for (std::size_t lane = 0; lane < u8::lanes; ++lane)
    {
        const std::size_t offset = lane * 3;
        c0.values[lane] = src[offset + 0];
        c1.values[lane] = src[offset + 1];
        c2.values[lane] = src[offset + 2];
    }
}

inline void store_u8(std::uint8_t* dst, const u8& value)
{
    for (std::size_t lane = 0; lane < u8::lanes; ++lane)
    {
        dst[lane] = value.values[lane];
    }
}

inline void expand_u8(const u8& value, u16& low, u16& high)
{
    for (std::size_t lane = 0; lane < u16::lanes; ++lane)
    {
        low.values[lane] = value.values[lane];
        high.values[lane] = value.values[lane + u16::lanes];
    }
}

inline u32 expand_low_u16(const u16& value)
{
    u32 result;
    for (std::size_t lane = 0; lane < u32::lanes; ++lane)
    {
        result.values[lane] = value.values[lane];
    }
    return result;
}

inline u32 expand_high_u16(const u16& value)
{
    u32 result;
    for (std::size_t lane = 0; lane < u32::lanes; ++lane)
    {
        result.values[lane] = value.values[lane + u32::lanes];
    }
    return result;
}

inline f32 add(const f32& lhs, const f32& rhs)
{
    f32 result;
    for (std::size_t lane = 0; lane < f32::lanes; ++lane)
    {
        result.values[lane] = lhs.values[lane] + rhs.values[lane];
    }
    return result;
}

inline u16 add(const u16& lhs, const u16& rhs)
{
    u16 result;
    for (std::size_t lane = 0; lane < u16::lanes; ++lane)
    {
        result.values[lane] = static_cast<std::uint16_t>(lhs.values[lane] + rhs.values[lane]);
    }
    return result;
}

inline u32 add(const u32& lhs, const u32& rhs)
{
    u32 result;
    for (std::size_t lane = 0; lane < u32::lanes; ++lane)
    {
        result.values[lane] = lhs.values[lane] + rhs.values[lane];
    }
    return result;
}

inline f32 sub(const f32& lhs, const f32& rhs)
{
    f32 result;
    for (std::size_t lane = 0; lane < f32::lanes; ++lane)
    {
        result.values[lane] = lhs.values[lane] - rhs.values[lane];
    }
    return result;
}

inline f32 mul(const f32& lhs, const f32& rhs)
{
    f32 result;
    for (std::size_t lane = 0; lane < f32::lanes; ++lane)
    {
        result.values[lane] = lhs.values[lane] * rhs.values[lane];
    }
    return result;
}

inline u32 mul(const u32& lhs, const u32& rhs)
{
    u32 result;
    for (std::size_t lane = 0; lane < u32::lanes; ++lane)
    {
        result.values[lane] = lhs.values[lane] * rhs.values[lane];
    }
    return result;
}

inline void mul_expand_u16(const u16& lhs, const u16& rhs, u32& low, u32& high)
{
    for (std::size_t lane = 0; lane < u32::lanes; ++lane)
    {
        low.values[lane] = static_cast<std::uint32_t>(lhs.values[lane]) *
                           static_cast<std::uint32_t>(rhs.values[lane]);
        high.values[lane] = static_cast<std::uint32_t>(lhs.values[lane + u32::lanes]) *
                            static_cast<std::uint32_t>(rhs.values[lane + u32::lanes]);
    }
}

inline f32 min(const f32& lhs, const f32& rhs)
{
    f32 result;
    for (std::size_t lane = 0; lane < f32::lanes; ++lane)
    {
        result.values[lane] = std::min(lhs.values[lane], rhs.values[lane]);
    }
    return result;
}

inline f32 max(const f32& lhs, const f32& rhs)
{
    f32 result;
    for (std::size_t lane = 0; lane < f32::lanes; ++lane)
    {
        result.values[lane] = std::max(lhs.values[lane], rhs.values[lane]);
    }
    return result;
}

inline float reduce_sum(const f32& value)
{
    float sum = 0.0f;
    for (float lane : value.values)
    {
        sum += lane;
    }
    return sum;
}

inline constexpr std::size_t f32_lanes()
{
    return f32::lanes;
}

template <int shift>
inline u16 rshr_pack_u32_to_u16(const u32& low, const u32& high)
{
    static_assert(shift > 0, "rounding right shift requires a positive shift");

    u16 result;
    constexpr std::uint32_t round = std::uint32_t{1} << (shift - 1);
    for (std::size_t lane = 0; lane < u32::lanes; ++lane)
    {
        const std::uint32_t low_value = (low.values[lane] + round) >> shift;
        const std::uint32_t high_value = (high.values[lane] + round) >> shift;
        result.values[lane] = static_cast<std::uint16_t>(
            std::min(low_value, static_cast<std::uint32_t>(std::numeric_limits<std::uint16_t>::max())));
        result.values[lane + u32::lanes] = static_cast<std::uint16_t>(
            std::min(high_value, static_cast<std::uint32_t>(std::numeric_limits<std::uint16_t>::max())));
    }
    return result;
}

template <int shift>
inline u8 rshr_pack_u16_to_u8(const u16& low, const u16& high)
{
    static_assert(shift > 0, "rounding right shift requires a positive shift");

    u8 result;
    constexpr std::uint16_t round = std::uint16_t{1} << (shift - 1);
    for (std::size_t lane = 0; lane < u16::lanes; ++lane)
    {
        const std::uint16_t low_value = static_cast<std::uint16_t>((low.values[lane] + round) >> shift);
        const std::uint16_t high_value = static_cast<std::uint16_t>((high.values[lane] + round) >> shift);
        result.values[lane] = static_cast<std::uint8_t>(
            std::min(low_value, static_cast<std::uint16_t>(std::numeric_limits<std::uint8_t>::max())));
        result.values[lane + u16::lanes] = static_cast<std::uint8_t>(
            std::min(high_value, static_cast<std::uint16_t>(std::numeric_limits<std::uint8_t>::max())));
    }
    return result;
}

inline u8 pack_u16_to_u8(const u16& low, const u16& high)
{
    u8 result;
    for (std::size_t lane = 0; lane < u16::lanes; ++lane)
    {
        result.values[lane] = static_cast<std::uint8_t>(
            std::min(low.values[lane], static_cast<std::uint16_t>(std::numeric_limits<std::uint8_t>::max())));
        result.values[lane + u16::lanes] = static_cast<std::uint8_t>(
            std::min(high.values[lane], static_cast<std::uint16_t>(std::numeric_limits<std::uint8_t>::max())));
    }
    return result;
}

inline constexpr std::size_t u8_lanes()
{
    return u8::lanes;
}

}  // namespace scalar
}  // namespace simd
}  // namespace detail
}  // namespace cvh

#endif  // CVH_CORE_SIMD_SCALAR_ADAPTER_H
