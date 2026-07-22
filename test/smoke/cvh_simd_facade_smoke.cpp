#include "cvh/core/simd/simd.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

int main()
{
    namespace simd = cvh::detail::simd;

    constexpr std::size_t lanes = simd::f32_lanes();
    std::array<float, lanes> lhs {};
    std::array<float, lanes> rhs {};
    std::array<float, lanes> out {};

    for (std::size_t lane = 0; lane < lanes; ++lane)
    {
        lhs[lane] = static_cast<float>(lane + 1);
        rhs[lane] = static_cast<float>((lane + 1) * 2);
    }

    const simd::f32 lhs_vec = simd::load_f32(lhs.data());
    const simd::f32 rhs_vec = simd::load_f32(rhs.data());
    const simd::f32 computed =
        simd::max(simd::min(simd::add(simd::mul(lhs_vec, rhs_vec), simd::setall_f32(1.0f)),
                            simd::setall_f32(40.0f)),
                  simd::setzero_f32());

    simd::store_f32(out.data(), computed);

    float expected_sum = 0.0f;
    for (std::size_t lane = 0; lane < lanes; ++lane)
    {
        const float expected = std::max(std::min(lhs[lane] * rhs[lane] + 1.0f, 40.0f), 0.0f);
        if (std::fabs(out[lane] - expected) > 1e-5f)
        {
            return 1;
        }
        expected_sum += expected;
    }

    if (std::fabs(simd::reduce_sum(computed) - expected_sum) > 1e-5f)
    {
        return 2;
    }

    constexpr std::size_t u8_lanes = simd::u8_lanes();
    std::array<std::uint8_t, u8_lanes * 3> interleaved {};
    std::array<std::uint8_t, u8_lanes> channel0 {};
    for (std::size_t lane = 0; lane < u8_lanes; ++lane)
    {
        interleaved[lane * 3 + 0] = static_cast<std::uint8_t>(lane * 7 + 3);
        interleaved[lane * 3 + 1] = static_cast<std::uint8_t>(lane * 5 + 11);
        interleaved[lane * 3 + 2] = static_cast<std::uint8_t>(lane * 3 + 19);
    }

    simd::u8 c0;
    simd::u8 c1;
    simd::u8 c2;
    simd::load_deinterleave3_u8(interleaved.data(), c0, c1, c2);

    simd::u16 c0_low;
    simd::u16 c0_high;
    simd::expand_u8(c0, c0_low, c0_high);

    const simd::u16 two = simd::setall_u16(2);
    simd::u32 c0_low_times2_low;
    simd::u32 c0_low_times2_high;
    simd::u32 c0_high_times2_low;
    simd::u32 c0_high_times2_high;
    simd::mul_expand_u16(c0_low, two, c0_low_times2_low, c0_low_times2_high);
    simd::mul_expand_u16(c0_high, two, c0_high_times2_low, c0_high_times2_high);
    const simd::u16 packed_low = simd::rshr_pack_u32_to_u16<1>(
        c0_low_times2_low,
        c0_low_times2_high);
    const simd::u16 packed_high = simd::rshr_pack_u32_to_u16<1>(
        c0_high_times2_low,
        c0_high_times2_high);
    simd::store_u8(channel0.data(), simd::pack_u16_to_u8(packed_low, packed_high));

    for (std::size_t lane = 0; lane < u8_lanes; ++lane)
    {
        if (channel0[lane] != interleaved[lane * 3])
        {
            return 3;
        }
    }

    return simd::backend_name()[0] != '\0' ? 0 : 4;
}
