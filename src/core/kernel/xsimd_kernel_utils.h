#ifndef CVH_XSIMD_KERNEL_UTILS_H
#define CVH_XSIMD_KERNEL_UTILS_H

#include "cvh/core/define.h"

#include "xsimd/xsimd.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace cvh {
namespace cpu {

using XSimdBatch = xsimd::batch<float>;
constexpr std::size_t kXSimdBatchSize = XSimdBatch::size;

#if defined(__GNUC__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
template <std::size_t Lanes = kXSimdBatchSize, typename std::enable_if<Lanes == 4, int>::type = 0>
__attribute__((target("f16c")))
inline XSimdBatch load_hfloat_batch_f16c(const hfloat* src)
{
    std::array<std::uint16_t, 4> raw_bits {};
    std::memcpy(raw_bits.data(), src, sizeof(raw_bits));
    return XSimdBatch(_mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(raw_bits.data()))));
}

template <std::size_t Lanes = kXSimdBatchSize, typename std::enable_if<Lanes == 8, int>::type = 0>
__attribute__((target("avx,f16c")))
inline XSimdBatch load_hfloat_batch_f16c(const hfloat* src)
{
    std::array<std::uint16_t, 8> raw_bits {};
    std::memcpy(raw_bits.data(), src, sizeof(raw_bits));
    return XSimdBatch(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(raw_bits.data()))));
}
#endif

inline XSimdBatch load_hfloat_batch(const hfloat* src)
{
#if defined(__GNUC__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
    static_assert(sizeof(hfloat) == sizeof(std::uint16_t), "hfloat must use 16-bit storage");

    if constexpr (kXSimdBatchSize == 8)
    {
        if (__builtin_cpu_supports("avx") && __builtin_cpu_supports("f16c"))
        {
            return load_hfloat_batch_f16c(src);
        }
    }

    if constexpr (kXSimdBatchSize == 4)
    {
        if (__builtin_cpu_supports("f16c"))
        {
            return load_hfloat_batch_f16c(src);
        }
    }
#endif

    std::array<float, kXSimdBatchSize> tmp{};
    for (std::size_t idx = 0; idx < kXSimdBatchSize; ++idx)
    {
        tmp[idx] = static_cast<float>(src[idx]);
    }
    return XSimdBatch::load_unaligned(tmp.data());
}

inline XSimdBatch load_int8_batch(const int8_t* src)
{
    return xsimd::load_as<float>(src, xsimd::unaligned_mode {});
}

}  // namespace cpu
}  // namespace cvh

#endif  // CVH_XSIMD_KERNEL_UTILS_H
