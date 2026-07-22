#ifndef CVH_3RDPARTY_OPENCV_INTRIN_CVDEF_H
#define CVH_3RDPARTY_OPENCV_INTRIN_CVDEF_H

#include <cstdint>
#include <cstdlib>

#if defined(CV_NEON) && CV_NEON && (defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64))
#include <arm_neon.h>
#endif

#ifndef CVH_DEFINE_H
typedef unsigned char uchar;
typedef signed char schar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef std::int64_t int64;
typedef std::uint64_t uint64;
#endif

#ifndef __CV_EXPAND
#define __CV_EXPAND(x) x
#endif

#ifndef __CV_CAT
#define __CV_CAT__(x, y) x ## y
#define __CV_CAT_(x, y) __CV_CAT__(x, y)
#define __CV_CAT(x, y) __CV_CAT_(x, y)
#endif

#ifndef CV_INLINE
#define CV_INLINE static inline
#endif

#ifndef CV_ALWAYS_INLINE
#if defined(__GNUC__) || defined(__clang__)
#define CV_ALWAYS_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define CV_ALWAYS_INLINE __forceinline
#else
#define CV_ALWAYS_INLINE inline
#endif
#endif

#ifndef CV_DECL_ALIGNED
#if defined(__GNUC__) || defined(__clang__)
#define CV_DECL_ALIGNED(x) __attribute__((aligned(x)))
#elif defined(_MSC_VER)
#define CV_DECL_ALIGNED(x) __declspec(align(x))
#else
#define CV_DECL_ALIGNED(x)
#endif
#endif

#ifndef CV_STRONG_ALIGNMENT
#define CV_STRONG_ALIGNMENT 1
#endif

#ifndef CV_Assert
#define CV_Assert(expr) do { if (!(expr)) std::abort(); } while (0)
#endif

#ifndef CV_DbgAssert
#ifndef NDEBUG
#define CV_DbgAssert(expr) CV_Assert(expr)
#else
#define CV_DbgAssert(expr) do { } while (0)
#endif
#endif

#ifndef CV_UNUSED
#define CV_UNUSED(name) (void)(name)
#endif

#ifndef CV_SSE2
#define CV_SSE2 0
#endif
#ifndef CV_AVX2
#define CV_AVX2 0
#endif
#ifndef CV_AVX512_SKX
#define CV_AVX512_SKX 0
#endif
#ifndef CV_NEON
#define CV_NEON 0
#endif
#ifndef CV_FP16
#define CV_FP16 0
#endif
#ifndef CV_VSX
#define CV_VSX 0
#endif
#ifndef CV_MSA
#define CV_MSA 0
#endif
#ifndef CV_WASM_SIMD
#define CV_WASM_SIMD 0
#endif
#ifndef CV_RVV
#define CV_RVV 0
#endif
#ifndef CV_RVV071
#define CV_RVV071 0
#endif
#ifndef CV_LSX
#define CV_LSX 0
#endif
#ifndef CV_LASX
#define CV_LASX 0
#endif
#ifndef CV_SIMD_SCALABLE
#define CV_SIMD_SCALABLE 0
#endif
#ifndef CV_SIMD_SCALABLE_64F
#define CV_SIMD_SCALABLE_64F 0
#endif

namespace cv {

class hfloat
{
public:
    hfloat() : value_(0.0f) {}
    explicit hfloat(float value) : value_(value) {}
    operator float() const { return value_; }

private:
    float value_;
};

}  // namespace cv

#endif  // CVH_3RDPARTY_OPENCV_INTRIN_CVDEF_H
