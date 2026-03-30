//
// Created by mzh on 2023/11/2.
//

#ifndef CVH_DEFINE_H
#define CVH_DEFINE_H

#include <assert.h>
#include <stdio.h>
#include <cstdint>
#include <memory>

#if defined(_MSC_VER)
#if defined(BUILDING_M_DLL)
#define CV_EXPORTS __declspec(dllexport)
#elif defined(USING_M_DLL)
#define CV_EXPORTS __declspec(dllimport)
#else
#define CV_EXPORTS
#endif
#else
#define CV_EXPORTS __attribute__((visibility("default")))
#endif
#define STR_IMP(x) #x
#define STR(x) STR_IMP(x)
#define CV_VERSION_MAJOR 0
#define CV_VERSION_MINOR 0
#define CV_VERSION_PATCH 1
#define CV_VERSION_STATUS   "-dev"
#define CV_VERSION STR(CV_VERSION_MAJOR) "." STR(CV_VERSION_MINOR) "." STR(CV_VERSION_PATCH) CV_VERSION_STATUS

#ifdef CV_ROOT_PATH
#define CV_ROOT STR(CV_ROOT_PATH)
#endif

#ifndef CV_PI
#define CV_PI   3.1415926535897932384626433832795
#endif

#ifndef CV_2PI
#define CV_2PI  6.283185307179586476925286766559
#endif

#ifndef CV_LOG2
#define CV_LOG2 0.69314718055994530941723212145818
#endif

/****************************************************************************************\
*                                  Basic Data Type                                       *
\****************************************************************************************/

using uchar = unsigned char;
using schar = signed char;
using ushort = unsigned short;
using uint = unsigned int;
using uint64 = std::uint64_t;
using int64 = std::int64_t;

static_assert(sizeof(uint64) == 8, "cvh::uint64 must be 64-bit on all platforms");
static_assert(sizeof(int64) == 8, "cvh::int64 must be 64-bit on all platforms");

/****************************************************************************************\
*                                  Matrix type (Mat)                                     *
\****************************************************************************************/

// all Mat type
#define CV_8U   0   // - 1 byte
#define CV_8S   1   // - 1 byte
#define CV_16U  2   // - 2 byte
#define CV_16S  3   // - 2 byte
#define CV_32F  4   // - 4 byte
#define CV_32S  5   // - 4 byte
#define CV_32U  6   // - 4 byte
#define CV_16F  7   // - 2 byte
#define CV_64F  8   // - 8 byte
#define CV_16BF 9   // - 2 byte
#define CV_Bool 10  // - 1 byte
#define CV_64U  11  // - 8 byte
#define CV_64S  12  // - 8 byte

#define CV_CN_MAX     128
#define CV_CN_SHIFT   5
#define CV_DEPTH_MAX  (1 << CV_CN_SHIFT)

#define CV_MAT_DEPTH_MASK (CV_DEPTH_MAX - 1)
#define CV_MAT_CN_MASK ((CV_CN_MAX - 1) << CV_CN_SHIFT)

#define CV_MAT_DEPTH(flags) ((flags) & CV_MAT_DEPTH_MASK)
#define CV_MAT_CN(flags) ((((flags) & CV_MAT_CN_MASK) >> CV_CN_SHIFT) + 1)
#define CV_MAKETYPE(depth, cn) (CV_MAT_DEPTH(depth) + (((cn) - 1) << CV_CN_SHIFT))

#define CV_8UC(n) CV_MAKETYPE(CV_8U, (n))
#define CV_8SC(n) CV_MAKETYPE(CV_8S, (n))
#define CV_16UC(n) CV_MAKETYPE(CV_16U, (n))
#define CV_16SC(n) CV_MAKETYPE(CV_16S, (n))
#define CV_32FC(n) CV_MAKETYPE(CV_32F, (n))
#define CV_32SC(n) CV_MAKETYPE(CV_32S, (n))
#define CV_32UC(n) CV_MAKETYPE(CV_32U, (n))
#define CV_16FC(n) CV_MAKETYPE(CV_16F, (n))

#define CV_8UC1 CV_8UC(1)
#define CV_8UC2 CV_8UC(2)
#define CV_8UC3 CV_8UC(3)
#define CV_8UC4 CV_8UC(4)

#define CV_8SC1 CV_8SC(1)
#define CV_8SC2 CV_8SC(2)
#define CV_8SC3 CV_8SC(3)
#define CV_8SC4 CV_8SC(4)

#define CV_16UC1 CV_16UC(1)
#define CV_16UC2 CV_16UC(2)
#define CV_16UC3 CV_16UC(3)
#define CV_16UC4 CV_16UC(4)

#define CV_16SC1 CV_16SC(1)
#define CV_16SC2 CV_16SC(2)
#define CV_16SC3 CV_16SC(3)
#define CV_16SC4 CV_16SC(4)

#define CV_32FC1 CV_32FC(1)
#define CV_32FC2 CV_32FC(2)
#define CV_32FC3 CV_32FC(3)
#define CV_32FC4 CV_32FC(4)

#define CV_32SC1 CV_32SC(1)
#define CV_32SC2 CV_32SC(2)
#define CV_32SC3 CV_32SC(3)
#define CV_32SC4 CV_32SC(4)

#define CV_32UC1 CV_32UC(1)
#define CV_32UC2 CV_32UC(2)
#define CV_32UC3 CV_32UC(3)
#define CV_32UC4 CV_32UC(4)

#define CV_16FC1 CV_16FC(1)
#define CV_16FC2 CV_16FC(2)
#define CV_16FC3 CV_16FC(3)
#define CV_16FC4 CV_16FC(4)

#define CV_MAX CV_DEPTH_MAX

#define CV_ELEM_SIZE1(type) ((int)((0x8812824442211ULL >> (CV_MAT_DEPTH(type) * 4)) & 15))
#define CV_ELEM_SIZE(type) (CV_MAT_CN(type) * CV_ELEM_SIZE1(type))

// Comparing flag
#define CV_CMP_EQ   0
#define CV_CMP_GT   1
#define CV_CMP_GE   2
#define CV_CMP_LT   3
#define CV_CMP_LE   4
#define CV_CMP_NE   5

/****************************************************************************************\
*          exchange-add operation for atomic operations on reference counters            *
\****************************************************************************************/
// take from opencv2/core/cvdef.h Line 706
#ifdef CV_XADD
// allow to use user-defined macro
#elif defined __GNUC__ || defined __clang__
#  if defined __clang__ && __clang_major__ >= 3 && !defined __ANDROID__ && !defined __EMSCRIPTEN__ && !defined(__CUDACC__)  && !defined __INTEL_COMPILER
#    ifdef __ATOMIC_ACQ_REL
#      define CV_XADD(addr, delta) __c11_atomic_fetch_add((_Atomic(int)*)(addr), delta, __ATOMIC_ACQ_REL)
#    else
#      define CV_XADD(addr, delta) __atomic_fetch_add((_Atomic(int)*)(addr), delta, 4)
#    endif
#  else
#    if defined __ATOMIC_ACQ_REL && !defined __clang__
// version for gcc >= 4.7
#      define CV_XADD(addr, delta) (int)__atomic_fetch_add((unsigned*)(addr), (unsigned)(delta), __ATOMIC_ACQ_REL)
#    else
#      define CV_XADD(addr, delta) (int)__sync_fetch_and_add((unsigned*)(addr), (unsigned)(delta))
#    endif
#  endif
#elif defined _MSC_VER && !defined RC_INVOKED
#  include <intrin.h>
#  define CV_XADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile*)addr, delta)
#else
#error "Atomic operations are not supported"
#endif


/****************************************************************************************\
*                                  Float16 Define                                        *
\****************************************************************************************/
#ifdef __cplusplus

typedef union Cv32suf
{
    int i;
    unsigned u;
    float f;
}
Cv32suf;

class hfloat
{
public:
#if M_WITH_ARM // TODO add arm
    hfloat() : h(0) {}
    explicit hfloat(float x) { h = (__fp16)x; }
    operator float() const { return (float)h; }
    void* get_ptr() { return &h; }
protected:
    __fp16 h;

#else
    void* get_ptr() { return &w; }
    hfloat() : w(0) {}
    explicit hfloat(float x)
    {
#if CV_FP16 && CV_AVX2
        __m128 v = _mm_load_ss(&x);
        w = (ushort)_mm_cvtsi128_si32(_mm_cvtps_ph(v, 0));
#else
        Cv32suf in;
        in.f = x;
        unsigned sign = in.u & 0x80000000;
        in.u ^= sign;

        if( in.u >= 0x47800000 )
            w = (ushort)(in.u > 0x7f800000 ? 0x7e00 : 0x7c00);
        else
        {
            if (in.u < 0x38800000)
            {
                in.f += 0.5f;
                w = (ushort)(in.u - 0x3f000000);
            }
            else
            {
                unsigned t = in.u + 0xc8000fff;
                w = (ushort)((t + ((in.u >> 13) & 1)) >> 13);
            }
        }

        w = (ushort)(w | (sign >> 16));
#endif
    }

    operator float() const
    {
    // TODO convert to float with intrinsics, and use xsimd to convert a batch of hfloat to float, and use xsimd to convert a batch of float to hfloat, and make sure the conversion is correct with unit test.
#if CV_FP16 && CV_AVX2
        float f;
        _mm_store_ss(&f, _mm_cvtph_ps(_mm_cvtsi32_si128(w)));
        return f;
#else
        Cv32suf out;

        unsigned t = ((w & 0x7fff) << 13) + 0x38000000;
        unsigned sign = (w & 0x8000) << 16;
        unsigned e = w & 0x7c00;

        out.u = t + (1 << 23);
        out.u = (e >= 0x7c00 ? t + 0x38000000 :
                 e == 0 ? (static_cast<void>(out.f -= 6.103515625e-05f), out.u) : t) | sign;
        return out.f;
#endif
    }

protected:
    ushort w;
#endif
};

#else
#error "Fp16 must compile with c++"
#endif

#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

/** min & max without jumps */
#define CV_IMIN(a, b)  ((a) ^ (((a)^(b)) & (((a) < (b)) - 1)))
#define CV_IMAX(a, b)  ((a) ^ (((a)^(b)) & (((a) > (b)) - 1)))
#define CV_SWAP(a,b,t) ((t) = (a), (a) = (b), (b) = (t))
#define CV_CMP(a,b)    (((a) > (b)) - ((a) < (b)))
#define CV_SIGN(a)     CV_CMP((a),0)

#endif //CVH_DEFINE_H
