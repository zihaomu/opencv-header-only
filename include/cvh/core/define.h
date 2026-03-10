//
// Created by mzh on 2023/11/2.
//

#ifndef CVH_DEFINE_H
#define CVH_DEFINE_H

#include <assert.h>
#include <stdio.h>
#include <cstdint>
#include <memory>

#define M_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

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

#ifdef M_ROOT_PATH
#define M_ROOT STR(M_ROOT_PATH)
#endif
// ERROR CODE
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
using uint64 = unsigned long int;
using int64 = long int;

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

#define CV_MAX  12 // Equal to MAX Data Type

#define CV_ELEM_SIZE(type) ((int)((0x8812824442211ULL >> (type * 4)) & 15))

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



#endif //CVH_DEFINE_H
