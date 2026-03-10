//
// Created by mzh on 2024/1/31.
//

#include "cvh/core/mat.h"
#include "cvh/core/system.h"
#include "cvh/core/saturate.h"
#include "cvh/core/utils.h"

#include <climits>
#include <cfloat>
#include <vector>
#include <limits>
#include <cstring>


namespace cvh
{
// TODO：目前的convert是没考虑对齐状态的，只有内存对齐速度才最快，未来为了能够在访存上最优，需要考虑对齐。
template<typename _Ts, typename _Td> static inline void
convert(const _Ts* src, _Td* dst, size_t length)
{
    for (size_t i = 0; i < length; i++)
    {
        dst[i] = saturate_cast<_Td>(src[i]);
        // std::cout<<"convert: src["<<i<<"] = "<<(float)src[i]<<" to dst["<<i<<"] = "<<(float)dst[i]<<std::endl;
    }
}

template<> inline
void convert(const hfloat* src, float* dst, size_t length)
{
    for (size_t i = 0; i < length; i++)
    {
        dst[i] = (float)(src[i]);
        // std::cout<<"convert: src["<<i<<"] = "<<(float)src[i]<<" to dst["<<i<<"] = "<<(float)dst[i]<<std::endl;
    }
}

template<> inline
void convert(const float* src, hfloat* dst, size_t length)
{
    for (size_t i = 0; i < length; i++)
    {
        dst[i] = (hfloat)(src[i]);
        // std::cout<<"convert: src["<<i<<"] = "<<(float)src[i]<<" to dst["<<i<<"] = "<<(float)dst[i]<<std::endl;
    }
}

#define CONVERT_FUNC(suffix, func, _Ts, _Td) \
static void convert_##suffix(const uchar* src_, uchar* dst_, size_t length) \
{ \
    const _Ts* src = (const _Ts*)src_; \
    _Td* dst = (_Td*)dst_; \
    func<_Ts, _Td>(src, dst, length); \
}

// 8u
CONVERT_FUNC(8u8s,  convert, uchar, char  )
CONVERT_FUNC(8u16u, convert, uchar, ushort)
CONVERT_FUNC(8u16s, convert, uchar, short )
CONVERT_FUNC(8u32s, convert, uchar, int   )
CONVERT_FUNC(8u32f, convert, uchar, float )
CONVERT_FUNC(8u32u, convert, uchar, uint  )

// 8s
CONVERT_FUNC(8s8u,  convert, char, uchar )
CONVERT_FUNC(8s16u, convert, char, ushort)
CONVERT_FUNC(8s16s, convert, char, short )
CONVERT_FUNC(8s32s, convert, char, int   )
CONVERT_FUNC(8s32f, convert, char, float )
CONVERT_FUNC(8s32u, convert, char, uint  )

// 16u
CONVERT_FUNC(16u8s,  convert, ushort, char  )
CONVERT_FUNC(16u8u,  convert, ushort, uchar )
CONVERT_FUNC(16u16s, convert, ushort, short )
CONVERT_FUNC(16u32s, convert, ushort, int   )
CONVERT_FUNC(16u32f, convert, ushort, float )
CONVERT_FUNC(16u32u, convert, ushort, uint  )

// 16s
CONVERT_FUNC(16s8s,  convert, short, char  )
CONVERT_FUNC(16s8u,  convert, short, uchar )
CONVERT_FUNC(16s16u, convert, short, ushort)
CONVERT_FUNC(16s32s, convert, short, int   )
CONVERT_FUNC(16s32f, convert, short, float )
CONVERT_FUNC(16s32u, convert, short, uint  )

// 32f
CONVERT_FUNC(32f8s,  convert, float, char  )
CONVERT_FUNC(32f8u,  convert, float, uchar )
CONVERT_FUNC(32f16u, convert, float, ushort)
CONVERT_FUNC(32f16s, convert, float, short )
CONVERT_FUNC(32f32s, convert, float, int   )
CONVERT_FUNC(32f32u, convert, float, uint  )
CONVERT_FUNC(32f16f, convert, float, hfloat)

// 32s
CONVERT_FUNC(32s8s,  convert, int, char  )
CONVERT_FUNC(32su8,  convert, int, uchar )
CONVERT_FUNC(32s16u, convert, int, ushort)
CONVERT_FUNC(32s16s, convert, int, short )
CONVERT_FUNC(32s32f, convert, int, float )
CONVERT_FUNC(32s32u, convert, int, uint  )

// 32u
CONVERT_FUNC(32u8s,  convert, uint, char  )
CONVERT_FUNC(32u8u,  convert, uint, uchar )
CONVERT_FUNC(32u16u, convert, uint, ushort)
CONVERT_FUNC(32u16s, convert, uint, short )
CONVERT_FUNC(32u32s, convert, uint, int   )
CONVERT_FUNC(32u32f, convert, uint, float )

// 16f
CONVERT_FUNC(16f8s,  convert, hfloat, char  )
CONVERT_FUNC(16f8u,  convert, hfloat, uchar )
CONVERT_FUNC(16f16u, convert, hfloat, ushort)
CONVERT_FUNC(16f16s, convert, hfloat, short )
CONVERT_FUNC(16f32s, convert, hfloat, int   )
CONVERT_FUNC(16f32f, convert, hfloat, float )
CONVERT_FUNC(16f32u, convert, hfloat, uint )


static void convert_8u(const uchar* src_, uchar* dst_, size_t length)
{
    memcpy(dst_, src_, length * sizeof(uchar));
}

static void convert_16u(const uchar* src_, uchar* dst_, size_t length)
{
    memcpy(dst_, src_, length * sizeof(ushort));
}

static void convert_32u(const uchar* src_, uchar* dst_, size_t length)
{
    memcpy(dst_, src_, length * sizeof(uint));
}

typedef void (*BinaryFunc)(const uchar* src1, uchar* dst, size_t length);
BinaryFunc getConvertFunc(int stype, int dtype)
{
    // The order of Data type can not be changed it is strictly correspond.
    static BinaryFunc funcTab[CV_MAX][CV_MAX] =
    {
        // 8u = 0
        {
            (convert_8u),
            (convert_8u8s),
            (convert_8u16u),
            (convert_8u16s),
            (convert_8u32f),
            (convert_8u32s),
            (convert_8u32u),
        },

        // 8s = 1
        {
            (convert_8s8u),
            (convert_8u),
            (convert_8s16u),
            (convert_8s16s),
            (convert_8s32f),
            (convert_8s32s),
            (convert_8s32u),
        },

        // 16u = 2
        {
            (convert_16u8s),
            (convert_16u8u),
            (convert_16u),
            (convert_16u16s),
            (convert_16u32f),
            (convert_16u32s),
            (convert_16u32u),
        },

        // 16s = 3
        {
            (convert_16s8u),
            (convert_16s8s),
            (convert_16s16u),
            (convert_16u),
            (convert_16s32f),
            (convert_16s32s),
            (convert_16s32u),
        },
        
        // 32f = 4
        {
            (convert_32f8u),
            (convert_32f8s),
            (convert_32f16u),
            (convert_32f16s),
            (convert_32u),
            (convert_32f32s),
            (convert_32f32u),
            (convert_32f16f),
        },

        // 32s = 5
        {
            (convert_32su8),
            (convert_32s8s),
            (convert_32s16u),
            (convert_32s16s),
            (convert_32s32f),
            (convert_32u),
            (convert_32s32u),
        },

        // 32u = 6
        {
            (convert_32u8u),
            (convert_32u8s),
            (convert_32u16u),
            (convert_32u16s),
            (convert_32u32f),
            (convert_32u32s),
            (convert_32u),
        },

        // 16f = 7
     {
            (convert_16f8s),
            (convert_16f8u),
            (convert_16f16u),
            (convert_16f16s),
            (convert_16f32f),
            (convert_16f32s),
            (convert_16f32u),
        }
    };

    return funcTab[stype][dtype];
}

void Mat::convertTo(minfer::Mat &m, int type_) const
{
    if (empty())
    {
        m.release();
        return;
    }

    int stype = type();
    int dtype = type_;

    if (stype == dtype)
    {
        copyTo(m);
        return;
    }

    // allocate new memory
    m.create(dims, size.p, dtype);

    auto a = m.total();

    BinaryFunc func = getConvertFunc(stype, dtype);
    func(this->data, m.data, m.total());
}

}
