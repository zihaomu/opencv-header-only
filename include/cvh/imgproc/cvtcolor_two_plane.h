#ifndef CVH_IMGPROC_CVTCOLOR_TWO_PLANE_H
#define CVH_IMGPROC_CVTCOLOR_TWO_PLANE_H

#include "cvtcolor.h"

namespace cvh
{

inline void cvtColorTwoPlane(const Mat& y,
                             const Mat& uv,
                             Mat& dst,
                             int code)
{
    if (y.empty() || y.dims != 2 || y.type() != CV_8UC1 ||
        uv.empty() || uv.dims != 2 || uv.type() != CV_8UC2)
    {
        CV_Error(
            Error::StsBadArg,
            "cvtColorTwoPlane expects CV_8UC1 Y and CV_8UC2 UV");
    }
    if ((y.size.p[0] & 1) != 0 || (y.size.p[1] & 1) != 0 ||
        uv.size.p[0] * 2 != y.size.p[0] ||
        uv.size.p[1] * 2 != y.size.p[1])
    {
        CV_Error(
            Error::StsBadSize,
            "cvtColorTwoPlane requires even Y dimensions and half-size UV");
    }
    const bool nv21 =
        code == COLOR_YUV2BGR_NV21 ||
        code == COLOR_YUV2RGB_NV21;
    const bool rgb =
        code == COLOR_YUV2RGB_NV12 ||
        code == COLOR_YUV2RGB_NV21;
    if (!nv21 &&
        code != COLOR_YUV2BGR_NV12 &&
        code != COLOR_YUV2RGB_NV12)
    {
        CV_Error(
            Error::StsBadFlag,
            "cvtColorTwoPlane unsupported conversion code");
    }

    const Mat y_source = y.data == dst.data ? y.clone() : y;
    const Mat uv_source = uv.data == dst.data ? uv.clone() : uv;
    dst.create(y_source.shape(), CV_8UC3);
    for (int row = 0; row < y_source.size.p[0]; ++row)
    {
        const uchar* y_row =
            y_source.data +
            static_cast<size_t>(row) * y_source.step(0);
        const uchar* uv_row =
            uv_source.data +
            static_cast<size_t>(row / 2) * uv_source.step(0);
        uchar* output =
            dst.data + static_cast<size_t>(row) * dst.step(0);
        for (int col = 0; col < y_source.size.p[1]; ++col)
        {
            const size_t uv_index =
                static_cast<size_t>(col / 2) * 2;
            const int first = uv_row[uv_index];
            const int second = uv_row[uv_index + 1];
            const int uu = nv21 ? second : first;
            const int vv = nv21 ? first : second;
            const uchar blue =
                detail::cvtcolor_yuv420sp_channel_u8(
                    y_row[col], uu, vv, 0);
            const uchar green =
                detail::cvtcolor_yuv420sp_channel_u8(
                    y_row[col], uu, vv, 1);
            const uchar red =
                detail::cvtcolor_yuv420sp_channel_u8(
                    y_row[col], uu, vv, 2);
            const size_t output_index =
                static_cast<size_t>(col) * 3;
            output[output_index + (rgb ? 0 : 2)] = red;
            output[output_index + 1] = green;
            output[output_index + (rgb ? 2 : 0)] = blue;
        }
    }
}

}  // namespace cvh

#endif  // CVH_IMGPROC_CVTCOLOR_TWO_PLANE_H
