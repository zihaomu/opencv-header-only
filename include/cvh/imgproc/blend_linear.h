#ifndef CVH_IMGPROC_BLEND_LINEAR_H
#define CVH_IMGPROC_BLEND_LINEAR_H

#include "detail/common.h"

#include <cmath>

namespace cvh
{

inline void blendLinear(const Mat& src1,
                        const Mat& src2,
                        const Mat& weights1,
                        const Mat& weights2,
                        Mat& dst)
{
    if (src1.empty() || src1.dims != 2 ||
        (src1.depth() != CV_8U && src1.depth() != CV_32F) ||
        (src1.channels() != 1 && src1.channels() != 3 &&
         src1.channels() != 4))
    {
        CV_Error(Error::StsBadArg, "blendLinear unsupported source");
    }
    const auto matches_source_shape = [&](const Mat& mat) {
        return mat.dims == 2 &&
               mat.size.p[0] == src1.size.p[0] &&
               mat.size.p[1] == src1.size.p[1];
    };
    if (!matches_source_shape(src2) || src2.type() != src1.type())
    {
        CV_Error(Error::StsBadArg, "blendLinear sources must match");
    }
    if (!matches_source_shape(weights1) ||
        !matches_source_shape(weights2) ||
        weights1.type() != CV_32FC1 ||
        weights2.type() != CV_32FC1)
    {
        CV_Error(
            Error::StsBadArg,
            "blendLinear weights must be matching CV_32FC1 matrices");
    }

    const Mat source1 = src1.data == dst.data ? src1.clone() : src1;
    const Mat source2 = src2.data == dst.data ? src2.clone() : src2;
    const Mat weight1 =
        weights1.data == dst.data ? weights1.clone() : weights1;
    const Mat weight2 =
        weights2.data == dst.data ? weights2.clone() : weights2;
    dst.create(source1.shape(), source1.type());
    const int rows = source1.size.p[0];
    const int cols = source1.size.p[1];
    const int channels = source1.channels();
    for (int y = 0; y < rows; ++y)
    {
        const uchar* input1 =
            source1.data + static_cast<size_t>(y) * source1.step(0);
        const uchar* input2 =
            source2.data + static_cast<size_t>(y) * source2.step(0);
        const float* first_weight = reinterpret_cast<const float*>(
            weight1.data + static_cast<size_t>(y) * weight1.step(0));
        const float* second_weight = reinterpret_cast<const float*>(
            weight2.data + static_cast<size_t>(y) * weight2.step(0));
        uchar* output =
            dst.data + static_cast<size_t>(y) * dst.step(0);
        for (int x = 0; x < cols; ++x)
        {
            const float w1 = first_weight[x];
            const float w2 = second_weight[x];
            const float denominator = w1 + w2 + 1e-5f;
            for (int ch = 0; ch < channels; ++ch)
            {
                const size_t index =
                    static_cast<size_t>(x) * channels +
                    static_cast<size_t>(ch);
                if (source1.depth() == CV_8U)
                {
                    const float value =
                        (static_cast<float>(input1[index]) * w1 +
                         static_cast<float>(input2[index]) * w2) /
                        denominator;
                    output[index] = saturate_cast<uchar>(
                        static_cast<int>(std::lrint(value)));
                }
                else
                {
                    const float* first =
                        reinterpret_cast<const float*>(input1);
                    const float* second =
                        reinterpret_cast<const float*>(input2);
                    float* destination =
                        reinterpret_cast<float*>(output);
                    destination[index] =
                        (first[index] * w1 + second[index] * w2) /
                        denominator;
                }
            }
        }
    }
}

}  // namespace cvh

#endif  // CVH_IMGPROC_BLEND_LINEAR_H
