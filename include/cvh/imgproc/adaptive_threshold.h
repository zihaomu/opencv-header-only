#ifndef CVH_IMGPROC_ADAPTIVE_THRESHOLD_H
#define CVH_IMGPROC_ADAPTIVE_THRESHOLD_H

#include "box_filter.h"
#include "gaussian_blur.h"

#include <cmath>

namespace cvh
{

inline void adaptiveThreshold(const Mat& src,
                              Mat& dst,
                              double maxValue,
                              int adaptiveMethod,
                              int thresholdType,
                              int blockSize,
                              double C)
{
    if (src.empty() || src.dims != 2 || src.type() != CV_8UC1)
    {
        CV_Error(Error::StsBadArg, "adaptiveThreshold expects non-empty CV_8UC1 source");
    }
    if (blockSize <= 1 || (blockSize & 1) == 0)
    {
        CV_Error(Error::StsBadSize, "adaptiveThreshold blockSize must be odd and greater than 1");
    }
    if (!std::isfinite(maxValue) || !std::isfinite(C))
    {
        CV_Error(Error::StsBadArg, "adaptiveThreshold maxValue and C must be finite");
    }
    if (adaptiveMethod != ADAPTIVE_THRESH_MEAN_C &&
        adaptiveMethod != ADAPTIVE_THRESH_GAUSSIAN_C)
    {
        CV_Error(Error::StsBadFlag, "adaptiveThreshold unsupported adaptive method");
    }
    if (thresholdType != THRESH_BINARY &&
        thresholdType != THRESH_BINARY_INV)
    {
        CV_Error(Error::StsBadFlag, "adaptiveThreshold unsupported threshold type");
    }

    const Mat source = src.data == dst.data ? src.clone() : src;
    dst.create(source.shape(), source.type());
    if (maxValue < 0.0)
    {
        dst.setTo(Scalar::all(0.0));
        return;
    }

    Mat local_mean;
    const int border = BORDER_REPLICATE | BORDER_ISOLATED;
    if (adaptiveMethod == ADAPTIVE_THRESH_MEAN_C)
    {
        boxFilter(
            source,
            local_mean,
            CV_8U,
            Size(blockSize, blockSize),
            Point(-1, -1),
            true,
            border);
    }
    else
    {
        GaussianBlur(
            source,
            local_mean,
            Size(blockSize, blockSize),
            0.0,
            0.0,
            border);
    }

    const int delta =
        thresholdType == THRESH_BINARY
            ? static_cast<int>(std::ceil(C))
            : static_cast<int>(std::floor(C));
    const uchar maximum = saturate_cast<uchar>(maxValue);
    for (int y = 0; y < source.size.p[0]; ++y)
    {
        const uchar* input =
            source.data + static_cast<size_t>(y) * source.step(0);
        const uchar* mean =
            local_mean.data + static_cast<size_t>(y) * local_mean.step(0);
        uchar* output =
            dst.data + static_cast<size_t>(y) * dst.step(0);
        for (int x = 0; x < source.size.p[1]; ++x)
        {
            const bool above =
                static_cast<int>(input[x]) - static_cast<int>(mean[x]) >
                -delta;
            output[x] =
                thresholdType == THRESH_BINARY
                    ? (above ? maximum : 0)
                    : (above ? 0 : maximum);
        }
    }
}

}  // namespace cvh

#endif  // CVH_IMGPROC_ADAPTIVE_THRESHOLD_H
