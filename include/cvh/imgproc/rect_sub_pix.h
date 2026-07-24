#ifndef CVH_IMGPROC_RECT_SUB_PIX_H
#define CVH_IMGPROC_RECT_SUB_PIX_H

#include "detail/geometric_sampling.hpp"

#include <cmath>

namespace cvh {
namespace detail {

template<typename SourceT, typename DestinationT>
inline void rect_sub_pix_typed(const Mat& source,
                               Mat& patch,
                               Point2f center)
{
    const double origin_x =
        static_cast<double>(center.x) -
        (static_cast<double>(patch.size[1]) - 1.0) * 0.5;
    const double origin_y =
        static_cast<double>(center.y) -
        (static_cast<double>(patch.size[0]) - 1.0) * 0.5;
    for (int row = 0; row < patch.size[0]; ++row)
    {
        DestinationT* output = reinterpret_cast<DestinationT*>(
            patch.data + static_cast<size_t>(row) * patch.step(0));
        for (int col = 0; col < patch.size[1]; ++col)
        {
            const double source_x = origin_x + col;
            const double source_y = origin_y + row;
            const int integer_x =
                static_cast<int>(std::floor(source_x));
            const int integer_y =
                static_cast<int>(std::floor(source_y));
            geometric_write_linear_as<SourceT, DestinationT>(
                source,
                output +
                    static_cast<size_t>(col) * source.channels(),
                integer_x,
                integer_y,
                source_x - integer_x,
                source_y - integer_y,
                BORDER_REPLICATE,
                Scalar());
        }
    }
}

}  // namespace detail

inline void getRectSubPix(const Mat& image,
                          Size patchSize,
                          Point2f center,
                          Mat& patch,
                          int patchType = -1)
{
    if (image.empty() || image.dims != 2 ||
        patchSize.width <= 0 || patchSize.height <= 0)
    {
        CV_Error(
            Error::StsBadArg,
            "getRectSubPix expects a non-empty 2D source and positive patch");
    }
    if (!std::isfinite(center.x) || !std::isfinite(center.y) ||
        center.x < 0.0f || center.y < 0.0f ||
        center.x >= image.size[1] ||
        center.y >= image.size[0])
    {
        CV_Error(
            Error::StsOutOfRange,
            "getRectSubPix center must be inside the source image");
    }
    if (image.channels() != 1 &&
        image.channels() != 3 &&
        image.channels() != 4)
    {
        CV_Error(
            Error::StsUnsupportedFormat,
            "getRectSubPix supports C1/C3/C4 source");
    }
    const int output_depth =
        patchType < 0 ? image.depth() : CV_MAT_DEPTH(patchType);
    const bool type_supported =
        (image.depth() == CV_8U &&
         (output_depth == CV_8U || output_depth == CV_32F)) ||
        (image.depth() == CV_32F && output_depth == CV_32F);
    if (!type_supported)
    {
        CV_Error(
            Error::StsUnsupportedFormat,
            "getRectSubPix supports U8->U8/F32 and F32->F32");
    }
    const Mat source =
        image.data == patch.data ? image.clone() : image;
    patch.create(
        {patchSize.height, patchSize.width},
        CV_MAKETYPE(output_depth, source.channels()));
    if (source.depth() == CV_8U && output_depth == CV_8U)
    {
        detail::rect_sub_pix_typed<uchar, uchar>(
            source,
            patch,
            center);
    }
    else if (source.depth() == CV_8U)
    {
        detail::rect_sub_pix_typed<uchar, float>(
            source,
            patch,
            center);
    }
    else
    {
        detail::rect_sub_pix_typed<float, float>(
            source,
            patch,
            center);
    }
}

}  // namespace cvh

#endif  // CVH_IMGPROC_RECT_SUB_PIX_H
