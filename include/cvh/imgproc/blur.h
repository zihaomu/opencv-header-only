#ifndef CVH_IMGPROC_BLUR_H
#define CVH_IMGPROC_BLUR_H

#include "box_filter.h"

namespace cvh {

inline void blur(const Mat& src,
                 Mat& dst,
                 Size ksize,
                 Point anchor = Point(-1, -1),
                 int borderType = BORDER_DEFAULT)
{
    boxFilter(src, dst, -1, ksize, anchor, true, borderType);
}

}  // namespace cvh

#endif  // CVH_IMGPROC_BLUR_H
