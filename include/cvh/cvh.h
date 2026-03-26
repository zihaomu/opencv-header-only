// main header of this project, include this in every source file instead of including opencv headers directly
// cvh means "opencv header only", it is a header-only library that provides a subset of opencv functionalities, and it is designed to be easy to use and integrate into other projects.

#ifndef CVH_OPENCV_HEADER_H
#define CVH_OPENCV_HEADER_H

#include "detail/config.h"
#include "core/define.h"
#include "core/types.h"
#include "core/mat.h"
#include "core/basic_op.h"
#include "core/utils.h"

#include "imgproc/imgproc.h"
#include "imgcodecs/imgcodecs.h"

// TODO: add more opencv functionalities, such as image processing, video processing, etc. and make sure all the opencv functionalities are implemented in a header-only way, and do not include any opencv headers in the implementation files, only include this header file.


#endif //CVH_OPENCV_HEADER_H
