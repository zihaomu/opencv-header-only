#include "cvh/imgproc/blur.h"
#include "cvh/imgproc/box_filter.h"
#include "cvh/imgproc/canny.h"
#include "cvh/imgproc/copy_make_border.h"
#include "cvh/imgproc/cvtcolor.h"
#include "cvh/imgproc/filter2d.h"
#include "cvh/imgproc/gaussian_blur.h"
#include "cvh/imgproc/lut.h"
#include "cvh/imgproc/morphology.h"
#include "cvh/imgproc/resize.h"
#include "cvh/imgproc/sep_filter2d.h"
#include "cvh/imgproc/sobel.h"
#include "cvh/imgproc/threshold.h"
#include "cvh/imgproc/warp_affine.h"
#include "gtest/gtest.h"

TEST(ImgprocHeaderLayout_TEST, operator_headers_are_individually_includable)
{
    SUCCEED();
}
