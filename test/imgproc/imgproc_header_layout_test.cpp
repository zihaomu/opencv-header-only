#include "cvh/imgproc/blur.h"
#include "cvh/imgproc/box_filter.h"
#include "cvh/imgproc/cvtcolor.h"
#include "cvh/imgproc/gaussian_blur.h"
#include "cvh/imgproc/resize.h"
#include "cvh/imgproc/threshold.h"
#include "gtest/gtest.h"

TEST(ImgprocHeaderLayout_TEST, operator_headers_are_individually_includable)
{
    SUCCEED();
}
