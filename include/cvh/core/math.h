#ifndef CVH_CORE_MATH_H
#define CVH_CORE_MATH_H

#include "mat.h"

#include <cfloat>

namespace cvh
{

void scaleAdd(const Mat& src1, double alpha, const Mat& src2, Mat& dst);
void convertScaleAbs(const Mat& src, Mat& dst, double alpha = 1.0, double beta = 0.0);
void convertFp16(const Mat& src, Mat& dst);

void sqrt(const Mat& src, Mat& dst);
void pow(const Mat& src, double power, Mat& dst);
void exp(const Mat& src, Mat& dst);
void log(const Mat& src, Mat& dst);

bool checkRange(const Mat& src,
                bool quiet = true,
                Point* pos = nullptr,
                double minVal = -DBL_MAX,
                double maxVal = DBL_MAX);

void patchNaNs(Mat& src, double value = 0.0);

}  // namespace cvh

#include "detail/math_impl.hpp"

#endif  // CVH_CORE_MATH_H
