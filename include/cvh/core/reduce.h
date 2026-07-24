#ifndef CVH_CORE_REDUCE_H
#define CVH_CORE_REDUCE_H

#include "mat.h"

#include <vector>

namespace cvh
{

enum NormTypes
{
    NORM_INF = 1,
    NORM_L1 = 2,
    NORM_L2 = 4,
    NORM_MINMAX = 32,
};

enum ReduceTypes
{
    REDUCE_SUM = 0,
    REDUCE_AVG = 1,
    REDUCE_MAX = 2,
    REDUCE_MIN = 3,
    REDUCE_SUM2 = 4,
};

Scalar sum(const Mat& src);
Scalar mean(const Mat& src, const Mat& mask = Mat());
void meanStdDev(const Mat& src,
                Scalar& mean_value,
                Scalar& stddev_value,
                const Mat& mask = Mat());

double norm(const Mat& src, int normType = NORM_L2, const Mat& mask = Mat());
double norm(const Mat& src1,
            const Mat& src2,
            int normType = NORM_L2,
            const Mat& mask = Mat());

int countNonZero(const Mat& src);
bool hasNonZero(const Mat& src);
void findNonZero(const Mat& src, std::vector<Point>& indices);
void findNonZero(const Mat& src, Mat& indices);

void minMaxIdx(const Mat& src,
               double* minVal,
               double* maxVal = nullptr,
               int* minIdx = nullptr,
               int* maxIdx = nullptr,
               const Mat& mask = Mat());
void minMaxLoc(const Mat& src,
               double* minVal,
               double* maxVal = nullptr,
               Point* minLoc = nullptr,
               Point* maxLoc = nullptr,
               const Mat& mask = Mat());

void reduce(const Mat& src, Mat& dst, int dim, int rtype, int dtype = -1);
void reduceArgMin(const Mat& src, Mat& dst, int axis, bool lastIndex = false);
void reduceArgMax(const Mat& src, Mat& dst, int axis, bool lastIndex = false);

void normalize(const Mat& src,
               Mat& dst,
               double alpha = 1.0,
               double beta = 0.0,
               int normType = NORM_L2,
               int dtype = -1,
               const Mat& mask = Mat());

}  // namespace cvh

#include "detail/reduce_impl.hpp"

#endif  // CVH_CORE_REDUCE_H
