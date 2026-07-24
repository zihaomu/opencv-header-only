#ifndef CVH_CORE_ARRAY_H
#define CVH_CORE_ARRAY_H

#include "mat.h"

#include <vector>

namespace cvh
{

enum BorderTypes
{
    BORDER_CONSTANT = 0,
    BORDER_REPLICATE = 1,
    BORDER_REFLECT = 2,
    BORDER_WRAP = 3,
    BORDER_REFLECT_101 = 4,
    BORDER_TRANSPARENT = 5,

    BORDER_REFLECT101 = BORDER_REFLECT_101,
    BORDER_DEFAULT = BORDER_REFLECT_101,
    BORDER_ISOLATED = 16,
};

enum RotateFlags
{
    ROTATE_90_CLOCKWISE = 0,
    ROTATE_180 = 1,
    ROTATE_90_COUNTERCLOCKWISE = 2,
};

int borderInterpolate(int p, int len, int borderType);

void copyTo(const Mat& src, Mat& dst, const Mat& mask = Mat());

void extractChannel(const Mat& src, Mat& dst, int coi);
void insertChannel(const Mat& src, Mat& dst, int coi);

void mixChannels(const Mat* src,
                 size_t nsrcs,
                 Mat* dst,
                 size_t ndsts,
                 const int* fromTo,
                 size_t npairs);
void mixChannels(const std::vector<Mat>& src,
                 std::vector<Mat>& dst,
                 const std::vector<int>& fromTo);

void flip(const Mat& src, Mat& dst, int flipCode);
void flipND(const Mat& src, Mat& dst, int axis);
void rotate(const Mat& src, Mat& dst, int rotateCode);
void repeat(const Mat& src, int ny, int nx, Mat& dst);

void hconcat(const Mat* src, size_t nsrc, Mat& dst);
void hconcat(const Mat& src1, const Mat& src2, Mat& dst);
void hconcat(const std::vector<Mat>& src, Mat& dst);
void vconcat(const Mat* src, size_t nsrc, Mat& dst);
void vconcat(const Mat& src1, const Mat& src2, Mat& dst);
void vconcat(const std::vector<Mat>& src, Mat& dst);

void broadcast(const Mat& src, const std::vector<int>& shape, Mat& dst);
void broadcast(const Mat& src, const Mat& shape, Mat& dst);

void swap(Mat& a, Mat& b);

}  // namespace cvh

#include "detail/array_impl.hpp"

#endif  // CVH_CORE_ARRAY_H
