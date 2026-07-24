#ifndef CVH_IMGPROC_ACCUMULATE_H
#define CVH_IMGPROC_ACCUMULATE_H

#include "detail/common.h"

#include <cmath>

namespace cvh
{
namespace accumulate_detail
{

enum class Operation
{
    Add,
    Square,
    Product,
    Weighted,
};

inline void validate_source(const Mat& src, const char* name)
{
    if (src.empty() || src.dims != 2 ||
        (src.depth() != CV_8U && src.depth() != CV_32F) ||
        (src.channels() != 1 && src.channels() != 3 &&
         src.channels() != 4))
    {
        CV_Error_(Error::StsBadArg, ("%s unsupported source", name));
    }
}

inline void validate_destination(const Mat& src,
                                 const Mat& dst,
                                 const char* name)
{
    if (dst.empty() || dst.dims != 2 || dst.depth() != CV_32F ||
        dst.channels() != src.channels() ||
        dst.size.p[0] != src.size.p[0] ||
        dst.size.p[1] != src.size.p[1])
    {
        CV_Error_(
            Error::StsBadArg,
            ("%s requires preinitialized matching CV_32F destination",
             name));
    }
}

inline void validate_mask(const Mat& src,
                          const Mat& mask,
                          const char* name)
{
    if (!mask.empty() &&
        (mask.dims != 2 || mask.type() != CV_8UC1 ||
         mask.size.p[0] != src.size.p[0] ||
         mask.size.p[1] != src.size.p[1]))
    {
        CV_Error_(Error::StsBadArg, ("%s invalid mask", name));
    }
}

inline double read_source(const Mat& src,
                          const uchar* row,
                          size_t index)
{
    return src.depth() == CV_8U
               ? static_cast<double>(row[index])
               : static_cast<double>(
                     reinterpret_cast<const float*>(row)[index]);
}

inline void run(const Mat& src1,
                const Mat* src2,
                Mat& dst,
                const Mat& mask,
                Operation operation,
                double alpha)
{
    const int rows = src1.size.p[0];
    const int cols = src1.size.p[1];
    const int channels = src1.channels();
    for (int y = 0; y < rows; ++y)
    {
        const uchar* input1 =
            src1.data + static_cast<size_t>(y) * src1.step(0);
        const uchar* input2 =
            src2
                ? src2->data + static_cast<size_t>(y) * src2->step(0)
                : nullptr;
        const uchar* mask_row =
            mask.empty()
                ? nullptr
                : mask.data + static_cast<size_t>(y) * mask.step(0);
        float* output = reinterpret_cast<float*>(
            dst.data + static_cast<size_t>(y) * dst.step(0));
        for (int x = 0; x < cols; ++x)
        {
            if (mask_row && mask_row[x] == 0)
            {
                continue;
            }
            for (int ch = 0; ch < channels; ++ch)
            {
                const size_t index =
                    static_cast<size_t>(x) * channels +
                    static_cast<size_t>(ch);
                const double value1 =
                    read_source(src1, input1, index);
                double value = value1;
                switch (operation)
                {
                    case Operation::Add:
                        output[index] += static_cast<float>(value);
                        break;
                    case Operation::Square:
                        output[index] +=
                            static_cast<float>(value * value);
                        break;
                    case Operation::Product:
                        value *= read_source(*src2, input2, index);
                        output[index] += static_cast<float>(value);
                        break;
                    case Operation::Weighted:
                        output[index] = static_cast<float>(
                            (1.0 - alpha) *
                                static_cast<double>(output[index]) +
                            alpha * value);
                        break;
                }
            }
        }
    }
}

}  // namespace accumulate_detail

inline void accumulate(const Mat& src,
                       Mat& dst,
                       const Mat& mask = Mat())
{
    accumulate_detail::validate_source(src, "accumulate");
    accumulate_detail::validate_destination(src, dst, "accumulate");
    accumulate_detail::validate_mask(src, mask, "accumulate");
    accumulate_detail::run(
        src,
        nullptr,
        dst,
        mask,
        accumulate_detail::Operation::Add,
        0.0);
}

inline void accumulateSquare(const Mat& src,
                             Mat& dst,
                             const Mat& mask = Mat())
{
    accumulate_detail::validate_source(src, "accumulateSquare");
    accumulate_detail::validate_destination(
        src, dst, "accumulateSquare");
    accumulate_detail::validate_mask(src, mask, "accumulateSquare");
    accumulate_detail::run(
        src,
        nullptr,
        dst,
        mask,
        accumulate_detail::Operation::Square,
        0.0);
}

inline void accumulateProduct(const Mat& src1,
                              const Mat& src2,
                              Mat& dst,
                              const Mat& mask = Mat())
{
    accumulate_detail::validate_source(src1, "accumulateProduct");
    if (src2.empty() || src2.type() != src1.type() ||
        src2.dims != src1.dims ||
        src2.size.p[0] != src1.size.p[0] ||
        src2.size.p[1] != src1.size.p[1])
    {
        CV_Error(
            Error::StsBadArg,
            "accumulateProduct requires matching sources");
    }
    accumulate_detail::validate_destination(
        src1, dst, "accumulateProduct");
    accumulate_detail::validate_mask(
        src1, mask, "accumulateProduct");
    accumulate_detail::run(
        src1,
        &src2,
        dst,
        mask,
        accumulate_detail::Operation::Product,
        0.0);
}

inline void accumulateWeighted(const Mat& src,
                               Mat& dst,
                               double alpha,
                               const Mat& mask = Mat())
{
    accumulate_detail::validate_source(src, "accumulateWeighted");
    accumulate_detail::validate_destination(
        src, dst, "accumulateWeighted");
    accumulate_detail::validate_mask(src, mask, "accumulateWeighted");
    if (!std::isfinite(alpha))
    {
        CV_Error(
            Error::StsBadArg,
            "accumulateWeighted alpha must be finite");
    }
    accumulate_detail::run(
        src,
        nullptr,
        dst,
        mask,
        accumulate_detail::Operation::Weighted,
        alpha);
}

}  // namespace cvh

#endif  // CVH_IMGPROC_ACCUMULATE_H
