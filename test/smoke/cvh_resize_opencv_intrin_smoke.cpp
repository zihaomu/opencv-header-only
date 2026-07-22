#include "cvh.h"

#if !CVH_ENABLE_OPENCV_INTRIN
#error "cvh_resize_opencv_intrin_smoke must be compiled with CVH_ENABLE_OPENCV_INTRIN=1"
#endif

#include <vector>

namespace {

void fill_u8(cvh::Mat& mat, int salt)
{
    const int channels = mat.channels();
    for (int y = 0; y < mat.size[0]; ++y)
    {
        for (int x = 0; x < mat.size[1]; ++x)
        {
            for (int c = 0; c < channels; ++c)
            {
                mat.at<uchar>(y, x, c) = static_cast<uchar>(
                    (y * 37 + x * 19 + c * 53 + salt) & 0xff);
            }
        }
    }
}

uchar resize_half_reference(const cvh::Mat& src, int y, int x)
{
    const int sy = y * 2;
    const int sx = x * 2;
    return static_cast<uchar>(
        (static_cast<int>(src.at<uchar>(sy, sx, 0)) +
         static_cast<int>(src.at<uchar>(sy, sx + 1, 0)) +
         static_cast<int>(src.at<uchar>(sy + 1, sx, 0)) +
         static_cast<int>(src.at<uchar>(sy + 1, sx + 1, 0)) +
         2) >> 2);
}

bool same_bytes_2d(const cvh::Mat& lhs, const cvh::Mat& rhs)
{
    if (lhs.dims != 2 ||
        rhs.dims != 2 ||
        lhs.type() != rhs.type() ||
        lhs.size[0] != rhs.size[0] ||
        lhs.size[1] != rhs.size[1])
    {
        return false;
    }

    const size_t row_bytes = static_cast<size_t>(lhs.size[1]) * lhs.elemSize();
    for (int y = 0; y < lhs.size[0]; ++y)
    {
        const uchar* lhs_row = lhs.data + static_cast<size_t>(y) * lhs.step(0);
        const uchar* rhs_row = rhs.data + static_cast<size_t>(y) * rhs.step(0);
        for (size_t x = 0; x < row_bytes; ++x)
        {
            if (lhs_row[x] != rhs_row[x])
            {
                return false;
            }
        }
    }
    return true;
}

int check_half_case(const cvh::Mat& src, int dst_rows, int dst_cols, int error_base)
{
    if (!cvh::detail::resize_linear_u8c1_downsample2_opencv_intrin_supported(
            src,
            dst_rows,
            dst_cols,
            cvh::INTER_LINEAR))
    {
        return error_base;
    }

    cvh::Mat public_dst(std::vector<int>{2, 2, 2}, CV_8UC1);
    cvh::resize(src, public_dst, cvh::Size(dst_cols, dst_rows), 0.0, 0.0, cvh::INTER_LINEAR);

    cvh::Mat direct_dst(std::vector<int>{3, 3, 3}, CV_8UC1);
    cvh::detail::resize_linear_u8c1_downsample2_opencv_intrin_impl(src, direct_dst);

    cvh::Mat scalar_expected(std::vector<int>{dst_rows, dst_cols}, CV_8UC1);
    cvh::detail::resize_fallback_impl_typed<uchar>(src, scalar_expected, cvh::INTER_LINEAR);

    if (public_dst.type() != CV_8UC1 ||
        public_dst.dims != 2 ||
        public_dst.size[0] != dst_rows ||
        public_dst.size[1] != dst_cols)
    {
        return error_base + 1;
    }
    if (direct_dst.type() != CV_8UC1 ||
        direct_dst.dims != 2 ||
        direct_dst.size[0] != dst_rows ||
        direct_dst.size[1] != dst_cols)
    {
        return error_base + 2;
    }

    for (int y = 0; y < dst_rows; ++y)
    {
        for (int x = 0; x < dst_cols; ++x)
        {
            const uchar expected = resize_half_reference(src, y, x);
            if (public_dst.at<uchar>(y, x, 0) != expected)
            {
                return error_base + 3;
            }
            if (direct_dst.at<uchar>(y, x, 0) != expected)
            {
                return error_base + 4;
            }
        }
    }

    if (!same_bytes_2d(public_dst, direct_dst))
    {
        return error_base + 5;
    }
    if (!same_bytes_2d(public_dst, scalar_expected))
    {
        return error_base + 6;
    }
    if (!same_bytes_2d(direct_dst, scalar_expected))
    {
        return error_base + 7;
    }

    return 0;
}

int check_public_fallback_u8(const cvh::Mat& src,
                             int dst_rows,
                             int dst_cols,
                             int interpolation,
                             int error_base)
{
    if (cvh::detail::resize_linear_u8c1_downsample2_opencv_intrin_supported(
            src,
            dst_rows,
            dst_cols,
            interpolation))
    {
        return error_base;
    }

    cvh::Mat actual;
    cvh::resize(src, actual, cvh::Size(dst_cols, dst_rows), 0.0, 0.0, interpolation);

    cvh::Mat expected(std::vector<int>{dst_rows, dst_cols}, src.type());
    cvh::detail::resize_fallback_impl_typed<uchar>(src, expected, interpolation);

    if (!same_bytes_2d(actual, expected))
    {
        return error_base + 1;
    }

    return 0;
}

}  // namespace

int main()
{
    struct HalfCase
    {
        int dst_rows;
        int dst_cols;
        int salt;
    };

    const std::vector<HalfCase> half_cases = {
        {1, 1, 9},
        {3, 16, 11},
        {5, 17, 13},
        {5, 37, 17},
        {4, 64, 19},
    };

    int case_index = 0;
    for (const HalfCase& half_case : half_cases)
    {
        cvh::Mat src({half_case.dst_rows * 2, half_case.dst_cols * 2}, CV_8UC1);
        fill_u8(src, half_case.salt);

        const int result = check_half_case(
            src,
            half_case.dst_rows,
            half_case.dst_cols,
            100 + case_index * 10);
        if (result != 0)
        {
            return result;
        }
        ++case_index;
    }

    cvh::Mat roi_base({14, 90}, CV_8UC1);
    fill_u8(roi_base, 23);
    cvh::Mat roi = roi_base(cvh::Range(2, 12), cvh::Range(5, 79));
    if (roi.isContinuous())
    {
        return 200;
    }
    int result = check_half_case(roi, 5, 37, 210);
    if (result != 0)
    {
        return result;
    }

    cvh::Mat non_2x_u8c1({11, 75}, CV_8UC1);
    fill_u8(non_2x_u8c1, 29);
    result = check_public_fallback_u8(non_2x_u8c1, 5, 37, cvh::INTER_LINEAR, 300);
    if (result != 0)
    {
        return result;
    }

    cvh::Mat exact_2x_u8c3({10, 74}, CV_8UC3);
    fill_u8(exact_2x_u8c3, 31);
    result = check_public_fallback_u8(exact_2x_u8c3, 5, 37, cvh::INTER_LINEAR, 310);
    if (result != 0)
    {
        return result;
    }

    cvh::Mat exact_2x_nearest({10, 74}, CV_8UC1);
    fill_u8(exact_2x_nearest, 37);
    result = check_public_fallback_u8(exact_2x_nearest, 5, 37, cvh::INTER_NEAREST, 320);
    if (result != 0)
    {
        return result;
    }

    cvh::Mat tensor_like({10, 74, 1}, CV_8UC1);
    if (cvh::detail::resize_linear_u8c1_downsample2_opencv_intrin_supported(
            tensor_like,
            5,
            37,
            cvh::INTER_LINEAR))
    {
        return 330;
    }

    return 0;
}
