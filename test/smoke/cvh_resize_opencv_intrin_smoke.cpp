#include "cvh.h"

#if !CVH_ENABLE_OPENCV_INTRIN
#error "cvh_resize_opencv_intrin_smoke must be compiled with CVH_ENABLE_OPENCV_INTRIN=1"
#endif

namespace {

uchar resize_half_reference(const cvh::Mat& src, int y, int x)
{
    const int sy = y * 2;
    const int sx = x * 2;
    return static_cast<uchar>(
        (static_cast<int>(src.at<uchar>(sy, sx)) +
         static_cast<int>(src.at<uchar>(sy, sx + 1)) +
         static_cast<int>(src.at<uchar>(sy + 1, sx)) +
         static_cast<int>(src.at<uchar>(sy + 1, sx + 1)) +
         2) >> 2);
}

}  // namespace

int main()
{
    constexpr int dst_rows = 5;
    constexpr int dst_cols = 37;
    constexpr int src_rows = dst_rows * 2;
    constexpr int src_cols = dst_cols * 2;

    cvh::Mat src({src_rows, src_cols}, CV_8UC1);
    for (int y = 0; y < src_rows; ++y)
    {
        for (int x = 0; x < src_cols; ++x)
        {
            src.at<uchar>(y, x) = static_cast<uchar>((y * 31 + x * 17 + 9) & 0xff);
        }
    }

    cvh::Mat public_dst;
    cvh::resize(src, public_dst, cvh::Size(dst_cols, dst_rows), 0.0, 0.0, cvh::INTER_LINEAR);

    cvh::Mat direct_dst;
    cvh::detail::resize_linear_u8c1_downsample2_opencv_intrin_impl(src, direct_dst);

    if (public_dst.type() != CV_8UC1 ||
        public_dst.size[0] != dst_rows ||
        public_dst.size[1] != dst_cols)
    {
        return 1;
    }
    if (direct_dst.type() != CV_8UC1 ||
        direct_dst.size[0] != dst_rows ||
        direct_dst.size[1] != dst_cols)
    {
        return 2;
    }

    for (int y = 0; y < dst_rows; ++y)
    {
        for (int x = 0; x < dst_cols; ++x)
        {
            const uchar expected = resize_half_reference(src, y, x);
            if (public_dst.at<uchar>(y, x) != expected)
            {
                return 3;
            }
            if (direct_dst.at<uchar>(y, x) != expected)
            {
                return 4;
            }
        }
    }

    return 0;
}
