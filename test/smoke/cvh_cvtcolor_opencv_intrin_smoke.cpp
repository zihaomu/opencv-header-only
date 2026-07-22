#include "cvh.h"

#if !CVH_ENABLE_OPENCV_INTRIN
#error "cvh_cvtcolor_opencv_intrin_smoke must be compiled with CVH_ENABLE_OPENCV_INTRIN=1"
#endif

namespace {

uchar bgr2gray_reference(uchar bb, uchar gg, uchar rr)
{
    constexpr int kB = 7471;
    constexpr int kG = 38470;
    constexpr int kR = 19595;
    constexpr int kRound = 1 << 15;
    return static_cast<uchar>(
        (kB * static_cast<int>(bb) +
         kG * static_cast<int>(gg) +
         kR * static_cast<int>(rr) +
         kRound) >> 16);
}

}  // namespace

int main()
{
    constexpr int rows = 5;
    constexpr int cols = 37;

    cvh::Mat src({rows, cols}, CV_8UC3);
    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            src.at<uchar>(y, x, 0) = static_cast<uchar>((y * 31 + x * 17 + 3) & 0xff);
            src.at<uchar>(y, x, 1) = static_cast<uchar>((y * 13 + x * 29 + 7) & 0xff);
            src.at<uchar>(y, x, 2) = static_cast<uchar>((y * 19 + x * 11 + 23) & 0xff);
        }
    }

    cvh::Mat gray;
    cvh::cvtColor(src, gray, cvh::COLOR_BGR2GRAY);
    cvh::Mat rgb_gray;
    cvh::cvtColor(src, rgb_gray, cvh::COLOR_RGB2GRAY);

    if (gray.type() != CV_8UC1 || gray.size[0] != rows || gray.size[1] != cols)
    {
        return 1;
    }
    if (rgb_gray.type() != CV_8UC1 || rgb_gray.size[0] != rows || rgb_gray.size[1] != cols)
    {
        return 3;
    }

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            const uchar expected = bgr2gray_reference(
                src.at<uchar>(y, x, 0),
                src.at<uchar>(y, x, 1),
                src.at<uchar>(y, x, 2));
            if (gray.at<uchar>(y, x) != expected)
            {
                return 2;
            }
            const uchar rgb_expected = bgr2gray_reference(
                src.at<uchar>(y, x, 2),
                src.at<uchar>(y, x, 1),
                src.at<uchar>(y, x, 0));
            if (rgb_gray.at<uchar>(y, x) != rgb_expected)
            {
                return 4;
            }
        }
    }

    return 0;
}
