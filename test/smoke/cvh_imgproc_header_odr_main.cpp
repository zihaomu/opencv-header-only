#include "cvh.h"

#include <cstdint>

std::uint64_t cvh_imgproc_header_odr_peer();

int main()
{
    cvh::Mat src({4, 4}, CV_8UC1);
    for (int y = 0; y < 4; ++y)
    {
        for (int x = 0; x < 4; ++x)
        {
            src.at<uchar>(y, x) = static_cast<uchar>(y * 4 + x);
        }
    }

    cvh::Mat resized;
    cvh::resize(src, resized, cvh::Size(2, 2), 0.0, 0.0, cvh::INTER_LINEAR);

    cvh::Mat bgr;
    cvh::cvtColor(resized, bgr, cvh::COLOR_GRAY2BGR);

    const bool main_ok =
        resized.type() == CV_8UC1 &&
        resized.size[0] == 2 &&
        resized.size[1] == 2 &&
        bgr.type() == CV_8UC3 &&
        bgr.at<uchar>(0, 0, 0) == resized.at<uchar>(0, 0);

    return main_ok && cvh_imgproc_header_odr_peer() != 0 ? 0 : 1;
}
