#include "cvh.h"

#include <cstdint>

std::uint64_t cvh_imgproc_header_odr_peer()
{
    cvh::Mat src({5, 5}, CV_8UC1);
    for (int y = 0; y < 5; ++y)
    {
        for (int x = 0; x < 5; ++x)
        {
            src.at<uchar>(y, x) = static_cast<uchar>(y * 17 + x * 3);
        }
    }

    cvh::Mat blurred;
    cvh::GaussianBlur(src, blurred, cvh::Size(3, 3), 0.0, 0.0, cvh::BORDER_REPLICATE);

    cvh::Mat edges;
    cvh::Canny(blurred, edges, 8.0, 24.0);

    std::uint64_t checksum = 1469598103934665603ull;
    for (std::size_t i = 0; i < edges.total(); ++i)
    {
        checksum ^= static_cast<std::uint64_t>(edges.data[i]);
        checksum *= 1099511628211ull;
    }
    return checksum;
}
