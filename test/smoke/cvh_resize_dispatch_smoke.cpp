#include "cvh/cvh.h"

int main()
{
    cvh::Mat src_gray({2, 2}, CV_8UC1);
    src_gray.at<uchar>(0, 0) = 1;
    src_gray.at<uchar>(0, 1) = 2;
    src_gray.at<uchar>(1, 0) = 3;
    src_gray.at<uchar>(1, 1) = 4;

    cvh::Mat resized;
    cvh::resize(src_gray, resized, cvh::Size(3, 3), 0.0, 0.0, cvh::INTER_NEAREST);

    if (resized.empty() || resized.size[0] != 3 || resized.size[1] != 3 || resized.type() != CV_8UC1)
    {
        return 1;
    }

    const uchar expected[9] = {
        1, 1, 2,
        1, 1, 2,
        3, 3, 4,
    };

    for (int y = 0; y < 3; ++y)
    {
        for (int x = 0; x < 3; ++x)
        {
            if (resized.at<uchar>(y, x) != expected[y * 3 + x])
            {
                return 2;
            }
        }
    }

    cvh::Mat src_bgr({2, 2}, CV_8UC3);
    src_bgr.at<uchar>(0, 0, 0) = 10;  src_bgr.at<uchar>(0, 0, 1) = 20;  src_bgr.at<uchar>(0, 0, 2) = 30;
    src_bgr.at<uchar>(0, 1, 0) = 100; src_bgr.at<uchar>(0, 1, 1) = 110; src_bgr.at<uchar>(0, 1, 2) = 120;
    src_bgr.at<uchar>(1, 0, 0) = 0;   src_bgr.at<uchar>(1, 0, 1) = 0;   src_bgr.at<uchar>(1, 0, 2) = 255;
    src_bgr.at<uchar>(1, 1, 0) = 255; src_bgr.at<uchar>(1, 1, 1) = 0;   src_bgr.at<uchar>(1, 1, 2) = 0;

    cvh::Mat gray;
    cvh::cvtColor(src_bgr, gray, cvh::COLOR_BGR2GRAY);
    if (gray.empty() || gray.type() != CV_8UC1 || gray.size[0] != 2 || gray.size[1] != 2)
    {
        return 3;
    }

    const uchar expected_gray[4] = {22, 112, 76, 29};
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 2; ++x)
        {
            if (gray.at<uchar>(y, x) != expected_gray[y * 2 + x])
            {
                return 4;
            }
        }
    }

    cvh::Mat th;
    const double ret = cvh::threshold(gray, th, 80.0, 255.0, cvh::THRESH_BINARY);
    if (ret != 80.0 || th.empty() || th.type() != CV_8UC1)
    {
        return 5;
    }

    const uchar expected_th[4] = {0, 255, 0, 0};
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 2; ++x)
        {
            if (th.at<uchar>(y, x) != expected_th[y * 2 + x])
            {
                return 6;
            }
        }
    }

#if defined(CVH_EXPECT_FULL)
    if (!cvh::detail::is_resize_backend_registered())
    {
        return 7;
    }
    if (!cvh::detail::is_cvtcolor_backend_registered())
    {
        return 8;
    }
    if (!cvh::detail::is_threshold_backend_registered())
    {
        return 9;
    }
#else
    if (cvh::detail::is_resize_backend_registered())
    {
        return 10;
    }
    if (cvh::detail::is_cvtcolor_backend_registered())
    {
        return 11;
    }
    if (cvh::detail::is_threshold_backend_registered())
    {
        return 12;
    }
#endif

    return 0;
}
