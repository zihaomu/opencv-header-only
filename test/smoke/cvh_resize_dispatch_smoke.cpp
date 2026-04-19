#include "cvh/cvh.h"

#include <cstring>

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
    if (!cvh::detail::is_lut_backend_registered())
    {
        return 33;
    }
    if (!cvh::detail::is_sobel_backend_registered())
    {
        return 17;
    }
    if (!cvh::detail::is_canny_image_backend_registered() || !cvh::detail::is_canny_deriv_backend_registered())
    {
        return 26;
    }
    if (!cvh::detail::is_copy_make_border_backend_registered())
    {
        return 29;
    }
    if (!cvh::detail::is_filter2d_backend_registered())
    {
        return 41;
    }
    if (!cvh::detail::is_sep_filter2d_backend_registered())
    {
        return 45;
    }
    if (!cvh::detail::is_warp_affine_backend_registered())
    {
        return 37;
    }
    if (!cvh::detail::is_erode_backend_registered())
    {
        return 18;
    }
    if (!cvh::detail::is_dilate_backend_registered())
    {
        return 19;
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
    if (cvh::detail::is_lut_backend_registered())
    {
        return 34;
    }
    if (cvh::detail::is_sobel_backend_registered())
    {
        return 20;
    }
    if (cvh::detail::is_canny_image_backend_registered() || cvh::detail::is_canny_deriv_backend_registered())
    {
        return 27;
    }
    if (cvh::detail::is_copy_make_border_backend_registered())
    {
        return 30;
    }
    if (cvh::detail::is_filter2d_backend_registered())
    {
        return 42;
    }
    if (cvh::detail::is_sep_filter2d_backend_registered())
    {
        return 46;
    }
    if (cvh::detail::is_warp_affine_backend_registered())
    {
        return 38;
    }
    if (cvh::detail::is_erode_backend_registered())
    {
        return 21;
    }
    if (cvh::detail::is_dilate_backend_registered())
    {
        return 22;
    }
#endif

    cvh::Mat blur_dst;
    cvh::blur(src_gray, blur_dst, cvh::Size(3, 3), cvh::Point(-1, -1), cvh::BORDER_REPLICATE);
    const char* expected_box_path =
#if defined(CVH_EXPECT_FULL)
        "box3x3";
#else
        "fallback";
#endif
    if (std::strcmp(cvh::detail::last_boxfilter_dispatch_path(), expected_box_path) != 0)
    {
        return 13;
    }

    cvh::Mat blur5_dst;
    cvh::blur(src_gray, blur5_dst, cvh::Size(5, 5), cvh::Point(-1, -1), cvh::BORDER_REPLICATE);
    const char* expected_box5_path =
#if defined(CVH_EXPECT_FULL)
        "box_generic";
#else
        "fallback";
#endif
    if (std::strcmp(cvh::detail::last_boxfilter_dispatch_path(), expected_box5_path) != 0)
    {
        return 14;
    }

    cvh::Mat gauss_dst;
    cvh::GaussianBlur(src_gray, gauss_dst, cvh::Size(5, 5), 0.0, 0.0, cvh::BORDER_REPLICATE);
    const char* expected_gauss_path =
#if defined(CVH_EXPECT_FULL)
        "gauss_separable";
#else
        "fallback";
#endif
    if (std::strcmp(cvh::detail::last_gaussianblur_dispatch_path(), expected_gauss_path) != 0)
    {
        return 15;
    }

    cvh::Mat gauss3_dst;
    cvh::GaussianBlur(src_gray, gauss3_dst, cvh::Size(3, 3), 0.0, 0.0, cvh::BORDER_REPLICATE);
    const char* expected_gauss3_path =
#if defined(CVH_EXPECT_FULL)
        "gauss3x3";
#else
        "fallback";
#endif
    if (std::strcmp(cvh::detail::last_gaussianblur_dispatch_path(), expected_gauss3_path) != 0)
    {
        return 16;
    }

    cvh::Mat sobel_dst;
    cvh::Sobel(src_gray, sobel_dst, CV_32F, 1, 0, 3, 1.0, 0.0, cvh::BORDER_REPLICATE);
    if (sobel_dst.empty() || sobel_dst.type() != CV_32FC1 || sobel_dst.size[0] != 2 || sobel_dst.size[1] != 2)
    {
        return 23;
    }

    cvh::Mat erode_dst;
    cvh::erode(src_gray, erode_dst, cvh::Mat(), cvh::Point(-1, -1), 1, cvh::BORDER_REPLICATE);
    if (erode_dst.empty() || erode_dst.type() != CV_8UC1 || erode_dst.at<uchar>(0, 0) != 1 ||
        erode_dst.at<uchar>(0, 1) != 1 || erode_dst.at<uchar>(1, 0) != 1 || erode_dst.at<uchar>(1, 1) != 1)
    {
        return 24;
    }

    cvh::Mat dilate_dst;
    cvh::dilate(src_gray, dilate_dst, cvh::Mat(), cvh::Point(-1, -1), 1, cvh::BORDER_REPLICATE);
    if (dilate_dst.empty() || dilate_dst.type() != CV_8UC1 || dilate_dst.at<uchar>(0, 0) != 4 ||
        dilate_dst.at<uchar>(0, 1) != 4 || dilate_dst.at<uchar>(1, 0) != 4 || dilate_dst.at<uchar>(1, 1) != 4)
    {
        return 25;
    }

    cvh::Mat canny_dst;
    cvh::Canny(src_gray, canny_dst, 5.0, 10.0, 3, false);
    if (canny_dst.empty() || canny_dst.type() != CV_8UC1 || canny_dst.size[0] != 2 || canny_dst.size[1] != 2)
    {
        return 28;
    }

    cvh::Mat border_dst;
    cvh::copyMakeBorder(src_gray, border_dst, 1, 1, 2, 2, cvh::BORDER_REPLICATE);
    if (border_dst.empty() || border_dst.type() != CV_8UC1 || border_dst.size[0] != 4 || border_dst.size[1] != 6)
    {
        return 31;
    }
    if (border_dst.at<uchar>(0, 0) != 1 || border_dst.at<uchar>(0, 5) != 2 ||
        border_dst.at<uchar>(3, 0) != 3 || border_dst.at<uchar>(3, 5) != 4 ||
        border_dst.at<uchar>(1, 2) != 1 || border_dst.at<uchar>(2, 3) != 4)
    {
        return 32;
    }

    cvh::Mat lut_table({1, 256}, CV_8UC1);
    for (int i = 0; i < 256; ++i)
    {
        lut_table.at<uchar>(0, i) = static_cast<uchar>(255 - i);
    }

    cvh::Mat lut_dst;
    cvh::LUT(src_gray, lut_table, lut_dst);
    if (lut_dst.empty() || lut_dst.type() != CV_8UC1 || lut_dst.size[0] != 2 || lut_dst.size[1] != 2)
    {
        return 35;
    }
    if (lut_dst.at<uchar>(0, 0) != 254 || lut_dst.at<uchar>(0, 1) != 253 ||
        lut_dst.at<uchar>(1, 0) != 252 || lut_dst.at<uchar>(1, 1) != 251)
    {
        return 36;
    }

    cvh::Mat kernel({3, 3}, CV_32FC1);
    kernel.setTo(0.0f);
    kernel.at<float>(1, 1) = 1.0f;

    cvh::Mat filter_dst;
    cvh::filter2D(src_gray, filter_dst, -1, kernel, cvh::Point(-1, -1), 0.0, cvh::BORDER_REPLICATE);
    if (filter_dst.empty() || filter_dst.type() != CV_8UC1 || filter_dst.size[0] != 2 || filter_dst.size[1] != 2)
    {
        return 43;
    }
    if (filter_dst.at<uchar>(0, 0) != 1 || filter_dst.at<uchar>(0, 1) != 2 ||
        filter_dst.at<uchar>(1, 0) != 3 || filter_dst.at<uchar>(1, 1) != 4)
    {
        return 44;
    }

    cvh::Mat kernel_x({1, 3}, CV_32FC1);
    kernel_x.setTo(0.0f);
    kernel_x.at<float>(0, 1) = 1.0f;
    cvh::Mat kernel_y({3, 1}, CV_32FC1);
    kernel_y.setTo(0.0f);
    kernel_y.at<float>(1, 0) = 1.0f;

    cvh::Mat sep_dst;
    cvh::sepFilter2D(src_gray, sep_dst, -1, kernel_x, kernel_y, cvh::Point(-1, -1), 0.0, cvh::BORDER_REPLICATE);
    if (sep_dst.empty() || sep_dst.type() != CV_8UC1 || sep_dst.size[0] != 2 || sep_dst.size[1] != 2)
    {
        return 47;
    }
    if (sep_dst.at<uchar>(0, 0) != 1 || sep_dst.at<uchar>(0, 1) != 2 ||
        sep_dst.at<uchar>(1, 0) != 3 || sep_dst.at<uchar>(1, 1) != 4)
    {
        return 48;
    }

    cvh::Mat M({2, 3}, CV_32FC1);
    M.setTo(0.0f);
    M.at<float>(0, 0) = 1.0f;
    M.at<float>(1, 1) = 1.0f;

    cvh::Mat warp_dst;
    cvh::warpAffine(src_gray, warp_dst, M, cvh::Size(2, 2), cvh::INTER_NEAREST, cvh::BORDER_CONSTANT, cvh::Scalar::all(0.0));
    if (warp_dst.empty() || warp_dst.type() != CV_8UC1 || warp_dst.size[0] != 2 || warp_dst.size[1] != 2)
    {
        return 39;
    }
    if (warp_dst.at<uchar>(0, 0) != 1 || warp_dst.at<uchar>(0, 1) != 2 ||
        warp_dst.at<uchar>(1, 0) != 3 || warp_dst.at<uchar>(1, 1) != 4)
    {
        return 40;
    }

    return 0;
}
