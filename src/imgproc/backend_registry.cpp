#include "cvh/imgproc/imgproc.h"

namespace cvh
{
namespace detail
{
void resize_backend_impl(const Mat& src, Mat& dst, Size dsize, double fx, double fy, int interpolation);
void cvtColor_backend_impl(const Mat& src, Mat& dst, int code);
double threshold_backend_impl(const Mat& src, Mat& dst, double thresh, double maxval, int type);
void lut_backend_impl(const Mat& src, const Mat& lut, Mat& dst);
void boxFilter_backend_impl(const Mat& src, Mat& dst, int ddepth, Size ksize, Point anchor, bool normalize, int borderType);
void gaussianBlur_backend_impl(const Mat& src, Mat& dst, Size ksize, double sigmaX, double sigmaY, int borderType);
void sobel_backend_impl(const Mat& src, Mat& dst, int ddepth, int dx, int dy, int ksize, double scale, double delta, int borderType);
void canny_image_backend_impl(const Mat& image, Mat& edges, double threshold1, double threshold2, int apertureSize, bool L2gradient);
void canny_deriv_backend_impl(const Mat& dx, const Mat& dy, Mat& edges, double threshold1, double threshold2, bool L2gradient);
void copy_make_border_backend_impl(const Mat& src,
                                   Mat& dst,
                                   int top,
                                   int bottom,
                                   int left,
                                   int right,
                                   int borderType,
                                   const Scalar& value);
void filter2d_backend_impl(const Mat& src, Mat& dst, int ddepth, const Mat& kernel, Point anchor, double delta, int borderType);
void sep_filter2d_backend_impl(const Mat& src,
                               Mat& dst,
                               int ddepth,
                               const Mat& kernelX,
                               const Mat& kernelY,
                               Point anchor,
                               double delta,
                               int borderType);
void warp_affine_backend_impl(const Mat& src,
                              Mat& dst,
                              const Mat& M,
                              Size dsize,
                              int flags,
                              int borderMode,
                              const Scalar& borderValue);
void erode_backend_impl(const Mat& src,
                        Mat& dst,
                        const Mat& kernel,
                        Point anchor,
                        int iterations,
                        int borderType,
                        const Scalar& borderValue);
void dilate_backend_impl(const Mat& src,
                         Mat& dst,
                         const Mat& kernel,
                         Point anchor,
                         int iterations,
                         int borderType,
                         const Scalar& borderValue);
} // namespace detail

void register_all_backends()
{
    static bool initialized = []() {
        detail::register_resize_backend(&detail::resize_backend_impl);
        detail::register_cvtcolor_backend(&detail::cvtColor_backend_impl);
        detail::register_threshold_backend(&detail::threshold_backend_impl);
        detail::register_lut_backend(&detail::lut_backend_impl);
        detail::register_boxfilter_backend(&detail::boxFilter_backend_impl);
        detail::register_gaussianblur_backend(&detail::gaussianBlur_backend_impl);
        detail::register_sobel_backend(&detail::sobel_backend_impl);
        detail::register_canny_image_backend(&detail::canny_image_backend_impl);
        detail::register_canny_deriv_backend(&detail::canny_deriv_backend_impl);
        detail::register_copy_make_border_backend(&detail::copy_make_border_backend_impl);
        detail::register_filter2d_backend(&detail::filter2d_backend_impl);
        detail::register_sep_filter2d_backend(&detail::sep_filter2d_backend_impl);
        detail::register_warp_affine_backend(&detail::warp_affine_backend_impl);
        detail::register_erode_backend(&detail::erode_backend_impl);
        detail::register_dilate_backend(&detail::dilate_backend_impl);
        return true;
    }();
    (void)initialized;
}

} // namespace cvh
