#include "cvh/imgproc/imgproc.h"

namespace cvh
{
namespace detail
{

void resize_backend_impl(const Mat& src, Mat& dst, Size dsize, double fx, double fy, int interpolation)
{
    // Placeholder full-backend entry. Keep behavior identical to fallback for PR-2 sample.
    resize_fallback(src, dst, dsize, fx, fy, interpolation);
}

void cvtColor_backend_impl(const Mat& src, Mat& dst, int code)
{
    cvtColor_fallback(src, dst, code);
}

double threshold_backend_impl(const Mat& src, Mat& dst, double thresh, double maxval, int type)
{
    return threshold_fallback(src, dst, thresh, maxval, type);
}

void boxFilter_backend_impl(const Mat& src,
                            Mat& dst,
                            int ddepth,
                            Size ksize,
                            Point anchor,
                            bool normalize,
                            int borderType)
{
    boxFilter_fallback(src, dst, ddepth, ksize, anchor, normalize, borderType);
}

void gaussianBlur_backend_impl(const Mat& src, Mat& dst, Size ksize, double sigmaX, double sigmaY, int borderType)
{
    gaussian_blur_fallback(src, dst, ksize, sigmaX, sigmaY, borderType);
}

} // namespace detail

void register_all_backends()
{
    static bool initialized = []() {
        detail::register_resize_backend(&detail::resize_backend_impl);
        detail::register_cvtcolor_backend(&detail::cvtColor_backend_impl);
        detail::register_threshold_backend(&detail::threshold_backend_impl);
        detail::register_boxfilter_backend(&detail::boxFilter_backend_impl);
        detail::register_gaussianblur_backend(&detail::gaussianBlur_backend_impl);
        return true;
    }();
    (void)initialized;
}

} // namespace cvh
