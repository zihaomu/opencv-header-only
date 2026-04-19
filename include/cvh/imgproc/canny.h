#ifndef CVH_IMGPROC_CANNY_H
#define CVH_IMGPROC_CANNY_H

#include "detail/common.h"
#include "sobel.h"

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

namespace cvh {
namespace detail {

using CannyImageFn = void (*)(const Mat&, Mat&, double, double, int, bool);
using CannyDerivFn = void (*)(const Mat&, const Mat&, Mat&, double, double, bool);

inline void canny_from_derivatives_fallback(const Mat& dx,
                                            const Mat& dy,
                                            Mat& edges,
                                            double threshold1,
                                            double threshold2,
                                            bool L2gradient)
{
    CV_Assert(!dx.empty() && "Canny(dx,dy): dx can not be empty");
    CV_Assert(!dy.empty() && "Canny(dx,dy): dy can not be empty");
    CV_Assert(dx.dims == 2 && dy.dims == 2 && "Canny(dx,dy): only 2D Mat is supported");
    CV_Assert(dx.type() == CV_16SC1 && dy.type() == CV_16SC1 &&
              "Canny(dx,dy): only CV_16SC1 derivatives are supported");
    CV_Assert(dx.size[0] == dy.size[0] && dx.size[1] == dy.size[1] && "Canny(dx,dy): dx/dy size mismatch");

    const int rows = dx.size[0];
    const int cols = dx.size[1];
    const size_t dx_step = dx.step(0);
    const size_t dy_step = dy.step(0);

    const float low_threshold = static_cast<float>(std::min(threshold1, threshold2));
    const float high_threshold = static_cast<float>(std::max(threshold1, threshold2));
    const double tan_pi_8 = std::tan(CV_PI / 8.0);
    const double tan_3pi_8 = std::tan(CV_PI * 3.0 / 8.0);

    Mat magnitude({rows, cols}, CV_32FC1);
    const size_t mag_step = magnitude.step(0);
    for (int y = 0; y < rows; ++y)
    {
        const short* dx_row = reinterpret_cast<const short*>(dx.data + static_cast<size_t>(y) * dx_step);
        const short* dy_row = reinterpret_cast<const short*>(dy.data + static_cast<size_t>(y) * dy_step);
        float* mag_row = reinterpret_cast<float*>(magnitude.data + static_cast<size_t>(y) * mag_step);
        for (int x = 0; x < cols; ++x)
        {
            const int gx = dx_row[x];
            const int gy = dy_row[x];
            if (L2gradient)
            {
                mag_row[x] = static_cast<float>(std::sqrt(static_cast<double>(gx) * gx + static_cast<double>(gy) * gy));
            }
            else
            {
                mag_row[x] = static_cast<float>(std::abs(gx) + std::abs(gy));
            }
        }
    }

    for (int y = 0; y < rows; ++y)
    {
        const short* dx_row = reinterpret_cast<const short*>(dx.data + static_cast<size_t>(y) * dx_step);
        const short* dy_row = reinterpret_cast<const short*>(dy.data + static_cast<size_t>(y) * dy_step);
        float* mag_row = reinterpret_cast<float*>(magnitude.data + static_cast<size_t>(y) * mag_step);
        for (int x = 0; x < cols; ++x)
        {
            const float a = mag_row[x];
            if (a <= low_threshold)
            {
                continue;
            }

            const int gx = dx_row[x];
            const int gy = dy_row[x];
            const double tg = gx ? static_cast<double>(gy) / gx : DBL_MAX * (gy >= 0 ? 1.0 : -1.0);

            int x1 = 0;
            int y1 = 0;
            int x2 = 0;
            int y2 = 0;
            if (std::abs(tg) < tan_pi_8)
            {
                y1 = y;
                y2 = y;
                x1 = x + 1;
                x2 = x - 1;
            }
            else if (tan_pi_8 <= tg && tg <= tan_3pi_8)
            {
                y1 = y + 1;
                y2 = y - 1;
                x1 = x + 1;
                x2 = x - 1;
            }
            else if (-tan_3pi_8 <= tg && tg <= -tan_pi_8)
            {
                y1 = y - 1;
                y2 = y + 1;
                x1 = x + 1;
                x2 = x - 1;
            }
            else
            {
                x1 = x;
                x2 = x;
                y1 = y + 1;
                y2 = y - 1;
            }

            float b = 0.0f;
            float c = 0.0f;
            if (static_cast<unsigned>(x1) < static_cast<unsigned>(cols) &&
                static_cast<unsigned>(y1) < static_cast<unsigned>(rows))
            {
                const float* row = reinterpret_cast<const float*>(magnitude.data + static_cast<size_t>(y1) * mag_step);
                b = std::fabs(row[x1]);
            }
            if (static_cast<unsigned>(x2) < static_cast<unsigned>(cols) &&
                static_cast<unsigned>(y2) < static_cast<unsigned>(rows))
            {
                const float* row = reinterpret_cast<const float*>(magnitude.data + static_cast<size_t>(y2) * mag_step);
                c = std::fabs(row[x2]);
            }

            if (!((a > b || (a == b && ((x1 == x + 1 && y1 == y) || (x1 == x && y1 == y + 1)))) && a > c))
            {
                mag_row[x] = -a;
            }
        }
    }

    edges.create(std::vector<int>{rows, cols}, CV_8UC1);
    const size_t edge_step = edges.step(0);
    for (int y = 0; y < rows; ++y)
    {
        uchar* edge_row = edges.data + static_cast<size_t>(y) * edge_step;
        std::fill(edge_row, edge_row + cols, static_cast<uchar>(0));
    }

    static const int kOffsets[8][2] = {
        {1, 0}, {1, -1}, {0, -1}, {-1, -1},
        {-1, 0}, {-1, 1}, {0, 1}, {1, 1},
    };

    std::vector<Point> stack;
    stack.reserve(static_cast<size_t>(rows) * static_cast<size_t>(cols) / 8 + 8);

    for (int y = 0; y < rows; ++y)
    {
        const float* mag_row = reinterpret_cast<const float*>(magnitude.data + static_cast<size_t>(y) * mag_step);
        for (int x = 0; x < cols; ++x)
        {
            if (mag_row[x] <= high_threshold)
            {
                continue;
            }

            uchar* edge_seed_row = edges.data + static_cast<size_t>(y) * edge_step;
            if (edge_seed_row[x] != 0)
            {
                continue;
            }

            edge_seed_row[x] = 255;
            stack.push_back(Point(x, y));

            while (!stack.empty())
            {
                const Point p = stack.back();
                stack.pop_back();

                for (int k = 0; k < 8; ++k)
                {
                    const int nx = p.x + kOffsets[k][0];
                    const int ny = p.y + kOffsets[k][1];
                    if (static_cast<unsigned>(nx) >= static_cast<unsigned>(cols) ||
                        static_cast<unsigned>(ny) >= static_cast<unsigned>(rows))
                    {
                        continue;
                    }

                    uchar* edge_row = edges.data + static_cast<size_t>(ny) * edge_step;
                    if (edge_row[nx] != 0)
                    {
                        continue;
                    }

                    const float* mag_n_row = reinterpret_cast<const float*>(magnitude.data + static_cast<size_t>(ny) * mag_step);
                    if (mag_n_row[nx] > low_threshold)
                    {
                        edge_row[nx] = 255;
                        stack.push_back(Point(nx, ny));
                    }
                }
            }
        }
    }
}

inline void canny_fallback(const Mat& image,
                           Mat& edges,
                           double threshold1,
                           double threshold2,
                           int apertureSize,
                           bool L2gradient)
{
    CV_Assert(!image.empty() && "Canny: source image can not be empty");
    CV_Assert(image.dims == 2 && "Canny: only 2D Mat is supported");
    CV_Assert(image.type() == CV_8UC1 && "Canny: only CV_8UC1 source is supported");
    CV_Assert((apertureSize == 3 || apertureSize == 5) && "Canny: only apertureSize=3/5 is supported");

    Mat src_local;
    const Mat* src_ref = &image;
    if (image.data == edges.data)
    {
        src_local = image.clone();
        src_ref = &src_local;
    }

    Mat dx;
    Mat dy;
    // OpenCV Canny behaves as ROI-isolated for boundary sampling.
    const int sobel_border = BORDER_REPLICATE | BORDER_ISOLATED;
    Sobel(*src_ref, dx, CV_16S, 1, 0, apertureSize, 1.0, 0.0, sobel_border);
    Sobel(*src_ref, dy, CV_16S, 0, 1, apertureSize, 1.0, 0.0, sobel_border);

    canny_from_derivatives_fallback(dx, dy, edges, threshold1, threshold2, L2gradient);
}

inline CannyImageFn& canny_image_dispatch()
{
    static CannyImageFn fn = &canny_fallback;
    return fn;
}

inline CannyDerivFn& canny_deriv_dispatch()
{
    static CannyDerivFn fn = &canny_from_derivatives_fallback;
    return fn;
}

inline void register_canny_image_backend(CannyImageFn fn)
{
    if (fn)
    {
        canny_image_dispatch() = fn;
    }
}

inline void register_canny_deriv_backend(CannyDerivFn fn)
{
    if (fn)
    {
        canny_deriv_dispatch() = fn;
    }
}

inline bool is_canny_image_backend_registered()
{
    return canny_image_dispatch() != &canny_fallback;
}

inline bool is_canny_deriv_backend_registered()
{
    return canny_deriv_dispatch() != &canny_from_derivatives_fallback;
}

}  // namespace detail

inline void Canny(const Mat& image,
                  Mat& edges,
                  double threshold1,
                  double threshold2,
                  int apertureSize = 3,
                  bool L2gradient = false)
{
    detail::ensure_backends_registered_once();
    detail::canny_image_dispatch()(image, edges, threshold1, threshold2, apertureSize, L2gradient);
}

inline void Canny(const Mat& dx,
                  const Mat& dy,
                  Mat& edges,
                  double threshold1,
                  double threshold2,
                  bool L2gradient = false)
{
    detail::ensure_backends_registered_once();
    detail::canny_deriv_dispatch()(dx, dy, edges, threshold1, threshold2, L2gradient);
}

}  // namespace cvh

#endif  // CVH_IMGPROC_CANNY_H
