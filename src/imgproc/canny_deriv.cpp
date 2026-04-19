#include "fastpath_common.h"

namespace cvh
{
namespace detail
{

namespace
{
bool try_canny_from_derivatives_fastpath_s16(const Mat& dx,
                                             const Mat& dy,
                                             Mat& edges,
                                             double threshold1,
                                             double threshold2,
                                             bool L2gradient)
{
    if (dx.empty() || dy.empty())
    {
        return false;
    }
    if (dx.dims != 2 || dy.dims != 2)
    {
        return false;
    }
    if (dx.type() != CV_16SC1 || dy.type() != CV_16SC1)
    {
        return false;
    }
    if (dx.size[0] != dy.size[0] || dx.size[1] != dy.size[1])
    {
        return false;
    }

    const int rows = dx.size[0];
    const int cols = dx.size[1];
    if (rows <= 0 || cols <= 0)
    {
        return false;
    }

    const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    const std::size_t dx_step = dx.step(0);
    const std::size_t dy_step = dy.step(0);
    const float low_threshold = static_cast<float>(std::min(threshold1, threshold2));
    const float high_threshold = static_cast<float>(std::max(threshold1, threshold2));
    const double tan_pi_8 = std::tan(CV_PI / 8.0);
    const double tan_3pi_8 = std::tan(CV_PI * 3.0 / 8.0);

    std::vector<float> magnitude(count, 0.0f);
    const bool do_parallel_mag = should_parallelize_filter_rows(rows, cols, 1, 1);
    parallel_for_index_if(do_parallel_mag, rows, [&](int y) {
        const short* dx_row = reinterpret_cast<const short*>(dx.data + static_cast<std::size_t>(y) * dx_step);
        const short* dy_row = reinterpret_cast<const short*>(dy.data + static_cast<std::size_t>(y) * dy_step);
        float* mag_row = magnitude.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(cols);
        for (int x = 0; x < cols; ++x)
        {
            const int gx = dx_row[x];
            const int gy = dy_row[x];
            if (L2gradient)
            {
                mag_row[x] =
                    static_cast<float>(std::sqrt(static_cast<double>(gx) * gx + static_cast<double>(gy) * gy));
            }
            else
            {
                mag_row[x] = static_cast<float>(std::abs(gx) + std::abs(gy));
            }
        }
    });

    std::vector<float> nms = magnitude;
    const bool do_parallel_nms = should_parallelize_filter_rows(rows, cols, 1, 3);
    parallel_for_index_if(do_parallel_nms, rows, [&](int y) {
        const short* dx_row = reinterpret_cast<const short*>(dx.data + static_cast<std::size_t>(y) * dx_step);
        const short* dy_row = reinterpret_cast<const short*>(dy.data + static_cast<std::size_t>(y) * dy_step);
        for (int x = 0; x < cols; ++x)
        {
            const std::size_t idx = static_cast<std::size_t>(y) * static_cast<std::size_t>(cols) +
                                    static_cast<std::size_t>(x);
            const float a = magnitude[idx];
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
                b = magnitude[static_cast<std::size_t>(y1) * static_cast<std::size_t>(cols) +
                              static_cast<std::size_t>(x1)];
            }
            if (static_cast<unsigned>(x2) < static_cast<unsigned>(cols) &&
                static_cast<unsigned>(y2) < static_cast<unsigned>(rows))
            {
                c = magnitude[static_cast<std::size_t>(y2) * static_cast<std::size_t>(cols) +
                              static_cast<std::size_t>(x2)];
            }

            if (!((a > b || (a == b && ((x1 == x + 1 && y1 == y) || (x1 == x && y1 == y + 1)))) && a > c))
            {
                nms[idx] = -a;
            }
        }
    });

    std::vector<uchar> edge_map(count, static_cast<uchar>(0));
    static const int kOffsets[8][2] = {
        {1, 0}, {1, -1}, {0, -1}, {-1, -1},
        {-1, 0}, {-1, 1}, {0, 1}, {1, 1},
    };

    std::vector<int> stack;
    stack.reserve(count / 8u + 8u);

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            const int seed_idx = y * cols + x;
            if (nms[static_cast<std::size_t>(seed_idx)] <= high_threshold)
            {
                continue;
            }
            if (edge_map[static_cast<std::size_t>(seed_idx)] != 0)
            {
                continue;
            }

            edge_map[static_cast<std::size_t>(seed_idx)] = 255;
            stack.push_back(seed_idx);

            while (!stack.empty())
            {
                const int p = stack.back();
                stack.pop_back();
                const int px = p % cols;
                const int py = p / cols;

                for (int k = 0; k < 8; ++k)
                {
                    const int nx = px + kOffsets[k][0];
                    const int ny = py + kOffsets[k][1];
                    if (static_cast<unsigned>(nx) >= static_cast<unsigned>(cols) ||
                        static_cast<unsigned>(ny) >= static_cast<unsigned>(rows))
                    {
                        continue;
                    }

                    const int nidx = ny * cols + nx;
                    if (edge_map[static_cast<std::size_t>(nidx)] != 0)
                    {
                        continue;
                    }

                    if (nms[static_cast<std::size_t>(nidx)] > low_threshold)
                    {
                        edge_map[static_cast<std::size_t>(nidx)] = 255;
                        stack.push_back(nidx);
                    }
                }
            }
        }
    }

    edges.create(std::vector<int>{rows, cols}, CV_8UC1);
    const std::size_t edge_step = edges.step(0);
    for (int y = 0; y < rows; ++y)
    {
        std::memcpy(edges.data + static_cast<std::size_t>(y) * edge_step,
                    edge_map.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(cols),
                    static_cast<std::size_t>(cols));
    }

    return true;
}


} // namespace

void canny_deriv_backend_impl(const Mat& dx,
                              const Mat& dy,
                              Mat& edges,
                              double threshold1,
                              double threshold2,
                              bool L2gradient)
{
    if (try_canny_from_derivatives_fastpath_s16(dx, dy, edges, threshold1, threshold2, L2gradient))
    {
        return;
    }

    canny_from_derivatives_fallback(dx, dy, edges, threshold1, threshold2, L2gradient);
}

} // namespace detail
} // namespace cvh
