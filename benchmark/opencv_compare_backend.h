#ifndef CVH_BENCHMARK_OPENCV_COMPARE_BACKEND_H
#define CVH_BENCHMARK_OPENCV_COMPARE_BACKEND_H

#include <cstdint>

namespace cvh_bench_compare {

enum class DepthId
{
    U8 = 0,
    F32 = 1,
};

double bench_opencv_add(int rows,
                        int cols,
                        DepthId depth,
                        int channels,
                        int warmup,
                        int iters,
                        int repeats,
                        std::uint32_t seed_a,
                        std::uint32_t seed_b);

double bench_opencv_sub(int rows,
                        int cols,
                        DepthId depth,
                        int channels,
                        int warmup,
                        int iters,
                        int repeats,
                        std::uint32_t seed_a,
                        std::uint32_t seed_b);

double bench_opencv_gemm(int m,
                         int k,
                         int n,
                         int warmup,
                         int iters,
                         int repeats,
                         std::uint32_t seed_a,
                         std::uint32_t seed_b);

double bench_opencv_gaussian(int rows,
                             int cols,
                             DepthId depth,
                             int channels,
                             int ksize,
                             int warmup,
                             int iters,
                             int repeats,
                             std::uint32_t seed);

double bench_opencv_box(int rows,
                        int cols,
                        DepthId depth,
                        int channels,
                        int ksize,
                        int warmup,
                        int iters,
                        int repeats,
                        std::uint32_t seed);

double bench_opencv_sobel(int rows, int cols, int channels, int warmup, int iters, int repeats, std::uint32_t seed);
double bench_opencv_erode(int rows, int cols, int channels, int warmup, int iters, int repeats, std::uint32_t seed);
double bench_opencv_dilate(int rows, int cols, int channels, int warmup, int iters, int repeats, std::uint32_t seed);

}  // namespace cvh_bench_compare

#endif  // CVH_BENCHMARK_OPENCV_COMPARE_BACKEND_H
