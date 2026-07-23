#ifndef CVH_BENCHMARK_OPENCV_COMPARE_BACKEND_H
#define CVH_BENCHMARK_OPENCV_COMPARE_BACKEND_H

#include <cstdint>

namespace cvh_bench_compare {

enum class DepthId
{
    U8 = 0,
    F32 = 1,
};

enum class MatOpId
{
    CreateReuse = 0,
    Clone,
    CopyTo,
    SetTo,
    ConvertTo,
    Reshape,
};

enum class CoreBinaryOpId
{
    Add = 0,
    Subtract,
    Multiply,
    Divide,
};

enum class ImgprocRoiOpId
{
    ResizeLinear = 0,
    CvtColorBgr2Gray,
    ThresholdF32,
    Box,
    Gaussian,
    Filter2D,
    SepFilter2D,
};

enum class ImgprocColorOpId
{
    Bgr2Rgb = 0,
    Bgr2Bgra,
    Bgra2Gray,
    Bgr2Gray,
    Bgr2Yuv,
    Yuv2Bgr,
    Bgr2YuvI420,
    YuvI420ToBgr,
    Bgr2YuvYuy2,
    YuvYuy2ToBgr,
    YuvNv12ToBgr,
};

void configure_opencv_threads(int threads);

double bench_opencv_mat_op(MatOpId op,
                           int rows,
                           int cols,
                           DepthId depth,
                           int channels,
                           int warmup,
                           int iters,
                           int repeats,
                           std::uint32_t seed);

double bench_opencv_binary(CoreBinaryOpId op,
                           int rows,
                           int cols,
                           DepthId depth,
                           int channels,
                           int warmup,
                           int iters,
                           int repeats,
                           std::uint32_t seed_a,
                           std::uint32_t seed_b);

double bench_opencv_transpose(int rows,
                              int cols,
                              DepthId depth,
                              int channels,
                              int warmup,
                              int iters,
                              int repeats,
                              std::uint32_t seed);

bool validate_opencv_binary(CoreBinaryOpId op,
                            int rows,
                            int cols,
                            DepthId depth,
                            int channels,
                            std::uint32_t seed_a,
                            std::uint32_t seed_b,
                            const void* cvh_data,
                            std::uint64_t cvh_bytes);

bool validate_opencv_transpose(int rows,
                               int cols,
                               DepthId depth,
                               int channels,
                               std::uint32_t seed,
                               const void* cvh_data,
                               std::uint64_t cvh_bytes);

bool validate_opencv_gemm(int m,
                          int k,
                          int n,
                          std::uint32_t seed_a,
                          std::uint32_t seed_b,
                          const void* cvh_data,
                          std::uint64_t cvh_bytes);

double bench_opencv_gemm(int m,
                         int k,
                         int n,
                         int warmup,
                         int iters,
                         int repeats,
                         std::uint32_t seed_a,
                         std::uint32_t seed_b);

double bench_opencv_gemm_prepack(int m,
                                 int k,
                                 int n,
                                 int warmup,
                                 int iters,
                                 int repeats,
                                 std::uint32_t seed_a,
                                 std::uint32_t seed_b);

double bench_opencv_resize_linear_half(int dst_rows,
                                        int dst_cols,
                                        int warmup,
                                        int iters,
                                        int repeats,
                                        std::uint32_t seed);

double bench_opencv_resize(int src_rows,
                           int src_cols,
                           int dst_rows,
                           int dst_cols,
                           DepthId depth,
                           int channels,
                           int interpolation,
                           int warmup,
                           int iters,
                           int repeats,
                           std::uint32_t seed);

double bench_opencv_cvtcolor(ImgprocColorOpId op,
                             int rows,
                             int cols,
                             DepthId depth,
                             int warmup,
                             int iters,
                             int repeats,
                             std::uint32_t seed);

double bench_opencv_cvtcolor_bgr2gray(int rows,
                                      int cols,
                                      int warmup,
                                      int iters,
                                      int repeats,
                                      std::uint32_t seed);

double bench_opencv_threshold_binary(int rows,
                                      int cols,
                                      int warmup,
                                      int iters,
                                      int repeats,
                                      std::uint32_t seed);

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

double bench_opencv_lut(int rows,
                        int cols,
                        int channels,
                        int warmup,
                        int iters,
                        int repeats,
                        std::uint32_t seed);

double bench_opencv_copy_make_border(int rows,
                                     int cols,
                                     DepthId depth,
                                     int channels,
                                     int top,
                                     int bottom,
                                     int left,
                                     int right,
                                     int warmup,
                                     int iters,
                                     int repeats,
                                     std::uint32_t seed);

double bench_opencv_filter2d(int rows,
                             int cols,
                             DepthId depth,
                             int channels,
                             int warmup,
                             int iters,
                             int repeats,
                             std::uint32_t seed);

double bench_opencv_sep_filter2d(int rows,
                                 int cols,
                                 DepthId depth,
                                 int channels,
                                 int warmup,
                                 int iters,
                                 int repeats,
                                 std::uint32_t seed);

double bench_opencv_warp_affine(int rows,
                                int cols,
                                DepthId depth,
                                int channels,
                                int warmup,
                                int iters,
                                int repeats,
                                std::uint32_t seed);

double bench_opencv_sobel(int rows, int cols, int channels, int warmup, int iters, int repeats, std::uint32_t seed);
double bench_opencv_canny(int rows,
                          int cols,
                          int warmup,
                          int iters,
                          int repeats,
                          std::uint32_t seed,
                          double threshold1,
                          double threshold2,
                          int aperture_size,
                          bool l2gradient);
double bench_opencv_erode(int rows, int cols, int channels, int warmup, int iters, int repeats, std::uint32_t seed);
double bench_opencv_dilate(int rows, int cols, int channels, int warmup, int iters, int repeats, std::uint32_t seed);

double bench_opencv_imgproc_roi(ImgprocRoiOpId op,
                                int rows,
                                int cols,
                                int warmup,
                                int iters,
                                int repeats,
                                std::uint32_t seed);

}  // namespace cvh_bench_compare

#endif  // CVH_BENCHMARK_OPENCV_COMPARE_BACKEND_H
