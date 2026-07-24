#ifndef CVH_TEST_OPENCV_CONTRACT_BACKEND_H
#define CVH_TEST_OPENCV_CONTRACT_BACKEND_H

#include <cstddef>
#include <cstdint>

namespace cvh_test_opencv_contract {

enum class CoreArrayOpId
{
    AbsDiff,
    BitwiseAnd,
    BitwiseNot,
    BitwiseOr,
    BitwiseXor,
    InRange,
    Min,
    Max,
};

enum class CoreDepthId
{
    U8,
    S8,
    U16,
    S16,
    S32,
    F32,
    F64,
};

enum class CoreMathOpId
{
    Sqrt,
    Pow,
    Exp,
    Log,
};

enum class CoreLayoutOpId
{
    CopyMask,
    ExtractLastChannel,
    FlipHorizontal,
    FlipVertical,
    FlipBoth,
    RotateClockwise,
    Rotate180,
    RotateCounterclockwise,
    Repeat2x3,
    HConcat,
    VConcat,
};

enum class ImgprocIntensityOpId
{
    MedianU8,
    BilateralU8,
    StackBlurU8,
    AdaptiveMeanU8,
    AdaptiveGaussianU8,
    ThresholdMaskU8,
    EqualizeHistU8,
    ColorMapJetU8,
    ColorMapUserU8,
};

enum class ImgprocPyramidColorOpId
{
    AccumulateU8,
    AccumulateSquareU8,
    AccumulateProductU8,
    AccumulateWeightedU8,
    BlendLinearU8,
    PyrDownU8,
    PyrUpU8,
    TwoPlaneNv12U8,
    TwoPlaneNv21U8,
    DemosaicBgU8,
    DemosaicGbU8,
    DemosaicRgU8,
    DemosaicGrU8,
};

enum class ImgprocGeometrySamplingOpId
{
    RemapFloatU8,
    RemapFixedU8,
    WarpPerspectiveU8,
    RectSubPixU8,
    RectSubPixU8F32,
};

struct CoreReductionSummary
{
    double sums[4];
    double means[4];
    double stddevs[4];
    double norm_inf;
    double norm_l1;
    double norm_l2;
    double min_value;
    double max_value;
    int count_non_zero;
    int min_x;
    int min_y;
    int max_x;
    int max_y;
};

bool validate_core_convert_u8_to_f64(int rows,
                                     int cols,
                                     int channels,
                                     std::uint32_t seed,
                                     const void* actual_data,
                                     std::size_t actual_bytes);

bool validate_core_array_op(CoreArrayOpId op,
                            CoreDepthId depth,
                            int rows,
                            int cols,
                            int channels,
                            std::uint32_t seed_a,
                            std::uint32_t seed_b,
                            const void* actual_data,
                            std::size_t actual_bytes);

bool validate_core_float_edge_op(CoreArrayOpId op,
                                 const void* actual_data,
                                 std::size_t actual_bytes);

bool validate_core_double_edge_op(CoreArrayOpId op,
                                  const void* actual_data,
                                  std::size_t actual_bytes);

bool validate_convert_scale_abs_edges(const void* actual_data,
                                      std::size_t actual_bytes);

bool validate_convert_fp16_edges(const void* actual_data,
                                 std::size_t actual_bytes);

bool validate_core_math_op(CoreMathOpId op,
                           CoreDepthId depth,
                           int rows,
                           int cols,
                           int channels,
                           std::uint32_t seed,
                           const void* actual_data,
                           std::size_t actual_bytes);

bool validate_core_reduction_summary(CoreDepthId depth,
                                     int rows,
                                     int cols,
                                     int channels,
                                     std::uint32_t seed,
                                     bool use_mask,
                                     const CoreReductionSummary& actual);

bool validate_core_nonzero_locations(CoreDepthId depth,
                                     int rows,
                                     int cols,
                                     std::uint32_t seed,
                                     const int* actual_xy,
                                     std::size_t actual_count);

bool validate_core_reduce_f64(CoreDepthId depth,
                              int rows,
                              int cols,
                              int channels,
                              std::uint32_t seed,
                              int dim,
                              int reduce_type,
                              const void* actual_data,
                              std::size_t actual_bytes);

bool validate_core_reduce_arg(CoreDepthId depth,
                              int rows,
                              int cols,
                              std::uint32_t seed,
                              int axis,
                              bool find_max,
                              bool last_index,
                              const void* actual_data,
                              std::size_t actual_bytes);

bool validate_core_normalize_l2_f64(CoreDepthId depth,
                                    int rows,
                                    int cols,
                                    int channels,
                                    std::uint32_t seed,
                                    bool use_mask,
                                    const void* actual_data,
                                    std::size_t actual_bytes);

bool validate_core_border_interpolate(int p,
                                      int len,
                                      int border_type,
                                      int actual);

bool validate_core_layout_op(CoreLayoutOpId op,
                             CoreDepthId depth,
                             int rows,
                             int cols,
                             int channels,
                             std::uint32_t seed,
                             const void* actual_data,
                             std::size_t actual_bytes);

bool validate_core_mix_channels(CoreDepthId depth,
                                int rows,
                                int cols,
                                std::uint32_t seed,
                                const void* actual_bgr,
                                std::size_t actual_bgr_bytes,
                                const void* actual_alpha,
                                std::size_t actual_alpha_bytes);

bool validate_core_flip_nd(CoreDepthId depth,
                           std::uint32_t seed,
                           int axis,
                           const void* actual_data,
                           std::size_t actual_bytes);

bool validate_core_broadcast(CoreDepthId depth,
                             std::uint32_t seed,
                             const void* actual_data,
                             std::size_t actual_bytes);

bool validate_imgproc_structuring_element(int shape,
                                          int width,
                                          int height,
                                          int anchor_x,
                                          int anchor_y,
                                          const void* actual_data,
                                          std::size_t actual_bytes);

bool validate_imgproc_gaussian_kernel(int ksize,
                                      double sigma,
                                      CoreDepthId depth,
                                      const void* actual_data,
                                      std::size_t actual_bytes);

bool validate_imgproc_deriv_kernels(int dx,
                                    int dy,
                                    int ksize,
                                    bool normalize,
                                    CoreDepthId depth,
                                    const void* actual_kx,
                                    std::size_t actual_kx_bytes,
                                    const void* actual_ky,
                                    std::size_t actual_ky_bytes);

bool validate_imgproc_gabor_kernel(int width,
                                   int height,
                                   CoreDepthId depth,
                                   const void* actual_data,
                                   std::size_t actual_bytes);

bool validate_imgproc_hanning_window(int width,
                                     int height,
                                     CoreDepthId depth,
                                     const void* actual_data,
                                     std::size_t actual_bytes);

bool validate_imgproc_integral_u8(int rows,
                                  int cols,
                                  int channels,
                                  std::uint32_t seed,
                                  CoreDepthId output_depth,
                                  const void* actual_data,
                                  std::size_t actual_bytes);

bool validate_imgproc_derivative_filters_u8(int rows,
                                            int cols,
                                            int channels,
                                            std::uint32_t seed,
                                            const void* actual_scharr,
                                            std::size_t actual_scharr_bytes,
                                            const void* actual_laplacian,
                                            std::size_t actual_laplacian_bytes);

bool validate_imgproc_spatial_gradient_u8(int rows,
                                          int cols,
                                          std::uint32_t seed,
                                          const void* actual_dx,
                                          std::size_t actual_dx_bytes,
                                          const void* actual_dy,
                                          std::size_t actual_dy_bytes);

bool validate_imgproc_sqr_box_filter_u8(int rows,
                                        int cols,
                                        int channels,
                                        std::uint32_t seed,
                                        const void* actual_data,
                                        std::size_t actual_bytes);

bool validate_imgproc_intensity_u8(ImgprocIntensityOpId op,
                                   int rows,
                                   int cols,
                                   std::uint32_t seed,
                                   const void* actual_data,
                                   std::size_t actual_bytes);

bool validate_imgproc_pyramid_color_u8(ImgprocPyramidColorOpId op,
                                       int rows,
                                       int cols,
                                       std::uint32_t seed,
                                       const void* actual_data,
                                       std::size_t actual_bytes);

bool validate_imgproc_geometry_matrices(const double* actual_rotation,
                                        const double* actual_affine,
                                        const double* actual_perspective,
                                        const double* actual_inverse);

bool validate_imgproc_geometry_sampling(
    ImgprocGeometrySamplingOpId op,
    std::uint32_t seed,
    const void* actual_data,
    std::size_t actual_bytes);

bool validate_imgproc_resize_linear_u8(int src_rows,
                                       int src_cols,
                                       int dst_rows,
                                       int dst_cols,
                                       int channels,
                                       std::uint32_t seed,
                                       const void* actual_data,
                                       std::size_t actual_bytes);

}  // namespace cvh_test_opencv_contract

#endif  // CVH_TEST_OPENCV_CONTRACT_BACKEND_H
