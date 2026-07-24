#ifndef CVH_BENCHMARK_OPENCV_COMPARE_PHASE1_BENCHMARK_H
#define CVH_BENCHMARK_OPENCV_COMPARE_PHASE1_BENCHMARK_H

#include <cstdint>
#include <string>
#include <vector>

namespace cvh_bench_compare {

enum class Phase1OpId
{
    Absdiff = 0,
    BitwiseAnd,
    BitwiseNot,
    BitwiseOr,
    BitwiseXor,
    InRange,
    Min,
    Max,
    ScaleAdd,
    ConvertScaleAbs,
    ConvertFp16,
    Sqrt,
    Pow,
    Exp,
    Log,
    CheckRange,
    PatchNaNs,
    Norm,
    Sum,
    Mean,
    MeanStdDev,
    CountNonZero,
    HasNonZero,
    FindNonZero,
    MinMaxIdx,
    MinMaxLoc,
    Reduce,
    ReduceArgMax,
    ReduceArgMin,
    Normalize,
    BorderInterpolate,
    CopyTo,
    ExtractChannel,
    InsertChannel,
    MixChannels,
    Flip,
    FlipND,
    Rotate,
    Repeat,
    Hconcat,
    Vconcat,
    Broadcast,
    Swap,
    GetStructuringElement,
    GetGaussianKernel,
    GetDerivKernels,
    GetGaborKernel,
    CreateHanningWindow,
    Integral,
    Scharr,
    Laplacian,
    SpatialGradient,
    SqrBoxFilter,
    MedianBlur,
    BilateralFilter,
    StackBlur,
    AdaptiveThreshold,
    ThresholdWithMask,
    EqualizeHist,
    ApplyColorMap,
    Accumulate,
    AccumulateProduct,
    AccumulateSquare,
    AccumulateWeighted,
    BlendLinear,
    PyrDown,
    PyrUp,
    BuildPyramid,
    CvtColorTwoPlane,
    Demosaicing,
    Remap,
    ConvertMaps,
    WarpPerspective,
    GetAffineTransform,
    GetPerspectiveTransform,
    GetRotationMatrix2D,
    GetRotationMatrix2DUnderscore,
    InvertAffineTransform,
    GetRectSubPix,
};

struct Phase1BenchmarkConfig
{
    std::string profile;
    int warmup = 1;
    int iters = 1;
    int repeats = 1;
};

struct Phase1BenchmarkResult
{
    std::string suite;
    std::string op;
    std::string variant;
    std::string dispatch_path;
    std::string depth;
    int channels = 1;
    std::string layout = "continuous";
    std::string shape;
    double cvh_ms = 0.0;
    double opencv_ms = 0.0;
    std::string note;
};

double bench_opencv_phase1(Phase1OpId op,
                           int rows,
                           int cols,
                           int warmup,
                           int iters,
                           int repeats,
                           std::uint32_t seed);

std::vector<Phase1BenchmarkResult> run_phase1_benchmarks(
    const Phase1BenchmarkConfig& config);

}  // namespace cvh_bench_compare

#endif  // CVH_BENCHMARK_OPENCV_COMPARE_PHASE1_BENCHMARK_H
