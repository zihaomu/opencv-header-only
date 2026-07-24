# OpenCV Core / Imgproc API Coverage

Last updated: 2026-07-24

## 1. Purpose

This document tracks the CPU C++ API gap between OpenCV upstream and
`opencv-header-only`. It answers three separate questions:

1. How many public operation families are exposed by upstream `core` and
   `imgproc`?
2. Which operation families have a callable `cvh` counterpart today?
3. Where does the current `cvh` implementation cover only a subset of the
   upstream types, overloads, or parameter space?

`cvh::headers` and `cvh::headers_fast` expose the same API surface.
`cvh::headers_fast` only changes validated dispatch paths, so it must not be
counted as additional API coverage.

The three-phase operator support scope is tracked in
[opencv-core-imgproc-three-phase-support-plan.md](opencv-core-imgproc-three-phase-support-plan.md).

## 2. Upstream Snapshot And Counting Rules

The inventory is based on the local OpenCV checkout:

- Repository: `/Users/zmu/work/my_project/ocvh/opencv`
- Commit: `d48bf69f65`
- Version header: `4.14.0-pre`
- Core source:
  `modules/core/include/opencv2/core.hpp`
- Imgproc source:
  `modules/imgproc/include/opencv2/imgproc.hpp`

The upstream checkout is dirty outside these two primary headers. Neither
inventory source header has a local modification at the time of this audit.

The primary count uses these rules:

- Count public exported free-function names in `opencv2/core.hpp` and
  `opencv2/imgproc.hpp`.
- Collapse overloads into one operation family. For example, all `divide`
  overloads count as one API family.
- Collapse binding-only aliases such as `wrapperEMD` into the public operation
  family `EMD`.
- Keep distinct public names such as `getRotationMatrix2D` and
  `getRotationMatrix2D_` distinct.
- Exclude C APIs, macros, enums, constructors, class methods, HAL, Universal
  Intrinsics, CUDA, OpenCL, OpenGL, and other platform integration headers.
- Track major class and infrastructure gaps separately in section 5.

This gives two useful counts:

| Module | Exported declarations, overloads included | Unique operation families |
|---|---:|---:|
| `core` | 120 | 97 |
| `imgproc` | 152 | 123 |
| **Total** | **272** | **220** |

The number 220 is the coverage denominator used below. It is an
operator-family count, not a claim that OpenCV has only 220 total C++ symbols.
Including every `Mat`, `SparseMat`, persistence, utility, and algorithm class
method would produce a much larger and less stable number.

## 3. Status Definitions

| Status | Meaning |
|---|---|
| `Available (subset)` | A public, header-defined `cvh` counterpart is callable, but it does not cover the full upstream overload, type, or parameter matrix. |
| `Declared only` | A public declaration exists in `cvh`, but no accepted inline definition exists. It is not part of the usable header-only contract. |
| `Missing` | No public `cvh` counterpart exists. |

Current summary:

| Module | Upstream families | Available (subset) | Declared only | Missing | Callable family coverage |
|---|---:|---:|---:|---:|---:|
| `core` | 97 | 57 | 0 | 40 | 58.8% |
| `imgproc` | 123 | 50 | 0 | 73 | 40.7% |
| **Total** | **220** | **107** | **0** | **113** | **48.6%** |

The percentages measure name-level callable coverage only. They do not measure
type coverage, numerical compatibility, performance, or overload parity.

## 4. Core Operation Families

### 4.1 Available Core Families

The module column follows the upstream classification. `LUT` and
`copyMakeBorder` live under `include/cvh/imgproc/` in this project but are
counted as `core` because OpenCV declares them in `opencv2/core.hpp`.

| Upstream API | Status | Current `cvh` scope |
|---|---|---|
| `LUT` | Available (subset) | `CV_8U` source; 256-entry LUT; one LUT channel or the same channel count as the source. |
| `absdiff` | Available (subset) | `Mat/Mat`, `Mat/Scalar`, and `Scalar/Mat`; current cvh depths, continuous and ROI paths. |
| `add` | Available (subset) | `Mat/Mat`, `Mat/Scalar`, and `Scalar/Mat`; no upstream mask or output-depth arguments. |
| `addWeighted` | Available (subset) | `a * alpha + b * beta`; the upstream `gamma` and `dtype` arguments are absent. |
| `borderInterpolate` | Available (subset) | Constant, replicate, reflect, reflect-101 and wrap modes, including isolated-bit normalization. |
| `bitwise_and` | Available (subset) | `Mat/Mat`, `Mat/Scalar`, and `Scalar/Mat`; optional `CV_8UC1` mask and raw floating-point bit patterns. |
| `bitwise_not` | Available (subset) | Unary Mat input with optional `CV_8UC1` mask; raw floating-point bit patterns. |
| `bitwise_or` | Available (subset) | `Mat/Mat`, `Mat/Scalar`, and `Scalar/Mat`; optional `CV_8UC1` mask and raw floating-point bit patterns. |
| `bitwise_xor` | Available (subset) | `Mat/Mat`, `Mat/Scalar`, and `Scalar/Mat`; optional `CV_8UC1` mask and raw floating-point bit patterns. |
| `checkRange` | Available (subset) | Current cvh depths, multi-channel first-bad-pixel reporting for 2D input, quiet return and throwing modes. |
| `compare` | Available (subset) | `Mat/Mat`, `Mat/Scalar`, and `Scalar/Mat`; produces the comparison mask for the supported cvh depths. |
| `convertFp16` | Available (subset) | FP32 to legacy `CV_16S` FP16 bits and `CV_16S`/native `CV_16F` back to FP32. |
| `convertScaleAbs` | Available (subset) | Current cvh depths to U8 with OpenCV-aligned scale, delta, absolute value, even rounding, and saturation. |
| `copyMakeBorder` | Available (subset) | `CV_8U` and `CV_32F`; constant, replicate, reflect, reflect-101, and wrap borders. |
| `copyTo` | Available (subset) | Free-function Mat copy with optional `CV_8UC1` mask, ROI and overlap-safe source snapshots. |
| `countNonZero` | Available (subset) | Single-channel current cvh depths; empty input returns zero. |
| `divide` | Available (subset) | Element-wise `Mat/Mat`, `Mat/Scalar`, and `Scalar/Mat`; no upstream `scale` or `dtype` arguments. |
| `error` | Available (subset) | Header-defined `cvh::Exception` throwing helpers; upstream callback/redirection behavior is not reproduced. |
| `exp` | Available (subset) | Per-element `CV_32F`/`CV_64F` exponential with continuous, ROI, and in-place paths. |
| `extractChannel` | Available (subset) | Raw-byte channel extraction for current Mat depths and channel counts. |
| `findNonZero` | Available (subset) | Single-channel 2D input with row-major `std::vector<Point>` or `CV_32SC2` Mat output. |
| `flip` | Available (subset) | 2D horizontal, vertical and both-axis byte-preserving flips. |
| `flipND` | Available (subset) | N-D byte-preserving flip with positive or negative axis selection. |
| `gemm` | Available (subset) | Return-by-value cvh API; FP32 activation with FP32/FP16 weights, batched/broadcast paths, and cvh packed-B extensions. It is not signature-compatible with `cv::gemm`. |
| `hasNonZero` | Available (subset) | Single-channel current cvh depths with deterministic empty-input behavior. |
| `hconcat` | Available (subset) | Pointer/count, pair and vector overloads for compatible 2D Mat inputs. |
| `inRange` | Available (subset) | Mat or Scalar inclusive bounds; all source channels are combined into a `CV_8UC1` result. |
| `insertChannel` | Available (subset) | Inserts a single-channel Mat into a preallocated compatible destination channel. |
| `log` | Available (subset) | Per-element `CV_32F`/`CV_64F` natural logarithm with continuous, ROI, and in-place paths. |
| `max` | Available (subset) | `Mat/Mat` and `Mat/Scalar`; accepted cvh depths with OpenCV-aligned floating NaN/zero behavior. |
| `mean` | Available (subset) | C1 through C4, optional `CV_8UC1` mask, current cvh depths and ROI input. |
| `meanStdDev` | Available (subset) | C1 through C4 with stable population variance accumulation and optional mask. |
| `merge` | Available (subset) | Pointer/count and `std::vector<Mat>` inputs; no `InputArrayOfArrays` abstraction. |
| `min` | Available (subset) | `Mat/Mat` and `Mat/Scalar`; accepted cvh depths with OpenCV-aligned floating NaN/zero behavior. |
| `minMaxIdx` | Available (subset) | N-D extrema values and coordinates; locations and masks require single-channel input. |
| `minMaxLoc` | Available (subset) | 2D extrema and first row-major locations with optional single-channel mask. |
| `mixChannels` | Available (subset) | Pointer/count and vector channel routing, negative-source zero fill and alias-safe source snapshots. |
| `multiply` | Available (subset) | Element-wise `Mat/Mat`, `Mat/Scalar`, and `Scalar/Mat`; no upstream `scale` or `dtype` arguments. |
| `norm` | Available (subset) | One/two-input `NORM_INF/L1/L2`, optional mask, current cvh depths and channels. |
| `normalize` | Available (subset) | `NORM_INF/L1/L2/MINMAX`, optional mask, dtype conversion and supported in-place paths. |
| `patchNaNs` | Available (subset) | In-place `CV_32F` replacement, matching the pinned OpenCV CPU implementation; F64 is explicitly rejected. |
| `pow` | Available (subset) | Per-element `CV_32F`/`CV_64F` power; pinned CPU behavior returns NaN for negative bases with non-integer powers. |
| `reduce` | Available (subset) | 2D axis 0/1 with SUM/AVG/MAX/MIN/SUM2 and explicit output depth. |
| `reduceArgMax` | Available (subset) | Single-channel 2D axis 0/1 with first- or last-index tie handling. |
| `reduceArgMin` | Available (subset) | Single-channel 2D axis 0/1 with first- or last-index tie handling. |
| `repeat` | Available (subset) | Positive 2D vertical and horizontal repetition factors. |
| `rotate` | Available (subset) | 2D 90-degree clockwise, 180-degree and 90-degree counterclockwise rotation. |
| `scaleAdd` | Available (subset) | Same-shape/type Mat inputs across current cvh depths; no InputArray abstraction. |
| `split` | Available (subset) | Pointer and `std::vector<Mat>` outputs; no `OutputArrayOfArrays` abstraction. |
| `sqrt` | Available (subset) | Per-element `CV_32F`/`CV_64F` square root with continuous, ROI, and in-place paths. |
| `subtract` | Available (subset) | Unary negate plus `Mat/Mat`, `Mat/Scalar`, and scalar forms; no upstream mask or output-depth arguments. |
| `sum` | Available (subset) | C1 through C4 over current cvh depths, including non-contiguous ROI input. |
| `swap` | Available (subset) | Swaps Mat headers and ownership without copying pixel storage. |
| `transpose` | Available (subset) | Return-by-value blocked transpose; the public signature differs from OpenCV's output-argument API. |
| `transposeND` | Available (subset) | Return-by-value N-D axis permutation; the public signature differs from OpenCV's output-argument API. |
| `vconcat` | Available (subset) | Pointer/count, pair and vector overloads for compatible 2D Mat inputs. |
| `broadcast` | Available (subset) | Trailing-dimension N-D broadcast from vector or `CV_32SC1` shape input; cvh also preserves multi-channel elements. |

Public implementation sources:

- [`core/basic_op.h`](../include/cvh/core/basic_op.h)
- [`core/array.h`](../include/cvh/core/array.h)
- [`core/gemm.h`](../include/cvh/core/gemm.h)
- [`core/math.h`](../include/cvh/core/math.h)
- [`core/reduce.h`](../include/cvh/core/reduce.h)
- [`core/system.h`](../include/cvh/core/system.h)
- [`imgproc/lut.h`](../include/cvh/imgproc/lut.h)
- [`imgproc/copy_make_border.h`](../include/cvh/imgproc/copy_make_border.h)

### 4.2 Declared-Only Core Families

None.

### 4.3 Missing Core Families

- [ ] `Mahalanobis`
- [ ] `PCABackProject`
- [ ] `PCACompute`
- [ ] `PCAProject`
- [ ] `PSNR`
- [ ] `SVBackSubst`
- [ ] `SVDecomp`
- [ ] `batchDistance`
- [ ] `calcCovarMatrix`
- [ ] `cartToPolar`
- [ ] `completeSymm`
- [ ] `dct`
- [ ] `determinant`
- [ ] `dft`
- [ ] `eigen`
- [ ] `eigenNonSymmetric`
- [ ] `getOptimalDFTSize`
- [ ] `idct`
- [ ] `idft`
- [ ] `invert`
- [ ] `kmeans`
- [ ] `magnitude`
- [ ] `mulSpectrums`
- [ ] `mulTransposed`
- [ ] `perspectiveTransform`
- [ ] `phase`
- [ ] `polarToCart`
- [ ] `randShuffle`
- [ ] `randn`
- [ ] `randu`
- [ ] `setIdentity`
- [ ] `setRNGSeed`
- [ ] `solve`
- [ ] `solveCubic`
- [ ] `solvePoly`
- [ ] `sort`
- [ ] `sortIdx`
- [ ] `theRNG`
- [ ] `trace`
- [ ] `transform`

## 5. Core Class And Infrastructure Coverage

These items are intentionally outside the 97-family free-function count, but
they matter for source compatibility.

| Upstream area | Current `cvh` status | Notes |
|---|---|---|
| `Mat` | Available (subset) | Header-only ownership, external-data views, N-D shape/step handling, ROI, clone/copy, conversion, reshape, and typed access are present. Many OpenCV constructors, iterators, masks, and expression conveniences are absent. |
| `MatExpr` and arithmetic operators | Available (subset) | Basic arithmetic and comparison expressions exist. Bitwise, min/max, abs, and the broader OpenCV expression system are incomplete. |
| `Scalar`, `Range`, `Point`, `Size` | Available (subset) | Simplified types for the current operator surface; not full template/type-family parity. |
| Type/channel macros and `saturate_cast` | Available (subset) | cvh defines its accepted depths and channel encoding. Exact OpenCV ABI compatibility is not a goal. |
| `parallel_for_`, thread controls | Available (subset) | Header-only serial, standard-thread, and optional OpenMP runtime; backend semantics differ from OpenCV. |
| `Exception` and assertions | Available (subset) | Core throw/assert behavior needed by cvh is present. |
| `InputArray`, `OutputArray`, `InputOutputArray` | Missing | cvh public operators use concrete `Mat` references and values. |
| `UMat`, OpenCL objects | Missing by scope | GPU/OpenCL execution is outside the CPU header-only target. |
| `cuda::*`, OpenGL, DirectX, VA interop | Missing by scope | Hardware/runtime integration is outside this project scope. |
| `SparseMat` | Missing | No sparse matrix data model. |
| `FileStorage`, `FileNode`, persistence | Missing | No YAML/XML/JSON persistence layer. |
| `Algorithm`, `AsyncArray` | Missing | No OpenCV algorithm object model or async result abstraction. |
| `RNG`, `RNG_MT19937` | Missing | Random free functions and random engine classes are absent. |
| `PCA`, `SVD`, `LDA` and solver classes | Missing | Dense linear algebra beyond the current GEMM subset is absent. |
| `Affine`, `Quaternion`, `DualQuaternion` | Missing | Geometry helper classes are absent. |

The cvh-only APIs `gemm_pack_b`, `softmax`, `silu`, `rmsnorm`, and `rope` do
not increase upstream coverage. `gemm_pack_b` is implemented as a cvh
extension; the neural-network helpers remain outside the accepted header-only
contract unless they gain inline implementations and tests.

## 6. Imgproc Operation Families

### 6.1 Available Imgproc Families

Every currently available imgproc family is a subset of the upstream type and
parameter matrix.

| Upstream API | Status | Current `cvh` scope |
|---|---|---|
| `Canny` | Available (subset) | Image overload for `CV_8UC1`; derivative overload for `CV_16SC1`; aperture 3/5 and L1/L2 gradient. |
| `GaussianBlur` | Available (subset) | `CV_8U`/`CV_32F`, C1/C3/C4, odd kernels, and sigma-based separable processing. |
| `Laplacian` | Available (subset) | `CV_8U`/`CV_16S`/`CV_32F` input, `CV_16S`/`CV_32F` output and kernel sizes 1/3/5. |
| `Scharr` | Available (subset) | First x/y derivative over the existing Sobel input/output matrix and common borders. |
| `Sobel` | Available (subset) | `CV_8U`/`CV_16S`/`CV_32F` input, `CV_16S`/`CV_32F` output, first derivatives, kernel size 3/5. |
| `accumulate` | Available (subset) | U8/F32 C1/C3/C4 source into preinitialized F32 destination, with optional U8C1 mask. |
| `accumulateProduct` | Available (subset) | Matching U8/F32 C1/C3/C4 sources into F32 destination, with optional mask. |
| `accumulateSquare` | Available (subset) | U8/F32 C1/C3/C4 source squared into F32 destination, with optional mask. |
| `accumulateWeighted` | Available (subset) | U8/F32 C1/C3/C4 running average into F32 destination, with optional mask. |
| `adaptiveThreshold` | Available (subset) | `CV_8UC1`, mean/Gaussian methods, binary/binary-inverse, odd block sizes. |
| `applyColorMap` | Available (subset) | `CV_8UC1` input; AUTUMN/JET/WINTER/COOL/HOT and 256-entry U8 C1/C3 user LUT. |
| `bilateralFilter` | Available (subset) | `CV_8U`/`CV_32F`, C1/C3, circular neighborhood and selected borders; no in-place operation. |
| `blendLinear` | Available (subset) | Matching U8/F32 C1/C3/C4 images and F32C1 weight maps using upstream epsilon normalization. |
| `blur` | Available (subset) | `CV_8U`/`CV_32F`, C1/C3/C4; implemented as the normalized `boxFilter` wrapper. |
| `boxFilter` | Available (subset) | `CV_8U`/`CV_32F`, C1/C3/C4, common border modes; selected 3x3 and separable fast paths. |
| `cvtColor` | Available (subset) | Common gray/BGR/RGB/BGRA/RGBA conversions for `CV_8U`/`CV_32F`, plus the documented `CV_8U` YUV420/YUV422/YUV444 families. |
| `createHanningWindow` | Available (subset) | F32/F64 two-dimensional Hanning window with pinned upstream square-root product semantics. |
| `convertMaps` | Available (subset) | F32 pair/F32C2/fixed S16C2+U16 maps, including 5-bit linear fractions and nearest maps without map2. |
| `cvtColorTwoPlane` | Available (subset) | Separate U8C1 Y and half-size U8C2 UV planes for NV12/NV21 to BGR/RGB. |
| `demosaicing` | Available (subset) | U8 Bayer BG/GB/RG/GR to three-channel BGR/RGB aliases using bilinear interpolation. |
| `dilate` | Available (subset) | `CV_8U`, C1/C3/C4, custom kernel, iterations, and basic border handling. |
| `erode` | Available (subset) | `CV_8U`, C1/C3/C4, custom kernel, iterations, and basic border handling. |
| `equalizeHist` | Available (subset) | `CV_8UC1` 256-bin histogram equalization with constant-image and in-place handling. |
| `filter2D` | Available (subset) | `CV_8U`/`CV_32F` source, `CV_32FC1` kernel, selected destination depths and borders. |
| `getDerivKernels` | Available (subset) | F32/F64 Sobel sizes through 31 and first-order Scharr kernel generation. |
| `getGaborKernel` | Available (subset) | F32/F64 Gabor kernels with explicit or automatically derived dimensions. |
| `getGaussianKernel` | Available (subset) | Positive odd F32/F64 normalized kernels with OpenCV fixed coefficients for common automatic-sigma sizes. |
| `getStructuringElement` | Available (subset) | CV8U rectangle, cross, ellipse and diamond morphology kernels with anchor validation. |
| `getAffineTransform` | Available (subset) | Three Point2f/Point2d pairs to a 2x3 CV_64F matrix; degenerate point sets are rejected. |
| `getPerspectiveTransform` | Available (subset) | Four Point2f/Point2d pairs to a 3x3 CV_64F matrix using `DECOMP_LU`; degenerate point sets are rejected. |
| `getRectSubPix` | Available (subset) | U8/F32 C1/C3/C4 bilinear patch extraction with replicate borders and U8-to-F32 output conversion. |
| `getRotationMatrix2D` | Available (subset) | Point2f/Point2d center to an OpenCV-layout 2x3 CV_64F matrix. |
| `getRotationMatrix2D_` | Available (subset) | Point2f/Point2d center to the header-only `AffineMatrix2x3d` value type. |
| `integral` | Available (subset) | U8 C1/C3/C4 input to S32/F64 sum image with zero top/left border; sqsum/tilted overloads are absent. |
| `invertAffineTransform` | Available (subset) | 2x3 F32/F64 input with same-depth output, alias safety, and zero output for singular matrices. |
| `medianBlur` | Available (subset) | `CV_8U`/`CV_32F`, C1/C3/C4; F32 is limited to kernel sizes 3/5. |
| `morphologyEx` | Available (subset) | Erode, dilate, open, close, gradient, top-hat, black-hat, and hit-or-miss; hit-or-miss is limited to `CV_8UC1`. |
| `buildPyramid` | Available (subset) | U8/F32 C1/C3/C4 Gaussian pyramid built from the same pyrDown contract. |
| `pyrDown` | Available (subset) | U8/F32 C1/C3/C4 fixed 5x5 Gaussian downsampling with default/custom compatible sizes. |
| `pyrUp` | Available (subset) | U8/F32 C1/C3/C4 zero-insertion Gaussian upsampling with BORDER_DEFAULT. |
| `remap` | Available (subset) | U8/F32 C1/C3/C4 with float pair, float C2, or fixed maps; nearest/linear and four common borders. |
| `resize` | Available (subset) | `CV_8U`/`CV_32F`, C1/C3/C4, nearest, nearest-exact, and linear interpolation. |
| `sepFilter2D` | Available (subset) | `CV_8U`/`CV_32F` source with single-channel FP32 vector kernels and selected borders. |
| `spatialGradient` | Available (subset) | Kernel size 3 using paired Sobel outputs with reflect-101 or replicate borders. |
| `sqrBoxFilter` | Available (subset) | U8/F32 C1/C3/C4, U8/F32/F64 output subset, wide accumulation and common borders. |
| `stackBlur` | Available (subset) | `CV_8U`/`CV_32F`, C1/C3/C4, positive odd kernels and replicate-border triangular filtering. |
| `threshold` | Available (subset) | Fixed thresholds for `CV_8U`/`CV_32F`; Otsu/Triangle only for `CV_8UC1`. |
| `thresholdWithMask` | Available (subset) | Existing U8/F32 threshold subset with U8C1 mask and preserved unselected destination pixels. |
| `warpAffine` | Available (subset) | `CV_8U`/`CV_32F`, C1/C3/C4, nearest/linear, inverse-map flag, and selected borders. |
| `warpPerspective` | Available (subset) | U8/F32 C1/C3/C4, F32/F64 3x3 transform, nearest/linear, inverse-map flag, and four common borders. |

Public implementation source:

- [`imgproc/imgproc.h`](../include/cvh/imgproc/imgproc.h)
- [`imgproc/kernels.h`](../include/cvh/imgproc/kernels.h)
- [`imgproc/integral.h`](../include/cvh/imgproc/integral.h)
- [`imgproc/derivatives.h`](../include/cvh/imgproc/derivatives.h)
- [`imgproc/sqr_box_filter.h`](../include/cvh/imgproc/sqr_box_filter.h)
- [`imgproc/median_blur.h`](../include/cvh/imgproc/median_blur.h)
- [`imgproc/bilateral_filter.h`](../include/cvh/imgproc/bilateral_filter.h)
- [`imgproc/stack_blur.h`](../include/cvh/imgproc/stack_blur.h)
- [`imgproc/adaptive_threshold.h`](../include/cvh/imgproc/adaptive_threshold.h)
- [`imgproc/equalize_hist.h`](../include/cvh/imgproc/equalize_hist.h)
- [`imgproc/colormap.h`](../include/cvh/imgproc/colormap.h)
- [`imgproc/accumulate.h`](../include/cvh/imgproc/accumulate.h)
- [`imgproc/blend_linear.h`](../include/cvh/imgproc/blend_linear.h)
- [`imgproc/pyramid.h`](../include/cvh/imgproc/pyramid.h)
- [`imgproc/cvtcolor_two_plane.h`](../include/cvh/imgproc/cvtcolor_two_plane.h)
- [`imgproc/demosaicing.h`](../include/cvh/imgproc/demosaicing.h)
- [`imgproc/geometry_transform.h`](../include/cvh/imgproc/geometry_transform.h)
- [`imgproc/convert_maps.h`](../include/cvh/imgproc/convert_maps.h)
- [`imgproc/remap.h`](../include/cvh/imgproc/remap.h)
- [`imgproc/warp_perspective.h`](../include/cvh/imgproc/warp_perspective.h)
- [`imgproc/rect_sub_pix.h`](../include/cvh/imgproc/rect_sub_pix.h)
- [`imgproc/readme.md`](../include/cvh/imgproc/readme.md)

### 6.2 Missing Imgproc Families

- [ ] `EMD`
- [ ] `HoughCircles`
- [ ] `HoughLines`
- [ ] `HoughLinesP`
- [ ] `HoughLinesPointSet`
- [ ] `HuMoments`
- [ ] `approxPolyDP`
- [ ] `approxPolyN`
- [ ] `arcLength`
- [ ] `arrowedLine`
- [ ] `boundingRect`
- [ ] `boxPoints`
- [ ] `calcBackProject`
- [ ] `calcHist`
- [ ] `circle`
- [ ] `clipLine`
- [ ] `compareHist`
- [ ] `connectedComponents`
- [ ] `connectedComponentsWithStats`
- [ ] `contourArea`
- [ ] `convexHull`
- [ ] `convexityDefects`
- [ ] `cornerEigenValsAndVecs`
- [ ] `cornerHarris`
- [ ] `cornerMinEigenVal`
- [ ] `cornerSubPix`
- [ ] `createCLAHE`
- [ ] `createGeneralizedHoughBallard`
- [ ] `createGeneralizedHoughGuil`
- [ ] `createLineSegmentDetector`
- [ ] `distanceTransform`
- [ ] `divSpectrums`
- [ ] `drawContours`
- [ ] `drawMarker`
- [ ] `ellipse`
- [ ] `ellipse2Poly`
- [ ] `fillConvexPoly`
- [ ] `fillPoly`
- [ ] `findContours`
- [ ] `findContoursLinkRuns`
- [ ] `fitEllipse`
- [ ] `fitEllipseAMS`
- [ ] `fitEllipseDirect`
- [ ] `fitLine`
- [ ] `floodFill`
- [ ] `getClosestEllipsePoints`
- [ ] `getFontScaleFromHeight`
- [ ] `getTextSize`
- [ ] `goodFeaturesToTrack`
- [ ] `grabCut`
- [ ] `intersectConvexConvex`
- [ ] `isContourConvex`
- [ ] `line`
- [ ] `linearPolar`
- [ ] `logPolar`
- [ ] `matchShapes`
- [ ] `matchTemplate`
- [ ] `minAreaRect`
- [ ] `minEnclosingCircle`
- [ ] `minEnclosingConvexPolygon`
- [ ] `minEnclosingTriangle`
- [ ] `moments`
- [ ] `phaseCorrelate`
- [ ] `phaseCorrelateIterative`
- [ ] `pointPolygonTest`
- [ ] `polylines`
- [ ] `preCornerDetect`
- [ ] `putText`
- [ ] `pyrMeanShiftFiltering`
- [ ] `rectangle`
- [ ] `rotatedRectangleIntersection`
- [ ] `warpPolar`
- [ ] `watershed`

### 6.3 Imgproc Class-Only Gaps

Class methods are not included in the 123-family denominator. Important
class-level gaps include:

| Upstream class family | Current `cvh` status |
|---|---|
| `CLAHE` | Missing |
| `Subdiv2D` | Missing |
| `LineSegmentDetector` | Missing |
| `GeneralizedHough`, `GeneralizedHoughBallard`, `GeneralizedHoughGuil` | Missing |
| `IntelligentScissorsMB` | Missing |

## 7. What The Coverage Number Means

The present implementation covers a useful preprocessing subset, not the
general OpenCV `core` and `imgproc` surface:

- Current strength: matrix ownership/layout, basic element-wise arithmetic,
  GEMM, resize, color conversion, thresholding, common filtering, Canny, and
  morphology.
- Largest core gaps: reductions/statistics, bitwise operations, channel/layout
  transforms, dense linear algebra, spectral transforms, random utilities,
  and persistence/infrastructure.
- Largest imgproc gaps: histogram/equalization, geometric remap and
  perspective transforms, pyramids, advanced smoothing, contours/shapes,
  feature/corner detection, drawing, segmentation, and Hough transforms.

A practical next API tranche should favor broadly reusable primitives before
large algorithm families:

1. Core reductions and predicates: `sum`, `mean`, `countNonZero`,
   `minMaxLoc`, and an implemented `norm`.
2. Core element-wise utilities: `absdiff`, bitwise operations, `inRange`,
   `min`, and `max`.
3. Core layout utilities: `flip`, `rotate`, `hconcat`, `vconcat`,
   `extractChannel`, and `insertChannel`.
4. Imgproc high-frequency primitives: `adaptiveThreshold`, `equalizeHist`,
   `medianBlur`, `pyrDown`, `pyrUp`, `remap`, and `warpPerspective`.
5. Only then expand into contours, drawing, Hough, segmentation, and
   class-based algorithms.

## 8. Maintenance Rules

Update this document when any of the following happens:

- The pinned OpenCV checkout changes.
- A new public cvh operation gains a complete inline definition.
- A declared-only operation becomes usable or is removed.
- A supported operation expands its accepted depth, channel, overload, border,
  interpolation, or flag matrix.

An API moves from `Missing` to `Available (subset)` only when:

1. It is reachable from a public cvh umbrella header.
2. Its implementation is header-defined and requires no project `.cpp`.
3. Correctness tests cover the accepted parameter matrix.
4. Unsupported upstream combinations fail explicitly instead of silently
   producing different behavior.

Benchmark coverage is tracked separately. An operation can be correct and
available without having a fast path, and a benchmark does not by itself make
an API supported.
