# OpenCV Core / Imgproc API Coverage

Last updated: 2026-07-23

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

Implementation sequencing is tracked in
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
| `core` | 97 | 14 | 1 | 82 | 14.4% |
| `imgproc` | 123 | 14 | 0 | 109 | 11.4% |
| **Total** | **220** | **28** | **1** | **191** | **12.7%** |

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
| `add` | Available (subset) | `Mat/Mat`, `Mat/Scalar`, and `Scalar/Mat`; no upstream mask or output-depth arguments. |
| `addWeighted` | Available (subset) | `a * alpha + b * beta`; the upstream `gamma` and `dtype` arguments are absent. |
| `compare` | Available (subset) | `Mat/Mat`, `Mat/Scalar`, and `Scalar/Mat`; produces the comparison mask for the supported cvh depths. |
| `copyMakeBorder` | Available (subset) | `CV_8U` and `CV_32F`; constant, replicate, reflect, reflect-101, and wrap borders. |
| `divide` | Available (subset) | Element-wise `Mat/Mat`, `Mat/Scalar`, and `Scalar/Mat`; no upstream `scale` or `dtype` arguments. |
| `error` | Available (subset) | Header-defined `cvh::Exception` throwing helpers; upstream callback/redirection behavior is not reproduced. |
| `gemm` | Available (subset) | Return-by-value cvh API; FP32 activation with FP32/FP16 weights, batched/broadcast paths, and cvh packed-B extensions. It is not signature-compatible with `cv::gemm`. |
| `merge` | Available (subset) | Pointer/count and `std::vector<Mat>` inputs; no `InputArrayOfArrays` abstraction. |
| `multiply` | Available (subset) | Element-wise `Mat/Mat`, `Mat/Scalar`, and `Scalar/Mat`; no upstream `scale` or `dtype` arguments. |
| `split` | Available (subset) | Pointer and `std::vector<Mat>` outputs; no `OutputArrayOfArrays` abstraction. |
| `subtract` | Available (subset) | Unary negate plus `Mat/Mat`, `Mat/Scalar`, and scalar forms; no upstream mask or output-depth arguments. |
| `transpose` | Available (subset) | Return-by-value blocked transpose; the public signature differs from OpenCV's output-argument API. |
| `transposeND` | Available (subset) | Return-by-value N-D axis permutation; the public signature differs from OpenCV's output-argument API. |

Public implementation sources:

- [`core/basic_op.h`](../include/cvh/core/basic_op.h)
- [`core/gemm.h`](../include/cvh/core/gemm.h)
- [`core/system.h`](../include/cvh/core/system.h)
- [`imgproc/lut.h`](../include/cvh/imgproc/lut.h)
- [`imgproc/copy_make_border.h`](../include/cvh/imgproc/copy_make_border.h)

### 4.2 Declared-Only Core Families

| Upstream API | Status | Current issue |
|---|---|---|
| `norm` | Declared only | `cvh::norm` declarations remain in `core/basic_op.h`, but no accepted inline definition exists. |

### 4.3 Missing Core Families

- [ ] `Mahalanobis`
- [ ] `PCABackProject`
- [ ] `PCACompute`
- [ ] `PCAProject`
- [ ] `PSNR`
- [ ] `SVBackSubst`
- [ ] `SVDecomp`
- [ ] `absdiff`
- [ ] `batchDistance`
- [ ] `bitwise_and`
- [ ] `bitwise_not`
- [ ] `bitwise_or`
- [ ] `bitwise_xor`
- [ ] `borderInterpolate`
- [ ] `broadcast`
- [ ] `calcCovarMatrix`
- [ ] `cartToPolar`
- [ ] `checkRange`
- [ ] `completeSymm`
- [ ] `convertFp16`
- [ ] `convertScaleAbs`
- [ ] `copyTo`
- [ ] `countNonZero`
- [ ] `dct`
- [ ] `determinant`
- [ ] `dft`
- [ ] `eigen`
- [ ] `eigenNonSymmetric`
- [ ] `exp`
- [ ] `extractChannel`
- [ ] `findNonZero`
- [ ] `flip`
- [ ] `flipND`
- [ ] `getOptimalDFTSize`
- [ ] `hasNonZero`
- [ ] `hconcat`
- [ ] `idct`
- [ ] `idft`
- [ ] `inRange`
- [ ] `insertChannel`
- [ ] `invert`
- [ ] `kmeans`
- [ ] `log`
- [ ] `magnitude`
- [ ] `max`
- [ ] `mean`
- [ ] `meanStdDev`
- [ ] `min`
- [ ] `minMaxIdx`
- [ ] `minMaxLoc`
- [ ] `mixChannels`
- [ ] `mulSpectrums`
- [ ] `mulTransposed`
- [ ] `normalize`
- [ ] `patchNaNs`
- [ ] `perspectiveTransform`
- [ ] `phase`
- [ ] `polarToCart`
- [ ] `pow`
- [ ] `randShuffle`
- [ ] `randn`
- [ ] `randu`
- [ ] `reduce`
- [ ] `reduceArgMax`
- [ ] `reduceArgMin`
- [ ] `repeat`
- [ ] `rotate`
- [ ] `scaleAdd`
- [ ] `setIdentity`
- [ ] `setRNGSeed`
- [ ] `solve`
- [ ] `solveCubic`
- [ ] `solvePoly`
- [ ] `sort`
- [ ] `sortIdx`
- [ ] `sqrt`
- [ ] `sum`
- [ ] `swap`
- [ ] `theRNG`
- [ ] `trace`
- [ ] `transform`
- [ ] `vconcat`

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
| `Sobel` | Available (subset) | `CV_8U`/`CV_16S`/`CV_32F` input, `CV_16S`/`CV_32F` output, first derivatives, kernel size 3/5. |
| `blur` | Available (subset) | `CV_8U`/`CV_32F`, C1/C3/C4; implemented as the normalized `boxFilter` wrapper. |
| `boxFilter` | Available (subset) | `CV_8U`/`CV_32F`, C1/C3/C4, common border modes; selected 3x3 and separable fast paths. |
| `cvtColor` | Available (subset) | Common gray/BGR/RGB/BGRA/RGBA conversions for `CV_8U`/`CV_32F`, plus the documented `CV_8U` YUV420/YUV422/YUV444 families. |
| `dilate` | Available (subset) | `CV_8U`, C1/C3/C4, custom kernel, iterations, and basic border handling. |
| `erode` | Available (subset) | `CV_8U`, C1/C3/C4, custom kernel, iterations, and basic border handling. |
| `filter2D` | Available (subset) | `CV_8U`/`CV_32F` source, `CV_32FC1` kernel, selected destination depths and borders. |
| `morphologyEx` | Available (subset) | Erode, dilate, open, close, gradient, top-hat, black-hat, and hit-or-miss; hit-or-miss is limited to `CV_8UC1`. |
| `resize` | Available (subset) | `CV_8U`/`CV_32F`, C1/C3/C4, nearest, nearest-exact, and linear interpolation. |
| `sepFilter2D` | Available (subset) | `CV_8U`/`CV_32F` source with single-channel FP32 vector kernels and selected borders. |
| `threshold` | Available (subset) | Fixed thresholds for `CV_8U`/`CV_32F`; Otsu/Triangle only for `CV_8UC1`. |
| `warpAffine` | Available (subset) | `CV_8U`/`CV_32F`, C1/C3/C4, nearest/linear, inverse-map flag, and selected borders. |

Public implementation source:

- [`imgproc/imgproc.h`](../include/cvh/imgproc/imgproc.h)
- [`imgproc/readme.md`](../include/cvh/imgproc/readme.md)

### 6.2 Missing Imgproc Families

- [ ] `EMD`
- [ ] `HoughCircles`
- [ ] `HoughLines`
- [ ] `HoughLinesP`
- [ ] `HoughLinesPointSet`
- [ ] `HuMoments`
- [ ] `Laplacian`
- [ ] `Scharr`
- [ ] `accumulate`
- [ ] `accumulateProduct`
- [ ] `accumulateSquare`
- [ ] `accumulateWeighted`
- [ ] `adaptiveThreshold`
- [ ] `applyColorMap`
- [ ] `approxPolyDP`
- [ ] `approxPolyN`
- [ ] `arcLength`
- [ ] `arrowedLine`
- [ ] `bilateralFilter`
- [ ] `blendLinear`
- [ ] `boundingRect`
- [ ] `boxPoints`
- [ ] `buildPyramid`
- [ ] `calcBackProject`
- [ ] `calcHist`
- [ ] `circle`
- [ ] `clipLine`
- [ ] `compareHist`
- [ ] `connectedComponents`
- [ ] `connectedComponentsWithStats`
- [ ] `contourArea`
- [ ] `convertMaps`
- [ ] `convexHull`
- [ ] `convexityDefects`
- [ ] `cornerEigenValsAndVecs`
- [ ] `cornerHarris`
- [ ] `cornerMinEigenVal`
- [ ] `cornerSubPix`
- [ ] `createCLAHE`
- [ ] `createGeneralizedHoughBallard`
- [ ] `createGeneralizedHoughGuil`
- [ ] `createHanningWindow`
- [ ] `createLineSegmentDetector`
- [ ] `cvtColorTwoPlane`
- [ ] `demosaicing`
- [ ] `distanceTransform`
- [ ] `divSpectrums`
- [ ] `drawContours`
- [ ] `drawMarker`
- [ ] `ellipse`
- [ ] `ellipse2Poly`
- [ ] `equalizeHist`
- [ ] `fillConvexPoly`
- [ ] `fillPoly`
- [ ] `findContours`
- [ ] `findContoursLinkRuns`
- [ ] `fitEllipse`
- [ ] `fitEllipseAMS`
- [ ] `fitEllipseDirect`
- [ ] `fitLine`
- [ ] `floodFill`
- [ ] `getAffineTransform`
- [ ] `getClosestEllipsePoints`
- [ ] `getDerivKernels`
- [ ] `getFontScaleFromHeight`
- [ ] `getGaborKernel`
- [ ] `getGaussianKernel`
- [ ] `getPerspectiveTransform`
- [ ] `getRectSubPix`
- [ ] `getRotationMatrix2D`
- [ ] `getRotationMatrix2D_`
- [ ] `getStructuringElement`
- [ ] `getTextSize`
- [ ] `goodFeaturesToTrack`
- [ ] `grabCut`
- [ ] `integral`
- [ ] `intersectConvexConvex`
- [ ] `invertAffineTransform`
- [ ] `isContourConvex`
- [ ] `line`
- [ ] `linearPolar`
- [ ] `logPolar`
- [ ] `matchShapes`
- [ ] `matchTemplate`
- [ ] `medianBlur`
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
- [ ] `pyrDown`
- [ ] `pyrMeanShiftFiltering`
- [ ] `pyrUp`
- [ ] `rectangle`
- [ ] `remap`
- [ ] `rotatedRectangleIntersection`
- [ ] `spatialGradient`
- [ ] `sqrBoxFilter`
- [ ] `stackBlur`
- [ ] `thresholdWithMask`
- [ ] `warpPerspective`
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
