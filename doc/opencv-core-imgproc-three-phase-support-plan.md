# OpenCV Core / Imgproc Three-Phase Support Plan

Last updated: 2026-07-24

Status: planned

## 1. Goal

This plan sequences the gaps recorded in
[opencv-core-imgproc-api-coverage.md](opencv-core-imgproc-api-coverage.md) into
three implementation phases.

The tracked baseline is:

- 220 upstream CPU C++ operation families: 97 `core` and 123 `imgproc`.
- 28 families currently have a callable cvh subset.
- `norm` is declared but has no accepted header-only definition.
- 191 families are missing.

The three phases cover all 192 outstanding operation families:

| Phase | Theme | Core gaps | Imgproc gaps | Total added | Cumulative callable coverage |
|---|---|---:|---:|---:|---:|
| P-API-1 | Foundation and common pipelines | 43 | 36 | 79 | 107 / 220, 48.6% |
| P-API-2 | Numerical analysis, features, and shapes | 35 | 47 | 82 | 189 / 220, 85.9% |
| P-API-3 | Complex algorithms and long-tail APIs | 5 | 26 | 31 | 220 / 220, 100% |

The final percentage is operation-family name coverage under documented cvh
contracts. It is not full OpenCV overload, type, ABI, class-method, or backend
compatibility.

Each phase is a macro phase, not one pull request. A wave should normally add
no more than one coherent dependency cluster and should remain independently
reviewable.

### Progress Tracker

Allowed status values: `not started`, `in progress`, `complete`, `blocked`.

| Wave | Purpose | Status |
|---|---|---|
| P-API-1.0 | Type and test prerequisites | not started |
| P-API-1.1 | Core element-wise and logical primitives | not started |
| P-API-1.2 | Core conversion, math, and validation | not started |
| P-API-1.3 | Core reductions and statistics | not started |
| P-API-1.4 | Core layout, copy, and channel utilities | not started |
| P-API-1.5 | Imgproc kernel, filter, and intensity primitives | not started |
| P-API-1.6 | Imgproc accumulation, pyramids, and color input | not started |
| P-API-1.7 | Imgproc geometric transform foundation | not started |
| P-API-1.8 | P-API-1 phase gate | not started |
| P-API-2.0 | Type and object prerequisites | not started |
| P-API-2.1 | Core linear algebra and statistical analysis | not started |
| P-API-2.2 | Core coordinate, spectral, random, and ordering APIs | not started |
| P-API-2.3 | Imgproc histogram, spectral, region, and polar APIs | not started |
| P-API-2.4 | Imgproc corners and feature selection | not started |
| P-API-2.5 | Imgproc contours and shape analysis | not started |
| P-API-2.6 | P-API-2 phase gate | not started |
| P-API-3.0 | Class prerequisites | not started |
| P-API-3.1 | Core numerical long tail | not started |
| P-API-3.2 | Imgproc drawing and text | not started |
| P-API-3.3 | Imgproc Hough and detector objects | not started |
| P-API-3.4 | Imgproc segmentation and specialized algorithms | not started |
| P-API-3.5 | Full inventory gate | not started |

## 2. Product Constraints

All three phases must preserve the current product direction:

- Pure header-only public implementation. No accepted operator may require a
  project `.cpp`.
- CPU only.
- `cvh::headers` and `cvh::headers_fast` expose the same APIs.
- `cvh::headers_fast` only selects validated fast paths.
- Scalar/header C++ correctness lands before SIMD optimization.
- OpenCV Universal Intrinsics is the preferred SIMD dialect.
- Do not reintroduce xsimd as an optimization candidate.
- Current architecture focus is x86 SSE/AVX and ARM NEON.
- RVV remains a separate future TODO because scalable vectors need a dedicated
  design.
- UMat, CUDA, OpenCL, OpenGL, DirectX, and other runtime/hardware integration
  APIs remain outside this plan.

API-family support means that cvh exposes a public function for the operation
with a useful, explicit support matrix. It does not require reproducing
OpenCV's `InputArray` and `OutputArray` wrappers. Concrete `cvh::Mat` APIs are
acceptable where they keep the header-only design smaller and clearer.

## 3. Common Implementation Workflow

Every wave follows the same sequence.

### 3.1 Contract First

Before implementation:

- Record the upstream declaration, overloads, accepted depths, channels,
  layouts, flags, and in-place behavior.
- Define the initial cvh support matrix.
- Decide whether the public API belongs in an existing header or a new focused
  header.
- Add the public header to the correct umbrella header.
- Explicitly reject unsupported combinations.

Initial support should prioritize:

- Pixel operations: `CV_8U`, `CV_16S`, and `CV_32F`, then other existing cvh
  depths when semantics are clear.
- Image operations: C1, C3, and C4, including continuous and ROI inputs where
  OpenCV permits them.
- Numerical linear algebra: `CV_32F` and `CV_64F`.
- Drawing and display-oriented operations: `CV_8U` first.

### 3.2 Scalar Correctness

- Implement the complete accepted matrix in ODR-safe headers.
- Add focused contract tests under `test/core/` or `test/imgproc/`.
- Differentially compare against the local OpenCV checkout.
- Cover empty/invalid input, odd sizes, single-row/single-column cases, ROI,
  non-contiguous steps, aliasing, and in-place behavior where applicable.
- Use exact comparison for integer results and an operation-specific tolerance
  for floating-point results.

### 3.3 Performance

- Do not add SIMD merely because an API is new.
- Add Mode A regression rows for operations that can regress existing cvh
  pipelines.
- Add Mode B `cvh::headers_fast` versus OpenCV rows for CPU hotspots.
- Start OpenCV UI work only after profiling identifies the kernel cost.
- Preserve a scalar fallback and explicit SIMD tail handling.

### 3.4 Documentation And Status

After a wave passes:

- Move each accepted family in
  `opencv-core-imgproc-api-coverage.md` from `Missing` or `Declared only` to
  `Available (subset)`.
- Record its exact type, channel, flag, and layout matrix.
- Update the root README operator table when the public surface materially
  changes.
- Update benchmark reports only when comparable measurements exist.

## 4. P-API-1: Foundation And Common Pipelines

Status: planned

Goal: add the reusable primitives needed by most preprocessing pipelines and
by the later analytical algorithms.

P-API-1 adds 79 operation families: 43 `core` and 36 `imgproc`.

### P-API-1.0: Type And Test Prerequisites

Operation-family count: 0

Required foundation work:

- Make `CV_64F` a real `Mat` depth across allocation, element size, typed
  access, conversion, scalar fill, and tests. The macro exists today, but
  `Mat` currently rejects the depth.
- Add the minimum floating-point geometry types needed by upstream-style
  transforms, including `Point2f` and `Point2d`.
- Decide whether to generalize `Point`/`Size` as templates or add narrow
  aliases without destabilizing existing APIs.
- Add reusable OpenCV differential-test helpers for generated matrices,
  tolerances, ROI, and parameterized type/channel cases.
- Add API coverage metadata so tests can identify the family and accepted
  matrix they cover.

Exit gate:

- `CV_64F` Mat lifecycle and conversion tests pass.
- Existing Mat ABI assumptions used inside cvh remain intact.
- Header compile, include-only, ODR, and header-only contract checks pass.

### P-API-1.1: Core Element-Wise And Logical Primitives

Operation-family count: 8

APIs:

- `absdiff`
- `bitwise_and`
- `bitwise_not`
- `bitwise_or`
- `bitwise_xor`
- `inRange`
- `min`
- `max`

Dependency rationale:

- These are low-level pixel kernels reused by masks, morphology, thresholding,
  compositing, and later algorithms.
- They share iteration, scalar broadcasting, ROI, saturation, and SIMD
  dispatch infrastructure.

### P-API-1.2: Core Conversion, Math, And Validation

Operation-family count: 9

APIs:

- `scaleAdd`
- `convertScaleAbs`
- `convertFp16`
- `sqrt`
- `pow`
- `exp`
- `log`
- `checkRange`
- `patchNaNs`

Dependency rationale:

- Conversion and unary math establish reusable typed loops for normalization,
  numerical analysis, and image statistics.
- `checkRange` and `patchNaNs` provide validation and cleanup paths needed by
  later solvers.

### P-API-1.3: Core Reductions And Statistics

Operation-family count: 13

APIs:

- `norm`
- `sum`
- `mean`
- `meanStdDev`
- `countNonZero`
- `hasNonZero`
- `findNonZero`
- `minMaxIdx`
- `minMaxLoc`
- `reduce`
- `reduceArgMax`
- `reduceArgMin`
- `normalize`

Dependency rationale:

- This wave turns the current declared-only `norm` into a real header-only
  API.
- The operations share reduction, mask, accumulation precision, and
  parallel-partition logic.
- Reductions are prerequisites for statistics, feature scoring, histogram
  validation, and numerical tests in P-API-2.

### P-API-1.4: Core Layout, Copy, And Channel Utilities

Operation-family count: 13

APIs:

- `borderInterpolate`
- `copyTo`
- `extractChannel`
- `insertChannel`
- `mixChannels`
- `flip`
- `flipND`
- `rotate`
- `repeat`
- `hconcat`
- `vconcat`
- `broadcast`
- `swap`

Dependency rationale:

- These APIs share step-aware copy and channel-routing kernels.
- `borderInterpolate` removes duplicated border logic from imgproc filters.
- Layout primitives simplify pyramid, remap, histogram, and shape test data.

### P-API-1.5: Imgproc Kernel, Filter, And Intensity Primitives

Operation-family count: 17

APIs:

- `getStructuringElement`
- `getGaussianKernel`
- `getDerivKernels`
- `getGaborKernel`
- `createHanningWindow`
- `integral`
- `Scharr`
- `Laplacian`
- `spatialGradient`
- `sqrBoxFilter`
- `medianBlur`
- `bilateralFilter`
- `stackBlur`
- `adaptiveThreshold`
- `thresholdWithMask`
- `equalizeHist`
- `applyColorMap`

Dependency rationale:

- Kernel factories should be shared by existing filters and new derivative
  operators.
- `integral` supports adaptive thresholding and later region statistics.
- Scharr, Laplacian, and spatial gradients build on the existing Sobel and
  separable-filter infrastructure.
- Histogram equalization and color maps are common preprocessing and
  visualization operations with limited external dependencies.

### P-API-1.6: Imgproc Accumulation, Pyramids, And Color Input

Operation-family count: 10

APIs:

- `accumulate`
- `accumulateProduct`
- `accumulateSquare`
- `accumulateWeighted`
- `blendLinear`
- `pyrDown`
- `pyrUp`
- `buildPyramid`
- `cvtColorTwoPlane`
- `demosaicing`

Dependency rationale:

- Accumulation APIs share typed floating-point accumulation loops.
- Pyramid operations reuse Gaussian filtering and resize/layout primitives.
- Two-plane color conversion extends the current YUV implementation without
  introducing a new color-conversion engine.
- Demosaicing should reuse border and color-channel helpers but remain a
  distinct kernel.

### P-API-1.7: Imgproc Geometric Transform Foundation

Operation-family count: 9

APIs:

- `remap`
- `convertMaps`
- `warpPerspective`
- `getAffineTransform`
- `getPerspectiveTransform`
- `getRotationMatrix2D`
- `getRotationMatrix2D_`
- `invertAffineTransform`
- `getRectSubPix`

Dependency rationale:

- `remap` is the shared sampling engine.
- `warpPerspective` should reuse remap interpolation, border, and coordinate
  logic instead of creating a separate sampler.
- Matrix-construction helpers depend on the P-API-1.0 floating-point geometry
  and `CV_64F` work.
- The existing `warpAffine` path should be refactored only when sharing code
  reduces duplication without changing its accepted behavior.

### P-API-1.8: Phase Gate

P-API-1 is complete only when:

- All 79 families are public, header-defined, and contract-tested.
- The coverage document reports 107 / 220 callable families.
- Common pixel/reduction kernels have Mode A rows.
- At minimum, reductions, median/bilateral filtering, pyramids, remap, and
  warpPerspective have Mode B rows.
- No accepted API requires a `.cpp`.
- Existing operator tests and header-only smoke/ODR checks pass.

## 5. P-API-2: Numerical Analysis, Features, And Shapes

Status: planned

Goal: build the numerical and image-analysis layer on top of P-API-1.

P-API-2 adds 82 operation families: 35 `core` and 47 `imgproc`.

### P-API-2.0: Type And Object Prerequisites

Operation-family count: 0

Required foundation work:

- Add `Rect`/`Rect2f`, `RotatedRect`, `Moments`, `TermCriteria`, and the
  minimum fixed-size vector aliases required by accepted APIs.
- Add a minimal header-only `RNG` object before exposing `theRNG`, `randu`,
  `randn`, and `randShuffle`.
- Define stable result containers for contours, labels, histograms, and shape
  fitting without introducing `InputArray`/`OutputArray` wrappers by accident.
- Introduce minimal `PCA`, `SVD`, and `CLAHE` classes only where the public
  operation contract requires class behavior. Keep their first accepted
  surface narrow.

### P-API-2.1: Core Linear Algebra And Statistical Analysis

Operation-family count: 16

APIs:

- `setIdentity`
- `trace`
- `determinant`
- `completeSymm`
- `invert`
- `solve`
- `mulTransposed`
- `SVDecomp`
- `SVBackSubst`
- `calcCovarMatrix`
- `PCACompute`
- `PCAProject`
- `PCABackProject`
- `Mahalanobis`
- `PSNR`
- `batchDistance`

Dependency rationale:

- Matrix decomposition and solve kernels should share pivoting, workspace,
  and precision policy.
- PCA and covariance build on GEMM, reductions, transpose, and decomposition.
- PSNR and Mahalanobis build on reductions and linear algebra.
- Start with `CV_32F`/`CV_64F`; integer inputs may convert through explicit
  documented paths.

### P-API-2.2: Core Coordinate, Spectral, Random, And Ordering APIs

Operation-family count: 19

APIs:

- `cartToPolar`
- `polarToCart`
- `phase`
- `magnitude`
- `transform`
- `perspectiveTransform`
- `dft`
- `idft`
- `dct`
- `idct`
- `mulSpectrums`
- `getOptimalDFTSize`
- `randu`
- `randn`
- `randShuffle`
- `setRNGSeed`
- `theRNG`
- `sort`
- `sortIdx`

Dependency rationale:

- Coordinate transforms share vector math from P-API-1.
- Spectral transforms are prerequisites for phase correlation and fast
  template matching.
- Random generation and sorting are reusable test and clustering
  infrastructure.

### P-API-2.3: Imgproc Histogram, Spectral, Region, And Polar APIs

Operation-family count: 15

APIs:

- `calcHist`
- `calcBackProject`
- `compareHist`
- `createCLAHE`
- `matchTemplate`
- `phaseCorrelate`
- `phaseCorrelateIterative`
- `divSpectrums`
- `connectedComponents`
- `connectedComponentsWithStats`
- `distanceTransform`
- `floodFill`
- `linearPolar`
- `logPolar`
- `warpPolar`

Dependency rationale:

- Histogram storage and binning should be shared by histogram comparison,
  backprojection, equalization, and CLAHE.
- Phase correlation and spectrum division depend on P-API-2.2.
- Connected components, distance transform, and flood fill establish reusable
  region/label infrastructure.
- Polar transforms reuse the P-API-1 remap engine.

### P-API-2.4: Imgproc Corners And Feature Selection

Operation-family count: 6

APIs:

- `cornerEigenValsAndVecs`
- `cornerHarris`
- `cornerMinEigenVal`
- `cornerSubPix`
- `preCornerDetect`
- `goodFeaturesToTrack`

Dependency rationale:

- This cluster shares derivative, local covariance, reduction, sorting, and
  sub-pixel refinement infrastructure.
- It should reuse P-API-1 Scharr/Sobel and P-API-2 ordering utilities.

### P-API-2.5: Imgproc Contours And Shape Analysis

Operation-family count: 26

APIs:

- `HuMoments`
- `approxPolyDP`
- `approxPolyN`
- `arcLength`
- `boundingRect`
- `boxPoints`
- `contourArea`
- `convexHull`
- `convexityDefects`
- `findContours`
- `findContoursLinkRuns`
- `fitEllipse`
- `fitEllipseAMS`
- `fitEllipseDirect`
- `fitLine`
- `getClosestEllipsePoints`
- `intersectConvexConvex`
- `isContourConvex`
- `matchShapes`
- `minAreaRect`
- `minEnclosingCircle`
- `minEnclosingConvexPolygon`
- `minEnclosingTriangle`
- `moments`
- `pointPolygonTest`
- `rotatedRectangleIntersection`

Dependency rationale:

- Contour extraction lands before approximation, hull, moments, and fitting.
- Shape fitting depends on P-API-2 linear algebra.
- Geometry result types are established in P-API-2.0 and shared by all
  functions in this wave.
- This large wave should be split into contour extraction, polygon/hull,
  moments/matching, and fitting/enclosing pull requests.

### P-API-2.6: Phase Gate

P-API-2 is complete only when:

- All 82 families are public, header-defined, and contract-tested.
- The coverage document reports 189 / 220 callable families.
- FP32/FP64 numerical tolerances are documented per algorithm family.
- DFT, solve/decomposition, histogram, matchTemplate, connected components,
  and contour extraction have Mode B benchmark coverage.
- Shape APIs agree with OpenCV on representative degenerate and boundary
  inputs, not only normal images.
- P-API-1 regression and performance gates remain green.

## 6. P-API-3: Complex Algorithms And Long-Tail APIs

Status: planned

Goal: finish the operation-family inventory after the shared primitives,
numerical tools, region model, and geometry types are stable.

P-API-3 adds 31 operation families: 5 `core` and 26 `imgproc`.

The family count is smaller than earlier phases, but expected implementation
cost and algorithmic risk are higher.

### P-API-3.0: Class Prerequisites

Operation-family count: 0

Required foundation work:

- Finalize the minimum object model needed by `LineSegmentDetector` and
  `GeneralizedHough` factories.
- Schedule class-only `Subdiv2D` and `IntelligentScissorsMB` after the
  operation-family gate. They do not contribute to the 220 denominator.
- Do not introduce a general OpenCV `Algorithm` hierarchy unless multiple
  implemented classes gain a concrete benefit from it.
- Add deterministic workspace and allocation policies for iterative
  algorithms.

### P-API-3.1: Core Numerical Long Tail

Operation-family count: 5

APIs:

- `eigen`
- `eigenNonSymmetric`
- `solveCubic`
- `solvePoly`
- `kmeans`

Dependency rationale:

- These algorithms need the P-API-2 solver, decomposition, random, sorting,
  and convergence infrastructure.
- `kmeans` should land before `grabCut`.
- Non-symmetric eigen and polynomial solvers require independent numerical
  stability gates and must not be accepted from nominal examples alone.

### P-API-3.2: Imgproc Drawing And Text

Operation-family count: 15

APIs:

- `arrowedLine`
- `circle`
- `clipLine`
- `drawContours`
- `drawMarker`
- `ellipse`
- `ellipse2Poly`
- `fillConvexPoly`
- `fillPoly`
- `getFontScaleFromHeight`
- `getTextSize`
- `line`
- `polylines`
- `putText`
- `rectangle`

Dependency rationale:

- All drawing operations should share clipping, line rasterization, fill,
  color conversion, and anti-aliasing helpers.
- Text measurement and rendering must use one font-metric implementation.
- `drawContours` reuses P-API-2 contour containers.

### P-API-3.3: Imgproc Hough And Detector Objects

Operation-family count: 7

APIs:

- `HoughCircles`
- `HoughLines`
- `HoughLinesP`
- `HoughLinesPointSet`
- `createGeneralizedHoughBallard`
- `createGeneralizedHoughGuil`
- `createLineSegmentDetector`

Dependency rationale:

- Standard Hough implementations reuse the existing Canny path, P-API-2
  sorting, and P-API-3 geometry/vector types.
- Generalized Hough and line-segment objects should reuse common accumulator,
  voting, and result-filtering infrastructure.

### P-API-3.4: Imgproc Segmentation And Specialized Algorithms

Operation-family count: 4

APIs:

- `EMD`
- `grabCut`
- `pyrMeanShiftFiltering`
- `watershed`

Dependency rationale:

- `grabCut` depends on P-API-3 `kmeans` plus P-API-2 region and histogram
  primitives.
- Watershed depends on stable neighborhood, label, and queue handling.
- Pyramid mean shift depends on P-API-1 pyramids and color conversion.
- EMD is a standalone optimization algorithm and should not be allowed to
  distort the earlier foundational API design.

### P-API-3.5: Full Inventory Gate

P-API-3 is complete only when:

- All 31 families are public, header-defined, and contract-tested.
- The coverage document reports 220 / 220 callable operation families.
- The 220-family inventory is regenerated against the pinned upstream commit
  and has no unassigned names.
- Factories return usable header-only objects for their accepted subset.
- Complex algorithms include deterministic tests and bounded workspace
  behavior.
- Representative Hough, drawing, GrabCut, watershed, and mean-shift cases
  have Mode B measurements.
- No completion claim is made for excluded class methods, C APIs, GPU/runtime
  backends, or full overload/type parity.

## 7. Class And Infrastructure Policy

The three phases target the 220 free-function operation families. Class
coverage is tracked separately because counting each method as an operation
would make the denominator unstable.

| Class/infrastructure area | Planned handling |
|---|---|
| `Mat`, `MatExpr`, scalar/geometry types | Expand only when a scheduled operation requires it. |
| `RNG` | Minimal header-only object in P-API-2.0. |
| `PCA`, `SVD` | Minimal interfaces in P-API-2 if required by accepted APIs. |
| `CLAHE` | Minimal usable class with `createCLAHE` in P-API-2.3. |
| `LineSegmentDetector`, `GeneralizedHough*` | Minimal usable classes in P-API-3.3. |
| `Subdiv2D`, `IntelligentScissorsMB` | Follow-up work after the P-API-3 operation-family gate. |
| `InputArray`/`OutputArray` wrappers | Not required; concrete Mat APIs remain acceptable. |
| `SparseMat`, persistence, async, quaternion/affine helper families | Not covered by this operator plan; require separate demand and design review. |
| UMat/CUDA/OpenCL/OpenGL/DirectX/VA | Explicit non-goals for the CPU pure header-only product. |

## 8. Per-API Definition Of Done

An operation family is counted as callable only when all applicable items are
complete:

- Public declaration is reachable through the correct cvh umbrella header.
- Definition is inline/header-safe and passes multi-translation-unit ODR
  checks.
- Accepted types, channels, flags, borders, interpolation modes, and layouts
  are documented.
- Unsupported combinations fail explicitly.
- Correctness tests compare with local OpenCV for the accepted matrix.
- ROI/non-contiguous and in-place behavior is tested where supported.
- Integer and floating-point comparison policy is explicit.
- Existing tests remain green.
- Hot operations have benchmark coverage before fast-path work is accepted.
- Coverage and README status are updated.

Phase completion additionally requires:

```bash
./scripts/check_header_only_contract.sh
./scripts/ci_headers_all.sh
./scripts/ci_benchmark_headers_quick.sh
git diff --check
```

Operator-specific test and benchmark targets must be added to this minimum
gate as each wave lands.

## 9. Risk Controls

### Scope Inflation

Name-level coverage must not silently become a promise to implement every
OpenCV overload. Each operation starts from a useful support matrix and expands
only with tests and demand.

### Header Size And Compile Time

- Keep public headers focused by operation family.
- Put implementation details in `detail/*_impl.hpp`.
- Avoid broad inclusion of unrelated operators.
- Measure umbrella-header compile time and binary size at every phase gate.

### Numerical Stability

- Use `CV_64F` where algorithmic stability requires it.
- Port algorithms from the pinned OpenCV source only with source-path and
  commit attribution.
- Do not weaken tolerances merely to make differential tests pass.

### Performance Regression

- Correct scalar support is not automatically a fast path.
- SIMD should target measured kernels, not orchestration-heavy algorithms.
- Every fast path keeps scalar fallback and tail coverage.
- Regressions remain visible in the dated OpenCV comparison report.

### Plan Drift

When upstream adds or removes an API:

1. Update the coverage inventory and denominator.
2. Assign the new family to a phase or a follow-up queue.
3. Record dependency and scope decisions in this document.
4. Do not change historical phase counts without explaining the delta.
