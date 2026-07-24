# OpenCV Core / Imgproc 三阶段支持清单

更新时间：2026-07-24

## 1. 文档范围

本清单依据
[opencv-core-imgproc-api-coverage.md](opencv-core-imgproc-api-coverage.md)
中的 220 个 upstream CPU C++ 操作族进行分期。

第一阶段已完成后，当前共有 107 个可调用操作族，没有仅声明未实现的
第一阶段 API，另有 113 个操作族未支持。本清单只回答：

- 每个阶段支持哪些 `core` 算子；
- 每个阶段支持哪些 `imgproc` 算子；
- 为什么把这些算子放在该阶段。

本清单不描述实现方法、测试步骤、SIMD 策略或交付流程。第三阶段完成后的
`220 / 220` 表示操作族名称级覆盖，不表示完整覆盖 OpenCV 的全部重载、类型、
类成员或后端。

## 2. 三阶段总览

| 阶段 | 定位 | Core | Imgproc | 本阶段新增 | 累计覆盖 |
|---|---|---:|---:|---:|---:|
| 第一阶段 | 高频基础算子与通用图像流水线 | 43 | 36 | 79 | 107 / 220，48.6% |
| 第二阶段 | 数值分析、特征与形状分析 | 35 | 47 | 82 | 189 / 220，85.9% |
| 第三阶段 | 高复杂度算法与长尾接口 | 5 | 26 | 31 | 220 / 220，100% |

## 3. 第一阶段：高频基础算子与通用图像流水线

第一阶段优先补齐调用频率高、复用范围广，并且会被第二、三阶段反复依赖的
基础能力。

详细落地顺序与验收标准见
[opencv-core-imgproc-phase1-implementation-plan.md](opencv-core-imgproc-phase1-implementation-plan.md)。

### 3.1 Core 支持列表

<!-- P1_CORE_API_LIST_START -->
| 类别 | 算子 | 数量 | 放在第一阶段的原因 |
|---|---|---:|---|
| 逐元素与逻辑运算 | `absdiff`<br>`bitwise_and`<br>`bitwise_not`<br>`bitwise_or`<br>`bitwise_xor`<br>`inRange`<br>`min`<br>`max` | 8 | 掩码、阈值、形态学、合成和后续区域算法都会直接使用。 |
| 转换、数学与数据校验 | `scaleAdd`<br>`convertScaleAbs`<br>`convertFp16`<br>`sqrt`<br>`pow`<br>`exp`<br>`log`<br>`checkRange`<br>`patchNaNs` | 9 | 是归一化、数值预处理、浮点数据清理和后续统计计算的基础。 |
| 归约与统计 | `norm`<br>`sum`<br>`mean`<br>`meanStdDev`<br>`countNonZero`<br>`hasNonZero`<br>`findNonZero`<br>`minMaxIdx`<br>`minMaxLoc`<br>`reduce`<br>`reduceArgMax`<br>`reduceArgMin`<br>`normalize` | 13 | 使用频率高，并为特征评分、直方图、距离度量和数值算法提供公共能力。 |
| 布局、复制与通道操作 | `borderInterpolate`<br>`copyTo`<br>`extractChannel`<br>`insertChannel`<br>`mixChannels`<br>`flip`<br>`flipND`<br>`rotate`<br>`repeat`<br>`hconcat`<br>`vconcat`<br>`broadcast`<br>`swap` | 13 | 解决数据搬运和布局转换，能够被滤波、金字塔、几何变换及测试数据构造复用。 |
| **合计** |  | **43** |  |
<!-- P1_CORE_API_LIST_END -->

### 3.2 Imgproc 支持列表

<!-- P1_IMGPROC_API_LIST_START -->
| 类别 | 算子 | 数量 | 放在第一阶段的原因 |
|---|---|---:|---|
| 核生成、滤波与强度处理 | `getStructuringElement`<br>`getGaussianKernel`<br>`getDerivKernels`<br>`getGaborKernel`<br>`createHanningWindow`<br>`integral`<br>`Scharr`<br>`Laplacian`<br>`spatialGradient`<br>`sqrBoxFilter`<br>`medianBlur`<br>`bilateralFilter`<br>`stackBlur`<br>`adaptiveThreshold`<br>`thresholdWithMask`<br>`equalizeHist`<br>`applyColorMap` | 17 | 与现有滤波、Sobel、threshold 路径关联紧密，也是最常见的预处理能力。 |
| 累积、金字塔与颜色输入 | `accumulate`<br>`accumulateProduct`<br>`accumulateSquare`<br>`accumulateWeighted`<br>`blendLinear`<br>`pyrDown`<br>`pyrUp`<br>`buildPyramid`<br>`cvtColorTwoPlane`<br>`demosaicing` | 10 | 视频统计、图像融合、多尺度处理和相机/YUV 输入场景使用频率较高。 |
| 几何变换基础 | `remap`<br>`convertMaps`<br>`warpPerspective`<br>`getAffineTransform`<br>`getPerspectiveTransform`<br>`getRotationMatrix2D`<br>`getRotationMatrix2D_`<br>`invertAffineTransform`<br>`getRectSubPix` | 9 | remap 和变换矩阵是透视、极坐标、配准及后续几何算法的共同基础。 |
| **合计** |  | **36** |  |
<!-- P1_IMGPROC_API_LIST_END -->

## 4. 第二阶段：数值分析、特征与形状分析

第二阶段建立在第一阶段的归约、滤波、布局和几何能力之上，重点覆盖计算机
视觉分析流程中常用但依赖关系更复杂的算子。

### 4.1 Core 支持列表

<!-- P2_CORE_API_LIST_START -->
| 类别 | 算子 | 数量 | 放在第二阶段的原因 |
|---|---|---:|---|
| 线性代数与统计分析 | `setIdentity`<br>`trace`<br>`determinant`<br>`completeSymm`<br>`invert`<br>`solve`<br>`mulTransposed`<br>`SVDecomp`<br>`SVBackSubst`<br>`calcCovarMatrix`<br>`PCACompute`<br>`PCAProject`<br>`PCABackProject`<br>`Mahalanobis`<br>`PSNR`<br>`batchDistance` | 16 | PCA、拟合、距离和质量评估依赖矩阵分解、求解、GEMM 与第一阶段归约能力。 |
| 坐标、频域、随机与排序 | `cartToPolar`<br>`polarToCart`<br>`phase`<br>`magnitude`<br>`transform`<br>`perspectiveTransform`<br>`dft`<br>`idft`<br>`dct`<br>`idct`<br>`mulSpectrums`<br>`getOptimalDFTSize`<br>`randu`<br>`randn`<br>`randShuffle`<br>`setRNGSeed`<br>`theRNG`<br>`sort`<br>`sortIdx` | 19 | 频域算子支撑模板匹配和相位相关；随机与排序支撑聚类、特征选择和测试数据。 |
| **合计** |  | **35** |  |
<!-- P2_CORE_API_LIST_END -->

### 4.2 Imgproc 支持列表

<!-- P2_IMGPROC_API_LIST_START -->
| 类别 | 算子 | 数量 | 放在第二阶段的原因 |
|---|---|---:|---|
| 直方图、频域、区域与极坐标 | `calcHist`<br>`calcBackProject`<br>`compareHist`<br>`createCLAHE`<br>`matchTemplate`<br>`phaseCorrelate`<br>`phaseCorrelateIterative`<br>`divSpectrums`<br>`connectedComponents`<br>`connectedComponentsWithStats`<br>`distanceTransform`<br>`floodFill`<br>`linearPolar`<br>`logPolar`<br>`warpPolar` | 15 | 依赖第一阶段 remap/归约和本阶段 DFT，同时为分割、匹配和区域分析提供基础。 |
| 角点与特征选择 | `cornerEigenValsAndVecs`<br>`cornerHarris`<br>`cornerMinEigenVal`<br>`cornerSubPix`<br>`preCornerDetect`<br>`goodFeaturesToTrack` | 6 | 共用梯度、局部协方差、排序和亚像素计算，适合作为一个关联算子组。 |
| 轮廓与形状分析 | `HuMoments`<br>`approxPolyDP`<br>`approxPolyN`<br>`arcLength`<br>`boundingRect`<br>`boxPoints`<br>`contourArea`<br>`convexHull`<br>`convexityDefects`<br>`findContours`<br>`findContoursLinkRuns`<br>`fitEllipse`<br>`fitEllipseAMS`<br>`fitEllipseDirect`<br>`fitLine`<br>`getClosestEllipsePoints`<br>`intersectConvexConvex`<br>`isContourConvex`<br>`matchShapes`<br>`minAreaRect`<br>`minEnclosingCircle`<br>`minEnclosingConvexPolygon`<br>`minEnclosingTriangle`<br>`moments`<br>`pointPolygonTest`<br>`rotatedRectangleIntersection` | 26 | 轮廓提取是多边形、凸包、矩、拟合和包围几何的共同入口，内部依赖关系紧密。 |
| **合计** |  | **47** |  |
<!-- P2_IMGPROC_API_LIST_END -->

## 5. 第三阶段：高复杂度算法与长尾接口

第三阶段处理实现复杂度高、数值稳定性要求高、需要对象模型，或者相对低频的
算法。这些算子会复用前两阶段已经建立的矩阵、频域、轮廓、区域和几何能力。

### 5.1 Core 支持列表

<!-- P3_CORE_API_LIST_START -->
| 类别 | 算子 | 数量 | 放在第三阶段的原因 |
|---|---|---:|---|
| 高复杂度数值算法 | `eigen`<br>`eigenNonSymmetric`<br>`solveCubic`<br>`solvePoly`<br>`kmeans` | 5 | 对收敛性、数值稳定性和边界输入要求较高；kmeans 还会被 GrabCut 依赖。 |
| **合计** |  | **5** |  |
<!-- P3_CORE_API_LIST_END -->

### 5.2 Imgproc 支持列表

<!-- P3_IMGPROC_API_LIST_START -->
| 类别 | 算子 | 数量 | 放在第三阶段的原因 |
|---|---|---:|---|
| 绘制与文本 | `arrowedLine`<br>`circle`<br>`clipLine`<br>`drawContours`<br>`drawMarker`<br>`ellipse`<br>`ellipse2Poly`<br>`fillConvexPoly`<br>`fillPoly`<br>`getFontScaleFromHeight`<br>`getTextSize`<br>`line`<br>`polylines`<br>`putText`<br>`rectangle` | 15 | 对核心预处理性能影响较小，但需要共享光栅化、裁剪、填充和字体能力。 |
| Hough 与检测器 | `HoughCircles`<br>`HoughLines`<br>`HoughLinesP`<br>`HoughLinesPointSet`<br>`createGeneralizedHoughBallard`<br>`createGeneralizedHoughGuil`<br>`createLineSegmentDetector` | 7 | 依赖 Canny、排序、几何类型和累加器，并涉及较复杂的检测器对象接口。 |
| 分割与专用算法 | `EMD`<br>`grabCut`<br>`pyrMeanShiftFiltering`<br>`watershed` | 4 | 算法复杂、工作区大、依赖链长；GrabCut 依赖 kmeans，均适合在基础能力稳定后支持。 |
| **合计** |  | **26** |  |
<!-- P3_IMGPROC_API_LIST_END -->

## 6. 分期原则

| 原则 | 对应阶段 |
|---|---|
| 高频、通用、可被大量其他算子复用 | 第一阶段 |
| 依赖基础算子，面向分析、特征、匹配和形状计算 | 第二阶段 |
| 高复杂度、长依赖链、对象型接口或低频长尾能力 | 第三阶段 |

类成员 API、C API、`UMat`、CUDA、OpenCL、OpenGL、DirectX 等不在这份
三阶段算子清单中，其范围以
[opencv-core-imgproc-api-coverage.md](opencv-core-imgproc-api-coverage.md)
为准。
