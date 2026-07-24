# opencv-header-only vs OpenCV Upstream 性能对比（2026-07-24）

生成时间（UTC）：`2026-07-24 06:25:24Z`

## 当前项目状态

- `opencv-header-only` 当前公共定位是纯 header-only，不依赖项目内 `.cpp` 扩展层。
- Mode B 只比较当前 `cvh::headers_fast` 与同机编译的 upstream OpenCV；`cvh::headers_fast` 表示最快 header-only 构建配置。
- `cvh::headers_fast` 完整继承 `cvh::headers`。算子没有专用 fast-path 时继续执行继承的 header 实现并参与 benchmark，不因缺少 SIMD 特化而跳过。
- Core/Imgproc 第一阶段完成后，名称级可调用覆盖为 `107/220`：Core `57/97`，Imgproc `50/123`。
- Core 的 `add/subtract/multiply/divide/transpose/GEMM` 已迁入 ODR-safe headers；本报告通过公共 API 测量，不链接 legacy core 对象。
- OpenCV Universal Intrinsics 是默认 SIMD 方言，kernel 直接使用 OpenCV UI；项目已移除 xsimd 性能路径。
- ARM 当前关注 NEON，本次实测平台为 Apple ARM；x86 目标是 SSE/AVX 系列，RVV 因 scalable vector 设计问题暂缓。
- Imgproc legacy `.cpp` fast-path 已迁入 ODR-safe detail headers；resize/cvtColor UI、filter、LUT、border、Sobel、Canny 和 morphology 均从公共 header API 进入。
- 第一阶段新增的 `79` 个操作族已全部进入 Mode B，本报告包含 `92` 个 P1 性能 case。
- `full` profile 覆盖代表性的 `CV_8U` / `CV_32F`、C1/C3/C4、尺寸、布局与非连续 ROI 扩展。

## 第一阶段新增算子

本节记录第一阶段相对原有覆盖新增的 API 操作族。API 已实现不等于已经进入本次 Mode B 性能矩阵；只有建立了同输入、同参数 OpenCV 对照 case 的算子才计为“本报告实测”。

| 模块 | 第一阶段新增 | 本报告实测 | 已实现但本报告未测 |
| --- | ---: | ---: | ---: |
| Core | 43 | 43 | 0 |
| Imgproc | 36 | 36 | 0 |
| **合计** | **79** | **79** | **0** |

| 模块/类别 | 新增操作族 | 数量 | 本报告 Mode B 状态 |
| --- | --- | ---: | --- |
| Core：逐元素与逻辑 | `absdiff`、`bitwise_and`、`bitwise_not`、`bitwise_or`、`bitwise_xor`、`inRange`、`min`、`max` | 8 | 8/8 已实测 |
| Core：转换、数学与校验 | `scaleAdd`、`convertScaleAbs`、`convertFp16`、`sqrt`、`pow`、`exp`、`log`、`checkRange`、`patchNaNs` | 9 | 9/9 已实测 |
| Core：归约与统计 | `norm`、`sum`、`mean`、`meanStdDev`、`countNonZero`、`hasNonZero`、`findNonZero`、`minMaxIdx`、`minMaxLoc`、`reduce`、`reduceArgMax`、`reduceArgMin`、`normalize` | 13 | 13/13 已实测 |
| Core：布局、复制与通道 | `borderInterpolate`、`copyTo`、`extractChannel`、`insertChannel`、`mixChannels`、`flip`、`flipND`、`rotate`、`repeat`、`hconcat`、`vconcat`、`broadcast`、`swap` | 13 | 13/13 已实测 |
| Imgproc：核、滤波与强度 | `getStructuringElement`、`getGaussianKernel`、`getDerivKernels`、`getGaborKernel`、`createHanningWindow`、`integral`、`Scharr`、`Laplacian`、`spatialGradient`、`sqrBoxFilter`、`medianBlur`、`bilateralFilter`、`stackBlur`、`adaptiveThreshold`、`thresholdWithMask`、`equalizeHist`、`applyColorMap` | 17 | 17/17 已实测 |
| Imgproc：累积、金字塔与颜色 | `accumulate`、`accumulateProduct`、`accumulateSquare`、`accumulateWeighted`、`blendLinear`、`pyrDown`、`pyrUp`、`buildPyramid`、`cvtColorTwoPlane`、`demosaicing` | 10 | 10/10 已实测 |
| Imgproc：几何变换 | `remap`、`convertMaps`、`warpPerspective`、`getAffineTransform`、`getPerspectiveTransform`、`getRotationMatrix2D`、`getRotationMatrix2D_`、`invertAffineTransform`、`getRectSubPix` | 9 | 9/9 已实测 |

后续表中的 `ADD`、`GEMM`、`resize`、`cvtColor` 等仍是既有算子基线；带有 `P1 新增` 标记的行是本轮新增并已进入性能对比的算子。

## 高层优化结构

| 层次 | 当前实现 | 本报告中的含义 |
| --- | --- | --- |
| 公共 API | OpenCV-compatible header API | 所有 case 均从 `cvh::headers_fast` 公共入口调用 |
| SIMD 方言 | OpenCV Universal Intrinsics | 在 Apple ARM 上映射到 NEON |
| 专用 kernel | `cvtColor`、特定 `resize` UI kernel | 记录为 `dispatch_path=opencv_ui`；core 计算当前仍为 baseline |
| Header fast-path | 行并行 filter、LUT、border、Sobel、Canny、morphology | 记录为 `dispatch_path=header_fastpath` |
| 通用实现 | `cvh::headers` 中的 header baseline | 无专用 fast-path 时自动继承，记录为 `headers_baseline` 或 `public_header_scalar` |
| 对照实现 | upstream OpenCV `core` / `imgproc` | 相同输入、尺寸、border 和线程配置 |

## 运行配置

- Profile：`full`
- CVH 实现：`cvh_headers_fast`
- 采样：`warmup=1, iters=10, repeats=3`
- 线程数：`1`
- OpenMP：`dynamic=false, proc_bind=close`
- 主机：`Darwin arm64`
- CPU：`Apple M5`
- 编译器：`Apple clang version 21.0.0 (clang-2100.0.123.102)`
- 构建类型：`Release`
- CVH commit：`d575a4d64ac04d024c04df03fe019ae7b3602055` + dirty
- OpenCV：`4.14.0`，commit `d48bf69f65444a13f8a34b8982b083c1b78fa0e8` + dirty
- 原始数据：`2026-07-24-opencv-upstream-performance.csv`；元数据：`2026-07-24-opencv-upstream-performance.csv.meta.json`

## 汇总

- 总 case：`321`；有效：`320`；不支持：`1`。
- `OpenCV/CVH` 几何平均：`0.1861`；中位数：`0.2550`。
- CVH 更快：`28` 个；OpenCV 更快或相当：`292` 个。

| Suite | Cases | 几何平均 OpenCV/CVH | 中位数 | CVH 更快 | OpenCV 更快/相当 |
| --- | --- | --- | --- | --- | --- |
| core_mat | 153 | 0.1572 | 0.2503 | 9 | 144 |
| imgproc | 167 | 0.2171 | 0.2800 | 19 | 148 |

## 算子级概览

### `core_mat`

| Op | 阶段 | CVH dispatch | Cases | 几何平均 OpenCV/CVH | 领先方 |
| --- | --- | --- | --- | --- | --- |
| ABSDIFF | P1 新增 | public_header_baseline | 1 | 0.0773 | OpenCV `12.94x` |
| ADD | 既有 | headers_baseline | 16 | 0.1801 | OpenCV `5.55x` |
| BITWISE_AND | P1 新增 | public_header_baseline | 1 | 0.0765 | OpenCV `13.07x` |
| BITWISE_NOT | P1 新增 | public_header_baseline | 1 | 0.0505 | OpenCV `19.78x` |
| BITWISE_OR | P1 新增 | public_header_baseline | 1 | 0.0768 | OpenCV `13.02x` |
| BITWISE_XOR | P1 新增 | public_header_baseline | 1 | 0.0778 | OpenCV `12.86x` |
| BORDER_INTERPOLATE | P1 新增 | public_header_baseline | 1 | 0.8471 | OpenCV `1.18x` |
| BROADCAST | P1 新增 | public_header_baseline | 1 | 0.0003 | OpenCV `3773.58x` |
| CHECK_RANGE | P1 新增 | public_header_baseline | 1 | 0.6559 | OpenCV `1.52x` |
| CONVERT_FP16 | P1 新增 | public_header_baseline | 1 | 0.2503 | OpenCV `4.00x` |
| CONVERT_SCALE_ABS | P1 新增 | public_header_baseline | 1 | 0.1389 | OpenCV `7.20x` |
| COPY_TO | P1 新增 | public_header_baseline | 1 | 0.0063 | OpenCV `158.03x` |
| COUNT_NON_ZERO | P1 新增 | public_header_baseline | 1 | 0.0340 | OpenCV `29.37x` |
| DIVIDE | 既有 | headers_baseline | 16 | 0.3407 | OpenCV `2.94x` |
| EXP | P1 新增 | public_header_baseline | 1 | 0.2260 | OpenCV `4.42x` |
| EXTRACT_CHANNEL | P1 新增 | public_header_baseline | 1 | 0.0174 | OpenCV `57.57x` |
| FIND_NON_ZERO | P1 新增 | public_header_baseline | 1 | 0.3632 | OpenCV `2.75x` |
| FLIP | P1 新增 | public_header_baseline | 1 | 0.0048 | OpenCV `208.25x` |
| FLIP_ND | P1 新增 | public_header_baseline | 1 | 0.0105 | OpenCV `95.11x` |
| GEMM | 既有 | headers_baseline | 6 | 0.0128 | OpenCV `78.01x` |
| HAS_NON_ZERO | P1 新增 | public_header_baseline | 1 | 0.0000 | OpenCV `32258.06x` |
| HCONCAT | P1 新增 | public_header_baseline | 1 | 0.5058 | OpenCV `1.98x` |
| INSERT_CHANNEL | P1 新增 | public_header_baseline | 1 | 0.0175 | OpenCV `57.02x` |
| IN_RANGE | P1 新增 | public_header_baseline | 1 | 0.0755 | OpenCV `13.25x` |
| LOG | P1 新增 | public_header_baseline | 1 | 0.2850 | OpenCV `3.51x` |
| MAT_CLONE | 既有 | headers_baseline | 4 | 0.9565 | OpenCV `1.05x` |
| MAT_CONVERTTO | 既有 | headers_baseline | 4 | 0.9743 | OpenCV `1.03x` |
| MAT_COPYTO | 既有 | headers_baseline | 4 | 1.0029 | CVH `1.00x` |
| MAT_CREATE | 既有 | headers_baseline | 4 | 0.0749 | OpenCV `13.35x` |
| MAT_RESHAPE | 既有 | headers_baseline | 4 | 0.3413 | OpenCV `2.93x` |
| MAT_SETTO | 既有 | headers_baseline | 4 | 0.0132 | OpenCV `75.99x` |
| MAX | P1 新增 | public_header_baseline | 1 | 0.1089 | OpenCV `9.19x` |
| MEAN | P1 新增 | public_header_baseline | 1 | 0.2248 | OpenCV `4.45x` |
| MEAN_STD_DEV | P1 新增 | public_header_baseline | 1 | 0.1187 | OpenCV `8.42x` |
| MIN | P1 新增 | public_header_baseline | 1 | 0.1086 | OpenCV `9.20x` |
| MIN_MAX_IDX | P1 新增 | public_header_baseline | 1 | 0.0636 | OpenCV `15.73x` |
| MIN_MAX_LOC | P1 新增 | public_header_baseline | 1 | 0.0638 | OpenCV `15.68x` |
| MIX_CHANNELS | P1 新增 | public_header_baseline | 1 | 0.0176 | OpenCV `56.72x` |
| MULTIPLY | 既有 | headers_baseline | 16 | 0.1863 | OpenCV `5.37x` |
| NORM | P1 新增 | public_header_baseline | 1 | 0.0783 | OpenCV `12.77x` |
| NORMALIZE | P1 新增 | public_header_baseline | 1 | 0.0540 | OpenCV `18.52x` |
| PATCH_NANS | P1 新增 | public_header_baseline | 1 | 0.4350 | OpenCV `2.30x` |
| POW | P1 新增 | public_header_baseline | 1 | 0.2138 | OpenCV `4.68x` |
| REDUCE | P1 新增 | public_header_baseline | 1 | 0.0219 | OpenCV `45.77x` |
| REDUCE_ARG_MAX | P1 新增 | public_header_baseline | 1 | 0.1655 | OpenCV `6.04x` |
| REDUCE_ARG_MIN | P1 新增 | public_header_baseline | 1 | 0.1654 | OpenCV `6.04x` |
| REPEAT | P1 新增 | public_header_baseline | 1 | 0.0014 | OpenCV `693.00x` |
| ROTATE | P1 新增 | public_header_baseline | 1 | 0.0286 | OpenCV `35.00x` |
| SCALE_ADD | P1 新增 | public_header_baseline | 1 | 0.8208 | OpenCV `1.22x` |
| SQRT | P1 新增 | public_header_baseline | 1 | 0.9561 | OpenCV `1.05x` |
| SUBTRACT | 既有 | headers_baseline | 16 | 0.1897 | OpenCV `5.27x` |
| SUM | P1 新增 | public_header_baseline | 1 | 0.2274 | OpenCV `4.40x` |
| SWAP | P1 新增 | public_header_baseline | 1 | 0.2061 | OpenCV `4.85x` |
| TRANSPOSE | 既有 | headers_baseline | 16 | 0.5212 | OpenCV `1.92x` |
| VCONCAT | P1 新增 | public_header_baseline | 1 | 0.5459 | OpenCV `1.83x` |

### `imgproc`

| Op | 阶段 | CVH dispatch | Cases | 几何平均 OpenCV/CVH | 领先方 |
| --- | --- | --- | --- | --- | --- |
| ACCUMULATE | P1 新增 | public_header_baseline | 1 | 0.0784 | OpenCV `12.75x` |
| ACCUMULATE_PRODUCT | P1 新增 | public_header_baseline | 1 | 0.0721 | OpenCV `13.88x` |
| ACCUMULATE_SQUARE | P1 新增 | public_header_baseline | 1 | 0.0699 | OpenCV `14.31x` |
| ACCUMULATE_WEIGHTED | P1 新增 | public_header_baseline | 1 | 0.0696 | OpenCV `14.38x` |
| ADAPTIVE_THRESHOLD | P1 新增 | public_header_baseline | 1 | 0.4985 | OpenCV `2.01x` |
| APPLY_COLOR_MAP | P1 新增 | public_header_baseline | 1 | 0.3253 | OpenCV `3.07x` |
| BILATERAL_FILTER | P1 新增 | public_header_baseline | 1 | 0.0388 | OpenCV `25.75x` |
| BLEND_LINEAR | P1 新增 | public_header_baseline | 1 | 0.3962 | OpenCV `2.52x` |
| BOX_FILTER | 既有 | box3x3, header_fastpath | 10 | 0.2939 | OpenCV `3.40x` |
| BUILD_PYRAMID | P1 新增 | public_header_baseline | 1 | 0.0065 | OpenCV `154.80x` |
| CANNY | 既有 | header_fastpath | 4 | 0.9619 | OpenCV `1.04x` |
| CONVERT_MAPS | P1 新增 | public_header_baseline | 1 | 0.0047 | OpenCV `212.72x` |
| COPY_MAKE_BORDER | 既有 | header_fastpath | 9 | 0.3795 | OpenCV `2.64x` |
| CREATE_HANNING_WINDOW | P1 新增 | public_header_baseline | 1 | 0.0292 | OpenCV `34.29x` |
| CVTCOLOR | 既有 | header_fastpath, opencv_ui | 17 | 0.5589 | OpenCV `1.79x` |
| CVT_COLOR_TWO_PLANE | P1 新增 | public_header_baseline | 1 | 0.1600 | OpenCV `6.25x` |
| DEMOSAICING | P1 新增 | public_header_baseline | 1 | 0.0017 | OpenCV `576.37x` |
| DILATE | 既有 | header_fastpath | 6 | 0.1140 | OpenCV `8.78x` |
| EQUALIZE_HIST | P1 新增 | public_header_baseline | 1 | 0.4758 | OpenCV `2.10x` |
| ERODE | 既有 | header_fastpath | 6 | 0.1159 | OpenCV `8.63x` |
| FILTER2D | 既有 | header_fastpath | 10 | 0.3600 | OpenCV `2.78x` |
| GAUSSIAN | 既有 | gauss_separable, header_fastpath | 10 | 0.2772 | OpenCV `3.61x` |
| GET_AFFINE_TRANSFORM | P1 新增 | public_header_baseline | 1 | 2.0637 | CVH `2.06x` |
| GET_DERIV_KERNELS | P1 新增 | public_header_baseline | 1 | 0.4093 | OpenCV `2.44x` |
| GET_GABOR_KERNEL | P1 新增 | public_header_baseline | 1 | 0.3285 | OpenCV `3.04x` |
| GET_GAUSSIAN_KERNEL | P1 新增 | public_header_baseline | 1 | 4.0344 | CVH `4.03x` |
| GET_PERSPECTIVE_TRANSFORM | P1 新增 | public_header_baseline | 1 | 2.4557 | CVH `2.46x` |
| GET_RECT_SUB_PIX | P1 新增 | public_header_scalar | 4 | 0.0218 | OpenCV `45.85x` |
| GET_ROTATION_MATRIX_2D | P1 新增 | public_header_baseline | 1 | 1.1382 | CVH `1.14x` |
| GET_ROTATION_MATRIX_2D_ | P1 新增 | public_header_baseline | 1 | 1.1270 | CVH `1.13x` |
| GET_STRUCTURING_ELEMENT | P1 新增 | public_header_baseline | 1 | 0.2004 | OpenCV `4.99x` |
| INTEGRAL | P1 新增 | public_header_baseline | 1 | 0.0269 | OpenCV `37.22x` |
| INVERT_AFFINE_TRANSFORM | P1 新增 | public_header_baseline | 1 | 0.4741 | OpenCV `2.11x` |
| LAPLACIAN | P1 新增 | public_header_baseline | 1 | 0.0119 | OpenCV `83.77x` |
| LUT | 既有 | header_fastpath | 6 | 0.6255 | OpenCV `1.60x` |
| MEDIAN_BLUR | P1 新增 | public_header_baseline | 1 | 0.0049 | OpenCV `202.59x` |
| PYR_DOWN | P1 新增 | public_header_baseline | 1 | 0.0077 | OpenCV `129.08x` |
| PYR_UP | P1 新增 | public_header_baseline | 1 | 0.0071 | OpenCV `140.10x` |
| REMAP | P1 新增 | public_header_scalar | 8 | 0.0503 | OpenCV `19.88x` |
| RESIZE | 既有 | header_fastpath, headers_baseline, opencv_ui | 10 | 0.6610 | OpenCV `1.51x` |
| SCHARR | P1 新增 | public_header_baseline | 1 | 0.0116 | OpenCV `86.13x` |
| SEP_FILTER2D | 既有 | header_fastpath | 10 | 0.4908 | OpenCV `2.04x` |
| SOBEL | 既有 | header_fastpath | 6 | 1.5119 | CVH `1.51x` |
| SPATIAL_GRADIENT | P1 新增 | public_header_baseline | 1 | 0.1311 | OpenCV `7.63x` |
| SQR_BOX_FILTER | P1 新增 | public_header_baseline | 1 | 0.0211 | OpenCV `47.40x` |
| STACK_BLUR | P1 新增 | public_header_baseline | 1 | 0.0786 | OpenCV `12.72x` |
| THRESHOLD | 既有 | header_fastpath, headers_baseline | 5 | 0.0384 | OpenCV `26.02x` |
| THRESHOLD_WITH_MASK | P1 新增 | public_header_baseline | 1 | 1.0048 | CVH `1.00x` |
| WARP_AFFINE | 既有 | headers_baseline | 9 | 0.0866 | OpenCV `11.55x` |
| WARP_PERSPECTIVE | P1 新增 | public_header_scalar | 4 | 0.0953 | OpenCV `10.50x` |

## 详细结果

### `core_mat`

| Op | 阶段 | Variant | CVH dispatch | Depth | Ch | Layout | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ABSDIFF | P1 新增 | mat_mat_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 0.334842 | 0.025879 | 0.0773 | phase1_representative_case |
| ADD | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 1080x1920 | 0.657708 | 0.218079 | 0.3316 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 479x641 | 0.092412 | 0.026558 | 0.2874 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 480x640 | 0.095300 | 0.029579 | 0.3104 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 720x1280 | 0.286713 | 0.097142 | 0.3388 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 1080x1920 | 1.212100 | 0.642138 | 0.5298 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 479x641 | 0.168517 | 0.088938 | 0.5278 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 480x640 | 0.166196 | 0.089392 | 0.5379 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 720x1280 | 0.518483 | 0.272900 | 0.5263 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.789154 | 0.053304 | 0.0675 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.117725 | 0.007092 | 0.0602 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.119900 | 0.007154 | 0.0597 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.358392 | 0.016704 | 0.0466 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 1080x1920 | 1.393200 | 0.146600 | 0.1052 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 479x641 | 0.201767 | 0.022071 | 0.1094 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 480x640 | 0.206154 | 0.022408 | 0.1087 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 720x1280 | 0.650192 | 0.070958 | 0.1091 | correctness=upstream_pass |
| BITWISE_AND | P1 新增 | mat_mat_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 0.335146 | 0.025638 | 0.0765 | phase1_representative_case |
| BITWISE_NOT | P1 新增 | u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 0.334079 | 0.016887 | 0.0505 | phase1_representative_case |
| BITWISE_OR | P1 新增 | mat_mat_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 0.337129 | 0.025883 | 0.0768 | phase1_representative_case |
| BITWISE_XOR | P1 新增 | mat_mat_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 0.333725 | 0.025958 | 0.0778 | phase1_representative_case |
| BORDER_INTERPOLATE | P1 新增 | reflect101_batch4096 | public_header_baseline | S32 | 1 | continuous | micro_batch | 0.006284 | 0.005324 | 0.8471 | phase1_representative_case;micro_iterations=10000 |
| BROADCAST | P1 新增 | row_to_image_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 11.918167 | 0.003158 | 0.0003 | phase1_representative_case |
| CHECK_RANGE | P1 新增 | quiet_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.164554 | 0.107925 | 0.6559 | phase1_representative_case |
| CONVERT_FP16 | P1 新增 | f32c1_to_fp16 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.079479 | 0.019892 | 0.2503 | phase1_representative_case |
| CONVERT_SCALE_ABS | P1 新增 | f32c3_to_u8c3 | public_header_baseline | CV_32F | 3 | continuous | 480x640 | 0.484071 | 0.067221 | 0.1389 | phase1_representative_case |
| COPY_TO | P1 新增 | masked_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 3.513404 | 0.022233 | 0.0063 | phase1_representative_case |
| COUNT_NON_ZERO | P1 新增 | u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.270354 | 0.009204 | 0.0340 | phase1_representative_case |
| DIVIDE | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 1080x1920 | 0.630850 | 0.206921 | 0.3280 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 479x641 | 0.092762 | 0.029779 | 0.3210 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 480x640 | 0.091546 | 0.031796 | 0.3473 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 720x1280 | 0.279192 | 0.095037 | 0.3404 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 1080x1920 | 1.517521 | 0.619050 | 0.4079 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 479x641 | 0.220408 | 0.090467 | 0.4105 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 480x640 | 0.222121 | 0.092463 | 0.4163 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 720x1280 | 0.675104 | 0.269767 | 0.3996 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 1.802104 | 0.457492 | 0.2539 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.261250 | 0.068533 | 0.2623 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.271825 | 0.073550 | 0.2706 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.781108 | 0.200058 | 0.2561 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 1080x1920 | 3.593492 | 1.375738 | 0.3828 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 479x641 | 0.528104 | 0.200775 | 0.3802 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 480x640 | 0.536737 | 0.200750 | 0.3740 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 720x1280 | 1.605150 | 0.606487 | 0.3778 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| EXP | P1 新增 | bounded_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.359358 | 0.081217 | 0.2260 | phase1_representative_case |
| EXTRACT_CHANNEL | P1 新增 | channel1_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 2.433950 | 0.042279 | 0.0174 | phase1_representative_case |
| FIND_NON_ZERO | P1 新增 | u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.575646 | 0.209063 | 0.3632 | phase1_representative_case |
| FLIP | P1 新增 | horizontal_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 3.435121 | 0.016496 | 0.0048 | phase1_representative_case |
| FLIP_ND | P1 新增 | axis1_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 12.129483 | 0.127525 | 0.0105 | phase1_representative_case |
| GEMM | 既有 | fp32_nn_end_to_end | headers_baseline | CV_32F | 1 | continuous | 128x128x128 | 0.210391 | 0.003859 | 0.0183 | correctness=upstream_pass;iters=8 |
| GEMM | 既有 | fp32_nn_end_to_end | headers_baseline | CV_32F | 1 | continuous | 256x256x256 | 1.962875 | 0.025459 | 0.0130 | correctness=upstream_pass;iters=1 |
| GEMM | 既有 | fp32_nn_end_to_end | headers_baseline | CV_32F | 1 | continuous | 512x512x512 | 17.341708 | 0.163084 | 0.0094 | correctness=upstream_pass;iters=1 |
| GEMM | 既有 | fp32_nn_pack_once | headers_baseline | CV_32F | 1 | continuous | 128x128x128 | 0.225016 | 0.003776 | 0.0168 | correctness=upstream_pass;opencv_reuses_B_without_public_pack_handle;iters=8 |
| GEMM | 既有 | fp32_nn_pack_once | headers_baseline | CV_32F | 1 | continuous | 256x256x256 | 2.089958 | 0.025292 | 0.0121 | correctness=upstream_pass;opencv_reuses_B_without_public_pack_handle;iters=1 |
| GEMM | 既有 | fp32_nn_pack_once | headers_baseline | CV_32F | 1 | continuous | 512x512x512 | 17.159542 | 0.167542 | 0.0098 | correctness=upstream_pass;opencv_reuses_B_without_public_pack_handle;iters=1 |
| HAS_NON_ZERO | P1 新增 | u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.270879 | 0.000008 | 0.0000 | phase1_representative_case |
| HCONCAT | P1 新增 | two_halves_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.012317 | 0.006229 | 0.5058 | phase1_representative_case |
| INSERT_CHANNEL | P1 新增 | channel1_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 2.454079 | 0.043037 | 0.0175 | phase1_representative_case |
| IN_RANGE | P1 新增 | scalar_bounds_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 1.596029 | 0.120463 | 0.0755 | phase1_representative_case |
| LOG | P1 新增 | positive_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.479333 | 0.136612 | 0.2850 | phase1_representative_case |
| MAT_CLONE | 既有 | full_copy | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.025225 | 0.024925 | 0.9881 |  |
| MAT_CLONE | 既有 | full_copy | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.003837 | 0.003712 | 0.9674 |  |
| MAT_CLONE | 既有 | full_copy | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.003967 | 0.003721 | 0.9380 |  |
| MAT_CLONE | 既有 | full_copy | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.011450 | 0.010687 | 0.9334 |  |
| MAT_CONVERTTO | 既有 | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.081162 | 0.079608 | 0.9809 |  |
| MAT_CONVERTTO | 既有 | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.013042 | 0.013292 | 1.0192 |  |
| MAT_CONVERTTO | 既有 | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.013300 | 0.012071 | 0.9076 |  |
| MAT_CONVERTTO | 既有 | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.035467 | 0.035229 | 0.9933 |  |
| MAT_COPYTO | 既有 | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.024213 | 0.024329 | 1.0048 |  |
| MAT_COPYTO | 既有 | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.003646 | 0.003463 | 0.9497 |  |
| MAT_COPYTO | 既有 | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.003383 | 0.003454 | 1.0210 |  |
| MAT_COPYTO | 既有 | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.010233 | 0.010625 | 1.0383 |  |
| MAT_CREATE | 既有 | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.000016 | 0.000001 | 0.0741 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | 既有 | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.000016 | 0.000001 | 0.0746 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | 既有 | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.000015 | 0.000001 | 0.0738 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | 既有 | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.000015 | 0.000001 | 0.0771 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_RESHAPE | 既有 | to_column_view | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.000046 | 0.000016 | 0.3566 | micro_iters_x1000 |
| MAT_RESHAPE | 既有 | to_column_view | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.000044 | 0.000015 | 0.3495 | micro_iters_x1000 |
| MAT_RESHAPE | 既有 | to_column_view | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.000048 | 0.000016 | 0.3279 | micro_iters_x1000 |
| MAT_RESHAPE | 既有 | to_column_view | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.000047 | 0.000016 | 0.3318 | micro_iters_x1000 |
| MAT_SETTO | 既有 | scalar_all | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 1.994938 | 0.024304 | 0.0122 |  |
| MAT_SETTO | 既有 | scalar_all | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.325175 | 0.004379 | 0.0135 |  |
| MAT_SETTO | 既有 | scalar_all | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.292946 | 0.004179 | 0.0143 |  |
| MAT_SETTO | 既有 | scalar_all | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.873658 | 0.011192 | 0.0128 |  |
| MAX | P1 新增 | mat_mat_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 0.227712 | 0.024788 | 0.1089 | phase1_representative_case |
| MEAN | P1 新增 | f32c3 | public_header_baseline | CV_32F | 3 | continuous | 480x640 | 0.895779 | 0.201396 | 0.2248 | phase1_representative_case |
| MEAN_STD_DEV | P1 新增 | f32c3 | public_header_baseline | CV_32F | 3 | continuous | 480x640 | 1.160900 | 0.137800 | 0.1187 | phase1_representative_case |
| MIN | P1 新增 | mat_mat_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 0.228587 | 0.024833 | 0.1086 | phase1_representative_case |
| MIN_MAX_IDX | P1 新增 | f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.538996 | 0.034267 | 0.0636 | phase1_representative_case |
| MIN_MAX_LOC | P1 新增 | f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.538992 | 0.034371 | 0.0638 | phase1_representative_case |
| MIX_CHANNELS | P1 新增 | reverse_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 7.343867 | 0.129471 | 0.0176 | phase1_representative_case |
| MULTIPLY | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 1080x1920 | 0.629846 | 0.204500 | 0.3247 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 479x641 | 0.094058 | 0.029371 | 0.3123 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 480x640 | 0.092225 | 0.029658 | 0.3216 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 720x1280 | 0.305463 | 0.095962 | 0.3142 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 1080x1920 | 1.148462 | 0.649088 | 0.5652 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 479x641 | 0.166250 | 0.089883 | 0.5407 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 480x640 | 0.171825 | 0.092688 | 0.5394 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 720x1280 | 0.495683 | 0.296588 | 0.5983 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.800192 | 0.050775 | 0.0635 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.117779 | 0.007483 | 0.0635 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.118367 | 0.007175 | 0.0606 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.360642 | 0.022371 | 0.0620 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 1080x1920 | 1.380650 | 0.150075 | 0.1087 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 479x641 | 0.201921 | 0.022067 | 0.1093 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 480x640 | 0.205725 | 0.021917 | 0.1065 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 720x1280 | 0.613650 | 0.066933 | 0.1091 | correctness=upstream_pass |
| NORM | P1 新增 | l2_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.478996 | 0.037504 | 0.0783 | phase1_representative_case |
| NORMALIZE | P1 新增 | l2_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 1.023975 | 0.055283 | 0.0540 | phase1_representative_case |
| PATCH_NANS | P1 新增 | one_nan_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.045629 | 0.019850 | 0.4350 | phase1_representative_case |
| POW | P1 新增 | power_1_75_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 1.466950 | 0.313679 | 0.2138 | phase1_representative_case |
| REDUCE | P1 新增 | axis0_sum_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.669533 | 0.014629 | 0.0219 | phase1_representative_case |
| REDUCE_ARG_MAX | P1 新增 | axis0_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.882783 | 0.146138 | 0.1655 | phase1_representative_case |
| REDUCE_ARG_MIN | P1 新增 | axis0_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.870058 | 0.143937 | 0.1654 | phase1_representative_case |
| REPEAT | P1 新增 | two_by_two_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 3.404817 | 0.004913 | 0.0014 | phase1_representative_case |
| ROTATE | P1 新增 | clockwise90_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 3.434646 | 0.098138 | 0.0286 | phase1_representative_case |
| SCALE_ADD | P1 新增 | f32c3 | public_header_baseline | CV_32F | 3 | continuous | 480x640 | 0.120642 | 0.099021 | 0.8208 | phase1_representative_case |
| SQRT | P1 新增 | positive_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.041500 | 0.039679 | 0.9561 | phase1_representative_case |
| SUBTRACT | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 1080x1920 | 0.666021 | 0.207325 | 0.3113 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 479x641 | 0.091800 | 0.029712 | 0.3237 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 480x640 | 0.096308 | 0.029692 | 0.3083 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 720x1280 | 0.289596 | 0.096521 | 0.3333 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 1080x1920 | 1.144338 | 0.614788 | 0.5372 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 479x641 | 0.167583 | 0.089250 | 0.5326 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 480x640 | 0.170333 | 0.091146 | 0.5351 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 720x1280 | 0.507142 | 0.272571 | 0.5375 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.767983 | 0.048683 | 0.0634 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.110721 | 0.007225 | 0.0653 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.114296 | 0.007112 | 0.0622 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.341583 | 0.021421 | 0.0627 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 1080x1920 | 1.262375 | 0.149383 | 0.1183 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 479x641 | 0.182617 | 0.022383 | 0.1226 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 480x640 | 0.185742 | 0.022400 | 0.1206 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 720x1280 | 0.587037 | 0.068904 | 0.1174 | correctness=upstream_pass |
| SUM | P1 新增 | f32c3 | public_header_baseline | CV_32F | 3 | continuous | 480x640 | 0.886092 | 0.201521 | 0.2274 | phase1_representative_case |
| SWAP | P1 新增 | mat_headers | public_header_baseline | CV_8U | 1 | continuous | micro_batch | 0.000024 | 0.000005 | 0.2061 | phase1_representative_case;micro_iterations=10000 |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_32F | 1 | continuous | 1080x1920 | 0.740729 | 0.638462 | 0.8619 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_32F | 1 | continuous | 479x641 | 0.179979 | 0.031442 | 0.1747 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_32F | 1 | continuous | 480x640 | 0.178050 | 0.070729 | 0.3972 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_32F | 1 | continuous | 720x1280 | 0.400196 | 0.307362 | 0.7680 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_32F | 3 | continuous | 1080x1920 | 0.682250 | 1.575937 | 2.3099 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_32F | 3 | continuous | 479x641 | 0.126683 | 0.117254 | 0.9256 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_32F | 3 | continuous | 480x640 | 0.122592 | 0.150308 | 1.2261 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_32F | 3 | continuous | 720x1280 | 0.267629 | 0.548183 | 2.0483 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.557158 | 0.126421 | 0.2269 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.141683 | 0.009508 | 0.0671 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.137992 | 0.006792 | 0.0492 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.286425 | 0.028879 | 0.1008 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_8U | 3 | continuous | 1080x1920 | 0.409550 | 0.756317 | 1.8467 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_8U | 3 | continuous | 479x641 | 0.116658 | 0.083554 | 0.7162 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_8U | 3 | continuous | 480x640 | 0.119242 | 0.082025 | 0.6879 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_8U | 3 | continuous | 720x1280 | 0.224354 | 0.392042 | 1.7474 | correctness=upstream_pass |
| VCONCAT | P1 新增 | two_halves_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.007129 | 0.003892 | 0.5459 | phase1_representative_case |

### `imgproc`

| Op | 阶段 | Variant | CVH dispatch | Depth | Ch | Layout | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACCUMULATE | P1 新增 | u8c1_to_f32 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.231133 | 0.018129 | 0.0784 | phase1_representative_case |
| ACCUMULATE_PRODUCT | P1 新增 | u8c1_to_f32 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.289479 | 0.020862 | 0.0721 | phase1_representative_case |
| ACCUMULATE_SQUARE | P1 新增 | u8c1_to_f32 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.258942 | 0.018092 | 0.0699 | phase1_representative_case |
| ACCUMULATE_WEIGHTED | P1 新增 | alpha0_1_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.290313 | 0.020192 | 0.0696 | phase1_representative_case |
| ADAPTIVE_THRESHOLD | P1 新增 | mean11_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.412604 | 0.205679 | 0.4985 | phase1_representative_case |
| APPLY_COLOR_MAP | P1 新增 | jet_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.659846 | 0.214642 | 0.3253 | phase1_representative_case |
| BILATERAL_FILTER | P1 新增 | d5_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 22.240875 | 0.863675 | 0.0388 | phase1_representative_case |
| BLEND_LINEAR | P1 新增 | u8c3_f32_weights | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 0.518850 | 0.205571 | 0.3962 | phase1_representative_case |
| BOX_FILTER | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 1.511075 | 0.287617 | 0.1903 |  |
| BOX_FILTER | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.268871 | 0.044321 | 0.1648 |  |
| BOX_FILTER | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.271842 | 0.044933 | 0.1653 |  |
| BOX_FILTER | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.675008 | 0.128117 | 0.1898 |  |
| BOX_FILTER | 既有 | 3x3_replicate_f32c1 | box3x3 | CV_32F | 1 | continuous | 480x640 | 0.174946 | 0.107075 | 0.6120 |  |
| BOX_FILTER | 既有 | 3x3_replicate_f32c3 | box3x3 | CV_32F | 3 | continuous | 480x640 | 0.329967 | 0.318404 | 0.9650 |  |
| BOX_FILTER | 既有 | 3x3_replicate_f32c4 | box3x3 | CV_32F | 4 | continuous | 480x640 | 0.387271 | 0.399242 | 1.0309 |  |
| BOX_FILTER | 既有 | 3x3_replicate_u8c3 | box3x3 | CV_8U | 3 | continuous | 480x640 | 0.661471 | 0.133396 | 0.2017 |  |
| BOX_FILTER | 既有 | 3x3_replicate_u8c3_roi | box3x3 | CV_8U | 3 | roi | 479x641 | 0.669887 | 0.127092 | 0.1897 |  |
| BOX_FILTER | 既有 | 3x3_replicate_u8c4 | box3x3 | CV_8U | 4 | continuous | 480x640 | 0.812583 | 0.170508 | 0.2098 |  |
| BUILD_PYRAMID | P1 新增 | levels3_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 2.991329 | 0.019325 | 0.0065 | phase1_representative_case |
| CANNY | 既有 | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 29.282267 | 28.046071 | 0.9578 |  |
| CANNY | 既有 | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 479x641 | 4.630792 | 4.432187 | 0.9571 |  |
| CANNY | 既有 | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 480x640 | 4.421337 | 4.350050 | 0.9839 |  |
| CANNY | 既有 | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 13.031875 | 12.367783 | 0.9490 |  |
| CONVERT_MAPS | P1 新增 | f32_pair_to_fixed | public_header_baseline | CV_32F | 2 | continuous | 480x640 | 12.610437 | 0.059288 | 0.0047 | phase1_representative_case |
| COPY_MAKE_BORDER | 既有 | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.073308 | 0.045658 | 0.6228 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.059775 | 0.006750 | 0.1129 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.060092 | 0.007013 | 0.1167 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.064021 | 0.020096 | 0.3139 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate_f32c1 | header_fastpath | CV_32F | 1 | continuous | 480x640 | 0.067562 | 0.026354 | 0.3901 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate_f32c3 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.088454 | 0.084317 | 0.9532 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate_f32c4 | header_fastpath | CV_32F | 4 | continuous | 480x640 | 0.082842 | 0.113475 | 1.3698 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.066750 | 0.021142 | 0.3167 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.067179 | 0.026371 | 0.3925 |  |
| CREATE_HANNING_WINDOW | P1 新增 | 64x64_f32 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.038579 | 0.001125 | 0.0292 | phase1_representative_case |
| CVTCOLOR | 既有 | BGR2BGRA_u8 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.112667 | 0.020008 | 0.1776 |  |
| CVTCOLOR | 既有 | BGR2GRAY_f32 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.079408 | 0.043633 | 0.5495 |  |
| CVTCOLOR | 既有 | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 1080x1920 | 0.203658 | 0.203396 | 0.9987 |  |
| CVTCOLOR | 既有 | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 479x641 | 0.030996 | 0.030175 | 0.9735 |  |
| CVTCOLOR | 既有 | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.030758 | 0.030771 | 1.0004 |  |
| CVTCOLOR | 既有 | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 720x1280 | 0.090583 | 0.092579 | 1.0220 |  |
| CVTCOLOR | 既有 | BGR2GRAY_u8_roi | opencv_ui | CV_8U | 3 | roi | 479x641 | 0.035196 | 0.035154 | 0.9988 |  |
| CVTCOLOR | 既有 | BGR2I420_u8 | header_fastpath | CV_8U | 3 | yuv420_i420 | 480x640 | 0.129412 | 0.061175 | 0.4727 |  |
| CVTCOLOR | 既有 | BGR2RGB_f32 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.086692 | 0.065687 | 0.7577 |  |
| CVTCOLOR | 既有 | BGR2RGB_u8 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.089050 | 0.016575 | 0.1861 |  |
| CVTCOLOR | 既有 | BGR2YUV_u8 | header_fastpath | CV_8U | 3 | yuv444_interleaved | 480x640 | 0.124071 | 0.080983 | 0.6527 |  |
| CVTCOLOR | 既有 | BGR2YUY2_u8 | header_fastpath | CV_8U | 3 | yuv422_yuy2 | 480x640 | 0.125646 | 0.066021 | 0.5255 |  |
| CVTCOLOR | 既有 | BGRA2GRAY_u8 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.092637 | 0.039829 | 0.4299 |  |
| CVTCOLOR | 既有 | I420_TO_BGR_u8 | header_fastpath | CV_8U | 1 | yuv420_i420 | 480x640 | 0.244638 | 0.077137 | 0.3153 |  |
| CVTCOLOR | 既有 | NV12_TO_BGR_u8 | header_fastpath | CV_8U | 1 | yuv420_nv12 | 480x640 | 0.129879 | 0.076429 | 0.5885 |  |
| CVTCOLOR | 既有 | YUV2BGR_u8 | header_fastpath | CV_8U | 3 | yuv444_interleaved | 480x640 | 0.119400 | 0.061267 | 0.5131 |  |
| CVTCOLOR | 既有 | YUY2_TO_BGR_u8 | header_fastpath | CV_8U | 2 | yuv422_yuy2 | 480x640 | 0.134133 | 0.074942 | 0.5587 |  |
| CVT_COLOR_TWO_PLANE | P1 新增 | nv12_to_bgr | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.424388 | 0.067908 | 0.1600 | phase1_representative_case |
| DEMOSAICING | P1 新增 | bayer_bg_to_bgr | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 28.189779 | 0.048921 | 0.0017 | phase1_representative_case |
| DILATE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 1.104162 | 0.146629 | 0.1328 |  |
| DILATE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.277796 | 0.023096 | 0.0831 |  |
| DILATE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.267892 | 0.022642 | 0.0845 |  |
| DILATE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.566538 | 0.070458 | 0.1244 |  |
| DILATE | 既有 | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.493221 | 0.065013 | 0.1318 |  |
| DILATE | 既有 | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.618217 | 0.088500 | 0.1432 |  |
| EQUALIZE_HIST | P1 新增 | u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.181188 | 0.086204 | 0.4758 | phase1_representative_case |
| ERODE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 1.126917 | 0.142883 | 0.1268 |  |
| ERODE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.270813 | 0.023308 | 0.0861 |  |
| ERODE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.266946 | 0.023658 | 0.0886 |  |
| ERODE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.557646 | 0.066804 | 0.1198 |  |
| ERODE | 既有 | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.491558 | 0.068317 | 0.1390 |  |
| ERODE | 既有 | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.601067 | 0.090517 | 0.1506 |  |
| FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 2.483150 | 0.624292 | 0.2514 |  |
| FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.426446 | 0.098471 | 0.2309 |  |
| FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.400525 | 0.094537 | 0.2360 |  |
| FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 1.075004 | 0.275450 | 0.2562 |  |
| FILTER2D | 既有 | 3x3_replicate_f32c1 | header_fastpath | CV_32F | 1 | continuous | 480x640 | 0.412021 | 0.073779 | 0.1791 |  |
| FILTER2D | 既有 | 3x3_replicate_f32c3 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.461838 | 0.210796 | 0.4564 |  |
| FILTER2D | 既有 | 3x3_replicate_f32c4 | header_fastpath | CV_32F | 4 | continuous | 480x640 | 0.448908 | 0.267425 | 0.5957 |  |
| FILTER2D | 既有 | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.530637 | 0.288396 | 0.5435 |  |
| FILTER2D | 既有 | 3x3_replicate_u8c3_roi | header_fastpath | CV_8U | 3 | roi | 479x641 | 0.540329 | 0.302663 | 0.5601 |  |
| FILTER2D | 既有 | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.541354 | 0.380458 | 0.7028 |  |
| GAUSSIAN | 既有 | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 1.515483 | 0.226500 | 0.1495 |  |
| GAUSSIAN | 既有 | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.364329 | 0.032542 | 0.0893 |  |
| GAUSSIAN | 既有 | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.345329 | 0.031712 | 0.0918 |  |
| GAUSSIAN | 既有 | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.750000 | 0.094279 | 0.1257 |  |
| GAUSSIAN | 既有 | 5x5_replicate_f32c1 | gauss_separable | CV_32F | 1 | continuous | 480x640 | 0.306133 | 0.115300 | 0.3766 |  |
| GAUSSIAN | 既有 | 5x5_replicate_f32c3 | gauss_separable | CV_32F | 3 | continuous | 480x640 | 0.400037 | 0.336325 | 0.8407 |  |
| GAUSSIAN | 既有 | 5x5_replicate_f32c4 | gauss_separable | CV_32F | 4 | continuous | 480x640 | 0.372287 | 0.434908 | 1.1682 |  |
| GAUSSIAN | 既有 | 5x5_replicate_u8c3 | gauss_separable | CV_8U | 3 | continuous | 480x640 | 0.494725 | 0.102862 | 0.2079 |  |
| GAUSSIAN | 既有 | 5x5_replicate_u8c3_roi | gauss_separable | CV_8U | 3 | roi | 479x641 | 0.476363 | 0.360304 | 0.7564 |  |
| GAUSSIAN | 既有 | 5x5_replicate_u8c4 | gauss_separable | CV_8U | 4 | continuous | 480x640 | 0.442617 | 0.132350 | 0.2990 |  |
| GET_AFFINE_TRANSFORM | P1 新增 | three_points | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000125 | 0.000259 | 2.0637 | phase1_representative_case;micro_iterations=10000 |
| GET_DERIV_KERNELS | P1 新增 | dx1_ksize5_f32 | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000208 | 0.000085 | 0.4093 | phase1_representative_case;micro_iterations=10000 |
| GET_GABOR_KERNEL | P1 新增 | 15x15_f32 | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.002937 | 0.000965 | 0.3285 | phase1_representative_case;micro_iterations=10000 |
| GET_GAUSSIAN_KERNEL | P1 新增 | ksize15_f32 | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000234 | 0.000942 | 4.0344 | phase1_representative_case;micro_iterations=10000 |
| GET_PERSPECTIVE_TRANSFORM | P1 新增 | four_points_lu | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000271 | 0.000666 | 2.4557 | phase1_representative_case;micro_iterations=10000 |
| GET_RECT_SUB_PIX | P1 新增 | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 1080x1920 | 44.652250 | 0.983050 | 0.0220 | no qualified SIMD fast path |
| GET_RECT_SUB_PIX | P1 新增 | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 479x641 | 6.621904 | 0.143717 | 0.0217 | no qualified SIMD fast path |
| GET_RECT_SUB_PIX | P1 新增 | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 480x640 | 6.618337 | 0.144329 | 0.0218 | no qualified SIMD fast path |
| GET_RECT_SUB_PIX | P1 新增 | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 720x1280 | 19.822429 | 0.430621 | 0.0217 | no qualified SIMD fast path |
| GET_ROTATION_MATRIX_2D | P1 新增 | point_angle_scale | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000065 | 0.000074 | 1.1382 | phase1_representative_case;micro_iterations=10000 |
| GET_ROTATION_MATRIX_2D_ | P1 新增 | matx23d | public_header_baseline | CV_64F | 1 | continuous | micro_batch | 0.000004 | 0.000004 | 1.1270 | phase1_representative_case;micro_iterations=10000 |
| GET_STRUCTURING_ELEMENT | P1 新增 | ellipse7x7 | public_header_baseline | CV_8U | 1 | continuous | micro_batch | 0.000412 | 0.000083 | 0.2004 | phase1_representative_case;micro_iterations=10000 |
| INTEGRAL | P1 新增 | u8c1_to_s32 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 1.503104 | 0.040379 | 0.0269 | phase1_representative_case |
| INVERT_AFFINE_TRANSFORM | P1 新增 | f64_2x3 | public_header_baseline | CV_64F | 1 | continuous | micro_batch | 0.000127 | 0.000060 | 0.4741 | phase1_representative_case;micro_iterations=10000 |
| LAPLACIAN | P1 新增 | ksize3_u8_to_f32 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 9.276783 | 0.110733 | 0.0119 | phase1_representative_case |
| LUT | 既有 | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.187342 | 0.193887 | 1.0349 |  |
| LUT | 既有 | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.077046 | 0.028846 | 0.3744 |  |
| LUT | 既有 | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.084558 | 0.028596 | 0.3382 |  |
| LUT | 既有 | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.115837 | 0.085550 | 0.7385 |  |
| LUT | 既有 | invert_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.118075 | 0.087438 | 0.7405 |  |
| LUT | 既有 | invert_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.131662 | 0.110067 | 0.8360 |  |
| MEDIAN_BLUR | P1 新增 | ksize5_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 50.581192 | 0.249654 | 0.0049 | phase1_representative_case |
| PYR_DOWN | P1 新增 | u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 7.698496 | 0.059637 | 0.0077 | phase1_representative_case |
| PYR_UP | P1 新增 | u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 20.281146 | 0.144767 | 0.0071 | phase1_representative_case |
| REMAP | P1 新增 | fixed_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 1080x1920 | 94.061583 | 4.890642 | 0.0520 | no qualified SIMD fast path |
| REMAP | P1 新增 | fixed_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 479x641 | 15.069962 | 0.748446 | 0.0497 | no qualified SIMD fast path |
| REMAP | P1 新增 | fixed_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 480x640 | 13.995971 | 0.683829 | 0.0489 | no qualified SIMD fast path |
| REMAP | P1 新增 | fixed_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 720x1280 | 42.014967 | 2.106712 | 0.0501 | no qualified SIMD fast path |
| REMAP | P1 新增 | float_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 1080x1920 | 97.281388 | 5.180188 | 0.0532 | no qualified SIMD fast path |
| REMAP | P1 新增 | float_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 479x641 | 14.442871 | 0.706487 | 0.0489 | no qualified SIMD fast path |
| REMAP | P1 新增 | float_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 480x640 | 14.444296 | 0.715087 | 0.0495 | no qualified SIMD fast path |
| REMAP | P1 新增 | float_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 720x1280 | 43.275887 | 2.173058 | 0.0502 | no qualified SIMD fast path |
| RESIZE | 既有 | linear_0.75_f32c1 | headers_baseline | CV_32F | 1 | continuous | 480x640 | 0.200058 | 0.089125 | 0.4455 |  |
| RESIZE | 既有 | linear_0.75_f32c3 | headers_baseline | CV_32F | 3 | continuous | 480x640 | 0.430171 | 0.266625 | 0.6198 |  |
| RESIZE | 既有 | linear_0.75_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.142825 | 0.065025 | 0.4553 |  |
| RESIZE | 既有 | linear_0.75_u8c3_roi | header_fastpath | CV_8U | 3 | roi | 479x641 | 0.137954 | 0.064458 | 0.4672 |  |
| RESIZE | 既有 | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.121254 | 0.081133 | 0.6691 |  |
| RESIZE | 既有 | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.018504 | 0.013021 | 0.7037 |  |
| RESIZE | 既有 | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.018908 | 0.012150 | 0.6426 |  |
| RESIZE | 既有 | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.056417 | 0.037342 | 0.6619 |  |
| RESIZE | 既有 | nearest_0.75_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.089875 | 0.098913 | 1.1006 |  |
| RESIZE | 既有 | nearest_exact_0.75_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.083517 | 0.102771 | 1.2305 |  |
| SCHARR | P1 新增 | dx1_u8_to_f32 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 9.252250 | 0.107417 | 0.0116 | phase1_representative_case |
| SEP_FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 1.557533 | 0.656754 | 0.4217 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.351721 | 0.098487 | 0.2800 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.334121 | 0.094238 | 0.2820 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.754387 | 0.273587 | 0.3627 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_f32c1 | header_fastpath | CV_32F | 1 | continuous | 480x640 | 0.348333 | 0.103288 | 0.2965 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_f32c3 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.417671 | 0.277196 | 0.6637 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_f32c4 | header_fastpath | CV_32F | 4 | continuous | 480x640 | 0.424521 | 0.372371 | 0.8772 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.452067 | 0.287312 | 0.6356 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_u8c3_roi | header_fastpath | CV_8U | 3 | roi | 479x641 | 0.445087 | 0.287842 | 0.6467 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.404138 | 0.382442 | 0.9463 |  |
| SOBEL | 既有 | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.389754 | 0.857625 | 2.2004 |  |
| SOBEL | 既有 | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.130300 | 0.120475 | 0.9246 |  |
| SOBEL | 既有 | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.114933 | 0.120092 | 1.0449 |  |
| SOBEL | 既有 | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.213475 | 0.341533 | 1.5999 |  |
| SOBEL | 既有 | dx1_ksize3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.176196 | 0.331925 | 1.8838 |  |
| SOBEL | 既有 | dx1_ksize3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.240137 | 0.447650 | 1.8641 |  |
| SPATIAL_GRADIENT | P1 新增 | ksize3_u8_to_s16 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.247729 | 0.032467 | 0.1311 | phase1_representative_case |
| SQR_BOX_FILTER | P1 新增 | 3x3_u8_to_f32 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 8.154929 | 0.172054 | 0.0211 | phase1_representative_case |
| STACK_BLUR | P1 新增 | 5x5_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 1.512908 | 0.118983 | 0.0786 | phase1_representative_case |
| THRESHOLD | 既有 | binary_f32c3_roi | header_fastpath | CV_32F | 3 | roi | 479x641 | 0.417292 | 0.068717 | 0.1647 |  |
| THRESHOLD | 既有 | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.989663 | 0.033121 | 0.0335 |  |
| THRESHOLD | 既有 | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.213292 | 0.004829 | 0.0226 |  |
| THRESHOLD | 既有 | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.173008 | 0.004942 | 0.0286 |  |
| THRESHOLD | 既有 | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.646983 | 0.015221 | 0.0235 |  |
| THRESHOLD_WITH_MASK | P1 新增 | binary_masked_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.167108 | 0.167908 | 1.0048 | phase1_representative_case |
| WARP_AFFINE | 既有 | linear_inverse_replicate | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 21.943467 | 1.953354 | 0.0890 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate | headers_baseline | CV_8U | 1 | continuous | 479x641 | 3.502650 | 0.320108 | 0.0914 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate | headers_baseline | CV_8U | 1 | continuous | 480x640 | 3.303125 | 0.285412 | 0.0864 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 9.861008 | 0.864525 | 0.0877 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate_f32c1 | headers_baseline | CV_32F | 1 | continuous | 480x640 | 2.897337 | 0.519508 | 0.1793 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate_f32c3 | headers_baseline | CV_32F | 3 | continuous | 480x640 | 7.497908 | 0.722967 | 0.0964 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate_f32c4 | headers_baseline | CV_32F | 4 | continuous | 480x640 | 9.850775 | 0.790296 | 0.0802 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate_u8c3 | headers_baseline | CV_8U | 3 | continuous | 480x640 | 9.766058 | 0.786604 | 0.0805 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate_u8c4 | headers_baseline | CV_8U | 4 | continuous | 480x640 | 12.969271 | 0.515604 | 0.0398 |  |
| WARP_PERSPECTIVE | P1 新增 | projective_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 1080x1920 | 67.114429 | 6.502475 | 0.0969 | no qualified SIMD fast path |
| WARP_PERSPECTIVE | P1 新增 | projective_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 479x641 | 10.776096 | 0.985254 | 0.0914 | no qualified SIMD fast path |
| WARP_PERSPECTIVE | P1 新增 | projective_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 480x640 | 9.873088 | 0.948237 | 0.0960 | no qualified SIMD fast path |
| WARP_PERSPECTIVE | P1 新增 | projective_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 720x1280 | 29.751379 | 2.879333 | 0.0968 | no qualified SIMD fast path |

## 不支持用例

| Suite | Op | Variant | Shape | Status | Note |
| --- | --- | --- | --- | --- | --- |
| imgproc | CVTCOLOR | BGR2NV12_u8 | 480x640 | UNSUPPORTED | upstream OpenCV has NV12 decode but no single-call BGR-to-NV12 encoder |

## 说明

- 比值统一为 `OpenCV耗时 / CVH耗时`：大于 `1` 表示 CVH 更快，小于 `1` 表示 OpenCV 更快。
- 表内耗时取各 repeat 的最小单次耗时，用于降低系统抖动影响；本报告不是跨机器排名。
- Mat case 对比相同的分配/复用语义；imgproc case 对齐输入尺寸、类型、kernel、border 和主要参数。
- `headers_baseline` 不等于跳过优化，它表示 `cvh::headers_fast` 当前继承了 `cvh::headers` 的通用实现。
- 原始 CSV 和 metadata 是可再生成的运行产物，日期命名 Markdown 是阶段性快照。
