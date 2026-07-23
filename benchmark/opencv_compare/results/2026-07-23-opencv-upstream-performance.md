# opencv-header-only vs OpenCV Upstream 性能对比（2026-07-23）

生成时间（UTC）：`2026-07-23 11:55:46Z`

## 当前项目状态

- `opencv-header-only` 当前公共定位是纯 header-only，不依赖项目内 `.cpp` 扩展层。
- Mode B 只比较当前 `cvh::headers_fast` 与同机编译的 upstream OpenCV；`cvh::headers_fast` 表示最快 header-only 构建配置。
- `cvh::headers_fast` 完整继承 `cvh::headers`。算子没有专用 fast-path 时继续执行继承的 header 实现并参与 benchmark，不因缺少 SIMD 特化而跳过。
- Core 的 `add/subtract/multiply/divide/transpose/GEMM` 已迁入 ODR-safe headers；本报告通过公共 API 测量，不链接 legacy core 对象。
- OpenCV Universal Intrinsics 是默认 SIMD 方言，kernel 直接使用 OpenCV UI；项目已移除 xsimd 性能路径。
- ARM 当前关注 NEON，本次实测平台为 Apple ARM；x86 目标是 SSE/AVX 系列，RVV 因 scalable vector 设计问题暂缓。
- Imgproc legacy `.cpp` fast-path 已迁入 ODR-safe detail headers；resize/cvtColor UI、filter、LUT、border、Sobel、Canny 和 morphology 均从公共 header API 进入。
- Stable imgproc 矩阵覆盖代表性的 `CV_8U` / `CV_32F` 与 C1/C3/C4；full profile 额外覆盖非连续 ROI。

## 高层优化结构

| 层次 | 当前实现 | 本报告中的含义 |
| --- | --- | --- |
| 公共 API | OpenCV-compatible header API | 所有 case 均从 `cvh::headers_fast` 公共入口调用 |
| SIMD 方言 | OpenCV Universal Intrinsics | 在 Apple ARM 上映射到 NEON |
| 专用 kernel | `cvtColor`、特定 `resize` UI kernel | 记录为 `dispatch_path=opencv_ui`；core 计算当前仍为 baseline |
| Header fast-path | 行并行 filter、LUT、border、Sobel、Canny、morphology | 记录为 `dispatch_path=header_fastpath` |
| 通用实现 | `cvh::headers` 中的 header baseline | 无专用 fast-path 时自动继承，记录为 `headers_baseline` |
| 对照实现 | upstream OpenCV `core` / `imgproc` | 相同输入、尺寸、border 和线程配置 |

## 运行配置

- Profile：`stable`
- CVH 实现：`cvh_headers_fast`
- 采样：`warmup=2, iters=20, repeats=5`
- 线程数：`1`
- OpenMP：`dynamic=false, proc_bind=close`
- 主机：`Darwin arm64`
- CPU：`Apple M5`
- 编译器：`Apple clang version 21.0.0 (clang-2100.0.123.102)`
- 构建类型：`Release`
- CVH commit：`ac9bac79a6c4e288bc12fffd7ec29052be612ab3` + dirty
- OpenCV：`4.14.0`，commit `d48bf69f65444a13f8a34b8982b083c1b78fa0e8` + dirty
- 原始数据：`2026-07-23-opencv-upstream-performance.csv`；元数据：`2026-07-23-opencv-upstream-performance.csv.meta.json`

## 汇总

- 总 case：`176`；有效：`176`；不支持：`0`。
- `OpenCV/CVH` 几何平均：`0.2799`；中位数：`0.3336`。
- CVH 更快：`22` 个；OpenCV 更快或相当：`154` 个。

| Suite | Cases | 几何平均 OpenCV/CVH | 中位数 | CVH 更快 | OpenCV 更快/相当 |
| --- | --- | --- | --- | --- | --- |
| core_mat | 84 | 0.2101 | 0.3255 | 10 | 74 |
| imgproc | 92 | 0.3637 | 0.3776 | 12 | 80 |

## 算子级概览

### `core_mat`

| Op | CVH dispatch | Cases | 几何平均 OpenCV/CVH | 领先方 |
| --- | --- | --- | --- | --- |
| ADD | headers_baseline | 12 | 0.1829 | OpenCV `5.47x` |
| DIVIDE | headers_baseline | 12 | 0.3406 | OpenCV `2.94x` |
| GEMM | headers_baseline | 6 | 0.0126 | OpenCV `79.63x` |
| MAT_CLONE | headers_baseline | 3 | 0.9929 | OpenCV `1.01x` |
| MAT_CONVERTTO | headers_baseline | 3 | 0.9839 | OpenCV `1.02x` |
| MAT_COPYTO | headers_baseline | 3 | 0.9787 | OpenCV `1.02x` |
| MAT_CREATE | headers_baseline | 3 | 0.0692 | OpenCV `14.46x` |
| MAT_RESHAPE | headers_baseline | 3 | 0.4005 | OpenCV `2.50x` |
| MAT_SETTO | headers_baseline | 3 | 0.0108 | OpenCV `92.64x` |
| MULTIPLY | headers_baseline | 12 | 0.1858 | OpenCV `5.38x` |
| SUBTRACT | headers_baseline | 12 | 0.1902 | OpenCV `5.26x` |
| TRANSPOSE | headers_baseline | 12 | 0.5639 | OpenCV `1.77x` |

### `imgproc`

| Op | CVH dispatch | Cases | 几何平均 OpenCV/CVH | 领先方 |
| --- | --- | --- | --- | --- |
| BOX_FILTER | box3x3, header_fastpath | 8 | 0.3267 | OpenCV `3.06x` |
| CANNY | header_fastpath | 3 | 0.9298 | OpenCV `1.08x` |
| COPY_MAKE_BORDER | header_fastpath | 8 | 0.4406 | OpenCV `2.27x` |
| CVTCOLOR | header_fastpath, opencv_ui | 10 | 0.5215 | OpenCV `1.92x` |
| DILATE | header_fastpath | 5 | 0.1150 | OpenCV `8.70x` |
| ERODE | header_fastpath | 5 | 0.1143 | OpenCV `8.75x` |
| FILTER2D | header_fastpath | 8 | 0.3275 | OpenCV `3.05x` |
| GAUSSIAN | gauss_separable, header_fastpath | 8 | 0.2769 | OpenCV `3.61x` |
| LUT | header_fastpath | 5 | 0.6737 | OpenCV `1.48x` |
| RESIZE | header_fastpath, headers_baseline, opencv_ui | 8 | 0.6905 | OpenCV `1.45x` |
| SEP_FILTER2D | header_fastpath | 8 | 0.4458 | OpenCV `2.24x` |
| SOBEL | header_fastpath | 5 | 1.5600 | CVH `1.56x` |
| THRESHOLD | headers_baseline | 3 | 0.0301 | OpenCV `33.27x` |
| WARP_AFFINE | headers_baseline | 8 | 0.2774 | OpenCV `3.60x` |

## 详细结果

### `core_mat`

| Op | Variant | CVH dispatch | Depth | Ch | Layout | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADD | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 1080x1920 | 0.646119 | 0.215081 | 0.3329 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 480x640 | 0.090685 | 0.029723 | 0.3278 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 720x1280 | 0.298360 | 0.096675 | 0.3240 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 1080x1920 | 1.134265 | 0.669950 | 0.5906 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 480x640 | 0.167225 | 0.090092 | 0.5387 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 720x1280 | 0.513298 | 0.277773 | 0.5412 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.799015 | 0.050113 | 0.0627 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.121735 | 0.007021 | 0.0577 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.355054 | 0.019406 | 0.0547 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 1080x1920 | 1.373456 | 0.150758 | 0.1098 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 480x640 | 0.203383 | 0.022371 | 0.1100 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 720x1280 | 0.609960 | 0.058838 | 0.0965 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 1080x1920 | 0.613531 | 0.205112 | 0.3343 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 480x640 | 0.091358 | 0.029442 | 0.3223 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 720x1280 | 0.275158 | 0.091410 | 0.3322 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 1080x1920 | 1.495783 | 0.644338 | 0.4308 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 480x640 | 0.221106 | 0.091190 | 0.4124 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 720x1280 | 0.678883 | 0.282127 | 0.4156 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 1.777837 | 0.456154 | 0.2566 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.262667 | 0.067696 | 0.2577 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.794117 | 0.201648 | 0.2539 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 1080x1920 | 3.633004 | 1.364558 | 0.3756 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 480x640 | 0.526702 | 0.201554 | 0.3827 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 720x1280 | 1.584323 | 0.606506 | 0.3828 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| GEMM | fp32_nn_end_to_end | headers_baseline | CV_32F | 1 | continuous | 128x128x128 | 0.214630 | 0.003594 | 0.0167 | correctness=upstream_pass;iters=8 |
| GEMM | fp32_nn_end_to_end | headers_baseline | CV_32F | 1 | continuous | 256x256x256 | 2.029333 | 0.024959 | 0.0123 | correctness=upstream_pass;iters=1 |
| GEMM | fp32_nn_end_to_end | headers_baseline | CV_32F | 1 | continuous | 512x512x512 | 17.274625 | 0.171500 | 0.0099 | correctness=upstream_pass;iters=1 |
| GEMM | fp32_nn_pack_once | headers_baseline | CV_32F | 1 | continuous | 128x128x128 | 0.207739 | 0.003474 | 0.0167 | correctness=upstream_pass;opencv_reuses_B_without_public_pack_handle;iters=8 |
| GEMM | fp32_nn_pack_once | headers_baseline | CV_32F | 1 | continuous | 256x256x256 | 2.100875 | 0.025334 | 0.0121 | correctness=upstream_pass;opencv_reuses_B_without_public_pack_handle;iters=1 |
| GEMM | fp32_nn_pack_once | headers_baseline | CV_32F | 1 | continuous | 512x512x512 | 17.813125 | 0.169417 | 0.0095 | correctness=upstream_pass;opencv_reuses_B_without_public_pack_handle;iters=1 |
| MAT_CLONE | full_copy | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.026796 | 0.025112 | 0.9372 |  |
| MAT_CLONE | full_copy | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.003681 | 0.003700 | 1.0051 |  |
| MAT_CLONE | full_copy | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.010392 | 0.010800 | 1.0393 |  |
| MAT_CONVERTTO | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.078673 | 0.079252 | 1.0074 |  |
| MAT_CONVERTTO | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.012567 | 0.011844 | 0.9425 |  |
| MAT_CONVERTTO | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.035202 | 0.035313 | 1.0031 |  |
| MAT_COPYTO | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.024365 | 0.024700 | 1.0138 |  |
| MAT_COPYTO | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.003646 | 0.003413 | 0.9360 |  |
| MAT_COPYTO | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.010225 | 0.010102 | 0.9880 |  |
| MAT_CREATE | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.000023 | 0.000001 | 0.0510 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.000015 | 0.000001 | 0.0795 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.000014 | 0.000001 | 0.0816 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_RESHAPE | to_column_view | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.000037 | 0.000015 | 0.4097 | micro_iters_x1000 |
| MAT_RESHAPE | to_column_view | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.000036 | 0.000015 | 0.4260 | micro_iters_x1000 |
| MAT_RESHAPE | to_column_view | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.000069 | 0.000025 | 0.3681 | micro_iters_x1000 |
| MAT_SETTO | scalar_all | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 2.423992 | 0.024319 | 0.0100 |  |
| MAT_SETTO | scalar_all | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.339958 | 0.004067 | 0.0120 |  |
| MAT_SETTO | scalar_all | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 1.077067 | 0.011288 | 0.0105 |  |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 1080x1920 | 0.618452 | 0.205827 | 0.3328 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 480x640 | 0.093877 | 0.029413 | 0.3133 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 720x1280 | 0.280756 | 0.091550 | 0.3261 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 1080x1920 | 1.136031 | 0.629254 | 0.5539 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 480x640 | 0.170104 | 0.091644 | 0.5388 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 720x1280 | 0.501342 | 0.271787 | 0.5421 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.792438 | 0.049525 | 0.0625 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.123263 | 0.007027 | 0.0570 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.357831 | 0.022346 | 0.0624 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 1080x1920 | 1.372131 | 0.160912 | 0.1173 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 480x640 | 0.204160 | 0.022087 | 0.1082 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 720x1280 | 0.610062 | 0.066583 | 0.1091 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 1080x1920 | 0.623815 | 0.201033 | 0.3223 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 480x640 | 0.094473 | 0.030079 | 0.3184 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_32F | 1 | continuous | 720x1280 | 0.279558 | 0.090860 | 0.3250 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 1080x1920 | 1.140642 | 0.627402 | 0.5500 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 480x640 | 0.166894 | 0.089750 | 0.5378 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_32F | 3 | continuous | 720x1280 | 0.504719 | 0.270990 | 0.5369 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.755183 | 0.049615 | 0.0657 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.111731 | 0.007019 | 0.0628 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.335250 | 0.022046 | 0.0658 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 1080x1920 | 1.257227 | 0.136427 | 0.1085 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 480x640 | 0.183148 | 0.022035 | 0.1203 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_8U | 3 | continuous | 720x1280 | 0.554181 | 0.066185 | 0.1194 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_32F | 1 | continuous | 1080x1920 | 0.914429 | 0.580315 | 0.6346 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_32F | 1 | continuous | 480x640 | 0.196692 | 0.070440 | 0.3581 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_32F | 1 | continuous | 720x1280 | 0.436027 | 0.290258 | 0.6657 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_32F | 3 | continuous | 1080x1920 | 0.757554 | 1.625979 | 2.1464 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_32F | 3 | continuous | 480x640 | 0.125090 | 0.152044 | 1.2155 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_32F | 3 | continuous | 720x1280 | 0.317958 | 0.622754 | 1.9586 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.650027 | 0.121752 | 0.1873 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.154998 | 0.006844 | 0.0442 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.368762 | 0.026583 | 0.0721 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_8U | 3 | continuous | 1080x1920 | 0.417490 | 0.749192 | 1.7945 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_8U | 3 | continuous | 480x640 | 0.117181 | 0.083342 | 0.7112 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_8U | 3 | continuous | 720x1280 | 0.223671 | 0.392742 | 1.7559 | correctness=upstream_pass |

### `imgproc`

| Op | Variant | CVH dispatch | Depth | Ch | Layout | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BOX_FILTER | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 1.498410 | 0.288300 | 0.1924 |  |
| BOX_FILTER | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.283710 | 0.048423 | 0.1707 |  |
| BOX_FILTER | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.716035 | 0.128017 | 0.1788 |  |
| BOX_FILTER | 3x3_replicate_f32c1 | box3x3 | CV_32F | 1 | continuous | 480x640 | 0.180244 | 0.105077 | 0.5830 |  |
| BOX_FILTER | 3x3_replicate_f32c3 | box3x3 | CV_32F | 3 | continuous | 480x640 | 0.325140 | 0.300708 | 0.9249 |  |
| BOX_FILTER | 3x3_replicate_f32c4 | box3x3 | CV_32F | 4 | continuous | 480x640 | 0.386165 | 0.405533 | 1.0502 |  |
| BOX_FILTER | 3x3_replicate_u8c3 | box3x3 | CV_8U | 3 | continuous | 480x640 | 0.711210 | 0.133004 | 0.1870 |  |
| BOX_FILTER | 3x3_replicate_u8c4 | box3x3 | CV_8U | 4 | continuous | 480x640 | 0.847221 | 0.176710 | 0.2086 |  |
| CANNY | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 29.695425 | 28.140046 | 0.9476 |  |
| CANNY | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 480x640 | 4.610137 | 4.084144 | 0.8859 |  |
| CANNY | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 13.247215 | 12.683854 | 0.9575 |  |
| COPY_MAKE_BORDER | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.072060 | 0.044825 | 0.6220 |  |
| COPY_MAKE_BORDER | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.063008 | 0.007256 | 0.1152 |  |
| COPY_MAKE_BORDER | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.067379 | 0.020242 | 0.3004 |  |
| COPY_MAKE_BORDER | 2px_replicate_f32c1 | header_fastpath | CV_32F | 1 | continuous | 480x640 | 0.066206 | 0.026554 | 0.4011 |  |
| COPY_MAKE_BORDER | 2px_replicate_f32c3 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.078531 | 0.083829 | 1.0675 |  |
| COPY_MAKE_BORDER | 2px_replicate_f32c4 | header_fastpath | CV_32F | 4 | continuous | 480x640 | 0.085687 | 0.110912 | 1.2944 |  |
| COPY_MAKE_BORDER | 2px_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.070013 | 0.021265 | 0.3037 |  |
| COPY_MAKE_BORDER | 2px_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.070340 | 0.027556 | 0.3918 |  |
| CVTCOLOR | BGR2BGRA_u8 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.125319 | 0.021038 | 0.1679 |  |
| CVTCOLOR | BGR2GRAY_f32 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.084883 | 0.045098 | 0.5313 |  |
| CVTCOLOR | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 1080x1920 | 0.206381 | 0.209163 | 1.0135 |  |
| CVTCOLOR | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.031348 | 0.031298 | 0.9984 |  |
| CVTCOLOR | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 720x1280 | 0.097640 | 0.097402 | 0.9976 |  |
| CVTCOLOR | BGR2RGB_f32 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.093181 | 0.055721 | 0.5980 |  |
| CVTCOLOR | BGR2RGB_u8 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.091960 | 0.017331 | 0.1885 |  |
| CVTCOLOR | BGR2YUV_u8 | header_fastpath | CV_8U | 3 | yuv444_interleaved | 480x640 | 0.121271 | 0.081025 | 0.6681 |  |
| CVTCOLOR | BGRA2GRAY_u8 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.104569 | 0.041556 | 0.3974 |  |
| CVTCOLOR | YUV2BGR_u8 | header_fastpath | CV_8U | 3 | yuv444_interleaved | 480x640 | 0.119067 | 0.065731 | 0.5521 |  |
| DILATE | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 1.150373 | 0.152767 | 0.1328 |  |
| DILATE | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.316360 | 0.023354 | 0.0738 |  |
| DILATE | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.606900 | 0.068885 | 0.1135 |  |
| DILATE | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.532044 | 0.067771 | 0.1274 |  |
| DILATE | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.635044 | 0.090000 | 0.1417 |  |
| ERODE | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 1.164023 | 0.146748 | 0.1261 |  |
| ERODE | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.306758 | 0.022748 | 0.0742 |  |
| ERODE | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.627465 | 0.068775 | 0.1096 |  |
| ERODE | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.512254 | 0.067933 | 0.1326 |  |
| ERODE | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.624948 | 0.089679 | 0.1435 |  |
| FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 3.865379 | 0.646269 | 0.1672 |  |
| FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.423813 | 0.098627 | 0.2327 |  |
| FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 1.121237 | 0.286946 | 0.2559 |  |
| FILTER2D | 3x3_replicate_f32c1 | header_fastpath | CV_32F | 1 | continuous | 480x640 | 0.403981 | 0.078006 | 0.1931 |  |
| FILTER2D | 3x3_replicate_f32c3 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.586615 | 0.190810 | 0.3253 |  |
| FILTER2D | 3x3_replicate_f32c4 | header_fastpath | CV_32F | 4 | continuous | 480x640 | 0.451021 | 0.258117 | 0.5723 |  |
| FILTER2D | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.550640 | 0.303615 | 0.5514 |  |
| FILTER2D | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.571110 | 0.383246 | 0.6711 |  |
| GAUSSIAN | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 1.660952 | 0.233544 | 0.1406 |  |
| GAUSSIAN | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.348348 | 0.032548 | 0.0934 |  |
| GAUSSIAN | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.847433 | 0.095408 | 0.1126 |  |
| GAUSSIAN | 5x5_replicate_f32c1 | gauss_separable | CV_32F | 1 | continuous | 480x640 | 0.316377 | 0.115015 | 0.3635 |  |
| GAUSSIAN | 5x5_replicate_f32c3 | gauss_separable | CV_32F | 3 | continuous | 480x640 | 0.410219 | 0.333908 | 0.8140 |  |
| GAUSSIAN | 5x5_replicate_f32c4 | gauss_separable | CV_32F | 4 | continuous | 480x640 | 0.393492 | 0.426658 | 1.0843 |  |
| GAUSSIAN | 5x5_replicate_u8c3 | gauss_separable | CV_8U | 3 | continuous | 480x640 | 0.501933 | 0.108525 | 0.2162 |  |
| GAUSSIAN | 5x5_replicate_u8c4 | gauss_separable | CV_8U | 4 | continuous | 480x640 | 0.433998 | 0.146148 | 0.3367 |  |
| LUT | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.199317 | 0.193883 | 0.9727 |  |
| LUT | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.085569 | 0.029800 | 0.3483 |  |
| LUT | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.122315 | 0.085700 | 0.7007 |  |
| LUT | invert_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.118550 | 0.082540 | 0.6962 |  |
| LUT | invert_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.131883 | 0.110746 | 0.8397 |  |
| RESIZE | linear_0.75_f32c1 | headers_baseline | CV_32F | 1 | continuous | 480x640 | 0.215750 | 0.098906 | 0.4584 |  |
| RESIZE | linear_0.75_f32c3 | headers_baseline | CV_32F | 3 | continuous | 480x640 | 0.457298 | 0.281106 | 0.6147 |  |
| RESIZE | linear_0.75_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.143285 | 0.069029 | 0.4818 |  |
| RESIZE | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.121648 | 0.081746 | 0.6720 |  |
| RESIZE | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.018894 | 0.012254 | 0.6486 |  |
| RESIZE | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.060769 | 0.040323 | 0.6635 |  |
| RESIZE | nearest_0.75_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.088575 | 0.098885 | 1.1164 |  |
| RESIZE | nearest_exact_0.75_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.087527 | 0.103217 | 1.1793 |  |
| SEP_FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 2.045652 | 0.664065 | 0.3246 |  |
| SEP_FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.361617 | 0.098079 | 0.2712 |  |
| SEP_FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.787154 | 0.272838 | 0.3466 |  |
| SEP_FILTER2D | 3x3_replicate_f32c1 | header_fastpath | CV_32F | 1 | continuous | 480x640 | 0.337967 | 0.084440 | 0.2498 |  |
| SEP_FILTER2D | 3x3_replicate_f32c3 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.439181 | 0.223242 | 0.5083 |  |
| SEP_FILTER2D | 3x3_replicate_f32c4 | header_fastpath | CV_32F | 4 | continuous | 480x640 | 0.399321 | 0.285621 | 0.7153 |  |
| SEP_FILTER2D | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.480767 | 0.286415 | 0.5957 |  |
| SEP_FILTER2D | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.404471 | 0.382271 | 0.9451 |  |
| SOBEL | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.542604 | 0.877381 | 1.6170 |  |
| SOBEL | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.117931 | 0.124735 | 1.0577 |  |
| SOBEL | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.215373 | 0.346108 | 1.6070 |  |
| SOBEL | dx1_ksize3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.184840 | 0.331879 | 1.7955 |  |
| SOBEL | dx1_ksize3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.237367 | 0.444360 | 1.8720 |  |
| THRESHOLD | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 1.050917 | 0.033485 | 0.0319 |  |
| THRESHOLD | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.159004 | 0.005625 | 0.0354 |  |
| THRESHOLD | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.618712 | 0.014906 | 0.0241 |  |
| WARP_AFFINE | linear_inverse_replicate | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 11.202506 | 1.945302 | 0.1736 |  |
| WARP_AFFINE | linear_inverse_replicate | headers_baseline | CV_8U | 1 | continuous | 480x640 | 1.181904 | 0.308881 | 0.2613 |  |
| WARP_AFFINE | linear_inverse_replicate | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 3.614485 | 0.908406 | 0.2513 |  |
| WARP_AFFINE | linear_inverse_replicate_f32c1 | headers_baseline | CV_32F | 1 | continuous | 480x640 | 0.942817 | 0.513410 | 0.5445 |  |
| WARP_AFFINE | linear_inverse_replicate_f32c3 | headers_baseline | CV_32F | 3 | continuous | 480x640 | 1.969498 | 0.715492 | 0.3633 |  |
| WARP_AFFINE | linear_inverse_replicate_f32c4 | headers_baseline | CV_32F | 4 | continuous | 480x640 | 2.464494 | 0.761175 | 0.3089 |  |
| WARP_AFFINE | linear_inverse_replicate_u8c3 | headers_baseline | CV_8U | 3 | continuous | 480x640 | 2.502423 | 0.819900 | 0.3276 |  |
| WARP_AFFINE | linear_inverse_replicate_u8c4 | headers_baseline | CV_8U | 4 | continuous | 480x640 | 3.415600 | 0.524565 | 0.1536 |  |

## 说明

- 比值统一为 `OpenCV耗时 / CVH耗时`：大于 `1` 表示 CVH 更快，小于 `1` 表示 OpenCV 更快。
- 表内耗时取各 repeat 的最小单次耗时，用于降低系统抖动影响；本报告不是跨机器排名。
- Mat case 对比相同的分配/复用语义；imgproc case 对齐输入尺寸、类型、kernel、border 和主要参数。
- `headers_baseline` 不等于跳过优化，它表示 `cvh::headers_fast` 当前继承了 `cvh::headers` 的通用实现。
- 原始 CSV 和 metadata 是可再生成的运行产物，日期命名 Markdown 是阶段性快照。
