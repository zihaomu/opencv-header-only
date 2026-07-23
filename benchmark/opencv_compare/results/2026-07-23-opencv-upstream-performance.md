# opencv-header-only vs OpenCV Upstream 性能对比（2026-07-23）

生成时间（UTC）：`2026-07-23 09:20:56Z`

## 当前项目状态

- `opencv-header-only` 当前公共定位是纯 header-only，不依赖项目内 `.cpp` 扩展层。
- Mode B 只比较当前 `cvh::headers_fast` 与同机编译的 upstream OpenCV；`cvh::headers_fast` 表示最快 header-only 构建配置。
- `cvh::headers_fast` 完整继承 `cvh::headers`。算子没有专用 fast-path 时继续执行继承的 header 实现并参与 benchmark，不因缺少 SIMD 特化而跳过。
- OpenCV Universal Intrinsics 是默认 SIMD 方言，kernel 直接使用 OpenCV UI；项目已移除 xsimd 性能路径。
- ARM 当前关注 NEON，本次实测平台为 Apple ARM；x86 目标是 SSE/AVX 系列，RVV 因 scalable vector 设计问题暂缓。
- 当前已验证的 UI 专用路径主要包括 `BGR/RGB2GRAY` 和 `CV_8UC1` 精确 2x `INTER_LINEAR` 下采样；其余本报告算子多数仍走继承的 header baseline。

## 高层优化结构

| 层次 | 当前实现 | 本报告中的含义 |
| --- | --- | --- |
| 公共 API | OpenCV-compatible header API | 所有 case 均从 `cvh::headers_fast` 公共入口调用 |
| SIMD 方言 | OpenCV Universal Intrinsics | 在 Apple ARM 上映射到 NEON |
| 专用 kernel | `cvtColor`、特定 `resize` UI kernel | 记录为 `dispatch_path=opencv_ui` |
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
- CVH commit：`658fabdd0f51bd340ec62b81e1d6541375042b1c` + dirty
- OpenCV：`4.14.0`，commit `d48bf69f65444a13f8a34b8982b083c1b78fa0e8`
- 原始数据：`2026-07-23-opencv-upstream-performance.csv`；元数据：`2026-07-23-opencv-upstream-performance.meta.json`

## 汇总

- 总 case：`60`；有效：`60`；不支持：`0`。
- `OpenCV/CVH` 几何平均：`0.0614`；中位数：`0.0304`。
- CVH 更快：`5` 个；OpenCV 更快或相当：`55` 个。

| Suite | Cases | 几何平均 OpenCV/CVH | 中位数 | CVH 更快 | OpenCV 更快/相当 |
| --- | --- | --- | --- | --- | --- |
| core_mat | 18 | 0.2829 | 0.6214 | 5 | 13 |
| imgproc | 42 | 0.0319 | 0.0221 | 0 | 42 |

## 算子级概览

### `core_mat`

| Op | CVH dispatch | Cases | 几何平均 OpenCV/CVH | 领先方 |
| --- | --- | --- | --- | --- |
| MAT_CLONE | headers_baseline | 3 | 0.9399 | OpenCV `1.06x` |
| MAT_CONVERTTO | headers_baseline | 3 | 0.9997 | OpenCV `1.00x` |
| MAT_COPYTO | headers_baseline | 3 | 0.9645 | OpenCV `1.04x` |
| MAT_CREATE | headers_baseline | 3 | 0.1110 | OpenCV `9.01x` |
| MAT_RESHAPE | headers_baseline | 3 | 0.4236 | OpenCV `2.36x` |
| MAT_SETTO | headers_baseline | 3 | 0.0120 | OpenCV `83.08x` |

### `imgproc`

| Op | CVH dispatch | Cases | 几何平均 OpenCV/CVH | 领先方 |
| --- | --- | --- | --- | --- |
| BOX_FILTER | headers_baseline | 3 | 0.0088 | OpenCV `113.37x` |
| CANNY | headers_baseline | 3 | 0.2508 | OpenCV `3.99x` |
| COPY_MAKE_BORDER | headers_baseline | 3 | 0.0273 | OpenCV `36.62x` |
| CVTCOLOR | opencv_ui | 3 | 0.9974 | OpenCV `1.00x` |
| DILATE | headers_baseline | 3 | 0.0036 | OpenCV `274.57x` |
| ERODE | headers_baseline | 3 | 0.0042 | OpenCV `240.51x` |
| FILTER2D | headers_baseline | 3 | 0.0028 | OpenCV `355.89x` |
| GAUSSIAN | headers_baseline | 3 | 0.0075 | OpenCV `132.69x` |
| LUT | headers_baseline | 3 | 0.0196 | OpenCV `51.03x` |
| RESIZE | opencv_ui | 3 | 0.6621 | OpenCV `1.51x` |
| SEP_FILTER2D | headers_baseline | 3 | 0.0218 | OpenCV `45.95x` |
| SOBEL | headers_baseline | 3 | 0.0236 | OpenCV `42.45x` |
| THRESHOLD | headers_baseline | 3 | 0.0337 | OpenCV `29.65x` |
| WARP_AFFINE | headers_baseline | 3 | 0.2627 | OpenCV `3.81x` |

## 详细结果

### `core_mat`

| Op | Variant | CVH dispatch | Depth | Ch | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MAT_CLONE | full_copy | headers_baseline | CV_8U | 1 | 1080x1920 | 0.024579 | 0.024869 | 1.0118 |  |
| MAT_CLONE | full_copy | headers_baseline | CV_8U | 1 | 480x640 | 0.010215 | 0.008340 | 0.8164 |  |
| MAT_CLONE | full_copy | headers_baseline | CV_8U | 1 | 720x1280 | 0.011196 | 0.011254 | 1.0052 |  |
| MAT_CONVERTTO | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | 1080x1920 | 0.078244 | 0.078408 | 1.0021 |  |
| MAT_CONVERTTO | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | 480x640 | 0.013821 | 0.013727 | 0.9932 |  |
| MAT_CONVERTTO | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | 720x1280 | 0.033890 | 0.034015 | 1.0037 |  |
| MAT_COPYTO | continuous_reuse | headers_baseline | CV_8U | 1 | 1080x1920 | 0.024673 | 0.023919 | 0.9694 |  |
| MAT_COPYTO | continuous_reuse | headers_baseline | CV_8U | 1 | 480x640 | 0.007906 | 0.007306 | 0.9241 |  |
| MAT_COPYTO | continuous_reuse | headers_baseline | CV_8U | 1 | 720x1280 | 0.010869 | 0.010885 | 1.0015 |  |
| MAT_CREATE | reuse_same_shape | headers_baseline | CV_8U | 1 | 1080x1920 | 0.000010 | 0.000001 | 0.1088 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | reuse_same_shape | headers_baseline | CV_8U | 1 | 480x640 | 0.000023 | 0.000002 | 0.1057 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | reuse_same_shape | headers_baseline | CV_8U | 1 | 720x1280 | 0.000010 | 0.000001 | 0.1191 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_RESHAPE | to_column_view | headers_baseline | CV_8U | 1 | 1080x1920 | 0.000035 | 0.000015 | 0.4246 | micro_iters_x1000 |
| MAT_RESHAPE | to_column_view | headers_baseline | CV_8U | 1 | 480x640 | 0.000040 | 0.000017 | 0.4264 | micro_iters_x1000 |
| MAT_RESHAPE | to_column_view | headers_baseline | CV_8U | 1 | 720x1280 | 0.000036 | 0.000015 | 0.4199 | micro_iters_x1000 |
| MAT_SETTO | scalar_all | headers_baseline | CV_8U | 1 | 1080x1920 | 2.056369 | 0.023854 | 0.0116 |  |
| MAT_SETTO | scalar_all | headers_baseline | CV_8U | 1 | 480x640 | 0.403790 | 0.004852 | 0.0120 |  |
| MAT_SETTO | scalar_all | headers_baseline | CV_8U | 1 | 720x1280 | 0.860104 | 0.010762 | 0.0125 |  |

### `imgproc`

| Op | Variant | CVH dispatch | Depth | Ch | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BOX_FILTER | 3x3_replicate | headers_baseline | CV_8U | 1 | 1080x1920 | 30.136152 | 0.255810 | 0.0085 |  |
| BOX_FILTER | 3x3_replicate | headers_baseline | CV_8U | 1 | 480x640 | 4.368494 | 0.040248 | 0.0092 |  |
| BOX_FILTER | 3x3_replicate | headers_baseline | CV_8U | 1 | 720x1280 | 13.149825 | 0.115390 | 0.0088 |  |
| CANNY | aperture3_l1 | headers_baseline | CV_8U | 1 | 1080x1920 | 109.409635 | 27.869696 | 0.2547 |  |
| CANNY | aperture3_l1 | headers_baseline | CV_8U | 1 | 480x640 | 16.213369 | 4.034640 | 0.2488 |  |
| CANNY | aperture3_l1 | headers_baseline | CV_8U | 1 | 720x1280 | 49.388894 | 12.294381 | 0.2489 |  |
| COPY_MAKE_BORDER | 2px_replicate | headers_baseline | CV_8U | 1 | 1080x1920 | 1.480787 | 0.040454 | 0.0273 |  |
| COPY_MAKE_BORDER | 2px_replicate | headers_baseline | CV_8U | 1 | 480x640 | 0.228688 | 0.006210 | 0.0272 |  |
| COPY_MAKE_BORDER | 2px_replicate | headers_baseline | CV_8U | 1 | 720x1280 | 0.655996 | 0.018008 | 0.0275 |  |
| CVTCOLOR | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | 1080x1920 | 0.204588 | 0.203946 | 0.9969 |  |
| CVTCOLOR | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | 480x640 | 0.029977 | 0.029892 | 0.9971 |  |
| CVTCOLOR | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | 720x1280 | 0.090594 | 0.090423 | 0.9981 |  |
| DILATE | 3x3_replicate | headers_baseline | CV_8U | 1 | 1080x1920 | 38.498740 | 0.134348 | 0.0035 |  |
| DILATE | 3x3_replicate | headers_baseline | CV_8U | 1 | 480x640 | 5.654662 | 0.021083 | 0.0037 |  |
| DILATE | 3x3_replicate | headers_baseline | CV_8U | 1 | 720x1280 | 16.841927 | 0.062527 | 0.0037 |  |
| ERODE | 3x3_replicate | headers_baseline | CV_8U | 1 | 1080x1920 | 35.263546 | 0.147694 | 0.0042 |  |
| ERODE | 3x3_replicate | headers_baseline | CV_8U | 1 | 480x640 | 5.023438 | 0.021046 | 0.0042 |  |
| ERODE | 3x3_replicate | headers_baseline | CV_8U | 1 | 720x1280 | 15.129771 | 0.061975 | 0.0041 |  |
| FILTER2D | 3x3_replicate | headers_baseline | CV_8U | 1 | 1080x1920 | 198.340133 | 0.546250 | 0.0028 |  |
| FILTER2D | 3x3_replicate | headers_baseline | CV_8U | 1 | 480x640 | 29.330762 | 0.084458 | 0.0029 |  |
| FILTER2D | 3x3_replicate | headers_baseline | CV_8U | 1 | 720x1280 | 87.948033 | 0.245975 | 0.0028 |  |
| GAUSSIAN | 5x5_replicate | headers_baseline | CV_8U | 1 | 1080x1920 | 24.958365 | 0.194754 | 0.0078 |  |
| GAUSSIAN | 5x5_replicate | headers_baseline | CV_8U | 1 | 480x640 | 3.787877 | 0.027540 | 0.0073 |  |
| GAUSSIAN | 5x5_replicate | headers_baseline | CV_8U | 1 | 720x1280 | 11.173417 | 0.084308 | 0.0075 |  |
| LUT | invert_u8 | headers_baseline | CV_8U | 1 | 1080x1920 | 8.877044 | 0.173358 | 0.0195 |  |
| LUT | invert_u8 | headers_baseline | CV_8U | 1 | 480x640 | 1.298350 | 0.025579 | 0.0197 |  |
| LUT | invert_u8 | headers_baseline | CV_8U | 1 | 720x1280 | 3.924090 | 0.076742 | 0.0196 |  |
| RESIZE | linear_half_u8c1 | opencv_ui | CV_8U | 1 | 1080x1920 | 0.121106 | 0.081115 | 0.6698 |  |
| RESIZE | linear_half_u8c1 | opencv_ui | CV_8U | 1 | 480x640 | 0.018221 | 0.011940 | 0.6553 |  |
| RESIZE | linear_half_u8c1 | opencv_ui | CV_8U | 1 | 720x1280 | 0.056463 | 0.037342 | 0.6614 |  |
| SEP_FILTER2D | 3x3_replicate | headers_baseline | CV_8U | 1 | 1080x1920 | 25.606185 | 0.568740 | 0.0222 |  |
| SEP_FILTER2D | 3x3_replicate | headers_baseline | CV_8U | 1 | 480x640 | 3.833652 | 0.082771 | 0.0216 |  |
| SEP_FILTER2D | 3x3_replicate | headers_baseline | CV_8U | 1 | 720x1280 | 11.449658 | 0.246104 | 0.0215 |  |
| SOBEL | dx1_ksize3_replicate | headers_baseline | CV_8U | 1 | 1080x1920 | 30.648769 | 0.769804 | 0.0251 |  |
| SOBEL | dx1_ksize3_replicate | headers_baseline | CV_8U | 1 | 480x640 | 4.539521 | 0.107496 | 0.0237 |  |
| SOBEL | dx1_ksize3_replicate | headers_baseline | CV_8U | 1 | 720x1280 | 13.973321 | 0.307085 | 0.0220 |  |
| THRESHOLD | binary_u8 | headers_baseline | CV_8U | 1 | 1080x1920 | 0.967400 | 0.033142 | 0.0343 |  |
| THRESHOLD | binary_u8 | headers_baseline | CV_8U | 1 | 480x640 | 0.142719 | 0.004767 | 0.0334 |  |
| THRESHOLD | binary_u8 | headers_baseline | CV_8U | 1 | 720x1280 | 0.429875 | 0.014415 | 0.0335 |  |
| WARP_AFFINE | linear_inverse_replicate | headers_baseline | CV_8U | 1 | 1080x1920 | 7.877475 | 1.991402 | 0.2528 |  |
| WARP_AFFINE | linear_inverse_replicate | headers_baseline | CV_8U | 1 | 480x640 | 1.101960 | 0.295358 | 0.2680 |  |
| WARP_AFFINE | linear_inverse_replicate | headers_baseline | CV_8U | 1 | 720x1280 | 3.321802 | 0.888648 | 0.2675 |  |

## 说明

- 比值统一为 `OpenCV耗时 / CVH耗时`：大于 `1` 表示 CVH 更快，小于 `1` 表示 OpenCV 更快。
- 表内耗时取各 repeat 的最小单次耗时，用于降低系统抖动影响；本报告不是跨机器排名。
- Mat case 对比相同的分配/复用语义；imgproc case 对齐输入尺寸、类型、kernel、border 和主要参数。
- `headers_baseline` 不等于跳过优化，它表示 `cvh::headers_fast` 当前继承了 `cvh::headers` 的通用实现。
- 原始 CSV 和 metadata 是可再生成的运行产物，日期命名 Markdown 是阶段性快照。
