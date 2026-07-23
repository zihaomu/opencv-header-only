# opencv-header-only vs OpenCV Upstream 性能对比（2026-07-23）

生成时间（UTC）：`2026-07-23 10:22:30Z`

## 当前项目状态

- `opencv-header-only` 当前公共定位是纯 header-only，不依赖项目内 `.cpp` 扩展层。
- Mode B 只比较当前 `cvh::headers_fast` 与同机编译的 upstream OpenCV；`cvh::headers_fast` 表示最快 header-only 构建配置。
- `cvh::headers_fast` 完整继承 `cvh::headers`。算子没有专用 fast-path 时继续执行继承的 header 实现并参与 benchmark，不因缺少 SIMD 特化而跳过。
- Core 的 `add/subtract/multiply/divide/transpose/GEMM` 已迁入 ODR-safe headers；本报告通过公共 API 测量，不链接 legacy core 对象。
- OpenCV Universal Intrinsics 是默认 SIMD 方言，kernel 直接使用 OpenCV UI；项目已移除 xsimd 性能路径。
- ARM 当前关注 NEON，本次实测平台为 Apple ARM；x86 目标是 SSE/AVX 系列，RVV 因 scalable vector 设计问题暂缓。
- Imgproc legacy `.cpp` fast-path 已迁入 ODR-safe detail headers；resize/cvtColor UI、filter、LUT、border、Sobel、Canny 和 morphology 均从公共 header API 进入。

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
- CVH commit：`46825e2f45e2680cb86f72934ca2a2654fe49c30` + dirty
- OpenCV：`4.14.0`，commit `d48bf69f65444a13f8a34b8982b083c1b78fa0e8` + dirty
- 原始数据：`2026-07-23-opencv-upstream-performance.csv`；元数据：`2026-07-23-opencv-upstream-performance.csv.meta.json`

## 汇总

- 总 case：`126`；有效：`126`；不支持：`0`。
- `OpenCV/CVH` 几何平均：`0.2403`；中位数：`0.3231`。
- CVH 更快：`12` 个；OpenCV 更快或相当：`114` 个。

| Suite | Cases | 几何平均 OpenCV/CVH | 中位数 | CVH 更快 | OpenCV 更快/相当 |
| --- | --- | --- | --- | --- | --- |
| core_mat | 84 | 0.2182 | 0.3255 | 8 | 76 |
| imgproc | 42 | 0.2913 | 0.2608 | 4 | 38 |

## 算子级概览

### `core_mat`

| Op | CVH dispatch | Cases | 几何平均 OpenCV/CVH | 领先方 |
| --- | --- | --- | --- | --- |
| ADD | headers_baseline | 12 | 0.1850 | OpenCV `5.41x` |
| DIVIDE | headers_baseline | 12 | 0.3391 | OpenCV `2.95x` |
| GEMM | headers_baseline | 6 | 0.0125 | OpenCV `79.95x` |
| MAT_CLONE | headers_baseline | 3 | 0.9997 | OpenCV `1.00x` |
| MAT_CONVERTTO | headers_baseline | 3 | 0.9806 | OpenCV `1.02x` |
| MAT_COPYTO | headers_baseline | 3 | 0.9236 | OpenCV `1.08x` |
| MAT_CREATE | headers_baseline | 3 | 0.1077 | OpenCV `9.29x` |
| MAT_RESHAPE | headers_baseline | 3 | 0.4012 | OpenCV `2.49x` |
| MAT_SETTO | headers_baseline | 3 | 0.0118 | OpenCV `84.74x` |
| MULTIPLY | headers_baseline | 12 | 0.1859 | OpenCV `5.38x` |
| SUBTRACT | headers_baseline | 12 | 0.1898 | OpenCV `5.27x` |
| TRANSPOSE | headers_baseline | 12 | 0.6501 | OpenCV `1.54x` |

### `imgproc`

| Op | CVH dispatch | Cases | 几何平均 OpenCV/CVH | 领先方 |
| --- | --- | --- | --- | --- |
| BOX_FILTER | header_fastpath | 3 | 0.1816 | OpenCV `5.51x` |
| CANNY | header_fastpath | 3 | 0.9306 | OpenCV `1.07x` |
| COPY_MAKE_BORDER | header_fastpath | 3 | 0.2823 | OpenCV `3.54x` |
| CVTCOLOR | opencv_ui | 3 | 0.9964 | OpenCV `1.00x` |
| DILATE | header_fastpath | 3 | 0.1055 | OpenCV `9.48x` |
| ERODE | header_fastpath | 3 | 0.1096 | OpenCV `9.12x` |
| FILTER2D | header_fastpath | 3 | 0.2428 | OpenCV `4.12x` |
| GAUSSIAN | header_fastpath | 3 | 0.1182 | OpenCV `8.46x` |
| LUT | header_fastpath | 3 | 0.6380 | OpenCV `1.57x` |
| RESIZE | opencv_ui | 3 | 0.6657 | OpenCV `1.50x` |
| SEP_FILTER2D | header_fastpath | 3 | 0.3385 | OpenCV `2.95x` |
| SOBEL | header_fastpath | 3 | 1.5737 | CVH `1.57x` |
| THRESHOLD | headers_baseline | 3 | 0.0344 | OpenCV `29.03x` |
| WARP_AFFINE | headers_baseline | 3 | 0.2575 | OpenCV `3.88x` |

## 详细结果

### `core_mat`

| Op | Variant | CVH dispatch | Depth | Ch | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADD | mat_mat_continuous | headers_baseline | CV_32F | 1 | 1080x1920 | 0.665592 | 0.215648 | 0.3240 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_32F | 1 | 480x640 | 0.094988 | 0.030756 | 0.3238 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_32F | 1 | 720x1280 | 0.286219 | 0.096492 | 0.3371 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_32F | 3 | 1080x1920 | 1.200540 | 0.617377 | 0.5142 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_32F | 3 | 480x640 | 0.169750 | 0.091354 | 0.5382 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_32F | 3 | 720x1280 | 0.533600 | 0.288327 | 0.5403 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_8U | 1 | 1080x1920 | 0.793056 | 0.049835 | 0.0628 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_8U | 1 | 480x640 | 0.119223 | 0.007023 | 0.0589 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_8U | 1 | 720x1280 | 0.358260 | 0.022483 | 0.0628 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_8U | 3 | 1080x1920 | 1.464775 | 0.160079 | 0.1093 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_8U | 3 | 480x640 | 0.217287 | 0.023773 | 0.1094 | correctness=upstream_pass |
| ADD | mat_mat_continuous | headers_baseline | CV_8U | 3 | 720x1280 | 0.651075 | 0.071052 | 0.1091 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_32F | 1 | 1080x1920 | 0.620308 | 0.203044 | 0.3273 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_32F | 1 | 480x640 | 0.090102 | 0.029823 | 0.3310 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_32F | 1 | 720x1280 | 0.291869 | 0.094113 | 0.3224 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_32F | 3 | 1080x1920 | 1.492577 | 0.624817 | 0.4186 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_32F | 3 | 480x640 | 0.220208 | 0.092940 | 0.4221 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_32F | 3 | 720x1280 | 0.666206 | 0.269000 | 0.4038 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_8U | 1 | 1080x1920 | 1.766371 | 0.451944 | 0.2559 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_8U | 1 | 480x640 | 0.261873 | 0.066852 | 0.2553 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_8U | 1 | 720x1280 | 0.783988 | 0.200977 | 0.2564 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_8U | 3 | 1080x1920 | 3.569731 | 1.365075 | 0.3824 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_8U | 3 | 480x640 | 0.529658 | 0.200302 | 0.3782 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | headers_baseline | CV_8U | 3 | 720x1280 | 1.580215 | 0.604308 | 0.3824 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| GEMM | fp32_nn_end_to_end | headers_baseline | CV_32F | 1 | 128x128x128 | 0.219406 | 0.003646 | 0.0166 | correctness=upstream_pass;iters=8 |
| GEMM | fp32_nn_end_to_end | headers_baseline | CV_32F | 1 | 256x256x256 | 2.012708 | 0.024250 | 0.0120 | correctness=upstream_pass;iters=1 |
| GEMM | fp32_nn_end_to_end | headers_baseline | CV_32F | 1 | 512x512x512 | 17.714750 | 0.169583 | 0.0096 | correctness=upstream_pass;iters=1 |
| GEMM | fp32_nn_pack_once | headers_baseline | CV_32F | 1 | 128x128x128 | 0.209292 | 0.003630 | 0.0173 | correctness=upstream_pass;opencv_reuses_B_without_public_pack_handle;iters=8 |
| GEMM | fp32_nn_pack_once | headers_baseline | CV_32F | 1 | 256x256x256 | 2.116917 | 0.025250 | 0.0119 | correctness=upstream_pass;opencv_reuses_B_without_public_pack_handle;iters=1 |
| GEMM | fp32_nn_pack_once | headers_baseline | CV_32F | 1 | 512x512x512 | 17.514666 | 0.169084 | 0.0097 | correctness=upstream_pass;opencv_reuses_B_without_public_pack_handle;iters=1 |
| MAT_CLONE | full_copy | headers_baseline | CV_8U | 1 | 1080x1920 | 0.024323 | 0.024631 | 1.0127 |  |
| MAT_CLONE | full_copy | headers_baseline | CV_8U | 1 | 480x640 | 0.009556 | 0.009335 | 0.9769 |  |
| MAT_CLONE | full_copy | headers_baseline | CV_8U | 1 | 720x1280 | 0.011035 | 0.011146 | 1.0100 |  |
| MAT_CONVERTTO | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | 1080x1920 | 0.078417 | 0.078177 | 0.9969 |  |
| MAT_CONVERTTO | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | 480x640 | 0.013767 | 0.013221 | 0.9604 |  |
| MAT_CONVERTTO | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | 720x1280 | 0.034783 | 0.034252 | 0.9847 |  |
| MAT_COPYTO | continuous_reuse | headers_baseline | CV_8U | 1 | 1080x1920 | 0.024688 | 0.023467 | 0.9506 |  |
| MAT_COPYTO | continuous_reuse | headers_baseline | CV_8U | 1 | 480x640 | 0.008952 | 0.007350 | 0.8210 |  |
| MAT_COPYTO | continuous_reuse | headers_baseline | CV_8U | 1 | 720x1280 | 0.010792 | 0.010896 | 1.0097 |  |
| MAT_CREATE | reuse_same_shape | headers_baseline | CV_8U | 1 | 1080x1920 | 0.000011 | 0.000001 | 0.1098 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | reuse_same_shape | headers_baseline | CV_8U | 1 | 480x640 | 0.000025 | 0.000002 | 0.0966 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | reuse_same_shape | headers_baseline | CV_8U | 1 | 720x1280 | 0.000011 | 0.000001 | 0.1178 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_RESHAPE | to_column_view | headers_baseline | CV_8U | 1 | 1080x1920 | 0.000038 | 0.000015 | 0.4087 | micro_iters_x1000 |
| MAT_RESHAPE | to_column_view | headers_baseline | CV_8U | 1 | 480x640 | 0.000042 | 0.000017 | 0.3966 | micro_iters_x1000 |
| MAT_RESHAPE | to_column_view | headers_baseline | CV_8U | 1 | 720x1280 | 0.000039 | 0.000015 | 0.3985 | micro_iters_x1000 |
| MAT_SETTO | scalar_all | headers_baseline | CV_8U | 1 | 1080x1920 | 2.063096 | 0.024148 | 0.0117 |  |
| MAT_SETTO | scalar_all | headers_baseline | CV_8U | 1 | 480x640 | 0.433113 | 0.004746 | 0.0110 |  |
| MAT_SETTO | scalar_all | headers_baseline | CV_8U | 1 | 720x1280 | 0.860688 | 0.011027 | 0.0128 |  |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_32F | 1 | 1080x1920 | 0.621506 | 0.203281 | 0.3271 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_32F | 1 | 480x640 | 0.092496 | 0.029954 | 0.3238 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_32F | 1 | 720x1280 | 0.292933 | 0.098312 | 0.3356 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_32F | 3 | 1080x1920 | 1.123100 | 0.624696 | 0.5562 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_32F | 3 | 480x640 | 0.168033 | 0.091367 | 0.5437 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_32F | 3 | 720x1280 | 0.501708 | 0.270042 | 0.5382 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_8U | 1 | 1080x1920 | 0.795583 | 0.049662 | 0.0624 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_8U | 1 | 480x640 | 0.117512 | 0.007015 | 0.0597 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_8U | 1 | 720x1280 | 0.353473 | 0.022046 | 0.0624 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_8U | 3 | 1080x1920 | 1.375587 | 0.148854 | 0.1082 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_8U | 3 | 480x640 | 0.205998 | 0.022367 | 0.1086 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | headers_baseline | CV_8U | 3 | 720x1280 | 0.616008 | 0.066606 | 0.1081 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_32F | 1 | 1080x1920 | 0.651073 | 0.201571 | 0.3096 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_32F | 1 | 480x640 | 0.095460 | 0.030335 | 0.3178 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_32F | 1 | 720x1280 | 0.294881 | 0.097100 | 0.3293 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_32F | 3 | 1080x1920 | 1.126167 | 0.612829 | 0.5442 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_32F | 3 | 480x640 | 0.168819 | 0.089779 | 0.5318 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_32F | 3 | 720x1280 | 0.512710 | 0.270492 | 0.5276 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_8U | 1 | 1080x1920 | 0.757923 | 0.049548 | 0.0654 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_8U | 1 | 480x640 | 0.111754 | 0.007010 | 0.0627 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_8U | 1 | 720x1280 | 0.339008 | 0.022031 | 0.0650 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_8U | 3 | 1080x1920 | 1.248821 | 0.149371 | 0.1196 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_8U | 3 | 480x640 | 0.189331 | 0.022510 | 0.1189 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | headers_baseline | CV_8U | 3 | 720x1280 | 0.589396 | 0.068673 | 0.1165 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_32F | 1 | 1080x1920 | 0.795994 | 0.640852 | 0.8051 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_32F | 1 | 480x640 | 0.167521 | 0.073208 | 0.4370 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_32F | 1 | 720x1280 | 0.386779 | 0.306204 | 0.7917 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_32F | 3 | 1080x1920 | 0.719985 | 1.491444 | 2.0715 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_32F | 3 | 480x640 | 0.116462 | 0.161650 | 1.3880 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_32F | 3 | 720x1280 | 0.281771 | 0.564048 | 2.0018 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_8U | 1 | 1080x1920 | 0.562894 | 0.130887 | 0.2325 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_8U | 1 | 480x640 | 0.142617 | 0.007083 | 0.0497 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_8U | 1 | 720x1280 | 0.286875 | 0.027515 | 0.0959 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_8U | 3 | 1080x1920 | 0.369525 | 0.752077 | 2.0353 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_8U | 3 | 480x640 | 0.107313 | 0.089940 | 0.8381 | correctness=upstream_pass |
| TRANSPOSE | continuous | headers_baseline | CV_8U | 3 | 720x1280 | 0.209225 | 0.393410 | 1.8803 | correctness=upstream_pass |

### `imgproc`

| Op | Variant | CVH dispatch | Depth | Ch | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BOX_FILTER | 3x3_replicate | header_fastpath | CV_8U | 1 | 1080x1920 | 1.485917 | 0.285096 | 0.1919 |  |
| BOX_FILTER | 3x3_replicate | header_fastpath | CV_8U | 1 | 480x640 | 0.279456 | 0.048327 | 0.1729 |  |
| BOX_FILTER | 3x3_replicate | header_fastpath | CV_8U | 1 | 720x1280 | 0.710406 | 0.128123 | 0.1804 |  |
| CANNY | aperture3_l1 | header_fastpath | CV_8U | 1 | 1080x1920 | 29.211146 | 27.765369 | 0.9505 |  |
| CANNY | aperture3_l1 | header_fastpath | CV_8U | 1 | 480x640 | 4.498719 | 3.986340 | 0.8861 |  |
| CANNY | aperture3_l1 | header_fastpath | CV_8U | 1 | 720x1280 | 12.775906 | 12.226521 | 0.9570 |  |
| COPY_MAKE_BORDER | 2px_replicate | header_fastpath | CV_8U | 1 | 1080x1920 | 0.071781 | 0.044625 | 0.6217 |  |
| COPY_MAKE_BORDER | 2px_replicate | header_fastpath | CV_8U | 1 | 480x640 | 0.060448 | 0.007254 | 0.1200 |  |
| COPY_MAKE_BORDER | 2px_replicate | header_fastpath | CV_8U | 1 | 720x1280 | 0.067146 | 0.020242 | 0.3015 |  |
| CVTCOLOR | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | 1080x1920 | 0.204833 | 0.203369 | 0.9929 |  |
| CVTCOLOR | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | 480x640 | 0.031317 | 0.031292 | 0.9992 |  |
| CVTCOLOR | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | 720x1280 | 0.097717 | 0.097431 | 0.9971 |  |
| DILATE | 3x3_replicate | header_fastpath | CV_8U | 1 | 1080x1920 | 1.120983 | 0.152135 | 0.1357 |  |
| DILATE | 3x3_replicate | header_fastpath | CV_8U | 1 | 480x640 | 0.310481 | 0.023667 | 0.0762 |  |
| DILATE | 3x3_replicate | header_fastpath | CV_8U | 1 | 720x1280 | 0.594831 | 0.067437 | 0.1134 |  |
| ERODE | 3x3_replicate | header_fastpath | CV_8U | 1 | 1080x1920 | 1.129387 | 0.153785 | 0.1362 |  |
| ERODE | 3x3_replicate | header_fastpath | CV_8U | 1 | 480x640 | 0.288023 | 0.023150 | 0.0804 |  |
| ERODE | 3x3_replicate | header_fastpath | CV_8U | 1 | 720x1280 | 0.582444 | 0.070108 | 0.1204 |  |
| FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | 1080x1920 | 2.474373 | 0.607200 | 0.2454 |  |
| FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | 480x640 | 0.418602 | 0.097994 | 0.2341 |  |
| FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | 720x1280 | 1.154090 | 0.287560 | 0.2492 |  |
| GAUSSIAN | 5x5_replicate | header_fastpath | CV_8U | 1 | 1080x1920 | 1.591298 | 0.226360 | 0.1422 |  |
| GAUSSIAN | 5x5_replicate | header_fastpath | CV_8U | 1 | 480x640 | 0.348154 | 0.032085 | 0.0922 |  |
| GAUSSIAN | 5x5_replicate | header_fastpath | CV_8U | 1 | 720x1280 | 0.777175 | 0.097854 | 0.1259 |  |
| LUT | invert_u8 | header_fastpath | CV_8U | 1 | 1080x1920 | 0.186646 | 0.192821 | 1.0331 |  |
| LUT | invert_u8 | header_fastpath | CV_8U | 1 | 480x640 | 0.083548 | 0.029792 | 0.3566 |  |
| LUT | invert_u8 | header_fastpath | CV_8U | 1 | 720x1280 | 0.121519 | 0.085671 | 0.7050 |  |
| RESIZE | linear_half_u8c1 | opencv_ui | CV_8U | 1 | 1080x1920 | 0.120900 | 0.081110 | 0.6709 |  |
| RESIZE | linear_half_u8c1 | opencv_ui | CV_8U | 1 | 480x640 | 0.019617 | 0.013002 | 0.6628 |  |
| RESIZE | linear_half_u8c1 | opencv_ui | CV_8U | 1 | 720x1280 | 0.060796 | 0.040337 | 0.6635 |  |
| SEP_FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | 1080x1920 | 1.637252 | 0.636471 | 0.3887 |  |
| SEP_FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | 480x640 | 0.337090 | 0.097760 | 0.2900 |  |
| SEP_FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | 720x1280 | 0.795321 | 0.273598 | 0.3440 |  |
| SOBEL | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | 1080x1920 | 0.377156 | 0.861335 | 2.2838 |  |
| SOBEL | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | 480x640 | 0.119388 | 0.123717 | 1.0363 |  |
| SOBEL | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | 720x1280 | 0.209063 | 0.344292 | 1.6468 |  |
| THRESHOLD | binary_u8 | headers_baseline | CV_8U | 1 | 1080x1920 | 0.963977 | 0.033242 | 0.0345 |  |
| THRESHOLD | binary_u8 | headers_baseline | CV_8U | 1 | 480x640 | 0.147296 | 0.005208 | 0.0354 |  |
| THRESHOLD | binary_u8 | headers_baseline | CV_8U | 1 | 720x1280 | 0.440723 | 0.014769 | 0.0335 |  |
| WARP_AFFINE | linear_inverse_replicate | headers_baseline | CV_8U | 1 | 1080x1920 | 7.731581 | 1.940660 | 0.2510 |  |
| WARP_AFFINE | linear_inverse_replicate | headers_baseline | CV_8U | 1 | 480x640 | 1.177133 | 0.308823 | 0.2624 |  |
| WARP_AFFINE | linear_inverse_replicate | headers_baseline | CV_8U | 1 | 720x1280 | 3.337954 | 0.865481 | 0.2593 |  |

## 说明

- 比值统一为 `OpenCV耗时 / CVH耗时`：大于 `1` 表示 CVH 更快，小于 `1` 表示 OpenCV 更快。
- 表内耗时取各 repeat 的最小单次耗时，用于降低系统抖动影响；本报告不是跨机器排名。
- Mat case 对比相同的分配/复用语义；imgproc case 对齐输入尺寸、类型、kernel、border 和主要参数。
- `headers_baseline` 不等于跳过优化，它表示 `cvh::headers_fast` 当前继承了 `cvh::headers` 的通用实现。
- 原始 CSV 和 metadata 是可再生成的运行产物，日期命名 Markdown 是阶段性快照。
